# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compressed C4 Lightning Indexer for Hy4-flash indexed MLA layers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from b12x.attention import sparse_mla
from b12x.preparation import PreparedCall

from hyperflare_recipes.components.fp8_indexer import PreparedFP8Indexer, write_fp8_index_keys
from hyperflare_recipes.components.linear import prepare_projection
from hyperflare_recipes.components.paged_batch import rotary_positions
from hyperflare_recipes.components.rms_norm import PreparedRMSNorm
from ignition.components.state import gather_rows, save_hidden
from ignition.preparation.tuning import declare

from .config import (
    GROUP_TOPK,
    INDEX_HEAD_DIM,
    INDEX_HEADS,
    INDEX_KPOOL,
    INDEX_TOPK,
    QK_ROPE_HEAD_DIM,
)
from .pool import INDEX_PAGE_RECORDS, PAGE_TOKENS

# Bound per-launch prefill indexer workspace while amortizing selection kernels.
INDEXER_QUERY_ROWS = 512
PHYSICAL_WIDTH = GROUP_TOPK * INDEX_KPOOL + (INDEX_KPOOL - 1)  # 2051


@dataclass(frozen=True)
class IndexScratch:
    query: torch.Tensor
    weights: torch.Tensor
    token_keys: torch.Tensor
    pool_lengths: torch.Tensor
    prior_pools: torch.Tensor
    pools: torch.Tensor
    physical: torch.Tensor
    physical_topk: torch.Tensor
    selected_lengths: torch.Tensor

    @classmethod
    def allocate(cls, tokens: int, device: torch.device) -> IndexScratch:
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            query=zeros(tokens, INDEX_HEADS, INDEX_HEAD_DIM),
            weights=zeros(tokens, INDEX_HEADS),
            token_keys=zeros(tokens, INDEX_HEAD_DIM),
            pool_lengths=zeros(tokens, dtype=torch.int32),
            prior_pools=torch.full((tokens, GROUP_TOPK), -1, dtype=torch.int32, device=device),
            pools=torch.full((tokens, GROUP_TOPK), -1, dtype=torch.int32, device=device),
            physical=torch.full((tokens, PHYSICAL_WIDTH), -1, dtype=torch.int32, device=device),
            physical_topk=torch.full((tokens, INDEX_TOPK), -1, dtype=torch.int32, device=device),
            selected_lengths=zeros(tokens, dtype=torch.int32),
        )

    def rows(self, tokens: int) -> IndexScratch:
        return IndexScratch(
            query=self.query[:tokens],
            weights=self.weights[:tokens],
            token_keys=self.token_keys[:tokens],
            pool_lengths=self.pool_lengths[:tokens],
            prior_pools=self.prior_pools[:tokens],
            pools=self.pools[:tokens],
            physical=self.physical[:tokens],
            physical_topk=self.physical_topk[:tokens],
            selected_lengths=self.selected_lengths[:tokens],
        )


def _expand_3d_positions(rope_pos: torch.Tensor) -> torch.Tensor:
    """Normalize ``[T]``, ``[T, 1]``, or ``[T, 3]`` rotary positions to ``[T, 3]`` int64."""
    if rope_pos.ndim == 1:
        return rope_pos[:, None].expand(-1, 3)
    if rope_pos.shape[1] == 1:
        return rope_pos.expand(-1, 3)
    return rope_pos


def expand_pooled_topk_to_logical_tokens(
    pools: torch.Tensor,
    positions: torch.Tensor,
    *,
    pool_size: int = INDEX_KPOOL,
    width: int = INDEX_TOPK,
) -> torch.Tensor:
    """Expand packed pooled top-k group IDs and partial tail into logical token positions ``[T, width]``.

    Matches ``b12x.attention.sparse_mla.expand_pooled_topk_to_physical_slots`` contiguous-prefix
    slot ordering while keeping indices logical (page-table independent) for checkpoint-safe
    continuation state. Valid pool IDs are compacted to the leading columns before appending the
    live partial tail so ordering does not rely on caller-side sentinel placement.
    """
    tokens = pools.shape[0]
    device = pools.device
    live = positions >= 0
    safe_pos = positions.clamp_min(0)
    raw_valid = live[:, None] & (pools >= 0)
    # Compact valid pools into the leading columns so the partial tail never overwrites a valid pool.
    order = raw_valid.to(torch.int64).argsort(dim=1, descending=True, stable=True)
    compacted_pools = pools.gather(1, order)
    valid_pool = raw_valid.gather(1, order)
    offsets = torch.arange(pool_size, dtype=torch.int32, device=device)[None, None, :]
    pool_tokens = torch.where(
        valid_pool[:, :, None],
        compacted_pools[:, :, None].to(torch.int32) * pool_size + offsets,
        -1,
    ).reshape(tokens, -1)[:, :width]
    num_valid_pools = valid_pool.sum(dim=1, dtype=torch.int64)
    seq_len = safe_pos + 1
    tail_start = (seq_len // pool_size) * pool_size
    tail_len = seq_len - tail_start
    tail_off = torch.arange(pool_size - 1, dtype=torch.int64, device=device)[None, :]
    tail_valid = live[:, None] & (tail_off < tail_len[:, None])
    tail_col = (num_valid_pools[:, None] * pool_size + tail_off).clamp(0, width - 1)
    tail_tok = torch.where(tail_valid, (tail_start[:, None] + tail_off).to(torch.int32), -1)
    out = pool_tokens.clone()
    existing = out.gather(1, tail_col)
    out.scatter_(1, tail_col, torch.where(tail_valid, tail_tok, existing))
    return out


class PreparedCompressedIndexer:
    """Incremental C4 key compression, trailing-64 3D M-RoPE, and pooled FP8 selection.

    Matches ``HYV4FlashCompressedIndexer``: every four consecutive token keys are
    compressed in FP32 via softmax-gated linear projections plus ``intra_group_ape``,
    normalized with ``key_norm``, rotated on the trailing 64 channels at the group's
    first token 3D M-RoPE position, and stored into 64-record FP8 index pages. Each
    query selects up to 511 strictly prior complete groups plus its own group and
    expands them into physical slots for sparse MLA.
    """

    def __init__(
        self,
        layer: int,
        weights: dict,
        source: torch.Tensor,
        query_residual: torch.Tensor,
        scratch: IndexScratch,
        metadata,
        pool,
        rotary,
        *,
        mode: str,
        eps: float,
        session,
        name: str,
        workspace,
        max_cache_pages: int | None = None,
        examples: dict | None = None,
    ):
        self.layer = layer
        self.pool = pool
        self.rotary = rotary
        self.mode = mode
        self.eps = eps
        self.deferred = mode == "verify"
        tokens = source.shape[0]
        device = source.device
        self.index_cache = pool.index_cache(layer)
        self.pending_keys_region = pool.region(f"layer{layer}.pending_keys")
        self.pending_pos_region = pool.region(f"layer{layer}.pending_group_pos")

        self.query_projection = prepare_projection(
            query_residual,
            weights["query_projection"],
            scratch.query.view(tokens, -1),
            session=session,
            name=f"{name}.query",
            workspace=workspace,
            activation_mode="auto",
        ).functional(session=session)
        self.head_weight_projection = (
            None
            if weights.get("head_weight_projection") is None
            else prepare_projection(
                source,
                weights["head_weight_projection"],
                scratch.weights,
                session=session,
                name=f"{name}.head_weight",
                workspace=workspace,
                activation_mode="auto",
            ).functional(session=session)
        )
        self.token_key_projection = (
            None
            if weights.get("token_key_projection") is None
            else prepare_projection(
                source,
                weights["token_key_projection"],
                scratch.token_keys,
                session=session,
                name=f"{name}.token_key",
                workspace=workspace,
                activation_mode="auto",
            ).functional(session=session)
        )
        self.token_key_norm = PreparedRMSNorm(
            scratch.token_keys,
            weights["token_key_norm"],
            scratch.token_keys,
            eps=eps,
            session=session,
            name=f"{name}.token_key_norm",
        ).functional()

        self.key_vg_projection = weights.get("key_vg_projection")  # [256, 128] FP32 when fused
        if self.key_vg_projection is None:
            self.key_value_projection = weights["key_value_projection"]  # [128, 128] FP32
            self.key_gate_projection = weights["key_gate_projection"]  # [128, 128] FP32
        else:
            self.key_value_projection = None
            self.key_gate_projection = None
        self.intra_group_ape = weights["intra_group_ape"]  # [4, 128] FP32
        self.key_norm = weights["key_norm"]  # [128] FP32 or BF16

        step = INDEXER_QUERY_ROWS if mode == "prefill" else tokens
        select_mode = "prefill" if mode == "prefill" else "decode"
        self.slices = tuple((s, min(tokens, s + step)) for s in range(0, tokens, step))
        self.selectors = tuple(
            PreparedFP8Indexer(
                scratch.query[s:e],
                scratch.weights[s:e],
                self.index_cache,
                metadata.token_page_table[s:e],
                scratch.pool_lengths[s:e],
                None,
                scratch.prior_pools[s:e],
                mode=select_mode,
                session=session,
                name=f"{name}.select" if e - s == tokens else f"{name}.select{s}",
                workspace=workspace,
                topk=GROUP_TOPK,
                head_weight_scale=None,
                hadamard=False,
                key_power_of_two=False,
                examples=examples,
                example_role=None if examples is None else "hy4.indexer",
            ).functional(session=session)
            for s, e in self.slices
        )

        # Pooled physical expansion plan (pool_size=4, block_size=256).
        req_i32 = metadata.requests.to(torch.int32)
        self.expand_options = {
            "pool_size": INDEX_KPOOL,
            "block_size": PAGE_TOKENS,
            "block_stride_rows": PAGE_TOKENS,
            "num_cache_blocks": self.index_cache.shape[0],
        }
        self.expand_plan = sparse_mla.plan_pooled_selection(
            device=device,
            max_rows=tokens,
            page_size=PAGE_TOKENS,
            max_page_table_width=metadata.page_table.shape[1],
            num_cache_blocks=max(self.index_cache.shape[0], max_cache_pages or 0),
        )
        expand_args = (
            scratch.pools,
            metadata.positions,
            req_i32,
            metadata.page_table,
            scratch.physical,
            scratch.selected_lengths,
        )
        expand_opts = self.expand_options
        declare(
            session,
            (
                self.expand_plan.request(
                    name=f"{name}.expand",
                    prepare_call=lambda state: PreparedCall(run=lambda: state.run(*expand_args, **expand_opts)),
                ),
            ),
        )

    def rebind_pages(self) -> None:
        self.index_cache = self.pool.index_cache(self.layer)
        self.expand_options["num_cache_blocks"] = self.index_cache.shape[0]
        for selector in self.selectors:
            selector.cache = self.index_cache
            selector.selection.rebind_cache(self.index_cache)

    def __call__(
        self,
        source: torch.Tensor,
        query_residual: torch.Tensor,
        metadata,
        *,
        q_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        token_keys_raw: torch.Tensor | None = None,
        head_weights_raw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        m = metadata
        tokens = source.shape[0]
        rows = m.starts.numel()
        device = source.device
        dtype = source.dtype

        # 1. Project & normalize per-token keys and gather partial-group state.
        if token_keys_raw is None:
            token_keys_raw = self.token_key_projection(source)
        token_keys = self.token_key_norm(token_keys_raw.contiguous())  # [T, 128] BF16
        pending_keys = torch.zeros((rows, INDEX_KPOOL - 1, INDEX_HEAD_DIM), dtype=dtype, device=device)
        pending_pos = torch.zeros((rows, 3), dtype=torch.int64, device=device)
        gather_rows(self.pending_keys_region, m.state_rows, pending_keys)
        gather_rows(self.pending_pos_region, m.state_rows, pending_pos)

        t_idx = torch.arange(tokens, dtype=torch.int64, device=device)
        req = m.requests.long()
        req_begin = m.query_start_loc[:-1].long()[req]
        local_off = t_idx - req_begin
        pos = m.positions  # -1 for inactive tokens
        live = pos >= 0
        safe_pos = pos.clamp_min(0)
        rope_pos3 = _expand_3d_positions(rotary_positions(m))  # [T, 3]

        # 2. Assemble 4-token groups for any token that closes a group (pos % 4 == 3).
        k_slots = []
        group_pos = pending_pos[req]
        for slot_in_group in range(INDEX_KPOOL):
            delta = (INDEX_KPOOL - 1) - slot_in_group  # 3, 2, 1, 0
            if delta == 0:
                k_slots.append(token_keys)
            else:
                in_batch = local_off >= delta
                src_idx = (t_idx - delta).clamp(0, tokens - 1)
                k_slot = torch.where(in_batch[:, None], token_keys[src_idx], pending_keys[req, slot_in_group])
                k_slots.append(k_slot)
                if slot_in_group == 0:
                    group_pos = torch.where(in_batch[:, None], rope_pos3[src_idx], pending_pos[req])

        # FP32 softmax-gated pooling matching HYV4FlashCompressedIndexer.pool_groups.
        grouped = torch.stack(k_slots, dim=1).float()  # [T, 4, 128]
        if self.key_vg_projection is not None:
            values, gates_raw = F.linear(grouped, self.key_vg_projection).chunk(2, dim=-1)
        else:
            values = F.linear(grouped, self.key_value_projection)
            gates_raw = F.linear(grouped, self.key_gate_projection)
        gates = gates_raw + self.intra_group_ape[None, :, :]
        pooled = (values * gates.softmax(dim=1)).sum(dim=1)  # [T, 128] FP32
        inv_rms = torch.rsqrt(pooled.square().mean(dim=-1, keepdim=True) + self.eps)
        group_keys = (pooled * inv_rms * self.key_norm.float()).to(dtype)  # [T, 128] BF16

        # Apply trailing-64 interleaved 3D M-RoPE at each group's first token position.
        nope_dim = INDEX_HEAD_DIM - QK_ROPE_HEAD_DIM
        g_cos, g_sin = self.rotary.cos_sin(group_pos, indexed=True, dtype=dtype)
        k_pass, k_rot = group_keys.split((nope_dim, QK_ROPE_HEAD_DIM), dim=-1)
        k_rot = self.rotary.apply(k_rot, g_cos, g_sin)
        rotated_keys = torch.cat((k_pass, k_rot), dim=-1).contiguous()

        completes = live & ((safe_pos & 3) == 3)
        page_col = (safe_pos // PAGE_TOKENS).clamp(0, m.page_table.shape[1] - 1)
        pages = m.page_table[req, page_col].long()
        group_slots = torch.where(completes & (pages >= 0), pages * INDEX_PAGE_RECORDS + (safe_pos % PAGE_TOKENS) // INDEX_KPOOL, -1)
        write_fp8_index_keys(rotated_keys, group_slots, self.index_cache, hadamard=False, power_of_two=False)

        pending_record = (token_keys, rope_pos3, pending_keys, pending_pos)
        if not self.deferred:
            self.commit(m.lengths, pending_record, m)

        # 3. Project query + trailing-64 interleaved 3D M-RoPE and select top-511 prior groups.
        query = self.query_projection(query_residual).view(tokens, INDEX_HEADS, INDEX_HEAD_DIM)
        q_cos, q_sin = self.rotary.cos_sin(rope_pos3, indexed=True, dtype=dtype) if q_cos_sin is None else q_cos_sin
        q_pass, q_rot = query.split((nope_dim, QK_ROPE_HEAD_DIM), dim=-1)
        q_rot = self.rotary.apply(q_rot, q_cos, q_sin)
        rotated_q = torch.cat((q_pass, q_rot), dim=-1).contiguous()

        head_weights = (
            self.head_weight_projection(source) if head_weights_raw is None else head_weights_raw.contiguous()
        )  # [T, 16] BF16
        group_id = safe_pos // INDEX_KPOOL
        pool_lengths = torch.where(live, group_id, 0).to(torch.int32)

        if len(self.selectors) == 1:
            prior_pools = self.selectors[0](rotated_q, head_weights, m.token_page_table, pool_lengths, None, cache=self.index_cache)
        else:
            chunks = [
                selector(
                    rotated_q[s:e],
                    head_weights[s:e],
                    m.token_page_table[s:e],
                    pool_lengths[s:e],
                    None,
                    cache=self.index_cache,
                )
                for selector, (s, e) in zip(self.selectors, self.slices, strict=True)
            ]
            prior_pools = torch.cat(chunks, dim=0)

        # 4. Keep at most 511 prior groups plus the own group (when completing at pos % 4 == 3)
        # so expanded selection never exceeds 2048 tokens and matches HYV4FlashCompressedIndexer.
        pools = prior_pools.clone()
        own_col = group_id.clamp(max=GROUP_TOPK - 1).unsqueeze(1)
        replace_val = torch.where(
            completes,
            group_id.to(torch.int32),
            torch.full_like(group_id, -1, dtype=torch.int32),
        ).unsqueeze(1)
        existing_col = pools.gather(1, own_col)
        should_write = (completes | (group_id >= GROUP_TOPK)).unsqueeze(1)
        pools.scatter_(1, own_col, torch.where(should_write, replace_val, existing_col))

        physical = torch.full((tokens, PHYSICAL_WIDTH), -1, dtype=torch.int32, device=device)
        selected_lengths = torch.zeros((tokens,), dtype=torch.int32, device=device)
        sparse_mla.expand_pooled_topk_to_physical_slots(
            pools,
            pos,
            m.requests.to(torch.int32),
            m.page_table,
            physical,
            selected_lengths,
            **self.expand_options,
            plan=self.expand_plan,
        )
        # At most 511 prior groups plus the own group are active, so all valid slots
        # fit inside the first INDEX_TOPK (2048) columns required by GLM_NSA sparse MLA.
        # Return logical pooled group IDs alongside physical slots so continuation state
        # persists page-table-independent logical token positions for DFly2 reuse.
        return physical[:, :INDEX_TOPK].contiguous(), selected_lengths, pending_record, pools

    def commit(
        self,
        counts: torch.Tensor,
        record: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        metadata,
    ) -> None:
        """Persist trailing 0..3 unpooled token keys and partial-group start 3D position."""
        m = metadata
        token_keys, rope_pos3, pending_keys, pending_pos = record
        tokens = token_keys.shape[0]
        device = token_keys.device
        active_req = counts > 0
        starts = m.starts
        ends = starts + counts.long()
        new_rem = ends % INDEX_KPOOL
        tail_start = ends - new_rem
        req_begin = m.query_start_loc[:-1].long()

        slots = torch.arange(INDEX_KPOOL - 1, dtype=torch.int64, device=device)[None, :]  # [1, 3]
        slot_pos = tail_start[:, None] + slots  # [R, 3]
        in_new = slot_pos >= starts[:, None]
        token_idx = (req_begin[:, None] + (slot_pos - starts[:, None])).clamp(0, tokens - 1)
        needed = slots < new_rem[:, None]

        new_pending = torch.where(
            in_new[:, :, None],
            token_keys[token_idx],
            pending_keys,
        )
        new_pending = torch.where(needed[:, :, None], new_pending, 0)

        first_in_new = in_new[:, 0]
        new_pos = torch.where(first_in_new[:, None], rope_pos3[token_idx[:, 0]], pending_pos)
        new_pos = torch.where((new_rem > 0)[:, None], new_pos, 0)

        save_hidden(new_pending.reshape(m.starts.numel(), -1), self.pending_keys_region.reshape(self.pending_keys_region.shape[0], -1), m.state_rows, active_req)
        save_hidden(new_pos, self.pending_pos_region, m.state_rows, active_req)
