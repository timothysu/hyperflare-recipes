# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Hybrid sparse-indexed (32-head) and sliding-window-512 (64-head) gated MLA for Hy4-flash."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from b12x.attention import dense_mla, sparse_mla

from hyperflare_recipes.components.dense_mla import PreparedDenseMLA
from hyperflare_recipes.components.linear import (
    BatchedQuantizedProjection,
    BatchedUnquantizedProjection,
    prepare_projection,
)
from hyperflare_recipes.components.paged_batch import rotary_positions
from hyperflare_recipes.components.rms_norm import PreparedRMSNorm
from hyperflare_recipes.components.sparse_mla import PreparedSparseMLA
from ignition.components.state import gather_rows, scatter_rows
from ignition.preparation.persistent import persistent_copy

from .config import (
    HIDDEN_SIZE,
    INDEX_HEAD_DIM,
    INDEX_HEADS,
    INDEX_TOPK,
    KV_LORA_RANK,
    QK_NOPE_HEAD_DIM,
    QK_ROPE_HEAD_DIM,
    Q_LORA_RANK,
    SPARSE_HEADS,
    SWA_HEADS,
    SWA_WINDOW,
    V_HEAD_DIM,
)
from .indexer import IndexScratch, PreparedCompressedIndexer
from .pool import PAGE_TOKENS, SPARSE_RECORD_BYTES, SWA_RECORD_WIDTH

QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 256
MLA_SCALE = QK_HEAD_DIM ** -0.5  # 1 / 16
SWA_PAGE_SIZE = 64
_FP8_MAX = 448.0
_QKV_BASE_WIDTH = Q_LORA_RANK + KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 1600
_QKV_INDEXER_WIDTH = _QKV_BASE_WIDTH + INDEX_HEAD_DIM + INDEX_HEADS  # 1744
# Worst-case fused qkv_a width: SWA with fused gate (1600 + 16384 = 17984) vs sparse with fused gate + indexer (9936).
_QKV_MAX_WIDTH = max(
    _QKV_BASE_WIDTH + SWA_HEADS * V_HEAD_DIM,
    _QKV_INDEXER_WIDTH + SPARSE_HEADS * V_HEAD_DIM,
)


def _projection_out_features(weight) -> int:
    return int(weight.shape[0]) if isinstance(weight, torch.Tensor) else int(weight.out_features)


def prepare_swa_plan(metadata, stage_len: int, max_query_tokens: int) -> tuple:
    """Precompute shared SWA-512 direct-staging and ring-commit indices once per forward."""
    m = metadata
    tokens = m.positions.numel()
    device = m.positions.device
    hist_len = m.starts.clamp(0, SWA_WINDOW - 1)  # [R]
    hist_start = (m.starts - hist_len).clamp_min(0)  # [R]
    offsets = torch.arange(stage_len, dtype=torch.int64, device=device)[None, :]
    ring_idx = (hist_start[:, None] + offsets) % SWA_WINDOW  # [R, stage_len]
    is_hist = offsets < hist_len[:, None]  # [R, stage_len]
    src_idx = (m.query_start_loc[:-1, None].long() + offsets - hist_len[:, None]).clamp(0, tokens - 1)
    valid_stage = (m.lengths[:, None] > 0) & (offsets < hist_len[:, None] + m.lengths[:, None])
    cache_seqlens = torch.where(m.lengths > 0, hist_len + m.lengths.long(), 0).to(torch.int32)
    live = m.positions >= 0
    t_idx = torch.arange(tokens, dtype=torch.int64, device=device)
    req = m.requests.long()
    local_off = t_idx - m.query_start_loc[:-1].long()[req]
    ring_write_slots = req * (SWA_WINDOW + max_query_tokens) + SWA_WINDOW + local_off
    return (ring_idx, is_hist, src_idx, valid_stage, cache_seqlens, live, ring_write_slots)


def prepare_sparse_slots(metadata) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute shared paged sparse-MLA physical slots and visible lengths once per forward."""
    m = metadata
    live = m.positions >= 0
    safe_pos = m.positions.clamp_min(0)
    req = m.requests.long()
    page_col = (safe_pos // PAGE_TOKENS).clamp(0, m.page_table.shape[1] - 1)
    pages = m.page_table[req, page_col].long()
    token_slots = torch.where(live & (pages >= 0), pages * PAGE_TOKENS + (safe_pos % PAGE_TOKENS), -1)
    visible = torch.where(live, safe_pos + 1, 0).to(torch.int32)
    return (live, token_slots, visible)


def split_kv_b_weights(kv_b, heads: int):
    """Split ``kv_b_proj`` into batched query-absorption and value-expansion operands."""
    if isinstance(kv_b, torch.Tensor):
        view = kv_b.view(heads, QK_NOPE_HEAD_DIM + V_HEAD_DIM, KV_LORA_RANK)
        return view[:, :QK_NOPE_HEAD_DIM], view[:, QK_NOPE_HEAD_DIM:], BatchedUnquantizedProjection
    packed = kv_b.weight
    values = packed.values.view(heads, QK_NOPE_HEAD_DIM + V_HEAD_DIM, KV_LORA_RANK)
    scales = packed.scale_rows.view(heads, QK_NOPE_HEAD_DIM + V_HEAD_DIM, KV_LORA_RANK // 32)
    return (
        (values[:, :QK_NOPE_HEAD_DIM], scales[:, :QK_NOPE_HEAD_DIM]),
        (values[:, QK_NOPE_HEAD_DIM:], scales[:, QK_NOPE_HEAD_DIM:]),
        BatchedQuantizedProjection,
    )


def pack_glm_nsa_records(kv_nope: torch.Tensor, k_rope: torch.Tensor) -> torch.Tensor:
    """Pack BF16 ``kv_nope [T, 512]`` and ``k_rope [T, 64]`` into 656-byte GLM_NSA records.

    Layout: 512 bytes of E4M3 quantized NoPE (4 x 128-element tiles), 16 bytes of
    inline FP32 tile scales (``absmax / 448.0``), and 128 bytes of raw BF16 RoPE.
    """
    tokens = kv_nope.shape[0]
    blocks = kv_nope.float().view(tokens, 4, 128)
    scales = blocks.abs().amax(dim=-1) / _FP8_MAX  # [T, 4] FP32
    scales = torch.where(scales > 0.0, scales, 1.0)
    quant = (blocks / scales.unsqueeze(-1)).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    quant_bytes = quant.view(torch.uint8).reshape(tokens, 512)
    scale_bytes = scales.contiguous().view(torch.uint8).reshape(tokens, 16)
    rope_bytes = k_rope.contiguous().view(torch.uint8).reshape(tokens, 128)
    return torch.cat((quant_bytes, scale_bytes, rope_bytes), dim=1).contiguous()


def swa_stage_tokens(max_query_tokens: int) -> int:
    """Linear staging tokens per request (511 prior ring tokens + max_query_tokens, 64-aligned)."""
    needed = (SWA_WINDOW - 1) + max_query_tokens
    return ((needed + SWA_PAGE_SIZE - 1) // SWA_PAGE_SIZE) * SWA_PAGE_SIZE


@dataclass(frozen=True)
class AttentionScratch:
    projected: torch.Tensor
    projected_fused: torch.Tensor
    q_resid: torch.Tensor
    kv_nope: torch.Tensor
    q_sparse: torch.Tensor
    q_swa: torch.Tensor
    q_abs_sparse: torch.Tensor
    q_abs_swa: torch.Tensor
    attended_sparse: torch.Tensor
    lse_sparse: torch.Tensor
    attended_swa: torch.Tensor
    lse_swa: torch.Tensor
    expanded_sparse: torch.Tensor
    expanded_swa: torch.Tensor
    gate_sparse: torch.Tensor
    gate_swa: torch.Tensor
    o_proj_out: torch.Tensor
    output: torch.Tensor
    visible: torch.Tensor
    swa_staged: torch.Tensor
    swa_lengths: torch.Tensor
    index: IndexScratch

    @classmethod
    def allocate(cls, tokens: int, rows: int, max_query_tokens: int, device: torch.device) -> AttentionScratch:
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)

        stage_len = swa_stage_tokens(max_query_tokens)
        projected_fused = zeros(tokens * _QKV_MAX_WIDTH)
        q_swa = zeros(tokens, SWA_HEADS, QK_HEAD_DIM)
        q_abs_swa = zeros(tokens, SWA_HEADS, SWA_RECORD_WIDTH)
        attended_swa = zeros(tokens, SWA_HEADS, KV_LORA_RANK)
        lse_swa = zeros(tokens, SWA_HEADS, dtype=torch.float32)
        expanded_swa = zeros(tokens, SWA_HEADS, V_HEAD_DIM)
        gate_swa = zeros(tokens, SWA_HEADS * V_HEAD_DIM)
        return cls(
            projected=projected_fused[: tokens * _QKV_BASE_WIDTH].view(tokens, _QKV_BASE_WIDTH),
            projected_fused=projected_fused,
            q_resid=zeros(tokens, Q_LORA_RANK),
            kv_nope=zeros(tokens, KV_LORA_RANK),
            q_sparse=q_swa.view(-1)[: tokens * SPARSE_HEADS * QK_HEAD_DIM].view(tokens, SPARSE_HEADS, QK_HEAD_DIM),
            q_swa=q_swa,
            q_abs_sparse=q_abs_swa.view(-1)[: tokens * SPARSE_HEADS * SWA_RECORD_WIDTH].view(
                tokens, SPARSE_HEADS, SWA_RECORD_WIDTH
            ),
            q_abs_swa=q_abs_swa,
            attended_sparse=attended_swa.view(-1)[: tokens * SPARSE_HEADS * KV_LORA_RANK].view(
                tokens, SPARSE_HEADS, KV_LORA_RANK
            ),
            lse_sparse=lse_swa.view(-1)[: tokens * SPARSE_HEADS].view(tokens, SPARSE_HEADS),
            attended_swa=attended_swa,
            lse_swa=lse_swa,
            expanded_sparse=expanded_swa.view(-1)[: tokens * SPARSE_HEADS * V_HEAD_DIM].view(
                tokens, SPARSE_HEADS, V_HEAD_DIM
            ),
            expanded_swa=expanded_swa,
            gate_sparse=gate_swa.view(-1)[: tokens * SPARSE_HEADS * V_HEAD_DIM].view(tokens, SPARSE_HEADS * V_HEAD_DIM),
            gate_swa=gate_swa,
            o_proj_out=zeros(tokens, HIDDEN_SIZE),
            output=zeros(tokens, HIDDEN_SIZE),
            visible=zeros(tokens, dtype=torch.int32),
            swa_staged=zeros(rows, stage_len, SWA_RECORD_WIDTH),
            swa_lengths=zeros(rows, dtype=torch.int32),
            index=IndexScratch.allocate(tokens, device),
        )

    def rows(self, tokens: int, rows: int, max_query_tokens: int) -> AttentionScratch:
        stage_len = swa_stage_tokens(max_query_tokens)
        needed_rows = rows * stage_len
        if needed_rows > self.swa_staged.shape[0] * self.swa_staged.shape[1] or rows > self.swa_lengths.shape[0]:
            raise ValueError("SWA staging rows/tokens exceed allocated attention scratch")
        swa_staged = self.swa_staged.view(-1, SWA_RECORD_WIDTH)[:needed_rows].view(rows, stage_len, SWA_RECORD_WIDTH)
        return AttentionScratch(
            projected=self.projected_fused[: tokens * _QKV_BASE_WIDTH].view(tokens, _QKV_BASE_WIDTH),
            projected_fused=self.projected_fused[: tokens * _QKV_MAX_WIDTH],
            q_resid=self.q_resid[:tokens],
            kv_nope=self.kv_nope[:tokens],
            q_sparse=self.q_swa.view(-1)[: tokens * SPARSE_HEADS * QK_HEAD_DIM].view(tokens, SPARSE_HEADS, QK_HEAD_DIM),
            q_swa=self.q_swa[:tokens],
            q_abs_sparse=self.q_abs_swa.view(-1)[: tokens * SPARSE_HEADS * SWA_RECORD_WIDTH].view(
                tokens, SPARSE_HEADS, SWA_RECORD_WIDTH
            ),
            q_abs_swa=self.q_abs_swa[:tokens],
            attended_sparse=self.attended_swa.view(-1)[: tokens * SPARSE_HEADS * KV_LORA_RANK].view(
                tokens, SPARSE_HEADS, KV_LORA_RANK
            ),
            lse_sparse=self.lse_swa.view(-1)[: tokens * SPARSE_HEADS].view(tokens, SPARSE_HEADS),
            attended_swa=self.attended_swa[:tokens],
            lse_swa=self.lse_swa[:tokens],
            expanded_sparse=self.expanded_swa.view(-1)[: tokens * SPARSE_HEADS * V_HEAD_DIM].view(
                tokens, SPARSE_HEADS, V_HEAD_DIM
            ),
            expanded_swa=self.expanded_swa[:tokens],
            gate_sparse=self.gate_swa.view(-1)[: tokens * SPARSE_HEADS * V_HEAD_DIM].view(
                tokens, SPARSE_HEADS * V_HEAD_DIM
            ),
            gate_swa=self.gate_swa[:tokens],
            o_proj_out=self.o_proj_out[:tokens],
            output=self.output[:tokens],
            visible=self.visible[:tokens],
            swa_staged=swa_staged,
            swa_lengths=self.swa_lengths[:rows],
            index=self.index.rows(tokens),
        )

    def projected_view(self, tokens: int, width: int) -> torch.Tensor:
        if width == _QKV_BASE_WIDTH:
            return self.projected[:tokens]
        if tokens * width <= self.projected_fused.numel():
            return self.projected_fused[: tokens * width].view(tokens, width)
        raise ValueError(
            f"projected_view({tokens}, {width}) exceeds allocated attention scratch ({self.projected_fused.numel()} elements)"
        )


class PreparedSparseAttention:
    """32-head sparse MLA over 656-byte GLM_NSA pages with C4 indexing, sink, and output gate."""

    def __init__(
        self,
        layer: int,
        weights: dict,
        indexer_weights: dict,
        source: torch.Tensor,
        scratch: AttentionScratch,
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
        self.cache = pool.latent_cache(layer)
        tokens = source.shape[0]
        device = source.device

        self.fused_gate = bool(weights.get("fused_gate", False))
        self.fused_indexer_projs = bool(weights.get("fused_indexer_projs", False))
        qkv_out_dim = _projection_out_features(weights["qkv_a"])
        proj_buf = scratch.projected_view(tokens, qkv_out_dim)
        self.qkv_a = prepare_projection(
            source,
            weights["qkv_a"],
            proj_buf,
            session=session,
            name=f"{name}.qkv_a",
            workspace=workspace,
            activation_mode="auto",
        ).functional(session=session)
        self.q_norm = PreparedRMSNorm(
            scratch.projected[:, :Q_LORA_RANK],
            weights["q_norm"],
            scratch.q_resid,
            eps=eps,
            session=session,
            name=f"{name}.q_norm",
        ).functional()
        self.kv_norm = PreparedRMSNorm(
            scratch.projected[:, Q_LORA_RANK : Q_LORA_RANK + KV_LORA_RANK],
            weights["kv_norm"],
            scratch.kv_nope,
            eps=eps,
            session=session,
            name=f"{name}.kv_norm",
        ).functional()
        self.q_b = prepare_projection(
            scratch.q_resid,
            weights["q_b"],
            scratch.q_sparse.view(tokens, -1),
            session=session,
            name=f"{name}.q_b",
            workspace=workspace,
            activation_mode="auto",
        ).functional(session=session)

        absorb_w, expand_w, batched_cls = split_kv_b_weights(weights["kv_b"], SPARSE_HEADS)
        self.query_absorption = batched_cls(
            scratch.q_sparse[:, :, :QK_NOPE_HEAD_DIM].transpose(0, 1),
            absorb_w,
            scratch.q_abs_sparse[:, :, :KV_LORA_RANK].transpose(0, 1),
            major="n",
            session=session,
            name=f"{name}.absorb_query",
        ).functional(session=session)
        self.value_expansion = batched_cls(
            scratch.attended_sparse.transpose(0, 1),
            expand_w,
            scratch.expanded_sparse.transpose(0, 1),
            major="k",
            session=session,
            name=f"{name}.expand_value",
        ).functional(session=session)

        self.gate_proj = (
            None
            if self.fused_gate
            else prepare_projection(
                source,
                weights["gate"],
                scratch.gate_sparse,
                session=session,
                name=f"{name}.gate",
                workspace=workspace,
                activation_mode="auto",
            ).functional(session=session)
        )
        self.o_proj = prepare_projection(
            scratch.expanded_sparse.view(tokens, -1),
            weights["o_proj"],
            scratch.o_proj_out,
            session=session,
            name=f"{name}.o_proj",
            workspace=workspace,
            activation_mode="auto",
        ).functional(session=session)
        self.post_proj_norm = PreparedRMSNorm(
            scratch.o_proj_out,
            weights["post_proj_norm"],
            scratch.output,
            eps=eps,
            session=session,
            name=f"{name}.post_proj_norm",
        ).functional()

        self.indexer = PreparedCompressedIndexer(
            layer,
            indexer_weights,
            source,
            scratch.q_resid,
            scratch.index,
            metadata,
            pool,
            rotary,
            mode=mode,
            eps=eps,
            session=session,
            name=f"{name}.indexer",
            workspace=workspace,
            max_cache_pages=max_cache_pages,
            examples=examples,
        )

        self.sink = weights["sink"]  # [32] FP32
        self.fused_sink = mode in ("decode", "verify")
        max_kv_rows = max(self.cache.shape[0], 4 * (max_cache_pages or 0)) * 64
        caps = sparse_mla.Caps(
            device=device,
            num_q_heads=SPARSE_HEADS,
            max_q_rows=tokens,
            max_width=INDEX_TOPK,
            softmax_scale=MLA_SCALE,
            max_batch=tokens,
            max_kv_rows=max_kv_rows,
            kv_dtype=torch.uint8,
            head_dim=SWA_RECORD_WIDTH,
            v_head_dim=KV_LORA_RANK,
            page_size=64,
            model_type=sparse_mla.ModelType.GLM_NSA,
            cache_record_bytes=SPARSE_RECORD_BYTES,
            return_lse=not self.fused_sink,
            # Prepared b12x sparse_mla extend launches run_unified_prefill directly, which
            # emits base-2 LSE; request base2 explicitly and scale by ln(2) for the sink.
            lse_scale="base2",
            has_attention_sink=self.fused_sink,
            mode="decode" if self.fused_sink else "extend",
        )
        self.attention = PreparedSparseMLA(
            scratch.q_abs_sparse,
            self.cache,
            scratch.index.physical_topk,
            scratch.visible,
            scratch.index.selected_lengths,
            scratch.attended_sparse,
            caps=caps,
            session=session,
            name=f"{name}.attention",
            workspace=workspace,
            attention_sink=self.sink if self.fused_sink else None,
            lse=None if self.fused_sink else scratch.lse_sparse,
        ).functional(session=session)

    def rebind_pages(self) -> None:
        self.cache = self.pool.latent_cache(self.layer)
        self.attention.rebind_cache(self.cache)
        self.indexer.rebind_pages()

    def __call__(
        self,
        source: torch.Tensor,
        metadata,
        *,
        cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        sparse_slots: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, object, torch.Tensor]:
        m = metadata
        tokens = source.shape[0]
        dtype = source.dtype

        projected = self.qkv_a(source)
        q_resid = self.q_norm(projected[:, :Q_LORA_RANK].contiguous())
        kv_nope = self.kv_norm(projected[:, Q_LORA_RANK : Q_LORA_RANK + KV_LORA_RANK].contiguous())
        k_rope = projected[:, Q_LORA_RANK + KV_LORA_RANK : _QKV_BASE_WIDTH].contiguous()
        col = _QKV_BASE_WIDTH
        if self.fused_gate:
            gate_w = SPARSE_HEADS * V_HEAD_DIM
            gate = projected[:, col : col + gate_w]
            col += gate_w
        else:
            gate = None
        if self.fused_indexer_projs:
            token_keys_raw = projected[:, col : col + INDEX_HEAD_DIM]
            col += INDEX_HEAD_DIM
            head_weights_raw = projected[:, col : col + INDEX_HEADS]
        else:
            token_keys_raw = None
            head_weights_raw = None

        q_raw = self.q_b(q_resid).view(tokens, SPARSE_HEADS, QK_HEAD_DIM)
        q_nope, q_rope = q_raw.split((QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM), dim=-1)

        if cos_sin is None:
            cos, sin = self.rotary.cos_sin(rotary_positions(m), indexed=True, dtype=dtype)
        else:
            cos, sin = cos_sin
        q_rope = self.rotary.apply(q_rope, cos, sin)
        k_rope = self.rotary.apply(k_rope, cos, sin)

        q_abs_nope = self.query_absorption(q_nope.transpose(0, 1).contiguous()).transpose(0, 1)
        q_absorbed = torch.cat((q_abs_nope, q_rope), dim=-1).contiguous()

        # Pack and scatter 656-byte GLM_NSA records into paged latent storage.
        if sparse_slots is None:
            live, token_slots, visible = prepare_sparse_slots(m)
        else:
            live, token_slots, visible = sparse_slots
        packed = pack_glm_nsa_records(kv_nope, k_rope)
        scatter_rows(packed, token_slots, self.cache.view(-1, SPARSE_RECORD_BYTES), token_slots >= 0)

        physical, selected_lengths, pending_record, pools = self.indexer(
            source,
            q_resid,
            m,
            q_cos_sin=(cos, sin),
            token_keys_raw=token_keys_raw,
            head_weights_raw=head_weights_raw,
        )
        if self.fused_sink:
            attended = self.attention(q_absorbed, physical, visible, selected_lengths, cache=self.cache)
        else:
            attended, lse = self.attention(q_absorbed, physical, visible, selected_lengths, cache=self.cache)
            # Apply exact post-hoc learnable attention sink in FP32 after converting base-2 LSE.
            sink_scale = torch.sigmoid(lse * math.log(2.0) - self.sink[None, :]).unsqueeze(-1)
            attended = (attended.float() * sink_scale).to(dtype)
            attended = torch.where(live[:, None, None], attended, 0)

        expanded = self.value_expansion(attended.transpose(0, 1).contiguous()).transpose(0, 1).reshape(tokens, -1)
        if gate is None:
            gate = self.gate_proj(source)
        gated = (expanded * torch.sigmoid(gate)).contiguous()
        output = self.post_proj_norm(self.o_proj(gated))
        return output, pending_record, pools

    def commit(self, counts: torch.Tensor, record, metadata) -> None:
        self.indexer.commit(counts, record, metadata)


def apply_attention_sink(
    attended: torch.Tensor,
    lse: torch.Tensor,
    sink: torch.Tensor,
    live: torch.Tensor,
) -> torch.Tensor:
    """Apply exact post-hoc learnable attention sink in FP32 from natural-log LSE."""
    sink_scale = torch.sigmoid(lse - sink[None, :]).unsqueeze(-1)
    scaled = (attended.float() * sink_scale).to(attended.dtype)
    return torch.where(live[:, None, None], scaled, 0)


class PreparedSlidingAttention:
    """64-head BF16 sliding-window-512 MLA with unwrapped linear staging and exact FP32 sink."""

    def __init__(
        self,
        layer: int,
        weights: dict,
        source: torch.Tensor,
        scratch: AttentionScratch,
        metadata,
        pool,
        rotary,
        *,
        mode: str,
        eps: float,
        session,
        name: str,
        workspace,
    ):
        self.layer = layer
        self.pool = pool
        self.rotary = rotary
        self.mode = mode
        self.deferred = mode == "verify"
        self.window = pool.window(layer)  # [max_rows, 512, 576] BF16
        self.sink = weights["sink"]  # [64] FP32
        tokens = source.shape[0]
        rows = metadata.starts.numel()
        device = source.device
        self.stage_len = swa_stage_tokens(metadata.max_query_tokens)
        self.max_query_tokens = metadata.max_query_tokens
        self.pages_per_row = self.stage_len // SWA_PAGE_SIZE
        self.page_table = persistent_copy(
            torch.arange(rows * self.pages_per_row, dtype=torch.int32, device=device).view(rows, self.pages_per_row)
        )

        self.fused_gate = bool(weights.get("fused_gate", False))
        qkv_out_dim = _projection_out_features(weights["qkv_a"])
        proj_buf = scratch.projected_view(tokens, qkv_out_dim)
        self.qkv_a = prepare_projection(
            source,
            weights["qkv_a"],
            proj_buf,
            session=session,
            name=f"{name}.qkv_a",
            workspace=workspace,
            activation_mode="auto",
        ).functional(session=session)
        self.q_norm = PreparedRMSNorm(
            scratch.projected[:, :Q_LORA_RANK],
            weights["q_norm"],
            scratch.q_resid,
            eps=eps,
            session=session,
            name=f"{name}.q_norm",
        ).functional()
        self.kv_norm = PreparedRMSNorm(
            scratch.projected[:, Q_LORA_RANK : Q_LORA_RANK + KV_LORA_RANK],
            weights["kv_norm"],
            scratch.kv_nope,
            eps=eps,
            session=session,
            name=f"{name}.kv_norm",
        ).functional()
        self.q_b = prepare_projection(
            scratch.q_resid,
            weights["q_b"],
            scratch.q_swa.view(tokens, -1),
            session=session,
            name=f"{name}.q_b",
            workspace=workspace,
            activation_mode="auto",
        ).functional(session=session)

        absorb_w, expand_w, batched_cls = split_kv_b_weights(weights["kv_b"], SWA_HEADS)
        # b12x MXFP8 batched GEMM qualifies batch sizes {8, 16, 32}; partition 64 SWA
        # heads into two 32-head groups sharing the same packed MXFP8 weight storage.
        if batched_cls is BatchedQuantizedProjection:
            half = SWA_HEADS // 2
            self.query_absorption = tuple(
                batched_cls(
                    scratch.q_swa[:, h0:h1, :QK_NOPE_HEAD_DIM].transpose(0, 1),
                    (absorb_w[0][h0:h1], absorb_w[1][h0:h1]),
                    scratch.q_abs_swa[:, h0:h1, :KV_LORA_RANK].transpose(0, 1),
                    major="n",
                    session=session,
                    name=f"{name}.absorb_query{idx}",
                ).functional(session=session)
                for idx, (h0, h1) in enumerate(((0, half), (half, SWA_HEADS)))
            )
            self.value_expansion = tuple(
                batched_cls(
                    scratch.attended_swa[:, h0:h1].transpose(0, 1),
                    (expand_w[0][h0:h1], expand_w[1][h0:h1]),
                    scratch.expanded_swa[:, h0:h1].transpose(0, 1),
                    major="k",
                    session=session,
                    name=f"{name}.expand_value{idx}",
                ).functional(session=session)
                for idx, (h0, h1) in enumerate(((0, half), (half, SWA_HEADS)))
            )
        else:
            self.query_absorption = (
                batched_cls(
                    scratch.q_swa[:, :, :QK_NOPE_HEAD_DIM].transpose(0, 1),
                    absorb_w,
                    scratch.q_abs_swa[:, :, :KV_LORA_RANK].transpose(0, 1),
                    major="n",
                    session=session,
                    name=f"{name}.absorb_query",
                ).functional(session=session),
            )
            self.value_expansion = (
                batched_cls(
                    scratch.attended_swa.transpose(0, 1),
                    expand_w,
                    scratch.expanded_swa.transpose(0, 1),
                    major="k",
                    session=session,
                    name=f"{name}.expand_value",
                ).functional(session=session),
            )

        self.gate_proj = (
            None
            if self.fused_gate
            else prepare_projection(
                source,
                weights["gate"],
                scratch.gate_swa,
                session=session,
                name=f"{name}.gate",
                workspace=workspace,
                activation_mode="auto",
            ).functional(session=session)
        )
        self.o_proj = prepare_projection(
            scratch.expanded_swa.view(tokens, -1),
            weights["o_proj"],
            scratch.o_proj_out,
            session=session,
            name=f"{name}.o_proj",
            workspace=workspace,
            activation_mode="auto",
        ).functional(session=session)
        self.post_proj_norm = PreparedRMSNorm(
            scratch.o_proj_out,
            weights["post_proj_norm"],
            scratch.output,
            eps=eps,
            session=session,
            name=f"{name}.post_proj_norm",
        ).functional()

        staged_pages = scratch.swa_staged.view(rows * self.pages_per_row, SWA_PAGE_SIZE, SWA_RECORD_WIDTH)
        caps = dense_mla.Caps(
            device=device,
            mode="decode" if mode == "decode" else "extend",
            kv_dtype=torch.bfloat16,
            q_dtype=torch.bfloat16,
            num_q_heads=SWA_HEADS,
            page_size=SWA_PAGE_SIZE,
            max_total_q=tokens,
            max_batch=rows,
            max_cache_tokens=self.stage_len,
            max_page_table_width=self.pages_per_row,
            num_cache_pages=rows * self.pages_per_row,
            head_dim=SWA_RECORD_WIDTH,
            v_head_dim=KV_LORA_RANK,
            window_size=SWA_WINDOW,
            causal=True,
            partial_dtype=torch.float32,
        )
        self.attention = PreparedDenseMLA(
            scratch.q_abs_swa,
            staged_pages,
            scratch.attended_swa,
            self.page_table,
            scratch.swa_lengths,
            metadata.query_start_loc,
            caps=caps,
            scale=MLA_SCALE,
            session=session,
            name=f"{name}.attention",
            workspace=workspace,
            lse=scratch.lse_swa,
        ).functional(session=session, cache=None)

    def __call__(
        self,
        source: torch.Tensor,
        metadata,
        *,
        cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        swa_plan: tuple | None = None,
        window_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, object, None]:
        m = metadata
        tokens = source.shape[0]
        rows = m.starts.numel()
        dtype = source.dtype

        projected = self.qkv_a(source)
        q_resid = self.q_norm(projected[:, :Q_LORA_RANK].contiguous())
        kv_nope = self.kv_norm(projected[:, Q_LORA_RANK : Q_LORA_RANK + KV_LORA_RANK].contiguous())
        k_rope = projected[:, Q_LORA_RANK + KV_LORA_RANK : _QKV_BASE_WIDTH].contiguous()
        gate = projected[:, _QKV_BASE_WIDTH : _QKV_BASE_WIDTH + SWA_HEADS * V_HEAD_DIM] if self.fused_gate else None

        q_raw = self.q_b(q_resid).view(tokens, SWA_HEADS, QK_HEAD_DIM)
        q_nope, q_rope = q_raw.split((QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM), dim=-1)

        if cos_sin is None:
            cos, sin = self.rotary.cos_sin(rotary_positions(m), indexed=False, dtype=dtype)
        else:
            cos, sin = cos_sin
        q_rope = self.rotary.apply(q_rope, cos, sin)
        k_rope = self.rotary.apply(k_rope, cos, sin)

        if len(self.query_absorption) == 1:
            q_abs_nope = self.query_absorption[0](q_nope.transpose(0, 1).contiguous()).transpose(0, 1)
        else:
            half = SWA_HEADS // 2
            q_lo = self.query_absorption[0](q_nope[:, :half].transpose(0, 1).contiguous()).transpose(0, 1)
            q_hi = self.query_absorption[1](q_nope[:, half:].transpose(0, 1).contiguous()).transpose(0, 1)
            q_abs_nope = torch.cat((q_lo, q_hi), dim=1)
        q_absorbed = torch.cat((q_abs_nope, q_rope), dim=-1).contiguous()
        kv_record = torch.cat((kv_nope, k_rope), dim=-1).contiguous()  # [T, 576] BF16

        # Stage up to 511 prior ring tokens plus this step's KV records directly into
        # chronological order in one fused pass without full-ring scratch copies.
        if swa_plan is None:
            ring_idx, is_hist, src_idx, valid_stage, cache_seqlens, live, ring_write_slots = prepare_swa_plan(
                m, self.stage_len, self.max_query_tokens
            )
        else:
            ring_idx, is_hist, src_idx, valid_stage, cache_seqlens, live, ring_write_slots = swa_plan

        previous = self.window[m.state_rows[:, None], ring_idx]
        staged = torch.where(
            valid_stage.unsqueeze(-1),
            torch.where(is_hist.unsqueeze(-1), previous, kv_record[src_idx]),
            0,
        ).contiguous()
        staged_pages = staged.view(rows * self.pages_per_row, SWA_PAGE_SIZE, SWA_RECORD_WIDTH)
        attended, lse = self.attention(
            q_absorbed,
            page_table=self.page_table,
            lengths=cache_seqlens,
            starts=m.query_start_loc,
            cache=staged_pages,
        )

        # Exact post-hoc learnable attention sink using FP32 natural-log LSE.
        attended = apply_attention_sink(attended, lse, self.sink, live)

        if window_cache is not None:
            # Stage only this step's new records for batched PreparedRingCommit across all 30 SWA layers.
            wc_bf16 = window_cache.view(torch.bfloat16).view(
                rows, SWA_WINDOW + self.max_query_tokens, SWA_RECORD_WIDTH
            )
            if self.mode == "decode":
                wc_bf16[:, SWA_WINDOW].copy_(kv_record)
            else:
                scatter_rows(kv_record, ring_write_slots, wc_bf16.view(-1, SWA_RECORD_WIDTH), live)
            commit_record = None
        else:
            commit_record = kv_record
            if not self.deferred:
                self.commit(m.lengths, commit_record, m)

        if len(self.value_expansion) == 1:
            expanded = self.value_expansion[0](attended.transpose(0, 1).contiguous()).transpose(0, 1).reshape(tokens, -1)
        else:
            half = SWA_HEADS // 2
            e_lo = self.value_expansion[0](attended[:, :half].transpose(0, 1).contiguous()).transpose(0, 1)
            e_hi = self.value_expansion[1](attended[:, half:].transpose(0, 1).contiguous()).transpose(0, 1)
            expanded = torch.cat((e_lo, e_hi), dim=1).reshape(tokens, -1)
        if gate is None:
            gate = self.gate_proj(source)
        gated = (expanded * torch.sigmoid(gate)).contiguous()
        output = self.post_proj_norm(self.o_proj(gated))
        return output, commit_record, None

    def commit(self, counts: torch.Tensor, record: torch.Tensor | None, metadata) -> None:
        """Write only the last min(count, 512) accepted tokens per request into the circular ring."""
        if record is None:
            return
        kv_record = record
        m = metadata
        rows = m.starts.numel()
        tokens = kv_record.shape[0]
        device = kv_record.device
        ring = torch.zeros((rows, SWA_WINDOW, SWA_RECORD_WIDTH), dtype=kv_record.dtype, device=device)
        gather_rows(self.window, m.state_rows, ring)
        t_idx = torch.arange(tokens, dtype=torch.int64, device=device)
        req = m.requests.long()
        local_off = t_idx - m.query_start_loc[:-1].long()[req]
        pos = m.positions
        state_row = m.state_rows[req]
        end_pos = m.starts[req] + counts.long()[req]
        keep = (
            (pos >= 0)
            & (state_row > 0)
            & (local_off < counts.long()[req])
            & (pos >= end_pos - SWA_WINDOW)
        )
        local_slots = req * SWA_WINDOW + (pos.clamp_min(0) % SWA_WINDOW)
        scatter_rows(kv_record, local_slots, ring.view(-1, SWA_RECORD_WIDTH), keep)
        scatter_rows(ring, m.state_rows, self.window, (counts > 0) & (m.state_rows > 0))
