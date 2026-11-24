# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""5-layer KV-Share/Index-Share DFly2 draft model and stochastic speculative round."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from b12x.attention import sparse_mla

from ignition.components.collectives import PreparedAllReduce
from hyperflare_recipes.components.draft_sampling import PreparedDraftDistribution
from hyperflare_recipes.components.embedding import PreparedEmbedding
from hyperflare_recipes.components.linear import prepare_projection
from hyperflare_recipes.components.mlp import MLPScratch, PreparedMLP
from hyperflare_recipes.components.paged_batch import Metadata, prepare_batch, row_rotary_offsets
from hyperflare_recipes.components.rms_norm import PreparedRMSNorm
from hyperflare_recipes.components.sparse_mla import PreparedSparseMLA
from hyperflare_recipes.components.speculation import (
    DRAFT_RNG_DOMAIN,
    PreparedSpeculativeCommit,
    PreparedSpeculativeDistributions,
    PreparedStochasticVerification,
)
from ignition.components.state import gather_rows, save_hidden
from hyperflare_recipes.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation.tuning import register_finalizer

from .attention import MLA_SCALE, QK_HEAD_DIM, split_kv_b_weights
from .config import (
    HIDDEN_SIZE,
    KV_LORA_RANK,
    QK_NOPE_HEAD_DIM,
    QK_ROPE_HEAD_DIM,
    Q_LORA_RANK,
    SPARSE_HEADS,
    VOCABULARY,
    V_HEAD_DIM,
)
from .pool import PAGE_TOKENS, SPARSE_RECORD_BYTES, SWA_RECORD_WIDTH
from .rotary import InterleavedMRoPE


def combine_three_way_lse(
    out_hist: torch.Tensor,
    lse_hist: torch.Tensor,
    out_blk: torch.Tensor,
    lse_blk: torch.Tensor,
    sink: torch.Tensor,
    selected_lengths: torch.Tensor,
) -> torch.Tensor:
    """Combine target history attention, causal draft block attention, and learnable sink in FP32."""
    has_hist = (selected_lengths > 0)[:, None] & torch.isfinite(lse_hist)
    safe_lse_hist = torch.where(has_hist, lse_hist, float("-inf"))
    sink_row = sink[None, :]
    max_lse = torch.maximum(torch.maximum(safe_lse_hist, lse_blk), sink_row)
    w_hist = torch.where(has_hist, torch.exp(safe_lse_hist - max_lse), 0.0)
    w_blk = torch.exp(lse_blk - max_lse)
    w_sink = torch.exp(sink_row - max_lse)
    denom = (w_hist + w_blk + w_sink).unsqueeze(-1)
    return ((w_hist.unsqueeze(-1) * out_hist.float() + w_blk.unsqueeze(-1) * out_blk.float()) / denom).to(
        out_hist.dtype
    )


def resolve_logical_slots(
    logical_tokens: torch.Tensor,
    page_tables: torch.Tensor,
    active: torch.Tensor,
    *,
    page_tokens: int = PAGE_TOKENS,
) -> torch.Tensor:
    """Resolve logical token positions ``[B, K]`` through live ``page_tables [B, blocks]`` to physical KV slots."""
    valid = active[:, None] & (logical_tokens >= 0)
    safe_tok = logical_tokens.clamp_min(0).long()
    blocks = page_tables.shape[1]
    page_idx = safe_tok // page_tokens
    in_bounds = page_idx < blocks
    phys_page = page_tables.gather(1, page_idx.clamp(0, blocks - 1)).long()
    valid = valid & in_bounds & (phys_page >= 0)
    slots = phys_page * page_tokens + (safe_tok % page_tokens)
    return torch.where(valid, slots.to(torch.int32), -1)


@dataclass(frozen=True)
class DFlyScratch:
    """Reusable scratch buffers for 5-layer DFly2 block proposal and hidden correction."""

    embedding: torch.Tensor
    mtp_concat: torch.Tensor
    mtp_fused: torch.Tensor
    hidden: torch.Tensor
    normalized: torch.Tensor
    projected: torch.Tensor
    projected_fused: torch.Tensor
    q_resid: torch.Tensor
    kv_nope: torch.Tensor
    q_raw: torch.Tensor
    q_abs: torch.Tensor
    attended_hist: torch.Tensor
    lse_hist: torch.Tensor
    expanded: torch.Tensor
    gate: torch.Tensor
    o_proj_out: torch.Tensor
    attn_out: torch.Tensor
    mlp: MLPScratch
    visible: torch.Tensor
    selected_slots: torch.Tensor
    selected_lengths: torch.Tensor
    hc_concat: torch.Tensor
    hc_gate_up: torch.Tensor
    hc_inter: torch.Tensor
    hc_out: torch.Tensor
    step_hidden: torch.Tensor
    step_local_logits: torch.Tensor
    step_logits: torch.Tensor

    @classmethod
    def allocate(
        cls,
        batch: int,
        width: int,
        intermediate_size: int,
        hc_intermediate_size: int,
        topk: int,
        device: torch.device,
        *,
        shard,
    ) -> DFlyScratch:
        tokens = batch * width
        _, local_vocab = shard.padded_partition(VOCABULARY, alignment=64)

        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            embedding=zeros(tokens, HIDDEN_SIZE),
            mtp_concat=zeros(batch, 2 * HIDDEN_SIZE),
            mtp_fused=zeros(batch, HIDDEN_SIZE),
            hidden=zeros(tokens, HIDDEN_SIZE),
            normalized=zeros(tokens, HIDDEN_SIZE),
            projected=zeros(tokens, Q_LORA_RANK + KV_LORA_RANK + QK_ROPE_HEAD_DIM),
            projected_fused=zeros(tokens, Q_LORA_RANK + KV_LORA_RANK + QK_ROPE_HEAD_DIM + SPARSE_HEADS * V_HEAD_DIM),
            q_resid=zeros(tokens, Q_LORA_RANK),
            kv_nope=zeros(tokens, KV_LORA_RANK),
            q_raw=zeros(tokens, SPARSE_HEADS, QK_HEAD_DIM),
            q_abs=zeros(tokens, SPARSE_HEADS, SWA_RECORD_WIDTH),
            attended_hist=zeros(tokens, SPARSE_HEADS, KV_LORA_RANK),
            lse_hist=zeros(tokens, SPARSE_HEADS, dtype=torch.float32),
            expanded=zeros(tokens, SPARSE_HEADS, V_HEAD_DIM),
            gate=zeros(tokens, SPARSE_HEADS * V_HEAD_DIM),
            o_proj_out=zeros(tokens, HIDDEN_SIZE),
            attn_out=zeros(tokens, HIDDEN_SIZE),
            mlp=MLPScratch.allocate(tokens, HIDDEN_SIZE, intermediate_size, device),
            visible=zeros(tokens, dtype=torch.int32),
            selected_slots=torch.full((tokens, topk), -1, dtype=torch.int32, device=device),
            selected_lengths=zeros(tokens, dtype=torch.int32),
            hc_concat=zeros(batch, 2 * HIDDEN_SIZE),
            hc_gate_up=zeros(batch, 2 * hc_intermediate_size),
            hc_inter=zeros(batch, hc_intermediate_size),
            hc_out=zeros(batch, HIDDEN_SIZE),
            step_hidden=zeros(batch, HIDDEN_SIZE),
            step_local_logits=zeros(batch, local_vocab, dtype=torch.bfloat16),
            step_logits=zeros(batch, VOCABULARY, dtype=torch.bfloat16),
        )


class PreparedDFlyLayer:
    """One causal DFly2 layer reusing target KV cache and sparse indices without mutating them."""

    def __init__(
        self,
        draft_index: int,
        target_kv_layer: int,
        weights,
        scratch: DFlyScratch,
        pool,
        rotary: InterleavedMRoPE,
        *,
        batch: int,
        width: int,
        topk: int,
        eps: float,
        session,
        name: str,
        workspace,
        max_cache_pages: int | None = None,
    ):
        self.draft_index = draft_index
        self.target_kv_layer = target_kv_layer
        self.pool = pool
        self.rotary = rotary
        self.batch = batch
        self.width = width
        self.cache = pool.latent_cache(target_kv_layer)
        self.sink = weights.attention["sink"]  # [32] FP32
        tokens = batch * width
        device = scratch.hidden.device

        self.input_norm = PreparedRMSNorm(
            scratch.hidden,
            weights.input_norm,
            scratch.normalized,
            eps=eps,
            session=session,
            name=f"{name}.input_norm",
        ).functional()
        a = weights.attention
        self.fused_gate = bool(a.get("fused_gate", False))
        proj_buf = scratch.projected_fused if self.fused_gate else scratch.projected
        self.qkv_a = prepare_projection(
            scratch.normalized,
            a["qkv_a"],
            proj_buf,
            session=session,
            name=f"{name}.qkv_a",
            workspace=workspace,
            activation_mode="auto",
        ).functional(session=session)
        self.q_norm = PreparedRMSNorm(
            scratch.projected[:, :Q_LORA_RANK],
            a["q_norm"],
            scratch.q_resid,
            eps=eps,
            session=session,
            name=f"{name}.q_norm",
        ).functional()
        self.kv_norm = PreparedRMSNorm(
            scratch.projected[:, Q_LORA_RANK : Q_LORA_RANK + KV_LORA_RANK],
            a["kv_norm"],
            scratch.kv_nope,
            eps=eps,
            session=session,
            name=f"{name}.kv_norm",
        ).functional()
        self.q_b = prepare_projection(
            scratch.q_resid,
            a["q_b"],
            scratch.q_raw.view(tokens, -1),
            session=session,
            name=f"{name}.q_b",
            workspace=workspace,
            activation_mode="auto",
        ).functional(session=session)

        absorb_w, expand_w, batched_cls = split_kv_b_weights(a["kv_b"], SPARSE_HEADS)
        self.query_absorption = batched_cls(
            scratch.q_raw[:, :, :QK_NOPE_HEAD_DIM].transpose(0, 1),
            absorb_w,
            scratch.q_abs[:, :, :KV_LORA_RANK].transpose(0, 1),
            major="n",
            session=session,
            name=f"{name}.absorb_query",
        ).functional(session=session)
        self.value_expansion = batched_cls(
            scratch.attended_hist.transpose(0, 1),
            expand_w,
            scratch.expanded.transpose(0, 1),
            major="k",
            session=session,
            name=f"{name}.expand_value",
        ).functional(session=session)

        self.gate_proj = (
            None
            if self.fused_gate
            else prepare_projection(
                scratch.normalized,
                a["gate"],
                scratch.gate,
                session=session,
                name=f"{name}.gate",
                workspace=workspace,
                activation_mode="auto",
            ).functional(session=session)
        )
        self.o_proj = prepare_projection(
            scratch.expanded.view(tokens, -1),
            a["o_proj"],
            scratch.o_proj_out,
            session=session,
            name=f"{name}.o_proj",
            workspace=workspace,
            activation_mode="auto",
        ).functional(session=session)
        self.post_proj_norm = PreparedRMSNorm(
            scratch.o_proj_out,
            a["post_proj_norm"],
            scratch.attn_out,
            eps=eps,
            session=session,
            name=f"{name}.post_proj_norm",
        ).functional()

        max_kv_rows = max(self.cache.shape[0], 4 * (max_cache_pages or 0)) * 64
        caps = sparse_mla.Caps(
            device=device,
            num_q_heads=SPARSE_HEADS,
            max_q_rows=tokens,
            max_width=topk,
            softmax_scale=MLA_SCALE,
            max_batch=tokens,
            max_kv_rows=max_kv_rows,
            kv_dtype=torch.uint8,
            head_dim=SWA_RECORD_WIDTH,
            v_head_dim=KV_LORA_RANK,
            page_size=64,
            model_type=sparse_mla.ModelType.GLM_NSA,
            cache_record_bytes=SPARSE_RECORD_BYTES,
            return_lse=True,
            lse_scale="natural",
            has_attention_sink=False,
            mode="decode",
        )
        self.history_attention = PreparedSparseMLA(
            scratch.q_abs,
            self.cache,
            scratch.selected_slots,
            scratch.visible,
            scratch.selected_lengths,
            scratch.attended_hist,
            caps=caps,
            session=session,
            name=f"{name}.history_attn",
            workspace=workspace,
            lse=scratch.lse_hist,
        ).functional(session=session)

        self.post_attn_norm = PreparedRMSNorm(
            scratch.hidden,
            weights.post_attn_norm,
            scratch.normalized,
            eps=eps,
            session=session,
            name=f"{name}.post_attn_norm",
        ).functional()
        self.mlp = PreparedMLP(
            scratch.normalized,
            weights.mlp,
            scratch.mlp.gate_up,
            scratch.mlp.intermediate,
            scratch.mlp.output,
            session=session,
            name=f"{name}.mlp",
            workspace=workspace,
            round_silu=False,
            activation_mode="auto",
        ).functional(session=session)

    def rebind_pages(self) -> None:
        self.cache = self.pool.latent_cache(self.target_kv_layer)
        self.history_attention.rebind_cache(self.cache)

    def __call__(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        selected_slots: torch.Tensor,
        visible: torch.Tensor,
        selected_lengths: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens = hidden.shape[0]
        b, w = self.batch, self.width
        dtype = hidden.dtype

        normed = self.input_norm(hidden)
        projected = self.qkv_a(normed)
        base_end = Q_LORA_RANK + KV_LORA_RANK + QK_ROPE_HEAD_DIM
        q_resid = self.q_norm(projected[:, :Q_LORA_RANK].contiguous())
        kv_nope = self.kv_norm(projected[:, Q_LORA_RANK : Q_LORA_RANK + KV_LORA_RANK].contiguous())
        k_rope = projected[:, Q_LORA_RANK + KV_LORA_RANK : base_end].contiguous()
        gate = projected[:, base_end : base_end + SPARSE_HEADS * V_HEAD_DIM] if self.fused_gate else None

        q_raw = self.q_b(q_resid).view(tokens, SPARSE_HEADS, QK_HEAD_DIM)
        q_nope, q_rope = q_raw.split((QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM), dim=-1)
        q_rope = self.rotary.apply(q_rope, cos, sin)
        k_rope = self.rotary.apply(k_rope, cos, sin)

        q_abs_nope = self.query_absorption(q_nope.transpose(0, 1).contiguous()).transpose(0, 1)
        q_absorbed = torch.cat((q_abs_nope, q_rope), dim=-1).contiguous()

        # 1. Sparse history attention over the reused target KV cache (no target mutation).
        out_hist, lse_hist = self.history_attention(
            q_absorbed, selected_slots, visible, selected_lengths, cache=self.cache
        )

        # 2. Exact FP32 causal self-attention over this layer's own [B, W] draft block.
        q_blk = q_absorbed.view(b, w, SPARSE_HEADS, SWA_RECORD_WIDTH).transpose(1, 2).float()  # [B, 32, W, 576]
        k_blk = torch.cat((kv_nope, k_rope), dim=-1).view(b, 1, w, SWA_RECORD_WIDTH).float()  # [B, 1, W, 576]
        v_blk = kv_nope.view(b, 1, w, KV_LORA_RANK).float()  # [B, 1, W, 512]
        scores_blk = torch.matmul(q_blk, k_blk.transpose(-2, -1)) * MLA_SCALE  # [B, 32, W, W]
        scores_blk = scores_blk.masked_fill(~causal_mask[None, None, :, :], float("-inf"))
        lse_blk = torch.logsumexp(scores_blk, dim=-1)  # [B, 32, W]
        probs_blk = torch.exp(scores_blk - lse_blk.unsqueeze(-1))
        out_blk = torch.matmul(probs_blk, v_blk).transpose(1, 2).reshape(tokens, SPARSE_HEADS, KV_LORA_RANK)
        lse_blk = lse_blk.transpose(1, 2).reshape(tokens, SPARSE_HEADS)

        # 3. Exact 3-way LogSumExp combine of target history, causal draft block, and learnable sink.
        attended = combine_three_way_lse(out_hist, lse_hist, out_blk, lse_blk, self.sink, selected_lengths)

        expanded = self.value_expansion(attended.transpose(0, 1).contiguous()).transpose(0, 1).reshape(tokens, -1)
        if gate is None:
            gate = self.gate_proj(normed)
        gated = (expanded * torch.sigmoid(gate)).contiguous()
        hidden = hidden + self.post_proj_norm(self.o_proj(gated))

        hidden = hidden + self.mlp(self.post_attn_norm(hidden))
        return hidden


class PreparedDFlyDraft:
    """5-layer DFly2 block proposal backbone + sequential hidden-correction logit head."""

    def __init__(
        self,
        weights,
        pool,
        *,
        batch: int,
        width: int,
        session,
        name: str,
        workspace,
        vocabulary_runtime=None,
        coordinator=None,
        max_cache_pages: int | None = None,
    ):
        assert weights.draft is not None and weights.config.draft is not None
        draft_cfg = weights.config.draft
        draft_w = weights.draft
        self.weights = weights
        self.pool = pool
        self.batch = batch
        self.width = width
        self.mask_token_id = draft_cfg.mask_token_id
        self.topk = draft_cfg.target_index_topk
        self.eps = float(weights.config.text["rms_norm_eps"])
        device = weights.embedding.device
        tokens = batch * width

        self.scratch = DFlyScratch.allocate(
            batch,
            width,
            draft_cfg.intermediate_size,
            draft_cfg.hidden_correction_intermediate_size,
            self.topk,
            device,
            shard=weights.shard,
        )
        self.rotary = InterleavedMRoPE(weights.config, device)
        self.causal_mask = torch.tril(torch.ones((width, width), dtype=torch.bool, device=device))

        self.block_embed = PreparedEmbedding(
            weights.embedding,
            torch.zeros(tokens, dtype=torch.int64, device=device),
            self.scratch.embedding,
            session=session,
            name=f"{name}.embed",
        ).functional(session=session, dynamic_inputs=True)
        self.step_embed = PreparedEmbedding(
            weights.embedding,
            torch.zeros(batch, dtype=torch.int64, device=device),
            self.scratch.step_hidden,
            session=session,
            name=f"{name}.step_embed",
        ).functional(session=session, dynamic_inputs=True)

        # Anchor slot-0 fusion: anchors + gate * eh_proj(cat(enorm(anchors), hnorm(target_hidden)))
        self.mtp_enorm = PreparedRMSNorm(
            self.scratch.step_hidden,
            draft_w.mtp_enorm,
            self.scratch.step_hidden,
            eps=self.eps,
            session=session,
            name=f"{name}.mtp_enorm",
        ).functional()
        self.mtp_hnorm = PreparedRMSNorm(
            self.scratch.step_hidden,
            draft_w.mtp_hnorm,
            self.scratch.step_hidden,
            eps=self.eps,
            session=session,
            name=f"{name}.mtp_hnorm",
        ).functional()
        self.mtp_eh_proj = prepare_projection(
            self.scratch.mtp_concat,
            draft_w.mtp_eh_proj,
            self.scratch.mtp_fused,
            session=session,
            name=f"{name}.mtp_eh_proj",
            workspace=workspace,
            activation_mode="auto",
        ).functional(session=session)
        self.mtp_gate = draft_w.mtp_gate

        self.layers = tuple(
            PreparedDFlyLayer(
                d,
                draft_cfg.target_kv_layers[d],
                draft_w.layers[d],
                self.scratch,
                pool,
                self.rotary,
                batch=batch,
                width=width,
                topk=self.topk,
                eps=self.eps,
                session=session,
                name=f"{name}.l{d}",
                workspace=workspace,
                max_cache_pages=max_cache_pages,
            )
            for d in range(draft_cfg.layers)
        )
        self.final_norm = PreparedRMSNorm(
            self.scratch.hidden,
            draft_w.final_norm,
            self.scratch.hidden,
            eps=self.eps,
            session=session,
            name=f"{name}.final_norm",
        ).functional()

        # Hidden correction: hidden + down(silu(gate(cat)) * up(cat))
        self.hc_hidden_norm = PreparedRMSNorm(
            self.scratch.step_hidden,
            draft_w.hc_hidden_norm,
            self.scratch.step_hidden,
            eps=self.eps,
            session=session,
            name=f"{name}.hc_hnorm",
        ).functional()
        self.hc_embed_norm = PreparedRMSNorm(
            self.scratch.step_hidden,
            draft_w.hc_embed_norm,
            self.scratch.step_hidden,
            eps=self.eps,
            session=session,
            name=f"{name}.hc_enorm",
        ).functional()
        self.hc_gate_up = prepare_projection(
            self.scratch.hc_concat,
            draft_w.hc_gate_up,
            self.scratch.hc_gate_up,
            session=session,
            name=f"{name}.hc_gate_up",
            workspace=workspace,
            activation_mode="auto",
        ).functional(session=session)
        self.hc_down = prepare_projection(
            self.scratch.hc_inter,
            draft_w.hc_down,
            self.scratch.hc_out,
            session=session,
            name=f"{name}.hc_down",
            workspace=workspace,
            activation_mode="auto",
        ).functional(session=session)

        draft_head = weights.head if draft_w.draft_head is None else draft_w.draft_head
        self.head = PreparedVocabularyProjection(
            self.scratch.step_hidden,
            draft_head,
            self.scratch.step_local_logits,
            self.scratch.step_logits,
            runtime=vocabulary_runtime,
            coordinator=coordinator,
            session=session,
            name=f"{name}.head",
            activation_mode="a16",
            workspace=workspace,
        ).functional(session=session)
        register_finalizer(session, self._release_examples)

    def _release_examples(self) -> None:
        self.scratch = None

    def rebind_pages(self) -> None:
        for layer in self.layers:
            layer.rebind_pages()

    def encode_block(
        self,
        anchor_tokens: torch.Tensor,
        before: torch.Tensor,
        rotary_offsets: torch.Tensor | None,
        state_rows: torch.Tensor,
        page_tables: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Run the 5 DFly2 layers over ``[anchor, MASK, ..., MASK]`` and return ``[B, W, 3072]``."""
        b, w = self.batch, self.width
        device = anchor_tokens.device

        # Gather target last_hidden and reused top-2048 logical token positions per draft layer,
        # resolving through each request's live page table so restored/offloaded rows stay valid.
        target_hidden = torch.zeros((b, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
        gather_rows(self.pool.last_hidden, state_rows, target_hidden)
        saved_logical = torch.full((b, len(self.layers), self.topk), -1, dtype=torch.int32, device=device)
        gather_rows(self.pool.region("dfly_target_indices"), state_rows, saved_logical)

        block_ids = torch.full((b, w), self.mask_token_id, dtype=torch.int64, device=device)
        block_ids[:, 0].copy_(anchor_tokens.clamp(0, VOCABULARY - 1))
        embeds = self.block_embed(token_ids=block_ids.reshape(-1)).view(b, w, HIDDEN_SIZE)

        # Fuse anchor embedding with target_hidden at slot 0.
        anchors = embeds[:, 0].contiguous()
        cat = torch.cat((self.mtp_enorm(anchors), self.mtp_hnorm(target_hidden)), dim=-1).contiguous()
        fused_anchor = anchors + self.mtp_gate * self.mtp_eh_proj(cat)
        embeds = torch.cat((fused_anchor.unsqueeze(1), embeds[:, 1:]), dim=1).reshape(b * w, HIDDEN_SIZE).contiguous()

        cols = torch.arange(w, dtype=torch.int64, device=device)[None, :]
        logical_pos = before[:, None] + cols  # [B, W]
        rope_pos = logical_pos if rotary_offsets is None else (logical_pos + rotary_offsets[:, None])
        rope_pos = torch.where(active[:, None], rope_pos, 0).reshape(-1)
        cos, sin = self.rotary.cos_sin(rope_pos, indexed=True, dtype=embeds.dtype)

        visible = torch.where(active[:, None], before[:, None].clamp_min(0), 0).expand(b, w).reshape(-1).to(torch.int32)
        hidden = embeds
        for d, layer in enumerate(self.layers):
            layer_slots = resolve_logical_slots(saved_logical[:, d, :], page_tables, active)
            sel_len = (layer_slots >= 0).sum(dim=-1, dtype=torch.int32)  # [B]
            slots_rep = layer_slots[:, None, :].expand(b, w, self.topk).reshape(b * w, self.topk).contiguous()
            len_rep = sel_len[:, None].expand(b, w).reshape(b * w).contiguous()
            hidden = layer(hidden, cos, sin, slots_rep, visible, len_rep, self.causal_mask)

        return self.final_norm(hidden).view(b, w, HIDDEN_SIZE)

    def step_logits(self, slot_hidden: torch.Tensor, prev_tokens: torch.Tensor) -> torch.Tensor:
        """Apply DFly2 hidden correction conditioned on ``prev_tokens`` and project through shared lm_head."""
        prev_embed = self.step_embed(token_ids=prev_tokens.clamp(0, VOCABULARY - 1))
        combined = torch.cat((self.hc_hidden_norm(slot_hidden), self.hc_embed_norm(prev_embed)), dim=-1).contiguous()
        gate, up = self.hc_gate_up(combined).chunk(2, dim=-1)
        corrected = slot_hidden + self.hc_down((F.silu(gate) * up).contiguous())
        return self.head(corrected)


class PreparedDFlySpeculativeRound:
    """Captured DFly2 speculative round: propose K drafts, verify with target, commit accepted prefix."""

    def __init__(
        self,
        owner,
        batch: int,
        width: int,
        name: str,
    ):
        self.owner = owner
        self.batch = batch
        self.width = width
        self.name = name
        o, b, w = owner, batch, width
        device = o.pool.backing.device
        blocks = o.inputs.page_tables.shape[1]

        self.metadata = Metadata.allocate(b * w, b, blocks, device, samples=b * w, max_query_tokens=w)
        self.validator = prepare_batch(
            o.inputs,
            o.requests,
            Metadata.allocate(b, b, blocks, device),
            mode="decode",
            shard=o.weights.shard,
            pool=o.pool,
            verification_width=w,
            target_token_capacity=b * w,
            page_tokens=o.page_tokens,
        )
        self.validation = PreparedAllReduce(
            self.validator.validation_input,
            self.validator.validation_sum,
            runtime=o.runtime,
            coordinator=o.coordinator,
            session=o.session,
            name=f"{name}.validation",
        )
        self.decoder = o.decoder_factory(
            o.weights,
            o.tables,
            o.scratch.rows(b * w, b, b * w, w),
            self.metadata,
            o.pool,
            mode="decode" if w == 1 else "verify",
            runtime=o.runtime,
            vocabulary_runtime=o.vocabulary_runtime,
            coordinator=o.coordinator,
            session=o.session,
            name=f"{name}.target",
            workspace=o.workspace,
            max_cache_pages=o.max_cache_pages,
            examples=o.examples,
        )
        self.draft = (
            None
            if w == 1
            else PreparedDFlyDraft(
                o.weights,
                o.pool,
                batch=b,
                width=w,
                session=o.session,
                name=f"{name}.dfly",
                workspace=o.workspace,
                vocabulary_runtime=o.vocabulary_runtime,
                coordinator=o.coordinator,
                max_cache_pages=o.max_cache_pages,
            )
        )

        self.candidates = torch.zeros((b, w), dtype=torch.int64, device=device)
        self.drafts = torch.zeros((b, w - 1), dtype=torch.int64, device=device)
        self.probabilities = torch.zeros((b, w - 1, VOCABULARY), dtype=torch.float32, device=device)
        self.valid = torch.zeros((b, w - 1), dtype=torch.bool, device=device)
        self.step_logits_buf = torch.zeros((b, 1, VOCABULARY), dtype=torch.float32, device=device)
        self.seed = torch.zeros(b, dtype=torch.int64, device=device)
        self.offset = torch.zeros(b, dtype=torch.int64, device=device)
        self.sampled = torch.zeros(b, dtype=torch.int64, device=device)
        self.operations = None

    def rebind_pages(self) -> None:
        self.decoder.rebind_pages()
        if self.draft is not None:
            self.draft.rebind_pages()

    def prepare(self) -> None:
        if self.operations is not None:
            return
        o, b, w = self.owner, self.batch, self.width
        args = (o.requests, o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows)

        target = PreparedSpeculativeDistributions(
            self.decoder.scratch.logits.view(b, w, -1),
            *args,
            self.drafts,
            workspace=None,
        )
        target.prepare()

        proposals = tuple(
            PreparedDraftDistribution(
                self.step_logits_buf,
                *args,
                self.drafts,
                prefix_offset=i,
                seed=self.seed,
                offset=self.offset,
                sampled=self.sampled,
                workspace=None,
            )
            for i in range(w - 1)
        )
        for proposal in proposals:
            proposal.prepare()

        verify = PreparedStochasticVerification(
            self.probabilities,
            self.drafts,
            target.probabilities,
            *args,
            rng=o.rng,
            draft_valid=self.valid,
            target_valid=target.valid,
            token_limits=o.inputs.query_lengths[:b],
            workspace=None,
        )
        verify.prepare()

        commit = PreparedSpeculativeCommit(
            o.requests,
            verify.output,
            self.candidates,
            o.inputs.row_slot_ids[:b],
            o.inputs.row_generations[:b],
            o.inputs.live_rows,
            o.completion,
            o.frame,
            o.sequence,
            logits=self.decoder.scratch.logits.view(b, w, -1),
            rng=o.rng,
            draft_context_lag=0,
            token_limits=o.inputs.query_lengths[:b],
        )
        commit.prepare()
        if commit.logprobs is not None:
            commit.logprobs.release_logits()

        self.decoder.release_outputs()
        self.validator.functional()
        self.validation.functional(session=o.session)
        target.functional(session=o.session)
        for proposal in proposals:
            proposal.functional(session=o.session)
        verify.functional(session=o.session)
        commit.functional(session=o.session)
        self.operations = (target, proposals, verify, commit)

        self.metadata = self.candidates = self.drafts = self.probabilities = self.valid = None
        self.step_logits_buf = self.seed = self.offset = self.sampled = None

    def __call__(self) -> None:
        o, b, w = self.owner, self.batch, self.width
        target, proposals, verify, commit = self.operations
        device = o.requests.histogram.device

        val_in, present = self.validator.validate()
        validated = self.validator.begin(self.validation(val_in), present=present)
        active = validated.lengths > 0
        rotary_offsets = row_rotary_offsets(validated)
        slots = o.inputs.row_slot_ids[:b].long()
        safe = slots.clamp(0, o.requests.capacity - 1)
        before = o.requests.consumed_tokens[safe]
        limits = torch.where(active, o.inputs.query_lengths[:b], 0)
        states = o.inputs.request_state_rows[safe, 0]

        candidates = torch.zeros((b, w), dtype=torch.int64, device=device)
        drafts = torch.zeros((b, w - 1), dtype=torch.int64, device=device)
        probabilities = torch.zeros((b, w - 1, VOCABULARY), dtype=torch.float32, device=device)
        valid = torch.zeros((b, w - 1), dtype=torch.bool, device=device)
        candidates[:, 0].copy_(o.requests.pending_token[safe])

        if self.draft is not None:
            base_offset = torch.where(o.generation[safe] == o.requests.generation[safe], o.offset[safe], 0)
            seed = o.requests.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN
            block_hidden = self.draft.encode_block(
                candidates[:, 0], before, rotary_offsets, states, o.inputs.page_tables[safe], active
            )
            prev_tokens = torch.where(active, candidates[:, 0], 0)
            sampled = torch.zeros(b, dtype=torch.int64, device=device)
            for step, sampler in enumerate(proposals):
                step_logits = self.draft.step_logits(block_hidden[:, step + 1].contiguous(), prev_tokens)
                scratch = {
                    name: torch.empty(shape, dtype=dtype, device=device)
                    for name, (shape, dtype) in sampler.layout.items()
                }
                step_probs = sampler.run(
                    scratch,
                    step_logits.float().unsqueeze(1).contiguous(),
                    candidates=drafts,
                    seed=seed,
                    offset=base_offset + 4 * step,
                    sampled=sampled,
                )
                drafts[:, step].copy_(sampled)
                probabilities[:, step].copy_(step_probs[:, 0])
                valid[:, step].copy_(scratch["valid"][:, 0] & active & (step < limits - 1))
                prev_tokens = sampled.clone()
            candidates[:, 1:].copy_(drafts)

            slot_ids = torch.arange(o.requests.capacity, device=device)
            matches = active[None, :] & (slots[None, :] == slot_ids[:, None])
            base = torch.where(matches, base_offset[None, :], 0).sum(1)
            o.offset.copy_(torch.where(matches.any(1), base + 4 * (w - 1), o.offset))
            o.generation.copy_(torch.where(matches.any(1), o.requests.generation, o.generation))

        metadata = Metadata.pack(
            before,
            limits,
            states,
            o.inputs.page_tables[safe],
            candidates,
            o.inputs.live_rows,
            token_capacity=b * w,
            sample_count=b * w,
            max_query_tokens=w,
            rotary_offset=rotary_offsets,
        )
        hidden, logits, _, records = self.decoder(metadata)
        target_scratch = {
            name: torch.empty(shape, dtype=dtype, device=device)
            for name, (shape, dtype) in target.layout.items()
        }
        verify_scratch = {
            name: torch.empty(shape, dtype=dtype, device=device)
            for name, (shape, dtype) in verify.layout.items()
        }
        target.run(target_scratch, logits.view(b, w, -1), candidates=drafts)
        acceptance = verify.run(
            verify_scratch,
            probabilities,
            target_scratch["probabilities"],
            target_scratch["valid"],
            disjoint=True,
            drafts=drafts,
            draft_valid=valid,
        )
        committed = commit(logits.view(b, w, -1), acceptance=acceptance, inputs=candidates)
        counts = committed.emitted_count
        if w > 1:
            self.decoder.commit(counts, records, metadata)
        row_ids = torch.arange(b, device=device)
        accepted_hidden = hidden.view(b, w, HIDDEN_SIZE)[row_ids, (counts - 1).long().clamp_min(0)]
        save_hidden(accepted_hidden, o.pool.last_hidden, metadata.state_rows, counts > 0)
