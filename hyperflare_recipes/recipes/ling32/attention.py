# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared Ling-3.2-Air Gated MLA over BF16 paged latent records."""

import torch
from b12x.attention import dense_mla

from hyperflare_recipes.components.dense_mla import PreparedDenseMLA
from hyperflare_recipes.components.head_gate import PreparedHeadGate
from hyperflare_recipes.components.linear import BatchedUnquantizedProjection, prepare_projection
from hyperflare_recipes.components.rms_norm import PreparedRMSNorm
from ignition.components.state import scatter_rows
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.tuning import register_finalizer

from .config import PAGE_TOKENS, RECORD_DIM
from .weights import MLAWeights


def latent_token_slots(positions: torch.Tensor, token_page_table: torch.Tensor) -> torch.Tensor:
    """Map per-token positions and page tables to flat BF16 cache slots (-1 when inactive)."""
    clamped = positions.clamp_min(0)
    col = (clamped // PAGE_TOKENS).clamp(0, token_page_table.shape[1] - 1)
    page = token_page_table.gather(1, col[:, None].long()).squeeze(1).long()
    live = (positions >= 0) & (page >= 0)
    return torch.where(live, page * PAGE_TOKENS + (clamped % PAGE_TOKENS), -1)


class PreparedGatedMLA:
    """Gated MLA with interleaved RoPE, BF16 paged latent cache, and head-wise sigmoid gate."""

    def __init__(
        self,
        weights: MLAWeights,
        source: torch.Tensor,
        metadata,
        cache: torch.Tensor,
        rotary,
        *,
        eps: float,
        mode: str,
        session,
        name: str,
        workspace,
        buffers: dict,
        cache_only: bool = False,
    ):
        tokens, hidden = source.shape
        heads = weights.key.shape[0]
        device = source.device
        self.tokens, self.heads = tokens, heads
        self.cache = cache
        self.rotary = rotary
        self.cache_only = bool(cache_only)

        def buf(key: str, *shape, dtype=torch.bfloat16):
            role = "ling32.hidden_output" if key == "output" else "mla." + key
            return declaration_tensor(buffers, role, (tokens, *shape), dtype, device)

        projected = buf("projected", heads * 192 + RECORD_DIM)
        kv_norm_out = buf("kv_norm_out", 512)
        self.q_kv_a = prepare_projection(
            source, weights.q_kv_a, projected, session=session, name=name + ".q_kv_a", workspace=workspace
        ).functional(session=session)
        self.kv_norm = PreparedRMSNorm(
            projected[:, heads * 192 : heads * 192 + 512],
            weights.kv_norm,
            kv_norm_out,
            eps=eps,
            session=session,
            name=name + ".kv_norm",
            zero_centered=False,
        ).functional()

        if cache_only:
            self.g_proj = self.absorb = self.attention = self.expand = self.head_gate = self.output = None
            return

        gate = buf("gate", heads)
        absorbed = buf("absorbed", heads, RECORD_DIM)
        attended = buf("attended", heads, 512)
        expanded = buf("expanded", heads, 128)
        q_view = projected[:, : heads * 192].view(tokens, heads, 192)

        self.g_proj = prepare_projection(
            source, weights.g_proj, gate, session=session, name=name + ".g_proj", workspace=workspace
        ).functional(session=session)
        self.absorb = BatchedUnquantizedProjection(
            q_view[:, :, :128].transpose(0, 1),
            weights.key,
            absorbed[:, :, :512].transpose(0, 1),
            major="n",
            session=session,
            name=name + ".absorb",
        ).functional(session=session)

        width = metadata.page_table.shape[1]
        self._lengths_example = torch.zeros_like(metadata.lengths)
        # Cap split-KV partial_o scratch at 512 MiB (partial_rows = max_total_q * max_splits,
        # each row holding heads * 512 FP32 elements = 64 KiB at 32 heads -> 8,192 rows).
        # Decode and verify remain unconstrained (128..2048 splits) while 4096-token extend stays
        # well below 2^31 partial_o elements and preserves VRAM for KV pages.
        max_partial_rows = max(tokens, (512 << 20) // (heads * 512 * 4))
        budget = dense_mla.Budget(max_partial_rows=max_partial_rows)
        caps = dense_mla.Caps(
            device=device,
            mode=mode if mode in ("decode", "verify") else "extend",
            q_dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            num_q_heads=heads,
            max_total_q=tokens,
            max_batch=metadata.lengths.numel(),
            max_cache_tokens=width * PAGE_TOKENS,
            max_page_table_width=width,
            num_cache_pages=cache.shape[0],
            page_size=PAGE_TOKENS,
            use_cuda_graph=True,
            partial_dtype=torch.float32,
            budget=budget,
        )
        self.attention = PreparedDenseMLA(
            absorbed,
            cache,
            attended,
            metadata.page_table,
            self._lengths_example,
            metadata.query_start_loc,
            caps=caps,
            scale=192.0 ** -0.5,
            session=session,
            name=name + ".attention",
            workspace=workspace,
        ).functional(session=session)
        self.expand = BatchedUnquantizedProjection(
            attended.transpose(0, 1),
            weights.value,
            expanded.transpose(0, 1),
            major="n",
            session=session,
            name=name + ".expand",
        ).functional(session=session)
        self.head_gate = PreparedHeadGate(
            expanded,
            gate,
            session=session,
            name=name + ".head_gate",
        ).functional(session=session)
        self.output = prepare_projection(
            expanded.view(tokens, heads * 128),
            weights.output,
            buf("output", hidden),
            session=session,
            name=name + ".dense",
            workspace=workspace,
        ).functional(session=session)
        register_finalizer(session, self._release_examples)

    def _release_examples(self):
        self._lengths_example = None

    def __call__(self, source: torch.Tensor, metadata, angles: torch.Tensor):
        tokens, heads = source.shape[0], self.heads
        projected = self.q_kv_a(source)
        q_raw = projected[:, : heads * 192].view(tokens, heads, 192)
        kv_a = projected[:, heads * 192 :]
        latent = torch.empty((tokens, RECORD_DIM), dtype=source.dtype, device=source.device)
        latent[:, :512].copy_(self.kv_norm(kv_a[:, :512]))
        latent[:, 512:].copy_(kv_a[:, 512:])
        self.rotary.apply(latent, angles)
        slots = latent_token_slots(metadata.positions, metadata.token_page_table)
        scatter_rows(latent, slots, self.cache.view(-1, RECORD_DIM), slots >= 0)
        if self.cache_only:
            return None

        gate = self.g_proj(source)
        self.rotary.apply(q_raw, angles)
        absorbed = torch.empty((tokens, heads, RECORD_DIM), dtype=source.dtype, device=source.device)
        self.absorb(q_raw[:, :, :128].transpose(0, 1), absorbed[:, :, :512].transpose(0, 1))
        absorbed[:, :, 512:].copy_(q_raw[:, :, 128:])
        lengths = torch.where(metadata.lengths > 0, metadata.starts + metadata.lengths, 0).to(torch.int32)
        attended = self.attention(
            absorbed,
            page_table=metadata.page_table,
            lengths=lengths,
            starts=metadata.query_start_loc,
        )
        expanded = torch.empty((tokens, heads, 128), dtype=source.dtype, device=source.device)
        self.expand(attended.transpose(0, 1), expanded.transpose(0, 1))
        self.head_gate(expanded, gate)
        return self.output(expanded.reshape(tokens, heads * 128))
