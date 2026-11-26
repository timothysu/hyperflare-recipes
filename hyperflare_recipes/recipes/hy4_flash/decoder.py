# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""42-layer Hy4-flash decoder with 4-stream iHC, Engram memory, hybrid MLA, and LatentMoE."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hyperflare_recipes.components.embedding import PreparedEmbedding
from hyperflare_recipes.components.linear import UnquantizedProjection
from hyperflare_recipes.components.paged_batch import rotary_positions
from hyperflare_recipes.components.ring_commit import PreparedRingCommit
from hyperflare_recipes.components.rms_norm import PreparedRMSNorm
from ignition.components.state import save_hidden
from hyperflare_recipes.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation.tuning import register_finalizer

from .attention import (
    AttentionScratch,
    PreparedSlidingAttention,
    PreparedSparseAttention,
    prepare_sparse_slots,
    prepare_swa_plan,
    swa_stage_tokens,
)
from .config import HC_MULT, HIDDEN_SIZE, INDEX_TOPK, SWA_WINDOW, VOCABULARY
from .engram import EngramScratch, PreparedEngram, PreparedTableRequests
from .ihc import ihc_head, ihc_post, ihc_pre
from .indexer import expand_pooled_topk_to_logical_tokens
from .moe import MoEScratch, PreparedDenseMLP, PreparedLatentMoE
from .pool import SWA_RECORD_WIDTH
from .rotary import InterleavedMRoPE


@dataclass(frozen=True)
class Scratch:
    """BF16 decoder activations and FP32 target vocabulary logits."""

    logits_dtype = torch.float32

    embedding: torch.Tensor
    streams: torch.Tensor
    hc_mixes: torch.Tensor
    head_streams: torch.Tensor
    head_hc_mixes: torch.Tensor
    branch_in: torch.Tensor
    normalized: torch.Tensor
    engram: EngramScratch
    attention: AttentionScratch
    moe: MoEScratch
    collapsed: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor
    ring_staging: torch.Tensor | None = None

    @classmethod
    def allocate(
        cls,
        tokens: int,
        rows: int,
        samples: int,
        max_query_tokens: int,
        device: torch.device,
        *,
        shard,
        ring_layers: int = 0,
        logits_dtype: torch.dtype | None = None,
    ) -> Scratch:
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)

        resolved_logits_dtype = cls.logits_dtype if logits_dtype is None else logits_dtype
        _, local_vocab = shard.padded_partition(VOCABULARY, alignment=64)
        ring_staging = (
            zeros(
                ring_layers,
                rows,
                (SWA_WINDOW + max_query_tokens) * SWA_RECORD_WIDTH * 2,
                dtype=torch.uint8,
            )
            if ring_layers > 0
            else None
        )
        return cls(
            embedding=zeros(tokens, HIDDEN_SIZE),
            streams=zeros(tokens, HC_MULT, HIDDEN_SIZE),
            hc_mixes=zeros(tokens, 2 * HC_MULT, dtype=torch.float32),
            head_streams=zeros(samples, HC_MULT, HIDDEN_SIZE),
            head_hc_mixes=zeros(samples, HC_MULT, dtype=torch.float32),
            branch_in=zeros(tokens, HIDDEN_SIZE),
            normalized=zeros(tokens, HIDDEN_SIZE),
            engram=EngramScratch.allocate(tokens, device),
            attention=AttentionScratch.allocate(tokens, rows, max_query_tokens, device),
            moe=MoEScratch.allocate(tokens, device),
            collapsed=zeros(samples, HIDDEN_SIZE),
            hidden=zeros(samples, HIDDEN_SIZE),
            local_logits=zeros(samples, local_vocab, dtype=resolved_logits_dtype),
            logits=zeros(samples, VOCABULARY, dtype=resolved_logits_dtype),
            ring_staging=ring_staging,
        )

    def rows(self, tokens: int, rows: int, samples: int, max_query_tokens: int) -> Scratch:
        return Scratch(
            embedding=self.embedding[:tokens],
            streams=self.streams[:tokens],
            hc_mixes=self.hc_mixes[:tokens],
            head_streams=self.head_streams[:samples],
            head_hc_mixes=self.head_hc_mixes[:samples],
            branch_in=self.branch_in[:tokens],
            normalized=self.normalized[:tokens],
            engram=self.engram.rows(tokens),
            attention=self.attention.rows(tokens, rows, max_query_tokens),
            moe=self.moe.rows(tokens),
            collapsed=self.collapsed[:samples],
            hidden=self.hidden[:samples],
            local_logits=self.local_logits[:samples],
            logits=self.logits[:samples],
            ring_staging=self.ring_staging,
        )

    def ring_staging_view(self, regions: int, rows: int, max_query_tokens: int) -> torch.Tensor | None:
        if self.ring_staging is None:
            return None
        width = (SWA_WINDOW + max_query_tokens) * SWA_RECORD_WIDTH * 2
        needed = regions * rows * width
        if needed > self.ring_staging.numel():
            raise ValueError(
                f"SWA ring staging requires {needed} bytes across {regions}x{rows}, "
                f"exceeding allocated {self.ring_staging.numel()} bytes"
            )
        return self.ring_staging.view(-1)[:needed].view(regions, rows, width)


class PreparedLayer:
    """One Hy4-flash decoder layer: optional Engram, iHC-wrapped hybrid MLA, iHC-wrapped FFN/LatentMoE."""

    def __init__(
        self,
        index: int,
        weights,
        tables,
        scratch: Scratch,
        metadata,
        pool,
        rotary: InterleavedMRoPE,
        *,
        mode: str,
        session,
        name: str,
        workspace,
        coordinator=None,
        max_cache_pages: int | None = None,
        examples: dict | None = None,
    ):
        config = pool.config
        text = config.text
        self.index = index
        self.eps = float(text["rms_norm_eps"])
        self.hc_eps = float(text["hc_eps"])
        self.hc_magnitude = float(text["hc_magnitude"])
        self.residual = weights.residual

        self.engram = None
        if weights.engram is not None:
            self.engram = PreparedEngram(
                index,
                weights.engram,
                tables,
                scratch.streams,
                metadata,
                scratch=scratch.engram,
                session=session,
                name=f"{name}.engram",
                workspace=workspace,
                coordinator=coordinator,
            )

        tokens = scratch.streams.shape[0]
        self.attn_hc_proj = UnquantizedProjection(
            scratch.streams.view(tokens, -1),
            weights.residual["attn_fn"],
            scratch.hc_mixes,
            session=session,
            name=f"{name}.hc_attn",
        ).functional(session=session)
        self.mlp_hc_proj = UnquantizedProjection(
            scratch.streams.view(tokens, -1),
            weights.residual["mlp_fn"],
            scratch.hc_mixes,
            session=session,
            name=f"{name}.hc_mlp",
        ).functional(session=session)

        self.input_norm = PreparedRMSNorm(
            scratch.branch_in,
            weights.residual["input_norm"],
            scratch.normalized,
            eps=self.eps,
            session=session,
            name=f"{name}.input_norm",
        ).functional()
        if config.is_indexed(index):
            self.attention = PreparedSparseAttention(
                index,
                weights.attention,
                weights.indexer,
                scratch.normalized,
                scratch.attention,
                metadata,
                pool,
                rotary,
                mode=mode,
                eps=self.eps,
                session=session,
                name=f"{name}.attention",
                workspace=workspace,
                max_cache_pages=max_cache_pages,
                examples=examples,
            )
        else:
            self.attention = PreparedSlidingAttention(
                index,
                weights.attention,
                scratch.normalized,
                scratch.attention,
                metadata,
                pool,
                rotary,
                mode=mode,
                eps=self.eps,
                session=session,
                name=f"{name}.attention",
                workspace=workspace,
            )

        self.post_attn_norm = PreparedRMSNorm(
            scratch.branch_in,
            weights.residual["post_attn_norm"],
            scratch.normalized,
            eps=self.eps,
            session=session,
            name=f"{name}.post_attn_norm",
        ).functional()
        if config.is_sparse_mlp(index):
            self.mlp = PreparedLatentMoE(
                weights.mlp,
                scratch.normalized,
                scratch.moe,
                config=config,
                mode=mode,
                session=session,
                name=f"{name}.moe",
                workspace=workspace,
            )
        else:
            self.mlp = PreparedDenseMLP(
                weights.mlp,
                scratch.normalized,
                scratch.moe,
                eps=self.eps,
                session=session,
                name=f"{name}.mlp",
                workspace=workspace,
            )

    def rebind_pages(self) -> None:
        if isinstance(self.attention, PreparedSparseAttention):
            self.attention.rebind_pages()

    def __call__(
        self,
        streams: torch.Tensor,
        metadata,
        *,
        cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        sparse_slots: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        swa_plan: tuple | None = None,
        window_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, object, torch.Tensor | None]:
        if self.engram is not None:
            streams = self.engram(streams, metadata)

        r = self.residual
        tokens = streams.shape[0]
        attn_mixes = self.attn_hc_proj(streams.view(tokens, -1))
        attn_post, attn_in = ihc_pre(
            streams,
            r["attn_fn"],
            r["attn_base"],
            r["attn_scale"],
            rms_eps=self.eps,
            hc_eps=self.hc_eps,
            hc_magnitude=self.hc_magnitude,
            mixes=attn_mixes,
        )
        if isinstance(self.attention, PreparedSparseAttention):
            attn_out, attn_record, physical = self.attention(
                self.input_norm(attn_in),
                metadata,
                cos_sin=cos_sin,
                sparse_slots=sparse_slots,
            )
        else:
            attn_out, attn_record, physical = self.attention(
                self.input_norm(attn_in),
                metadata,
                cos_sin=cos_sin,
                swa_plan=swa_plan,
                window_cache=window_cache,
            )
        streams = ihc_post(streams, attn_out, attn_post)

        mlp_mixes = self.mlp_hc_proj(streams.view(tokens, -1))
        mlp_post, mlp_in = ihc_pre(
            streams,
            r["mlp_fn"],
            r["mlp_base"],
            r["mlp_scale"],
            rms_eps=self.eps,
            hc_eps=self.hc_eps,
            hc_magnitude=self.hc_magnitude,
            mixes=mlp_mixes,
        )
        mlp_out = self.mlp(self.post_attn_norm(mlp_in))
        streams = ihc_post(streams, mlp_out, mlp_post)
        return streams, attn_record, physical


class PreparedDecoder:
    """Full 42-layer Hy4-flash decoder with Engram mailboxes, vision injection, and DFly2 reuse capture."""

    def __init__(
        self,
        weights,
        tables,
        scratch: Scratch,
        metadata,
        pool,
        *,
        mode: str,
        runtime=None,
        vocabulary_runtime=None,
        coordinator=None,
        session,
        name: str,
        workspace,
        vision=None,
        scoring=None,
        max_cache_pages: int | None = None,
        examples: dict | None = None,
    ):
        self.weights = weights
        self.auxiliary = None  # Required by PreparedGeneration context_factory protocol.
        self.scratch = scratch
        self.metadata = metadata
        self.pool = pool
        self.mode = mode
        self.vision = vision if mode == "prefill" else None
        text = weights.config.text
        self.eps = float(text["rms_norm_eps"])
        self.hc_eps = float(text["hc_eps"])

        self.table_requests = PreparedTableRequests(tables, metadata, pool)
        self.rotary = InterleavedMRoPE(weights.config, scratch.embedding.device)
        self.embedding = PreparedEmbedding(
            weights.embedding,
            metadata.token_ids,
            scratch.embedding,
            num_rows=metadata.live_counts[:1],
            session=session,
            name=f"{name}.embedding",
        ).functional(session=session, dynamic_inputs=True)

        self.dfly_index_layers: tuple[int, ...] = (
            weights.config.draft.target_index_layers
            if weights.draft is not None and weights.config.draft is not None
            else ()
        )
        self.dfly_topk = (
            weights.config.draft.target_index_topk
            if weights.draft is not None and weights.config.draft is not None
            else INDEX_TOPK
        )

        self.swa_indices: dict[int, int] = {
            layer_idx: pos for pos, layer_idx in enumerate(weights.config.swa_layers)
        }
        self.swa_stage_len = swa_stage_tokens(metadata.max_query_tokens)
        self.swa_max_query_tokens = metadata.max_query_tokens
        self.window_commit = None
        self.staging_shape = None
        if mode in ("decode", "verify") and self.swa_indices:
            self.window_commit = PreparedRingCommit(
                pool.reserved_backing,
                [pool.layout.region(f"layer{layer_idx}.window").offset for layer_idx in weights.config.swa_layers],
                metadata.state_rows,
                metadata.starts,
                token_capacity=metadata.max_query_tokens,
                window_tokens=SWA_WINDOW,
                record_bytes=SWA_RECORD_WIDTH * 2,
                staging=scratch.ring_staging_view(
                    len(weights.config.swa_layers),
                    metadata.state_rows.numel(),
                    metadata.max_query_tokens,
                ),
            )
            self.staging_shape = tuple(self.window_commit.staging.shape)
            self.window_commit.functional()

        self.layers = tuple(
            PreparedLayer(
                index,
                layer_weights,
                tables,
                scratch,
                metadata,
                pool,
                self.rotary,
                mode=mode,
                session=session,
                name=f"{name}.layer{index}",
                workspace=workspace,
                coordinator=coordinator,
                max_cache_pages=max_cache_pages,
                examples=examples,
            )
            for index, layer_weights in enumerate(weights.layers)
        )

        if scoring is not None:
            head_in = scratch.streams.view(scratch.streams.shape[0], -1)
            head_out = torch.zeros((scratch.streams.shape[0], HC_MULT), dtype=torch.float32, device=head_in.device)
        else:
            head_in = scratch.head_streams.view(scratch.head_streams.shape[0], -1)
            head_out = scratch.head_hc_mixes
        self.head_hc_proj = UnquantizedProjection(
            head_in,
            weights.hc_head["fn"],
            head_out,
            session=session,
            name=f"{name}.hc_head",
        ).functional(session=session)

        self.final_norm = PreparedRMSNorm(
            scratch.collapsed,
            weights.norm,
            scratch.hidden,
            eps=self.eps,
            session=session,
            name=f"{name}.final_norm",
        ).functional()
        self.scoring_norm = (
            None
            if scoring is None
            else PreparedRMSNorm(
                scratch.embedding,
                weights.norm,
                scratch.embedding,
                eps=self.eps,
                session=session,
                name=f"{name}.scoring_norm",
            ).functional()
        )
        self.scoring = (
            None
            if scoring is None
            else scoring.prepare(
                scratch.embedding,
                weights.head,
                vocabulary=VOCABULARY,
                session=session,
                coordinator=coordinator,
                name=f"{name}.prompt",
                logits_dtype=scratch.logits.dtype,
                activation_mode="a16" if not isinstance(weights.head, torch.Tensor) else "auto",
            )
        )
        self.head = PreparedVocabularyProjection(
            scratch.hidden,
            weights.head,
            scratch.local_logits,
            scratch.logits,
            runtime=vocabulary_runtime,
            coordinator=coordinator,
            session=session,
            name=f"{name}.head",
            workspace=workspace,
        ).functional(session=session)
        register_finalizer(session, self.release_outputs)

    def release_outputs(self) -> None:
        self.scratch = self.metadata = None

    def rebind_pages(self) -> None:
        for layer in self.layers:
            layer.rebind_pages()

    def _save_dfly_indices(
        self,
        counts: torch.Tensor,
        dfly_pools: dict[int, torch.Tensor],
        metadata,
    ) -> None:
        if not self.dfly_index_layers or not dfly_pools:
            return
        m = metadata
        rows = m.starts.numel()
        active = counts > 0
        last_tok = (m.query_start_loc[:-1].long() + counts.long() - 1).clamp(0, m.positions.numel() - 1)
        last_pos = torch.where(active, m.positions[last_tok], -1)
        stacked = torch.stack(
            [
                expand_pooled_topk_to_logical_tokens(
                    dfly_pools[layer][last_tok],
                    last_pos,
                    width=self.dfly_topk,
                )
                for layer in self.dfly_index_layers
            ],
            dim=1,
        )  # [R, 5, 2048] int32 logical token positions
        region = self.pool.region("dfly_target_indices")
        save_hidden(
            stacked.reshape(rows, -1),
            region.reshape(region.shape[0], -1),
            m.state_rows,
            active,
        )

    def __call__(self, metadata):
        m = metadata
        raw_history = self.table_requests(m)
        tokens = m.token_ids.clamp(0, self.weights.embedding.shape[0] - 1)
        embedding = self.embedding(token_ids=tokens, num_rows=m.live_counts[:1])
        if self.vision is not None:
            self.vision.inject(m, embedding)

        # Precompute shared rotary tables, sparse physical slots, and SWA staging indices once per forward.
        rope_pos = rotary_positions(m)
        cos_sin_idx = self.rotary.cos_sin(rope_pos, indexed=True, dtype=embedding.dtype)
        cos_sin_swa = self.rotary.cos_sin(rope_pos, indexed=False, dtype=embedding.dtype)
        sparse_slots = prepare_sparse_slots(m)
        swa_plan = prepare_swa_plan(m, self.swa_stage_len, self.swa_max_query_tokens) if self.swa_indices else None
        ring_staging = (
            None
            if self.staging_shape is None
            else torch.empty(self.staging_shape, dtype=torch.uint8, device=embedding.device)
        )

        streams = embedding[:, None, :].expand(-1, HC_MULT, -1).contiguous()
        layer_records = []
        dfly_pools: dict[int, torch.Tensor] = {}
        for index, layer in enumerate(self.layers):
            swa_pos = self.swa_indices.get(index)
            if swa_pos is None:
                streams, record, pools = layer(
                    streams,
                    m,
                    cos_sin=cos_sin_idx,
                    sparse_slots=sparse_slots,
                )
            else:
                streams, record, pools = layer(
                    streams,
                    m,
                    cos_sin=cos_sin_swa,
                    swa_plan=swa_plan,
                    window_cache=None if ring_staging is None else ring_staging[swa_pos],
                )
            layer_records.append(record)
            if pools is not None and index in self.dfly_index_layers:
                dfly_pools[index] = pools

        samples = m.sample_indices
        hc_head_w = self.weights.hc_head
        if self.scoring is not None:
            all_mixes = self.head_hc_proj(streams.view(streams.shape[0], -1))
            all_collapsed = ihc_head(
                streams,
                hc_head_w["fn"],
                hc_head_w["base"],
                hc_head_w["scale"],
                rms_eps=self.eps,
                hc_eps=self.hc_eps,
                mixes=all_mixes,
            )
            all_normed = self.scoring_norm(all_collapsed)
            self.scoring(all_normed)
            hidden = all_normed[samples].contiguous()
        else:
            sampled_streams = streams[samples].contiguous()
            head_mixes = self.head_hc_proj(sampled_streams.view(sampled_streams.shape[0], -1))
            collapsed = ihc_head(
                sampled_streams,
                hc_head_w["fn"],
                hc_head_w["base"],
                hc_head_w["scale"],
                rms_eps=self.eps,
                hc_eps=self.hc_eps,
                mixes=head_mixes,
            )
            hidden = self.final_norm(collapsed)

        logits = self.head(hidden)
        if self.mode != "verify":
            if self.window_commit is not None and ring_staging is not None:
                self.window_commit(m.lengths, staging=ring_staging, rows=m.state_rows, starts=m.starts)
            self.table_requests.commit(m.lengths, raw_history, m)
            self._save_dfly_indices(m.lengths, dfly_pools, m)
        return hidden, logits, None, (ring_staging, tuple(layer_records), raw_history, dfly_pools)

    def commit(self, counts: torch.Tensor, records, metadata) -> None:
        ring_staging, layer_records, raw_history, dfly_pools = records
        if self.window_commit is not None and ring_staging is not None:
            self.window_commit(counts, staging=ring_staging, rows=metadata.state_rows, starts=metadata.starts)
        for layer, record in zip(self.layers, layer_records, strict=True):
            layer.attention.commit(counts, record, metadata)
        self.table_requests.commit(counts, raw_history, metadata)
        self._save_dfly_indices(counts, dfly_pools, metadata)
