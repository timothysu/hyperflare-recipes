# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""42-layer Hy4-flash decoder with 4-stream iHC, Engram memory, hybrid MLA, and LatentMoE."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hyperflare_recipes.components.embedding import PreparedEmbedding
from hyperflare_recipes.components.ihc import PreparedIHC, post_norm
from hyperflare_recipes.components.paged_batch import rotary_positions
from hyperflare_recipes.components.ring_commit import PreparedRingCommit
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
    normalized: torch.Tensor
    engram: EngramScratch
    attention: AttentionScratch
    moe: MoEScratch
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
            normalized=zeros(tokens, HIDDEN_SIZE),
            engram=EngramScratch.allocate(tokens, device),
            attention=AttentionScratch.allocate(tokens, rows, max_query_tokens, device),
            moe=MoEScratch.allocate(tokens, device),
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
            normalized=self.normalized[:tokens],
            engram=self.engram.rows(tokens),
            attention=self.attention.rows(tokens, rows, max_query_tokens),
            moe=self.moe.rows(tokens),
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

        def boundary(prefix, norm):
            return PreparedIHC(
                scratch.streams, scratch.hc_mixes, weights.residual[f"{prefix}_fn"],
                weights.residual[f"{prefix}_base"], weights.residual[f"{prefix}_scale"],
                weights.residual[norm], eps=self.eps, hc_eps=self.hc_eps, magnitude=self.hc_magnitude,
                session=session, name=f"{name}.hc_{prefix}",
            )

        self.attn_ihc = boundary("attn", "input_norm")
        self.mlp_ihc = boundary("mlp", "post_attn_norm")
        self.attn_output_gain = weights.attention["post_proj_norm"]
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
                apply_post_norm=False,
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
                apply_post_norm=False,
                eps=self.eps,
                session=session,
                name=f"{name}.attention",
                workspace=workspace,
            )

        if config.is_sparse_mlp(index):
            self.mlp_output_gain = weights.mlp.combine_norm
            self.mlp = PreparedLatentMoE(
                weights.mlp,
                scratch.normalized,
                scratch.moe,
                apply_combine_norm=False,
                config=config,
                mode=mode,
                session=session,
                name=f"{name}.moe",
                workspace=workspace,
            )
        else:
            self.mlp_output_gain = weights.mlp.post_fc2_norm
            self.mlp = PreparedDenseMLP(
                weights.mlp,
                scratch.normalized,
                scratch.moe,
                apply_post_norm=False,
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
        previous,
        cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        sparse_slots: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        swa_plan: tuple | None = None,
        window_cache: torch.Tensor | None = None,
    ):
        if self.engram is not None:
            if previous:
                streams = post_norm(streams, *previous, self.eps)
            streams = self.engram(streams, metadata)
            previous = ()
        streams, attn_post, attn_in = self.attn_ihc(streams, *previous)
        if isinstance(self.attention, PreparedSparseAttention):
            attn_out, attn_record, physical = self.attention(
                attn_in,
                metadata,
                cos_sin=cos_sin,
                sparse_slots=sparse_slots,
            )
        else:
            attn_out, attn_record, physical = self.attention(
                attn_in,
                metadata,
                cos_sin=cos_sin,
                swa_plan=swa_plan,
                window_cache=window_cache,
            )
        streams, post, mlp_in = self.mlp_ihc(streams, attn_out, None, attn_post, self.attn_output_gain)
        output = self.mlp(mlp_in)
        branch, shared = output if isinstance(output, tuple) else (output, None)
        return streams, (branch, shared, post, self.mlp_output_gain), attn_record, physical


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

        head_streams = scratch.streams if scoring is not None else scratch.head_streams
        head_out = (torch.empty((head_streams.shape[0], HC_MULT), dtype=torch.float32, device=head_streams.device)
                    if scoring is not None else scratch.head_hc_mixes)
        self.head_ihc = PreparedIHC(
            head_streams, head_out, weights.hc_head["fn"], weights.hc_head["base"],
            weights.hc_head["scale"], weights.norm, eps=self.eps, hc_eps=self.hc_eps,
            magnitude=1., session=session, name=f"{name}.hc_head",
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

    @staticmethod
    def _last_tokens(counts, metadata):
        return (metadata.query_start_loc[:-1].long() + counts.long() - 1).clamp(0, metadata.positions.numel() - 1)

    def _save_dfly_indices(
        self,
        counts: torch.Tensor,
        dfly_pools: dict[int, torch.Tensor],
        metadata,
        last_tok: torch.Tensor | None = None,
    ) -> None:
        if not self.dfly_index_layers or not dfly_pools:
            return
        m = metadata
        rows = m.starts.numel()
        active = counts > 0
        if last_tok is None:
            last_tok = self._last_tokens(counts, m)
            dfly_pools = {layer: pools.index_select(0, last_tok) for layer, pools in dfly_pools.items()}
        last_pos = torch.where(active, m.positions[last_tok], -1)
        stacked = torch.stack(
            [
                expand_pooled_topk_to_logical_tokens(
                    dfly_pools[layer],
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
        last_tok = self._last_tokens(m.lengths, m) if self.mode != "verify" and self.dfly_index_layers else None
        previous = ()
        for index, layer in enumerate(self.layers):
            swa_pos = self.swa_indices.get(index)
            if swa_pos is None:
                streams, previous, record, pools = layer(
                    streams,
                    m,
                    previous=previous,
                    cos_sin=cos_sin_idx,
                    sparse_slots=sparse_slots,
                )
            else:
                streams, previous, record, pools = layer(
                    streams,
                    m,
                    previous=previous,
                    cos_sin=cos_sin_swa,
                    swa_plan=swa_plan,
                    window_cache=None if ring_staging is None else ring_staging[swa_pos],
                )
            layer_records.append(record)
            if pools is not None and index in self.dfly_index_layers:
                dfly_pools[index] = pools if last_tok is None else pools.index_select(0, last_tok)

        samples = m.sample_indices
        if self.scoring is not None:
            _, _, all_normed = self.head_ihc(streams, *previous)
            self.scoring(all_normed)
            hidden = all_normed[samples].contiguous()
        else:
            branch, shared, post, gain = previous
            _, _, hidden = self.head_ihc(streams[samples], branch[samples], None if shared is None else shared[samples],
                                        post[samples], gain)

        logits = self.head(hidden)
        if self.mode != "verify":
            if self.window_commit is not None and ring_staging is not None:
                self.window_commit(m.lengths, staging=ring_staging, rows=m.state_rows, starts=m.starts)
            self.table_requests.commit(m.lengths, raw_history, m)
            self._save_dfly_indices(m.lengths, dfly_pools, m, last_tok)
        return hidden, logits, None, (ring_staging, tuple(layer_records), raw_history, dfly_pools)

    def commit(self, counts: torch.Tensor, records, metadata) -> None:
        ring_staging, layer_records, raw_history, dfly_pools = records
        if self.window_commit is not None and ring_staging is not None:
            self.window_commit(counts, staging=ring_staging, rows=metadata.state_rows, starts=metadata.starts)
        for layer, record in zip(self.layers, layer_records, strict=True):
            layer.attention.commit(counts, record, metadata)
        self.table_requests.commit(counts, raw_history, metadata)
        self._save_dfly_indices(counts, dfly_pools, metadata)
