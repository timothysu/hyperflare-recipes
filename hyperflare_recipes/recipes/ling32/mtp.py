# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared Ling-3.2-Air Layer-42 MTP predictor, teacher-forced absorb, and causal proposals."""

import torch

from hyperflare_recipes.components.embedding import PreparedEmbedding
from hyperflare_recipes.components.linear import prepare_projection
from hyperflare_recipes.components.paged_batch import (
    Metadata,
    gather_rotary_offset,
    rotary_offset_region,
    rotary_positions,
)
from hyperflare_recipes.components.rms_norm import PreparedPairedRMSNorm, PreparedRMSNorm
from hyperflare_recipes.components.rotary import PreparedRotaryAngles
from ignition.components.state import gather_rows, save_hidden
from hyperflare_recipes.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.persistent import persistent_copy

from .attention import PreparedGatedMLA
from .config import mrope_axis_indices
from .decoder import mrope_angles, rotary_frequencies
from .moe import PreparedFeedForward


class PreparedMTP:
    """Layer-42 MTP: paired enorm/hnorm -> eh_proj -> Gated MLA -> 512-expert MoE -> final_norm -> head."""

    def __init__(
        self,
        weights,
        pool,
        metadata,
        *,
        mode: str,
        vocabulary_runtime=None,
        coordinator=None,
        session,
        name: str,
        workspace,
        examples: dict,
        cache_only: bool = False,
        overlap: bool = True,
    ):
        config = weights.config
        draft = weights.draft
        device = metadata.token_ids.device
        tokens = metadata.token_ids.numel()
        hidden = config.hidden
        self.tokens, self.hidden = tokens, hidden
        self.cache_only = bool(cache_only)

        def buf(key: str, *shape, dtype=torch.bfloat16):
            return declaration_tensor(examples, "mtp." + key, (tokens, *shape), dtype, device)

        embedded = buf("embedded", hidden)
        prev_hidden = buf("prev_hidden", hidden)
        joined = buf("joined", 2 * hidden)
        residual = buf("residual", hidden)
        norm_out = buf("norm_out", hidden)

        self.embedding = PreparedEmbedding(
            weights.embedding,
            metadata.token_ids,
            embedded,
            num_rows=metadata.live_counts[:1],
            session=session,
            name=name + ".embedding",
        ).functional(session=session, dynamic_inputs=True)
        self.join = PreparedPairedRMSNorm(
            embedded,
            prev_hidden,
            (draft.enorm, draft.hnorm),
            joined,
            eps=config.eps,
            session=session,
            name=name + ".enorm_hnorm",
            zero_centered=False,
        ).functional()
        self.eh_proj = prepare_projection(
            joined,
            draft.eh_proj,
            residual,
            session=session,
            name=name + ".eh_proj",
            workspace=workspace,
        ).functional(session=session)
        self.input_norm = PreparedRMSNorm(
            residual,
            draft.layer.input_norm,
            norm_out,
            eps=config.eps,
            session=session,
            name=name + ".input_norm",
            zero_centered=False,
        ).functional()
        self.rotary_frequencies = rotary_frequencies(config.rope_theta, device)
        self.rotary = PreparedRotaryAngles(
            metadata.positions,
            self.rotary_frequencies,
        ).functional()
        self.mrope_axes = (
            persistent_copy(torch.tensor(mrope_axis_indices(config.mrope_section), dtype=torch.int64, device=device))
            if rotary_offset_region(pool) is not None
            else None
        )
        self.attention = PreparedGatedMLA(
            draft.layer.attention,
            norm_out,
            metadata,
            pool.draft_attention(),
            self.rotary,
            eps=config.eps,
            mode=mode,
            session=session,
            name=name + ".mla",
            workspace=workspace,
            buffers=examples,
            cache_only=cache_only,
        )
        if cache_only:
            self.post_norm = self.ffn = self.final_norm = self.head = None
            return

        self.post_norm = PreparedRMSNorm(
            residual,
            draft.layer.post_attention_norm,
            norm_out,
            eps=config.eps,
            session=session,
            name=name + ".post_norm",
            zero_centered=False,
            residual=norm_out,
        ).functional()
        self.ffn = PreparedFeedForward(
            config.layers,
            draft.layer,
            config,
            norm_out,
            mode=mode,
            session=session,
            name=name + ".ffn",
            workspace=workspace,
            buffers=examples,
            overlap=overlap,
        )
        self.final_norm = PreparedRMSNorm(
            residual,
            draft.final_norm,
            norm_out,
            eps=config.eps,
            session=session,
            name=name + ".final_norm",
            zero_centered=False,
            residual=norm_out,
        ).functional()
        head_weight = draft.draft_head if draft.draft_head is not None else weights.head
        local_logits = buf("local_logits", config.vocabulary)
        logits = buf("logits", config.vocabulary)
        self.head = PreparedVocabularyProjection(
            norm_out,
            head_weight,
            local_logits,
            logits,
            runtime=vocabulary_runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".head",
            workspace=workspace,
        ).functional(session=session)

    def __call__(self, metadata, prev_hidden: torch.Tensor, *, vision_state=None, vision_indices=None):
        embedded = self.embedding(token_ids=metadata.token_ids, num_rows=metadata.live_counts[:1])
        if vision_state is not None and vision_indices is not None:
            slot = vision_state.feature_slot.long().clamp(0, vision_state.arena.shape[0] - 1)
            safe_idx = vision_indices.long().clamp(0, vision_state.arena.shape[1] - 1)
            embedded = torch.where(
                (metadata.token_ids == vision_state.image_token)[:, None],
                vision_state.arena[slot, safe_idx],
                embedded,
            )
        joined = self.join(embedded, prev_hidden)
        residual = self.eh_proj(joined)
        hidden = self.input_norm(residual)
        angles = (
            self.rotary(metadata.positions)
            if self.mrope_axes is None
            else mrope_angles(rotary_positions(metadata), metadata.positions, self.rotary_frequencies, self.mrope_axes)
        )
        if self.cache_only:
            self.attention(hidden, metadata, angles)
            return None, None
        attended = self.attention(hidden, metadata, angles)
        hidden = self.post_norm(residual, attended)
        mlp_out = self.ffn(hidden)
        out_hidden = self.final_norm(residual, mlp_out)
        return out_hidden, self.head(out_hidden)


class PreparedAcceptedContext:
    """Cache-only Layer-42 MTP absorption of accepted target pairs and target_hidden update."""

    def __init__(
        self,
        weights,
        metadata,
        pool,
        *,
        draft_tables: torch.Tensor,
        inputs,
        examples: dict,
        session,
        name: str,
        workspace,
        is_prefill: bool = False,
        vision_state=None,
        runtime=None,
        vocabulary_runtime=None,
        coordinator=None,
    ):
        del runtime
        self.pool = pool
        self.draft_tables = draft_tables
        self.inputs = inputs
        self.tokens = metadata.token_ids.numel()
        self.rows = metadata.starts.numel()
        self.hidden = weights.config.hidden
        self.is_prefill = bool(is_prefill)
        self.vision_state = vision_state if self.is_prefill else None
        blocks = draft_tables.shape[1]
        device = metadata.token_ids.device
        absorb_meta = Metadata.allocate(self.tokens, self.rows, blocks, device, samples=self.rows)
        self.mtp = PreparedMTP(
            weights,
            pool,
            absorb_meta,
            mode="prefill" if self.is_prefill else "decode",
            vocabulary_runtime=vocabulary_runtime,
            coordinator=coordinator,
            session=session,
            name=name,
            workspace=workspace,
            examples=examples,
            cache_only=True,
            overlap=False,
        )

    def functional(self):
        return self

    def __call__(self, auxiliary: torch.Tensor, counts: torch.Tensor, metadata):
        device = auxiliary.device
        capacity = self.draft_tables.shape[0]
        slots = self.inputs.row_slot_ids[: self.rows].long().clamp(0, capacity - 1)
        pages = self.draft_tables[slots]
        target_hist = self.pool.draft_target_hidden()
        prev_hidden = torch.empty((self.rows, self.hidden), dtype=torch.bfloat16, device=device)
        gather_rows(target_hist, metadata.state_rows, prev_hidden)

        if self.is_prefill:
            before = metadata.starts[0]
            length = counts[0].long()
            skip = (before == 0).long()
            rows_idx = torch.arange(self.tokens, device=device)
            src_tok = (rows_idx + skip).clamp_max(self.tokens - 1)
            mtp_len = (length - skip).clamp_min(0)
            mtp_start = (before - 1).clamp_min(0)
            live = rows_idx < mtp_len
            mtp_tokens = torch.where(live, metadata.token_ids[src_tok], 0)
            mtp_hidden = torch.where(
                ((rows_idx == 0) & (before > 0))[:, None],
                prev_hidden,
                auxiliary[(src_tok - 1).clamp_min(0)],
            )
            mtp_hidden = torch.where(live[:, None], mtp_hidden, 0)
            mtp_pos = torch.where(live, mtp_start + rows_idx, -1)
            absorb_rope = (
                None
                if metadata.rotary_positions is None
                else torch.where(live[:, None], metadata.rotary_positions[src_tok], -1)
            )
            vision_indices = (
                None
                if self.vision_state is None
                else self.vision_state.prefill.vision_token_indices[: self.tokens][src_tok]
            )
            qsl = torch.cat((torch.zeros(1, dtype=torch.int32, device=device), mtp_len.view(1).to(torch.int32)))
            live_counts = torch.stack((mtp_len.to(torch.int32), (mtp_len > 0).to(torch.int32)))
            absorb_meta = Metadata(
                token_ids=mtp_tokens,
                sample_indices=torch.zeros(1, dtype=torch.int64, device=device),
                query_start_loc=qsl,
                state_rows=metadata.state_rows,
                starts=mtp_start.view(1),
                lengths=mtp_len.view(1).to(torch.int32),
                requests=torch.zeros(self.tokens, dtype=torch.int64, device=device),
                positions=mtp_pos,
                page_table=pages,
                token_page_table=pages.expand(self.tokens, -1),
                live_counts=live_counts,
                max_query_tokens=self.tokens,
                rotary_positions=absorb_rope,
            )
            self.mtp(absorb_meta, mtp_hidden, vision_state=self.vision_state, vision_indices=vision_indices)
            last_idx = (length - 1).clamp(0, self.tokens - 1).view(1)
            save_hidden(auxiliary[last_idx], target_hist, metadata.state_rows, counts > 0)
            return

        # Decode / speculative verification absorption (before >= 1 for all active requests).
        offsets = torch.arange(self.tokens, device=device)
        req = metadata.requests
        local = offsets - metadata.query_start_loc[req].long()
        active_tok = (metadata.positions >= 1) & (local < counts[req].long())
        mtp_pos = torch.where(active_tok, metadata.positions - 1, -1)
        absorb_rope = (
            None
            if metadata.rotary_positions is None
            else torch.where(active_tok[:, None], metadata.rotary_positions, -1)
        )
        mtp_tokens = torch.where(active_tok, metadata.token_ids, 0)
        mtp_hidden = torch.where(
            (local == 0)[:, None],
            prev_hidden[req],
            auxiliary[(offsets - 1).clamp_min(0)],
        )
        mtp_hidden = torch.where(active_tok[:, None], mtp_hidden, 0)
        live_rows = (counts > 0).sum().to(torch.int32)
        embed_rows = torch.full((), self.tokens, dtype=torch.int32, device=device)
        absorb_meta = Metadata(
            token_ids=mtp_tokens,
            sample_indices=metadata.sample_indices[: self.rows],
            query_start_loc=metadata.query_start_loc,
            state_rows=metadata.state_rows,
            starts=(metadata.starts - 1).clamp_min(0),
            lengths=counts.to(torch.int32),
            requests=req,
            positions=mtp_pos,
            page_table=pages,
            token_page_table=pages[req],
            live_counts=torch.stack((embed_rows, live_rows)),
            max_query_tokens=metadata.max_query_tokens,
            rotary_positions=absorb_rope,
        )
        self.mtp(absorb_meta, mtp_hidden)
        last_pos = (metadata.query_start_loc[:-1].long() + counts.long() - 1).clamp(0, self.tokens - 1)
        save_hidden(auxiliary[last_pos], target_hist, metadata.state_rows, counts > 0)


class PreparedProposals:
    """Autoregressive Layer-42 MTP proposer for ``PreparedCausalDraftRound``."""

    def __init__(
        self,
        weights,
        pool,
        batch: int,
        depth: int,
        *,
        draft_tables: torch.Tensor,
        inputs,
        requests,
        examples: dict,
        session,
        name: str,
        workspace,
        runtime=None,
        vocabulary_runtime=None,
        coordinator=None,
    ):
        del runtime
        self.pool = pool
        self.batch = batch
        self.depth = depth
        self.hidden = weights.config.hidden
        self.draft_tables = draft_tables
        self.inputs = inputs
        self.requests = requests
        self.rotary_offset = rotary_offset_region(pool)
        self.logit_shape = (batch, weights.config.vocabulary)
        self.logit_dtype = torch.bfloat16
        self.mtp = None
        if depth > 0:
            device = draft_tables.device
            blocks = draft_tables.shape[1]
            step_meta = Metadata.allocate(batch, batch, blocks, device, samples=batch)
            self.mtp = PreparedMTP(
                weights,
                pool,
                step_meta,
                mode="decode",
                vocabulary_runtime=vocabulary_runtime,
                coordinator=coordinator,
                session=session,
                name=name + ".step",
                workspace=workspace,
                examples=examples,
                cache_only=False,
                overlap=True,
            )

    def release_outputs(self):
        pass

    def begin(self, states: torch.Tensor) -> torch.Tensor:
        if self.depth == 0:
            return states
        prev = torch.empty((self.batch, self.hidden), dtype=torch.bfloat16, device=states.device)
        gather_rows(self.pool.draft_target_hidden(), states, prev)
        return prev

    def step(
        self,
        draft_state: torch.Tensor,
        index: int,
        tokens: torch.Tensor,
        before: torch.Tensor,
        states: torch.Tensor,
        active: torch.Tensor,
        proposing: torch.Tensor,
    ):
        del active
        device = tokens.device
        slots = self.inputs.row_slot_ids[: self.batch].long().clamp(0, self.requests.capacity - 1)
        pages = self.draft_tables[slots]
        pos = torch.where(proposing, before - 1 + index, -1)
        rope = None
        if self.rotary_offset is not None:
            offset = gather_rotary_offset(self.rotary_offset, torch.where(proposing, states, 0))
            rope = torch.where(proposing, before + index + offset, -1)[:, None]
        starts = (before - 1 + index).clamp_min(0)
        lengths = proposing.to(torch.int32)
        qsl = torch.arange(self.batch + 1, dtype=torch.int32, device=device)
        req = torch.arange(self.batch, dtype=torch.int64, device=device)
        live = torch.full((2,), self.batch, dtype=torch.int32, device=device)
        meta = Metadata(
            token_ids=torch.where(proposing, tokens, 0),
            sample_indices=req,
            query_start_loc=qsl,
            state_rows=torch.where(proposing, states, 0),
            starts=starts,
            lengths=lengths,
            requests=req,
            positions=pos,
            page_table=pages,
            token_page_table=pages,
            live_counts=live,
            max_query_tokens=1,
            rotary_positions=rope,
        )
        next_hidden, logits = self.mtp(meta, draft_state)
        return logits, next_hidden

    def finish(self, draft_state, before, states, active):
        del draft_state, before, states, active
