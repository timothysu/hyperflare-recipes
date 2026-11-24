# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared 42-layer Ling-3.2-Air decoder composing KDA, Gated MLA, and MoE."""

from dataclasses import dataclass

import torch
import triton

from hyperflare_recipes.components.embedding import PreparedEmbedding
from hyperflare_recipes.components.kda_speculation import KDARecords, PreparedKDAReplay
from hyperflare_recipes.components.paged_batch import rotary_offset_region, rotary_positions
from hyperflare_recipes.components.recurrent import DeltaMetadata, _commit_convolution
from hyperflare_recipes.components.rms_norm import PreparedRMSNorm
from hyperflare_recipes.components.rotary import PreparedRotaryAngles
from hyperflare_recipes.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.persistent import persistent_copy

from .attention import PreparedGatedMLA
from .config import mrope_axis_indices
from .kda import PreparedKDA
from .moe import PreparedFeedForward


def rotary_frequencies(rope_theta: float, device: torch.device) -> torch.Tensor:
    dims = torch.arange(32, dtype=torch.float32, device=device)
    return persistent_copy(1.0 / (float(rope_theta) ** (dims / 32.0)))


def mrope_angles(
    positions: torch.Tensor,
    logical_positions: torch.Tensor,
    frequencies: torch.Tensor,
    axis_indices: torch.Tensor | None,
) -> torch.Tensor:
    """Compute contiguous FP32 [tokens, 32, 2] cos/sin pairs for 1D or Bailing 3D M-RoPE."""
    if positions.ndim == 1:
        pos = positions.clamp_min(0).float()[:, None]
    else:
        live = logical_positions >= 0
        if positions.shape[1] == 1:
            pos = torch.where(live, positions[:, 0], 0).float()[:, None]
        else:
            if axis_indices is None:
                raise ValueError("three-axis rotary positions require Bailing M-RoPE axis indices")
            # Centered H/W coordinates can be negative on live image tokens; mask only inactive rows.
            pos = torch.where(live[:, None], positions[:, axis_indices], 0).float()
    angle = pos * frequencies[None, :]
    return torch.stack((angle.cos(), angle.sin()), dim=-1).contiguous()


@dataclass(frozen=True)
class DecoderScratch:
    activations: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, tokens: int, rows: int, config, device, *, examples=None):
        def tensor(role: str, shape):
            return declaration_tensor(examples, role, shape, torch.bfloat16, device)

        return cls(
            tensor("ling32.hidden_input", (tokens, config.hidden)),
            tensor("ling32.hidden", (rows, config.hidden)),
            tensor("ling32.local_logits", (rows, config.vocabulary)),
            tensor("ling32.logits", (rows, config.vocabulary)),
        )

    def rows(self, tokens: int, rows: int) -> "DecoderScratch":
        return type(self)(
            self.activations[:tokens],
            self.hidden[:rows],
            self.local_logits[:rows],
            self.logits[:rows],
        )


class PreparedDecoder:
    """42-layer hybrid decoder (35 KDA + 7 Gated MLA, 2 dense + 40 MoE) with KDA verify commit."""

    def __init__(
        self,
        weights,
        scratch: DecoderScratch,
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
        overlap: bool = True,
        kda_records: KDARecords | None = None,
        scoring=None,
        vision_state=None,
        examples=None,
    ):
        del runtime
        if mode not in ("prefill", "decode", "verify") or (mode == "verify" and kda_records is None):
            raise ValueError("Ling-3.2-Air verification requires KDA candidate recording storage")
        self.weights = weights
        self.scratch = scratch
        self.pool = pool
        self.mode = mode
        self.vision_state = vision_state if mode == "prefill" else None
        config = weights.config
        source = scratch.activations
        tokens, hidden = source.shape
        device = source.device
        self.tokens, self.hidden = tokens, hidden
        self.sequences = metadata.starts.numel()
        self.verification_columns = 8 if mode == "verify" else 1
        examples = {} if examples is None else examples

        self.embedding = PreparedEmbedding(
            weights.embedding,
            metadata.token_ids,
            source,
            num_rows=metadata.live_counts[:1],
            session=session,
            name=name + ".embedding",
        ).functional(session=session, dynamic_inputs=True)

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

        delta = DeltaMetadata.allocate(self.sequences, device, verification_columns=self.verification_columns)
        norm_out = declaration_tensor(examples, "ling32.norm_out", (tokens, hidden), torch.bfloat16, device)
        residual_example = declaration_tensor(examples, "ling32.residual", (tokens, hidden), torch.bfloat16, device)

        self.auxiliary = (
            None
            if weights.draft is None
            else declaration_tensor(examples, "ling32.auxiliary", (tokens, hidden), torch.bfloat16, device)
        )

        self.layers = []
        for index, w in enumerate(weights.layers):
            prefix = f"{name}.layer{index}"
            kind = config.layer_types[index]
            input_norm = PreparedRMSNorm(
                residual_example,
                w.input_norm,
                norm_out,
                eps=config.eps,
                session=session,
                name=prefix + ".input_norm",
                zero_centered=False,
                residual=source if index > 0 else None,
            ).functional()
            if kind == "kda":
                attention = PreparedKDA(
                    w.attention,
                    norm_out,
                    pool.recurrent(index),
                    delta,
                    mode=mode,
                    eps=config.eps,
                    lower_bound=config.kda_lower_bound,
                    session=session,
                    name=prefix + ".kda",
                    workspace=workspace,
                    buffers=examples,
                    records=kda_records.layer(index) if mode == "verify" else None,
                )
            else:
                attention = PreparedGatedMLA(
                    w.attention,
                    norm_out,
                    metadata,
                    pool.attention(index),
                    self.rotary,
                    eps=config.eps,
                    mode=mode,
                    session=session,
                    name=prefix + ".mla",
                    workspace=workspace,
                    buffers=examples,
                )
            post_norm = PreparedRMSNorm(
                residual_example,
                w.post_attention_norm,
                norm_out,
                eps=config.eps,
                session=session,
                name=prefix + ".post_norm",
                zero_centered=False,
                residual=source,
            ).functional()
            ffn = PreparedFeedForward(
                index,
                w,
                config,
                norm_out,
                mode=mode,
                session=session,
                name=prefix + ".ffn",
                workspace=workspace,
                buffers=examples,
                overlap=overlap,
            )
            self.layers.append((kind, input_norm, attention, post_norm, ffn))

        self.final_norm = PreparedRMSNorm(
            residual_example,
            weights.norm,
            norm_out,
            eps=config.eps,
            session=session,
            name=name + ".final_norm",
            zero_centered=False,
            residual=source,
        ).functional()

        self.scoring = (
            None
            if scoring is None
            else scoring.prepare(
                source,
                weights.head,
                vocabulary=config.vocabulary,
                session=session,
                coordinator=coordinator,
                name=name + ".prompt",
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
            name=name + ".head",
        ).functional(session=session)

        self.replay = None
        if mode == "verify":
            indices = kda_records.layers
            states = [pool.recurrent(i).recurrent[: kda_records.corrections.shape[1]] for i in indices]
            self.record_layers = indices
            self.record_shapes = (tuple(kda_records.corrections.shape), tuple(kda_records.keys_and_gates.shape))
            self.record_constants = (kda_records.A_log, kda_records.dt_bias)
            first_kda = self.layers[indices[0]][2]
            self.replay = PreparedKDAReplay(
                pool.backing, states, kda_records, first_kda.plan, session=session
            ).functional()
            regions = [pool.layout.region(f"layer{i}.convolution") for i in indices]
            self.convolution_offsets = persistent_copy(
                torch.tensor([r.offset // 2 for r in regions], dtype=torch.int64, device=device)
            )
            self.convolution_shape = regions[0].shape

    def release_outputs(self):
        self.scratch = self.auxiliary = None

    def commit(self, counts: torch.Tensor, verification, metadata):
        records, delta = verification
        self.replay(counts, records, delta)
        channels, columns = self.convolution_shape
        torch.library.wrap_triton(_commit_convolution)[
            (self.convolution_offsets.numel(), counts.numel(), triton.cdiv(channels, 128))
        ](
            self.pool.backing.view(torch.bfloat16),
            self.convolution_offsets,
            metadata.state_rows,
            counts,
            self.pool.layout.row_bytes // 2,
            channels,
            columns,
            128,
        )

    def __call__(self, metadata):
        m = metadata
        device = m.token_ids.device
        d = DeltaMetadata.allocate(self.sequences, device, verification_columns=self.verification_columns)
        active = m.lengths > 0
        d.query_start_loc.copy_(m.query_start_loc)
        d.convolution_slots.copy_(torch.where(active, m.state_rows, -1))
        d.fresh.copy_(m.starts == 0)
        d.state_indices[:, 0].copy_(m.state_rows)
        d.initial_state_indices.copy_(torch.where(m.starts > 0, m.state_rows, 0))
        d.final_state_indices.copy_(m.state_rows)
        d.num_tokens.copy_(m.live_counts[:1])
        d.num_seqs.copy_(m.live_counts[1:])

        records = None
        if self.replay is not None:
            records = KDARecords(
                self.record_layers,
                torch.zeros(self.record_shapes[0], dtype=torch.float32, device=device),
                torch.zeros(self.record_shapes[1], dtype=torch.bfloat16, device=device),
                *self.record_constants,
            )

        angles = (
            self.rotary(m.positions)
            if self.mrope_axes is None
            else mrope_angles(rotary_positions(m), m.positions, self.rotary_frequencies, self.mrope_axes)
        )
        residual = self.embedding(token_ids=m.token_ids, num_rows=m.live_counts[:1])
        if self.vision_state is not None:
            self.vision_state.inject(m, residual)
        update = None
        for index, (kind, input_norm, attention, post_norm, ffn) in enumerate(self.layers):
            hidden = input_norm(residual) if update is None else input_norm(residual, update)
            if kind == "kda":
                attended = attention(hidden, d, records=None if records is None else records.layer(index))
            else:
                attended = attention(hidden, m, angles)
            hidden = post_norm(residual, attended)
            update = ffn(hidden)

        normalized = self.final_norm(residual, update)
        if self.scoring is not None:
            self.scoring(normalized)
        sampled = normalized[m.sample_indices]
        logits = self.head(sampled)
        auxiliary = normalized if self.weights.draft is not None else None
        return sampled, logits, auxiliary, (records, d)
