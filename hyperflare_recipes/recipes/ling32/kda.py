# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared Ling-3.2-Air KDA with fused 5-branch input projections (no LoRA)."""

from types import SimpleNamespace

import torch
from b12x.sequence import gdn_decode, kda_prefill, short_conv
from b12x.sequence.gdn_decode._kernels import _gdn_decode_op as kda_decode_out
from b12x.sequence.kda_prefill._ops import _run as kda_prefill_out

from hyperflare_recipes.components.gated_norm import PreparedGatedRMSNorm
from hyperflare_recipes.components.kda_speculation import _record_fused, register_recording
from hyperflare_recipes.components.linear import prepare_projection
from hyperflare_recipes.components.recurrent import prepare_sequence_binding, run_short_conv
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.tuning import register_finalizer

from .config import KDA_CONV_KERNEL, KDA_HEAD_DIM
from .weights import KDAWeights

# L2-normalization epsilon passed to b12x kda_prefill_out for Q/K unit-norm scaling.
KDA_L2_NORM_EPS = 1e-6


class PreparedKDA:
    """One Ling-3.2-Air KDA layer: fused QKVFG MXFP8 GEMM, BF16 beta, short conv, and recurrence."""

    def __init__(
        self,
        weights: KDAWeights,
        source: torch.Tensor,
        state,
        metadata,
        *,
        mode: str,
        eps: float,
        lower_bound: float,
        session,
        name: str,
        workspace,
        buffers: dict,
        records=None,
    ):
        exec_mode = "decode" if mode == "verify" else mode
        self.recording = records is not None
        self.state = state
        self.weights = weights
        self.eps = float(eps)
        self.lower_bound = float(lower_bound)
        self.mode = exec_mode
        self.fused = None
        tokens, hidden = source.shape
        heads = state.recurrent.shape[1]
        width = heads * KDA_HEAD_DIM
        device = source.device
        self.tokens, self.heads, self.width = tokens, heads, width

        def buf(key: str, channels: int):
            role = "ling32.hidden_output" if key == "output" else "kda." + key
            return declaration_tensor(buffers, role, (tokens, channels), torch.bfloat16, device)

        projected = buf("projected", 5 * width)
        beta = buf("beta", heads)
        mixed = buf("mixed", 3 * width)
        raw_g = buf("raw_g", width)
        recurrence = buf("recurrence", width)
        gated = buf("gated", width)
        qkv = projected[:, : 3 * width]
        gate = projected[:, 4 * width : 5 * width]

        self.qkvfg = prepare_projection(
            source, weights.qkvfg, projected, session=session, name=name + ".qkvfg", workspace=workspace
        ).functional(session=session)
        self.b_proj = prepare_projection(
            source, weights.b_proj, beta, session=session, name=name + ".b_proj", workspace=workspace
        ).functional(session=session)
        self.output = prepare_projection(
            gated, weights.output, buf("output", hidden), session=session, name=name + ".o_proj", workspace=workspace
        ).functional(session=session)

        sequences = metadata.query_start_loc.numel() - 1
        conv_args = dict(
            x=qkv,
            conv_weight=weights.convolution,
            query_start_loc=metadata.query_start_loc,
            state_slot_ids=metadata.convolution_slots,
            state_is_fresh=metadata.fresh,
            num_accepted_tokens=metadata.accepted_tokens,
            num_seqs=metadata.num_seqs,
            num_tokens=metadata.num_tokens,
            conv_state=state.convolution,
            out=mixed,
        )
        conv_caps = short_conv.Caps(
            device=device,
            mode=exec_mode,
            channels=3 * width,
            kernel_size=KDA_CONV_KERNEL,
            max_tokens=tokens,
            max_seqs=sequences,
            max_state_slots=state.convolution.shape[0],
            max_speculative_tokens=state.convolution.shape[-1] - (KDA_CONV_KERNEL - 1),
        )
        self.conv_plan = short_conv.plan(conv_caps, invocation=short_conv.invocation_from_tensors(**conv_args))
        prepare_sequence_binding(
            self.conv_plan,
            conv_args,
            session=session,
            name=name + ".conv",
            mode=exec_mode,
            public_bind=short_conv.bind,
            bind_plan=True,
            workspace=workspace,
            set_binding=lambda binding: setattr(self, "convolution", binding),
        )

        recurrent = state.recurrent if records is None else state.recurrent[: records[0].shape[0]]
        self.recurrent_state = recurrent
        dt_bias = weights.dt_bias.view(heads, 128)
        common = dict(
            raw_g=raw_g.view(tokens, heads, 128),
            raw_beta=beta,
            A_log=weights.A_log,
            dt_bias=dt_bias,
            recurrent_state=recurrent,
            num_seqs=metadata.num_seqs,
            num_tokens=metadata.num_tokens,
            output=(gated if exec_mode == "decode" else recurrence).view(tokens, heads, 128),
        )
        options = {"lower_bound": self.lower_bound}
        if exec_mode == "decode":
            options["eps"] = self.eps
            rec_caps = gdn_decode.Caps(
                device=device,
                max_tokens=tokens,
                max_seqs=sequences,
                max_state_slots=recurrent.shape[0],
                key_heads=heads,
                value_heads=heads,
                state_index_columns=metadata.state_indices.shape[1],
                gate_activation="sigmoid",
                null_state_index=0,
                recover_speculative_state=records is not None,
            )
            rec_args = dict(
                common,
                mixed_qkv=mixed,
                z=gate.view(tokens, heads, 128),
                norm_weight=weights.o_norm,
                query_start_loc=metadata.query_start_loc,
                num_accepted_tokens=metadata.accepted_tokens,
                state_indices=metadata.state_indices,
            )
            if records is not None:
                rec_args.update(correction_cache=records[0], kg_cache=records[1])
                self.handle = register_recording(self)
            rec_plan = gdn_decode.plan(rec_caps, invocation=gdn_decode.invocation_from_tensors(rec_caps, **rec_args))
            bind_fn, bind_method = gdn_decode.bind_kda, "bind_kda"
            self.norm = None
        else:
            rec_caps = kda_prefill.Caps(
                device=device,
                max_tokens=tokens,
                max_seqs=sequences,
                max_state_slots=state.recurrent.shape[0],
                heads=heads,
                null_state_index=0,
            )
            q, k, v = (x.view(tokens, heads, 128) for x in mixed.split(width, dim=-1))
            rec_args = dict(
                common,
                q=q,
                k=k,
                v=v,
                cu_seqlens=metadata.query_start_loc,
                initial_state_indices=metadata.initial_state_indices,
                final_state_indices=metadata.final_state_indices,
                checkpoint_state_indices=metadata.checkpoint_state_indices,
                checkpoint_offsets=metadata.checkpoint_offsets,
            )
            rec_plan = kda_prefill.plan(rec_caps, invocation=kda_prefill.invocation_from_tensors(**rec_args))
            bind_fn, bind_method = kda_prefill.bind, "bind"
            self.norm = PreparedGatedRMSNorm(
                recurrence.view(tokens, heads, 128),
                gate.view(tokens, heads, 128),
                weights.o_norm,
                gated.view(tokens, heads, 128),
                metadata.num_tokens,
                eps=self.eps,
                activation="sigmoid",
            ).functional(session=session, dynamic_live=True)

        self.plan = rec_plan
        prepare_sequence_binding(
            rec_plan,
            rec_args,
            session=session,
            name=name + ".recurrence",
            mode=exec_mode,
            public_bind=bind_fn,
            bind_method=bind_method,
            workspace=workspace,
            run_options=options,
            set_binding=lambda binding: setattr(self, "recurrence", binding),
        )
        register_finalizer(session, self._finalize)
        if workspace is not None:
            workspace.register_preparation_release(self._release_bindings)

    def _finalize(self):
        self.conv_bytes = self.conv_plan.scratch_specs()[0].nbytes
        self.recurrence_bytes = sum(spec.nbytes for spec in self.plan.scratch_specs())
        if self.recording:
            self.fused = gdn_decode.prepare_kda_fused_verify(
                self.recurrence, self.convolution, retain_examples=False
            )

    def _release_bindings(self):
        self.convolution = self.recurrence = None

    def __call__(self, source: torch.Tensor, metadata, *, records=None) -> torch.Tensor:
        if self.recording and records is None:
            raise ValueError("recording KDA execution requires invocation-local records")
        tokens, heads, width = source.shape[0], self.heads, self.width
        projected = self.qkvfg(source)
        beta = self.b_proj(source)
        qkv = projected[:, : 3 * width]
        raw_g = torch.empty((tokens, heads, KDA_HEAD_DIM), dtype=source.dtype, device=source.device)
        raw_g.copy_(projected[:, 3 * width : 4 * width].view(tokens, heads, KDA_HEAD_DIM))
        gate = projected[:, 4 * width : 5 * width].view(tokens, heads, KDA_HEAD_DIM)
        dt_bias = self.weights.dt_bias.view(heads, KDA_HEAD_DIM)

        if self.recording:
            if self.fused is None:
                raise RuntimeError("recording KDA execution requires finalized fused verification")
            output = torch.empty((tokens, heads, KDA_HEAD_DIM), dtype=source.dtype, device=source.device)
            _record_fused(
                [
                    qkv,
                    raw_g,
                    beta,
                    gate,
                    self.weights.A_log,
                    dt_bias,
                    self.weights.o_norm,
                    self.weights.convolution,
                    metadata.query_start_loc,
                    metadata.convolution_slots,
                    metadata.fresh,
                    metadata.accepted_tokens,
                    metadata.state_indices,
                    metadata.num_seqs,
                    metadata.num_tokens,
                ],
                self.recurrent_state,
                output,
                *records,
                self.state.convolution,
                self.handle,
            )
            return self.output(output.reshape(tokens, -1))

        mixed = torch.empty((tokens, 3 * width), dtype=source.dtype, device=source.device)
        conv_scratch = torch.empty(self.conv_bytes, dtype=torch.uint8, device=source.device)
        run_short_conv(
            self.conv_plan,
            x=qkv,
            conv_weight=self.weights.convolution,
            query_start_loc=metadata.query_start_loc,
            state_slot_ids=metadata.convolution_slots,
            state_is_fresh=metadata.fresh,
            num_accepted_tokens=metadata.accepted_tokens,
            num_seqs=metadata.num_seqs,
            num_tokens=metadata.num_tokens,
            conv_state=self.state.convolution,
            out=mixed,
            scratch=conv_scratch,
        )
        output = torch.empty((tokens, heads, KDA_HEAD_DIM), dtype=source.dtype, device=source.device)
        if self.mode == "decode":
            kda_decode_out(
                mixed,
                raw_g,
                beta,
                gate,
                self.weights.A_log,
                dt_bias,
                self.weights.o_norm,
                self.recurrent_state,
                metadata.query_start_loc,
                metadata.accepted_tokens,
                metadata.state_indices,
                metadata.num_seqs,
                metadata.num_tokens,
                output,
                self.eps,
                KDA_HEAD_DIM ** -0.5,
                self.lower_bound,
                self.plan.handle,
            )
        else:
            scratch = torch.empty(self.recurrence_bytes, dtype=torch.uint8, device=source.device)
            q, k, v = (x.view(tokens, heads, KDA_HEAD_DIM) for x in mixed.split(width, dim=-1))
            kda_prefill_out(
                [
                    q,
                    k,
                    v,
                    raw_g,
                    beta,
                    self.weights.A_log,
                    dt_bias,
                    metadata.query_start_loc,
                    metadata.initial_state_indices,
                    metadata.final_state_indices,
                    metadata.checkpoint_state_indices,
                    metadata.checkpoint_offsets,
                    metadata.num_seqs,
                    metadata.num_tokens,
                ],
                scratch,
                self.recurrent_state,
                output,
                self.plan.handle,
                self.lower_bound,
                None,
                KDA_L2_NORM_EPS,
                None,
                None,
            )
            output = self.norm(output, gate, live=metadata.num_tokens)
        return self.output(output.reshape(tokens, -1))

    def run_fused(self, reads, recurrent_state, output, corrections, keys_gates, convolution_state):
        (
            qkv,
            raw_g,
            raw_beta,
            z,
            A_log,
            dt_bias,
            norm_weight,
            conv_weight,
            query_start_loc,
            convolution_slots,
            fresh,
            accepted_tokens,
            state_indices,
            num_seqs,
            num_tokens,
        ) = reads
        binding = SimpleNamespace(
            raw_g=raw_g,
            raw_beta=raw_beta,
            A_log=A_log,
            dt_bias=dt_bias,
            recurrent_state=recurrent_state,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            num_accepted_tokens=accepted_tokens,
            num_seqs=num_seqs,
            num_tokens=num_tokens,
            output=output,
            z=z,
            norm_weight=norm_weight,
            correction_cache=corrections,
            kg_cache=keys_gates,
        )
        convolution = SimpleNamespace(
            x=qkv,
            conv_weight=conv_weight,
            state_slot_ids=convolution_slots,
            state_is_fresh=fresh,
            num_accepted_tokens=accepted_tokens,
            conv_state=convolution_state,
            query_start_loc=query_start_loc,
            num_seqs=num_seqs,
            num_tokens=num_tokens,
        )
        self.fused.run_dynamic(
            binding, convolution, scale=KDA_HEAD_DIM ** -0.5, lower_bound=self.lower_bound, eps=self.eps
        )
