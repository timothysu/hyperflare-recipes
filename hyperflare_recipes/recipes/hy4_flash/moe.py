# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Layer-0 dense FFN and layers 1..41 LatentMoE (512 experts, top-16, 1536 latent width)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hyperflare_recipes.components.expert_overlap import ConcurrentSharedExpert
from hyperflare_recipes.components.linear import QuantizedProjection, UnquantizedProjection, prepare_projection
from hyperflare_recipes.components.mlp import MLPScratch, PreparedMLP
from hyperflare_recipes.components.moe import PreparedRoutedExperts, PreparedRouter, RouterScratch
from hyperflare_recipes.components.rms_norm import PreparedRMSNorm
from ignition.preparation.tuning import declare

from .config import HIDDEN_SIZE


@dataclass(frozen=True)
class MoEScratch:
    dense_mlp: MLPScratch
    shared_mlp: MLPScratch
    router_logits: torch.Tensor
    router: RouterScratch
    latent_in: torch.Tensor
    latent_out: torch.Tensor
    routed_out: torch.Tensor
    combined: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, tokens: int, device: torch.device, *, dense_width: int = 13824,
                 latent_dim: int = 1536, shared_dim: int = 1536, experts: int = 512, top_k: int = 16) -> MoEScratch:
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            dense_mlp=MLPScratch.allocate(tokens, HIDDEN_SIZE, dense_width, device),
            shared_mlp=MLPScratch.allocate(tokens, HIDDEN_SIZE, shared_dim, device),
            router_logits=zeros(tokens, experts, dtype=torch.float32),
            router=RouterScratch.allocate(tokens, device, experts=experts, top_k=top_k),
            latent_in=zeros(tokens, latent_dim),
            latent_out=zeros(tokens, latent_dim),
            routed_out=zeros(tokens, HIDDEN_SIZE),
            combined=zeros(tokens, HIDDEN_SIZE),
            output=zeros(tokens, HIDDEN_SIZE),
        )

    def rows(self, tokens: int) -> MoEScratch:
        return MoEScratch(
            dense_mlp=self.dense_mlp.rows(tokens),
            shared_mlp=self.shared_mlp.rows(tokens),
            router_logits=self.router_logits[:tokens],
            router=self.router.rows(tokens),
            latent_in=self.latent_in[:tokens],
            latent_out=self.latent_out[:tokens],
            routed_out=self.routed_out[:tokens],
            combined=self.combined[:tokens],
            output=self.output[:tokens],
        )


class PreparedDenseMLP:
    """Layer-0 dense SwiGLU FFN closed with ``post_fc2_norm``."""

    def __init__(self, weights, source: torch.Tensor, scratch: MoEScratch, *, eps: float,
                 session, name: str, workspace, apply_post_norm: bool = True):
        self.mlp = PreparedMLP(
            source,
            weights.mlp,
            scratch.dense_mlp.gate_up,
            scratch.dense_mlp.intermediate,
            scratch.dense_mlp.output,
            session=session,
            name=f"{name}.mlp",
            workspace=workspace,
            round_silu=False,
            activation_mode="auto",
        ).functional(session=session)
        self.post_norm = (
            PreparedRMSNorm(
                scratch.dense_mlp.output,
                weights.post_fc2_norm,
                scratch.output,
                eps=eps,
                session=session,
                name=f"{name}.post_fc2_norm",
            ).functional()
            if apply_post_norm
            else None
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        return out if self.post_norm is None else self.post_norm(out)


class PreparedLatentMoE:
    """LatentMoE block: full-width router/shared expert + 1536-wide NVFP4 routed experts."""

    def __init__(self, weights, source: torch.Tensor, scratch: MoEScratch, *, config, mode: str,
                 session, name: str, workspace, apply_combine_norm: bool = True):
        text = config.text
        eps = float(text["rms_norm_eps"])
        self.router_proj = UnquantizedProjection(
            source,
            weights.router,
            scratch.router_logits,
            session=session,
            name=f"{name}.router",
        )
        self.router = PreparedRouter(
            scratch.router_logits,
            scratch.router,
            session=session,
            name=f"{name}.topk",
            score_func="sigmoid",
            renormalize=bool(text["norm_topk_prob"]),
            routed_scaling_factor=float(text["routed_scaling_factor"]),
            correction_bias=weights.correction,
        )
        from b12x.gemm.blockscaled._linear import MXFP8LinearWeight

        use_concurrent = mode in ("decode", "verify") and source.shape[0] <= 2 and workspace.overlap
        self._shared_encoded = (
            mode == "prefill"
            and source.shape[0] >= 128
            and isinstance(weights.down_proj, MXFP8LinearWeight)
            and isinstance(weights.shared.gate_up, MXFP8LinearWeight)
        )
        if self._shared_encoded:
            from b12x.norm.hyperconnection import api as hyperconnection
            from b12x.preparation import PreparedCall

            self.down_proj = QuantizedProjection(
                source,
                weights.down_proj,
                scratch.latent_in,
                session=session,
                name=f"{name}.pre_expert",
                workspace=workspace,
                activation_mode="quantized",
                mxfp8_min_amax=1e-4,
            )
            self.shared_gate_up = QuantizedProjection(
                source,
                weights.shared.gate_up,
                scratch.shared_mlp.gate_up,
                session=session,
                name=f"{name}.shared.up",
                workspace=workspace,
                activation_mode="quantized",
                encoded=self.down_proj.encoded,
            )
            self.shared_act_plan = hyperconnection.plan(
                hyperconnection.Caps(
                    device=source.device,
                    max_tokens=source.shape[0],
                    hidden_size=scratch.shared_mlp.intermediate.shape[1],
                    streams=1,
                    lowrank=1,
                    dtype=source.dtype,
                ),
                invocation={"operation": "swiglu", "limit": float("inf"), "round_silu": False},
            )
            declare(
                session,
                (
                    self.shared_act_plan.request(
                        name=f"{name}.shared.activation",
                        prepare_call=lambda state: PreparedCall(
                            run=lambda: state.launch(
                                scratch.shared_mlp.gate_up,
                                scratch.shared_mlp.intermediate,
                                limit=float("inf"),
                                round_silu=False,
                            )
                        ),
                    ),
                ),
            )
            self.shared_down = prepare_projection(
                scratch.shared_mlp.intermediate,
                weights.shared.down,
                scratch.shared_mlp.output,
                session=session,
                name=f"{name}.shared.down",
                workspace=workspace,
                activation_mode="auto",
            )
            self._shared_inter_shape = tuple(scratch.shared_mlp.intermediate.shape)
            self.shared = None
        else:
            self.down_proj = prepare_projection(
                source,
                weights.down_proj,
                scratch.latent_in,
                session=session,
                name=f"{name}.pre_expert",
                workspace=workspace,
                activation_mode="auto",
            )
            self.shared_gate_up = self.shared_act_plan = self.shared_down = None
            self.shared = PreparedMLP(
                source,
                weights.shared,
                scratch.shared_mlp.gate_up,
                scratch.shared_mlp.intermediate,
                scratch.shared_mlp.output,
                session=session,
                name=f"{name}.shared",
                workspace=workspace,
                round_silu=False,
                activation_mode="auto",
            )
        self.experts = PreparedRoutedExperts(
            scratch.latent_in,
            weights.experts,
            scratch.router.ids,
            scratch.router.weights,
            scratch.latent_out,
            session=session,
            name=f"{name}.experts",
            workspace=workspace,
            router_plan=self.router.plan,
            a4_prefill=(mode == "prefill") and weights.experts_a4_prefill,
            reserve_sms=16 if use_concurrent else 0,
        )
        self.up_proj = prepare_projection(
            scratch.latent_out,
            weights.up_proj,
            scratch.routed_out,
            session=session,
            name=f"{name}.post_expert",
            workspace=workspace,
            activation_mode="auto",
        )
        self.concurrent_shared = None
        if use_concurrent:
            self.concurrent_shared = ConcurrentSharedExpert(
                self.shared,
                parallel_plans=(
                    self.router_proj.plan,
                    self.router.plan,
                    self.down_proj.plan,
                    self.experts.plan,
                    self.up_proj.plan,
                ),
                session=session,
                workspace=workspace,
                defer_launch=False,
            )
        self.combine_norm = (
            PreparedRMSNorm(
                scratch.combined,
                weights.combine_norm,
                scratch.output,
                eps=eps,
                session=session,
                name=f"{name}.combine_norm",
            ).functional()
            if apply_combine_norm
            else None
        )

        self.router_proj.functional(session=session)
        self.router.functional(session=session)
        self.down_proj.functional(session=session)
        self.experts.functional(session=session)
        self.up_proj.functional(session=session)
        if self._shared_encoded:
            self.shared_gate_up.functional(session=session)
            self.shared_down.functional(session=session)
        elif self.concurrent_shared is None:
            self.shared.functional(session=session)
        else:
            self.concurrent_shared.functional(session=session)
            self.shared = None

    def __call__(self, x: torch.Tensor):
        pending = None if self.concurrent_shared is None else self.concurrent_shared.start(x)
        router_logits = self.router_proj(x)
        route = self.router(router_logits)
        if self._shared_encoded:
            bases = tuple(
                torch.empty(shape, dtype=dtype, device=x.device)
                for shape, dtype in self.down_proj.encoded_layout
            )
            latent = self.down_proj(x, encoded_bases=bases)
            shared_gu = self.shared_gate_up(x, encoded_bases=bases)
            shared_inter = torch.empty(self._shared_inter_shape, dtype=x.dtype, device=x.device)
            torch.ops.b12x.hyperconnection_swiglu(
                shared_gu, shared_inter, float("inf"), False, self.shared_act_plan.handle
            )
            shared = self.shared_down(shared_inter)
        else:
            latent = self.down_proj(x)
        routed_latent = self.experts(
            latent,
            route.ids,
            route.weights,
            router_logits=router_logits,
            selected_logits=route.selected_logits,
        )
        routed = self.up_proj(routed_latent)
        if not self._shared_encoded:
            shared = (
                self.shared(x)
                if self.concurrent_shared is None
                else self.concurrent_shared.finish(pending, routed)
            )
        if self.combine_norm is None:
            return routed, shared
        return self.combine_norm(routed + shared)
