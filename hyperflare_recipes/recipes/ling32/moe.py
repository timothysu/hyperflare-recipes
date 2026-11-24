# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared Ling-3.2-Air dense MLPs and group-limited 512-expert MoE blocks."""

import torch
import torch.nn.functional as F

from hyperflare_recipes.components.expert_overlap import ConcurrentSharedExpert
from hyperflare_recipes.components.mlp import PreparedMLP
from hyperflare_recipes.components.moe import PreparedRoutedExperts
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import register_finalizer

from .weights import LayerWeights


def group_limited_route(
    source: torch.Tensor,
    router_weight: torch.Tensor,
    expert_bias: torch.Tensor,
    *,
    n_group: int,
    topk_group: int,
    top_k: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact ``BailingMoeV3Gate`` FP32 sigmoid + bias group-limited top-k routing."""
    tokens = source.shape[0]
    num_experts = router_weight.shape[0]
    logits = F.linear(source.float(), router_weight)
    scores = torch.sigmoid(logits)
    biased = scores + expert_bias
    group_scores = biased.view(tokens, n_group, -1).topk(2, dim=-1).values.sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(tokens, n_group, num_experts // n_group)
        .reshape(tokens, num_experts)
    )
    masked = biased.masked_fill(~score_mask.bool(), float("-inf"))
    topk_idx = torch.topk(masked, k=top_k, dim=-1).indices
    selected = torch.gather(scores, dim=1, index=topk_idx)
    weights = (selected / (selected.sum(dim=-1, keepdim=True) + 1e-20)) * routed_scaling_factor
    return topk_idx.to(torch.int32), weights


class PreparedFeedForward:
    """Dense SwiGLU MLP (layers 0-1) or routed 512-expert MoE with overlapping shared expert."""

    def __init__(
        self,
        layer: int,
        weights: LayerWeights,
        config,
        source: torch.Tensor,
        *,
        mode: str,
        session,
        name: str,
        workspace,
        buffers: dict,
        overlap: bool = True,
    ):
        tokens, hidden = source.shape
        device = source.device
        c = config.text
        routed = weights.experts is not None
        inter = c["moe_shared_expert_intermediate_size"] if routed else c["intermediate_size"]
        limit = config.shared_swiglu_limit(layer) if routed else None
        swiglu_limit = float("inf") if limit is None else limit

        def buf(key: str, width: int, dtype=torch.bfloat16):
            role = "ling32.hidden_output" if key == "output" else "ffn." + key
            return declaration_tensor(buffers, role, (tokens, width), dtype, device)

        gate_up = buf("gate_up", 2 * inter)
        intermediate = buf("intermediate", inter)
        mlp_out = buf("shared_out" if routed else "output", hidden)

        self.mlp = PreparedMLP(
            source,
            weights.mlp,
            gate_up,
            intermediate,
            mlp_out,
            session=session,
            name=name + (".shared" if routed else ".dense"),
            workspace=workspace,
            swiglu_limit=swiglu_limit,
            round_silu=False,
            clamp_after_silu=True,
            activation_mode="auto",
        )
        self.experts = self.concurrent = None
        self.router_weight = weights.router
        self.expert_bias = weights.expert_bias
        self.n_group = int(c["n_group"])
        self.topk_group = int(c["topk_group"])
        self.top_k = int(c["num_experts_per_tok"])
        self.routed_scaling_factor = float(c["routed_scaling_factor"])

        if routed:
            record_numeric_tensor(weights.router)
            record_numeric_tensor(weights.expert_bias)
            route_ids = buf("route_ids", self.top_k, torch.int32)
            route_weights = buf("route_weights", self.top_k, torch.float32).fill_(1.0 / self.top_k)
            routed_out = buf("output", hidden)
            self.experts = PreparedRoutedExperts(
                source,
                weights.experts,
                route_ids,
                route_weights,
                routed_out,
                session=session,
                name=name + ".experts",
                workspace=workspace,
                a4_prefill=(mode == "prefill"),
            ).functional(session=session)
            if overlap and workspace is not None and workspace.overlap and tokens <= 256:
                self.concurrent = ConcurrentSharedExpert(
                    self.mlp,
                    parallel_plans=(self.experts.plan,),
                    session=session,
                    workspace=workspace,
                ).functional(session=session)

                def _finalize_overlap():
                    binding = getattr(self.experts, "binding", None)
                    self.concurrent.defer_launch = (
                        tokens <= 2 and binding is not None and binding.implementation == "micro"
                    )

                register_finalizer(session, _finalize_overlap, workspace=workspace)
            else:
                self.mlp.functional(session=session)
        else:
            self.mlp.functional(session=session)

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        if self.experts is None:
            return self.mlp(source)
        pending = self.concurrent.start(source) if self.concurrent is not None else None
        route_ids, route_weights = group_limited_route(
            source,
            self.router_weight,
            self.expert_bias,
            n_group=self.n_group,
            topk_group=self.topk_group,
            top_k=self.top_k,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        routed = self.experts(source, route_ids, route_weights)
        shared = self.concurrent.finish(pending, routed) if self.concurrent is not None else self.mlp(source)
        return routed + shared
