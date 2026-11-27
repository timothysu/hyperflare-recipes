# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Heterogeneous latent ReLU-squared experts with full-width shared experts."""

import torch

from hyperflare_recipes.components.expert_overlap import ConcurrentSharedExpert
from hyperflare_recipes.components.linear import UnquantizedProjection, prepare_projection
from hyperflare_recipes.components.mlp import PreparedUngatedMLP
from hyperflare_recipes.components.moe import PreparedRouter, PreparedRoutedExperts, RouterScratch
from ignition.components.collectives import PreparedAllReduce
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.examples import declaration_tensor


_FUSED_UP_ADD_MIN_TOKENS = 128


@torch.library.custom_op("flare::bf16_addmm_inplace", mutates_args=("accumulator",))
def _bf16_addmm_inplace(accumulator: torch.Tensor, source: torch.Tensor, weight_t: torch.Tensor) -> None:
    torch.addmm(accumulator, source, weight_t, out=accumulator)


@_bf16_addmm_inplace.register_fake
def _fake_bf16_addmm_inplace(accumulator, source, weight_t):
    pass


class PreparedLatentMoE:
    def __init__(self, weights, source, config, index, *, session, name, workspace,
                 shard=None, runtime=None, coordinator=None, num_valid_tokens=None, examples=None):
        c, tokens = config.text, source.shape[0]
        block = config.block_config(index)

        def empty(role, width, dtype=torch.bfloat16):
            return declaration_tensor(examples, "nemotron.target.moe." + role, (tokens, width), dtype, source.device)

        logits = empty("logits", block["n_routed_experts"], torch.float32)
        routes = RouterScratch.allocate(tokens, source.device, experts=block["n_routed_experts"],
            top_k=block["num_experts_per_tok"])
        self.router_projection = UnquantizedProjection(source, weights["router"], logits,
            session=session, name=name + ".router_logits")
        self.router = PreparedRouter(logits, routes, score_func="sigmoid", correction_bias=weights["correction"],
            renormalize=True, routed_scaling_factor=c["routed_scaling_factor"],
            num_valid_tokens=num_valid_tokens, session=session, name=name + ".route")
        latent, expert_output, output = (empty("latent", c["moe_latent_size"]),
            empty("expert_output", c["moe_latent_size"]), empty("routed_output", config.hidden))
        self.down = prepare_projection(source, weights["latent_down"], latent,
            session=session, name=name + ".latent_down", activation_mode="auto", workspace=workspace)
        self.experts = PreparedRoutedExperts(latent, weights["experts"], routes.ids, routes.weights, expert_output,
            session=session, name=name + ".experts", workspace=workspace)
        self.reduce = None if runtime is None else PreparedAllReduce(
            output, empty("reduced", config.hidden), runtime=runtime, coordinator=coordinator,
            session=session, name=name + ".reduce", has_addend=True)
        shared_width = config.local_shared_intermediate(shard)
        self.shared = PreparedUngatedMLP(source, weights["shared_up"], weights["shared_down"],
            empty("shared_projected", shared_width), empty("shared_intermediate", shared_width),
            empty("shared_output", config.hidden), activation="relu2",
            session=session, name=name + ".shared", workspace=workspace)
        self.concurrent_shared = None
        self.fuse_up_add = (
            tokens >= _FUSED_UP_ADD_MIN_TOKENS
            and isinstance(weights["latent_up"], torch.Tensor)
            and weights["latent_up"].dtype == torch.bfloat16
            and (self.reduce is None or not self.reduce.fused_addend)
        )
        if self.fuse_up_add:
            self.up = None
            self.up_weight = weights["latent_up"]
            record_numeric_tensor(self.up_weight)
        else:
            self.up = prepare_projection(expert_output, weights["latent_up"], output,
                session=session, name=name + ".latent_up", activation_mode="auto", workspace=workspace)
            self.up_weight = None
            if tokens <= 64 and isinstance(workspace, ScratchArena) and workspace.overlap:
                self.concurrent_shared = ConcurrentSharedExpert(self.shared,
                    parallel_plans=(self.router_projection.plan, self.router.plan, self.down.plan,
                                    self.experts.plan, self.up.plan),
                    session=session, workspace=workspace, defer_launch=1 < tokens <= 4)

    def functional(self, *, session):
        self.router_projection = self.router_projection.functional(session=session)
        self.router.functional(session=session)
        self.down.functional(session=session)
        self.experts.functional(session=session)
        if self.up is not None:
            self.up.functional(session=session)
        self.shared.functional(session=session)
        if self.concurrent_shared is not None:
            self.concurrent_shared.functional(session=session)
        if self.reduce is not None:
            self.reduce.functional(session=session)
        return self

    def __call__(self, source, metadata):
        pending = self.concurrent_shared.start(source) if self.concurrent_shared is not None else None
        logits = self.router_projection(source)
        routes = self.router(logits, num_valid_tokens=metadata.live_counts[:1])
        latent = self.down(source)
        expert_output = self.experts(latent, routes.ids, routes.weights)
        if self.fuse_up_add:
            shared = self.shared(source)
            _bf16_addmm_inplace(shared, expert_output, self.up_weight.t())
            return shared if self.reduce is None else self.reduce(shared)
        routed = self.up(expert_output)
        shared = (self.shared(source) if self.concurrent_shared is None
                  else self.concurrent_shared.finish(pending, routed))
        if self.reduce is None:
            return routed + shared
        return self.reduce(routed, addend=shared)
