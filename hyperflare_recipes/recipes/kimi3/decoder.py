# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 attention residuals and latent MoE composed into complete forwards."""

from dataclasses import dataclass

import torch
import triton

from ignition.components.collectives import (
    ConcurrentAllReduce,
    ConcurrentGatherShards,
    ConcurrentReduceScatter,
    PreparedChunkedAllGather,
    PreparedChunkedAllReduce,
    PreparedGatherShards,
    PreparedReduceScatter,
    PreparedRowToColAllToAll,
    row_shard_surfaces,
)
from hyperflare_recipes.components.embedding import PreparedShardedEmbedding
from hyperflare_recipes.components.linear import UnquantizedProjection
from hyperflare_recipes.components.moe import PreparedRouter, PreparedRoutedExperts, RouterScratch
from hyperflare_recipes.components.projection_overlap import ConcurrentProjectionChain
from hyperflare_recipes.components.column_pair_gather import PreparedColumnPairGather
from hyperflare_recipes.components.recurrent import DeltaMetadata, _commit_convolution
from hyperflare_recipes.components.vocabulary import PreparedVocabularyProjection
from hyperflare_recipes.components.kda_speculation import KDARecords, PreparedKDAReplay
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.persistent import persistent_copy, persistent_zeros
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import register_finalizer

from .attention import PreparedMLA
from .kda import PreparedKDA
from .pointwise import rms_norm_columns, rms_norm_rows, scale_columns, situ
from .projections import prepare_projection
from .residual import ADD, REPLACE, ResidualStream
from .routing import MAX_ROWS as ROUTE_ROWS, PreparedRouteGather


def _projection_plans(*projections):
    plans = []
    for projection in projections:
        for attr in ("plan", "quantizer_plan"):
            plan = getattr(projection, attr, None)
            if plan is not None:
                plans.append(plan)
    return tuple(plans)


def _is_dma_runtime(runtime):
    if runtime is None:
        return False
    from b12x.comm import pcie

    return isinstance(runtime, pcie.DmaAllReduce)


def _route_max_rows(runtime, config, world):
    if runtime is None:
        return 0
    from ignition.components.collectives import _island_row_shard

    if _is_dma_runtime(runtime):
        c = config.text
        cols = (c["routed_expert_hidden_size"] + 2 * c["num_experts"]) // world
        return runtime.shard_capacity // (cols * 2) if cols > 0 else 0
    if _island_row_shard(runtime):
        c = config.text
        cols = (c["routed_expert_hidden_size"] + 2 * c["num_experts"]) // world
        return (runtime._island_rs.max_elements // world) // cols if cols > 0 else 0
    return getattr(runtime, "max_batch_size", 0)


def _runtime_accepts_chunk(runtime, tensor):
    if runtime is None:
        return False
    from b12x.comm import pcie
    from ignition.components.collectives import _roce

    if _roce(runtime):
        return False
    if isinstance(runtime, pcie.DmaAllReduce):
        return runtime.should_allreduce(tensor)
    if getattr(runtime, "algorithm", None) == "oneshot":
        return runtime.for_stream().should_allreduce(tensor)
    return runtime.should_allreduce(tensor)


# A concurrent shared-expert MLP forks onto the auxiliary stream after its input is written and joins before its
# output is added; the explicit mutated outputs order both against the routed branch through compilation.
@torch.library.custom_op("flare::kimi3_start_shared_expert", mutates_args=("outputs", "workspace"))
def _start_shared_expert(source: torch.Tensor, weights: list[torch.Tensor], scales: list[torch.Tensor | None],
                         outputs: list[torch.Tensor], workspace: torch.Tensor, plans: list[int], auxiliary_stream: int) -> None:
    from hyperflare_recipes.components.sm103_projection import bf16_out, packed_out_workspace

    def project(value, index, output):
        if scales[index] is None:
            bf16_out(value, weights[index], output, plans[index])
        else:
            packed_out_workspace(value, weights[index], scales[index], None, None, output,
                                 workspace, "none", None, plans[index], "any")

    main = torch.cuda.current_stream(source.device)
    auxiliary = torch.cuda.ExternalStream(auxiliary_stream, device=source.device)
    auxiliary.wait_stream(main)
    gate_up, intermediate, projected = outputs
    with torch.cuda.stream(auxiliary):
        project(source, 0, gate_up)
        situ(gate_up, intermediate)
        project(intermediate, 1, projected)


@_start_shared_expert.register_fake
def _fake_start_shared_expert(source, weights, scales, outputs, workspace, plans, auxiliary_stream) -> None:
    pass


@torch.library.custom_op("flare::kimi3_join_shared_expert", mutates_args=("outputs", "workspace"))
def _join_shared_expert(source: torch.Tensor, routed: torch.Tensor, outputs: list[torch.Tensor],
                        workspace: torch.Tensor, auxiliary_stream: int) -> None:
    main = torch.cuda.current_stream(source.device)
    main.wait_stream(torch.cuda.ExternalStream(auxiliary_stream, device=source.device))


@_join_shared_expert.register_fake
def _fake_join_shared_expert(source, routed, outputs, workspace, auxiliary_stream) -> None:
    pass


@dataclass(frozen=True)
class DecoderStorage:
    hidden_columns: torch.Tensor
    latent_begin: torch.Tensor
    row_tuning: object = None
    dual_tc_flags: torch.Tensor | None = None

    @classmethod
    def allocate(cls, config, shard, device):
        n_a = config.text.get("routed_expert_hidden_size", 0) // shard.world_size
        n_b = config.text.get("num_experts", 0) // shard.world_size
        flags_n = max(64, (n_a + 15) // 16 + (n_b + 15) // 16)
        return cls(persistent_copy(torch.arange(*shard.partition(config.hidden), device=device)),
            persistent_copy(torch.tensor(shard.partition(config.text["routed_expert_hidden_size"])[:1],
                                         dtype=torch.int32, device=device)),
            dual_tc_flags=persistent_zeros((flags_n,), torch.uint32, device))


@dataclass(frozen=True)
class Scratch:
    activations: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, tokens, rows, config, world_size, device, *, examples=None):
        def tensor(role, shape):
            return declaration_tensor(examples, role, shape, torch.bfloat16, device)
        return cls(tensor("kimi.hidden_input", (tokens, config.hidden)),
                   tensor("kimi.hidden", (rows, config.hidden)),
                   tensor("kimi.local_logits", (rows, config.vocabulary // world_size)),
                   tensor("kimi.logits", (rows, config.vocabulary)))

    def rows(self, tokens, rows):
        return type(self)(self.activations[:tokens], self.hidden[:rows], self.local_logits[:rows], self.logits[:rows])


class PreparedMLP:
    """SiTU projections with invocation-local intermediates and explicit side-stream joins."""

    def __init__(self, weights, source, *, session, name, workspace=None, parallel_plans=(), examples=None):
        rows, device = source.shape[0], source.device
        width = (weights["gate_up"].shape[0] if isinstance(weights["gate_up"], torch.Tensor)
                 else weights["gate_up"].out_features)
        examples = {} if examples is None else examples
        def tensor(role, columns):
            key = "kimi.hidden_output" if role == "output" else "kimi.mlp." + role
            return declaration_tensor(examples, key, (rows, columns), source.dtype, device)
        intermediate = tensor("intermediate", width // 2)
        self.gate_up = prepare_projection(source, weights["gate_up"], tensor("gate_up", width),
            session=session, name=name + ".gate_up", workspace=workspace)
        self.down = prepare_projection(intermediate, weights["down"], tensor("output", source.shape[1]),
            session=session, name=name + ".down", workspace=workspace)
        self.intermediate_width = width // 2
        self.concurrent = bool(parallel_plans) and workspace is not None and workspace.overlap
        if self.concurrent:
            self.stream = workspace.auxiliary_stream
            self.stream_handle = self.stream.cuda_stream
            self.weights, self.scales = [], []
            for projection in (self.gate_up, self.down):
                if isinstance(projection, UnquantizedProjection):
                    self.weights.append(projection.weight)
                    self.scales.append(None)
                else:
                    values, scales, _ = projection.parts
                    self.weights.append(values)
                    self.scales.append(scales.view(torch.uint8))
                for main in parallel_plans:
                    workspace.register_parallel(main, projection.plan)
            register_finalizer(session, self.refresh)
        self.gate_up.functional(session=session)
        self.down.functional(session=session)
        self.all_plans = _projection_plans(self.gate_up, self.down)

    def refresh(self):
        self.workspace_nbytes = max(p.plan.memory_requirements().scratch_nbytes
                                   for p in (self.gate_up, self.down))

    def __call__(self, source):
        gate_up = self.gate_up(source)
        intermediate = torch.empty((source.shape[0], self.intermediate_width), dtype=source.dtype, device=source.device)
        situ(gate_up, intermediate)
        return self.down(intermediate)

    def start(self, source):
        outputs = [torch.empty((source.shape[0], width), dtype=source.dtype, device=source.device)
                   for width in (2 * self.intermediate_width, self.intermediate_width, source.shape[1])]
        workspace = torch.empty(self.workspace_nbytes, dtype=torch.uint8, device=source.device)
        _start_shared_expert(source, self.weights, self.scales, outputs, workspace,
            [self.gate_up.plan.handle, self.down.plan.handle], self.stream_handle)
        return source, outputs, workspace

    def finish(self, pending, routed):
        source, outputs, workspace = pending
        _join_shared_expert(source, routed, outputs, workspace, self.stream_handle)
        return outputs[-1]


class PreparedLatentMoE:
    def __init__(self, weights, source, config, shard, *, runtime, router_runtime, coordinator,
                 session, name, workspace, storage, route_runtime=None, prefill_route_runtime=None, examples=None,
                 chunk_rows=None, row_shard_prefill=False, expert_rs_runtime=None, split_moe=False):
        tokens, hidden = source.shape
        c, device = config.text, source.device
        latent = c["routed_expert_hidden_size"]
        examples = {} if examples is None else examples
        rs_runtime = runtime if expert_rs_runtime is None else expert_rs_runtime
        candidate_chunk = chunk_rows
        can_pipeline = (
            prefill_route_runtime is not None
            and candidate_chunk is not None
            and 0 < candidate_chunk < tokens
            and tokens == 2 * candidate_chunk
            and workspace is not None
            and workspace.overlap
            and rs_runtime is not None
        )
        use_row_shard = False
        if can_pipeline:
            expert_probe = declaration_tensor(
                examples, "kimi.moe.expert_output_chunk", (candidate_chunk, latent), source.dtype, device,
            )
            if row_shard_prefill and shard.world_size > 1 and candidate_chunk % shard.world_size == 0:
                shard_probe = declaration_tensor(
                    examples, "kimi.moe.expert_shard",
                    (candidate_chunk // shard.world_size, latent), source.dtype, device,
                )
                up_probe = declaration_tensor(
                    examples, "kimi.moe.up_input", (candidate_chunk, latent // shard.world_size), source.dtype, device,
                )
                use_row_shard = (
                    PreparedReduceScatter.fits(rs_runtime, expert_probe, shard_probe)
                    and PreparedRowToColAllToAll.fits(runtime, shard_probe, up_probe)
                )
                can_pipeline = use_row_shard
            else:
                can_pipeline = _runtime_accepts_chunk(rs_runtime, expert_probe)
        self.pipelined = can_pipeline
        self.row_shard_prefill = use_row_shard
        self.split_moe = bool(split_moe and self.pipelined and self.row_shard_prefill)
        work_tokens = candidate_chunk if self.pipelined else tokens
        work_source = source[:work_tokens]

        def buffer(role, width, dtype=torch.bfloat16):
            key = "kimi.hidden_output" if role == "output" else "kimi.moe." + role
            return declaration_tensor(examples, key, (work_tokens, width), dtype, device)

        def full_buffer(role, width, dtype=torch.bfloat16):
            return declaration_tensor(examples, "kimi.moe." + role, (tokens, width), dtype, device)

        self.hidden_columns, self.latent_begin = storage.hidden_columns, storage.latent_begin
        self.world, self.latent_width = shard.world_size, latent
        self.latent_norm, self.eps = weights["latent_norm"], c["rms_norm_eps"]
        latent_chunk = buffer("latent_chunk" if self.pipelined else "latent", latent)
        latent_full = full_buffer("latent", latent) if (self.pipelined and not self.split_moe) else latent_chunk
        expert_output_chunk = buffer("expert_output_chunk" if self.pipelined else "expert_output", latent)
        expert_output_full = (
            full_buffer("expert_output", latent) if (self.pipelined and not self.split_moe) else expert_output_chunk
        )
        up_input = buffer("up_input", latent // shard.world_size)
        self.up = prepare_projection(up_input, weights["latent_up"], buffer("output", hidden),
            session=session, name=name + ".latent_up", workspace=workspace).functional(session=session)
        self.expert_reduce = self.expert_rs = self.expert_a2a = None
        if self.row_shard_prefill:
            expert_shard = declaration_tensor(
                examples, "kimi.moe.expert_shard",
                (work_tokens // shard.world_size, latent), source.dtype, device,
            )
            self.expert_rs = PreparedReduceScatter(
                expert_output_chunk, expert_shard, runtime=rs_runtime, coordinator=coordinator,
                session=session, name=name + ".expert_rs",
            )
            self._concurrent_expert_rs_op = (
                PreparedReduceScatter(
                    expert_output_chunk, expert_shard, runtime=rs_runtime, coordinator=coordinator,
                    session=session, name=name + ".expert_rs.concurrent", concurrent=True,
                )
                if self.pipelined
                else None
            )
            self.expert_a2a = PreparedRowToColAllToAll(
                expert_shard, up_input, runtime=runtime, coordinator=coordinator,
                session=session, name=name + ".expert_a2a",
            )
        else:
            self.expert_reduce = PreparedChunkedAllReduce(expert_output_chunk, buffer("latent_sum", latent),
                runtime=rs_runtime, coordinator=coordinator, session=session,
                name=name + ".expert_reduce")
        self.row_norm = None
        if not self.row_shard_prefill and 1 <= work_tokens <= 64 and latent == 3584 and storage.row_tuning is not None:
            from .row_reductions import RowGeometry
            self.row_norm = storage.row_tuning.register(self.expert_reduce, RowGeometry(latent, rows=work_tokens))
        local_logits = buffer("local_router", weights["router"].shape[0], torch.float32)
        top_k = c["num_experts_per_token"]
        route_buf = buffer if self.split_moe else full_buffer
        routes = RouterScratch(route_buf("selected_logits", top_k, torch.float32),
            route_buf("route_ids", top_k, torch.int32),
            route_buf("route_weights", top_k, torch.float32).fill_(1 / top_k))
        self.route_gather = self.prefill_route_gather = self.prefill_route_gather_sm = None
        self.dual_down_router = None
        use_dual = False
        if route_runtime is not None and work_tokens <= ROUTE_ROWS:
            import importlib.util
            from hyperflare_recipes.components import sm103_projection
            w_down = weights["latent_down_rows"]
            w_router = weights["router"]
            use_dual = (
                1 <= work_tokens <= 32
                and hidden % 64 == 0
                and device.type == "cuda"
                and not sm103_projection.is_sm103(device)
                and work_source.dtype == torch.bfloat16
                and isinstance(w_down, torch.Tensor)
                and isinstance(w_router, torch.Tensor)
                and w_down.dtype == torch.bfloat16
                and w_router.dtype == torch.bfloat16
                and w_down.is_contiguous()
                and w_router.is_contiguous()
                and w_down.data_ptr() % 16 == 0
                and w_router.data_ptr() % 16 == 0
                and importlib.util.find_spec("b12x.gemm.bf16_gemv._dual_tc") is not None
            )
        if use_dual:
            from b12x.gemm.bf16_gemv._dual_tc import (
                compile_dual_tc_projection,
                dual_tc_flags_numel,
                dual_tc_scratch_numel,
                dual_tc_split_k,
            )
            w_down = weights["latent_down_rows"]
            w_router = weights["router"]
            record_numeric_tensor(w_down)
            record_numeric_tensor(w_router)
            n_a, n_b = int(w_down.shape[0]), int(w_router.shape[0])
            split_k = dual_tc_split_k(n_a, n_b, hidden, work_tokens)
            flags_needed = dual_tc_flags_numel(n_a, n_b)
            flags = storage.dual_tc_flags
            if flags is None or flags.numel() < flags_needed or flags.device != device:
                flags = persistent_zeros((flags_needed,), torch.uint32, device)
            # Precompile on the active CUDA context so native export captures the cubin.
            self._dual_program = compile_dual_tc_projection(
                int(device.index or 0), work_tokens, split_k, n_a, n_b, hidden
            )
            self.dual_down_router = (
                w_down,
                w_router,
                flags,
                n_a,
                n_b,
                work_tokens,
                split_k,
                dual_tc_scratch_numel(n_a, n_b, work_tokens, split_k),
            )
            self.router_projection = self.down = None
            down_output = buffer("local_down", latent // shard.world_size)
            self.route_gather = PreparedRouteGather(down_output, local_logits, weights["correction"],
                latent_full, routes.weights, routes.ids, runtime=route_runtime, session=session,
                coordinator=coordinator, name=name + ".route_gather")
        else:
            self.router_projection = UnquantizedProjection(work_source, weights["router"], local_logits,
                session=session, name=name + ".router").functional(session=session)
            if route_runtime is not None and work_tokens <= ROUTE_ROWS:
                down_output = buffer("local_down", latent // shard.world_size)
                self.down = prepare_projection(work_source, weights["latent_down_rows"], down_output,
                    session=session, name=name + ".latent_down_rows", workspace=workspace)
                self.route_gather = PreparedRouteGather(down_output, local_logits, weights["correction"],
                    latent_full, routes.weights, routes.ids, runtime=route_runtime, session=session,
                    coordinator=coordinator, name=name + ".route_gather")
            else:
                logits_chunk = buffer("router_chunk" if self.pipelined else "router", c["num_experts"], torch.float32)
                logits_full = (
                    full_buffer("router", c["num_experts"], torch.float32)
                    if (self.pipelined and not self.split_moe)
                    else logits_chunk
                )
                if prefill_route_runtime is not None:
                    from ignition.components.collectives import _island_row_shard

                    down_output = buffer("local_down", latent // shard.world_size)
                    self.down = prepare_projection(work_source, weights["latent_down_rows"], down_output,
                        session=session, name=name + ".latent_down_rows", workspace=workspace)
                    paired_buffers = examples.setdefault("kimi.moe.paired_columns", {})
                    self.prefill_route_gather = PreparedColumnPairGather(down_output, local_logits, latent_chunk, logits_chunk,
                        runtime=prefill_route_runtime, session=session, coordinator=coordinator,
                        name=name + ".latent_router_gather",
                        buffers=paired_buffers,
                        concurrent=self.pipelined)
                    if self.row_shard_prefill and self.pipelined and _island_row_shard(prefill_route_runtime):
                        self.prefill_route_gather_sm = PreparedColumnPairGather(down_output, local_logits, latent_chunk, logits_chunk,
                            runtime=prefill_route_runtime, session=session, coordinator=coordinator,
                            name=name + ".latent_router_gather.sm",
                            buffers=paired_buffers,
                            concurrent=False)
                else:
                    self.down = prepare_projection(buffer("source", hidden // shard.world_size), weights["latent_down"],
                        buffer("down", latent), session=session, name=name + ".latent_down", workspace=workspace)
                    self.down_reduce = PreparedChunkedAllReduce(self.down.output, latent_full, runtime=runtime,
                        coordinator=coordinator, session=session, name=name + ".down_reduce").functional(session=session)
                    self.local_experts = local_logits.shape[1]
                    self.router_width = (self.local_experts + 31) // 32 * 32
                    router_input = buffer("router_input", self.router_width, torch.float32)
                    gathered = buffer("gathered_router", self.router_width * shard.world_size, torch.float32)
                    self.router_gather = PreparedChunkedAllGather(router_input.view(torch.bfloat16), gathered.view(torch.bfloat16),
                        runtime=router_runtime, coordinator=coordinator, session=session,
                        name=name + ".router_gather").functional(session=session)
                self.router = PreparedRouter(logits_full, routes, score_func="sigmoid", correction_bias=weights["correction"],
                    renormalize=True, routed_scaling_factor=c["routed_scaling_factor"], session=session,
                    name=name + ".route").functional(session=session)
        # Split MoE overlaps the cooperative persistent W4A16 kernel with auxiliary-stream
        # collectives; reserving 4 SMs keeps the cooperative grid below the full SM count so
        # concurrent collective CTAs can always be scheduled without barrier deadlock.
        self.experts = PreparedRoutedExperts(latent_full, weights["experts"], routes.ids, routes.weights,
            expert_output_full, workspace=workspace, session=session, name=name + ".experts",
            reserve_sms=4 if self.split_moe else 0).functional(session=session)
        shared_parallel = ()
        if self.route_gather is not None or (self.prefill_route_gather is not None and not self.pipelined):
            shared_parallel = ((self.experts.plan, self.up.plan) if self.dual_down_router is not None
                               else (self.router_projection.plan, self.down.plan, self.experts.plan, self.up.plan))
        self.shared = PreparedMLP(weights["shared"], work_source, session=session, name=name + ".shared", workspace=workspace,
            parallel_plans=shared_parallel, examples=examples)
        self.concurrent_expert_reduce = None
        self.concurrent_expert_rs = None
        self.all_plans = ()
        if self.pipelined:
            self.all_plans = (
                *_projection_plans(
                    self.router_projection,
                    self.down,
                    self.up,
                    self.shared.gate_up,
                    self.shared.down,
                ),
                self.router.plan,
                self.experts.plan,
            )
            self.prefill_route_gather.enable_concurrent(
                workspace=workspace, session=session, parallel_plans=self.all_plans)
            if self.prefill_route_gather_sm is not None:
                self.prefill_route_gather_sm.enable_concurrent(
                    workspace=workspace, session=session, parallel_plans=self.all_plans)
            if self.row_shard_prefill:
                self.concurrent_expert_rs = ConcurrentReduceScatter(
                    self._concurrent_expert_rs_op, workspace=workspace, session=session,
                    parallel_plans=self.all_plans)
            else:
                self.concurrent_expert_reduce = ConcurrentAllReduce(
                    self.expert_reduce.parts[0], workspace=workspace, session=session,
                    parallel_plans=self.all_plans)
        if self.expert_reduce is not None:
            self.expert_reduce.functional(session=session)
        if self.concurrent_expert_reduce is not None:
            self.concurrent_expert_reduce.functional(session=session)
        if self.expert_rs is not None:
            self.expert_rs.functional(session=session)
        if self.concurrent_expert_rs is not None:
            self.concurrent_expert_rs.functional(session=session)
        if self.expert_a2a is not None:
            self.expert_a2a.functional(session=session)
        if self.prefill_route_gather is not None:
            self.prefill_route_gather.functional(session=session)
        if self.prefill_route_gather_sm is not None:
            self.prefill_route_gather_sm.functional(session=session)
        self.concurrent_down = (ConcurrentProjectionChain((self.down,), parallel_plans=(self.router_projection.plan,),
            session=session, workspace=workspace).functional(session=session)
            if self.dual_down_router is None and self.route_gather is not None and workspace is not None and workspace.overlap else None)
        if self.dual_down_router is None and self.concurrent_down is None:
            self.down.functional(session=session)

    def post_moe(self, latent_sum, shared):
        up_input = torch.empty((latent_sum.shape[0], self.latent_width // self.world),
                               dtype=latent_sum.dtype, device=latent_sum.device)
        rms_norm_columns(latent_sum, self.latent_norm, up_input, self.eps, begin=self.latent_begin)
        return self.up(up_input) + shared

    def norm_expert_shard(self, shard_sum):
        normed = torch.empty_like(shard_sum)
        rms_norm_rows(shard_sum, normed, self.eps)
        return normed

    def post_moe_sharded(self, col_normed, shared):
        up_input = torch.empty_like(col_normed)
        scale_columns(col_normed, self.latent_norm, up_input)
        return self.up(up_input) + shared

    def __call__(self, source, *, fuse_add=False):
        dual = getattr(self, "dual_down_router", None)
        if dual is not None:
            from b12x.gemm.bf16_gemv._dual_tc import bf16_dual_gemv_out
            w_down, w_router, flags, n_a, n_b, max_rows, split_k, scratch_numel = dual
            down = torch.empty((source.shape[0], n_a), dtype=torch.bfloat16, device=source.device)
            logits = torch.empty((source.shape[0], n_b), dtype=torch.float32, device=source.device)
            scratch = torch.empty((scratch_numel,), dtype=torch.float32, device=source.device)
            bf16_dual_gemv_out(source, w_down, w_router, down, logits, scratch, flags, max_rows, split_k)
        else:
            pending_down = None if self.concurrent_down is None else self.concurrent_down.start(source)
            logits = self.router_projection(source)
            if self.route_gather is not None:
                down = (self.down(source) if self.concurrent_down is None else
                        self.concurrent_down.finish(pending_down, logits))
            elif self.prefill_route_gather is not None:
                down = self.down(source)
        pending_shared = self.shared.start(source) if self.shared.concurrent else None
        if self.route_gather is not None:
            latent, ids, route_weights = self.route_gather(down, logits)
        else:
            if self.prefill_route_gather is not None:
                latent, logits = self.prefill_route_gather(down, logits)
            else:
                router_input = torch.zeros((source.shape[0], self.router_width), dtype=torch.float32, device=source.device)
                router_input[:, :self.local_experts].copy_(logits)
                gathered = self.router_gather(router_input.view(torch.bfloat16)).view(torch.float32)
                logits = gathered.view(source.shape[0], self.world, self.router_width)[:, :, :self.local_experts].reshape(source.shape[0], -1)
                latent = self.down_reduce(self.down(source.index_select(1, self.hidden_columns)))
            routes = self.router(logits)
            ids, route_weights = routes.ids, routes.weights
        partial = self.experts(latent, ids, route_weights)
        if self.row_norm is None:
            latent_sum = self.expert_reduce(partial)
            up_input = torch.empty((source.shape[0], self.latent_width // self.world), dtype=source.dtype, device=source.device)
            rms_norm_columns(latent_sum, self.latent_norm, up_input, self.eps, begin=self.latent_begin)
        else:
            up_input = self.row_norm.norm(partial, self.latent_norm, self.latent_begin, self.eps)
        output = self.up(up_input)
        shared = self.shared.finish(pending_shared, output) if self.shared.concurrent else self.shared(source)
        if fuse_add:
            return output, shared
        return output + shared


class PreparedDecoder:
    """Compose Kimi layers with local residuals and verification records."""

    def __init__(self, weights, scratch, metadata, pool, *, mode, runtime, vocabulary_runtime,
                 embedding_runtime, router_runtime, vision_state, coordinator, session, name,
                 workspace, buffers, kda_records=None, route_runtime=None, context=None,
                 latent_runtimes=(), scoring=None, prefill_route_runtime=None, examples=None,
                 row_shard_prefill=False, prefill_runtime=None):
        if mode not in ("prefill", "decode", "verify") or mode == "verify" and kda_records is None:
            raise ValueError("Kimi verification requires candidate recording storage")
        self.weights, self.scratch, self.pool = weights, scratch, pool
        self.mode = mode
        self.vision_state = vision_state if mode == "prefill" else None
        config, source = weights.config, scratch.activations
        tokens, hidden = source.shape
        self.tokens, self.hidden = tokens, hidden
        self.shared_index_type, self.index_shards = None, None
        if mode in ("decode", "verify") and context is not None and "dsa" in config.layers:
            from hyperflare_recipes.recipes.kimi31.indexer import SharedIndexMetadata

            self.shared_index_type, self.index_shards = SharedIndexMetadata, context.shards
        self.sequences = metadata.starts.numel()
        self.verification_columns = 8 if mode == "verify" else 1
        examples = {} if examples is None else examples
        local_embedding = declaration_tensor(examples, "embedding", (tokens, weights.embedding.shape[1]), source.dtype, source.device)
        self.embedding = PreparedShardedEmbedding(weights.embedding, metadata.token_ids, local_embedding, source,
            runtime=embedding_runtime, coordinator=coordinator, num_rows=metadata.live_counts[:1],
            session=session, name=name + ".embedding").functional(session=session, dynamic_inputs=True)
        delta = DeltaMetadata.allocate(self.sequences, source.device, verification_columns=self.verification_columns)
        self.layers = []
        self.residual_blocks = -(-len(weights.layers) // config.text["attn_res_block_size"])
        self.auxiliary_layers = config.draft.auxiliary_layers if weights.draft is not None else ()
        self.auxiliary_stream = config.draft.auxiliary_stream if self.auxiliary_layers else None
        self.auxiliary = (None if not self.auxiliary_layers else declaration_tensor(examples, "auxiliary",
            (tokens, len(self.auxiliary_layers) * hidden), source.dtype, source.device))
        world = weights.shard.world_size
        rs_runtime = (
            prefill_runtime
            if (prefill_runtime is not None and row_shard_prefill and mode == "prefill")
            else runtime
        )
        pipeline_chunk = None
        reduce_step = None
        if (mode == "prefill" and rs_runtime is not None and prefill_route_runtime is not None
                and workspace is not None and workspace.overlap and tokens >= 2048):
            step = tokens
            while step > 1 and not _runtime_accepts_chunk(rs_runtime, source[:step]):
                step = (step + 1) // 2
            max_route_rows = _route_max_rows(prefill_route_runtime, config, world)
            latent = config.text.get("routed_expert_hidden_size", hidden)
            half = tokens // 2
            # Split prefill projections and collectives into two halves (up to 2048 rows).
            if (0 < step <= half <= 2048
                    and tokens % 2 == 0
                    and half % step == 0
                    and max_route_rows > 0
                    and half % min(half, max_route_rows) == 0
                    and half * latent <= step * hidden):
                pipeline_chunk = half
                reduce_step = step
        self.pipeline_chunk = pipeline_chunk
        self.reduce_step = reduce_step
        self.subchunks = 0 if (pipeline_chunk is None or reduce_step is None) else pipeline_chunk // reduce_step
        self.row_shard_prefill = False
        self.shard_rows = None
        self.final_gather = None
        if (
            row_shard_prefill
            and mode == "prefill"
            and scoring is None
            and pipeline_chunk is not None
            and reduce_step is not None
            and tokens // pipeline_chunk == 2
            and world > 1
            and reduce_step % world == 0
            and row_shard_surfaces(rs_runtime)
        ):
            sub_st = reduce_step // world
            half_st = pipeline_chunk // world
            latent = config.text["routed_expert_hidden_size"]
            shard_sub = declaration_tensor(examples, "kimi.row_shard.sub", (sub_st, hidden), source.dtype, source.device)
            full_sub = source[:reduce_step]
            moe_ok = True
            if any("mlp" not in w for w in weights.layers):
                exp_full = declaration_tensor(examples, "kimi.moe.expert_output_chunk", (pipeline_chunk, latent), source.dtype, source.device)
                exp_sh = declaration_tensor(examples, "kimi.moe.expert_shard", (half_st, latent), source.dtype, source.device)
                up_in = declaration_tensor(examples, "kimi.moe.up_input", (pipeline_chunk, latent // world), source.dtype, source.device)
                moe_ok = (
                    PreparedReduceScatter.fits(rs_runtime, exp_full, exp_sh)
                    and PreparedRowToColAllToAll.fits(runtime, exp_sh, up_in)
                )
            if (
                moe_ok
                and PreparedReduceScatter.fits(rs_runtime, full_sub, shard_sub)
                and PreparedGatherShards.fits(rs_runtime, shard_sub, full_sub)
            ):
                self.row_shard_prefill = True
                rank = weights.shard.rank
                self.shard_rows = tuple(
                    persistent_copy(torch.cat([
                        torch.arange(
                            half_idx * pipeline_chunk + j * reduce_step + rank * sub_st,
                            half_idx * pipeline_chunk + j * reduce_step + (rank + 1) * sub_st,
                            dtype=torch.int64,
                            device=source.device,
                        )
                        for j in range(self.subchunks)
                    ]))
                    for half_idx in range(2)
                )
                total_sub = declaration_tensor(examples, "kimi.hidden_output", (reduce_step, hidden), source.dtype, source.device)
                self.final_gather = PreparedGatherShards(
                    shard_sub, total_sub, runtime=rs_runtime, coordinator=coordinator,
                    session=session, name=name + ".final_gather",
                ).functional(session=session)
        use_split_moe = self.row_shard_prefill
        self.split_kda = self.row_shard_prefill and self.sequences == 1 and use_split_moe
        for index, w in enumerate(weights.layers):
            prefix, kind = f"{name}.layer{index}", config.layers[index]
            kwargs = dict(mode=mode, session=session, name=prefix + ".attention", workspace=workspace,
                          buffers=examples, chunk_rows=pipeline_chunk)
            if kind == "kda":
                kda_source = source[:pipeline_chunk] if self.split_kda else source
                kda_kwargs = {**kwargs, "chunk_rows": None} if self.split_kda else kwargs
                attention = PreparedKDA(w["attention"], kda_source, pool.recurrent(index), delta,
                    eps=config.text["rms_norm_eps"], records=kda_records.layer(index) if mode == "verify" else None,
                    **kda_kwargs)
            else:
                attention_type, extra = PreparedMLA, {}
                if kind == "dsa":
                    from hyperflare_recipes.recipes.kimi31.attention import PreparedDSA
                    attention_type, extra = PreparedDSA, dict(index_cache=pool.indexer_pages(index))
                attention = attention_type(w["attention"], source, metadata, pool.attention(index),
                    eps=config.mla_norm_eps, context=context,
                    peers=None if context is None else context.peers[index], coordinator=coordinator,
                    latents=next((r for r in latent_runtimes if tokens <= r.max_batch_size),
                                 latent_runtimes[-1] if latent_runtimes else None), **extra, **kwargs)
            if "mlp" in w:
                mlp_source = source if pipeline_chunk is None else source[:pipeline_chunk]
                mlp = PreparedMLP(w["mlp"], mlp_source, session=session, name=prefix + ".mlp", workspace=workspace,
                    examples=examples)
            else:
                mlp = PreparedLatentMoE(w, source, config, weights.shard, runtime=runtime,
                    router_runtime=router_runtime, route_runtime=route_runtime if mode != "prefill" else None,
                    prefill_route_runtime=prefill_route_runtime if mode == "prefill" else None,
                    coordinator=coordinator, session=session, name=prefix + ".moe", workspace=workspace, storage=buffers,
                    examples=examples, chunk_rows=pipeline_chunk, row_shard_prefill=self.row_shard_prefill,
                    expert_rs_runtime=rs_runtime if self.row_shard_prefill else None, split_moe=use_split_moe)
            total = declaration_tensor(examples, "kimi.hidden_output", source.shape, source.dtype, source.device)
            reduce = None
            concurrent_reduce = None
            row_shard_ops = None
            if self.row_shard_prefill:
                sub_st = reduce_step // world
                shard_sub = declaration_tensor(examples, "kimi.row_shard.sub", (sub_st, hidden), source.dtype, source.device)
                n_sub = 2 * self.subchunks
                rs_parts = tuple(
                    PreparedReduceScatter(
                        source[j * reduce_step:(j + 1) * reduce_step], shard_sub,
                        runtime=rs_runtime, coordinator=coordinator, session=session,
                        name=f"{prefix}.rs.{j}",
                    )
                    for j in range(n_sub)
                )
                ag_parts = tuple(
                    PreparedGatherShards(
                        shard_sub, total[j * reduce_step:(j + 1) * reduce_step],
                        runtime=rs_runtime, coordinator=coordinator, session=session,
                        name=f"{prefix}.ag.{j}",
                    )
                    for j in range(n_sub)
                )
                crs_ops = tuple(
                    PreparedReduceScatter(
                        source[j * reduce_step:(j + 1) * reduce_step], shard_sub,
                        runtime=rs_runtime, coordinator=coordinator, session=session,
                        name=f"{prefix}.crs.{j}", concurrent=True,
                    )
                    for j in range(n_sub)
                )
                cag_ops = tuple(
                    PreparedGatherShards(
                        shard_sub, total[j * reduce_step:(j + 1) * reduce_step],
                        runtime=rs_runtime, coordinator=coordinator, session=session,
                        name=f"{prefix}.cag.{j}", concurrent=True,
                    )
                    for j in range(n_sub)
                )
                attn_plans = (
                    attention.all_plans()
                    if hasattr(attention, "all_plans")
                    else (*attention.input_plans(), *attention.output_plans())
                )
                rs_parallel = (*attn_plans, *mlp.all_plans)
                ag_parallel = (*attn_plans, *mlp.all_plans)
                crs = tuple(
                    ConcurrentReduceScatter(p, workspace=workspace, session=session, parallel_plans=rs_parallel).functional(session=session)
                    for p in crs_ops
                )
                cag = tuple(
                    ConcurrentGatherShards(p, workspace=workspace, session=session, parallel_plans=ag_parallel).functional(session=session)
                    for p in cag_ops
                )
                crs_sm = tuple(
                    ConcurrentReduceScatter(p, workspace=workspace, session=session, parallel_plans=rs_parallel).functional(session=session)
                    for p in rs_parts
                )
                cag_sm = tuple(
                    ConcurrentGatherShards(p, workspace=workspace, session=session, parallel_plans=ag_parallel).functional(session=session)
                    for p in ag_parts
                )
                if isinstance(mlp, PreparedLatentMoE) and mlp.prefill_route_gather is not None and mlp.prefill_route_gather.concurrent is not None:
                    for item in mlp.prefill_route_gather.concurrent:
                        for local_plan in attn_plans:
                            workspace.register_parallel(item.gather.plan, local_plan)
                row_shard_ops = (rs_parts, ag_parts, crs, cag, crs_sm, cag_sm)
            else:
                reduce = PreparedChunkedAllReduce(source, total, runtime=runtime, coordinator=coordinator,
                    session=session, name=prefix + ".reduce")
                mlp_ready = isinstance(mlp, PreparedMLP) or (isinstance(mlp, PreparedLatentMoE) and mlp.pipelined)
                if (pipeline_chunk is not None and reduce_step is not None and mlp_ready
                        and len(reduce.parts) == tokens // reduce_step
                        and all(end - begin == reduce_step for begin, end in reduce.bounds)):
                    layer_parallel_plans = (*attention.output_plans(), *mlp.all_plans)
                    concurrent_reduce = tuple(
                        ConcurrentAllReduce(part, workspace=workspace, session=session,
                            parallel_plans=layer_parallel_plans)
                        for part in reduce.parts
                    )
                reduce.functional(session=session)
                if concurrent_reduce is not None:
                    for cr in concurrent_reduce:
                        cr.functional(session=session)
            self.layers.append((w, attention, mlp, reduce, concurrent_reduce, row_shard_ops))
        for index in range(1, len(self.layers)):
            _, _, prev_mlp, _, prev_cr, prev_rs_ops = self.layers[index - 1]
            _, next_attn, _, _, next_cr, next_rs_ops = self.layers[index]
            next_in_plans = (
                next_attn.all_plans()
                if hasattr(next_attn, "all_plans")
                else next_attn.input_plans()
            )
            if prev_rs_ops is not None and next_rs_ops is not None:
                _, _, prev_crs, _, prev_crs_sm, _ = prev_rs_ops
                _, _, _, next_cag, _, next_cag_sm = next_rs_ops
                for crs in (*prev_crs, *prev_crs_sm):
                    for local_plan in next_in_plans:
                        workspace.register_parallel(crs.reduce.plan, local_plan)
                if isinstance(prev_mlp, PreparedLatentMoE) and prev_mlp.concurrent_expert_rs is not None:
                    for local_plan in next_in_plans:
                        workspace.register_parallel(prev_mlp.concurrent_expert_rs.reduce.plan, local_plan)
                for cag in (*next_cag, *next_cag_sm):
                    for local_plan in prev_mlp.all_plans:
                        workspace.register_parallel(cag.gather.plan, local_plan)
                continue
            if prev_cr is None or next_cr is None or not next_in_plans:
                continue
            for cr in prev_cr:
                for local_plan in next_in_plans:
                    workspace.register_parallel(cr.reduce.plan, local_plan)
            if isinstance(prev_mlp, PreparedLatentMoE) and prev_mlp.concurrent_expert_reduce is not None:
                for local_plan in next_in_plans:
                    workspace.register_parallel(prev_mlp.concurrent_expert_reduce.reduce.plan, local_plan)
        self.row_mixes = None
        if 1 <= tokens <= 64 and hidden == 7168 and buffers.row_tuning is not None:
            from .row_reductions import hidden_row_choices
            self.row_mixes = hidden_row_choices(buffers.row_tuning,
                [(w, a, m, r) for w, a, m, r, _, _ in self.layers], config.text["attn_res_block_size"],
                self.auxiliary_layers if self.auxiliary_stream == "attn_res" else (), self.residual_blocks, rows=tokens)
        self.scoring = None if scoring is None else scoring.prepare(source, weights.head,
            vocabulary=config.text["vocab_size"], session=session, coordinator=coordinator, name=name + ".prompt")
        self.compact_head = scratch.hidden.shape[0] > tokens and scoring is None
        head_rows = tokens if self.compact_head else scratch.hidden.shape[0]
        self.head = PreparedVocabularyProjection(scratch.hidden[:head_rows], weights.head,
            scratch.local_logits[:head_rows], scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head",
            row_indices=metadata.sample_indices if self.compact_head else None).functional(
            session=session, dynamic_indices=True)
        self.replay = None
        if mode == "verify":
            indices = kda_records.layers
            states = [pool.recurrent(i).recurrent[:kda_records.corrections.shape[1]] for i in indices]
            self.record_layers = indices
            self.record_shapes = (tuple(kda_records.corrections.shape), tuple(kda_records.keys_and_gates.shape))
            self.record_constants = (kda_records.A_log, kda_records.dt_bias)
            self.replay = PreparedKDAReplay(pool.backing, states, kda_records, self.layers[indices[0]][1].plan,
                session=session).functional()
            regions = [pool.layout.region(f"layer{i}.convolution") for i in indices]
            self.convolution_offsets = persistent_copy(torch.tensor([r.offset // 2 for r in regions],
                dtype=torch.int64, device=source.device))
            self.convolution_shape = regions[0].shape

    def release_outputs(self):
        self.scratch = self.auxiliary = None

    def commit(self, counts, verification, metadata):
        records, delta = verification
        self.replay(counts, records, delta)
        channels, columns = self.convolution_shape
        torch.library.wrap_triton(_commit_convolution)[
            (self.convolution_offsets.numel(), counts.numel(), triton.cdiv(channels, 128))](
            self.pool.backing.view(torch.bfloat16), self.convolution_offsets, metadata.state_rows, counts,
            self.pool.layout.row_bytes // 2, channels, columns, 128)

    def __call__(self, metadata):
        m, c = metadata, self.weights.config.text
        d = DeltaMetadata.allocate(self.sequences, m.token_ids.device, verification_columns=self.verification_columns)
        active = m.lengths > 0
        d.query_start_loc.copy_(m.query_start_loc)
        d.convolution_slots.copy_(torch.where(active, m.state_rows, -1))
        d.fresh.copy_(m.starts == 0)
        d.state_indices[:, 0].copy_(m.state_rows)
        d.initial_state_indices.copy_(torch.where(m.starts > 0, m.state_rows, 0))
        d.final_state_indices.copy_(m.state_rows)
        d.num_tokens.copy_(m.live_counts[:1])
        d.num_seqs.copy_(m.live_counts[1:])
        split_kda = bool(getattr(self, "split_kda", False))
        d0 = d1 = None
        if split_kda:
            h_tok = self.pipeline_chunk
            dev = m.token_ids.device
            live_n = m.live_counts[:1]
            n0 = torch.clamp(live_n, max=h_tok)
            n1 = torch.clamp(live_n - h_tok, min=0)
            act0 = n0 > 0
            act1 = n1 > 0
            d0 = DeltaMetadata.allocate(1, dev, verification_columns=self.verification_columns)
            d0.query_start_loc[1:].copy_(n0)
            d0.convolution_slots.copy_(torch.where(act0, m.state_rows, -1))
            d0.fresh.copy_(m.starts == 0)
            d0.state_indices[:, 0].copy_(m.state_rows)
            d0.initial_state_indices.copy_(torch.where(m.starts > 0, m.state_rows, 0))
            d0.final_state_indices.copy_(torch.where(act0, m.state_rows, 0))
            d0.num_tokens.copy_(n0)
            d0.num_seqs.copy_(act0.to(torch.int32))
            d1 = DeltaMetadata.allocate(1, dev, verification_columns=self.verification_columns)
            d1.query_start_loc[1:].copy_(n1)
            d1.convolution_slots.copy_(torch.where(act1, m.state_rows, -1))
            d1.fresh.zero_()
            d1.state_indices[:, 0].copy_(m.state_rows)
            d1.initial_state_indices.copy_(torch.where(act1, m.state_rows, 0))
            d1.final_state_indices.copy_(torch.where(act1, m.state_rows, 0))
            d1.num_tokens.copy_(n1)
            d1.num_seqs.copy_(act1.to(torch.int32))
        shared_index = (None if self.shared_index_type is None else
                        self.shared_index_type.create(m, self.index_shards, self.tokens))
        records = None
        if self.replay is not None:
            records = KDARecords(self.record_layers,
                torch.zeros(self.record_shapes[0], dtype=torch.float32, device=m.token_ids.device),
                torch.zeros(self.record_shapes[1], dtype=torch.bfloat16, device=m.token_ids.device), *self.record_constants)
        local = self.embedding.lookup(m.token_ids, num_rows=m.live_counts[:1])
        if self.vision_state is not None:
            self.vision_state.inject(m, local)
        size, eps = c["attn_res_block_size"], c["rms_norm_eps"]
        delta, mode = self.embedding.gather(local), REPLACE
        preceding, preceding_addend = None, None
        pipelined_deltas, pending_tail = None, None
        sharded_deltas = None
        if self.row_shard_prefill:
            world = self.weights.shard.world_size
            total_st = self.tokens // world
            sharded_deltas = [
                torch.index_select(delta, 0, self.shard_rows[0]),
                torch.index_select(delta, 0, self.shard_rows[1]),
            ]
            prefix = torch.zeros((total_st, self.hidden), dtype=delta.dtype, device=delta.device)
            blocks = torch.zeros((total_st, self.residual_blocks, self.hidden), dtype=delta.dtype, device=delta.device)
            residual = ResidualStream(prefix, blocks, torch.empty_like(prefix))
            auxiliary = (None if not self.auxiliary_layers else
                torch.empty((total_st, len(self.auxiliary_layers) * self.hidden), dtype=delta.dtype, device=delta.device))
            delta = None
        else:
            prefix = torch.zeros_like(delta)
            blocks = torch.zeros((self.tokens, self.residual_blocks, self.hidden), dtype=delta.dtype, device=delta.device)
            residual = ResidualStream(prefix, blocks, torch.empty_like(delta))
            auxiliary = (None if not self.auxiliary_layers else
                torch.empty((self.tokens, len(self.auxiliary_layers) * self.hidden), dtype=delta.dtype, device=delta.device))
        for index, (w, attention, mlp, reduce, concurrent_reduce, row_shard_ops) in enumerate(self.layers):
            committed, boundary = -(-index // size), index % size == 0
            captured = None
            if index in self.auxiliary_layers:
                column = self.auxiliary_layers.index(index) * self.hidden
                captured = auxiliary[:, column:column + self.hidden]
            if row_shard_ops is not None:
                if len(row_shard_ops) == 6:
                    rs_parts, ag_parts, crs, cag, crs_sm, cag_sm = row_shard_ops
                else:
                    rs_parts, ag_parts, crs, cag = row_shard_ops
                    crs_sm, cag_sm = crs, cag
                h = self.pipeline_chunk
                r = self.reduce_step
                sc = self.subchunks
                world = self.weights.shard.world_size
                sub_st = r // world
                half_st = h // world
                kind = self.weights.config.layers[index]
                write_block = index // size if boundary else -1
                crs_halves = (crs[:sc], crs[sc:])
                cag_halves = (cag[:sc], cag[sc:])
                crs_sm_halves = (crs_sm[:sc], crs_sm[sc:])
                cag_sm_halves = (cag_sm[:sc], cag_sm[sc:])
                rs_halves = (rs_parts[:sc], rs_parts[sc:])
                ag_halves = (ag_parts[:sc], ag_parts[sc:])

                def start_rs_half(crs_half, tensor_half, *, dependencies=()):
                    pending = []
                    deps = tuple(dependencies)
                    for j, op in enumerate(crs_half):
                        p = op.start(tensor_half[j * r:(j + 1) * r], dependencies=deps, clone=False)
                        pending.append(p)
                        deps = (p.output,)
                    return tuple(pending)

                def finish_rs_half(crs_half, pending, dependency):
                    outs = []
                    dep = dependency
                    for op, p in zip(crs_half, pending, strict=True):
                        out = op.finish(p, dep)
                        outs.append(out)
                        dep = out
                    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)

                def run_rs_half_sync(rs_half, tensor_half, *, dependencies=()):
                    outs = []
                    deps = tuple(dependencies)
                    for j, op in enumerate(rs_half):
                        out = op(tensor_half[j * r:(j + 1) * r], dependencies=deps)
                        outs.append(out)
                        deps = (out,)
                    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)

                def start_ag_half(cag_half, shard_half, *, dependencies=()):
                    pending = []
                    deps = tuple(dependencies)
                    for j, op in enumerate(cag_half):
                        p = op.start(shard_half[j * sub_st:(j + 1) * sub_st], dependencies=deps, clone=False)
                        pending.append(p)
                        deps = (p.output,)
                    return tuple(pending)

                def finish_ag_half(cag_half, pending, dependency):
                    outs = []
                    dep = dependency
                    for op, p in zip(cag_half, pending, strict=True):
                        out = op.finish(p, dep)
                        outs.append(out)
                        dep = out
                    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)

                def run_ag_half_sync(ag_half, shard_half, *, dependencies=()):
                    outs = []
                    deps = tuple(dependencies)
                    for j, op in enumerate(ag_half):
                        out = op(shard_half[j * sub_st:(j + 1) * sub_st], dependencies=deps)
                        outs.append(out)
                        deps = (out,)
                    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)

                def pre_attn_mix_sh(half_idx, delta_sh_half):
                    begin, end = half_idx * half_st, (half_idx + 1) * half_st
                    cap_chunk = None if captured is None else captured[begin:end]
                    chunk_stream = ResidualStream(residual.prefix[begin:end], residual.blocks[begin:end],
                        torch.empty((half_st, self.hidden), dtype=delta_sh_half.dtype, device=delta_sh_half.device))
                    mixed = chunk_stream.mix(delta_sh_half, mode, sources=committed, write=write_block,
                        score=w["attention_res_score"], norm=w["input_layernorm"], eps=eps, output_eps=eps,
                        mixture=cap_chunk if self.auxiliary_stream == "attn_res" else None)
                    if cap_chunk is not None and self.auxiliary_stream == "prefix":
                        cap_chunk.copy_(residual.prefix[begin:end])
                    return mixed

                att_mode = REPLACE if boundary else ADD
                att_sources = committed + boundary
                defer_to_next = index + 1 < len(self.layers) and self.layers[index + 1][5] is not None

                def post_attn_mix_sh(half_idx, att_sh_half):
                    begin, end = half_idx * half_st, (half_idx + 1) * half_st
                    return ResidualStream(residual.prefix[begin:end], residual.blocks[begin:end],
                        torch.empty((half_st, self.hidden), dtype=att_sh_half.dtype, device=att_sh_half.device)).mix(
                        att_sh_half, att_mode, sources=att_sources, write=-1,
                        score=w["mlp_res_score"], norm=w["post_attention_layernorm"], eps=eps, output_eps=eps)

                # Steady-state two-microbatch split-KDA + MoE pipeline:
                # overlaps prev mlp_rs(1) || KDA(0), att_ag(1) || r+d+s(0), rg(0) || KDA(1),
                # att_rs(1) || MoE(0), er(0) || r+d+s(1), rg(1) || post_moe(0), mlp_rs(0) || MoE(1),
                # and next att_ag(0) || post_moe(1).
                if (
                    split_kda
                    and kind == "kda"
                    and not isinstance(mlp, PreparedMLP)
                    and pending_tail is not None
                ):
                    _, prev_delta_sh_0, prev_col_1, prev_shared_1, prev_mlp, prev_crs_1, prev_rs_1, prev_last_comm = pending_tail
                    sharded_deltas, pending_tail = None, None
                    rg = mlp.prefill_route_gather
                    rg_sm = getattr(mlp, "prefill_route_gather_sm", None) or rg
                    er_rs = mlp.concurrent_expert_rs

                    h_sh_0 = pre_attn_mix_sh(0, prev_delta_sh_0)
                    p_ag_0 = start_ag_half(cag_sm_halves[0], h_sh_0, dependencies=(prev_last_comm,))
                    out_1_prev = prev_mlp.post_moe_sharded(prev_col_1, prev_shared_1)
                    hidden_0 = finish_ag_half(cag_sm_halves[0], p_ag_0, out_1_prev)

                    p_prev_rs_1 = start_rs_half(prev_crs_1, out_1_prev, dependencies=(hidden_0,))
                    att_out_0 = attention(hidden_0, d0)
                    delta_sh_1 = finish_rs_half(prev_crs_1, p_prev_rs_1, att_out_0)

                    h_sh_1 = pre_attn_mix_sh(1, delta_sh_1)
                    att_sh_0 = run_rs_half_sync(rs_halves[0], att_out_0, dependencies=(delta_sh_1,))
                    m_sh_0 = post_attn_mix_sh(0, att_sh_0)
                    mlp_in_0 = run_ag_half_sync(ag_halves[0], m_sh_0, dependencies=(att_sh_0,))

                    p_ag_1 = start_ag_half(cag_halves[1], h_sh_1, dependencies=(mlp_in_0,))
                    logits_0 = mlp.router_projection(mlp_in_0)
                    down_0 = mlp.down(mlp_in_0)
                    shared_0 = mlp.shared(mlp_in_0)
                    hidden_1 = finish_ag_half(cag_halves[1], p_ag_1, shared_0)

                    p_rg_0 = rg.start(down_0, logits_0, dependencies=(hidden_1,))
                    att_out_1 = attention(hidden_1, d1)
                    latent_0, full_logits_0 = rg.finish(p_rg_0, att_out_1)

                    p_att_rs_1 = start_rs_half(crs_halves[1], att_out_1, dependencies=(latent_0,))
                    routes_0 = mlp.router(full_logits_0)
                    partial_0 = mlp.experts(latent_0, routes_0.ids, routes_0.weights)
                    att_sh_1 = finish_rs_half(crs_halves[1], p_att_rs_1, partial_0)

                    m_sh_1 = post_attn_mix_sh(1, att_sh_1)
                    mlp_in_1 = run_ag_half_sync(ag_halves[1], m_sh_1, dependencies=(att_sh_1,))

                    p_er_0 = er_rs.start(partial_0, dependencies=(mlp_in_1,), clone=False)
                    logits_1 = mlp.router_projection(mlp_in_1)
                    down_1 = mlp.down(mlp_in_1)
                    shared_1 = mlp.shared(mlp_in_1)
                    normed_0_raw = er_rs.finish(p_er_0, shared_1)

                    normed_0 = mlp.norm_expert_shard(normed_0_raw)
                    col_0 = mlp.expert_a2a(normed_0, dependencies=(normed_0_raw,))

                    p_rg_1 = rg_sm.start(down_1, logits_1, dependencies=(col_0,))
                    out_0 = mlp.post_moe_sharded(col_0, shared_0)
                    latent_1, full_logits_1 = rg_sm.finish(p_rg_1, out_0)

                    p_mlp_rs_0 = start_rs_half(crs_halves[0], out_0, dependencies=(latent_1,))
                    routes_1 = mlp.router(full_logits_1)
                    partial_1 = mlp.experts(latent_1, routes_1.ids, routes_1.weights)
                    delta_sh_0 = finish_rs_half(crs_halves[0], p_mlp_rs_0, partial_1)

                    normed_1_raw = mlp.expert_rs(partial_1, dependencies=(delta_sh_0,))
                    normed_1 = mlp.norm_expert_shard(normed_1_raw)
                    col_1 = mlp.expert_a2a(normed_1, dependencies=(normed_1_raw,))

                    if defer_to_next:
                        pending_tail = ("post_moe_1", delta_sh_0, col_1, shared_1, mlp, crs_halves[1], rs_halves[1], col_1)
                    else:
                        out_1 = mlp.post_moe_sharded(col_1, shared_1)
                        delta_sh_1 = run_rs_half_sync(rs_halves[1], out_1, dependencies=(col_1,))
                        sharded_deltas = [delta_sh_0, delta_sh_1]
                    preceding, preceding_addend = None, None
                    mode = ADD
                    continue

                if split_kda and kind == "kda":
                    # Entry Split-KDA layer (Layer 0 dense or Layer 1 MoE with pending_tail is None).
                    h_sh_0 = pre_attn_mix_sh(0, sharded_deltas[0])
                    h_sh_1 = pre_attn_mix_sh(1, sharded_deltas[1])
                    hidden_0 = run_ag_half_sync(ag_halves[0], h_sh_0, dependencies=(sharded_deltas[1],))
                    p_ag_1 = start_ag_half(cag_halves[1], h_sh_1, dependencies=(hidden_0,))
                    att_out_0 = attention(hidden_0, d0)
                    hidden_1 = finish_ag_half(cag_halves[1], p_ag_1, att_out_0)
                    p_att_rs_0 = start_rs_half(crs_halves[0], att_out_0, dependencies=(hidden_1,))
                    att_out_1 = attention(hidden_1, d1)
                    att_sh_0 = finish_rs_half(crs_halves[0], p_att_rs_0, att_out_1)
                    sharded_deltas, pending_tail = None, None
                else:
                    last_comm = None
                    if pending_tail is None:
                        h_sh_0 = pre_attn_mix_sh(0, sharded_deltas[0])
                        h_sh_1 = pre_attn_mix_sh(1, sharded_deltas[1])
                        hidden_0 = run_ag_half_sync(ag_halves[0], h_sh_0, dependencies=(sharded_deltas[1],))
                        p_ag_1 = start_ag_half(cag_halves[1], h_sh_1, dependencies=(hidden_0,))
                        proj_0 = attention.project_input_chunk(hidden_0)
                        hidden_1 = finish_ag_half(cag_halves[1], p_ag_1, proj_0)
                        proj_1 = attention.project_input_chunk(hidden_1)
                        last_comm = hidden_1
                    else:
                        _, prev_delta_sh_0, prev_col_1, prev_shared_1, prev_mlp, _, prev_rs_1, prev_last_comm = pending_tail
                        h_sh_0 = pre_attn_mix_sh(0, prev_delta_sh_0)
                        p_ag_0 = start_ag_half(cag_sm_halves[0], h_sh_0, dependencies=(prev_last_comm,))
                        out_1_prev = prev_mlp.post_moe_sharded(prev_col_1, prev_shared_1)
                        hidden_0 = finish_ag_half(cag_sm_halves[0], p_ag_0, out_1_prev)
                        delta_sh_1 = run_rs_half_sync(prev_rs_1, out_1_prev, dependencies=(hidden_0,))
                        h_sh_1 = pre_attn_mix_sh(1, delta_sh_1)
                        p_ag_1 = start_ag_half(cag_halves[1], h_sh_1, dependencies=(delta_sh_1,))
                        proj_0 = attention.project_input_chunk(hidden_0)
                        hidden_1 = finish_ag_half(cag_halves[1], p_ag_1, proj_0)
                        proj_1 = attention.project_input_chunk(hidden_1)
                        last_comm = hidden_1
                    sharded_deltas, pending_tail = None, None
                    projected = torch.cat((proj_0, proj_1), dim=0)
                    if kind == "kda":
                        gated = attention.run_core(projected, d, records=None if records is None else records.layer(index))
                        att_out_0 = attention.project_output_chunk(gated[:h])
                        p_att_rs_0 = start_rs_half(crs_sm_halves[0], att_out_0,
                            dependencies=() if last_comm is None else (last_comm,))
                        att_out_1 = attention.project_output_chunk(gated[h:])
                    else:
                        hidden_full = torch.cat((hidden_0, hidden_1), dim=0)
                        if kind == "dsa":
                            core_out, gate_full = attention.run_core(hidden_full, projected, m, shared_index=shared_index)
                        else:
                            core_out, gate_full = attention.run_core(hidden_full, projected, m)
                        att_out_0 = attention.finish_chunk(core_out[:h], gate_full[:h])
                        p_att_rs_0 = start_rs_half(crs_sm_halves[0], att_out_0,
                            dependencies=() if last_comm is None else (last_comm,))
                        att_out_1 = attention.finish_chunk(core_out[h:], gate_full[h:])
                    att_sh_0 = finish_rs_half(crs_sm_halves[0], p_att_rs_0, att_out_1)

                if isinstance(mlp, PreparedMLP):
                    att_sh_1 = run_rs_half_sync(rs_halves[1], att_out_1, dependencies=(att_sh_0,))
                    m_sh_0 = post_attn_mix_sh(0, att_sh_0)
                    m_sh_1 = post_attn_mix_sh(1, att_sh_1)
                    mlp_in_0 = run_ag_half_sync(ag_halves[0], m_sh_0, dependencies=(att_sh_1,))
                    p_att_ag_1 = start_ag_half(cag_halves[1], m_sh_1, dependencies=(mlp_in_0,))
                    out_0 = mlp(mlp_in_0)
                    mlp_in_1 = finish_ag_half(cag_halves[1], p_att_ag_1, out_0)
                    p_mlp_rs_0 = start_rs_half(crs_halves[0], out_0, dependencies=(mlp_in_1,))
                    out_1 = mlp(mlp_in_1)
                    delta_sh_0 = finish_rs_half(crs_halves[0], p_mlp_rs_0, out_1)
                    delta_sh_1 = run_rs_half_sync(rs_halves[1], out_1, dependencies=(delta_sh_0,))
                    sharded_deltas = [delta_sh_0, delta_sh_1]
                elif getattr(mlp, "split_moe", False):
                    rg = mlp.prefill_route_gather
                    rg_sm = getattr(mlp, "prefill_route_gather_sm", None) or rg
                    er_rs = mlp.concurrent_expert_rs
                    m_sh_0 = post_attn_mix_sh(0, att_sh_0)
                    mlp_in_0 = run_ag_half_sync(ag_halves[0], m_sh_0, dependencies=(att_sh_0,))

                    p_att_rs_1 = start_rs_half(crs_halves[1], att_out_1, dependencies=(mlp_in_0,))
                    logits_0 = mlp.router_projection(mlp_in_0)
                    down_0 = mlp.down(mlp_in_0)
                    shared_0 = mlp.shared(mlp_in_0)
                    att_sh_1 = finish_rs_half(crs_halves[1], p_att_rs_1, shared_0)

                    m_sh_1 = post_attn_mix_sh(1, att_sh_1)
                    latent_0, full_logits_0 = rg_sm(down_0, logits_0, dependencies=(att_sh_1,))

                    p_mlp_ag_1 = start_ag_half(cag_halves[1], m_sh_1, dependencies=(latent_0,))
                    routes_0 = mlp.router(full_logits_0)
                    partial_0 = mlp.experts(latent_0, routes_0.ids, routes_0.weights)
                    mlp_in_1 = finish_ag_half(cag_halves[1], p_mlp_ag_1, partial_0)

                    p_er_0 = er_rs.start(partial_0, dependencies=(mlp_in_1,), clone=False)
                    logits_1 = mlp.router_projection(mlp_in_1)
                    down_1 = mlp.down(mlp_in_1)
                    shared_1 = mlp.shared(mlp_in_1)
                    normed_0_raw = er_rs.finish(p_er_0, shared_1)

                    normed_0 = mlp.norm_expert_shard(normed_0_raw)
                    col_0 = mlp.expert_a2a(normed_0, dependencies=(normed_0_raw,))

                    p_rg_1 = rg_sm.start(down_1, logits_1, dependencies=(col_0,))
                    out_0 = mlp.post_moe_sharded(col_0, shared_0)
                    latent_1, full_logits_1 = rg_sm.finish(p_rg_1, out_0)

                    p_mlp_rs_0 = start_rs_half(crs_halves[0], out_0, dependencies=(latent_1,))
                    routes_1 = mlp.router(full_logits_1)
                    partial_1 = mlp.experts(latent_1, routes_1.ids, routes_1.weights)
                    delta_sh_0 = finish_rs_half(crs_halves[0], p_mlp_rs_0, partial_1)

                    normed_1_raw = mlp.expert_rs(partial_1, dependencies=(delta_sh_0,))
                    normed_1 = mlp.norm_expert_shard(normed_1_raw)
                    col_1 = mlp.expert_a2a(normed_1, dependencies=(normed_1_raw,))

                    if defer_to_next:
                        pending_tail = ("post_moe_1", delta_sh_0, col_1, shared_1, mlp, crs_halves[1], rs_halves[1], col_1)
                    else:
                        out_1 = mlp.post_moe_sharded(col_1, shared_1)
                        delta_sh_1 = run_rs_half_sync(rs_halves[1], out_1, dependencies=(col_1,))
                        sharded_deltas = [delta_sh_0, delta_sh_1]
                else:
                    rg = mlp.prefill_route_gather
                    er_rs = mlp.concurrent_expert_rs
                    att_sh_1 = run_rs_half_sync(rs_halves[1], att_out_1, dependencies=(att_sh_0,))
                    m_sh_0 = post_attn_mix_sh(0, att_sh_0)
                    m_sh_1 = post_attn_mix_sh(1, att_sh_1)
                    mlp_in_0 = run_ag_half_sync(ag_halves[0], m_sh_0, dependencies=(att_sh_1,))

                    p_att_ag_1 = start_ag_half(cag_halves[1], m_sh_1, dependencies=(mlp_in_0,))
                    logits_0 = mlp.router_projection(mlp_in_0)
                    down_0 = mlp.down(mlp_in_0)
                    shared_0 = mlp.shared(mlp_in_0)
                    mlp_in_1 = finish_ag_half(cag_halves[1], p_att_ag_1, shared_0)

                    p_rg_0 = rg.start(down_0, logits_0, dependencies=(mlp_in_1,))
                    logits_1 = mlp.router_projection(mlp_in_1)
                    down_1 = mlp.down(mlp_in_1)
                    latent_0, full_logits_0 = rg.finish(p_rg_0, down_1)

                    p_rg_1 = rg.start(down_1, logits_1, dependencies=(latent_0,))
                    shared_1 = mlp.shared(mlp_in_1)
                    latent_1, full_logits_1 = rg.finish(p_rg_1, shared_1)

                    # Keep routed MoE unsplit so 896-expert W4 weights are read once per layer.
                    latent = torch.cat((latent_0, latent_1), dim=0)
                    full_logits = torch.cat((full_logits_0, full_logits_1), dim=0)
                    routes = mlp.router(full_logits)
                    partial = mlp.experts(latent, routes.ids, routes.weights)

                    normed_0_raw = mlp.expert_rs(partial[:h], dependencies=(mlp_in_1, latent_1))
                    normed_0 = mlp.norm_expert_shard(normed_0_raw)
                    col_0 = mlp.expert_a2a(normed_0, dependencies=(normed_0_raw,))

                    p_er_1 = er_rs.start(partial[h:], dependencies=(col_0,), clone=False)
                    out_0 = mlp.post_moe_sharded(col_0, shared_0)
                    normed_1_raw = er_rs.finish(p_er_1, out_0)

                    normed_1 = mlp.norm_expert_shard(normed_1_raw)
                    col_1 = mlp.expert_a2a(normed_1, dependencies=(normed_1_raw,))

                    p_mlp_rs_0 = start_rs_half(crs_sm_halves[0], out_0, dependencies=(col_1,))
                    out_1 = mlp.post_moe_sharded(col_1, shared_1)
                    delta_sh_0 = finish_rs_half(crs_sm_halves[0], p_mlp_rs_0, out_1)

                    delta_sh_1 = run_rs_half_sync(rs_halves[1], out_1, dependencies=(delta_sh_0,))
                    sharded_deltas = [delta_sh_0, delta_sh_1]
                preceding, preceding_addend = None, None
                mode = ADD
                continue
            if concurrent_reduce is not None:
                h = self.pipeline_chunk
                r = self.reduce_step
                sc = self.subchunks
                cr0, cr1 = concurrent_reduce[:sc], concurrent_reduce[sc:]
                reduce_groups = (reduce.parts[:sc], reduce.parts[sc:])
                kind = self.weights.config.layers[index]
                write_block = index // size if boundary else -1

                def start_ar_chunk(cr_group, tensor_chunk, *, dependencies=()):
                    pending = []
                    deps = tuple(dependencies)
                    for j, op in enumerate(cr_group):
                        p = op.start(tensor_chunk[j * r:(j + 1) * r], dependencies=deps)
                        pending.append(p)
                        deps = (p.output,)
                    return tuple(pending)

                def finish_ar_chunk(cr_group, pending, dependency):
                    outs = []
                    dep = dependency
                    for op, p in zip(cr_group, pending, strict=True):
                        out = op.finish(p, dep)
                        outs.append(out)
                        dep = out
                    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)

                def run_ar_chunk_sync(parts_group, tensor_chunk, *, dependencies=()):
                    outs = []
                    deps = tuple(dependencies)
                    for j, op in enumerate(parts_group):
                        out = op(tensor_chunk[j * r:(j + 1) * r], dependencies=deps)
                        outs.append(out)
                        deps = (out,)
                    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)

                def mix_and_project_in(chunk_index, delta_chunk):
                    begin, end = chunk_index * h, (chunk_index + 1) * h
                    cap_chunk = None if captured is None else captured[begin:end]
                    chunk_stream = ResidualStream(residual.prefix[begin:end], residual.blocks[begin:end],
                        torch.empty((h, self.hidden), dtype=delta_chunk.dtype, device=delta_chunk.device))
                    hidden_chunk = chunk_stream.mix(delta_chunk, mode, sources=committed, write=write_block,
                        score=w["attention_res_score"], norm=w["input_layernorm"], eps=eps, output_eps=eps,
                        mixture=cap_chunk if self.auxiliary_stream == "attn_res" else None)
                    if cap_chunk is not None and self.auxiliary_stream == "prefix":
                        cap_chunk.copy_(residual.prefix[begin:end])
                    return hidden_chunk, attention.project_input_chunk(hidden_chunk)

                last_comm = None
                if pending_tail is None:
                    src_deltas = (pipelined_deltas if pipelined_deltas is not None
                                  else [delta[:h], delta[h:]])
                    hidden_0, proj_0 = mix_and_project_in(0, src_deltas[0])
                    hidden_1, proj_1 = mix_and_project_in(1, src_deltas[1])
                else:
                    _, prev_cr_1, p_mlp_1 = pending_tail
                    hidden_0, proj_0 = mix_and_project_in(0, pipelined_deltas[0])
                    delta_1 = finish_ar_chunk(prev_cr_1, p_mlp_1, proj_0)
                    hidden_1, proj_1 = mix_and_project_in(1, delta_1)
                    last_comm = delta_1
                pipelined_deltas, pending_tail = None, None
                projected = torch.cat((proj_0, proj_1), dim=0)
                if kind == "kda":
                    gated = attention.run_core(projected, d, records=None if records is None else records.layer(index))
                    att_out_0 = attention.project_output_chunk(gated[:h])
                    p_att_0 = start_ar_chunk(cr0, att_out_0,
                        dependencies=() if last_comm is None else (last_comm,))
                    att_out_1 = attention.project_output_chunk(gated[h:])
                else:
                    hidden_full = torch.cat((hidden_0, hidden_1), dim=0)
                    if kind == "dsa":
                        core_out, gate_full = attention.run_core(hidden_full, projected, m, shared_index=shared_index)
                    else:
                        core_out, gate_full = attention.run_core(hidden_full, projected, m)
                    att_out_0 = attention.finish_chunk(core_out[:h], gate_full[:h])
                    p_att_0 = start_ar_chunk(cr0, att_out_0,
                        dependencies=() if last_comm is None else (last_comm,))
                    att_out_1 = attention.finish_chunk(core_out[h:], gate_full[h:])
                att_0 = finish_ar_chunk(cr0, p_att_0, att_out_1)
                att_mode = REPLACE if boundary else ADD
                att_sources = committed + boundary
                defer_to_next = index + 1 < len(self.layers) and self.layers[index + 1][4] is not None

                def post_attn_mix(chunk_index, att_chunk):
                    begin, end = chunk_index * h, (chunk_index + 1) * h
                    return ResidualStream(residual.prefix[begin:end], residual.blocks[begin:end],
                        torch.empty((h, self.hidden), dtype=att_chunk.dtype, device=att_chunk.device)).mix(
                        att_chunk, att_mode, sources=att_sources, write=-1,
                        score=w["mlp_res_score"], norm=w["post_attention_layernorm"], eps=eps, output_eps=eps)

                if isinstance(mlp, PreparedMLP):
                    p_att_1 = start_ar_chunk(cr1, att_out_1, dependencies=(att_0,))
                    out_0 = mlp(post_attn_mix(0, att_0))
                    att_1 = finish_ar_chunk(cr1, p_att_1, out_0)
                    p_mlp_0 = start_ar_chunk(cr0, out_0, dependencies=(att_1,))
                    out_1 = mlp(post_attn_mix(1, att_1))
                    delta_0 = finish_ar_chunk(cr0, p_mlp_0, out_1)
                    if defer_to_next:
                        p_mlp_1 = start_ar_chunk(cr1, out_1, dependencies=(delta_0,))
                        pipelined_deltas = [delta_0]
                        pending_tail = ("mlp_last", cr1, p_mlp_1)
                    else:
                        delta_1 = run_ar_chunk_sync(reduce_groups[1], out_1, dependencies=(delta_0,))
                        delta = torch.cat((delta_0, delta_1), dim=0)
                else:
                    rg, er = mlp.prefill_route_gather, mlp.concurrent_expert_reduce
                    p_att_1 = start_ar_chunk(cr1, att_out_1, dependencies=(att_0,))
                    hidden_0 = post_attn_mix(0, att_0)
                    logits_0 = mlp.router_projection(hidden_0)
                    down_0 = mlp.down(hidden_0)
                    shared_0 = mlp.shared(hidden_0)
                    att_1 = finish_ar_chunk(cr1, p_att_1, shared_0)

                    p_rg_0 = rg.start(down_0, logits_0, dependencies=(att_1,))
                    hidden_1 = post_attn_mix(1, att_1)
                    logits_1 = mlp.router_projection(hidden_1)
                    down_1 = mlp.down(hidden_1)
                    latent_0, full_logits_0 = rg.finish(p_rg_0, down_1)

                    p_rg_1 = rg.start(down_1, logits_1, dependencies=(latent_0,))
                    shared_1 = mlp.shared(hidden_1)
                    latent_1, full_logits_1 = rg.finish(p_rg_1, shared_1)

                    # Keep routed MoE unsplit so 896-expert W4 weights are read once per layer.
                    latent = torch.cat((latent_0, latent_1), dim=0)
                    full_logits = torch.cat((full_logits_0, full_logits_1), dim=0)
                    routes = mlp.router(full_logits)
                    partial = mlp.experts(latent, routes.ids, routes.weights)

                    latent_sum_0 = mlp.expert_reduce.parts[0](partial[:h], dependencies=(att_1, latent_1))
                    p_er_1 = er.start(partial[h:], dependencies=(latent_sum_0,))
                    out_0 = mlp.post_moe(latent_sum_0, shared_0)
                    latent_sum_1 = er.finish(p_er_1, out_0)

                    p_mlp_0 = start_ar_chunk(cr0, out_0, dependencies=(latent_sum_1,))
                    out_1 = mlp.post_moe(latent_sum_1, shared_1)
                    delta_0 = finish_ar_chunk(cr0, p_mlp_0, out_1)

                    if defer_to_next:
                        p_mlp_1 = start_ar_chunk(cr1, out_1, dependencies=(delta_0,))
                        pipelined_deltas = [delta_0]
                        pending_tail = ("mlp_last", cr1, p_mlp_1)
                    else:
                        delta_1 = run_ar_chunk_sync(reduce_groups[1], out_1, dependencies=(delta_0,))
                        delta = torch.cat((delta_0, delta_1), dim=0)
                preceding, preceding_addend = None, None
                mode = ADD
                continue
            hidden = residual.mix(delta, mode, sources=committed, write=index // size if boundary else -1,
                score=w["attention_res_score"], norm=w["input_layernorm"], eps=eps, output_eps=eps,
                mixture=captured if self.auxiliary_stream == "attn_res" else None, output=torch.empty_like(delta),
                reduction_choice=preceding, addend=preceding_addend)
            if captured is not None and self.auxiliary_stream == "prefix":
                captured.copy_(residual.prefix)
            if self.weights.config.layers[index] == "kda":
                attended = attention(hidden, d, records=None if records is None else records.layer(index))
            elif self.weights.config.layers[index] == "dsa":
                attended = attention(hidden, m, shared_index=shared_index)
            else:
                attended = attention(hidden, m)
            after_attention, preceding = (None, None) if self.row_mixes is None else self.row_mixes[index]
            if after_attention is None:
                attended = reduce(attended)
            hidden = residual.mix(attended, REPLACE if boundary else ADD, sources=committed + boundary, write=-1,
                score=w["mlp_res_score"], norm=w["post_attention_layernorm"], eps=eps, output_eps=eps,
                output=torch.empty_like(delta), reduction_choice=after_attention)
            if preceding is not None and preceding.fused and isinstance(mlp, PreparedLatentMoE):
                delta, preceding_addend = mlp(hidden, fuse_add=True)
            else:
                delta, preceding_addend = mlp(hidden), None
            mode = ADD
            if preceding is None:
                delta = reduce(delta)
        if self.row_shard_prefill:
            delta_sh = torch.cat(sharded_deltas, dim=0)
            hidden_sh = residual.mix(delta_sh, mode, sources=-(-len(self.layers) // size), write=-1,
                score=self.weights.output_res_score, norm=self.weights.norm, eps=eps, output_eps=eps,
                output=torch.empty_like(delta_sh))
            world = self.weights.shard.world_size
            sub_st = self.reduce_step // world
            n_sub = 2 * self.subchunks

            def gather_all_subchunks(sharded_tensor, deps=()):
                parts = []
                cur_deps = deps
                for s_idx in range(n_sub):
                    sub = sharded_tensor[s_idx * sub_st:(s_idx + 1) * sub_st]
                    gathered = self.final_gather(sub, dependencies=cur_deps)
                    parts.append(gathered)
                    cur_deps = (gathered,)
                return torch.cat(parts, dim=0), cur_deps

            hidden, gather_deps = gather_all_subchunks(hidden_sh)
            if auxiliary is not None:
                aux_cols = []
                for aux_idx in range(len(self.auxiliary_layers)):
                    col = aux_idx * self.hidden
                    aux_slice = auxiliary[:, col:col + self.hidden].contiguous()
                    gathered_col, gather_deps = gather_all_subchunks(aux_slice, deps=gather_deps)
                    aux_cols.append(gathered_col)
                auxiliary = torch.cat(aux_cols, dim=1)
        else:
            hidden = residual.mix(delta, mode, sources=-(-len(self.layers) // size), write=-1,
                score=self.weights.output_res_score, norm=self.weights.norm, eps=eps, output_eps=eps,
                output=torch.empty_like(delta), reduction_choice=preceding, addend=preceding_addend)
        if self.scoring is not None:
            self.scoring(hidden)
        if self.compact_head:
            logits = self.head(hidden, row_indices=m.sample_indices)
            sampled = hidden[m.sample_indices]
            return sampled, logits, auxiliary, (records, d)
        sampled = hidden[m.sample_indices]
        return sampled, self.head(sampled), auxiliary, (records, d)
