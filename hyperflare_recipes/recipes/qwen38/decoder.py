# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Exact-checkpoint decoder weight ownership and canonical module parameters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from hyperflare_recipes.components.parallel import TensorParallelShard
from ignition.planning import SUPPORTED_ARCHITECTURES, SUPPORTED_CAPABILITIES
from hyperflare_recipes.recipes.qwen38.checkpoint import CheckpointManifest
from ignition.preparation.host_memory import device_memory_info
from ignition.preparation.persistent import allocated_tensor_bytes
from .checkpoint_schema import TEXT_PREFIX
from ignition.preparation.scratch import ScratchArena

from .hyperconnection import packed_down_projection, hyperconnection_extents
from .model import create_model_tree
from .moe import QwenMoeWeights
from .vision import VisionWeights
from .weights import QwenWeightLoader
from ignition.preparation.rebuild_manifest import weight_container


@weight_container(('attention', 'moe'))
@dataclass(frozen=True)
class DecoderLayerWeights:
    attention: dict[str, object]
    moe: QwenMoeWeights


@dataclass(frozen=True)
class DecoderAuxiliary:
    residual: torch.Tensor
    draft_features: torch.Tensor | None


torch.utils._pytree.register_dataclass(DecoderAuxiliary)


class PreparedDecoderForward:
    def __init__(self, decoder):
        self.decoder = decoder

    def __call__(self):
        decoder = self.decoder
        errors = None if decoder.embedding_injection is None else decoder.embedding_injection.validate()
        runtime = decoder.state.begin_runtime()
        runtime.injection_errors = errors
        decoder(runtime)


@dataclass(frozen=True)
class ModelWeights:
    """Own the canonical tree and every packed GPU weight used by its recipes.

    Canonical quantized Linear and expert parameters remain on meta storage;
    their prepared replacements own the checkpoint's packed representations.
    PLE table ownership is separate because those tables reside in mapped host
    memory. Expert layers expand their CSF block scales once at load.
    """

    tree: torch.nn.Module
    layers: tuple[DecoderLayerWeights, ...]
    vision: VisionWeights
    allocated_bytes: int
    shard: TensorParallelShard = TensorParallelShard(0, 1)

    @classmethod
    def load(cls, manifest: CheckpointManifest, device: torch.device,
             *, progress: Callable[[str], None] | None = None,
             shard: TensorParallelShard = TensorParallelShard(0, 1),
             shard_hyperconnections: bool = False, force_moe_decode_a16: bool = False) -> ModelWeights:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("model weights require a CUDA device")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
            raise ValueError(f"the Qwen recipe requires {SUPPORTED_ARCHITECTURES}")
        if force_moe_decode_a16 and torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
            raise ValueError("force_moe_decode_a16 requires SM120/121 packed NVFP4 experts")
        free_bytes, _ = device_memory_info(device)
        stored = sum(tensor.nbytes for tensor in manifest.tensors.values()
                     if tensor.component not in ("mtp", "ple_tables"))
        # Additional layouts, state, graph memory, and cache pages are planned separately.
        if free_bytes < stored // shard.world_size + (8 << 30):
            raise MemoryError(f"model weight loading requires at least {stored // shard.world_size + (8 << 30)} free CUDA bytes; found {free_bytes}")
        tree = create_model_tree(manifest)
        before = allocated_tensor_bytes(device)
        layers = []
        with QwenWeightLoader(manifest, device, progress=progress, shard=shard,
                              force_moe_decode_a16=force_moe_decode_a16) as loader:
            def parameters(paths, extents=None, *, temporary=False):
                for path, value in zip(paths, loader.load_tensors(paths, extents=extents, temporary=temporary), strict=True):
                    parent, _, attribute = path.rpartition(".")
                    setattr(tree.get_submodule(parent), attribute, torch.nn.Parameter(value, requires_grad=False))

            def hyperconnection(prefix: str):
                names = ("hc_norm", "input_mix_weight_down", "input_mix_weight_up")
                if tree.get_submodule(prefix).block_inject_weight is not None:
                    names += ("block_inject_weight",)
                extents = hyperconnection_extents(shard, shard_hyperconnections)
                packed = tree.get_submodule(prefix).block_inject_weight is not None
                for temporary in (False, True):
                    selected = tuple(name for name in names
                        if (packed and name in ("input_mix_weight_down", "block_inject_weight")) == temporary)
                    if selected:
                        parameters(tuple(f"{prefix}.{name}.weight" for name in selected),
                                   tuple(extents.get(f"{name}.weight") for name in selected), temporary=temporary)
                packed_down_projection(tree.get_submodule(prefix))

            parameters((f"{TEXT_PREFIX}.embed_tokens.weight", "lm_head.weight"),
                       (None, slice(*shard.partition(248320))))
            hyperconnection(f"{TEXT_PREFIX}.hyper_connection_mixer")
            for layer in range(48):
                prefix = f"{TEXT_PREFIX}.layers.{layer}"
                hyperconnection(f"{prefix}.attn_hyper_connection")
                hyperconnection(f"{prefix}.mlp_hyper_connection")
                if layer % 4 == 3:
                    attention_prefix = f"{prefix}.self_attn"
                    load_projections = loader.load_qsa_projections
                    names = ("q_norm.weight", "k_norm.weight", "indexer.q_layernorm.weight",
                             "indexer.k_layernorm.weight")
                else:
                    attention_prefix = f"{prefix}.linear_attn"
                    load_projections = loader.load_gdn_projections
                    names = ("A_log", "dt_bias", "conv1d.weight", "norm.weight")
                if layer % 4 == 3:
                    parameters(tuple(f"{attention_prefix}.{name}" for name in names))
                else:
                    heads = slice(*shard.partition(48))
                    parameters(tuple(f"{attention_prefix}.{name}" for name in ("A_log", "dt_bias", "norm.weight")),
                               (heads, heads, None))
                    channels = shard.fused_partitions((2048, 2048, 6144))
                    from ignition.preparation.persistent import persistent_cat

                    convolution = persistent_cat(loader.load_tensors((f"{attention_prefix}.conv1d.weight",) * 3,
                        extents=tuple(slice(*part) for part in channels), temporary=True))
                    module = tree.get_submodule(attention_prefix)
                    module.conv1d.weight = torch.nn.Parameter(convolution, requires_grad=False)
                    module.num_k_heads //= shard.world_size
                    module.num_v_heads //= shard.world_size
                    module.conv_dim //= shard.world_size
                attention = load_projections(attention_prefix)
                moe = QwenMoeWeights.load(loader, layer)
                module = tree.get_submodule(f"{prefix}.mlp")
                module.gate.weight = torch.nn.Parameter(moe.router, requires_grad=False)
                module.shared_expert_gate.weight = torch.nn.Parameter(moe.shared.gate, requires_grad=False)
                layers.append(DecoderLayerWeights(attention, moe))

            prefix = f"{TEXT_PREFIX}.layers.1.ple"
            parameters(tuple(f"{prefix}.{name}" for name in
                ("key_proj.weight", "value_proj.weight", "conv1d.weight", "norm_key.weight",
                 "norm_query.weight", "norm_conv.weight")))
            vision = VisionWeights.load(tree.model.visual, loader)
        torch.cuda.synchronize(device)
        return cls(tree, tuple(layers), vision,
                   allocated_tensor_bytes(device) - before, shard)


@dataclass(frozen=True)
class DecoderScratch:
    """Tensor examples used to declare and tune decoder operations."""

    embedding: torch.Tensor
    residual: torch.Tensor
    hidden: torch.Tensor
    sample_hidden: torch.Tensor
    logits: torch.Tensor
    hyperconnection: object
    gdn: object
    qsa: object
    ple: object
    moe: object
    local_logits: torch.Tensor | None = None

    @classmethod
    def allocate(cls, tokens: int, device, *, sample_rows: int = 4, world_size: int = 1,
                 hyperconnection_lowrank: int = 320):
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("decoder token capacity must be a positive integer")
        if type(sample_rows) is not int or sample_rows <= 0:
            raise ValueError("decoder sampling capacity must be positive")
        from .attention import QSAScratch
        from hyperflare_recipes.components.gated_delta import GDNScratch
        from .hyperconnection import HyperconnectionScratch
        from .moe import QwenMoeScratch
        from .ple import PLEScratch

        def zeros(*shape):
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        embedding, residual, hidden = zeros(tokens, 2560), zeros(tokens, 10240), zeros(tokens, 2560)
        sample_hidden, logits = zeros(sample_rows, 2560), zeros(sample_rows, 248320)
        local_logits = zeros(sample_rows, 248320 // world_size) if world_size > 1 else None
        block_input, block_output = zeros(tokens, 2560), zeros(tokens, 2560)
        projected = zeros(tokens, (hyperconnection_lowrank + 4 + 15) // 16 * 16)
        ple = PLEScratch.allocate(tokens, device)
        gdn = GDNScratch.allocate(tokens, device, world_size=world_size, output=block_output)
        qsa = QSAScratch.allocate(tokens, device, world_size=world_size, output=block_output)
        moe = QwenMoeScratch.allocate(tokens, device, intermediate=640 // world_size,
                                    output=block_output)
        hc = HyperconnectionScratch.allocate(tokens, device,
                                            projected=projected, block_input=block_input, lowrank=hyperconnection_lowrank)
        return cls(embedding, residual, hidden, sample_hidden, logits, hc, gdn, qsa, ple, moe, local_logits)

    def rows(self, tokens: int, sequences: int):
        if (type(tokens) is not int or type(sequences) is not int
                or not 0 < tokens <= self.embedding.shape[0]
                or not 1 <= sequences <= self.sample_hidden.shape[0]):
            raise ValueError("decoder view exceeds persistent scratch capacity")

        def rows(value):
            return type(value)(*(getattr(value, field)[:tokens] for field in value.__dataclass_fields__))

        return type(self)(self.embedding[:tokens], self.residual[:tokens], self.hidden[:tokens],
            self.sample_hidden[:sequences], self.logits[:sequences], self.hyperconnection.rows(tokens),
            self.gdn.rows(tokens), self.qsa.rows(tokens), rows(self.ple), self.moe.rows(tokens),
            None if self.local_logits is None else self.local_logits[:sequences])


# Minimum prefill chunk capacity for 2-microbatch DMA pipelining. Splitting 512
# routed experts into two halves doubles per-layer expert-weight reads, so
# overlapping copy-engine all-reduce latency requires chunks of at least 4096
# tokens.
_PIPELINED_PREFILL_MIN_TOKENS = 4096


def _is_dma_runtime(runtime) -> bool:
    if runtime is None:
        return False
    from b12x.comm import pcie

    return isinstance(runtime, pcie.DmaAllReduce)


def _non_none(*plans) -> tuple:
    return tuple(plan for plan in plans if plan is not None)


def _hc_plans(hc) -> tuple:
    return _non_none(*hc.plans.values(), hc.down.plan, getattr(hc.gate, "plan", None))


def _attention_plans(attention, *, is_qsa: bool) -> tuple:
    plans = []
    for proj in attention.projections.values():
        plans.append(proj.plan)
        plans.append(getattr(proj, "quantizer_plan", None))
    if is_qsa:
        plans.extend((attention.plan, attention.writer, getattr(attention.sharded, "select_plan", None)))
    else:
        plans.extend((
            getattr(attention, "conv_plan", None),
            getattr(attention, "recurrent_plan", None),
            getattr(getattr(attention, "chunked", None), "plan", None),
        ))
    return _non_none(*plans)


def _moe_plans(moe) -> tuple:
    return _non_none(
        moe.router_projection.plan,
        moe.routing.plan,
        moe.experts.plan,
        getattr(moe.experts, "scale_plan", None),
        moe.shared.up_projection.plan,
        getattr(moe.shared.up_projection, "quantizer_plan", None),
        moe.shared.activation_plan,
        moe.shared.down_projection.plan,
        getattr(moe.shared.down_projection, "quantizer_plan", None),
        moe.shared.gate_projection.plan,
    )


def _split_prefill_metadata(gdn, qsa, half: int):
    """Derive half-chunk GDN and QSA metadata for [0, half) and [half, 2 * half) on device."""
    from hyperflare_recipes.components.recurrent import DeltaMetadata
    from .attention import QSAMetadata

    total = gdn.num_tokens
    has_seq = gdn.num_seqs > 0
    one_i32 = torch.ones_like(total)
    len_a = total.clamp(max=half)
    len_b = (total - half).clamp(min=0)
    active_a = (len_a > 0) & has_seq
    active_b = (len_b > 0) & has_seq
    # Build [0, len] with a pointwise select rather than cat so Inductor does not
    # alias len_a/len_b into cu_a/cu_b storage that dies after cu_a/cu_b is snapshotted.
    cu_mask = torch.arange(2, dtype=torch.int32, device=total.device) == 1
    cu_a = torch.where(cu_mask, len_a, 0)
    cu_b = torch.where(cu_mask, len_b, 0)
    conv_a = torch.where(active_a, gdn.convolution_slots, -1)
    conv_b = torch.where(active_b, gdn.convolution_slots, -1)
    fresh_b = torch.zeros_like(gdn.fresh)
    init_a = torch.where(active_a, gdn.initial_state_indices, 0)
    final_a = torch.where(active_a, gdn.final_state_indices, 0)
    init_b = torch.where(active_b, gdn.final_state_indices, 0)
    final_b = torch.where(active_b, gdn.final_state_indices, 0)
    gdn_a = DeltaMetadata(
        cu_a, conv_a, gdn.fresh, gdn.accepted_tokens, gdn.state_indices,
        init_a, final_a, gdn.checkpoint_state_indices, gdn.checkpoint_offsets,
        active_a.to(torch.int32), len_a,
    )
    gdn_b = DeltaMetadata(
        cu_b, conv_b, fresh_b, one_i32, gdn.state_indices,
        init_b, final_b, gdn.checkpoint_state_indices, gdn.checkpoint_offsets,
        active_b.to(torch.int32), len_b,
    )
    seq_len_a = torch.where(active_a, qsa.sequence_lengths - len_b, 0).to(torch.int32)
    seq_len_b = torch.where(active_b, qsa.sequence_lengths, 0).to(torch.int32)
    qsa_a = QSAMetadata(
        qsa.main_block_table, qsa.compressed_block_table,
        torch.where(active_a, qsa.raw_state_slot_ids, -1),
        qsa.request_ids[:half], qsa.query_positions[:half], qsa.rope_positions[:half],
        seq_len_a, cu_a, qsa.num_accepted_tokens, qsa.is_prefilling & active_a,
        qsa.slot_mapping[:half],
    )
    qsa_b = QSAMetadata(
        qsa.main_block_table, qsa.compressed_block_table,
        torch.where(active_b, qsa.raw_state_slot_ids, -1),
        qsa.request_ids[half:], qsa.query_positions[half:], qsa.rope_positions[half:],
        seq_len_b, cu_b, one_i32, qsa.is_prefilling & active_b,
        qsa.slot_mapping[half:],
    )
    return gdn_a, gdn_b, qsa_a, qsa_b


class PreparedDecoder:
    """Capture the canonical 48-layer residual composition with prepared operators.

    Weights, hybrid state, paged attention caches and all escaping tensors have
    explicit owners. Scratch and workspaces are shared across serialized layer
    calls and graph variants. Finalize scratch bindings before capture.
    Python executes this object
    only during preparation and capture.
    """

    def __init__(self, weights: ModelWeights, scratch: DecoderScratch, state,
                 qsa_states, rotary, ple_table, workspaces: ScratchArena,
                 completion, frame_index, submission_sequence,
                 *, session, name: str, progress=None,
                 layer_outputs: torch.Tensor | None = None, embedding_injection=None,
                 max_cache_pages: int | None = None,
                 residual_output: torch.Tensor | None = None, verification: bool = False,
                 gdn_records=None, gdn_in_place: bool = False, runtime=None, barrier_runtime=None, vocabulary_runtime=None,
                 result_runtime=None, qsa_gather_runtime=None, qsa_shard_mode: str | None = None,
                 coordinator=None, sampling_workspace=None, scoring=None, sample: bool = True):
        from hyperflare_recipes.components.embedding import PreparedEmbedding
        from hyperflare_recipes.components.linear import UnquantizedProjection
        from hyperflare_recipes.components.recurrent import DeltaMetadata

        from .attention import PreparedAttention, QSAMetadata
        from hyperflare_recipes.components.gated_delta import PreparedGDN
        from .hyperconnection import PreparedHyperconnection
        from .moe import PreparedQwenMoe
        from .ple import PreparedPLE

        if not sample and (state.mode != "prefill" or verification or scoring is not None):
            raise ValueError("non-sampling decoders require unverified, unscored prefill state")
        self.sample = bool(sample)

        if verification and scratch.sample_hidden.shape == scratch.hidden.shape:
            scratch = replace(scratch, sample_hidden=scratch.hidden)
        self.weights, self.scratch, self.state = weights, scratch, state
        self.record_layout = None if gdn_records is None else (
            gdn_records.layers, gdn_records.delta.shape[1], gdn_records.keys.shape[2],
            gdn_records.delta.shape[2], gdn_records.delta.shape[3])
        self.verification = verification
        if residual_output is not None and (
                residual_output.shape != scratch.residual.shape
                or residual_output.dtype != torch.bfloat16
                or residual_output.device != scratch.residual.device):
            raise ValueError("target feedback requires persistent BF16 [tokens,10240] storage")
        self.residual_output = residual_output
        if layer_outputs is not None and (layer_outputs.shape != (48, state.tokens, 10240)
                or layer_outputs.dtype != torch.bfloat16 or layer_outputs.device != scratch.hidden.device):
            raise ValueError("layer trace requires BF16 [48,token_capacity,10240] on the model device")
        self.layer_outputs = layer_outputs
        self.context_capture, self.context_layers = None, ()
        self.embedding_injection = embedding_injection
        self.workspaces, self.ple_table, self.qsa_states = workspaces, ple_table, qsa_states
        language = weights.tree.model.language_model
        sharded_hc = language.layers[0].attn_hyper_connection.input_mix_weight_down.weight.shape[0] < 320
        half = state.tokens // 2
        shard = getattr(weights, "shard", None)
        row_sharded_qsa_fits = (
            qsa_shard_mode is None
            or (qsa_shard_mode == "row" and shard is not None and half % (32 * shard.world_size) == 0)
        )
        self.pipelined_prefill = (
            state.mode == "prefill"
            and state.rows == 1
            and getattr(state, "gdn_partitions", None) is None
            and gdn_records is None
            and not sharded_hc
            and isinstance(workspaces, ScratchArena)
            and workspaces.overlap
            and _is_dma_runtime(runtime)
            and state.tokens >= _PIPELINED_PREFILL_MIN_TOKENS
            and state.tokens % 2 == 0
            and row_sharded_qsa_fits
            and runtime.should_allreduce(scratch.hidden[:half])
        )
        self.half_tokens = half if self.pipelined_prefill else 0
        if self.pipelined_prefill:
            layer_scratch = scratch.rows(half, state.rows)
            layer_gdn_meta = DeltaMetadata.allocate(state.rows, scratch.hidden.device)
            layer_qsa_meta = QSAMetadata.allocate(
                tokens=half, batch=state.rows,
                context=state.qsa.main_block_table.shape[1] * 64,
                device=scratch.hidden.device,
            )
        else:
            layer_scratch = scratch
            layer_gdn_meta = state.gdn
            layer_qsa_meta = state.qsa
        source = layer_scratch.hyperconnection.block_input
        self.embedding = PreparedEmbedding(language.embed_tokens.weight, state.ple.token_ids,
            scratch.embedding, num_rows=state.gdn.num_tokens, session=session, name=f"{name}.embedding")
        self.layers, self.reductions, self.pipelined_reductions = [], [], []
        for index, (module, layer_weights) in enumerate(zip(language.layers, weights.layers, strict=True)):
            prefix = f"{name}.layer{index}"
            attn_hc = PreparedHyperconnection(module.attn_hyper_connection, layer_scratch.hyperconnection,
                priming_state=layer_scratch.residual, session=session, name=f"{prefix}.attn_hc",
                runtime=runtime, coordinator=coordinator)
            mlp_hc = PreparedHyperconnection(module.mlp_hyper_connection, layer_scratch.hyperconnection,
                priming_state=layer_scratch.residual, session=session, name=f"{prefix}.mlp_hc",
                runtime=runtime, coordinator=coordinator)
            if index % 4 == 3:
                attention = PreparedAttention(module.self_attn, layer_weights.attention, source,
                    layer_scratch.qsa, qsa_states[index], layer_qsa_meta, rotary, session=session, name=f"{prefix}.qsa",
                    workspace=workspaces, projection_workspace=workspaces,
                    max_cache_pages=max_cache_pages, draft_tokens=state.pool.layout.speculative_tokens,
                    shard=shard, qsa_gather_runtime=qsa_gather_runtime,
                    qsa_shard_mode=qsa_shard_mode, coordinator=coordinator)
            else:
                attention = PreparedGDN(module.linear_attn, layer_weights.attention, source,
                    layer_scratch.gdn, state.pool.gdn[index], layer_gdn_meta, mode=state.mode,
                    session=session, name=f"{prefix}.gdn", checkpoint_export=False,
                    workspace=workspaces, projection_workspace=workspaces,
                    records=None if gdn_records is None else gdn_records[index], in_place=gdn_in_place,
                    partitions=getattr(state, "gdn_partitions", None))
            moe = PreparedQwenMoe(source, layer_weights.moe, session=session, name=f"{prefix}.moe",
                scratch=layer_scratch.moe, workspace=workspaces,
                projection_workspace=workspaces, prefill=state.mode == "prefill")
            self.layers.append((attn_hc, attention, mlp_hc, moe))
            if self.pipelined_prefill:
                from ignition.components.collectives import ConcurrentAllReduce, PreparedAllReduce

                att_reduce_0 = PreparedAllReduce(attention.scratch.output, layer_scratch.hidden,
                    runtime=runtime, session=session, coordinator=coordinator, name=f"{prefix}.attention_reduce.0")
                att_reduce_1 = PreparedAllReduce(attention.scratch.output, layer_scratch.hidden,
                    runtime=runtime, session=session, coordinator=coordinator, name=f"{prefix}.attention_reduce.1")
                mlp_reduce_0 = PreparedAllReduce(moe.output, layer_scratch.hidden,
                    runtime=runtime, session=session, coordinator=coordinator, name=f"{prefix}.moe_reduce.0")
                mlp_reduce_1 = PreparedAllReduce(moe.output, layer_scratch.hidden,
                    runtime=runtime, session=session, coordinator=coordinator, name=f"{prefix}.moe_reduce.1")
                attn_plans = _hc_plans(attn_hc) + _attention_plans(attention, is_qsa=index % 4 == 3)
                moe_plans = _hc_plans(mlp_hc) + _moe_plans(moe)
                self.pipelined_reductions.append((
                    ConcurrentAllReduce(att_reduce_0, workspace=workspaces, session=session,
                        parallel_plans=attn_plans),
                    ConcurrentAllReduce(att_reduce_1, workspace=workspaces, session=session,
                        parallel_plans=moe_plans),
                    ConcurrentAllReduce(mlp_reduce_0, workspace=workspaces, session=session,
                        parallel_plans=moe_plans),
                    ConcurrentAllReduce(mlp_reduce_1, workspace=workspaces, session=session,
                        parallel_plans=_hc_plans(mlp_hc)),
                ))
            elif runtime is not None:
                from ignition.components.collectives import PreparedAllReduce

                self.reductions.append(tuple(PreparedAllReduce(value, scratch.hidden,
                    runtime=runtime, session=session, coordinator=coordinator, name=f"{prefix}.{label}_reduce")
                    for label, value in (("attention", attention.scratch.output), ("moe", moe.output))))
            if progress is not None:
                progress(f"prepared {name} layer {index + 1}/48")
        module = language.layers[1].ple
        self.ple = PreparedPLE(module,
            {key: getattr(module, key).weight for key in ("key_proj", "value_proj")},
            scratch.residual, scratch.ple, state.pool.ple_convolution, state.ple, ple_table,
            mode=state.mode, session=session, name=f"{name}.ple", workspace=workspaces,
            partitions=getattr(state, "ple_partitions", None),
            runtime=runtime if barrier_runtime is None else barrier_runtime, coordinator=coordinator)
        self.final_mixer = (
            PreparedHyperconnection(language.hyper_connection_mixer,
                replace(scratch.hyperconnection, block_input=scratch.hidden), priming_state=scratch.residual,
                session=session, name=f"{name}.final_mixer", runtime=runtime, coordinator=coordinator)
            if self.sample else None
        )
        for index, (attn_hc, _, mlp_hc, _) in enumerate(self.layers):
            attn_hc.prepare_combine_norm(mlp_hc, priming_state=layer_scratch.residual,
                session=session, name=f"{name}.layer{index}.attn_combine_norm")
            if index != 0 and (self.sample or index + 1 < len(self.layers)):
                following = self.layers[index + 1][0] if index + 1 < len(self.layers) else self.final_mixer
                mlp_hc.prepare_combine_norm(following, priming_state=layer_scratch.residual,
                    session=session, name=f"{name}.layer{index}.mlp_combine_norm")
        if self.pipelined_prefill:
            for index, (attn_hc, attention, mlp_hc, _) in enumerate(self.layers):
                cr_att_0, cr_att_1, cr_mlp_0, cr_mlp_1 = self.pipelined_reductions[index]
                attn_cn = attn_hc.plans["combine_norm"]
                workspaces.register_parallel(cr_att_1.reduce.plan, attn_cn)
                workspaces.register_parallel(cr_mlp_0.reduce.plan, attn_cn)
                if index > 0 and "combine_norm" in mlp_hc.plans:
                    workspaces.register_parallel(cr_mlp_1.reduce.plan, mlp_hc.plans["combine_norm"])
                if index >= 2:
                    prev_cr_mlp_1 = self.pipelined_reductions[index - 1][3]
                    prev_mlp_cn = self.layers[index - 1][2].plans["combine_norm"]
                    workspaces.register_parallel(cr_att_0.reduce.plan, prev_mlp_cn)
                    for plan in _hc_plans(attn_hc) + _attention_plans(attention, is_qsa=index % 4 == 3):
                        workspaces.register_parallel(prev_cr_mlp_1.reduce.plan, plan)
        if self.sample:
            state.prepare_hidden(scratch.hidden, scratch.sample_hidden)
        else:
            state.prepare_rope_delta()
        state.prepare_residual(scratch.residual)
        if not self.sample:
            self.lm_head = None
        elif vocabulary_runtime is None:
            self.lm_head = UnquantizedProjection(scratch.sample_hidden, weights.tree.lm_head.weight,
                scratch.logits, session=session, name=f"{name}.lm_head")
        else:
            from hyperflare_recipes.components.vocabulary import PreparedVocabularyProjection

            self.lm_head = PreparedVocabularyProjection(scratch.sample_hidden, weights.tree.lm_head.weight,
                scratch.local_logits, scratch.logits, runtime=vocabulary_runtime, session=session,
                coordinator=coordinator, name=f"{name}.lm_head")
        self.scoring = None if scoring is None else scoring.prepare(scratch.hidden, weights.tree.lm_head.weight,
            vocabulary=248320, session=session, coordinator=coordinator, name=name + ".prompt")
        self.sampler = None
        self.completion_recorder = None
        if not verification:
            state, inputs = self.state, self.state.inputs
            if self.sample:
                from ignition.components.sampling import PreparedSampling

                self.sampler = PreparedSampling(self.scratch.logits, state.requests,
                    inputs.row_slot_ids[:state.rows], inputs.row_generations[:state.rows],
                    inputs.live_rows, completion, frame_index, submission_sequence,
                    sample_enabled=inputs.sample_enabled[:state.rows], workspace=sampling_workspace)
                self.sampler.prepare()
                self.sampler.prepare_broadcast(runtime=vocabulary_runtime if result_runtime is None else result_runtime,
                    session=session,
                    coordinator=coordinator, name=name + ".sampling")
            else:
                from ignition.components.sampling import PreparedCompletion

                self.completion_recorder = PreparedCompletion(state.requests,
                    inputs.row_slot_ids[:state.rows], inputs.row_generations[:state.rows],
                    inputs.live_rows, completion, frame_index, submission_sequence)
                self.completion_recorder.prepare()

    def capture_context(self, metadata, draft_config, *, compact_prefill=False):
        from hyperflare_recipes.components.dflash2 import PreparedContextCapture

        self.context_layers = draft_config.target_layers
        self.context_capture = PreparedContextCapture(metadata,
            2560 * len(self.context_layers), draft_config.attention.window,
            compact_prefill=compact_prefill)
        self.auxiliary = torch.empty(self.context_capture.shape, dtype=torch.bfloat16,
                                     device=self.context_capture.device)

    def functional(self, *, session):
        if getattr(self, "_functional_requested", False):
            return self
        self._functional_requested = True
        self.embedding.functional(session=session, dynamic_inputs=True)
        for index, (attn_hc, attention, mlp_hc, moe) in enumerate(self.layers):
            attn_hc.functional(session=session)
            if index % 4 != 3:
                attention.functional(session=session, dynamic_metadata=True)
            else:
                attention.functional(session=session)
            mlp_hc.functional(session=session)
            moe.functional(session=session)
        for pair in self.reductions:
            for reduction in pair:
                reduction.functional(session=session)
        for group in self.pipelined_reductions:
            for reduction in group:
                reduction.functional(session=session)
        self.ple.functional(session=session)
        if self.final_mixer is not None:
            self.final_mixer.functional(session=session)
        if self.lm_head is not None:
            self.lm_head.functional(session=session)
        if self.sampler is not None:
            self.sampler.functional(session=session)
        if self.context_capture is not None:
            self.context_capture = self.context_capture.functional()
        if self.embedding_injection is not None:
            self.embedding_injection.functional()
        return self

    def release_outputs(self):
        self.state.functional()
        self.scratch = self.residual_output = None
        self.auxiliary = None

    def _run_pipelined(self, runtime, residual, capture):
        h = self.half_tokens
        gdn_a, gdn_b, qsa_a, qsa_b = _split_prefill_metadata(runtime.gdn, runtime.qsa, h)
        res_a, res_b = residual[:h], residual[h:]
        norm_a = norm_b = None
        normalized = None
        pending_prev = None
        last_layer = len(self.layers) - 1
        for index, (attn_hc, attention, mlp_hc, moe) in enumerate(self.layers):
            cr_att_0, cr_att_1, cr_mlp_0, cr_mlp_1 = self.pipelined_reductions[index]
            is_qsa = index % 4 == 3
            has_row_gather = is_qsa and attention.sharded is not None and attention.sharded.mode == "row"
            if index == 1:
                residual.add_(self.ple(residual, runtime.ple, getattr(runtime, "ple_partitions", None)))
                res_a, res_b = residual[:h], residual[h:]
                norm_a = norm_b = None
            prev_feat_a = None
            if pending_prev is not None:
                prev_cr_mlp_1, prev_mlp_hc, prev_p_mlp_b, prev_res_a, prev_mlp_a, prev_inj_a, prev_res_b, prev_inj_b = pending_prev
                res_a, norm_a = prev_mlp_hc.combine_norm_runtime(prev_res_a, prev_mlp_a, prev_inj_a)
                if self.layer_outputs is not None:
                    self.layer_outputs[index - 1][:h].copy_(res_a)
                if (index - 1) in self.context_layers:
                    prev_feat_a = res_a.view(-1, 4, 2560).mean(dim=1)
            mixed_a, attn_inj_a = attn_hc.mix_runtime(res_a, norm_a)
            if not is_qsa:
                out_a = attention(mixed_a, metadata=gdn_a)
                if pending_prev is not None:
                    prev_mlp_b = prev_cr_mlp_1.finish(prev_p_mlp_b, out_a)
                    att0_deps = (prev_mlp_b,)
                else:
                    att0_deps = ()
            elif has_row_gather:
                stage_a = attention.pre_gather_runtime(mixed_a, qsa_a)
                if pending_prev is not None:
                    prev_mlp_b = prev_cr_mlp_1.finish(prev_p_mlp_b, stage_a.send)
                    gather_deps_a = (prev_mlp_b,)
                else:
                    gather_deps_a = ()
                out_a = attention.post_gather_runtime(stage_a, gather_dependencies=gather_deps_a)
                att0_deps = (out_a,)
            else:
                out_a = attention(mixed_a, qsa_a)
                if pending_prev is not None:
                    prev_mlp_b = prev_cr_mlp_1.finish(prev_p_mlp_b, out_a)
                    att0_deps = (prev_mlp_b,)
                else:
                    att0_deps = ()

            p_att_a = cr_att_0.start(out_a, dependencies=att0_deps)
            if pending_prev is not None:
                res_b, norm_b = prev_mlp_hc.combine_norm_runtime(prev_res_b, prev_mlp_b, prev_inj_b)
                if self.layer_outputs is not None:
                    self.layer_outputs[index - 1][h:].copy_(res_b)
                if (index - 1) in self.context_layers:
                    prev_feat_b = res_b.view(-1, 4, 2560).mean(dim=1)
                    self.context_capture(
                        capture,
                        torch.cat((prev_feat_a, prev_feat_b), dim=0),
                        self.context_layers.index(index - 1) * 2560,
                    )
                pending_prev = None
            mixed_b, attn_inj_b = attn_hc.mix_runtime(res_b, norm_b)
            if not is_qsa:
                out_b = attention(mixed_b, metadata=gdn_b)
                att_a = cr_att_0.finish(p_att_a, out_b)
            elif has_row_gather:
                stage_b = attention.pre_gather_runtime(mixed_b, qsa_b)
                att_a = cr_att_0.finish(p_att_a, stage_b.send)
                out_b = attention.post_gather_runtime(stage_b, gather_dependencies=(att_a,))
            else:
                out_b = attention(mixed_b, qsa_b)
                att_a = cr_att_0.finish(p_att_a, out_b)

            p_att_b = cr_att_1.start(out_b, dependencies=(att_a,))
            res_a, norm_a = attn_hc.combine_norm_runtime(res_a, att_a, attn_inj_a)
            mixed_a, mlp_inj_a = mlp_hc.mix_runtime(res_a, norm_a)
            moe_a = moe(mixed_a)
            att_b = cr_att_1.finish(p_att_b, moe_a)

            p_mlp_a = cr_mlp_0.start(moe_a, dependencies=(att_b,))
            res_b, norm_b = attn_hc.combine_norm_runtime(res_b, att_b, attn_inj_b)
            mixed_b, mlp_inj_b = mlp_hc.mix_runtime(res_b, norm_b)
            moe_b = moe(mixed_b)
            mlp_a = cr_mlp_0.finish(p_mlp_a, moe_b)

            p_mlp_b = cr_mlp_1.start(moe_b, dependencies=(mlp_a,))
            if index == 0:
                res_a = mlp_hc.combine_runtime(res_a, mlp_a, mlp_inj_a)
                mlp_b = cr_mlp_1.finish(p_mlp_b, res_a)
                res_b = mlp_hc.combine_runtime(res_b, mlp_b, mlp_inj_b)
                residual = torch.cat((res_a, res_b), dim=0)
                if self.layer_outputs is not None:
                    self.layer_outputs[0].copy_(residual)
                if 0 in self.context_layers:
                    features = residual.view(-1, 4, 2560).mean(dim=1)
                    self.context_capture(capture, features, self.context_layers.index(0) * 2560)
            elif index < last_layer:
                pending_prev = (cr_mlp_1, mlp_hc, p_mlp_b, res_a, mlp_a, mlp_inj_a, res_b, mlp_inj_b)
            else:
                if self.sample:
                    res_a, norm_a = mlp_hc.combine_norm_runtime(res_a, mlp_a, mlp_inj_a)
                    mlp_b = cr_mlp_1.finish(p_mlp_b, res_a)
                    res_b, norm_b = mlp_hc.combine_norm_runtime(res_b, mlp_b, mlp_inj_b)
                    residual = torch.cat((res_a, res_b), dim=0)
                    normalized = torch.cat((norm_a, norm_b), dim=0)
                else:
                    res_a = mlp_hc.combine_runtime(res_a, mlp_a, mlp_inj_a)
                    mlp_b = cr_mlp_1.finish(p_mlp_b, res_a)
                    res_b = mlp_hc.combine_runtime(res_b, mlp_b, mlp_inj_b)
                    residual = torch.cat((res_a, res_b), dim=0)
                    normalized = None
                if self.layer_outputs is not None:
                    self.layer_outputs[index].copy_(residual)
                if index in self.context_layers:
                    features = residual.view(-1, 4, 2560).mean(dim=1)
                    self.context_capture(capture, features, self.context_layers.index(index) * 2560)
        return residual, normalized

    def __call__(self, runtime):
        from hyperflare_recipes.components.gdn_recovery import GDNLayerRecords
        self.ple.embedding_lookup.publish(runtime.ple)
        embedding = self.embedding(token_ids=runtime.ple.token_ids, num_rows=runtime.gdn.num_tokens)
        if self.embedding_injection is not None:
            self.embedding_injection.inject(embedding, runtime.injection_errors)
        residual = torch.empty((embedding.shape[0], 10240), dtype=embedding.dtype, device=embedding.device)
        residual.view(-1, 4, 2560).copy_(embedding[:, None, :])
        records = None
        if self.record_layout is not None:
            layers, rows, key_heads, value_heads, width = self.record_layout
            records = GDNLayerRecords.allocate(layers, rows, embedding.device,
                key_heads=key_heads, value_heads=value_heads, width=width)
        capture = None if self.context_capture is None else self.context_capture.begin(runtime.capture_metadata)
        if self.pipelined_prefill:
            residual, normalized = self._run_pipelined(runtime, residual, capture)
        else:
            normalized = None
            for index, (attn_hc, attention, mlp_hc, moe) in enumerate(self.layers):
                if index == 1:
                    residual.add_(self.ple(residual, runtime.ple, getattr(runtime, "ple_partitions", None)))
                    normalized = None
                mixed, injection = attn_hc.mix_runtime(residual, normalized)
                if index % 4 != 3:
                    layer_records = None if records is None or index not in records.layers else records[index]
                    block = attention(mixed, records=layer_records, metadata=runtime.gdn,
                                      partitions=getattr(runtime, "gdn_partitions", None))
                else:
                    block = attention(mixed, runtime.qsa)
                if self.reductions:
                    block = self.reductions[index][0](block)
                residual, normalized = attn_hc.combine_norm_runtime(residual, block, injection)
                mixed, injection = mlp_hc.mix_runtime(residual, normalized)
                block = moe(mixed)
                if self.reductions:
                    block = self.reductions[index][1](block)
                if index == 0 or (not self.sample and index + 1 == len(self.layers)):
                    residual = mlp_hc.combine_runtime(residual, block, injection)
                    normalized = None
                else:
                    residual, normalized = mlp_hc.combine_norm_runtime(residual, block, injection)
                if self.layer_outputs is not None:
                    self.layer_outputs[index].copy_(residual)
                if index in self.context_layers:
                    features = residual.view(-1, 4, 2560).mean(dim=1)
                    self.context_capture(capture, features, self.context_layers.index(index) * 2560)
        if not self.sample:
            self.state.finish_runtime(runtime, None, residual)
            if self.completion_recorder is not None:
                self.completion_recorder()
            return None, None, DecoderAuxiliary(residual, None if capture is None else capture[0]), records
        hidden, _ = self.final_mixer.mix_runtime(residual, normalized)
        if self.scoring is not None:
            self.scoring(hidden)
        sample = self.state.finish_runtime(runtime, hidden, residual)
        logits = self.lm_head(sample)
        if self.sampler is not None:
            self.sampler.run(logits)
        return sample, logits, DecoderAuxiliary(residual, None if capture is None else capture[0]), records
