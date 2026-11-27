# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare Inkling 2 text, vision and complete DSpark generation graphs."""

import contextlib
import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import torch

from ignition.components.capacity import batch_capacities, prefill_capacities, request_capacity, image_capacity, parallel_speculative_shapes, serving_modes
from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
from hyperflare_recipes.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from hyperflare_recipes.components.paged_generation import PreparedGeneration
from hyperflare_recipes.components.paged_maintenance import PreparedMaintenance
from hyperflare_recipes.components.paged_resume import PreparedResume
from hyperflare_recipes.components.parallel import TensorParallelShard
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.capture import capture_forwards
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.collectives import (CollectivePreparation, create_all_reduce_runtime, create_all_gather_runtime,
                                              create_roce_runtime)
from ignition.preparation.execution import GraphSignature
from hyperflare_recipes.fixtures.recurrent import register_benchmark_state
from ignition.preparation.host_memory import release_unused
from ignition.preparation.model import PreparedModel
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation.rank_group import minimum_across_ranks
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.preparation.tuning_replicas import TuningReplica
from ignition.progress import phase

from .config import ModelConfig
from .contract import prepare as prepare_contract
from .decoder import PreparedDecoder, Scratch
from .weights import ModelWeights
from .pool import PAGE_TOKENS, ContinuationPool, RankGeometry, continuation_layout, page_specs
from .recipe import Inkling2Parameters, RECIPE
from ignition.preparation.bootstrap import BasePreparationConfig, validate_host_topology


def _default_parameters() -> Inkling2Parameters:
    return replace(RECIPE.parameters, autotune_effort="off", compiled_forwards=False)


@dataclass(frozen=True)
class PreparationConfig(BasePreparationConfig):
    parameters: Inkling2Parameters = field(default_factory=_default_parameters)
    devices: tuple[int, ...] = (0, 1, 2, 3)
    # Device ordinals are host-local when each rank runs on its own host.
    hosts: tuple[str, ...] | None = None
    context: int = 1048576
    decoder_layers: int | None = None
    benchmark_selection: object | None = None

    @property
    def native_dense_activations(self) -> bool:
        return self.parameters.native_dense_activations.enabled

    @property
    def drafting(self) -> bool:
        return self.parameters.speculation is not None

    @property
    def vision(self) -> bool:
        return self.parameters.vision is not None and not self.parameters.vision.language_model_only

    @property
    def draft_head_quantization(self) -> str | None:
        if self.parameters.draft_quantization is None:
            return None
        return self.parameters.draft_quantization.draft_head_quantization

    @property
    def markov_head_quantization(self) -> str | None:
        if self.parameters.draft_quantization is None:
            return None
        return self.parameters.draft_quantization.markov_head_quantization

    def __post_init__(self):
        if not isinstance(self.parameters, Inkling2Parameters):
            raise TypeError("Inkling 2 PreparationConfig requires Inkling2Parameters")
        placements = tuple(zip(self.hosts or ("",) * len(self.devices), self.devices))
        if (not isinstance(self.devices, tuple) or len(self.devices) not in (1, 4)
                or len(set(placements)) != len(self.devices)
                or self.hosts is not None and (not isinstance(self.hosts, tuple) or len(self.hosts) != len(self.devices))
                or any(type(d) is not int or d < 0 for d in self.devices)):
            raise ValueError("Inkling 2 preparation requires 1 or 4 distinct GPUs")
        validate_host_topology(self.roce_ring, self.roce_chain, self.hosts)
        request_capacity(self.max_active)
        if not 1 <= self.max_active <= 32:
            raise ValueError("Inkling 2 request capacity must be in [1, 32]")
        if not PAGE_TOKENS <= self.context <= 1048576 or self.context % PAGE_TOKENS:
            raise ValueError("Inkling 2 context must be 128-aligned in [128, 1048576]")
        if not PAGE_TOKENS <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % PAGE_TOKENS:
            raise ValueError("Inkling 2 prefill chunks must be 128-aligned and fit the context")
        if self.prompt_logprobs:
            raise ValueError("Inkling 2 does not capture prompt scoring graphs")
        if self.decoder_layers is not None and (type(self.decoder_layers) is not int or not 6 <= self.decoder_layers <= 48):
            raise ValueError("decoder layer count must be in [6, 48]")
        if self.drafting and self.decoder_layers not in (None, 48):
            raise ValueError("the bundled DSpark drafter requires all target layers")
        if self.mtp_only and (not self.drafting or max(self.speculative_widths, default=1) <= 1
                              or self.speculative_fixed_width == 1):
            raise ValueError("--mtp-only requires enabled learned speculation with positive draft depth")
        if self.drafting and (not self.speculative_widths or not set(self.speculative_widths) <= set(range(1, 9))):
            raise ValueError("Inkling 2 verification widths must be in [1, 8]")
        if self.draft_head_quantization not in (None, "nvfp4", "mxfp8"):
            raise ValueError(f"unsupported DSpark draft head quantization: {self.draft_head_quantization!r}")
        if self.markov_head_quantization not in (None, "nvfp4"):
            raise ValueError(f"unsupported DSpark Markov head quantization: {self.markov_head_quantization!r}")
        if (self.draft_head_quantization is not None or self.markov_head_quantization is not None) and not self.drafting:
            raise ValueError("DSpark proposal head quantization requires drafting")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")
        if self.vision:
            image_capacity(self.max_active, self.image_concurrency)
        _ = self.speculative_round_shapes

    @property
    def speculative_round_shapes(self):
        if self.benchmark_selection is not None:
            return self.benchmark_selection.speculative_round_shapes
        return (parallel_speculative_shapes(self.max_active, self.speculative_widths,
            self.speculative_token_capacity, self.speculative_graph_budget or 64,
            fixed_width=self.speculative_fixed_width)
            if self.drafting else ())

    @property
    def draft_tokens(self):
        return max(self.speculative_widths) - 1 if self.drafting else 0

    @property
    def decode_tokens(self):
        return max((p for _, _, p in self.speculative_round_shapes), default=request_capacity(self.max_active))

    @property
    def decode_samples(self):
        return max((b * w for b, w, _ in self.speculative_round_shapes), default=request_capacity(self.max_active))


def _logit_gather_bytes(rows, width):
    """Per-rank bytes of an FP32 `[rows, width]` vocabulary-shard gather."""
    return rows * width * 4


def _collective_runtimes(config, group, device, *, coordinator, core_storage, hidden, local_vocabulary, draft_block):
    """Allocate the hidden reduction and the vocabulary, sampling and log-probability gather channels.

    `hidden` is the model width, `local_vocabulary` one rank's padded vocabulary columns and `draft_block`
    the positions the drafter reduces per request. Ranks on separate hosts run every collective in
    rank-serial order over one RoCEnante runtime, which all four roles share; the sampling and
    log-probability channels are None without drafting.
    """
    slots, world = request_capacity(config.max_active), len(config.devices)
    if world == 1:
        return None, None, None, None
    tokens = max(config.prefill_chunk_size, config.decode_tokens)
    vocabulary_rows = max(config.decode_samples, slots)
    if coordinator.spans_hosts:
        # One-shot reductions cover decode and verification rows and the drafter's block of every request;
        # prefill chunks reduce through the pipelined two-shot path. The largest gather is each rank's FP32
        # vocabulary partition of every sampled row, carried as BF16 column pairs: 4 * local_vocabulary
        # bytes per row. A row of the other gathers is at most 1024 bytes (the FP32 log-probability summary),
        # and the packed sampling results of all rows fit in one vocabulary row.
        runtime = create_roce_runtime(group, device,
            max_tokens=max(config.decode_tokens, slots * draft_block if config.drafting else 0), pipe_tokens=tokens,
            hidden_size=hidden, gather_bytes=vocabulary_rows * local_vocabulary * 4,
            ring=config.roce_ring, ring_open=config.roce_chain, core_storage=core_storage)
        drafting = runtime if config.drafting else None
        return runtime, runtime, drafting, drafting
    ipc = core_storage.peer_storage
    runtime = create_all_reduce_runtime(group, device, max_tokens=min(tokens, 1024),
        hidden_size=hidden, element_bytes=4, ipc=ipc)
    vocabulary = create_all_gather_runtime(group, device, max_rows=vocabulary_rows,
        columns=2 * local_vocabulary * world, ipc=ipc)
    sampling = (create_all_gather_runtime(group, device, max_rows=config.decode_samples,
        columns=64 * world, ipc=ipc) if config.drafting else None)
    logprob = (create_all_gather_runtime(group, device, max_rows=config.decode_samples,
        columns=512 * world, ipc=ipc) if config.drafting else None)
    return runtime, vocabulary, sampling, logprob


class _Softcapped:
    def __init__(self, projection, softcap):
        self.projection, self.softcap = projection, softcap

    def __call__(self, source):
        return self.softcap * torch.tanh(self.projection(source) / self.softcap)


class InklingResume(PreparedResume):
    """Resumed sampling from the last hidden state through the softcapped FP32 vocabulary shards.

    The gathered logits stop at the valid vocabulary size, so the sampler never
    reads the masked IDs.
    """

    def __init__(self, weights, *args, **kwargs):
        super().__init__(weights, *args, logits_dtype=torch.float32, **kwargs)
        self.projection = _Softcapped(self.projection, weights.config.text.softcap)


@torch.no_grad()
@with_autotune_effort
def prepare_rank(snapshot, config, *, rank, group, progress=None, core_storage):
    from .frontend import prepare as prepare_frontend, runtime_assets
    from .recipe import RECIPE, MINI_RECIPE

    snapshot = Path(snapshot).resolve()
    world = len(config.devices)
    shard = TensorParallelShard(rank, world)
    device = torch.device("cuda", config.devices[rank])
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
        raise ValueError("Inkling 2 preparation requires SM120 or SM121 GPUs")
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
        headers = [entry.header_sha256 for entry in manifest.shards]
        model = ModelConfig.load(snapshot)
        model.validate_shard(shard)
        recipe = MINI_RECIPE if model.text.loop is not None else RECIPE
        decoder_type = PreparedDecoder
        if model.text.loop is not None:
            from .mini_decoder import PreparedLoopDecoder
            decoder_type = PreparedLoopDecoder
    # Continuation rows and pages hold the prepared layers only.
    text = (model.text if config.decoder_layers is None else model.text.prefix(config.decoder_layers)).cache_config
    if not text.global_layers:
        raise ValueError("a decoder-layer prefix must include a global-attention layer, whose K/V use pages")
    with phase(progress, "Load weights"):
        weights = ModelWeights.load(manifest, device, shard=shard, layers=config.decoder_layers,
                                    progress=progress, drafting=config.drafting, vision=config.vision,
                                    draft_head_quantization=config.draft_head_quantization,
                                    markov_head_quantization=config.markov_head_quantization)
        weights_bytes = torch.cuda.memory_allocated(device)
    if torch.cuda.get_device_properties(device).is_integrated:
        from ignition.preparation.host_memory import release_page_cache

        # Unified memory counts cached checkpoint pages against the GPU; drop them after loading.
        release_page_cache(manifest.shard_paths)
    del manifest
    release_unused()
    transient = contextlib.ExitStack()
    transient.enter_context(core_storage.transient())
    geometry = RankGeometry.of(text, world)
    draft = model.draft if config.drafting else None
    layout = continuation_layout(text, geometry, draft)
    page_bytes = sum(shape[0] for _, shape in page_specs(text, geometry, draft))
    snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
    if snapshots < 1:
        raise ValueError("snapshot budget must hold a complete Inkling 2 continuation")
    slots, blocks = request_capacity(config.max_active), config.context // PAGE_TOKENS
    tokens = max(config.prefill_chunk_size, config.decode_tokens)
    state_rows = slots + snapshots
    pages = -(-config.prefill_chunk_size // PAGE_TOKENS) + 2 * slots
    total_memory = minimum_across_ranks(torch.cuda.mem_get_info(device)[1], group)
    vocabulary = text.valid_vocabulary
    with phase(progress, "Allocate model state"):
        pool = ContinuationPool.allocate(text, geometry, state_rows + 1, pages, device, storage=core_storage,
            maximum_pages=total_memory // page_bytes, maximum_rows=total_memory // layout.row_bytes,
            backed_slots=1 + slots, draft=draft)
        requests = DeviceRequestState.allocate(device, vocab_size=vocabulary, capacity=slots)
        inputs = BatchInputs.allocate(tokens, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=vocabulary, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        width = max((w for _, w, _ in config.speculative_round_shapes), default=1)
        completion = CompletionOutputs.allocate((2, slots, VECTOR_TOKENS_FIELD + width), device=device)
        coordinator = CollectivePreparation(group) if group is not None else None
        runtime, vocabulary_runtime, sampling_runtime, logprob_runtime = _collective_runtimes(
            config, group, device, coordinator=coordinator, core_storage=core_storage,
            hidden=text.hidden, local_vocabulary=weights.head.shape[0],
            draft_block=model.draft.block)
        identity_payload = {"text": model.text.__dict__ | {"local_layers": sorted(model.text.local_layers)},
            "quantization": model.quantization, "headers": headers, "world_size": world,
            "decoder_layers": len(weights.layers), "contract": "inkling2.paged-text.v1"}
        if config.draft_head_quantization is not None:
            identity_payload["draft_head_quantization"] = config.draft_head_quantization
        if config.markov_head_quantization is not None:
            identity_payload["markov_head_quantization"] = config.markov_head_quantization
        identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True, default=str).encode()).hexdigest()
        session = RecordedPreparationSession(compile_workers=config.compile_workers,
            native_dense_activations=config.native_dense_activations, lock_gpu_clocks=config.lock_gpu_clocks,
            tuning_replica=config.tuning_replica,
            device=device, checkpoint=identity, workload_schema="inkling2", numerical_policy="default",
            coordinator=coordinator, headroom_bytes=config.headroom_bytes, benchmark_storage=core_storage,
            autotune_effort=config.autotune_effort)
        session.__enter__()
        register_benchmark_state(session, pool.reserved_backing, storage=pool.reserved_backing,
                                 restore=pool.clear_for_preparation)
        arena = ScratchArena(device)
        session.register_scratch(arena)
        from ignition.components.vision_state import PrefillVisionInputs, VisionState
        from .frontend import IMAGE_TOKEN_ID
        vision_inputs = vision_prefill = vision_state = None
        patches = ()
        if config.vision:
            from .vision_state import FEATURES, PATCH_CAPACITIES, VisionInputs
            patches = PATCH_CAPACITIES
            vision_inputs = VisionInputs.allocate(device)
            vision_prefill = PrefillVisionInputs.allocate(config.prefill_chunk_size, device)
            vision_state = VisionState(inputs, vision_prefill, requests, image_token=IMAGE_TOKEN_ID,
                features=FEATURES, hidden=text.hidden, image_slots=config.image_concurrency)
        buffers = dict(inputs.buffers(), **commands.buffers(),
            **({} if vision_inputs is None else vision_inputs.buffers()),
            **({} if vision_prefill is None else vision_prefill.buffers()),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity,
            request_pending_token=requests.pending_token, sampling_grammar_mask=requests.grammar_mask,
            frame_index=frame, submission_sequence=sequence, **completion.buffers())
    forwards, signatures = {}, {}
    prefills, batches = prefill_capacities(config.prefill_chunk_size), batch_capacities(config.max_active)
    examples = {}
    prefill_storage = Scratch.allocate(model, shard, config.prefill_chunk_size, 1, device)
    decode_storage = Scratch.allocate(model, shard, config.decode_tokens, config.decode_samples, device)
    collectives = dict(runtime=runtime, vocabulary_runtime=vocabulary_runtime, coordinator=coordinator,
                       vision_state=vision_state, examples=examples)
    speculation = None
    if config.drafting:
        from .dspark import SpeculativePreparation
        speculation = SpeculativePreparation(weights, pool, requests, inputs, decode_storage, decoder=decoder_type,
            completion=completion, frame=frame, sequence=sequence, session=session, workspace=arena,
            collectives=collectives, draft_tokens=config.draft_tokens, sampling_runtime=sampling_runtime,
            logprob_runtime=logprob_runtime)
    with phase(progress, "Construct graph catalog"):
        def add(name, forward, kind, rows, tokens, fields, enabled=None, packed=0):
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields),
                speculation_enabled=enabled, packed_token_capacity=packed)

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        for enabled in serving_modes(config.drafting, config.mtp_only):
            mode_weights = weights if enabled else replace(weights, draft=None)
            suffix = "-no-mtp" if config.drafting and not enabled else ""
            for mode, sizes in (("prefill", reversed(prefills)), ("decode", () if enabled else reversed(batches))):
                for capacity in sizes:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{mode}{capacity}" + (suffix if mode == "prefill" else "")
                    m = Metadata.allocate(capacity, rows, blocks, device)
                    storage = prefill_storage if mode == "prefill" else decode_storage
                    decoder = decoder_type(mode_weights, None, storage.rows(capacity, rows), m, pool,
                        mode=mode, session=session, workspace=arena, name=name, **collectives)
                    batch = PreparedBatch(inputs, requests, m, mode=mode, shard=shard, pool=pool,
                        validate_prefill=vision_state.validate_prefill if mode == "prefill" and vision_state else None)
                    forward = PreparedGeneration(decoder, batch, completion, frame, sequence, runtime=runtime,
                        coordinator=coordinator, session=session, name=name + ".generation", workspace=arena,
                        context_factory=speculation.forwards.context if enabled else None)
                    fields = (*forward_inputs, *(vision_prefill.buffers() if mode == "prefill" and vision_prefill else ()))
                    add(name, forward, mode, rows, capacity if mode == "prefill" else 1, fields, enabled)
        if speculation is not None:
            from hyperflare_recipes.components.parallel_draft import PreparedParallelDraftRound
            for batch, width, packed in sorted(config.speculative_round_shapes, key=lambda x: (x[2], x[0]), reverse=True):
                name = f"speculative{batch}x{width}-tokens{packed}"
                forward = PreparedParallelDraftRound(speculation, batch, width, name, target_token_capacity=packed)
                forward.prepare()
                add(name, forward, "speculative", batch, width, forward_inputs, True, packed)
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in serving_modes(config.drafting, config.mtp_only):
            suffix = "-no-mtp" if config.drafting and not enabled else ""
            for capacity in reversed(batches):
                name = f"resume{capacity}" + suffix
                forward = InklingResume(weights, requests, pool, inputs, completion, frame, sequence, rows=capacity,
                    runtime=runtime, vocabulary_runtime=vocabulary_runtime, coordinator=coordinator,
                    session=session, name=name, workspace=arena)
                add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"), enabled)
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    forward = PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame, sequence,
                        runtime=runtime, coordinator=coordinator, session=session, name=name, capacity=capacity,
                        admission_buffers=admission, draft_state=enabled)
                    fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                    add(name, forward, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()), enabled)
        for capacity in reversed(patches):
            from transformers.models.inkling2.configuration_inkling2 import Inkling2VisionConfig
            from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel
            from .vision import PreparedVision, VisionMetadata, VisionScratch
            from .vision_state import PreparedVisionGeneration
            raw = json.loads((snapshot / "config.json").read_text())
            with torch.device("meta"):
                module = Inkling2VisionModel(Inkling2VisionConfig.from_dict(
                    raw["vision_config"], **(dict(text_hidden_size=model.text.hidden) if model.text.loop is not None else {})))
            name = f"vision{capacity}"
            encoder = PreparedVision(module, VisionScratch.allocate(capacity, device, hidden=model.text.hidden),
                VisionMetadata.allocate(capacity, 1, device), weights.vision,
                session=session, name=name, projection_workspace=arena, attention_workspace=arena)
            encoder.functional(session=session)
            forward = PreparedVisionGeneration(encoder, vision_inputs, vision_state, completion, frame, sequence)
            add(name, forward, "vision", 1, capacity, (*vision_inputs.buffers(), "vision_feature_slot"))
    if speculation is not None:
        speculation.release_examples()
    examples.clear()
    prefill_storage = decode_storage = None
    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None and rank == 0, console=stage.console)
    stream, graph_pool = torch.cuda.Stream(device=device), core_storage.graph_pool.id
    arena.release_preparation()
    graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device, stream=stream,
        pool=graph_pool, collective=runtime, coordinator=coordinator, compile_decoder=config.compile_decoder,
        progress=progress, cache_root=config.kernel_cache, retain_graphs=config.retain_graphs,
        materialize=pool.clear_for_preparation, warmup_storage=core_storage,
        compile_kinds=("prefill", "decode", "resume", "init", "restore", "snapshot", "speculative", "vision"))
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    release_unused()
    server_config = dict(identity=identity, model_id=recipe.served_model_name, max_sequence_tokens=config.context,
        prefill_chunk_size=config.prefill_chunk_size, pages=pages, state_rows=state_rows, snapshots=snapshots,
        page_tokens=PAGE_TOKENS, frontend=prepare_frontend(snapshot, vision=config.vision), assets=runtime_assets(snapshot),
        sampling_defaults=recipe.sampling_defaults.as_dict(),
        model_contract=prepare_contract(signatures, buffers, pool, vocabulary=vocabulary,
            image_slots=config.image_concurrency, vision=config.vision, drafting=config.drafting))
    if config.drafting:
        server_config.update(speculative_widths=tuple(sorted({w for _, w, _ in config.speculative_round_shapes})),
            speculative_token_capacity=config.speculative_token_capacity,
            speculative_fixed_width=config.speculative_fixed_width)
    free, total = torch.cuda.mem_get_info(device)
    memory = dict(rank=rank, world_size=world, device=config.devices[rank], decoder_layers=len(weights.layers),
        weights_bytes=weights_bytes, mapped_host_bytes=0, offloaded_bytes=0, continuation_bytes=pool.backing.nbytes,
        state_rows=state_rows, preparation_state_rows=int(pool.memory_capacity[0].item()),
        preparation_state_bytes=int(pool.memory_capacity[0].item()) * layout.row_bytes, pages=pages,
        page_bytes=page_bytes, page_tokens=PAGE_TOKENS, kv_bytes=pages * page_bytes, kv_dtype="float8_e4m3fn",
        operator_scratch_bytes=arena.nbytes, graph_buffer_bytes=0, private_buffer_bytes=0, graphs=len(graphs),
        prefill_capacities=prefills, free_device_bytes=free, total_device_bytes=total,
        allocated_tensor_bytes=torch.cuda.memory_allocated(device),
        reserved_tensor_bytes=torch.cuda.memory_reserved(device), headroom_bytes=config.headroom_bytes,
        tuning=tuning, compiled_decoder=config.compile_decoder)
    # Each decoder layer's weights form one stable payload; vocabulary shards, norms and the output query share misc.
    components = ("embedding", "embedding_norm", "final_norm", "output_query", "unembedding", "draft", "vision")
    if weights.cross_pass_queries_t is not None:
        components = (*components, "cross_pass_queries_t")
    rebuild_weights = declare_weights(weights, components=components)
    return PreparedModel(rebuild_weights=rebuild_weights, device=config.devices[rank], graphs=graphs, buffers=buffers,
        signatures={name: signatures[name] for name in graphs}, max_active=config.max_active, session=session,
        server_config=server_config, memory_report=memory,
        forwards={name: compiled.get(name, forwards[name]) for name in graphs},
        export_zero_ranges=completion.zero_ranges(),
        resources=(weights, pool, requests, inputs, admission, commands, completion, forwards, arena, session,
                   compiled, stream, graph_pool, coordinator, runtime, vocabulary_runtime, sampling_runtime, logprob_runtime, speculation, vision_state),
        # Ranks on separate hosts share one runtime across these roles; close it once.
        close_resources=tuple(dict.fromkeys(
            r for r in (runtime, vocabulary_runtime, sampling_runtime, logprob_runtime) if r is not None)))
