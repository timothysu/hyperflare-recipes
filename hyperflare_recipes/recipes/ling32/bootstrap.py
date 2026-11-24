# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare Ling-3.2-Air TP=1 text and Layer-42 MTP graphs for native Helios serving."""

import contextlib
from dataclasses import dataclass, replace
from functools import partial
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from ignition.components.capacity import (
    batch_capacities,
    image_capacity,
    prefill_capacities,
    request_capacity,
    serving_modes,
    speculative_shapes,
    speculative_token_capacity,
)
from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
from hyperflare_recipes.components.kda_speculation import KDARecords
from hyperflare_recipes.components.paged_batch import (
    BatchInputs,
    Metadata,
    PreparedBatch,
    RotaryBatch,
    RotaryBatchInputs,
)
from hyperflare_recipes.components.paged_generation import PreparedGeneration
from hyperflare_recipes.components.paged_maintenance import PreparedMaintenance
from hyperflare_recipes.components.paged_resume import PreparedResume
from hyperflare_recipes.components.parallel import TensorParallelShard
from hyperflare_recipes.components.prompt_logprobs import PreparedScoredPrefill, PromptScoring, prefill_modes
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from hyperflare_recipes.components.speculation import SpeculativeRngState
from ignition.components.state_transfer import StateCommands
from ignition.components.vision_state import PrefillVisionInputs, VisionState
from ignition.preparation.bootstrap import BasePreparationConfig
from ignition.preparation.capture import capture_forwards
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from hyperflare_recipes.fixtures.recurrent import register_benchmark_state
from ignition.preparation.host_memory import release_unused
from ignition.preparation.model import PreparedModel
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.progress import phase

from .config import IMAGE_TOKEN_ID, ModelConfig, PAGE_TOKENS
from .contract import PATCH_CAPACITIES, prepare as prepare_contract
from .decoder import DecoderScratch, PreparedDecoder
from .frontend import prepare_frontend, runtime_assets
from .mtp import PreparedAcceptedContext
from .pool import ContinuationPool, continuation_layout, page_bytes
from .speculative import PreparedRound
from .vision import (
    FEATURES,
    PreparedVision,
    PreparedVisionGeneration,
    VisionInputs,
    VisionMetadata,
    VisionScratch,
    vision_math_mode,
)
from .weights import ModelWeights


@dataclass(frozen=True)
class PreparationConfig(BasePreparationConfig):
    prompt_logprobs: bool = False
    devices: tuple[int, ...] = (0,)
    context: int = 262144
    # Image request slots; zero prepares the language model only.
    image_concurrency: int = 0
    mtp: bool = True
    speculative_widths: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    stream_overlap: bool = True
    expert_scale_compression: str | None = "csf"
    draft_head_quantization: str | None = None

    def __post_init__(self):
        if self.mtp_only and (
            not self.mtp or max(self.speculative_widths, default=1) <= 1 or self.speculative_fixed_width == 1
        ):
            raise ValueError("--mtp-only requires enabled learned speculation with positive draft depth")
        if self.expert_scale_compression not in (None, "csf"):
            raise ValueError("Ling-3.2-Air expert scale compression must be None or 'csf'")
        if self.draft_head_quantization not in (None, "mxfp8", "nvfp4"):
            raise ValueError("Ling-3.2-Air draft head quantization must be None, 'mxfp8', or 'nvfp4'")
        if self.draft_head_quantization is not None and not self.mtp:
            raise ValueError("Ling-3.2-Air draft head quantization requires MTP")
        if len(self.devices) != 1 or any(type(d) is not int or d < 0 for d in self.devices):
            raise ValueError("Ling-3.2-Air preparation requires one CUDA device (TP=1)")
        request_capacity(self.max_active)
        if type(self.image_concurrency) is not int or self.image_concurrency < 0:
            raise ValueError("image concurrency must be a non-negative integer")
        if self.image_concurrency:
            image_capacity(self.max_active, self.image_concurrency)
        if not PAGE_TOKENS <= self.context <= 262144 or self.context % PAGE_TOKENS:
            raise ValueError("Ling-3.2-Air context must be 128-aligned in [128, 262144]")
        if not 128 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 128:
            raise ValueError("Ling-3.2-Air prefill chunk must be 128-aligned and fit the context")
        if self.mtp:
            speculative_shapes(self.max_active, self.speculative_widths, self.speculative_token_capacity)
            if not self.speculative_widths or max(self.speculative_widths) > 8:
                raise ValueError("Ling-3.2-Air MTP supports verification widths in 1..8")
            if self.speculative_fixed_width is not None and self.speculative_fixed_width not in self.speculative_widths:
                raise ValueError("fixed verification width must belong to the prepared widths")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")

    @property
    def speculative_shapes(self):
        return (
            speculative_shapes(self.max_active, self.speculative_widths, self.speculative_token_capacity)
            if self.mtp
            else ()
        )

    @property
    def draft_tokens(self) -> int:
        return max(self.speculative_widths) - 1 if self.mtp else 0

    @property
    def completion_token_capacity(self) -> int:
        return max((width for _, width in self.speculative_shapes), default=1)


@torch.no_grad()
@with_autotune_effort
def prepare_rank(snapshot, config: PreparationConfig, *, rank=0, group=None, progress=None, core_storage=None):
    from .recipe import RECIPE

    snapshot = Path(snapshot).resolve()
    shard = TensorParallelShard(rank, len(config.devices))
    device = torch.device("cuda", config.devices[rank])
    torch.cuda.set_device(device)

    patches = PATCH_CAPACITIES if config.image_concurrency else ()
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
        headers = [s.header_sha256 for s in manifest.shards]
        model_config = ModelConfig.load(snapshot)
        assets = runtime_assets(snapshot)
        frontend = prepare_frontend(snapshot, eos_token_ids=model_config.text["eos_token_id"], vision=bool(patches))

    weights = ModelWeights.load(
        manifest,
        device,
        shard=shard,
        progress=progress,
        mtp=config.mtp,
        vision=bool(patches),
        expert_scale_compression=config.expert_scale_compression,
        draft_head_quantization=config.draft_head_quantization,
        core_storage=core_storage,
    )
    del manifest
    release_unused()

    transient = contextlib.ExitStack()
    if core_storage is not None:
        transient.enter_context(core_storage.transient())

    drafting = weights.draft is not None
    layout = continuation_layout(
        weights.config,
        mtp=drafting,
        speculative_tokens=config.draft_tokens if drafting else 0,
        rotary_offset=bool(patches),
    )
    snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
    if snapshots < 1:
        raise ValueError("snapshot budget must hold at least one complete Ling-3.2-Air continuation")
    slots = request_capacity(config.max_active)
    blocks = config.context // PAGE_TOKENS
    shapes = config.speculative_shapes if drafting else ()
    decode_tokens = max((b * w for b, w in shapes), default=slots)
    tokens = max(config.prefill_chunk_size, decode_tokens)
    state_rows = slots + snapshots
    pages = config.prefill_chunk_size // PAGE_TOKENS + 2 * slots
    total_memory = torch.cuda.mem_get_info(device)[1]
    per_page = page_bytes(weights.config, mtp=drafting)

    with phase(progress, "Allocate model state"):
        pool = ContinuationPool.for_model(
            weights.config,
            state_rows + 1,
            pages,
            device,
            mtp=drafting,
            speculative_tokens=config.draft_tokens if drafting else 0,
            rotary_offset=bool(patches),
            storage=core_storage,
            maximum_pages=total_memory // per_page,
            maximum_rows=total_memory // layout.row_bytes,
            backed_slots=min(state_rows + 1, 1 + 2 * slots) if core_storage is not None else None,
        )
        vocabulary = weights.config.vocabulary
        requests = DeviceRequestState.allocate(device, vocab_size=vocabulary, capacity=slots)
        inputs = (RotaryBatchInputs if patches else BatchInputs).allocate(tokens, blocks, device, capacity=slots)
        draft_tables = torch.full_like(inputs.page_tables, -1) if drafting else None
        admission = AdmissionBuffers.allocate(device, vocab_size=vocabulary, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        completion = CompletionOutputs.allocate(
            (2, slots, VECTOR_TOKENS_FIELD + config.completion_token_capacity),
            device=device,
        )

        identity = hashlib.sha256(
            json.dumps(
                {
                    "config": weights.config.text,
                    "quantization": weights.config.quantization,
                    "headers": headers,
                    "world_size": 1,
                    "draft_head_quantization": config.draft_head_quantization,
                    **(
                        {}
                        if not patches
                        else {"vision": {"image_slots": config.image_concurrency, "patch_capacities": patches}}
                    ),
                    "contract": "ling32.graph-inputs.vision" if patches else "ling32.graph-inputs.text",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

        session = RecordedPreparationSession(
            compile_workers=config.compile_workers,
            collective_tuning=config.collective_tuning,
            collective_ladder=config.collective_ladder,
            autotune_effort=config.autotune_effort,
            tuning_replica=config.tuning_replica,
            native_dense_activations=config.native_dense_activations,
            lock_gpu_clocks=config.lock_gpu_clocks,
            device=device,
            checkpoint=identity,
            workload_schema="ling32",
            numerical_policy="default",
            coordinator=None,
            headroom_bytes=config.headroom_bytes,
            benchmark_storage=core_storage,
        )
        session.__enter__()
        register_benchmark_state(
            session,
            pool.reserved_backing,
            storage=pool.reserved_backing,
            restore=pool.clear_for_preparation,
        )
        arena = ScratchArena(device, overlap=config.stream_overlap)
        session.register_scratch(arena)

        examples: dict = {}
        prefill_scratch = DecoderScratch.allocate(
            config.prefill_chunk_size, 1, weights.config, device, examples=examples
        )
        decode_scratch = DecoderScratch.allocate(
            decode_tokens, decode_tokens, weights.config, device, examples=examples
        )
        vision_state = vision_inputs = vision_prefill = None
        if patches:
            vision_inputs = VisionInputs.allocate(device)
            vision_prefill = PrefillVisionInputs.allocate(config.prefill_chunk_size, device)
            vision_state = VisionState(
                inputs,
                vision_prefill,
                requests,
                image_token=IMAGE_TOKEN_ID,
                features=FEATURES,
                hidden=weights.config.hidden,
                image_slots=config.image_concurrency,
            )

        buffers = dict(
            inputs.buffers(),
            **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing,
            memory_capacity=pool.memory_capacity,
            request_pending_token=requests.pending_token,
            sampling_grammar_mask=requests.grammar_mask,
            frame_index=frame,
            submission_sequence=sequence,
            **completion.buffers(),
            **({} if not patches else {**vision_inputs.buffers(), **vision_prefill.buffers()}),
        )
        if draft_tables is not None:
            buffers["draft_page_tables"] = draft_tables

        collectives = {"runtime": None, "vocabulary_runtime": None, "coordinator": None}
        speculation = None
        if drafting:
            kda_attn_weights = {
                i: {"A_log": w.attention.A_log, "dt_bias": w.attention.dt_bias}
                for i, w in enumerate(weights.layers)
                if weights.config.layer_types[i] == "kda"
            }
            kda_records = KDARecords.allocate(
                kda_attn_weights,
                slots + 1,
                weights.config.heads,
                device,
                window=8,
            )
            speculation = SimpleNamespace(
                weights=weights,
                pool=pool,
                inputs=inputs,
                requests=requests,
                scratch=decode_scratch,
                draft_scratch=None,
                draft_tables=draft_tables,
                kda_records=kda_records,
                examples=examples,
                session=session,
                workspace=arena,
                collectives=collectives,
                overlap=config.stream_overlap,
                completion=completion,
                frame=frame,
                sequence=sequence,
                rng=SpeculativeRngState.allocate(device, capacity=slots),
                offset=torch.zeros(slots, dtype=torch.int64, device=device),
                generation=torch.zeros(slots, dtype=torch.int64, device=device),
            )

    scoring = PromptScoring.allocate(
        config, device, buffers, frame=frame, group=group, ipc=None, runtime=None, coordinator=None
    )
    forwards, signatures, metadata = {}, {}, {}
    prefills = prefill_capacities(config.prefill_chunk_size)
    batches = batch_capacities(config.max_active)
    modes = serving_modes(drafting, config.mtp_only)

    with phase(progress, "Construct graph catalog"):
        def add(name, forward, kind, rows, cap_tokens, fields, enabled=None):
            if kind == "scored_prefill":
                forward = PreparedScoredPrefill(forward, requests, inputs)
                fields = (*fields, "prompt_target_tokens")
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, cap_tokens, tuple(fields), speculation_enabled=enabled)

        base_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        spec_inputs = (*base_inputs, "draft_page_tables") if drafting else base_inputs
        # Bailing centered M-RoPE advances post-image text positions from the temporal axis (0).
        batch_cls = partial(RotaryBatch, advance_axis=0) if patches else PreparedBatch

        for enabled in modes:
            suffix = "-no-mtp" if drafting and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            mode_inputs = spec_inputs if enabled else base_inputs
            for kind, mode, sizes in prefill_modes(prefills, () if enabled else reversed(batches), scoring):
                for capacity in sizes:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{kind}{capacity}" + (suffix if mode == "prefill" else "")
                    m = metadata[name] = Metadata.allocate(capacity, rows, blocks, device)
                    scratch = (prefill_scratch if mode == "prefill" else decode_scratch).rows(capacity, rows)
                    prefill_vision = vision_state if mode == "prefill" else None
                    decoder = PreparedDecoder(
                        mode_weights,
                        scratch,
                        m,
                        pool,
                        mode=mode,
                        scoring=scoring if kind == "scored_prefill" else None,
                        vision_state=prefill_vision,
                        session=session,
                        name=name,
                        workspace=arena,
                        overlap=config.stream_overlap,
                        examples=examples,
                        **collectives,
                    )
                    batch = batch_cls(
                        inputs,
                        requests,
                        m,
                        mode=mode,
                        shard=shard,
                        pool=pool,
                        validate_prefill=None if prefill_vision is None else prefill_vision.validate_prefill,
                    )

                    def _prefill_context(
                        w_arg, aux_arg, meta_arg, pool_arg, *, session, name, workspace, pv=prefill_vision
                    ):
                        del aux_arg
                        return PreparedAcceptedContext(
                            w_arg,
                            meta_arg,
                            pool_arg,
                            draft_tables=draft_tables,
                            inputs=inputs,
                            examples=examples,
                            session=session,
                            name=name,
                            workspace=workspace,
                            is_prefill=True,
                            vision_state=pv,
                            **collectives,
                        )

                    forward = PreparedGeneration(
                        decoder,
                        batch,
                        completion,
                        frame,
                        sequence,
                        runtime=None,
                        coordinator=None,
                        session=session,
                        name=name + ".generation",
                        workspace=arena,
                        context_factory=_prefill_context if (enabled and kind != "scored_prefill") else None,
                    )
                    fields = (
                        tuple(f for f in mode_inputs if f not in ("token_ids", "prefill_positions"))
                        if mode == "decode"
                        else (*mode_inputs, *(vision_prefill.buffers() if prefill_vision is not None else ()))
                    )
                    add(name, forward, kind, rows, capacity if mode == "prefill" else 1, fields, enabled)

        vision_owners = ()
        if patches:
            vision_fields = (*vision_inputs.buffers(), "vision_feature_slot")
            owners = []
            for capacity in reversed(patches):
                s = VisionScratch.allocate(capacity, device, weights.config.hidden)
                encoder = PreparedVision(
                    weights.vision,
                    s,
                    VisionMetadata.allocate(capacity, 1, device),
                    session=session,
                    name=f"vision{capacity}",
                    attention_workspace=arena,
                    math_mode=vision_math_mode(device),
                )
                add(
                    f"vision{capacity}",
                    PreparedVisionGeneration(encoder, vision_inputs, vision_state, completion, frame, sequence),
                    "vision",
                    1,
                    capacity,
                    vision_fields,
                    None,
                )
                owners.append(encoder)
            del s
            vision_owners = tuple(owners)

        if speculation is not None:
            for batch_sz, width in sorted(shapes, key=lambda s: (s[0] * s[1], s[0]), reverse=True):
                name = f"speculative{batch_sz}x{width}"
                forward = PreparedRound(speculation, batch_sz, width, name)
                forward.prepare()
                fields = tuple(f for f in spec_inputs if f not in ("token_ids", "prefill_positions"))
                add(name, forward, "speculative", batch_sz, width, fields, True)

        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in modes:
            suffix = "-no-mtp" if drafting and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for capacity in reversed(batches):
                name = f"resume{capacity}" + suffix
                forward = PreparedResume(
                    mode_weights,
                    requests,
                    pool,
                    inputs,
                    completion,
                    frame,
                    sequence,
                    rows=capacity,
                    runtime=None,
                    vocabulary_runtime=None,
                    coordinator=None,
                    session=session,
                    name=name,
                    workspace=arena,
                )
                add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"), enabled)
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    forward = PreparedMaintenance(
                        mode,
                        requests,
                        pool,
                        commands,
                        inputs,
                        completion,
                        frame,
                        sequence,
                        runtime=None,
                        coordinator=None,
                        session=session,
                        name=name,
                        capacity=capacity,
                        admission_buffers=admission,
                        draft_state=enabled,
                        page_tokens=PAGE_TOKENS,
                        trailing_pages=pool.draft_pages() if enabled else (),
                        trailing_tables=draft_tables if enabled else None,
                    )
                    fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                    extra_tables = ("draft_page_tables",) if enabled else ()
                    add(name, forward, mode, capacity, 1, (*fields, "page_tables", *extra_tables, *commands.buffers()), enabled)

    examples.clear()
    metadata.clear()
    prefill_scratch = decode_scratch = None
    if speculation is not None:
        speculation.scratch = speculation.kda_records = None

    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None and rank == 0, console=stage.console)

    arena.release_preparation()
    stream = torch.cuda.Stream(device=device)
    graph_pool = torch.cuda.graph_pool_handle() if core_storage is None else core_storage.graph_pool.id
    graphs, compiled = capture_forwards(
        forwards,
        signatures,
        session=session,
        device=device,
        stream=stream,
        pool=graph_pool,
        collective=None,
        coordinator=None,
        compile_decoder=config.compile_decoder,
        progress=progress,
        cache_root=config.kernel_cache,
        retain_graphs=config.retain_graphs,
        warmup_storage=core_storage,
        compile_kinds=("prefill", "scored_prefill", "decode", "resume", "speculative", "vision"),
    )
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    release_unused()

    server_config = {
        "identity": identity,
        "model_id": snapshot.name,
        "max_sequence_tokens": config.context,
        "prefill_chunk_size": config.prefill_chunk_size,
        "pages": pages,
        "state_rows": state_rows,
        "snapshots": snapshots,
        "page_tokens": PAGE_TOKENS,
        "frontend": frontend,
        "sampling_defaults": RECIPE.sampling_defaults.as_dict(),
        "assets": assets,
        "model_contract": prepare_contract(
            signatures,
            buffers,
            pool,
            vocabulary=vocabulary,
            page_tokens=PAGE_TOKENS,
            draft_lag=1 if drafting else None,
            image_slots=config.image_concurrency,
        ),
    }
    if drafting:
        server_config.update(
            speculative_widths=config.speculative_widths,
            speculative_fixed_width=config.speculative_fixed_width,
            speculative_token_capacity=speculative_token_capacity(
                config.max_active, max(config.speculative_widths), config.speculative_token_capacity
            ),
        )

    free, total = torch.cuda.mem_get_info(device)
    memory = {
        "rank": rank,
        "world_size": 1,
        "device": config.devices[rank],
        "weights_bytes": weights.allocated_bytes,
        "expert_scale_compression": config.expert_scale_compression,
        "mapped_host_bytes": 0,
        "offloaded_bytes": 0,
        "continuation_bytes": pool.backing.nbytes,
        "state_rows": state_rows,
        "preparation_state_rows": int(pool.memory_capacity[0].item()),
        "preparation_state_bytes": int(pool.memory_capacity[0].item()) * layout.row_bytes,
        "pages": pages,
        "page_bytes": per_page,
        "page_tokens": PAGE_TOKENS,
        "kv_bytes": pages * per_page,
        "completion_output_bytes": completion.nbytes,
        "operator_scratch_bytes": arena.nbytes,
        "graphs": len(graphs),
        "kv_dtype": "bfloat16",
        "prefill_capacities": prefills,
        "free_device_bytes": free,
        "total_device_bytes": total,
        "headroom_bytes": config.headroom_bytes,
        "tuning": tuning,
        "compiled_decoder": config.compile_decoder,
        "speculative_shapes": shapes,
        "speculative_token_capacity": server_config.get("speculative_token_capacity", 0),
        **({} if vision_state is None else {"vision_feature_bytes": vision_state.arena.nbytes}),
    }
    if scoring is not None:
        memory.update(scoring.memory_report())

    disposable = [arena.tensor, weights.scale_scratch]
    if vision_state is not None:
        disposable.append(vision_state.arena)
    for forward in forwards.values():
        if isinstance(forward, PreparedScoredPrefill):
            forward = forward.forward
        if hasattr(forward, "completion") and hasattr(forward.completion, "scratch_tensors"):
            disposable.append(forward.completion.scratch_tensors)

    weight_components = ("embedding", "norm", "head", "draft", *(("vision",) if weights.vision is not None else ()))
    return PreparedModel(
        rebuild_weights=declare_weights(weights, components=weight_components),
        device=config.devices[rank],
        graphs=graphs,
        buffers=buffers,
        signatures=signatures,
        max_active=config.max_active,
        session=session,
        server_config=server_config,
        memory_report=memory,
        export_zero_ranges=(
            *(() if scoring is None else scoring.zero_ranges()),
            *storage_ranges(disposable),
            *completion.zero_ranges(),
        ),
        resources=(
            scoring,
            weights,
            pool,
            requests,
            inputs,
            draft_tables,
            admission,
            commands,
            arena,
            forwards,
            compiled,
            stream,
            graph_pool,
            session,
            speculation,
            vision_state,
            vision_inputs,
            vision_prefill,
            vision_owners,
        ),
        close_resources=(scoring,) if scoring is not None else (),
    )
