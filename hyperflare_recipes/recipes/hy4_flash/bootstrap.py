# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare Hy4-flash text, Engram, DFly2, and SigLIP2 vision graphs on a single Blackwell GPU."""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import dataclass, replace
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
from hyperflare_recipes.components.paged_batch import Metadata, RotaryBatchInputs, prepare_batch
from hyperflare_recipes.components.paged_generation import PreparedGeneration
from hyperflare_recipes.components.paged_maintenance import PreparedMaintenance
from hyperflare_recipes.components.paged_resume import PreparedResume
from hyperflare_recipes.components.parallel import TensorParallelShard
from hyperflare_recipes.components.prompt_logprobs import (
    PreparedScoredPrefill,
    PromptScoring,
    prefill_graph_count,
    prefill_modes,
)
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from hyperflare_recipes.components.speculation import SpeculativeRngState
from ignition.components.state_transfer import StateCommands
from ignition.components.vision_state import PrefillVisionInputs
from ignition.preparation.bootstrap import BasePreparationConfig
from ignition.preparation.capture import capture_forwards
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.frontend import annotated_template_asset, prepare_frontend
from ignition.preparation.host_memory import release_unused
from ignition.preparation.model import PreparedModel
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.progress import phase

from .config import ModelConfig, VOCABULARY
from .contract import FEATURES, PATCH_CAPACITIES, prepare as prepare_contract
from .decoder import PreparedDecoder, Scratch
from .dfly import PreparedDFlySpeculativeRound
from .engram import Tables
from .pool import ContinuationPool, PAGE_TOKENS, continuation_layout, page_bytes
from .vision import PreparedVision, VisionInputs, VisionScratch
from .vision_state import PreparedVisionGeneration, VisionState
from .weights import ModelWeights

ASSETS = Path(__file__).with_name("assets")
_COMPILED_KINDS = ("prefill", "scored_prefill", "decode", "resume", "speculative", "vision")


def chat_template_asset(snapshot: Path) -> dict:
    """Annotate the bundled Hy4-flash chat template with exact message-boundary marks."""
    source = (Path(snapshot) / "chat_template.jinja").read_bytes()
    return annotated_template_asset(source, ASSETS / "chat_template.patch.json")


class _NoOpDraftContext:
    """Satisfies PreparedGeneration's required context_factory when weights.draft is present.

    PreparedGeneration (hyperflare_recipes/components/paged_generation.py) requires a
    context_factory whenever sampling with weights.draft is enabled; Hy4-flash saves
    DFly2 target logical indices directly inside PreparedDecoder rather than through a
    separate draft-context object.
    """

    def __init__(self, *args, **kwargs):
        pass

    def functional(self):
        return self

    def __call__(self, *args, **kwargs):
        return None


def _collect_disposable_storage(
    *,
    arena_tensor,
    scale_scratch,
    engram_buffers,
    vision_arena,
    forwards: dict,
) -> list:
    """Collect post-capture disposable buffers that export may zero-fill."""
    disposable: list = [
        buf for buf in (arena_tensor, scale_scratch, engram_buffers, vision_arena) if buf is not None
    ]
    for forward in forwards.values():
        if isinstance(forward, PreparedScoredPrefill):
            forward = forward.forward
        if isinstance(forward, PreparedGeneration):
            disposable.append(forward.completion.scratch_tensors)
    return disposable


@dataclass(frozen=True)
class PreparationConfig(BasePreparationConfig):
    prompt_logprobs: bool = False
    devices: tuple[int, ...] = (0,)
    context: int = 1048576
    prefill_chunk_size: int = 4096
    max_active: int = 8
    minimum_pages: int = 8
    kv_pages: int | None = None
    image_concurrency: int | None = None
    language_model_only: bool = False
    dfly: bool = True
    speculative_widths: tuple[int, ...] = tuple(range(1, 7))
    speculative_token_capacity: int | None = 48
    draft_head_quantization: str | None = None
    online_mxfp8: bool = False
    online_mxfp8_scope: str = "all"
    stream_overlap: bool = True
    expert_scale_compression: str | None = "csf"
    chat_template: dict | None = None

    def __post_init__(self):
        if self.mtp_only and (
            not self.dfly
            or max(self.speculative_widths, default=1) <= 1
            or self.speculative_fixed_width == 1
        ):
            raise ValueError("--mtp-only requires enabled learned speculation with positive draft depth")
        if self.draft_head_quantization not in (None, "nvfp4", "mxfp8"):
            raise ValueError("Hy4-flash draft head quantization must be omitted, 'nvfp4', or 'mxfp8'")
        if self.draft_head_quantization is not None and (
            not self.dfly or max(self.speculative_widths, default=1) <= 1
        ):
            raise ValueError("--draft-head-quantization requires enabled learned speculation with positive draft depth")
        if self.online_mxfp8_scope not in ("all", "draft"):
            raise ValueError("Hy4-flash online MXFP8 scope must be 'all' or 'draft'")
        if self.online_mxfp8_scope != "all" and not self.online_mxfp8:
            raise ValueError("--online-mxfp8-scope requires --online-mxfp8")
        if len(self.devices) != 1 or any(type(d) is not int or d < 0 for d in self.devices):
            raise ValueError("Hy4-flash preparation requires a single CUDA device (TP=1)")
        if self.expert_scale_compression not in (None, "csf"):
            raise ValueError("Hy4-flash expert scale compression must be None or 'csf'")
        if not PAGE_TOKENS <= self.context <= 1048576 or self.context % PAGE_TOKENS:
            raise ValueError(f"Hy4-flash context must be {PAGE_TOKENS}-aligned in [{PAGE_TOKENS},1048576]")
        if not 64 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 64:
            raise ValueError("Hy4-flash prefill chunk size must be 64-aligned in [64,context]")
        request_capacity(self.max_active)
        if self.speculative_widths and max(self.speculative_widths) > 8:
            raise ValueError("Hy4-flash DFly2 supports at most seven draft tokens")
        self.speculative_shapes
        if self.speculative_fixed_width is not None and self.speculative_fixed_width not in self.speculative_widths:
            raise ValueError("fixed speculative width must belong to the prepared widths")
        if not self.language_model_only:
            object.__setattr__(self, "image_concurrency", image_capacity(self.max_active, self.image_concurrency))
        if self.minimum_pages <= 0 or self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("page and snapshot budgets must be positive and headroom nonnegative")
        if self.kv_pages is not None and (type(self.kv_pages) is not int or self.kv_pages < self.minimum_pages):
            raise ValueError("kv_pages must cover the minimum page capacity")

    @property
    def request_capacity(self) -> int:
        return request_capacity(self.max_active)

    @property
    def prefill_capacities(self) -> tuple[int, ...]:
        return prefill_capacities(self.prefill_chunk_size)

    @property
    def batch_capacities(self) -> tuple[int, ...]:
        return tuple(reversed(batch_capacities(self.max_active)))

    @property
    def speculative_shapes(self) -> tuple[tuple[int, int], ...]:
        return (
            speculative_shapes(self.max_active, self.speculative_widths, self.speculative_token_capacity)
            if self.dfly
            else ()
        )

    @property
    def decode_tokens(self) -> int:
        return max(self.request_capacity, max((b * w for b, w in self.speculative_shapes), default=0))

    @property
    def completion_token_capacity(self) -> int:
        return max((w for _, w in self.speculative_shapes), default=1) if self.dfly else 1


@torch.no_grad()
@with_autotune_effort
def prepare_rank(
    snapshot: Path,
    config: PreparationConfig,
    *,
    rank: int = 0,
    group=None,
    progress=None,
    core_storage=None,
) -> PreparedModel:
    """Load, tune, and capture Hy4-flash text, DFly2, and vision graphs on one GPU."""
    from .recipe import RECIPE

    if rank != 0 or group is not None:
        raise ValueError("Hy4-flash recipe requires TP=1 on rank 0")
    snapshot = Path(snapshot).resolve()
    shard = TensorParallelShard(0, 1)
    device = torch.device("cuda", config.devices[0])
    torch.cuda.set_device(device)

    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
        metadata_path = manifest.metadata_path
        model_config = ModelConfig.load(metadata_path, manifest=manifest)
        template = config.chat_template or chat_template_asset(metadata_path)
        frontend = prepare_frontend(
            metadata_path,
            eos_token_ids=model_config.text["eos_token_id"],
            image_profile=None if config.language_model_only else "row_newline",
            preprocessor=metadata_path / "processor_config.json",
        )

    with (core_storage.transient() if core_storage is not None else contextlib.nullcontext()):
        weights = ModelWeights.load(
            manifest,
            device,
            shard=shard,
            progress=progress,
            dfly=config.dfly,
            vision=not config.language_model_only,
            expert_scale_compression=config.expert_scale_compression,
            draft_head_quantization=config.draft_head_quantization,
            online_mxfp8=config.online_mxfp8,
            online_mxfp8_scope=config.online_mxfp8_scope,
        )
        release_unused()

    transient = contextlib.ExitStack()
    if core_storage is not None:
        transient.enter_context(core_storage.transient())
    if progress is not None:
        progress(f"Resident model tensors: {weights.allocated_bytes / (1 << 30):.3f} GiB")

    sequence = torch.zeros(1, dtype=torch.int64, device=device)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    max_tokens = max(config.prefill_chunk_size, config.decode_tokens)
    host_alloc = (
        core_storage.host_allocate
        if core_storage is not None
        else _pinned_host_allocator(device)
    )
    tables = Tables.prepare(
        manifest,
        weights.config,
        tokens=max_tokens,
        device=device,
        sequence=sequence,
        frame=frame,
        host_allocator=host_alloc,
        progress=progress,
    )

    with phase(progress, "Allocate model state"):
        dfly_active = config.dfly and weights.draft is not None
        layout = continuation_layout(weights.config, dfly=dfly_active)
        snapshots = (
            1
            if core_storage is not None
            else max(1, layout.snapshot_capacity(config.snapshot_budget_bytes))
        )
        slots = config.request_capacity
        blocks = config.context // PAGE_TOKENS
        state_rows = snapshots + slots
        bytes_per_page = page_bytes(weights.config)
        prep_pages = (
            config.kv_pages
            if config.kv_pages is not None
            else (config.prefill_chunk_size + PAGE_TOKENS - 1) // PAGE_TOKENS + 2 * slots
        )
        total_memory = torch.cuda.mem_get_info(device)[1]
        max_cache_pages = max(prep_pages, total_memory // bytes_per_page)
        pool = ContinuationPool.allocate(
            weights.config,
            state_rows + 1,
            prep_pages,
            device,
            storage=core_storage,
            maximum_pages=max_cache_pages,
            maximum_rows=max(state_rows + 1, total_memory // layout.row_bytes),
            backed_slots=min(state_rows + 1, 1 + 2 * slots) if core_storage is not None else None,
            dfly=dfly_active,
        )
        requests = DeviceRequestState.allocate(device, vocab_size=VOCABULARY, capacity=slots)
        inputs = RotaryBatchInputs.allocate(config.prefill_chunk_size, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=VOCABULARY, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        completion = CompletionOutputs.allocate(
            (2, slots, VECTOR_TOKENS_FIELD + config.completion_token_capacity),
            device=device,
        )
        identity_payload = {
            "config": json.loads((metadata_path / "config.json").read_text()),
            "headers": [s.header_sha256 for s in manifest.shards],
            "world_size": 1,
            "contract": "hy4_flash.graph-inputs.v1",
            "vision": weights.vision is not None,
            "draft_head_quantization": config.draft_head_quantization,
        }
        if config.online_mxfp8:
            identity_payload["online_mxfp8"] = True
            if config.online_mxfp8_scope != "all":
                identity_payload["online_mxfp8_scope"] = config.online_mxfp8_scope
        identity = hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True).encode()
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
            workload_schema="hy4-flash-tp1-v1",
            numerical_policy="default",
            coordinator=None,
            headroom_bytes=config.headroom_bytes,
            benchmark_storage=core_storage,
        )
        session.__enter__()
        session.register_benchmark_resource("continuations", pool, restore=pool.clear_for_preparation)
        arena = ScratchArena(device, overlap=config.stream_overlap)
        session.register_scratch(arena)
        examples: dict = {}

        max_verify_w = max((w for _, w in config.speculative_shapes), default=1) if dfly_active else 1
        prefill_storage = Scratch.allocate(
            config.prefill_chunk_size, 1, 1, config.prefill_chunk_size, device, shard=shard
        )
        decode_storage = Scratch.allocate(
            config.decode_tokens,
            slots,
            config.decode_tokens,
            max_verify_w,
            device,
            shard=shard,
            ring_layers=len(weights.config.swa_layers),
        )

        vision_state = vision_inputs = vision_prefill = vision_storage = feature_pool = None
        if weights.vision is not None:
            vision_inputs = VisionInputs.allocate(device, patches=max(PATCH_CAPACITIES))
            vision_prefill = PrefillVisionInputs.allocate(config.prefill_chunk_size, device)
            image_newline = weights.vision["vit.perceive.image_newline"]
            if core_storage is not None:
                feature_pool = core_storage.scratch_pool()
                with feature_pool:
                    feature_arena = torch.zeros(
                        (config.image_concurrency, FEATURES, weights.embedding.shape[1]),
                        dtype=torch.bfloat16,
                        device=device,
                    )
            else:
                feature_arena = None
            vision_state = VisionState(
                image_newline,
                inputs,
                vision_prefill,
                requests,
                features=FEATURES,
                image_slots=config.image_concurrency,
                arena=feature_arena,
            )
            vision_storage = VisionScratch.allocate(max(PATCH_CAPACITIES), device)

        buffers = dict(
            inputs.buffers(),
            **tables.buffers(),
            **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing,
            memory_capacity=pool.memory_capacity,
            request_pending_token=requests.pending_token,
            sampling_grammar_mask=requests.grammar_mask,
            frame_index=frame,
            submission_sequence=sequence,
            **completion.buffers(),
        )
        if vision_state is not None:
            buffers.update(vision_inputs.buffers(), **vision_prefill.buffers())

        speculation = None
        if dfly_active:
            speculation = SimpleNamespace(
                weights=weights,
                tables=tables,
                pool=pool,
                requests=requests,
                inputs=inputs,
                scratch=decode_storage,
                decoder_factory=PreparedDecoder,
                completion=completion,
                frame=frame,
                sequence=sequence,
                session=session,
                workspace=arena,
                examples=examples,
                runtime=None,
                vocabulary_runtime=None,
                coordinator=None,
                page_tokens=PAGE_TOKENS,
                max_cache_pages=max_cache_pages,
                rng=SpeculativeRngState.allocate(device, capacity=slots),
                offset=torch.zeros(slots, dtype=torch.int64, device=device),
                generation=torch.zeros(slots, dtype=torch.int64, device=device),
            )

    scoring = PromptScoring.allocate(
        config,
        device,
        buffers,
        frame=frame,
        group=None,
        ipc=None if core_storage is None else core_storage.peer_storage,
        runtime=None,
        coordinator=None,
    )
    del manifest

    forwards, signatures, metadata_map = {}, {}, {}
    storage = decoder = batch = encoder = None
    modes = serving_modes(dfly_active, config.mtp_only)
    graph_count = (
        len(modes) * (prefill_graph_count(config.prefill_capacities, scoring) + 4 * len(config.batch_capacities))
        + (len(config.batch_capacities) if False in modes else 0)
        + (len(config.speculative_shapes) if dfly_active else 0)
        + (len(PATCH_CAPACITIES) if vision_state is not None else 0)
    )
    with phase(progress, "Construct graph catalog", total=graph_count) as stage:
        def add(name, forward, kind, rows, tokens, fields, *, enabled=None):
            if kind == "scored_prefill":
                forward = PreparedScoredPrefill(forward, requests, inputs)
                fields = (*fields, "prompt_target_tokens")
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields), speculation_enabled=enabled)
            stage.update(advance=1)

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        for enabled in modes:
            suffix = "-no-mtp" if dfly_active and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for kind, mode, capacities in prefill_modes(
                config.prefill_capacities, () if enabled else config.batch_capacities, scoring
            ):
                for capacity in capacities:
                    rows = 1 if mode == "prefill" else capacity
                    max_q = capacity if mode == "prefill" else 1
                    name = f"{kind}{capacity}" + (suffix if mode == "prefill" else "")
                    m = metadata_map[name] = Metadata.allocate(capacity, rows, blocks, device, max_query_tokens=max_q)
                    storage = prefill_storage if mode == "prefill" else decode_storage
                    decoder = PreparedDecoder(
                        mode_weights,
                        tables,
                        storage.rows(capacity, rows, rows, max_q),
                        m,
                        pool,
                        mode=mode,
                        scoring=scoring if kind == "scored_prefill" else None,
                        session=session,
                        name=name,
                        workspace=arena,
                        vision=vision_state,
                        max_cache_pages=max_cache_pages,
                        examples=examples,
                    )
                    batch = prepare_batch(
                        inputs,
                        requests,
                        m,
                        mode=mode,
                        shard=shard,
                        pool=pool,
                        page_tokens=PAGE_TOKENS,
                        validate_prefill=vision_state.validate_prefill
                        if vision_state is not None and mode == "prefill"
                        else None,
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
                        name=f"{name}.generation",
                        workspace=arena,
                        context_factory=_NoOpDraftContext if enabled else None,
                    )
                    fields = forward_inputs
                    if mode == "prefill" and vision_state is not None:
                        fields = (*fields, "vision_token_indices", "vision_feature_slot")
                    if mode == "decode":
                        fields = tuple(f for f in fields if f not in ("token_ids", "prefill_positions", "query_lengths"))
                    add(name, forward, kind, rows, max_q, fields, enabled=enabled)

        if speculation is not None:
            spec_inputs = tuple(f for f in forward_inputs if f not in ("token_ids", "prefill_positions"))
            for batch, width in sorted(config.speculative_shapes, key=lambda s: s[0] * s[1], reverse=True):
                name = f"speculative{batch}x{width}"
                operation = PreparedDFlySpeculativeRound(speculation, batch, width, name)
                operation.prepare()
                add(name, operation, "speculative", batch, width, spec_inputs, enabled=True)

        if vision_state is not None:
            for capacity in reversed(PATCH_CAPACITIES):
                name = f"vision{capacity}"
                encoder = PreparedVision(
                    weights.vision,
                    vision_storage.rows(capacity),
                    vision_inputs,
                    config=weights.config,
                    session=session,
                    workspace=arena,
                    name=name,
                )
                forward = PreparedVisionGeneration(encoder, vision_state, vision_inputs, completion, frame, sequence)
                add(name, forward, "vision", 1, capacity, (*vision_inputs.buffers(), "vision_feature_slot"))

        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in modes:
            suffix = "-no-mtp" if dfly_active and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for capacity in config.batch_capacities:
                name = f"resume{capacity}" + suffix
                resume = PreparedResume(
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
                    logits_dtype=torch.float32,
                    workspace=arena,
                )
                add(name, resume, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"), enabled=enabled)
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    operation = PreparedMaintenance(
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
                    )
                    fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                    add(name, operation, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()), enabled=enabled)

    transient.close()
    metadata_map.clear()
    release_unused()

    def materialize():
        if feature_pool is not None:
            feature_pool.materialize()
        pool.clear_for_preparation()

    if feature_pool is not None:
        feature_pool.release_backing()
    try:
        with phase(progress, "Prepare and tune kernels") as stage, (
            core_storage.transient() if core_storage is not None else contextlib.nullcontext()
        ):
            tuning = session.tune(show_progress=progress is not None, console=stage.console)
        stream = torch.cuda.Stream(device=device)
        graph_pool = torch.cuda.graph_pool_handle() if core_storage is None else core_storage.graph_pool.id
        if speculation is not None:
            speculation.scratch = None
            speculation.examples = None
        examples.clear()
        prefill_storage = decode_storage = vision_storage = storage = decoder = batch = encoder = None
        arena.release_preparation()
        release_unused()
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
            materialize=materialize,
            warmup_storage=core_storage,
            compile_kinds=_COMPILED_KINDS,
        )
    finally:
        if feature_pool is not None:
            feature_pool.materialize()

    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    release_unused()

    free, total = torch.cuda.mem_get_info(device)
    assets = {
        "tokenizer": {"path": str(metadata_path / "tokenizer.json")},
        "chat_template": template,
    }
    server_config = {
        "identity": identity,
        "model_id": snapshot.name,
        "max_sequence_tokens": config.context,
        "prefill_chunk_size": config.prefill_chunk_size,
        "pages": prep_pages,
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
            page_tokens=PAGE_TOKENS,
            draft_lag=0 if dfly_active else None,
            rows=1,
            embedding_tables=range(len(tables.sources)),
            image_slots=config.image_concurrency or 0,
        ),
    }
    if dfly_active:
        server_config.update(
            speculative_widths=config.speculative_widths,
            speculative_fixed_width=config.speculative_fixed_width,
            speculative_token_capacity=speculative_token_capacity(
                config.max_active,
                max(config.speculative_widths),
                config.speculative_token_capacity,
            ),
        )

    memory = {
        "rank": 0,
        "world_size": 1,
        "device": config.devices[0],
        "weights_bytes": weights.allocated_bytes,
        "expert_scale_compression": config.expert_scale_compression,
        "mapped_host_bytes": 0,
        "offloaded_bytes": sum(src.nbytes for pair in tables.sources.values() for src in pair),
        "continuation_bytes": pool.backing.nbytes,
        "state_rows": state_rows,
        "state_row_bytes": layout.row_bytes,
        "pages": prep_pages,
        "page_bytes": bytes_per_page,
        "page_tokens": PAGE_TOKENS,
        "kv_bytes": prep_pages * bytes_per_page,
        "completion_output_bytes": completion.nbytes,
        "operator_scratch_bytes": arena.nbytes,
        "graphs": len(graphs),
        "prefill_capacities": config.prefill_capacities,
        "free_device_bytes": free,
        "total_device_bytes": total,
        "headroom_bytes": config.headroom_bytes,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "tuning": tuning,
        "compiled_decoder": config.compile_decoder,
    }
    if scoring is not None:
        memory.update(scoring.memory_report())

    disposable = _collect_disposable_storage(
        arena_tensor=arena.tensor,
        scale_scratch=weights.scale_scratch,
        engram_buffers=tables.buffers(),
        vision_arena=None if vision_state is None else vision_state.arena,
        forwards=forwards,
    )

    return PreparedModel(
        rebuild_weights=declare_weights(
            weights, components=("embedding", "norm", "hc_head", "head", "draft", "vision")
        ),
        device=config.devices[0],
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
        export_events=tables.events(),
        embedding_catalog=tables.resident_catalog(),
        core_data=tables,
        resources=(
            scoring,
            weights,
            tables,
            pool,
            requests,
            inputs,
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
            feature_pool,
        ),
        close_resources=((scoring,) if scoring is not None else ()),
    )


def _pinned_host_allocator(device: torch.device):
    """Fallback mapped-pinned host allocator for in-process preparation without core_storage."""

    def allocate(name: str, shape: tuple[int, ...], dtype: torch.dtype):
        del name
        host = torch.zeros(shape, dtype=dtype, pin_memory=True)
        return SimpleNamespace(host_view=host, device_view=host)

    return allocate
