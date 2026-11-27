# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 MXFP4-CSF preparation capacities, parameters, and serving defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from ignition.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, SamplingRange
from ignition.errors import UsageError
from ignition.parameters import (
    ParameterContext,
    ParameterScope,
    PromptLogprobsParameters,
    RecipeParameters,
    SpeculationParameters,
    VisionParameters,
    param,
)
from ignition.preparation.bootstrap import common_preparation_kwargs

from hyperflare_recipes.parameters import (
    ContextParallelParameters,
    ExpertScaleCompressionParameters,
    NativeDenseActivationsParameters,
    OnlineMXFP8Parameters,
    StreamOverlapParameters,
)


@dataclass(frozen=True)
class Kimi3Parameters(RecipeParameters):
    """Preparation parameters for Kimi K3."""

    native_dense_activations: NativeDenseActivationsParameters = NativeDenseActivationsParameters()
    stream_overlap: StreamOverlapParameters = StreamOverlapParameters()
    context_parallel: ContextParallelParameters = ContextParallelParameters()
    expert_scale_compression: ExpertScaleCompressionParameters = ExpertScaleCompressionParameters()
    online_mxfp8: OnlineMXFP8Parameters = OnlineMXFP8Parameters(
        default_scope="all",
        supported_scopes=("all", "kda-input"),
    )
    row_shard_prefill: bool = param(
        False,
        flag="--row-shard-prefill",
        group="advanced options",
        scope=ParameterScope.DEPLOYMENT,
        default_label="full-row all-reduce",
        help="reduce eligible prefill branch outputs into one contiguous row shard per rank,"
             " run row-local residual mixing on that shard, and gather normalized branch inputs back to full rows",
        status_label=lambda v: "row-sharded prefill" if v else None,
    )
    # Recipe capability constraint (never a CLI flag or manifest entry).
    row_shard_prefill_world_sizes: tuple[int, ...] = (16,)

    def __post_init__(self):
        if type(self.row_shard_prefill) is not bool:
            raise TypeError("row_shard_prefill must be a boolean")

    def filter_world_sizes(self, world_sizes: tuple[int, ...], context: ParameterContext) -> tuple[int, ...]:
        sizes = super().filter_world_sizes(world_sizes, context)
        if self.row_shard_prefill:
            sizes = tuple(size for size in sizes if size in self.row_shard_prefill_world_sizes)
        return sizes

    def validate(self, context: ParameterContext) -> None:
        super().validate(context)
        if self.row_shard_prefill:
            if context.world_size is not None:
                if context.world_size < 2:
                    raise UsageError("--row-shard-prefill requires tensor parallelism")
                if context.world_size not in self.row_shard_prefill_world_sizes:
                    supported = ", ".join(map(str, self.row_shard_prefill_world_sizes))
                    raise UsageError(
                        f"--row-shard-prefill is not supported at TP{context.world_size} for {context.alias};"
                        f" supported tensor-parallel sizes: {supported}"
                    )
            if any(gpu.capability == (10, 3) for gpu in context.gpus):
                raise UsageError("--row-shard-prefill is not qualified on SM103; select SM120 or SM121 GPUs")


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None, recipe=None, draft_source=None, draft_directory=None):
    from ignition.preparation.rank_group import prepare_group
    from .bootstrap import PreparationConfig, prepare_rank
    from .config import DRAFT_DIRECTORY, DSPARK_REPOSITORY, DSPARK_REVISION

    recipe = RECIPE if recipe is None else recipe
    if recipe.speculation is None and plan.speculation is not None:
        raise ValueError(f"{recipe.served_model_name} preparation does not support speculation")
    if ranks is not None:
        raise ValueError("Kimi K3 preparation requires GPUs on one host")
    draft_checkpoint = None
    if plan.speculation is not None:
        directory = DRAFT_DIRECTORY if draft_directory is None else draft_directory
        if (Path(snapshot) / directory).is_dir():
            draft_checkpoint = str(Path(snapshot) / directory)
        else:
            from huggingface_hub import snapshot_download

            repository, revision = (DSPARK_REPOSITORY, DSPARK_REVISION) if draft_source is None else draft_source
            draft_checkpoint = snapshot_download(
                repository, revision=revision, cache_dir=plan.hub_cache,
                allow_patterns=("config.json", "model.safetensors"),
            )
    config = PreparationConfig(
        **common_preparation_kwargs(
            plan, compile_forwards=compile_forwards, kernel_cache=kernel_cache, retain_graphs=retain_graphs,
        ),
        devices=tuple(range(plan.world_size)),
        benchmark_selection=plan.benchmark_selection,
        draft_checkpoint=draft_checkpoint,
        sampling_defaults=recipe.sampling_defaults.as_dict(),
        frontend_template=str(Path(__file__).parent.parent / recipe.name / "assets" / "chat_template.jinja"),
    )
    if plan.world_size == 1:
        return prepare_rank(snapshot, config, rank=0, group=None, progress=progress, core_storage=core_storage)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, core_storage=core_storage,
                         checkpoint=plan.checkpoint)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="kimi3",
    served_model_name="Kimi-K3",
    prepare=prepare,
    usage=usage,
    # Agentic defaults; top-p spans single-step/agentic presets: https://huggingface.co/moonshotai/Kimi-K3#3-evaluation-results.
    sampling_defaults=SamplingDefaults(
        temperature=1., top_p=1., top_k=-1,
        ranges={"top_p": SamplingRange(.95, 1.)},
    ),
    checkpoints=(
        Checkpoint(
            alias="kimi-k3-mxfp4-csf", repository="/data/models/Kimi-K3-MXFP4-CSF",
            revision="local", architecture="KimiK3ForConditionalGeneration",
            stored_bytes=1_491_026_420_236, device_bytes=1_498_926_420_236, draft_bytes=7_900_000_000,
            scale_compressed_device_bytes={"csf": 1_498_926_420_236},
        ),
        # The native checkpoint's block scales are compressed while loading, as resident as the CSF copy's.
        Checkpoint(
            alias="kimi-k3", repository="moonshotai/Kimi-K3",
            revision="2496450e92e425c886db095102a52a6682ca3970", architecture="KimiK3ForConditionalGeneration",
            stored_bytes=1_560_860_324_864, device_bytes=1_498_926_420_236, draft_bytes=7_900_000_000,
            scale_compressed_device_bytes={"csf": 1_498_926_420_236},
        ),
        # A local target checkpoint carrying Red Hat's DSpark weights in dspark/.
        Checkpoint(
            alias="kimi-k3-dspark", repository="/data/kimi-k3-dspark", revision="local",
            architecture="KimiK3ForConditionalGeneration", stored_bytes=1_500_516_228_062,
            device_bytes=1_498_926_420_236, draft_bytes=7_900_000_000,
            scale_compressed_device_bytes={"csf": 1_498_926_420_236},
        ),
    ),
    context=1048576,
    page_tokens=128,
    parameters=Kimi3Parameters(
        max_concurrency=32,
        max_concurrency_limit=64,
        prefill_chunk=4096,
        speculation=SpeculationParameters(
            draft_tokens=7, max_draft_tokens=7, token_capacity=128, packed_verification=True, graph_budget=64,
        ),
        vision=VisionParameters(separable_encoder=True),
        prompt_logprobs=PromptLogprobsParameters(),
    ),
    max_images=4,
    image_bytes=32 << 20,
    verification_rows=0,
    # vLLM's Kimi K3 launch settings for b12x's PCIe all-reduce (island reduce-scatter, deferred slab
    # consumption, 256 threads), its decode context exchange and its small-M W4A16 experts.
    b12x_environment=MappingProxyType({
        "B12X_PCIE_ALLREDUCE_ALGORITHM": "island_rs",
        "B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION": "1",
        "B12X_PCIE_HIERARCHICAL_THREADS": "256",
        "B12X_PCIE_DCP_THREADS": "512",
        "B12X_PCIE_DCP_BLOCK_LIMIT": "8",
        "B12X_W4A16_SMALL_M_HOST_BARRIER_RESET": "0",
    }),
    # Target MLA pages are context-parallel; DSpark keeps its sliding KV windows in continuation rows.
    ranks=MappingProxyType({16: RankCosts(state_row_bytes=34_716_928, page_bytes=1_769_472,
                                          graph_bytes=9 << 29, page_tokens=128 * 16)}),
    plain_ranks=MappingProxyType({16: RankCosts(state_row_bytes=29_869_824, page_bytes=1_769_472,
                                                graph_bytes=9 << 29, page_tokens=128 * 16)}),
)
