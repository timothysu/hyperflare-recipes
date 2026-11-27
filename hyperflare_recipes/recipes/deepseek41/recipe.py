# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek V4.1 Flash checkpoint, parameters, and tensor-parallel memory geometry."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

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
    DraftQuantizationParameters,
    ExpertScaleCompressionParameters,
    NativeDenseActivationsParameters,
    StreamOverlapParameters,
)
from .config import replicated_attention_bytes


@dataclass(frozen=True)
class DeepSeek41Parameters(RecipeParameters):
    """Preparation parameters for DeepSeek V4.1 Flash."""

    native_dense_activations: NativeDenseActivationsParameters = NativeDenseActivationsParameters()
    stream_overlap: StreamOverlapParameters = StreamOverlapParameters()
    expert_scale_compression: ExpertScaleCompressionParameters = ExpertScaleCompressionParameters(encoding="csf")
    draft_quantization: DraftQuantizationParameters = DraftQuantizationParameters(
        draft_head_choices=("nvfp4", "mxfp8"),
        markov_head_choices=("nvfp4", "fp8"),
    )
    mla_compute_mode: str = param(
        "bf16",
        flag="--mla-compute-mode",
        group="advanced options",
        choices=("auto", "bf16"),
        scope=ParameterScope.COMBINE,
        repro_always=True,
        help="constrain MLA arithmetic while retaining packed KV storage and compatible kernel tuning",
    )
    vocabulary_parallel_sampling: bool = param(
        False,
        flag="--vocabulary-parallel-sampling",
        group="advanced options",
        scope=ParameterScope.COMBINE,
        manifest_key="speculation.vocabulary_parallel_sampling",
        requires_speculation=True,
        default_label="full-vocabulary sampling",
        help="prepare vocabulary-sharded DSpark proposals and block rejection sampling",
        status_label=lambda v: "vocabulary-parallel block sampling" if v else None,
        in_dict=False,
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
    row_shard_attention: bool = param(
        False,
        flag="--row-shard-attention",
        group="advanced options",
        scope=ParameterScope.DEPLOYMENT,
        default_label="off",
        help="also run eligible prefill attention layers on each rank's row shard using"
             " replicated attention projections and KV gathers; requires --row-shard-prefill",
        status_label=lambda v: "row-sharded attention" if v else None,
    )
    # Recipe capability constraints (never CLI flags or manifest entries).
    vocabulary_parallel_world_sizes: tuple[int, ...] = (4, 6)
    row_shard_prefill_world_sizes: tuple[int, ...] = (4,)

    def __post_init__(self):
        if type(self.row_shard_prefill) is not bool:
            raise TypeError("row_shard_prefill must be a boolean")
        if type(self.row_shard_attention) is not bool:
            raise TypeError("row_shard_attention must be a boolean")
        if self.vocabulary_parallel_sampling and (
            self.speculation is None or self.speculation.draft_tokens < 1
        ):
            raise ValueError("vocabulary-parallel sampling requires positive learned draft depth")
        if self.vocabulary_parallel_sampling and self.draft_quantization.markov_head_quantization == "nvfp4":
            raise ValueError("vocabulary-parallel sampling requires BF16 or row-scaled FP8 Markov weights")

    def filter_world_sizes(self, world_sizes: tuple[int, ...], context: ParameterContext) -> tuple[int, ...]:
        sizes = super().filter_world_sizes(world_sizes, context)
        if self.vocabulary_parallel_sampling:
            sizes = tuple(size for size in sizes if size in self.vocabulary_parallel_world_sizes)
        if self.row_shard_prefill:
            sizes = tuple(size for size in sizes if size in self.row_shard_prefill_world_sizes)
        return sizes

    def validate(self, context: ParameterContext) -> None:
        super().validate(context)
        if self.vocabulary_parallel_sampling:
            if self.speculation is None or self.speculation.draft_tokens < 1:
                raise UsageError("vocabulary-parallel sampling requires positive learned draft depth")
            if self.draft_quantization.markov_head_quantization == "nvfp4":
                raise UsageError("vocabulary-parallel sampling requires BF16 or row-scaled FP8 Markov weights")
            if context.recipe is not None:
                sampling = context.recipe.sampling_defaults
                if (
                    sampling.top_p != 1 or sampling.top_k > 0 or sampling.min_p != 0
                    or {"top_p", "top_k", "min_p"} & sampling.ranges.keys()
                ):
                    raise UsageError(
                        "--vocabulary-parallel-sampling requires a sampling policy without"
                        " top-k, top-p, or min-p filtering"
                    )
            if context.world_size is not None and context.world_size not in self.vocabulary_parallel_world_sizes:
                allowed = " or ".join(map(str, self.vocabulary_parallel_world_sizes))
                raise UsageError(
                    f"--vocabulary-parallel-sampling has no TP{context.world_size} gather; select {allowed} GPUs"
                )
        if self.row_shard_attention and not self.row_shard_prefill:
            raise UsageError("--row-shard-attention requires --row-shard-prefill")
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

    def adjust_weights_bytes(self, weights_bytes: int, context: ParameterContext) -> int:
        weights_bytes = super().adjust_weights_bytes(weights_bytes, context)
        if self.row_shard_attention and context.world_size is not None:
            replicated = replicated_attention_bytes(
                context.world_size,
                prompt_logprobs=bool(self.prompt_logprobs and self.prompt_logprobs.enabled),
                mtp_only=bool(self.speculation and self.speculation.mtp_only),
            )
            if type(replicated) is not int or replicated < 0:
                raise ValueError("replicated_attention_bytes must return a non-negative integer")
            return weights_bytes + replicated
        return weights_bytes

    def manifest_parameters(self, context: ParameterContext) -> list[tuple[str, ParameterScope, Any]]:
        entries = super().manifest_parameters(context)
        if self.speculation is None:
            entries = [item for item in entries if item[0] != "speculation.vocabulary_parallel_sampling"]
        return entries

    def to_dict(self, context: ParameterContext | None = None) -> dict[str, Any]:
        result = super().to_dict(context)
        if self.speculation is not None and isinstance(result.get("speculation"), dict):
            result["speculation"] = {
                **result["speculation"],
                "vocabulary_parallel_sampling": self.vocabulary_parallel_sampling,
            }
        return result


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None):
    from ignition.preparation.rank_group import prepare_group
    from .bootstrap import PreparationConfig, prepare_rank

    config = PreparationConfig(
        **common_preparation_kwargs(
            plan, compile_forwards=compile_forwards, kernel_cache=kernel_cache, retain_graphs=retain_graphs,
        ),
        devices=tuple(range(plan.world_size)) if ranks is None else (0,) * plan.world_size,
        hosts=None if ranks is None else ranks.hosts,
        minimum_pages=plan.minimum_pages,
    )
    return prepare_group(snapshot, config, prepare_rank, progress=progress, core_storage=core_storage,
                         ranks=ranks, checkpoint=plan.checkpoint)


def usage(prepared):
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in prepared.memory_report["ranks"])


RECIPE = Recipe(
    name="deepseek41",
    served_model_name="DeepSeek-V4.1-Flash",
    prepare=prepare,
    usage=usage,
    # Recommended sampling, including top-p 1.0: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017#minimal-inference.
    # Repetition-penalty bounds enable measured use over prompt + output, matching vLLM/HF semantics.
    sampling_defaults=SamplingDefaults(
        temperature=1., top_p=1., top_k=-1,
        ranges={"repetition_penalty": SamplingRange(1., 1.1)},
    ),
    checkpoints=(
        Checkpoint(
            alias="deepseek-v4.1-flash",
            repository="deepseek-ai/DeepSeek-V4.1-Flash-1017",
            revision="1bae0e1088993b9b82de08f59de5be5a34e42170",
            architecture="DeepseekV41ForCausalLM",
            stored_bytes=510_286_023_000,
            device_bytes=306_557_453_640,
            draft_bytes=7_932_874_632,
            # Estimate half of backbone scale bytes saved; retain the full draft budget.
            scale_compressed_device_bytes={"csf": 298_063_988_040},
        ),
    ),
    context=1048576,
    page_tokens=128,
    parameters=DeepSeek41Parameters(
        max_concurrency=32,
        max_concurrency_limit=64,
        prefill_chunk=4096,
        speculation=SpeculationParameters(
            draft_tokens=7,
            max_draft_tokens=7,
            token_capacity=128,
            packed_verification=True,
            draft_prefix_selection="gpu",
            graph_budget=64,
        ),
        vision=VisionParameters(),
        prompt_logprobs=PromptLogprobsParameters(),
    ),
    max_images=4,
    image_bytes=32 << 20,
    ranks=MappingProxyType({world: RankCosts(3_043_840, 113_920, 8 << 30) for world in (3, 4, 6)}),
    plain_ranks=MappingProxyType({world: RankCosts(2_841_088, 113_920, 8 << 30) for world in (3, 4, 6)}),
)
