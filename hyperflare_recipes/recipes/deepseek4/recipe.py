# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek V4 Flash Vision checkpoint with integrated DSpark proposals."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from ignition.declaration import (
    Checkpoint,
    RankCosts,
    RankUsage,
    Recipe,
    SamplingDefaults,
)
from ignition.parameters import (
    PromptLogprobsParameters,
    RecipeParameters,
    SpeculationParameters,
    VisionParameters,
)
from ignition.preparation.bootstrap import common_preparation_kwargs

from hyperflare_recipes.parameters import (
    DraftQuantizationParameters,
    NativeDenseActivationsParameters,
    StreamOverlapParameters,
)


@dataclass(frozen=True)
class DeepSeek4Parameters(RecipeParameters):
    """Preparation parameters for DeepSeek V4 Flash Vision."""

    native_dense_activations: NativeDenseActivationsParameters = NativeDenseActivationsParameters()
    stream_overlap: StreamOverlapParameters = StreamOverlapParameters()
    draft_quantization: DraftQuantizationParameters = DraftQuantizationParameters(
        draft_head_choices=("nvfp4",),
        markov_head_choices=("nvfp4",),
    )


def prepare(
    snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False, core_storage=None,
    ranks=None,
):
    """Prepare the plan's catalog on process-local GPUs 0 through N-1.

    With `ranks`, rank zero's multi-host rendezvous, each rank uses its own host's GPU 0.
    """
    from ignition.preparation.rank_group import prepare_group
    from .bootstrap import PreparationConfig, prepare_rank

    config = PreparationConfig(
        **common_preparation_kwargs(
            plan, compile_forwards=compile_forwards, kernel_cache=kernel_cache, retain_graphs=retain_graphs,
        ),
        devices=tuple(range(plan.world_size)) if ranks is None else (0,) * plan.world_size,
        hosts=None if ranks is None else ranks.hosts,
    )
    return prepare_group(snapshot, config, prepare_rank, progress=progress, core_storage=core_storage,
                         ranks=ranks, checkpoint=plan.checkpoint)


def usage(prepared):
    return tuple(
        RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in prepared.memory_report["ranks"]
    )


RECIPE = Recipe(
    name="deepseek4",
    served_model_name="DeepSeek-V4-Flash-Vision-Exp",
    # Agent evaluation sampling: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp#introduction.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1),
    prepare=prepare,
    usage=usage,
    checkpoints=(
        Checkpoint(
            alias="deepseek-v4-flash",
            repository="local-inference-lab/DeepSeek-V4-Flash-Vision-Exp-lossless-CSF",
            revision="45fd997626821b131a904b1ae2304bb673c9458d",
            architecture="DeepseekV4ForCausalLM",
            stored_bytes=160_372_641_076,
            device_bytes=160_360_333_756,
            draft_bytes=10_862_841_372,
        ),
    ),
    context=1048576,
    page_tokens=256,
    parameters=DeepSeek4Parameters(
        max_concurrency=32,
        max_concurrency_limit=64,
        prefill_chunk=4096,
        speculation=SpeculationParameters(
            draft_tokens=5, max_draft_tokens=5, token_capacity=128, packed_verification=True, graph_budget=64,
        ),
        vision=VisionParameters(),
        prompt_logprobs=PromptLogprobsParameters(),
    ),
    max_images=4,
    image_bytes=32 << 20,
    # Continuation rows and latent pages are replicated, so their costs do not depend on the rank count.
    ranks=MappingProxyType({
        2: RankCosts(16_668_416, 998_208, 11 << 30),
        4: RankCosts(16_668_416, 998_208, 24 << 30),
    }),
    plain_ranks=MappingProxyType({
        2: RankCosts(16_443_392, 998_208, 12 << 30),
        4: RankCosts(16_443_392, 998_208, 24 << 30),
    }),
)
