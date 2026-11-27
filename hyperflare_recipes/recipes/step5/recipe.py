# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Step-5 Preview NVFP4 text, causal MTP, and perception-encoder preparation."""

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
    ExpertScaleCompressionParameters,
    NativeDenseActivationsParameters,
    StreamOverlapParameters,
)


@dataclass(frozen=True)
class Step5Parameters(RecipeParameters):
    """Preparation parameters for Step-5 Preview."""

    native_dense_activations: NativeDenseActivationsParameters = NativeDenseActivationsParameters()
    stream_overlap: StreamOverlapParameters = StreamOverlapParameters()
    expert_scale_compression: ExpertScaleCompressionParameters = ExpertScaleCompressionParameters()


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None):
    from ignition.preparation.rank_group import prepare_group
    from .bootstrap import PreparationConfig, prepare_rank

    if ranks is not None:
        raise ValueError("Step-5 preparation requires GPUs on one host")
    config = PreparationConfig(
        **common_preparation_kwargs(
            plan, compile_forwards=compile_forwards, kernel_cache=kernel_cache, retain_graphs=retain_graphs,
        ),
        devices=tuple(range(plan.world_size)),
    )
    return prepare_group(snapshot, config, prepare_rank, progress=progress, core_storage=core_storage,
                         checkpoint=plan.checkpoint)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="step5",
    served_model_name="Step-5-Preview",
    prepare=prepare,
    usage=usage,
    # Recommended sampling: https://huggingface.co/stepfun-ai/Step-5-Preview-NVFP4#evaluation-notes.
    # The card's top_k=0 and Flare's top_k=-1 both disable the filter.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1),
    checkpoints=(
        Checkpoint(
            alias="step5-preview-nvfp4", repository="stepfun-ai/Step-5-Preview-NVFP4",
            revision="bc246cc2848e17b37f377eda8fd825c5b3018d1f", architecture="Step5ForConditionalGeneration",
            # W4A16 experts and MXFP8 gate/up include maximum scale padding across TP4/8/16.
            stored_bytes=394_164_831_232, device_bytes=367_482_051_584, draft_bytes=4_713_730_560,
            # Planning assumes 6% resident-weight savings; loading measures the encoded allocation.
            scale_compressed_device_bytes={"csf": 345_326_999_491},
            # Replicated MTP projections add up to fifteen copies of three BF16 4096x8192 matrices.
            expansion_bytes=(2 << 30) + 3 * (4096 * 8192 * 2) * 15,
        ),
    ),
    context=1048576,
    page_tokens=128,
    parameters=Step5Parameters(
        max_concurrency=32,
        max_concurrency_limit=64,
        prefill_chunk=4096,
        speculation=SpeculationParameters(draft_tokens=3, max_draft_tokens=3, token_capacity=128),
        vision=VisionParameters(),
        prompt_logprobs=PromptLogprobsParameters(),
    ),
    max_images=4,
    image_bytes=32 << 20,
    verification_rows=0,
    ranks=MappingProxyType({world: RankCosts(15_667_968, 1_224_704, 8 << 30) for world in (4, 8, 16)}),
    plain_ranks=MappingProxyType({world: RankCosts(14_987_520, 1_224_704, 8 << 30) for world in (4, 8, 16)}),
)
