# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax M3 mixed MXFP8/NVFP4 preparation on four SM120/SM121 GPUs."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from ignition.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults
from ignition.parameters import (
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
class MiniMaxM3Parameters(RecipeParameters):
    """Preparation parameters for MiniMax M3."""

    native_dense_activations: NativeDenseActivationsParameters = NativeDenseActivationsParameters()
    stream_overlap: StreamOverlapParameters = StreamOverlapParameters()
    expert_scale_compression: ExpertScaleCompressionParameters = ExpertScaleCompressionParameters()


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
    )
    return prepare_group(snapshot, config, prepare_rank, progress=progress, core_storage=core_storage,
                         ranks=ranks, checkpoint=plan.checkpoint)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="minimax_m3",
    served_model_name="MiniMax-M3.1-Flash",
    prepare=prepare,
    usage=usage,
    # Recommended sampling: https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash#usage.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1),
    checkpoints=(
        Checkpoint(
            alias="minimax-m3-1-flash", repository="MiniMaxAI/MiniMax-M3.1-Flash",
            revision="9e7ebacac350497ad4a1272a2e817639bf11561c", architecture="MiniMaxM3SparseForConditionalGeneration",
            stored_bytes=252_405_263_874, device_bytes=252_405_263_874, draft_bytes=2_301_852_674,
            expansion_bytes=2 << 30,
            scale_compressed_device_bytes={"csf": 252_405_263_874},
        ),
    ),
    context=1048576,
    page_tokens=128,
    parameters=MiniMaxM3Parameters(
        max_concurrency=32,
        max_concurrency_limit=64,
        prefill_chunk=4096,
        speculation=SpeculationParameters(
            draft_tokens=5,
            max_draft_tokens=7,
            token_capacity=128,
            packed_verification=True,
            draft_prefix_selection="gpu",
            graph_budget=64,
        ),
        vision=VisionParameters(separable_encoder=True),
    ),
    verification_rows=0,
    max_images=4,
    image_bytes=32 << 20,
    ranks=MappingProxyType({4: RankCosts(8_265_984, 2_949_120, 8 << 30)}),
    plain_ranks=MappingProxyType({4: RankCosts(3_023_104, 2_949_120, 8 << 30)}),
)
