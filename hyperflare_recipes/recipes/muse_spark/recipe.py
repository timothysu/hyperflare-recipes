# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Spark Open text preparation, DFlash2 drafts, and FP8 attention caches."""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from ignition.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults
from ignition.parameters import (
    PromptLogprobsParameters,
    RecipeParameters,
    SpeculationParameters,
)
from ignition.preparation.bootstrap import common_preparation_kwargs

from hyperflare_recipes.parameters import NativeDenseActivationsParameters


@dataclass(frozen=True)
class MuseSparkParameters(RecipeParameters):
    """Preparation parameters for Muse Spark Open checkpoints."""

    native_dense_activations: NativeDenseActivationsParameters = NativeDenseActivationsParameters()


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None):
    from ignition.preparation.rank_group import prepare_group
    from .bootstrap import PreparationConfig, prepare_rank

    if ranks is not None:
        raise ValueError("Muse Spark preparation requires eight GPUs on one host")
    draft_snapshot = None
    if plan.speculation is not None:
        from huggingface_hub import snapshot_download
        from .draft_config import REPOSITORY, REVISION

        draft_snapshot = Path(snapshot_download(REPOSITORY, revision=REVISION,
            cache_dir=plan.hub_cache, allow_patterns=("config.json", "model.safetensors")))
    config = PreparationConfig(
        **common_preparation_kwargs(plan, compile_forwards=compile_forwards,
                                    kernel_cache=kernel_cache, retain_graphs=retain_graphs),
        devices=tuple(range(plan.world_size)),
        draft_snapshot=draft_snapshot,
    )
    return prepare_group(snapshot, config, prepare_rank, progress=progress, core_storage=core_storage,
                         checkpoint=plan.checkpoint)


def usage(prepared):
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"])
                 for rank in prepared.memory_report.get("ranks", (prepared.memory_report,)))


RECIPE = Recipe(name="muse_spark", served_model_name="Muse-Spark-Open-816B-A42B",
    # Recommended sampling: https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=1., top_k=-1),
    prepare=prepare, usage=usage,
    checkpoints=(Checkpoint(alias="muse-spark-open-nvfp4", repository="meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
        revision="4239e4db19376904f2e82c167e78f70abb5455fe", architecture="MuseSparkForConditionalGeneration",
        stored_bytes=533_010_718_720, device_bytes=545_401_038_848,
        draft_bytes=12_768_520_192, expansion_bytes=1 << 30),),
    context=262144, page_tokens=128, verification_rows=0,
    parameters=MuseSparkParameters(
        max_concurrency=4, max_concurrency_limit=4, prefill_chunk=4096,
        speculation=SpeculationParameters(draft_tokens=7, max_draft_tokens=7, token_capacity=32,
                                          packed_verification=True, graph_budget=64),
        prompt_logprobs=PromptLogprobsParameters(),
    ),
    # Five FP8 draft key/value rings: 2048 positions, two local KV heads, 64 elements per head.
    ranks=MappingProxyType({8: RankCosts(27_525_120, 524_288, 8 << 30)}),
    plain_ranks=MappingProxyType({8: RankCosts(24_903_680, 524_288, 8 << 30)}))
