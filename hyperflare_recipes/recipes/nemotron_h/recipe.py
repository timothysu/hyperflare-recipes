# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-H Bolt and Super VL preparation capacities and serving defaults."""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from ignition.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults
from ignition.parameters import (
    PromptLogprobsParameters,
    RecipeParameters,
    SpeculationParameters,
    VisionParameters,
)
from ignition.preparation.bootstrap import common_preparation_kwargs

from hyperflare_recipes.parameters import NativeDenseActivationsParameters, StreamOverlapParameters


@dataclass(frozen=True)
class NemotronHParameters(RecipeParameters):
    """Preparation parameters for Nemotron-H checkpoints."""

    native_dense_activations: NativeDenseActivationsParameters = NativeDenseActivationsParameters()
    stream_overlap: StreamOverlapParameters = StreamOverlapParameters()


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None):
    from .bootstrap import PreparationConfig, prepare as prepare_model, prepare_rank

    if plan.world_size not in (1, 2):
        raise ValueError("Nemotron-H preparation requires one or two GPUs")
    if plan.world_size == 1 and ranks is not None:
        raise ValueError("single-rank Nemotron-H preparation does not use a multi-host rendezvous")
    draft = None
    if plan.speculation is not None:
        from huggingface_hub import snapshot_download
        from .draft_config import draft_source

        repo, rev = draft_source(plan.checkpoint.repository, Path(snapshot))
        draft = snapshot_download(repo, revision=rev, cache_dir=plan.hub_cache,
                                  allow_patterns=("config.json", "model.safetensors"))
    config = PreparationConfig(
        **common_preparation_kwargs(plan, compile_forwards=compile_forwards,
                                    kernel_cache=kernel_cache, retain_graphs=retain_graphs),
        devices=tuple(range(plan.world_size)) if ranks is None else (0,) * plan.world_size,
        hosts=None if ranks is None else ranks.hosts,
        benchmark_selection=plan.benchmark_selection,
        hub_cache=plan.hub_cache,
        draft_checkpoint=draft,
    )
    if plan.world_size == 1:
        return prepare_model(snapshot, config, progress=progress, core_storage=core_storage)
    from ignition.preparation.rank_group import prepare_group

    return prepare_group(snapshot, config, prepare_rank, progress=progress, core_storage=core_storage,
                         ranks=ranks, checkpoint=plan.checkpoint)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="nemotron_h", prepare=prepare, usage=usage,
    served_model_name="NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",
    # Text sampling defaults verified against both model cards:
    # https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB
    # https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1, min_p=0.,
        repetition_penalty=1., presence_penalty=0., frequency_penalty=0.),
    checkpoints=(
        Checkpoint(alias="nemotron-bolt-24gb", repository="nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",
            revision="c6cc2e8ffa353403213d6b9b18aaf8eb684d705a", architecture="NemotronH_Omni_Reasoning_V3",
            stored_bytes=24_079_678_824, device_bytes=25_063_117_344, draft_bytes=983_438_520,
            ranks={1: RankCosts(107_365_120, 524_288, 4 << 30)},
            plain_ranks={1: RankCosts(86_876_416, 524_288, 4 << 30)}),
        # Super VL NVFP4 skips the 1.65 GiB MTP tensors and video embedder, loading backbone,
        # C-RADIO vision (with vision_final_layernorm), and LiLiCorr draft weights. Plain dicts
        # keep checkpoint metadata picklable across multi-host preparation ranks.
        Checkpoint(alias="nemotron-3.5-super-vl-nvfp4", repository="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4",
            revision="ae1368329b1fc76b71534c5d04b87b35d9cc6e15", architecture="NemotronH_Omni_Reasoning_V3",
            served_model_name="NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4",
            stored_bytes=74_986_817_096, device_bytes=74_531_164_588, draft_bytes=1_317_933_948,
            expansion_bytes=6 << 30, decode_weight_bytes=9_765_491_976,
            ranks={
                1: RankCosts(188_629_760, 524_288, 6 << 30),
                2: RankCosts(94_323_456, 262_144, 6 << 30),
            },
            plain_ranks={
                1: RankCosts(86_876_416, 524_288, 6 << 30),
                2: RankCosts(43_442_432, 262_144, 6 << 30),
            }),
    ),
    context=262144, page_tokens=128, verification_rows=0, max_images=4, image_bytes=32 << 20,
    parameters=NemotronHParameters(
        max_concurrency=32, max_concurrency_limit=64, prefill_chunk=4096,
        speculation=SpeculationParameters(draft_tokens=7, max_draft_tokens=7, token_capacity=128,
                                          packed_verification=True, graph_budget=64),
        vision=VisionParameters(),
        prompt_logprobs=PromptLogprobsParameters(),
    ),
    benchmark_world_sizes=(1,),
    ranks=MappingProxyType({
        1: RankCosts(107_365_120, 524_288, 4 << 30),
        2: RankCosts(94_323_456, 262_144, 6 << 30),
    }),
    plain_ranks=MappingProxyType({
        1: RankCosts(86_876_416, 524_288, 4 << 30),
        2: RankCosts(43_442_432, 262_144, 6 << 30),
    }),
)
