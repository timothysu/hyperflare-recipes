# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling 2 checkpoint declarations and preparation entry points."""

from dataclasses import dataclass, replace
from types import MappingProxyType

from ignition.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults
from ignition.parameters import RecipeParameters, SpeculationParameters, VisionParameters
from ignition.preparation.bootstrap import common_preparation_kwargs

from hyperflare_recipes.parameters import DraftQuantizationParameters, NativeDenseActivationsParameters

from .frontend import SAMPLING_DEFAULTS


@dataclass(frozen=True)
class Inkling2Parameters(RecipeParameters):
    """Preparation parameters for Inkling 2 checkpoints."""

    native_dense_activations: NativeDenseActivationsParameters = NativeDenseActivationsParameters()
    draft_quantization: DraftQuantizationParameters | None = None


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None):
    """Prepare Mini on one GPU or Small on four GPUs.

    With `ranks`, rank zero's multi-host rendezvous, each rank uses its own host's GPU 0.
    """
    from ignition.preparation.rank_group import prepare_group
    from .bootstrap import PreparationConfig, prepare_rank

    if plan.parameters.prompt_logprobs is not None and plan.parameters.prompt_logprobs.enabled:
        raise ValueError("Inkling 2 preparation does not capture prompt scoring graphs")
    config = PreparationConfig(
        **common_preparation_kwargs(plan, compile_forwards=compile_forwards,
                                    kernel_cache=kernel_cache, retain_graphs=retain_graphs),
        devices=tuple(range(plan.world_size)) if ranks is None else (0,) * plan.world_size,
        hosts=None if ranks is None else ranks.hosts,
        benchmark_selection=plan.benchmark_selection,
    )
    if plan.world_size == 1:
        return prepare_rank(snapshot, config, rank=0, group=None, progress=progress, core_storage=core_storage)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, core_storage=core_storage,
                         ranks=ranks, checkpoint=plan.checkpoint)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(name="inkling2", served_model_name="Inkling-2-Small-NVFP4", prepare=prepare, usage=usage,
    sampling_defaults=SAMPLING_DEFAULTS,
    checkpoints=(Checkpoint(alias="inkling-2-small-nvfp4", repository="/data/models/Inkling-2-Small-NVFP4",
        revision="local", architecture="Inkling2ForConditionalGeneration", stored_bytes=344_366_116_864,
        device_bytes=344_366_116_864, expansion_bytes=5_673_717_760, draft_bytes=2_684_203_392),),
    context=1048576, page_tokens=128, max_images=4, image_bytes=32 << 20, verification_rows=0,
    parameters=Inkling2Parameters(
        max_concurrency=8, max_concurrency_limit=32, prefill_chunk=4096,
        speculation=SpeculationParameters(draft_tokens=7, max_draft_tokens=7, token_capacity=128,
                                          packed_verification=True, graph_budget=64),
        vision=VisionParameters(separable_encoder=True),
    ),
    ranks=MappingProxyType({4: RankCosts(31_293_440, 655_360, 5 << 30)}),
    plain_ranks=MappingProxyType({4: RankCosts(26_935_296, 524_288, 5 << 30)}))


# Mini benchmark defaults: temperature 1, top_p 1, no top-k
# filter, and neutral penalties. The TML encoder supplies reasoning effort 0.9.
MINI_SAMPLING_DEFAULTS = SamplingDefaults(temperature=1., top_p=1., top_k=-1)

MINI_RECIPE = replace(RECIPE, name="inkling2_mini", served_model_name="inkling-2-mini",
    sampling_defaults=MINI_SAMPLING_DEFAULTS,
    parameters=replace(RECIPE.parameters, max_concurrency_limit=8, prefill_chunk=8192,
        speculation=SpeculationParameters(draft_tokens=7, max_draft_tokens=7, token_capacity=64,
                                          packed_verification=True, graph_budget=64),
        draft_quantization=DraftQuantizationParameters(
            draft_head_choices=("nvfp4", "mxfp8"),
            markov_head_choices=("nvfp4",),
        )),
    checkpoints=(Checkpoint(alias="inkling-2-mini", repository="thinkingmachines/Inkling-2-Mini",
        revision="ed40e7da625f594fc9c4990f5f66be24873c22d7", architecture="Inkling2LoopForConditionalGeneration",
        stored_bytes=19_525_870_752, device_bytes=20_263_810_112, draft_bytes=737_939_360),),
    ranks=MappingProxyType({1: RankCosts(107_347_968, 2_621_440, 3 << 30)}),
    plain_ranks=MappingProxyType({1: RankCosts(98_435_072, 2_097_152, 3 << 30)}))
