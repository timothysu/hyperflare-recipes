# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU Qwen3.8-27B NVFP4 text and image preparation without host-mapped storage."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from ignition.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, SamplingRange
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
    OnlineMXFP8Parameters,
)

# Image feature slot: 16,384 BF16 vectors at the language-model width 5,120.
VISION_SLOT_BYTES = 16_384 * 5_120 * 2  # 167,772,160
# Float32 patch upload and its interpolation, position, and counter metadata.
VISION_INPUT_BYTES = 101_711_900
# BF16 vision tower weights under `model.visual.*` in the NVFP4 checkpoint.
VISION_TOWER_BYTES = 921_460_192
# Fixed persistent vision storage (weights and upload buffers); per-slot feature
# arenas scale with `VISION_SLOT_BYTES` via `RankCosts.with_image_slots`.
VISION_BYTES = VISION_TOWER_BYTES + VISION_INPUT_BYTES


@dataclass(frozen=True)
class Qwen38DenseParameters(RecipeParameters):
    """Preparation parameters for Qwen3.8-27B dense."""

    native_dense_activations: NativeDenseActivationsParameters = NativeDenseActivationsParameters()
    online_mxfp8: OnlineMXFP8Parameters = OnlineMXFP8Parameters()
    draft_quantization: DraftQuantizationParameters = DraftQuantizationParameters(
        draft_head_choices=("mxfp8", "nvfp4"),
    )


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None):
    from .bootstrap import PreparationConfig, prepare as prepare_model

    if plan.world_size != 1 or ranks is not None:
        raise ValueError("Qwen dense preparation supports one local GPU")
    speculation = plan.speculation
    draft_snapshot = None
    if speculation is not None:
        from huggingface_hub import snapshot_download
        from .draft_config import REPOSITORY, REVISION

        draft_snapshot = Path(snapshot_download(
            REPOSITORY, revision=REVISION, cache_dir=plan.hub_cache,
            allow_patterns=("config.json", "model.safetensors"),
        ))
    selection = plan.benchmark_selection
    kwargs = common_preparation_kwargs(
        plan, compile_forwards=compile_forwards, kernel_cache=kernel_cache, retain_graphs=retain_graphs,
    )
    if selection is not None:
        params = kwargs["parameters"]
        vision = (
            replace(params.vision, language_model_only=True, image_concurrency=None)
            if params.vision is not None
            else None
        )
        kwargs["parameters"] = replace(params, max_concurrency=selection.max_batch, vision=vision)
    config = PreparationConfig(
        **kwargs,
        benchmark_selection=selection,
        draft_snapshot=draft_snapshot,
    )
    return prepare_model(snapshot, config, progress=progress, core_storage=core_storage)


def usage(prepared):
    return (RankUsage(prepared.memory_report["pages"], prepared.memory_report["free_device_bytes"]),)


RECIPE = Recipe(
    name="qwen38_dense",
    served_model_name="Qwen3.8-27B",
    prepare=prepare,
    usage=usage,
    # Thinking defaults; bounds span thinking/instruct presets: https://huggingface.co/Qwen/Qwen3.8-27B#best-practices.
    sampling_defaults=SamplingDefaults(
        temperature=1., top_p=.95, top_k=20, min_p=0.,
        presence_penalty=0., repetition_penalty=1., frequency_penalty=0.,
        ranges={"temperature": SamplingRange(.7, 1.), "top_p": SamplingRange(.8, .95),
                "presence_penalty": SamplingRange(0., 2.)},
    ),
    checkpoints=(
        Checkpoint(
            alias="qwen3.8-27b", repository="nvidia/Qwen3.8-27B-NVFP4",
            revision="482ca0f3832238542f8f5295dde86b5f22711d80", architecture="Qwen3_5ForConditionalGeneration",
            stored_bytes=21_921_428_072, device_bytes=24_104_235_656, draft_bytes=3_953_666_560,
            expansion_bytes=307_871_744,
        ),
        Checkpoint(
            alias="qwen3.8-27b-qad", repository="local-inference-lab/Qwen3.8-27B-NVFP4-QAD",
            revision="b9389787cbb5a09fbcf3517bdf673b5a56b16b02", architecture="Qwen3_5ForConditionalGeneration",
            stored_bytes=25_578_292_192, device_bytes=27_761_099_776, draft_bytes=3_953_666_560,
            expansion_bytes=1_619_001_344,
        ),
    ),
    context=262144,
    page_tokens=128,
    parameters=Qwen38DenseParameters(
        max_concurrency=32,
        max_concurrency_limit=64,
        prefill_chunk=4096,
        speculation=SpeculationParameters(
            draft_tokens=7,
            max_draft_tokens=15,
            token_capacity=128,
            packed_verification=True,
            graph_budget=64,
            method="dflash2",
            supported_methods=("dflash2",),
        ),
        vision=VisionParameters(separable_encoder=True),
        prompt_logprobs=PromptLogprobsParameters(),
    ),
    verification_rows=0,
    benchmark_world_sizes=(1,),
    speculation_study=True,
    benchmark_language_model_only=True,
    max_images=4,
    image_bytes=32 << 20,
    # Reserve continuation memory for the supported sixteen-token verification capacity.
    # A plan with image slots stores the three-axis rotary offset in a row 256 bytes larger.
    ranks=MappingProxyType({1: RankCosts(
        193_865_984, 4_194_304, 4 << 30,
        vision_bytes=VISION_BYTES, vision_slot_bytes=VISION_SLOT_BYTES,
        image_state_row_bytes=193_866_240,
    )}),
    plain_ranks=MappingProxyType({1: RankCosts(
        158_148_864, 4_194_304, 4 << 30,
        vision_bytes=VISION_BYTES, vision_slot_bytes=VISION_SLOT_BYTES,
        image_state_row_bytes=158_149_120,
    )}),
)
