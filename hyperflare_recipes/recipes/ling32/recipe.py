# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Ling-3.2-Air TP=1 preparation capacities, memory costs, and serving defaults."""

from types import MappingProxyType

from ignition.declaration import (
    Checkpoint,
    RankCosts,
    RankUsage,
    Recipe,
    SamplingDefaults,
    Speculation,
)

# Seven target MLA layers and one Layer-42 MTP MLA layer, each storing 128 BF16
# 576-element latent records per page (147,456 bytes/layer/page).
_PAGE_LAYER_BYTES = 128 * 576 * 2
_TARGET_MLA_LAYERS = 7
_GRAPH_BYTES = 3 << 30
# 256-byte-aligned continuation row sizes for 35 KDA layers (32 heads x 128x128
# FP32 recurrent + 12288 x (3 + drafts) BF16 conv), 7 target MLA partial pages,
# optional 2-page MTP partial buffer + draft target hidden, last_hidden, and consumed_tokens.
# Speculative row sizes reserve the max_draft_tokens=7 upper bound (at the default
# draft_tokens=5, continuation_layout allocates 81,619,200 bytes for text-only and
# 81,619,456 bytes with vision). Enabling vision adds one 256-byte-aligned int64
# rotary_offset slot per continuation row.
_PLAIN_STATE_ROW_BYTES = 77_018_368
_PLAIN_IMAGE_STATE_ROW_BYTES = 77_018_624
_SPECULATIVE_STATE_ROW_BYTES = 83_339_520
_SPECULATIVE_IMAGE_STATE_ROW_BYTES = 83_339_776

# Image feature slot: 16,384 BF16 vectors at the language-model width 2,560.
VISION_SLOT_BYTES = 16_384 * 2_560 * 2  # 83,886,080
# Float32 patch upload and its interpolation, position, and counter metadata.
VISION_INPUT_BYTES = 101_711_900
# BF16 vision tower (`model.visual.*`) and projector (`linear_proj.*`) weights.
VISION_TOWER_BYTES = 868_497_888
# Fixed persistent vision storage (weights and upload buffers); per-slot feature
# arenas scale with `VISION_SLOT_BYTES` via `RankCosts.with_image_slots`.
VISION_BYTES = VISION_TOWER_BYTES + VISION_INPUT_BYTES


def prepare(
    snapshot,
    plan,
    *,
    compile_forwards,
    kernel_cache,
    progress,
    retain_graphs=False,
    core_storage=None,
    ranks=None,
):
    from .bootstrap import PreparationConfig, prepare_rank

    if ranks is not None or plan.world_size != 1:
        raise ValueError("Ling-3.2-Air preparation supports TP=1 on one GPU")

    speculation = plan.speculation
    drafting = (
        {}
        if speculation is None
        else {
            "speculative_widths": speculation.widths,
            "speculative_fixed_width": speculation.fixed_width,
            "speculative_token_capacity": plan.speculative_token_capacity,
            "draft_head_quantization": speculation.draft_head_quantization,
        }
    )
    config = PreparationConfig(
        mtp_only=speculation is not None and speculation.mtp_only,
        collective_tuning=plan.collective_tuning,
        collective_ladder=plan.collective_ladder,
        prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers,
        autotune_effort=plan.autotune_effort,
        tuning_replica=plan.tuning_replica,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        devices=(0,),
        context=plan.recipe.context,
        max_active=plan.concurrency,
        image_concurrency=0 if plan.language_model_only else plan.image_concurrency,
        prefill_chunk_size=plan.prefill_chunk,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes,
        compile_decoder=compile_forwards,
        retain_graphs=retain_graphs,
        kernel_cache=kernel_cache,
        stream_overlap=plan.stream_overlap,
        mtp=speculation is not None,
        expert_scale_compression=plan.expert_scale_compression,
        **drafting,
    )
    return prepare_rank(snapshot, config, rank=0, group=None, progress=progress, core_storage=core_storage)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="ling32",
    served_model_name="Ling-3.2-Air",
    prepare=prepare,
    usage=usage,
    # Official sampling recommendations from https://huggingface.co/inclusionAI/Ling-3.2-Air
    # model card ("Deployment" / "Quickstart": temperature=1.0, top_p=0.95, top_k=20).
    sampling_defaults=SamplingDefaults(temperature=1.0, top_p=0.95, top_k=20),
    checkpoints=(
        Checkpoint(
            alias="ling-3.2-air-nvfp4",
            repository="timbuksu/Ling-3.2-Air-NVFP4",
            revision="a95def604aa78f3111e6eac10a65f2651d5da828",
            architecture="BailingMoeV3VLForConditionalGeneration",
            stored_bytes=75_290_852_960,
            device_bytes=75_254_142_560,
            draft_bytes=1_773_380_608,
            scale_compressed_device_bytes={"csf": 71_523_427_000},
            decode_weight_bytes=4_960_000_000,
        ),
    ),
    context=262144,
    page_tokens=128,
    concurrency=32,
    max_concurrency=64,
    prefill_chunk=4096,
    speculation=Speculation(draft_tokens=5, token_capacity=128),
    max_draft_tokens=7,
    verification_rows=0,
    stream_overlap=True,
    expert_scale_compression="csf",
    draft_head_quantizations=("mxfp8", "nvfp4"),
    max_images=4,
    image_bytes=32 << 20,
    language_model_only=True,
    b12x_environment=MappingProxyType({"B12X_W4A16_A4_PREFILL": "1"}),
    ranks=MappingProxyType(
        {
            1: RankCosts(
                state_row_bytes=_SPECULATIVE_STATE_ROW_BYTES,
                page_bytes=(_TARGET_MLA_LAYERS + 1) * _PAGE_LAYER_BYTES,
                graph_bytes=_GRAPH_BYTES,
                page_tokens=128,
                vision_bytes=VISION_BYTES,
                vision_slot_bytes=VISION_SLOT_BYTES,
                image_state_row_bytes=_SPECULATIVE_IMAGE_STATE_ROW_BYTES,
            ),
        }
    ),
    plain_ranks=MappingProxyType(
        {
            1: RankCosts(
                state_row_bytes=_PLAIN_STATE_ROW_BYTES,
                page_bytes=_TARGET_MLA_LAYERS * _PAGE_LAYER_BYTES,
                graph_bytes=_GRAPH_BYTES,
                page_tokens=128,
                vision_bytes=VISION_BYTES,
                vision_slot_bytes=VISION_SLOT_BYTES,
                image_state_row_bytes=_PLAIN_IMAGE_STATE_ROW_BYTES,
            ),
        }
    ),
)
