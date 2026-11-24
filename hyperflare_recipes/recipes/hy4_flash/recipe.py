# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Hy4-flash recipe declaration, serving range, and TP=1 Blackwell memory costs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from ignition.declaration import (
    Checkpoint,
    RankCosts,
    RankUsage,
    Recipe,
    SamplingDefaults,
    Speculation,
)

# 12 sparse indexed layers x (256 * 656 B GLM_NSA latent + 64 * 132 B C4 index) per 256-token page.
_PAGE_BYTES = 12 * (256 * 656 + 64 * 132)  # 2,116,608 B (2.02 MiB)
# Continuation row bytes (aligned via pool.continuation_layout): 30 SWA-512 BF16 rings
# (30 * 512 * 576 * 2 B = 17,694,720 B) + 12 sparse partial pages (2,116,608 B) +
# C4 pending keys/positions + last_hidden/consumed/rotary_offset/engram_history; DFly2
# adds [5, 2048] int32 logical target indices (40,960 B).
_PLAIN_STATE_ROW_BYTES = 19_830_528
_DFLY_STATE_ROW_BYTES = 19_871_488
# Captured CUDA graph pool, SigLIP2 vision weights (~933 MiB), and 16,384 x 3,072 BF16 feature slot.
_GRAPH_BYTES = 3 << 30
_VISION_BYTES = 933 << 20
_VISION_SLOT_BYTES = 16384 * 3072 * 2


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
    """Prepare the Hy4-flash TP=1 graph catalog on local GPU 0."""
    from .bootstrap import PreparationConfig, chat_template_asset, prepare_rank

    if plan.world_size != 1 or ranks is not None:
        raise ValueError("Hy4-flash recipe requires TP=1 on a single host")

    speculation = plan.speculation
    drafting = (
        {}
        if speculation is None
        else {
            "speculative_widths": speculation.widths,
            "speculative_token_capacity": plan.speculative_token_capacity,
            "speculative_fixed_width": speculation.fixed_width,
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
        compile_decoder=compile_forwards,
        retain_graphs=retain_graphs,
        image_concurrency=plan.image_concurrency,
        language_model_only=plan.language_model_only,
        online_mxfp8=plan.online_mxfp8,
        online_mxfp8_scope=plan.online_mxfp8_scope,
        max_active=plan.concurrency,
        stream_overlap=plan.stream_overlap,
        context=plan.recipe.context,
        prefill_chunk_size=plan.prefill_chunk,
        minimum_pages=plan.minimum_pages,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes,
        kernel_cache=kernel_cache,
        expert_scale_compression=plan.expert_scale_compression,
        dfly=speculation is not None,
        **drafting,
    )
    config = replace(config, chat_template=chat_template_asset(Path(snapshot).resolve()))
    return prepare_rank(snapshot, config, rank=0, group=None, progress=progress, core_storage=core_storage)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="hy4_flash",
    served_model_name="Hy4-Flash",
    prepare=prepare,
    usage=usage,
    expert_scale_compression="csf",
    draft_head_quantizations=("nvfp4", "mxfp8"),
    online_mxfp8=True,
    online_mxfp8_scope="all",
    online_mxfp8_scopes=("all", "draft"),
    # Sampling defaults from Tencent Hy4-flash model card quickstart
    # (https://huggingface.co/tencent/Hy4-flash#quickstart: temperature=0.9, top_p=1.0).
    sampling_defaults=SamplingDefaults(temperature=0.9, top_p=1.0, top_k=-1),
    checkpoints=(
        Checkpoint(
            alias="hy4-flash-nvfp4",
            repository="timbuksu/Hy4-flash-NVFP4",
            revision="2bb816400ae80b1d1f7ac5f39c4e82691514e7ae",
            architecture="HYV4FlashForConditionalGeneration",
            stored_bytes=131_762_366_182,
            # Static weight/scale-compression estimates; exact SM120 peak measurement is tracked in docs/TODO.txt.
            device_bytes=94_100_000_000,
            draft_bytes=1_950_000_000,
            scale_compressed_device_bytes={"csf": 89_600_000_000},
        ),
    ),
    context=1048576,
    page_tokens=256,
    concurrency=8,
    max_concurrency=16,
    prefill_chunk=4096,
    speculation=Speculation(draft_tokens=5, adaptive=True, token_capacity=48),
    max_draft_tokens=7,
    verification_rows=0,
    stream_overlap=True,
    max_images=4,
    image_bytes=32 << 20,
    language_model_only=True,
    ranks=MappingProxyType(
        {
            1: RankCosts(
                _DFLY_STATE_ROW_BYTES,
                _PAGE_BYTES,
                _GRAPH_BYTES,
                vision_bytes=_VISION_BYTES,
                vision_slot_bytes=_VISION_SLOT_BYTES,
            ),
        }
    ),
    plain_ranks=MappingProxyType(
        {
            1: RankCosts(
                _PLAIN_STATE_ROW_BYTES,
                _PAGE_BYTES,
                _GRAPH_BYTES,
                vision_bytes=_VISION_BYTES,
                vision_slot_bytes=_VISION_SLOT_BYTES,
            ),
        }
    ),
)
