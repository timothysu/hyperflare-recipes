# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Bailing V3 chat envelope, XML thinking tags, and GLM argument-tag tools."""

import hashlib
from pathlib import Path

from tokenizers import Tokenizer

from ignition.preparation import contract as c
from ignition.preparation import runtime_core_pb2 as p
from ignition.preparation.frontend import (
    QwenPositionScheme,
    QwenResizeKernel,
    annotated_template_asset,
    prepare_frontend as _prepare_frontend,
)

from .config import IMAGE_TOKEN_ID, VIDEO_TOKEN_ID

ENDOFTEXT_TOKEN_ID = 156892
ROLE_END_TOKEN_ID = 156895
VISION_START_TOKEN_ID = 157158
VISION_END_TOKEN_ID = 157159
VIDEO_END_TOKEN_ID = 157160

# Video placeholder tokens are always rejected (image-only multimodal support).
VIDEO_PLACEHOLDER_TOKENS = (VIDEO_TOKEN_ID, VIDEO_END_TOKEN_ID)
# Vision placeholder tokens rejected during text-only serving.
VISION_PLACEHOLDER_TOKENS = (
    VIDEO_TOKEN_ID,
    IMAGE_TOKEN_ID,
    VISION_START_TOKEN_ID,
    VISION_END_TOKEN_ID,
    VIDEO_END_TOKEN_ID,
)

# Explicit structural parameters passed to the core Qwen image profile.
BAILING_QWEN_IMAGE_PARAMS = dict(
    qwen_model_type="bailing_moe_v3_vl",
    qwen_processor_class="BailingMM2Processor",
    qwen_image_processor_type="BailingMM2ImageProcessor",
    qwen_channel_normalization=True,
    qwen_position_scheme=QwenPositionScheme.CENTERED_STEP,
    qwen_resize_kernel=QwenResizeKernel.PILLOW_BICUBIC,
    qwen_tokens={"image_token": "<|image_pad|>", "video_token": "<|video_pad|>"},
)


def frontend(vision: bool = False):
    """Bailing V3 chat syntax; with ``vision``, Qwen-style patch preparation and 3D M-RoPE."""
    images = {}
    if vision:
        images = dict(
            image_tokens=dict(feature_token=IMAGE_TOKEN_ID, span=None),
            image_preparation=p.QWEN_IMAGE_PREPARATION,
            image_layout=p.QWEN_IMAGE_LAYOUT,
        )
    return c.frontend(
        tools=p.GLM_ARGUMENT_TAGS,
        assistant_marker="<role>ASSISTANT</role>",
        message_end_markers=["<|role_end|>"],
        rejected_tokens=list(VIDEO_PLACEHOLDER_TOKENS if vision else VISION_PLACEHOLDER_TOKENS),
        prefix_boundary=c.prefix_boundary(
            ["system"],
            append_role="user",
            minimum_messages=0,
            required_start="<role>SYSTEM</role>",
            required_marker="<|role_end|>",
        ),
        position_dimensions=3 if vision else 0,
        **images,
    )


FRONTEND = frontend(vision=False)


def prepare_frontend(snapshot, *, eos_token_ids, vision: bool = False):
    """Validate Bailing V3 control tokens and build Helios frontend settings."""
    snapshot = Path(snapshot)
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    pinned_markers = {
        "<|endoftext|>": ENDOFTEXT_TOKEN_ID,
        "<|role_end|>": ROLE_END_TOKEN_ID,
    }
    if vision:
        pinned_markers.update(
            {
                "<|image_pad|>": IMAGE_TOKEN_ID,
                "<|video_pad|>": VIDEO_TOKEN_ID,
                "<|vision_start|>": VISION_START_TOKEN_ID,
                "<|vision_end|>": VISION_END_TOKEN_ID,
            }
        )
    for marker, expected in pinned_markers.items():
        if tokenizer.token_to_id(marker) != expected:
            raise ValueError(f"Ling-3.2-Air tokenizer marker {marker!r} must map to token ID {expected}")
    for marker in ("<role>", "</role>", "<tool_call>", "</tool_call>"):
        if tokenizer.token_to_id(marker) is None:
            raise ValueError(f"Ling-3.2-Air tokenizer is missing required control token {marker!r}")

    effective = [eos_token_ids] if isinstance(eos_token_ids, int) else list(eos_token_ids)
    for required_eos in (ENDOFTEXT_TOKEN_ID, ROLE_END_TOKEN_ID):
        if required_eos not in effective:
            effective.append(required_eos)
    return _prepare_frontend(
        snapshot,
        eos_token_ids=list(dict.fromkeys(effective)),
        image_profile="qwen" if vision else None,
        **(BAILING_QWEN_IMAGE_PARAMS if vision else {}),
    )


def runtime_assets(snapshot):
    """Return validated tokenizer and message-marked chat template assets."""
    snapshot = Path(snapshot)
    tokenizer = snapshot / "tokenizer.json"
    with tokenizer.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "tokenizer": {"path": str(tokenizer), "sha256": digest},
        "chat_template": annotated_template_asset(
            (snapshot / "chat_template.jinja").read_bytes(),
            Path(__file__).with_name("assets") / "chat_template.patch.json",
        ),
    }
