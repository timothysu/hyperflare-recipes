# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared graph inputs, paged latent/index storage, and frontend syntax for Hy4-flash."""

from __future__ import annotations

from ignition.preparation import contract as c
from ignition.preparation import runtime_core_pb2 as p

from .config import (
    IMAGE_END_TOKEN,
    IMAGE_NEWLINE_TOKEN,
    IMAGE_START_TOKEN,
    IMAGE_TOKEN,
    INDEX_KPOOL,
    VIDEO_TOKEN,
    VOCABULARY,
)

PATCH_CAPACITIES = (256, 1024, 4096, 16384)
FEATURES = 16384
VISION = dict(
    features_per_request=FEATURES,
    max_images=4,
    patches_per_feature=4,
    patch_capacities=list(PATCH_CAPACITIES),
)
TEXT_VISION = dict(
    features_per_request=0,
    max_images=0,
    patches_per_feature=0,
    patch_capacities=[],
)

ASSISTANT_MARKER = "<\uff5chy_start:opensource\uff5c>assistant<\uff5chy_middle:opensource\uff5c>"
MESSAGE_END_MARKER = "<\uff5chy_end:opensource\uff5c>"


def frontend(*, vision: bool = True) -> dict:
    """Hy4-flash chat markers, HyThink reasoning, HyArgumentTags tools, and RowNewline vision."""
    images = {}
    rejected = [VIDEO_TOKEN] if vision else [
        IMAGE_START_TOKEN,
        IMAGE_END_TOKEN,
        IMAGE_TOKEN,
        IMAGE_NEWLINE_TOKEN,
        VIDEO_TOKEN,
    ]
    if vision:
        images = dict(
            image_tokens=dict(feature_token=IMAGE_TOKEN, span=None),
            image_preparation=p.ROW_NEWLINE_IMAGE_PREPARATION,
            image_layout=p.ROW_NEWLINE_IMAGE_LAYOUT,
        )
    spec = c.frontend(
        tools=p.HY_ARGUMENT_TAGS,
        assistant_marker=ASSISTANT_MARKER,
        message_end_markers=[MESSAGE_END_MARKER],
        prefix_boundary=c.prefix_boundary(["system"]),
        # Both multimodal and text-only graphs bind RotaryBatchInputs.prefill_positions ([T, 3]).
        position_dimensions=3,
        rejected_tokens=rejected,
        **images,
    )
    spec["reasoning"] = dict(
        kind=p.HY_THINK,
        open="<think:opensource>",
        close="</think:opensource>",
        wrapping_newlines=0,
        terminal_markers=[],
    )
    return spec


def prepare(
    signatures,
    buffers,
    pool,
    *,
    page_tokens: int,
    draft_lag: int | None,
    rows: int = 1,
    embedding_tables=(),
    regions=(),
    image_slots: int = 4,
) -> dict:
    """Build the runtime-core ModelContract for Hy4-flash."""
    regions = list(regions)
    for name, tensor in pool.pages.items():
        tokens_per_record = INDEX_KPOOL if name.endswith(".index_page") else 1
        regions.append(
            c.history_region(
                buffers,
                "cache." + name,
                tensor,
                p.PAGED_LATENT,
                tokens_per_record=tokens_per_record,
            )
        )
    has_vision = any(s.kind == "vision" for s in signatures.values())
    inputs = {kind: c.text_inputs(kind) for kind in {s.kind for s in signatures.values()} if kind != "vision"}
    if has_vision:
        inputs["vision"] = c.vision_inputs(
            {field for s in signatures.values() if s.kind == "vision" for field in s.inputs}
        )
    for kind, values in inputs.items():
        if kind in ("decode", "speculative"):
            values.pop("token_ids", None)
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    consumed = pool.layout.region("consumed_tokens")
    cursor = (
        None
        if draft_lag is None
        else dict(
            state=c.state_view(
                "state_pool",
                pool.layout.row_bytes,
                offset=consumed.offset,
                nbytes=consumed.nbytes,
            ),
            lag_tokens=draft_lag,
        )
    )
    pending = c.state_view("request_pending_token", 8)
    graphs = c.graph_contracts(
        signatures,
        inputs,
        continuation=continuation,
        pending=pending,
        cursor=cursor,
        rows=rows,
        embedding_tables=embedding_tables,
    )
    return c.model_contract(
        vocabulary=VOCABULARY,
        input_vocabulary=VOCABULARY,
        page_tokens=page_tokens,
        page_tables=[dict(buffer="page_tables", page_offset=0)],
        regions=[
            dict(kind=p.CONTINUATION_STATE, storage=continuation, tokens_per_record=1, window_tokens=0),
            *regions,
        ],
        graphs=graphs,
        vision={**VISION, "feature_slots": image_slots} if has_vision else TEXT_VISION,
        frontend=frontend(vision=has_vision),
    )
