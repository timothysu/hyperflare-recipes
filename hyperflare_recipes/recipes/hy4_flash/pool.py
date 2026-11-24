# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared 256-token sparse MLA/C4-index pages and per-request continuation state."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hyperflare_recipes.components.paged_state import PagedStateStorage
from ignition.components.state import StateLayout

from .config import (
    HIDDEN_SIZE,
    INDEX_HEAD_DIM,
    INDEX_KPOOL,
    KV_LORA_RANK,
    QK_ROPE_HEAD_DIM,
    SWA_WINDOW,
)

# One logical page covers 256 tokens: four 64-record sparse latent sub-pages (656 B
# per record) and one exact 64-record FP8 C4 index page (132 B per record).
PAGE_TOKENS = 256
LATENT_SUBPAGE_TOKENS = 64
SPARSE_RECORD_BYTES = 656
INDEX_PAGE_RECORDS = PAGE_TOKENS // INDEX_KPOOL  # 64
INDEX_RECORD_BYTES = 132
SWA_RECORD_WIDTH = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576 BF16 elements


def page_specs(config):
    """Per-page storage layout across all 12 sparse indexed layers."""
    for layer in config.sparse_layers:
        yield f"layer{layer}.latent_page", (PAGE_TOKENS * SPARSE_RECORD_BYTES,)
        yield f"layer{layer}.index_page", (INDEX_PAGE_RECORDS * INDEX_RECORD_BYTES,)


def page_bytes(config) -> int:
    return sum(shape[0] for _, shape in page_specs(config))


def continuation_layout(config, *, dfly: bool = False) -> StateLayout:
    """Complete immutable continuation state for one active request or snapshot."""
    regions = []
    # 30 SWA layers: 512-token BF16 latent ring [512, 576].
    for layer in config.swa_layers:
        regions.append((f"layer{layer}.window", (SWA_WINDOW, SWA_RECORD_WIDTH), torch.bfloat16))
    # 12 sparse layers: pending C4 compressor tokens (at most 3) + group start 3D M-RoPE position.
    for layer in config.sparse_layers:
        regions.append((f"layer{layer}.pending_keys", (INDEX_KPOOL - 1, INDEX_HEAD_DIM), torch.bfloat16))
        regions.append((f"layer{layer}.pending_group_pos", (3,), torch.int64))
    # Partial-page snapshots of the 12 sparse layers for prefix-cache checkpointing.
    regions.extend((name, shape, torch.uint8) for name, shape in page_specs(config))
    if dfly and config.draft is not None:
        regions.append(
            ("dfly_target_indices", (config.draft.layers, config.draft.target_index_topk), torch.int32)
        )
    regions.extend((
        ("last_hidden", (HIDDEN_SIZE,), torch.bfloat16),
        ("consumed_tokens", (1,), torch.int64),
        ("rotary_offset", (1,), torch.int64),
        ("engram_history", (2,), torch.int64),
    ))
    return StateLayout.from_specs(regions, rotary_region="rotary_offset")


@dataclass(frozen=True)
class ContinuationPool(PagedStateStorage):
    config: object

    @classmethod
    def allocate(
        cls,
        config,
        slots: int,
        pages: int,
        device: torch.device,
        *,
        storage=None,
        maximum_pages: int | None = None,
        maximum_rows: int | None = None,
        backed_slots: int | None = None,
        dfly: bool = False,
    ) -> ContinuationPool:
        backing = PagedStateStorage.allocate(
            continuation_layout(config, dfly=dfly),
            page_specs(config),
            slots,
            pages,
            device,
            storage=storage,
            maximum_pages=maximum_pages,
            maximum_rows=maximum_rows,
            backed_slots=backed_slots,
        )
        return cls(
            backing.backing,
            backing.layout,
            backing.memory_capacity,
            backing.pages,
            backing.reserved_backing,
            config,
        )

    def window(self, layer: int) -> torch.Tensor:
        """Reserved SWA ring storage ``[max_rows, 512, 576]`` BF16."""
        return self.reserved_region(f"layer{layer}.window")

    def latent_cache(self, layer: int) -> torch.Tensor:
        """View ``layer{L}.latent_page`` as ``[pages * 4, 64, 656]`` uint8 for b12x sparse MLA."""
        raw = self.pages[f"layer{layer}.latent_page"]
        return raw.view(raw.shape[0] * (PAGE_TOKENS // LATENT_SUBPAGE_TOKENS), LATENT_SUBPAGE_TOKENS, SPARSE_RECORD_BYTES)

    def index_cache(self, layer: int) -> torch.Tensor:
        """View ``layer{L}.index_page`` as ``[pages, 64 * 132]`` uint8 for PreparedFP8Indexer."""
        return self.pages[f"layer{layer}.index_page"]
