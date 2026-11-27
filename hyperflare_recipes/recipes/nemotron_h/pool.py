# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mamba continuations and paged FP8 key/value histories."""

import torch

from hyperflare_recipes.components.paged_state import PagedStateStorage
from hyperflare_recipes.components.recurrent import DeltaState
from ignition.components.state import StateLayout

from .config import PAGE_TOKENS

REPLAY_RING = 16


def page_specs(config, shard=None):
    width = config.local_kv_heads(shard) * config.text["head_dim"]
    for index, kind in enumerate(config.layers):
        if kind == "attention":
            for field in ("key", "value"):
                yield f"layer{index}.{field}", (PAGE_TOKENS * width,)


def page_bytes(config, shard=None):
    return sum(shape[0] for _, shape in page_specs(config, shard))


def continuation_layout(config, shard=None):
    c, regions = config.text, []
    heads, groups = config.local_mamba_heads(shard), config.local_n_groups(shard)
    conv_width = config.local_convolution_width(shard)
    for index, kind in enumerate(config.layers):
        if kind == "mamba":
            columns = c["conv_kernel"] - 1 + (config.draft.block_size - 1 if config.draft is not None else 0)
            regions.extend(((f"layer{index}.convolution", (conv_width, columns), torch.bfloat16),
                (f"layer{index}.recurrent", (heads, c["mamba_head_dim"], c["ssm_state_size"]), torch.bfloat16)))
            if config.draft is not None:
                regions.extend(((f"layer{index}.replay_x", (heads, REPLAY_RING, c["mamba_head_dim"]), torch.bfloat16),
                    (f"layer{index}.replay_b", (groups, REPLAY_RING, c["ssm_state_size"]), torch.bfloat16),
                    (f"layer{index}.replay_dt", (heads, REPLAY_RING), torch.float32)))
    if config.draft is not None:
        regions.extend((("mamba_ring_start", (1,), torch.int32), ("mamba_pending_count", (1,), torch.int32),
                        ("draft_anchor", (config.hidden,), torch.bfloat16)))
        draft = config.draft.text
        draft_kv = config.draft.local_kv_heads(shard)
        for index in range(config.draft.layers):
            for field in ("key", "value"):
                regions.append((f"draft{index}.{field}",
                    (draft["sliding_window"], draft_kv, draft["head_dim"]), torch.float8_e4m3fn))
    regions.extend((name, shape, torch.uint8) for name, shape in page_specs(config, shard))
    regions.extend((("last_hidden", (config.hidden,), torch.bfloat16), ("consumed_tokens", (1,), torch.int64)))
    return StateLayout.from_specs(regions)


class ContinuationPool(PagedStateStorage):
    @classmethod
    def for_model(cls, config, slots, pages, device, *, shard=None, **kwargs):
        return cls.allocate(continuation_layout(config, shard), page_specs(config, shard), slots, pages, device, **kwargs)

    def recurrent(self, layer):
        return DeltaState(self.reserved_region(f"layer{layer}.convolution"), self.reserved_region(f"layer{layer}.recurrent"))

    def attention(self, layer, config, shard=None):
        c = config.text
        return tuple(self.pages[f"layer{layer}.{field}"].view(torch.float8_e4m3fn).view(
            -1, PAGE_TOKENS, config.local_kv_heads(shard), c["head_dim"]) for field in ("key", "value"))

    def replay_records(self, layer):
        return tuple(self.reserved_region(f"layer{layer}.replay_{field}") for field in ("x", "b", "dt"))
