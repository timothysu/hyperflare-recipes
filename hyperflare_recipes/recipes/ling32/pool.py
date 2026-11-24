# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""KDA continuation rows, BF16 paged MLA latent pools, and trailing MTP pages."""

import torch

from hyperflare_recipes.components.paged_batch import ROTARY_OFFSET
from hyperflare_recipes.components.paged_state import PagedStateStorage
from hyperflare_recipes.components.recurrent import DeltaState
from ignition.components.state import StateLayout

from .config import PAGE_TOKENS, RECORD_BYTES, RECORD_DIM


def page_specs(config, *, mtp=True):
    """Target and optional Layer-42 MTP BF16 latent page regions."""
    page_nbytes = PAGE_TOKENS * RECORD_BYTES
    for index, kind in enumerate(config.layer_types):
        if kind == "mla":
            yield f"layer{index}.latent", (page_nbytes,)
    if mtp and config.num_mtp_layers > 0:
        yield "draft.latent", (page_nbytes,)


def page_bytes(config, *, mtp=True) -> int:
    return sum(shape[0] for _, shape in page_specs(config, mtp=mtp))


def continuation_layout(config, *, mtp=True, speculative_tokens=0, rotary_offset=False) -> StateLayout:
    """One continuation row holds all 35 KDA states, target partial pages, and MTP feedback.

    When MTP is enabled, ``draft.latent`` reserves two full page records to hold the
    one-token-lagged boundary and active draft pages across prefix snapshots, and
    ``draft.target_hidden`` retains the last committed post-norm target hidden state.
    When ``rotary_offset`` is enabled, one int64 offset per continuation row tracks
    the three-axis M-RoPE offset across image prompts and continuation snapshots.
    """
    if type(rotary_offset) is not bool:
        raise TypeError("rotary offset selection must be a boolean")
    heads = config.heads
    width = heads * 128
    regions = []
    for index, kind in enumerate(config.layer_types):
        if kind == "kda":
            regions.extend((
                (f"layer{index}.convolution", (3 * width, 3 + speculative_tokens), torch.bfloat16),
                (f"layer{index}.recurrent", (heads, 128, 128), torch.float32),
            ))
        else:
            regions.append((f"layer{index}.latent", (PAGE_TOKENS * RECORD_BYTES,), torch.uint8))
    if mtp and config.num_mtp_layers > 0:
        regions.append(("draft.latent", (2, PAGE_TOKENS * RECORD_BYTES), torch.uint8))
        regions.append(("draft.target_hidden", (config.hidden,), torch.bfloat16))
    regions.extend((
        ("last_hidden", (config.hidden,), torch.bfloat16),
        ("consumed_tokens", (1,), torch.int64),
    ))
    if rotary_offset:
        regions.append((ROTARY_OFFSET, (1,), torch.int64))
    return StateLayout.from_specs(
        regions,
        rotary_region=ROTARY_OFFSET if rotary_offset else None,
    )


class ContinuationPool(PagedStateStorage):
    @classmethod
    def for_model(
        cls,
        config,
        slots,
        pages,
        device,
        *,
        mtp=True,
        speculative_tokens=0,
        rotary_offset=False,
        **kwargs,
    ):
        return cls.allocate(
            continuation_layout(
                config,
                mtp=mtp,
                speculative_tokens=speculative_tokens,
                rotary_offset=rotary_offset,
            ),
            page_specs(config, mtp=mtp),
            slots,
            pages,
            device,
            **kwargs,
        )

    def recurrent(self, layer: int) -> DeltaState:
        return DeltaState(
            self.reserved_region(f"layer{layer}.convolution"),
            self.reserved_region(f"layer{layer}.recurrent"),
        )

    def attention(self, layer: int) -> torch.Tensor:
        return self.pages[f"layer{layer}.latent"].view(torch.bfloat16).view(-1, PAGE_TOKENS, RECORD_DIM)

    def draft_attention(self) -> torch.Tensor:
        return self.pages["draft.latent"].view(torch.bfloat16).view(-1, PAGE_TOKENS, RECORD_DIM)

    def draft_target_hidden(self) -> torch.Tensor:
        return self.reserved_region("draft.target_hidden")

    def partial_pages(self):
        """Target MLA partial-page planes synchronized through ``page_tables``."""
        return tuple((name, tensor) for name, tensor in self.pages.items() if name != "draft.latent")

    def draft_pages(self):
        """One-token-lagged MTP draft page plane synchronized through ``draft_page_tables``."""
        if "draft.latent" not in self.pages:
            return ()
        return (("draft.latent", self.pages["draft.latent"]),)
