# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared vision-encoder helpers for Blackwell math-mode selection and varlen FlashAttention."""

from __future__ import annotations

import torch


def vision_math_mode(device: torch.device | int | str) -> str:
    """Use canonical patch convolution and Torch FlashAttention on SM120."""
    return "torch_flash" if torch.cuda.get_device_capability(device) == (12, 0) else "flash"


class TorchFlashAttention:
    """Variable-length FlashAttention wrapper over ``torch.nn.attention.varlen.varlen_attn_out``."""

    scratch = None

    def __init__(self, capacity: int):
        self.capacity = capacity

    def functional(self, *, session):
        del session
        return self

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        from torch.nn.attention.varlen import varlen_attn_out

        # Inactive rows are outside the cumulative lengths and need defined output.
        output = torch.zeros_like(query)
        varlen_attn_out(output, query, key, value, cu_seqlens, cu_seqlens, self.capacity, self.capacity)
        return output
