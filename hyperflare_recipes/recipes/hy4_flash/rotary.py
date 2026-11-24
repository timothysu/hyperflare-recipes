# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Interleaved 3D M-RoPE for Hy4-flash indexed, sliding, indexer, and DFly2 layers."""

from __future__ import annotations

import torch

from ignition.preparation.persistent import persistent_copy

from .config import QK_ROPE_HEAD_DIM


class InterleavedMRoPE:
    """Precomputed inverse frequencies and interleaved pair rotation for 3D M-RoPE.

    Supports both 1-column text positions ``[T, 1]`` (broadcast across all three axes)
    and 3-column multimodal positions ``[T, 3]`` (time, height, width), interleaving
    the frequency planes according to ``mrope_section`` (``(12, 10, 10)`` for 64-dim
    RoPE). Matches ``HYV4FlashRotaryEmbedding`` and ``apply_rotary_pos_emb``.
    """

    def __init__(self, config, device: torch.device):
        half = QK_ROPE_HEAD_DIM // 2
        steps = torch.arange(0, QK_ROPE_HEAD_DIM, 2, dtype=torch.float32, device=device) / QK_ROPE_HEAD_DIM
        self.inv_freq_indexed = persistent_copy(1.0 / (config.rope_theta_indexed ** steps))
        self.inv_freq_sliding = persistent_copy(1.0 / (config.rope_theta_sliding ** steps))
        # Map each of the 32 frequency pairs to its axis in {0: T, 1: H, 2: W}.
        axes = [0] * half
        for axis, offset in ((1, 1), (2, 2)):
            for i in range(offset, config.mrope_section[axis] * 3, 3):
                axes[i] = axis
        self.axis_index = persistent_copy(torch.tensor(axes, dtype=torch.int64, device=device))

    def cos_sin(
        self,
        positions: torch.Tensor,
        *,
        indexed: bool,
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ``(cos, sin)`` of shape ``[T, 64]`` in ``dtype`` for interleaved pairs."""
        inv_freq = self.inv_freq_indexed if indexed else self.inv_freq_sliding
        if positions.ndim == 1:
            pos = positions.clamp_min(0).float()[:, None]
            angles = pos * inv_freq[None, :]
        elif positions.shape[1] == 1:
            pos = positions[:, 0].clamp_min(0).float()[:, None]
            angles = pos * inv_freq[None, :]
        else:
            pos3 = positions.clamp_min(0).float()  # [T, 3]
            per_pair_pos = pos3[:, self.axis_index]  # [T, 32]
            angles = per_pair_pos * inv_freq[None, :]
        cos = angles.cos().to(dtype).repeat_interleave(2, dim=-1)
        sin = angles.sin().to(dtype).repeat_interleave(2, dim=-1)
        return cos, sin

    @staticmethod
    def apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply interleaved pair rotation in FP32 and cast back to ``x.dtype``.

        ``x`` has shape ``[T, ..., 64]``; ``cos`` and ``sin`` have shape ``[T, 64]``.
        """
        view_shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (x.shape[-1],)
        c = cos.view(view_shape)
        s = sin.view(view_shape)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return ((x.float() * c) + (rot.float() * s)).to(x.dtype)
