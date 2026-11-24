# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Four-stream independent Hyper-Connections (iHC) in forced FP32 arithmetic."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ihc_pre(
    streams: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    hc_magnitude: float,
    mixes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute post-gate weights and stream-collapsed branch input.

    Matches ``HYV4FlashHyperConnection.forward``: unweighted FP32 RMS normalization
    of the flattened 4-stream vector, linear mix projection, sigmoid pre/post gates
    (no Sinkhorn/comb stream mixing), and BF16 collapsed branch output. When
    ``mixes`` is precomputed (via a prepared BF16xFP32->FP32 projection), the
    redundant FP32 ``F.linear`` is skipped.
    """
    hc_mult = streams.shape[1]
    flat = streams.flatten(1).float()
    inv_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + rms_eps)
    raw_mixes = F.linear(flat, fn.float()) if mixes is None else mixes.float()
    scaled_mixes = raw_mixes * inv_rms
    scale_vec = scale.float().repeat_interleave(hc_mult)
    sig = torch.sigmoid(scaled_mixes * scale_vec + base.float())
    pre_sig, post_sig = sig.split(hc_mult, dim=-1)
    pre = pre_sig + hc_eps
    post = hc_magnitude * post_sig + hc_eps
    collapsed = (pre.unsqueeze(-1) * streams.float()).sum(dim=1).to(streams.dtype)
    return post, collapsed


def ihc_post(
    residual: torch.Tensor,
    branch: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    """Add post-gated branch output back into the 4-stream residual in FP32."""
    return (post.float().unsqueeze(-1) * branch.float().unsqueeze(1) + residual.float()).to(residual.dtype)


def ihc_head(
    streams: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    mixes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Collapse the 4 hidden streams before the final shared RMSNorm."""
    flat = streams.flatten(1).float()
    inv_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + rms_eps)
    raw_mixes = F.linear(flat, fn.float()) if mixes is None else mixes.float()
    scaled_mixes = raw_mixes * inv_rms
    pre = torch.sigmoid(scaled_mixes * scale.float() + base.float()) + hc_eps
    return (pre.unsqueeze(-1) * streams.float()).sum(dim=1).to(streams.dtype)
