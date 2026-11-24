# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured SigLIP2 27-layer vision tower and static 2x2 Conv2d perceive projector."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hyperflare_recipes.components.linear import UnquantizedProjection, prepare_projection
from hyperflare_recipes.components.varlen_attention import PreparedVarlenAttention

from .config import HIDDEN_SIZE

VISION_HIDDEN = 1152
VISION_INTERMEDIATE = 4304
VISION_HEADS = 16
VISION_HEAD_DIM = VISION_HIDDEN // VISION_HEADS  # 72
VISION_LAYERS = 27
VISION_PIXEL_WIDTH = 3 * 16 * 16  # 768
VISION_POS_GRID = 128 * 128  # 16384


@dataclass(frozen=True)
class VisionInputs:
    """Host-uploaded inputs matching Helios ``ImageLayoutKind::RowNewline`` packing."""

    vision_pixel_values: torch.Tensor
    vision_interpolation_indices: torch.Tensor
    vision_interpolation_weights: torch.Tensor
    # Uploaded by Helios's shared spatial_merge packer alongside interpolation weights.
    vision_position_ids: torch.Tensor
    vision_merge_indices: torch.Tensor
    vision_cu_seqlens: torch.Tensor
    vision_num_patches: torch.Tensor
    vision_num_features: torch.Tensor
    vision_slot: torch.Tensor
    vision_generation: torch.Tensor
    vision_feature_offset: torch.Tensor

    @classmethod
    def allocate(cls, device: torch.device, *, patches: int) -> VisionInputs:
        if patches <= 0 or patches % 4:
            raise ValueError("Hy4-flash vision requires positive patch capacity divisible by four")

        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            zeros((patches, VISION_PIXEL_WIDTH), torch.float32),
            zeros((patches, 4), torch.int64),
            zeros((patches, 4), torch.float32),
            zeros((patches, 2), torch.int64),
            zeros((patches // 4, 4), torch.int32),
            zeros(2),
            zeros(1),
            zeros(1),
            zeros(1) - 1,
            zeros(1, torch.int64),
            zeros(1),
        )

    def buffers(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class VisionScratch:
    patches: torch.Tensor
    hidden: torch.Tensor
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    projected: torch.Tensor
    intermediate: torch.Tensor
    merged: torch.Tensor
    perceive_hidden: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, patches: int, device: torch.device, *, out_hidden: int = HIDDEN_SIZE) -> VisionScratch:
        if patches <= 0 or patches % 4:
            raise ValueError("vision patch capacity must be positive and divisible by four")

        def zeros(*shape):
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)

        return cls(
            zeros(patches, VISION_PIXEL_WIDTH),
            zeros(patches, VISION_HIDDEN),
            zeros(patches, 3 * VISION_HIDDEN),
            zeros(patches, VISION_HEADS, VISION_HEAD_DIM),
            zeros(patches, VISION_HEADS, VISION_HEAD_DIM),
            zeros(patches, VISION_HEADS, VISION_HEAD_DIM),
            zeros(patches, VISION_HIDDEN),
            zeros(patches, VISION_HIDDEN),
            zeros(patches, VISION_INTERMEDIATE),
            zeros(patches // 4, 4 * VISION_HIDDEN),
            zeros(patches // 4, 4 * VISION_HIDDEN),
            zeros(patches // 4, out_hidden),
        )

    def rows(self, patches: int) -> VisionScratch:
        if patches <= 0 or patches % 4 or patches > self.patches.shape[0]:
            raise ValueError("vision patch slice exceeds allocated scratch capacity")
        features = patches // 4
        return VisionScratch(
            patches=self.patches[:patches],
            hidden=self.hidden[:patches],
            qkv=self.qkv[:patches],
            query=self.query[:patches],
            key=self.key[:patches],
            value=self.value[:patches],
            attended=self.attended[:patches],
            projected=self.projected[:patches],
            intermediate=self.intermediate[:patches],
            merged=self.merged[:features],
            perceive_hidden=self.perceive_hidden[:features],
            output=self.output[:features],
        )


class PreparedVision:
    """27-layer SigLIP2 ViT with bilinearly interpolated 128x128 pos-embed and 2x2 perceive merger."""

    def __init__(self, weights: dict[str, torch.Tensor], scratch: VisionScratch, inputs: VisionInputs,
                 *, config, session, workspace, name: str):
        self.capacity = scratch.patches.shape[0]
        self.ln_eps = config.vision.layer_norm_eps
        self.rms_eps = config.vision.rms_norm_eps
        device = scratch.patches.device
        cu = torch.zeros(2, dtype=torch.int32, device=device)

        # Patch embedding: Conv2d(3 -> 1152, k=16, s=16) flattened to Linear(768 -> 1152),
        # prepacked once by Hy4WeightLoader.load_vision so all vision graphs share tensors.
        self.patch_embed = UnquantizedProjection(
            scratch.patches, weights["patch_w"], scratch.hidden, bias=weights["patch_b"],
            session=session, name=f"{name}.patch_embed",
        ).functional(session=session)
        # Index 0 is the unused CLS slot; indices 1..16384 form the 128x128 spatial grid.
        self.pos_embed = weights["pos_embed"]

        self.blocks = []
        for i in range(VISION_LAYERS):
            b = weights["blocks"][i]
            qkv = prepare_projection(
                scratch.hidden,
                b["qkv_w"],
                scratch.qkv,
                bias=b["qkv_b"],
                session=session,
                name=f"{name}.l{i}.qkv",
                workspace=workspace,
                activation_mode="auto",
            ).functional(session=session)
            o_proj = prepare_projection(
                scratch.attended,
                b["o_w"],
                scratch.projected,
                bias=b["o_b"],
                session=session,
                name=f"{name}.l{i}.o_proj",
                workspace=workspace,
                activation_mode="auto",
            ).functional(session=session)
            fc1 = prepare_projection(
                scratch.hidden,
                b["fc1_w"],
                scratch.intermediate,
                bias=b["fc1_b"],
                session=session,
                name=f"{name}.l{i}.fc1",
                workspace=workspace,
                activation_mode="auto",
            ).functional(session=session)
            fc2 = prepare_projection(
                scratch.intermediate,
                b["fc2_w"],
                scratch.projected,
                bias=b["fc2_b"],
                session=session,
                name=f"{name}.l{i}.fc2",
                workspace=workspace,
                activation_mode="auto",
            ).functional(session=session)
            self.blocks.append((b["ln1_w"], b["ln1_b"], qkv, o_proj, b["ln2_w"], b["ln2_b"], fc1, fc2))

        self.attention = PreparedVarlenAttention(
            scratch.query,
            scratch.key,
            scratch.value,
            cu,
            scratch.attended.view(self.capacity, VISION_HEADS, VISION_HEAD_DIM),
            session=session,
            workspace=workspace,
            name=f"{name}.attention",
        ).functional(session=session)

        # Static 2x2 perceive merger: Conv2d(1152->4608, k=2, s=2) -> GELU -> Conv2d(4608->3072, k=1).
        self.before_rms = weights["before_rms"]
        self.perceive_conv0 = UnquantizedProjection(
            scratch.merged, weights["conv0_w"], scratch.perceive_hidden, bias=weights["conv0_b"],
            session=session, name=f"{name}.perceive0",
        ).functional(session=session)
        self.perceive_conv2 = UnquantizedProjection(
            scratch.perceive_hidden, weights["conv2_w"], scratch.output, bias=weights["conv2_b"],
            session=session, name=f"{name}.perceive2",
        ).functional(session=session)

    def __call__(
        self,
        pixels: torch.Tensor,
        interp_indices: torch.Tensor,
        interp_weights: torch.Tensor,
        cu_seqlens: torch.Tensor,
        merge_indices: torch.Tensor,
        num_features: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.patch_embed(pixels.to(torch.bfloat16))
        safe_idx = interp_indices.clamp(0, VISION_POS_GRID - 1)
        pos = (self.pos_embed[safe_idx].float() * interp_weights.unsqueeze(-1)).sum(dim=1).to(hidden.dtype)
        hidden = hidden + pos

        for ln1_w, ln1_b, qkv, o_proj, ln2_w, ln2_b, fc1, fc2 in self.blocks:
            normed = F.layer_norm(hidden, (VISION_HIDDEN,), ln1_w, ln1_b, eps=self.ln_eps)
            q, k, v = qkv(normed).view(-1, 3, VISION_HEADS, VISION_HEAD_DIM).unbind(1)
            attn = self.attention(q.contiguous(), k.contiguous(), v.contiguous(), cu_seqlens).reshape(-1, VISION_HIDDEN)
            hidden = hidden + o_proj(attn)

            normed2 = F.layer_norm(hidden, (VISION_HIDDEN,), ln2_w, ln2_b, eps=self.ln_eps)
            hidden = hidden + fc2(F.gelu(fc1(normed2)))

        # vit.perceive: before_rms -> 2x2 gather -> Conv2d(k=2, s=2) -> GELU -> Conv2d(k=1)
        hf = hidden.float()
        normed = (hf * torch.rsqrt(hf.square().mean(dim=-1, keepdim=True) + self.rms_eps)).to(hidden.dtype)
        normed = normed * self.before_rms
        idx = merge_indices.long()  # [F, 4]
        gathered = normed[idx.clamp(0, self.capacity - 1)]  # [F, 4, 1152]
        gathered = torch.where(idx[:, :, None] >= 0, gathered, 0)
        # Channel-major [C_in, 2, 2] matches Conv2d weight layout [C_out, C_in, 2, 2].
        merged = gathered.transpose(1, 2).reshape(idx.shape[0], 4 * VISION_HIDDEN).contiguous()
        out = self.perceive_conv2(F.gelu(self.perceive_conv0(merged)))
        rows = torch.arange(idx.shape[0], device=pixels.device)
        return torch.where((rows < num_features)[:, None], out, 0)
