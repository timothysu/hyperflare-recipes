# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared Ling-3.2-Air 27-layer ViT encoder, linear projector, and feature publication."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hyperflare_recipes.components.linear import UnquantizedProjection
from hyperflare_recipes.components.varlen_attention import PreparedVarlenAttention
from hyperflare_recipes.components.vision import TorchFlashAttention, vision_math_mode
from ignition.components.vision_delivery import PreparedFeaturePublication
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.rebuild_manifest import weight_container
from ignition.preparation.tensor_metadata import record_numeric_tensor

PATCH_CAPACITIES = (256, 1024, 4096, 16384)
PATCHES = 16384
FEATURES = 16384


@dataclass(frozen=True)
class VisionMetadata:
    interpolation_indices: torch.Tensor
    interpolation_weights: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    num_patches: torch.Tensor

    @classmethod
    def allocate(cls, patches: int, images: int, device: torch.device) -> VisionMetadata:
        if patches <= 0 or patches % 4 or images <= 0:
            raise ValueError("vision requires positive image capacity and a patch capacity divisible by four")

        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            zeros((patches, 4), torch.int64),
            zeros((patches, 4), torch.float32),
            zeros((patches, 2), torch.int64),
            zeros((images + 1,), torch.int32),
            zeros((1,), torch.int32),
        )


@dataclass(frozen=True)
class VisionInputs:
    vision_pixel_values: torch.Tensor
    vision_interpolation_indices: torch.Tensor
    vision_interpolation_weights: torch.Tensor
    vision_position_ids: torch.Tensor
    vision_cu_seqlens: torch.Tensor
    vision_num_patches: torch.Tensor
    vision_slot: torch.Tensor
    vision_generation: torch.Tensor
    vision_feature_offset: torch.Tensor

    @classmethod
    def allocate(cls, device: torch.device) -> VisionInputs:
        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            zeros((PATCHES, 1536), torch.float32),
            zeros((PATCHES, 4), torch.int64),
            zeros((PATCHES, 4), torch.float32),
            zeros((PATCHES, 2), torch.int64),
            zeros(2, torch.int32),
            zeros(1, torch.int32),
            torch.full((1,), -1, dtype=torch.int32, device=device),
            zeros(1, torch.int64),
            zeros(1, torch.int32),
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
    attention: torch.Tensor
    projected: torch.Tensor
    intermediate: torch.Tensor
    merged: torch.Tensor
    proj_hidden: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, patches: int, device: torch.device, out_hidden_size: int = 2560) -> VisionScratch:
        if patches <= 0 or patches % 4:
            raise ValueError("vision patch capacity must be positive and divisible by four")

        def zeros(*shape):
            return torch.zeros(shape, device=device, dtype=torch.bfloat16)

        query, key, value = (zeros(patches, 16, 72) for _ in range(3))
        return cls(
            zeros(patches, 1536),
            zeros(patches, 1152),
            zeros(patches, 3456),
            query,
            key,
            value,
            zeros(patches, 1152),
            zeros(patches, 1152),
            zeros(patches, 4304),
            zeros(patches // 4, 4608),
            zeros(patches // 4, out_hidden_size),
            zeros(patches // 4, out_hidden_size),
        )


@weight_container(
    (
        "norm1_weight",
        "norm1_bias",
        "qkv_weight",
        "qkv_bias",
        "proj_weight",
        "proj_bias",
        "norm2_weight",
        "norm2_bias",
        "fc1_weight",
        "fc1_bias",
        "fc2_weight",
        "fc2_bias",
    )
)
@dataclass(frozen=True)
class VisionBlockWeights:
    norm1_weight: torch.Tensor
    norm1_bias: torch.Tensor
    qkv_weight: torch.Tensor
    qkv_bias: torch.Tensor
    proj_weight: torch.Tensor
    proj_bias: torch.Tensor
    norm2_weight: torch.Tensor
    norm2_bias: torch.Tensor
    fc1_weight: torch.Tensor
    fc1_bias: torch.Tensor
    fc2_weight: torch.Tensor
    fc2_bias: torch.Tensor


@weight_container(
    (
        "patch_embed_weight",
        "patch_embed_bias",
        "pos_embed",
        "blocks",
        "merger_norm_weight",
        "merger_norm_bias",
        "linear_proj_0_weight",
        "linear_proj_0_bias",
        "linear_proj_2_weight",
        "linear_proj_2_bias",
    )
)
@dataclass(frozen=True)
class VisionWeights:
    patch_embed_weight: torch.Tensor
    patch_embed_bias: torch.Tensor
    pos_embed: torch.Tensor
    blocks: tuple[VisionBlockWeights, ...]
    merger_norm_weight: torch.Tensor
    merger_norm_bias: torch.Tensor
    linear_proj_0_weight: torch.Tensor
    linear_proj_0_bias: torch.Tensor
    linear_proj_2_weight: torch.Tensor
    linear_proj_2_bias: torch.Tensor


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_vision_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    qf, kf = q.float(), k.float()
    q_rot = (qf * cos + _rotate_half(qf) * sin).to(q.dtype).contiguous()
    k_rot = (kf * cos + _rotate_half(kf) * sin).to(k.dtype).contiguous()
    return q_rot, k_rot


class PreparedVision:
    """Prepared 27-layer Ling-3.2-Air ViT tower plus external 2-layer GELU projector."""

    def __init__(
        self,
        weights: VisionWeights,
        scratch: VisionScratch,
        metadata: VisionMetadata,
        *,
        session,
        name: str = "vision",
        attention_workspace=None,
        math_mode: str = "flash",
    ):
        if math_mode not in ("flash", "eager_bf16", "torch_flash"):
            raise ValueError(f"unknown vision math mode {math_mode!r}")
        capacity = scratch.hidden.shape[0]
        device = scratch.hidden.device
        if metadata.interpolation_indices.shape[0] != capacity:
            raise ValueError("vision metadata and scratch capacities differ")
        if metadata.num_patches.item() != 0 or bool(metadata.cu_seqlens.any().item()):
            raise ValueError("vision preparation requires inactive metadata")
        self.weights = weights
        self.capacity = capacity
        self.out_hidden_size = scratch.output.shape[1]

        record_numeric_tensor(weights.pos_embed)
        record_numeric_tensor(weights.merger_norm_weight)
        record_numeric_tensor(weights.merger_norm_bias)
        self.inv_freq = persistent_copy(
            1.0 / (10000.0 ** (torch.arange(0, 36, 2, dtype=torch.float32, device=device) / 36.0))
        )
        record_numeric_tensor(self.inv_freq)

        self.patch_embed = UnquantizedProjection(
            scratch.patches,
            weights.patch_embed_weight,
            scratch.hidden,
            session=session,
            name=f"{name}.patch_embed",
            bias=weights.patch_embed_bias,
        ).functional(session=session)

        if math_mode == "torch_flash":
            self.attention = TorchFlashAttention(capacity)
        else:
            self.attention = PreparedVarlenAttention(
                scratch.query,
                scratch.key,
                scratch.value,
                metadata.cu_seqlens,
                scratch.attention.view_as(scratch.query),
                session=session,
                workspace=attention_workspace,
                name=f"{name}.attn",
                math_mode=math_mode,
            ).functional(session=session)

        self.blocks = []
        for idx, blk in enumerate(weights.blocks):
            prefix = f"{name}.blocks.{idx}"
            record_numeric_tensor(blk.norm1_weight)
            record_numeric_tensor(blk.norm1_bias)
            record_numeric_tensor(blk.norm2_weight)
            record_numeric_tensor(blk.norm2_bias)
            qkv = UnquantizedProjection(
                scratch.hidden,
                blk.qkv_weight,
                scratch.qkv,
                session=session,
                name=f"{prefix}.attn.qkv",
                bias=blk.qkv_bias,
            ).functional(session=session)
            proj = UnquantizedProjection(
                scratch.attention,
                blk.proj_weight,
                scratch.projected,
                session=session,
                name=f"{prefix}.attn.proj",
                bias=blk.proj_bias,
            ).functional(session=session)
            fc1 = UnquantizedProjection(
                scratch.hidden,
                blk.fc1_weight,
                scratch.intermediate,
                session=session,
                name=f"{prefix}.mlp.fc1",
                bias=blk.fc1_bias,
            ).functional(session=session)
            fc2 = UnquantizedProjection(
                scratch.intermediate,
                blk.fc2_weight,
                scratch.projected,
                session=session,
                name=f"{prefix}.mlp.fc2",
                bias=blk.fc2_bias,
            ).functional(session=session)
            self.blocks.append((blk, qkv, proj, fc1, fc2))

        self.linear_proj_0 = UnquantizedProjection(
            scratch.merged,
            weights.linear_proj_0_weight,
            scratch.proj_hidden,
            session=session,
            name=f"{name}.linear_proj.0",
            bias=weights.linear_proj_0_bias,
        ).functional(session=session)
        self.linear_proj_2 = UnquantizedProjection(
            scratch.proj_hidden,
            weights.linear_proj_2_weight,
            scratch.output,
            session=session,
            name=f"{name}.linear_proj.2",
            bias=weights.linear_proj_2_bias,
        ).functional(session=session)

    def __call__(self, pixels: torch.Tensor, metadata: VisionMetadata) -> torch.Tensor:
        hidden = self.patch_embed(pixels.to(torch.bfloat16))
        pos = (
            F.embedding(metadata.interpolation_indices, self.weights.pos_embed)
            * metadata.interpolation_weights.to(hidden.dtype)[:, :, None]
        ).sum(1)
        hidden = hidden + pos
        freqs = (metadata.position_ids.float().unsqueeze(-1) * self.inv_freq).flatten(1)
        cos = freqs.cos()
        sin = freqs.sin()
        cos = torch.cat((cos, cos), dim=-1)[:, None, :]
        sin = torch.cat((sin, sin), dim=-1)[:, None, :]

        for blk, qkv_proj, out_proj, fc1, fc2 in self.blocks:
            norm1 = F.layer_norm(hidden, (1152,), blk.norm1_weight, blk.norm1_bias, eps=1e-6)
            qkv = qkv_proj(norm1).view(self.capacity, 3, 16, 72)
            q, k = apply_vision_rope(qkv[:, 0], qkv[:, 1], cos, sin)
            v = qkv[:, 2].contiguous()
            attn = self.attention(q, k, v, metadata.cu_seqlens).view(self.capacity, 1152)
            hidden = hidden + out_proj(attn)
            norm2 = F.layer_norm(hidden, (1152,), blk.norm2_weight, blk.norm2_bias, eps=1e-6)
            hidden = hidden + fc2(F.gelu(fc1(norm2), approximate="tanh"))

        merged = F.layer_norm(
            hidden,
            (1152,),
            self.weights.merger_norm_weight,
            self.weights.merger_norm_bias,
            eps=1e-6,
        ).view(self.capacity // 4, 4608)
        result = self.linear_proj_2(F.gelu(self.linear_proj_0(merged)))
        output_rows = torch.arange(self.capacity // 4, device=pixels.device)
        result.masked_fill_((output_rows >= metadata.num_patches // 4)[:, None], 0)
        return result


class PreparedVisionGeneration:
    """Validate uploaded patches, run ``PreparedVision``, and publish image features."""

    def __init__(self, encoder: PreparedVision, inputs: VisionInputs, state, completion, frame, sequence):
        self.encoder, self.inputs, self.state = encoder, inputs, state
        self.publish = PreparedFeaturePublication(state, inputs, completion, frame, sequence)

    def __call__(self):
        self.forward()

    def forward(self):
        e, i, s = self.encoder, self.inputs, self.state
        r, capacity = s.requests, e.capacity
        slot = i.vision_slot.long().clamp(0, r.capacity - 1)
        count, offset = i.vision_num_patches, i.vision_feature_offset
        features = count // 4
        current = (i.vision_slot >= 0) & (i.vision_slot < r.capacity)
        current &= (i.vision_generation > 0) & (i.vision_generation == r.generation[slot]) & (r.status[slot] == 1)
        valid = current & (count > 0) & (count <= capacity) & (count % 4 == 0)
        same_epoch = s.generation[slot] == i.vision_generation
        valid &= (offset >= 0) & (offset.long() + features.long() <= s.arena.shape[1])
        valid &= offset == torch.where(same_epoch, s.count[slot], 0)
        valid &= (i.vision_cu_seqlens[0] == 0) & (i.vision_cu_seqlens[1] == count)
        valid &= (s.feature_slot >= 0) & (s.feature_slot < s.arena.shape[0])
        live = torch.arange(capacity, device=count.device) < count
        pixels = i.vision_pixel_values[:capacity]
        indices = i.vision_interpolation_indices[:capacity]
        weights = i.vision_interpolation_weights[:capacity]
        positions = i.vision_position_ids[:capacity]
        valid &= (~live[:, None] | torch.isfinite(pixels)).all()
        valid &= (~live[:, None] | ((indices >= 0) & (indices < 2304))).all()
        valid &= (~live[:, None] | ((weights >= 0.0) & (weights <= 1.0))).all()
        valid &= (~live[:, None] | ((positions >= 0) & (positions < PATCHES))).all()
        errors = (~valid).to(torch.int32)
        m = VisionMetadata(
            torch.where(valid & live[:, None], indices, 0),
            torch.where(valid & live[:, None], weights, 0.0),
            torch.where(valid & live[:, None], positions, 0),
            torch.where(valid, i.vision_cu_seqlens, 0),
            torch.where(valid, count, 0),
        )
        output = e(torch.where(valid & live[:, None], pixels, 0.0).to(torch.bfloat16), m)
        live_features = m.num_patches // 4
        feature_live = torch.arange(capacity // 4, device=count.device) < live_features
        finite = (~feature_live[:, None] | torch.isfinite(output)).all()
        errors = errors | (~finite).to(torch.int32) * 64
        self.publish(output, live_features, errors)
        return output, errors
