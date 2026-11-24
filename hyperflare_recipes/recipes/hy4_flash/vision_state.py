# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Per-request image feature arena, vision graph wrapper, and image/newline injection."""

from __future__ import annotations

import torch

from ignition.components.sampling import RUNNING
from ignition.components.vision_delivery import PreparedFeaturePublication
from ignition.components.vision_state import PrefillVisionInputs
from ignition.preparation.persistent import persistent_copy

from .config import HIDDEN_SIZE, IMAGE_NEWLINE_TOKEN, IMAGE_TOKEN
from .vision import VISION_POS_GRID


class VisionState:
    """Generation-scoped vision feature arena plus learned row-newline embedding."""

    def __init__(self, image_newline: torch.Tensor, inputs, prefill: PrefillVisionInputs, requests,
                 *, features: int, image_slots: int | None = None, arena: torch.Tensor | None = None):
        from ignition.components.capacity import image_capacity

        self.inputs = inputs
        self.prefill = prefill
        self.requests = requests
        self.feature_slot = prefill.vision_feature_slot
        self.image_newline = persistent_copy(image_newline.to(torch.bfloat16).reshape(1, HIDDEN_SIZE))
        device = inputs.token_ids.device
        shape = (image_capacity(requests.capacity, image_slots), features, HIDDEN_SIZE)
        if arena is None:
            arena = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        elif arena.shape != shape or arena.dtype != torch.bfloat16 or arena.device != device:
            raise ValueError("Hy4-flash vision feature arena shape or dtype mismatch")
        self.arena = arena
        self.generation = torch.zeros(requests.capacity, dtype=torch.int64, device=device)
        self.count = torch.zeros(requests.capacity, dtype=torch.int32, device=device)

    def validate_prefill(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        b = self.inputs
        slot = b.row_slot_ids[:1].long().clamp(0, self.requests.capacity - 1)
        index = self.prefill.vision_token_indices[: tokens.numel()]
        live = torch.arange(tokens.numel(), device=tokens.device) < lengths[:1]
        is_image = tokens == IMAGE_TOKEN
        is_newline = tokens == IMAGE_NEWLINE_TOKEN
        valid = (~is_image & (index == -1)) | (is_image & (index >= 0) & (index < self.count[slot]))
        valid &= ~(is_image | is_newline) | (self.generation[slot] == b.row_generations[:1])
        valid &= ~is_image | ((self.feature_slot >= 0) & (self.feature_slot < self.arena.shape[0]))
        return (~live | valid).all()

    def inject(self, metadata, embedding: torch.Tensor) -> None:
        slot = self.feature_slot.long().clamp(0, self.arena.shape[0] - 1)
        tokens = metadata.token_ids
        indices = self.prefill.vision_token_indices[: embedding.shape[0]].long().clamp(0, self.arena.shape[1] - 1)
        out = torch.where((tokens == IMAGE_TOKEN)[:, None], self.arena[slot, indices], embedding)
        out = torch.where((tokens == IMAGE_NEWLINE_TOKEN)[:, None], self.image_newline, out)
        embedding.copy_(out)


class PreparedVisionGeneration:
    """Validate host-uploaded image inputs, run SigLIP2 + perceive, and publish features."""

    def __init__(self, encoder, state: VisionState, inputs, completion, frame, sequence):
        self.encoder = encoder
        self.state = state
        self.inputs = inputs
        self.publish = PreparedFeaturePublication(state, inputs, completion, frame, sequence)

    def __call__(self):
        i, s = self.inputs, self.state
        r = s.requests
        cap = self.encoder.capacity
        fcap = cap // 4
        slot = i.vision_slot.long().clamp(0, r.capacity - 1)
        epoch = i.vision_generation
        patches = i.vision_num_patches
        features = i.vision_num_features
        offset = i.vision_feature_offset
        fslot = s.feature_slot

        active = (i.vision_slot != -1) | (patches != 0)
        slot_ok = (i.vision_slot >= 0) & (i.vision_slot < r.capacity) & (fslot >= 0) & (fslot < s.arena.shape[0])
        epoch_ok = (epoch > 0) & (r.generation[slot] == epoch) & (r.status[slot] == RUNNING)
        geom_ok = (
            (patches > 0)
            & (patches <= cap)
            & (patches % 4 == 0)
            & (features == patches // 4)
            & (i.vision_cu_seqlens[0] == 0)
            & (i.vision_cu_seqlens[1] == patches)
        )
        same_epoch = s.generation[slot] == epoch
        prev_count = torch.where(same_epoch, s.count[slot], 0)
        append_ok = (offset == prev_count) & (offset >= 0) & (offset.long() + features.long() <= s.arena.shape[1])

        p_idx = torch.arange(cap, device=slot.device)
        p_live = active & (p_idx < patches)
        f_idx = torch.arange(fcap, device=slot.device)
        f_live = active & (f_idx < features)

        interp_ok = (
            (i.vision_interpolation_indices[:cap] >= 0)
            & (i.vision_interpolation_indices[:cap] < VISION_POS_GRID)
            & (i.vision_interpolation_weights[:cap] >= 0.0)
            & (i.vision_interpolation_weights[:cap] <= 1.0)
        ).all(dim=1)
        merge_ok = (
            (i.vision_merge_indices[:fcap] >= 0) & (i.vision_merge_indices[:fcap] < patches)
        ).all(dim=1)
        pix_ok = torch.isfinite(i.vision_pixel_values[:cap]).all(dim=1)
        meta_ok = (~p_live | interp_ok).all() & (~f_live | merge_ok).all()
        pixels_ok = (~p_live | pix_ok).all()

        err = torch.where(active & ~slot_ok, 1, 0)
        err = err | torch.where(active & ~epoch_ok, 2, 0)
        err = err | torch.where(active & ~geom_ok, 4, 0)
        err = err | torch.where(active & ~append_ok, 8, 0)
        err = err | torch.where(active & ~meta_ok, 16, 0)
        err = err | torch.where(active & ~pixels_ok, 32, 0)

        ok = err == 0
        safe_patches = torch.where(ok, patches, 0)
        safe_features = torch.where(ok, features, 0)
        safe_cu = torch.cat((torch.zeros_like(safe_patches), safe_patches))
        safe_pixels = torch.where((p_idx < safe_patches)[:, None], i.vision_pixel_values[:cap], 0.0)
        safe_interp_idx = torch.where((p_idx < safe_patches)[:, None], i.vision_interpolation_indices[:cap], 0)
        safe_interp_w = torch.where((p_idx < safe_patches)[:, None], i.vision_interpolation_weights[:cap], 0.0)
        safe_merge = torch.where((f_idx < safe_features)[:, None], i.vision_merge_indices[:fcap], -1)

        output = self.encoder(
            safe_pixels,
            safe_interp_idx,
            safe_interp_w,
            safe_cu,
            safe_merge,
            safe_features,
        )
        out_ok = (~(f_idx < safe_features) | torch.isfinite(output).all(dim=1)).all()
        err = err | torch.where(active & ok & ~out_ok, 64, 0)
        self.publish(output, safe_features.to(torch.int32), err.to(torch.int32))
