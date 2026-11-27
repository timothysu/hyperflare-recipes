# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-H checkpoint transfers preserving each declared weight format."""

from dataclasses import dataclass, replace

import torch

from hyperflare_recipes.components.weights import DirectWeightLoader
from ignition.preparation.persistent import allocated_tensor_bytes, persistent_cat, persistent_copy
from ignition.preparation.quantization import expert_format

from .config import ModelConfig

PREFIX = "language_model.backbone"


def _ungated_nvfp4_expert_scales(sources, prefixes):
    w1, w2, a1, a2 = [], [], [], []
    for prefix in prefixes:
        values = {}
        for projection in ("up_proj", "down_proj"):
            for suffix in ("weight_scale_2", "input_scale"):
                name = f"{prefix}.{projection}.{suffix}"
                value = sources[name]
                if (value.device.type != "cpu" or value.dtype != torch.float32 or value.numel() != 1
                        or not bool((torch.isfinite(value) & (value > 0)).all())):
                    raise ValueError(f"{name}: expected a finite positive FP32 scalar")
                values[projection, suffix] = value.item()
        w1.append(values["up_proj", "weight_scale_2"])
        w2.append(values["down_proj", "weight_scale_2"])
        a1.append(values["up_proj", "input_scale"])
        a2.append(values["down_proj", "input_scale"])
    return (torch.tensor(w1, dtype=torch.float32), torch.tensor(w2, dtype=torch.float32),
            torch.tensor(a1, dtype=torch.float32).reciprocal(), torch.tensor(a2, dtype=torch.float32).reciprocal())


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    shard: object
    embedding: torch.Tensor
    norm: torch.Tensor
    head: object
    layers: tuple[dict, ...]
    allocated_bytes: int
    vision: object = None
    draft: object = None

    @classmethod
    def load(cls, manifest, device, *, shard, progress=None, draft_checkpoint=None):
        before = allocated_tensor_bytes(device)
        with NemotronWeightLoader(manifest, device, shard=shard, progress=progress) as loader:
            config = loader.config
            embedding, norm = loader.load_tensors((PREFIX + ".embeddings.weight", PREFIX + ".norm_f.weight"))
            if embedding.dtype != torch.bfloat16 or embedding.shape != (config.vocabulary, config.hidden):
                raise ValueError("Nemotron-H requires a BF16 embedding matching the configured vocabulary")
            head = loader.load_linear("language_model.lm_head", rows=shard.partition(config.vocabulary))
            layers = tuple(loader.load_layer(i) for i in range(len(config.layers)))
        from .vision import VisionWeights
        vision = VisionWeights.load(manifest, device, progress=progress)
        draft = None
        if draft_checkpoint is not None:
            from .draft_config import DraftConfig
            from .draft_weights import DraftWeights

            config = replace(config, draft=DraftConfig.load(draft_checkpoint, config))
            config.validate_shard(shard)
            draft = DraftWeights.load(draft_checkpoint, config.draft, device, shard=shard, progress=progress)
        return cls(config, shard, embedding, norm, head, layers, allocated_tensor_bytes(device) - before, vision=vision, draft=draft)


class NemotronWeightLoader(DirectWeightLoader):
    def __init__(self, manifest, device, *, shard, progress=None, prefixes=None):
        super().__init__(manifest, device, prefixes=prefixes, progress=progress)
        self.config, self.shard = ModelConfig.load(manifest.snapshot), shard
        self.config.validate_shard(shard)

    def _selected_names(self):
        return {name for name in super()._selected_names()
                if name.startswith(("language_model.backbone.", "language_model.lm_head."))}

    def load_layer(self, index):
        c, shard = self.config.text, self.shard
        prefix = f"{PREFIX}.layers.{index}"
        mixer = prefix + ".mixer"
        result = dict(norm=self.load_tensor(prefix + ".norm.weight"))
        kind = self.config.layers[index]
        if kind == "mamba":
            heads, channels = c["mamba_num_heads"], self.config.convolution_width
            width = self.config.mamba_width
            group_width = c["n_groups"] * c["ssm_state_size"]
            head_range = shard.partition(heads)
            width_range = shard.partition(width)
            if (self._sources[mixer + ".A_log"].shape != (heads,)
                    or self._sources[mixer + ".D"].shape != (heads,)
                    or self._sources[mixer + ".dt_bias"].shape != (heads,)
                    or self._sources[mixer + ".conv1d.weight"].shape != (channels, 1, c["conv_kernel"])
                    or self._sources[mixer + ".conv1d.bias"].shape != (channels,)
                    or self._sources[mixer + ".norm.weight"].shape != (width,)):
                raise ValueError(f"{mixer}: Mamba parameter geometry differs from configuration")
            head_slice = slice(*head_range)
            width_slice = slice(*width_range)
            a = self.load_tensor(mixer + ".A_log", dtype=torch.float32, extent=head_slice, temporary=True)
            d, bias, norm = self.load_tensors(
                (mixer + ".D", mixer + ".dt_bias", mixer + ".norm.weight"),
                extents=(head_slice, head_slice, width_slice),
            )
            if shard.world_size == 1:
                convolution, conv_bias = self.load_tensors((mixer + ".conv1d.weight", mixer + ".conv1d.bias"))
                convolution = convolution.squeeze(1)
                inp = self.load_linear(mixer + ".in_proj")
            else:
                conv_ranges = shard.fused_partitions((width, group_width, group_width))
                conv_slices = tuple(slice(*part) for part in conv_ranges)
                conv_parts = self.load_tensors((mixer + ".conv1d.weight",) * 3, extents=conv_slices, temporary=True)
                bias_parts = self.load_tensors((mixer + ".conv1d.bias",) * 3, extents=conv_slices, temporary=True)
                convolution = persistent_cat(conv_parts, dim=0).squeeze(1)
                conv_bias = persistent_cat(bias_parts, dim=0)
                in_ranges = shard.fused_partitions((width, width, group_width, group_width, heads))
                inp = self.load_linear_group((mixer + ".in_proj",) * 5, row_ranges=in_ranges)
            result.update(input=inp, output=self.load_linear(mixer + ".out_proj", columns=width_range),
                A=persistent_copy(-a.exp()), D=d, dt_bias=bias,
                convolution=convolution, convolution_bias=conv_bias, output_norm=norm)
        elif kind == "attention":
            dim = c["head_dim"]
            q_range = shard.partition(c["num_attention_heads"] * dim, alignment=dim)
            kv_heads = shard.replicated_partition(c["num_key_value_heads"])
            kv_range = (kv_heads[0] * dim, kv_heads[1] * dim)
            result.update(
                qkv=self.load_mixed_linear_group(
                    tuple(mixer + "." + name for name in ("q_proj", "k_proj", "v_proj")),
                    row_ranges=(q_range, kv_range, kv_range),
                ),
                output=self.load_linear(mixer + ".o_proj", columns=q_range),
            )
        else:
            block = self.config.block_config(index)
            router, correction = self.load_tensors((mixer + ".gate.weight", mixer + ".gate.e_score_correction_bias"))
            count = block["n_routed_experts"]
            inter = block["moe_intermediate_size"]
            latent = c["moe_latent_size"]
            if router.shape != (count, self.config.hidden) or correction.shape != (count,):
                raise ValueError(f"{mixer}: router geometry differs from block configuration")
            shared_range = shard.partition(c["moe_shared_expert_intermediate_size"], alignment=16)
            prefixes = tuple(f"{mixer}.experts.{i}" for i in range(count))
            fmt, mode = expert_format(self.quantization, f"{prefixes[0]}.up_proj")
            if fmt in ("iq2_xxs", "q8_0"):
                experts = self.load_block_quant_experts(
                    prefixes, hidden_size=latent, intermediate_size=inter,
                    intermediate_range=shard.partition(inter, alignment=256 if fmt == "iq2_xxs" else 32),
                )
            else:
                scales = _ungated_nvfp4_expert_scales(self._sources, prefixes) if fmt == "nvfp4" and mode == "a4" else None
                experts = self.load_fp4_experts(
                    prefixes, hidden_size=latent, intermediate_size=inter,
                    intermediate_range=shard.partition(inter, alignment=16 if fmt == "nvfp4" else 32),
                    format=fmt, activation_mode=mode, projection_names=("up_proj", "down_proj"),
                    activation="relu2", scale_vectors=scales,
                )
            result.update(router=router, correction=correction,
                latent_down=self.load_linear(mixer + ".fc1_latent_proj"),
                latent_up=self.load_linear(mixer + ".fc2_latent_proj"),
                shared_up=self.load_linear(mixer + ".shared_experts.up_proj", rows=shared_range),
                shared_down=self.load_linear(mixer + ".shared_experts.down_proj", columns=shared_range),
                experts=experts)
        return result
