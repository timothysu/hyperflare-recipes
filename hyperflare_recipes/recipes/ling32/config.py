# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Validated Ling-3.2-Air checkpoint geometry, layer schedule, and quantization metadata."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

from ignition.preparation.quantization import quantized_layers

PAGE_TOKENS = 128
RECORD_DIM = 576  # 512 compressed KV latent + 64 RoPE key dimensions
RECORD_BYTES = RECORD_DIM * 2  # BF16 paged latent storage
KDA_HEAD_DIM = 128
KDA_CONV_KERNEL = 4
IMAGE_TOKEN_ID = 157157
VIDEO_TOKEN_ID = 156909
MROPE_SECTION = (8, 12, 12)


def mrope_axis_indices(mrope_section: tuple[int, int, int] = (8, 12, 12)) -> tuple[int, ...]:
    """Map the 32 interleaved RoPE pairs to Bailing ``apply_video_rope`` (T, H, W) axes.

    Bailing reverses ``[T, H + W]`` to place ``H + W`` alternating pairs first
    (``H = 1, W = 2``) followed by ``T = 0`` block pairs.
    """
    t_sec, h_sec, w_sec = mrope_section
    spatial = h_sec + w_sec
    if t_sec <= 0 or h_sec <= 0 or w_sec <= 0 or t_sec + spatial != 32:
        raise ValueError(f"Ling-3.2-Air requires positive mrope_section summing to 32, got {mrope_section!r}")
    return tuple((1 if j % 2 == 0 else 2) for j in range(spatial)) + (0,) * t_sec


@dataclass(frozen=True)
class VisionConfig:
    """Validated ``qwen3_moe_vit`` vision tower geometry for Ling-3.2-Air."""

    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    num_position_embeddings: int = 2304
    out_hidden_size: int = 2560

    @classmethod
    def from_raw(cls, raw: dict, *, out_hidden_size: int) -> "VisionConfig":
        required = {
            "model_type": "qwen3_moe_vit",
            "depth": 27,
            "hidden_size": 1152,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 4304,
            "num_heads": 16,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 2304,
            "disable_merger_proj": True,
        }
        for key, expected in required.items():
            actual = raw.get(key)
            if actual != expected or (type(expected) is bool and type(actual) is not bool):
                raise ValueError(f"Ling-3.2-Air requires vision_config.{key} == {expected!r}, got {actual!r}")
        return cls(out_hidden_size=out_hidden_size)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size

    @property
    def merged_hidden_size(self) -> int:
        return self.hidden_size * (self.spatial_merge_size ** 2)


@dataclass(frozen=True)
class ModelConfig:
    """Validated configuration of a Ling-3.2-Air (``bailing_moe_v3_vl``) checkpoint."""

    raw: dict
    text: dict
    quantization: dict[str, dict]
    layer_types: tuple[str, ...]
    vision: VisionConfig | None = None
    mrope_section: tuple[int, int, int] = (8, 12, 12)
    image_token_id: int = IMAGE_TOKEN_ID
    video_token_id: int = VIDEO_TOKEN_ID

    @classmethod
    def load(cls, snapshot: Path) -> "ModelConfig":
        raw = json.loads((Path(snapshot) / "config.json").read_text())
        if raw.get("model_type") != "bailing_moe_v3_vl":
            raise ValueError(f"Ling-3.2-Air recipe requires model_type 'bailing_moe_v3_vl', got {raw.get('model_type')!r}")
        if raw.get("tie_word_embeddings", False):
            raise ValueError("Ling-3.2-Air requires untied word embeddings and lm_head")
        text = raw.get("text_config")
        if not isinstance(text, dict):
            raise ValueError("Ling-3.2-Air config.json requires a text_config object")

        required = {
            "num_hidden_layers": 42,
            "hidden_size": 2560,
            "intermediate_size": 6144,
            "first_k_dense_replace": 2,
            "moe_intermediate_size": 768,
            "moe_shared_expert_intermediate_size": 768,
            "num_experts": 512,
            "num_experts_per_tok": 8,
            "n_group": 8,
            "topk_group": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "head_dim": 128,
            "q_lora_rank": None,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "layer_group_size": 6,
            "short_conv_kernel_size": 4,
            "vocab_size": 157184,
            "score_function": "sigmoid",
            "moe_router_enable_expert_bias": True,
            "norm_topk_prob": True,
            "gated_attention_proj_granularity_type": "head_wise",
            "no_kda_lora": True,
            "kda_safe_gate": True,
            "mtp_use_kda": False,
            "use_qk_norm": True,
        }
        for key, expected in required.items():
            actual = text.get(key)
            if actual != expected or (type(expected) is bool and type(actual) is not bool):
                raise ValueError(f"Ling-3.2-Air requires text_config.{key} == {expected!r}, got {actual!r}")

        # Validate switches assumed by the decoder when explicitly set in text_config.
        if text.get("rope_interleave") is False:
            raise ValueError("Ling-3.2-Air requires interleaved RoPE (rope_interleave != False)")
        if text.get("use_qkv_bias") is True:
            raise ValueError("Ling-3.2-Air does not support QKV projection bias")
        if text.get("hidden_act") not in (None, "silu"):
            raise ValueError(f"Ling-3.2-Air requires silu activation, got {text.get('hidden_act')!r}")
        if text.get("num_shared_experts") not in (None, 1):
            raise ValueError(f"Ling-3.2-Air requires one shared expert, got {text.get('num_shared_experts')!r}")

        if text.get("num_nextn_predict_layers", 0) not in (0, 1):
            raise ValueError("Ling-3.2-Air supports zero or one MTP layer")
        if float(text.get("kda_lower_bound", -5.0)) != -5.0:
            raise ValueError("Ling-3.2-Air requires kda_lower_bound == -5.0")
        for positive_key in ("routed_scaling_factor", "rope_theta"):
            raw_val = text.get(positive_key)
            if type(raw_val) not in (int, float) or not math.isfinite(raw_val) or float(raw_val) <= 0.0:
                raise ValueError(f"Ling-3.2-Air requires positive finite {positive_key}, got {raw_val!r}")
        if text.get("rope_scaling") is not None:
            raise ValueError("Ling-3.2-Air recipe expects default unscaled RoPE")

        num_layers = text["num_hidden_layers"]
        group_size = text["layer_group_size"]
        layer_types = tuple(
            "mla"
            if (idx + 1) % group_size == 0 or idx >= (num_layers // group_size) * group_size
            else "kda"
            for idx in range(num_layers)
        )
        # Upstream inclusionAI/Ling-3.2-Air main defines mrope_section = [8, 12, 12] in text_config,
        # whereas upstream revision 1613b1a4 and the pinned checkpoint
        # timbuksu/Ling-3.2-Air-NVFP4@a95def604aa78f3111e6eac10a65f2651d5da828 carry it at the top
        # level of config.json. Accept either location (rejecting disagreement or absence) and
        # strictly require [8, 12, 12] since mrope_axis_indices alternates equal H/W spatial pairs.
        text_mrope = text.get("mrope_section")
        top_mrope = raw.get("mrope_section")
        if text_mrope is not None and top_mrope is not None and list(text_mrope) != list(top_mrope):
            raise ValueError(
                f"Ling-3.2-Air config has conflicting mrope_section values: "
                f"text_config={text_mrope!r} vs top-level={top_mrope!r}"
            )
        mrope_raw = text_mrope if text_mrope is not None else top_mrope
        if (
            not isinstance(mrope_raw, (list, tuple))
            or len(mrope_raw) != 3
            or any(type(v) is not int for v in mrope_raw)
            or tuple(mrope_raw) != MROPE_SECTION
        ):
            raise ValueError(
                f"Ling-3.2-Air requires mrope_section == {list(MROPE_SECTION)} "
                f"(in text_config or top-level config.json), got {mrope_raw!r}"
            )
        mrope_section = (mrope_raw[0], mrope_raw[1], mrope_raw[2])
        mrope_axis_indices(mrope_section)
        image_token_id = raw.get("image_token_id", IMAGE_TOKEN_ID)
        video_token_id = raw.get("video_token_id", VIDEO_TOKEN_ID)
        if image_token_id != IMAGE_TOKEN_ID or video_token_id != VIDEO_TOKEN_ID:
            raise ValueError(
                f"Ling-3.2-Air requires image_token_id={IMAGE_TOKEN_ID} and video_token_id={VIDEO_TOKEN_ID}"
            )
        vision_raw = raw.get("vision_config")
        vision = (
            VisionConfig.from_raw(vision_raw, out_hidden_size=int(text["hidden_size"]))
            if isinstance(vision_raw, dict)
            else None
        )
        quant = quantized_layers(raw)
        return cls(
            raw=raw,
            text=text,
            quantization=quant,
            layer_types=layer_types,
            vision=vision,
            mrope_section=mrope_section,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
        )

    @property
    def layers(self) -> int:
        return int(self.text["num_hidden_layers"])

    @property
    def num_mtp_layers(self) -> int:
        return int(self.text.get("num_nextn_predict_layers", 0))

    @property
    def hidden(self) -> int:
        return int(self.text["hidden_size"])

    @property
    def vocabulary(self) -> int:
        return int(self.text["vocab_size"])

    @property
    def heads(self) -> int:
        return int(self.text["num_attention_heads"])

    @property
    def eps(self) -> float:
        return float(self.text["rms_norm_eps"])

    @property
    def rope_theta(self) -> float:
        return float(self.text["rope_theta"])

    @property
    def kda_lower_bound(self) -> float:
        return float(self.text.get("kda_lower_bound", -5.0))

    def is_moe(self, layer: int) -> bool:
        return layer >= int(self.text["first_k_dense_replace"])

    def expert_swiglu_limit(self, layer: int) -> float | None:
        limits = self.text.get("expert_swiglu_limit_list")
        if not limits or not 0 <= layer < len(limits):
            return None
        val = limits[layer]
        return float(val) if val else None

    def shared_swiglu_limit(self, layer: int) -> float | None:
        limits = self.text.get("share_expert_swiglu_limit_list")
        if not limits or not 0 <= layer < len(limits):
            return None
        val = limits[layer]
        return float(val) if val else None

    def validate_shard(self, shard) -> None:
        if shard.world_size != 1:
            raise ValueError(f"Ling-3.2-Air recipe supports TP=1, got world_size={shard.world_size}")
