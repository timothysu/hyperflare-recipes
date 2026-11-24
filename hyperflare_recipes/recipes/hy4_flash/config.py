# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Validated Hy4-flash text, Engram, DFly2, and SigLIP2 vision configuration."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from ignition.preparation.quantization import linear_format, quantized_layers

VOCABULARY = 120832
HIDDEN_SIZE = 3072
HC_MULT = 4
Q_LORA_RANK = 1024
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 192
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 256
SPARSE_HEADS = 32
SWA_HEADS = 64
SWA_WINDOW = 512
INDEX_HEADS = 16
INDEX_HEAD_DIM = 128
INDEX_KPOOL = 4
INDEX_TOPK = 2048
GROUP_TOPK = INDEX_TOPK // INDEX_KPOOL
ENGRAM_HEADS = 16
ENGRAM_HEAD_DIM = 96
ENGRAM_WIDTH = ENGRAM_HEADS * ENGRAM_HEAD_DIM
ENGRAM_MAX_NGRAM = 3
ENGRAM_HEADS_PER_NGRAM = 8
IMAGE_START_TOKEN = 120118
IMAGE_END_TOKEN = 120119
IMAGE_TOKEN = 120120
IMAGE_NEWLINE_TOKEN = 120121
VIDEO_TOKEN = 120124


@dataclass(frozen=True)
class DFlyConfig:
    layers: int
    intermediate_size: int
    hidden_correction_intermediate_size: int
    mask_token_id: int
    target_kv_layers: tuple[int, ...]
    target_index_layers: tuple[int, ...]
    target_index_topk: int
    heads: int = SPARSE_HEADS


@dataclass(frozen=True)
class VisionConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    patch_size: int
    max_image_size: int
    spatial_merge_size: int
    out_hidden_size: int
    layer_norm_eps: float
    rms_norm_eps: float


@dataclass(frozen=True)
class ModelConfig:
    raw: dict
    text: dict
    sparse_layers: tuple[int, ...]
    swa_layers: tuple[int, ...]
    engram_layers: tuple[int, ...]  # 0-based layer indices: (1, 11, 21, 31)
    engram_head_vocab_sizes: dict[int, tuple[int, ...]]  # keyed by 0-based layer index
    rope_theta_indexed: float
    rope_theta_sliding: float
    mrope_section: tuple[int, int, int]
    quantized: dict[str, dict] | None
    draft: DFlyConfig | None
    vision: VisionConfig | None

    @property
    def layers(self) -> int:
        return int(self.text["num_hidden_layers"])

    def is_indexed(self, layer: int) -> bool:
        return layer in self.sparse_layers

    def is_sparse_mlp(self, layer: int) -> bool:
        types = self.text.get("mlp_layer_types")
        return (layer > 0) if types is None else (types[layer] == "sparse")

    def linear_format(self, prefix: str) -> str:
        if self.quantized is None or prefix not in self.quantized:
            return "bfloat16"
        return linear_format(self.quantized, prefix)

    @classmethod
    def load(cls, metadata_path: Path, *, manifest=None) -> ModelConfig:
        raw = json.loads((Path(metadata_path) / "config.json").read_text())
        if raw.get("model_type") != "hy_v4_flash":
            raise ValueError(f"expected model_type='hy_v4_flash', got {raw.get('model_type')!r}")
        text = raw.get("text_config", raw)

        expected_dims = (
            ("vocab_size", VOCABULARY),
            ("hidden_size", HIDDEN_SIZE),
            ("num_hidden_layers", 42),
            ("hc_mult", HC_MULT),
            ("q_lora_rank", Q_LORA_RANK),
            ("kv_lora_rank", KV_LORA_RANK),
            ("qk_nope_head_dim", QK_NOPE_HEAD_DIM),
            ("qk_rope_head_dim", QK_ROPE_HEAD_DIM),
            ("v_head_dim", V_HEAD_DIM),
            ("num_attention_heads", SPARSE_HEADS),
            ("n_routed_experts", 512),
            ("num_experts_per_tok", 16),
            ("routed_expert_hidden_size", 1536),
            ("moe_intermediate_size", 1536),
            ("intermediate_size", 13824),
        )
        for key, expected in expected_dims:
            if int(text.get(key, -1)) != expected:
                raise ValueError(f"Hy4-flash config {key}={text.get(key)!r} differs from {expected}")

        if int(text.get("swa_num_attention_heads", -1)) != SWA_HEADS:
            raise ValueError(f"Hy4-flash requires swa_num_attention_heads={SWA_HEADS}")
        if int(text.get("sliding_window", -1)) != SWA_WINDOW:
            raise ValueError(f"Hy4-flash requires sliding_window={SWA_WINDOW}")
        for required_key in (
            "rms_norm_eps",
            "hc_eps",
            "hc_magnitude",
            "routed_scaling_factor",
            "norm_topk_prob",
            "swiglu_limit",
            "eos_token_id",
            "engram_pad_id",
            "engram_seed",
        ):
            if required_key not in text:
                raise ValueError(f"Hy4-flash text_config is missing required key {required_key!r}")
        for pos_key in ("rms_norm_eps", "hc_eps", "hc_magnitude", "swiglu_limit"):
            val = float(text[pos_key])
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"Hy4-flash text_config {pos_key} must be positive and finite, got {val!r}")
        if not math.isfinite(float(text["routed_scaling_factor"])):
            raise ValueError("Hy4-flash text_config routed_scaling_factor must be finite")

        layer_types = text.get("layer_types")
        if not isinstance(layer_types, list) or len(layer_types) != 42:
            raise ValueError("Hy4-flash text_config requires a 42-entry layer_types list")
        sparse_layers = tuple(
            i for i, t in enumerate(layer_types) if t in ("indexed_attention", "deepseek_sparse_attention")
        )
        swa_layers = tuple(i for i, t in enumerate(layer_types) if t == "sliding_attention")
        if len(sparse_layers) != 12 or len(swa_layers) != 30:
            raise ValueError(
                f"Hy4-flash requires 12 indexed_attention and 30 sliding_attention layers, "
                f"got {len(sparse_layers)} indexed and {len(swa_layers)} sliding"
            )

        engram_ids_1based = tuple(int(x) for x in (text.get("engram_layer_ids") or ()))
        engram_layers = tuple(x - 1 for x in engram_ids_1based)
        head_vocab_sizes: dict[int, tuple[int, ...]] = {}
        raw_hvs = text.get("engram_layer_head_vocab_sizes") or {}
        for lid_1 in engram_ids_1based:
            orders = raw_hvs.get(str(lid_1))
            if not isinstance(orders, list) or len(orders) != ENGRAM_MAX_NGRAM - 1:
                raise ValueError(f"invalid engram_layer_head_vocab_sizes for layer {lid_1}")
            flat = tuple(int(p) for order in orders for p in order)
            if len(flat) != ENGRAM_HEADS:
                raise ValueError(f"Hy4-flash Engram requires {ENGRAM_HEADS} hash heads per layer")
            head_vocab_sizes[lid_1 - 1] = flat

        rp = text.get("rope_parameters")
        if not isinstance(rp, dict):
            raise ValueError("Hy4-flash text_config requires rope_parameters")
        idx_rp = rp.get("indexed_attention") or rp.get("deepseek_sparse_attention")
        swa_rp = rp.get("sliding_attention")
        if not isinstance(idx_rp, dict) or not isinstance(swa_rp, dict):
            raise ValueError("Hy4-flash rope_parameters must declare indexed_attention and sliding_attention")
        if "rope_theta" not in idx_rp or "rope_theta" not in swa_rp or "mrope_section" not in idx_rp:
            raise ValueError("Hy4-flash rope_parameters must declare rope_theta and mrope_section")
        rope_theta_indexed = float(idx_rp["rope_theta"])
        rope_theta_sliding = float(swa_rp["rope_theta"])
        if not (math.isfinite(rope_theta_indexed) and rope_theta_indexed > 0.0) or not (
            math.isfinite(rope_theta_sliding) and rope_theta_sliding > 0.0
        ):
            raise ValueError("Hy4-flash rope_theta values must be positive and finite")
        half = QK_ROPE_HEAD_DIM // 2
        raw_mrope = idx_rp["mrope_section"]
        if not isinstance(raw_mrope, (list, tuple)) or len(raw_mrope) != 3 or any(
            type(v) is not int or v <= 0 for v in raw_mrope
        ):
            raise ValueError(f"invalid mrope_section {raw_mrope!r}")
        mrope = (raw_mrope[0], raw_mrope[1], raw_mrope[2])
        if sum(mrope) != half or mrope[1] * 3 > half + 1 or mrope[2] * 3 > half:
            raise ValueError(
                f"invalid mrope_section {mrope!r}: must sum to {half} and fit interleaved axis bounds"
            )

        quantized = dict(quantized_layers(raw)) if "quantization_config" in raw else None

        draft = None
        dfly_layers = raw.get("dfly_num_hidden_layers")
        if dfly_layers is not None:
            if type(dfly_layers) is not int or dfly_layers != 5:
                raise ValueError(f"Hy4-flash DFly2 requires dfly_num_hidden_layers == 5, got {dfly_layers!r}")
            kv_targets = tuple(int(x) for x in raw["dfly_target_kv_layers"])
            idx_targets = raw["dfly_target_index_layer"]
            idx_targets = (
                (int(idx_targets),) * dfly_layers
                if isinstance(idx_targets, int)
                else tuple(int(x) for x in idx_targets)
            )
            if len(kv_targets) != dfly_layers or len(idx_targets) != dfly_layers:
                raise ValueError("DFly2 target KV/index layer counts must match dfly_num_hidden_layers")
            if any(layer not in sparse_layers for layer in kv_targets) or any(
                layer not in sparse_layers for layer in idx_targets
            ):
                raise ValueError("DFly2 target KV and index layers must be sparse indexed layers")
            dfly_inter = int(raw.get("dfly_intermediate_size") or 9216)
            if manifest is not None and "model.dfly_layers.0.mlp.gate_proj.weight" in manifest.tensors:
                dfly_inter = int(manifest.tensors["model.dfly_layers.0.mlp.gate_proj.weight"].shape[0])
            hc_inter = int(raw.get("dfly_hidden_correction_intermediate_size") or HIDDEN_SIZE)
            if manifest is not None and "model.dfly_layers.hidden_correction.gate_proj.weight" in manifest.tensors:
                hc_inter = int(manifest.tensors["model.dfly_layers.hidden_correction.gate_proj.weight"].shape[0])
            if "dfly_mask_token_id" not in raw or "dfly_target_index_topk" not in raw:
                raise ValueError("DFly2 config requires dfly_mask_token_id and dfly_target_index_topk")
            dfly_topk = raw["dfly_target_index_topk"]
            if type(dfly_topk) is not int or dfly_topk != INDEX_TOPK:
                raise ValueError(
                    f"Hy4-flash DFly2 requires dfly_target_index_topk == {INDEX_TOPK}, got {dfly_topk!r}"
                )
            draft = DFlyConfig(
                layers=dfly_layers,
                intermediate_size=dfly_inter,
                hidden_correction_intermediate_size=hc_inter,
                mask_token_id=int(raw["dfly_mask_token_id"]),
                target_kv_layers=kv_targets,
                target_index_layers=idx_targets,
                target_index_topk=dfly_topk,
            )

        vision = None
        vcfg = raw.get("vision_config")
        if isinstance(vcfg, dict):
            expected_vision = (
                ("hidden_size", 1152),
                ("intermediate_size", 4304),
                ("num_hidden_layers", 27),
                ("num_attention_heads", 16),
                ("patch_size", 16),
                ("max_image_size", 2048),
                ("spatial_merge_size", 2),
                ("out_hidden_size", HIDDEN_SIZE),
            )
            for key, expected in expected_vision:
                if int(vcfg.get(key, -1)) != expected:
                    raise ValueError(f"Hy4-flash vision_config {key}={vcfg.get(key)!r} differs from {expected}")
            if "rms_norm_eps" not in vcfg:
                raise ValueError("Hy4-flash vision_config requires rms_norm_eps")
            vision = VisionConfig(
                hidden_size=1152,
                intermediate_size=4304,
                num_hidden_layers=27,
                num_attention_heads=16,
                patch_size=16,
                max_image_size=2048,
                spatial_merge_size=2,
                out_hidden_size=HIDDEN_SIZE,
                layer_norm_eps=float(vcfg.get("layer_norm_eps", 1e-6)),
                rms_norm_eps=float(vcfg["rms_norm_eps"]),
            )

        return cls(
            raw=raw,
            text=text,
            sparse_layers=sparse_layers,
            swa_layers=swa_layers,
            engram_layers=engram_layers,
            engram_head_vocab_sizes=head_vocab_sizes,
            rope_theta_indexed=rope_theta_indexed,
            rope_theta_sliding=rope_theta_sliding,
            mrope_section=mrope,
            quantized=quantized,
            draft=draft,
            vision=vision,
        )
