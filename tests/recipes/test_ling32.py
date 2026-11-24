# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for Ling-3.2-Air configuration, state layout, routing, and contract."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from hyperflare_recipes.components.paged_batch import Metadata, RotaryBatch
from ignition.preparation import runtime_core_pb2 as p
from ignition.preparation.frontend import QwenPositionScheme, QwenResizeKernel, image_settings
from hyperflare_recipes.recipes import resolve
from hyperflare_recipes.recipes.ling32.config import (
    IMAGE_TOKEN_ID,
    ModelConfig,
    PAGE_TOKENS,
    RECORD_BYTES,
    VIDEO_TOKEN_ID,
    mrope_axis_indices,
)
from hyperflare_recipes.recipes.ling32.decoder import mrope_angles, rotary_frequencies
from hyperflare_recipes.recipes.ling32.frontend import (
    BAILING_QWEN_IMAGE_PARAMS,
    FRONTEND,
    VIDEO_PLACEHOLDER_TOKENS,
    VISION_PLACEHOLDER_TOKENS,
    frontend,
)
from hyperflare_recipes.recipes.ling32.moe import group_limited_route
from hyperflare_recipes.recipes.ling32.pool import ContinuationPool, continuation_layout, page_bytes, page_specs
from hyperflare_recipes.recipes.ling32.recipe import (
    RECIPE,
    VISION_BYTES,
    VISION_SLOT_BYTES,
    _PLAIN_IMAGE_STATE_ROW_BYTES,
    _PLAIN_STATE_ROW_BYTES,
    _SPECULATIVE_IMAGE_STATE_ROW_BYTES,
    _SPECULATIVE_STATE_ROW_BYTES,
)
from hyperflare_recipes.recipes.ling32.weights import Ling32WeightLoader


def _sample_config() -> dict:
    return {
        "model_type": "bailing_moe_v3_vl",
        "architectures": ["BailingMoeV3VLForConditionalGeneration"],
        "tie_word_embeddings": False,
        "image_token_id": IMAGE_TOKEN_ID,
        "video_token_id": VIDEO_TOKEN_ID,
        "vision_config": {
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
        },
        "text_config": {
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
            "rms_norm_eps": 1e-6,
            "rope_theta": 6000000,
            "mrope_section": [8, 12, 12],
            "eos_token_id": 156895,
            "score_function": "sigmoid",
            "moe_router_enable_expert_bias": True,
            "norm_topk_prob": True,
            "routed_scaling_factor": 2.5,
            "gated_attention_proj_granularity_type": "head_wise",
            "no_kda_lora": True,
            "kda_safe_gate": True,
            "kda_lower_bound": -5.0,
            "mtp_use_kda": False,
            "use_qk_norm": True,
            "num_nextn_predict_layers": 1,
            "expert_swiglu_limit_list": [0] * 35 + [4] * 7,
            "share_expert_swiglu_limit_list": [0] * 34 + [5] * 6 + [7] * 2,
        },
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {},
        },
    }


def test_config_layer_schedule_and_swiglu_limits(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_sample_config()))
    cfg = ModelConfig.load(tmp_path)
    assert cfg.layers == 42
    assert cfg.num_mtp_layers == 1
    mla_indices = [i for i, k in enumerate(cfg.layer_types) if k == "mla"]
    kda_indices = [i for i, k in enumerate(cfg.layer_types) if k == "kda"]
    assert mla_indices == [5, 11, 17, 23, 29, 35, 41]
    assert len(kda_indices) == 35
    assert not cfg.is_moe(0) and not cfg.is_moe(1)
    assert all(cfg.is_moe(i) for i in range(2, 43))
    assert cfg.expert_swiglu_limit(34) is None
    assert cfg.expert_swiglu_limit(35) == 4.0
    assert cfg.expert_swiglu_limit(41) == 4.0
    assert cfg.expert_swiglu_limit(42) is None
    assert cfg.shared_swiglu_limit(33) is None
    assert cfg.shared_swiglu_limit(34) == 5.0
    assert cfg.shared_swiglu_limit(40) == 7.0
    assert cfg.shared_swiglu_limit(42) is None


def test_continuation_layout_and_page_costs_match_declared_recipe(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_sample_config()))
    cfg = ModelConfig.load(tmp_path)

    plain = continuation_layout(cfg, mtp=False, speculative_tokens=0)
    spec = continuation_layout(cfg, mtp=True, speculative_tokens=RECIPE.max_draft_tokens)
    plain_img = continuation_layout(cfg, mtp=False, speculative_tokens=0, rotary_offset=True)
    spec_img = continuation_layout(cfg, mtp=True, speculative_tokens=RECIPE.max_draft_tokens, rotary_offset=True)
    assert plain.row_bytes == _PLAIN_STATE_ROW_BYTES
    assert spec.row_bytes == _SPECULATIVE_STATE_ROW_BYTES
    assert plain_img.row_bytes == _PLAIN_IMAGE_STATE_ROW_BYTES
    assert spec_img.row_bytes == _SPECULATIVE_IMAGE_STATE_ROW_BYTES
    assert plain_img.rotary_region == "rotary_offset"
    assert spec_img.rotary_region == "rotary_offset"
    assert RECIPE.plain_ranks[1].state_row_bytes == plain.row_bytes
    assert RECIPE.ranks[1].state_row_bytes == spec.row_bytes
    assert RECIPE.plain_ranks[1].image_state_row_bytes == plain_img.row_bytes
    assert RECIPE.ranks[1].image_state_row_bytes == spec_img.row_bytes
    with_slots = RECIPE.ranks[1].with_image_slots(4)
    assert with_slots.state_row_bytes == spec_img.row_bytes
    assert with_slots.graph_bytes == RECIPE.ranks[1].graph_bytes + VISION_BYTES + 4 * VISION_SLOT_BYTES

    assert page_bytes(cfg, mtp=False) == 7 * PAGE_TOKENS * RECORD_BYTES
    assert page_bytes(cfg, mtp=True) == 8 * PAGE_TOKENS * RECORD_BYTES
    assert RECIPE.plain_ranks[1].page_bytes == page_bytes(cfg, mtp=False)
    assert RECIPE.ranks[1].page_bytes == page_bytes(cfg, mtp=True)
    assert [name for name, _ in page_specs(cfg, mtp=True)] == [
        "layer5.latent",
        "layer11.latent",
        "layer17.latent",
        "layer23.latent",
        "layer29.latent",
        "layer35.latent",
        "layer41.latent",
        "draft.latent",
    ]


def test_group_limited_route_matches_bailing_reference():
    torch.manual_seed(7)
    tokens, hidden, num_experts = 17, 2560, 512
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    w = torch.randn(num_experts, hidden, dtype=torch.float32) * 0.02
    bias = torch.randn(num_experts, dtype=torch.float32) * 0.1

    ids, weights = group_limited_route(
        x, w, bias, n_group=8, topk_group=4, top_k=8, routed_scaling_factor=2.5
    )

    # Direct reference from BailingMoeV3Gate.forward
    logits = F.linear(x.float(), w.float())
    scores = torch.sigmoid(logits)
    biased = scores + bias
    group_scores = biased.view(tokens, 8, 64).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=4, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(-1).expand(tokens, 8, 64).reshape(tokens, num_experts)
    masked = biased.masked_fill(~score_mask.bool(), float("-inf"))
    _, ref_ids = torch.topk(masked, k=8, dim=-1)
    ref_sel = torch.gather(scores, dim=1, index=ref_ids)
    ref_weights = (ref_sel / (ref_sel.sum(dim=-1, keepdim=True) + 1e-20)) * 2.5

    assert ids.dtype == torch.int32
    assert weights.dtype == torch.float32
    assert torch.equal(ids.long(), ref_ids)
    torch.testing.assert_close(weights, ref_weights, rtol=0.0, atol=1e-6)


def test_recipe_registration_and_frontend_contract():
    resolved = resolve("ling-3.2-air-nvfp4")
    assert resolved.recipe.name == "ling32"
    assert resolved.recipe.sampling_defaults.as_dict() == {
        "seed": 0,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
    assert FRONTEND["assistant_marker"] == "<role>ASSISTANT</role>"
    assert FRONTEND["message_end_markers"] == ["<|role_end|>"]
    assert FRONTEND["rejected_tokens"] == list(VISION_PLACEHOLDER_TOKENS)
    mm_frontend = frontend(vision=True)
    assert mm_frontend["image_preparation"] == p.QWEN_IMAGE_PREPARATION
    assert mm_frontend["image_layout"] == p.QWEN_IMAGE_LAYOUT
    assert mm_frontend["position_dimensions"] == 3
    assert mm_frontend["image_tokens"] == {"feature_token": IMAGE_TOKEN_ID, "span": None}
    assert mm_frontend["rejected_tokens"] == list(VIDEO_PLACEHOLDER_TOKENS)
    patch = json.loads(
        (Path(__file__).resolve().parents[2] / "hyperflare_recipes/recipes/ling32/assets/chat_template.patch.json").read_text()
    )
    assert len(patch["insertions"]) == 2


def test_bailing_mrope_matches_upstream_apply_video_rope():
    freqs = rotary_frequencies(6_000_000.0, torch.device("cpu"))
    axes = torch.tensor(mrope_axis_indices((8, 12, 12)), dtype=torch.int64)
    # Include negative centered H/W coordinates on live tokens and an inactive padding row.
    pos3 = torch.tensor(
        [
            [0, 0, 0],
            [1, -2, -1],
            [1, 3, 2],
            [2, 2, 2],
            [-1, -1, -1],
        ],
        dtype=torch.int64,
    )
    logical = torch.tensor([0, 1, 2, 3, -1], dtype=torch.int64)
    got = mrope_angles(pos3, logical, freqs, axes)

    # Direct reference from BailingMoeV3VLRotaryEmbedding.apply_video_rope (first 32 pairs).
    ref_pos = pos3[:4].T.float()  # (3, seq)
    ref_freqs = ref_pos[:, :, None] * freqs[None, None, :]  # (3, seq, 32)
    ref_cos_3 = ref_freqs.cos()
    ref_sin_3 = ref_freqs.sin()
    sec_cos = [ref_cos_3[1 if j % 2 == 0 else 2, :, j : j + 1] for j in range(24)]
    sec_cos.append(ref_cos_3[0, :, 24:32])
    sec_sin = [ref_sin_3[1 if j % 2 == 0 else 2, :, j : j + 1] for j in range(24)]
    sec_sin.append(ref_sin_3[0, :, 24:32])
    expected = torch.stack((torch.cat(sec_cos, dim=-1), torch.cat(sec_sin, dim=-1)), dim=-1)

    torch.testing.assert_close(got[:4], expected, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(got[4, :, 0], torch.ones(32), rtol=0.0, atol=0.0)
    torch.testing.assert_close(got[4, :, 1], torch.zeros(32), rtol=0.0, atol=0.0)


def test_bailing_rotary_batch_commit_advances_on_temporal_axis(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_sample_config()))
    cfg = ModelConfig.load(tmp_path)
    small = ModelConfig(
        raw=cfg.raw,
        text={**cfg.text, "num_hidden_layers": 1, "layer_group_size": 1},
        quantization={},
        layer_types=("mla",),
        vision=cfg.vision,
    )
    pool = ContinuationPool.for_model(small, 4, 4, "cpu", mtp=False, rotary_offset=True)
    batch = RotaryBatch.__new__(RotaryBatch)
    batch.rotary_offset = pool.region("rotary_offset")
    batch.advance_axis = 0

    # Prompt: 2 text [0, 1] + 4x4 image at T=2 (H,W in [1..4]) split into two chunks.
    img_coords = [[2, 2 + y - 1, 2 + x - 1] for y in range(4) for x in range(4)]
    all_pos = [[0, 0, 0], [1, 1, 1], *img_coords, [3, 3, 3], [4, 4, 4]]

    # Chunk 0: 10 tokens (ends mid-image at T=2, H=3, W=4).
    c0_meta = Metadata.allocate(16, 1, 4, "cpu")
    c0_meta.set(
        starts=torch.tensor([0]),
        lengths=torch.tensor([10], dtype=torch.int32),
        state_rows=torch.tensor([1]),
        pages=torch.zeros((1, 4), dtype=torch.int32),
        tokens=torch.zeros((1, 10), dtype=torch.int64),
        live_rows=torch.tensor([1], dtype=torch.int32),
    )
    rope0 = torch.full((16, 3), -1, dtype=torch.int64)
    rope0[:10] = torch.tensor(all_pos[:10], dtype=torch.int64)
    batch.commit_rotary(Metadata(**{**c0_meta.__dict__, "rotary_positions": rope0}))
    # Temporal maximum is 2, so the next position is 3 and the saved offset is 3 - 10 = -7.
    assert pool.region("rotary_offset")[1, 0].item() == -7

    # Chunk 1: remaining 10 tokens (8 image tokens at T=2 + 2 text at T=3,4).
    c1_meta = Metadata.allocate(16, 1, 4, "cpu")
    c1_meta.set(
        starts=torch.tensor([10]),
        lengths=torch.tensor([10], dtype=torch.int32),
        state_rows=torch.tensor([1]),
        pages=torch.zeros((1, 4), dtype=torch.int32),
        tokens=torch.zeros((1, 10), dtype=torch.int64),
        live_rows=torch.tensor([1], dtype=torch.int32),
    )
    rope1 = torch.full((16, 3), -1, dtype=torch.int64)
    rope1[:10] = torch.tensor(all_pos[10:], dtype=torch.int64)
    batch.commit_rotary(Metadata(**{**c1_meta.__dict__, "rotary_positions": rope1}))
    # Total prompt = 20 tokens, T_max = 4 -> next decode rotary = 5 -> offset = 5 - 20 = -15.
    assert pool.region("rotary_offset")[1, 0].item() == -15


def test_bailing_image_settings_extraction():
    preproc = {
        "processor_class": "BailingMM2Processor",
        "image_processor_type": "BailingMM2ImageProcessor",
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "min_pixels": 4096,
        "max_pixels": 4194304,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "image_token": "<|image_pad|>",
        "video_token": "<|video_pad|>",
    }
    extracted = image_settings(preproc, _sample_config(), "qwen", **BAILING_QWEN_IMAGE_PARAMS)
    assert extracted == {
        "qwen": {
            "min_pixels": 4096,
            "max_pixels": 4194304,
            "position_side": 48,
            "image_token_id": IMAGE_TOKEN_ID,
            "video_token_id": VIDEO_TOKEN_ID,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "position_scheme": int(QwenPositionScheme.CENTERED_STEP),
            "resize_kernel": int(QwenResizeKernel.PILLOW_BICUBIC),
        }
    }


@pytest.mark.parametrize(
    "override,match",
    [
        ({"routed_scaling_factor": math.nan}, "routed_scaling_factor"),
        ({"routed_scaling_factor": math.inf}, "routed_scaling_factor"),
        ({"rope_theta": math.nan}, "rope_theta"),
        ({"rope_theta": math.inf}, "rope_theta"),
        ({"mrope_section": None}, "mrope_section"),
        ({"mrope_section": [8, 8, 16]}, "mrope_section"),
        ({"rope_interleave": False}, "rope_interleave"),
        ({"use_qkv_bias": True}, "QKV projection bias"),
        ({"hidden_act": "gelu"}, "silu"),
        ({"num_shared_experts": 2}, "one shared expert"),
        ({"use_qk_norm": False}, "use_qk_norm"),
    ],
)
def test_config_rejects_malformed_or_unsupported_fields(tmp_path, override, match):
    raw = _sample_config()
    for key, val in override.items():
        if val is None:
            raw["text_config"].pop(key, None)
        else:
            raw["text_config"][key] = val
    (tmp_path / "config.json").write_text(json.dumps(raw, allow_nan=True))
    with pytest.raises(ValueError, match=match):
        ModelConfig.load(tmp_path)


def test_config_accepts_top_level_mrope_section_and_rejects_conflicts(tmp_path):
    # Pinned timbuksu/Ling-3.2-Air-NVFP4 and upstream 1613b1a4 place mrope_section at top level.
    raw = _sample_config()
    raw["mrope_section"] = raw["text_config"].pop("mrope_section")
    (tmp_path / "config.json").write_text(json.dumps(raw))
    cfg = ModelConfig.load(tmp_path)
    assert cfg.mrope_section == (8, 12, 12)

    # Unequal spatial split at top level is still rejected.
    bad_top = _sample_config()
    bad_top["text_config"].pop("mrope_section")
    bad_top["mrope_section"] = [8, 8, 16]
    (tmp_path / "config.json").write_text(json.dumps(bad_top))
    with pytest.raises(ValueError, match="mrope_section"):
        ModelConfig.load(tmp_path)

    # Conflicting top-level and text_config values are rejected.
    conflict = _sample_config()
    conflict["mrope_section"] = [8, 10, 14]
    (tmp_path / "config.json").write_text(json.dumps(conflict))
    with pytest.raises(ValueError, match="conflicting mrope_section"):
        ModelConfig.load(tmp_path)


@pytest.mark.parametrize(
    "quant_entry,match",
    [
        (None, "missing expert quantization declaration"),
        ({"quant_algo": "MXFP8", "group_size": 32}, "NVFP4"),
        ({"quant_algo": "W4A16_NVFP4", "group_size": 16}, "NVFP4"),
        ({"quant_algo": "NVFP4", "group_size": 32}, "unsupported linear quantization"),
    ],
)
def test_expert_loader_validates_declared_quantization_before_loading(tmp_path, quant_entry, match):
    raw = _sample_config()
    prefix = "model.layers.2.mlp.experts"
    layers_q = {
        f"{prefix}.up_proj": {"quant_algo": "NVFP4", "group_size": 16},
        f"{prefix}.gate_proj": {"quant_algo": "NVFP4", "group_size": 16},
        f"{prefix}.down_proj": {"quant_algo": "NVFP4", "group_size": 16},
    }
    if quant_entry is None:
        del layers_q[f"{prefix}.down_proj"]
    else:
        layers_q[f"{prefix}.down_proj"] = quant_entry
    raw["quantization_config"]["quantized_layers"] = layers_q
    (tmp_path / "config.json").write_text(json.dumps(raw))
    cfg = ModelConfig.load(tmp_path)

    loader = Ling32WeightLoader.__new__(Ling32WeightLoader)
    loader.config = cfg
    loader._sources = {}
    with pytest.raises(ValueError, match=match):
        loader._load_experts(prefix, None)
