# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit and mathematical-parity tests for the Hy4-flash recipe (CPU-safe)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hyperflare_recipes.recipes import resolve
from hyperflare_recipes.recipes.hy4_flash.attention import pack_glm_nsa_records
from hyperflare_recipes.recipes.hy4_flash.config import (
    ENGRAM_HEADS,
    HC_MULT,
    HIDDEN_SIZE,
    ModelConfig,
    QK_ROPE_HEAD_DIM,
    VOCABULARY,
)
from hyperflare_recipes.recipes.hy4_flash.engram import (
    _FP4_SIGNED_VALUES,
    _draw_hash_multipliers,
    _head_offsets,
    dequantize_nvfp4_engram,
)
from hyperflare_recipes.recipes.hy4_flash.ihc import ihc_head, ihc_post, ihc_pre
from hyperflare_recipes.recipes.hy4_flash.pool import (
    PAGE_TOKENS,
    continuation_layout,
    page_bytes,
)
from hyperflare_recipes.recipes.hy4_flash.recipe import (
    RECIPE,
    _DFLY_STATE_ROW_BYTES,
    _PAGE_BYTES,
    _PLAIN_STATE_ROW_BYTES,
)
from hyperflare_recipes.recipes.hy4_flash.rotary import InterleavedMRoPE


def _synthetic_config_dict() -> dict:
    engram_hvs = {
        str(lid): [[101, 103, 107, 109, 113, 127, 131, 137], [139, 149, 151, 157, 163, 167, 173, 179]]
        for lid in (2, 12, 22, 32)
    }
    layer_types = [
        "indexed_attention" if i < 2 or (i - 1) % 4 == 0 else "sliding_attention"
        for i in range(42)
    ]
    return {
        "model_type": "hy_v4_flash",
        "architectures": ["HYV4FlashForConditionalGeneration"],
        "image_token_id": 120120,
        "image_newline_token_id": 120121,
        "video_token_id": 120124,
        "dfly_num_hidden_layers": 5,
        "dfly_target_kv_layers": [25, 29, 33, 37, 41],
        "dfly_target_index_layer": [25, 29, 33, 37, 41],
        "dfly_target_index_topk": 2048,
        "dfly_mask_token_id": 120009,
        "text_config": {
            "vocab_size": VOCABULARY,
            "hidden_size": HIDDEN_SIZE,
            "num_hidden_layers": 42,
            "hc_mult": HC_MULT,
            "hc_eps": 1e-6,
            "hc_magnitude": 2.0,
            "rms_norm_eps": 1e-5,
            "q_lora_rank": 1024,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
            "v_head_dim": 256,
            "num_attention_heads": 32,
            "swa_num_attention_heads": 64,
            "sliding_window": 512,
            "n_routed_experts": 512,
            "num_experts_per_tok": 16,
            "routed_expert_hidden_size": 1536,
            "moe_intermediate_size": 1536,
            "intermediate_size": 13824,
            "routed_scaling_factor": 3.998,
            "norm_topk_prob": True,
            "swiglu_limit": 10.0,
            "eos_token_id": 120025,
            "pad_token_id": 120002,
            "engram_pad_id": 120002,
            "engram_seed": 0,
            "engram_layer_ids": [2, 12, 22, 32],
            "engram_layer_head_vocab_sizes": engram_hvs,
            "layer_types": layer_types,
            "rope_parameters": {
                "indexed_attention": {"rope_theta": 10000000.0, "mrope_section": [12, 10, 10]},
                "sliding_attention": {"rope_theta": 10000.0, "mrope_section": [12, 10, 10]},
            },
        },
        "vision_config": {
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "patch_size": 16,
            "num_channels": 3,
            "max_image_size": 2048,
            "spatial_merge_size": 2,
            "out_hidden_size": HIDDEN_SIZE,
            "layer_norm_eps": 1e-6,
            "rms_norm_eps": 1e-5,
        },
    }


def test_recipe_layout_bytes_match_declared_rank_costs(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_synthetic_config_dict()))
    cfg = ModelConfig.load(tmp_path)
    assert len(cfg.sparse_layers) == 12
    assert len(cfg.swa_layers) == 30
    assert cfg.engram_layers == (1, 11, 21, 31)
    assert page_bytes(cfg) == _PAGE_BYTES
    assert continuation_layout(cfg, dfly=False).row_bytes == _PLAIN_STATE_ROW_BYTES
    assert continuation_layout(cfg, dfly=True).row_bytes == _DFLY_STATE_ROW_BYTES
    assert RECIPE.costs(1, None).state_row_bytes == _PLAIN_STATE_ROW_BYTES
    assert RECIPE.costs(1, RECIPE.speculation).state_row_bytes == _DFLY_STATE_ROW_BYTES
    assert RECIPE.costs(1, RECIPE.speculation).page_bytes == _PAGE_BYTES
    assert RECIPE.parameters.draft_quantization.draft_head_choices == ("nvfp4", "mxfp8")
    assert RECIPE.parameters.online_mxfp8.enabled is True
    assert RECIPE.parameters.online_mxfp8.effective_scope == "all"
    assert RECIPE.parameters.online_mxfp8.supported_scopes == ("all", "draft")
    assert resolve("hy4-flash-nvfp4").recipe is RECIPE


def test_draft_head_quantization_config_validation():
    from dataclasses import replace
    from hyperflare_recipes.recipes.hy4_flash.bootstrap import PreparationConfig

    base = RECIPE.parameters
    for fmt in (None, "nvfp4", "mxfp8"):
        cfg = PreparationConfig(
            parameters=replace(base, draft_quantization=replace(base.draft_quantization, draft_head_quantization=fmt))
        )
        assert cfg.draft_head_quantization == fmt
    with pytest.raises(ValueError, match="draft head quantization"):
        PreparationConfig(
            parameters=replace(base, draft_quantization=replace(base.draft_quantization, draft_head_quantization="fp8"))
        )
    with pytest.raises(ValueError, match="--draft-head-quantization requires"):
        PreparationConfig(
            parameters=replace(
                base,
                speculation=None,
                draft_quantization=replace(base.draft_quantization, draft_head_quantization="nvfp4"),
            )
        )

    for scope in ("all", "draft"):
        cfg = PreparationConfig(
            parameters=replace(base, online_mxfp8=replace(base.online_mxfp8, enabled=True, scope=scope))
        )
        assert cfg.online_mxfp8 is True and cfg.online_mxfp8_scope == scope
    with pytest.raises(ValueError, match="online MXFP8 scope"):
        PreparationConfig(
            parameters=replace(base, online_mxfp8=replace(base.online_mxfp8, enabled=True, scope="other"))
        )
    with pytest.raises(ValueError, match="--online-mxfp8-scope requires"):
        PreparationConfig(
            parameters=replace(base, online_mxfp8=replace(base.online_mxfp8, enabled=False, scope="draft"))
        )


def test_ihc_matches_reference_fp32_hyper_connection():
    torch.manual_seed(7)
    tokens, d, m = 5, 64, 4
    streams = torch.randn(tokens, m, d, dtype=torch.bfloat16)
    branch = torch.randn(tokens, d, dtype=torch.bfloat16)
    fn = torch.randn(2 * m, m * d, dtype=torch.float32) * 0.05
    base = torch.randn(2 * m, dtype=torch.float32)
    scale = torch.tensor([0.02, 0.03], dtype=torch.float32)

    post, collapsed = ihc_pre(streams, fn, base, scale, rms_eps=1e-5, hc_eps=1e-6, hc_magnitude=2.0)
    updated = ihc_post(streams, branch, post)

    # Exact HYV4FlashHyperConnection reference.
    flat = streams.flatten(1).float()
    inv = torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + 1e-5)
    mix = F.linear(flat, fn) * inv
    pre_ref = torch.sigmoid(mix[:, :m] * scale[0] + base[:m]) + 1e-6
    post_ref = 2.0 * torch.sigmoid(mix[:, m:] * scale[1] + base[m:]) + 1e-6
    collapsed_ref = (pre_ref.unsqueeze(-1) * streams.float()).sum(dim=1).to(torch.bfloat16)
    updated_ref = (post_ref.unsqueeze(-1) * branch.float().unsqueeze(1) + streams.float()).to(torch.bfloat16)

    assert torch.equal(collapsed, collapsed_ref)
    assert torch.equal(updated, updated_ref)

    h_fn = torch.randn(m, m * d, dtype=torch.float32) * 0.05
    h_base = torch.randn(m, dtype=torch.float32)
    h_scale = torch.tensor([0.01], dtype=torch.float32)
    head_out = ihc_head(streams, h_fn, h_base, h_scale, rms_eps=1e-5, hc_eps=1e-6)
    h_mix = F.linear(flat, h_fn) * inv
    h_pre = torch.sigmoid(h_mix * h_scale + h_base) + 1e-6
    head_ref = (h_pre.unsqueeze(-1) * streams.float()).sum(dim=1).to(torch.bfloat16)
    assert torch.equal(head_out, head_ref)


def test_interleaved_mrope_matches_transformers_3d_recomposition(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_synthetic_config_dict()))
    cfg = ModelConfig.load(tmp_path)
    rope = InterleavedMRoPE(cfg, torch.device("cpu"))

    torch.manual_seed(11)
    tokens = 7
    pos3 = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 1],
            [1, 0, 2],
            [1, 1, 1],
            [1, 1, 2],
            [2, 2, 2],
            [15, 15, 15],
        ],
        dtype=torch.int64,
    )
    x = torch.randn(tokens, 4, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16)

    for indexed, theta in ((True, 10000000.0), (False, 10000.0)):
        cos, sin = rope.cos_sin(pos3, indexed=indexed, dtype=torch.bfloat16)
        out = rope.apply(x, cos, sin)

        # Reference HYV4FlashRotaryEmbedding + apply_rotary_pos_emb.
        inv_freq = 1.0 / (theta ** (torch.arange(0, QK_ROPE_HEAD_DIM, 2, dtype=torch.float32) / QK_ROPE_HEAD_DIM))
        freqs = pos3.T[:, None, :, None].float() * inv_freq  # [3, 1, T, 32]
        cos_f, sin_f = freqs.cos(), freqs.sin()
        for f in (cos_f, sin_f):
            f_thw = f[0].clone()
            for dim, offset in ((1, 1), (2, 2)):
                idx = slice(offset, cfg.mrope_section[dim] * 3, 3)
                f_thw[..., idx] = f[dim][..., idx]
            f[0].copy_(f_thw)
        ref_cos = cos_f[0, 0].to(torch.bfloat16).repeat_interleave(2, dim=-1)
        ref_sin = sin_f[0, 0].to(torch.bfloat16).repeat_interleave(2, dim=-1)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        ref_out = ((x.float() * ref_cos[:, None, :]) + (rot.float() * ref_sin[:, None, :])).to(torch.bfloat16)
        assert torch.equal(out, ref_out)


def test_pack_glm_nsa_records_roundtrips_e4m3_scales_and_rope_bytes():
    torch.manual_seed(13)
    k_nope = torch.randn(9, 512, dtype=torch.bfloat16) * 3.0
    k_nope[4].zero_()  # Exercise zero-scale edge case.
    k_rope = torch.randn(9, 64, dtype=torch.bfloat16)
    packed = pack_glm_nsa_records(k_nope, k_rope)
    assert packed.shape == (9, 656) and packed.dtype == torch.uint8

    # Unpack and verify exact E4M3 quantization + inline FP32 scales + raw BF16 RoPE bytes.
    q = packed[:, :512].view(torch.float8_e4m3fn).float().view(9, 4, 128)
    scales = packed[:, 512:528].contiguous().view(torch.float32).view(9, 4)
    rope = packed[:, 528:].contiguous().view(torch.bfloat16).view(9, 64)
    assert torch.equal(rope, k_rope)
    assert torch.all(scales[4] == 1.0)
    deq = (q * scales.unsqueeze(-1)).view(9, 512)
    rel_err = (deq - k_nope.float()).abs().max() / k_nope.float().abs().max()
    assert rel_err < 0.08


def test_engram_hash_multipliers_match_numpy_reference():
    compressed_vocab = 60211
    for layer_1based in (2, 12, 22, 32):
        mults = _draw_hash_multipliers(layer_1based, compressed_vocab, seed=0)
        assert len(mults) == 3
        assert all(m % 2 == 1 and m > 0 for m in mults)
        # Recompute directly from numpy PCG64 matching HYV4FlashEngramLayer.draw_hash_multipliers.
        half_bound = max(1, np.iinfo(np.int64).max // compressed_vocab // 2)
        rng = np.random.default_rng(0 + 10007 * layer_1based)
        expected = tuple(int(v) * 2 + 1 for v in rng.integers(0, half_bound, size=3, dtype=np.int64).tolist())
        assert mults == expected
    primes = (11, 13, 17)
    assert _head_offsets(primes) == (0, 11, 24)


def test_engram_nvfp4_dequantization_matches_e2m1_block16_reference():
    torch.manual_seed(29)
    lut = torch.tensor(_FP4_SIGNED_VALUES, dtype=torch.float32)
    codes = torch.randint(0, 16, (3, ENGRAM_HEADS, 96), dtype=torch.uint8)
    packed = (codes[..., 0::2] & 0x0F) | ((codes[..., 1::2] & 0x0F) << 4)
    scales_f8 = (torch.rand(3, ENGRAM_HEADS, 6) * 4.0 + 0.25).to(torch.float8_e4m3fn)
    global_scale = 0.0125
    actual = dequantize_nvfp4_engram(
        packed,
        scales_f8.view(torch.uint8),
        global_scale,
        lut,
        dtype=torch.bfloat16,
    )
    expected = (
        lut[codes.long()].view(3, ENGRAM_HEADS, 6, 16)
        * scales_f8.float().unsqueeze(-1)
        * global_scale
    ).reshape(3, ENGRAM_HEADS, 96).to(torch.bfloat16)
    assert torch.equal(actual, expected)


def test_swa_sink_and_dfly_three_way_lse_combine_are_exact():
    from hyperflare_recipes.recipes.hy4_flash.attention import apply_attention_sink
    from hyperflare_recipes.recipes.hy4_flash.dfly import combine_three_way_lse

    torch.manual_seed(19)
    b, h, w, hist = 2, 32, 6, 17
    d = 576
    dv = 512
    scale = 256 ** -0.5

    q = torch.randn(b, h, w, d)
    k_hist = torch.randn(b, 1, hist, d)
    v_hist = torch.randn(b, 1, hist, dv)
    k_blk = torch.randn(b, 1, w, d)
    v_blk = torch.randn(b, 1, w, dv)
    sink = torch.randn(h) * 2.0
    causal = torch.tril(torch.ones(w, w, dtype=torch.bool))

    # Full combined attention with sink.
    s_hist = torch.matmul(q, k_hist.transpose(-2, -1)) * scale
    s_blk = (torch.matmul(q, k_blk.transpose(-2, -1)) * scale).masked_fill(~causal[None, None], float("-inf"))
    s_sink = sink[None, :, None, None].expand(b, h, w, 1)
    all_scores = torch.cat((s_hist, s_blk, s_sink), dim=-1)
    probs = F.softmax(all_scores, dim=-1)
    ref_out = torch.matmul(probs[..., :hist], v_hist) + torch.matmul(probs[..., hist : hist + w], v_blk)

    # 1. Post-hoc SWA sink helper on history alone (including an inactive token masked to 0).
    lse_h = torch.logsumexp(s_hist, dim=-1)
    out_h = torch.matmul(torch.exp(s_hist - lse_h.unsqueeze(-1)), v_hist)
    ref_h_sink = torch.matmul(F.softmax(torch.cat((s_hist, s_sink), dim=-1), dim=-1)[..., :hist], v_hist)
    t_out_h = out_h.transpose(1, 2).reshape(b * w, h, dv)
    t_lse_h = lse_h.transpose(1, 2).reshape(b * w, h)
    live = torch.ones(b * w, dtype=torch.bool)
    live[-1] = False
    posthoc_h = apply_attention_sink(t_out_h, t_lse_h, sink, live)
    ref_h_flat = ref_h_sink.transpose(1, 2).reshape(b * w, h, dv)
    assert (posthoc_h[:-1] - ref_h_flat[:-1]).abs().max().item() < 1e-5
    assert torch.all(posthoc_h[-1] == 0)

    # 2. DFly2 3-way LSE combine helper (history + causal block + sink, plus zero-history fallback).
    lse_b = torch.logsumexp(s_blk, dim=-1)
    out_b = torch.matmul(torch.exp(s_blk - lse_b.unsqueeze(-1)), v_blk)
    t_out_b = out_b.transpose(1, 2).reshape(b * w, h, dv)
    t_lse_b = lse_b.transpose(1, 2).reshape(b * w, h)
    sel_lens = torch.full((b * w,), hist, dtype=torch.int32)
    combined = combine_three_way_lse(t_out_h, t_lse_h, t_out_b, t_lse_b, sink, sel_lens)
    assert (combined - ref_out.transpose(1, 2).reshape(b * w, h, dv)).abs().max().item() < 1e-5

    # Zero-history tokens ignore out_hist/lse_hist (even with NaN/inf LSE) and combine only block + sink.
    sel_zero = torch.zeros(b * w, dtype=torch.int32)
    zero_combined = combine_three_way_lse(t_out_h, torch.full_like(t_lse_h, float("nan")), t_out_b, t_lse_b, sink, sel_zero)
    blk_only_ref = torch.matmul(F.softmax(torch.cat((s_blk, s_sink), dim=-1), dim=-1)[..., :w], v_blk)
    assert (zero_combined - blk_only_ref.transpose(1, 2).reshape(b * w, h, dv)).abs().max().item() < 1e-5


def test_hy4_snapshot_config_template_and_frontend_metadata():
    raw = os.environ.get("FLARE_HY4_FLASH_SNAPSHOT")
    if not raw:
        pytest.skip("set FLARE_HY4_FLASH_SNAPSHOT to run Hy4-flash snapshot metadata tests")
    snapshot = Path(raw)
    if not (snapshot / "config.json").is_file():
        pytest.skip(f"Hy4-flash metadata snapshot not present at {snapshot}")

    from ignition.preparation.frontend import prepare_frontend
    from hyperflare_recipes.recipes.hy4_flash.bootstrap import chat_template_asset

    cfg = ModelConfig.load(snapshot)
    assert len(cfg.sparse_layers) == 12
    assert len(cfg.swa_layers) == 30
    assert cfg.engram_layers == (1, 11, 21, 31)
    assert cfg.draft is not None and cfg.draft.layers == 5
    assert cfg.draft.target_kv_layers == (25, 29, 33, 37, 41)
    assert cfg.vision is not None and cfg.vision.num_hidden_layers == 27

    tpl = chat_template_asset(snapshot)
    assert tpl["sha256"] == "7d9d42c9771eaf9ea4cd94db73863d85052765e31cb37611ee0f3fd61c4e27ef"

    fe = prepare_frontend(
        snapshot,
        eos_token_ids=cfg.text["eos_token_id"],
        image_profile="row_newline",
        preprocessor=snapshot / "processor_config.json",
    )
    assert fe["eos_token_ids"] == [120025]
    assert fe["image"]["row_newline"]["position_side"] == 128
    assert fe["image"]["row_newline"]["image_token_id"] == 120120
    assert fe["image"]["row_newline"]["newline_token_id"] == 120121


@pytest.mark.parametrize("language_model_only", [False, True])
def test_hy4_export_geometry_passes_native_bootstrap_validation(tmp_path, language_model_only):
    import hashlib
    import flare_heliopause
    from ignition.preparation.frontend import prepare_frontend
    from hyperflare_recipes.recipes.hy4_flash.bootstrap import chat_template_asset
    from tests.preparation.test_native_recipe_contract import _recipe_metadata

    raw = os.environ.get("FLARE_HY4_FLASH_SNAPSHOT")
    if not raw:
        pytest.skip("set FLARE_HY4_FLASH_SNAPSHOT to run Hy4-flash native bootstrap validation")
    snapshot = Path(raw)
    if not (snapshot / "tokenizer.json").is_file() or not (snapshot / "chat_template.jinja").is_file():
        pytest.skip(f"Hy4-flash snapshot assets not present at {snapshot}")

    cfg = ModelConfig.load(snapshot)
    config, contract, manifest = _recipe_metadata(
        "hy4_flash",
        all_variants=False,
        config_options=dict(language_model_only=language_model_only),
    )
    fe = prepare_frontend(
        snapshot,
        eos_token_ids=cfg.text["eos_token_id"],
        image_profile=None if language_model_only else "row_newline",
        preprocessor=snapshot / "processor_config.json",
    )
    tpl = chat_template_asset(snapshot)
    tpl_path = tmp_path / "chat_template.jinja"
    tpl_path.write_text(tpl["text"], encoding="utf-8")
    server_config = dict(
        identity="00" * 32,
        model_id=RECIPE.served_model_name,
        max_sequence_tokens=config.context,
        prefill_chunk_size=config.prefill_chunk_size,
        pages=4096,
        state_rows=16,
        snapshots=8,
        page_tokens=PAGE_TOKENS,
        frontend=fe,
        sampling_defaults=RECIPE.sampling_defaults.as_dict(),
        assets={
            "tokenizer": {
                "path": str(snapshot / "tokenizer.json"),
                "sha256": hashlib.sha256((snapshot / "tokenizer.json").read_bytes()).hexdigest(),
            },
            "chat_template": {
                "path": str(tpl_path),
                "sha256": tpl["sha256"],
            },
        },
        model_contract=contract,
        speculative_widths=list(config.speculative_widths),
        speculative_token_capacity=config.speculative_token_capacity,
    )
    flare_heliopause.validate_prepared_catalog(manifest, json.dumps(server_config))


def test_dfly_logical_indices_survive_page_table_relocation_and_mask_inactive_rows():
    from hyperflare_recipes.recipes.hy4_flash.dfly import resolve_logical_slots
    from hyperflare_recipes.recipes.hy4_flash.indexer import expand_pooled_topk_to_logical_tokens

    # Row 0: pos=515 (page 2, completes group 128) with 2 selected pools across page 0 & page 2.
    # Row 1: inactive (pos=-1).
    # Row 2: pos=261 (page 1, partial tail tokens 260, 261) with pool 0 and pool 64.
    pools = torch.full((3, 512), -1, dtype=torch.int32)
    pools[0, 0] = 2     # logical tokens 8..11 (page 0)
    pools[0, 1] = 128   # logical tokens 512..515 (page 2)
    pools[1, 0] = 0
    pools[2, 0] = 0     # logical tokens 0..3 (page 0)
    pools[2, 1] = 64    # logical tokens 256..259 (page 1)
    positions = torch.tensor([515, -1, 261], dtype=torch.int64)

    logical = expand_pooled_topk_to_logical_tokens(pools, positions, width=16)
    assert logical[0, :8].tolist() == [8, 9, 10, 11, 512, 513, 514, 515]
    assert torch.all(logical[0, 8:] == -1)
    assert torch.all(logical[1] == -1)
    assert logical[2, :10].tolist() == [0, 1, 2, 3, 256, 257, 258, 259, 260, 261]
    assert torch.all(logical[2, 10:] == -1)

    # Relocate restored rows onto completely disjoint physical page IDs.
    orig_pages = torch.tensor([[7, 19, 3, -1], [5, 6, 7, 8], [10, 11, -1, -1]], dtype=torch.int32)
    restored_pages = torch.tensor([[102, 55, 88, -1], [9, 8, 7, 6], [42, 77, -1, -1]], dtype=torch.int32)
    active = torch.tensor([True, False, True])

    orig_slots = resolve_logical_slots(logical, orig_pages, active)
    restored_slots = resolve_logical_slots(logical, restored_pages, active)
    assert orig_slots[0, :8].tolist() == [
        7 * 256 + 8, 7 * 256 + 9, 7 * 256 + 10, 7 * 256 + 11,
        3 * 256 + 0, 3 * 256 + 1, 3 * 256 + 2, 3 * 256 + 3,
    ]
    assert restored_slots[0, :8].tolist() == [
        102 * 256 + 8, 102 * 256 + 9, 102 * 256 + 10, 102 * 256 + 11,
        88 * 256 + 0, 88 * 256 + 1, 88 * 256 + 2, 88 * 256 + 3,
    ]
    assert torch.all(restored_slots[1] == -1)
    assert restored_slots[2, :10].tolist() == [
        42 * 256 + 0, 42 * 256 + 1, 42 * 256 + 2, 42 * 256 + 3,
        77 * 256 + 0, 77 * 256 + 1, 77 * 256 + 2, 77 * 256 + 3,
        77 * 256 + 4, 77 * 256 + 5,
    ]

    # Full-cap case: 511 prior pools + 3 tail tokens (pos=2050) fits cleanly in 2048 slots,
    # even when valid pools are not placed in leading columns.
    full_pools = torch.full((1, 512), -1, dtype=torch.int32)
    full_pools[0, 1:512] = torch.arange(511, dtype=torch.int32)
    full_logical = expand_pooled_topk_to_logical_tokens(full_pools, torch.tensor([2050], dtype=torch.int64))
    assert (full_logical[0] >= 0).sum().item() == 2047
    assert full_logical[0, 2044:2048].tolist() == [2048, 2049, 2050, -1]


def test_dfly_decoder_save_and_encode_block_handoff_survives_page_relocation(tmp_path):
    """Exercise PreparedDecoder._save_dfly_indices -> ContinuationPool -> PreparedDFlyDraft.encode_block."""
    from types import SimpleNamespace
    from hyperflare_recipes.recipes.hy4_flash.decoder import PreparedDecoder
    from hyperflare_recipes.recipes.hy4_flash.dfly import PreparedDFlyDraft

    (tmp_path / "config.json").write_text(json.dumps(_synthetic_config_dict()))
    cfg = ModelConfig.load(tmp_path)
    region_tensor = torch.full((4, 5, 2048), -1, dtype=torch.int32)
    last_hidden_tensor = torch.zeros((4, HIDDEN_SIZE), dtype=torch.bfloat16)
    pool = SimpleNamespace(
        region=lambda name: region_tensor if name == "dfly_target_indices" else None,
        last_hidden=last_hidden_tensor,
    )

    # 3 requests (row 1 active with 2 tokens accepted, row 2 inactive, row 3 active with 3 tokens accepted).
    # Last token for row 1 is at packed index 1 (pos=265 -> 256*1 + 9, tail pos 264, 265);
    # last token for row 3 is at packed index 5 (pos=514 -> 256*2 + 2, tail pos 512, 513, 514).
    dec = object.__new__(PreparedDecoder)
    dec.dfly_index_layers = (5, 11, 17, 25, 37)
    dec.dfly_topk = 2048
    dec.pool = pool

    pools_by_layer = {}
    for layer in dec.dfly_index_layers:
        p = torch.full((6, 512), -1, dtype=torch.int32)
        p[1, 0], p[1, 2] = 64, 1  # non-compacted valid pools for row 1's accepted last token
        p[5, 0], p[5, 1] = 128, 0  # pools for row 3's accepted last token
        pools_by_layer[layer] = p

    meta = SimpleNamespace(
        starts=torch.tensor([264, 0, 512], dtype=torch.int64),
        query_start_loc=torch.tensor([0, 2, 3, 6], dtype=torch.int32),
        positions=torch.tensor([264, 265, -1, 512, 513, 514], dtype=torch.int64),
        state_rows=torch.tensor([1, 2, 3], dtype=torch.int32),
    )
    counts = torch.tensor([2, 0, 3], dtype=torch.int32)
    dec._save_dfly_indices(counts, pools_by_layer, meta)

    # Exercise PreparedDFlyDraft.encode_block with lightweight capture stubs recording resolved slots.
    captured_slots: list[torch.Tensor] = []

    def _capture_layer(hidden, cos, sin, slots_rep, visible, len_rep, causal_mask):
        captured_slots.append(slots_rep.clone())
        return hidden

    draft = object.__new__(PreparedDFlyDraft)
    draft.pool = pool
    draft.batch = 3
    draft.width = 2
    draft.mask_token_id = cfg.draft.mask_token_id
    draft.topk = 2048
    draft.causal_mask = torch.tril(torch.ones((2, 2), dtype=torch.bool))
    draft.rotary = InterleavedMRoPE(cfg, torch.device("cpu"))
    draft.block_embed = lambda *, token_ids: torch.zeros((token_ids.numel(), HIDDEN_SIZE), dtype=torch.bfloat16)
    draft.mtp_enorm = lambda x: x
    draft.mtp_hnorm = lambda x: x
    draft.mtp_eh_proj = lambda x: x[:, :HIDDEN_SIZE]
    draft.mtp_gate = torch.tensor(0.0, dtype=torch.bfloat16)
    draft.final_norm = lambda x: x
    draft.layers = [_capture_layer for _ in range(5)]

    active = torch.tensor([True, False, True])
    orig_pages = torch.tensor([[3, 7, 11], [1, 2, 3], [5, 9, 13]], dtype=torch.int32)
    reloc_pages = torch.tensor([[41, 88, 99], [1, 2, 3], [60, 70, 80]], dtype=torch.int32)

    draft.encode_block(
        torch.tensor([10, 0, 20], dtype=torch.int64),
        torch.tensor([266, 0, 515], dtype=torch.int64),
        None,
        torch.tensor([1, 2, 3], dtype=torch.int32),
        orig_pages,
        active,
    )
    orig_resolved = captured_slots[0].view(3, 2, 2048)
    # Row 0: pool 64 -> logical 256..259 (page 1 -> phys 7), pool 1 -> logical 4..7 (page 0 -> phys 3), tail 264, 265 (phys 7).
    assert orig_resolved[0, 0, :10].tolist() == [
        7 * 256 + 0, 7 * 256 + 1, 7 * 256 + 2, 7 * 256 + 3,
        3 * 256 + 4, 3 * 256 + 5, 3 * 256 + 6, 3 * 256 + 7,
        7 * 256 + 8, 7 * 256 + 9,
    ]
    assert torch.all(orig_resolved[1] == -1)

    captured_slots.clear()
    draft.encode_block(
        torch.tensor([10, 0, 20], dtype=torch.int64),
        torch.tensor([266, 0, 515], dtype=torch.int64),
        None,
        torch.tensor([1, 2, 3], dtype=torch.int32),
        reloc_pages,
        active,
    )
    reloc_resolved = captured_slots[0].view(3, 2, 2048)
    assert reloc_resolved[0, 0, :10].tolist() == [
        88 * 256 + 0, 88 * 256 + 1, 88 * 256 + 2, 88 * 256 + 3,
        41 * 256 + 4, 41 * 256 + 5, 41 * 256 + 6, 41 * 256 + 7,
        88 * 256 + 8, 88 * 256 + 9,
    ]
    assert torch.all(reloc_resolved[1] == -1)
    assert reloc_resolved[2, 0, :11].tolist() == [
        80 * 256 + 0, 80 * 256 + 1, 80 * 256 + 2, 80 * 256 + 3,
        60 * 256 + 0, 60 * 256 + 1, 60 * 256 + 2, 60 * 256 + 3,
        80 * 256 + 0, 80 * 256 + 1, 80 * 256 + 2,
    ]


def test_config_rejects_invalid_layer_types_mrope_bounds_and_missing_keys(tmp_path):
    import copy

    def _load(mutator):
        cfg = copy.deepcopy(_synthetic_config_dict())
        mutator(cfg)
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        return ModelConfig.load(tmp_path)

    with pytest.raises(ValueError, match="42-entry layer_types"):
        _load(lambda c: c["text_config"].pop("layer_types"))
    with pytest.raises(ValueError, match="12 indexed_attention and 30 sliding_attention"):
        _load(lambda c: c["text_config"]["layer_types"].__setitem__(2, "indexed_attention"))
    with pytest.raises(ValueError, match="interleaved axis bounds"):
        _load(lambda c: c["text_config"]["rope_parameters"]["indexed_attention"].__setitem__("mrope_section", [10, 12, 10]))
    with pytest.raises(ValueError, match="interleaved axis bounds"):
        _load(lambda c: c["text_config"]["rope_parameters"]["indexed_attention"].__setitem__("mrope_section", [10, 11, 11]))
    with pytest.raises(ValueError, match="rope_theta"):
        _load(lambda c: c["text_config"]["rope_parameters"]["indexed_attention"].__setitem__("rope_theta", float("nan")))
    with pytest.raises(ValueError, match="missing required key"):
        _load(lambda c: c["text_config"].pop("engram_pad_id"))
    with pytest.raises(ValueError, match="vision_config hidden_size"):
        _load(lambda c: c["vision_config"].__setitem__("hidden_size", 1024))
    with pytest.raises(ValueError, match="dfly_num_hidden_layers == 5"):
        _load(lambda c: c.__setitem__("dfly_num_hidden_layers", 4))
    with pytest.raises(ValueError, match="dfly_target_index_topk == 2048"):
        _load(lambda c: c.__setitem__("dfly_target_index_topk", 1024))
    with pytest.raises(ValueError, match="dfly_target_index_topk == 2048"):
        _load(lambda c: c.__setitem__("dfly_target_index_topk", 4096))


def test_model_weights_rejects_missing_requested_dfly_or_vision(tmp_path):
    from types import SimpleNamespace
    from hyperflare_recipes.recipes.hy4_flash.weights import ModelWeights

    cfg = _synthetic_config_dict()
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    manifest_no_dfly = SimpleNamespace(metadata_path=tmp_path, tensors={"vit.embeddings.patch_embedding.weight": object()})
    with pytest.raises(ValueError, match="DFly2 speculation was requested"):
        ModelWeights.load(manifest_no_dfly, torch.device("cpu"), shard=SimpleNamespace(world_size=1), dfly=True, vision=False)

    manifest_no_vit = SimpleNamespace(metadata_path=tmp_path, tensors={"model.dfly_layers.0.input_layernorm.weight": object()})
    with pytest.raises(ValueError, match="Vision preparation was requested"):
        ModelWeights.load(manifest_no_vit, torch.device("cpu"), shard=SimpleNamespace(world_size=1), dfly=False, vision=True)


def test_attention_scratch_projected_view_rejects_overflow():
    from hyperflare_recipes.recipes.hy4_flash.attention import AttentionScratch, _QKV_MAX_WIDTH

    scratch = AttentionScratch.allocate(4, 1, 4, torch.device("cpu"))
    view = scratch.projected_view(4, _QKV_MAX_WIDTH)
    assert view.shape == (4, _QKV_MAX_WIDTH)
    assert view.data_ptr() == scratch.projected_fused.data_ptr()
    with pytest.raises(ValueError, match="exceeds allocated attention scratch"):
        scratch.projected_view(4, _QKV_MAX_WIDTH + 1)


def test_engram_table_requests_hash_continuity_across_chunk_boundaries(monkeypatch):
    from types import SimpleNamespace
    from hyperflare_recipes.components.paged_batch import Metadata
    from hyperflare_recipes.recipes.hy4_flash import engram as engram_mod
    from hyperflare_recipes.recipes.hy4_flash.config import IMAGE_NEWLINE_TOKEN, IMAGE_TOKEN, VIDEO_TOKEN

    published = []
    monkeypatch.setattr(engram_mod, "publish_requests", lambda hashes, mb, seq: published.append(hashes.clone()))

    pad_id = 120002
    token_map = [i % 97 for i in range(VOCABULARY)]
    token_map[pad_id] = 0
    primes = (101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179)
    spec = engram_mod.LayerHashSpec(
        layer=1,
        rows=sum(primes),
        primes=primes,
        offsets=_head_offsets(primes),
        multipliers=_draw_hash_multipliers(2, 97, 0),
    )

    def _run_sequence(chunks):
        published.clear()
        history = torch.zeros((4, 2), dtype=torch.int64)
        pool = SimpleNamespace(region=lambda name: history if name == "engram_history" else None)
        max_len = max(len(c) for c in chunks)
        hashes_buf = torch.full((max_len, ENGRAM_HEADS), -1, dtype=torch.int64)
        tables = SimpleNamespace(
            specs=(spec,),
            token_map=tuple(token_map),
            pad_token_id=pad_id,
            device_token_map=torch.tensor(token_map, dtype=torch.int64),
            inputs={1: SimpleNamespace(hashes=hashes_buf)},
            mailboxes={1: SimpleNamespace(device_view=torch.zeros(max_len * ENGRAM_HEADS + 1, dtype=torch.int64))},
            sequence=torch.zeros(1, dtype=torch.int64),
        )
        all_hashes = []
        offset = 0
        for chunk in chunks:
            n = len(chunk)
            meta = Metadata.pack(
                starts=torch.tensor([offset], dtype=torch.int64),
                lengths=torch.tensor([n], dtype=torch.int32),
                state_rows=torch.tensor([1], dtype=torch.int64),
                pages=torch.zeros((1, 1), dtype=torch.int32),
                tokens=torch.tensor([chunk], dtype=torch.int64),
                live_rows=torch.tensor([1], dtype=torch.int32),
                token_capacity=n,
            )
            req = engram_mod.PreparedTableRequests(tables, meta, pool)
            raw_hist = req(meta)
            req.commit(torch.tensor([n], dtype=torch.int32), raw_hist, meta)
            all_hashes.append(published[-1].clone())
            offset += n
        return torch.cat(all_hashes, dim=0)

    seq = [11, 22, IMAGE_TOKEN, IMAGE_NEWLINE_TOKEN, 33, 44, VIDEO_TOKEN, 55]
    single = _run_sequence([seq])
    chunked = _run_sequence([seq[:3], seq[3:5], seq[5:]])
    assert torch.equal(single, chunked)
    for masked_idx in (2, 3, 6):
        assert torch.all(single[masked_idx] == -1)
    for active_idx in (0, 1, 4, 5, 7):
        assert torch.all(single[active_idx] >= 0)


def test_collect_disposable_storage_handles_prepared_generation_scored_prefill_and_decoder():
    """Verify post-capture export zero-range collection works on real Hy4 forward wrapper types."""
    from types import SimpleNamespace
    from hyperflare_recipes.components.paged_generation import PreparedGeneration
    from hyperflare_recipes.components.prompt_logprobs import PreparedScoredPrefill
    from hyperflare_recipes.recipes.hy4_flash.bootstrap import _collect_disposable_storage
    from hyperflare_recipes.recipes.hy4_flash.decoder import PreparedDecoder
    from hyperflare_recipes.recipes.hy4_flash.vision_state import PreparedVisionGeneration
    from ignition.preparation.disposable import storage_ranges

    arena_t = torch.zeros(16, dtype=torch.uint8)
    scale_t = torch.zeros(8, dtype=torch.uint8)
    engram_t = torch.zeros(12, dtype=torch.uint8)
    vision_t = torch.zeros(24, dtype=torch.uint8)
    gen_scratch = torch.zeros(32, dtype=torch.uint8)
    scored_scratch = torch.zeros(40, dtype=torch.uint8)

    dec = object.__new__(PreparedDecoder)
    gen = object.__new__(PreparedGeneration)
    gen.decoder = dec
    gen.completion = SimpleNamespace(scratch_tensors=(gen_scratch,))

    scored_inner = object.__new__(PreparedGeneration)
    scored_inner.decoder = dec
    scored_inner.completion = SimpleNamespace(scratch_tensors=(scored_scratch,))
    scored = object.__new__(PreparedScoredPrefill)
    scored.forward = scored_inner

    vis = object.__new__(PreparedVisionGeneration)

    disposable = _collect_disposable_storage(
        arena_tensor=arena_t,
        scale_scratch=scale_t,
        engram_buffers={"e": engram_t},
        vision_arena=vision_t,
        forwards={"prefill": gen, "scored_prefill": scored, "decode_raw": dec, "vision": vis},
    )
    ranges = storage_ranges(disposable)
    ptrs = {start for start, _end in ranges}
    for tensor in (arena_t, scale_t, engram_t, vision_t, gen_scratch, scored_scratch):
        assert tensor.data_ptr() in ptrs


def test_c4_select_pools_evicts_lowest_score_and_compacts_physical_tail():
    """Verify score-based pool eviction past 2,048 tokens and partial-tail compaction."""
    from hyperflare_recipes.recipes.hy4_flash.indexer import (
        compact_physical_topk,
        expand_pooled_topk_to_logical_tokens,
        select_c4_pools,
    )

    # Rows cover inactive (-1), pre-cap (0, 3, 2046, 2047), exact-cap partial (2048, 2049, 2050),
    # exact-cap completing (2051), and post-cap partial/completing (2054, 2055).
    positions = torch.tensor([-1, 0, 3, 2046, 2047, 2048, 2049, 2050, 2051, 2054, 2055], dtype=torch.int64)
    rows = positions.numel()
    prior_pools = torch.full((rows, 512), -1, dtype=torch.int32)
    prior_scores = torch.full((rows, 512), float("-inf"), dtype=torch.float32)

    for r, pos in enumerate(positions.tolist()):
        if pos < 0:
            continue
        n_prior = min(pos // 4, 512)
        if n_prior > 0:
            # Place highest score in column 511 and lowest score in column 7 so fixed-column eviction fails.
            prior_pools[r, :n_prior] = torch.arange(n_prior, dtype=torch.int32)
            scores = torch.linspace(1.0, 10.0, n_prior, dtype=torch.float32)
            if n_prior == 512:
                scores[7] = -5.0
                scores[511] = 99.0
            prior_scores[r, :n_prior] = scores

    pools = select_c4_pools(prior_pools, prior_scores, positions)

    # Inactive row stays all -1.
    assert torch.all(pools[0] == -1)
    # Pre-cap rows preserve all prior pools and append completing own group without evicting.
    assert torch.all(pools[1] == -1)
    assert pools[2, 0].item() == 0 and torch.all(pools[2, 1:] == -1)
    assert pools[3, :511].tolist() == list(range(511)) and pools[3, 511].item() == -1
    assert pools[4].tolist() == list(range(512))

    # At or past position 2048, column 7 (score -5.0) must be evicted while column 511 (score 99.0) survives.
    for r in (5, 6, 7, 8, 9, 10):
        pos = int(positions[r].item())
        completes = (pos % 4) == 3
        kept_prior = set(pools[r, :511].tolist())
        assert len(kept_prior) == 511
        assert 7 not in kept_prior
        assert 511 in kept_prior
        assert all(p >= 0 for p in kept_prior)
        assert pools[r, 511].item() == (pos // 4 if completes else -1)

    # Emulate b12x expand_pooled_topk_to_physical_slots (1:1 identity physical slots) and verify compaction.
    physical = torch.full((rows, 2051), -1, dtype=torch.int32)
    raw_lengths = torch.zeros((rows,), dtype=torch.int32)
    for r, pos in enumerate(positions.tolist()):
        seq_len = pos + 1
        complete_pools = seq_len // 4
        sel_pools = min(complete_pools, 512)
        for p_idx in range(sel_pools):
            pid = int(pools[r, p_idx].item())
            if pid >= 0:
                for off in range(4):
                    physical[r, p_idx * 4 + off] = pid * 4 + off
        tail_start = complete_pools * 4
        tail_count = seq_len - tail_start
        for off in range(tail_count):
            physical[r, sel_pools * 4 + off] = tail_start + off
        raw_lengths[r] = sel_pools * 4 + tail_count

    compacted, lengths = compact_physical_topk(physical, raw_lengths, positions)
    logical = expand_pooled_topk_to_logical_tokens(pools, positions)

    expected_lengths = [0, 1, 4, 2047, 2048, 2045, 2046, 2047, 2048, 2047, 2048]
    assert lengths.tolist() == expected_lengths
    for r, (pos, exp_len) in enumerate(zip(positions.tolist(), expected_lengths, strict=True)):
        row_slots = compacted[r]
        # Valid slots form a contiguous prefix with zero -1 holes and match logical expansion exactly.
        assert torch.all(row_slots[:exp_len] >= 0)
        assert torch.all(row_slots[exp_len:] == -1)
        assert torch.equal(row_slots, logical[r])
        if pos >= 0:
            assert pos in row_slots[:exp_len].tolist()
            tail_start = (pos // 4) * 4
            expected_tail = list(range(tail_start, pos + 1))
            assert row_slots[exp_len - len(expected_tail) : exp_len].tolist() == expected_tail


def test_dfly_speculative_round_uses_validated_state_rows_and_width_one_lengths():
    """Verify PreparedDFlySpeculativeRound passes validated state_rows/page_table and width-1 lengths."""
    from types import SimpleNamespace
    from hyperflare_recipes.components.paged_batch import Metadata
    from hyperflare_recipes.recipes.hy4_flash.dfly import PreparedDFlySpeculativeRound

    capacity, b, w, blocks = 4, 2, 2, 2
    captured = {}

    class _Validator:
        def validate(self):
            return torch.zeros((1, 8)), torch.tensor([False, True, False, False])

        def begin(self, totals, *, present):
            del totals, present
            # Lane 0 rejected (state_row=0, length=0, pages=-1); lane 1 valid (state_row=2, length=1, page=3).
            m = Metadata.allocate(b, b, blocks, torch.device("cpu"))
            m.lengths.copy_(torch.tensor([0, 1], dtype=torch.int32))
            m.state_rows.copy_(torch.tensor([0, 2], dtype=torch.int64))
            m.page_table.copy_(torch.tensor([[-1, -1], [3, -1]], dtype=torch.int32))
            return m

    class _Draft:
        def encode_block(self, anchor, before, rotary_offsets, state_rows, page_tables, active):
            del anchor, before, rotary_offsets, active
            captured["draft_state_rows"] = state_rows.clone()
            captured["draft_page_tables"] = page_tables.clone()
            return torch.zeros((b, w, HIDDEN_SIZE), dtype=torch.bfloat16)

        def step_logits(self, slot_hidden, prev_tokens):
            del slot_hidden, prev_tokens
            return torch.zeros((b, VOCABULARY), dtype=torch.bfloat16)

    class _Sampler:
        layout = {"valid": ((b, 1), torch.bool)}

        def run(self, scratch, logits, *, candidates, seed, offset, sampled):
            del logits, candidates, seed, offset
            scratch["valid"] = torch.ones((b, 1), dtype=torch.bool)
            sampled.fill_(5)
            return torch.zeros((b, 1, VOCABULARY), dtype=torch.float32)

    class _Decoder:
        def __call__(self, metadata):
            captured["decoder_state_rows"] = metadata.state_rows.clone()
            captured["decoder_page_table"] = metadata.page_table.clone()
            captured["decoder_lengths"] = metadata.lengths.clone()
            toks = metadata.token_ids.numel()
            return (
                torch.zeros((toks, HIDDEN_SIZE), dtype=torch.bfloat16),
                torch.zeros((toks, VOCABULARY), dtype=torch.bfloat16),
                None,
                None,
            )

        def commit(self, counts, records, metadata):
            del counts, records, metadata

    class _Target:
        def __init__(self, width: int):
            self.width = width
            self.layout = {"probabilities": ((b, width, VOCABULARY), torch.float32), "valid": ((b, width), torch.bool)}

        def run(self, scratch, logits, *, candidates):
            del logits, candidates
            scratch["probabilities"] = torch.zeros((b, self.width, VOCABULARY), dtype=torch.float32)
            scratch["valid"] = torch.ones((b, self.width), dtype=torch.bool)

    class _Verify:
        layout = {}

        def run(self, *args, **kwargs):
            del args, kwargs
            return torch.zeros((b, rnd.width), dtype=torch.bool)

    def _commit(logits, *, acceptance, inputs):
        del logits, acceptance, inputs
        return SimpleNamespace(emitted_count=torch.tensor([0, 1], dtype=torch.int32))

    # Lane 0's raw request_state_rows holds an out-of-range sentinel (999999) that must never reach encode_block.
    raw_state_rows = torch.zeros((capacity, 8), dtype=torch.int64)
    raw_state_rows[0, 0] = 999999
    raw_state_rows[1, 0] = 2
    raw_pages = torch.tensor([[999, 999], [3, -1], [-1, -1], [-1, -1]], dtype=torch.int32)
    owner = SimpleNamespace(
        inputs=SimpleNamespace(
            row_slot_ids=torch.tensor([0, 1], dtype=torch.int32),
            query_lengths=torch.tensor([2, 2], dtype=torch.int32),
            request_state_rows=raw_state_rows,
            page_tables=raw_pages,
            live_rows=torch.tensor([2], dtype=torch.int32),
        ),
        requests=SimpleNamespace(
            capacity=capacity,
            histogram=torch.zeros((1,), dtype=torch.int32),
            consumed_tokens=torch.tensor([10, 20, 0, 0], dtype=torch.int64),
            pending_token=torch.tensor([1, 2, 0, 0], dtype=torch.int64),
            generation=torch.tensor([1, 1, 0, 0], dtype=torch.int64),
            seed=torch.zeros((capacity,), dtype=torch.int64),
        ),
        pool=SimpleNamespace(last_hidden=torch.zeros((4, HIDDEN_SIZE), dtype=torch.bfloat16)),
        offset=torch.zeros((capacity,), dtype=torch.int64),
        generation=torch.zeros((capacity,), dtype=torch.int64),
    )

    rnd = object.__new__(PreparedDFlySpeculativeRound)
    rnd.owner, rnd.batch, rnd.width = owner, b, w
    rnd.validator = _Validator()
    rnd.validation = lambda x: x
    rnd.draft = _Draft()
    rnd.decoder = _Decoder()
    rnd.operations = (_Target(w), (_Sampler(),), _Verify(), _commit)
    rnd()

    assert captured["draft_state_rows"].tolist() == [0, 2]
    assert captured["draft_page_tables"].tolist() == [[-1, -1], [3, -1]]
    assert captured["decoder_state_rows"].tolist() == [0, 2]
    assert captured["decoder_page_table"].tolist() == [[-1, -1], [3, -1]]
    assert captured["decoder_lengths"].tolist() == [0, 2]

    # Width-1 speculative rounds must use validated lengths (1) even if raw query_lengths is larger.
    captured.clear()
    owner.inputs.query_lengths.fill_(6)
    rnd.width = 1
    rnd.draft = None
    rnd.operations = (_Target(1), (), _Verify(), _commit)
    rnd()
    assert captured["decoder_lengths"].tolist() == [0, 1]
