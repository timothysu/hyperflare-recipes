# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CUDA parity, capture, and real-checkpoint tests for the Hy4-flash TP=1 recipe."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from hyperflare_recipes.components.linear import UnquantizedProjection
from hyperflare_recipes.components.paged_batch import Metadata
from hyperflare_recipes.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from hyperflare_recipes.recipes.hy4_flash.attention import (
    AttentionScratch,
    MLA_SCALE,
    PreparedSlidingAttention,
    PreparedSparseAttention,
    pack_glm_nsa_records,
)
from hyperflare_recipes.recipes.hy4_flash.config import (
    HIDDEN_SIZE,
    INDEX_HEAD_DIM,
    INDEX_HEADS,
    INDEX_KPOOL,
    KV_LORA_RANK,
    ModelConfig,
    QK_NOPE_HEAD_DIM,
    QK_ROPE_HEAD_DIM,
    Q_LORA_RANK,
    SPARSE_HEADS,
    SWA_HEADS,
    SWA_WINDOW,
    V_HEAD_DIM,
)
from hyperflare_recipes.recipes.hy4_flash.dfly import DFlyScratch, PreparedDFlyLayer
from hyperflare_recipes.recipes.hy4_flash.ihc import ihc_post, ihc_pre
from hyperflare_recipes.recipes.hy4_flash.moe import MoEScratch, PreparedDenseMLP, PreparedLatentMoE
from hyperflare_recipes.recipes.hy4_flash.pool import ContinuationPool, PAGE_TOKENS
from hyperflare_recipes.recipes.hy4_flash.rotary import InterleavedMRoPE
from hyperflare_recipes.recipes.hy4_flash.vision import (
    VISION_HIDDEN,
    VISION_INTERMEDIATE,
    VISION_LAYERS,
    VISION_PIXEL_WIDTH,
    PreparedVision,
    VisionInputs,
    VisionScratch,
)
from hyperflare_recipes.recipes.hy4_flash.weights import Hy4WeightLoader

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="explicit CUDA assignment required",
)

def _quant_snapshot() -> Path:
    raw = os.environ.get("FLARE_HY4_FLASH_SNAPSHOT")
    if not raw:
        pytest.skip("set FLARE_HY4_FLASH_SNAPSHOT to run real-checkpoint Hy4-flash CUDA tests")
    path = Path(raw)
    if not (path / "model.safetensors.index.json").is_file():
        pytest.skip(f"quantized Hy4-flash checkpoint not present at {path}")
    return path


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denom = expected.float().norm().clamp_min(1e-12)
    return float((actual.float() - expected.float()).norm() / denom)


def _synthetic_config(tmp_path: Path) -> ModelConfig:
    engram_hvs = {
        str(lid): [[101, 103, 107, 109, 113, 127, 131, 137], [139, 149, 151, 157, 163, 167, 173, 179]]
        for lid in (2, 12, 22, 32)
    }
    layer_types = [
        "indexed_attention" if i < 2 or (i - 1) % 4 == 0 else "sliding_attention"
        for i in range(42)
    ]
    cfg = {
        "model_type": "hy_v4_flash",
        "dfly_num_hidden_layers": 5,
        "dfly_target_kv_layers": [25, 29, 33, 37, 41],
        "dfly_target_index_layer": [25, 29, 33, 37, 41],
        "dfly_target_index_topk": 2048,
        "dfly_mask_token_id": 120009,
        "text_config": {
            "vocab_size": 120832,
            "hidden_size": HIDDEN_SIZE,
            "num_hidden_layers": 42,
            "hc_mult": 4,
            "hc_eps": 1e-6,
            "hc_magnitude": 2.0,
            "rms_norm_eps": 1e-5,
            "q_lora_rank": Q_LORA_RANK,
            "kv_lora_rank": KV_LORA_RANK,
            "qk_nope_head_dim": QK_NOPE_HEAD_DIM,
            "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
            "v_head_dim": V_HEAD_DIM,
            "num_attention_heads": SPARSE_HEADS,
            "swa_num_attention_heads": SWA_HEADS,
            "sliding_window": SWA_WINDOW,
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
            "hidden_size": VISION_HIDDEN,
            "intermediate_size": VISION_INTERMEDIATE,
            "num_hidden_layers": VISION_LAYERS,
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
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return ModelConfig.load(tmp_path)


def _single_request_metadata(positions: list[int], *, start: int, state_row: int, pages: int, device: torch.device):
    tokens = len(positions)
    m = Metadata.allocate(tokens, 1, pages, device)
    pos_t = torch.tensor(positions, dtype=torch.int64, device=device)
    live_count = int((pos_t >= 0).sum().item())
    m.positions.copy_(pos_t)
    m.requests.zero_()
    m.starts.fill_(start)
    m.state_rows.fill_(state_row)
    m.lengths.fill_(live_count)
    m.query_start_loc[0] = 0
    m.query_start_loc[1] = live_count
    m.page_table[0].copy_(torch.arange(pages, dtype=torch.int32, device=device))
    m.token_page_table.copy_(m.page_table[0].unsqueeze(0).expand(tokens, -1))
    return m


def _random_attention_weights(heads: int, device: torch.device, seed: int) -> dict:
    gen = torch.Generator(device=device).manual_seed(seed)
    bf16 = lambda *s: (torch.randn(s, generator=gen, device=device, dtype=torch.float32) * 0.03).to(torch.bfloat16)
    return {
        "qkv_a": bf16(Q_LORA_RANK + KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN_SIZE),
        "q_b": bf16(heads * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM), Q_LORA_RANK),
        "kv_b": bf16(heads * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK),
        "gate": bf16(heads * V_HEAD_DIM, HIDDEN_SIZE),
        "o_proj": bf16(HIDDEN_SIZE, heads * V_HEAD_DIM),
        "q_norm": torch.ones(Q_LORA_RANK, dtype=torch.bfloat16, device=device),
        "kv_norm": torch.ones(KV_LORA_RANK, dtype=torch.bfloat16, device=device),
        "post_proj_norm": torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16, device=device),
        "sink": torch.randn(heads, generator=gen, device=device, dtype=torch.float32) * 0.5,
    }


def _random_indexer_weights(device: torch.device, seed: int) -> dict:
    gen = torch.Generator(device=device).manual_seed(seed)
    bf16 = lambda *s: (torch.randn(s, generator=gen, device=device, dtype=torch.float32) * 0.04).to(torch.bfloat16)
    fp32 = lambda *s: torch.randn(s, generator=gen, device=device, dtype=torch.float32) * 0.04
    return {
        "query_projection": bf16(INDEX_HEADS * INDEX_HEAD_DIM, Q_LORA_RANK),
        "head_weight_projection": bf16(INDEX_HEADS, HIDDEN_SIZE),
        "token_key_projection": bf16(INDEX_HEAD_DIM, HIDDEN_SIZE),
        "token_key_norm": torch.ones(INDEX_HEAD_DIM, dtype=torch.bfloat16, device=device),
        "key_value_projection": fp32(INDEX_HEAD_DIM, INDEX_HEAD_DIM),
        "key_gate_projection": fp32(INDEX_HEAD_DIM, INDEX_HEAD_DIM),
        "intra_group_ape": fp32(INDEX_KPOOL, INDEX_HEAD_DIM),
        "key_norm": torch.ones(INDEX_HEAD_DIM, dtype=torch.float32, device=device),
    }


def _dequantize_glm_nsa_records(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize 656-byte GLM_NSA records into FP32 NoPE ``[T, 512]`` and RoPE ``[T, 64]``."""
    tokens = packed.shape[0]
    nope_q = packed[:, :KV_LORA_RANK].contiguous().view(torch.float8_e4m3fn).float().view(tokens, 4, 128)
    scales = packed[:, KV_LORA_RANK : KV_LORA_RANK + 16].contiguous().view(torch.float32).view(tokens, 4, 1)
    rope = packed[:, KV_LORA_RANK + 16 :].contiguous().view(torch.bfloat16).float().view(tokens, QK_ROPE_HEAD_DIM)
    return (nope_q * scales).reshape(tokens, KV_LORA_RANK), rope


def _reference_mla_attention(
    x: torch.Tensor,
    weights: dict,
    rotary: InterleavedMRoPE,
    *,
    heads: int,
    indexed: bool,
    window: int | None,
    eps: float = 1e-5,
    quantize_glm_nsa: bool = False,
) -> torch.Tensor:
    tokens = x.shape[0]
    device = x.device
    rms = lambda t, w: (
        (t.float() * torch.rsqrt(t.float().square().mean(-1, keepdim=True) + eps)).to(t.dtype) * w
    )
    proj = F.linear(x, weights["qkv_a"])
    q_r = rms(proj[:, :Q_LORA_RANK], weights["q_norm"])
    kv_nope = rms(proj[:, Q_LORA_RANK : Q_LORA_RANK + KV_LORA_RANK], weights["kv_norm"])
    k_rope = proj[:, Q_LORA_RANK + KV_LORA_RANK :]

    q_raw = F.linear(q_r, weights["q_b"]).view(tokens, heads, QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)
    q_nope, q_rope = q_raw.split((QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM), dim=-1)
    pos = torch.arange(tokens, dtype=torch.int64, device=device)
    cos, sin = rotary.cos_sin(pos, indexed=indexed, dtype=x.dtype)
    q_rope = rotary.apply(q_rope, cos, sin)
    k_rope = rotary.apply(k_rope, cos, sin)

    if quantize_glm_nsa:
        kv_nope_f, k_rope_f = _dequantize_glm_nsa_records(pack_glm_nsa_records(kv_nope, k_rope))
    else:
        kv_nope_f, k_rope_f = kv_nope.float(), k_rope.float()

    kv_b = weights["kv_b"].view(heads, QK_NOPE_HEAD_DIM + V_HEAD_DIM, KV_LORA_RANK)
    k_nope = torch.einsum("td,hnd->thn", kv_nope_f, kv_b[:, :QK_NOPE_HEAD_DIM].float())
    v = torch.einsum("td,hvd->thv", kv_nope_f, kv_b[:, QK_NOPE_HEAD_DIM:].float())
    q = torch.cat((q_nope.float(), q_rope.float()), dim=-1).transpose(0, 1)  # [H, T, 256]
    k = torch.cat((k_nope, k_rope_f[:, None, :].expand(-1, heads, -1)), dim=-1).transpose(0, 1)
    v = v.transpose(0, 1)  # [H, T, 256]

    scores = torch.matmul(q, k.transpose(-2, -1)) * MLA_SCALE
    dist = pos[:, None] - pos[None, :]
    valid = dist >= 0 if window is None else (dist >= 0) & (dist < window)
    scores = scores.masked_fill(~valid[None, :, :], float("-inf"))
    sink = weights["sink"].float()[:, None, None].expand(-1, tokens, 1)
    probs = F.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[..., :-1]
    out = torch.matmul(probs, v).transpose(0, 1).reshape(tokens, -1).to(x.dtype)
    gate = F.linear(x, weights["gate"])
    gated = out * torch.sigmoid(gate)
    return rms(F.linear(gated, weights["o_proj"]), weights["post_proj_norm"])


def _reference_dfly_layer(
    hidden: torch.Tensor,
    hist_cache_records: torch.Tensor,
    dfly_w,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary: InterleavedMRoPE,
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Exact FP32 reference for one DFly2 layer attending over dequantized target history + causal draft block."""
    w = hidden.shape[0]
    rms = lambda t, weight: (
        (t.float() * torch.rsqrt(t.float().square().mean(-1, keepdim=True) + eps)).to(t.dtype) * weight
    )
    a = dfly_w.attention
    normed = rms(hidden, dfly_w.input_norm)
    proj = F.linear(normed, a["qkv_a"])
    q_r = rms(proj[:, :Q_LORA_RANK], a["q_norm"])
    kv_nope = rms(proj[:, Q_LORA_RANK : Q_LORA_RANK + KV_LORA_RANK], a["kv_norm"])
    k_rope = rotary.apply(proj[:, Q_LORA_RANK + KV_LORA_RANK :], cos, sin)

    q_raw = F.linear(q_r, a["q_b"]).view(w, SPARSE_HEADS, QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)
    q_nope, q_rope = q_raw.split((QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM), dim=-1)
    q_rope = rotary.apply(q_rope, cos, sin)
    kv_b = a["kv_b"].view(SPARSE_HEADS, QK_NOPE_HEAD_DIM + V_HEAD_DIM, KV_LORA_RANK).float()
    q_abs_nope = torch.einsum("thn,hnd->thd", q_nope.float(), kv_b[:, :QK_NOPE_HEAD_DIM])
    q_abs = torch.cat((q_abs_nope, q_rope.float()), dim=-1).transpose(0, 1)  # [H, W, 576]

    hist_nope, hist_rope = _dequantize_glm_nsa_records(hist_cache_records)
    k_all = torch.cat(
        (torch.cat((hist_nope, hist_rope), dim=-1), torch.cat((kv_nope.float(), k_rope.float()), dim=-1)),
        dim=0,
    )  # [H_len + W, 576]
    v_all = torch.cat((hist_nope, kv_nope.float()), dim=0)  # [H_len + W, 512]
    h_len = hist_nope.shape[0]

    scores = torch.matmul(q_abs, k_all.T.unsqueeze(0)) * MLA_SCALE  # [H, W, H_len + W]
    causal = torch.ones((w, h_len + w), dtype=torch.bool, device=hidden.device)
    causal[:, h_len:] = torch.tril(torch.ones((w, w), dtype=torch.bool, device=hidden.device))
    scores = scores.masked_fill(~causal[None, :, :], float("-inf"))
    sink = a["sink"].float()[:, None, None].expand(-1, w, 1)
    probs = F.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[..., :-1]
    attended = torch.matmul(probs, v_all.unsqueeze(0))  # [H, W, 512]
    expanded = torch.einsum("htd,hvd->thv", attended, kv_b[:, QK_NOPE_HEAD_DIM:]).reshape(w, -1).to(hidden.dtype)
    gated = expanded * torch.sigmoid(F.linear(normed, a["gate"]))
    hidden = hidden + rms(F.linear(gated, a["o_proj"]), a["post_proj_norm"])
    m_in = rms(hidden, dfly_w.post_attn_norm)
    gate_up = F.linear(m_in, dfly_w.mlp.gate_up)
    g, u = gate_up.chunk(2, dim=-1)
    return hidden + F.linear((F.silu(g.float()) * u.float()).to(hidden.dtype), dfly_w.mlp.down)


@torch.no_grad()
def test_swa_512_ring_wrap_and_post_hoc_sink_cuda(tmp_path):
    """Verify SWA-512 ring wraparound, post-hoc sink scaling, Inductor compile, and graph capture."""
    device = torch.device("cuda:0")
    cfg = _synthetic_config(tmp_path)
    rotary = InterleavedMRoPE(cfg, device)
    weights = _random_attention_weights(SWA_HEADS, device, seed=101)

    total_tokens = 640  # Exceeds 512 so ring buffer wraps around 128 positions.
    chunk = 160
    torch.manual_seed(202)
    x_full = (torch.randn(total_tokens, HIDDEN_SIZE, device=device) * 0.25).to(torch.bfloat16)
    expected = _reference_mla_attention(x_full, weights, rotary, heads=SWA_HEADS, indexed=False, window=SWA_WINDOW)

    pool = ContinuationPool.allocate(cfg, 3, 4, device, dfly=False)
    scratch = AttentionScratch.allocate(chunk, 1, chunk, device)
    meta = _single_request_metadata(list(range(chunk)), start=0, state_row=1, pages=4, device=device)
    arena = ScratchArena(device)

    with RecordedPreparationSession(
        device=device, checkpoint="hy4-swa-unit", workload_schema="swa-ring", numerical_policy="default"
    ) as session:
        session.register_scratch(arena)
        attn = PreparedSlidingAttention(
            layer=2,
            weights=weights,
            source=x_full[:chunk],
            scratch=scratch,
            metadata=meta,
            pool=pool,
            rotary=rotary,
            mode="prefill",
            eps=1e-5,
            session=session,
            name="swa2",
            workspace=arena,
        )
        session.tune()

        # 1. Eager chunked prefill across ring wraparound (4 x 160 = 640 > 512).
        pool.clear_for_preparation()
        outputs = []
        for c in range(4):
            s = c * chunk
            m = _single_request_metadata(list(range(s, s + chunk)), start=s, state_row=1, pages=4, device=device)
            out, _, _ = attn(x_full[s : s + chunk], m)
            outputs.append(out.clone())
        eager_out = torch.cat(outputs, dim=0)
        assert _relative_error(eager_out, expected) < 0.015
        # Inactive rows 0 and 2 must remain untouched.
        ring = pool.window(2)
        assert not ring[0].any() and not ring[2].any()

        # 2. Captured CUDA graph + zeroed-scratch replay on the wrapped final chunk.
        pool.clear_for_preparation()
        for c in range(3):
            s = c * chunk
            m = _single_request_metadata(list(range(s, s + chunk)), start=s, state_row=1, pages=4, device=device)
            attn(x_full[s : s + chunk], m)
        saved_ring = ring.clone()

        static_x = x_full[3 * chunk : 4 * chunk].clone()
        static_m = _single_request_metadata(list(range(3 * chunk, 4 * chunk)), start=3 * chunk, state_row=1, pages=4, device=device)
        static_out = torch.empty_like(outputs[3])

        def step():
            o, _, _ = attn(static_x, static_m)
            static_out.copy_(o)

        compiled = compile_forward(step)
        ring.copy_(saved_ring)
        compiled()
        assert _relative_error(static_out, expected[3 * chunk : 4 * chunk]) < 0.015

        ring.copy_(saved_ring)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        # Zero all scratch buffers before graph replay to verify no stale state dependencies.
        for tensor in scratch.__dict__.values():
            if isinstance(tensor, torch.Tensor):
                tensor.zero_()
        ring.copy_(saved_ring)
        graph.replay()
        torch.cuda.synchronize(device)
        assert _relative_error(static_out, expected[3 * chunk : 4 * chunk]) < 0.015


@torch.no_grad()
def test_sparse_attention_c4_indexer_and_dfly_zero_target_mutation_cuda(tmp_path):
    """Verify 656B GLM_NSA sparse MLA, C4 indexer chunk boundaries, and DFly2 3-way LSE reuse."""
    device = torch.device("cuda:0")
    cfg = _synthetic_config(tmp_path)
    rotary = InterleavedMRoPE(cfg, device)
    attn_w = _random_attention_weights(SPARSE_HEADS, device, seed=303)
    idx_w = _random_indexer_weights(device, seed=404)
    dfly_w = type(
        "DFlyLayerW",
        (),
        {
            "attention": _random_attention_weights(SPARSE_HEADS, device, seed=505),
            "mlp": type(
                "MLPW",
                (),
                {
                    "gate_up": (torch.randn(2 * 1024, HIDDEN_SIZE, device=device) * 0.02).to(torch.bfloat16),
                    "down": (torch.randn(HIDDEN_SIZE, 1024, device=device) * 0.02).to(torch.bfloat16),
                },
            )(),
            "input_norm": torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16, device=device),
            "post_attn_norm": torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16, device=device),
        },
    )()

    # Prefill 33 tokens in irregular chunks crossing C4 group boundaries (3, 2, 6, 9, 13).
    chunks = (3, 2, 6, 9, 13)
    total_tokens = sum(chunks)
    max_chunk = max(chunks)
    torch.manual_seed(606)
    x_full = (torch.randn(total_tokens, HIDDEN_SIZE, device=device) * 0.25).to(torch.bfloat16)
    expected = _reference_mla_attention(
        x_full, attn_w, rotary, heads=SPARSE_HEADS, indexed=True, window=None, quantize_glm_nsa=True
    )
    expected_fp32 = _reference_mla_attention(
        x_full, attn_w, rotary, heads=SPARSE_HEADS, indexed=True, window=None, quantize_glm_nsa=False
    )

    pool = ContinuationPool.allocate(cfg, 3, 4, device, dfly=True)
    scratch = AttentionScratch.allocate(max_chunk, 1, max_chunk, device)
    meta_cap = _single_request_metadata(list(range(max_chunk)), start=0, state_row=1, pages=4, device=device)
    arena = ScratchArena(device)
    draft_width = 6
    dfly_scratch = DFlyScratch.allocate(
        1, draft_width, 1024, HIDDEN_SIZE, 2048, device, shard=TensorParallelShard(0, 1)
    )

    with RecordedPreparationSession(
        device=device, checkpoint="hy4-sparse-dfly-unit", workload_schema="sparse-dfly", numerical_policy="default"
    ) as session:
        session.register_scratch(arena)
        sparse_attn = PreparedSparseAttention(
            layer=25,
            weights=attn_w,
            indexer_weights=idx_w,
            source=x_full[:max_chunk],
            scratch=scratch,
            metadata=meta_cap,
            pool=pool,
            rotary=rotary,
            mode="prefill",
            eps=1e-5,
            session=session,
            name="sparse25",
            workspace=arena,
        )
        dfly_layer = PreparedDFlyLayer(
            draft_index=0,
            target_kv_layer=25,
            weights=dfly_w,
            scratch=dfly_scratch,
            pool=pool,
            rotary=rotary,
            batch=1,
            width=draft_width,
            topk=2048,
            eps=1e-5,
            session=session,
            name="dfly0",
            workspace=arena,
        )
        session.tune()

        pool.clear_for_preparation()
        start = 0
        outputs = []
        last_physical = None
        for count in chunks:
            padded_pos = list(range(start, start + count)) + [-1] * (max_chunk - count)
            padded_x = torch.zeros((max_chunk, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
            padded_x[:count].copy_(x_full[start : start + count])
            m = _single_request_metadata(padded_pos, start=start, state_row=1, pages=4, device=device)
            out, _, physical = sparse_attn(padded_x, m)
            outputs.append(out[:count].clone())
            last_physical = physical[count - 1].clone()
            start += count
        actual = torch.cat(outputs, dim=0)
        # Match exact dequantized GLM_NSA cache reference tightly (verifies base-2 LSE * ln(2) sink;
        # ~1.5% is BF16/FP8 kernel vs FP32 einsum tolerance) and stay within E4M3 noise vs FP32 ref.
        assert _relative_error(actual, expected) < 0.02
        assert _relative_error(actual, expected_fp32) < 0.035
        # Verify C4 pending key state for token 32 (33 % 4 == 1) and clean inactive rows 0 & 2.
        pending = pool.region("layer25.pending_keys")
        assert not pending[0].any() and not pending[2].any()
        assert pending[1, 0].abs().sum().item() > 0.0

        # Snapshot target KV cache before DFly2 draft execution and verify exact 3-way LSE merge + zero mutation.
        target_kv_before = pool.latent_cache(25).clone()
        dfly_in = (torch.randn(draft_width, HIDDEN_SIZE, device=device) * 0.25).to(torch.bfloat16)
        pos_draft = torch.arange(total_tokens, total_tokens + draft_width, dtype=torch.int64, device=device)
        cos_d, sin_d = rotary.cos_sin(pos_draft, indexed=True, dtype=torch.bfloat16)
        sel_slots = torch.full((draft_width, 2048), -1, dtype=torch.int32, device=device)
        sel_slots[:, :total_tokens] = torch.arange(total_tokens, dtype=torch.int32, device=device)[None, :]
        visible = torch.full((draft_width,), total_tokens, dtype=torch.int32, device=device)
        sel_lens = torch.full((draft_width,), total_tokens, dtype=torch.int32, device=device)
        causal_mask = torch.tril(torch.ones((draft_width, draft_width), dtype=torch.bool, device=device))

        dfly_out = dfly_layer(dfly_in, cos_d, sin_d, sel_slots, visible, sel_lens, causal_mask)
        dfly_ref = _reference_dfly_layer(
            dfly_in, target_kv_before.view(-1, 656)[:total_tokens], dfly_w, cos_d, sin_d, rotary
        )
        assert torch.isfinite(dfly_out).all()
        assert _relative_error(dfly_out, dfly_ref) < 0.02
        assert torch.equal(pool.latent_cache(25), target_kv_before), "DFly2 mutated target KV cache!"


@torch.no_grad()
def test_real_checkpoint_layers_and_vision_cuda():
    """Verify real Hy4-flash-NVFP4 layers (0 dense, 1 sparse LatentMoE+CSF, 2 SWA, ViT) on SM120."""
    snapshot = _quant_snapshot()

    device = torch.device("cuda:0")
    shard = TensorParallelShard(0, 1)
    manifest = inspect_indexed_checkpoint(snapshot)
    cfg = ModelConfig.load(manifest.metadata_path, manifest=manifest)
    rotary = InterleavedMRoPE(cfg, device)

    # Load layers 0, 1, 2 and ViT via Hy4WeightLoader with CSF expert scale compression.
    with Hy4WeightLoader(
        manifest,
        device,
        config=cfg,
        shard=shard,
        dfly=False,
        vision=True,
        expert_scale_compression="csf",
    ) as loader:
        l0 = loader.load_layer(0)
        l1 = loader.load_layer(1)
        l2 = loader.load_layer(2)
        vit_packed = loader.load_vision()
        vit_w = {name: loader.load_tensor(name) for name in sorted(loader._sources) if name.startswith("vit.")}
        scale_scratch = loader.scale_scratch

    tokens = 64
    pool = ContinuationPool.allocate(cfg, 2, 4, device, dfly=False)
    attn_scratch = AttentionScratch.allocate(tokens, 1, tokens, device)
    moe_scratch = MoEScratch.allocate(tokens, device)
    meta = _single_request_metadata(list(range(tokens)), start=0, state_row=1, pages=4, device=device)
    arena = ScratchArena(device)
    torch.manual_seed(909)
    streams = (torch.randn(tokens, 4, HIDDEN_SIZE, device=device) * 0.2).to(torch.bfloat16)
    proto_x = streams[:, 0].contiguous()
    hc_mixes = torch.zeros(tokens, 8, dtype=torch.float32, device=device)

    with RecordedPreparationSession(
        device=device, checkpoint="hy4-real-layers", workload_schema="real-layers", numerical_policy="default"
    ) as session:
        session.register_scratch(arena)
        hc_projs = {}
        for idx, lw in ((0, l0), (1, l1), (2, l2)):
            hc_projs[(idx, "attn")] = UnquantizedProjection(
                streams.view(tokens, -1), lw.residual["attn_fn"], hc_mixes,
                session=session, name=f"l{idx}.hc_attn",
            ).functional(session=session)
            hc_projs[(idx, "mlp")] = UnquantizedProjection(
                streams.view(tokens, -1), lw.residual["mlp_fn"], hc_mixes,
                session=session, name=f"l{idx}.hc_mlp",
            ).functional(session=session)
        attn0 = PreparedSparseAttention(
            0, l0.attention, l0.indexer, proto_x, attn_scratch, meta, pool, rotary,
            mode="prefill", eps=1e-5, session=session, name="l0.attn", workspace=arena,
        )
        mlp0 = PreparedDenseMLP(l0.mlp, proto_x, moe_scratch, eps=1e-5, session=session, name="l0.mlp", workspace=arena)
        attn1 = PreparedSparseAttention(
            1, l1.attention, l1.indexer, proto_x, attn_scratch, meta, pool, rotary,
            mode="prefill", eps=1e-5, session=session, name="l1.attn", workspace=arena,
        )
        moe1 = PreparedLatentMoE(l1.mlp, proto_x, moe_scratch, config=cfg, mode="prefill", session=session, name="l1.moe", workspace=arena)
        attn2 = PreparedSlidingAttention(
            2, l2.attention, proto_x, attn_scratch, meta, pool, rotary,
            mode="prefill", eps=1e-5, session=session, name="l2.attn", workspace=arena,
        )
        moe2 = PreparedLatentMoE(l2.mlp, proto_x, moe_scratch, config=cfg, mode="prefill", session=session, name="l2.moe", workspace=arena)
        session.tune()

        def run_three_layers(x_streams: torch.Tensor) -> torch.Tensor:
            h = x_streams
            for idx, layer_w, attn_mod, ffn_mod in ((0, l0, attn0, mlp0), (1, l1, attn1, moe1), (2, l2, attn2, moe2)):
                r = layer_w.residual
                attn_mix = hc_projs[(idx, "attn")](h.reshape(tokens, -1))
                post_a, x_a = ihc_pre(h, r["attn_fn"], r["attn_base"], r["attn_scale"], rms_eps=1e-5, hc_eps=1e-6, hc_magnitude=2.0, mixes=attn_mix)
                x_a = (x_a.float() * torch.rsqrt(x_a.float().square().mean(-1, keepdim=True) + 1e-5)).to(x_a.dtype) * r["input_norm"]
                out_a = attn_mod(x_a, meta)[0]
                h = ihc_post(h, out_a, post_a)
                mlp_mix = hc_projs[(idx, "mlp")](h.reshape(tokens, -1))
                post_m, x_m = ihc_pre(h, r["mlp_fn"], r["mlp_base"], r["mlp_scale"], rms_eps=1e-5, hc_eps=1e-6, hc_magnitude=2.0, mixes=mlp_mix)
                x_m = (x_m.float() * torch.rsqrt(x_m.float().square().mean(-1, keepdim=True) + 1e-5)).to(x_m.dtype) * r["post_attn_norm"]
                out_m = ffn_mod(x_m)
                h = ihc_post(h, out_m, post_m)
            return h

        pool.clear_for_preparation()
        eager_out = run_three_layers(streams).clone()
        assert torch.isfinite(eager_out).all()
        assert eager_out.abs().mean().item() > 0.01

        # Measure native top-16 MoE atomic-add repeatability on a fresh pool so compiled and
        # captured graph checks enforce parity within the hardware reduction noise floor.
        pool.clear_for_preparation()
        eager_repeat = run_three_layers(streams).clone()
        repeat_err = _relative_error(eager_repeat, eager_out)
        bound = max(0.005, 2.0 * repeat_err + 0.002)

        compiled = compile_forward(lambda: run_three_layers(streams))
        pool.clear_for_preparation()
        compiled_out = compiled().clone()
        assert _relative_error(compiled_out, eager_out) < bound

        static_out = torch.empty_like(eager_out)
        pool.clear_for_preparation()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out.copy_(compiled())
        for t in scale_scratch:
            t.zero_()
        pool.clear_for_preparation()
        graph.replay()
        torch.cuda.synchronize(device)
        assert _relative_error(static_out, eager_out) < bound

    # Verify real SigLIP2 vision encoder + vit.perceive projector against direct Conv2d reference.
    v_patches = 64  # 8 x 8 patch grid -> 4 x 4 = 16 merged features
    v_inputs = VisionInputs.allocate(device, patches=v_patches)
    v_scratch = VisionScratch.allocate(v_patches, device)
    torch.manual_seed(1111)
    pixels = torch.randn(v_patches, VISION_PIXEL_WIDTH, dtype=torch.float32, device=device) * 0.2
    v_inputs.vision_pixel_values.copy_(pixels)
    # 8x8 grid with index-0 learned position weights exercising PreparedVision against Conv2d reference.
    v_inputs.vision_interpolation_indices.zero_()
    v_inputs.vision_interpolation_weights[:, 0].fill_(1.0)
    v_inputs.vision_cu_seqlens[0] = 0
    v_inputs.vision_cu_seqlens[1] = v_patches
    v_inputs.vision_num_patches.fill_(v_patches)
    v_inputs.vision_num_features.fill_(v_patches // 4)
    # 2x2 merge groups over an 8x8 spatial patch grid
    merge = []
    for r in range(0, 8, 2):
        for c in range(0, 8, 2):
            merge.append([r * 8 + c, r * 8 + c + 1, (r + 1) * 8 + c, (r + 1) * 8 + c + 1])
    v_inputs.vision_merge_indices.copy_(torch.tensor(merge, dtype=torch.int32, device=device))

    arena_v = ScratchArena(device)
    with RecordedPreparationSession(
        device=device, checkpoint="hy4-real-vision", workload_schema="real-vision", numerical_policy="default"
    ) as session:
        session.register_scratch(arena_v)
        vis = PreparedVision(vit_packed, v_scratch, v_inputs, config=cfg, session=session, workspace=arena_v, name="vis64")
        session.tune()
        vis_out = vis(
            v_inputs.vision_pixel_values,
            v_inputs.vision_interpolation_indices,
            v_inputs.vision_interpolation_weights,
            v_inputs.vision_cu_seqlens,
            v_inputs.vision_merge_indices,
            v_inputs.vision_num_features,
        )
        assert vis_out.shape == (16, HIDDEN_SIZE) and torch.isfinite(vis_out).all()
        assert vis_out.abs().mean().item() > 0.01

        # Exact PyTorch Conv2d + 27-layer SigLIP2 + Conv2d vit.perceive reference.
        pix_4d = pixels.to(torch.bfloat16).view(v_patches, 3, 16, 16)
        ref_h = F.conv2d(
            pix_4d,
            vit_w["vit.embeddings.patch_embedding.weight"],
            vit_w["vit.embeddings.patch_embedding.bias"],
            stride=16,
        ).flatten(1)
        ref_h = ref_h + vit_w["vit.embeddings.position_embedding.weight"][1:2]
        for i in range(VISION_LAYERS):
            p = f"vit.layers.{i}"
            n1 = F.layer_norm(
                ref_h, (VISION_HIDDEN,), vit_w[f"{p}.input_layernorm.weight"], vit_w[f"{p}.input_layernorm.bias"], eps=cfg.vision.layer_norm_eps
            )
            q = F.linear(n1, vit_w[f"{p}.self_attn.q_proj.weight"], vit_w[f"{p}.self_attn.q_proj.bias"]).view(v_patches, 16, 72).transpose(0, 1)
            k = F.linear(n1, vit_w[f"{p}.self_attn.k_proj.weight"], vit_w[f"{p}.self_attn.k_proj.bias"]).view(v_patches, 16, 72).transpose(0, 1)
            v = F.linear(n1, vit_w[f"{p}.self_attn.v_proj.weight"], vit_w[f"{p}.self_attn.v_proj.bias"]).view(v_patches, 16, 72).transpose(0, 1)
            a = F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=False).squeeze(0).transpose(0, 1).reshape(v_patches, VISION_HIDDEN)
            o_w = vit_w[f"{p}.self_attn.o_proj.weight"]
            if o_w.dtype == torch.float8_e4m3fn:
                o_s = vit_w[f"{p}.self_attn.o_proj.weight_scale"].view(torch.uint8).to(torch.float32) - 127.0
                o_w = (o_w.float().view(VISION_HIDDEN, -1, 32) * torch.exp2(o_s).unsqueeze(-1)).reshape(VISION_HIDDEN, VISION_HIDDEN).to(torch.bfloat16)
            ref_h = ref_h + F.linear(a, o_w, vit_w[f"{p}.self_attn.o_proj.bias"])
            n2 = F.layer_norm(
                ref_h, (VISION_HIDDEN,), vit_w[f"{p}.post_attention_layernorm.weight"], vit_w[f"{p}.post_attention_layernorm.bias"], eps=cfg.vision.layer_norm_eps
            )
            ff = F.gelu(F.linear(n2, vit_w[f"{p}.mlp.dense_h_to_4h.weight"], vit_w[f"{p}.mlp.dense_h_to_4h.bias"]))
            ref_h = ref_h + F.linear(ff, vit_w[f"{p}.mlp.dense_4h_to_h.weight"], vit_w[f"{p}.mlp.dense_4h_to_h.bias"])
        rf = ref_h.float()
        ref_n = (rf * torch.rsqrt(rf.square().mean(-1, keepdim=True) + cfg.vision.rms_norm_eps)).to(ref_h.dtype) * vit_w["vit.perceive.norm.weight"]
        ref_4d = ref_n.view(1, 8, 8, VISION_HIDDEN).permute(0, 3, 1, 2).contiguous()
        ref_p = F.conv2d(ref_4d, vit_w["vit.perceive.proj.0.weight"], vit_w["vit.perceive.proj.0.bias"], stride=2)
        ref_p = F.gelu(ref_p)
        ref_p = F.conv2d(ref_p, vit_w["vit.perceive.proj.2.weight"], vit_w["vit.perceive.proj.2.bias"], stride=1)
        ref_out = ref_p.permute(0, 2, 3, 1).reshape(16, HIDDEN_SIZE)
        # Isolated vit.perceive Linear equivalence to Conv2d(k=2, s=2) -> GELU -> Conv2d(k=1).
        iso_merged = ref_n[v_inputs.vision_merge_indices.long()].transpose(1, 2).reshape(16, 4 * VISION_HIDDEN).contiguous()
        iso_out = vis.perceive_conv2(F.gelu(vis.perceive_conv0(iso_merged)))
        assert _relative_error(iso_out, ref_out) < 0.01
        # End-to-end 27-layer SigLIP2 + perceive parity (o_proj is MXFP8 in the quantized checkpoint).
        cos_sim = float(F.cosine_similarity(vis_out.flatten().float(), ref_out.flatten().float(), dim=0))
        assert cos_sim > 0.995
        assert _relative_error(vis_out, ref_out) < 0.08


@torch.no_grad()
def test_online_mxfp8_loader_scopes_and_vision_cuda():
    snapshot = _quant_snapshot()

    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight

    device = torch.device("cuda:0")
    shard = TensorParallelShard(0, 1)
    manifest = inspect_indexed_checkpoint(snapshot)
    cfg = ModelConfig.load(manifest.metadata_path, manifest=manifest)

    # scope='draft' converts DFly2 dense weights to MXFP8 while leaving target and vision BF16.
    with Hy4WeightLoader(
        manifest,
        device,
        config=cfg,
        shard=shard,
        dfly=True,
        vision=True,
        expert_scale_compression="csf",
        online_mxfp8=True,
        online_mxfp8_scope="draft",
    ) as loader:
        l0_draft = loader.load_layer(0)
        dfly_draft = loader.load_dfly()
        assert isinstance(l0_draft.attention["qkv_a"], torch.Tensor)
        assert isinstance(l0_draft.mlp.mlp.gate_up, torch.Tensor)
        assert isinstance(dfly_draft.layers[0].attention["qkv_a"], MXFP8LinearWeight)
        assert isinstance(dfly_draft.layers[0].attention["kv_b"], torch.Tensor)
        assert isinstance(dfly_draft.layers[0].mlp.gate_up, MXFP8LinearWeight)
        assert isinstance(dfly_draft.mtp_eh_proj, MXFP8LinearWeight)
        assert isinstance(dfly_draft.hc_gate_up, MXFP8LinearWeight)
    del l0_draft, dfly_draft
    torch.cuda.empty_cache()

    # scope='all' converts eligible target, DFly2, and SigLIP2 dense weights while preserving
    # kv_b_proj, sparse indexer projections, and vit.fc2 (4304 % 32 != 0) in BF16.
    with Hy4WeightLoader(
        manifest,
        device,
        config=cfg,
        shard=shard,
        dfly=False,
        vision=True,
        expert_scale_compression="csf",
        online_mxfp8=True,
        online_mxfp8_scope="all",
    ) as loader:
        l0 = loader.load_layer(0)
        l1 = loader.load_layer(1)
        vis_mx = loader.load_vision()
        assert isinstance(l0.attention["qkv_a"], MXFP8LinearWeight)
        assert isinstance(l0.attention["kv_b"], torch.Tensor)
        assert l0.attention["fused_indexer_projs"] is False
        assert isinstance(l0.indexer["token_key_projection"], torch.Tensor)
        assert isinstance(l0.mlp.mlp.gate_up, MXFP8LinearWeight)
        assert isinstance(l1.mlp.down_proj, MXFP8LinearWeight)
        assert isinstance(l1.mlp.shared.gate_up, MXFP8LinearWeight)
        assert l1.mlp.experts_a4_prefill is False
        assert isinstance(l1.engram["kv_proj"], MXFP8LinearWeight)
        b0 = vis_mx["blocks"][0]
        assert isinstance(b0["qkv_w"], MXFP8LinearWeight)
        assert isinstance(b0["o_w"], MXFP8LinearWeight)
        assert isinstance(b0["fc1_w"], MXFP8LinearWeight)
        assert isinstance(b0["fc2_w"], torch.Tensor)
    del l0, l1

    # Execute PreparedVision end-to-end with online-MXFP8 SigLIP2 weights.
    v_patches = 64
    v_inputs = VisionInputs.allocate(device, patches=v_patches)
    v_scratch = VisionScratch.allocate(v_patches, device)
    torch.manual_seed(2026)
    v_inputs.vision_pixel_values.copy_(torch.randn(v_patches, VISION_PIXEL_WIDTH, device=device) * 0.2)
    v_inputs.vision_interpolation_indices.zero_()
    v_inputs.vision_interpolation_weights[:, 0].fill_(1.0)
    v_inputs.vision_cu_seqlens[0] = 0
    v_inputs.vision_cu_seqlens[1] = v_patches
    v_inputs.vision_num_patches.fill_(v_patches)
    v_inputs.vision_num_features.fill_(v_patches // 4)
    merge = [
        [r * 8 + c, r * 8 + c + 1, (r + 1) * 8 + c, (r + 1) * 8 + c + 1]
        for r in range(0, 8, 2)
        for c in range(0, 8, 2)
    ]
    v_inputs.vision_merge_indices.copy_(torch.tensor(merge, dtype=torch.int32, device=device))

    arena_v = ScratchArena(device)
    with RecordedPreparationSession(
        device=device, checkpoint="hy4-online-mxfp8-vision", workload_schema="online-mxfp8-vision", numerical_policy="default"
    ) as session:
        session.register_scratch(arena_v)
        vis = PreparedVision(vis_mx, v_scratch, v_inputs, config=cfg, session=session, workspace=arena_v, name="vis64_mx")
        session.tune()
        vis_out = vis(
            v_inputs.vision_pixel_values,
            v_inputs.vision_interpolation_indices,
            v_inputs.vision_interpolation_weights,
            v_inputs.vision_cu_seqlens,
            v_inputs.vision_merge_indices,
            v_inputs.vision_num_features,
        )
        assert vis_out.shape == (16, HIDDEN_SIZE) and torch.isfinite(vis_out).all()
        assert vis_out.abs().mean().item() > 0.01


@torch.no_grad()
def test_c4_indexer_above_2048_and_dfly_page_relocation_cuda(tmp_path):
    """Verify >2,048-token C4 top-k selection, _save_dfly_indices handoff, and DFly2 page relocation on CUDA."""
    from types import SimpleNamespace
    from hyperflare_recipes.recipes.hy4_flash.decoder import PreparedDecoder
    from hyperflare_recipes.recipes.hy4_flash.dfly import resolve_logical_slots

    device = torch.device("cuda:0")
    cfg = _synthetic_config(tmp_path)
    rotary = InterleavedMRoPE(cfg, device)
    attn_w = _random_attention_weights(SPARSE_HEADS, device, seed=707)
    idx_w = _random_indexer_weights(device, seed=808)
    dfly_w = type(
        "DFlyLayerW",
        (),
        {
            "attention": _random_attention_weights(SPARSE_HEADS, device, seed=909),
            "mlp": type(
                "MLPW",
                (),
                {
                    "gate_up": (torch.randn(2 * 1024, HIDDEN_SIZE, device=device) * 0.02).to(torch.bfloat16),
                    "down": (torch.randn(HIDDEN_SIZE, 1024, device=device) * 0.02).to(torch.bfloat16),
                },
            )(),
            "input_norm": torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16, device=device),
            "post_attn_norm": torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16, device=device),
        },
    )()

    # 2,115 tokens = 528 complete C4 groups (> 511 top-k pool cap) + 3-token partial tail.
    total_tokens = 2115
    chunk_size = 512
    num_pages_needed = (total_tokens + PAGE_TOKENS - 1) // PAGE_TOKENS  # 9 pages
    total_pages = num_pages_needed * 2 + 2  # room for original pages + disjoint relocated pages
    torch.manual_seed(1234)
    x_full = (torch.randn(total_tokens, HIDDEN_SIZE, device=device) * 0.2).to(torch.bfloat16)

    pool = ContinuationPool.allocate(cfg, 3, total_pages, device, dfly=True)
    scratch = AttentionScratch.allocate(chunk_size, 1, chunk_size, device)
    meta_cap = _single_request_metadata(list(range(chunk_size)), start=0, state_row=1, pages=total_pages, device=device)
    arena = ScratchArena(device)
    draft_width = 4
    dfly_scratch = DFlyScratch.allocate(
        2, draft_width, 1024, HIDDEN_SIZE, 2048, device, shard=TensorParallelShard(0, 1)
    )

    with RecordedPreparationSession(
        device=device, checkpoint="hy4-c4-2048-reloc", workload_schema="c4-2048-reloc", numerical_policy="default"
    ) as session:
        session.register_scratch(arena)
        sparse_attn = PreparedSparseAttention(
            layer=25,
            weights=attn_w,
            indexer_weights=idx_w,
            source=x_full[:chunk_size],
            scratch=scratch,
            metadata=meta_cap,
            pool=pool,
            rotary=rotary,
            mode="prefill",
            eps=1e-5,
            session=session,
            name="sparse25_long",
            workspace=arena,
        )
        dfly_layer = PreparedDFlyLayer(
            draft_index=0,
            target_kv_layer=25,
            weights=dfly_w,
            scratch=dfly_scratch,
            pool=pool,
            rotary=rotary,
            batch=2,
            width=draft_width,
            topk=2048,
            eps=1e-5,
            session=session,
            name="dfly0_long",
            workspace=arena,
        )
        session.tune()

        pool.clear_for_preparation()
        start = 0
        last_pools = None
        last_meta = None
        last_count = 0
        while start < total_tokens:
            count = min(chunk_size, total_tokens - start)
            padded_pos = list(range(start, start + count)) + [-1] * (chunk_size - count)
            padded_x = torch.zeros((chunk_size, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
            padded_x[:count].copy_(x_full[start : start + count])
            m = _single_request_metadata(padded_pos, start=start, state_row=1, pages=total_pages, device=device)
            out, _, pools = sparse_attn(padded_x, m)
            assert torch.isfinite(out[:count]).all()
            last_pools = pools.clone()
            last_meta = m
            last_count = count
            start += count

        # Save logical token indices via PreparedDecoder._save_dfly_indices for the committed prefix.
        dec = object.__new__(PreparedDecoder)
        dec.dfly_index_layers = (25, 25, 25, 25, 25)
        dec.dfly_topk = 2048
        dec.pool = pool
        save_meta = SimpleNamespace(
            starts=last_meta.starts,
            query_start_loc=last_meta.query_start_loc,
            positions=last_meta.positions,
            state_rows=last_meta.state_rows,
        )
        dec._save_dfly_indices(torch.tensor([last_count], dtype=torch.int32, device=device), {25: last_pools}, save_meta)

        saved_logical = pool.region("dfly_target_indices")[1, 0].clone()  # [2048]
        # >2,048 tokens crosses top-k cap: at most 511*4 + 3 = 2,047 valid logical indices, including partial tail.
        valid_logical = saved_logical[saved_logical >= 0]
        assert valid_logical.numel() == 2047
        assert valid_logical[-3:].tolist() == [2112, 2113, 2114]

        # Resolve through original contiguous page table [0..num_pages_needed-1] alongside an inactive row.
        orig_pages = torch.full((2, num_pages_needed), -1, dtype=torch.int32, device=device)
        orig_pages[0] = torch.arange(num_pages_needed, dtype=torch.int32, device=device)
        active = torch.tensor([True, False], device=device)
        logical_batch = torch.stack((saved_logical, saved_logical), dim=0)
        orig_slots = resolve_logical_slots(logical_batch, orig_pages, active)
        assert torch.equal(orig_slots[0, :2047], valid_logical)
        assert torch.all(orig_slots[1] == -1)

        # Run DFly2 draft on original pages.
        torch.manual_seed(4321)
        dfly_in = (torch.randn(2 * draft_width, HIDDEN_SIZE, device=device) * 0.25).to(torch.bfloat16)
        pos_draft = torch.arange(total_tokens, total_tokens + draft_width, dtype=torch.int64, device=device)
        pos_batch = torch.cat((pos_draft, torch.zeros_like(pos_draft)), dim=0)
        cos_d, sin_d = rotary.cos_sin(pos_batch, indexed=True, dtype=torch.bfloat16)
        visible = torch.tensor([total_tokens] * draft_width + [0] * draft_width, dtype=torch.int32, device=device)
        causal_mask = torch.tril(torch.ones((draft_width, draft_width), dtype=torch.bool, device=device))

        orig_sel_len = (orig_slots >= 0).sum(dim=-1, dtype=torch.int32)[:, None].expand(2, draft_width).reshape(-1)
        orig_slots_rep = orig_slots[:, None, :].expand(2, draft_width, 2048).reshape(2 * draft_width, 2048).contiguous()
        out_uninterrupted = dfly_layer(dfly_in, cos_d, sin_d, orig_slots_rep, visible, orig_sel_len, causal_mask).clone()

        # Relocate every 256-token KV page onto disjoint reversed physical page IDs and clear original pages.
        reloc_map = torch.arange(total_pages - 1, total_pages - 1 - num_pages_needed, -1, dtype=torch.int32, device=device)
        kv_pages = pool.pages["layer25.latent_page"]
        kv_records = kv_pages.view(-1, pool.latent_cache(25).shape[-1])
        expected_selected_kv = kv_records[orig_slots[0, :2047].long()].clone()
        for logical_page in range(num_pages_needed):
            kv_pages[int(reloc_map[logical_page].item())].copy_(kv_pages[logical_page])
            kv_pages[logical_page].zero_()

        reloc_pages = torch.full((2, num_pages_needed), -1, dtype=torch.int32, device=device)
        reloc_pages[0] = reloc_map
        reloc_slots = resolve_logical_slots(logical_batch, reloc_pages, active)
        assert torch.all(reloc_slots[1] == -1)
        assert torch.equal(kv_records[reloc_slots[0, :2047].long()], expected_selected_kv)
        reloc_sel_len = (reloc_slots >= 0).sum(dim=-1, dtype=torch.int32)[:, None].expand(2, draft_width).reshape(-1)
        reloc_slots_rep = reloc_slots[:, None, :].expand(2, draft_width, 2048).reshape(2 * draft_width, 2048).contiguous()
        out_relocated = dfly_layer(dfly_in, cos_d, sin_d, reloc_slots_rep, visible, reloc_sel_len, causal_mask).clone()

        assert torch.isfinite(out_relocated[:draft_width]).all()
        assert _relative_error(out_relocated[:draft_width], out_uninterrupted[:draft_width]) < 1e-4


@torch.no_grad()
def test_swa_decode_multistream_inactive_leading_lane_ring_commit_cuda(tmp_path):
    """Verify width-1 SWA decode ring staging under B>1 when an earlier lane is inactive."""
    from hyperflare_recipes.components.ring_commit import PreparedRingCommit

    device = torch.device("cuda:0")
    cfg = _synthetic_config(tmp_path)
    rotary = InterleavedMRoPE(cfg, device)
    weights = _random_attention_weights(SWA_HEADS, device, seed=2101)

    # Three distinct streams (one past 512-token ring wraparound) on distinct continuation rows 1, 2, 3.
    prefix_lens = (19, 520, 67)
    max_prefill = max(prefix_lens)
    torch.manual_seed(2102)
    prefixes = [
        (torch.randn(n, HIDDEN_SIZE, device=device) * 0.25).to(torch.bfloat16) for n in prefix_lens
    ]
    step_x = [
        (torch.randn(2, HIDDEN_SIZE, device=device) * 0.25).to(torch.bfloat16) for _ in prefix_lens
    ]

    def _run_isolated(req_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        n = prefix_lens[req_idx]
        pool = ContinuationPool.allocate(cfg, 3, 4, device, dfly=False)
        arena = ScratchArena(device)
        p_scratch = AttentionScratch.allocate(n, 1, n, device)
        d_scratch = AttentionScratch.allocate(1, 1, 1, device)
        p_meta = _single_request_metadata(list(range(n)), start=0, state_row=1, pages=4, device=device)
        d_meta = Metadata.pack(
            starts=torch.tensor([n], dtype=torch.int64, device=device),
            lengths=torch.tensor([1], dtype=torch.int32, device=device),
            state_rows=torch.tensor([1], dtype=torch.int64, device=device),
            pages=torch.arange(4, dtype=torch.int32, device=device).unsqueeze(0),
            tokens=torch.ones((1, 1), dtype=torch.int64, device=device),
            live_rows=torch.tensor([1], dtype=torch.int32, device=device),
            token_capacity=1,
            sample_count=1,
            max_query_tokens=1,
        )
        with RecordedPreparationSession(
            device=device, checkpoint=f"hy4-swa-iso-{req_idx}", workload_schema="swa-iso", numerical_policy="default"
        ) as session:
            session.register_scratch(arena)
            prefill = PreparedSlidingAttention(
                2, weights, prefixes[req_idx], p_scratch, p_meta, pool, rotary,
                mode="prefill", eps=1e-5, session=session, name="p", workspace=arena,
            )
            decode = PreparedSlidingAttention(
                2, weights, step_x[req_idx][:1], d_scratch, d_meta, pool, rotary,
                mode="decode", eps=1e-5, session=session, name="d", workspace=arena,
            )
            ring_commit = PreparedRingCommit(
                pool.reserved_backing,
                [pool.layout.region("layer2.window").offset],
                d_meta.state_rows,
                d_meta.starts,
                token_capacity=1,
                window_tokens=SWA_WINDOW,
                record_bytes=576 * 2,
            )
            staging_shape = tuple(ring_commit.staging.shape)
            ring_commit.functional()
            session.tune()

            pool.clear_for_preparation()
            prefill(prefixes[req_idx], p_meta)
            # Step 1: decode + ring commit
            staging = torch.empty(staging_shape, dtype=torch.uint8, device=device)
            decode(step_x[req_idx][:1], d_meta, window_cache=staging[0])
            ring_commit(d_meta.lengths, staging=staging, rows=d_meta.state_rows, starts=d_meta.starts)
            ring_after_step1 = pool.window(2)[1].clone()
            # Step 2: verify next-step attention observes the committed ring
            d_meta2 = Metadata.pack(
                starts=torch.tensor([n + 1], dtype=torch.int64, device=device),
                lengths=torch.tensor([1], dtype=torch.int32, device=device),
                state_rows=torch.tensor([1], dtype=torch.int64, device=device),
                pages=torch.arange(4, dtype=torch.int32, device=device).unsqueeze(0),
                tokens=torch.ones((1, 1), dtype=torch.int64, device=device),
                live_rows=torch.tensor([1], dtype=torch.int32, device=device),
                token_capacity=1,
                sample_count=1,
                max_query_tokens=1,
            )
            out_step2, _, _ = decode(step_x[req_idx][1:2], d_meta2, window_cache=staging[0])
            return ring_after_step1, out_step2.clone()

    iso_rings, iso_out2 = zip(*(_run_isolated(i) for i in range(3)), strict=True)

    # Multi-stream B=3 width-1 decode with lane 0 inactive ([0, 1, 1]) and then ([0, 1, 0]),
    # testing both eager and CUDA-graph-captured execution.
    pool = ContinuationPool.allocate(cfg, 5, 12, device, dfly=False)
    arena = ScratchArena(device)
    p_scratch = AttentionScratch.allocate(max_prefill, 1, max_prefill, device)
    d_scratch = AttentionScratch.allocate(3, 3, 1, device)
    pages_b3 = torch.arange(12, dtype=torch.int32, device=device).view(3, 4)
    d_meta_cap = Metadata.pack(
        starts=torch.tensor(prefix_lens, dtype=torch.int64, device=device),
        lengths=torch.tensor([1, 1, 1], dtype=torch.int32, device=device),
        state_rows=torch.tensor([1, 2, 3], dtype=torch.int64, device=device),
        pages=pages_b3,
        tokens=torch.ones((3, 1), dtype=torch.int64, device=device),
        live_rows=torch.tensor([3], dtype=torch.int32, device=device),
        token_capacity=3,
        sample_count=3,
        max_query_tokens=1,
    )
    p_meta_cap = _single_request_metadata(list(range(max_prefill)), start=0, state_row=1, pages=4, device=device)
    with RecordedPreparationSession(
        device=device, checkpoint="hy4-swa-b3-decode", workload_schema="swa-b3-decode", numerical_policy="default"
    ) as session:
        session.register_scratch(arena)
        prefill = PreparedSlidingAttention(
            2, weights, torch.zeros((max_prefill, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
            p_scratch, p_meta_cap, pool, rotary,
            mode="prefill", eps=1e-5, session=session, name="p3", workspace=arena,
        )
        decode = PreparedSlidingAttention(
            2, weights, torch.zeros((3, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
            d_scratch, d_meta_cap, pool, rotary,
            mode="decode", eps=1e-5, session=session, name="d3", workspace=arena,
        )
        ring_commit = PreparedRingCommit(
            pool.reserved_backing,
            [pool.layout.region("layer2.window").offset],
            d_meta_cap.state_rows,
            d_meta_cap.starts,
            token_capacity=1,
            window_tokens=SWA_WINDOW,
            record_bytes=576 * 2,
        )
        staging_shape = tuple(ring_commit.staging.shape)
        ring_commit.functional()
        session.tune()

        def _seed_prefixes():
            pool.clear_for_preparation()
            for r, n in enumerate(prefix_lens):
                padded_x = torch.zeros((max_prefill, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
                padded_x[:n].copy_(prefixes[r])
                padded_pos = list(range(n)) + [-1] * (max_prefill - n)
                pm = _single_request_metadata(padded_pos, start=0, state_row=r + 1, pages=4, device=device)
                prefill(padded_x, pm)

        for use_graph in (False, True):
            _seed_prefixes()
            lane0_ring_before = pool.window(2)[1].clone()
            # Packed decode step with lengths=[0, 1, 1]: packed idx 0 is req 1, packed idx 1 is req 2, idx 2 is dead.
            packed_x1 = torch.zeros((3, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
            packed_x1[0].copy_(step_x[1][0])
            packed_x1[1].copy_(step_x[2][0])
            m1 = Metadata.pack(
                starts=torch.tensor(prefix_lens, dtype=torch.int64, device=device),
                lengths=torch.tensor([0, 1, 1], dtype=torch.int32, device=device),
                state_rows=torch.tensor([1, 2, 3], dtype=torch.int64, device=device),
                pages=pages_b3,
                tokens=torch.ones((3, 1), dtype=torch.int64, device=device),
                live_rows=torch.tensor([2], dtype=torch.int32, device=device),
                token_capacity=3,
                sample_count=3,
                max_query_tokens=1,
            )
            if use_graph:
                saved_backing = pool.backing.clone()

                def _step():
                    staging = torch.empty(staging_shape, dtype=torch.uint8, device=device)
                    decode(packed_x1, m1, window_cache=staging[0])
                    ring_commit(m1.lengths, staging=staging, rows=m1.state_rows, starts=m1.starts)

                compiled = compile_forward(_step)
                pool.backing.copy_(saved_backing)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    compiled()
                pool.backing.copy_(saved_backing)
                graph.replay()
                torch.cuda.synchronize(device)
            else:
                staging = torch.empty(staging_shape, dtype=torch.uint8, device=device)
                decode(packed_x1, m1, window_cache=staging[0])
                ring_commit(m1.lengths, staging=staging, rows=m1.state_rows, starts=m1.starts)

            # Inactive lane 0 and null row 0 must remain untouched; lanes 1 & 2 must match isolated rings.
            assert not pool.window(2)[0].any()
            assert torch.equal(pool.window(2)[1], lane0_ring_before)
            assert _relative_error(pool.window(2)[2], iso_rings[1]) < 1e-3
            assert _relative_error(pool.window(2)[3], iso_rings[2]) < 1e-3

            # Step 2: run lengths=[0, 1, 0] and confirm lane 1's output matches isolated step 2.
            packed_x2 = torch.zeros((3, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
            packed_x2[0].copy_(step_x[1][1])
            m2 = Metadata.pack(
                starts=torch.tensor([prefix_lens[0], prefix_lens[1] + 1, prefix_lens[2] + 1], dtype=torch.int64, device=device),
                lengths=torch.tensor([0, 1, 0], dtype=torch.int32, device=device),
                state_rows=torch.tensor([1, 2, 3], dtype=torch.int64, device=device),
                pages=pages_b3,
                tokens=torch.ones((3, 1), dtype=torch.int64, device=device),
                live_rows=torch.tensor([1], dtype=torch.int32, device=device),
                token_capacity=3,
                sample_count=3,
                max_query_tokens=1,
            )
            staging2 = torch.empty(staging_shape, dtype=torch.uint8, device=device)
            out2, _, _ = decode(packed_x2, m2, window_cache=staging2[0])
            # Same M=3 kernel run on the clean isolated ring must match identically; M=1 vs M=3 BF16 stays within 1 ULP.
            pool.window(2)[2].copy_(iso_rings[1])
            out2_iso_ring, _, _ = decode(packed_x2, m2, window_cache=staging2[0])
            assert _relative_error(out2[0:1], out2_iso_ring[0:1]) < 1e-3
            assert _relative_error(out2[0:1], iso_out2[1]) < 0.01


@torch.no_grad()
def test_c4_indexer_physical_tail_and_b1_vs_b2_score_eviction_cuda(tmp_path):
    """Verify >2,048-token physical tail compaction and score-based eviction across fused (M=6) and tiled (M=12)."""
    from hyperflare_recipes.recipes.hy4_flash.indexer import expand_pooled_topk_to_logical_tokens

    device = torch.device("cuda:0")
    cfg = _synthetic_config(tmp_path)
    rotary = InterleavedMRoPE(cfg, device)
    attn_w = _random_attention_weights(SPARSE_HEADS, device, seed=3101)
    idx_w = _random_indexer_weights(device, seed=3102)

    # 2,060 tokens = 515 complete C4 groups (> 512) so positions 2047..2059 exercise pre-cap,
    # exact-cap (512 prior groups), and post-cap partial/completing boundaries.
    prefix_tokens = 2047
    verify_width = 6
    chunk = 512
    pages_per_req = (prefix_tokens + verify_width + PAGE_TOKENS - 1) // PAGE_TOKENS  # 9 pages
    total_pages = pages_per_req * 2 + 2

    torch.manual_seed(3103)
    x_req0 = (torch.randn(prefix_tokens + verify_width, HIDDEN_SIZE, device=device) * 0.25).to(torch.bfloat16)
    x_req1 = (torch.randn(prefix_tokens + verify_width, HIDDEN_SIZE, device=device) * 0.25).to(torch.bfloat16)

    pool = ContinuationPool.allocate(cfg, 4, total_pages, device, dfly=True)
    arena = ScratchArena(device)
    p_scratch = AttentionScratch.allocate(chunk, 1, chunk, device)
    v1_scratch = AttentionScratch.allocate(verify_width, 1, verify_width, device)
    v2_scratch = AttentionScratch.allocate(2 * verify_width, 2, verify_width, device)

    # Disjoint shuffled physical pages for request 0 and request 1.
    pages_r0 = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15, 17], dtype=torch.int32, device=device)
    pages_r1 = torch.tensor([18, 16, 14, 12, 10, 8, 6, 4, 2], dtype=torch.int32, device=device)
    assert pages_r0.numel() == pages_per_req and pages_r1.numel() == pages_per_req

    p_meta_cap = _single_request_metadata(list(range(chunk)), start=0, state_row=1, pages=pages_per_req, device=device)
    v1_meta = Metadata.pack(
        starts=torch.tensor([prefix_tokens], dtype=torch.int64, device=device),
        lengths=torch.tensor([verify_width], dtype=torch.int32, device=device),
        state_rows=torch.tensor([1], dtype=torch.int64, device=device),
        pages=pages_r0.unsqueeze(0),
        tokens=torch.ones((1, verify_width), dtype=torch.int64, device=device),
        live_rows=torch.tensor([1], dtype=torch.int32, device=device),
        token_capacity=verify_width,
        sample_count=verify_width,
        max_query_tokens=verify_width,
    )
    v2_meta = Metadata.pack(
        starts=torch.tensor([prefix_tokens, prefix_tokens], dtype=torch.int64, device=device),
        lengths=torch.tensor([verify_width, verify_width], dtype=torch.int32, device=device),
        state_rows=torch.tensor([1, 2], dtype=torch.int64, device=device),
        pages=torch.stack((pages_r0, pages_r1), dim=0),
        tokens=torch.ones((2, verify_width), dtype=torch.int64, device=device),
        live_rows=torch.tensor([2], dtype=torch.int32, device=device),
        token_capacity=2 * verify_width,
        sample_count=2 * verify_width,
        max_query_tokens=verify_width,
    )

    with RecordedPreparationSession(
        device=device, checkpoint="hy4-c4-b1-b2", workload_schema="c4-b1-b2", numerical_policy="default"
    ) as session:
        session.register_scratch(arena)
        prefill = PreparedSparseAttention(
            25, attn_w, idx_w, torch.zeros((chunk, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
            p_scratch, p_meta_cap, pool, rotary,
            mode="prefill", eps=1e-5, session=session, name="sp_p", workspace=arena,
        )
        # B=1 verify (6 rows -> paged_fused) vs B=2 verify (12 rows -> paged_tiled)
        verify1 = PreparedSparseAttention(
            25, attn_w, idx_w, torch.zeros((verify_width, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
            v1_scratch, v1_meta, pool, rotary,
            mode="verify", eps=1e-5, session=session, name="sp_v1", workspace=arena,
        )
        verify2 = PreparedSparseAttention(
            25, attn_w, idx_w, torch.zeros((2 * verify_width, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
            v2_scratch, v2_meta, pool, rotary,
            mode="verify", eps=1e-5, session=session, name="sp_v2", workspace=arena,
        )
        session.tune()

        pool.clear_for_preparation()
        for r_idx, (x_seq, p_map) in enumerate(((x_req0, pages_r0), (x_req1, pages_r1))):
            start = 0
            while start < prefix_tokens:
                n = min(chunk, prefix_tokens - start)
                padded_x = torch.zeros((chunk, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
                padded_x[:n].copy_(x_seq[start : start + n])
                padded_pos = list(range(start, start + n)) + [-1] * (chunk - n)
                pm = _single_request_metadata(padded_pos, start=start, state_row=r_idx + 1, pages=pages_per_req, device=device)
                pm.page_table[0].copy_(p_map)
                prefill(padded_x, pm)
                start += n

        # Inspect physical slots and selected_lengths directly for positions [2047, 2048, 2049, 2050, 2051, 2052].
        q_res1 = verify1.q_norm(verify1.qkv_a(x_req0[prefix_tokens:])[:, :Q_LORA_RANK].contiguous())
        phys1, len1, _, pools1 = verify1.indexer(x_req0[prefix_tokens:], q_res1, v1_meta)
        x_b2 = torch.cat((x_req0[prefix_tokens:], x_req1[prefix_tokens:]), dim=0)
        q_res2 = verify2.q_norm(verify2.qkv_a(x_b2)[:, :Q_LORA_RANK].contiguous())
        phys2, len2, _, pools2 = verify2.indexer(x_b2, q_res2, v2_meta)

        expected_lens = [2048, 2045, 2046, 2047, 2048, 2045]
        assert len1.tolist() == expected_lens
        assert len2[:verify_width].tolist() == expected_lens
        assert len2[verify_width:].tolist() == expected_lens

        for tok_i, (pos, exp_len) in enumerate(zip(range(prefix_tokens, prefix_tokens + verify_width), expected_lens, strict=True)):
            for label, phys_row, pool_row in (("fused", phys1[tok_i], pools1[tok_i]), ("tiled", phys2[tok_i], pools2[tok_i])):
                # Zero -1 holes before exp_len; all -1 from exp_len onward.
                assert torch.all(phys_row[:exp_len] >= 0), f"{label} hole at pos={pos}"
                assert torch.all(phys_row[exp_len:] == -1), f"{label} trailing slot at pos={pos}"
                # Current token and its full partial/completing group must be present in physical slots.
                group_start = (pos // INDEX_KPOOL) * INDEX_KPOOL
                expected_tail_phys = [
                    int(pages_r0[t // PAGE_TOKENS].item()) * PAGE_TOKENS + (t % PAGE_TOKENS)
                    for t in range(group_start, pos + 1)
                ]
                assert phys_row[exp_len - len(expected_tail_phys) : exp_len].tolist() == expected_tail_phys, (
                    f"{label} missing partial tail at pos={pos}"
                )
                # Logical expansion (DFly2) and physical expansion (target sparse MLA) must represent the exact same slots.
                logical = expand_pooled_topk_to_logical_tokens(
                    pool_row.unsqueeze(0), torch.tensor([pos], dtype=torch.int64, device=device)
                )[0]
                valid_log = logical[logical >= 0]
                assert valid_log.numel() == exp_len
                resolved_log = [
                    int(pages_r0[t // PAGE_TOKENS].item()) * PAGE_TOKENS + (t % PAGE_TOKENS)
                    for t in valid_log.tolist()
                ]
                assert resolved_log == phys_row[:exp_len].tolist()

            # Fused (M=6) and tiled (M=12) must select the same set of physical slots for request 0.
            set_fused = set(phys1[tok_i, :exp_len].tolist())
            set_tiled = set(phys2[tok_i, :exp_len].tolist())
            assert set_fused == set_tiled, f"fused vs tiled pool mismatch at pos={pos}: diff={set_fused ^ set_tiled}"

        # End-to-end sparse attention output for request 0 must match between B=1 (M=6) and B=2 (M=12).
        out1, _, _ = verify1(x_req0[prefix_tokens:], v1_meta)
        out2, _, _ = verify2(x_b2, v2_meta)
        assert _relative_error(out2[:verify_width], out1) < 0.01


@torch.no_grad()
def test_real_checkpoint_multistream_layers_b1_vs_b2_b4_b8_cuda():
    """Verify real Hy4-flash-NVFP4 online-MXFP8 + CSF LatentMoE + stream_overlap layers match B=1 across B=2,4,8."""
    snapshot = _quant_snapshot()

    device = torch.device("cuda:0")
    shard = TensorParallelShard(0, 1)
    manifest = inspect_indexed_checkpoint(snapshot)
    cfg = ModelConfig.load(manifest.metadata_path, manifest=manifest)
    rotary = InterleavedMRoPE(cfg, device)

    with Hy4WeightLoader(
        manifest,
        device,
        config=cfg,
        shard=shard,
        dfly=False,
        vision=False,
        expert_scale_compression="csf",
        online_mxfp8=True,
        online_mxfp8_scope="all",
    ) as loader:
        l0 = loader.load_layer(0)  # sparse + dense FFN
        l1 = loader.load_layer(1)  # sparse + LatentMoE (CSF)
        l2 = loader.load_layer(2)  # SWA + LatentMoE (CSF)

    from hyperflare_recipes.components.ring_commit import PreparedRingCommit

    verify_w = 6
    max_b = 8
    prefix_lens = tuple(37 + 4 * i for i in range(max_b))  # distinct lengths crossing C4 boundaries
    max_prefill = max(prefix_lens)
    total_pages = max_b * 2 + 2

    torch.manual_seed(4101)
    prefill_streams = [
        (torch.randn(n, 4, HIDDEN_SIZE, device=device) * 0.2).to(torch.bfloat16) for n in prefix_lens
    ]
    verify_streams = [
        (torch.randn(verify_w, 4, HIDDEN_SIZE, device=device) * 0.2).to(torch.bfloat16) for _ in range(max_b)
    ]
    # Disjoint shuffled physical pages (2 pages per request).
    req_pages = torch.randperm(max_b * 2, generator=torch.Generator().manual_seed(4102), dtype=torch.int32).to(device).view(max_b, 2)

    pool = ContinuationPool.allocate(cfg, max_b + 2, total_pages, device, dfly=False)
    arena = ScratchArena(device, overlap=True)
    p_attn_scratch = AttentionScratch.allocate(max_prefill, 1, max_prefill, device)
    p_moe_scratch = MoEScratch.allocate(max_prefill, device)
    p_meta_cap = _single_request_metadata(list(range(max_prefill)), start=0, state_row=1, pages=2, device=device)

    batch_sizes = (1, 2, 4, 8)
    v_scratches = {
        b: (AttentionScratch.allocate(b * verify_w, b, verify_w, device), MoEScratch.allocate(b * verify_w, device))
        for b in batch_sizes
    }
    v_metas_cap = {
        b: Metadata.pack(
            starts=torch.tensor(prefix_lens[:b], dtype=torch.int64, device=device),
            lengths=torch.full((b,), verify_w, dtype=torch.int32, device=device),
            state_rows=torch.arange(1, b + 1, dtype=torch.int64, device=device),
            pages=req_pages[:b],
            tokens=torch.ones((b, verify_w), dtype=torch.int64, device=device),
            live_rows=torch.tensor([b], dtype=torch.int32, device=device),
            token_capacity=b * verify_w,
            sample_count=b * verify_w,
            max_query_tokens=verify_w,
        )
        for b in batch_sizes
    }

    def _build_three_layers(session, *, tokens, mode, attn_scratch, moe_scratch, meta, tag):
        proto_streams = torch.zeros((tokens, 4, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
        proto_x = proto_streams[:, 0].contiguous()
        hc_mixes = torch.zeros((tokens, 8), dtype=torch.float32, device=device)
        hc_projs = {}
        for idx, lw in ((0, l0), (1, l1), (2, l2)):
            hc_projs[(idx, "attn")] = UnquantizedProjection(
                proto_streams.view(tokens, -1), lw.residual["attn_fn"], hc_mixes,
                session=session, name=f"{tag}.l{idx}.hc_attn",
            ).functional(session=session)
            hc_projs[(idx, "mlp")] = UnquantizedProjection(
                proto_streams.view(tokens, -1), lw.residual["mlp_fn"], hc_mixes,
                session=session, name=f"{tag}.l{idx}.hc_mlp",
            ).functional(session=session)
        a0 = PreparedSparseAttention(
            0, l0.attention, l0.indexer, proto_x, attn_scratch, meta, pool, rotary,
            mode=mode, eps=1e-5, session=session, name=f"{tag}.l0.attn", workspace=arena,
        )
        m0 = PreparedDenseMLP(l0.mlp, proto_x, moe_scratch, eps=1e-5, session=session, name=f"{tag}.l0.mlp", workspace=arena)
        a1 = PreparedSparseAttention(
            1, l1.attention, l1.indexer, proto_x, attn_scratch, meta, pool, rotary,
            mode=mode, eps=1e-5, session=session, name=f"{tag}.l1.attn", workspace=arena,
        )
        m1 = PreparedLatentMoE(l1.mlp, proto_x, moe_scratch, config=cfg, mode=mode, session=session, name=f"{tag}.l1.moe", workspace=arena)
        a2 = PreparedSlidingAttention(
            2, l2.attention, proto_x, attn_scratch, meta, pool, rotary,
            mode=mode, eps=1e-5, session=session, name=f"{tag}.l2.attn", workspace=arena,
        )
        m2 = PreparedLatentMoE(l2.mlp, proto_x, moe_scratch, config=cfg, mode=mode, session=session, name=f"{tag}.l2.moe", workspace=arena)
        ring_commit = None
        staging_shape = None
        if mode == "verify":
            ring_commit = PreparedRingCommit(
                pool.reserved_backing,
                [pool.layout.region("layer2.window").offset],
                meta.state_rows,
                meta.starts,
                token_capacity=verify_w,
                window_tokens=SWA_WINDOW,
                record_bytes=576 * 2,
            )
            staging_shape = tuple(ring_commit.staging.shape)
            ring_commit.functional()
        return hc_projs, ((0, l0, a0, m0), (1, l1, a1, m1), (2, l2, a2, m2)), ring_commit, staging_shape

    def _run_stack(x_streams, meta, bundle, *, capture_trace: bool = False):
        hc_projs, layers, _, staging_shape = bundle
        tokens = x_streams.shape[0]
        h = x_streams
        ring_staging = None if staging_shape is None else torch.empty(staging_shape, dtype=torch.uint8, device=device)
        trace = {}
        for idx, lw, attn_mod, ffn_mod in layers:
            r = lw.residual
            attn_mix = hc_projs[(idx, "attn")](h.reshape(tokens, -1))
            post_a, x_a = ihc_pre(h, r["attn_fn"], r["attn_base"], r["attn_scale"], rms_eps=1e-5, hc_eps=1e-6, hc_magnitude=2.0, mixes=attn_mix)
            x_a = (x_a.float() * torch.rsqrt(x_a.float().square().mean(-1, keepdim=True) + 1e-5)).to(x_a.dtype) * r["input_norm"]
            if isinstance(attn_mod, PreparedSlidingAttention) and ring_staging is not None:
                out_a = attn_mod(x_a, meta, window_cache=ring_staging[0])[0]
            else:
                out_a = attn_mod(x_a, meta)[0]
            h = ihc_post(h, out_a, post_a)
            mlp_mix = hc_projs[(idx, "mlp")](h.reshape(tokens, -1))
            post_m, x_m = ihc_pre(h, r["mlp_fn"], r["mlp_base"], r["mlp_scale"], rms_eps=1e-5, hc_eps=1e-6, hc_magnitude=2.0, mixes=mlp_mix)
            x_m = (x_m.float() * torch.rsqrt(x_m.float().square().mean(-1, keepdim=True) + 1e-5)).to(x_m.dtype) * r["post_attn_norm"]
            out_m = ffn_mod(x_m)
            h = ihc_post(h, out_m, post_m)
            if capture_trace:
                trace[f"l{idx}.attn"] = out_a.clone()
                trace[f"l{idx}.mlp"] = out_m.clone()
                trace[f"l{idx}.h"] = h.clone()
        return h, trace

    with RecordedPreparationSession(
        device=device, checkpoint="hy4-real-multistream", workload_schema="real-multistream", numerical_policy="default"
    ) as session:
        session.register_scratch(arena)
        prefill_bundle = _build_three_layers(
            session, tokens=max_prefill, mode="prefill",
            attn_scratch=p_attn_scratch, moe_scratch=p_moe_scratch, meta=p_meta_cap, tag="p",
        )
        verify_bundles = {
            b: _build_three_layers(
                session, tokens=b * verify_w, mode="verify",
                attn_scratch=v_scratches[b][0], moe_scratch=v_scratches[b][1], meta=v_metas_cap[b], tag=f"v{b}",
            )
            for b in batch_sizes
        }
        session.tune()

        # Prefill all 8 distinct streams into disjoint continuation rows (1..8) and pages.
        pool.clear_for_preparation()
        for r, n in enumerate(prefix_lens):
            padded_s = torch.zeros((max_prefill, 4, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
            padded_s[:n].copy_(prefill_streams[r])
            padded_pos = list(range(n)) + [-1] * (max_prefill - n)
            pm = _single_request_metadata(padded_pos, start=0, state_row=r + 1, pages=2, device=device)
            pm.page_table[0].copy_(req_pages[r])
            _run_stack(padded_s, pm, prefill_bundle)
        saved_backing = pool.backing.clone()
        saved_pages = {k: v.clone() for k, v in pool.pages.items()}

        def _restore_pool():
            pool.backing.copy_(saved_backing)
            for k, v in saved_pages.items():
                pool.pages[k].copy_(v)

        # Baseline isolated B=1 (M=6) outputs + measured MoE reduction repeatability for each request.
        iso_out = []
        iso_traces = []
        max_repeat = 0.0
        for r in range(max_b):
            m1 = Metadata.pack(
                starts=torch.tensor([prefix_lens[r]], dtype=torch.int64, device=device),
                lengths=torch.tensor([verify_w], dtype=torch.int32, device=device),
                state_rows=torch.tensor([r + 1], dtype=torch.int64, device=device),
                pages=req_pages[r : r + 1],
                tokens=torch.ones((1, verify_w), dtype=torch.int64, device=device),
                live_rows=torch.tensor([1], dtype=torch.int32, device=device),
                token_capacity=verify_w,
                sample_count=verify_w,
                max_query_tokens=verify_w,
            )
            _restore_pool()
            h1, tr1 = _run_stack(verify_streams[r], m1, verify_bundles[1], capture_trace=True)
            _restore_pool()
            h1_rep, _ = _run_stack(verify_streams[r], m1, verify_bundles[1])
            max_repeat = max(max_repeat, _relative_error(h1_rep, h1))
            iso_out.append(h1.clone())
            iso_traces.append(tr1)

        bound = max(0.01, 3.0 * max_repeat + 0.005)

        # Compare B=2 (M=12), B=4 (M=24), and B=8 (M=48) — both eager and CUDA-graph captured — against B=1.
        for b in (2, 4, 8):
            x_b = torch.cat(verify_streams[:b], dim=0).contiguous()
            mb = v_metas_cap[b]
            _restore_pool()
            h_b, tr_b = _run_stack(x_b, mb, verify_bundles[b], capture_trace=True)
            for r in range(b):
                sl = slice(r * verify_w, (r + 1) * verify_w)
                for key in ("l0.attn", "l0.mlp", "l1.attn", "l1.mlp", "l2.attn", "l2.mlp", "l2.h"):
                    err = _relative_error(tr_b[key][sl], iso_traces[r][key])
                    assert err < bound, f"B={b} req={r} {key} diverged from B=1: rel_err={err:.5f} (bound={bound:.5f})"
                assert _relative_error(h_b[sl], iso_out[r]) < bound

            # Compiled + CUDA-graph captured replay at B=2 (M=12) and B=8 (M=48).
            if b in (2, 8):
                static_out = torch.empty_like(h_b)
                compiled = compile_forward(lambda: _run_stack(x_b, mb, verify_bundles[b])[0])
                _restore_pool()
                compiled()
                _restore_pool()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_out.copy_(compiled())
                _restore_pool()
                graph.replay()
                torch.cuda.synchronize(device)
                for r in range(b):
                    sl = slice(r * verify_w, (r + 1) * verify_w)
                    assert _relative_error(static_out[sl], iso_out[r]) < bound

        # Also test B=2 with unequal offered lengths [2, 6] and inactive leading lane [0, 6].
        for lengths_pair in ((2, 6), (0, 6)):
            x_packed = torch.zeros((2 * verify_w, 4, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
            off = 0
            for r, ln in enumerate(lengths_pair):
                if ln > 0:
                    x_packed[off : off + ln].copy_(verify_streams[r][:ln])
                    off += ln
            m_var = Metadata.pack(
                starts=torch.tensor(prefix_lens[:2], dtype=torch.int64, device=device),
                lengths=torch.tensor(lengths_pair, dtype=torch.int32, device=device),
                state_rows=torch.tensor([1, 2], dtype=torch.int64, device=device),
                pages=req_pages[:2],
                tokens=torch.ones((2, verify_w), dtype=torch.int64, device=device),
                live_rows=torch.tensor([sum(1 for ln in lengths_pair if ln > 0)], dtype=torch.int32, device=device),
                token_capacity=2 * verify_w,
                sample_count=2 * verify_w,
                max_query_tokens=verify_w,
            )
            _restore_pool()
            h_var, _ = _run_stack(x_packed, m_var, verify_bundles[2])
            off = 0
            for r, ln in enumerate(lengths_pair):
                if ln > 0:
                    assert _relative_error(h_var[off : off + ln], iso_out[r][:ln]) < bound
                    off += ln


@torch.no_grad()
def test_multistream_mixed_rotary_offsets_and_interleaved_dead_rows_cuda(tmp_path):
    """Verify B=4 (M=24) and B=8 (M=48) sparse + SWA verify with mixed rotary offsets and dead rows."""
    from dataclasses import replace
    from hyperflare_recipes.components.ring_commit import PreparedRingCommit

    device = torch.device("cuda:0")
    cfg = _synthetic_config(tmp_path)
    rotary = InterleavedMRoPE(cfg, device)
    sp_w = _random_attention_weights(SPARSE_HEADS, device, seed=5101)
    idx_w = _random_indexer_weights(device, seed=5102)
    swa_w = _random_attention_weights(SWA_HEADS, device, seed=5103)

    # Stream 0: short text (< 2048), rotary_offset=0
    # Stream 1: long text at exact C4 cap (pos=2048, partial tail), rotary_offset=0
    # Stream 2: multimodal image+text (> 2048, 3D M-RoPE prefill, negative rotary_offset=-759)
    # Stream 3: inactive row (length=0, state_row=0, page_table=-1)
    prefix_lens = (45, 2048, 2055, 0)
    rot_offsets = (0, 0, -759, 0)
    verify_w = 6
    chunk = 512
    pages_per_req = (max(prefix_lens) + verify_w + PAGE_TOKENS - 1) // PAGE_TOKENS  # 9
    total_pages = 3 * pages_per_req + 2

    torch.manual_seed(5104)
    seqs = [
        (torch.randn(n + verify_w, HIDDEN_SIZE, device=device) * 0.25).to(torch.bfloat16) if n > 0 else None
        for n in prefix_lens
    ]
    page_maps = [
        torch.arange(1 + i * pages_per_req, 1 + (i + 1) * pages_per_req, dtype=torch.int32, device=device)
        for i in range(3)
    ]
    page_maps.append(torch.full((pages_per_req,), -1, dtype=torch.int32, device=device))

    pool = ContinuationPool.allocate(cfg, 6, total_pages, device, dfly=True)
    arena = ScratchArena(device, overlap=True)
    p_scratch = AttentionScratch.allocate(chunk, 1, chunk, device)
    v1_scratch = AttentionScratch.allocate(verify_w, 1, verify_w, device)
    v4_scratch = AttentionScratch.allocate(4 * verify_w, 4, verify_w, device)
    v8_scratch = AttentionScratch.allocate(8 * verify_w, 8, verify_w, device)

    p_meta_cap = _single_request_metadata(list(range(chunk)), start=0, state_row=1, pages=pages_per_req, device=device)
    v1_meta_cap = Metadata.pack(
        starts=torch.tensor([prefix_lens[1]], dtype=torch.int64, device=device),
        lengths=torch.tensor([verify_w], dtype=torch.int32, device=device),
        state_rows=torch.tensor([2], dtype=torch.int64, device=device),
        pages=page_maps[1].unsqueeze(0),
        tokens=torch.ones((1, verify_w), dtype=torch.int64, device=device),
        live_rows=torch.tensor([1], dtype=torch.int32, device=device),
        token_capacity=verify_w,
        sample_count=verify_w,
        max_query_tokens=verify_w,
        rotary_offset=torch.tensor([0], dtype=torch.int64, device=device),
    )
    v4_meta = Metadata.pack(
        starts=torch.tensor(prefix_lens, dtype=torch.int64, device=device),
        lengths=torch.tensor([verify_w, verify_w, verify_w, 0], dtype=torch.int32, device=device),
        state_rows=torch.tensor([1, 2, 3, 0], dtype=torch.int64, device=device),
        pages=torch.stack(page_maps, dim=0),
        tokens=torch.ones((4, verify_w), dtype=torch.int64, device=device),
        live_rows=torch.tensor([3], dtype=torch.int32, device=device),
        token_capacity=4 * verify_w,
        sample_count=4 * verify_w,
        max_query_tokens=verify_w,
        rotary_offset=torch.tensor(rot_offsets, dtype=torch.int64, device=device),
    )
    v8_meta = Metadata.pack(
        starts=torch.tensor([0, prefix_lens[0], 0, prefix_lens[1], prefix_lens[2], 0, 0, 0], dtype=torch.int64, device=device),
        lengths=torch.tensor([0, verify_w, 0, verify_w, verify_w, 0, 0, 0], dtype=torch.int32, device=device),
        state_rows=torch.tensor([0, 1, 0, 2, 3, 0, 0, 0], dtype=torch.int64, device=device),
        pages=torch.stack(
            [page_maps[3], page_maps[0], page_maps[3], page_maps[1], page_maps[2], page_maps[3], page_maps[3], page_maps[3]],
            dim=0,
        ),
        tokens=torch.ones((8, verify_w), dtype=torch.int64, device=device),
        live_rows=torch.tensor([5], dtype=torch.int32, device=device),
        token_capacity=8 * verify_w,
        sample_count=8 * verify_w,
        max_query_tokens=verify_w,
        rotary_offset=torch.tensor([0, 0, 0, 0, -759, 0, 0, 0], dtype=torch.int64, device=device),
    )

    with RecordedPreparationSession(
        device=device, checkpoint="hy4-mixed-rope-dead", workload_schema="mixed-rope-dead", numerical_policy="default"
    ) as session:
        session.register_scratch(arena)
        sp_p = PreparedSparseAttention(
            25, sp_w, idx_w, torch.zeros((chunk, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
            p_scratch, p_meta_cap, pool, rotary, mode="prefill", eps=1e-5, session=session, name="sp_p", workspace=arena,
        )
        sp_v1 = PreparedSparseAttention(
            25, sp_w, idx_w, torch.zeros((verify_w, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
            v1_scratch, v1_meta_cap, pool, rotary, mode="verify", eps=1e-5, session=session, name="sp_v1", workspace=arena,
        )
        sp_v4 = PreparedSparseAttention(
            25, sp_w, idx_w, torch.zeros((4 * verify_w, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
            v4_scratch, v4_meta, pool, rotary, mode="verify", eps=1e-5, session=session, name="sp_v4", workspace=arena,
        )
        sp_v8 = PreparedSparseAttention(
            25, sp_w, idx_w, torch.zeros((8 * verify_w, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
            v8_scratch, v8_meta, pool, rotary, mode="verify", eps=1e-5, session=session, name="sp_v8", workspace=arena,
        )
        swa_p = PreparedSlidingAttention(
            2, swa_w, torch.zeros((chunk, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
            p_scratch, p_meta_cap, pool, rotary, mode="prefill", eps=1e-5, session=session, name="swa_p", workspace=arena,
        )
        swa_v1 = PreparedSlidingAttention(
            2, swa_w, torch.zeros((verify_w, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
            v1_scratch, v1_meta_cap, pool, rotary, mode="verify", eps=1e-5, session=session, name="swa_v1", workspace=arena,
        )
        swa_v4 = PreparedSlidingAttention(
            2, swa_w, torch.zeros((4 * verify_w, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
            v4_scratch, v4_meta, pool, rotary, mode="verify", eps=1e-5, session=session, name="swa_v4", workspace=arena,
        )
        swa_v8 = PreparedSlidingAttention(
            2, swa_w, torch.zeros((8 * verify_w, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
            v8_scratch, v8_meta, pool, rotary, mode="verify", eps=1e-5, session=session, name="swa_v8", workspace=arena,
        )
        rc1 = PreparedRingCommit(
            pool.reserved_backing, [pool.layout.region("layer2.window").offset],
            v1_meta_cap.state_rows, v1_meta_cap.starts,
            token_capacity=verify_w, window_tokens=SWA_WINDOW, record_bytes=576 * 2,
        )
        rc4 = PreparedRingCommit(
            pool.reserved_backing, [pool.layout.region("layer2.window").offset],
            v4_meta.state_rows, v4_meta.starts,
            token_capacity=verify_w, window_tokens=SWA_WINDOW, record_bytes=576 * 2,
        )
        rc8 = PreparedRingCommit(
            pool.reserved_backing, [pool.layout.region("layer2.window").offset],
            v8_meta.state_rows, v8_meta.starts,
            token_capacity=verify_w, window_tokens=SWA_WINDOW, record_bytes=576 * 2,
        )
        rc1_shape, rc4_shape, rc8_shape = tuple(rc1.staging.shape), tuple(rc4.staging.shape), tuple(rc8.staging.shape)
        rc1.functional()
        rc4.functional()
        rc8.functional()
        session.tune()

        pool.clear_for_preparation()
        for r in range(3):
            n = prefix_lens[r]
            start = 0
            while start < n:
                cnt = min(chunk, n - start)
                px = torch.zeros((chunk, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
                px[:cnt].copy_(seqs[r][start : start + cnt])
                ppos = list(range(start, start + cnt)) + [-1] * (chunk - cnt)
                pm = _single_request_metadata(ppos, start=start, state_row=r + 1, pages=pages_per_req, device=device)
                pm.page_table[0].copy_(page_maps[r])
                if r == 2:
                    rope3 = torch.full((chunk, 3), -1, dtype=torch.int64, device=device)
                    for k in range(cnt):
                        p_abs = start + k
                        if 10 <= p_abs < 802:
                            rel = p_abs - 10
                            rope3[k, 0] = 10
                            rope3[k, 1] = 10 + rel // 33
                            rope3[k, 2] = 10 + rel % 33
                        else:
                            rope3[k] = p_abs if p_abs < 10 else p_abs - 759
                    pm = replace(pm, rotary_positions=rope3)
                sp_p(px, pm)
                swa_p(px, pm)
                start += cnt

        saved_backing = pool.backing.clone()
        saved_pages = {k: v.clone() for k, v in pool.pages.items()}

        def _restore():
            pool.backing.copy_(saved_backing)
            for k, v in saved_pages.items():
                pool.pages[k].copy_(v)

        iso_sp, iso_swa, iso_rings = [], [], []
        for r in range(3):
            _restore()
            m1 = Metadata.pack(
                starts=torch.tensor([prefix_lens[r]], dtype=torch.int64, device=device),
                lengths=torch.tensor([verify_w], dtype=torch.int32, device=device),
                state_rows=torch.tensor([r + 1], dtype=torch.int64, device=device),
                pages=page_maps[r].unsqueeze(0),
                tokens=torch.ones((1, verify_w), dtype=torch.int64, device=device),
                live_rows=torch.tensor([1], dtype=torch.int32, device=device),
                token_capacity=verify_w,
                sample_count=verify_w,
                max_query_tokens=verify_w,
                rotary_offset=torch.tensor([rot_offsets[r]], dtype=torch.int64, device=device),
            )
            o_sp, _, _ = sp_v1(seqs[r][prefix_lens[r] :], m1)
            st1 = torch.empty(rc1_shape, dtype=torch.uint8, device=device)
            o_swa, _, _ = swa_v1(seqs[r][prefix_lens[r] :], m1, window_cache=st1[0])
            rc1(m1.lengths, staging=st1, rows=m1.state_rows, starts=m1.starts)
            iso_sp.append(o_sp.clone())
            iso_swa.append(o_swa.clone())
            iso_rings.append(pool.window(2)[r + 1].clone())

        # B=4 verify (M=24, 3 active + 1 trailing dead row) + ring commit.
        _restore()
        x4 = torch.zeros((4 * verify_w, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
        for r in range(3):
            x4[r * verify_w : (r + 1) * verify_w].copy_(seqs[r][prefix_lens[r] :])
        o4_sp, _, _ = sp_v4(x4, v4_meta)
        st4 = torch.empty(rc4_shape, dtype=torch.uint8, device=device)
        o4_swa, _, _ = swa_v4(x4, v4_meta, window_cache=st4[0])
        rc4(v4_meta.lengths, staging=st4, rows=v4_meta.state_rows, starts=v4_meta.starts)
        assert torch.isfinite(o4_sp).all() and torch.isfinite(o4_swa).all()
        assert not pool.window(2)[0].any()
        for r in range(3):
            sl = slice(r * verify_w, (r + 1) * verify_w)
            assert _relative_error(o4_sp[sl], iso_sp[r]) < 0.01
            assert _relative_error(o4_swa[sl], iso_swa[r]) < 0.01
            assert _relative_error(pool.window(2)[r + 1], iso_rings[r]) < 1e-3

        # B=8 verify (M=48, leading dead + middle dead + 3 trailing dead) + ring commit.
        _restore()
        x8 = torch.zeros((8 * verify_w, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
        for r in range(3):
            x8[r * verify_w : (r + 1) * verify_w].copy_(seqs[r][prefix_lens[r] :])
        o8_sp, _, _ = sp_v8(x8, v8_meta)
        st8 = torch.empty(rc8_shape, dtype=torch.uint8, device=device)
        o8_swa, _, _ = swa_v8(x8, v8_meta, window_cache=st8[0])
        rc8(v8_meta.lengths, staging=st8, rows=v8_meta.state_rows, starts=v8_meta.starts)
        assert torch.isfinite(o8_sp).all() and torch.isfinite(o8_swa).all()
        assert not pool.window(2)[0].any()
        for r in range(3):
            sl = slice(r * verify_w, (r + 1) * verify_w)
            assert _relative_error(o8_sp[sl], iso_sp[r]) < 0.01
            assert _relative_error(o8_swa[sl], iso_swa[r]) < 0.01
            assert _relative_error(pool.window(2)[r + 1], iso_rings[r]) < 1e-3


@torch.no_grad()
def test_mxfp8_target_head_and_prompt_scoring_cuda():
    """Verify MXFP8 target lm_head with BF16 logits and A16 prompt-logprob scoring on SM120."""
    from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch
    from b12x.preparation.types import require_prepared
    from hyperflare_recipes.components.prompt_logprobs import PromptScoring
    from hyperflare_recipes.components.vocabulary import PreparedVocabularyProjection, mxfp8_draft_head
    from hyperflare_recipes.recipes.hy4_flash.config import VOCABULARY
    from hyperflare_recipes.recipes.hy4_flash.decoder import Scratch

    device = torch.device("cuda:0")
    shard = TensorParallelShard(0, 1)
    torch.manual_seed(20261126)
    head_bf16 = (torch.randn(VOCABULARY, HIDDEN_SIZE, device=device, dtype=torch.float32) * 0.04).to(torch.bfloat16)
    head_mx = mxfp8_draft_head(head_bf16)
    head_dequant = dequantize_mxfp8_rows_torch(
        head_mx.weight.values, head_mx.weight.scale_rows
    )[:, : head_mx.in_features].float()

    scratch_v = Scratch.allocate(6, 1, 6, 6, device, shard=shard, logits_dtype=torch.bfloat16)
    assert scratch_v.local_logits.dtype == torch.bfloat16
    assert scratch_v.logits.dtype == torch.bfloat16

    h_verify = (torch.randn(6, HIDDEN_SIZE, device=device, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    h_prompt = (torch.randn(64, HIDDEN_SIZE, device=device, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    scoring = PromptScoring(64, device, frame=frame)
    targets = torch.randint(0, VOCABULARY, (64,), dtype=torch.int32, device=device)
    scoring.outputs.targets.copy_(targets)

    arena = ScratchArena(device)
    with RecordedPreparationSession(
        device=device, checkpoint="hy4-mxfp8-head-unit", workload_schema="mxfp8-head", numerical_policy="default"
    ) as session:
        session.register_scratch(arena)
        vocab_proj = PreparedVocabularyProjection(
            scratch_v.hidden,
            head_mx,
            scratch_v.local_logits,
            scratch_v.logits,
            runtime=None,
            coordinator=None,
            session=session,
            name="target_head",
            workspace=arena,
        ).functional(session=session)
        prompt_op = scoring.prepare(
            h_prompt,
            head_mx,
            vocabulary=VOCABULARY,
            session=session,
            coordinator=None,
            name="target_prompt",
            logits_dtype=torch.bfloat16,
            activation_mode="a16",
        )
        session.tune()

        # Verify both target generation and T=64 prompt scoring prepared in A16 mode (not A8).
        for proj, src in ((vocab_proj.projection, h_verify), (prompt_op.parts[0][2], h_prompt)):
            state = require_prepared(proj.plan, "gemm.blockscaled_precision", device)
            if hasattr(state, "resolve"):
                state = state.resolve(src)
            assert state.config.mode == "a16", state.config.mode

        # Verify generation logits (BF16 output, A16 activation mode) against dequantized MXFP8 and BF16 refs.
        logits = vocab_proj(h_verify)
        assert logits.dtype == torch.bfloat16 and logits.shape == (6, VOCABULARY)
        assert torch.isfinite(logits).all()
        ref_v_mx = F.linear(h_verify.float(), head_dequant)
        ref_v_bf16 = F.linear(h_verify.float(), head_bf16.float())
        assert _relative_error(logits, ref_v_mx) < 0.008
        assert _relative_error(logits, ref_v_bf16) < 0.035
        p_ref = F.softmax(ref_v_bf16, dim=-1)
        kl = float((p_ref * (F.log_softmax(ref_v_bf16, dim=-1) - F.log_softmax(logits.float(), dim=-1))).sum(-1).mean())
        assert kl < 0.005
        assert torch.equal(logits.argmax(dim=-1), ref_v_mx.argmax(dim=-1))

        # Verify blocked prompt scoring at T=64 against exact dequantized MXFP8 log-softmax.
        prompt_op(h_prompt)
        torch.cuda.synchronize(device)
        out = scoring.outputs.values[0]
        assert torch.equal(out[:, 0], targets)
        actual_lp = out[:, 1].view(torch.float32)
        actual_top1_id = out[:, 3]
        actual_top1_lp = out[:, 4].view(torch.float32)
        assert torch.isfinite(actual_lp).all() and torch.isfinite(actual_top1_lp).all()

        ref_p_mx = F.log_softmax(F.linear(h_prompt.float(), head_dequant), dim=-1)
        ref_target_lp = ref_p_mx[torch.arange(64, device=device), targets.long()]
        ref_top1_lp, ref_top1_id = ref_p_mx.max(dim=-1)
        assert float((actual_lp - ref_target_lp).abs().max()) < 0.02
        assert float((actual_top1_lp - ref_top1_lp).abs().max()) < 0.02
        # Every selected top-1 token must match the FP32 dequantized maximum within BF16 logit rounding.
        actual_top1_ref_lp = ref_p_mx[torch.arange(64, device=device), actual_top1_id.long()]
        assert float((ref_top1_lp - actual_top1_ref_lp).abs().max()) < 0.02


