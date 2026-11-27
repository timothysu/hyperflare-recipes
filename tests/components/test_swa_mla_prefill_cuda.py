# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CUDA parity, ragged-batch, padding, and graph-replay checks for direct-ring SWA-512 MLA prefill."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from hyperflare_recipes.components.swa_mla_prefill import PreparedSwaMlaPrefill, commit_swa_ring
from ignition.preparation.compilation import compile_forward

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device",
)

D_NOPE = 512
D_ROPE = 64
D_QK = D_NOPE + D_ROPE
HEADS = 64
WINDOW = 512
MLA_SCALE = 256 ** -0.5


def _reference_swa_mla_prefill(
    query: torch.Tensor,
    window_buf: torch.Tensor,
    kv_record: torch.Tensor,
    sink: torch.Tensor,
    state_rows: torch.Tensor,
    starts: torch.Tensor,
    query_start_loc: torch.Tensor,
    lengths: torch.Tensor,
    *,
    window_size: int = WINDOW,
    scale: float = MLA_SCALE,
    round_before_sink: bool = True,
) -> torch.Tensor:
    """Exact FP32 sliding-window MLA reference with optional pre-sink BF16 rounding."""
    tokens, heads, _ = query.shape
    out = torch.zeros((tokens, heads, D_NOPE), dtype=torch.bfloat16, device=query.device)
    rows = starts.numel()
    for b in range(rows):
        state_row = int(state_rows[b].item())
        start_pos = int(starts[b].item())
        q_base = int(query_start_loc[b].item())
        q_end = int(query_start_loc[b + 1].item())
        span_len = max(0, q_end - q_base)
        seq_len = min(max(int(lengths[b].item()), 0), span_len) if state_row > 0 else 0
        if seq_len <= 0:
            continue

        hist_len = min(max(start_pos, 0), window_size - 1)
        hist_start = max(0, start_pos - hist_len)
        kv_chunk = kv_record[q_base : q_base + seq_len].float()
        if hist_len > 0:
            ring_idx = (hist_start + torch.arange(hist_len, device=query.device, dtype=torch.int64)) % window_size
            kv_hist = window_buf[state_row, ring_idx].float()
            kv_all = torch.cat((kv_hist, kv_chunk), dim=0)
            k_global = torch.arange(hist_start, start_pos + seq_len, device=query.device, dtype=torch.int64)
        else:
            kv_all = kv_chunk
            k_global = torch.arange(start_pos, start_pos + seq_len, device=query.device, dtype=torch.int64)

        q_slice = query[q_base : q_base + seq_len].float()
        q_global = start_pos + torch.arange(seq_len, device=query.device, dtype=torch.int64)
        valid_q = q_global >= 0
        valid_k = k_global >= 0
        dist = q_global[:, None] - k_global[None, :]
        mask = valid_q[:, None] & valid_k[None, :] & (dist >= 0) & (dist < window_size)

        scores = torch.einsum("qhd,kd->hqk", q_slice, kv_all) * scale
        scores = scores.masked_fill(~mask[None, :, :], float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)  # [H, seq_len]
        safe_lse = torch.where(lse == float("-inf"), 0.0, lse)
        probs = torch.where(mask[None, :, :], torch.exp(scores - safe_lse.unsqueeze(-1)), 0.0)
        att = torch.einsum("hqk,kd->qhd", probs, kv_all[:, :D_NOPE])  # [seq_len, H, D_NOPE]
        if round_before_sink:
            att = att.to(torch.bfloat16).float()
        sink_scale = torch.sigmoid(lse.transpose(0, 1) - sink[None, :]).unsqueeze(-1)
        scaled = (att * sink_scale).to(torch.bfloat16)
        scaled = torch.where(valid_q[:, None, None], scaled, 0)
        out[q_base : q_base + seq_len] = scaled
    return out


def _make_metadata(
    state_rows: list[int],
    starts: list[int],
    query_start_loc: list[int],
    lengths: list[int],
    *,
    tokens: int,
    max_query_tokens: int,
    device: torch.device,
) -> SimpleNamespace:
    positions = torch.full((tokens,), -1, dtype=torch.int64, device=device)
    for b, (sr, st, q0, ln) in enumerate(zip(state_rows, starts, query_start_loc[:-1], lengths, strict=True)):
        if sr > 0 and ln > 0:
            positions[q0 : q0 + ln] = st + torch.arange(ln, dtype=torch.int64, device=device)
    return SimpleNamespace(
        state_rows=torch.tensor(state_rows, dtype=torch.int64, device=device),
        starts=torch.tensor(starts, dtype=torch.int64, device=device),
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32, device=device),
        lengths=torch.tensor(lengths, dtype=torch.int32, device=device),
        positions=positions,
        max_query_tokens=max_query_tokens,
    )


@pytest.mark.parametrize("start_pos", [0, 63, 511, 512, 513, 1234])
@pytest.mark.parametrize("seq_len", [1, 64, 65, 512, 513, 700])
@torch.no_grad()
def test_swa_mla_prefill_single_request_boundaries_and_ring_wraparound(start_pos: int, seq_len: int):
    device = torch.device("cuda:0")
    gen = torch.Generator(device=device).manual_seed(10000 + start_pos * 17 + seq_len)
    capacity = 768

    query = (torch.randn((capacity, HEADS, D_QK), generator=gen, device=device, dtype=torch.float32) * 0.25).to(
        torch.bfloat16
    )
    # Poison padded query/KV rows to verify padded positions never leak or affect active rows.
    query[seq_len:].fill_(float("nan"))
    window_buf = (torch.randn((4, WINDOW, D_QK), generator=gen, device=device, dtype=torch.float32) * 0.25).to(
        torch.bfloat16
    )
    window_buf[0].fill_(float("nan"))
    kv_record = (torch.randn((capacity, D_QK), generator=gen, device=device, dtype=torch.float32) * 0.25).to(
        torch.bfloat16
    )
    kv_record[seq_len:].fill_(float("nan"))
    sink = torch.randn((HEADS,), generator=gen, device=device, dtype=torch.float32) * 0.5
    out = torch.full((capacity, HEADS, D_NOPE), 123.0, dtype=torch.bfloat16, device=device)

    meta = _make_metadata([2], [start_pos], [0, seq_len], [seq_len], tokens=capacity, max_query_tokens=capacity, device=device)
    op = PreparedSwaMlaPrefill(
        query, window_buf, kv_record, sink, out, meta, scale=MLA_SCALE, window_size=WINDOW
    ).functional()

    actual = torch.full_like(out, 123.0)
    op(query, kv_record, meta, window=window_buf, output=actual)

    ref_bf16_sink = _reference_swa_mla_prefill(
        query, window_buf, kv_record, sink, meta.state_rows, meta.starts, meta.query_start_loc, meta.lengths,
        round_before_sink=True,
    )
    ref_fp32 = _reference_swa_mla_prefill(
        query, window_buf, kv_record, sink, meta.state_rows, meta.starts, meta.query_start_loc, meta.lengths,
        round_before_sink=False,
    )

    assert torch.isfinite(actual).all()
    diff_sink = (actual[:seq_len].float() - ref_bf16_sink[:seq_len].float()).abs()
    diff_fp32 = (actual[:seq_len].float() - ref_fp32[:seq_len].float()).abs()
    rel_l2_sink = diff_sink.norm().item() / max(ref_bf16_sink[:seq_len].float().norm().item(), 1e-6)
    rel_l2_fp32 = diff_fp32.norm().item() / max(ref_fp32[:seq_len].float().norm().item(), 1e-6)
    pad_zero_max = actual[seq_len:].float().abs().max().item() if seq_len < capacity else 0.0

    assert pad_zero_max == 0.0
    assert diff_sink.max().item() < 5e-3
    assert rel_l2_sink < 3e-3
    assert diff_fp32.max().item() < 8e-3
    assert rel_l2_fp32 < 4e-3


@torch.no_grad()
def test_swa_mla_prefill_ragged_b4_zero_lengths_inactive_rows_and_strided_window():
    device = torch.device("cuda:0")
    gen = torch.Generator(device=device).manual_seed(90210)
    tokens = 1536
    max_q = 700

    # Embed the 512x576 window inside a wider strided backing buffer like ContinuationPool.
    row_stride_elems = WINDOW * D_QK + 4096
    raw_backing = (torch.randn((6, row_stride_elems), generator=gen, device=device, dtype=torch.float32) * 0.25).to(
        torch.bfloat16
    )
    raw_backing[0].fill_(float("nan"))
    window_buf = raw_backing[:, 512 : 512 + WINDOW * D_QK].view(6, WINDOW, D_QK)
    assert not window_buf.is_contiguous() and window_buf.stride(-1) == 1

    query = (torch.randn((tokens, HEADS, D_QK), generator=gen, device=device, dtype=torch.float32) * 0.25).to(
        torch.bfloat16
    )
    kv_record = (torch.randn((tokens, D_QK), generator=gen, device=device, dtype=torch.float32) * 0.25).to(
        torch.bfloat16
    )
    sink = torch.randn((HEADS,), generator=gen, device=device, dtype=torch.float32) * 0.5
    decl_out = torch.empty((tokens, HEADS, D_NOPE), dtype=torch.bfloat16, device=device)

    decl_meta = _make_metadata(
        [1, 2, 3, 4], [0, 0, 0, 0], [0, 64, 64, 577, 1277], [64, 0, 513, 700],
        tokens=tokens, max_query_tokens=max_q, device=device,
    )
    op = PreparedSwaMlaPrefill(
        query, window_buf, kv_record, sink, decl_out, decl_meta, scale=MLA_SCALE, window_size=WINDOW
    ).functional()

    cases = [
        # Mixed lengths with 0-length request, ring wraparound, and trailing ragged padding.
        ([3, 1, 5, 2], [0, 999, 511, 1234], [0, 65, 65, 578, 1278], [65, 0, 513, 700]),
        # Inactive request (state_row=0 with non-zero span), negative start_pos prefix, and all-zero tail.
        ([0, 4, 2, 0], [512, -3, 63, 100], [0, 64, 129, 130, 130], [64, 65, 1, 0]),
        # All four requests empty/inactive -> entire [tokens, H, D_NOPE] output must be exact 0.0.
        ([1, 0, 3, 2], [512, 0, 1234, 63], [0, 0, 0, 0, 0], [0, 0, 0, 0]),
    ]
    for state_rows, starts, qsl, lengths in cases:
        m = _make_metadata(state_rows, starts, qsl, lengths, tokens=tokens, max_query_tokens=max_q, device=device)
        q_run = query.clone()
        kv_run = kv_record.clone()
        # Poison unowned tail rows and zero-length gaps with NaNs.
        q_run[qsl[-1] :].fill_(float("nan"))
        kv_run[qsl[-1] :].fill_(float("nan"))
        actual = torch.full((tokens, HEADS, D_NOPE), -77.0, dtype=torch.bfloat16, device=device)
        op(q_run, kv_run, m, window=window_buf, output=actual)

        ref = _reference_swa_mla_prefill(
            query, window_buf, kv_record, sink, m.state_rows, m.starts, m.query_start_loc, m.lengths,
            round_before_sink=True,
        )
        assert torch.isfinite(actual).all()
        assert actual[qsl[-1] :].float().abs().max().item() == 0.0
        for b, (sr, st, q0, q1, ln) in enumerate(zip(state_rows, starts, qsl[:-1], qsl[1:], lengths, strict=True)):
            if sr <= 0 or ln <= 0:
                if q1 > q0:
                    assert actual[q0:q1].float().abs().max().item() == 0.0
            else:
                neg_prefix = max(0, -st)
                if neg_prefix > 0:
                    assert actual[q0 : q0 + min(ln, neg_prefix)].float().abs().max().item() == 0.0
                active_slice = slice(q0 + min(ln, neg_prefix), q0 + ln)
                if active_slice.start < active_slice.stop:
                    diff = (actual[active_slice].float() - ref[active_slice].float()).abs()
                    assert diff.max().item() < 5e-3, f"batch {b} max_abs={diff.max().item()}"


@torch.no_grad()
def test_swa_mla_prefill_and_ring_commit_compile_and_cudagraph_replay():
    device = torch.device("cuda:0")
    gen = torch.Generator(device=device).manual_seed(424242)
    tokens = 640
    max_q = 513

    row_stride_elems = WINDOW * D_QK + 2048
    backing = torch.zeros((4, row_stride_elems), dtype=torch.bfloat16, device=device)
    window_buf = backing[:, 256 : 256 + WINDOW * D_QK].view(4, WINDOW, D_QK)
    window_seed = (torch.randn((4, WINDOW, D_QK), generator=gen, device=device, dtype=torch.float32) * 0.2).to(
        torch.bfloat16
    )
    window_seed[0].zero_()
    window_buf.copy_(window_seed)

    query = (torch.randn((tokens, HEADS, D_QK), generator=gen, device=device, dtype=torch.float32) * 0.2).to(
        torch.bfloat16
    )
    kv_record = (torch.randn((tokens, D_QK), generator=gen, device=device, dtype=torch.float32) * 0.2).to(
        torch.bfloat16
    )
    sink = torch.randn((HEADS,), generator=gen, device=device, dtype=torch.float32) * 0.5
    static_out = torch.empty((tokens, HEADS, D_NOPE), dtype=torch.bfloat16, device=device)

    meta = _make_metadata([1, 2], [511, 1234], [0, 513, 578], [513, 65], tokens=tokens, max_query_tokens=max_q, device=device)
    op = PreparedSwaMlaPrefill(
        query, window_buf, kv_record, sink, static_out, meta, scale=MLA_SCALE, window_size=WINDOW
    ).functional()

    def step():
        attended = op(query, kv_record, meta, window=window_buf)
        commit_swa_ring(
            kv_record,
            window_buf,
            meta.state_rows,
            meta.starts,
            meta.query_start_loc,
            meta.positions,
            meta.lengths,
            WINDOW,
        )
        static_out.copy_(attended)

    # 1. Verify torch.compile(fullgraph=True) parity before graph capture.
    torch_compiled = torch.compile(
        lambda q, kv: op(q, kv, meta, window=window_buf),
        fullgraph=True,
    )
    tc_out = torch_compiled(query, kv_record)
    ref_out = _reference_swa_mla_prefill(
        query, window_seed, kv_record, sink, meta.state_rows, meta.starts, meta.query_start_loc, meta.lengths
    )
    assert (tc_out[:578].float() - ref_out[:578].float()).abs().max().item() < 5e-3
    assert tc_out[578:].float().abs().max().item() == 0.0

    # 2. Verify Flare compile_forward + CUDA graph capture + zero-allocation replay + exact ring commit.
    compiled = compile_forward(step)
    window_buf.copy_(window_seed)
    compiled()
    torch.cuda.synchronize(device)

    window_buf.copy_(window_seed)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()

    for scale_factor in (1.0, -0.75):
        query.mul_(scale_factor)
        kv_record.mul_(scale_factor)
        window_buf.copy_(window_seed)
        static_out.fill_(float("nan"))
        expected_out = _reference_swa_mla_prefill(
            query, window_seed, kv_record, sink, meta.state_rows, meta.starts, meta.query_start_loc, meta.lengths
        )
        expected_win = window_seed.clone()
        for b, (sr, st, q0, ln) in enumerate(zip([1, 2], [511, 1234], [0, 513], [513, 65], strict=True)):
            for off in range(max(0, ln - WINDOW), ln):
                expected_win[sr, (st + off) % WINDOW] = kv_record[q0 + off]

        torch.cuda.synchronize(device)
        before = torch.cuda.memory_stats(device)
        graph.replay()
        torch.cuda.synchronize(device)
        after = torch.cuda.memory_stats(device)
        for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
            assert before[key] == after[key], key

        assert torch.isfinite(static_out).all()
        assert (static_out[:578].float() - expected_out[:578].float()).abs().max().item() < 5e-3
        assert static_out[578:].float().abs().max().item() == 0.0
        assert torch.equal(window_buf[0], window_seed[0])
        assert torch.equal(window_buf[3], window_seed[3])
        assert torch.equal(window_buf[1], expected_win[1])
        assert torch.equal(window_buf[2], expected_win[2])
    graph.reset()


@torch.no_grad()
def test_sliding_attention_fused_gate_strided_norms_and_skip_post_norm_cuda(tmp_path):
    """Verify PreparedSlidingAttention with fused_gate=True strided norms and apply_post_norm=False."""
    import json
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from hyperflare_recipes.components.paged_batch import Metadata
    from hyperflare_recipes.recipes.hy4_flash.attention import AttentionScratch, PreparedSlidingAttention
    from hyperflare_recipes.recipes.hy4_flash.config import (
        HIDDEN_SIZE,
        KV_LORA_RANK,
        QK_NOPE_HEAD_DIM,
        QK_ROPE_HEAD_DIM,
        Q_LORA_RANK,
        SWA_HEADS,
        SWA_WINDOW,
        V_HEAD_DIM,
        ModelConfig,
    )
    from hyperflare_recipes.recipes.hy4_flash.pool import ContinuationPool
    from hyperflare_recipes.recipes.hy4_flash.rotary import InterleavedMRoPE

    engram_hvs = {
        str(lid): [[101, 103, 107, 109, 113, 127, 131, 137], [139, 149, 151, 157, 163, 167, 173, 179]]
        for lid in (2, 12, 22, 32)
    }
    layer_types = ["indexed_attention" if i < 2 or (i - 1) % 4 == 0 else "sliding_attention" for i in range(42)]
    cfg_dict = {
        "model_type": "hy_v4_flash",
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
            "num_attention_heads": 32,
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
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg_dict))
    cfg = ModelConfig.load(tmp_path)
    device = torch.device("cuda:0")
    rotary = InterleavedMRoPE(cfg, device)
    gen = torch.Generator(device=device).manual_seed(777)
    bf16 = lambda *s: (torch.randn(s, generator=gen, device=device, dtype=torch.float32) * 0.04).to(torch.bfloat16)

    qkv_base = bf16(Q_LORA_RANK + KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN_SIZE)
    gate_w = bf16(SWA_HEADS * V_HEAD_DIM, HIDDEN_SIZE)
    post_norm_w = (torch.rand(HIDDEN_SIZE, generator=gen, device=device, dtype=torch.float32) + 0.5).to(torch.bfloat16)
    base_weights = {
        "q_b": bf16(SWA_HEADS * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM), Q_LORA_RANK),
        "kv_b": bf16(SWA_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK),
        "o_proj": bf16(HIDDEN_SIZE, SWA_HEADS * V_HEAD_DIM),
        "q_norm": torch.ones(Q_LORA_RANK, dtype=torch.bfloat16, device=device),
        "kv_norm": torch.ones(KV_LORA_RANK, dtype=torch.bfloat16, device=device),
        "post_proj_norm": post_norm_w,
        "sink": torch.randn(SWA_HEADS, generator=gen, device=device, dtype=torch.float32) * 0.5,
    }
    weights_unfused = {**base_weights, "qkv_a": qkv_base, "gate": gate_w, "fused_gate": False}
    weights_fused = {**base_weights, "qkv_a": torch.cat((qkv_base, gate_w), dim=0), "gate": None, "fused_gate": True}

    tokens = 128
    x = (torch.randn(tokens, HIDDEN_SIZE, generator=gen, device=device, dtype=torch.float32) * 0.25).to(torch.bfloat16)
    pool_a = ContinuationPool.allocate(cfg, 3, 4, device, dfly=False)
    pool_b = ContinuationPool.allocate(cfg, 3, 4, device, dfly=False)
    scratch_a = AttentionScratch.allocate(tokens, 1, tokens, device)
    scratch_b = AttentionScratch.allocate(tokens, 1, tokens, device)
    meta = Metadata.pack(
        starts=torch.tensor([0], dtype=torch.int64, device=device),
        lengths=torch.tensor([tokens], dtype=torch.int32, device=device),
        state_rows=torch.tensor([1], dtype=torch.int64, device=device),
        pages=torch.arange(4, dtype=torch.int32, device=device).unsqueeze(0),
        tokens=torch.ones((1, tokens), dtype=torch.int64, device=device),
        live_rows=torch.tensor([1], dtype=torch.int32, device=device),
        token_capacity=tokens,
        sample_count=1,
        max_query_tokens=tokens,
    )
    arena = ScratchArena(device)
    with RecordedPreparationSession(
        device=device, checkpoint="hy4-swa-fused-post", workload_schema="hy4-swa-fused-post", numerical_policy="default"
    ) as session:
        from hyperflare_recipes.components.rms_norm import PreparedRMSNorm

        session.register_scratch(arena)
        attn_normed = PreparedSlidingAttention(
            2, weights_unfused, x, scratch_a, meta, pool_a, rotary,
            mode="prefill", eps=1e-5, session=session, name="unfused_norm", workspace=arena, apply_post_norm=True,
        )
        attn_raw_unfused = PreparedSlidingAttention(
            2, weights_unfused, x, scratch_a, meta, pool_a, rotary,
            mode="prefill", eps=1e-5, session=session, name="unfused_raw", workspace=arena, apply_post_norm=False,
        )
        attn_raw_fused = PreparedSlidingAttention(
            2, weights_fused, x, scratch_b, meta, pool_b, rotary,
            mode="prefill", eps=1e-5, session=session, name="fused_raw", workspace=arena, apply_post_norm=False,
        )
        standalone_norm = PreparedRMSNorm(
            scratch_a.o_proj_out, post_norm_w, scratch_a.output, eps=1e-5, session=session, name="ext_post_norm"
        ).functional()
        session.tune()

        pool_a.clear_for_preparation()
        out_normed, _, _ = attn_normed(x, meta)
        win_normed = pool_a.window(2)[1].clone()

        pool_a.clear_for_preparation()
        out_raw_u, _, _ = attn_raw_unfused(x, meta)
        assert torch.equal(standalone_norm(out_raw_u), out_normed)
        assert torch.equal(pool_a.window(2)[1], win_normed)

        pool_b.clear_for_preparation()
        out_raw_f, _, _ = attn_raw_fused(x, meta)
        rel_err = (out_raw_f.float() - out_raw_u.float()).norm().item() / out_raw_u.float().norm().item()
        assert rel_err < 5e-3
        ring_rel = (pool_b.window(2)[1, :tokens].float() - win_normed[:tokens].float()).norm().item() / max(
            win_normed[:tokens].float().norm().item(), 1e-6
        )
        assert ring_rel < 5e-3
