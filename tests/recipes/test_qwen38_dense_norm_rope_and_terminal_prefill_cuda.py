# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CUDA parity tests for fused Q/K RMSNorm+RoPE+KV-write and Qwen3.8-27B TerminalPrefill split."""

from __future__ import annotations

from functools import partial
import os
from pathlib import Path

import pytest
import torch

from hyperflare_recipes.recipes import resolve

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1",
)


@pytest.mark.parametrize("layout", ["packed", "unpacked"])
@pytest.mark.parametrize("q_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("cache_dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("zero_centered", [True, False])
@pytest.mark.parametrize("tokens", [1, 7, 32, 128, 512])
@torch.no_grad()
def test_qk_head_norm_rope_kv_write_matches_canonical_and_kv_writer(
    layout, q_dtype, cache_dtype, weight_dtype, zero_centered, tokens
):
    from b12x.preparation import PreparationSession
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm, apply_rotary_pos_emb
    from hyperflare_recipes.components.head_norm_rope import head_norm_rope, qk_head_norm_rope_kv_write
    from hyperflare_recipes.components.paged_attention import PreparedPagedKVWriter
    from hyperflare_recipes.recipes.qwen38_dense.config import rotary_axes
    from hyperflare_recipes.recipes.qwen38_dense.decoder import rotary_embeddings

    device = torch.device("cuda:0")
    heads, kv_heads, dim, page_size = 24, 4, 256, 128
    rotary_cfg = {"partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10], "rope_theta": 10000000}
    inv_freq = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=device) / 64))
    axes = rotary_axes(rotary_cfg, 32).to(device)

    gen = torch.Generator(device="cpu").manual_seed(
        2026 + tokens * 13 + int(layout == "packed") * 7 + int(zero_centered) * 3
    )
    if zero_centered:
        q_norm = Qwen3_5RMSNorm(dim, eps=1e-6).to(device=device, dtype=weight_dtype).eval()
        k_norm = Qwen3_5RMSNorm(dim, eps=1e-6).to(device=device, dtype=weight_dtype).eval()
        q_norm.weight.data.copy_((torch.randn((dim,), generator=gen) * 0.25).to(weight_dtype).to(device))
        k_norm.weight.data.copy_((torch.randn((dim,), generator=gen) * 0.25).to(weight_dtype).to(device))
    else:
        q_norm = torch.nn.RMSNorm(dim, eps=1e-6, device=device, dtype=weight_dtype).eval()
        k_norm = torch.nn.RMSNorm(dim, eps=1e-6, device=device, dtype=weight_dtype).eval()
        q_norm.weight.data.copy_((1.0 + torch.randn((dim,), generator=gen) * 0.25).to(weight_dtype).to(device))
        k_norm.weight.data.copy_((1.0 + torch.randn((dim,), generator=gen) * 0.25).to(weight_dtype).to(device))

    q_width, kv_width = heads * dim, kv_heads * dim
    if layout == "packed":
        qgkv = (torch.randn((tokens, 2 * q_width + 2 * kv_width), generator=gen) * 1.5).bfloat16().to(device)
        query, _, key_flat, value_flat = qgkv.split((q_width, q_width, kv_width, kv_width), dim=-1)
        query = query.view(tokens, heads, dim)
        key = key_flat.view(tokens, kv_heads, dim)
        value = value_flat.view(tokens, kv_heads, dim)
    else:
        qg = (torch.randn((tokens, 2 * q_width), generator=gen) * 1.5).bfloat16().to(device)
        query, _ = qg.view(tokens, heads, 2 * dim).chunk(2, dim=-1)
        key = (torch.randn((tokens, kv_heads, dim), generator=gen) * 1.5).bfloat16().to(device)
        value = (torch.randn((tokens, kv_heads, dim), generator=gen) * 1.5).bfloat16().to(device)

    # Mix 3-axis MRoPE positions, negative slots, and out-of-range slots to verify guard parity.
    positions = torch.randint(0, 8192, (tokens, 3), generator=gen, dtype=torch.int64).to(device)
    cos, sin = rotary_embeddings(positions, inv_freq, axes, 3)

    pages = max(2, (tokens + page_size - 1) // page_size + 1)
    slots = torch.arange(tokens, dtype=torch.int64, device=device)
    if tokens >= 4:
        slots[-2] = -1
        slots[-1] = pages * page_size + 5

    # Inject FP8 saturation / special values into row 0 of value when cache is FP8.
    if cache_dtype == torch.float8_e4m3fn:
        specials = torch.tensor(
            [0.0, 448.0, 500.0, -500.0, 1e4, -1e4, float("inf"), float("-inf"), float("nan")],
            dtype=torch.bfloat16,
            device=device,
        )
        value[0, 0, : specials.numel()].copy_(specials)

    kc_ref = torch.zeros((pages, page_size, kv_heads, dim), dtype=cache_dtype, device=device)
    vc_ref = torch.zeros_like(kc_ref)
    kc_fused = torch.zeros_like(kc_ref)
    vc_fused = torch.zeros_like(vc_ref)
    kc_cache_only = torch.zeros_like(kc_ref)
    vc_cache_only = torch.zeros_like(vc_ref)

    q_fused = torch.empty((tokens, heads, dim), dtype=q_dtype, device=device)
    qk_head_norm_rope_kv_write(
        query, key, value, q_norm.weight, k_norm.weight, cos, sin, slots,
        q_fused, kc_fused, vc_fused, eps=1e-6, zero_centered=zero_centered,
    )
    qk_head_norm_rope_kv_write(
        None, key, value, None, k_norm.weight, cos, sin, slots,
        None, kc_cache_only, vc_cache_only, eps=1e-6, zero_centered=zero_centered,
    )

    # Cache-only path must match full path bit-for-bit on K and V caches.
    assert torch.equal(kc_cache_only.view(torch.uint8), kc_fused.view(torch.uint8))
    assert torch.equal(vc_cache_only.view(torch.uint8), vc_fused.view(torch.uint8))

    # Build canonical reference + PreparedPagedKVWriter.
    with PreparationSession(device=device, autotune=False) as session:
        q_ref_bf16, k_ref_bf16 = apply_rotary_pos_emb(q_norm(query), k_norm(key), cos, sin)
        q_ref_bf16, k_ref_bf16 = q_ref_bf16.contiguous(), k_ref_bf16.contiguous()
        writer = PreparedPagedKVWriter(
            k_ref_bf16, value.contiguous(), kc_ref, vc_ref, slots,
            session=session, name="ref_kv_write", caller_owned_placeholders=True,
        ).functional()
        writer(k_ref_bf16, value.contiguous(), slots)

    # Value cache must be 100% bit-exact vs PreparedPagedKVWriter (including FP8 saturation/NaN/guards).
    assert torch.equal(vc_fused.view(torch.uint8), vc_ref.view(torch.uint8))
    # Untouched sentinel page must remain all-zero.
    assert int(kc_fused[-1].view(torch.uint8).count_nonzero().item()) == 0
    assert int(vc_fused[-1].view(torch.uint8).count_nonzero().item()) == 0

    if q_dtype == torch.bfloat16:
        rel_l2 = ((q_fused.float() - q_ref_bf16.float()).norm() / q_ref_bf16.float().norm().clamp_min(1e-6)).item()
        frac_diff = (q_fused != q_ref_bf16).float().mean().item()
        torch.testing.assert_close(q_fused, q_ref_bf16, rtol=2e-2, atol=2e-2)
        assert rel_l2 < 1e-4, f"q rel_l2={rel_l2:.6g}"
        assert frac_diff < 5e-4, f"q frac_diff={frac_diff:.6%}"
    elif zero_centered:
        q_hnr = torch.empty((tokens, heads, dim), dtype=q_dtype, device=device)
        head_norm_rope(query, q_norm.weight, cos, sin, q_hnr, eps=1e-6)
        assert torch.equal(q_fused.view(torch.uint8), q_hnr.view(torch.uint8))

    # Key cache parity against canonical K + PreparedPagedKVWriter.
    k_rel_l2 = ((kc_fused.float() - kc_ref.float()).norm() / kc_ref.float().norm().clamp_min(1e-6)).item()
    k_frac_diff = (kc_fused.view(torch.uint8) != kc_ref.view(torch.uint8)).float().mean().item()
    torch.testing.assert_close(kc_fused.float(), kc_ref.float(), rtol=2e-2, atol=2e-2)
    assert k_rel_l2 < 5e-4, f"k_cache rel_l2={k_rel_l2:.6g}"
    assert k_frac_diff < 5e-4, f"k_cache frac_diff={k_frac_diff:.6%}"


@torch.no_grad()
def test_qk_head_norm_rope_kv_write_compiled_cuda_graph():
    from ignition.preparation.compilation import compile_forward
    from hyperflare_recipes.components.head_norm_rope import qk_head_norm_rope_kv_write
    from hyperflare_recipes.recipes.qwen38_dense.decoder import rotary_embeddings

    device = torch.device("cuda:0")
    tokens, heads, kv_heads, dim, page_size = 64, 24, 4, 256, 128
    q_width, kv_width = heads * dim, kv_heads * dim
    inv_freq = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=device) / 64))
    qgkv = torch.randn((tokens, 2 * q_width + 2 * kv_width), dtype=torch.bfloat16, device=device)
    qw = torch.randn((dim,), dtype=torch.float32, device=device) * 0.2
    kw = torch.randn((dim,), dtype=torch.float32, device=device) * 0.2
    positions = torch.arange(tokens, dtype=torch.int64, device=device)
    slots = torch.arange(tokens, dtype=torch.int64, device=device)
    slots[-3:] = -1
    k_cache = torch.zeros((2, page_size, kv_heads, dim), dtype=torch.float8_e4m3fn, device=device)
    v_cache = torch.zeros_like(k_cache)
    q_captured = torch.empty((tokens, heads, dim), dtype=torch.bfloat16, device=device)

    def run():
        cos, sin = rotary_embeddings(positions, inv_freq)
        query, _, key, value = qgkv.split((q_width, q_width, kv_width, kv_width), dim=-1)
        q_local = torch.empty((tokens, heads, dim), dtype=torch.bfloat16, device=device)
        qk_head_norm_rope_kv_write(
            query.view(tokens, heads, dim), key.view(tokens, kv_heads, dim), value.view(tokens, kv_heads, dim),
            qw, kw, cos, sin, slots, q_local, k_cache, v_cache, eps=1e-6, zero_centered=True,
        )
        q_captured.copy_(q_local)

    compiled = compile_forward(run)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        compiled()
    stream.synchronize()
    k_cache.zero_()
    v_cache.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        compiled()
    stream.synchronize()

    # Mutate inputs in place and replay with zero allocation growth.
    qgkv.normal_(std=1.2)
    positions.copy_(torch.arange(100, 100 + tokens, dtype=torch.int64, device=device))
    slots.copy_(torch.arange(page_size, page_size + tokens, dtype=torch.int64, device=device))
    slots[-2:] = -1
    k_cache.zero_()
    v_cache.zero_()
    allocated = torch.cuda.memory_allocated(device)
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    assert torch.cuda.memory_allocated(device) == allocated
    graph.reset()

    k_eager = torch.zeros_like(k_cache)
    v_eager = torch.zeros_like(v_cache)
    q_eager = torch.empty_like(q_captured)
    cos, sin = rotary_embeddings(positions, inv_freq)
    query, _, key, value = qgkv.split((q_width, q_width, kv_width, kv_width), dim=-1)
    qk_head_norm_rope_kv_write(
        query.view(tokens, heads, dim), key.view(tokens, kv_heads, dim), value.view(tokens, kv_heads, dim),
        qw, kw, cos, sin, slots, q_eager, k_eager, v_eager, eps=1e-6, zero_centered=True,
    )
    assert torch.equal(q_captured, q_eager)
    assert torch.equal(k_cache.view(torch.uint8), k_eager.view(torch.uint8))
    assert torch.equal(v_cache.view(torch.uint8), v_eager.view(torch.uint8))


@pytest.mark.parametrize("checkpoint,online_mxfp8", [("qwen3.8-27b-qad", True), ("qwen3.8-27b", False)])
@torch.no_grad()
def test_qwen38_dense_nonterminal_and_terminal_prefill_checkpoint(checkpoint, online_mxfp8):
    """Non-terminal prefill (sample=False, Layer 63 cache-only) + terminal prefill match full prefill."""
    from b12x.preparation import PreparationSession
    from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
    from ignition.components.sampling import DeviceRequestState, RequestAdmission
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from hyperflare_recipes.components.dflash2 import PreparedContext
    from hyperflare_recipes.components.paged_batch import Metadata, RotaryBatchInputs, prepare_batch
    from hyperflare_recipes.components.paged_generation import PreparedGeneration
    from hyperflare_recipes.recipes.qwen38_dense.decoder import PreparedDecoder, Scratch
    from hyperflare_recipes.recipes.qwen38_dense.pool import ContinuationPool
    from hyperflare_recipes.recipes.qwen38_dense.weights import ModelWeights

    device = torch.device("cuda:0")
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    target = resolve(checkpoint).snapshot(hub)
    draft = hub / "models--incoai--Qwen3.8-27B-DFlash2/snapshots/015e795645c74b1a0eeef3b570031fb62e769bc5"
    weights = ModelWeights.load(
        inspect_indexed_checkpoint(target), device, mtp=True, draft_snapshot=draft, online_mxfp8=online_mxfp8,
    )

    tokens, slots, blocks, pages = 512, 2, 16, 12

    def build_env():
        pool = ContinuationPool.for_model(
            weights.config, slots + 1, pages, device, mtp=True, verification_width=8, rotary_offset=True,
        )
        requests = DeviceRequestState.allocate(device, vocab_size=weights.config.vocabulary, capacity=slots)
        inputs = RotaryBatchInputs.allocate(tokens, blocks, device, capacity=slots)
        completion = CompletionOutputs.allocate((2, slots, VECTOR_TOKENS_FIELD + 8), device=device)
        return pool, requests, inputs, completion

    ref_pool, ref_req, ref_in, ref_comp = build_env()
    split_pool, split_req, split_in, split_comp = build_env()
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.zeros(1, dtype=torch.int64, device=device)

    with PreparationSession(device=device, autotune=False) as session:
        arena = ScratchArena(device)
        ctx_factory = partial(PreparedContext, runtime=None, coordinator=None, compact_prefill=True)

        ref_m = Metadata.allocate(tokens, 1, blocks, device)
        ref_s = Scratch.allocate(tokens, 1, weights.config, device)
        ref_dec = PreparedDecoder(
            weights, ref_s, ref_m, ref_pool, mode="prefill",
            vocabulary_runtime=None, coordinator=None, session=session, name="ref512", workspace=arena,
            sample=True,
        )
        ref_batch = prepare_batch(ref_in, ref_req, ref_m, mode="prefill", shard=weights.shard, pool=ref_pool)
        ref_gen = PreparedGeneration(
            ref_dec, ref_batch, ref_comp, frame, sequence, runtime=None,
            coordinator=None, session=session, name="ref512.gen", workspace=arena,
            context_factory=ctx_factory, sample=True, absorb_context=True,
        )

        nonterm_m = Metadata.allocate(tokens, 1, blocks, device)
        nonterm_s = Scratch.allocate(tokens, 1, weights.config, device)
        nonterm_dec = PreparedDecoder(
            weights, nonterm_s, nonterm_m, split_pool, mode="prefill",
            vocabulary_runtime=None, coordinator=None, session=session, name="nonterm512", workspace=arena,
            sample=False,
        )
        assert nonterm_dec.head is None
        assert nonterm_dec.layers[-1].cache_only is True
        assert nonterm_dec.layers[-1].layer.self_attn.cache_only is True
        assert nonterm_dec.layers[-1].layer.self_attn.attention is None
        if not online_mxfp8:
            assert "q_proj" not in nonterm_dec.layers[-1].layer.self_attn.projections
        nonterm_batch = prepare_batch(split_in, split_req, nonterm_m, mode="prefill", shard=weights.shard, pool=split_pool)
        nonterm_gen = PreparedGeneration(
            nonterm_dec, nonterm_batch, split_comp, frame, sequence, runtime=None,
            coordinator=None, session=session, name="nonterm512.gen", workspace=arena,
            context_factory=ctx_factory, sample=False, absorb_context=True,
        )
        assert nonterm_gen.context is not None
        assert nonterm_gen.compilation_variant == "record_context"

        term_m = Metadata.allocate(tokens, 1, blocks, device)
        term_s = Scratch.allocate(tokens, 1, weights.config, device)
        term_dec = PreparedDecoder(
            weights, term_s, term_m, split_pool, mode="prefill",
            vocabulary_runtime=None, coordinator=None, session=session, name="term512", workspace=arena,
            sample=True,
        )
        assert term_dec.head is not None
        assert term_dec.layers[-1].cache_only is False
        term_batch = prepare_batch(split_in, split_req, term_m, mode="prefill", shard=weights.shard, pool=split_pool)
        term_gen = PreparedGeneration(
            term_dec, term_batch, split_comp, frame, sequence, runtime=None,
            coordinator=None, session=session, name="term512.gen", workspace=arena,
            context_factory=ctx_factory, sample=True, absorb_context=True,
        )
        arena.finalize()

        compiled = {
            "ref": compile_forward(ref_gen),
            "nonterm": compile_forward(nonterm_gen),
            "term": compile_forward(term_gen),
        }

        def init_env(req, pool, comp, prompt_tokens):
            pool.clear_for_preparation()
            comp.records.zero_()
            req.admit(0, RequestAdmission(generation=7, seed=12345, max_new_tokens=16, top_k=1), prompt_tokens)

        def stage_chunk(inp, *, consumed, live_len, sample_enabled, token_values, rope_offset):
            inp.request_state_rows.zero_()
            inp.request_state_rows[0, 0] = 1
            inp.row_slot_ids.fill_(-1)
            inp.row_slot_ids[0] = 0
            inp.row_generations.zero_()
            inp.row_generations[0] = 7
            inp.live_rows[0] = 1 if live_len > 0 else 0
            inp.query_lengths.zero_()
            inp.query_lengths[0] = live_len
            inp.sample_enabled.zero_()
            inp.sample_enabled[0] = int(sample_enabled)
            inp.token_ids.zero_()
            inp.prefill_positions.zero_()
            inp.page_tables.fill_(-1)
            if live_len > 0:
                inp.token_ids[:live_len].copy_(token_values[:live_len])
                base_pos = torch.arange(consumed, consumed + live_len, dtype=torch.int64, device=device)
                inp.prefill_positions[:live_len, 0].copy_(base_pos + rope_offset)
                inp.prefill_positions[:live_len, 1].copy_(base_pos + rope_offset + 5)
                inp.prefill_positions[:live_len, 2].copy_(base_pos + rope_offset + 9)
                needed_pages = (consumed + live_len + 127) // 128
                inp.page_tables[0, :needed_pages] = torch.arange(needed_pages, dtype=torch.int32, device=device)

        gen = torch.Generator(device="cpu").manual_seed(777)
        tok0 = torch.randint(10, 50000, (tokens,), generator=gen, dtype=torch.int64).to(device)
        chunk1_len = 291
        tok1 = torch.randint(10, 50000, (chunk1_len,), generator=gen, dtype=torch.int64).to(device)
        full_prompt = torch.cat((tok0, tok1)).cpu().tolist()

        # Capture CUDA graphs.
        stream = torch.cuda.Stream(device=device)
        init_env(ref_req, ref_pool, ref_comp, full_prompt)
        init_env(split_req, split_pool, split_comp, full_prompt)
        graphs = {}
        for key in ("ref", "nonterm", "term"):
            with torch.cuda.stream(stream):
                compiled[key]()
            stream.synchronize()
            g = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(g, stream=stream):
                compiled[key]()
            stream.synchronize()
            graphs[key] = g

        try:
            init_env(ref_req, ref_pool, ref_comp, full_prompt)
            init_env(split_req, split_pool, split_comp, full_prompt)
            # Poison last_hidden on split_pool to verify non-terminal chunk leaves it untouched.
            split_pool.last_hidden[1].fill_(3.25)
            poisoned_hidden = split_pool.last_hidden[1].clone()

            # Chunk 0: non-terminal (512 tokens, sample_enabled=0, 3-axis MRoPE offset=13).
            sequence.fill_(11)
            stage_chunk(ref_in, consumed=0, live_len=tokens, sample_enabled=False, token_values=tok0, rope_offset=13)
            stage_chunk(split_in, consumed=0, live_len=tokens, sample_enabled=False, token_values=tok0, rope_offset=13)
            allocated = torch.cuda.memory_allocated(device)
            with torch.cuda.stream(stream):
                graphs["ref"].replay()
                graphs["nonterm"].replay()
            stream.synchronize()
            assert torch.cuda.memory_allocated(device) == allocated

            assert torch.equal(split_comp.records, ref_comp.records)
            assert torch.equal(split_req.consumed_tokens, ref_req.consumed_tokens)
            assert int(split_req.consumed_tokens[0].item()) == tokens
            assert torch.equal(split_req.pending_token, ref_req.pending_token)
            assert int(split_req.pending_token[0].item()) == -1
            assert torch.equal(split_req.rng_counter, ref_req.rng_counter)
            assert torch.equal(split_pool.last_hidden[1], poisoned_hidden), "non-terminal mutated last_hidden"
            assert torch.equal(
                split_pool.reserved_region("rotary_offset"),
                ref_pool.reserved_region("rotary_offset"),
            )
            assert int(split_pool.reserved_region("rotary_offset")[1, 0].item()) == 22

            # Verify ALL 16 full-attention KV caches (including Layer 63 cache-only!) and DFlash2 draft rings match ref.
            for name, page in split_pool.pages.items():
                ref_page = ref_pool.pages[name]
                diff = (page.view(torch.float8_e4m3fn).float() - ref_page.view(torch.float8_e4m3fn).float()).norm()
                denom = ref_page.view(torch.float8_e4m3fn).float().norm().clamp_min(1e-6)
                rel = (diff / denom).item()
                assert rel < 1e-3, f"chunk 0 page {name} diverged: rel_l2={rel:.6g}"
            for layer_idx in range(5):
                for kind in ("key", "value"):
                    reg_name = f"draft{layer_idx}.{kind}"
                    s_reg = split_pool.reserved_region(reg_name)[1].float()
                    r_reg = ref_pool.reserved_region(reg_name)[1].float()
                    assert s_reg.count_nonzero() > 0, f"non-terminal chunk did not populate {reg_name}"
                    rel = ((s_reg - r_reg).norm() / r_reg.norm().clamp_min(1e-6)).item()
                    assert rel < 1e-3, f"chunk 0 draft ring {reg_name} diverged: rel_l2={rel:.6g}"

            # Inactive replay on non-terminal graph must preserve all persistent state bit-for-bit.
            snap_backing = split_pool.backing.clone()
            snap_pages = {name: value.clone() for name, value in split_pool.pages.items()}
            stage_chunk(split_in, consumed=tokens, live_len=0, sample_enabled=False, token_values=tok1, rope_offset=0)
            with torch.cuda.stream(stream):
                graphs["nonterm"].replay()
            stream.synchronize()
            assert torch.equal(split_pool.backing, snap_backing), "inactive non-terminal mutated pool"
            for name, snap in snap_pages.items():
                assert torch.equal(split_pool.pages[name], snap), f"inactive non-terminal mutated page {name}"

            # Chunk 1: terminal chunk (291 tokens, sample_enabled=1) — samples token and writes last_hidden.
            sequence.fill_(12)
            stage_chunk(ref_in, consumed=tokens, live_len=chunk1_len, sample_enabled=True, token_values=tok1, rope_offset=20)
            stage_chunk(split_in, consumed=tokens, live_len=chunk1_len, sample_enabled=True, token_values=tok1, rope_offset=20)
            with torch.cuda.stream(stream):
                graphs["ref"].replay()
                graphs["term"].replay()
            stream.synchronize()

            assert torch.equal(split_req.consumed_tokens, ref_req.consumed_tokens)
            assert int(split_req.consumed_tokens[0].item()) == tokens + chunk1_len
            assert torch.equal(
                split_pool.reserved_region("rotary_offset"),
                ref_pool.reserved_region("rotary_offset"),
            )
            assert int(split_pool.reserved_region("rotary_offset")[1, 0].item()) == 29
            assert int(split_req.pending_token[0].item()) >= 0
            assert int(split_req.generated_tokens[0].item()) == 1
            assert torch.equal(split_req.pending_token, ref_req.pending_token)
            hidden_rel = (
                (split_pool.last_hidden[1].float() - ref_pool.last_hidden[1].float()).norm()
                / ref_pool.last_hidden[1].float().norm().clamp_min(1e-6)
            ).item()
            assert hidden_rel < 2e-3, f"chunk 1 last_hidden diverged: rel_l2={hidden_rel:.6g}"
        finally:
            stream.synchronize()
            for g in graphs.values():
                g.reset()
