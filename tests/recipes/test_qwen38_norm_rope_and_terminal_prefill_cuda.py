# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CUDA parity tests for fused QSA Q/K RMSNorm + 3-axis MRoPE and non-terminal prefill decoders."""

from __future__ import annotations

from functools import partial
import os
import socket
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _requires_cuda():
    if os.environ.get("FLARE_RUN_CUDA_TESTS") != "1":
        pytest.skip("requires FLARE_RUN_CUDA_TESTS=1")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")


def _requires_tp2():
    _requires_cuda()
    if os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1":
        pytest.skip("requires FLARE_RUN_TP_CUDA_TESTS=1")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least two CUDA devices")


@pytest.mark.parametrize("q_heads,kv_heads", [(24, 2), (12, 1), (6, 1)])
@pytest.mark.parametrize("tokens", [1, 7, 128, 512, 2048])
@pytest.mark.parametrize("zero_centered", [True, False])
def test_qk_head_norm_mrope_matches_canonical(q_heads, kv_heads, tokens, zero_centered):
    """Fused Q/K RMSNorm + 3-axis interleaved MRoPE matches canonical PyTorch within BF16 rounding."""
    _requires_cuda()
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextRMSNorm,
        apply_rotary_pos_emb,
    )
    from hyperflare_recipes.components.head_norm_rope import qk_head_norm_mrope
    from hyperflare_recipes.recipes.qwen38.attention import RotaryTables

    device = torch.device("cuda:0")
    cfg = SimpleNamespace(
        head_dim=256,
        rope_parameters={
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "rope_theta": 10000000,
        },
    )
    rotary = RotaryTables.allocate(cfg, 8192, device)
    gen = torch.Generator(device="cpu").manual_seed(1000 + tokens * 17 + q_heads + int(zero_centered))
    if zero_centered:
        q_norm = Qwen4ExpTextRMSNorm(256, eps=1e-6).to(device=device, dtype=torch.bfloat16).eval()
        k_norm = Qwen4ExpTextRMSNorm(256, eps=1e-6).to(device=device, dtype=torch.bfloat16).eval()
        q_norm.weight.data.copy_((torch.randn((256,), generator=gen) * 0.25).bfloat16().to(device))
        k_norm.weight.data.copy_((torch.randn((256,), generator=gen) * 0.25).bfloat16().to(device))
    else:
        q_norm = torch.nn.RMSNorm(256, eps=1e-6, device=device, dtype=torch.bfloat16).eval()
        k_norm = torch.nn.RMSNorm(256, eps=1e-6, device=device, dtype=torch.bfloat16).eval()
        q_norm.weight.data.copy_((1.0 + torch.randn((256,), generator=gen) * 0.25).bfloat16().to(device))
        k_norm.weight.data.copy_((1.0 + torch.randn((256,), generator=gen) * 0.25).bfloat16().to(device))

    q_width = q_heads * 512
    kv_width = kv_heads * 256
    qkv = (torch.randn((tokens, q_width + 2 * kv_width), generator=gen) * 1.5).bfloat16().to(device)
    positions = torch.randint(0, 8192, (tokens, 3), generator=gen, dtype=torch.int64).to(device)
    if tokens >= 4:
        positions[-2:] = -1

    cos, sin = rotary.gather(positions)
    projected_q = qkv[:, :q_width].view(tokens, q_heads, 512)
    q_ref, k_ref = apply_rotary_pos_emb(
        q_norm(projected_q[..., :256]),
        k_norm(qkv[:, q_width:q_width + kv_width].view(tokens, kv_heads, 256)),
        cos,
        sin,
        unsqueeze_dim=1,
    )
    q_ref, k_ref = q_ref.contiguous(), k_ref.contiguous()

    q_fused, k_fused = qk_head_norm_mrope(
        qkv,
        q_norm.weight,
        k_norm.weight,
        positions,
        rotary.cos,
        rotary.sin,
        q_heads=q_heads,
        kv_heads=kv_heads,
        eps=1e-6,
        zero_centered=zero_centered,
    )
    for actual, expected in ((q_fused, q_ref), (k_fused, k_ref)):
        rel_l2 = ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-6)).item()
        frac_diff = (actual != expected).float().mean().item()
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        assert rel_l2 < 1e-4, f"rel_l2={rel_l2:.6g}"
        assert frac_diff < 5e-4, f"frac_diff={frac_diff:.6%}"
    # Negative inactive positions must clamp identically to position zero.
    if tokens >= 4:
        zero_pos = positions.clone()
        zero_pos[-2:] = 0
        q_zero, k_zero = qk_head_norm_mrope(
            qkv, q_norm.weight, k_norm.weight, zero_pos, rotary.cos, rotary.sin,
            q_heads=q_heads, kv_heads=kv_heads, eps=1e-6, zero_centered=zero_centered,
        )
        assert torch.equal(q_fused[-2:], q_zero[-2:])
        assert torch.equal(k_fused[-2:], k_zero[-2:])


def test_qk_head_norm_mrope_compiled_cuda_graph_matches_canonical():
    """Fused Q/K RMSNorm + 3-axis MRoPE survives torch.compile and CUDA graph capture."""
    _requires_cuda()
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextRMSNorm,
        apply_rotary_pos_emb,
    )
    from ignition.preparation.compilation import compile_forward
    from hyperflare_recipes.components.head_norm_rope import qk_head_norm_mrope
    from hyperflare_recipes.recipes.qwen38.attention import RotaryTables

    device = torch.device("cuda:0")
    tokens, q_heads, kv_heads = 512, 12, 1
    cfg = SimpleNamespace(
        head_dim=256,
        rope_parameters={
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "rope_theta": 10000000,
        },
    )
    rotary = RotaryTables.allocate(cfg, 4096, device)
    q_norm = Qwen4ExpTextRMSNorm(256, eps=1e-6).to(device=device, dtype=torch.bfloat16).eval()
    k_norm = Qwen4ExpTextRMSNorm(256, eps=1e-6).to(device=device, dtype=torch.bfloat16).eval()
    q_norm.weight.data.normal_(std=0.2)
    k_norm.weight.data.normal_(std=0.2)
    qkv = torch.randn((tokens, q_heads * 512 + kv_heads * 512), dtype=torch.bfloat16, device=device)
    positions = torch.randint(0, 4096, (tokens, 3), dtype=torch.int64, device=device)
    positions[-5:] = -1

    def run():
        return qk_head_norm_mrope(
            qkv, q_norm.weight, k_norm.weight, positions, rotary.cos, rotary.sin,
            q_heads=q_heads, kv_heads=kv_heads, eps=1e-6,
        )

    compiled = compile_forward(run)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        compiled()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        gq, gk = compiled()
    stream.synchronize()

    # Mutate inputs in place and replay the captured graph.
    qkv.normal_(std=1.2)
    positions.random_(0, 4096)
    positions[-3:] = -1
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    graph.reset()

    q_eager, k_eager = run()
    assert torch.equal(gq, q_eager)
    assert torch.equal(gk, k_eager)
    cos, sin = rotary.gather(positions)
    pq = qkv[:, :q_heads * 512].view(tokens, q_heads, 512)
    q_ref, k_ref = apply_rotary_pos_emb(
        q_norm(pq[..., :256]),
        k_norm(qkv[:, q_heads * 512:q_heads * 512 + kv_heads * 256].view(tokens, kv_heads, 256)),
        cos, sin, unsqueeze_dim=1,
    )
    torch.testing.assert_close(gq, q_ref.contiguous(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(gk, k_ref.contiguous(), rtol=2e-2, atol=2e-2)


def _run_decoder_forward(forward):
    forward()


def _run_decoder_forward_with_capture(decoder, out_draft):
    from tests.recipes.test_qwen38_pipelined_prefill_cuda import _CaptureMetadata

    runtime = decoder.state.begin_runtime()
    s = decoder.state
    runtime.capture_metadata = _CaptureMetadata(
        positions=runtime.qsa.query_positions,
        starts=runtime.values["positions"],
        lengths=s.inputs.query_lengths[:1],
        requests=runtime.qsa.request_ids,
        state_rows=runtime.gdn.convolution_slots,
        query_start_loc=runtime.gdn.query_start_loc,
    )
    _, _, aux, _ = decoder(runtime)
    out_draft.copy_(aux.draft_features)


def _tp2_terminal_prefill_worker(rank: int, world: int, port: int, tokens: int, cache_dir: str, error_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache_dir, f"inductor_{rank}")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    group = dist.group.WORLD

    from hyperflare_recipes.components.parallel import TensorParallelShard
    from hyperflare_recipes.recipes.qwen38.attention import QSAState, RotaryTables
    from hyperflare_recipes.recipes.qwen38.decoder import DecoderScratch, PreparedDecoder, PreparedDecoderForward
    from hyperflare_recipes.recipes.qwen38.decoder_state import DecoderInputs, PreparedDecoderState
    from hyperflare_recipes.recipes.qwen38.sampling import DeviceRequestState
    from hyperflare_recipes.recipes.qwen38.state import HybridStatePool
    from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
    from ignition.components.readiness import ReadinessTimings
    from ignition.preparation.aot import ForwardCache
    from ignition.preparation.collectives import (
        CollectivePreparation,
        create_all_gather_runtime,
        create_all_reduce_runtime,
        create_dma_all_reduce_runtime,
        create_result_broadcast_runtime,
    )
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from tests.recipes.test_qwen38_pipelined_prefill_cuda import (
        _make_mini_model_weights,
        _make_synthetic_ple_table,
    )

    collective = dma_rt = vocab_rt = result_rt = qsa_gather_rt = table = None
    sessions = []
    try:
        coordinator = CollectivePreparation(group)
        shard = TensorParallelShard(rank, world)
        weights = _make_mini_model_weights(device, shard, num_layers=5)
        context = 16384
        num_pages = 256
        slots = 4

        result_layouts = (
            dict(
                sampled=((slots,), torch.int64),
                valid=((slots,), torch.bool),
                maximum=((slots,), torch.float32),
                mass=((slots, 1), torch.float32),
            ),
        )
        collective = create_all_reduce_runtime(
            group, device, max_tokens=tokens, hidden_size=2560
        )
        dma_rt = (
            create_dma_all_reduce_runtime(group, device, max_tokens=tokens, hidden_size=2560)
            if tokens >= 1024
            else None
        )
        vocab_rt = create_all_gather_runtime(group, device, max_rows=1, columns=248320)
        result_rt = create_result_broadcast_runtime(group, device, layouts=result_layouts)
        qsa_gather_rt = create_all_gather_runtime(
            group, device, max_rows=tokens // world, columns=1024 * world
        )

        table = _make_synthetic_ple_table(device, tokens=tokens)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        table.sequence = sequence
        table.timings = ReadinessTimings.allocate(frame, 1)
        rotary = RotaryTables.allocate(weights.tree.config.text_config, context, device)

        def build_env():
            pool = HybridStatePool.allocate(slots + 1, device, speculative_tokens=7, world_size=world)
            requests = DeviceRequestState.allocate(device, capacity=slots)
            inputs = DecoderInputs.allocate(device, context=context, tokens=tokens, capacity=slots)
            cache = {3: QSAState.allocate(pages=1, slots=slots + 1, device=device, draft_tokens=7, kv_heads=1)}
            completion = CompletionOutputs.allocate((2, slots, VECTOR_TOKENS_FIELD + 1), device=device)
            return pool, requests, inputs, cache, completion

        ref_pool, ref_req, ref_in, ref_cache, ref_comp = build_env()
        split_pool, split_req, split_in, split_cache, split_comp = build_env()
        draft_cfg = SimpleNamespace(target_layers=(0, 1, 2, 3, 4), attention=SimpleNamespace(window=256))
        ref_draft_out = torch.empty((256, 5 * 2560), dtype=torch.bfloat16, device=device)
        split_draft_out = torch.empty((256, 5 * 2560), dtype=torch.bfloat16, device=device)

        session = RecordedPreparationSession(
            device=device,
            checkpoint=f"test_qwen38_terminal_{tokens}",
            workload_schema=f"test_qwen38_terminal_{tokens}",
            numerical_policy="default",
            coordinator=coordinator,
            autotune_effort="off",
        )
        session.__enter__()
        sessions.append(session)
        arena = ScratchArena(device, overlap=True)
        session.register_scratch(arena)

        dec_collectives = dict(
            runtime=dma_rt if dma_rt is not None else collective,
            barrier_runtime=collective if dma_rt is not None else None,
            vocabulary_runtime=vocab_rt,
            result_runtime=result_rt,
            qsa_gather_runtime=qsa_gather_rt,
            qsa_shard_mode="row",
            coordinator=coordinator,
        )

        # Reject invalid sample=False configurations.
        bad_decode_state = PreparedDecoderState(ref_in, ref_req, ref_pool, tokens=1, rows=1, mode="decode", maintain_draft_state=False)
        bad_scratch = DecoderScratch.allocate(1, device, sample_rows=1, world_size=world)
        with pytest.raises(ValueError, match="non-sampling"):
            PreparedDecoder(
                weights, bad_scratch, bad_decode_state, ref_cache, rotary, table, arena,
                ref_comp, frame, sequence, session=session, name="bad_decode",
                max_cache_pages=num_pages, sample=False, **dec_collectives,
            )

        ref_scratch = DecoderScratch.allocate(tokens, device, sample_rows=1, world_size=world)
        ref_state = PreparedDecoderState(ref_in, ref_req, ref_pool, tokens=tokens, rows=1, mode="prefill", maintain_draft_state=True)
        ref_dec = PreparedDecoder(
            weights, ref_scratch, ref_state, ref_cache, rotary, table, arena,
            ref_comp, frame, sequence, session=session, name=f"ref_term_{tokens}",
            max_cache_pages=num_pages, sample=True, **dec_collectives,
        )
        ref_dec.capture_context(
            SimpleNamespace(
                positions=ref_state.qsa.query_positions, starts=ref_state.token_state.positions,
                lengths=ref_state.inputs.query_lengths[:1], requests=ref_state.qsa.request_ids,
                state_rows=ref_state.gdn.convolution_slots, query_start_loc=ref_state.gdn.query_start_loc,
            ),
            draft_cfg,
            compact_prefill=True,
        )
        ref_dec.functional(session=session)
        ref_dec.release_outputs()

        split_scratch = DecoderScratch.allocate(tokens, device, sample_rows=1, world_size=world)
        nonterm_state = PreparedDecoderState(split_in, split_req, split_pool, tokens=tokens, rows=1, mode="prefill", maintain_draft_state=True)
        nonterm_dec = PreparedDecoder(
            weights, split_scratch, nonterm_state, split_cache, rotary, table, arena,
            split_comp, frame, sequence, session=session, name=f"nonterm_{tokens}",
            max_cache_pages=num_pages, sample=False, **dec_collectives,
        )
        assert nonterm_dec.final_mixer is None
        assert nonterm_dec.lm_head is None
        assert nonterm_dec.sampler is None
        assert nonterm_dec.completion_recorder is not None
        assert nonterm_dec.pipelined_prefill is (tokens >= 4096)
        nonterm_dec.capture_context(
            SimpleNamespace(
                positions=nonterm_state.qsa.query_positions, starts=nonterm_state.token_state.positions,
                lengths=nonterm_state.inputs.query_lengths[:1], requests=nonterm_state.qsa.request_ids,
                state_rows=nonterm_state.gdn.convolution_slots, query_start_loc=nonterm_state.gdn.query_start_loc,
            ),
            draft_cfg,
            compact_prefill=True,
        )
        nonterm_dec.functional(session=session)
        nonterm_dec.release_outputs()

        term_state = PreparedDecoderState(split_in, split_req, split_pool, tokens=tokens, rows=1, mode="prefill", maintain_draft_state=True)
        term_dec = PreparedDecoder(
            weights, split_scratch, term_state, split_cache, rotary, table, arena,
            split_comp, frame, sequence, session=session, name=f"term_{tokens}",
            max_cache_pages=num_pages, sample=True, **dec_collectives,
        )
        assert term_dec.final_mixer is not None
        assert term_dec.lm_head is not None
        assert term_dec.sampler is not None
        assert term_dec.completion_recorder is None
        assert term_dec.pipelined_prefill is (tokens >= 4096)
        term_dec.functional(session=session)
        term_dec.release_outputs()
        term_fwd = PreparedDecoderForward(term_dec)

        session.tune(show_progress=False)
        arena.release_preparation()

        final_ref_cache = {3: ref_cache[3].resize_pages_for_preparation(num_pages)}
        final_split_cache = {3: split_cache[3].resize_pages_for_preparation(num_pages)}
        ref_dec.layers[3][1].rebind_pages_for_preparation(final_ref_cache[3])
        nonterm_dec.layers[3][1].rebind_pages_for_preparation(final_split_cache[3])
        term_dec.layers[3][1].rebind_pages_for_preparation(final_split_cache[3])

        fwd_cache = ForwardCache(
            namespace=f"test_qwen38_terminal_{tokens}",
            device=device,
            root=os.path.join(cache_dir, "aot"),
            compile_workers=1,
        )
        compiled = fwd_cache.prepare_many(
            {
                "ref": partial(_run_decoder_forward_with_capture, ref_dec, ref_draft_out),
                "nonterm": partial(_run_decoder_forward_with_capture, nonterm_dec, split_draft_out),
                "term": partial(_run_decoder_forward, term_fwd),
            },
            {"ref": ("ref", tokens), "nonterm": ("nonterm", tokens), "term": ("term", tokens)},
            coordinator=coordinator,
        )

        stream = torch.cuda.Stream(device=device)
        graphs = {}
        for key in ("ref", "nonterm", "term"):
            dist.barrier(group)
            with torch.cuda.stream(stream):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=stream):
                    compiled[key]()
                stream.synchronize()
            graphs[key] = g

        def init_env(req, inp, pool, qsa_st, comp):
            pool.backing.zero_()
            qsa_st.main_k_cache.zero_()
            qsa_st.main_v_cache.zero_()
            qsa_st.compressed_k_cache.zero_()
            qsa_st.raw_k_ring.zero_()
            qsa_st.raw_logical_positions.fill_(-1)
            qsa_st.raw_rope_positions.fill_(-1)
            qsa_st.raw_interval_start_positions.fill_(-1)
            comp.records.zero_()
            req.generation.zero_()
            req.status.zero_()
            req.error_code.zero_()
            req.consumed_tokens.zero_()
            req.generated_tokens.zero_()
            req.pending_token.fill_(-1)
            req.ple_history.fill_(248044)
            req.rng_counter.zero_()
            req.seed.fill_(12345)
            req.parameters.zero_()
            req.parameters[0].copy_(torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0], device=device))
            req.top_k.fill_(1)
            req.max_new_tokens.fill_(16)
            req.eos_count.zero_()
            req.histogram.zero_()
            req.grammar_enabled.zero_()
            req.status[0] = 1
            req.generation[0] = 7

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
            inp.token_ids.fill_(248044)
            inp.prefill_positions.zero_()
            inp.page_tables.fill_(-1)
            if live_len > 0:
                inp.token_ids[:live_len].copy_(token_values[:live_len])
                base_pos = torch.arange(consumed, consumed + live_len, dtype=torch.int64, device=device)
                # Use distinct 3-axis MRoPE positions with offset > 0 to exercise rope_delta update.
                inp.prefill_positions[:live_len, 0].copy_(base_pos + rope_offset)
                inp.prefill_positions[:live_len, 1].copy_(base_pos + rope_offset + 5)
                inp.prefill_positions[:live_len, 2].copy_(base_pos + rope_offset + 9)
                pages = (consumed + live_len + 63) // 64
                inp.page_tables[0, :pages] = torch.arange(pages, dtype=torch.int32, device=device)

        try:
            gen = torch.Generator(device="cpu").manual_seed(9001 + tokens)
            tok0 = torch.randint(10, 50000, (tokens,), generator=gen, dtype=torch.int64).to(device)
            tok1 = torch.randint(10, 50000, (tokens,), generator=gen, dtype=torch.int64).to(device)

            # Measure reference repeatability across the exact same 2-chunk sequence.
            init_env(ref_req, ref_in, ref_pool, final_ref_cache[3], ref_comp)
            sequence.fill_(11)
            stage_chunk(ref_in, consumed=0, live_len=tokens, sample_enabled=False, token_values=tok0, rope_offset=13)
            dist.barrier(group)
            with torch.cuda.stream(stream):
                graphs["ref"].replay()
            stream.synchronize()
            ref_chunk0_residual_repeat = ref_pool.region("target_residual")[1].clone()
            ref_chunk0_draft_repeat = ref_draft_out.clone()
            chunk1_len = tokens // 2 + 17
            sequence.fill_(12)
            stage_chunk(ref_in, consumed=tokens, live_len=chunk1_len, sample_enabled=True, token_values=tok1, rope_offset=20)
            dist.barrier(group)
            with torch.cuda.stream(stream):
                graphs["ref"].replay()
            stream.synchronize()
            ref_hidden_repeat_sample = ref_pool.last_hidden[1].clone()

            # Reset and run ref vs split (nonterm chunk 0 + term chunk 1) side by side.
            init_env(ref_req, ref_in, ref_pool, final_ref_cache[3], ref_comp)
            init_env(split_req, split_in, split_pool, final_split_cache[3], split_comp)
            # Poison last_hidden on split_pool to verify non-terminal chunk leaves it untouched.
            split_pool.last_hidden[1].fill_(3.25)
            poisoned_hidden = split_pool.last_hidden[1].clone()

            # Chunk 0: non-terminal (sample_enabled=0) with MRoPE offset = 13.
            sequence.fill_(11)
            stage_chunk(ref_in, consumed=0, live_len=tokens, sample_enabled=False, token_values=tok0, rope_offset=13)
            stage_chunk(split_in, consumed=0, live_len=tokens, sample_enabled=False, token_values=tok0, rope_offset=13)
            dist.barrier(group)
            with torch.cuda.stream(stream):
                graphs["ref"].replay()
                graphs["nonterm"].replay()
            stream.synchronize()

            assert torch.equal(split_comp.records, ref_comp.records), "chunk 0 completion records diverged"
            assert torch.equal(split_pool.rope_delta, ref_pool.rope_delta), "chunk 0 rope_delta diverged"
            assert int(split_pool.rope_delta[1, 0].item()) == 22, (
                f"expected rope_delta=22 (offset 13 + axis2 +9), got {split_pool.rope_delta[1, 0].item()}"
            )
            assert torch.equal(split_req.consumed_tokens, ref_req.consumed_tokens)
            assert torch.equal(split_req.ple_history, ref_req.ple_history)
            assert torch.equal(split_req.pending_token, ref_req.pending_token)
            assert int(split_req.pending_token[0].item()) == -1
            assert torch.equal(split_req.rng_counter, ref_req.rng_counter)
            assert torch.equal(split_pool.last_hidden[1], poisoned_hidden), "non-terminal prefill mutated last_hidden"
            # Verify MTP target_residual/target_residual_position and DFlash2 context capture on sample=False.
            assert torch.equal(
                split_pool.region("target_residual_position")[1],
                ref_pool.region("target_residual_position")[1],
            ), "chunk 0 target_residual_position diverged"
            res0_repeat = (
                (ref_chunk0_residual_repeat.float() - ref_pool.region("target_residual")[1].float()).norm()
                / ref_pool.region("target_residual")[1].float().norm().clamp_min(1e-6)
            ).item()
            res0_rel = (
                (split_pool.region("target_residual")[1].float() - ref_pool.region("target_residual")[1].float()).norm()
                / ref_pool.region("target_residual")[1].float().norm().clamp_min(1e-6)
            ).item()
            assert res0_rel <= max(2e-3, 2.5 * res0_repeat) and res0_rel < 0.15, (
                f"chunk 0 target_residual diverged: rel_l2={res0_rel:.6f} (repeat={res0_repeat:.6f})"
            )
            draft0_repeat = (
                (ref_chunk0_draft_repeat.float() - ref_draft_out.float()).norm()
                / ref_draft_out.float().norm().clamp_min(1e-6)
            ).item()
            draft0_rel = (
                (split_draft_out.float() - ref_draft_out.float()).norm()
                / ref_draft_out.float().norm().clamp_min(1e-6)
            ).item()
            assert draft0_rel <= max(2e-3, 2.5 * draft0_repeat) and draft0_rel < 0.15, (
                f"chunk 0 DFlash2 draft_features diverged: rel_l2={draft0_rel:.6f} (repeat={draft0_repeat:.6f})"
            )
            assert torch.equal(final_split_cache[3].raw_logical_positions, final_ref_cache[3].raw_logical_positions)
            assert torch.equal(final_split_cache[3].raw_rope_positions, final_ref_cache[3].raw_rope_positions)
            assert torch.equal(
                final_split_cache[3].raw_interval_start_positions,
                final_ref_cache[3].raw_interval_start_positions,
            )

            # Inactive replay on non-terminal graph must preserve all persistent state bit-for-bit.
            snap_backing = split_pool.backing.clone()
            snap_mk = final_split_cache[3].main_k_cache.clone()
            snap_mv = final_split_cache[3].main_v_cache.clone()
            snap_ck = final_split_cache[3].compressed_k_cache.clone()
            snap_rk = final_split_cache[3].raw_k_ring.clone()
            stage_chunk(split_in, consumed=tokens, live_len=0, sample_enabled=False, token_values=tok1, rope_offset=0)
            dist.barrier(group)
            with torch.cuda.stream(stream):
                graphs["nonterm"].replay()
            stream.synchronize()
            assert torch.equal(split_pool.backing, snap_backing), "inactive non-terminal replay mutated pool"
            assert torch.equal(final_split_cache[3].main_k_cache, snap_mk)
            assert torch.equal(final_split_cache[3].main_v_cache, snap_mv)
            assert torch.equal(final_split_cache[3].compressed_k_cache, snap_ck)
            assert torch.equal(final_split_cache[3].raw_k_ring, snap_rk)

            # Chunk 1: terminal chunk (sample_enabled=1) — samples token and saves last_hidden.
            sequence.fill_(12)
            stage_chunk(ref_in, consumed=tokens, live_len=chunk1_len, sample_enabled=True, token_values=tok1, rope_offset=20)
            stage_chunk(split_in, consumed=tokens, live_len=chunk1_len, sample_enabled=True, token_values=tok1, rope_offset=20)
            dist.barrier(group)
            with torch.cuda.stream(stream):
                graphs["ref"].replay()
                graphs["term"].replay()
            stream.synchronize()

            assert torch.equal(split_pool.rope_delta, ref_pool.rope_delta), "chunk 1 rope_delta diverged"
            assert int(split_pool.rope_delta[1, 0].item()) == 29
            assert torch.equal(split_req.consumed_tokens, ref_req.consumed_tokens)
            assert torch.equal(split_req.ple_history, ref_req.ple_history)
            assert int(split_req.pending_token[0].item()) >= 0, "terminal prefill did not sample a token"
            assert int(split_req.generated_tokens[0].item()) == 1
            assert torch.isfinite(split_pool.last_hidden[1]).all()
            hidden_repeat = (
                (ref_hidden_repeat_sample.float() - ref_pool.last_hidden[1].float()).norm()
                / ref_pool.last_hidden[1].float().norm().clamp_min(1e-6)
            ).item()
            hidden_rel = (
                (split_pool.last_hidden[1].float() - ref_pool.last_hidden[1].float()).norm()
                / ref_pool.last_hidden[1].float().norm().clamp_min(1e-6)
            ).item()
            assert hidden_rel <= max(2e-3, 2.5 * hidden_repeat) and hidden_rel < 0.15, (
                f"chunk 1 last_hidden diverged: rel_l2={hidden_rel:.6f} (repeat={hidden_repeat:.6f})"
            )
            # Verify identical sampled token across TP ranks.
            local_tok = split_req.pending_token[0:1].cpu()
            gathered_tok = [torch.empty_like(local_tok) for _ in range(world)]
            dist.all_gather(gathered_tok, local_tok, group=group)
            assert torch.equal(gathered_tok[0], gathered_tok[1]), "TP ranks sampled different tokens"
        finally:
            stream.synchronize()
            for g in graphs.values():
                g.reset()
    except BaseException as exc:  # noqa: BLE001
        error_queue.put((rank, repr(exc)))
        raise
    finally:
        for active in reversed(sessions):
            active.__exit__(None, None, None)
        if table is not None:
            table.close()
        for rt in (qsa_gather_rt, result_rt, vocab_rt, dma_rt, collective):
            if rt is not None:
                rt.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("tokens", [512, 4096])
def test_qwen38_nonterminal_and_terminal_prefill_tp2(tokens):
    """Non-terminal prefill (sample=False) + terminal prefill (sample=True) match full prefill on TP2."""
    _requires_tp2()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = mp.get_context("spawn")
        errors = ctx.Queue()
        mp.spawn(
            _tp2_terminal_prefill_worker,
            args=(2, port, tokens, tmpdir, errors),
            nprocs=2,
            join=True,
        )
        assert errors.empty(), errors.get()
