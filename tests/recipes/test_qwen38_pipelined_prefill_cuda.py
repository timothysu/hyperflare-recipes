# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP2 CUDA parity tests for Qwen3.8 2-microbatch DMA prefill pipelining."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import os
import socket
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hyperflare_recipes.components.recurrent import DeltaMetadata
from hyperflare_recipes.recipes.qwen38.attention import QSAMetadata
from hyperflare_recipes.recipes.qwen38.decoder import _split_prefill_metadata


@dataclass(frozen=True)
class _CaptureMetadata:
    positions: torch.Tensor
    starts: torch.Tensor
    lengths: torch.Tensor
    requests: torch.Tensor
    state_rows: torch.Tensor
    query_start_loc: torch.Tensor


def _run_decoder_with_capture(dec):
    runtime = dec.state.begin_runtime()
    s = dec.state
    runtime.capture_metadata = _CaptureMetadata(
        positions=runtime.qsa.query_positions,
        starts=runtime.values["positions"],
        lengths=s.inputs.query_lengths[:1],
        requests=runtime.qsa.request_ids,
        state_rows=runtime.gdn.convolution_slots,
        query_start_loc=runtime.gdn.query_start_loc,
    )
    sample, logits, aux, _ = dec(runtime)
    return sample, logits, aux.residual, aux.draft_features


def _requires_tp2():
    if os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1":
        pytest.skip("requires FLARE_RUN_CUDA_TESTS=1 and FLARE_RUN_TP_CUDA_TESTS=1")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least two CUDA devices")


def _make_mxfp8_weight(out_f: int, in_f: int, device: torch.device, gen: torch.Generator, scale: float = 0.02):
    from b12x.gemm.blockscaled import pack_weight

    w = torch.randn((out_f, in_f), generator=gen) * scale
    return pack_weight(
        w.to(torch.float8_e4m3fn).to(device),
        torch.full((out_f, in_f // 32), 127, dtype=torch.uint8, device=device),
        recipe="mxfp8",
    )


def _make_nvfp4_moe_weights(device: torch.device, world: int, seed: int, *, rank: int = 0):
    from b12x.moe import fused_moe as moe
    from b12x._lib.intrinsics.reference import swizzle_block_scale
    from hyperflare_recipes.recipes.qwen38.moe import QwenMoeWeights, SharedExpertWeights

    gen_replicated = torch.Generator(device="cpu").manual_seed(seed)
    gen_shard = torch.Generator(device="cpu").manual_seed(seed + rank * 10000)
    local = 640 // world
    num_experts, hidden = 512, 2560
    plan = moe.plan_weights(
        source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=moe.ActivationSpec(mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16),
        geometry=moe.MoEGeometry(num_experts=num_experts, hidden_size=hidden, intermediate_size=local),
    )
    # Random valid packed FP4 nibbles (avoid high magnitudes so BF16 outputs stay well-conditioned).
    w13 = torch.randint(0, 256, (num_experts, 2 * local, hidden // 2), generator=gen_shard, dtype=torch.uint8).to(device)
    w2 = torch.randint(0, 256, (num_experts, hidden, local // 2), generator=gen_shard, dtype=torch.uint8).to(device)
    s13 = torch.full((num_experts, 2 * local, hidden // 16), 0.25, dtype=torch.float8_e4m3fn, device=device)
    s2 = torch.full((num_experts, hidden, local // 16), 0.25, dtype=torch.float8_e4m3fn, device=device)
    g13 = torch.full((num_experts,), 0.01, dtype=torch.float32, device=device)
    g2 = torch.full((num_experts,), 0.01, dtype=torch.float32, device=device)
    a1 = torch.full((num_experts,), 1.0, dtype=torch.float32, device=device)
    a2 = torch.full((num_experts,), 1.0, dtype=torch.float32, device=device)
    packed = moe.PackedWeights(
        w13=w13,
        w2=w2,
        w13_block_scales=swizzle_block_scale(s13),
        w2_block_scales=swizzle_block_scale(s2),
        w13_global_scales=g13,
        w2_global_scales=g2,
        input_scale=a1,
        intermediate_scale=a2,
        immutable_input_scales=True,
    )
    experts = moe.prepare_weights(plan=plan, weights=packed)
    router = (torch.randn((num_experts, hidden), generator=gen_replicated) * 0.05).bfloat16().to(device)
    shared = SharedExpertWeights(
        gate_up=_make_mxfp8_weight(2 * local, hidden, device, gen_shard),
        down=_make_mxfp8_weight(hidden, local, device, gen_shard),
        gate=(torch.randn((1, hidden), generator=gen_replicated) * 0.05).bfloat16().to(device),
    )
    return QwenMoeWeights(router=router, experts=experts, shared=shared)


def _make_synthetic_ple_table(device: torch.device, tokens: int):
    from b12x.sequence import ple_embedding
    from b12x.sequence._shared.disk_table import MappedHostAllocation
    from hyperflare_recipes.recipes.qwen38.ple import PLETable, _caps
    from ignition.preparation.persistent import persistent_empty

    config = {
        "eos_token_id": 248044,
        "vocab_size": 248320,
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "ngram_vocab_size_base": 20000000,
        "ple_embed_dim": 2560,
        "make_ngram_vocab_size_divisible_by": 128,
    }
    caps = _caps(config, device, 1, 1, rank=0, world_size=1)
    prime_list = []
    cand = 19999999
    while len(prime_list) < 16:
        if all(cand % d != 0 for d in range(2, int(cand**0.5) + 1)):
            prime_list.append(cand)
        cand += 2
    primes = torch.tensor(prime_list, dtype=torch.int64)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), primes[:-1].cumsum(0)))
    multipliers = torch.tensor([1000003, 1000033, 1000037], dtype=torch.int64)
    geometry = ple_embedding.compute_geometry(
        caps, prime_sizes=primes, table_offsets=offsets, multipliers=multipliers
    )
    layout = ple_embedding.storage_layout(caps, geometry=geometry)
    heads = layout.caps.head_count
    mailbox = MappedHostAllocation((tokens * heads + 1,), torch.int64, device)
    mailbox.host_view.zero_()
    ready = torch.cuda.Event(external=True)
    ready.record(torch.cuda.current_stream(device))
    scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    return PLETable(
        config=config,
        layout=layout,
        geometry=ple_embedding.allocate_geometry(layout.geometry, device=device, allocate=persistent_empty),
        scale=persistent_empty(scale.shape, scale.dtype, device).copy_(scale),
        weights=torch.zeros((tokens * heads, 80), dtype=torch.uint8, device=device),
        scales=torch.zeros((tokens * heads, 10), dtype=torch.uint8, device=device),
        hashes=torch.full((tokens, heads), -1, dtype=torch.int64, device=device),
        mailbox=mailbox,
        ready=ready,
        sources=((), ()),
        global_scale=1.0,
    )


def _make_mini_model_weights(device: torch.device, shard, num_layers: int = 5):
    """Build exact-geometry Qwen3.8 ModelWeights (layers 0..num_layers-1) directly on CUDA."""
    from transformers import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration
    from hyperflare_recipes.components.model import canonical_tree
    from hyperflare_recipes.recipes.qwen38.decoder import DecoderLayerWeights, ModelWeights
    from hyperflare_recipes.recipes.qwen38.hyperconnection import packed_down_projection

    layer_types = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(num_layers)]
    text_cfg = {
        "model_type": "qwen4_exp_text",
        "vocab_size": 248320,
        "hidden_size": 2560,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 4,
        "layer_types": layer_types,
        "hc_count": 4,
        "hc_lowrank": 320,
        "rms_norm_eps": 1e-6,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "output_gate_type": "sigmoid",
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "indexer_head_dim": 128,
        "indexer_kv_heads": 1,
        "indexer_n_heads": 4,
        "moe_intermediate_size": 640,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "ple_conv_kernel_size": 4,
        "ple_embed_dim": 2560,
        "ple_layer_ids": [2],
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "ngram_vocab_size_base": 20000000,
        "make_ngram_vocab_size_divisible_by": 128,
        "eos_token_id": 248044,
        "bos_token_id": 248044,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "mtp_num_hidden_layers": 0,
    }
    config = Qwen4ExpConfig(text_config=text_cfg)
    tree = canonical_tree(Qwen4ExpForConditionalGeneration, config)
    language = tree.model.language_model
    world = shard.world_size

    def init_hc(hc_mod, seed_val: int):
        g = torch.Generator(device="cpu").manual_seed(seed_val)
        hc_mod.hc_norm.weight = torch.nn.Parameter(
            (torch.ones((10240,)) + torch.randn((10240,), generator=g) * 0.05).float().to(device), requires_grad=False
        )
        hc_mod.input_mix_weight_down.weight = torch.nn.Parameter(
            (torch.randn((320, 10240), generator=g) * 0.01).bfloat16().to(device), requires_grad=False
        )
        hc_mod.input_mix_weight_up.weight = torch.nn.Parameter(
            (torch.randn((10240, 320), generator=g) * 0.01).bfloat16().to(device), requires_grad=False
        )
        if hc_mod.block_inject_weight is not None:
            hc_mod.block_inject_weight.weight = torch.nn.Parameter(
                (torch.randn((4, 10240), generator=g) * 0.01).bfloat16().to(device), requires_grad=False
            )
        packed_down_projection(hc_mod)

    g_top = torch.Generator(device="cpu").manual_seed(11)
    language.embed_tokens.weight = torch.nn.Parameter(
        (torch.randn((248320, 2560), generator=g_top) * 0.05).bfloat16().to(device), requires_grad=False
    )
    local_vocab = 248320 // world
    tree.lm_head.weight = torch.nn.Parameter(
        (torch.randn((local_vocab, 2560), generator=torch.Generator(device="cpu").manual_seed(12 + shard.rank)) * 0.02)
        .bfloat16()
        .to(device),
        requires_grad=False,
    )
    init_hc(language.hyper_connection_mixer, 19)

    layers = []
    for idx in range(num_layers):
        layer_mod = language.layers[idx]
        init_hc(layer_mod.attn_hyper_connection, 100 + idx * 10)
        init_hc(layer_mod.mlp_hyper_connection, 105 + idx * 10)
        g_layer = torch.Generator(device="cpu").manual_seed(200 + idx * 17 + shard.rank * 1000)
        if idx % 4 == 3:
            sa = layer_mod.self_attn
            sa.q_norm.weight = torch.nn.Parameter(torch.zeros((256,), dtype=torch.bfloat16, device=device), requires_grad=False)
            sa.k_norm.weight = torch.nn.Parameter(torch.zeros((256,), dtype=torch.bfloat16, device=device), requires_grad=False)
            sa.indexer.q_layernorm.weight = torch.nn.Parameter(torch.zeros((128,), dtype=torch.bfloat16, device=device), requires_grad=False)
            sa.indexer.k_layernorm.weight = torch.nn.Parameter(torch.zeros((128,), dtype=torch.bfloat16, device=device), requires_grad=False)
            q_heads = 24 // world
            kv_heads = max(1, 2 // world)
            g_idx = torch.Generator(device="cpu").manual_seed(300 + idx)  # replicated index projection across ranks
            attention = {
                "qkv_proj": _make_mxfp8_weight(q_heads * 512 + 2 * kv_heads * 256, 2560, device, g_layer),
                "indexer.index_qk_proj": _make_mxfp8_weight(640, 2560, device, g_idx),
                "o_proj": _make_mxfp8_weight(2560, q_heads * 256, device, g_layer),
            }
        else:
            la = layer_mod.linear_attn
            la.num_k_heads //= world
            la.num_v_heads //= world
            la.conv_dim //= world
            la.A_log = torch.nn.Parameter(
                (torch.randn((48 // world,), generator=g_layer) * 0.1 - 1.0).float().to(device), requires_grad=False
            )
            la.dt_bias = torch.nn.Parameter(
                (torch.randn((48 // world,), generator=g_layer) * 0.1).bfloat16().to(device), requires_grad=False
            )
            la.norm.weight = torch.nn.Parameter(
                torch.ones((128,), dtype=torch.float32, device=device), requires_grad=False
            )
            la.conv1d.weight = torch.nn.Parameter(
                (torch.randn((10240 // world, 1, 4), generator=g_layer) * 0.1).bfloat16().to(device), requires_grad=False
            )
            attention = {
                "in_proj_qkvz": _make_mxfp8_weight(16384 // world, 2560, device, g_layer),
                "in_proj_ba": (torch.randn((96 // world, 2560), generator=g_layer) * 0.02).bfloat16().to(device),
                "out_proj": _make_mxfp8_weight(2560, 6144 // world, device, g_layer),
            }
        moe_weights = _make_nvfp4_moe_weights(device, world, seed=500 + idx * 31, rank=shard.rank)
        layer_mod.mlp.gate.weight = torch.nn.Parameter(moe_weights.router, requires_grad=False)
        layer_mod.mlp.shared_expert_gate.weight = torch.nn.Parameter(moe_weights.shared.gate, requires_grad=False)
        layers.append(DecoderLayerWeights(attention, moe_weights))

    ple = language.layers[1].ple
    g_ple = torch.Generator(device="cpu").manual_seed(900)
    ple.key_proj.weight = torch.nn.Parameter((torch.randn((10240, 2560), generator=g_ple) * 0.01).bfloat16().to(device), requires_grad=False)
    ple.value_proj.weight = torch.nn.Parameter((torch.randn((2560, 2560), generator=g_ple) * 0.01).bfloat16().to(device), requires_grad=False)
    ple.conv1d.weight = torch.nn.Parameter((torch.randn((10240, 1, 4), generator=g_ple) * 0.05).bfloat16().to(device), requires_grad=False)
    for norm_name in ("norm_key", "norm_query", "norm_conv"):
        getattr(ple, norm_name).weight = torch.nn.Parameter(torch.zeros((10240,), dtype=torch.float32, device=device), requires_grad=False)

    torch.cuda.synchronize(device)
    return ModelWeights(tree, tuple(layers), None, 0, shard)


@pytest.mark.parametrize(
    "tokens,start,live",
    [
        (1024, 0, 0),
        (1024, 0, 1),
        (1024, 0, 511),
        (1024, 0, 512),
        (1024, 0, 513),
        (1024, 0, 515),
        (1024, 0, 1024),
        (2048, 3, 2048),
        (2048, 515, 1025),
        (2048, 515, 700),
    ],
)
def test_split_prefill_metadata_handles_boundaries_and_inactive_halves(tokens, start, live):
    """Half-chunk GDN and QSA metadata preserve exact state handoff across all boundary lengths."""
    half = tokens // 2
    gdn = DeltaMetadata.allocate(1, torch.device("cpu"))
    qsa = QSAMetadata.allocate(tokens=tokens, batch=1, context=8192, device=torch.device("cpu"))
    has_live = live > 0
    gdn.query_start_loc[1] = live
    gdn.convolution_slots[0] = 3 if has_live else -1
    gdn.fresh[0] = has_live and start == 0
    gdn.accepted_tokens[0] = 1
    gdn.initial_state_indices[0] = 3 if (has_live and start > 0) else 0
    gdn.final_state_indices[0] = 3 if has_live else 0
    gdn.num_seqs[0] = 1 if has_live else 0
    gdn.num_tokens[0] = live

    qsa.raw_state_slot_ids[0] = 1 if has_live else -1
    qsa.sequence_lengths[0] = start + live if has_live else 0
    qsa.query_start_loc[1] = live
    qsa.num_accepted_tokens[0] = 1
    qsa.is_prefilling[0] = has_live
    if has_live:
        positions = torch.arange(start, start + live, dtype=torch.int64)
        qsa.request_ids[:live] = 0
        qsa.query_positions[:live] = positions
        qsa.rope_positions[:live] = positions[:, None].expand(-1, 3)
        qsa.slot_mapping[:live] = positions

    gdn_a, gdn_b, qsa_a, qsa_b = _split_prefill_metadata(gdn, qsa, half)
    exp_a = min(live, half)
    exp_b = max(live - half, 0)

    assert gdn_a.query_start_loc.tolist() == [0, exp_a]
    assert gdn_b.query_start_loc.tolist() == [0, exp_b]
    assert gdn_a.num_tokens.item() == exp_a
    assert gdn_b.num_tokens.item() == exp_b
    assert gdn_a.num_seqs.item() == int(exp_a > 0)
    assert gdn_b.num_seqs.item() == int(exp_b > 0)
    assert gdn_a.convolution_slots.item() == (3 if exp_a > 0 else -1)
    assert gdn_b.convolution_slots.item() == (3 if exp_b > 0 else -1)
    assert gdn_a.fresh.item() is (has_live and start == 0)
    assert gdn_b.fresh.item() is False
    assert gdn_b.accepted_tokens.item() == 1
    assert gdn_a.initial_state_indices.item() == (3 if (exp_a > 0 and start > 0) else 0)
    assert gdn_a.final_state_indices.item() == (3 if exp_a > 0 else 0)
    assert gdn_b.initial_state_indices.item() == (3 if exp_b > 0 else 0)
    assert gdn_b.final_state_indices.item() == (3 if exp_b > 0 else 0)

    assert qsa_a.sequence_lengths.item() == (start + exp_a if exp_a > 0 else 0)
    assert qsa_b.sequence_lengths.item() == (start + live if exp_b > 0 else 0)
    assert qsa_a.raw_state_slot_ids.item() == (1 if exp_a > 0 else -1)
    assert qsa_b.raw_state_slot_ids.item() == (1 if exp_b > 0 else -1)
    assert qsa_a.is_prefilling.item() is (exp_a > 0)
    assert qsa_b.is_prefilling.item() is (exp_b > 0)
    assert qsa_a.request_ids.shape == (half,) and qsa_b.request_ids.shape == (half,)
    assert torch.equal(qsa_a.query_positions, qsa.query_positions[:half])
    assert torch.equal(qsa_b.query_positions, qsa.query_positions[half:])


def _tp2_pipelined_prefill_worker(rank: int, world: int, port: int, tokens: int, cache_dir: str, error_queue):
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
    from hyperflare_recipes.recipes.qwen38.decoder import DecoderScratch, PreparedDecoder
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
    )
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    collective = dma_rt = vocab_rt = qsa_gather_rt = table = None
    sessions = []
    try:
        coordinator = CollectivePreparation(group)
        shard = TensorParallelShard(rank, world)
        weights = _make_mini_model_weights(device, shard, num_layers=5)
        context = 16384
        num_pages = 256
        slots = 4

        collective = create_all_reduce_runtime(
            group, device, max_tokens=tokens, hidden_size=2560
        )
        dma_rt = create_dma_all_reduce_runtime(group, device, max_tokens=tokens, hidden_size=2560)
        vocab_rt = create_all_gather_runtime(group, device, max_rows=1, columns=248320)
        qsa_gather_rt = create_all_gather_runtime(
            group, device, max_rows=tokens // world, columns=1024 * world
        )

        table = _make_synthetic_ple_table(device, tokens=tokens)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        table.sequence = sequence
        table.timings = ReadinessTimings.allocate(frame, 1)

        rotary = RotaryTables.allocate(weights.tree.config.text_config, context, device)
        draft_config = SimpleNamespace(
            target_layers=(0, 1, 2, 3, 4),
            attention=SimpleNamespace(window=512),
        )

        def build_env():
            pool = HybridStatePool.allocate(slots + 1, device, speculative_tokens=0, world_size=world)
            requests = DeviceRequestState.allocate(device, capacity=slots)
            inputs = DecoderInputs.allocate(device, context=context, tokens=tokens, capacity=slots)
            cache = {3: QSAState.allocate(pages=1, slots=slots + 1, device=device, draft_tokens=0, kv_heads=1)}
            completion = CompletionOutputs.allocate((2, slots, VECTOR_TOKENS_FIELD + 1), device=device)
            state = PreparedDecoderState(
                inputs, requests, pool, tokens=tokens, rows=1, mode="prefill", maintain_draft_state=False
            )
            scratch = DecoderScratch.allocate(tokens, device, sample_rows=1, world_size=world)
            return pool, requests, inputs, cache, completion, state, scratch

        ref_pool, ref_req, ref_in, ref_cache, ref_comp, ref_state, ref_scratch = build_env()
        pipe_pool, pipe_req, pipe_in, pipe_cache, pipe_comp, pipe_state, pipe_scratch = build_env()

        ref_session = RecordedPreparationSession(
            device=device,
            checkpoint=f"test_qwen38_ref_{tokens}",
            workload_schema=f"test_qwen38_ref_{tokens}",
            numerical_policy="default",
            coordinator=coordinator,
            autotune_effort="off",
        )
        pipe_session = RecordedPreparationSession(
            device=device,
            checkpoint=f"test_qwen38_pipe_{tokens}",
            workload_schema=f"test_qwen38_pipe_{tokens}",
            numerical_policy="default",
            coordinator=coordinator,
            autotune_effort="off",
        )
        ref_session.__enter__()
        sessions.append(ref_session)
        ref_arena = ScratchArena(device, overlap=False)
        ref_session.register_scratch(ref_arena)
        ref_decoder = PreparedDecoder(
            weights, ref_scratch, ref_state, ref_cache, rotary, table, ref_arena,
            ref_comp, frame, sequence, session=ref_session, name=f"ref_{tokens}",
            max_cache_pages=num_pages, verification=True,
            runtime=dma_rt, barrier_runtime=collective, vocabulary_runtime=vocab_rt,
            qsa_gather_runtime=qsa_gather_rt, qsa_shard_mode="row", coordinator=coordinator,
        )
        assert ref_decoder.pipelined_prefill is False
        s = ref_decoder.state
        ref_decoder.capture_context(
            SimpleNamespace(
                positions=s.qsa.query_positions, starts=s.token_state.positions,
                lengths=s.inputs.query_lengths[:1], requests=s.qsa.request_ids,
                state_rows=s.gdn.convolution_slots, query_start_loc=s.gdn.query_start_loc,
            ),
            draft_config,
            compact_prefill=True,
        )
        ref_decoder.functional(session=ref_session)
        ref_decoder.release_outputs()
        ref_session.tune(show_progress=False)
        ref_arena.release_preparation()

        pipe_session.__enter__()
        sessions.append(pipe_session)
        pipe_arena = ScratchArena(device, overlap=True)
        pipe_session.register_scratch(pipe_arena)
        pipe_decoder = PreparedDecoder(
            weights, pipe_scratch, pipe_state, pipe_cache, rotary, table, pipe_arena,
            pipe_comp, frame, sequence, session=pipe_session, name=f"pipe_{tokens}",
            max_cache_pages=num_pages, verification=True,
            runtime=dma_rt, barrier_runtime=collective, vocabulary_runtime=vocab_rt,
            qsa_gather_runtime=qsa_gather_rt, qsa_shard_mode="row", coordinator=coordinator,
        )
        assert pipe_decoder.pipelined_prefill is True
        s = pipe_decoder.state
        pipe_decoder.capture_context(
            SimpleNamespace(
                positions=s.qsa.query_positions, starts=s.token_state.positions,
                lengths=s.inputs.query_lengths[:1], requests=s.qsa.request_ids,
                state_rows=s.gdn.convolution_slots, query_start_loc=s.gdn.query_start_loc,
            ),
            draft_config,
            compact_prefill=True,
        )
        pipe_decoder.functional(session=pipe_session)
        pipe_decoder.release_outputs()
        pipe_session.tune(show_progress=False)
        pipe_arena.release_preparation()

        final_ref_cache = {3: ref_cache[3].resize_pages_for_preparation(num_pages)}
        final_pipe_cache = {3: pipe_cache[3].resize_pages_for_preparation(num_pages)}
        ref_decoder.layers[3][1].rebind_pages_for_preparation(final_ref_cache[3])
        pipe_decoder.layers[3][1].rebind_pages_for_preparation(final_pipe_cache[3])

        fwd_cache = ForwardCache(
            namespace=f"test_qwen38_pipelined_{tokens}",
            device=device,
            root=os.path.join(cache_dir, "aot"),
            compile_workers=1,
        )
        compiled_pipe = fwd_cache.prepare_many(
            {"pipe": partial(_run_decoder_with_capture, pipe_decoder)},
            {"pipe": ("pipe", tokens)},
            coordinator=coordinator,
        )["pipe"]

        stream = torch.cuda.Stream(device=device)
        dist.barrier(group)
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                g_sample, g_logits, g_residual, g_draft = compiled_pipe()
            stream.synchronize()

        half = tokens // 2
        scenarios = [
            ("full_chunk", True, tokens),
            ("first_half_only", True, half - 100),
            ("cross_half_plus_1", True, half + 1),
            ("cross_half_plus_3", True, half + 3),
            ("continuation_non_4_aligned", False, half + 88),
            ("inactive_zero", False, 0),
        ]

        def setup_request(req, inp, pool, qsa_st, *, reset, consumed, live_len, token_values):
            if reset:
                pool.backing.zero_()
                qsa_st.main_k_cache.zero_()
                qsa_st.main_v_cache.zero_()
                qsa_st.compressed_k_cache.zero_()
                qsa_st.raw_k_ring.zero_()
                qsa_st.raw_logical_positions.fill_(-1)
                qsa_st.raw_rope_positions.fill_(-1)
                qsa_st.raw_interval_start_positions.fill_(-1)
                req.generation.zero_()
                req.status.zero_()
                req.error_code.zero_()
                req.consumed_tokens.zero_()
                req.pending_token.fill_(-1)
                req.ple_history.fill_(248044)
            req.status[0] = 1
            req.generation[0] = 7
            req.consumed_tokens[0] = consumed
            req.pending_token[0] = -1
            inp.request_state_rows.zero_()
            inp.request_state_rows[0, 0] = 1
            inp.row_slot_ids.fill_(-1)
            inp.row_slot_ids[0] = 0
            inp.row_generations.zero_()
            inp.row_generations[0] = 7
            inp.live_rows[0] = 1 if live_len > 0 else 0
            inp.query_lengths.zero_()
            inp.query_lengths[0] = live_len
            inp.token_ids.fill_(248044)
            inp.prefill_positions.zero_()
            inp.page_tables.fill_(-1)
            if live_len > 0:
                inp.token_ids[:live_len].copy_(token_values[:live_len])
                pos = torch.arange(consumed, consumed + live_len, dtype=torch.int64, device=device)
                inp.prefill_positions[:live_len].copy_(pos[:, None].expand(-1, 3))
                pages = (consumed + live_len + 63) // 64
                inp.page_tables[0, :pages] = torch.arange(pages, dtype=torch.int32, device=device)

        def snapshot_state(pool, qsa_st):
            return (
                pool.backing.clone(),
                qsa_st.main_k_cache.clone(),
                qsa_st.main_v_cache.clone(),
                qsa_st.compressed_k_cache.clone(),
                qsa_st.raw_k_ring.clone(),
                qsa_st.raw_logical_positions.clone(),
                qsa_st.raw_rope_positions.clone(),
                qsa_st.raw_interval_start_positions.clone(),
            )

        def restore_state(pool, qsa_st, snap):
            (
                backing, mk, mv, ck, rk, rlp, rrp, risp
            ) = snap
            pool.backing.copy_(backing)
            qsa_st.main_k_cache.copy_(mk)
            qsa_st.main_v_cache.copy_(mv)
            qsa_st.compressed_k_cache.copy_(ck)
            qsa_st.raw_k_ring.copy_(rk)
            qsa_st.raw_logical_positions.copy_(rlp)
            qsa_st.raw_rope_positions.copy_(rrp)
            qsa_st.raw_interval_start_positions.copy_(risp)

        def rel_l2(actual, expected):
            return (
                (actual.float() - expected.float()).norm()
                / expected.float().norm().clamp_min(1e-6)
            ).item()

        consumed = 0
        try:
            for idx, (label, reset, live_len) in enumerate(scenarios):
                if reset:
                    consumed = 0
                gen = torch.Generator(device="cpu").manual_seed(1234 + idx)
                tok_vals = torch.randint(10, 50000, (tokens,), generator=gen, dtype=torch.int64).to(device)
                setup_request(
                    ref_req, ref_in, ref_pool, final_ref_cache[3],
                    reset=reset, consumed=consumed, live_len=live_len, token_values=tok_vals,
                )
                setup_request(
                    pipe_req, pipe_in, pipe_pool, final_pipe_cache[3],
                    reset=reset, consumed=consumed, live_len=live_len, token_values=tok_vals,
                )
                pre_ref = snapshot_state(ref_pool, final_ref_cache[3])
                pre_pipe = snapshot_state(pipe_pool, final_pipe_cache[3])

                dist.barrier(group)
                with torch.cuda.stream(stream):
                    _, ref_logits_2, ref_res_2, ref_draft_2 = _run_decoder_with_capture(ref_decoder)
                stream.synchronize()
                ref_conv_2 = {l: ref_pool.gdn[l].convolution[1].clone() for l in (0, 1, 2, 4)}
                ref_rec_2 = {l: ref_pool.gdn[l].recurrent[1].clone() for l in (0, 1, 2, 4)}
                ref_comp_k_2 = final_ref_cache[3].compressed_k_cache.clone()

                restore_state(ref_pool, final_ref_cache[3], pre_ref)
                setup_request(
                    ref_req, ref_in, ref_pool, final_ref_cache[3],
                    reset=False, consumed=consumed, live_len=live_len, token_values=tok_vals,
                )
                dist.barrier(group)
                with torch.cuda.stream(stream):
                    ref_sample, ref_logits, ref_res, ref_draft = _run_decoder_with_capture(ref_decoder)
                    graph.replay()
                stream.synchronize()

                consumed += live_len

                local_res = g_residual.cpu()
                gathered_res = [torch.empty_like(local_res) for _ in range(world)]
                dist.all_gather(gathered_res, local_res, group=group)
                assert torch.equal(gathered_res[0], gathered_res[1]), f"{label}: TP ranks diverged"

                if live_len == 0:
                    post_pipe = snapshot_state(pipe_pool, final_pipe_cache[3])
                    for pre_t, post_t in zip(pre_pipe, post_pipe, strict=True):
                        assert torch.equal(post_t, pre_t), f"{label}: inactive replay mutated persistent state"
                    continue

                assert torch.isfinite(g_logits).all(), f"{label}: non-finite logits"
                assert torch.isfinite(g_residual[:live_len]).all(), f"{label}: non-finite residual"
                res_repeat = rel_l2(ref_res_2[:live_len], ref_res[:live_len])
                res_rel = rel_l2(g_residual[:live_len], ref_res[:live_len])
                assert res_rel <= max(2e-3, 2.5 * res_repeat) and res_rel < 0.15, (
                    f"{label}: residual rel_l2={res_rel:.6f} (repeat={res_repeat:.6f})"
                )
                logit_repeat = rel_l2(ref_logits_2, ref_logits)
                logit_rel = rel_l2(g_logits, ref_logits)
                assert logit_rel <= max(2e-3, 2.5 * logit_repeat) and logit_rel < 0.15, (
                    f"{label}: logits rel_l2={logit_rel:.6f} (repeat={logit_repeat:.6f})"
                )
                draft_repeat = rel_l2(ref_draft_2, ref_draft)
                draft_rel = rel_l2(g_draft, ref_draft)
                assert draft_rel <= max(2e-3, 2.5 * draft_repeat) and draft_rel < 0.15, (
                    f"{label}: draft features rel_l2={draft_rel:.6f} (repeat={draft_repeat:.6f})"
                )

                for layer_idx in (0, 1, 2, 4):
                    conv_ref = ref_pool.gdn[layer_idx].convolution[1]
                    conv_pipe = pipe_pool.gdn[layer_idx].convolution[1]
                    conv_repeat = rel_l2(ref_conv_2[layer_idx], conv_ref)
                    conv_rel = rel_l2(conv_pipe, conv_ref)
                    conv_limit = 5e-3 if layer_idx == 0 else min(0.15, max(5e-3, 2.5 * max(conv_repeat, res_repeat)))
                    assert conv_rel <= conv_limit, (
                        f"{label} layer {layer_idx} conv rel={conv_rel:.6f} (repeat={conv_repeat:.6f})"
                    )

                    rec_ref = ref_pool.gdn[layer_idx].recurrent[1]
                    rec_pipe = pipe_pool.gdn[layer_idx].recurrent[1]
                    rec_repeat = rel_l2(ref_rec_2[layer_idx], rec_ref)
                    rec_rel = rel_l2(rec_pipe, rec_ref)
                    rec_limit = 5e-3 if layer_idx == 0 else min(0.20, max(5e-3, 2.5 * max(rec_repeat, res_repeat)))
                    assert rec_rel <= rec_limit, (
                        f"{label} layer {layer_idx} rec rel={rec_rel:.6f} (repeat={rec_repeat:.6f})"
                    )

                q_ref, q_pipe = final_ref_cache[3], final_pipe_cache[3]
                assert torch.equal(q_pipe.raw_logical_positions, q_ref.raw_logical_positions), f"{label}: raw tags"
                assert torch.equal(q_pipe.raw_rope_positions, q_ref.raw_rope_positions), f"{label}: raw rope"
                assert torch.equal(
                    q_pipe.raw_interval_start_positions, q_ref.raw_interval_start_positions
                ), f"{label}: raw anchor"
                c_repeat = rel_l2(ref_comp_k_2, q_ref.compressed_k_cache)
                c_rel = rel_l2(q_pipe.compressed_k_cache, q_ref.compressed_k_cache)
                assert c_rel <= min(0.15, max(5e-3, 2.5 * max(c_repeat, res_repeat))), (
                    f"{label}: compressed_k_cache rel={c_rel:.6f} (repeat={c_repeat:.6f})"
                )
        finally:
            stream.synchronize()
            graph.reset()
    except BaseException as exc:  # noqa: BLE001
        error_queue.put((rank, repr(exc)))
        raise
    finally:
        for active in reversed(sessions):
            active.__exit__(None, None, None)
        if table is not None:
            table.close()
        for rt in (qsa_gather_rt, vocab_rt, dma_rt, collective):
            if rt is not None:
                rt.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("tokens", [4096])
def test_qwen38_pipelined_prefill_matches_unpipelined_tp2(tokens):
    """2-microbatch DMA pipelined prefill matches unpipelined prefill across GDN, PLE, QSA, MoE, and capture."""
    _requires_tp2()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = mp.get_context("spawn")
        errors = ctx.Queue()
        mp.spawn(
            _tp2_pipelined_prefill_worker,
            args=(2, port, tokens, tmpdir, errors),
            nprocs=2,
            join=True,
        )
        assert errors.empty(), errors.get()
