# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CUDA parity, KDA rollback, MTP alignment, and real-checkpoint tests for Ling-3.2-Air."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import triton

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device",
)


def _reference_mla(x_all, q_w, kv_a_w, g_w, kv_norm_w, kv_b_w, dense_w, *, freqs, eps=1e-6):
    seq_len, heads = x_all.shape[0], 32
    pos = torch.arange(seq_len, dtype=torch.float32, device=x_all.device)
    ang = torch.outer(pos, freqs)
    cos, sin = ang.cos(), ang.sin()

    def rope_interleave(t):
        pairs = t.float().view(*t.shape[:-1], 32, 2)
        c = cos.view(*([1] * (pairs.ndim - 3)), seq_len, 32)
        s = sin.view(*([1] * (pairs.ndim - 3)), seq_len, 32)
        return torch.stack((pairs[..., 0] * c - pairs[..., 1] * s, pairs[..., 1] * c + pairs[..., 0] * s), dim=-1).flatten(-2).to(t.dtype)

    q_ref = F.linear(x_all, q_w).view(1, seq_len, heads, 192).transpose(1, 2)
    q_nope, q_rot = q_ref.split((128, 64), dim=-1)
    kv_a_ref = F.linear(x_all, kv_a_w)
    k_pass, k_rot = kv_a_ref.split((512, 64), dim=-1)
    kp_f = k_pass.float()
    k_norm = (kp_f * torch.rsqrt(kp_f.pow(2).mean(dim=-1, keepdim=True) + eps)).to(torch.bfloat16) * kv_norm_w
    kv_ref = F.linear(k_norm, kv_b_w).view(1, seq_len, heads, 256).transpose(1, 2)
    k_nope, v_ref = kv_ref.split((128, 128), dim=-1)
    q_rot = rope_interleave(q_rot)
    k_rot = rope_interleave(k_rot.view(1, 1, seq_len, 64)).expand(1, heads, seq_len, 64)
    attn_ref = F.scaled_dot_product_attention(
        torch.cat((q_nope, q_rot), dim=-1),
        torch.cat((k_nope, k_rot), dim=-1),
        v_ref,
        is_causal=True,
        scale=192.0 ** -0.5,
    ).transpose(1, 2).contiguous()
    gate_ref = torch.sigmoid(F.linear(x_all, g_w).float()).to(torch.bfloat16).view(1, seq_len, heads, 1)
    return F.linear((attn_ref * gate_ref).reshape(seq_len, heads * 128), dense_w)


def _reference_kda(x_all, qkvfg_w, b_w, conv_w, o_norm_w, A_log, dt_bias, o_w, *, eps=1e-6, lower_bound=-5.0):
    from b12x.sequence.kda_prefill.reference import prefill_kda

    tokens, heads, width = x_all.shape[0], 32, 4096
    proj = F.linear(x_all, qkvfg_w)
    beta = F.linear(x_all, b_w)
    qkv = proj[:, : 3 * width]
    raw_g = proj[:, 3 * width : 4 * width].contiguous().view(tokens, heads, 128)
    gate = proj[:, 4 * width : 5 * width].contiguous().view(tokens, heads, 128)

    # Causal depthwise conv1d (kernel=4, left-pad 3) + SiLU.
    qkv_t = qkv.transpose(0, 1).unsqueeze(0)  # [1, 3*width, T]
    conv_out = F.conv1d(F.pad(qkv_t, (3, 0)), conv_w.unsqueeze(1), groups=3 * width)
    mixed = F.silu(conv_out.squeeze(0).transpose(0, 1)).to(torch.bfloat16)
    q, k, v = (t.contiguous().view(tokens, heads, 128) for t in mixed.split(width, dim=-1))

    cu = torch.tensor([0, tokens], dtype=torch.int32, device=x_all.device)
    init = torch.tensor([0], dtype=torch.int64, device=x_all.device)
    final = torch.tensor([1], dtype=torch.int64, device=x_all.device)
    ckpt_idx = torch.zeros((1,), dtype=torch.int64, device=x_all.device)
    ckpt_off = torch.zeros((1,), dtype=torch.int32, device=x_all.device)
    n_seqs = torch.tensor([1], dtype=torch.int32, device=x_all.device)
    n_toks = torch.tensor([tokens], dtype=torch.int32, device=x_all.device)
    state = torch.zeros((2, heads, 128, 128), dtype=torch.float32, device=x_all.device)
    rec = prefill_kda(
        q, k, v, raw_g, beta, A_log, dt_bias.view(heads, 128),
        cu_seqlens=cu, initial_state_indices=init, final_state_indices=final,
        recurrent_state=state,
        checkpoint_state_indices=ckpt_idx, checkpoint_offsets=ckpt_off,
        num_seqs=n_seqs, num_tokens=n_toks,
        lower_bound=lower_bound,
    )
    if isinstance(rec, tuple):
        rec = rec[0]
    rf = rec.float()
    normed = rf * torch.rsqrt(rf.pow(2).mean(dim=-1, keepdim=True) + eps) * o_norm_w.float()
    gated = (normed * torch.sigmoid(gate.float())).to(torch.bfloat16)
    return F.linear(gated.reshape(tokens, width), o_w)


@torch.no_grad()
def test_gated_mla_prefill_decode_and_cache_only_parity():
    from hyperflare_recipes.components.paged_batch import Metadata
    from hyperflare_recipes.components.rotary import PreparedRotaryAngles
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from hyperflare_recipes.recipes.ling32.attention import PreparedGatedMLA
    from hyperflare_recipes.recipes.ling32.config import PAGE_TOKENS, RECORD_DIM
    from hyperflare_recipes.recipes.ling32.decoder import rotary_frequencies
    from hyperflare_recipes.recipes.ling32.weights import MLAWeights

    torch.manual_seed(11)
    dev = torch.device("cuda:0")
    hidden, heads, eps = 2560, 32, 1e-6
    q_w = (torch.randn(heads * 192, hidden, device=dev) * 0.02).to(torch.bfloat16)
    kv_a_w = (torch.randn(RECORD_DIM, hidden, device=dev) * 0.02).to(torch.bfloat16)
    g_w = (torch.randn(heads, hidden, device=dev) * 0.02).to(torch.bfloat16)
    kv_norm_w = torch.ones(512, dtype=torch.bfloat16, device=dev) + torch.randn(512, device=dev).to(torch.bfloat16) * 0.05
    kv_b_w = (torch.randn(heads * 256, 512, device=dev) * 0.02).to(torch.bfloat16)
    dense_w = (torch.randn(hidden, heads * 128, device=dev) * 0.02).to(torch.bfloat16)

    kv_b_h = kv_b_w.view(heads, 256, 512)
    mla_w = MLAWeights(
        q_kv_a=torch.cat((q_w, kv_a_w), dim=0).contiguous(),
        g_proj=g_w,
        kv_norm=kv_norm_w,
        key=kv_b_h[:, :128, :].contiguous(),
        value=kv_b_h[:, 128:, :].transpose(1, 2).contiguous(),
        output=dense_w,
    )

    seq_len = 16
    x_all = (torch.randn(seq_len, hidden, device=dev) * 0.5).to(torch.bfloat16)
    freqs = rotary_frequencies(6_000_000.0, dev)
    ref_out = _reference_mla(x_all, q_w, kv_a_w, g_w, kv_norm_w, kv_b_w, dense_w, freqs=freqs, eps=eps)

    cache = torch.zeros(4, PAGE_TOKENS, RECORD_DIM, dtype=torch.bfloat16, device=dev)
    cache_only_pool = torch.zeros_like(cache)
    meta_pre = Metadata.allocate(15, 1, 2, dev)
    meta_dec = Metadata.allocate(1, 1, 2, dev)
    arena = ScratchArena(dev)
    bufs = {}
    with RecordedPreparationSession(
        device=dev, checkpoint="test_ling32_mla", workload_schema="ling32_test",
        numerical_policy="default", compile_workers=1, autotune_effort="off",
    ) as sess:
        sess.register_scratch(arena)
        rot_pre = PreparedRotaryAngles(meta_pre.positions, freqs).functional()
        rot_dec = PreparedRotaryAngles(meta_dec.positions, freqs).functional()
        mla_pre = PreparedGatedMLA(
            mla_w, torch.zeros(15, hidden, dtype=torch.bfloat16, device=dev),
            meta_pre, cache, rot_pre, eps=eps, mode="prefill",
            session=sess, name="mla_pre", workspace=arena, buffers=bufs,
        )
        mla_co = PreparedGatedMLA(
            mla_w, torch.zeros(15, hidden, dtype=torch.bfloat16, device=dev),
            meta_pre, cache_only_pool, rot_pre, eps=eps, mode="prefill",
            session=sess, name="mla_co", workspace=arena, buffers=bufs, cache_only=True,
        )
        mla_dec = PreparedGatedMLA(
            mla_w, torch.zeros(1, hidden, dtype=torch.bfloat16, device=dev),
            meta_dec, cache, rot_dec, eps=eps, mode="decode",
            session=sess, name="mla_dec", workspace=arena, buffers=bufs,
        )
        sess.tune()

        pages = torch.tensor([[0, 1]], dtype=torch.int32, device=dev)
        meta_pre.set(
            starts=torch.tensor([0], dtype=torch.int64, device=dev),
            lengths=torch.tensor([15], dtype=torch.int32, device=dev),
            state_rows=torch.tensor([1], dtype=torch.int64, device=dev),
            pages=pages,
            tokens=torch.zeros((1, 15), dtype=torch.int64, device=dev),
            live_rows=torch.tensor([1], dtype=torch.int32, device=dev),
        )
        out_pre = mla_pre(x_all[:15], meta_pre, rot_pre(meta_pre.positions))
        assert mla_co(x_all[:15], meta_pre, rot_pre(meta_pre.positions)) is None
        torch.testing.assert_close(cache_only_pool[0, :15], cache[0, :15], rtol=0.0, atol=0.0)

        meta_dec.set(
            starts=torch.tensor([15], dtype=torch.int64, device=dev),
            lengths=torch.tensor([1], dtype=torch.int32, device=dev),
            state_rows=torch.tensor([1], dtype=torch.int64, device=dev),
            pages=pages,
            tokens=torch.zeros((1, 1), dtype=torch.int64, device=dev),
            live_rows=torch.tensor([1], dtype=torch.int32, device=dev),
        )
        out_dec = mla_dec(x_all[15:16], meta_dec, rot_dec(meta_dec.positions))

        got = torch.cat((out_pre, out_dec), dim=0)
        torch.testing.assert_close(got, ref_out, rtol=3e-2, atol=2e-2)


@torch.no_grad()
def test_kda_prefill_decode_fused_verify_and_rollback_parity():
    from hyperflare_recipes.components.kda_speculation import KDARecords, PreparedKDAReplay
    from hyperflare_recipes.components.recurrent import DeltaMetadata, DeltaState, _commit_convolution
    from ignition.components.state import StateLayout, StateStorage
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from hyperflare_recipes.recipes.ling32.kda import PreparedKDA
    from hyperflare_recipes.recipes.ling32.weights import KDAWeights

    torch.manual_seed(17)
    dev = torch.device("cuda:0")
    hidden, heads, width, eps, lower_bound = 2560, 32, 4096, 1e-6, -5.0
    qkvfg_w = (torch.randn(5 * width, hidden, device=dev) * 0.015).to(torch.bfloat16)
    b_w = (torch.randn(heads, hidden, device=dev) * 0.02).to(torch.bfloat16)
    o_w = (torch.randn(hidden, width, device=dev) * 0.015).to(torch.bfloat16)
    conv_w = (torch.randn(3 * width, 4, device=dev) * 0.1).to(torch.bfloat16)
    o_norm_w = (torch.ones(128, device=dev) + torch.randn(128, device=dev) * 0.05).to(torch.bfloat16)
    A_log = torch.randn(heads, dtype=torch.float32, device=dev) * 0.2 - 1.0
    dt_bias = torch.randn(width, dtype=torch.float32, device=dev) * 0.1

    kda_w = KDAWeights(
        qkvfg=qkvfg_w, b_proj=b_w, output=o_w, convolution=conv_w,
        o_norm=o_norm_w, A_log=A_log, dt_bias=dt_bias,
    )
    layout = StateLayout.from_specs((
        ("layer0.convolution", (3 * width, 3 + 7), torch.bfloat16),
        ("layer0.recurrent", (heads, 128, 128), torch.float32),
    ))
    storage = StateStorage.allocate(layout, 4, dev)
    state = DeltaState(storage.region("layer0.convolution"), storage.region("layer0.recurrent"))
    kda_records = KDARecords.allocate({0: {"A_log": A_log, "dt_bias": dt_bias}}, 4, heads, dev, window=8)

    d_pre = DeltaMetadata.allocate(1, dev, verification_columns=1)
    d_ver = DeltaMetadata.allocate(1, dev, verification_columns=8)
    d_dec = DeltaMetadata.allocate(1, dev, verification_columns=1)
    arena = ScratchArena(dev)
    bufs = {}
    with RecordedPreparationSession(
        device=dev, checkpoint="test_ling32_kda", workload_schema="ling32_kda",
        numerical_policy="default", compile_workers=1, autotune_effort="off",
    ) as sess:
        sess.register_scratch(arena)
        kda_pre = PreparedKDA(
            kda_w, torch.zeros(10, hidden, dtype=torch.bfloat16, device=dev),
            state, d_pre, mode="prefill", eps=eps, lower_bound=lower_bound,
            session=sess, name="kda_pre", workspace=arena, buffers=bufs,
        )
        kda_ver = PreparedKDA(
            kda_w, torch.zeros(4, hidden, dtype=torch.bfloat16, device=dev),
            state, d_ver, mode="verify", eps=eps, lower_bound=lower_bound,
            session=sess, name="kda_ver", workspace=arena, buffers=bufs,
            records=kda_records.layer(0),
        )
        kda_dec = PreparedKDA(
            kda_w, torch.zeros(1, hidden, dtype=torch.bfloat16, device=dev),
            state, d_dec, mode="decode", eps=eps, lower_bound=lower_bound,
            session=sess, name="kda_dec", workspace=arena, buffers=bufs,
        )
        replay = PreparedKDAReplay(storage.backing, [state.recurrent], kda_records, kda_ver.plan, session=sess).functional()
        sess.tune()

        assert kda_ver.fused is not None, "Ling-3.2-Air KDA verify must use fused short_conv + recovery kernel"

        # Sequence: 10 prefill + 4 verify (accept only first 2, reject last 2) + 1 decode of true next token.
        x_pre = (torch.randn(10, hidden, device=dev) * 0.3).to(torch.bfloat16)
        x_acc = (torch.randn(2, hidden, device=dev) * 0.3).to(torch.bfloat16)
        x_rej = (torch.randn(2, hidden, device=dev) * 0.3).to(torch.bfloat16)
        x_next = (torch.randn(1, hidden, device=dev) * 0.3).to(torch.bfloat16)

        # 1. Prefill 10 tokens into row 1.
        d_pre.query_start_loc.copy_(torch.tensor([0, 10], dtype=torch.int32, device=dev))
        d_pre.convolution_slots.fill_(1)
        d_pre.fresh.fill_(True)
        d_pre.initial_state_indices.fill_(0)
        d_pre.final_state_indices.fill_(1)
        d_pre.num_seqs.fill_(1)
        d_pre.num_tokens.fill_(10)
        out_pre = kda_pre(x_pre, d_pre)

        # 2. Verify 4 candidate tokens [x_acc, x_rej] on row 1, then commit ONLY counts=2!
        d_ver.query_start_loc.copy_(torch.tensor([0, 4], dtype=torch.int32, device=dev))
        d_ver.convolution_slots.fill_(1)
        d_ver.fresh.fill_(False)
        d_ver.state_indices[:, 0].fill_(1)
        d_ver.num_seqs.fill_(1)
        d_ver.num_tokens.fill_(4)
        out_ver = kda_ver(torch.cat((x_acc, x_rej), dim=0), d_ver, records=kda_records.layer(0))

        counts = torch.tensor([2], dtype=torch.int32, device=dev)
        replay(counts, kda_records, d_ver)
        conv_region = layout.region("layer0.convolution")
        conv_offsets = torch.tensor([conv_region.offset // 2], dtype=torch.int64, device=dev)
        channels, columns = conv_region.shape
        torch.library.wrap_triton(_commit_convolution)[(1, 1, triton.cdiv(channels, 128))](
            storage.backing.view(torch.bfloat16), conv_offsets,
            torch.tensor([1], dtype=torch.int64, device=dev), counts,
            layout.row_bytes // 2, channels, columns, 128,
        )

        # 3. Decode 1 token (x_next) after rolling back rejected tokens!
        d_dec.query_start_loc.copy_(torch.tensor([0, 1], dtype=torch.int32, device=dev))
        d_dec.convolution_slots.fill_(1)
        d_dec.fresh.fill_(False)
        d_dec.state_indices[:, 0].fill_(1)
        d_dec.num_seqs.fill_(1)
        d_dec.num_tokens.fill_(1)
        out_dec = kda_dec(x_next, d_dec)

    # Reference on the true accepted sequence [x_pre, x_acc, x_next].
    ref_true = _reference_kda(
        torch.cat((x_pre, x_acc, x_next), dim=0),
        qkvfg_w, b_w, conv_w, o_norm_w, A_log, dt_bias, o_w, eps=eps, lower_bound=lower_bound,
    )
    # Also check rejected verification tokens matched [x_pre, x_acc, x_rej].
    ref_ver_all = _reference_kda(
        torch.cat((x_pre, x_acc, x_rej), dim=0),
        qkvfg_w, b_w, conv_w, o_norm_w, A_log, dt_bias, o_w, eps=eps, lower_bound=lower_bound,
    )

    torch.testing.assert_close(out_pre, ref_true[:10], rtol=3e-2, atol=2e-2)
    torch.testing.assert_close(out_ver, ref_ver_all[10:14], rtol=3e-2, atol=2e-2)
    torch.testing.assert_close(out_dec, ref_true[12:13], rtol=3e-2, atol=2e-2)


@torch.no_grad()
def test_mtp_prefill_absorb_and_causal_proposals_parity():
    from hyperflare_recipes.components.paged_batch import BatchInputs, Metadata
    from ignition.components.sampling import DeviceRequestState
    from hyperflare_recipes.components.weights import MLPWeights
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from hyperflare_recipes.recipes.ling32.config import ModelConfig, PAGE_TOKENS, RECORD_DIM
    from hyperflare_recipes.recipes.ling32.decoder import rotary_frequencies
    from hyperflare_recipes.recipes.ling32.mtp import PreparedAcceptedContext, PreparedMTP, PreparedProposals
    from hyperflare_recipes.recipes.ling32.pool import ContinuationPool
    from hyperflare_recipes.recipes.ling32.weights import LayerWeights, MLAWeights, MTPWeights

    torch.manual_seed(23)
    dev = torch.device("cuda:0")
    hidden, heads, vocab, eps = 2560, 32, 256, 1e-6

    # Minimal 1-MLA-layer config to construct a lightweight ContinuationPool with draft.latent + draft.target_hidden.
    text = {
        "num_hidden_layers": 1,
        "hidden_size": hidden,
        "intermediate_size": 512,
        "first_k_dense_replace": 100,  # dense MLP on layer 42 for pure BF16 reference parity
        "moe_intermediate_size": 768,
        "moe_shared_expert_intermediate_size": 768,
        "num_experts": 512,
        "num_experts_per_tok": 8,
        "n_group": 8,
        "topk_group": 4,
        "num_attention_heads": heads,
        "vocab_size": vocab,
        "rms_norm_eps": eps,
        "rope_theta": 6_000_000.0,
        "routed_scaling_factor": 2.5,
        "num_nextn_predict_layers": 1,
    }
    cfg = ModelConfig(raw={"text_config": text}, text=text, quantization={}, layer_types=("mla",))
    pool = ContinuationPool.for_model(cfg, 4, 8, dev, mtp=True, speculative_tokens=5)

    embed_w = (torch.randn(vocab, hidden, device=dev) * 0.1).to(torch.bfloat16)
    head_w = (torch.randn(vocab, hidden, device=dev) * 0.02).to(torch.bfloat16)
    enorm_w = (torch.ones(hidden, device=dev) + torch.randn(hidden, device=dev) * 0.05).to(torch.bfloat16)
    hnorm_w = (torch.ones(hidden, device=dev) + torch.randn(hidden, device=dev) * 0.05).to(torch.bfloat16)
    eh_w = (torch.randn(hidden, 2 * hidden, device=dev) * 0.02).to(torch.bfloat16)
    in_norm_w = (torch.ones(hidden, device=dev) + torch.randn(hidden, device=dev) * 0.05).to(torch.bfloat16)
    post_norm_w = (torch.ones(hidden, device=dev) + torch.randn(hidden, device=dev) * 0.05).to(torch.bfloat16)
    final_norm_w = (torch.ones(hidden, device=dev) + torch.randn(hidden, device=dev) * 0.05).to(torch.bfloat16)
    q_w = (torch.randn(heads * 192, hidden, device=dev) * 0.02).to(torch.bfloat16)
    kv_a_w = (torch.randn(RECORD_DIM, hidden, device=dev) * 0.02).to(torch.bfloat16)
    g_w = (torch.randn(heads, hidden, device=dev) * 0.02).to(torch.bfloat16)
    kv_norm_w = (torch.ones(512, device=dev) + torch.randn(512, device=dev) * 0.05).to(torch.bfloat16)
    kv_b_w = (torch.randn(heads * 256, 512, device=dev) * 0.02).to(torch.bfloat16)
    dense_w = (torch.randn(hidden, heads * 128, device=dev) * 0.02).to(torch.bfloat16)
    gate_w = (torch.randn(512, hidden, device=dev) * 0.02).to(torch.bfloat16)
    up_w = (torch.randn(512, hidden, device=dev) * 0.02).to(torch.bfloat16)
    down_w = (torch.randn(hidden, 512, device=dev) * 0.02).to(torch.bfloat16)

    kv_b_h = kv_b_w.view(heads, 256, 512)
    mla_w = MLAWeights(
        q_kv_a=torch.cat((q_w, kv_a_w), dim=0).contiguous(),
        g_proj=g_w, kv_norm=kv_norm_w,
        key=kv_b_h[:, :128, :].contiguous(),
        value=kv_b_h[:, 128:, :].transpose(1, 2).contiguous(),
        output=dense_w,
    )
    layer_w = LayerWeights(
        attention=mla_w, input_norm=in_norm_w, post_attention_norm=post_norm_w,
        mlp=MLPWeights(torch.cat((gate_w, up_w), dim=0).contiguous(), down_w),
        router=None, expert_bias=None, experts=None,
    )
    weights = SimpleNamespace(
        config=cfg, embedding=embed_w, head=head_w,
        draft=MTPWeights(enorm=enorm_w, hnorm=hnorm_w, eh_proj=eh_w, layer=layer_w, final_norm=final_norm_w),
    )

    inputs = BatchInputs.allocate(8, 2, dev, capacity=2)
    draft_tables = torch.full_like(inputs.page_tables, -1)
    draft_tables[0, 0] = 0
    inputs.row_slot_ids[0] = 0
    requests = DeviceRequestState.allocate(dev, vocab_size=vocab, capacity=2)

    meta_c0 = Metadata.allocate(4, 1, 2, dev)
    meta_c1 = Metadata.allocate(1, 1, 2, dev)  # explicitly exercises prefill1!
    arena = ScratchArena(dev)
    bufs = {}
    with RecordedPreparationSession(
        device=dev, checkpoint="test_ling32_mtp", workload_schema="ling32_mtp",
        numerical_policy="default", compile_workers=1, autotune_effort="off",
    ) as sess:
        sess.register_scratch(arena)
        ctx_c0 = PreparedAcceptedContext(
            weights, meta_c0, pool, draft_tables=draft_tables, inputs=inputs,
            examples=bufs, session=sess, name="ctx0", workspace=arena, is_prefill=True,
        ).functional()
        ctx_c1 = PreparedAcceptedContext(
            weights, meta_c1, pool, draft_tables=draft_tables, inputs=inputs,
            examples=bufs, session=sess, name="ctx1", workspace=arena, is_prefill=True,
        ).functional()
        proposer = PreparedProposals(
            weights, pool, 1, 2, draft_tables=draft_tables, inputs=inputs,
            requests=requests, examples=bufs, session=sess, name="prop", workspace=arena,
        )
        sess.tune()

        # Prompt of 5 tokens split into chunk 0 (4 tokens: t0..t3) + chunk 1 (1 token: t4), then propose with pending t5 + draft0.
        prompt_ids = torch.tensor([5, 11, 23, 37, 41], dtype=torch.int64, device=dev)
        pending_tok = torch.tensor([53], dtype=torch.int64, device=dev)
        draft0_tok = torch.tensor([67], dtype=torch.int64, device=dev)
        target_h = (torch.randn(5, hidden, device=dev) * 0.4).to(torch.bfloat16)

        pages = torch.tensor([[0, -1]], dtype=torch.int32, device=dev)
        # Chunk 0: positions 0..3 -> MTP absorbs positions 0..2 with (t1, h0), (t2, h1), (t3, h2); saves h3.
        meta_c0.set(
            starts=torch.tensor([0], dtype=torch.int64, device=dev),
            lengths=torch.tensor([4], dtype=torch.int32, device=dev),
            state_rows=torch.tensor([1], dtype=torch.int64, device=dev),
            pages=pages, tokens=prompt_ids[:4].view(1, 4),
            live_rows=torch.tensor([1], dtype=torch.int32, device=dev),
        )
        ctx_c0(target_h[:4], torch.tensor([4], dtype=torch.int32, device=dev), meta_c0)

        # Chunk 1 (prefill1!): position 4 -> MTP absorbs position 3 with (t4, h3); saves h4.
        meta_c1.set(
            starts=torch.tensor([4], dtype=torch.int64, device=dev),
            lengths=torch.tensor([1], dtype=torch.int32, device=dev),
            state_rows=torch.tensor([1], dtype=torch.int64, device=dev),
            pages=pages, tokens=prompt_ids[4:5].view(1, 1),
            live_rows=torch.tensor([1], dtype=torch.int32, device=dev),
        )
        ctx_c1(target_h[4:5], torch.tensor([1], dtype=torch.int32, device=dev), meta_c1)

        # Propose step 0 at before=5 (MTP pos 4, inputs (t5, h4)) and step 1 (MTP pos 5, inputs (draft0, h_mtp4)).
        states = torch.tensor([1], dtype=torch.int64, device=dev)
        before = torch.tensor([5], dtype=torch.int64, device=dev)
        active = torch.tensor([True], device=dev)
        dstate = proposer.begin(states)
        logits0, dstate1 = proposer.step(dstate, 0, pending_tok, before, states, active, active)
        logits1, _ = proposer.step(dstate1, 1, draft0_tok, before, states, active, active)

    # Reference BailingMoeV3MTPLayer over the full 6-position MTP sequence:
    # pos 0..4 use (prompt_ids[1..4], pending_tok) with target_h[0..4]; pos 5 uses (draft0_tok, ref_mtp_h[4]).
    def rms(t, w):
        f = t.float()
        return (f * torch.rsqrt(f.pow(2).mean(dim=-1, keepdim=True) + eps)).to(torch.bfloat16) * w

    def mtp_full(tok_seq, h_seq):
        joined = torch.cat((rms(embed_w[tok_seq], enorm_w), rms(h_seq, hnorm_w)), dim=-1)
        res = F.linear(joined, eh_w)
        att = _reference_mla(
            rms(res, in_norm_w), q_w, kv_a_w, g_w, kv_norm_w, kv_b_w, dense_w,
            freqs=rotary_frequencies(6_000_000.0, dev), eps=eps,
        )
        res = res + att
        n = rms(res, post_norm_w)
        mlp = F.linear((F.silu(F.linear(n, gate_w).float()) * F.linear(n, up_w).float()).to(torch.bfloat16), down_w)
        out_h = rms(res + mlp, final_norm_w)
        return out_h, F.linear(out_h, head_w)

    toks_5 = torch.cat((prompt_ids[1:], pending_tok))
    ref_h5, ref_l5 = mtp_full(toks_5, target_h)
    torch.testing.assert_close(logits0[0], ref_l5[4], rtol=3e-2, atol=2e-2)

    toks_6 = torch.cat((toks_5, draft0_tok))
    ref_h6, ref_l6 = mtp_full(toks_6, torch.cat((target_h, ref_h5[4:5]), dim=0))
    torch.testing.assert_close(logits1[0], ref_l6[5], rtol=3e-2, atol=2e-2)


@torch.no_grad()
def test_real_checkpoint_layers_clamped_moe_and_helios_catalog_validation():
    snapshot = os.environ.get("FLARE_LING32_SNAPSHOT")
    if not snapshot or not Path(snapshot).is_dir():
        pytest.skip("requires FLARE_LING32_SNAPSHOT pointing to a quantized Ling-3.2-Air checkpoint")

    import flare_heliopause
    from hyperflare_recipes.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from hyperflare_recipes.recipes.ling32.frontend import prepare_frontend, runtime_assets
    from hyperflare_recipes.recipes.ling32.moe import PreparedFeedForward
    from hyperflare_recipes.recipes.ling32.recipe import RECIPE
    from hyperflare_recipes.recipes.ling32.weights import Ling32WeightLoader
    from tests.preparation.test_native_recipe_contract import _recipe_metadata

    dev = torch.device("cuda:0")
    snap = Path(snapshot)
    manifest = inspect_indexed_checkpoint(snap)
    shard = TensorParallelShard(0, 1)
    with Ling32WeightLoader(manifest, dev, shard=shard, mtp=True, expert_scale_compression="csf") as loader:
        l0 = loader.load_layer(0)    # KDA + dense MLP
        l5 = loader.load_layer(5)    # MLA + unclamped MoE
        l35 = loader.load_layer(35)  # MLA + clamped MoE (limit=4.0, shared=5.0)
        head = loader.load_tensor("lm_head.weight")
        mtp = loader.load_mtp(head)
        cfg = loader.config
        assert l0.experts is None
        assert l5.experts is not None and l5.router.dtype == torch.float32
        assert l35.experts is not None and cfg.expert_swiglu_limit(35) == 4.0 and cfg.shared_swiglu_limit(35) == 5.0
        assert mtp.layer.experts is not None

    # Exercise PreparedFeedForward on real layer 35 in both decode (W4A16 + overlap) and prefill (W4A4).
    arena = ScratchArena(dev, overlap=True)
    bufs = {}
    with RecordedPreparationSession(
        device=dev, checkpoint="test_ling32_real_l35", workload_schema="ling32_real",
        numerical_policy="default", compile_workers=1, autotune_effort="off",
    ) as sess:
        sess.register_scratch(arena)
        ffn_dec = PreparedFeedForward(
            35, l35, cfg, torch.zeros(4, 2560, dtype=torch.bfloat16, device=dev),
            mode="decode", session=sess, name="l35.dec", workspace=arena, buffers=bufs, overlap=True,
        )
        ffn_pre = PreparedFeedForward(
            35, l35, cfg, torch.zeros(64, 2560, dtype=torch.bfloat16, device=dev),
            mode="prefill", session=sess, name="l35.pre", workspace=arena, buffers=bufs, overlap=False,
        )
        sess.tune()

        x_dec = (torch.randn(4, 2560, device=dev) * 0.5).to(torch.bfloat16)
        y_dec = ffn_dec(x_dec)
        assert y_dec.shape == (4, 2560) and torch.isfinite(y_dec).all()

        x_pre = (torch.randn(64, 2560, device=dev) * 0.5).to(torch.bfloat16)
        y_pre = ffn_pre(x_pre)
        assert y_pre.shape == (64, 2560) and torch.isfinite(y_pre).all()

    # Validate frontend extraction + patched chat template + Helios catalog when qualification bindings exist.
    from ignition.preparation.frontend import QwenPositionScheme, QwenResizeKernel

    frontend = prepare_frontend(snap, eos_token_ids=cfg.text["eos_token_id"])
    mm_frontend = prepare_frontend(snap, eos_token_ids=cfg.text["eos_token_id"], vision=True)
    assert mm_frontend["image"]["qwen"]["position_scheme"] == int(QwenPositionScheme.CENTERED_STEP)
    assert mm_frontend["image"]["qwen"]["resize_kernel"] == int(QwenResizeKernel.PILLOW_BICUBIC)
    assert mm_frontend["image"]["qwen"]["image_token_id"] == 157157
    assets = runtime_assets(snap)
    assert set(frontend["eos_token_ids"]) == {156892, 156895}
    assert "{%- mark -%}" in assets["chat_template"]["text"]
    if hasattr(flare_heliopause, "validate_prepared_catalog"):
        config_prep, contract, cat_manifest = _recipe_metadata("ling32", all_variants=False)
        server_value = dict(
            identity="00" * 32,
            model_id=RECIPE.served_model_name,
            max_sequence_tokens=config_prep.context,
            prefill_chunk_size=config_prep.prefill_chunk_size,
            pages=4096,
            state_rows=65,
            snapshots=32,
            page_tokens=128,
            frontend=frontend,
            sampling_defaults=RECIPE.sampling_defaults.as_dict(),
            assets=assets,
            model_contract=contract,
            speculative_widths=list(config_prep.speculative_widths),
            speculative_token_capacity=config_prep.speculative_token_capacity,
        )
        flare_heliopause.validate_prepared_catalog(cat_manifest, json.dumps(server_value))


@torch.no_grad()
def test_real_checkpoint_vision_encoder_matches_pytorch_reference():
    snapshot = os.environ.get("FLARE_LING32_SNAPSHOT")
    if not snapshot or not Path(snapshot).is_dir():
        pytest.skip("requires FLARE_LING32_SNAPSHOT pointing to a quantized Ling-3.2-Air checkpoint")

    from hyperflare_recipes.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from hyperflare_recipes.recipes.ling32.vision import (
        PreparedVision,
        VisionMetadata,
        VisionScratch,
        apply_vision_rope,
        vision_math_mode,
    )
    from hyperflare_recipes.recipes.ling32.weights import Ling32WeightLoader

    dev = torch.device("cuda:0")
    snap = Path(snapshot)
    manifest = inspect_indexed_checkpoint(snap)
    with Ling32WeightLoader(manifest, dev, shard=TensorParallelShard(0, 1), mtp=False, vision=True) as loader:
        vw = loader.load_vision()

    capacity = 256
    # 128x128 image -> 8x8 = 64 patches -> 16 merged visual tokens inside a 256-patch graph.
    grid_h, grid_w = 8, 8
    num_patches = grid_h * grid_w
    torch.manual_seed(42)
    pixels = torch.zeros((capacity, 1536), dtype=torch.bfloat16, device=dev)
    pixels[:num_patches] = (torch.randn(num_patches, 1536, device=dev) * 0.5).to(torch.bfloat16)

    # Build spatially merged patch coordinates and bilinear 48x48 interpolation matching Helios qwen.rs.
    pos_list, idx_list, wt_list = [], [], []
    side = 48
    for by in range(grid_h // 2):
        for bx in range(grid_w // 2):
            for dy in range(2):
                for dx in range(2):
                    r, c = by * 2 + dy, bx * 2 + dx
                    pos_list.append([r, c])
                    sy = r * (side - 1) / (grid_h - 1)
                    sx = c * (side - 1) / (grid_w - 1)
                    y0, x0 = int(sy), int(sx)
                    y1, x1 = min(y0 + 1, side - 1), min(x0 + 1, side - 1)
                    wy, wx = sy - y0, sx - x0
                    idx_list.append([y0 * side + x0, y0 * side + x1, y1 * side + x0, y1 * side + x1])
                    wt_list.append([(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx])

    meta = VisionMetadata.allocate(capacity, 1, dev)
    arena = ScratchArena(dev)
    with RecordedPreparationSession(
        device=dev, checkpoint="test_ling32_vision", workload_schema="ling32_vision",
        numerical_policy="default", compile_workers=1, autotune_effort="off",
    ) as sess:
        sess.register_scratch(arena)
        scratch = VisionScratch.allocate(capacity, dev, 2560)
        encoder = PreparedVision(
            vw, scratch, meta, session=sess, name="vision256",
            attention_workspace=arena, math_mode=vision_math_mode(dev),
        )
        sess.tune()

        meta.interpolation_indices[:num_patches].copy_(torch.tensor(idx_list, dtype=torch.int64, device=dev))
        meta.interpolation_weights[:num_patches].copy_(torch.tensor(wt_list, dtype=torch.float32, device=dev))
        meta.position_ids[:num_patches].copy_(torch.tensor(pos_list, dtype=torch.int64, device=dev))
        meta.cu_seqlens[0] = 0
        meta.cu_seqlens[1] = num_patches
        meta.num_patches[0] = num_patches
        got = encoder(pixels, meta)

    # Independent FP32 eager PyTorch reference matching BailingMoeV3VLVisionModel + linear_proj.
    def _rotate_half_f32(t):
        x1, x2 = t.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    x = F.linear(pixels[:num_patches].float(), vw.patch_embed_weight.float(), vw.patch_embed_bias.float())
    p_idx = torch.tensor(idx_list, dtype=torch.int64, device=dev)
    p_wt = torch.tensor(wt_list, dtype=torch.float32, device=dev)
    x = x + (F.embedding(p_idx, vw.pos_embed.float()) * p_wt[:, :, None]).sum(1)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 36, 2, dtype=torch.float32, device=dev) / 36.0))
    freqs = (torch.tensor(pos_list, dtype=torch.float32, device=dev).unsqueeze(-1) * inv_freq).flatten(1)
    cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1)[:, None, :]
    sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1)[:, None, :]
    for blk in vw.blocks:
        h = F.layer_norm(x, (1152,), blk.norm1_weight.float(), blk.norm1_bias.float(), eps=1e-6)
        qkv = F.linear(h, blk.qkv_weight.float(), blk.qkv_bias.float()).view(num_patches, 3, 16, 72)
        q = qkv[:, 0] * cos + _rotate_half_f32(qkv[:, 0]) * sin
        k = qkv[:, 1] * cos + _rotate_half_f32(qkv[:, 1]) * sin
        v = qkv[:, 2]
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            is_causal=False,
        ).squeeze(0).transpose(0, 1).reshape(num_patches, 1152)
        x = x + F.linear(o, blk.proj_weight.float(), blk.proj_bias.float())
        h2 = F.layer_norm(x, (1152,), blk.norm2_weight.float(), blk.norm2_bias.float(), eps=1e-6)
        x = x + F.linear(
            F.gelu(F.linear(h2, blk.fc1_weight.float(), blk.fc1_bias.float()), approximate="tanh"),
            blk.fc2_weight.float(),
            blk.fc2_bias.float(),
        )
    merged = F.layer_norm(
        x, (1152,), vw.merger_norm_weight.float(), vw.merger_norm_bias.float(), eps=1e-6
    ).view(num_patches // 4, 4608)
    ref = F.linear(
        F.gelu(F.linear(merged, vw.linear_proj_0_weight.float(), vw.linear_proj_0_bias.float())),
        vw.linear_proj_2_weight.float(),
        vw.linear_proj_2_bias.float(),
    )

    live_out = got[: num_patches // 4]
    assert torch.isfinite(live_out).all()
    assert (got[num_patches // 4 :] == 0).all()
    # Across 27 BF16 ViT layers + 2-layer projector (~7-bit mantissa compounded across 29 residual
    # GEMM blocks), allow BF16-vs-FP32 rounding within cos > 0.999 and atol=2.5e-2 / rtol=3e-2.
    cos_sim = F.cosine_similarity(live_out.float().flatten(), ref.flatten(), dim=0).item()
    assert cos_sim > 0.999, f"vision encoder cosine similarity too low: {cos_sim:.6f}"
    torch.testing.assert_close(live_out.float(), ref, rtol=3e-2, atol=2.5e-2)


@torch.no_grad()
def test_snapshot_restore_multimodal_rotary_offset_mtp_differential_cuda():
    """Differential CUDA check: production snapshot/restore of an image-prefilled row beside a second offset and inactive row.

    Exercises ContinuationPool.for_model, RotaryBatch.commit_rotary(advance_axis=0),
    PreparedMaintenance ("init"/"snapshot"/"restore" with trailing draft.latent pages),
    Metadata.pack(..., rotary_offset=...), PreparedAcceptedContext, and PreparedProposals.
    Compares the next MTP draft logits, committed draft state, trailing draft pages,
    second active row, and inactive row against uninterrupted execution with identical weights.
    """
    from dataclasses import replace

    from hyperflare_recipes.components.paged_batch import (
        Metadata,
        RotaryBatch,
        RotaryBatchInputs,
        gather_rotary_offset,
        rotary_offset_region,
    )
    from hyperflare_recipes.components.paged_maintenance import PreparedMaintenance
    from hyperflare_recipes.components.parallel import TensorParallelShard
    from ignition.components.completion import CompletionOutputs
    from ignition.components.sampling import (
        AdmissionBuffers,
        DeviceRequestState,
        RequestAdmission,
    )
    from ignition.components.state_transfer import StateCommands
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from hyperflare_recipes.recipes.ling32.config import (
        MROPE_SECTION,
        PAGE_TOKENS,
        RECORD_DIM,
        ModelConfig,
    )
    from hyperflare_recipes.recipes.ling32.mtp import (
        PreparedAcceptedContext,
        PreparedProposals,
    )
    from hyperflare_recipes.recipes.ling32.pool import ContinuationPool
    from hyperflare_recipes.recipes.ling32.weights import (
        LayerWeights,
        MLAWeights,
        MLPWeights,
        MTPWeights,
    )

    dev = torch.device("cuda:0")
    hidden, heads, vocab, eps = 2560, 32, 256, 1e-6
    text = {
        "hidden_size": hidden,
        "num_hidden_layers": 1,
        "first_k_dense_replace": 1,
        "layer_group_size": 1,
        "moe_intermediate_size": 768,
        "intermediate_size": 512,
        "num_experts": 16,
        "num_experts_per_tok": 4,
        "n_group": 8,
        "topk_group": 4,
        "num_attention_heads": heads,
        "vocab_size": vocab,
        "rms_norm_eps": eps,
        "rope_theta": 6_000_000.0,
        "routed_scaling_factor": 2.5,
        "num_nextn_predict_layers": 1,
        "mrope_section": list(MROPE_SECTION),
    }
    cfg = ModelConfig(
        raw={"text_config": text},
        text=text,
        quantization={},
        layer_types=("mla",),
        mrope_section=MROPE_SECTION,
    )

    torch.manual_seed(2026)
    embed_w = (torch.randn(vocab, hidden, device=dev) * 0.1).to(torch.bfloat16)
    head_w = (torch.randn(vocab, hidden, device=dev) * 0.02).to(torch.bfloat16)
    enorm_w = (torch.ones(hidden, device=dev) + torch.randn(hidden, device=dev) * 0.05).to(torch.bfloat16)
    hnorm_w = (torch.ones(hidden, device=dev) + torch.randn(hidden, device=dev) * 0.05).to(torch.bfloat16)
    eh_w = (torch.randn(hidden, 2 * hidden, device=dev) * 0.02).to(torch.bfloat16)
    in_norm_w = (torch.ones(hidden, device=dev) + torch.randn(hidden, device=dev) * 0.05).to(torch.bfloat16)
    post_norm_w = (torch.ones(hidden, device=dev) + torch.randn(hidden, device=dev) * 0.05).to(torch.bfloat16)
    final_norm_w = (torch.ones(hidden, device=dev) + torch.randn(hidden, device=dev) * 0.05).to(torch.bfloat16)
    q_w = (torch.randn(heads * 192, hidden, device=dev) * 0.02).to(torch.bfloat16)
    kv_a_w = (torch.randn(RECORD_DIM, hidden, device=dev) * 0.02).to(torch.bfloat16)
    g_w = (torch.randn(heads, hidden, device=dev) * 0.02).to(torch.bfloat16)
    kv_norm_w = (torch.ones(512, device=dev) + torch.randn(512, device=dev) * 0.05).to(torch.bfloat16)
    kv_b_w = (torch.randn(heads * 256, 512, device=dev) * 0.02).to(torch.bfloat16)
    dense_w = (torch.randn(hidden, heads * 128, device=dev) * 0.02).to(torch.bfloat16)
    gate_w = (torch.randn(512, hidden, device=dev) * 0.02).to(torch.bfloat16)
    up_w = (torch.randn(512, hidden, device=dev) * 0.02).to(torch.bfloat16)
    down_w = (torch.randn(hidden, 512, device=dev) * 0.02).to(torch.bfloat16)

    kv_b_h = kv_b_w.view(heads, 256, 512)
    mla_w = MLAWeights(
        q_kv_a=torch.cat((q_w, kv_a_w), dim=0).contiguous(),
        g_proj=g_w,
        kv_norm=kv_norm_w,
        key=kv_b_h[:, :128, :].contiguous(),
        value=kv_b_h[:, 128:, :].transpose(1, 2).contiguous(),
        output=dense_w,
    )
    layer_w = LayerWeights(
        attention=mla_w,
        input_norm=in_norm_w,
        post_attention_norm=post_norm_w,
        mlp=MLPWeights(torch.cat((gate_w, up_w), dim=0).contiguous(), down_w),
        router=None,
        expert_bias=None,
        experts=None,
    )
    weights = SimpleNamespace(
        config=cfg,
        embedding=embed_w,
        head=head_w,
        draft=MTPWeights(
            enorm=enorm_w,
            hnorm=hnorm_w,
            eh_proj=eh_w,
            layer=layer_w,
            final_norm=final_norm_w,
        ),
    )

    def _centered_step_positions(prefix_text: int, image_hw: tuple[int, int], suffix_text: int) -> torch.Tensor:
        """Build 3-axis M-RoPE coordinates matching Helios QwenPositionScheme::CenteredStep (qwen.rs)."""
        rows = [[i, i, i] for i in range(prefix_text)]
        cur = prefix_text
        h, w = image_hw
        h_center, w_center = (h - 1) // 2, (w - 1) // 2
        for y in range(h):
            for x in range(w):
                rows.append([cur, cur + y - h_center, cur + x - w_center])
        cur += 1
        for k in range(suffix_text):
            rows.append([cur + k, cur + k, cur + k])
        return torch.tensor(rows, dtype=torch.int64, device=dev)

    # Fixed prompt/verify payloads reused identically in uninterrupted vs snapshot/restore runs.
    # Slot 0 has 131 tokens: chunk 0 = 121 text + 2x4 centered-step image (129 tokens, crossing
    # PAGE_TOKENS=128, max(T)=121 < max(W)=123 -> advance_axis=0 commits rotary_offset=-7);
    # chunk 1 = 2 trailing text tokens at [122, 123] keeping rotary_offset=-7.
    # Crossing PAGE_TOKENS ensures PreparedMaintenance copies BOTH trailing draft.latent pages.
    torch.manual_seed(777)
    prompt0_ids = torch.randint(1, vocab, (131,), dtype=torch.int64, device=dev)
    prompt0_h = (torch.randn(131, hidden, device=dev) * 0.3).to(torch.bfloat16)
    rope0_full = _centered_step_positions(121, (2, 4), 2)
    rope0_c0 = rope0_full[:129].contiguous()
    rope0_c1 = rope0_full[129:].contiguous()

    # Slot 1 has 8 tokens: 2 text + 2x2 centered-step image + 2 text -> distinct rotary_offset=-3.
    prompt1_ids = torch.tensor([21, 22, 23, 24, 25, 26, 27, 28], dtype=torch.int64, device=dev)
    prompt1_h = (torch.randn(8, hidden, device=dev) * 0.3).to(torch.bfloat16)
    rope1 = _centered_step_positions(2, (2, 2), 2)

    toks_ver = torch.tensor([[31, 32, 33], [41, 42, 43], [0, 0, 0]], dtype=torch.int64, device=dev)
    h_ver = (torch.randn(9, hidden, device=dev) * 0.3).to(torch.bfloat16)
    pending = torch.tensor([51, 61, 0], dtype=torch.int64, device=dev)
    shard = TensorParallelShard(0, 1)

    def _run_sequence(*, snapshot_and_restore: bool):
        pool = ContinuationPool.for_model(
            cfg,
            8,
            8,
            dev,
            mtp=True,
            speculative_tokens=3,
            rotary_offset=True,
        )
        inputs = RotaryBatchInputs.allocate(129, 4, dev, capacity=4)
        draft_tables = torch.full_like(inputs.page_tables, -1)
        requests = DeviceRequestState.allocate(dev, vocab_size=vocab, capacity=4)
        admission = AdmissionBuffers.allocate(dev, vocab_size=vocab, capacity=4)
        commands = StateCommands.allocate(dev, capacity=4)
        completion = CompletionOutputs.allocate((2, 4, 32), device=dev)
        frame = torch.zeros(1, dtype=torch.int32, device=dev)
        sequence = torch.ones(1, dtype=torch.int64, device=dev)

        # Assign distinct state rows and disjoint physical draft pages for slots 0, 1, 2, and restore slot 3.
        inputs.request_state_rows[0, 0] = 1
        inputs.request_state_rows[1, 0] = 2
        inputs.request_state_rows[2, 0] = 3
        inputs.request_state_rows[3, 0] = 5
        inputs.page_tables[0, :2] = torch.tensor([0, 1], dtype=torch.int32, device=dev)
        inputs.page_tables[1, :1] = torch.tensor([2], dtype=torch.int32, device=dev)
        inputs.page_tables[2, :1] = torch.tensor([3], dtype=torch.int32, device=dev)
        inputs.page_tables[3, :2] = torch.tensor([0, 5], dtype=torch.int32, device=dev)
        draft_tables[0, :2] = torch.tensor([2, 3], dtype=torch.int32, device=dev)
        draft_tables[1, :2] = torch.tensor([4, 5], dtype=torch.int32, device=dev)
        draft_tables[2, :1] = torch.tensor([0], dtype=torch.int32, device=dev)
        draft_tables[3, :2] = torch.tensor([6, 7], dtype=torch.int32, device=dev)

        meta_c0 = Metadata.allocate(129, 1, 4, dev)
        meta_c1 = Metadata.allocate(2, 1, 4, dev)
        meta_s1 = Metadata.allocate(8, 1, 4, dev)
        meta_ver = Metadata.allocate(9, 3, 4, dev, samples=9)

        arena = ScratchArena(dev)
        bufs = {}
        with RecordedPreparationSession(
            device=dev,
            checkpoint=f"ling32_mtp_rotary_diff_{int(snapshot_and_restore)}",
            workload_schema="ling32_mtp_rotary_diff",
            numerical_policy="default",
            compile_workers=1,
            autotune_effort="off",
        ) as sess:
            sess.register_scratch(arena)
            maint = {
                mode: PreparedMaintenance(
                    mode,
                    requests,
                    pool,
                    commands,
                    inputs,
                    completion,
                    frame,
                    sequence,
                    runtime=None,
                    coordinator=None,
                    session=sess,
                    name=f"maint.{mode}",
                    capacity=4,
                    admission_buffers=admission,
                    draft_state=True,
                    page_tokens=PAGE_TOKENS,
                    trailing_pages=pool.draft_pages(),
                    trailing_tables=draft_tables,
                )
                for mode in ("init", "snapshot", "restore")
            }
            rotary_c0 = RotaryBatch(
                inputs, requests, meta_c0, mode="prefill", shard=shard, pool=pool, advance_axis=0
            )
            rotary_c1 = RotaryBatch(
                inputs, requests, meta_c1, mode="prefill", shard=shard, pool=pool, advance_axis=0
            )
            rotary_s1 = RotaryBatch(
                inputs, requests, meta_s1, mode="prefill", shard=shard, pool=pool, advance_axis=0
            )
            ctx_c0 = PreparedAcceptedContext(
                weights, meta_c0, pool, draft_tables=draft_tables, inputs=inputs,
                examples=bufs, session=sess, name="ctx_c0", workspace=arena, is_prefill=True,
            ).functional()
            ctx_c1 = PreparedAcceptedContext(
                weights, meta_c1, pool, draft_tables=draft_tables, inputs=inputs,
                examples=bufs, session=sess, name="ctx_c1", workspace=arena, is_prefill=True,
            ).functional()
            ctx_s1 = PreparedAcceptedContext(
                weights, meta_s1, pool, draft_tables=draft_tables, inputs=inputs,
                examples=bufs, session=sess, name="ctx_s1", workspace=arena, is_prefill=True,
            ).functional()
            ctx_ver = PreparedAcceptedContext(
                weights, meta_ver, pool, draft_tables=draft_tables, inputs=inputs,
                examples=bufs, session=sess, name="ctx_ver", workspace=arena, is_prefill=False,
            ).functional()
            proposer = PreparedProposals(
                weights, pool, 3, 1, draft_tables=draft_tables, inputs=inputs,
                requests=requests, examples=bufs, session=sess, name="prop", workspace=arena,
            )
            sess.tune()

            def _set_cmd(slot: int, src: int, dst: int, endpoint: int, op: int) -> None:
                commands.request_slots[0] = slot
                commands.source_slots[0] = src
                commands.destination_slots[0] = dst
                commands.expected_consumed[0] = endpoint
                commands.operation[0] = op
                commands.num_ops.fill_(1)

            # Initialize slots 0 (state 1), 1 (state 2), and inactive slot 2 (state 3) via production init.
            for slot, srow, seed, ptoks in (
                (0, 1, 13, prompt0_ids.tolist()),
                (1, 2, 29, prompt1_ids.tolist()),
                (2, 3, 47, [1]),
            ):
                admission.admission_count.fill_(1)
                admission.stage(0, slot, RequestAdmission(1, seed, max_new_tokens=16), ptoks)
                _set_cmd(slot, -1, srow, 0, 1)
                maint["init"]()
                assert not commands.errors.count_nonzero()

            # Slot 0 prefill chunk 0 (129 tokens, non-square centered-step image M-RoPE -> offset -7).
            inputs.row_slot_ids[0] = 0
            meta_c0.set(
                starts=torch.tensor([0], dtype=torch.int64, device=dev),
                lengths=torch.tensor([129], dtype=torch.int32, device=dev),
                state_rows=torch.tensor([1], dtype=torch.int64, device=dev),
                pages=inputs.page_tables[0:1],
                tokens=prompt0_ids[:129].view(1, 129),
                live_rows=torch.tensor([1], dtype=torch.int32, device=dev),
            )
            meta_c0_rope = replace(meta_c0, rotary_positions=rope0_c0)
            ctx_c0(prompt0_h[:129], torch.tensor([129], dtype=torch.int32, device=dev), meta_c0_rope)
            rotary_c0.commit_rotary(meta_c0_rope)
            assert int(rotary_offset_region(pool)[1].item()) == -7

            # Slot 0 prefill chunk 1 (2 tokens across PAGE_TOKENS boundary -> 131 total consumed).
            meta_c1.set(
                starts=torch.tensor([129], dtype=torch.int64, device=dev),
                lengths=torch.tensor([2], dtype=torch.int32, device=dev),
                state_rows=torch.tensor([1], dtype=torch.int64, device=dev),
                pages=inputs.page_tables[0:1],
                tokens=prompt0_ids[129:131].view(1, 2),
                live_rows=torch.tensor([1], dtype=torch.int32, device=dev),
            )
            meta_c1_rope = replace(meta_c1, rotary_positions=rope0_c1)
            ctx_c1(prompt0_h[129:131], torch.tensor([2], dtype=torch.int32, device=dev), meta_c1_rope)
            rotary_c1.commit_rotary(meta_c1_rope)
            requests.consumed_tokens[0] = 131
            assert int(rotary_offset_region(pool)[1].item()) == -7

            # Slot 1 prefill (8 tokens, 2x2 centered-step image -> offset -3).
            inputs.row_slot_ids[0] = 1
            meta_s1.set(
                starts=torch.tensor([0], dtype=torch.int64, device=dev),
                lengths=torch.tensor([8], dtype=torch.int32, device=dev),
                state_rows=torch.tensor([2], dtype=torch.int64, device=dev),
                pages=inputs.page_tables[1:2],
                tokens=prompt1_ids.view(1, 8),
                live_rows=torch.tensor([1], dtype=torch.int32, device=dev),
            )
            meta_s1_rope = replace(meta_s1, rotary_positions=rope1)
            ctx_s1(prompt1_h, torch.tensor([8], dtype=torch.int32, device=dev), meta_s1_rope)
            rotary_s1.commit_rotary(meta_s1_rope)
            requests.consumed_tokens[1] = 8
            assert int(rotary_offset_region(pool)[2].item()) == -3

            active_slot_0, active_state_0 = 0, 1
            active_draft_slice_0 = slice(2, 4)
            if snapshot_and_restore:
                orig_draft_pages = pool.pages["draft.latent"][2:4].clone()
                # Snapshot slot 0 (state 1) -> immutable row 4 via production PreparedMaintenance.
                inputs.row_slot_ids[0] = 0
                inputs.row_generations[0] = 1
                inputs.live_rows.fill_(1)
                _set_cmd(0, 1, 4, 131, 0)
                maint["snapshot"]()
                assert not commands.errors.count_nonzero()

                # Wipe original continuation row and both original draft physical pages before restore.
                pool.backing[1].zero_()
                pool.pages["draft.latent"][2:4].zero_()

                # Restore immutable row 4 -> slot 3 (state 5) with disjoint physical draft pages [6, 7].
                admission.admission_count.fill_(1)
                admission.stage(0, 3, RequestAdmission(1, 99, max_new_tokens=16), prompt0_ids.tolist())
                _set_cmd(3, 4, 5, 131, 0)
                maint["restore"]()
                assert not commands.errors.count_nonzero()
                assert int(requests.consumed_tokens[3].item()) == 131
                assert int(rotary_offset_region(pool)[5].item()) == -7
                assert torch.equal(pool.pages["draft.latent"][6:8], orig_draft_pages)
                active_slot_0, active_state_0 = 3, 5
                active_draft_slice_0 = slice(6, 8)

            # Packed partial-acceptance verification commit: row 0 accepts 2/3, row 1 accepts 1/3, row 2 inactive.
            inputs.row_slot_ids[:3] = torch.tensor([active_slot_0, 1, 2], dtype=torch.int32, device=dev)
            ver_rows = torch.tensor([active_state_0, 2, 3], dtype=torch.int64, device=dev)
            ver_offsets = gather_rotary_offset(rotary_offset_region(pool), ver_rows)
            ver_meta = Metadata.pack(
                starts=torch.tensor([131, 8, 0], dtype=torch.int64, device=dev),
                lengths=torch.tensor([3, 3, 0], dtype=torch.int32, device=dev),
                state_rows=ver_rows,
                pages=inputs.page_tables[[active_slot_0, 1, 2]],
                tokens=toks_ver,
                live_rows=torch.tensor([2], dtype=torch.int32, device=dev),
                token_capacity=9,
                sample_count=9,
                max_query_tokens=3,
                rotary_offset=ver_offsets,
            )
            ctx_ver(h_ver, torch.tensor([2, 1, 0], dtype=torch.int32, device=dev), ver_meta)

            # Propose next draft step consuming restored rotary_offset and restored trailing draft pages.
            before = torch.tensor([133, 9, 0], dtype=torch.int64, device=dev)
            active = torch.tensor([True, True, False], device=dev)
            dstate = proposer.begin(ver_rows)
            logits, _ = proposer.step(dstate, 0, pending, before, ver_rows, active, active)
            return (
                logits[:2].clone(),
                pool.draft_target_hidden()[active_state_0].clone(),
                rotary_offset_region(pool)[active_state_0].clone(),
                pool.pages["draft.latent"][active_draft_slice_0].clone(),
                pool.backing[2].clone(),
                pool.backing[3].clone(),
            )

    ref_logits, ref_hid0, ref_off0, ref_dpages0, ref_row1, ref_inactive = _run_sequence(snapshot_and_restore=False)
    res_logits, res_hid0, res_off0, res_dpages0, res_row1, res_inactive = _run_sequence(snapshot_and_restore=True)
    torch.testing.assert_close(res_logits, ref_logits, rtol=1e-3, atol=1e-3)
    assert torch.equal(res_hid0, ref_hid0)
    assert torch.equal(res_off0, ref_off0)
    assert torch.equal(res_dpages0, ref_dpages0)
    assert torch.equal(res_row1, ref_row1)
    assert torch.equal(res_inactive, ref_inactive)
    assert not res_inactive.count_nonzero()
