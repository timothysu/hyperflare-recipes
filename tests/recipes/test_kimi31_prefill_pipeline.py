# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Exactness, dependency ordering, and cross-layer tail overlap for Kimi M-microbatch prefill pipelining."""

from types import SimpleNamespace

import pytest
import torch

from hyperflare_recipes.components.paged_batch import Metadata
from hyperflare_recipes.recipes.kimi3 import decoder
from hyperflare_recipes.recipes.kimi3.residual import ResidualStream


def _reference_mix(prefix, blocks, delta, mode, *, sources, write, score, norm, eps, mixture=None):
    # Pure-PyTorch CPU equivalent of ResidualStream.mix so CPU tests verify exact tensor flow.
    if mode == 2:
        current = delta.clone()
    elif mode == 1:
        current = (prefix.float() + delta.float()).to(prefix.dtype)
    else:
        current = prefix.clone()
    if mode != 0:
        prefix.copy_(current)
    if write >= 0:
        blocks[:, write].copy_(current)
    if sources == 0:
        mixed = current.float()
    else:
        candidates = [blocks[:, i].float() for i in range(sources)] + [current.float()]
        stacked = torch.stack(candidates, dim=1)
        scale = torch.rsqrt(stacked.square().mean(dim=-1, keepdim=True) + eps)
        scores = (stacked * score.float()[None, None, :]).sum(dim=-1) * scale.squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        mixed = (weights[:, :, None] * stacked).sum(dim=1).to(prefix.dtype).float()
    if mixture is not None:
        mixture.copy_(mixed.to(mixture.dtype))
    normalized = (mixed * torch.rsqrt(mixed.square().mean(dim=-1, keepdim=True) + eps)).to(prefix.dtype)
    return (norm.float() * normalized.float()).to(prefix.dtype)


class _TrackingCollective:
    """Ordered mock collective that checks serialization and returns exact deterministic results."""

    def __init__(self, log, label, transform):
        self.log = log
        self.label = label
        self.transform = transform
        self.in_flight = []

    def __call__(self, source, output=None, *, dependencies=()):
        self.log.append(("sync", self.label, tuple(source.shape), len(dependencies)))
        result = self.transform(source)
        if output is not None:
            output.copy_(result)
            return output
        return result

    def start(self, *args, dependencies=()):
        self.log.append(("start", self.label, tuple(args[0].shape), len(dependencies)))
        token = SimpleNamespace(output=self.transform(*args), args=args, dependencies=dependencies)
        self.in_flight.append(token)
        return token

    def finish(self, pending, dependency):
        assert self.in_flight and self.in_flight[0] is pending
        assert dependency is not None
        self.in_flight.pop(0)
        self.log.append(("finish", self.label))
        return pending.output


class _FakeKDA:
    def __init__(self, hidden, width, chunk_rows):
        gen = torch.Generator().manual_seed(11)
        self.w_in = torch.randn((width, hidden), generator=gen, dtype=torch.bfloat16) * 0.05
        self.w_out = torch.randn((hidden, width), generator=gen, dtype=torch.bfloat16) * 0.05
        self.chunk_rows = chunk_rows

    def project_input_chunk(self, x):
        assert x.shape[0] == self.chunk_rows
        return x @ self.w_in.T

    def run_core(self, projected, delta_meta, *, records=None):
        # Cross-row recurrence surrogate: cumulative weighted mix across rows.
        shifted = torch.roll(projected, shifts=1, dims=0)
        shifted[0].zero_()
        return (projected.float() * 0.75 + shifted.float() * 0.25).to(projected.dtype)

    def project_output_chunk(self, gated):
        assert gated.shape[0] == self.chunk_rows
        return gated @ self.w_out.T

    def __call__(self, x, delta_meta, *, records=None):
        return self.run_core(x @ self.w_in.T, delta_meta, records=records) @ self.w_out.T


class _FakeDSA:
    def __init__(self, hidden, width, chunk_rows):
        gen = torch.Generator().manual_seed(23)
        self.w_in = torch.randn((2 * width, hidden), generator=gen, dtype=torch.bfloat16) * 0.05
        self.w_out = torch.randn((hidden, width), generator=gen, dtype=torch.bfloat16) * 0.05
        self.width = width
        self.chunk_rows = chunk_rows

    def project_input_chunk(self, x):
        assert x.shape[0] == self.chunk_rows
        return x @ self.w_in.T

    def run_core(self, source, local, metadata, *, shared_index=None):
        assert source.shape[0] == local.shape[0]
        latent = local[:, :self.width]
        gate = local[:, self.width:]
        # Incorporate source-dependent cross-row mix (matching indexer dependence on full source).
        mean_tag = source[:, :self.width].float().mean(dim=0, keepdim=True)
        attended = (latent.float() * 0.8 + mean_tag * 0.2).to(latent.dtype)
        return attended, gate

    def finish_chunk(self, attended_chunk, gate_chunk):
        assert attended_chunk.shape[0] == self.chunk_rows
        gated = (attended_chunk.float() * torch.sigmoid(gate_chunk.float()).to(attended_chunk.dtype).float()).to(attended_chunk.dtype)
        return gated @ self.w_out.T

    def __call__(self, x, metadata, *, shared_index=None):
        local = x @ self.w_in.T
        attended, gate = self.run_core(x, local, metadata, shared_index=shared_index)
        gated = (attended.float() * torch.sigmoid(gate.float()).to(attended.dtype).float()).to(attended.dtype)
        return gated @ self.w_out.T


class _FakeDenseMLP(decoder.PreparedMLP):
    def __init__(self, hidden):
        gen = torch.Generator().manual_seed(31)
        self.w = torch.randn((hidden, hidden), generator=gen, dtype=torch.bfloat16) * 0.05

    def __call__(self, x):
        return x @ self.w.T


class _FakeLatentMoE(decoder.PreparedLatentMoE):
    def __init__(self, hidden, latent, experts_n, log, layer_index):
        gen = torch.Generator().manual_seed(41 + layer_index)
        self.hidden = hidden
        self.latent_width = latent
        self.world = 2
        self.w_router = torch.randn((experts_n // 2, hidden), generator=gen, dtype=torch.bfloat16) * 0.05
        self.w_down = torch.randn((latent // 2, hidden), generator=gen, dtype=torch.bfloat16) * 0.05
        self.w_shared = torch.randn((hidden, hidden), generator=gen, dtype=torch.bfloat16) * 0.05
        self.w_exp = torch.randn((latent, latent), generator=gen, dtype=torch.bfloat16) * 0.05
        self.w_up = torch.randn((hidden, latent // 2), generator=gen, dtype=torch.bfloat16) * 0.05
        self.latent_norm = torch.ones((latent // 2,), dtype=torch.bfloat16)
        self.eps = 1e-5
        self.prefill_route_gather = _TrackingCollective(
            log, f"l{layer_index}.rg",
            lambda down, logits: (torch.cat((down, down * 0.5), dim=-1), torch.cat((logits, logits * 0.5), dim=-1)),
        )
        self.concurrent_expert_reduce = _TrackingCollective(
            log, f"l{layer_index}.er", lambda partial: (partial.float() * 1.25).to(partial.dtype)
        )
        self.expert_reduce = SimpleNamespace(parts=[self.concurrent_expert_reduce])
        self.expert_call_rows = []

    def router_projection(self, x):
        return (x @ self.w_router.T).float()

    def down(self, x):
        return x @ self.w_down.T

    def shared(self, x):
        return x @ self.w_shared.T

    def router(self, logits):
        weights, ids = torch.topk(torch.sigmoid(logits), k=2, dim=-1)
        return SimpleNamespace(ids=ids.to(torch.int32), weights=weights)

    def experts(self, latent, ids, weights):
        self.expert_call_rows.append(latent.shape[0])
        scale = weights.sum(dim=-1, keepdim=True).to(latent.dtype)
        return (latent @ self.w_exp.T) * scale

    def post_moe(self, latent_sum, shared):
        normed = (latent_sum.float() * torch.rsqrt(latent_sum.float().square().mean(dim=-1, keepdim=True) + self.eps)).to(latent_sum.dtype)
        up_in = (normed[:, :self.latent_width // 2].float() * self.latent_norm.float()).to(latent_sum.dtype)
        return (up_in @ self.w_up.T) + shared

    def __call__(self, source, *, fuse_add=False):
        logits = self.router_projection(source)
        down = self.down(source)
        shared = self.shared(source)
        latent, full_logits = self.prefill_route_gather.transform(down, logits)
        routes = self.router(full_logits)
        partial = self.experts(latent, routes.ids, routes.weights)
        latent_sum = self.concurrent_expert_reduce.transform(partial)
        return self.post_moe(latent_sum, shared)


@pytest.mark.parametrize("subchunks", [1, 2])
@pytest.mark.parametrize("auxiliary_stream", ["attn_res", "prefix"])
def test_pipelined_decoder_matches_serial_reference_across_dense_kda_and_dsa(
    monkeypatch, subchunks, auxiliary_stream
):
    # Patch ResidualStream.mix to CPU PyTorch implementation for device-free verification.
    def cpu_mix(self, delta, mode, *, sources, write, score, norm, eps, output_eps, mixture=None,
                output=None, reduction_choice=None, addend=None):
        out = self.output if output is None else output
        result = _reference_mix(self.prefix, self.blocks, delta, mode, sources=sources, write=write,
                                score=score, norm=norm, eps=eps, mixture=mixture)
        out.copy_(result)
        return out

    monkeypatch.setattr(ResidualStream, "mix", cpu_mix)

    reduce_step, hidden, latent = 8, 32, 16
    h = reduce_step * subchunks
    tokens = 2 * h
    layer_kinds = ("kda", "kda", "dsa", "kda")
    block_size = 2
    aux_layers = (1, 3)

    gen = torch.Generator().manual_seed(7)
    emb_values = torch.randn((tokens, hidden), generator=gen, dtype=torch.bfloat16)

    layer_weights = []
    for _ in layer_kinds:
        layer_weights.append({
            "attention_res_score": torch.randn((hidden,), generator=gen, dtype=torch.float32) * 0.1,
            "input_layernorm": torch.ones((hidden,), dtype=torch.bfloat16),
            "mlp_res_score": torch.randn((hidden,), generator=gen, dtype=torch.float32) * 0.1,
            "post_attention_layernorm": torch.ones((hidden,), dtype=torch.bfloat16),
        })

    def build_decoder(pipelined):
        log = []
        dec = decoder.PreparedDecoder.__new__(decoder.PreparedDecoder)
        dec.tokens, dec.hidden = tokens, hidden
        dec.sequences = 1
        dec.verification_columns = 1
        dec.vision_state = None
        dec.shared_index_type = dec.index_shards = None
        dec.replay = None
        dec.row_mixes = None
        dec.scoring = None
        dec.compact_head = False
        dec.pipeline_chunk = h if pipelined else None
        dec.reduce_step = reduce_step if pipelined else None
        dec.subchunks = subchunks if pipelined else 0
        dec.row_shard_prefill = False
        dec.shard_rows = None
        dec.final_gather = None
        dec.residual_blocks = -(-len(layer_kinds) // block_size)
        dec.auxiliary_layers = aux_layers
        dec.auxiliary_stream = auxiliary_stream
        dec.weights = SimpleNamespace(
            config=SimpleNamespace(
                text={"attn_res_block_size": block_size, "rms_norm_eps": 1e-5},
                layers=layer_kinds,
            ),
            output_res_score=torch.ones((hidden,), dtype=torch.float32) * 0.1,
            norm=torch.ones((hidden,), dtype=torch.bfloat16),
        )
        dec.embedding = SimpleNamespace(
            lookup=lambda token_ids, num_rows: emb_values,
            gather=lambda local: local.clone(),
        )
        dec.head = lambda sampled: sampled.float()
        dec.layers = []
        for idx, (w, kind) in enumerate(zip(layer_weights, layer_kinds, strict=True)):
            attn = (_FakeKDA(hidden, latent, h) if kind == "kda" else _FakeDSA(hidden, latent, h))
            mlp = (_FakeDenseMLP(hidden) if idx == 0 else _FakeLatentMoE(hidden, latent, 8, log, idx))
            scale_fn = lambda x, factor=1.125: (x.float() * factor).to(x.dtype)
            parts = [_TrackingCollective(log, f"l{idx}.r{c}", scale_fn) for c in range(2 * subchunks)]
            # Make reduce callable on full [tokens, hidden] for the serial path.
            class _FullReduce:
                def __init__(self, p, fn):
                    self.parts = p
                    self.fn = fn

                def __call__(self, x):
                    return self.fn(x)

            full_reduce = _FullReduce(parts, scale_fn)
            cr = tuple(parts) if pipelined else None
            dec.layers.append((w, attn, mlp, full_reduce, cr, None))
        return dec, log

    meta = Metadata.allocate(tokens, 1, 4, "cpu")
    meta.lengths.fill_(tokens)
    meta.live_counts.copy_(torch.tensor([tokens, 1], dtype=torch.int32))
    meta.sample_indices[0] = tokens - 1

    ref_dec, _ = build_decoder(pipelined=False)
    pipe_dec, pipe_log = build_decoder(pipelined=True)

    ref_sampled, ref_logits, ref_aux, _ = ref_dec(meta)
    pipe_sampled, pipe_logits, pipe_aux, _ = pipe_dec(meta)

    torch.testing.assert_close(pipe_sampled, ref_sampled, rtol=0, atol=0)
    torch.testing.assert_close(pipe_logits, ref_logits, rtol=0, atol=0)
    torch.testing.assert_close(pipe_aux, ref_aux, rtol=0, atol=0)

    # Every started collective must have been joined, and routed experts stay unsplit.
    starts = [entry for entry in pipe_log if entry[0] == "start"]
    finishes = [entry for entry in pipe_log if entry[0] == "finish"]
    assert len(starts) == len(finishes) and len(starts) > 0
    for _, _, mlp, _, cr, _ in pipe_dec.layers:
        for part in cr:
            assert not part.in_flight
        if isinstance(mlp, _FakeLatentMoE):
            assert not mlp.prefill_route_gather.in_flight
            assert not mlp.concurrent_expert_reduce.in_flight
            assert mlp.expert_call_rows == [tokens]
    # Every started collective after layer 0's first attention reduce carries explicit ordering dependencies.
    assert all(entry[3] >= 1 for entry in starts[1:])


def test_prepared_latent_moe_does_not_pipeline_when_chunk_rows_is_none_at_2048(monkeypatch):
    """When PreparedDecoder leaves pipeline_chunk=None for 2048 tokens, MoE stays unpipelined."""
    captured = {}

    class _StubPlan:
        def __init__(self, label):
            self.label = label

    class _StubOp:
        def __init__(self, label, output=None):
            self.plan = _StubPlan(label)
            self.output = output

        def functional(self, **kwargs):
            return self

    def fake_prepare_projection(source, weight, output, **kwargs):
        captured.setdefault("projections", []).append((kwargs["name"], tuple(source.shape), tuple(output.shape)))
        return _StubOp(kwargs["name"], output)

    class _FakeUnquantizedProjection(_StubOp):
        def __init__(self, source, weight, output, **kwargs):
            captured["router_proj"] = (tuple(source.shape), tuple(output.shape))
            super().__init__(kwargs["name"], output)

    class _FakeChunkedAR(_StubOp):
        def __init__(self, source, output, **kwargs):
            captured["expert_reduce"] = (tuple(source.shape), tuple(output.shape))
            self.parts = [_StubOp(kwargs["name"] + ".0", output)]
            super().__init__(kwargs["name"], output)

    class _FakePairGather(_StubOp):
        def __init__(self, first, second, out_first, out_second, **kwargs):
            captured["pair_gather"] = (
                tuple(first.shape),
                tuple(second.shape),
                tuple(out_first.shape),
                tuple(out_second.shape),
                kwargs.get("concurrent"),
            )
            super().__init__(kwargs["name"], out_first)

    class _FakeRouter(_StubOp):
        def __init__(self, logits, routes, **kwargs):
            captured["router"] = (tuple(logits.shape), tuple(routes.ids.shape))
            super().__init__(kwargs["name"])

    class _FakeExperts(_StubOp):
        def __init__(self, latent, weights, ids, rweights, output, **kwargs):
            captured["experts"] = (tuple(latent.shape), tuple(ids.shape), tuple(output.shape), kwargs.get("reserve_sms"))
            super().__init__(kwargs["name"], output)

    class _FakeShared(_StubOp):
        def __init__(self, weights, source, **kwargs):
            captured["shared"] = tuple(source.shape)
            self.gate_up = _StubOp("shared.gate_up")
            self.down = _StubOp("shared.down")
            super().__init__(kwargs["name"])

    monkeypatch.setattr(decoder, "prepare_projection", fake_prepare_projection)
    monkeypatch.setattr(decoder, "UnquantizedProjection", _FakeUnquantizedProjection)
    monkeypatch.setattr(decoder, "PreparedChunkedAllReduce", _FakeChunkedAR)
    monkeypatch.setattr(decoder, "PreparedColumnPairGather", _FakePairGather)
    monkeypatch.setattr(decoder, "PreparedRouter", _FakeRouter)
    monkeypatch.setattr(decoder, "PreparedRoutedExperts", _FakeExperts)
    monkeypatch.setattr(decoder, "PreparedMLP", _FakeShared)

    tokens, hidden, latent, num_experts, world = 2048, 64, 32, 32, 16
    source = torch.empty((tokens, hidden), dtype=torch.bfloat16)
    weights = {
        "latent_norm": torch.ones((latent // world,), dtype=torch.bfloat16),
        "latent_up": torch.empty((hidden, latent // world), dtype=torch.bfloat16),
        "latent_down_rows": torch.empty((latent // world, hidden), dtype=torch.bfloat16),
        "router": torch.empty((num_experts // world, hidden), dtype=torch.bfloat16),
        "correction": torch.zeros((num_experts,), dtype=torch.float32),
        "experts": object(),
        "shared": object(),
    }
    config = SimpleNamespace(
        text={
            "routed_expert_hidden_size": latent,
            "rms_norm_eps": 1e-6,
            "num_experts_per_token": 4,
            "num_experts": num_experts,
            "routed_scaling_factor": 1.0,
        }
    )
    shard = SimpleNamespace(world_size=world, rank=0)
    storage = SimpleNamespace(hidden_columns=None, latent_begin=None, row_tuning=None)
    workspace = SimpleNamespace(overlap=True)
    runtime = SimpleNamespace(should_allreduce=lambda tensor: True)

    moe = decoder.PreparedLatentMoE(
        weights,
        source,
        config,
        shard,
        runtime=runtime,
        router_runtime=None,
        coordinator=None,
        session=None,
        name="prefill2048.layer1.moe",
        workspace=workspace,
        storage=storage,
        prefill_route_runtime=runtime,
        chunk_rows=None,
        row_shard_prefill=False,
    )
    assert moe.pipelined is False
    assert captured["pair_gather"][3] == (tokens, num_experts)
    assert captured["router"][0] == (tokens, num_experts)
    assert captured["experts"] == ((tokens, latent), (tokens, 4), (tokens, latent), 0)
    assert captured["pair_gather"][4] is False
