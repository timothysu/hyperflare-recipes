# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek V4.1 PCIe row-sharded encoder prefill eligibility, shard clamping, and Engram selection."""

from dataclasses import dataclass
import os
from types import SimpleNamespace

import pytest
import torch

from b12x.comm import pcie
from hyperflare_recipes.components.mhc import MHCScratch
from hyperflare_recipes.components.mlp import MLPScratch
from hyperflare_recipes.components.moe import RouteRecord, RouterScratch
from hyperflare_recipes.components.paged_batch import Metadata
from hyperflare_recipes.components.parallel import TensorParallelShard
from hyperflare_recipes.recipes.deepseek41 import attention as ds_attention, decoder, engram as ds_engram, recipe as ds_recipe
from hyperflare_recipes.recipes.deepseek41.bootstrap import PreparationConfig
from ignition.planning import Gpu, Plan, UsageError
from ignition.preparation.scratch import ScratchArena

_RUN_CUDA_TESTS = os.environ.get("FLARE_RUN_CUDA_TESTS") == "1"


def _sm120_gpus(count: int, capability: tuple[int, int] = (12, 0)) -> tuple[Gpu, ...]:
    return tuple(
        Gpu(i, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{i}", 96 << 30, 90 << 30, capability)
        for i in range(count)
    )


def _make_plan(*, gpus: tuple[Gpu, ...], row_shard_prefill: bool) -> Plan:
    from dataclasses import replace

    return Plan(
        recipe=ds_recipe.RECIPE,
        checkpoint=ds_recipe.RECIPE.checkpoints[0],
        gpus=gpus,
        parameters=replace(ds_recipe.RECIPE.parameters, row_shard_prefill=row_shard_prefill),
        cache_target=8192,
        cache_states=256,
        pages=1024,
        weights_bytes=1,
        graph_bytes=1,
    )


def test_recipe_and_plan_gate_row_shard_prefill_to_tp4(monkeypatch):
    from dataclasses import replace

    assert ds_recipe.RECIPE.parameters.row_shard_prefill_world_sizes == (4,)

    default_cfg = PreparationConfig()
    assert default_cfg.row_shard_prefill is False
    assert PreparationConfig(
        parameters=replace(ds_recipe.RECIPE.parameters, row_shard_prefill=True),
    ).row_shard_prefill is True
    for invalid in (1, 0, "true", None):
        with pytest.raises(TypeError, match="row_shard_prefill must be a boolean"):
            replace(ds_recipe.RECIPE.parameters, row_shard_prefill=invalid)

    valid_plan = _make_plan(gpus=_sm120_gpus(4), row_shard_prefill=True)
    assert valid_plan.parameters.row_shard_prefill is True

    captured = {}

    def fake_prepare_group(snapshot, config, prepare_rank, **kwargs):
        captured["config"] = config
        return "prepared"

    import ignition.preparation.rank_group as rank_group

    monkeypatch.setattr(rank_group, "prepare_group", fake_prepare_group)
    result = ds_recipe.prepare(
        "/tmp/snapshot",
        valid_plan,
        compile_forwards=True,
        kernel_cache=None,
        progress=None,
    )
    assert result == "prepared"
    assert captured["config"].row_shard_prefill is True

    for world in (3, 6):
        with pytest.raises(UsageError, match="--row-shard-prefill is not supported at TP"):
            _make_plan(gpus=_sm120_gpus(world), row_shard_prefill=True)

    with pytest.raises(UsageError, match="not qualified on SM103"):
        _make_plan(gpus=_sm120_gpus(4, capability=(10, 3)), row_shard_prefill=True)


class _FakeDmaRuntime:
    def __init__(self, world_size: int = 4, *, max_bytes: int = 64 << 20, shard_capacity: int = 16 << 20,
                 with_surfaces: bool = True):
        self.world_size = world_size
        self.max_bytes = max_bytes
        self.shard_capacity = shard_capacity
        if with_surfaces:
            self.should_reduce_scatter = lambda tensor: True
            self.should_all_gather = lambda tensor: True

    def should_allreduce(self, tensor):
        return True


def _build_decoder_for_eligibility(
    monkeypatch,
    *,
    tokens: int = 1024,
    rank: int = 0,
    world_size: int = 4,
    mode: str = "prefill",
    row_shard_prefill: bool = True,
    overlap: bool = True,
    scoring=None,
    runtime="default",
    encoder_only: bool = True,
):
    monkeypatch.setattr(pcie, "DmaAllReduce", _FakeDmaRuntime)
    if runtime == "default":
        runtime = _FakeDmaRuntime(world_size=world_size)

    built_layers = []
    built_boundaries = []
    layer_inputs = []

    class FakeTableRequests:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, metadata):
            return "history"

        def commit(self, lengths, raw_history, metadata):
            pass

    class FakeEmbedding:
        def __init__(self, *args, **kwargs):
            pass

        def functional(self, **kwargs):
            return self

        def __call__(self, *, token_ids, num_rows):
            idx = torch.arange(token_ids.numel(), dtype=torch.float32)
            out = torch.zeros((token_ids.numel(), 5120), dtype=torch.bfloat16)
            out[:, 0] = (idx // 256).to(torch.bfloat16)
            out[:, 1] = (idx % 256).to(torch.bfloat16)
            return out

    class FakeIndices:
        def __init__(self, *args, **kwargs):
            pass

        def functional(self):
            return self

        def __call__(self, metadata, **kwargs):
            return "indices"

    class FakeLayer:
        def __init__(self, index, *args, **kwargs):
            self.index = index
            self.has_image_mask = False
            self.pipelined_ffn = kwargs["pipelined_ffn"]
            self.pipelined_attn = kwargs["pipelined_attn"]
            self.row_shard_dma = kwargs["row_shard_dma"]
            self.shard_rows = kwargs["shard_rows"]
            self.boundary = kwargs["boundary"]
            self.engram = None
            self.attention = SimpleNamespace(selectors=[object()], plans=lambda: ())
            self.post = None if kwargs["defer_post"] else object()
            self.cr_mlp_1 = SimpleNamespace(reduce=SimpleNamespace(plan=SimpleNamespace()))
            self.attention_hc = SimpleNamespace(plan=None)
            built_layers.append(self)

        def __call__(self, incoming, pre_mix, metadata, indices, selection, **kwargs):
            if self.index == 0:
                layer_inputs.append((incoming, pre_mix))
            return incoming, pre_mix, f"rec{self.index}", selection

    class FakeBoundary:
        def __init__(self, index, layer, source, target, packing, pool, rotary, **kwargs):
            self.index = index
            self.packing = packing
            self.row_shard_dma = kwargs.get("row_shard_dma", False)
            built_boundaries.append(self)

        def __call__(self, incoming, pre_mix, metadata, indices, indexer=None):
            if self.packing is None:
                return None, None, None
            suffix = self.packing.begin(metadata)
            zeros_4d = torch.zeros((128, 4, 5120), dtype=torch.bfloat16)
            state = (
                zeros_4d,
                torch.zeros((128, 4), dtype=torch.float32),
                torch.zeros((128, 4, 4), dtype=torch.float32),
                torch.zeros((128, 5120), dtype=torch.bfloat16),
            )
            return suffix, state, torch.zeros((128, 4), dtype=torch.float32)

    monkeypatch.setattr(decoder, "PreparedTableRequests", FakeTableRequests)
    monkeypatch.setattr(decoder, "PreparedEmbedding", FakeEmbedding)
    monkeypatch.setattr(decoder, "Rotary", lambda *args, **kwargs: None)
    monkeypatch.setattr(decoder, "AttentionIndices", FakeIndices)
    monkeypatch.setattr(decoder, "PreparedLayer", FakeLayer)
    monkeypatch.setattr(decoder, "PreparedDecoderBoundary", FakeBoundary)
    monkeypatch.setattr(decoder, "register_finalizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(decoder.PreparedDecoder, "_prepare_head", lambda *args, **kwargs: setattr(args[0], "head", lambda h, **kw: torch.zeros((1, 16))))

    @dataclass(frozen=True)
    class FakeLayerWeights:
        engram: object
        attention: dict

    layers = tuple(
        FakeLayerWeights(engram=None, attention={"compressor": object()} if i % 2 else {})
        for i in range(40)
    )
    weights = SimpleNamespace(
        config=SimpleNamespace(
            decoder_start=20,
            sources=(SimpleNamespace(layer=2), SimpleNamespace(layer=20)),
            text={"compress_ratios": [0, 0, *([2] * 18), *([1] * 20)], "rms_norm_eps": 1e-6},
            draft=SimpleNamespace(target_layers=(35,)),
        ),
        draft=None,
        layers=layers,
        shard=TensorParallelShard(rank, world_size),
        embedding=torch.zeros((16, 5120), dtype=torch.bfloat16),
        norm=torch.ones(5120, dtype=torch.bfloat16),
        head=torch.zeros((16, 5120), dtype=torch.bfloat16),
    )
    scratch = SimpleNamespace(
        embedding=torch.zeros((tokens, 5120), dtype=torch.bfloat16),
        residual=torch.zeros((tokens, 4, 5120), dtype=torch.bfloat16),
        attention=SimpleNamespace(output=torch.zeros((tokens, 5120), dtype=torch.bfloat16)),
        attention_output=torch.zeros((tokens, 5120), dtype=torch.bfloat16),
        attention_hc=SimpleNamespace(normalized=torch.zeros((tokens, 5120), dtype=torch.bfloat16)),
        hidden=torch.zeros((1, 5120), dtype=torch.bfloat16),
        local_logits=torch.zeros((1, 16), dtype=torch.float32),
        logits=torch.zeros((1, 16), dtype=torch.float32),
    )
    rows = 1 if mode == "prefill" else tokens
    metadata = Metadata.allocate(tokens, rows, max(2, tokens // 128 + 2), "cpu")
    metadata.lengths.fill_(tokens)
    metadata.live_counts.copy_(torch.tensor([tokens, rows], dtype=torch.int32))
    arena = ScratchArena("cpu", overlap=overlap)
    scoring_obj = None if scoring is None else SimpleNamespace(prepare=lambda *a, **kw: (lambda x: None))

    dec = decoder.PreparedDecoder(
        weights,
        None,
        scratch,
        metadata,
        None,
        mode=mode,
        runtime=runtime,
        vocabulary_runtime=None,
        coordinator=None,
        session=None,
        name=f"test_{tokens}",
        workspace=arena,
        scoring=scoring_obj,
        encoder_only=encoder_only,
        row_shard_prefill=row_shard_prefill,
    )
    return dec, metadata, built_layers, built_boundaries, layer_inputs


@pytest.mark.parametrize("tokens", [1024, 4096])
def test_prepared_decoder_row_shard_eligibility_and_embedding_partition(monkeypatch, tokens):
    half = tokens // 2
    shard_tokens = half // 4
    for rank in range(4):
        dec, metadata, built_layers, built_boundaries, layer_inputs = _build_decoder_for_eligibility(
            monkeypatch, tokens=tokens, rank=rank, encoder_only=True
        )
        assert dec.row_shard_dma is True
        assert dec.half_tokens == half
        assert dec.shard_tokens == shard_tokens
        s0 = rank * shard_tokens
        assert dec.shard_rows_a.tolist() == list(range(s0, s0 + shard_tokens))
        assert dec.shard_rows_b.tolist() == list(range(half + s0, half + s0 + shard_tokens))
        assert dec.shard_offset_a.tolist() == [s0]
        assert dec.shard_offset_b.tolist() == [half + s0]
        assert len(built_layers) == 20 and all(layer.row_shard_dma for layer in built_layers)
        assert len(built_boundaries) == 1 and built_boundaries[0].row_shard_dma is True

        dec(metadata)
        (res_a, res_b), (pre_mix_a, pre_mix_b) = layer_inputs[0]
        assert res_a.shape == (shard_tokens, 4, 5120)
        assert res_b.shape == (shard_tokens, 4, 5120)
        decoded_a = (res_a[:, 0, 0].float() * 256.0 + res_a[:, 0, 1].float()).to(torch.int64).tolist()
        decoded_b = (res_b[:, 0, 0].float() * 256.0 + res_b[:, 0, 1].float()).to(torch.int64).tolist()
        assert decoded_a == list(range(s0, s0 + shard_tokens))
        assert decoded_b == list(range(half + s0, half + s0 + shard_tokens))
        assert torch.all(pre_mix_a[:, 0] == 1.0) and torch.all(pre_mix_a[:, 1:] == 0.0)
        assert torch.all(pre_mix_b[:, 0] == 1.0) and torch.all(pre_mix_b[:, 1:] == 0.0)

    # Terminal prefill (encoder + bounded 128-row decoder suffix) row-shards encoder layers 0..19 only.
    term_dec, _, term_layers, term_boundaries, _ = _build_decoder_for_eligibility(
        monkeypatch, tokens=tokens, rank=1, encoder_only=False
    )
    assert term_dec.row_shard_dma is True
    assert all(layer.row_shard_dma for layer in term_layers[:20])
    assert all(not layer.row_shard_dma for layer in term_layers[20:])
    assert len(term_boundaries) == 1 and term_boundaries[0].row_shard_dma is True


@pytest.mark.parametrize(
    "override",
    [
        {"row_shard_prefill": False},
        {"tokens": 512},
        {"tokens": 1536},  # >= 1024 and divisible by 8, but per-rank half-shard 192 % 128 != 0
        {"scoring": True},
        {"mode": "decode", "tokens": 32, "encoder_only": False},
        {"overlap": False},
        {"world_size": 1, "rank": 0},
        {"runtime": None},
        {"runtime": "no_surfaces"},
        {"runtime": "tiny_capacity"},
    ],
)
def test_prepared_decoder_rejects_ineligible_row_shard_configurations(monkeypatch, override):
    kwargs = dict(tokens=1024, rank=0, world_size=4, mode="prefill", row_shard_prefill=True, overlap=True)
    kwargs.update(override)
    if kwargs.get("runtime") == "no_surfaces":
        kwargs["runtime"] = _FakeDmaRuntime(world_size=4, with_surfaces=False)
    elif kwargs.get("runtime") == "tiny_capacity":
        kwargs["runtime"] = _FakeDmaRuntime(world_size=4, max_bytes=1024)
    if kwargs.get("scoring"):
        kwargs["encoder_only"] = False

    dec, _, built_layers, built_boundaries, _ = _build_decoder_for_eligibility(monkeypatch, **kwargs)
    assert dec.row_shard_dma is False
    assert dec.shard_tokens == 0
    assert dec.shard_rows_a is None and dec.shard_offset_a is None
    assert all(not layer.row_shard_dma for layer in built_layers)
    assert all(not boundary.row_shard_dma for boundary in built_boundaries)


def test_prepared_layer_row_shard_clamps_partial_tails_and_reconstructs_mxfp8_scales():
    tokens, world, hidden, top_k = 1024, 4, 256, 6
    half = tokens // 2
    shard_tokens = half // world

    class PassCollective:
        def __init__(self, fn):
            self.fn = fn

        def start(self, source, *, dependencies=()):
            out = self.fn(source)
            return SimpleNamespace(source=source, output=out, dependencies=dependencies)

        def finish(self, pending, dependency):
            return pending.output

    router_calls = []

    def make_rank_layer(rank: int):
        layer = decoder.PreparedLayer.__new__(decoder.PreparedLayer)
        s0 = rank * shard_tokens
        layer.row_shard_dma = True
        layer.pipelined_attn_hc = False
        layer.pipelined_attn = "full"
        layer.defer_mlp_b = False
        layer.half_tokens = half
        layer.shard_tokens = shard_tokens
        layer.shard_rows_a = torch.arange(s0, s0 + shard_tokens, dtype=torch.int64)
        layer.shard_rows_b = torch.arange(half + s0, half + s0 + shard_tokens, dtype=torch.int64)
        layer.shard_offset_a = torch.tensor([s0], dtype=torch.int32)
        layer.shard_offset_b = torch.tensor([half + s0], dtype=torch.int32)
        layer.engram = None
        layer.has_image_mask = True
        layer.attn_quantizer_plan = None
        layer.attn_local_shapes = None
        layer.attn_vals_bf16_cols = 0
        layer.ffn_quantizer_plan = None
        layer.ffn_local_shapes = None
        layer.ffn_meta_scale_cols = 0

        layer.attention_hc = lambda src, *, pre_mix, pre_out: (
            src,
            torch.zeros((shard_tokens, 4), dtype=torch.float32),
            torch.zeros((shard_tokens, 4, 4), dtype=torch.float32),
            src[:, 0, :],
        )
        layer.cr_attn_gather_0 = PassCollective(lambda sh: sh.repeat(world, 1))
        layer.cr_attn_gather_1 = PassCollective(lambda sh: sh.repeat(world, 1))
        layer.attention = SimpleNamespace(
            mode="prefill",
            init_window_cache=lambda metadata, dev: "wc",
            run_half=lambda norm, metadata, selection, **kw: norm,
            commit=lambda lengths, records, metadata: None,
        )
        layer.cr_att_0 = PassCollective(lambda full: full[s0 : s0 + shard_tokens].clone())
        layer.cr_att_1 = PassCollective(lambda full: full[s0 : s0 + shard_tokens].clone())
        layer.ffn_hc = lambda att, res, post, comb, *, pre_mix, pre_out: (res, post, comb, att)
        layer.router_projection = lambda norm: torch.zeros((norm.shape[0], 384), dtype=torch.float32)

        def fake_router(logits, *, image_mask=None, num_valid_tokens=None):
            valid = int(num_valid_tokens.item())
            router_calls.append((rank, valid, image_mask.clone()))
            ids = torch.full((logits.shape[0], top_k), rank + 10, dtype=torch.int32)
            weights = torch.full((logits.shape[0], top_k), 0.12345679 + rank * 0.01, dtype=torch.float32)
            if valid < logits.shape[0]:
                ids[valid:].fill_(-1)
                weights[valid:].zero_()
            return RouterScratch(torch.zeros_like(weights), ids, weights)

        layer.router = fake_router
        layer.cr_ffn_vals_0 = PassCollective(lambda sh: sh.repeat(world, 1))
        layer.cr_ffn_meta_0 = PassCollective(lambda sh: sh.repeat(world, 1))
        layer.cr_ffn_vals_1 = PassCollective(lambda sh: sh.repeat(world, 1))
        layer.cr_ffn_meta_1 = PassCollective(lambda sh: sh.repeat(world, 1))
        layer.experts = lambda vals, ids, weights, **kw: vals
        layer.shared = lambda vals, **kw: vals
        layer.cr_mlp_0 = PassCollective(lambda full: full[s0 : s0 + shard_tokens].clone())
        layer.cr_mlp_1 = PassCollective(lambda full: full[s0 : s0 + shard_tokens].clone())
        layer.post = lambda mlp_sh, res, post, comb: res
        return layer

    metadata = Metadata.allocate(tokens, 1, 10, "cpu")
    # Place synthetic image tokens (>= 129280) at distinctive indices across both microbatch halves.
    image_positions = {0, 127, 128, 255, 300, 511, 512, 640, 900, 1023}
    metadata.token_ids.copy_(
        torch.tensor([130000 if idx in image_positions else (idx % 1000) for idx in range(tokens)], dtype=torch.int64)
    )

    rank_layers = [make_rank_layer(rank) for rank in range(world)]
    for live in (0, 1, 127, 128, 129, 300, 512, 513, 700, 1023, 1024):
        metadata.live_counts[0] = live
        metadata.lengths[0] = live
        router_calls.clear()
        for rank, layer in enumerate(rank_layers):
            inc_a = torch.ones((shard_tokens, 4, hidden), dtype=torch.bfloat16)
            inc_b = torch.ones((shard_tokens, 4, hidden), dtype=torch.bfloat16)
            pm_a = torch.zeros((shard_tokens, 4), dtype=torch.float32)
            pm_b = torch.zeros((shard_tokens, 4), dtype=torch.float32)
            (_, _), _, _, _ = layer._run_row_shard_dma(
                (inc_a, inc_b), (pm_a, pm_b), metadata, "indices", ("sel", None, None)
            )

        for rank in range(world):
            call_a = router_calls[2 * rank]
            call_b = router_calls[2 * rank + 1]
            expected_a = max(0, min(shard_tokens, live - rank * shard_tokens))
            expected_b = max(0, min(shard_tokens, live - (half + rank * shard_tokens)))
            assert call_a[0] == rank and call_a[1] == expected_a
            assert call_b[0] == rank and call_b[1] == expected_b
            s0 = rank * shard_tokens
            expected_mask_a = torch.tensor([idx in image_positions for idx in range(s0, s0 + shard_tokens)])
            expected_mask_b = torch.tensor([idx in image_positions for idx in range(half + s0, half + s0 + shard_tokens)])
            assert torch.equal(call_a[2], expected_mask_a)
            assert torch.equal(call_b[2], expected_mask_b)

    # Verify _run_ffn_gathered MXFP8 scale tile reconstruction and RouteRecord unpacking directly.
    mxfp8_layer = make_rank_layer(0)
    mxfp8_layer.ffn_quantizer_plan = object()
    scale_cols_u8 = hidden // 32
    mxfp8_layer.ffn_meta_scale_cols = scale_cols_u8 // 2  # uint8 pairs packed as BF16 columns

    full_vals_fp8 = torch.ones((half, hidden), dtype=torch.float8_e4m3fn)
    # Unique scale byte per (row, scale_col) to verify exact MMA tile transpose.
    srows_2d = ((torch.arange(half * scale_cols_u8, dtype=torch.int64) % 251) + 1).to(torch.uint8).reshape(half, scale_cols_u8)
    expected_ids = ((torch.arange(half * top_k, dtype=torch.int32) * 5) % 384).reshape(half, top_k)
    expected_weights = torch.linspace(0.05123457, 0.91234565, half * top_k, dtype=torch.float32).reshape(half, top_k)
    route_rec = RouteRecord.empty(half, top_k, "cpu")
    route_rec.write(expected_ids, expected_weights)
    meta_full = torch.cat((srows_2d.view(torch.bfloat16), route_rec.packed), dim=1)

    seen_experts, seen_shared = [], []

    def capture_experts(vals, ids, weights, **kwargs):
        seen_experts.append((vals, ids.clone(), weights.clone(), kwargs))
        return torch.ones((half, hidden), dtype=torch.bfloat16)

    def capture_shared(vals, **kwargs):
        seen_shared.append((vals, kwargs))
        return torch.full((half, hidden), 2.0, dtype=torch.bfloat16)

    mxfp8_layer.experts = capture_experts
    mxfp8_layer.shared = capture_shared
    summed, unpacked_ids = mxfp8_layer._run_ffn_gathered(full_vals_fp8.view(torch.bfloat16), meta_full)
    assert torch.all(summed == 3.0)
    assert torch.equal(unpacked_ids, expected_ids)
    exp_vals, exp_ids, exp_weights, exp_kw = seen_experts[0]
    assert exp_vals.dtype == torch.float8_e4m3fn and exp_vals.shape == (half, hidden)
    assert torch.equal(exp_ids, expected_ids)
    assert torch.equal(exp_weights.view(torch.int32), expected_weights.view(torch.int32))
    assert torch.equal(exp_kw["a_scales"], srows_2d.view(1, half, scale_cols_u8))

    sh_vals, sh_kw = seen_shared[0]
    bases_vals, bases_srows, bases_smma = sh_kw["encoded_bases"]
    assert sh_vals.dtype == torch.float8_e4m3fn
    assert torch.equal(bases_srows, srows_2d.view(1, half, scale_cols_u8))
    assert bases_smma.shape == (1, half // 128, hidden // 128, 32, 4, 4)
    # Spot-check tile mapping: row = m_tile*128 + r_outer*32 + r_inner, col = k_tile*4 + c_inner.
    for m_tile in range(half // 128):
        for k_tile in range(hidden // 128):
            for r_outer in range(4):
                for c_inner in range(4):
                    rows_idx = m_tile * 128 + r_outer * 32 + torch.arange(32)
                    col_idx = k_tile * 4 + c_inner
                    assert torch.equal(bases_smma[0, m_tile, k_tile, :, r_outer, c_inner], srows_2d[rows_idx, col_idx])


def test_prepared_engram_row_shard_eligibility_and_owner_row_selection(monkeypatch):
    monkeypatch.setattr(pcie, "DmaAllReduce", _FakeDmaRuntime)

    class FakeLookup:
        def __init__(self, *args, **kwargs):
            pass

        def functional(self, **kwargs):
            return self

        def __call__(self, source, count):
            return torch.zeros((source.shape[0], 6144), dtype=torch.bfloat16)

    class FakeAllGather:
        def __init__(self, source, output, **kwargs):
            self.source_shape = tuple(source.shape)
            self.output_shape = tuple(output.shape)

        def functional(self, **kwargs):
            return self

    class FakeChunkedAllReduce:
        def __init__(self, *args, **kwargs):
            pass

        def functional(self, **kwargs):
            return self

    projection_shapes = []

    class FakeQuantizedProjection:
        def __init__(self, source, weight, output, **kwargs):
            self.weight = weight
            projection_shapes.append((kwargs["name"], tuple(source.shape), tuple(output.shape)))

        def functional(self, **kwargs):
            return self

        def __call__(self, x):
            # Deterministic linear map from [rows, 6144] -> [rows, 25600] preserving row identity.
            row_tag = x[:, :1].float()
            return row_tag.expand(x.shape[0], 25600).to(torch.bfloat16).contiguous()

    mix_caps = []

    def fake_mix_plan(caps, *, invocation):
        mix_caps.append(caps.max_tokens)
        return SimpleNamespace(request=lambda **kwargs: SimpleNamespace(**kwargs))

    def fake_run_engram_mix(source_2d, projected, norm_weight, *, eps, plan, out, token_mask):
        delta = torch.where(token_mask[:, None], projected[:, : source_2d.shape[1]], torch.zeros_like(source_2d))
        out.copy_(source_2d + delta)

    monkeypatch.setattr(ds_engram, "PreparedEngramLookup", FakeLookup)
    monkeypatch.setattr(ds_engram, "PreparedAllGather", FakeAllGather)
    monkeypatch.setattr(ds_engram, "PreparedChunkedAllReduce", FakeChunkedAllReduce)
    monkeypatch.setattr(ds_engram, "QuantizedProjection", FakeQuantizedProjection)
    monkeypatch.setattr(ds_engram.hyperconnection, "plan", fake_mix_plan)
    monkeypatch.setattr(ds_engram.hyperconnection, "run_engram_mix", fake_run_engram_mix)
    monkeypatch.setattr(ds_engram, "declare", lambda session, reqs: None)

    tokens, world = 1024, 4
    half = tokens // 2
    shard_rows = half // world
    pool = SimpleNamespace(config=SimpleNamespace(text={"rms_norm_eps": 1e-6}))
    weights_full = {
        "projection_full": "proj_full_w",
        "projection": "proj_sharded_w",
        "weight": torch.ones(5120, dtype=torch.bfloat16),
    }

    def build_engram(*, tok: int, rank: int, w_size: int, runtime, weights, row_shard):
        tables = SimpleNamespace(
            columns=torch.arange(0, 6144 // w_size, dtype=torch.int64),
            shard=TensorParallelShard(rank, w_size),
            inputs={2: object()},
            token_map=(0, 1),
            geometry=object(),
            readiness_runtime=None,
        )
        meta = Metadata.allocate(tok, 1, 10, "cpu")
        scratch = ds_engram.EngramScratch.allocate(tok, "cpu", world=w_size)
        src = torch.zeros((tok, 4, 5120), dtype=torch.bfloat16)
        out = torch.zeros((tok, 4, 5120), dtype=torch.bfloat16)
        return (
            ds_engram.PreparedEngram(
                2, weights, tables, src, out, meta, pool,
                scratch=scratch, session=None, name=f"l2_r{rank}", workspace=None,
                runtime=runtime, row_shard=row_shard,
            ),
            meta,
        )

    dma_rt = _FakeDmaRuntime(world_size=world)

    # Reference full-chunk Engram (row_shard=None).
    projection_shapes.clear()
    mix_caps.clear()
    full_engram, meta = build_engram(
        tok=tokens, rank=0, w_size=world, runtime=dma_rt, weights=weights_full, row_shard=None
    )
    assert full_engram.use_all_gather is True
    assert full_engram.row_shard is None and full_engram.row_shard_indices is None
    assert projection_shapes[-1][1][0] == tokens and mix_caps[-1] == tokens

    # Populate distinct gathered head activations, padded tail positions (-1), and image tokens (>= 129280).
    meta.positions.copy_(torch.tensor([idx if idx < 700 else -1 for idx in range(tokens)], dtype=torch.int64))
    meta.token_ids.copy_(
        torch.tensor([130000 if idx in (5, 140, 520, 680) else (idx % 1000) for idx in range(tokens)], dtype=torch.int64)
    )
    # Chunk shape from PreparedAllGather is [world, tokens, 6144 // world].
    gathered_chunk = (
        torch.arange(tokens, dtype=torch.float32)[None, :, None]
        .expand(world, tokens, 6144 // world)
        .to(torch.bfloat16)
        .contiguous()
    )
    full_source = (
        (torch.arange(tokens, dtype=torch.float32)[:, None, None] * 2.0)
        .expand(tokens, 4, 5120)
        .to(torch.bfloat16)
        .contiguous()
    )
    expected_mixed = full_engram.mix(full_source, full_engram.project_gathered([gathered_chunk]), meta)

    reassembled_a, reassembled_b = [], []
    for rank in range(world):
        projection_shapes.clear()
        mix_caps.clear()
        sharded_engram, _ = build_engram(
            tok=tokens,
            rank=rank,
            w_size=world,
            runtime=dma_rt,
            weights=weights_full,
            row_shard=(rank, half, shard_rows),
        )
        assert sharded_engram.row_shard is True
        s0 = rank * shard_rows
        expected_indices = list(range(s0, s0 + shard_rows)) + list(range(half + s0, half + s0 + shard_rows))
        assert sharded_engram.row_shard_indices.tolist() == expected_indices
        assert projection_shapes[-1][1] == (2 * shard_rows, 6144)
        assert mix_caps[-1] == 2 * shard_rows

        local_source = torch.cat(
            (full_source[s0 : s0 + shard_rows], full_source[half + s0 : half + s0 + shard_rows]), dim=0
        )
        local_proj = sharded_engram.project_gathered([gathered_chunk])
        assert local_proj.shape == (2 * shard_rows, 25600)
        local_mixed = sharded_engram.mix(local_source, local_proj, meta)
        assert local_mixed.shape == (2 * shard_rows, 4, 5120)
        reassembled_a.append(local_mixed[:shard_rows])
        reassembled_b.append(local_mixed[shard_rows:])

    reassembled = torch.cat((*reassembled_a, *reassembled_b), dim=0)
    assert torch.equal(reassembled, expected_mixed)

    # Ineligible Engram configurations leave row_shard disabled.
    small_engram, _ = build_engram(
        tok=256, rank=0, w_size=world, runtime=dma_rt, weights=weights_full, row_shard=(0, 128, 32)
    )
    assert small_engram.use_all_gather is False and small_engram.row_shard is None

    no_full_weights = {"projection": "proj_sharded_w", "weight": torch.ones(5120, dtype=torch.bfloat16)}
    sharded_proj_engram, _ = build_engram(
        tok=tokens, rank=0, w_size=world, runtime=dma_rt, weights=no_full_weights, row_shard=(0, half, shard_rows)
    )
    assert sharded_proj_engram.use_all_gather is False and sharded_proj_engram.row_shard is None

    for bad_row_shard in ((4, half, shard_rows), (-1, half, shard_rows), (0, 256, shard_rows), (0, half, 0)):
        with pytest.raises(ValueError, match="Engram row shard must partition both microbatch halves"):
            build_engram(
                tok=tokens, rank=0, w_size=world, runtime=dma_rt, weights=weights_full, row_shard=bad_row_shard
            )


@pytest.mark.parametrize("pipelined_attn_hc", [False, True])
def test_prepared_layer_attn_mxfp8_gather_unpack_and_wiring(pipelined_attn_hc):
    tokens, world, hidden, top_k = 1024, 4, 256, 6
    half = tokens // 2
    shard_tokens = half // world
    scale_cols_u8 = hidden // 32
    vals_bf16_cols = hidden // 2
    scale_bf16_cols = scale_cols_u8 // 2

    class PassCollective:
        def __init__(self, fn):
            self.fn = fn
            self.seen_shapes = []

        def start(self, source, *, dependencies=()):
            self.seen_shapes.append(tuple(source.shape))
            out = self.fn(source)
            return SimpleNamespace(source=source, output=out, dependencies=dependencies)

        def finish(self, pending, dependency):
            return pending.output

    layer = decoder.PreparedLayer.__new__(decoder.PreparedLayer)
    layer.row_shard_dma = True
    layer.pipelined_attn_hc = pipelined_attn_hc
    layer.pipelined_attn = "full"
    layer.defer_mlp_b = False
    layer.half_tokens = half
    layer.shard_tokens = shard_tokens
    layer.shard_rows_a = torch.arange(0, shard_tokens, dtype=torch.int64)
    layer.shard_rows_b = torch.arange(half, half + shard_tokens, dtype=torch.int64)
    layer.shard_offset_a = torch.tensor([0], dtype=torch.int32)
    layer.shard_offset_b = torch.tensor([half], dtype=torch.int32)
    layer.engram = None
    layer.has_image_mask = False
    layer.attn_quantizer_plan = object()
    layer.attn_local_shapes = None
    layer.attn_vals_bf16_cols = vals_bf16_cols
    layer.ffn_quantizer_plan = None
    layer.ffn_local_shapes = None
    layer.ffn_meta_scale_cols = 0

    # Construct distinct per-(row, col) FP8 payloads and UE8M0 scales for both microbatch halves.
    vals_a = ((torch.arange(half * hidden, dtype=torch.int64) % 113) + 1).to(torch.uint8).view(torch.float8_e4m3fn).reshape(half, hidden)
    srows_a = ((torch.arange(half * scale_cols_u8, dtype=torch.int64) % 241) + 5).to(torch.uint8).reshape(half, scale_cols_u8)
    packed_a = torch.cat((vals_a.view(torch.bfloat16), srows_a.view(torch.bfloat16)), dim=1)

    vals_b = ((torch.arange(half * hidden, dtype=torch.int64) % 97) + 3).to(torch.uint8).view(torch.float8_e4m3fn).reshape(half, hidden)
    srows_b = ((torch.arange(half * scale_cols_u8, dtype=torch.int64) % 229) + 11).to(torch.uint8).reshape(half, scale_cols_u8)
    packed_b = torch.cat((vals_b.view(torch.bfloat16), srows_b.view(torch.bfloat16)), dim=1)

    # Verify _unpack_attn_gathered reproduces exact FP8 values, scale_rows, and MMA tile layout.
    unpacked_vals, bases = layer._unpack_attn_gathered(packed_a)
    assert bases is not None
    b_vals, b_srows, b_smma = bases
    assert torch.equal(unpacked_vals.view(torch.uint8), vals_a.view(torch.uint8))
    assert torch.equal(b_vals.view(torch.uint8), vals_a.view(torch.uint8))
    assert torch.equal(b_srows, srows_a.view(1, half, scale_cols_u8))
    assert b_smma.shape == (1, half // 128, hidden // 128, 32, 4, 4)
    for m_tile in range(half // 128):
        for k_tile in range(hidden // 128):
            for r_outer in range(4):
                for c_inner in range(4):
                    rows_idx = m_tile * 128 + r_outer * 32 + torch.arange(32)
                    col_idx = k_tile * 4 + c_inner
                    assert torch.equal(b_smma[0, m_tile, k_tile, :, r_outer, c_inner], srows_a[rows_idx, col_idx])

    # Sensitivity check: mutating one packed scale byte changes the exact mapped scale_mma slot.
    mutated = packed_a.clone()
    mutated_srows = mutated[:, vals_bf16_cols:].contiguous().view(torch.uint8)
    mutated_srows[165, 5] ^= 0x01
    mutated[:, vals_bf16_cols:] = mutated_srows.view(torch.bfloat16)
    _, (_, _, mutated_smma) = layer._unpack_attn_gathered(mutated)
    assert not torch.equal(mutated_smma, b_smma)
    diff_mask = mutated_smma != b_smma
    assert int(diff_mask.sum().item()) == 1
    assert bool(diff_mask[0, 1, 1, 5, 1, 1].item())

    # Wire _run_row_shard_dma and verify both half gathers carry packed width and forward qkv_encoded_bases.
    packed_shards = [packed_a[:shard_tokens], packed_b[:shard_tokens]]
    shard_idx = [0]

    def fake_prepare_attn_shard(normalized):
        out = packed_shards[shard_idx[0]]
        shard_idx[0] += 1
        return out

    layer._prepare_attn_shard = fake_prepare_attn_shard
    layer.attention_hc = lambda *args, **kwargs: (
        torch.zeros((shard_tokens, 4, hidden), dtype=torch.bfloat16),
        torch.zeros((shard_tokens, 4), dtype=torch.float32),
        torch.zeros((shard_tokens, 4, 4), dtype=torch.float32),
        torch.zeros((shard_tokens, hidden), dtype=torch.bfloat16),
    )
    layer.cr_attn_gather_0 = PassCollective(lambda sh: packed_a)
    layer.cr_attn_gather_1 = PassCollective(lambda sh: packed_b)
    seen_run_half = []

    def fake_run_half(source_half, metadata, selection, *, indices, window_cache, half_idx, qkv_encoded_bases=None):
        seen_run_half.append((half_idx, source_half, qkv_encoded_bases))
        return torch.zeros((half, hidden), dtype=torch.bfloat16)

    layer.attention = SimpleNamespace(
        mode="prefill",
        init_window_cache=lambda metadata, dev: "wc",
        run_half=fake_run_half,
        commit=lambda lengths, records, metadata: None,
    )
    layer.cr_att_0 = PassCollective(lambda full: torch.zeros((shard_tokens, hidden), dtype=torch.bfloat16))
    layer.cr_att_1 = PassCollective(lambda full: torch.zeros((shard_tokens, hidden), dtype=torch.bfloat16))
    layer.ffn_hc = lambda att, res, post, comb, *, pre_mix, pre_out: (res, post, comb, att)
    layer.router_projection = lambda norm: torch.zeros((norm.shape[0], 384), dtype=torch.float32)
    layer.router = lambda logits, **kw: RouterScratch(
        torch.zeros((logits.shape[0], top_k), dtype=torch.float32),
        torch.zeros((logits.shape[0], top_k), dtype=torch.int32),
        torch.ones((logits.shape[0], top_k), dtype=torch.float32),
    )
    layer.cr_ffn_vals_0 = PassCollective(lambda sh: sh.repeat(world, 1))
    layer.cr_ffn_meta_0 = PassCollective(lambda sh: sh.repeat(world, 1))
    layer.cr_ffn_vals_1 = PassCollective(lambda sh: sh.repeat(world, 1))
    layer.cr_ffn_meta_1 = PassCollective(lambda sh: sh.repeat(world, 1))
    layer.experts = lambda vals, ids, weights, **kw: vals
    layer.shared = lambda vals, **kw: vals
    layer.cr_mlp_0 = PassCollective(lambda full: full[:shard_tokens].clone())
    layer.cr_mlp_1 = PassCollective(lambda full: full[:shard_tokens].clone())
    layer.post = lambda mlp_sh, res, post, comb: res

    metadata = Metadata.allocate(tokens, 1, 10, "cpu")
    metadata.live_counts[0] = tokens
    metadata.lengths[0] = tokens
    pm_a = torch.zeros((shard_tokens, 4), dtype=torch.float32)
    pm_b = torch.zeros((shard_tokens, 4), dtype=torch.float32)
    if pipelined_attn_hc:
        zeros_res = torch.zeros((shard_tokens, 4, hidden), dtype=torch.bfloat16)
        zeros_post = torch.zeros((shard_tokens, 4), dtype=torch.float32)
        zeros_comb = torch.zeros((shard_tokens, 4, 4), dtype=torch.float32)
        mlp_a = torch.zeros((shard_tokens, hidden), dtype=torch.bfloat16)
        p_mlp_b = SimpleNamespace(output=torch.zeros((shard_tokens, hidden), dtype=torch.bfloat16))
        incoming = (mlp_a, zeros_res, zeros_post, zeros_comb, p_mlp_b, zeros_res, zeros_post, zeros_comb)
    else:
        incoming = (
            torch.zeros((shard_tokens, 4, hidden), dtype=torch.bfloat16),
            torch.zeros((shard_tokens, 4, hidden), dtype=torch.bfloat16),
        )

    layer._run_row_shard_dma(incoming, (pm_a, pm_b), metadata, "indices", ("sel", None, None))
    assert layer.cr_attn_gather_0.seen_shapes == [(shard_tokens, vals_bf16_cols + scale_bf16_cols)]
    assert layer.cr_attn_gather_1.seen_shapes == [(shard_tokens, vals_bf16_cols + scale_bf16_cols)]
    assert len(seen_run_half) == 2
    assert seen_run_half[0][0] == 0 and seen_run_half[1][0] == 1
    assert torch.equal(seen_run_half[0][1].view(torch.uint8), vals_a.view(torch.uint8))
    assert torch.equal(seen_run_half[1][1].view(torch.uint8), vals_b.view(torch.uint8))
    assert torch.equal(seen_run_half[0][2][1], srows_a.view(1, half, scale_cols_u8))
    assert torch.equal(seen_run_half[1][2][1], srows_b.view(1, half, scale_cols_u8))


def test_prepared_attention_preencoded_qkv_contract():
    # Reject preencoded qkv_a outside full pipelined attention.
    for bad_mode in (None, "mla_wo"):
        with pytest.raises(ValueError, match="preencoded qkv_a requires full pipelined attention"):
            ds_attention.PreparedAttention(
                0, {}, None, None, None, None, None, mode="prefill", world=4,
                session=None, name="attn", workspace=None, pipelined_mode=bad_mode,
                qkv_encoded=object(),
            )

    # Preencoded qkv_a (quantizer_plan is None) must not allocate uninitialized encoded_bases in run_half.
    seen_bases = []

    class FakeQkvProjection:
        def __init__(self, *, quantizer_plan):
            self.quantizer_plan = quantizer_plan
            self.encoded_layout = (((8, 16), torch.float8_e4m3fn),)

        def __call__(self, source, *, encoded_bases=None):
            seen_bases.append(encoded_bases)
            if self.quantizer_plan is None and encoded_bases is None:
                raise ValueError("functional preencoded projection requires encoded_bases")
            raise StopIteration("reached qkv call")

    attn = ds_attention.PreparedAttention.__new__(ds_attention.PreparedAttention)
    attn.half_tokens = 8

    attn.projections = {"qkv_a": FakeQkvProjection(quantizer_plan=None)}
    with pytest.raises(ValueError, match="functional preencoded projection requires encoded_bases"):
        attn.run_half(torch.zeros((8, 16), dtype=torch.bfloat16), None, None,
                      indices=None, window_cache=None, half_idx=0, qkv_encoded_bases=None)
    assert seen_bases == [None]

    supplied = (torch.zeros((8, 16), dtype=torch.float8_e4m3fn),)
    seen_bases.clear()
    with pytest.raises(StopIteration, match="reached qkv call"):
        attn.run_half(torch.zeros((8, 16), dtype=torch.bfloat16), None, None,
                      indices=None, window_cache=None, half_idx=0, qkv_encoded_bases=supplied)
    assert seen_bases == [supplied]

    # Projection-owned quantizer still allocates fresh encoded_bases when caller passes None.
    seen_bases.clear()
    attn.projections = {"qkv_a": FakeQkvProjection(quantizer_plan=object())}
    with pytest.raises(StopIteration, match="reached qkv call"):
        attn.run_half(torch.zeros((8, 16), dtype=torch.bfloat16), None, None,
                      indices=None, window_cache=None, half_idx=0)
    assert len(seen_bases) == 1 and seen_bases[0] is not None and seen_bases[0][0].shape == (8, 16)


@pytest.mark.skipif(
    not _RUN_CUDA_TESTS or not torch.cuda.is_available(),
    reason="requires CUDA GPU and FLARE_RUN_CUDA_TESTS=1",
)
def test_mxfp8_attn_shard_quantize_gather_matches_full_quantize_bit_exact_cuda():
    from b12x.quantization import mxfp8

    device = torch.device("cuda")
    half, world, hidden = 4096, 4, 5120
    st = half // world

    gen = torch.Generator(device=device).manual_seed(42)
    full_x = torch.randn((half, hidden), dtype=torch.bfloat16, device=device, generator=gen)
    # Exercise zero rows and sub-min_amax rows (below 1e-4) alongside normal activations.
    full_x[0].zero_()
    full_x[128:256].mul_(1e-5)
    full_x[1024:1152].mul_(50.0)

    # Reference: post-gather MXFP8 quantization with min_amax=1e-4 (matching PreparedAttention qkv_a).
    full_shapes = mxfp8.dense_rows_storage_shapes(half, hidden)
    ref_vals = torch.empty(full_shapes[0], dtype=torch.float8_e4m3fn, device=device)
    ref_srows = torch.empty(full_shapes[1], dtype=torch.uint8, device=device)
    ref_smma = torch.empty(full_shapes[2], dtype=torch.uint8, device=device)
    ref_encoded = mxfp8.dense_rows_from_bases(ref_vals, ref_srows, ref_smma)
    full_plan = mxfp8.plan(mxfp8.query_from_call(
        full_x, ref_encoded.values, ref_encoded.scale_rows, ref_encoded.scale_mma, min_amax=1e-4))
    mxfp8.quantize_dense_rows(full_x, ref_vals, ref_srows, ref_smma, plan=full_plan)

    # Treatment: per-shard quantization + pack -> gather -> unpack via PreparedLayer helpers.
    local_shapes = mxfp8.dense_rows_storage_shapes(st, hidden)
    local_vals_ex = torch.empty(local_shapes[0], dtype=torch.float8_e4m3fn, device=device)
    local_srows_ex = torch.empty(local_shapes[1], dtype=torch.uint8, device=device)
    local_smma_ex = torch.empty(local_shapes[2], dtype=torch.uint8, device=device)
    local_encoded = mxfp8.dense_rows_from_bases(local_vals_ex, local_srows_ex, local_smma_ex)
    shard_plan = mxfp8.plan(mxfp8.query_from_call(
        full_x[:st], local_encoded.values, local_encoded.scale_rows, local_encoded.scale_mma, min_amax=1e-4))

    layer = decoder.PreparedLayer.__new__(decoder.PreparedLayer)
    layer.attn_quantizer_plan = shard_plan
    layer.attn_local_shapes = local_shapes
    layer.attn_vals_bf16_cols = hidden // 2

    packed_shards = [layer._prepare_attn_shard(full_x[r * st : (r + 1) * st]) for r in range(world)]
    assert all(p.shape == (st, hidden // 2 + hidden // 64) and p.dtype == torch.bfloat16 for p in packed_shards)
    packed_full = torch.cat(packed_shards, dim=0)
    got_vals, bases = layer._unpack_attn_gathered(packed_full)
    assert bases is not None
    b_vals, got_srows, got_smma = bases

    assert torch.equal(got_vals.view(torch.uint8), ref_vals.view(torch.uint8))
    assert torch.equal(b_vals.view(torch.uint8), ref_vals.view(torch.uint8))
    assert torch.equal(got_srows, ref_srows)
    assert torch.equal(got_smma, ref_smma)
    got_encoded = mxfp8.dense_rows_from_bases(*bases)
    assert torch.equal(got_encoded.scale_mma.view(torch.uint8), ref_encoded.scale_mma.view(torch.uint8))

    # Sensitivity check: mismatched min_amax=0.0 must differ on the sub-1e-4 rows.
    wrong_plan = mxfp8.plan(mxfp8.query_from_call(
        full_x[:st], local_encoded.values, local_encoded.scale_rows, local_encoded.scale_mma, min_amax=0.0))
    layer.attn_quantizer_plan = wrong_plan
    wrong_full = torch.cat([layer._prepare_attn_shard(full_x[r * st : (r + 1) * st]) for r in range(world)], dim=0)
    _, (_, wrong_srows, _) = layer._unpack_attn_gathered(wrong_full)
    assert not torch.equal(wrong_srows, ref_srows)
