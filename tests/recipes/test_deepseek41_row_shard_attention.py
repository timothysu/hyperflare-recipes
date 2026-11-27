# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek V4.1 row-sharded encoder attention, mla_wo compression/indexing, Engram overlap, and boundary parity."""

from dataclasses import dataclass, replace
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from b12x.comm import pcie
from hyperflare_recipes.components.moe import RouterScratch
from hyperflare_recipes.components.paged_batch import Metadata
from hyperflare_recipes.components.parallel import TensorParallelShard
from hyperflare_recipes.recipes.deepseek41 import (
    attention as ds_attention,
    bootstrap as ds_bootstrap,
    config as ds_config,
    decoder,
    engram as ds_engram,
    recipe as ds_recipe,
    weights as ds_weights,
)
from hyperflare_recipes.recipes.deepseek41.bootstrap import PreparationConfig
from hyperflare_recipes.recipes.deepseek41.decoder_prefill import PreparedDecoderSuffix
from hyperflare_recipes.recipes.deepseek41.pool import ContinuationPool
from ignition.planning import Gpu, Plan, UsageError
from ignition.preparation.rebuild_manifest import (
    declare_weights,
    declared_views,
    weight_payload_declaration,
)
from ignition.preparation.scratch import ScratchArena


_RUN_CUDA_TESTS = os.environ.get("FLARE_RUN_CUDA_TESTS") == "1"
_SNAPSHOT_ENV = os.environ.get("FLARE_DEEPSEEK41_SNAPSHOT", "")

EXPECTED_FULL_ATTN_ENCODER_LAYERS = (
    0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19,
)
EXPECTED_ROW_SHARD_ENCODER_LAYERS = tuple(range(20))
EXPECTED_LAYER_BYTES = 124_780_800
EXPECTED_TOTAL_BYTES = 20 * EXPECTED_LAYER_BYTES  # 2,495,616,000 B


def _qualified_model_config(*, overrides: dict | None = None) -> ds_config.ModelConfig:
    text = {
        **ds_config.TEXT_GEOMETRY,
        **ds_config.QUALIFIED_ENCODER_SCHEDULE,
        "rms_norm_eps": 1e-20,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 20,
        "rope_theta": 10000.0,
        "compress_rope_theta": 10000.0,
        "rope_scaling": {
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40.0,
        },
    }
    if overrides:
        text.update(overrides)
    return ds_config.ModelConfig(text, {})


def _sm120_gpus(count: int, capability: tuple[int, int] = (12, 0)) -> tuple[Gpu, ...]:
    return tuple(
        Gpu(i, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{i}", 96 << 30, 90 << 30, capability)
        for i in range(count)
    )


def test_replicated_attention_layer_schedule_and_exact_byte_accounting(monkeypatch):
    cfg = _qualified_model_config()
    assert cfg.decoder_start == 20
    assert ds_attention.INDEXER_QUERY_ROWS == 512

    full_layers = ds_config.full_attention_encoder_layers(cfg)
    assert full_layers == EXPECTED_FULL_ATTN_ENCODER_LAYERS
    assert len(full_layers) == 17

    encoder_layers = ds_config.row_shard_attention_encoder_layers(cfg)
    assert encoder_layers == EXPECTED_ROW_SHARD_ENCODER_LAYERS
    assert len(encoder_layers) == 20
    for mla_wo_layer in (2, 8, 14):
        assert mla_wo_layer in encoder_layers
        assert mla_wo_layer not in full_layers
    for suffix_layer in range(20, 40):
        assert suffix_layer not in encoder_layers
    cfg.require_planned_attention_schedule()

    # Exact per-layer breakdown: FP8 values + BOTH MXFP8 scale layouts (scale_rows + scale_mma) + FP32 sinks.
    q_b_vals = 64 * 512 * 1280
    wo_a_vals = 8 * 1024 * 4096
    wo_b_vals = 5120 * (8 * 1024)
    all_vals = q_b_vals + wo_a_vals + wo_b_vals
    both_scales = 2 * (all_vals // 32)
    one_scale_only = all_vals // 32
    fp32_sinks = 64 * 4

    layer_bytes = ds_config.replicated_attention_layer_bytes()
    assert layer_bytes == EXPECTED_LAYER_BYTES
    assert layer_bytes == all_vals + both_scales + fp32_sinks
    # Sensitivity: omitting one scale layout, FP32 sinks, or the 3 mla_wo layers must disagree.
    assert layer_bytes != all_vals + one_scale_only + fp32_sinks
    assert layer_bytes != all_vals + both_scales
    assert EXPECTED_TOTAL_BYTES == 2_495_616_000
    assert EXPECTED_TOTAL_BYTES - 17 * EXPECTED_LAYER_BYTES == 374_342_400

    assert ds_config.replicated_attention_bytes(4) == EXPECTED_TOTAL_BYTES
    assert ds_config.replicated_attention_bytes(4, prompt_logprobs=True) == EXPECTED_TOTAL_BYTES
    assert ds_config.replicated_attention_bytes(4, mtp_only=True) == EXPECTED_TOTAL_BYTES

    # Zero charge on single-rank, when no encoder boundary exists, or when candidate selection precedes boundary.
    assert ds_config.replicated_attention_bytes(1) == 0
    no_encoder_cfg = _qualified_model_config(overrides={
        "compress_ratios": [1] * 40,
        "kv_source_layer_ids": [0],
        "index_source_layer_ids": [0, 20],
        "candidate_source_layer_id": 20,
    })
    assert no_encoder_cfg.decoder_start is None
    assert ds_config.row_shard_attention_encoder_layers(no_encoder_cfg) == ()
    assert ds_config.replicated_attention_bytes(4, config=no_encoder_cfg) == 0

    early_candidate_cfg = _qualified_model_config(overrides={
        "candidate_source_layer_id": 14,
        "index_source_layer_ids": [2, 8, 14],
    })
    assert ds_config.row_shard_attention_encoder_layers(early_candidate_cfg) == ()
    assert ds_config.replicated_attention_bytes(4, config=early_candidate_cfg) == 0
    with pytest.raises(ValueError, match="qualified encoder layer schedule"):
        early_candidate_cfg.require_planned_attention_schedule()

    # Sensitivity to changing source/indexer schedule or decoder boundary.
    sixteen_layer_cfg = _qualified_model_config(overrides={
        "index_source_layer_ids": [2, 5, 8, 14, 20, 24, 28, 32, 36],
    })
    assert len(ds_config.full_attention_encoder_layers(sixteen_layer_cfg)) == 16
    with pytest.raises(ValueError, match="qualified encoder layer schedule"):
        sixteen_layer_cfg.require_planned_attention_schedule()

    shifted_boundary_cfg = _qualified_model_config(overrides={
        "compress_ratios": [0, 0] + [2] * 16 + [1] * 22,
        "kv_source_layer_ids": [2, 8, 14, 18],
        "index_source_layer_ids": [2, 8, 14, 18, 24, 28, 32, 36],
        "candidate_source_layer_id": 18,
    })
    assert ds_config.replicated_attention_bytes(4, config=shifted_boundary_cfg) == 18 * EXPECTED_LAYER_BYTES
    with pytest.raises(ValueError, match="qualified encoder layer schedule"):
        shifted_boundary_cfg.require_planned_attention_schedule()

    wider_sink_text = {**ds_config.TEXT_GEOMETRY, "num_attention_heads": 128, "o_groups": 16}
    assert ds_config.replicated_attention_layer_bytes(wider_sink_text) > EXPECTED_LAYER_BYTES

    # PreparationConfig validation and safe plan extraction.
    from dataclasses import replace

    assert PreparationConfig().row_shard_attention is False
    valid_cfg = PreparationConfig(
        parameters=replace(ds_recipe.RECIPE.parameters, row_shard_prefill=True, row_shard_attention=True),
    )
    assert valid_cfg.row_shard_attention is True
    for bad in (1, 0, "true", None):
        with pytest.raises(TypeError, match="row_shard_attention must be a boolean"):
            replace(ds_recipe.RECIPE.parameters, row_shard_prefill=True, row_shard_attention=bad)
    with pytest.raises(ValueError, match="row_shard_attention requires row_shard_prefill"):
        PreparationConfig(
            parameters=replace(ds_recipe.RECIPE.parameters, row_shard_prefill=False, row_shard_attention=True),
        )

    captured = {}

    def fake_prepare_group(snapshot, config, prepare_rank, **kwargs):
        captured["config"] = config
        return "prepared"

    import ignition.preparation.rank_group as rank_group

    monkeypatch.setattr(rank_group, "prepare_group", fake_prepare_group)
    base_params = replace(ds_recipe.RECIPE.parameters, row_shard_prefill=True, row_shard_attention=False)
    base_plan = Plan(
        recipe=ds_recipe.RECIPE,
        checkpoint=ds_recipe.RECIPE.checkpoints[0],
        gpus=_sm120_gpus(4),
        parameters=base_params,
        cache_target=8192,
        cache_states=256,
        pages=1024,
        weights_bytes=1,
        graph_bytes=1,
    )
    ds_recipe.prepare("/tmp/snap", base_plan, compile_forwards=True, kernel_cache=None, progress=None)
    assert captured["config"].row_shard_prefill is True
    assert captured["config"].row_shard_attention is False

    rsa_params = replace(ds_recipe.RECIPE.parameters, row_shard_prefill=True, row_shard_attention=True)
    rsa_plan = Plan(
        recipe=ds_recipe.RECIPE,
        checkpoint=ds_recipe.RECIPE.checkpoints[0],
        gpus=_sm120_gpus(4),
        parameters=rsa_params,
        cache_target=8192,
        cache_states=256,
        pages=1024,
        weights_bytes=1,
        graph_bytes=1,
    )
    ds_recipe.prepare("/tmp/snap", rsa_plan, compile_forwards=True, kernel_cache=None, progress=None)
    assert captured["config"].row_shard_attention is True
    with pytest.raises(UsageError, match="--row-shard-attention requires --row-shard-prefill"):
        Plan(
            recipe=ds_recipe.RECIPE,
            checkpoint=ds_recipe.RECIPE.checkpoints[0],
            gpus=_sm120_gpus(4),
            parameters=replace(ds_recipe.RECIPE.parameters, row_shard_prefill=False, row_shard_attention=True),
            cache_target=8192,
            cache_states=256,
            pages=1024,
            weights_bytes=1,
            graph_bytes=1,
        )
    base_wb = ds_recipe.RECIPE.weights_bytes(ds_recipe.RECIPE.checkpoints[0], 4, base_params)
    rsa_wb = ds_recipe.RECIPE.weights_bytes(ds_recipe.RECIPE.checkpoints[0], 4, rsa_params)
    assert rsa_wb - base_wb == EXPECTED_TOTAL_BYTES


def test_model_weights_replicated_attention_loading_and_misc_component_routing(monkeypatch):
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight
    from b12x.quantization import mxfp8

    @dataclass(frozen=True)
    class FakePackedWo:
        wo_a: mxfp8.MXFP8Rows
        wo_b: mxfp8.MXFP8Rows
        groups: int
        group_width: int
        rank: int
        hidden: int

    def make_mxfp8_rows(rows: int, cols: int) -> mxfp8.MXFP8Rows:
        r, c = max(1, rows // 64), max(32, cols // 64)
        vals = torch.ones((r, c), dtype=torch.float8_e4m3fn)
        srows = torch.ones((1, r, max(1, c // 32)), dtype=torch.uint8)
        smma = torch.ones((1, 1, 1, 1, 1, max(1, (r * c) // 32)), dtype=torch.uint8)
        return mxfp8.MXFP8Rows(vals, srows, smma)

    def make_linear(rows: int, cols: int) -> MXFP8LinearWeight:
        packed = make_mxfp8_rows(rows, cols)
        obj = MXFP8LinearWeight.__new__(MXFP8LinearWeight)
        for k, v in (
            ("weight", packed),
            ("out_features", rows),
            ("in_features", cols),
            ("padded_in_features", cols),
        ):
            object.__setattr__(obj, k, v)
        return obj

    def make_wo(groups: int) -> FakePackedWo:
        return FakePackedWo(
            wo_a=make_mxfp8_rows(groups * 1024, 4096),
            wo_b=make_mxfp8_rows(5120, groups * 1024),
            groups=groups,
            group_width=4096,
            rank=1024,
            hidden=5120,
        )

    cfg = _qualified_model_config()
    replicated_calls = []
    orig_load_replicated = ds_weights.DeepSeekWeightLoader.load_replicated_attention

    class FakeLoader:
        def __init__(self, manifest, device, *, shard, **kwargs):
            self.config = cfg
            self.shard = shard
            self._sources = {}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def load_tensors(self, names):
            return (torch.ones((16, 64), dtype=torch.bfloat16), torch.ones(64, dtype=torch.bfloat16))

        def load_padded_tensor(self, name, *, rows, temporary=False):
            count = rows[1] - rows[0]
            if name.endswith(".attn_sink"):
                return torch.ones((count,), dtype=torch.float32)
            return torch.ones((max(1, count // 1024), 64), dtype=torch.bfloat16)

        def load_linear_group(self, names, *, row_ranges=None, pad=False):
            if row_ranges is None:
                return make_linear(1792, 5120)
            r0, r1 = row_ranges[0]
            return make_linear(r1 - r0, 1280)

        def load_attention_output(self, prefix, groups_range, *, pad=False):
            groups = (groups_range[1] - groups_range[0]) // 1024
            return make_wo(groups)

        def load_layer(self, layer, *, draft=False):
            first, last = self.shard.padded_partition(8)
            attn = {
                "qkv_a": self.load_linear_group((f"layers.{layer}.attn.wq_a",)),
                "q_b": self.load_linear_group((f"layers.{layer}.attn.wq_b",), row_ranges=((first * 4096, last * 4096),)),
                "output": self.load_attention_output(f"layers.{layer}.attn", (first * 1024, last * 1024)),
                "q_norm": torch.ones(1280, dtype=torch.bfloat16),
                "kv_norm": torch.ones(512, dtype=torch.bfloat16),
                "sink": self.load_padded_tensor(f"layers.{layer}.attn.attn_sink", rows=(first * 8, last * 8)),
            }
            shared = ds_weights.MLPWeights(make_linear(1152, 5120), make_linear(5120, 576))
            experts = ds_weights.ExpertWeights(
                router=torch.ones((384, 64), dtype=torch.bfloat16),
                correction=torch.zeros(384, dtype=torch.float32),
                experts=None,
                shared=shared,
            )
            residual = {k: torch.ones(4, dtype=torch.float32) for k in ("hc_attn_base", "hc_ffn_base")}
            return ds_weights.LayerWeights(attn, experts, residual, None)

        def load_replicated_attention(self, layer):
            replicated_calls.append(layer)
            return orig_load_replicated(self, layer)

    monkeypatch.setattr(ds_weights.ModelConfig, "load", classmethod(lambda cls, path: cfg))
    monkeypatch.setattr(ds_weights, "DeepSeekWeightLoader", FakeLoader)
    monkeypatch.setattr(ds_weights, "allocated_tensor_bytes", lambda dev: 0)
    manifest = SimpleNamespace(metadata_path=Path("/tmp/fake"))

    # 1. row_shard_attention=False -> replicated_attention is None.
    w_off = ds_weights.ModelWeights.load(
        manifest, "cpu", shard=TensorParallelShard(1, 4), dspark=False, row_shard_attention=False
    )
    assert w_off.replicated_attention is None
    assert replicated_calls == []

    # 2. world_size == 1 -> replicated_attention is None even if row_shard_attention=True.
    w_tp1 = ds_weights.ModelWeights.load(
        manifest, "cpu", shard=TensorParallelShard(0, 1), dspark=False, row_shard_attention=True
    )
    assert w_tp1.replicated_attention is None
    assert replicated_calls == []

    # 3. row_shard_attention=True at TP4 -> loads all 20 encoder layers (0..19), keeping decode copies intact.
    w_on = ds_weights.ModelWeights.load(
        manifest, "cpu", shard=TensorParallelShard(1, 4), dspark=False, row_shard_attention=True
    )
    assert tuple(replicated_calls) == EXPECTED_ROW_SHARD_ENCODER_LAYERS
    assert w_on.replicated_attention is not None
    assert tuple(w_on.replicated_attention) == EXPECTED_ROW_SHARD_ENCODER_LAYERS
    for layer_idx in EXPECTED_ROW_SHARD_ENCODER_LAYERS:
        rep = w_on.replicated_attention[layer_idx]
        assert set(rep) == {"q_b_full", "output_full", "sink_full"}
        assert rep["q_b_full"].out_features == 64 * 512
        assert rep["output_full"].groups == 8
        assert rep["sink_full"].shape == (64,) and rep["sink_full"].dtype == torch.float32
        part = w_on.layers[layer_idx].attention
        assert part["q_b"].out_features == 16 * 512
        assert part["output"].groups == 2
        assert part["sink"].shape == (16,)

    # 4. Verify declare_weights + declared_views routes replicated_attention to 'misc' while
    #    keeping all 40 layer-000..039 views untouched!
    decl_off = declare_weights(w_off, components=("embedding", "norm", "head", "vision", "draft"))
    decl_on = declare_weights(
        w_on, components=("embedding", "norm", "head", "vision", "draft", "replicated_attention")
    )
    views_off = weight_payload_declaration(decl_off)["views"]
    views_on = weight_payload_declaration(decl_on)["views"]

    layer_views_off = [
        (v["name"], v["group"], v["storage"], v["element_bytes"], v["shape"], v["stride"])
        for v in views_off if v["group"].startswith("layer-")
    ]
    layer_views_on = [
        (v["name"], v["group"], v["storage"], v["element_bytes"], v["shape"], v["stride"])
        for v in views_on if v["group"].startswith("layer-")
    ]
    assert len(layer_views_off) > 0
    assert layer_views_on == layer_views_off
    assert {v[1] for v in layer_views_on} == {f"layer-{i:03d}" for i in range(40)}

    rep_views = [v for v in views_on if "replicated_attention" in v["name"]]
    assert len(rep_views) == 20 * 10
    assert all(v["group"] == "misc" and v["storage"] == "stable" for v in rep_views)
    assert not any("replicated_attention" in name for name, *_ in declared_views(decl_off))


def test_prepared_attention_shard_validation_and_rank_invariant_execution(monkeypatch):
    cfg = _qualified_model_config()
    pool = ContinuationPool.allocate(cfg, 4, 4, "cpu")
    rotary = ds_attention.Rotary(cfg, "cpu")

    tokens, world = 1024, 4
    half = tokens // 2
    st = half // world
    meta = Metadata.allocate(tokens, 1, 16, "cpu", max_query_tokens=tokens)
    scr = ds_attention.AttentionScratch.allocate(tokens, world, "cpu")
    src = torch.zeros((tokens, 5120), dtype=torch.bfloat16)

    base_weights = {
        "qkv_a": SimpleNamespace(out_features=1792),
        "q_b": SimpleNamespace(out_features=16 * 512),
        "q_norm": torch.ones(1280, dtype=torch.bfloat16),
        "kv_norm": torch.ones(512, dtype=torch.bfloat16),
        "output": SimpleNamespace(groups=2),
        "sink": torch.zeros(16, dtype=torch.float32),
    }
    source_weights = {
        **base_weights,
        "compressor": {
            "norm": torch.ones(512, dtype=torch.float32),
            "values": torch.ones((512, 5120), dtype=torch.bfloat16),
            "gates": torch.ones((512, 5120), dtype=torch.bfloat16),
        },
        "index_k": torch.ones((128, 512), dtype=torch.bfloat16),
        "index_norm": torch.ones(128, dtype=torch.bfloat16),
        "index_q": SimpleNamespace(out_features=32 * 128),
        "index_weights": torch.ones((32, 5120), dtype=torch.bfloat16),
    }
    valid_rep = {
        "q_b_full": SimpleNamespace(out_features=64 * 512),
        "output_full": SimpleNamespace(groups=8),
        "sink_full": torch.zeros(64, dtype=torch.float32),
    }

    # Validation failure matrix.
    with pytest.raises(ValueError, match="requires both shard_tokens and replicated_weights"):
        ds_attention.PreparedAttention(
            3, base_weights, src, scr, meta, pool, rotary, mode="prefill", world=4,
            session=None, name="bad", workspace=None, pipelined_mode="full",
            shard_tokens=st, replicated_weights=None,
        )
    with pytest.raises(ValueError, match="requires both shard_tokens and replicated_weights"):
        ds_attention.PreparedAttention(
            3, base_weights, src, scr, meta, pool, rotary, mode="prefill", world=4,
            session=None, name="bad", workspace=None, pipelined_mode="full",
            shard_tokens=None, replicated_weights=valid_rep,
        )
    with pytest.raises(ValueError, match="requires prefill mode with pipelined attention"):
        ds_attention.PreparedAttention(
            3, base_weights, src, scr, meta, pool, rotary, mode="prefill", world=4,
            session=None, name="bad", workspace=None, pipelined_mode=None,
            shard_tokens=st, replicated_weights=valid_rep,
        )
    with pytest.raises(ValueError, match="mla_wo pipelined attention requires a source or indexer layer"):
        ds_attention.PreparedAttention(
            3, base_weights, src, scr, meta, pool, rotary, mode="prefill", world=4,
            session=None, name="bad", workspace=None, pipelined_mode="mla_wo",
            shard_tokens=st, replicated_weights=valid_rep,
        )
    with pytest.raises(ValueError, match="before candidate selection"):
        ds_attention.PreparedAttention(
            20, source_weights, src, scr, meta, pool, rotary, mode="prefill", world=4,
            session=None, name="bad", workspace=None, pipelined_mode="mla_wo",
            shard_tokens=st, replicated_weights=valid_rep,
        )
    with pytest.raises(ValueError, match="requires prefill mode with pipelined attention"):
        ds_attention.PreparedAttention(
            3, base_weights, src[:8], scr, Metadata.allocate(8, 8, 2, "cpu", max_query_tokens=1),
            pool, rotary, mode="decode", world=4, session=None, name="bad", workspace=None,
            pipelined_mode="full", shard_tokens=2, replicated_weights=valid_rep,
        )
    with pytest.raises(ValueError, match="rejects preencoded qkv_a"):
        ds_attention.PreparedAttention(
            3, base_weights, src, scr, meta, pool, rotary, mode="prefill", world=4,
            session=None, name="bad", workspace=None, pipelined_mode="full",
            qkv_encoded=object(), shard_tokens=st, replicated_weights=valid_rep,
        )
    for bad_st in (0, -128, 192, "128", True):
        with pytest.raises(ValueError, match="shard_tokens must be a positive divisor"):
            ds_attention.PreparedAttention(
                3, base_weights, src, scr, meta, pool, rotary, mode="prefill", world=4,
                session=None, name="bad", workspace=None, pipelined_mode="full",
                shard_tokens=bad_st, replicated_weights=valid_rep,
            )
    for bad_rep, pattern in (
        ({}, "requires q_b_full, output_full, and sink_full"),
        ({**valid_rep, "q_b_full": None}, "requires q_b_full, output_full, and sink_full"),
        ({**valid_rep, "q_b_full": SimpleNamespace(out_features=16 * 512)}, "must span all heads and output groups"),
        ({**valid_rep, "output_full": SimpleNamespace(groups=2)}, "must span all heads and output groups"),
        ({**valid_rep, "sink_full": torch.zeros(16, dtype=torch.float32)}, "must span all heads and output groups"),
        ({**valid_rep, "sink_full": torch.zeros(64, dtype=torch.bfloat16)}, "must span all heads and output groups"),
    ):
        with pytest.raises(ValueError, match=pattern):
            ds_attention.PreparedAttention(
                3, base_weights, src, scr, meta, pool, rotary, mode="prefill", world=4,
                session=None, name="bad", workspace=None, pipelined_mode="full",
                shard_tokens=st, replicated_weights=bad_rep,
            )

    declared = {}

    class FakeProj:
        def __init__(self, values, weight, out, **kwargs):
            self.name = kwargs["name"]
            self.weight = weight
            self.plan = object()
            self.quantizer_plan = None
            self.encoded = kwargs.get("encoded")
            self.encoded_layout = None
            declared[self.name] = (tuple(values.shape), tuple(out.shape), weight)

        def functional(self, **kwargs):
            return self

        def __call__(self, x, *, encoded_bases=None):
            if self.name.endswith(".qkv_a"):
                out = torch.zeros((x.shape[0], 1792), dtype=torch.bfloat16)
                out[:, 0] = x[:, 0]
                out[:, 1280] = x[:, 0]
                return out
            if self.name.endswith(".index_query"):
                out = torch.zeros((x.shape[0], 32, 128), dtype=torch.bfloat16)
                out[:, :, 0] = x[:, :1]
                return out.view(x.shape[0], -1)
            out = torch.zeros((x.shape[0], 64, 512), dtype=torch.bfloat16)
            out[:, :, 0] = x[:, :1]
            return out.view(x.shape[0], -1)

    class FakeUnquantProj:
        def __init__(self, values, weight, out, **kwargs):
            self.name = kwargs["name"]
            self.out_dtype = out.dtype
            self.out_cols = out.shape[1]
            self.plan = object()
            declared[self.name] = (tuple(values.shape), tuple(out.shape))

        def functional(self, **kwargs):
            return self

        def __call__(self, x):
            out = torch.zeros((x.shape[0], self.out_cols), dtype=self.out_dtype)
            out[:, 0] = x[:, 0].to(self.out_dtype)
            return out

    class FakeNorm:
        def __init__(self, values, weight, out, **kwargs):
            self.plan = object()
            declared[kwargs["name"]] = (tuple(values.shape), tuple(out.shape))

        def functional(self, **kwargs):
            return self

        def __call__(self, x):
            return x.clone()

    writer_calls = []

    class FakeWindowWriter:
        def __init__(self, values, cache, slots, **kwargs):
            self.plan = object()
            declared[kwargs["name"]] = (tuple(values.shape), tuple(slots.shape))

        def functional(self, **kwargs):
            return self

        def __call__(self, values, slots, *, cache=None, **kwargs):
            if cache is not None:
                writer_calls.append((values.clone(), slots.clone()))
                for idx, slot in enumerate(slots.tolist()):
                    if slot >= 0:
                        cache[0, slot] = int(values[idx, 0].item()) % 251

    selector_decls = {}

    class FakeSelector:
        def __init__(self, query, weights, cache, pages, lengths, active, output, **kwargs):
            self.plan = object()
            selector_decls[kwargs["name"]] = (tuple(query.shape), tuple(output.shape))

        def write_keys(self, keys, slots):
            pass

        def functional(self, **kwargs):
            return self

        def __call__(self, iq, iw, pt, ln, active):
            out = torch.full((iq.shape[0], 512), -1, dtype=torch.int32)
            out[:, 0] = iq[:, 0, 0].to(torch.int32) + pt[:, 0] + ln
            return out

    mla_calls = []

    class FakeMLA:
        def __init__(self, query, window, win_idx, win_len, sink, attended, **kwargs):
            self.plan = object()
            declared[kwargs["name"]] = (
                tuple(query.shape), tuple(attended.shape), tuple(sink.shape),
                None if "indexed_indices" not in kwargs else tuple(kwargs["indexed_indices"].shape),
            )

        def functional(self, **kwargs):
            return self

        def __call__(self, query, win_idx, win_len, *, indexed_indices=None, indexed_lengths=None,
                     page_table=None, window=None):
            mla_calls.append({
                "win_idx": win_idx.clone(),
                "win_len": win_len.clone(),
                "indexed_indices": None if indexed_indices is None else indexed_indices.clone(),
                "indexed_lengths": None if indexed_lengths is None else indexed_lengths.clone(),
                "page_table": None if page_table is None else page_table.clone(),
                "cache_byte_last_half": int(window[0, win_idx[:, -1].clamp_min(0)].float().sum().item()),
            })
            out = torch.zeros((query.shape[0], 64, 512), dtype=torch.bfloat16)
            extra = torch.zeros(query.shape[0], dtype=torch.float32)
            if indexed_indices is not None:
                extra = indexed_indices[:, 0].float()
            out[:, :, 0] = (query[:, 0, 0].float() + win_len.float() + extra).to(torch.bfloat16)[:, None]
            return out

    wo_calls = []

    class FakeWo:
        def __init__(self, attended, weight, output, **kwargs):
            self.plan = object()
            declared[kwargs["name"]] = (
                tuple(attended.shape), tuple(output.shape), weight.groups, kwargs["heads_per_group"],
                tuple(kwargs["positions"].shape), tuple(kwargs["cos_sin_cache"].shape),
            )

        def functional(self, **kwargs):
            return self

        def __call__(self, attended, *, positions=None, cos_sin_cache=None):
            wo_calls.append((positions.clone(), cos_sin_cache.clone()))
            out = torch.zeros((attended.shape[0], 5120), dtype=torch.bfloat16)
            out[:, 0] = attended[:, 0, 0]
            out[:, 1] = cos_sin_cache[:, 0].to(torch.bfloat16)
            return out

    monkeypatch.setattr(ds_attention.sm103_projection, "is_sm103", lambda dev: False)
    monkeypatch.setattr(ds_attention.mla_norm_mxfp8, "is_supported", lambda dev: False)
    monkeypatch.setattr(ds_attention, "QuantizedProjection", FakeProj)
    monkeypatch.setattr(ds_attention, "UnquantizedProjection", FakeUnquantProj)
    monkeypatch.setattr(ds_attention, "PreparedRMSNorm", FakeNorm)
    monkeypatch.setattr(ds_attention, "PreparedCompressedCacheWriter", FakeWindowWriter)
    monkeypatch.setattr(ds_attention, "PreparedMXFP4Indexer", FakeSelector)
    monkeypatch.setattr(ds_attention, "PreparedCompressedMLA", FakeMLA)
    monkeypatch.setattr(ds_attention, "PreparedGroupedOutputProjection", FakeWo)
    monkeypatch.setattr(ds_attention.mla_compress, "Caps", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(ds_attention.mla_compress, "plan",
                        lambda caps: SimpleNamespace(handle=0, request=lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setattr(ds_attention, "declare", lambda *a, **kw: None)
    monkeypatch.setattr(ds_attention, "register_finalizer", lambda *a, **kw: None)
    monkeypatch.setattr(
        ds_attention.PreparedRotaryAngles,
        "__call__",
        lambda self, pos: torch.zeros((pos.numel(), 64), dtype=torch.float32, device=pos.device),
    )

    # Verify INDEXER_QUERY_ROWS=512 selector batching across st = 128, 256, 512, 768, 1024, 2048!
    for test_st, expected_batches in (
        (128, [128]),
        (256, [256]),
        (512, [512]),
        (768, [512, 256]),
        (1024, [512, 512]),
        (2048, [512, 512, 512, 512]),
    ):
        selector_decls.clear()
        tok = test_st * 8
        m_t = Metadata.allocate(tok, 1, 16, "cpu", max_query_tokens=tok)
        s_t = ds_attention.AttentionScratch.allocate(tok, 4, "cpu", candidate_tokens=128)
        wm_t = ds_attention.AttentionIndices(m_t, mode="prefill", tokens=tok, page_size=128 + tok, width=128, ratios=(2,))
        attn_m = ds_attention.PreparedAttention(
            2, source_weights, torch.zeros((tok, 5120), dtype=torch.bfloat16), s_t, m_t, pool, rotary,
            mode="prefill", world=4, session=None, name=f"l2_{tok}", workspace=None,
            window_metadata=wm_t, examples={}, pipelined_mode="mla_wo",
            shard_tokens=test_st, replicated_weights=valid_rep,
        )
        assert attn_m.kv_pack_cols == 2560
        assert [v[0][0] for v in selector_decls.values()] == expected_batches
        assert declared[f"l2_{tok}.compress.values"] == ((test_st, 5120), (test_st, 512))
        assert declared[f"l2_{tok}.compress.gates"] == ((test_st, 5120), (test_st, 512))
        assert declared[f"l2_{tok}.compressed_write"] == ((tok // 2, 512), (tok // 2,))

    wm = ds_attention.AttentionIndices(meta, mode="prefill", tokens=tokens, page_size=128 + tokens,
                                       width=128, ratios=(2,))
    examples = {}
    attn = ds_attention.PreparedAttention(
        3, base_weights, src, scr, meta, pool, rotary, mode="prefill", world=4,
        session=None, name="l3", workspace=None, window_metadata=wm, examples=examples,
        pipelined_mode="full", shard_tokens=st, replicated_weights=valid_rep,
    )
    assert attn.kv_pack_cols == 512
    plain_attn = ds_attention.PreparedAttention(
        3, base_weights, src, scr, meta, pool, rotary, mode="prefill", world=4,
        session=None, name="l3_plain", workspace=None, window_metadata=wm, examples=examples,
        pipelined_mode="full",
    )
    for fn_name, call_args in (
        ("run_shard_qkv_q", (src[:st],)),
        ("run_shard_mla_wo_local", (src[:st],)),
    ):
        with pytest.raises(ValueError, match="requires row-sharded attention"):
            getattr(plain_attn, fn_name)(*call_args, indices=None, shard_rows=torch.arange(st))
    with pytest.raises(ValueError, match="requires row-sharded attention"):
        plain_attn.run_shard_mla_wo(
            torch.zeros((st, 64, 512), dtype=torch.bfloat16), torch.zeros((half, 512), dtype=torch.bfloat16),
            meta, (None, None, None), indices=None, window_cache=None, half_idx=0,
            shard_rows=torch.arange(st), layer_angles=torch.zeros((st, 64)),
        )

    attn.fused_qkv_norm_rope = False
    attn.query_norm = FakeNorm(scr.projected[:st, :1280], base_weights["q_norm"], scr.query_low[:st], name="l3.qn")
    attn.window_norm = FakeNorm(scr.projected[:st, 1280:], base_weights["kv_norm"], scr.window_values[:st], name="l3.kvn")
    monkeypatch.setattr(attn.angles, "apply", lambda tensor, angles: None)

    assert declared["l3.qkv_a"][:2] == ((st, 5120), (st, 1792))
    assert declared["l3.q_b"] == ((st, 1280), (st, 64 * 512), valid_rep["q_b_full"])
    assert declared["l3.window_write"] == ((half, 512), (half,))
    assert declared["l3.attention"] == ((st, 64, 512), (st, 64, 512), (64,), (st, 512))
    assert declared["l3.output"] == ((st, 64, 512), (st, 5120), 8, 8, (st,), (st, 64))

    wm.functional()
    meta.positions.copy_(torch.arange(tokens, dtype=torch.int32))
    meta.starts.zero_()
    meta.lengths.fill_(tokens)
    meta.query_start_loc.copy_(torch.tensor([0, tokens], dtype=torch.int32))
    meta.live_counts.copy_(torch.tensor([tokens, 1], dtype=torch.int32))
    meta.state_rows.fill_(1)
    meta.page_table[0, :16] = torch.arange(16, dtype=torch.int32)
    distinct_pt = (
        torch.arange(tokens, dtype=torch.int32)[:, None] * 17
        + torch.arange(meta.token_page_table.shape[1], dtype=torch.int32)[None, :]
    )
    object.__setattr__(meta, "token_page_table", distinct_pt)
    indices = wm(meta)
    indices.angles[(True, 0)][:, 0] = torch.arange(tokens, dtype=torch.float32) * 0.25

    x = torch.zeros((tokens, 5120), dtype=torch.bfloat16)
    x[:, 0] = torch.arange(1, tokens + 1, dtype=torch.float32).to(torch.bfloat16)
    selected = (torch.arange(tokens, dtype=torch.int32)[:, None] * 3 + 7).expand(tokens, 512).contiguous()

    rows_a = [torch.arange(r * st, (r + 1) * st, dtype=torch.int64) for r in range(world)]
    rows_b = [torch.arange(half + r * st, half + (r + 1) * st, dtype=torch.int64) for r in range(world)]

    qa_s, wva_s, anga_s = [], [], []
    qb_s, wvb_s, angb_s = [], [], []
    for r in range(world):
        qa, wva, aa = attn.run_shard_qkv_q(x[r * st : (r + 1) * st], indices=indices, shard_rows=rows_a[r])
        qb, wvb, ab = attn.run_shard_qkv_q(x[half + r * st : half + (r + 1) * st], indices=indices, shard_rows=rows_b[r])
        qa_s.append(qa)
        wva_s.append(wva)
        anga_s.append(aa)
        qb_s.append(qb)
        wvb_s.append(wvb)
        angb_s.append(ab)

    wva_full = torch.cat(wva_s, dim=0)
    wvb_full = torch.cat(wvb_s, dim=0)

    caches = []
    outs_a, outs_b = [], []
    mla_calls.clear()
    wo_calls.clear()
    for r in range(world):
        wc = torch.zeros((1, (128 + tokens) * 528), dtype=torch.uint8)
        oa = attn.run_shard_mla_wo(
            qa_s[r], wva_full, meta, (selected, None, None),
            indices=indices, window_cache=wc, half_idx=0, shard_rows=rows_a[r], layer_angles=anga_s[r],
        )
        ob = attn.run_shard_mla_wo(
            qb_s[r], wvb_full, meta, (selected, None, None),
            indices=indices, window_cache=wc, half_idx=1, shard_rows=rows_b[r], layer_angles=angb_s[r],
        )
        caches.append(wc)
        outs_a.append(oa)
        outs_b.append(ob)

    # Also verify hoisted ShardAttentionInputs + shard-local tuple `(sel_a_sh, sel_b_sh)` produces
    # bit-exact outputs with ZERO `torch.index_select` calls inside `run_shard_qkv_q` / `run_shard_mla_wo`!
    index_select_count = [0]
    real_index_select = torch.index_select

    def counting_index_select(*args, **kwargs):
        index_select_count[0] += 1
        return real_index_select(*args, **kwargs)

    for r in range(world):
        ra, rb = rows_a[r], rows_b[r]
        r2 = indices.ratios[2]
        sh_in = ds_attention.ShardAttentionInputs(
            window_indices=(real_index_select(indices.window_indices, 0, ra), real_index_select(indices.window_indices, 0, rb)),
            window_lengths=(real_index_select(indices.window_lengths, 0, ra), real_index_select(indices.window_lengths, 0, rb)),
            angles={(True, 0): (real_index_select(indices.angles[(True, 0)], 0, ra), real_index_select(indices.angles[(True, 0)], 0, rb))},
            index_lengths=(real_index_select(r2.index_lengths, 0, ra), real_index_select(r2.index_lengths, 0, rb)),
            compressed_lengths=(real_index_select(r2.lengths, 0, ra), real_index_select(r2.lengths, 0, rb)),
            page_table=(real_index_select(meta.token_page_table, 0, ra), real_index_select(meta.token_page_table, 0, rb)),
            active_width_a=r2.lengths[:half].amax().view(1),
        )
        local_sel_tuple = (selected[r * st : (r + 1) * st], selected[half + r * st : half + (r + 1) * st])
        monkeypatch.setattr(torch, "index_select", counting_index_select)
        index_select_count[0] = 0
        wc_hoisted = torch.zeros((1, (128 + tokens) * 528), dtype=torch.uint8)
        qa_h, _, aa_h = attn.run_shard_qkv_q(
            x[r * st : (r + 1) * st], indices=indices, shard_rows=ra, half_idx=0, shard_inputs=sh_in)
        qb_h, _, ab_h = attn.run_shard_qkv_q(
            x[half + r * st : half + (r + 1) * st], indices=indices, shard_rows=rb, half_idx=1, shard_inputs=sh_in)
        oa_h = attn.run_shard_mla_wo(
            qa_h, wva_full, meta, (local_sel_tuple, None, None),
            indices=indices, window_cache=wc_hoisted, half_idx=0, shard_rows=ra, layer_angles=aa_h, shard_inputs=sh_in)
        ob_h = attn.run_shard_mla_wo(
            qb_h, wvb_full, meta, (local_sel_tuple, None, None),
            indices=indices, window_cache=wc_hoisted, half_idx=1, shard_rows=rb, layer_angles=ab_h, shard_inputs=sh_in)
        monkeypatch.setattr(torch, "index_select", real_index_select)
        assert index_select_count[0] == 0
        assert torch.equal(oa_h, outs_a[r]) and torch.equal(ob_h, outs_b[r])
        assert torch.equal(wc_hoisted, caches[r])

    # Mutation sensitivity: perturbing row (2 * st + 5) in `selected` affects ONLY rank 2 Half A row 5.
    mut_selected = selected.clone()
    mut_selected[2 * st + 5, 0] += 64
    mut_outs_a = [
        attn.run_shard_mla_wo(
            qa_s[r], wva_full, meta, (mut_selected, None, None),
            indices=indices, window_cache=caches[r].clone(), half_idx=0,
            shard_rows=rows_a[r], layer_angles=anga_s[r],
        )
        for r in range(world)
    ]
    for r in range(world):
        if r == 2:
            assert not torch.equal(mut_outs_a[r], outs_a[r])
            assert torch.equal(mut_outs_a[r][:5], outs_a[r][:5])
            assert torch.equal(mut_outs_a[r][6:], outs_a[r][6:])
        else:
            assert torch.equal(mut_outs_a[r], outs_a[r])


class _FakeDmaRuntime:
    def __init__(self, world_size: int = 4, *, kv_gather_fits: bool = True, mla_wo_fits: bool = True,
                 max_bytes: int = 128 << 20, shard_capacity: int = 32 << 20):
        self.world_size = world_size
        self.max_bytes = max_bytes
        self.shard_capacity = shard_capacity
        self.should_reduce_scatter = lambda tensor: True
        self.should_all_gather = lambda tensor: (
            (kv_gather_fits or tensor.shape[1] != 512)
            and (mla_wo_fits or tensor.shape[1] != 2560)
        )

    def should_allreduce(self, tensor):
        return True


def test_prepared_decoder_row_shard_attention_layer_routing_and_kv_runtime(monkeypatch):
    monkeypatch.setattr(pcie, "DmaAllReduce", _FakeDmaRuntime)

    seen_layers = []

    class FakeLayer:
        def __init__(self, index, *args, **kwargs):
            self.index = index
            self.has_image_mask = False
            self.pipelined_ffn = kwargs["pipelined_ffn"]
            self.pipelined_attn = kwargs["pipelined_attn"]
            self.pipelined_attn_hc = kwargs["pipelined_attn_hc"]
            self.row_shard_dma = kwargs["row_shard_dma"]
            self.replicated_attention = kwargs.get("replicated_attention")
            self.kv_runtime = kwargs.get("kv_runtime")
            self.engram_min_gather_chunks = kwargs.get("engram_min_gather_chunks")
            self.row_shard_attention = bool(self.row_shard_dma and self.replicated_attention is not None)
            self.p2p_kv_gather = bool(
                self.row_shard_attention and self.pipelined_attn == "full"
                and self.pipelined_attn_hc and self.kv_runtime is not None
            )
            self.kv_gather_0 = SimpleNamespace(plan=SimpleNamespace(tag=f"kv0_{index}")) if self.p2p_kv_gather else None
            self.shard_prep_plans = (SimpleNamespace(tag=f"prep_{index}"),)
            self.ffn_compute_plans = (SimpleNamespace(tag=f"ffn_{index}"),)
            self.boundary = kwargs["boundary"]
            self.engram = None
            self.attention = SimpleNamespace(selectors=[object()], plans=lambda: (SimpleNamespace(tag=f"attn_{index}"),))
            self.post = None if kwargs["defer_post"] else object()
            self.cr_mlp_1 = SimpleNamespace(reduce=SimpleNamespace(plan=SimpleNamespace(tag=f"mlp1_{index}")))
            self.attention_hc = SimpleNamespace(plan=SimpleNamespace(tag=f"hc_{index}"))
            seen_layers.append(self)

    class FakeBoundary:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(decoder, "PreparedTableRequests", lambda *a, **kw: None)
    monkeypatch.setattr(decoder, "PreparedEmbedding", lambda *a, **kw: SimpleNamespace(functional=lambda **k: None))
    monkeypatch.setattr(decoder, "Rotary", lambda *a, **kw: None)
    monkeypatch.setattr(decoder, "AttentionIndices", lambda *a, **kw: SimpleNamespace(functional=lambda: None))
    monkeypatch.setattr(decoder, "PreparedLayer", FakeLayer)
    monkeypatch.setattr(decoder, "PreparedDecoderBoundary", FakeBoundary)
    monkeypatch.setattr(decoder, "register_finalizer", lambda *a, **kw: None)
    monkeypatch.setattr(decoder.PreparedDecoder, "_prepare_head", lambda *a, **kw: None)

    @dataclass(frozen=True)
    class FakeLayerWeights:
        engram: object
        attention: dict

    cfg = _qualified_model_config()
    kv_sources = set(cfg.text["kv_source_layer_ids"])
    index_sources = set(cfg.text["index_source_layer_ids"])
    layers = tuple(
        FakeLayerWeights(
            engram=object() if i in (1, 14) else None,
            attention=(
                {"compressor": object(), "index_q": object()} if i in kv_sources
                else ({"index_q": object()} if i in index_sources else {})
            ),
        )
        for i in range(40)
    )
    rep_map = {i: {"tag": i} for i in EXPECTED_ROW_SHARD_ENCODER_LAYERS}

    def build(*, tokens: int = 8192, row_shard_attention: bool = True, rep=rep_map,
              kv_fits: bool = True, mla_wo_fits: bool = True, kv_runtime=None, encoder_only: bool = False):
        seen_layers.clear()
        weights = SimpleNamespace(
            config=SimpleNamespace(
                decoder_start=20,
                sources=cfg.sources,
                text={**cfg.text, "rms_norm_eps": 1e-20},
                draft=SimpleNamespace(target_layers=(37, 38, 39)),
            ),
            draft=None,
            layers=layers,
            replicated_attention=rep,
            shard=TensorParallelShard(0, 4),
            embedding=torch.zeros((16, 5120), dtype=torch.bfloat16),
            norm=torch.ones(5120, dtype=torch.bfloat16),
            head=torch.zeros((16, 5120), dtype=torch.bfloat16),
        )
        scratch = SimpleNamespace(
            embedding=torch.zeros((tokens, 5120), dtype=torch.bfloat16),
            residual=torch.zeros((tokens, 4, 5120), dtype=torch.bfloat16),
            attention=SimpleNamespace(
                output=torch.zeros((tokens, 5120), dtype=torch.bfloat16),
                window_values=torch.zeros((tokens, 512), dtype=torch.bfloat16),
            ),
            attention_output=torch.zeros((tokens, 5120), dtype=torch.bfloat16),
            attention_hc=SimpleNamespace(normalized=torch.zeros((tokens, 5120), dtype=torch.bfloat16)),
            hidden=torch.zeros((1, 5120), dtype=torch.bfloat16),
            local_logits=torch.zeros((1, 16), dtype=torch.float32),
            logits=torch.zeros((1, 16), dtype=torch.float32),
        )
        meta = Metadata.allocate(tokens, 1, 10, "cpu", max_query_tokens=tokens)
        arena = ScratchArena("cpu", overlap=True)
        dec = decoder.PreparedDecoder(
            weights, None, scratch, meta, None, mode="prefill",
            runtime=_FakeDmaRuntime(4, kv_gather_fits=kv_fits, mla_wo_fits=mla_wo_fits,
                                    max_bytes=256 << 20, shard_capacity=64 << 20),
            vocabulary_runtime=None, coordinator=None, session=None, name="dec",
            workspace=arena, kv_runtime=kv_runtime,
            encoder_only=encoder_only, row_shard_prefill=True, row_shard_attention=row_shard_attention,
            examples={},
        )
        return dec, list(seen_layers), arena

    fake_kv_rt = SimpleNamespace(world_size=4, max_batch_size=2048, heads_per_rank=8, query_head_dim=64)
    expected_p2p_layers = (3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19)

    for tokens in (8192, 16384):
        dec_on, layers_on, arena = build(tokens=tokens, row_shard_attention=True, kv_runtime=fake_kv_rt)
        assert dec_on.row_shard_dma is True and dec_on.row_shard_attention is True
        sharded_indices = tuple(l.index for l in layers_on if l.replicated_attention is not None)
        assert sharded_indices == EXPECTED_ROW_SHARD_ENCODER_LAYERS
        p2p_indices = tuple(l.index for l in layers_on if l.kv_runtime is not None)
        assert p2p_indices == expected_p2p_layers
        assert layers_on[1].engram_min_gather_chunks == 2
        assert layers_on[14].engram_min_gather_chunks == 4
        # Verify ScratchArena.register_parallel covers both mla_wo carry-in (2, 8) and p2p_kv_gather (15 layers).
        for idx in (2, 8, *expected_p2p_layers):
            prev_plan = layers_on[idx - 1].cr_mlp_1.reduce.plan
            assert (prev_plan, layers_on[idx].attention_hc.plan) in arena._parallel_plans
            for p in layers_on[idx].attention.plans():
                assert (prev_plan, p) in arena._parallel_plans
            if idx in expected_p2p_layers:
                assert (prev_plan, layers_on[idx].kv_gather_0.plan) in arena._parallel_plans
                for p in layers_on[idx].shard_prep_plans:
                    assert (prev_plan, p) in arena._parallel_plans

    # Oversized chunk relative to kv_runtime.max_batch_size falls back cleanly to prefill_collective.
    small_kv_rt = SimpleNamespace(world_size=4, max_batch_size=1024, heads_per_rank=8, query_head_dim=64)
    dec_16k_fb, layers_16k_fb, _ = build(tokens=16384, row_shard_attention=True, kv_runtime=small_kv_rt)
    assert dec_16k_fb.row_shard_attention is True
    assert all(l.kv_runtime is None for l in layers_16k_fb)

    # Disabling overrides (including missing layer 2 from replicated_attention or mla_wo_fits=False).
    partial_rep = {k: v for k, v in rep_map.items() if k != 2}
    for override in (
        {"row_shard_attention": False},
        {"row_shard_attention": True, "rep": None},
        {"row_shard_attention": True, "rep": partial_rep},
        {"row_shard_attention": True, "kv_fits": False},
        {"row_shard_attention": True, "mla_wo_fits": False},
    ):
        dec_off, layers_off, _ = build(**override)
        assert dec_off.row_shard_dma is True and dec_off.row_shard_attention is False
        assert all(l.replicated_attention is None and l.kv_runtime is None for l in layers_off)

    # Verify bootstrap._collective_runtimes preserves its 5-tuple contract and _kv_gather_runtime
    # sizes DcpAllToAll dynamically at chunk=8192 (1024 rows) and chunk=16384 (2048 rows).
    ag_calls = []
    monkeypatch.setattr(ds_bootstrap, "create_all_reduce_runtime", lambda *a, **kw: "ar")
    monkeypatch.setattr(ds_bootstrap, "create_dma_all_reduce_runtime", lambda *a, **kw: "dma")
    monkeypatch.setattr(ds_bootstrap, "create_result_broadcast_runtime", lambda *a, **kw: "res")
    monkeypatch.setattr(
        ds_bootstrap, "create_all_gather_runtime",
        lambda group, dev, *, max_rows, columns, ipc=None: ag_calls.append((max_rows, columns)) or f"ag_{max_rows}_{columns}",
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda dev: SimpleNamespace(multi_processor_count=188),
    )
    coord_local = SimpleNamespace(spans_hosts=False)
    coord_remote = SimpleNamespace(spans_hosts=True)
    storage = SimpleNamespace(peer_storage="ipc")
    for chunk_sz, rsa, coord, expected_kv in (
        (8192, True, coord_local, "ag_1024_2048"),
        (16384, True, coord_local, "ag_2048_2048"),
        (8192, False, coord_local, None),
        (512, True, coord_local, None),
        (8192, True, coord_remote, None),
    ):
        ag_calls.clear()
        p_cfg = PreparationConfig(
            parameters=replace(
                PreparationConfig().parameters,
                prefill_chunk=chunk_sz,
                row_shard_prefill=True,
                row_shard_attention=rsa,
                speculation=None,
            ),
        )
        if not coord.spans_hosts:
            rets = ds_bootstrap._collective_runtimes(
                p_cfg, "grp", "cuda:0", coordinator=coord, core_storage=storage, verification_samples=32,
            )
            assert len(rets) == 5
        kv_rt = ds_bootstrap._kv_gather_runtime(
            p_cfg, "grp", "cuda:0", coordinator=coord, core_storage=storage,
        )
        assert kv_rt == expected_kv


@pytest.mark.parametrize("st", [128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("pipelined_attn_hc", [False, True])
@pytest.mark.parametrize("use_p2p_kv", [False, True])
def test_dcp_kv_gather_overlap_schedule_and_transpose_parity(st, pipelined_attn_hc, use_p2p_kv):
    world, hidden, top_k = 4, 256, 6
    half = st * world
    tokens = 2 * half
    events = []

    # 1. Verify _gather_kv_shard rank-major transpose parity across all st in 128..2048!
    t_layer = decoder.PreparedLayer.__new__(decoder.PreparedLayer)
    t_layer.world_size = world
    t_layer.half_tokens = half
    shards = [
        (r * 10000.0 + torch.arange(st, dtype=torch.float32)[:, None] * 3.0 + torch.arange(512, dtype=torch.float32)[None, :]).to(torch.bfloat16)
        for r in range(world)
    ]
    col_interleaved = torch.cat(shards, dim=1).contiguous()
    assert col_interleaved.shape == (st, 512 * world)

    class FakeP2PGather:
        single = False

        def __call__(self, wv_sh, *, dependencies=()):
            events.append("p2p_kv_gather")
            return col_interleaved

    got_full = t_layer._gather_kv_shard(FakeP2PGather(), shards[0])
    expected_full = torch.cat(shards, dim=0)
    assert got_full.shape == (half, 512) and got_full.is_contiguous()
    assert torch.equal(got_full.view(torch.uint16), expected_full.view(torch.uint16))

    # Sensitivity: perturbing rank 2 row 7 column 13 alters ONLY global row (2*st + 7) column 13.
    bad_col = col_interleaved.clone()
    bad_col.view(torch.uint16)[7, 2 * 512 + 13] ^= 1
    bad_full = t_layer._gather_kv_shard(
        type("G", (), {"single": False, "__call__": lambda self, x, **k: bad_col})(), shards[0]
    )
    assert not torch.equal(bad_full.view(torch.uint16), expected_full.view(torch.uint16))
    diff_locs = (bad_full.view(torch.uint16) != expected_full.view(torch.uint16)).nonzero(as_tuple=False).tolist()
    assert diff_locs == [[2 * st + 7, 13]]

    if st != 128:
        return

    # 2. Verify peer-to-peer KV overlap schedule vs fallback DMA schedule event ordering.
    events.clear()

    class LoggedCollective:
        def __init__(self, tag, fn):
            self.tag = tag
            self.fn = fn
            self.seen_shapes = []

        def start(self, source, *, dependencies=()):
            self.seen_shapes.append(tuple(source.shape))
            events.append(f"{self.tag}.start")
            return SimpleNamespace(source=source, output=self.fn(source), dependencies=dependencies)

        def finish(self, pending, dependency):
            events.append(f"{self.tag}.finish")
            return pending.output

    class LoggedSyncGather:
        single = False

        def __init__(self, tag):
            self.tag = tag
            self.seen_shapes = []

        def __call__(self, source, *, dependencies=()):
            self.seen_shapes.append(tuple(source.shape))
            events.append(self.tag)
            return source.repeat(1, world)

    layer = decoder.PreparedLayer.__new__(decoder.PreparedLayer)
    layer.row_shard_dma = True
    layer.row_shard_attention = True
    layer.pipelined_attn_hc = pipelined_attn_hc
    layer.pipelined_attn = "full"
    layer.p2p_kv_gather = bool(pipelined_attn_hc and use_p2p_kv)
    layer.world_size = world
    layer.defer_mlp_b = False
    layer.half_tokens = half
    layer.shard_tokens = st
    layer.shard_rows_a = torch.arange(0, st, dtype=torch.int64)
    layer.shard_rows_b = torch.arange(half, half + st, dtype=torch.int64)
    layer.shard_offset_a = torch.tensor([0], dtype=torch.int32)
    layer.shard_offset_b = torch.tensor([half], dtype=torch.int32)
    layer.engram = None
    layer.has_image_mask = False
    layer.attn_quantizer_plan = None
    layer.ffn_quantizer_plan = None
    layer.ffn_local_shapes = None
    layer.ffn_meta_scale_cols = 0
    layer.attn_gather_0 = layer.attn_gather_1 = None
    layer.cr_attn_gather_0 = layer.cr_attn_gather_1 = None
    layer.att_reduce_0 = layer.att_reduce_1 = None
    layer.cr_att_0 = layer.cr_att_1 = None

    hc_call = [0]

    def fake_attn_hc(*args, **kwargs):
        tag = "A" if hc_call[0] == 0 else "B"
        hc_call[0] += 1
        events.append(f"attn_hc_{tag}")
        return (
            torch.zeros((st, 4, hidden), dtype=torch.bfloat16),
            torch.zeros((st, 4), dtype=torch.float32),
            torch.zeros((st, 4, 4), dtype=torch.float32),
            torch.full((st, hidden), 1.0 if tag == "A" else 2.0, dtype=torch.bfloat16),
        )

    def fake_qkv_q(norm_sh, *, indices, shard_rows):
        tag = "A" if int(shard_rows[0].item()) == 0 else "B"
        events.append(f"qkv_q_{tag}")
        val = 10.0 if tag == "A" else 20.0
        return (
            torch.full((st, 64, 512), val, dtype=torch.bfloat16),
            torch.full((st, 512), val, dtype=torch.bfloat16),
            torch.full((st, 64), val, dtype=torch.float32),
        )

    def fake_mla_wo(q_sh, wv_full, metadata, selection, *, indices, window_cache, half_idx, shard_rows, layer_angles):
        tag = "A" if half_idx == 0 else "B"
        events.append(f"mla_wo_{tag}")
        assert wv_full.shape == (half, 512)
        return torch.full((st, hidden), 3.0 if half_idx == 0 else 4.0, dtype=torch.bfloat16)

    layer.attention_hc = fake_attn_hc
    layer.attention = SimpleNamespace(
        mode="prefill",
        init_window_cache=lambda metadata, dev: "wc",
        run_shard_qkv_q=fake_qkv_q,
        run_shard_mla_wo=fake_mla_wo,
        commit=lambda lengths, records, metadata: events.append("attn_commit"),
    )
    if layer.p2p_kv_gather:
        layer.kv_gather_0 = LoggedSyncGather("kv_gather_0")
        layer.kv_gather_1 = LoggedSyncGather("kv_gather_1")
        layer.cr_kv_gather_0 = layer.cr_kv_gather_1 = None
    else:
        layer.cr_kv_gather_0 = LoggedCollective("kv_gather_0", lambda sh: sh.repeat(world, 1))
        layer.cr_kv_gather_1 = LoggedCollective("kv_gather_1", lambda sh: sh.repeat(world, 1))

    ffn_hc_call = [0]

    def fake_ffn_hc(att, res, post, comb, *, pre_mix, pre_out):
        tag = "A" if ffn_hc_call[0] == 0 else "B"
        ffn_hc_call[0] += 1
        events.append(f"ffn_hc_{tag}")
        return res, post, comb, att

    layer.ffn_hc = fake_ffn_hc
    layer.router_projection = lambda norm: torch.zeros((norm.shape[0], 384), dtype=torch.float32)
    prep_call = [0]

    def fake_router(logits, **kw):
        tag = "A" if prep_call[0] == 0 else "B"
        prep_call[0] += 1
        events.append(f"ffn_prep_{tag}")
        return RouterScratch(
            torch.zeros((logits.shape[0], top_k), dtype=torch.float32),
            torch.zeros((logits.shape[0], top_k), dtype=torch.int32),
            torch.ones((logits.shape[0], top_k), dtype=torch.float32),
        )

    layer.router = fake_router
    layer.cr_ffn_vals_0 = LoggedCollective("ffn_vals_0", lambda sh: sh.repeat(world, 1))
    layer.cr_ffn_meta_0 = LoggedCollective("ffn_meta_0", lambda sh: sh.repeat(world, 1))
    layer.cr_ffn_vals_1 = LoggedCollective("ffn_vals_1", lambda sh: sh.repeat(world, 1))
    layer.cr_ffn_meta_1 = LoggedCollective("ffn_meta_1", lambda sh: sh.repeat(world, 1))
    layer.experts = lambda vals, ids, weights, **kw: (events.append("ffn_compute") or vals)
    layer.shared = lambda vals, **kw: vals
    layer.cr_mlp_0 = LoggedCollective("mlp_rs_0", lambda full: full[:st].clone())
    layer.cr_mlp_1 = LoggedCollective("mlp_rs_1", lambda full: full[:st].clone())
    layer.post = lambda mlp_sh, res, post, comb: res

    metadata = Metadata.allocate(tokens, 1, 10, "cpu")
    metadata.live_counts[0] = tokens
    metadata.lengths[0] = tokens
    pm_a = torch.zeros((st, 4), dtype=torch.float32)
    pm_b = torch.zeros((st, 4), dtype=torch.float32)
    if pipelined_attn_hc:
        zeros_res = torch.zeros((st, 4, hidden), dtype=torch.bfloat16)
        zeros_post = torch.zeros((st, 4), dtype=torch.float32)
        zeros_comb = torch.zeros((st, 4, 4), dtype=torch.float32)
        mlp_a = torch.zeros((st, hidden), dtype=torch.bfloat16)
        p_mlp_b = SimpleNamespace(output=torch.zeros((st, hidden), dtype=torch.bfloat16))
        incoming = (mlp_a, zeros_res, zeros_post, zeros_comb, p_mlp_b, zeros_res, zeros_post, zeros_comb)
    else:
        incoming = (
            torch.zeros((st, 4, hidden), dtype=torch.bfloat16),
            torch.zeros((st, 4, hidden), dtype=torch.bfloat16),
        )

    layer._run_row_shard_dma(incoming, (pm_a, pm_b), metadata, "indices", ("sel", None, None))

    if layer.p2p_kv_gather:
        assert layer.kv_gather_0.seen_shapes == [(st, 512)]
        assert layer.kv_gather_1.seen_shapes == [(st, 512)]
        expected_prefix = [
            "attn_hc_A", "qkv_q_A", "kv_gather_0", "mla_wo_A", "ffn_hc_A", "ffn_prep_A",
            "mlp_rs_1.finish", "ffn_vals_0.start", "ffn_meta_0.start",
            "attn_hc_B", "qkv_q_B", "kv_gather_1", "mla_wo_B", "attn_commit", "ffn_hc_B", "ffn_prep_B",
            "ffn_vals_0.finish", "ffn_meta_0.finish",
        ]
        assert events[:len(expected_prefix)] == expected_prefix
    else:
        assert layer.cr_kv_gather_0.seen_shapes == [(st, 512)]
        assert layer.cr_kv_gather_1.seen_shapes == [(st, 512)]
        idx = {ev: events.index(ev) for ev in (
            "attn_hc_A", "qkv_q_A", "kv_gather_0.start", "attn_hc_B", "qkv_q_B", "kv_gather_0.finish",
            "kv_gather_1.start", "mla_wo_A", "ffn_hc_A", "ffn_prep_A", "kv_gather_1.finish",
            "ffn_vals_0.start", "ffn_meta_0.start", "mla_wo_B", "attn_commit", "ffn_hc_B", "ffn_prep_B",
            "ffn_vals_0.finish", "ffn_meta_0.finish",
        )}
        if pipelined_attn_hc:
            first_mlp_finish = events.index("mlp_rs_1.finish")
            assert idx["attn_hc_A"] < idx["qkv_q_A"] < first_mlp_finish < idx["kv_gather_0.start"]
        assert idx["kv_gather_0.start"] < idx["attn_hc_B"] < idx["qkv_q_B"] < idx["kv_gather_0.finish"]
        assert idx["kv_gather_1.start"] < idx["mla_wo_A"] < idx["ffn_hc_A"] < idx["ffn_prep_A"] < idx["kv_gather_1.finish"]


@pytest.mark.parametrize("pipelined_attn_hc", [False, True])
def test_mla_wo_row_shard_dma_schedule_and_engram_ffn_window_overlap(monkeypatch, pipelined_attn_hc):
    tokens, world, hidden, top_k = 8192, 4, 256, 6
    half = tokens // 2
    st = half // world
    events = []

    class LoggedCollective:
        def __init__(self, tag, fn):
            self.tag = tag
            self.fn = fn
            self.seen_shapes = []

        def start(self, source, *, dependencies=()):
            self.seen_shapes.append(tuple(source.shape))
            events.append(f"{self.tag}.start")
            return SimpleNamespace(source=source, output=self.fn(source), dependencies=dependencies)

        def finish(self, pending, dependency):
            events.append(f"{self.tag}.finish")
            return pending.output

    layer = decoder.PreparedLayer.__new__(decoder.PreparedLayer)
    layer.row_shard_dma = True
    layer.row_shard_attention = True
    layer.pipelined_attn_hc = pipelined_attn_hc
    layer.pipelined_attn = "mla_wo"
    layer.p2p_kv_gather = False
    layer.world_size = world
    layer.defer_mlp_b = False
    layer.half_tokens = half
    layer.shard_tokens = st
    layer.shard_rows_a = torch.arange(0, st, dtype=torch.int64)
    layer.shard_rows_b = torch.arange(half, half + st, dtype=torch.int64)
    layer.shard_offset_a = torch.tensor([0], dtype=torch.int32)
    layer.shard_offset_b = torch.tensor([half], dtype=torch.int32)
    layer.engram = None
    layer.has_image_mask = False
    layer.attn_quantizer_plan = None
    layer.ffn_quantizer_plan = None
    layer.ffn_local_shapes = None
    layer.ffn_meta_scale_cols = 0
    layer.attn_gather_0 = layer.attn_gather_1 = None
    layer.cr_attn_gather_0 = layer.cr_attn_gather_1 = None
    layer.att_reduce_0 = layer.att_reduce_1 = None
    layer.cr_att_0 = layer.cr_att_1 = None

    hc_call = [0]

    def fake_attn_hc(*args, **kwargs):
        tag = "A" if hc_call[0] == 0 else "B"
        hc_call[0] += 1
        events.append(f"attn_hc_{tag}")
        return (
            torch.zeros((st, 4, hidden), dtype=torch.bfloat16),
            torch.zeros((st, 4), dtype=torch.float32),
            torch.zeros((st, 4, 4), dtype=torch.float32),
            torch.ones((st, hidden), dtype=torch.bfloat16),
        )

    def fake_local(norm_sh, *, indices, shard_rows, **kw):
        tag = "A" if int(shard_rows[0].item()) == 0 else "B"
        events.append(f"mla_wo_local_{tag}")
        return (
            torch.ones((st, 64, 512), dtype=torch.bfloat16),
            torch.ones((st, 2560), dtype=torch.bfloat16),
            torch.ones((st, 32, 128), dtype=torch.bfloat16),
            torch.ones((st, 32), dtype=torch.bfloat16),
            torch.ones((st, 64), dtype=torch.float32),
        )

    def fake_source(q_sh, kv_full, iq_sh, iw_sh, metadata, selection, *, half_idx, **kw):
        tag = "A" if half_idx == 0 else "B"
        events.append(f"mla_wo_source_{tag}")
        assert kv_full.shape == (half, 2560)
        sel_sh = torch.full((st, 512), 100 + half_idx, dtype=torch.int32)
        return torch.ones((st, hidden), dtype=torch.bfloat16), sel_sh

    layer.attention_hc = fake_attn_hc
    layer.attention = SimpleNamespace(
        mode="prefill",
        global_kv=SimpleNamespace(compressor=SimpleNamespace(init_pending=lambda m, d: "pending")),
        init_window_cache=lambda metadata, dev: "wc",
        run_shard_mla_wo_local=fake_local,
        run_shard_mla_wo_source=fake_source,
        commit=lambda lengths, records, metadata: events.append("attn_commit"),
    )
    layer.cr_kv_gather_0 = LoggedCollective("kv_gather_0", lambda sh: sh.repeat(world, 1))
    layer.cr_kv_gather_1 = LoggedCollective("kv_gather_1", lambda sh: sh.repeat(world, 1))
    layer.ffn_hc = lambda att, res, post, comb, *, pre_mix, pre_out: (events.append("ffn_hc") or (res, post, comb, att))
    layer.router_projection = lambda norm: torch.zeros((norm.shape[0], 384), dtype=torch.float32)
    layer.router = lambda logits, **kw: RouterScratch(
        torch.zeros((logits.shape[0], top_k), dtype=torch.float32),
        torch.zeros((logits.shape[0], top_k), dtype=torch.int32),
        torch.ones((logits.shape[0], top_k), dtype=torch.float32),
    )
    layer.cr_ffn_vals_0 = LoggedCollective("ffn_vals_0", lambda sh: sh.repeat(world, 1))
    layer.cr_ffn_meta_0 = LoggedCollective("ffn_meta_0", lambda sh: sh.repeat(world, 1))
    layer.cr_ffn_vals_1 = LoggedCollective("ffn_vals_1", lambda sh: sh.repeat(world, 1))
    layer.cr_ffn_meta_1 = LoggedCollective("ffn_meta_1", lambda sh: sh.repeat(world, 1))
    ffn_idx = [0]

    def fake_experts(vals, ids, weights, **kw):
        events.append(f"ffn_compute_{'A' if ffn_idx[0] == 0 else 'B'}")
        ffn_idx[0] += 1
        return vals

    layer.experts = fake_experts
    layer.shared = lambda vals, **kw: vals
    layer.cr_mlp_0 = LoggedCollective("mlp_rs_0", lambda full: full[:st].clone())
    layer.cr_mlp_1 = LoggedCollective("mlp_rs_1", lambda full: full[:st].clone())
    layer.post = lambda mlp_sh, res, post, comb: res

    # Attach 2 FFN Engram chunks + engram_project to verify exact Layer-0 -> Layer-1 overlap schedule!
    engram_state = {"partial": None, "chunks": []}
    fake_engram = SimpleNamespace(
        lookup_local=lambda src, m: (events.append("engram_lookup") or torch.arange(tokens * 1536, dtype=torch.float32).reshape(tokens, 1536).to(torch.bfloat16)),
        project_gathered=lambda chunks: (events.append("engram_project") or torch.cat(chunks, dim=1)),
    )
    cr_eng_0 = LoggedCollective("engram_ag_0", lambda sh: sh.unsqueeze(0).expand(world, -1, -1).contiguous())
    cr_eng_1 = LoggedCollective("engram_ag_1", lambda sh: sh.unsqueeze(0).expand(world, -1, -1).contiguous())
    ffn_overlaps = (
        (fake_engram, cr_eng_0, (0, half), engram_state, True),
        (None, cr_eng_1, (half, tokens), engram_state, True),
    )

    metadata = Metadata.allocate(tokens, 1, 10, "cpu")
    metadata.live_counts[0] = tokens
    metadata.lengths[0] = tokens
    pm = (torch.zeros((st, 4), dtype=torch.float32), torch.zeros((st, 4), dtype=torch.float32))
    if pipelined_attn_hc:
        zr = torch.zeros((st, 4, hidden), dtype=torch.bfloat16)
        zp = torch.zeros((st, 4), dtype=torch.float32)
        zc = torch.zeros((st, 4, 4), dtype=torch.float32)
        incoming = (torch.zeros((st, hidden), dtype=torch.bfloat16), zr, zp, zc,
                    SimpleNamespace(output=torch.zeros((st, hidden), dtype=torch.bfloat16)), zr, zp, zc)
    else:
        incoming = (torch.zeros((st, 4, hidden), dtype=torch.bfloat16), torch.zeros((st, 4, hidden), dtype=torch.bfloat16))

    _, _, _, out_sel = layer._run_row_shard_dma(
        incoming, pm, metadata, "indices", ((None, None), None, None),
        ffn_engram_overlaps=ffn_overlaps, engram_project=(fake_engram, engram_state),
    )

    assert layer.cr_kv_gather_0.seen_shapes == [(st, 2560)]
    assert layer.cr_kv_gather_1.seen_shapes == [(st, 2560)]
    assert isinstance(out_sel[0], tuple) and len(out_sel[0]) == 2
    assert torch.all(out_sel[0][0] == 100) and torch.all(out_sel[0][1] == 101)
    assert cr_eng_0.seen_shapes == [(half, 1536)]
    assert cr_eng_1.seen_shapes == [(half, 1536)]
    assert len(engram_state["chunks"]) == 2 and "projected" in engram_state

    # Verify exact FFN-window overlap ordering:
    # Window A: engram_lookup -> ffn_vals_1.start -> ffn_meta_1.start -> engram_ag_0.start -> ffn_compute_A -> finishes
    # Window B: mlp_rs_0.start -> engram_ag_1.start -> ffn_compute_B -> finishes -> mlp_rs_1.start -> engram_project -> mlp_rs_1.finish
    idx = {ev: events.index(ev) for ev in (
        "engram_lookup", "ffn_vals_1.start", "ffn_meta_1.start", "engram_ag_0.start", "ffn_compute_A",
        "engram_ag_0.finish", "mlp_rs_0.start", "engram_ag_1.start", "ffn_compute_B", "engram_ag_1.finish",
        "mlp_rs_1.start", "engram_project",
    )}
    assert idx["engram_lookup"] < idx["ffn_vals_1.start"] < idx["ffn_meta_1.start"] < idx["engram_ag_0.start"]
    assert idx["engram_ag_0.start"] < idx["ffn_compute_A"] < idx["engram_ag_0.finish"] < idx["mlp_rs_0.start"]
    assert idx["mlp_rs_0.start"] < idx["engram_ag_1.start"] < idx["ffn_compute_B"] < idx["engram_ag_1.finish"]
    assert idx["engram_ag_1.finish"] < idx["mlp_rs_1.start"] < idx["engram_project"] < len(events) - 1

    # Verify PreparedEngram min_gather_chunks & fits() at BOTH chunk=8192 and chunk=16384!
    monkeypatch.setattr(ds_engram, "PreparedEngramLookup", lambda *a, **k: SimpleNamespace(functional=lambda **kw: None))
    monkeypatch.setattr(ds_engram, "PreparedAllGather", lambda s, o, **k: SimpleNamespace(
        s=tuple(s.shape), o=tuple(o.shape), plan=SimpleNamespace(), single=False, roce=False,
        functional=lambda **kw: None,
    ))
    monkeypatch.setattr(ds_engram, "QuantizedProjection", lambda *a, **k: SimpleNamespace(
        plan=SimpleNamespace(), functional=lambda **kw: None,
    ))
    monkeypatch.setattr(ds_engram.hyperconnection, "plan", lambda *a, **k: SimpleNamespace(request=lambda **kw: None))
    monkeypatch.setattr(ds_engram, "declare", lambda *a, **k: None)
    monkeypatch.setattr(pcie, "DmaAllReduce", _FakeDmaRuntime)

    for chunk_sz in (8192, 16384):
        dma_tokens = ((chunk_sz * 6 + 4) // 5 + 63) // 64 * 64
        max_bytes = dma_tokens * 5120 * 2
        shard_cap = (max_bytes // 4) & ~255
        rt = _FakeDmaRuntime(4, max_bytes=max_bytes, shard_capacity=shard_cap)
        tables = SimpleNamespace(
            columns=torch.arange(1536), shard=TensorParallelShard(0, 4),
            inputs={1: object(), 14: object()}, token_map=(0, 1), geometry=object(), readiness_runtime=None,
        )
        m_e = Metadata.allocate(chunk_sz, 1, 10, "cpu")
        scr_e = ds_engram.EngramScratch.allocate(chunk_sz, "cpu", world=4)
        src_e = torch.zeros((chunk_sz, 4, 5120), dtype=torch.bfloat16)
        out_e = torch.zeros((chunk_sz, 4, 5120), dtype=torch.bfloat16)
        w_e = {"projection_full": "pf", "projection": "ps", "weight": torch.ones(5120, dtype=torch.bfloat16)}
        eng_1 = ds_engram.PreparedEngram(
            1, w_e, tables, src_e, out_e, m_e, SimpleNamespace(config=SimpleNamespace(text={"rms_norm_eps": 1e-6})),
            scratch=scr_e, session=None, name="e1", workspace=None, runtime=rt,
            defer_reduce_functional=True, row_shard=(0, chunk_sz // 2, chunk_sz // 8), min_gather_chunks=2,
        )
        eng_14 = ds_engram.PreparedEngram(
            14, w_e, tables, src_e, out_e, m_e, SimpleNamespace(config=SimpleNamespace(text={"rms_norm_eps": 1e-6})),
            scratch=scr_e, session=None, name="e14", workspace=None, runtime=rt,
            defer_reduce_functional=True, row_shard=(0, chunk_sz // 2, chunk_sz // 8), min_gather_chunks=4,
        )
        assert eng_1.gather_bounds == ((0, chunk_sz // 2), (chunk_sz // 2, chunk_sz))
        assert len(eng_14.gather_bounds) == 4
        assert all((e - b) * 1536 * 2 <= shard_cap for b, e in eng_1.gather_bounds + eng_14.gather_bounds)


@pytest.mark.parametrize("tokens", [8192, 16384])
def test_layer20_boundary_shard_csa1_and_128_row_suffix_exact_parity(monkeypatch, tokens):
    world = 4
    h = tokens // 2
    s = h // world
    cfg = _qualified_model_config()
    pool = ContinuationPool.allocate(cfg, 4, 2, "cpu")
    rotary = ds_attention.Rotary(cfg, "cpu")

    class Component(SimpleNamespace):
        def functional(self, **kwargs):
            return self

        def __call__(self, *args, **kwargs):
            fn = getattr(self, "call", None)
            return None if fn is None else fn(*args, **kwargs)

    monkeypatch.setattr(ds_attention.mla_compress, "Caps", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(ds_attention.mla_compress, "plan",
                        lambda caps: SimpleNamespace(handle=0, request=lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setattr(ds_attention, "declare", lambda *a, **kw: None)
    monkeypatch.setattr(ds_attention, "register_finalizer", lambda *a, **kw: None)
    monkeypatch.setattr(ds_attention, "UnquantizedProjection",
                        lambda src, w, out, **kw: Component(plan=object(), call=lambda x: x[:, :out.shape[1]].clone()))
    monkeypatch.setattr(ds_attention, "PreparedRMSNorm", lambda *a, **kw: Component(plan=object()))
    monkeypatch.setattr(ds_attention, "PreparedCompressedCacheWriter",
                        lambda *a, **kw: Component(plan=object()))
    monkeypatch.setattr(
        ds_attention.PreparedRotaryAngles,
        "__call__",
        lambda self, pos: torch.zeros((pos.numel(), 64), dtype=torch.float32, device=pos.device),
    )
    monkeypatch.setattr(decoder, "PreparedMHC", lambda *a, **kw: Component(plan=object()))

    seen_val_gathers = []
    seen_suffix_gathers = []

    class FakeGatherShards:
        single = False
        dma = False
        plan = None

        def __init__(self, src, out, **kw):
            self.name = kw["name"]
            self.src_shape = tuple(src.shape)
            self.out_shape = tuple(out.shape)

        def functional(self, **kw):
            return self

    class FakeChunkedGather:
        def __init__(self, src, out, **kw):
            self.src_shape = tuple(src.shape)
            self.out_shape = tuple(out.shape)

        def functional(self, **kw):
            return self

    monkeypatch.setattr(decoder, "PreparedGatherShards", FakeGatherShards)
    monkeypatch.setattr(decoder, "PreparedChunkedGatherShards", FakeChunkedGather)

    meta = Metadata.allocate(tokens, 1, 160, "cpu", max_query_tokens=tokens)
    packing = PreparedDecoderSuffix(meta)
    src_indices = ds_attention.AttentionIndices(
        meta, mode="prefill", tokens=tokens, page_size=128 + tokens, width=128, ratios=(1, 2)
    )
    sfx_indices = ds_attention.AttentionIndices(
        packing.metadata, mode="prefill", tokens=128, page_size=256, width=128,
        replay_start=packing.replay_start, ratios=(1,),
    )
    # Allocate lightweight minimal source scratch view matching tokens.
    src_scr = SimpleNamespace(
        residual=torch.zeros((tokens, 4, 5120), dtype=torch.bfloat16),
        mlp_output=torch.zeros((s, 5120), dtype=torch.bfloat16),
        pre_mix=torch.zeros((tokens, 4), dtype=torch.float32),
        attn_mix=torch.zeros((tokens, 4), dtype=torch.float32),
        attention=SimpleNamespace(
            output=torch.zeros((tokens, 5120), dtype=torch.bfloat16),
            index_keys=torch.zeros((tokens, 128), dtype=torch.bfloat16),
            normalized_keys=torch.zeros((tokens, 128), dtype=torch.bfloat16),
        ),
        attention_hc=SimpleNamespace(
            normalized=torch.zeros((tokens, 5120), dtype=torch.bfloat16),
            rows=lambda n: SimpleNamespace(normalized=torch.zeros((n, 5120), dtype=torch.bfloat16)),
        ),
        ffn_hc=SimpleNamespace(
            residual=torch.zeros((s, 4, 5120), dtype=torch.bfloat16),
            post=torch.zeros((s, 4), dtype=torch.float32),
            comb=torch.zeros((s, 4, 4), dtype=torch.float32),
        ),
    )
    layer_w = SimpleNamespace(
        attention={
            "compressor": {"norm": torch.ones(512, dtype=torch.float32), "values": torch.ones((512, 5120), dtype=torch.bfloat16)},
            "index_k": torch.ones((128, 512), dtype=torch.bfloat16),
            "index_norm": torch.ones(128, dtype=torch.bfloat16),
        },
        residual={k: torch.ones(4) for k in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm.weight")},
    )
    examples = {}
    ref_b = decoder.PreparedDecoderBoundary(
        20, layer_w, src_scr, None, packing, pool, rotary, carry=False,
        session=None, name="ref_b", workspace=None, window_metadata=sfx_indices,
        source_indices=src_indices, row_shard_dma=False, examples=examples,
    )
    rank_boundaries = [
        decoder.PreparedDecoderBoundary(
            20, layer_w, src_scr, None, packing, pool, rotary, carry=False,
            session=None, name=f"sh_b_{r}", workspace=None, window_metadata=sfx_indices,
            source_indices=src_indices, row_shard_dma=True, shard=TensorParallelShard(r, world),
            runtime="rt", examples=examples,
        )
        for r in range(world)
    ]
    for b in rank_boundaries:
        assert b.shard_csa1 is True
        assert b.norm_gather_0 is None and b.norm_gather_1 is None
        assert b.res_gather_0 is None and b.state_gather_0 is None
        assert b.val_gather_0.src_shape == (s, 512) and b.val_gather_0.out_shape == (h, 512)
        assert b.suffix_gather.src_shape == (128, 25648) and b.suffix_gather.out_shape == (world * 128, 25648)

    packing.functional()
    src_indices.functional()

    # Construct full ground-truth activations with -0.0 BF16 and NaN-like low-halfword FP32 bit patterns!
    gen = torch.Generator().manual_seed(1234 + tokens)
    full_res = torch.randn((tokens, 4, 5120), generator=gen).to(torch.bfloat16)
    full_norm = torch.randn((tokens, 5120), generator=gen).to(torch.bfloat16)
    # Inject -0.0 BF16 (0x8000) across boundaries!
    full_norm.view(torch.uint16)[::17, 0] = 0x8000
    full_res.view(torch.uint16)[::23, 1, 2] = 0x8000
    full_post = torch.randn((tokens, 4), generator=gen, dtype=torch.float32)
    full_comb = torch.randn((tokens, 4, 4), generator=gen, dtype=torch.float32)
    full_mix = torch.randn((tokens, 4), generator=gen, dtype=torch.float32)
    # Inject FP32 values whose low 16-bit halfword encodes BF16 NaN (0x7FC1) / -0.0 (0x8000)!
    full_post.view(torch.int32)[:, 0] = (full_post.view(torch.int32)[:, 0] & ~0xFFFF) | 0x7FC1
    full_comb.view(torch.int32)[:, 1, 2] = (full_comb.view(torch.int32)[:, 1, 2] & ~0xFFFF) | 0x8000
    full_mix.view(torch.int32)[:, 3] = -2147483648  # FP32 -0.0

    def _ref_mhc(*inc, pre_mix, pre_out):
        del inc, pre_mix
        pre_out.copy_(full_mix)
        return full_res, full_post, full_comb, full_norm

    ref_b.mhc = _ref_mhc
    ref_kv_seen = []
    ref_b.global_kv = Component(ratio=1, call=lambda n, idxr, m, g, **kw: ref_kv_seen.append(n.clone()))

    sh_kv_seen = [[] for _ in range(world)]
    for r, b in enumerate(rank_boundaries):
        s0_a, s0_b = r * s, h + r * s

        def make_mhc(ra=s0_a, rb=s0_b):
            call_i = [0]

            def _mhc(*inc, pre_mix, pre_out):
                if call_i[0] == 0:
                    call_i[0] = 1
                    pre_out.copy_(full_mix[ra : ra + s])
                    return full_res[ra : ra + s], full_post[ra : ra + s], full_comb[ra : ra + s], full_norm[ra : ra + s]
                call_i[0] = 0
                pre_out.copy_(full_mix[rb : rb + s])
                return full_res[rb : rb + s], full_post[rb : rb + s], full_comb[rb : rb + s], full_norm[rb : rb + s]

            return _mhc

        b.mhc = make_mhc()
        b.global_kv.compressor.project_shard = lambda norm_sh: (norm_sh[:, :512].contiguous(), None)
        b.global_kv.run_half = (
            lambda vh, gh, idxr, m, g, *, half_idx, rr=r, **kw: sh_kv_seen[rr].append((half_idx, vh.clone())) or vh
        )

    tail_cases = (37, 128, 500, 4096, 7500, 8192) if tokens == 8192 else (37, 128, 500, 4096, 7500, 8192, 16384)
    for live in tail_cases:
        meta.starts.fill_(127)
        meta.lengths.fill_(live)
        meta.query_start_loc.copy_(torch.tensor([0, live], dtype=torch.int32))
        meta.live_counts.copy_(torch.tensor([live, 1], dtype=torch.int32))
        meta.positions.fill_(-1)
        meta.positions[:live] = torch.arange(127, 127 + live, dtype=torch.int32)
        meta.token_ids.copy_(torch.arange(tokens, dtype=torch.int64) % 50000)
        meta.sample_indices.fill_(live - 1)
        ind = src_indices(meta)

        ref_kv_seen.clear()
        ref_sfx, (ref_r, ref_p, ref_c, ref_n), ref_m = ref_b((full_res,), full_mix, meta, ind, "idxr")

        # First collect each rank's local val_a_sh, val_b_sh, and local_128 suffix contribution, then simulate gather!
        val_a_all = torch.cat([full_norm[r * s : (r + 1) * s, :512] for r in range(world)], dim=0)
        val_b_all = torch.cat([full_norm[h + r * s : h + (r + 1) * s, :512] for r in range(world)], dim=0)
        local_128_per_rank = []
        for r, b in enumerate(rank_boundaries):
            sh_kv_seen[r].clear()
            b.val_gather_0 = lambda sh, **kw: (seen_val_gathers.append(sh.shape) or val_a_all)
            b.val_gather_1 = lambda sh, **kw: val_b_all
            captured_local = []
            b.suffix_gather = lambda loc, **kw: (
                captured_local.append(loc.clone()) or loc.repeat(world, 1)
            )
            b((full_res[:s], full_res[:s]), (full_mix[:s], full_mix[:s]), meta, ind, "idxr")
            local_128_per_rank.append(captured_local[0])
            assert torch.equal(
                torch.cat([sh_kv_seen[r][0][1], sh_kv_seen[r][1][1]], dim=0).view(torch.uint16),
                ref_kv_seen[0][:, :512].view(torch.uint16),
            )

        gathered_all_ranks = torch.cat(local_128_per_rank, dim=0)
        assert gathered_all_ranks.shape == (world * 128, 25648)
        seen_suffix_gathers.append(gathered_all_ranks.shape)

        for r, b in enumerate(rank_boundaries):
            b.suffix_gather = lambda loc, **kw: gathered_all_ranks
            sh_sfx, (sh_r, sh_p, sh_c, sh_n), sh_m = b(
                (full_res[:s], full_res[:s]), (full_mix[:s], full_mix[:s]), meta, ind, "idxr"
            )
            assert torch.equal(sh_sfx.indices, ref_sfx.indices)
            assert torch.equal(sh_sfx.live, ref_sfx.live)
            for got, exp in ((sh_r, ref_r), (sh_p, ref_p), (sh_c, ref_c), (sh_n, ref_n), (sh_m, ref_m)):
                assert torch.equal(got.view(torch.uint8), exp.view(torch.uint8))

        # Sensitivity: corrupting gathered_all_ranks on the owner of the final live token fails bitwise parity.
        last_idx = live - 1
        last_owner = (last_idx // s) if last_idx < h else ((last_idx - h) // s)
        last_slot = min(127, live - 1)
        bad_gathered = gathered_all_ranks.clone()
        bad_gathered.view(world, 128, 25648).view(torch.uint16)[last_owner, last_slot, 0] ^= 1
        rank_boundaries[0].suffix_gather = lambda loc, **kw: bad_gathered
        _, (bad_r, _, _, _), _ = rank_boundaries[0](
            (full_res[:s], full_res[:s]), (full_mix[:s], full_mix[:s]), meta, ind, "idxr"
        )
        assert not torch.equal(bad_r.view(torch.uint8), ref_r.view(torch.uint8))


@pytest.mark.skipif(
    not _RUN_CUDA_TESTS or not torch.cuda.is_available(),
    reason="requires CUDA GPU and FLARE_RUN_CUDA_TESTS=1",
)
@pytest.mark.parametrize(("layer_idx", "ratio"), [(2, 2), (20, 1)])
@pytest.mark.parametrize("start_pos", [0, 127])
def test_mla_wo_half_split_global_kv_bit_exact_cuda(layer_idx: int, ratio: int, start_pos: int):
    from b12x.preparation import autotune_effort as b12x_autotune_effort
    from hyperflare_recipes.components.mxfp4_indexer import PreparedMXFP4Indexer
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    cfg = _qualified_model_config()
    tokens, world = 1024, 4
    h = tokens // 2
    st = h // world
    gen = torch.Generator(device=device).manual_seed(202600 + layer_idx * 10 + start_pos)

    weights = {
        "compressor": {
            "norm": (torch.rand(512, dtype=torch.float32, device=device, generator=gen) + 0.5).contiguous(),
            "values": (torch.randn((512, 5120), dtype=torch.bfloat16, device=device, generator=gen) * 0.05).contiguous(),
        },
        "index_k": (torch.randn((128, 512), dtype=torch.bfloat16, device=device, generator=gen) * 0.05).contiguous(),
        "index_norm": (torch.rand(128, dtype=torch.bfloat16, device=device, generator=gen) + 0.5).contiguous(),
    }
    if ratio == 2:
        weights["compressor"]["gates"] = (
            torch.randn((512, 5120), dtype=torch.bfloat16, device=device, generator=gen) * 0.05
        ).contiguous()

    pool_ref = ContinuationPool.allocate(cfg, 4, 32, device, dspark=False)
    pool_sh = ContinuationPool.allocate(cfg, 4, 32, device, dspark=False)
    rotary = ds_attention.Rotary(cfg, device)
    meta = Metadata.allocate(tokens, 1, 32, device, max_query_tokens=tokens)
    scr = ds_attention.AttentionScratch.allocate(tokens, world, device, candidate_tokens=128)
    src = (torch.randn((tokens, 5120), dtype=torch.bfloat16, device=device, generator=gen) * 0.25).contiguous()

    with b12x_autotune_effort("off"):
        session = RecordedPreparationSession(
            device=device, checkpoint=f"test-gkv-{layer_idx}-{start_pos}",
            workload_schema="verify", numerical_policy="default", autotune_effort="off",
        )
        session.__enter__()
        arena = ScratchArena(device, overlap=False)
        session.register_scratch(arena)
        wm = ds_attention.AttentionIndices(
            meta, mode="prefill", tokens=tokens, page_size=128 + tokens, width=128, ratios=(ratio,),
        )
        geom_decl = wm.compressed(ratio)
        idx_ref = PreparedMXFP4Indexer(
            scr.index_query[:128], scr.index_weights[:128],
            pool_ref.pages[f"layer{layer_idx}.index_page"],
            meta.token_page_table[:128], geom_decl.lengths[:128], geom_decl.active_width,
            scr.selected[:128], page_size=128 // ratio, session=session, name="idx_ref",
            workspace=arena, mode="prefill",
        ).functional(session=session)
        idx_sh = PreparedMXFP4Indexer(
            scr.index_query[:128], scr.index_weights[:128],
            pool_sh.pages[f"layer{layer_idx}.index_page"],
            meta.token_page_table[:128], geom_decl.lengths[:128], geom_decl.active_width,
            scr.selected[:128], page_size=128 // ratio, session=session, name="idx_sh",
            workspace=arena, mode="prefill",
        ).functional(session=session)
        kv_ref = ds_attention.PreparedGlobalKV(
            layer_idx, weights, src, scr, meta, pool_ref, rotary,
            session=session, name="kv_ref", indices=wm, examples={},
        )
        kv_sh = ds_attention.PreparedGlobalKV(
            layer_idx, weights, src, scr, meta, pool_sh, rotary,
            session=session, name="kv_sh", indices=wm, examples={},
            proj_tokens=st, compress_tokens=h,
        )
        wm.functional()
        session.tune(show_progress=False)

    try:
        init_pv = torch.randn((4, 512), dtype=torch.float32, device=device, generator=gen)
        init_pg = torch.randn((4, 512), dtype=torch.float32, device=device, generator=gen)
        init_pos = start_pos - 1 if (start_pos > 0 and (start_pos - 1) % 2 == 0) else -1

        def reset_pools():
            for p in (pool_ref, pool_sh):
                p.backing.fill_(0x5A)
                p.pages[f"layer{layer_idx}.latent_page"].zero_()
                p.pages[f"layer{layer_idx}.index_page"].zero_()
                if ratio == 2:
                    p.region(f"layer{layer_idx}.pending_values").copy_(init_pv)
                    p.region(f"layer{layer_idx}.pending_gates").copy_(init_pg)
                    p.region(f"layer{layer_idx}.pending_position").fill_(init_pos)

        # Cover inactive (0), partial in Half A (1, 127, h), partial in Half B (h+1, h+189, 2h-1), and full (2h).
        for live in (0, 1, 127, h, h + 1, h + 189, tokens - 1, tokens):
            reset_pools()
            meta.starts.fill_(start_pos)
            meta.lengths.fill_(live)
            meta.query_start_loc.copy_(torch.tensor([0, live], dtype=torch.int32, device=device))
            meta.live_counts.copy_(torch.tensor([live, 1 if live > 0 else 0], dtype=torch.int32, device=device))
            meta.state_rows.fill_(1 if live > 0 else 0)
            meta.requests.zero_()
            meta.positions.fill_(-1)
            if live > 0:
                meta.positions[:live] = torch.arange(start_pos, start_pos + live, dtype=torch.int32, device=device)
            for p_idx in range(32):
                meta.page_table[0, p_idx] = p_idx
            if meta.token_page_table.stride(0) != 0:
                meta.token_page_table.copy_(meta.page_table[0].unsqueeze(0).expand(tokens, -1))

            ind = wm(meta)
            geom = ind.compressed(ratio)
            angles = ind.angles[(True, ratio)]

            vals_ref, gates_ref = kv_ref(src, idx_ref, meta, geom, angles=angles)

            # 1. Operator-level bit-exactness: feeding identical pre-projected (values, gates)
            #    into two-half compress_half + rotary + writer + write_keys is 100% bitwise equal!
            pending_exact = kv_sh.compressor.init_pending(meta, device)
            lat_a, sl_a = kv_sh.compressor.compress_half(
                vals_ref[:h], None if gates_ref is None else gates_ref[:h],
                meta, geom.slots[:h], half_idx=0, pending=pending_exact,
            )
            lat_b, sl_b = kv_sh.compressor.compress_half(
                vals_ref[h:], None if gates_ref is None else gates_ref[h:],
                meta, geom.slots[h:], half_idx=1, pending=pending_exact,
            )
            keys_full = kv_ref.index_key(torch.cat((lat_a, lat_b), dim=0))
            keys_a = ds_attention._index_norm_and_rotate(
                keys_full[:h], kv_sh.index_norm_weight, lat_a, angles[:h], kv_sh.eps)
            keys_b = ds_attention._index_norm_and_rotate(
                keys_full[h:], kv_sh.index_norm_weight, lat_b, angles[h:], kv_sh.eps)
            idx_sh.write_keys(keys_a, sl_a)
            idx_sh.write_keys(keys_b, sl_b)
            kv_sh.writer(lat_a, sl_a)
            kv_sh.writer(lat_b, sl_b)

            assert torch.equal(
                pool_sh.pages[f"layer{layer_idx}.latent_page"].view(torch.uint8),
                pool_ref.pages[f"layer{layer_idx}.latent_page"].view(torch.uint8),
            ), f"latent_page mismatch at ratio={ratio} start={start_pos} live={live}"
            assert torch.equal(
                pool_sh.pages[f"layer{layer_idx}.index_page"].view(torch.uint8),
                pool_ref.pages[f"layer{layer_idx}.index_page"].view(torch.uint8),
            ), f"index_page mismatch at ratio={ratio} start={start_pos} live={live}"
            assert torch.equal(
                pool_sh.backing.view(torch.uint8),
                pool_ref.backing.view(torch.uint8),
            ), f"ContinuationPool pending carry mismatch at ratio={ratio} start={start_pos} live={live}"

            # 2. End-to-end shard projection (`4 x [st, 5120]`) + `kv_sh.run_half` parity:
            pool_sh.backing.fill_(0x5A)
            pool_sh.pages[f"layer{layer_idx}.latent_page"].zero_()
            pool_sh.pages[f"layer{layer_idx}.index_page"].zero_()
            if ratio == 2:
                pool_sh.region(f"layer{layer_idx}.pending_values").copy_(init_pv)
                pool_sh.region(f"layer{layer_idx}.pending_gates").copy_(init_pg)
                pool_sh.region(f"layer{layer_idx}.pending_position").fill_(init_pos)

            val_a_parts, gate_a_parts = [], []
            val_b_parts, gate_b_parts = [], []
            for r in range(world):
                va, ga = kv_sh.compressor.project_shard(src[r * st : (r + 1) * st])
                vb, gb = kv_sh.compressor.project_shard(src[h + r * st : h + (r + 1) * st])
                val_a_parts.append(va)
                val_b_parts.append(vb)
                if ratio == 2:
                    gate_a_parts.append(ga)
                    gate_b_parts.append(gb)
            val_a = torch.cat(val_a_parts, dim=0)
            val_b = torch.cat(val_b_parts, dim=0)
            gate_a = torch.cat(gate_a_parts, dim=0) if ratio == 2 else None
            gate_b = torch.cat(gate_b_parts, dim=0) if ratio == 2 else None

            torch.testing.assert_close(
                torch.cat((val_a, val_b), dim=0), vals_ref,
                atol=1e-4 if ratio == 2 else 2e-2, rtol=1e-4 if ratio == 2 else 1e-2,
            )
            pending = kv_sh.compressor.init_pending(meta, device)
            lat_sh_a = kv_sh.run_half(val_a, gate_a, idx_sh, meta, geom, half_idx=0, pending=pending, angles=angles)
            lat_sh_b = kv_sh.run_half(val_b, gate_b, idx_sh, meta, geom, half_idx=1, pending=pending, angles=angles)
            lat_sh_full = torch.cat((lat_sh_a, lat_sh_b), dim=0).float()
            lat_ref_full = torch.cat((lat_a, lat_b), dim=0).float()
            lat_rel_rms = (lat_sh_full - lat_ref_full).square().mean().sqrt().item() / max(
                lat_ref_full.square().mean().sqrt().item(), 1e-12
            )
            # `mla_compress` rounds gated-pooled values to BF16 prior to RMSNorm, so a single half-ULP
            # GEMM tile difference (`M=128` vs `M=1024`) yields ~1.5e-4 relative RMS across 524,288 elements.
            assert lat_rel_rms < (5e-4 if ratio == 2 else 5e-3), f"lat_rel_rms={lat_rel_rms}"
            torch.testing.assert_close(lat_sh_full, lat_ref_full, atol=4e-2, rtol=2e-2)
            if ratio == 2:
                assert torch.equal(
                    pool_sh.region(f"layer{layer_idx}.pending_position"),
                    pool_ref.region(f"layer{layer_idx}.pending_position"),
                )
                torch.testing.assert_close(
                    pool_sh.region(f"layer{layer_idx}.pending_values"),
                    pool_ref.region(f"layer{layer_idx}.pending_values"),
                    atol=1e-4, rtol=1e-4,
                )
                torch.testing.assert_close(
                    pool_sh.region(f"layer{layer_idx}.pending_gates"),
                    pool_ref.region(f"layer{layer_idx}.pending_gates"),
                    atol=1e-4, rtol=1e-4,
                )

        # Sensitivity on odd start_pos=127 (`h-1` carry into Half B row 0!): perturbing `val_a[h - 1]` MUST
        # alter Half B's first emitted compressed KV slot!
        if ratio == 2 and start_pos == 127:
            bad_val_a = val_a.clone()
            bad_val_a[h - 1] += 10.0
            pool_sh.pages[f"layer{layer_idx}.latent_page"].zero_()
            pool_sh.region(f"layer{layer_idx}.pending_values").copy_(init_pv)
            pool_sh.region(f"layer{layer_idx}.pending_gates").copy_(init_pg)
            pool_sh.region(f"layer{layer_idx}.pending_position").fill_(init_pos)
            pending_bad = kv_sh.compressor.init_pending(meta, device)
            kv_sh.run_half(bad_val_a, gate_a, idx_sh, meta, geom, half_idx=0, pending=pending_bad, angles=angles)
            kv_sh.run_half(val_b, gate_b, idx_sh, meta, geom, half_idx=1, pending=pending_bad, angles=angles)
            assert not torch.equal(
                pool_sh.pages[f"layer{layer_idx}.latent_page"].view(torch.uint8),
                pool_ref.pages[f"layer{layer_idx}.latent_page"].view(torch.uint8),
            )
    finally:
        session.__exit__(None, None, None)


def _mxfp4_quantized_values(value: torch.Tensor) -> torch.Tensor:
    groups = value.float().reshape(*value.shape[:-1], 4, 32)
    maximum = groups.abs().amax(-1).clamp_min(6 * 2.0**-126)
    bits = (maximum / 6).contiguous().view(torch.int32)
    scale_bytes = ((bits >> 23) + ((bits & 0x7FFFFF) != 0)).to(torch.uint8)
    scale = (scale_bytes.int() << 23).view(torch.float32)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=value.device)
    even_first = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=value.device)
    nearest = ((groups / scale[..., None]).abs()[..., None] - levels[even_first]).abs().argmin(-1)
    magnitude = levels[even_first[nearest]] * scale[..., None]
    return torch.where(torch.signbit(groups), -magnitude, magnitude).bfloat16().reshape_as(value)


def _mxfp4_scores(query: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    q_mxfp4, k_mxfp4 = _mxfp4_quantized_values(query), _mxfp4_quantized_values(keys)
    reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    try:
        dot = torch.einsum("rhd,kd->rhk", q_mxfp4, k_mxfp4)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduction
    products = (dot.relu() * weights[..., None]).float()
    total = torch.zeros_like(products[:, 0])
    for head in range(products.shape[1]):
        total.add_(products[:, head])
    return total.bfloat16().float()


@pytest.mark.skipif(
    not _RUN_CUDA_TESTS or not torch.cuda.is_available(),
    reason="requires CUDA GPU and FLARE_RUN_CUDA_TESTS=1",
)
@pytest.mark.parametrize("st", [128, 256, 512, 768, 1024, 2048])
def test_shard_local_mxfp4_indexer_512_query_rows_bit_exact_cuda(st: int):
    from b12x.preparation import autotune_effort as b12x_autotune_effort
    from hyperflare_recipes.components.mxfp4_indexer import PreparedMXFP4Indexer
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    world = 4
    h = st * world
    tokens = 2 * h
    page_size = 64
    max_pages = 256
    # Live partial tail in Half B (tokens - 73 active tokens) starting after 1024 prefix tokens (512 compressed keys).
    live = tokens - 73
    pos = torch.arange(tokens, dtype=torch.int64, device=device) + 1024
    lengths = torch.where(torch.arange(tokens, device=device) < live, (pos + 1) // 2, 0).to(torch.int32)
    records = int(lengths.max().item()) + 64
    gen = torch.Generator(device=device).manual_seed(9000 + st)

    iq_full = (torch.randn((tokens, 32, 128), dtype=torch.bfloat16, device=device, generator=gen) * 0.25).contiguous()
    iw_full = (torch.randn((tokens, 32), dtype=torch.bfloat16, device=device, generator=gen) / 32.0).contiguous()
    keys = (torch.randn((records, 128), dtype=torch.bfloat16, device=device, generator=gen) * 0.25).contiguous()
    cache = torch.zeros((max_pages, page_size * 68), dtype=torch.uint8, device=device)
    pt = torch.arange(max_pages, dtype=torch.int32, device=device)[None, :].expand(tokens, -1).contiguous()
    active_full = lengths.amax().view(1)
    active_a = lengths[:h].amax().view(1)
    out_dummy = torch.empty((tokens, 512), dtype=torch.int32, device=device)

    with b12x_autotune_effort("off"):
        session = RecordedPreparationSession(
            device=device, checkpoint=f"test-idx512-{st}",
            workload_schema="verify", numerical_policy="default", autotune_effort="off",
        )
        session.__enter__()
        arena = ScratchArena(device, overlap=False)
        session.register_scratch(arena)

        ref_sels = []
        for first in range(0, tokens, ds_attention.INDEXER_QUERY_ROWS):
            last = min(tokens, first + ds_attention.INDEXER_QUERY_ROWS)
            ref_sels.append(
                PreparedMXFP4Indexer(
                    iq_full[first:last], iw_full[first:last], cache, pt[first:last],
                    lengths[first:last], active_full, out_dummy[first:last],
                    page_size=page_size, session=session, name=f"ref_{first}", workspace=arena, mode="prefill",
                ).functional(session=session)
            )

        sh_sels = []
        for first in range(0, st, ds_attention.INDEXER_QUERY_ROWS):
            last = min(st, first + ds_attention.INDEXER_QUERY_ROWS)
            sh_sels.append(
                PreparedMXFP4Indexer(
                    iq_full[first:last], iw_full[first:last], cache, pt[first:last],
                    lengths[first:last], active_full, out_dummy[first:last],
                    page_size=page_size, session=session, name=f"sh_{first}", workspace=arena, mode="prefill",
                ).functional(session=session)
            )
        session.tune(show_progress=False)

    try:
        slots = torch.arange(records, dtype=torch.int64, device=device)
        sh_sels[0].write_keys(keys, slots)

        ref_parts = []
        for ordinal, sel in enumerate(ref_sels):
            f = ordinal * ds_attention.INDEXER_QUERY_ROWS
            l = min(tokens, f + ds_attention.INDEXER_QUERY_ROWS)
            ref_parts.append(sel(iq_full[f:l], iw_full[f:l], pt[f:l], lengths[f:l], active_full))
        ref_out = torch.cat(ref_parts, dim=0)

        half_a_ranks, half_b_ranks = [], []
        for r in range(world):
            for half_idx, base, act, bucket in (
                (0, r * st, active_a, half_a_ranks),
                (1, h + r * st, active_full, half_b_ranks),
            ):
                del half_idx
                outs = []
                for ordinal, sel in enumerate(sh_sels):
                    f = ordinal * ds_attention.INDEXER_QUERY_ROWS
                    l = min(st, f + ds_attention.INDEXER_QUERY_ROWS)
                    outs.append(
                        sel(iq_full[base + f : base + l], iw_full[base + f : base + l],
                            pt[base + f : base + l], lengths[base + f : base + l], act)
                    )
                bucket.append(outs[0] if len(outs) == 1 else torch.cat(outs, dim=0))
        sh_out = torch.cat(half_a_ranks + half_b_ranks, dim=0)

        # Verify inactive tail rows emit exact -1 padding and valid rows are strictly ascending.
        assert torch.all(sh_out[live:] == -1)
        assert torch.equal(sh_out[live:], ref_out[live:])
        valid_sh = sh_out >= 0
        valid_ref = ref_out >= 0
        assert ((sh_out[:, 1:] > sh_out[:, :-1]) | ~valid_sh[:, 1:]).all()
        assert torch.equal(valid_sh.sum(1), valid_ref.sum(1))

        # Verify 100% bit-exact MXFP4 top-512 score multisets across all rows (equal-score cutoff ties allowed).
        sample_rows = (
            torch.arange(tokens, device=device)
            if tokens <= 2048
            else torch.cat((
                torch.arange(0, 256, device=device),
                torch.arange(h - 128, h + 128, device=device),
                torch.arange(live - 128, tokens, device=device),
            ))
        )
        scores = _mxfp4_scores(iq_full[sample_rows], keys, iw_full[sample_rows]).masked_fill(
            torch.arange(records, device=device)[None, :] >= lengths[sample_rows, None], -torch.inf
        )
        sh_sub = sh_out[sample_rows]
        ref_sub = ref_out[sample_rows]
        sh_scores = scores.gather(1, sh_sub.clamp_min(0).long()).masked_fill(sh_sub < 0, -torch.inf)
        ref_scores = scores.gather(1, ref_sub.clamp_min(0).long()).masked_fill(ref_sub < 0, -torch.inf)
        exp_scores = scores.topk(512, dim=1).values
        assert torch.equal(sh_scores.sort(dim=1).values, ref_scores.sort(dim=1).values)
        assert torch.equal(sh_scores.sort(dim=1).values, exp_scores.sort(dim=1).values)

        # Every index strictly above the 512th cutoff score must appear identically in sh_out and ref_out!
        cutoff = exp_scores[:, -1:]
        above_sh = torch.where(sh_scores > cutoff, sh_sub, -1).sort(dim=1).values
        above_ref = torch.where(ref_scores > cutoff, ref_sub, -1).sort(dim=1).values
        assert torch.equal(above_sh, above_ref)

        # Sensitivity: perturbing rank 1's query alters rank 1's selected indices and score multiset.
        bad_iq = iq_full[st : st + min(st, ds_attention.INDEXER_QUERY_ROWS)].neg().contiguous()
        bad_sel = sh_sels[0](bad_iq, iw_full[st : st + bad_iq.shape[0]],
                             pt[st : st + bad_iq.shape[0]], lengths[st : st + bad_iq.shape[0]], active_a)
        assert not torch.equal(bad_sel, half_a_ranks[1][:bad_iq.shape[0]])
    finally:
        session.__exit__(None, None, None)


@pytest.mark.skipif(
    not _RUN_CUDA_TESTS or not torch.cuda.is_available(),
    reason="requires CUDA GPU and FLARE_RUN_CUDA_TESTS=1",
)
@pytest.mark.parametrize(("start_pos", "live"), [(0, 1024), (127, 1024), (127, 700)])
def test_mla_wo_row_shard_attention_matches_tp4_reference_cuda(start_pos: int, live: int):
    from b12x.gemm import wo_projection
    from b12x.gemm.blockscaled import pack_weight
    from b12x.preparation import autotune_effort as b12x_autotune_effort
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    cfg = _qualified_model_config()
    layer_idx = 2
    tokens, world = 1024, 4
    h = tokens // 2
    st = h // world
    ratio = cfg.text["compress_ratios"][layer_idx]
    assert ratio == 2 and layer_idx in ds_config.row_shard_attention_encoder_layers(cfg)

    gen = torch.Generator(device=device).manual_seed(20261125 + start_pos + live)

    def rand_mxfp8_raw(rows: int, cols: int):
        w_bf16 = torch.randn((rows, cols), dtype=torch.bfloat16, device=device, generator=gen) * 0.02
        vals = w_bf16.to(torch.float8_e4m3fn)
        scales_32x32 = torch.full((rows // 32, cols // 32), 127, dtype=torch.uint8, device=device)
        return vals, scales_32x32

    def pack_mxfp8(vals: torch.Tensor, scales_32x32: torch.Tensor):
        return pack_weight(vals, scales_32x32.repeat_interleave(32, dim=0), recipe="mxfp8")

    qkv_vals, qkv_s = rand_mxfp8_raw(1792, 5120)
    qb_vals, qb_s = rand_mxfp8_raw(32768, 1280)
    iq_vals, iq_s = rand_mxfp8_raw(4096, 1280)
    woa_vals, woa_s = rand_mxfp8_raw(8192, 4096)
    wob_vals, wob_s = rand_mxfp8_raw(5120, 8192)
    q_norm = torch.ones(1280, dtype=torch.bfloat16, device=device)
    kv_norm = torch.ones(512, dtype=torch.bfloat16, device=device)
    sink_full = torch.linspace(-1.5, 1.5, 64, dtype=torch.float32, device=device)
    comp_w = {
        "norm": (torch.rand(512, dtype=torch.float32, device=device, generator=gen) + 0.5).contiguous(),
        "values": (torch.randn((512, 5120), dtype=torch.bfloat16, device=device, generator=gen) * 0.04).contiguous(),
        "gates": (torch.randn((512, 5120), dtype=torch.bfloat16, device=device, generator=gen) * 0.04).contiguous(),
    }
    ik_w = (torch.randn((128, 512), dtype=torch.bfloat16, device=device, generator=gen) * 0.04).contiguous()
    inorm_w = (torch.rand(128, dtype=torch.bfloat16, device=device, generator=gen) + 0.5).contiguous()
    iw_w = (torch.randn((32, 5120), dtype=torch.bfloat16, device=device, generator=gen) * 0.04).contiguous()
    iq_packed = pack_mxfp8(iq_vals, iq_s)
    qkv_packed = pack_mxfp8(qkv_vals, qkv_s)

    tp_weights = []
    for r in range(world):
        h0, h1 = r * 8192, (r + 1) * 8192
        g0, g1 = r * 2048, (r + 1) * 2048
        tp_weights.append({
            "qkv_a": qkv_packed,
            "q_b": pack_mxfp8(qb_vals[h0:h1], qb_s[h0 // 32 : h1 // 32]),
            "output": wo_projection.pack_weights(
                woa_vals[g0:g1], woa_s[g0 // 32 : g1 // 32].view(torch.float8_e8m0fnu),
                wob_vals[:, g0:g1].contiguous(), wob_s[:, g0 // 32 : g1 // 32].contiguous().view(torch.float8_e8m0fnu),
                groups=2, group_width=4096, rank=1024, hidden=5120, block_size=(32, 32),
            ),
            "q_norm": q_norm, "kv_norm": kv_norm,
            "sink": sink_full[r * 16 : (r + 1) * 16].contiguous(),
            "compressor": comp_w, "index_k": ik_w, "index_norm": inorm_w,
            "index_q": iq_packed, "index_weights": iw_w,
        })
    rep_weights = {
        "q_b_full": pack_mxfp8(qb_vals, qb_s),
        "output_full": wo_projection.pack_weights(
            woa_vals, woa_s.view(torch.float8_e8m0fnu), wob_vals, wob_s.view(torch.float8_e8m0fnu),
            groups=8, group_width=4096, rank=1024, hidden=5120, block_size=(32, 32),
        ),
        "sink_full": sink_full,
    }

    pool_ref = ContinuationPool.allocate(cfg, 4, 32, device, dspark=False)
    pool_sh = ContinuationPool.allocate(cfg, 4, 32, device, dspark=False)
    init_pv = torch.randn((4, 512), dtype=torch.float32, device=device, generator=gen)
    init_pg = torch.randn((4, 512), dtype=torch.float32, device=device, generator=gen)
    init_pos = start_pos - 1 if (start_pos > 0 and (start_pos - 1) % 2 == 0) else -1
    for p in (pool_ref, pool_sh):
        p.backing.zero_()
        p.pages["layer2.latent_page"].zero_()
        p.pages["layer2.index_page"].zero_()
        p.window(layer_idx).zero_()
        p.region("layer2.pending_values").copy_(init_pv)
        p.region("layer2.pending_gates").copy_(init_pg)
        p.region("layer2.pending_position").fill_(init_pos)

    rotary = ds_attention.Rotary(cfg, device)
    meta = Metadata.allocate(tokens, 1, 64, device, max_query_tokens=tokens)
    meta.starts.fill_(start_pos)
    meta.lengths.fill_(live)
    meta.query_start_loc.copy_(torch.tensor([0, live], dtype=torch.int32, device=device))
    meta.live_counts.copy_(torch.tensor([live, 1], dtype=torch.int32, device=device))
    meta.state_rows.fill_(1)
    meta.requests.zero_()
    meta.positions.fill_(-1)
    meta.positions[:live] = torch.arange(start_pos, start_pos + live, dtype=torch.int32, device=device)
    meta.page_table.zero_()
    for p_idx in range(32):
        meta.page_table[0, p_idx] = p_idx
    if meta.token_page_table.stride(0) != 0:
        meta.token_page_table.copy_(meta.page_table[0].unsqueeze(0).expand(tokens, -1))

    x = (torch.randn((tokens, 5120), dtype=torch.bfloat16, device=device, generator=gen) * 0.25).contiguous()
    init_sel = (torch.full((tokens, 512), -1, dtype=torch.int32, device=device), None, None)

    with b12x_autotune_effort("off"):
        session = RecordedPreparationSession(
            device=device, checkpoint=f"test-mlawo-{start_pos}-{live}",
            workload_schema="verify", numerical_policy="default", autotune_effort="off",
        )
        session.__enter__()
        arena = ScratchArena(device, overlap=False)
        session.register_scratch(arena)
        examples = {}
        wm = ds_attention.AttentionIndices(
            meta, mode="prefill", tokens=tokens, page_size=128 + tokens, width=128, ratios=(ratio,),
        )
        scr = ds_attention.AttentionScratch.allocate(tokens, 4, device, candidate_tokens=128)
        ref_attns = [
            ds_attention.PreparedAttention(
                layer_idx, tp_weights[r], x, scr, meta, pool_ref, rotary, mode="prefill", world=4,
                session=session, name=f"ref_r{r}", workspace=arena, mla_compute_mode="bf16",
                window_metadata=wm, examples=examples, pipelined_mode="mla_wo",
            )
            for r in range(4)
        ]
        sharded_attn = ds_attention.PreparedAttention(
            layer_idx, tp_weights[0], x, scr, meta, pool_sh, rotary, mode="prefill", world=4,
            session=session, name="sharded_mla_wo", workspace=arena, mla_compute_mode="bf16",
            window_metadata=wm, examples=examples, pipelined_mode="mla_wo",
            shard_tokens=st, replicated_weights=rep_weights,
        )
        wm.functional()
        session.tune(show_progress=False)

    try:
        indices = wm(meta)
        ref_out_a = torch.zeros((h, 5120), dtype=torch.float32, device=device)
        ref_out_b = torch.zeros((h, 5120), dtype=torch.float32, device=device)
        ref_wc = None
        ref_sel = None
        ref_kv_records = None
        written_wc_bytes = (128 + live) * 528
        for r in range(4):
            # Reset pool_ref carry before each rank so all 4 simulated ranks see identical pre-chunk state!
            pool_ref.region("layer2.pending_values").copy_(init_pv)
            pool_ref.region("layer2.pending_gates").copy_(init_pg)
            pool_ref.region("layer2.pending_position").fill_(init_pos)
            wc_r = ref_attns[r].init_window_cache(meta, device)
            pstate = ref_attns[r].run_prefix(x, meta, init_sel, indices=indices, window_cache=wc_r)
            oa = ref_attns[r].run_mla_wo_half(pstate, meta, indices=indices, half_idx=0)
            ob = ref_attns[r].run_mla_wo_half(pstate, meta, indices=indices, half_idx=1)
            ref_out_a += oa.float()
            ref_out_b += ob.float()
            if r == 0:
                ref_wc = wc_r.clone()
                ref_sel = pstate[2][0].clone()
                ref_kv_records = pstate[5]
            else:
                assert torch.equal(ref_wc[:, :written_wc_bytes], wc_r[:, :written_wc_bytes])
                assert torch.equal((ref_sel >= 0).sum(1), (pstate[2][0] >= 0).sum(1))
                assert torch.equal(ref_sel[live:], pstate[2][0][live:])
        ref_attns[0].commit(meta.lengths, (ref_wc, ref_kv_records), meta)
        ref_full = torch.cat((ref_out_a.bfloat16(), ref_out_b.bfloat16()), dim=0)

        rows_a = [torch.arange(r * st, (r + 1) * st, dtype=torch.int64, device=device) for r in range(4)]
        rows_b = [torch.arange(h + r * st, h + (r + 1) * st, dtype=torch.int64, device=device) for r in range(4)]
        loc_a = [sharded_attn.run_shard_mla_wo_local(x[r * st : (r + 1) * st], indices=indices, shard_rows=rows_a[r]) for r in range(4)]
        loc_b = [sharded_attn.run_shard_mla_wo_local(x[h + r * st : h + (r + 1) * st], indices=indices, shard_rows=rows_b[r]) for r in range(4)]
        kv_a_full = torch.cat([item[1] for item in loc_a], dim=0)
        kv_b_full = torch.cat([item[1] for item in loc_b], dim=0)
        assert kv_a_full.shape == (h, 2560) and kv_b_full.shape == (h, 2560)

        sh_out_a, sh_out_b = [], []
        sh_sel_a, sh_sel_b = [], []
        sh_wc = None
        for r in range(4):
            pool_sh.region("layer2.pending_values").copy_(init_pv)
            pool_sh.region("layer2.pending_gates").copy_(init_pg)
            pool_sh.region("layer2.pending_position").fill_(init_pos)
            wc_s = sharded_attn.init_window_cache(meta, device)
            pending_c = sharded_attn.global_kv.compressor.init_pending(meta, device)
            qa, _, iqa, iwa, aa = loc_a[r]
            qb, _, iqb, iwb, ab = loc_b[r]
            oa, sela = sharded_attn.run_shard_mla_wo_source(
                qa, kv_a_full, iqa, iwa, meta, ((None, None), None, None),
                indices=indices, window_cache=wc_s, half_idx=0, shard_rows=rows_a[r],
                layer_angles=aa, pending_compress=pending_c,
            )
            ob, selb = sharded_attn.run_shard_mla_wo_source(
                qb, kv_b_full, iqb, iwb, meta, ((None, None), None, None),
                indices=indices, window_cache=wc_s, half_idx=1, shard_rows=rows_b[r],
                layer_angles=ab, pending_compress=pending_c,
            )
            sh_out_a.append(oa)
            sh_out_b.append(ob)
            sh_sel_a.append(sela)
            sh_sel_b.append(selb)
            if r == 0:
                sh_wc = wc_s
            else:
                assert torch.equal(sh_wc[:, :written_wc_bytes], wc_s[:, :written_wc_bytes])
        sharded_attn.commit(meta.lengths, (sh_wc, None), meta)

        sh_full = torch.cat(sh_out_a + sh_out_b, dim=0)
        sh_sel_full = torch.cat(sh_sel_a + sh_sel_b, dim=0)

        # Bit-exact written SWA window cache, committed ring window, and pending_position!
        assert torch.equal(sh_wc[:, :written_wc_bytes], ref_wc[:, :written_wc_bytes])
        assert torch.equal(pool_sh.window(layer_idx), pool_ref.window(layer_idx))
        assert torch.equal(
            pool_sh.region("layer2.pending_position"),
            pool_ref.region("layer2.pending_position"),
        )
        torch.testing.assert_close(
            pool_sh.region("layer2.pending_values"),
            pool_ref.region("layer2.pending_values"),
            atol=1e-4, rtol=1e-4,
        )
        torch.testing.assert_close(
            pool_sh.region("layer2.pending_gates"),
            pool_ref.region("layer2.pending_gates"),
            atol=1e-4, rtol=1e-4,
        )
        assert torch.equal(sh_sel_full[live:], ref_sel[live:])
        assert torch.all(sh_sel_full[live:] == -1)
        assert torch.equal((sh_sel_full >= 0).sum(1), (ref_sel >= 0).sum(1))
        if start_pos == 0:
            assert torch.equal(sh_sel_full[:live].sort(dim=1).values, ref_sel[:live].sort(dim=1).values)
        lat_byte_mismatches = int((pool_sh.pages["layer2.latent_page"] != pool_ref.pages["layer2.latent_page"]).sum().item())
        assert lat_byte_mismatches <= 8, f"lat_byte_mismatches={lat_byte_mismatches}"

        # Compare live output rows within TP4 BF16 partial-sum and 4-bit latent quantization bounds.
        sh_live = sh_full[:live].float()
        ref_live = ref_full[:live].float()
        rel_rms = (sh_live - ref_live).square().mean().sqrt().item() / max(ref_live.square().mean().sqrt().item(), 1e-12)
        cos = torch.nn.functional.cosine_similarity(sh_live.flatten(), ref_live.flatten(), dim=0).item()
        assert rel_rms < (0.005 if start_pos == 0 else 0.015), f"rel_rms={rel_rms}"
        assert cos > 0.9999, f"cos={cos}"
    finally:
        session.__exit__(None, None, None)


@pytest.mark.skipif(
    not _RUN_CUDA_TESTS or not torch.cuda.is_available(),
    reason="requires CUDA GPU and FLARE_RUN_CUDA_TESTS=1",
)
def test_row_shard_attention_synthetic_mxfp8_matches_tp4_reference_cuda():
    from b12x.gemm import wo_projection
    from b12x.gemm.blockscaled import pack_weight
    from b12x.preparation import autotune_effort as b12x_autotune_effort
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    cfg = _qualified_model_config()
    layer_idx = 3
    tokens = 1024
    h = tokens // 2
    st = h // 4
    ratio = cfg.text["compress_ratios"][layer_idx]
    assert ratio == 2 and layer_idx in ds_config.full_attention_encoder_layers(cfg)

    gen = torch.Generator(device=device).manual_seed(20261125)

    def rand_mxfp8_raw(rows: int, cols: int):
        w_bf16 = torch.randn((rows, cols), dtype=torch.bfloat16, device=device, generator=gen) * 0.02
        vals = w_bf16.to(torch.float8_e4m3fn)
        scales_32x32 = torch.full((rows // 32, cols // 32), 127, dtype=torch.uint8, device=device)
        return vals, scales_32x32

    def pack_mxfp8(vals: torch.Tensor, scales_32x32: torch.Tensor):
        expanded = scales_32x32.repeat_interleave(32, dim=0)
        return pack_weight(vals, expanded, recipe="mxfp8")

    qkv_vals, qkv_s = rand_mxfp8_raw(1792, 5120)
    qb_vals, qb_s = rand_mxfp8_raw(32768, 1280)
    woa_vals, woa_s = rand_mxfp8_raw(8192, 4096)
    wob_vals, wob_s = rand_mxfp8_raw(5120, 8192)
    q_norm = torch.ones(1280, dtype=torch.bfloat16, device=device)
    kv_norm = torch.ones(512, dtype=torch.bfloat16, device=device)
    sink_full = torch.linspace(-1.5, 1.5, 64, dtype=torch.float32, device=device)

    qkv_packed = pack_mxfp8(qkv_vals, qkv_s)
    tp_weights = []
    for r in range(4):
        h0, h1 = r * 8192, (r + 1) * 8192
        g0, g1 = r * 2048, (r + 1) * 2048
        tp_weights.append({
            "qkv_a": qkv_packed,
            "q_b": pack_mxfp8(qb_vals[h0:h1], qb_s[h0 // 32 : h1 // 32]),
            "output": wo_projection.pack_weights(
                woa_vals[g0:g1],
                woa_s[g0 // 32 : g1 // 32].view(torch.float8_e8m0fnu),
                wob_vals[:, g0:g1].contiguous(),
                wob_s[:, g0 // 32 : g1 // 32].contiguous().view(torch.float8_e8m0fnu),
                groups=2, group_width=4096, rank=1024, hidden=5120, block_size=(32, 32),
            ),
            "q_norm": q_norm,
            "kv_norm": kv_norm,
            "sink": sink_full[r * 16 : (r + 1) * 16].contiguous(),
        })
    rep_weights = {
        "q_b_full": pack_mxfp8(qb_vals, qb_s),
        "output_full": wo_projection.pack_weights(
            woa_vals,
            woa_s.view(torch.float8_e8m0fnu),
            wob_vals,
            wob_s.view(torch.float8_e8m0fnu),
            groups=8, group_width=4096, rank=1024, hidden=5120, block_size=(32, 32),
        ),
        "sink_full": sink_full,
    }

    pool = ContinuationPool.allocate(cfg, 4, 32, device, dspark=False)
    pool.pages[f"layer{cfg.source(layer_idx).layer}.latent_page"].zero_()
    pool.window(layer_idx).zero_()
    rotary = ds_attention.Rotary(cfg, device)
    meta = Metadata.allocate(tokens, 1, 64, device, max_query_tokens=tokens)
    meta.positions.copy_(torch.arange(tokens, dtype=torch.int32, device=device))
    meta.starts.zero_()
    meta.lengths.fill_(tokens)
    meta.query_start_loc.copy_(torch.tensor([0, tokens], dtype=torch.int32, device=device))
    meta.live_counts.copy_(torch.tensor([tokens, 1], dtype=torch.int32, device=device))
    meta.state_rows.fill_(1)
    meta.requests.zero_()
    meta.page_table.zero_()
    for p in range(16):
        meta.page_table[0, p] = p
    if meta.token_page_table.stride(0) != 0:
        meta.token_page_table.copy_(meta.page_table[0].unsqueeze(0).expand(tokens, -1))

    x = (torch.randn((tokens, 5120), dtype=torch.bfloat16, device=device, generator=gen) * 0.25).contiguous()
    selected = torch.full((tokens, 512), -1, dtype=torch.int32, device=device)
    for t in range(tokens):
        k_avail = min(16, (t + 1) // 2)
        if k_avail > 0:
            selected[t, :k_avail] = torch.arange(k_avail, dtype=torch.int32, device=device)
    selection = (selected, None, None)

    with b12x_autotune_effort("off"):
        session = RecordedPreparationSession(
            device=device, checkpoint="test-rsa-synth", workload_schema="verify",
            numerical_policy="default", autotune_effort="off",
        )
        session.__enter__()
        arena = ScratchArena(device, overlap=False)
        session.register_scratch(arena)
        examples = {}
        wm = ds_attention.AttentionIndices(
            meta, mode="prefill", tokens=tokens, page_size=128 + tokens, width=128, ratios=(ratio,),
        )
        scr = ds_attention.AttentionScratch.allocate(tokens, 4, device)
        ref_attns = [
            ds_attention.PreparedAttention(
                layer_idx, tp_weights[r], x, scr, meta, pool, rotary, mode="prefill", world=4,
                session=session, name=f"ref_r{r}", workspace=arena, mla_compute_mode="bf16",
                window_metadata=wm, examples=examples, pipelined_mode="full",
            )
            for r in range(4)
        ]
        sharded_attn = ds_attention.PreparedAttention(
            layer_idx, tp_weights[0], x, scr, meta, pool, rotary, mode="prefill", world=4,
            session=session, name="sharded", workspace=arena, mla_compute_mode="bf16",
            window_metadata=wm, examples=examples, pipelined_mode="full",
            shard_tokens=st, replicated_weights=rep_weights,
        )
        wm.functional()
        session.tune(show_progress=False)

    try:
        indices = wm(meta)
        ref_out_a = torch.zeros((h, 5120), dtype=torch.float32, device=device)
        ref_out_b = torch.zeros((h, 5120), dtype=torch.float32, device=device)
        ref_wc = None
        for r in range(4):
            wc_r = ref_attns[r].init_window_cache(meta, device)
            oa = ref_attns[r].run_half(x[:h], meta, selection, indices=indices, window_cache=wc_r, half_idx=0)
            ob = ref_attns[r].run_half(x[h:], meta, selection, indices=indices, window_cache=wc_r, half_idx=1)
            ref_out_a += oa.float()
            ref_out_b += ob.float()
            if r == 0:
                ref_wc = wc_r
            else:
                assert torch.equal(ref_wc, wc_r)
        ref_full = torch.cat((ref_out_a.bfloat16(), ref_out_b.bfloat16()), dim=0)

        rows_a = [torch.arange(r * st, (r + 1) * st, dtype=torch.int64, device=device) for r in range(4)]
        rows_b = [torch.arange(h + r * st, h + (r + 1) * st, dtype=torch.int64, device=device) for r in range(4)]
        q_a_l, wv_a_l, ang_a_l = [], [], []
        q_b_l, wv_b_l, ang_b_l = [], [], []
        for r in range(4):
            qa, wva, aa = sharded_attn.run_shard_qkv_q(x[r * st : (r + 1) * st], indices=indices, shard_rows=rows_a[r])
            qb, wvb, ab = sharded_attn.run_shard_qkv_q(x[h + r * st : h + (r + 1) * st], indices=indices, shard_rows=rows_b[r])
            q_a_l.append(qa)
            wv_a_l.append(wva)
            ang_a_l.append(aa)
            q_b_l.append(qb)
            wv_b_l.append(wvb)
            ang_b_l.append(ab)

        wv_a_full = torch.cat(wv_a_l, dim=0)
        wv_b_full = torch.cat(wv_b_l, dim=0)
        sh_out_a, sh_out_b = [], []
        sh_wc = None
        for r in range(4):
            wc_s = sharded_attn.init_window_cache(meta, device)
            local_sel = ((selected[r * st : (r + 1) * st], selected[h + r * st : h + (r + 1) * st]), None, None)
            oa = sharded_attn.run_shard_mla_wo(
                q_a_l[r], wv_a_full, meta, local_sel, indices=indices, window_cache=wc_s,
                half_idx=0, shard_rows=rows_a[r], layer_angles=ang_a_l[r],
            )
            ob = sharded_attn.run_shard_mla_wo(
                q_b_l[r], wv_b_full, meta, local_sel, indices=indices, window_cache=wc_s,
                half_idx=1, shard_rows=rows_b[r], layer_angles=ang_b_l[r],
            )
            sh_out_a.append(oa)
            sh_out_b.append(ob)
            if r == 0:
                sh_wc = wc_s
            else:
                assert torch.equal(sh_wc, wc_s)
        sh_full = torch.cat(sh_out_a + sh_out_b, dim=0)

        assert torch.equal(ref_wc, sh_wc)
        diff = sh_full.float() - ref_full.float()
        rel_rms = diff.square().mean().sqrt().item() / max(ref_full.float().square().mean().sqrt().item(), 1e-12)
        cos = torch.nn.functional.cosine_similarity(sh_full.float().flatten(), ref_full.float().flatten(), dim=0).item()
        assert rel_rms < 0.005, f"rel_rms exceeded BF16 rounding budget: {rel_rms}"
        assert cos > 0.9999, f"cosine similarity below bound: {cos}"

        bad_wc = sharded_attn.init_window_cache(meta, device)
        bad_wv = wv_a_full.clone()
        bad_wv[st - 4 : st] += 5.0
        bad_oa_r1 = sharded_attn.run_shard_mla_wo(
            q_a_l[1], bad_wv, meta, selection, indices=indices, window_cache=bad_wc,
            half_idx=0, shard_rows=rows_a[1], layer_angles=ang_a_l[1],
        )
        assert not torch.equal(bad_wc, ref_wc[:1])
        bad_rel = (bad_oa_r1.float() - sh_out_a[1].float()).square().mean().sqrt().item() / max(
            sh_out_a[1].float().square().mean().sqrt().item(), 1e-12
        )
        assert bad_rel > 0.05
    finally:
        session.__exit__(None, None, None)


@pytest.mark.skipif(
    not _RUN_CUDA_TESTS or not torch.cuda.is_available() or not _SNAPSHOT_ENV,
    reason="requires CUDA GPU, FLARE_RUN_CUDA_TESTS=1, and FLARE_DEEPSEEK41_SNAPSHOT",
)
@pytest.mark.parametrize(("layer_idx", "tokens"), [(2, 1024), (3, 1024), (3, 4096), (0, 1024)])
def test_row_shard_attention_checkpoint_parity_cuda(layer_idx: int, tokens: int):
    from b12x.preparation import autotune_effort as b12x_autotune_effort
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.tuning import RecordedPreparationSession

    snapshot = Path(_SNAPSHOT_ENV)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    manifest = inspect_indexed_checkpoint(snapshot)
    config = ds_config.ModelConfig.load(manifest.metadata_path)

    tp_weights = []
    rep_weights = None
    for r in range(4):
        shard = TensorParallelShard(r, 4)
        with ds_weights.DeepSeekWeightLoader(manifest, device, shard=shard, dspark=False) as loader:
            tp_weights.append(loader.load_layer(layer_idx).attention)
            if r == 0:
                rep_weights = loader.load_replicated_attention(layer_idx)

    is_mla_wo = "compressor" in tp_weights[0] or "index_q" in tp_weights[0]
    pipelined_mode = "mla_wo" if is_mla_wo else "full"
    h = tokens // 2
    st = h // 4
    pool_ref = ContinuationPool.allocate(config, 4, 32, device, dspark=False)
    pool_sh = ContinuationPool.allocate(config, 4, 32, device, dspark=False)
    rotary = ds_attention.Rotary(config, device)
    meta = Metadata.allocate(tokens, 1, 64, device, max_query_tokens=tokens)
    meta.positions.copy_(torch.arange(tokens, dtype=torch.int32, device=device))
    meta.starts.zero_()
    meta.lengths.fill_(tokens)
    meta.query_start_loc.copy_(torch.tensor([0, tokens], dtype=torch.int32, device=device))
    meta.live_counts.copy_(torch.tensor([tokens, 1], dtype=torch.int32, device=device))
    meta.state_rows.fill_(1)
    meta.requests.zero_()
    meta.page_table.zero_()
    for p in range(16):
        meta.page_table[0, p] = p
    if meta.token_page_table.stride(0) != 0:
        meta.token_page_table.copy_(meta.page_table[0].unsqueeze(0).expand(tokens, -1))

    gen = torch.Generator(device=device).manual_seed(2026)
    ratio = config.text["compress_ratios"][layer_idx]
    for p in (pool_ref, pool_sh):
        if ratio:
            p.pages[f"layer{config.source(layer_idx).layer}.latent_page"].zero_()
            p.pages[f"layer{config.source(layer_idx).layer}.index_page"].zero_()
        p.window(layer_idx).zero_()
    x = (torch.randn((tokens, 5120), dtype=torch.bfloat16, device=device, generator=gen) * 0.25).contiguous()
    selected = torch.full((tokens, 512), -1, dtype=torch.int32, device=device)
    if not is_mla_wo:
        for t in range(tokens):
            k_avail = min(16, (t + 1) // 2)
            if k_avail > 0:
                selected[t, :k_avail] = torch.arange(k_avail, dtype=torch.int32, device=device)
    selection = (selected, None, None)

    with b12x_autotune_effort("off"):
        session = RecordedPreparationSession(
            device=device, checkpoint="verify-rsa", workload_schema="verify",
            numerical_policy="default", autotune_effort="off",
        )
        session.__enter__()
        arena = ScratchArena(device, overlap=False)
        session.register_scratch(arena)
        examples = {}
        wm = ds_attention.AttentionIndices(
            meta, mode="prefill", tokens=tokens, page_size=128 + tokens, width=128,
            ratios=(ratio,) if ratio else (),
        )
        scr = ds_attention.AttentionScratch.allocate(tokens, 4, device, candidate_tokens=128)
        ref_attns = [
            ds_attention.PreparedAttention(
                layer_idx, tp_weights[r], x, scr, meta, pool_ref, rotary, mode="prefill", world=4,
                session=session, name=f"ref_r{r}", workspace=arena, mla_compute_mode="bf16",
                window_metadata=wm, examples=examples, pipelined_mode=pipelined_mode,
            )
            for r in range(4)
        ]
        sharded_attn = ds_attention.PreparedAttention(
            layer_idx, tp_weights[0], x, scr, meta, pool_sh, rotary, mode="prefill", world=4,
            session=session, name="sharded", workspace=arena, mla_compute_mode="bf16",
            window_metadata=wm, examples=examples, pipelined_mode=pipelined_mode,
            shard_tokens=st, replicated_weights=rep_weights,
        )
        wm.functional()
        session.tune(show_progress=False)

    try:
        indices = wm(meta)
        ref_out_a = torch.zeros((h, 5120), dtype=torch.float32, device=device)
        ref_out_b = torch.zeros((h, 5120), dtype=torch.float32, device=device)
        ref_wc = None
        for r in range(4):
            wc_r = ref_attns[r].init_window_cache(meta, device)
            if is_mla_wo:
                pstate = ref_attns[r].run_prefix(x, meta, selection, indices=indices, window_cache=wc_r)
                oa = ref_attns[r].run_mla_wo_half(pstate, meta, indices=indices, half_idx=0)
                ob = ref_attns[r].run_mla_wo_half(pstate, meta, indices=indices, half_idx=1)
            else:
                oa = ref_attns[r].run_half(x[:h], meta, selection, indices=indices, window_cache=wc_r, half_idx=0)
                ob = ref_attns[r].run_half(x[h:], meta, selection, indices=indices, window_cache=wc_r, half_idx=1)
            ref_out_a += oa.float()
            ref_out_b += ob.float()
            if r == 0:
                ref_wc = wc_r
            else:
                assert torch.equal(ref_wc, wc_r)
        ref_full = torch.cat((ref_out_a.bfloat16(), ref_out_b.bfloat16()), dim=0)

        rows_a = [torch.arange(r * st, (r + 1) * st, dtype=torch.int64, device=device) for r in range(4)]
        rows_b = [torch.arange(h + r * st, h + (r + 1) * st, dtype=torch.int64, device=device) for r in range(4)]
        sh_out_a, sh_out_b = [], []
        sh_wc = None
        if is_mla_wo:
            loc_a = [sharded_attn.run_shard_mla_wo_local(x[r * st : (r + 1) * st], indices=indices, shard_rows=rows_a[r]) for r in range(4)]
            loc_b = [sharded_attn.run_shard_mla_wo_local(x[h + r * st : h + (r + 1) * st], indices=indices, shard_rows=rows_b[r]) for r in range(4)]
            kv_a_full = torch.cat([item[1] for item in loc_a], dim=0)
            kv_b_full = torch.cat([item[1] for item in loc_b], dim=0)
            for r in range(4):
                wc_s = sharded_attn.init_window_cache(meta, device)
                pending_c = sharded_attn.global_kv.compressor.init_pending(meta, device)
                qa, _, iqa, iwa, aa = loc_a[r]
                qb, _, iqb, iwb, ab = loc_b[r]
                oa, _ = sharded_attn.run_shard_mla_wo_source(
                    qa, kv_a_full, iqa, iwa, meta, selection, indices=indices, window_cache=wc_s,
                    half_idx=0, shard_rows=rows_a[r], layer_angles=aa, pending_compress=pending_c,
                )
                ob, _ = sharded_attn.run_shard_mla_wo_source(
                    qb, kv_b_full, iqb, iwb, meta, selection, indices=indices, window_cache=wc_s,
                    half_idx=1, shard_rows=rows_b[r], layer_angles=ab, pending_compress=pending_c,
                )
                sh_out_a.append(oa)
                sh_out_b.append(ob)
                sh_wc = wc_s if r == 0 else sh_wc
        else:
            q_a_l, wv_a_l, ang_a_l = [], [], []
            q_b_l, wv_b_l, ang_b_l = [], [], []
            for r in range(4):
                qa, wva, aa = sharded_attn.run_shard_qkv_q(x[r * st : (r + 1) * st], indices=indices, shard_rows=rows_a[r])
                qb, wvb, ab = sharded_attn.run_shard_qkv_q(x[h + r * st : h + (r + 1) * st], indices=indices, shard_rows=rows_b[r])
                q_a_l.append(qa)
                wv_a_l.append(wva)
                ang_a_l.append(aa)
                q_b_l.append(qb)
                wv_b_l.append(wvb)
                ang_b_l.append(ab)
            wv_a_full = torch.cat(wv_a_l, dim=0)
            wv_b_full = torch.cat(wv_b_l, dim=0)
            for r in range(4):
                wc_s = sharded_attn.init_window_cache(meta, device)
                oa = sharded_attn.run_shard_mla_wo(
                    q_a_l[r], wv_a_full, meta, selection, indices=indices, window_cache=wc_s,
                    half_idx=0, shard_rows=rows_a[r], layer_angles=ang_a_l[r],
                )
                ob = sharded_attn.run_shard_mla_wo(
                    q_b_l[r], wv_b_full, meta, selection, indices=indices, window_cache=wc_s,
                    half_idx=1, shard_rows=rows_b[r], layer_angles=ang_b_l[r],
                )
                sh_out_a.append(oa)
                sh_out_b.append(ob)
                sh_wc = wc_s if r == 0 else sh_wc
        sh_full = torch.cat(sh_out_a + sh_out_b, dim=0)
    finally:
        session.__exit__(None, None, None)

    assert torch.equal(ref_wc, sh_wc)
    diff = sh_full.float() - ref_full.float()
    rel_rms = diff.square().mean().sqrt().item() / max(ref_full.float().square().mean().sqrt().item(), 1e-12)
    cos = torch.nn.functional.cosine_similarity(sh_full.float().flatten(), ref_full.float().flatten(), dim=0).item()
    assert rel_rms < 0.005
    assert cos > 0.9999
