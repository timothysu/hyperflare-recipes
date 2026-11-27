# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Declaration tensors preserve geometry without owning serving temporaries."""

import pytest
import torch

from hyperflare_recipes.recipes.deepseek41.bootstrap import PreparationConfig
from hyperflare_recipes.recipes.deepseek41.decoder import Scratch


def _overlap(left, right):
    return max(left.data_ptr(), right.data_ptr()) < min(
        left.data_ptr() + left.nbytes, right.data_ptr() + right.nbytes)


def test_declaration_stages_do_not_overwrite_residuals_or_selections():
    scratch = Scratch.allocate(128, 1, 4, "cpu", candidate_tokens=16)
    retained = (scratch.residual, scratch.engram_residual, scratch.attention_hc.residual,
                scratch.ffn_hc.residual, scratch.attention.selected, scratch.attention.candidates,
                scratch.attention.candidate_counts, scratch.mlp_output)
    concurrent = (scratch.shared.gate_up, scratch.shared.intermediate, scratch.shared.output,
                  scratch.routed, scratch.mlp_partial)
    for index, value in enumerate(retained):
        value.fill_(index + 1)
    for index, value in enumerate(concurrent):
        assert all(not _overlap(value, other) for other in (*retained, *concurrent[:index]))
    assert all(not _overlap(value, other) for value in concurrent
               for other in (scratch.embedding, scratch.engram.gathered, scratch.engram.projected))
    engram = (scratch.engram.gathered, scratch.engram.local, scratch.engram.partial, scratch.engram.projected)
    for index, value in enumerate(engram):
        assert all(not _overlap(value, other) for other in engram[:index])
    for value in (scratch.embedding, *engram, *concurrent):
        value.fill_(42)
    for index, value in enumerate(retained):
        assert value.eq(index + 1).all()
    assert scratch.rows(64, 1).attention.candidates.shape == (16, 16384)
    assert scratch.rows(8, 1).attention.candidates.shape == (8, 16384)


def test_suffix_declarations_preserve_full_prefill_sources():
    prefill = Scratch.allocate(128, 1, 4, "cpu")
    suffix = Scratch.allocate(32, 1, 4, "cpu")
    pairs = [(getattr(prefill.attention_hc, name), getattr(suffix.attention_hc, name))
             for name in ("normalized", "residual", "post", "comb")]
    pairs.append((prefill.attn_mix, suffix.attn_mix))
    for source, destination in pairs:
        assert not _overlap(source, destination)
        source.fill_(7)
        destination.fill_(9)
        assert source.eq(7).all()
    first = prefill.rows(128, 1)
    second = prefill.rows(32, 1)
    assert first.residual.data_ptr() == second.residual.data_ptr() == prefill.residual.data_ptr()
    assert second.attention_hc.normalized.shape == (32, 5120)
    for source, destination in pairs:
        assert source.eq(7).all() and destination.eq(9).all()


@pytest.mark.parametrize("capacity,dspark,draft_width,expected", [
    (32, True, 6, 256), (32, False, 1, 256), (1, True, 6, 256),
])
def test_oneshot_capacity_excludes_prefills_assigned_to_dma(capacity, dspark, draft_width, expected):
    from dataclasses import replace
    from ignition.parameters import SpeculationParameters

    base = PreparationConfig()
    spec = None if not dspark else SpeculationParameters(
        draft_tokens=draft_width - 1, max_draft_tokens=7, token_capacity=64,
        packed_verification=True, draft_prefix_selection="gpu", graph_budget=64,
    )
    config = replace(base, parameters=replace(base.parameters, max_concurrency=capacity, speculation=spec))
    assert config.all_reduce_tokens == expected


def test_oneshot_capacity_tracks_draft_requirement_below_dma_cutoff():
    from dataclasses import replace
    from ignition.parameters import SpeculationParameters

    base = PreparationConfig()
    spec = SpeculationParameters(
        draft_tokens=5, max_draft_tokens=7, token_capacity=64,
        packed_verification=True, draft_prefix_selection="gpu", graph_budget=64,
    )
    config = replace(base, parameters=replace(base.parameters, max_concurrency=32, prefill_chunk=64, speculation=spec))
    assert config.all_reduce_tokens == 160


def test_short_prefill_still_fits_the_oneshot_channel():
    from dataclasses import replace

    base = PreparationConfig()
    config = replace(base, parameters=replace(base.parameters, max_concurrency=1, prefill_chunk=64, speculation=None))
    assert config.all_reduce_tokens == 64


def test_three_rank_scratch_preserves_grouped_attention_and_unpadded_experts():
    from dataclasses import replace
    from ignition.parameters import SpeculationParameters, VisionParameters

    base = PreparationConfig()
    config = PreparationConfig(
        devices=(0, 1, 2),
        parameters=replace(
            base.parameters,
            max_concurrency=32,
            speculation=SpeculationParameters(
                draft_tokens=7, max_draft_tokens=7, token_capacity=128,
                packed_verification=True, draft_prefix_selection="gpu", graph_budget=64,
            ),
            vision=VisionParameters(image_concurrency=4),
        ),
    )
    scratch = Scratch.allocate(128, 32, 3, "meta")
    assert config.draft_tokens == 7
    assert scratch.attention.query.shape == (128, 24, 512)
    assert scratch.attention.attended.view(128, 3, 4096).shape == (128, 3, 4096)
    assert scratch.local_logits.shape == (32, 43136)
    assert scratch.logits.shape == (32, 129280)
    assert scratch.shared.intermediate.shape == (128, 768)
    assert scratch.engram.local.shape == (128, 2048)


@pytest.mark.parametrize("world", [3, 4])
def test_target_logits_remain_fp32_across_capacity_views(world):
    scratch = Scratch.allocate(128, 32, world, "meta")
    assert scratch.hidden.dtype == torch.bfloat16
    for view in (scratch, scratch.rows(8, 8), scratch.rows(32, 1)):
        assert view.local_logits.dtype == torch.float32
        assert view.logits.dtype == torch.float32
        assert view.local_logits.element_size() == 4


def test_draft_logits_keep_bf16_storage_independently_of_target():
    from hyperflare_recipes.recipes.deepseek41.decoder import DraftScratch

    scratch = DraftScratch.allocate(32, 32, 4, "meta")
    assert scratch.local_logits.dtype == scratch.logits.dtype == torch.bfloat16
    sliced = scratch.rows(8, 8)
    assert isinstance(sliced, DraftScratch)
    assert sliced.local_logits.dtype == sliced.logits.dtype == torch.bfloat16


def _flatten_scratch_tensors(value):
    from dataclasses import fields, is_dataclass

    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from _flatten_scratch_tensors(getattr(value, field.name))


def test_deepseek41_vision_scratch_rows_share_max_capacity_storage():
    from hyperflare_recipes.components.deepseek_vision import VisionScratch
    from hyperflare_recipes.recipes.deepseek41.contract import PATCH_CAPACITIES

    vision_max = VisionScratch.allocate(max(PATCH_CAPACITIES), "cpu", hidden=5120)
    max_tensors = tuple(_flatten_scratch_tensors(vision_max))
    assert len(max_tensors) == 16
    assert len({tensor.untyped_storage()._cdata for tensor in max_tensors}) == len(max_tensors)

    views = {capacity: vision_max.rows(capacity) for capacity in reversed(PATCH_CAPACITIES)}
    assert len(views) == len(PATCH_CAPACITIES)
    for capacity, view in views.items():
        standalone = VisionScratch.allocate(capacity, "cpu", hidden=5120)
        view_tensors = tuple(_flatten_scratch_tensors(view))
        ref_tensors = tuple(_flatten_scratch_tensors(standalone))
        assert len(view_tensors) == len(max_tensors)
        assert len({tensor.untyped_storage()._cdata for tensor in view_tensors}) == len(view_tensors)
        for parent, actual, expected in zip(max_tensors, view_tensors, ref_tensors, strict=True):
            assert actual.shape == expected.shape
            assert actual.stride() == expected.stride()
            assert actual.dtype == expected.dtype
            assert actual.is_contiguous()
            assert actual.data_ptr() == parent.data_ptr()
            assert actual.untyped_storage()._cdata == parent.untyped_storage()._cdata


def test_compressor_and_global_kv_share_declaration_operands_across_layers_and_capacities(monkeypatch):
    from types import SimpleNamespace

    from hyperflare_recipes.components.paged_batch import Metadata
    from hyperflare_recipes.recipes.deepseek41 import attention, decoder
    from hyperflare_recipes.recipes.deepseek41.config import ModelConfig
    from hyperflare_recipes.recipes.deepseek41.decoder_prefill import PreparedDecoderSuffix
    from hyperflare_recipes.recipes.deepseek41.pool import ContinuationPool

    finalizers = []

    class Component(SimpleNamespace):
        def functional(self, **kwargs):
            return self

    monkeypatch.setattr(attention.mla_compress, "Caps", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(attention.mla_compress, "plan",
                        lambda caps: SimpleNamespace(request=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setattr(attention, "declare", lambda *args, **kwargs: None)
    monkeypatch.setattr(attention, "register_finalizer",
                        lambda session, callback, **kwargs: finalizers.append(callback))
    monkeypatch.setattr(attention, "UnquantizedProjection",
                        lambda source, weight, output, **kwargs: Component(plan=object(), source=source, output=output))
    monkeypatch.setattr(attention, "PreparedRMSNorm",
                        lambda *args, **kwargs: Component(plan=object()))
    monkeypatch.setattr(attention, "PreparedCompressedCacheWriter",
                        lambda values, cache, slots, **kwargs: Component(plan=object(), values=values, slots=slots))
    monkeypatch.setattr(decoder, "PreparedMHC", lambda *args, **kwargs: Component(plan=object()))

    model_cfg = ModelConfig({
        "num_hidden_layers": 40,
        "hidden_size": 5120,
        "sliding_window": 128,
        "rms_norm_eps": 1e-6,
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
        "candidate_source_layer_id": 20,
        "compress_ratios": [0, 0] + [2] * 18 + [1] * 20,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
    }, {})
    pool = ContinuationPool.allocate(model_cfg, 3, 2, "cpu")
    rotary = attention.Rotary(model_cfg, "cpu")

    def layer_weights(ratio):
        comp = {"norm": torch.ones(512, dtype=torch.bfloat16), "values": torch.ones((512, 5120), dtype=torch.bfloat16)}
        if ratio == 2:
            comp["gates"] = torch.ones((512, 5120), dtype=torch.bfloat16)
        return {
            "compressor": comp,
            "index_k": torch.ones((128, 512), dtype=torch.bfloat16),
            "index_norm": torch.ones(128, dtype=torch.bfloat16),
        }

    residual_weights = {
        "hc_attn_fn": torch.ones((5120, 4), dtype=torch.bfloat16),
        "hc_attn_scale": torch.ones(4, dtype=torch.float32),
        "hc_attn_base": torch.ones((4, 4), dtype=torch.float32),
        "attn_norm.weight": torch.ones(5120, dtype=torch.bfloat16),
    }

    def build_kv(layer, tokens, *, examples):
        meta = Metadata.allocate(tokens, 1, 2, "cpu", max_query_tokens=tokens)
        indices = attention.AttentionIndices(
            meta, mode="prefill", tokens=tokens, page_size=128 + tokens, width=128, ratios=(1, 2)
        )
        scr = attention.AttentionScratch.allocate(tokens, 4, "cpu")
        src = torch.zeros((tokens, 5120), dtype=torch.bfloat16)
        ratio = model_cfg.text["compress_ratios"][layer]
        return attention.PreparedGlobalKV(
            layer, layer_weights(ratio), src, scr, meta, pool, rotary,
            session=None, name=f"kv_{layer}_{tokens}", indices=indices, examples=examples,
        )

    examples = {}
    kv_256_l2 = build_kv(2, 256, examples=examples)
    kv_256_l8 = build_kv(8, 256, examples=examples)

    meta_256 = Metadata.allocate(256, 1, 2, "cpu", max_query_tokens=256)
    packing_256 = PreparedDecoderSuffix(meta_256)
    src_indices_256 = attention.AttentionIndices(
        meta_256, mode="prefill", tokens=256, page_size=128 + 256, width=128, ratios=(1, 2)
    )
    sfx_indices_256 = attention.AttentionIndices(
        packing_256.metadata, mode="prefill", tokens=128, page_size=256, width=128,
        replay_start=packing_256.replay_start, ratios=(1,),
    )
    full_scr_256 = Scratch.allocate(256, 1, 4, "cpu", candidate_tokens=128)
    sfx_scr_128 = Scratch.allocate(128, 1, 4, "cpu")
    boundary_256 = decoder.PreparedDecoderBoundary(
        20,
        SimpleNamespace(attention=layer_weights(1), residual=residual_weights),
        full_scr_256,
        sfx_scr_128,
        packing_256,
        pool,
        rotary,
        carry=False,
        session=None,
        name="boundary_256",
        workspace=None,
        window_metadata=sfx_indices_256,
        source_indices=src_indices_256,
        examples=examples,
    )
    kv_256_l20 = boundary_256.global_kv

    kv_64_l2 = build_kv(2, 64, examples=examples)
    kv_64_l20 = build_kv(20, 64, examples=examples)

    c256_2, c256_8, c256_20 = kv_256_l2.compressor, kv_256_l8.compressor, kv_256_l20.compressor
    c64_2, c64_20 = kv_64_l2.compressor, kv_64_l20.compressor

    live_r2 = (c256_2.values, c256_2.gates, c256_2.output, c256_2.emitted, c256_2.slots)
    assert len({tensor.untyped_storage()._cdata for tensor in live_r2}) == 5
    for name in ("values", "gates", "output", "emitted", "slots"):
        assert getattr(c256_2, name) is getattr(c256_8, name)
        small = getattr(c64_2, name)
        large = getattr(c256_2, name)
        assert small.data_ptr() == large.data_ptr()
        assert small.untyped_storage()._cdata == large.untyped_storage()._cdata
        assert small.shape[0] == 64 and small.is_contiguous()

    assert c256_20.values.dtype == torch.bfloat16 and c256_2.values.dtype == torch.float32
    assert c256_20.values.untyped_storage()._cdata != c256_2.values.untyped_storage()._cdata
    assert not hasattr(c256_20, "gates")
    for name in ("output", "emitted", "slots"):
        assert getattr(c256_20, name) is getattr(c256_2, name)
    assert c64_20.values.data_ptr() == c256_20.values.data_ptr()
    assert c64_20.values.shape == (64, 512) and c64_20.values.is_contiguous()

    solo_a = build_kv(2, 64, examples=None).compressor
    solo_b = build_kv(2, 64, examples=None).compressor
    for name in ("values", "gates", "output", "emitted", "slots"):
        left, right = getattr(solo_a, name), getattr(solo_b, name)
        assert left.untyped_storage()._cdata != right.untyped_storage()._cdata
        assert left.untyped_storage()._cdata != getattr(c256_2, name).untyped_storage()._cdata
    assert torch.all(solo_a.slots == -1) and torch.all(solo_b.slots == -1)

    for callback in finalizers:
        callback()
    assert c256_2.values is None and c256_2.gates is None and c256_2.output is None
    assert c256_2.emitted is None and c256_2.slots is None and c256_2.arguments is None


def test_attention_shares_window_cache_for_prefill_and_draft_while_preserving_decode_and_verify(monkeypatch):
    from types import SimpleNamespace

    from hyperflare_recipes.components.paged_batch import Metadata
    from hyperflare_recipes.recipes.deepseek41 import attention
    from hyperflare_recipes.recipes.deepseek41.config import ModelConfig
    from hyperflare_recipes.recipes.deepseek41.pool import ContinuationPool

    recorded_writers = {}
    recorded_mla = {}

    class Component(SimpleNamespace):
        def functional(self, **kwargs):
            return self

    def fake_writer(values, cache, slots, **kwargs):
        recorded_writers[kwargs["name"]] = cache
        return Component(plan=object())

    def fake_mla(query, window, *args, **kwargs):
        recorded_mla[kwargs["name"]] = window
        return Component(plan=object())

    monkeypatch.setattr(attention.sm103_projection, "is_sm103", lambda device: False)
    monkeypatch.setattr(attention.mla_norm_mxfp8, "is_supported", lambda device: False)
    monkeypatch.setattr(attention.mla_compress, "Caps", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(attention.mla_compress, "plan",
                        lambda caps: SimpleNamespace(request=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setattr(attention, "declare", lambda *args, **kwargs: None)
    monkeypatch.setattr(attention, "register_finalizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(attention, "QuantizedProjection",
                        lambda source, weight, output, **kwargs: Component(
                            plan=object(), output=output, encoded=None, encoded_layout=None, quantizer_plan=None))
    monkeypatch.setattr(attention, "UnquantizedProjection",
                        lambda source, weight, output, **kwargs: Component(plan=object(), output=output))
    for attr in ("PreparedRMSNorm", "PreparedGroupedOutputProjection"):
        monkeypatch.setattr(attention, attr, lambda *args, **kwargs: Component(plan=object()))
    monkeypatch.setattr(attention, "PreparedCompressedCacheWriter", fake_writer)
    monkeypatch.setattr(attention, "PreparedCompressedMLA", fake_mla)

    model_cfg = ModelConfig({
        "num_hidden_layers": 40,
        "hidden_size": 5120,
        "vocab_size": 129280,
        "sliding_window": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "compress_rope_theta": 10000.0,
        "rope_scaling": {
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40.0,
        },
        "candidate_source_layer_id": 20,
        "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "num_nextn_predict_layers": 3,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [37, 38, 39],
        "dspark_markov_rank": 256,
        "dspark_n_routed_experts": 128,
        "dspark_num_experts_per_tok": 3,
    }, {})
    pool = ContinuationPool.allocate(model_cfg, 10, 2, "cpu", dspark=True)
    rotary = attention.Rotary(model_cfg, "cpu")
    weights_source = {
        "qkv_a": object(),
        "q_b": object(),
        "q_norm": torch.ones(1280, dtype=torch.bfloat16),
        "kv_norm": torch.ones(512, dtype=torch.bfloat16),
        "output": object(),
        "sink": object(),
        "compressor": {
            "norm": torch.ones(512, dtype=torch.bfloat16),
            "values": torch.ones((512, 5120), dtype=torch.bfloat16),
            "gates": torch.ones((512, 5120), dtype=torch.bfloat16),
        },
        "index_k": torch.ones((128, 512), dtype=torch.bfloat16),
        "index_norm": torch.ones(128, dtype=torch.bfloat16),
    }
    weights_plain = {k: v for k, v in weights_source.items() if k not in ("compressor", "index_k", "index_norm")}

    def build_attn(name, layer, weights, *, mode, tokens, rows, max_query_tokens, examples, window_staging=None):
        meta = Metadata.allocate(tokens, rows, 2, "cpu", max_query_tokens=max_query_tokens)
        scr = attention.AttentionScratch.allocate(tokens, 4, "cpu")
        src = torch.zeros((tokens, 5120), dtype=torch.bfloat16)
        return attention.PreparedAttention(
            layer, weights, src, scr, meta, pool, rotary, mode=mode, world=4,
            session=None, name=name, workspace=None, window_staging=window_staging, examples=examples,
        )

    examples = {}
    prefill_512 = build_attn("p512", 2, weights_source, mode="prefill", tokens=512, rows=1,
                             max_query_tokens=512, examples=examples)
    prefill_128 = build_attn("p128", 8, weights_source, mode="prefill", tokens=128, rows=1,
                             max_query_tokens=128, examples=examples)
    draft_4x5 = build_attn("d4x5", 0, weights_plain, mode="draft", tokens=20, rows=4,
                           max_query_tokens=5, examples=examples)
    decode_8 = build_attn("dec8", 2, weights_source, mode="decode", tokens=8, rows=8,
                          max_query_tokens=1, examples=examples)
    explicit_staging = torch.zeros((4, (128 + 6) * 528), dtype=torch.uint8)
    verify_4x6 = build_attn("ver4x6", 2, weights_source, mode="verify", tokens=16, rows=4,
                            max_query_tokens=6, examples=examples, window_staging=explicit_staging)

    assert prefill_512.window_cache.shape == (1, (128 + 512) * 528) and prefill_512.window_cache.is_contiguous()
    assert prefill_128.window_cache.shape == (1, (128 + 128) * 528) and prefill_128.window_cache.is_contiguous()
    assert draft_4x5.window_cache.shape == (4, (128 + 5) * 528) and draft_4x5.window_cache.is_contiguous()
    shared_ptr = prefill_512.window_cache.data_ptr()
    shared_storage = prefill_512.window_cache.untyped_storage()._cdata
    assert prefill_128.window_cache.data_ptr() == shared_ptr
    assert draft_4x5.window_cache.data_ptr() == shared_ptr
    assert prefill_128.window_cache.untyped_storage()._cdata == shared_storage
    assert draft_4x5.window_cache.untyped_storage()._cdata == shared_storage
    for tag, op in (("p512", prefill_512), ("p128", prefill_128), ("d4x5", draft_4x5)):
        assert recorded_writers[f"{tag}.window_write"] is op.window_cache
        assert recorded_mla[f"{tag}.attention"] is op.window_cache

    assert (prefill_512.global_kv.compressor.values.data_ptr()
            == prefill_128.global_kv.compressor.values.data_ptr()
            == verify_4x6.global_kv.compressor.values.data_ptr())

    pool_window = pool.window(2, draft=False)
    assert decode_8.window_cache is decode_8.window
    assert decode_8.window_cache.data_ptr() == pool_window.data_ptr()
    assert decode_8.window_cache.shape == pool_window.shape
    assert decode_8.window_cache.stride() == pool_window.stride()
    assert decode_8.window_cache.untyped_storage()._cdata == pool_window.untyped_storage()._cdata
    assert decode_8.window_cache.untyped_storage()._cdata != shared_storage
    assert verify_4x6.window_cache is explicit_staging
    assert verify_4x6.window_cache.untyped_storage()._cdata != shared_storage

    for bad_mode, bad_staging in (
        ("prefill", torch.zeros((1, (128 + 128) * 528), dtype=torch.uint8)),
        ("verify", torch.zeros((2, (128 + 6) * 528), dtype=torch.uint8)),
        ("verify", torch.zeros((4, (128 + 6) * 528 + 16), dtype=torch.uint8)[:, :-16]),
    ):
        with pytest.raises(ValueError, match="verification window staging"):
            build_attn("bad", 0, weights_plain, mode=bad_mode, tokens=128 if bad_mode == "prefill" else 16,
                       rows=1 if bad_mode == "prefill" else 4,
                       max_query_tokens=128 if bad_mode == "prefill" else 6,
                       examples=examples, window_staging=bad_staging)

    solo_a = build_attn("sa", 0, weights_plain, mode="prefill", tokens=128, rows=1,
                        max_query_tokens=128, examples=None)
    solo_b = build_attn("sb", 0, weights_plain, mode="prefill", tokens=128, rows=1,
                        max_query_tokens=128, examples=None)
    assert solo_a.window_cache.untyped_storage()._cdata != solo_b.window_cache.untyped_storage()._cdata


def test_decoder_shares_verification_staging_and_suffix_output_scratch(monkeypatch):
    from dataclasses import dataclass
    from types import SimpleNamespace

    from hyperflare_recipes.components.paged_batch import Metadata
    from hyperflare_recipes.components.parallel import TensorParallelShard
    from hyperflare_recipes.recipes.deepseek41 import decoder
    from hyperflare_recipes.recipes.deepseek41.config import ModelConfig
    from hyperflare_recipes.recipes.deepseek41.pool import ContinuationPool
    from ignition.preparation.scratch import bind_scratch

    ring_commits = []
    layer_records = []
    boundary_examples = []

    class CpuRingCommit:
        def __init__(self, backing, offsets, rows, starts, *, token_capacity, window_tokens,
                     record_bytes, staging=None):
            shape = (len(offsets), rows.numel(), (window_tokens + token_capacity) * record_bytes)
            self.staging = bind_scratch(
                {"staging": (shape, torch.uint8)},
                device=backing.device,
                workspace=None if staging is None else {"staging": staging},
            )["staging"]
            self.staging.zero_()
            ring_commits.append(self.staging)

        def functional(self):
            self.staging = None
            return self

    class FakeLayer:
        def __init__(self, index, layer_weights, tables, layer_scratch, layer_metadata, *args, **kwargs):
            self.index = index
            self.has_image_mask = False
            self.pipelined_ffn = False
            self.pipelined_attn = None
            self.boundary = kwargs["boundary"]
            self.engram = None
            self.attention = SimpleNamespace(selectors=[object()], plans=lambda: ())
            ws = kwargs["window_staging"]
            if kwargs["mode"] == "verify":
                expected_shape = (layer_metadata.starts.numel(), (128 + layer_metadata.max_query_tokens) * 528)
                assert ws is not None and ws.shape == expected_shape
                assert ws.dtype == torch.uint8 and ws.is_contiguous()
            else:
                assert ws is None
            layer_records.append((kwargs["name"], index, layer_scratch, ws))

    class FakeBoundary:
        def __init__(self, index, layer, source, target, packing, pool, rotary, **kwargs):
            self.index = index
            self.packing = packing
            boundary_examples.append((target, kwargs.get("examples")))

    monkeypatch.setattr(decoder, "PreparedRingCommit", CpuRingCommit)
    monkeypatch.setattr(decoder, "PreparedTableRequests", lambda *args, **kwargs: None)
    monkeypatch.setattr(decoder, "PreparedEmbedding",
                        lambda *args, **kwargs: SimpleNamespace(functional=lambda **kw: None))
    monkeypatch.setattr(decoder, "Rotary", lambda *args, **kwargs: None)
    monkeypatch.setattr(decoder, "PreparedLayer", FakeLayer)
    monkeypatch.setattr(decoder, "PreparedDecoderBoundary", FakeBoundary)
    monkeypatch.setattr(decoder, "register_finalizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(decoder.PreparedDecoder, "_prepare_head", lambda *args, **kwargs: None)

    model_cfg = ModelConfig({
        "num_hidden_layers": 40,
        "hidden_size": 5120,
        "sliding_window": 128,
        "rms_norm_eps": 1e-6,
        "candidate_source_layer_id": 20,
        "compress_ratios": [0, 0] + [2] * 18 + [1] * 20,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "decoder_start_layer_id": 20,
    }, {})
    @dataclass(frozen=True)
    class FakeLayerWeights:
        engram: object
        attention: dict

    pool = ContinuationPool.allocate(model_cfg, 12, 2, "cpu")
    layers = tuple(FakeLayerWeights(engram=None, attention={}) for _ in range(40))
    weights = SimpleNamespace(
        config=model_cfg,
        draft=None,
        layers=layers,
        shard=TensorParallelShard(0, 4),
        embedding=torch.zeros((16, 5120), dtype=torch.bfloat16),
        norm=torch.ones(5120, dtype=torch.bfloat16),
        head=torch.zeros((16, 5120), dtype=torch.bfloat16),
    )

    examples = {}
    decoder_storage = Scratch.allocate(32, 48, 4, "cpu")
    verify_decoders = []
    for batch, packed_tokens in ((8, 32), (4, 16), (1, 2)):
        meta = Metadata.allocate(packed_tokens, batch, 2, "cpu", samples=batch * 6, max_query_tokens=6)
        verify_decoders.append(decoder.PreparedDecoder(
            weights, None, decoder_storage.rows(packed_tokens, batch * 6), meta, pool,
            mode="verify", runtime=None, vocabulary_runtime=None, coordinator=None,
            session=None, name=f"ver_{batch}_{packed_tokens}", workspace=None, examples=examples,
        ))

    assert len(ring_commits) == 3
    base_ptr = ring_commits[0].data_ptr()
    base_storage = ring_commits[0].untyped_storage()._cdata
    for dec_v, staged, (batch, _) in zip(verify_decoders, ring_commits, ((8, 32), (4, 16), (1, 2)), strict=True):
        assert dec_v.staging_shape == (40, batch, (128 + 6) * 528)
        assert dec_v.window_commit.staging is None
        assert staged.shape == dec_v.staging_shape and staged.is_contiguous()
        assert staged.data_ptr() == base_ptr
        assert staged.untyped_storage()._cdata == base_storage

    first_verify_layers = [ws for name, _, _, ws in layer_records if name.startswith("ver_8_32.")]
    assert len(first_verify_layers) == 40
    layer_stride = 8 * (128 + 6) * 528
    for idx, ws in enumerate(first_verify_layers):
        assert ws.data_ptr() == base_ptr + idx * layer_stride

    ring_commits.clear()
    for tag in ("solo_v1", "solo_v2"):
        meta = Metadata.allocate(8, 2, 2, "cpu", samples=12, max_query_tokens=6)
        decoder.PreparedDecoder(
            weights, None, decoder_storage.rows(8, 12), meta, pool,
            mode="verify", runtime=None, vocabulary_runtime=None, coordinator=None,
            session=None, name=tag, workspace=None, examples=None,
        )
    assert ring_commits[0].untyped_storage()._cdata != ring_commits[1].untyped_storage()._cdata

    layer_records.clear()
    boundary_examples.clear()
    prefill_storage = Scratch.allocate(1024, 1, 4, "cpu", candidate_tokens=128)
    prefill_decoders = {}
    for tokens in (1024, 256, 128):
        meta = Metadata.allocate(tokens, 1, 10, "cpu", max_query_tokens=tokens)
        prefill_decoders[tokens] = decoder.PreparedDecoder(
            weights, None, prefill_storage.rows(tokens, 1), meta, pool,
            mode="prefill", runtime=None, vocabulary_runtime=None, coordinator=None,
            session=None, name=f"tp_{tokens}", workspace=None, examples=examples, encoder_only=False,
        )

    shared_suffix = prefill_decoders[1024].output_scratch
    assert prefill_decoders[256].output_scratch is shared_suffix
    assert shared_suffix.embedding.shape == (128, 5120)
    assert prefill_decoders[128].packing is None
    assert prefill_decoders[128].output_scratch.embedding.data_ptr() == prefill_storage.embedding.data_ptr()
    for suffix_tensor, full_tensor in zip(
        _flatten_scratch_tensors(shared_suffix), _flatten_scratch_tensors(prefill_storage), strict=True
    ):
        assert not _overlap(suffix_tensor, full_tensor)

    tp256_layers = [(idx, scr) for name, idx, scr, _ in layer_records if name.startswith("tp_256.")]
    assert len(tp256_layers) == 40
    assert all(scr.embedding.shape[0] == 256 and scr.embedding.data_ptr() == prefill_storage.embedding.data_ptr()
               for idx, scr in tp256_layers[:20])
    assert all(scr is shared_suffix for idx, scr in tp256_layers[20:])
    assert boundary_examples == [(shared_suffix, examples), (shared_suffix, examples)]

    solo_prefills = []
    for tokens in (1024, 256):
        meta = Metadata.allocate(tokens, 1, 10, "cpu", max_query_tokens=tokens)
        solo_prefills.append(decoder.PreparedDecoder(
            weights, None, prefill_storage.rows(tokens, 1), meta, pool,
            mode="prefill", runtime=None, vocabulary_runtime=None, coordinator=None,
            session=None, name=f"solo_tp_{tokens}", workspace=None, examples=None, encoder_only=False,
        ))
    assert (solo_prefills[0].output_scratch.embedding.untyped_storage()._cdata
            != solo_prefills[1].output_scratch.embedding.untyped_storage()._cdata)
    assert (solo_prefills[0].output_scratch.embedding.untyped_storage()._cdata
            != shared_suffix.embedding.untyped_storage()._cdata)
