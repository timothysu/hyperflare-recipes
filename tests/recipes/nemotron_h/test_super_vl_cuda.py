# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Super VL NVFP4 layer numerics, vision final norm, LiLiCorr draft, and TP2 parity."""

from dataclasses import replace
import gc
import json
import os
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device")

SUPER_DEFAULT = "/data/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
DRAFT_DEFAULT = "/data/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr"


def _super_path():
    path = Path(os.environ.get("FLARE_NEMOTRON_SUPER_CHECKPOINT", SUPER_DEFAULT))
    if not path.is_dir():
        pytest.skip("set FLARE_NEMOTRON_SUPER_CHECKPOINT to the Super VL NVFP4 checkpoint")
    return path


def _draft_path():
    path = Path(os.environ.get("FLARE_NEMOTRON_SUPER_LILICORR_CHECKPOINT", DRAFT_DEFAULT))
    if not path.is_dir():
        pytest.skip("set FLARE_NEMOTRON_SUPER_LILICORR_CHECKPOINT to the Super VL LiLiCorr checkpoint")
    return path


def _dequant_tensor_fp8(reader, prefix, device):
    w = reader.read(prefix + ".weight").to(device)
    ws = reader.read(prefix + ".weight_scale").to(device).float().reshape(())
    return (w.float() * ws).bfloat16()


def _dequant_nvfp4(reader, prefix, device):
    from hyperflare_recipes.reference.qwen38_quant import decode_nvfp4

    w = reader.read(prefix + ".weight").to(device)
    s = reader.read(prefix + ".weight_scale").to(device)
    g = reader.read(prefix + ".weight_scale_2").to(device).float().reshape(())
    return decode_nvfp4(w, s, g).bfloat16()


def test_super_vl_config_pool_and_template_match_declaration():
    from hyperflare_recipes.components.parallel import TensorParallelShard
    from hyperflare_recipes.recipes import resolve
    from hyperflare_recipes.recipes.nemotron_h.config import ModelConfig
    from hyperflare_recipes.recipes.nemotron_h.draft_config import DraftConfig
    from hyperflare_recipes.recipes.nemotron_h.frontend import chat_template_asset
    from hyperflare_recipes.recipes.nemotron_h.pool import continuation_layout, page_bytes
    from hyperflare_recipes.recipes.nemotron_h.vision import VisionConfig

    path = _super_path()
    draft_path = _draft_path()
    model = resolve("nemotron-3.5-super-vl-nvfp4")
    config = ModelConfig.load(path)
    draft = DraftConfig.load(draft_path, config)
    config_draft = replace(config, draft=draft)
    vision = VisionConfig.from_dict(json.loads((path / "config.json").read_text()))
    assert vision.final_layernorm is True
    assert draft.has_own_lm_head is True
    assert draft.has_sink_bias is False
    assert draft.target_layers == (1, 22, 43, 64, 85)
    for world in (1, 2):
        shard = TensorParallelShard(0, world)
        config_draft.validate_shard(shard)
        plain = model.recipe.costs(world, None, model.checkpoint)
        spec = model.recipe.costs(world, model.recipe.speculation, model.checkpoint)
        assert continuation_layout(config, shard).row_bytes == plain.state_row_bytes
        assert continuation_layout(config_draft, shard).row_bytes == spec.state_row_bytes
        assert page_bytes(config, shard) == plain.page_bytes == spec.page_bytes
    with pytest.raises(ValueError, match="64-aligned"):
        config_draft.validate_shard(TensorParallelShard(0, 4))
    asset = chat_template_asset(path)
    assert "{%- mark -%}" in asset["text"]
    assert asset["text"].replace("{%- mark -%}", "").encode() == (path / "chat_template.jinja").read_bytes()


@torch.no_grad()
def test_super_vl_vision_final_layernorm_matches_reference(tmp_path):
    from b12x.preparation import PreparationSession
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compile_pool import CompilePool
    from ignition.preparation.scratch import ScratchArena
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from hyperflare_recipes.recipes.nemotron_h.vision import (
        FINAL_NORM_PREFIX, PREFIX, PreparedVision, VisionInputs, VisionScratch, VisionWeights,
    )
    from tests.recipes.nemotron_h.test_vision_cuda import VisionForward, set_inputs

    path = _super_path()
    weights = VisionWeights.load(inspect_indexed_checkpoint(path), "cuda:0")
    assert weights.config.final_layernorm is True
    radio_class = get_class_from_dynamic_module("modeling_radio.RadioModel", str(path))
    config_class = get_class_from_dynamic_module("configuration_radio.RadioConfig", str(path))
    config = config_class(**weights.config.radio)
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        reference = radio_class(config)
    reference.make_preprocessor_external()
    state = {
        "embeddings.patch_projection.weight": weights.patch,
        "embeddings.position_embedding": weights.tensors[PREFIX + "patch_generator.pos_embed"],
        "embeddings.cls_register_token": weights.tensors[PREFIX + "patch_generator.cls_token.token"],
        "summary_idxs": torch.tensor([0, 1], device="cuda", dtype=torch.int64),
    }
    for i in range(config.num_hidden_layers):
        prefix = PREFIX + f"blocks.{i}."
        for key, value in weights.tensors.items():
            if not key.startswith(prefix):
                continue
            name = key.removeprefix(prefix)
            target = f"encoder.layer.{i}."
            if name.startswith("attn.qkv."):
                suffix = name.removeprefix("attn.qkv.")
                for part, proj in zip(value.chunk(3, 0), ("query", "key", "value"), strict=True):
                    state[target + "attention.attention." + proj + "." + suffix] = part
            else:
                state[target + name.replace("attn.proj.", "attention.output.dense.")] = value
        for j in (1, 2):
            state[f"encoder.layer.{i}.layer_scale{j}.lambda1"] = torch.ones(1280, device="cuda", dtype=torch.bfloat16)
    reference.load_state_dict(state, strict=False, assign=True)
    reference.eval()

    torch.manual_seed(42)
    pixels = torch.randn((1, 3, 32 * 16, 32 * 16), device="cuda", dtype=torch.bfloat16)
    raw_features = reference(pixels).features[0]
    normed = F.layer_norm(
        raw_features, (1280,),
        weights.tensors[FINAL_NORM_PREFIX + "weight"],
        weights.tensors[FINAL_NORM_PREFIX + "bias"],
        config.layer_norm_eps,
    )
    merged = normed.reshape(16, 2, 16, 2, 1280).permute(0, 2, 1, 3, 4).reshape(-1, 5120).float()
    merged = (merged * torch.rsqrt(merged.square().mean(-1, keepdim=True) + 1e-5)
              * weights.tensors["mlp1.0.weight"].float()).bfloat16()
    expected = F.linear(F.linear(merged, weights.tensors["mlp1.1.weight"]).relu().square(),
                        weights.tensors["mlp1.3.weight"])
    ref32 = reference.float()
    raw32 = ref32(pixels.float()).features[0]
    normed32 = F.layer_norm(
        raw32, (1280,),
        weights.tensors[FINAL_NORM_PREFIX + "weight"].float(),
        weights.tensors[FINAL_NORM_PREFIX + "bias"].float(),
        config.layer_norm_eps,
    )
    merged32 = normed32.reshape(16, 2, 16, 2, 1280).permute(0, 2, 1, 3, 4).reshape(-1, 5120)
    merged32 = (merged32 * torch.rsqrt(merged32.square().mean(-1, keepdim=True) + 1e-5)
                * weights.tensors["mlp1.0.weight"].float())
    expected32 = F.linear(F.linear(merged32, weights.tensors["mlp1.1.weight"].float()).relu().square(),
                          weights.tensors["mlp1.3.weight"].float())
    bf16_floor = (expected.float() - expected32).norm() / expected32.norm()

    with PreparationSession(device="cuda:0", autotune=False) as session:
        arena = ScratchArena("cuda:0")
        encoder = PreparedVision(weights, VisionScratch.allocate(1024, "cuda:0"),
                                 VisionInputs.allocate("cuda:0"), session=session, workspace=arena, name="radio")
        arena.finalize()
        forward = VisionForward(encoder, 1024, "cuda:0")
        artifact_path = tmp_path / "radio_super.bin"
        with CompilePool(pixels.device) as compiler:
            compiler.submit(forward, artifact_path).result()
        with artifact_path.open("rb") as file:
            artifact = torch.compiler.load_compiled_function(file)
        count = set_inputs(forward, pixels)
        actual = artifact(forward)[:count]
    error32 = (actual.float() - expected32).norm() / expected32.norm()
    cos = F.cosine_similarity(actual.float().flatten(), expected32.flatten(), dim=0)
    assert error32 <= max(0.025, 1.15 * bf16_floor.item()), (error32.item(), bf16_floor.item())
    assert cos > 0.995, cos.item()


@pytest.mark.parametrize("tokens", [1, 17, 64, 128])
@pytest.mark.parametrize("schedule", ["serial", "shared_first"])
@torch.no_grad()
def test_super_vl_nvfp4_latent_moe_and_tp2_parity(tokens, schedule, tmp_path):
    if tokens >= 128 and schedule != "serial":
        pytest.skip("prefill MoE runs without shared-expert overlap")
    from b12x.preparation import PreparationSession
    from hyperflare_recipes.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compile_pool import CompilePool
    from ignition.preparation.scratch import ScratchArena
    from hyperflare_recipes.recipes.nemotron_h.moe import PreparedLatentMoE
    from hyperflare_recipes.recipes.nemotron_h.weights import NemotronWeightLoader
    from hyperflare_recipes.reference.qwen38_quant import CheckpointTensors

    path = _super_path()
    manifest = inspect_indexed_checkpoint(path)
    prefix = "language_model.backbone.layers.1.mixer"
    with NemotronWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 1),
                              prefixes=("language_model.backbone.layers.1.",)) as loader:
        weights_tp1, config = loader.load_layer(1), loader.config

    torch.manual_seed(123)
    source = torch.randn((tokens, config.hidden), device="cuda", dtype=torch.bfloat16) * 0.2
    count = torch.tensor([tokens], device="cuda", dtype=torch.int32)
    metadata = SimpleNamespace(live_counts=count)

    with PreparationSession(device="cuda:0", autotune=False) as session:
        arena = ScratchArena("cuda:0", overlap=schedule != "serial")
        block = PreparedLatentMoE(weights_tp1, source.clone(), config, 1, session=session, name="moe",
                                  workspace=arena, num_valid_tokens=count)
        block.functional(session=session)
        arena.finalize()
        actual_tp1 = block(source, metadata).clone()

    # Pure PyTorch dequantized NVFP4 reference.
    with CheckpointTensors(manifest, byte_budget=2 << 30) as reader:
        scores = (source.float() @ weights_tp1["router"].float().T).sigmoid()
        ids = (scores + weights_tp1["correction"].float()).topk(22, dim=-1).indices
        route_weights = scores.gather(1, ids)
        route_weights = route_weights / route_weights.sum(-1, keepdim=True) * config.text["routed_scaling_factor"]
        latent = F.linear(source, reader.read(prefix + ".fc1_latent_proj.weight").cuda())
        total = torch.zeros_like(latent, dtype=torch.float32)
        for expert in ids.unique().tolist():
            rows, cols = torch.where(ids == expert)
            w_up = _dequant_nvfp4(reader, f"{prefix}.experts.{expert}.up_proj", "cuda")
            w_down = _dequant_nvfp4(reader, f"{prefix}.experts.{expert}.down_proj", "cuda")
            hidden = F.linear(latent[rows], w_up).relu().square()
            down = F.linear(hidden, w_down)
            total.index_add_(0, rows, down.float() * route_weights[rows, cols, None])
        routed = F.linear(total.bfloat16(), reader.read(prefix + ".fc2_latent_proj.weight").cuda())
        w_sup = _dequant_nvfp4(reader, prefix + ".shared_experts.up_proj", "cuda")
        w_sdown = _dequant_nvfp4(reader, prefix + ".shared_experts.down_proj", "cuda")
        shared = F.linear(F.linear(source, w_sup).relu().square(), w_sdown)
        expected = routed + shared

    err = (actual_tp1.float() - expected.float()).norm() / expected.float().norm()
    assert err < 0.08, f"Super NVFP4 MoE vs dequantized ref error too high: {err.item():.4f}"

    # Verify TP2 mathematical equivalence by summing rank 0 and rank 1 partial outputs.
    if schedule == "serial" and tokens in (1, 17):
        del weights_tp1
        gc.collect()
        torch.cuda.empty_cache()
        partials = []
        for rank in (0, 1):
            shard = TensorParallelShard(rank, 2)
            with NemotronWeightLoader(manifest, "cuda:0", shard=shard,
                                      prefixes=("language_model.backbone.layers.1.",)) as loader:
                w_rank = loader.load_layer(1)
            with PreparationSession(device="cuda:0", autotune=False) as session:
                arena = ScratchArena("cuda:0", overlap=False)
                blk = PreparedLatentMoE(w_rank, source.clone(), config, 1, shard=shard,
                                        session=session, name=f"moe_r{rank}", workspace=arena,
                                        num_valid_tokens=count)
                blk.functional(session=session)
                arena.finalize()
                partials.append(blk(source, metadata).clone())
            del w_rank
        actual_tp2 = partials[0] + partials[1]
        tp_err = (actual_tp2.float() - actual_tp1.float()).norm() / actual_tp1.float().norm()
        assert tp_err < 0.03, f"TP2 vs TP1 MoE mismatch: {tp_err.item():.4f}"


@torch.no_grad()
def test_super_vl_mamba_and_attention_tp2_parity():
    from b12x.preparation import PreparationSession
    from hyperflare_recipes.components.paged_batch import Metadata
    from hyperflare_recipes.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.scratch import ScratchArena
    from hyperflare_recipes.recipes.nemotron_h.attention import PreparedAttention
    from hyperflare_recipes.recipes.nemotron_h.decoder import mamba_metadata
    from hyperflare_recipes.recipes.nemotron_h.mamba import PreparedMamba
    from hyperflare_recipes.recipes.nemotron_h.pool import ContinuationPool
    from hyperflare_recipes.recipes.nemotron_h.weights import NemotronWeightLoader

    path = _super_path()
    manifest = inspect_indexed_checkpoint(path)
    tokens = 16
    torch.manual_seed(321)

    # Layer 0: Mamba-2 with TensorFP8 in_proj & out_proj.
    with NemotronWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 1),
                              prefixes=("language_model.backbone.layers.0.",)) as loader:
        w0_tp1, config = loader.load_layer(0), loader.config
    source = torch.randn((tokens, config.hidden), device="cuda", dtype=torch.bfloat16) * 0.2
    m = Metadata.allocate(tokens, 1, 4, "cuda:0")
    m.set(torch.tensor([0], device="cuda"), torch.tensor([tokens], device="cuda"),
          torch.tensor([1], device="cuda"), torch.zeros((1, 4), device="cuda", dtype=torch.int32),
          torch.zeros((1, tokens), device="cuda", dtype=torch.int64),
          torch.tensor([1], device="cuda", dtype=torch.int32))
    rm = mamba_metadata(m)
    pool1 = ContinuationPool.for_model(config, 3, 4, "cuda:0")
    with PreparationSession(device="cuda:0", autotune=False) as session:
        arena = ScratchArena("cuda:0", overlap=False)
        blk = PreparedMamba(w0_tp1, source.clone(), pool1.recurrent(0), rm, config, mode="prefill",
                            session=session, name="mamba1", workspace=arena)
        blk.functional(session=session)
        arena.finalize()
        out_mamba_tp1 = blk(source, rm, pool1.recurrent(0)).clone()
    del w0_tp1, pool1

    partials = []
    for rank in (0, 1):
        shard = TensorParallelShard(rank, 2)
        with NemotronWeightLoader(manifest, "cuda:0", shard=shard,
                                  prefixes=("language_model.backbone.layers.0.",)) as loader:
            w0_r = loader.load_layer(0)
        pool_r = ContinuationPool.for_model(config, 3, 4, "cuda:0", shard=shard)
        with PreparationSession(device="cuda:0", autotune=False) as session:
            arena = ScratchArena("cuda:0", overlap=False)
            blk = PreparedMamba(w0_r, source.clone(), pool_r.recurrent(0), rm, config, mode="prefill",
                                shard=shard, session=session, name=f"mamba_r{rank}", workspace=arena)
            blk.functional(session=session)
            arena.finalize()
            partials.append(blk(source, rm, pool_r.recurrent(0)).clone())
        del w0_r, pool_r
    out_mamba_tp2 = partials[0] + partials[1]
    mamba_err = (out_mamba_tp2.float() - out_mamba_tp1.float()).norm() / out_mamba_tp1.float().norm()
    assert mamba_err < 0.01, f"Mamba TP2 vs TP1 error: {mamba_err.item():.4f}"

    # Layer 7: Attention with fused TensorFP8 QKV & TensorFP8 o_proj.
    from hyperflare_recipes.components.tensor_fp8 import TensorFP8Weight

    kv_scale = torch.ones(1, device="cuda", dtype=torch.float32)
    with NemotronWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 1),
                              prefixes=("language_model.backbone.layers.7.",)) as loader:
        w7_tp1 = loader.load_layer(7)
    assert isinstance(w7_tp1["qkv"], TensorFP8Weight)
    pool1 = ContinuationPool.for_model(config, 3, 4, "cuda:0")
    with PreparationSession(device="cuda:0", autotune=False) as session:
        arena = ScratchArena("cuda:0", overlap=False)
        k1, v1 = pool1.attention(7, config)
        blk = PreparedAttention(w7_tp1, source.clone(), m, k1, v1, config, mode="prefill",
                                key_scale=kv_scale, value_scale=kv_scale,
                                session=session, name="attn1", workspace=arena)
        blk.functional(session=session)
        arena.finalize()
        out_attn_tp1 = blk(source, m, k1, v1).clone()
    del w7_tp1, pool1

    partials = []
    for rank in (0, 1):
        shard = TensorParallelShard(rank, 2)
        with NemotronWeightLoader(manifest, "cuda:0", shard=shard,
                                  prefixes=("language_model.backbone.layers.7.",)) as loader:
            w7_r = loader.load_layer(7)
        pool_r = ContinuationPool.for_model(config, 3, 4, "cuda:0", shard=shard)
        with PreparationSession(device="cuda:0", autotune=False) as session:
            arena = ScratchArena("cuda:0", overlap=False)
            kr, vr = pool_r.attention(7, config, shard=shard)
            blk = PreparedAttention(w7_r, source.clone(), m, kr, vr, config, mode="prefill",
                                    shard=shard, key_scale=kv_scale, value_scale=kv_scale,
                                    session=session, name=f"attn_r{rank}", workspace=arena)
            blk.functional(session=session)
            arena.finalize()
            partials.append(blk(source, m, kr, vr).clone())
        del w7_r, pool_r
    out_attn_tp2 = partials[0] + partials[1]
    attn_err = (out_attn_tp2.float() - out_attn_tp1.float()).norm() / out_attn_tp1.float().norm()
    assert attn_err < 0.01, f"Attention TP2 vs TP1 error: {attn_err.item():.4f}"


@torch.no_grad()
def test_super_vl_lilicorr_draft_and_own_head():
    from b12x.preparation import PreparationSession
    from hyperflare_recipes.components.paged_batch import Metadata
    from hyperflare_recipes.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.scratch import ScratchArena
    from hyperflare_recipes.recipes.nemotron_h.config import ModelConfig
    from hyperflare_recipes.recipes.nemotron_h.decoder import Scratch
    from hyperflare_recipes.recipes.nemotron_h.dflash import PreparedContext, PreparedDraft
    from hyperflare_recipes.recipes.nemotron_h.draft_config import DraftConfig
    from hyperflare_recipes.recipes.nemotron_h.draft_weights import DraftWeights
    from hyperflare_recipes.recipes.nemotron_h.pool import ContinuationPool
    from hyperflare_recipes.recipes.nemotron_h.weights import NemotronWeightLoader

    target_path, draft_path = _super_path(), _draft_path()
    config = ModelConfig.load(target_path)
    config = replace(config, draft=DraftConfig.load(draft_path, config))
    with NemotronWeightLoader(inspect_indexed_checkpoint(target_path), "cuda:0", shard=TensorParallelShard(0, 1),
                              prefixes=("language_model.backbone.embeddings.", "language_model.lm_head.")) as loader:
        embedding = loader.load_tensor("language_model.backbone.embeddings.weight")
        target_head = loader.load_linear("language_model.lm_head")
    draft_weights = DraftWeights.load(draft_path, config.draft, "cuda:0")
    assert draft_weights.head is not None
    assert all(layer["sink"] is None for layer in draft_weights.layers)
    weights = SimpleNamespace(config=config, shard=TensorParallelShard(0, 1),
                              draft=draft_weights, embedding=embedding, head=target_head)

    batch, tokens = 2, 16
    torch.manual_seed(99)
    pool = ContinuationPool.for_model(config, batch + 2, 2, "cuda:0")
    starts = torch.tensor([0, 4], device="cuda", dtype=torch.int64)
    counts = torch.tensor([6, 8], device="cuda", dtype=torch.int64)
    rows = torch.arange(1, batch + 1, device="cuda", dtype=torch.int64)
    table = torch.zeros((batch, 4), device="cuda", dtype=torch.int32)
    ids = torch.randint(0, config.vocabulary, (batch, 8), device="cuda", dtype=torch.int64)
    live = torch.tensor([batch], device="cuda", dtype=torch.int32)
    m = Metadata.allocate(tokens, batch, 4, "cuda:0", samples=tokens)
    m.set(starts, torch.full_like(starts, 8), rows, table, ids, live)
    auxiliary = torch.randn((tokens, len(config.draft.target_layers) * config.hidden),
                            device="cuda", dtype=torch.bfloat16) * 0.1

    with PreparationSession(device="cuda:0", autotune=False) as session:
        arena = ScratchArena("cuda:0")
        context = PreparedContext(weights, auxiliary, m, pool, session=session, name="ctx", workspace=arena)
        scratch = Scratch.allocate(tokens, tokens, config, "cuda:0")
        draft = PreparedDraft(weights, scratch, m, pool, vocabulary_runtime=None, coordinator=None,
                              session=session, name="draft", workspace=arena)
        context.functional()
        draft.release_outputs()
        arena.finalize()
        normalized = context(auxiliary, counts, m)
        assert torch.isfinite(normalized).all()
        m.set(starts + counts, torch.full_like(starts, 8), rows, table, ids, live)
        hidden, logits, selector = draft.forward(m)
        assert torch.isfinite(hidden).all()
        assert torch.isfinite(logits).all()
        assert torch.isfinite(selector).all()
        # Verify draft used its own W4A16_NVFP4 lm_head rather than the target FP8 lm_head.
        from safetensors import safe_open
        from hyperflare_recipes.reference.qwen38_quant import decode_nvfp4
        with safe_open(draft_path / "model.safetensors", framework="pt", device="cuda:0") as reader:
            head_w = decode_nvfp4(reader.get_tensor("lm_head.weight"),
                                  reader.get_tensor("lm_head.weight_scale"),
                                  reader.get_tensor("lm_head.weight_scale_2")).bfloat16()
        expected_logits = F.linear(hidden, head_w)
        err = (logits.float() - expected_logits.float()).norm() / expected_logits.float().norm()
        assert err < 0.015, f"Draft own lm_head error: {err.item():.4f}"
        # Verify draft.__call__ reuses shortlist topk and direct score_shortlist matches apply_shortlist.
        call_logits, call_features = draft(m)
        assert isinstance(call_features, tuple) and len(call_features) == 3
        u_fast, id_fast = draft.selector.prepare_shortlists(call_logits.view(batch, 8, -1), features=call_features)
        u_slow, id_slow = draft.selector.prepare_shortlists(call_logits.view(batch, 8, -1))
        torch.testing.assert_close(u_fast, u_slow, rtol=0, atol=0)
        torch.testing.assert_close(id_fast, id_slow, rtol=0, atol=0)
        pred = id_fast[:, 1, 0]
        direct = draft.selector.score_shortlist(pred, 2, call_features, u_fast, id_fast)
        scattered = draft.selector.apply_shortlist(call_logits.view(batch, 8, -1)[:, 2].float(), pred, 2, call_features, u_fast, id_fast)
        torch.testing.assert_close(direct, scattered.gather(1, id_fast[:, 2]), rtol=0, atol=0)
        # Verify _stage_draft_kv matches stage_windows bit-for-bit on non-contiguous ContinuationPool
        # state rows (state_row > 0, where stride(0) spans the full state record) with dirty 0x7F scratch.
        from hyperflare_recipes.components.window_attention import stage_windows
        from hyperflare_recipes.recipes.nemotron_h.config import PAGE_TOKENS
        from hyperflare_recipes.recipes.nemotron_h.dflash import _stage_draft_kv

        hk0, hv0 = pool.reserved_region("draft0.key"), pool.reserved_region("draft0.value")
        assert not hk0.is_contiguous() and hk0.stride(0) > hk0[0].numel()
        hk0.copy_(torch.randn(hk0.shape, device="cuda", dtype=torch.bfloat16).clamp(-448, 448).to(torch.float8_e4m3fn))
        hv0.copy_(torch.randn(hv0.shape, device="cuda", dtype=torch.bfloat16).clamp(-448, 448).to(torch.float8_e4m3fn))
        win = config.draft.text["sliding_window"]
        kh, dim = hk0.shape[2], hk0.shape[3]
        wpages = (win + 8 + PAGE_TOKENS - 1) // PAGE_TOKENS
        cap = wpages * PAGE_TOKENS
        st_m = Metadata.allocate(16, 2, 4, "cuda:0", samples=16)
        st_m.set(torch.tensor([567, 1532], device="cuda", dtype=torch.int64),
                 torch.tensor([8, 8], device="cuda", dtype=torch.int32),
                 torch.tensor([1, 2], device="cuda", dtype=torch.int64),
                 torch.zeros((2, 4), device="cuda", dtype=torch.int32),
                 torch.zeros((2, 8), device="cuda", dtype=torch.int64),
                 torch.tensor([2], device="cuda", dtype=torch.int32))
        ck = torch.randn((16, kh, dim), device="cuda", dtype=torch.bfloat16) * 10.0
        cv = torch.randn((16, kh, dim), device="cuda", dtype=torch.bfloat16) * 10.0
        ref_k = torch.empty((2 * wpages, PAGE_TOKENS, kh, dim), device="cuda", dtype=torch.float8_e4m3fn)
        ref_v = torch.empty_like(ref_k)
        stage_windows(hk0, ck, ref_k, st_m)
        stage_windows(hv0, cv, ref_v, st_m)
        opt_k = torch.full((2 * wpages, PAGE_TOKENS, kh, dim), 0x7F, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn)
        opt_v = torch.full_like(opt_k.view(torch.uint8), 0x7F).view(torch.float8_e4m3fn)
        _stage_draft_kv(hk0, hv0, ck, cv, opt_k, opt_v, st_m.starts, st_m.state_rows, st_m.query_start_loc,
                        st_m.lengths, win, cap, kh * dim)
        for row_idx, total in enumerate((567 + 8, 1532 + 8)):
            pad32 = ((total + 31) // 32) * 32
            assert torch.equal(opt_k.view(2, cap, kh, dim)[row_idx, :pad32].view(torch.uint8),
                               ref_k.view(2, cap, kh, dim)[row_idx, :pad32].view(torch.uint8))
            assert torch.equal(opt_v.view(2, cap, kh, dim)[row_idx, :pad32].view(torch.uint8),
                               ref_v.view(2, cap, kh, dim)[row_idx, :pad32].view(torch.uint8))

        # Verify prefill-size (tokens=1024) PreparedContext on-the-fly NVFP4 dequant + torch.mm matches reference.
        pf_tokens = 1024
        pf_aux = torch.randn((pf_tokens, len(config.draft.target_layers) * config.hidden),
                             device="cuda", dtype=torch.bfloat16) * 0.1
        pf_m = Metadata.allocate(pf_tokens, 1, 16, "cuda:0", samples=1)
        pf_m.set(torch.tensor([0], device="cuda"), torch.tensor([pf_tokens], device="cuda"),
                 torch.tensor([1], device="cuda"), torch.zeros((1, 16), device="cuda", dtype=torch.int32),
                 torch.zeros((1, pf_tokens), device="cuda", dtype=torch.int64),
                 torch.tensor([1], device="cuda", dtype=torch.int32))
        pf_ctx = PreparedContext(weights, pf_aux, pf_m, pool, session=session, name="ctx_pf", workspace=arena)
        assert pf_ctx.dequant_fc is True
        pf_ctx.functional()
        pf_norm = pf_ctx(pf_aux, torch.tensor([pf_tokens], device="cuda", dtype=torch.int64), pf_m)
        with safe_open(draft_path / "model.safetensors", framework="pt", device="cuda:0") as reader:
            fc_w = decode_nvfp4(reader.get_tensor("fc.weight"),
                                reader.get_tensor("fc.weight_scale"),
                                reader.get_tensor("fc.weight_scale_2")).bfloat16()
            hn_w = reader.get_tensor("hidden_norm.weight")
        ref_proj = F.linear(pf_aux, fc_w).float()
        ref_norm = (ref_proj * torch.rsqrt(ref_proj.square().mean(-1, keepdim=True) + config.draft.text["rms_norm_eps"]) * hn_w.float()).bfloat16()
        pf_err = (pf_norm.float() - ref_norm.float()).norm() / ref_norm.float().norm()
        assert pf_err < 0.005, f"Prefill dequant draft_fc error: {pf_err.item():.4f}"
