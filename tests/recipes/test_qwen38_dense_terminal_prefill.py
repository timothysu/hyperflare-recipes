# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for Qwen3.8-27B TerminalPrefill catalog split, contract, and fused Q/K validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from hyperflare_recipes.components.head_norm_rope import qk_head_norm_rope_kv_write
from hyperflare_recipes.components.paged_batch import RotaryBatchInputs
from hyperflare_recipes.recipes.qwen38_dense import bootstrap, contract
from hyperflare_recipes.recipes.qwen38_dense.decoder import PreparedDecoder
from hyperflare_recipes.recipes.qwen38_dense.pool import ContinuationPool
from ignition.components.completion import CompletionOutputs
from ignition.components.sampling import AdmissionBuffers
from ignition.components.state_transfer import StateCommands
from ignition.preparation import runtime_core_pb2 as p
from ignition.preparation.execution import GraphSignature
from tests.recipes.test_qwen38_dense_text_only import small_config


@pytest.mark.parametrize("capacities,has_op,expected", [
    ((128, 256, 512, 1024, 2048, 4096), True, (4096,)),
    ((128, 256, 512), True, (512,)),
    ((128, 256), True, ()),
    ((128, 256, 512, 4096), False, ()),
    ((), True, ()),
])
def test_terminal_prefill_capacities_selection(capacities, has_op, expected):
    assert bootstrap._terminal_prefill_capacities(capacities, has_terminal_op=has_op) == expected


def test_non_sampling_decoder_rejects_decode_verify_and_scored():
    for mode, scoring in (("decode", None), ("verify", None), ("prefill", object())):
        with pytest.raises(ValueError, match="non-sampling decoders require unscored prefill"):
            PreparedDecoder(
                None, None, None, None,
                mode=mode, vocabulary_runtime=None, coordinator=None,
                session=None, name="bad", workspace=None, scoring=scoring, sample=False,
            )


def test_vision_contract_validates_prompt_positions_on_terminal_prefill():
    if not hasattr(p, "TERMINAL_PREFILL"):
        pytest.skip("requires TERMINAL_PREFILL protobuf operation")
    config = small_config(mtp=True)
    capacity, prefill, vocab = 2, 128, 128
    pool = ContinuationPool.for_model(config, capacity + 1, 2, "cpu", mtp=True, rotary_offset=True)
    inputs = RotaryBatchInputs.allocate(prefill, 2, "cpu", capacity=capacity)
    buffers = dict(inputs.buffers())
    admission = AdmissionBuffers.allocate("cpu", vocab_size=vocab, capacity=capacity)
    buffers.update({name: getattr(admission, name) for name in admission.__dataclass_fields__})
    buffers.update(StateCommands.allocate("cpu", capacity=capacity).buffers())
    buffers.update(CompletionOutputs.allocate((2, capacity, 24), device="cpu").buffers())
    buffers.update(
        state_pool=pool.backing,
        memory_capacity=pool.memory_capacity,
        request_pending_token=torch.zeros(capacity, dtype=torch.int64),
        sampling_grammar_mask=torch.zeros((capacity, (vocab + 31) // 32), dtype=torch.int32),
    )
    fields_with_pos = tuple(inputs.buffers())
    fields_without_pos = tuple(name for name in fields_with_pos if name != "prefill_positions")

    good_sigs = {
        "prefill128": GraphSignature("prefill", 1, prefill, fields_with_pos, speculation_enabled=True),
        "terminal_prefill128": GraphSignature("terminal_prefill", 1, prefill, fields_with_pos, speculation_enabled=True),
    }
    doc = contract.prepare(good_sigs, buffers, pool, vocabulary=vocab, image_slots=2)
    term_graph = next(g for g in doc["graphs"] if g["graph"] == "terminal_prefill128")
    assert term_graph["kind"] == p.TERMINAL_PREFILL
    assert any(inp["source"] == p.PROMPT_POSITION for inp in term_graph["inputs"])

    bad_sigs = {
        "prefill128": GraphSignature("prefill", 1, prefill, fields_with_pos, speculation_enabled=True),
        "terminal_prefill128": GraphSignature("terminal_prefill", 1, prefill, fields_without_pos, speculation_enabled=True),
    }
    with pytest.raises(ValueError, match="terminal_prefill128"):
        contract.prepare(bad_sigs, buffers, pool, vocabulary=vocab, image_slots=2)


def test_qk_head_norm_rope_kv_write_input_validation():
    tokens, q_heads, kv_heads, dim, rot = 4, 4, 2, 16, 8
    q = torch.zeros((tokens, q_heads, dim), dtype=torch.bfloat16)
    k = torch.zeros((tokens, kv_heads, dim), dtype=torch.bfloat16)
    v = torch.zeros((tokens, kv_heads, dim), dtype=torch.bfloat16)
    qw = torch.zeros((dim,), dtype=torch.bfloat16)
    kw = torch.zeros((dim,), dtype=torch.bfloat16)
    cos = torch.zeros((tokens, rot), dtype=torch.bfloat16)
    sin = torch.zeros((tokens, rot), dtype=torch.bfloat16)
    slots = torch.zeros((tokens,), dtype=torch.int64)
    q_out = torch.zeros((tokens, q_heads, dim), dtype=torch.bfloat16)
    kc = torch.zeros((2, 16, kv_heads, dim), dtype=torch.float8_e4m3fn)
    vc = torch.zeros((2, 16, kv_heads, dim), dtype=torch.float8_e4m3fn)

    # Partial None between query / q_weight / q_out must fail.
    with pytest.raises(ValueError, match="all be provided or all be None"):
        qk_head_norm_rope_kv_write(None, k, v, qw, kw, cos, sin, slots, q_out, kc, vc, eps=1e-6)
    with pytest.raises(ValueError, match="all be provided or all be None"):
        qk_head_norm_rope_kv_write(q, k, v, None, kw, cos, sin, slots, q_out, kc, vc, eps=1e-6)
    with pytest.raises(ValueError, match="all be provided or all be None"):
        qk_head_norm_rope_kv_write(q, k, v, qw, kw, cos, sin, slots, None, kc, vc, eps=1e-6)
    # Invalid eps.
    with pytest.raises(ValueError, match="eps"):
        qk_head_norm_rope_kv_write(q, k, v, qw, kw, cos, sin, slots, q_out, kc, vc, eps=-1e-6)
    # Non-power-of-two head_dim.
    k12 = torch.zeros((tokens, kv_heads, 12), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="power of two"):
        qk_head_norm_rope_kv_write(None, k12, k12, None, torch.zeros((12,), dtype=torch.bfloat16),
                                   cos, sin, slots, None,
                                   torch.zeros((2, 16, kv_heads, 12), dtype=torch.bfloat16),
                                   torch.zeros((2, 16, kv_heads, 12), dtype=torch.bfloat16), eps=1e-6)
    # Odd or oversized rotary_dim.
    with pytest.raises(ValueError, match="cosine and sine"):
        qk_head_norm_rope_kv_write(q, k, v, qw, kw, torch.zeros((tokens, 7), dtype=torch.bfloat16),
                                   torch.zeros((tokens, 7), dtype=torch.bfloat16), slots, q_out, kc, vc, eps=1e-6)
    # Non-contiguous q_out.
    with pytest.raises(ValueError, match="q_out"):
        qk_head_norm_rope_kv_write(q, k, v, qw, kw, cos, sin, slots, q_out.transpose(0, 1), kc, vc, eps=1e-6)
    # Mismatched caches.
    with pytest.raises(ValueError, match="k_cache and v_cache"):
        qk_head_norm_rope_kv_write(q, k, v, qw, kw, cos, sin, slots, q_out, kc,
                                   torch.zeros((2, 16, kv_heads, dim), dtype=torch.bfloat16), eps=1e-6)
    # Cross-device mismatch.
    with pytest.raises(ValueError, match="same device"):
        qk_head_norm_rope_kv_write(q, k, v, qw, kw.to("meta"), cos, sin, slots, q_out, kc, vc, eps=1e-6)
