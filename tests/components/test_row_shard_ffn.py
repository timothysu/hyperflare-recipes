# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Packed route records, prequantized MXFP8 expert lowering, and preencoded MLP forwarding."""

from types import SimpleNamespace
import weakref

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from b12x._lib.scratch import ScratchBufferSpec, scratch_tensor
from hyperflare_recipes.components import block_fp8, mlp, moe
from hyperflare_recipes.components.weights import MLPWeights
from hyperflare_recipes.fixtures import fused_moe as moe_fixtures
from ignition.preparation.scratch import ScratchArena


@pytest.mark.parametrize("rows", [1, 17, 128])
@pytest.mark.parametrize("top_k", [2, 4, 6, 8, 10])
def test_route_record_pack_unpack_preserves_ids_and_exact_fp32_weight_bits(rows, top_k):
    allocated = moe.RouteRecord.allocate(rows, top_k, "cpu")
    assert allocated.top_k == top_k
    assert allocated.words.dtype == torch.int32
    assert allocated.words.shape == (rows, 2 * top_k)
    assert not allocated.words.count_nonzero()
    assert allocated.packed.dtype == torch.bfloat16
    assert allocated.packed.shape == (rows, 4 * top_k)
    assert allocated.packed.is_contiguous()
    assert (allocated.packed.shape[1] * allocated.packed.element_size()) % moe.ROUTE_PACK_BYTES == 0

    rank_ids, rank_weights, rank_packed = [], [], []
    for rank in range(4):
        ids = ((torch.arange(rows * top_k, dtype=torch.int32).reshape(rows, top_k) * 7 + rank * 19) % 384)
        weights = torch.linspace(
            0.031250119 + rank * 0.01,
            0.968750358 + rank * 0.01,
            rows * top_k,
            dtype=torch.float32,
        ).reshape(rows, top_k)
        if rows > 1:
            ids[-1].fill_(-1)
            weights[-1].zero_()
        # Ensure test weights carry mantissa bits below BF16 precision so any numeric cast fails.
        assert not torch.equal(weights.to(torch.bfloat16).float(), weights)

        record = moe.RouteRecord.empty(rows, top_k, "cpu")
        record.write(ids, weights)
        rank_ids.append(ids)
        rank_weights.append(weights)
        rank_packed.append(record.packed.clone())

    gathered_bf16 = torch.cat(rank_packed, dim=0)
    expected_ids = torch.cat(rank_ids, dim=0)
    expected_weights = torch.cat(rank_weights, dim=0)
    unpacked_ids = torch.full_like(expected_ids, -99)
    unpacked_weights = torch.full_like(expected_weights, -1.0)
    moe.RouteRecord(gathered_bf16.view(torch.int32)).read(unpacked_ids, unpacked_weights)

    assert torch.equal(unpacked_ids, expected_ids)
    assert torch.equal(unpacked_weights.view(torch.int32), expected_weights.view(torch.int32))

    selected_logits = torch.zeros((rows, top_k), dtype=torch.float32)
    router_view = allocated.router_scratch(selected_logits)
    router_view.ids.copy_(rank_ids[0])
    router_view.weights.copy_(rank_weights[0])
    check_ids = torch.empty_like(rank_ids[0])
    check_weights = torch.empty_like(rank_weights[0])
    moe.RouteRecord(allocated.packed.view(torch.int32)).read(check_ids, check_weights)
    assert torch.equal(check_ids, rank_ids[0])
    assert torch.equal(check_weights.view(torch.int32), rank_weights[0].view(torch.int32))

    sliced = allocated.rows(1)
    assert sliced.words.shape == (1, 2 * top_k)
    assert torch.equal(sliced.packed, allocated.packed[:1])
    for invalid_count in (0, -1, rows + 1, True, 1.5):
        with pytest.raises(ValueError, match="route record rows exceed capacity"):
            allocated.rows(invalid_count)


@pytest.mark.parametrize(
    ("rows", "top_k"),
    [(0, 6), (-4, 6), (8, 0), (8, -2), (8, 1), (8, 3), (8, 5), (8, 7), (True, 6), (8, False), (8.0, 6), (8, 6.0)],
)
def test_route_record_rejects_unaligned_or_nonpositive_geometry(rows, top_k):
    for factory in (moe.RouteRecord.allocate, moe.RouteRecord.empty):
        with pytest.raises(ValueError, match="whole 16-byte packs"):
            factory(rows, top_k, "cpu")


@pytest.mark.parametrize("prequantized", [False, True])
def test_prequantized_mxfp8_experts_declare_mxfp8_capacity_and_bind_scales(monkeypatch, prequantized):
    tokens, hidden, top_k = 128, 256, 6
    source = (
        torch.ones((tokens, hidden), dtype=torch.float8_e4m3fn)
        if prequantized
        else torch.ones((tokens, hidden), dtype=torch.bfloat16)
    )
    a_scales = torch.full((1, tokens, hidden // 32), 121, dtype=torch.uint8) if prequantized else None
    ids = torch.zeros((tokens, top_k), dtype=torch.int32)
    weights = torch.zeros((tokens, top_k), dtype=torch.float32)
    output = torch.zeros((tokens, hidden), dtype=torch.bfloat16)
    planes = {
        name: torch.zeros(16, dtype=torch.uint8)
        for name in (
            "w1_fp4", "w2_fp4", "w1_blockscale", "w2_blockscale",
            "w1_alphas", "w2_alphas", "a1_gscale", "a2_gscale",
        )
    }
    experts = SimpleNamespace(
        num_experts=384,
        has_csf_blocks=False,
        plan=SimpleNamespace(activation=SimpleNamespace(outer_scale="token")),
        _impl=SimpleNamespace(**planes, representation=None, nvfp4_csf=None),
    )
    arena = ScratchArena("cpu")
    arena.reserve(256)
    arena.freeze()

    seen_capacities, bound_calls, finalizers, requests, resources = [], [], [], [], {}
    session = SimpleNamespace(prepare=lambda declarations: requests.extend(declarations))

    def specs(size):
        return (ScratchBufferSpec(name="workspace", shape=(size,), dtype=torch.uint8, device=torch.device("cpu")),)

    plan = SimpleNamespace(
        scratch_specs=lambda: specs(32),
        request=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    def plan_execution(**kwargs):
        seen_capacities.append(kwargs["capacity"])
        return plan

    def bind(plan_obj, **kwargs):
        bound_calls.append(("finalize", kwargs))
        return SimpleNamespace(**kwargs)

    def resource(session_obj, key, factory):
        if key not in resources:
            resources[key] = factory()
        return resources[key]

    monkeypatch.setattr(moe.fused_moe, "plan_execution", plan_execution)
    monkeypatch.setattr(moe.fused_moe, "bind", bind)
    monkeypatch.setattr(moe, "register_finalizer", lambda _, callback, **kwargs: finalizers.append(callback))
    monkeypatch.setattr(moe_fixtures, "benchmark_resource", resource)

    operator = moe.PreparedRoutedExperts(
        source,
        experts,
        ids,
        weights,
        output,
        session=session,
        name="prefill.layer0.experts",
        workspace=arena,
        a_scales=a_scales,
    )
    assert operator.prequantized_input is prequantized
    assert seen_capacities[0].input_format == ("mxfp8" if prequantized else "bf16")

    def candidate_state(tag):
        def bind_state(**kwargs):
            scratch_tensor(kwargs["scratch"], specs(32), owner="MoE")
            bound_calls.append((tag, kwargs))
            return SimpleNamespace(**kwargs)

        return SimpleNamespace(
            scratch=SimpleNamespace(scratch_specs=lambda: specs(32)),
            bind=bind_state,
            run=lambda binding: None,
        )

    bench = requests[0].benchmark_call(candidate_state("benchmark"))
    fixture = resources[("moe_inputs", "prefill.layer0.experts")]
    if prequantized:
        assert fixture.a_scales is not None and fixture.a_scales_fixture is not None
        assert fixture.a_scales.shape == a_scales.shape and fixture.a_scales.dtype == torch.uint8
        assert torch.all(fixture.a_scales_fixture == 127)
        fixture.a_scales.zero_()
        bench.produce()
        assert torch.all(fixture.a_scales == 127)
        assert bound_calls[-1][1]["a_scales"] is fixture.a_scales
    else:
        assert fixture.a_scales is None and fixture.a_scales_fixture is None
        assert "a_scales" not in bound_calls[-1][1]

    expected_tensors = (
        fixture.fixture,
        fixture.source,
        fixture.a_scales_fixture,
        fixture.a_scales,
        fixture.output,
        fixture.routes,
        *(getattr(fixture.router, field) for field in fixture.router.__dataclass_fields__),
        fixture.route_weights,
    )
    expected_storage = {
        tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
        for tensor in expected_tensors
        if tensor is not None
    }
    assert fixture.nbytes == sum(expected_storage.values())

    requests[0].prepare_call(candidate_state("prepare"))
    for callback in finalizers:
        callback()
    for tag, kwargs in bound_calls:
        if tag in ("prepare", "finalize"):
            if prequantized:
                assert kwargs["a_scales"] is a_scales
            else:
                assert "a_scales" not in kwargs

    source_ref = weakref.ref(source)
    scales_ref = None if a_scales is None else weakref.ref(a_scales)
    operator.functional(session=None)
    assert operator.source is None and operator.a_scales is None and operator.output is None
    assert source_ref() is not None  # local variable still alive
    assert scales_ref is None or scales_ref() is not None


def test_prequantized_mxfp8_functional_experts_lower_to_prepared_moe_mxfp8_out():
    def make_owner(prequantized: bool):
        owner = moe.PreparedRoutedExperts.__new__(moe.PreparedRoutedExperts)
        owner.output = owner.source = owner.output_source = owner.a_scales = None
        owner.prequantized_input = prequantized
        owner.output_layout = ((128, 256), (256, 1), torch.bfloat16)
        owner.workspace_nbytes = 512
        owner.scale_inputs, owner.scale_layouts = [], ()
        owner.integrated_routing, owner.grouped_output = False, False
        owner.num_experts, owner.precision = 384, {}
        owner.plan = SimpleNamespace(handle=7)
        return owner

    owner = make_owner(prequantized=True)
    source = torch.ones((128, 256), dtype=torch.float8_e4m3fn)
    a_scales = torch.full((1, 128, 8), 127, dtype=torch.uint8)
    ids = torch.zeros((128, 6), dtype=torch.int32)
    weights = torch.full((128, 6), 1.0 / 6.0, dtype=torch.float32)

    with pytest.raises(ValueError, match="require a_scales"):
        owner(source, ids, weights)

    plain_owner = make_owner(prequantized=False)
    with pytest.raises(ValueError, match="requires prequantized MXFP8 expert preparation"):
        plain_owner(torch.ones((128, 256), dtype=torch.bfloat16), ids, weights, a_scales=a_scales)

    graph = make_fx(
        lambda src, scales, topk_ids, topk_weights: owner(src, topk_ids, topk_weights, a_scales=scales),
        tracing_mode="fake",
    )(source, a_scales, ids, weights)
    mxfp8_calls = [
        node for node in graph.graph.nodes if node.target == torch.ops.b12x.prepared_moe_mxfp8_out.default
    ]
    bf16_calls = [
        node for node in graph.graph.nodes if node.target == torch.ops.b12x.prepared_moe_out.default
    ]
    assert len(mxfp8_calls) == 1 and not bf16_calls
    placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
    assert mxfp8_calls[0].args[:4] == (placeholders[0], placeholders[1], placeholders[3], placeholders[2])
    assert mxfp8_calls[0].args[4].target == torch.ops.aten.empty_strided.default
    assert mxfp8_calls[0].args[5].target == torch.ops.aten.empty.memory_format


def test_prepared_mlp_forwards_encoded_and_encoded_bases_to_gate_up_projection(monkeypatch):
    tokens, hidden, intermediate = 8, 16, 12
    constructed, calls = [], []

    class FakeQuantizedWeight:
        def __init__(self, matrix: torch.Tensor):
            self.matrix = matrix

    class RecordingQuantizedProjection:
        def __init__(self, source, weight, output, *, session, name, workspace,
                     activation_mode="auto", bias=None, encoded=None):
            self.weight = weight.matrix
            self.output = output
            self.name = name
            self.encoded = encoded
            constructed.append((name, encoded))

        def functional(self, *, session):
            self.output_layout = (tuple(self.output.shape), self.output.stride(), self.output.dtype)
            self.output = None
            return self

        def __call__(self, source, **kwargs):
            calls.append((self.name, kwargs))
            operand = kwargs["encoded_bases"][0] if "encoded_bases" in kwargs else source
            projected = (operand.float() @ self.weight.float().T).to(torch.bfloat16)
            if self.output is not None:
                return self.output.copy_(projected)
            shape, stride, dtype = self.output_layout
            out = torch.empty_strided(shape, stride, dtype=dtype, device=operand.device)
            return out.copy_(projected)

    def fake_plan(caps, *, invocation):
        return SimpleNamespace(
            caps=caps,
            invocation=invocation,
            request=lambda **kwargs: SimpleNamespace(**kwargs),
        )

    def fake_swiglu(gate_up, *, limit, round_silu, out, plan):
        width = out.shape[1]
        gate = torch.nn.functional.silu(gate_up[:, :width].float())
        up = gate_up[:, width:].float()
        if round_silu:
            gate = gate.to(torch.bfloat16).float()
        out.copy_((gate * up).clamp(max=limit).to(torch.bfloat16))

    monkeypatch.setattr(mlp, "QuantizedProjection", RecordingQuantizedProjection)
    monkeypatch.setattr(mlp.hyperconnection, "plan", fake_plan)
    monkeypatch.setattr(mlp.hyperconnection, "run_swiglu", fake_swiglu)

    torch.manual_seed(42)
    gate_up_mat = torch.randn((2 * intermediate, hidden), dtype=torch.bfloat16) * 0.25
    down_mat = torch.randn((hidden, intermediate), dtype=torch.bfloat16) * 0.25
    weights = MLPWeights(gate_up=FakeQuantizedWeight(gate_up_mat), down=FakeQuantizedWeight(down_mat))
    scratch = mlp.MLPScratch.allocate(tokens, hidden, intermediate, "cpu")
    source_template = torch.zeros((tokens, hidden), dtype=torch.bfloat16)
    encoded_template = object()
    declarations = []
    session = SimpleNamespace(prepare=lambda reqs: declarations.extend(reqs))

    owner = mlp.PreparedMLP(
        source_template,
        weights,
        scratch.gate_up,
        scratch.intermediate,
        scratch.output,
        session=session,
        name="layer0.shared",
        workspace=ScratchArena("cpu"),
        round_silu=False,
        encoded=encoded_template,
    )
    assert constructed == [("layer0.shared.gate_up", encoded_template), ("layer0.shared.down", None)]

    runtime_bf16 = torch.randn((tokens, hidden), dtype=torch.bfloat16)
    encoded_vals = runtime_bf16.to(torch.float8_e4m3fn)
    encoded_bases = (encoded_vals, torch.full((1, tokens, 1), 127, dtype=torch.uint8), torch.empty(0, dtype=torch.uint8))

    bound_plain = owner(runtime_bf16).clone()
    assert calls == [("layer0.shared.gate_up", {}), ("layer0.shared.down", {})]
    calls.clear()

    bound_preencoded = owner(encoded_vals, encoded_bases=encoded_bases).clone()
    assert calls == [
        ("layer0.shared.gate_up", {"encoded_bases": encoded_bases}),
        ("layer0.shared.down", {}),
    ]
    calls.clear()

    owner.functional(session=session)
    assert owner.gate_up is None and owner.intermediate is None and owner.output is None
    functional_plain = owner(runtime_bf16)
    functional_preencoded = owner(encoded_vals, encoded_bases=encoded_bases)
    assert calls == [
        ("layer0.shared.gate_up", {}),
        ("layer0.shared.down", {}),
        ("layer0.shared.gate_up", {"encoded_bases": encoded_bases}),
        ("layer0.shared.down", {}),
    ]
    assert torch.equal(functional_plain, bound_plain)
    assert torch.equal(functional_preencoded, bound_preencoded)
    assert not torch.equal(functional_plain, functional_preencoded)


def test_prepared_mlp_rejects_preencoded_inputs_for_block_fp8_and_unquantized_weights(monkeypatch):
    tokens, hidden, intermediate = 4, 256, 128
    scratch = mlp.MLPScratch.allocate(tokens, hidden, intermediate, "cpu")
    source = torch.zeros((tokens, hidden), dtype=torch.bfloat16)
    session = SimpleNamespace(prepare=lambda reqs: None)

    monkeypatch.setattr(
        mlp.hyperconnection,
        "plan",
        lambda caps, *, invocation: SimpleNamespace(request=lambda **kwargs: SimpleNamespace(**kwargs)),
    )

    bf16_weights = MLPWeights(
        gate_up=torch.zeros((2 * intermediate, hidden), dtype=torch.bfloat16),
        down=torch.zeros((hidden, intermediate), dtype=torch.bfloat16),
    )
    with pytest.raises(ValueError, match="unquantized MLP projections do not support preencoded MXFP8 inputs"):
        mlp.PreparedMLP(
            source, bf16_weights, scratch.gate_up, scratch.intermediate, scratch.output,
            session=session, name="bf16_mlp", workspace=None, encoded=object(),
        )

    block_fp8_weights = MLPWeights(
        gate_up=block_fp8.BlockFP8Weight(
            torch.zeros((2 * intermediate, hidden), dtype=torch.float8_e4m3fn),
            torch.ones((2 * intermediate // 128, hidden // 128), dtype=torch.float32),
        ),
        down=torch.zeros((hidden, intermediate), dtype=torch.bfloat16),
    )
    with pytest.raises(ValueError, match="block-FP8 MLP projections do not support bias or preencoded MXFP8 inputs"):
        mlp.PreparedMLP(
            source, block_fp8_weights, scratch.gate_up, scratch.intermediate, scratch.output,
            session=session, name="fp8_mlp", workspace=None, encoded=object(),
        )
