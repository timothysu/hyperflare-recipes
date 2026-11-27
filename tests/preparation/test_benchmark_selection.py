# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark geometry selects complete production speculative rounds."""

from dataclasses import replace
from pathlib import Path

import pytest

from ignition import planning
from ignition.errors import UsageError
from ignition.parameters import PromptLogprobsParameters, SpeculationParameters
from ignition.preparation.selection import BenchmarkSelection
from hyperflare_recipes.recipes import resolve


def _plan():
    model = resolve("qwen3.8-27b")
    gpu = planning.Gpu(0, "NVIDIA GB10", "GPU-test", 128 << 30, 120 << 30, (12, 1))
    return planning.plan(model.recipe, model.checkpoint, (gpu,), selected=(0,),
                         parameters=replace(model.recipe.parameters, max_concurrency=32),
                         reserve_bytes=0)


def test_complete_rounds_and_context_backing():
    selection = BenchmarkSelection()
    assert selection.speculative_round_shapes == ((1, 8, 8), (2, 8, 16), (4, 8, 32))
    assert selection.prefill_capacities(4096) == (512, 2048, 4096)
    assert selection.physical_pages(128) == 517
    assert selection.physical_pages(128) * 128 > 4 * (16384 + 8)


def test_kimi_tp16_eight_request_benchmark_plan():
    model = resolve("kimi-k3.1-pro")
    gpus = tuple(planning.Gpu(rank, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{rank}",
                             96 << 30, 95 << 30, (12, 0)) for rank in range(16))
    selection = BenchmarkSelection(batches=(8,), contexts=(2048, 8192), width=8)
    params = replace(
        model.recipe.parameters,
        max_concurrency=8,
        online_mxfp8=replace(model.recipe.parameters.online_mxfp8, enabled=True),
        speculation=replace(model.recipe.speculation, token_capacity=64),
    )
    plan = planning.plan(model.recipe, model.checkpoint, gpus, selected=tuple(range(16)),
                         parameters=params, reserve_bytes=0)
    selected = replace(plan, benchmark_selection=selection)
    assert selected.world_size == 16 and selected.concurrency == 8
    assert selected.benchmark_selection.speculative_round_shapes == ((8, 8, 64),)
    assert selected.speculative_token_capacity == 64
    assert selection.physical_pages(2048) == 41
    with pytest.raises(UsageError, match="speculation studies require"):
        replace(plan, benchmark_selection=BenchmarkSelection(batches=(1,), speculation_study=True))


def test_fifteen_draft_native_catalog_and_completion_capacity():
    from hyperflare_recipes.recipes.qwen38_dense.bootstrap import PreparationConfig

    selection = BenchmarkSelection(batches=(1,), contexts=(4096,), width=16)
    original = _plan()
    plan = replace(
        original,
        parameters=replace(
            original.parameters,
            max_concurrency=1,
            speculation=replace(original.speculation, draft_tokens=15, token_capacity=16),
            vision=replace(original.parameters.vision, language_model_only=True),
        ),
        benchmark_selection=selection,
    )
    config = PreparationConfig(parameters=plan.parameters, draft_snapshot=Path("draft"), benchmark_selection=selection)
    assert config.speculative_round_shapes == ((1, 16, 16),)
    assert config.decode_tokens == config.decode_samples == config.completion_token_capacity == 16
    assert selection.required_context == 4128
    assert selection.physical_pages(128) == 34
    assert not selection.selected_study_widths
    with pytest.raises(ValueError, match="width capacities"):
        replace(config, parameters=replace(config.parameters, speculation=replace(config.parameters.speculation, draft_tokens=7)))
    with pytest.raises(ValueError, match="token capacity"):
        replace(
            config,
            parameters=replace(
                config.parameters,
                speculation=SpeculationParameters(draft_tokens=15, max_draft_tokens=15, token_capacity=8,
                                                  packed_verification=True, graph_budget=64),
            ),
        )


def test_prefill_catalog_has_exact_chunk_and_prefix_remainder():
    selection = BenchmarkSelection(batches=(1,), contexts=(0, 512, 4608), prefill_chunk=2048)
    assert selection.prefill_capacities(2048) == (512, 2048)
    assert selection.speculative_round_shapes == ()
    assert selection.required_context == 6658
    assert selection.physical_pages(128) == 53
    with pytest.raises(ValueError, match="match"):
        selection.prefill_capacities(4096)


@pytest.mark.parametrize("mtp", [False, True])
def test_prefill_plan_and_recipe_accept_no_decode_catalog(mtp):
    from hyperflare_recipes.recipes.qwen38_dense.bootstrap import PreparationConfig

    selection = BenchmarkSelection(batches=(1,), contexts=(0, 512, 4096), prefill_chunk=2048)
    plan = _plan()
    plan = replace(
        plan,
        parameters=replace(
            plan.parameters,
            prefill_chunk=2048,
            speculation=plan.speculation if mtp else None,
            vision=replace(plan.parameters.vision, language_model_only=True),
        ),
        benchmark_selection=selection,
    )
    assert plan.benchmark_selection.prefill_chunk == 2048
    config = PreparationConfig(
        parameters=replace(plan.parameters, speculation=SpeculationParameters(draft_tokens=0, max_draft_tokens=15) if mtp else None),
        draft_snapshot=Path("draft") if mtp else None,
        benchmark_selection=selection,
    )
    assert config.speculative_round_shapes == ()
    with pytest.raises(ValueError, match="match"):
        replace(config, parameters=replace(config.parameters, prefill_chunk=4096))
    with pytest.raises(UsageError, match="prepared context"):
        replace(plan, recipe=replace(plan.recipe, context=6144))
    with pytest.raises(UsageError, match="match"):
        replace(plan, parameters=replace(plan.parameters, prefill_chunk=4096))


@pytest.mark.parametrize("values", [dict(batches=()), dict(batches=(1, 3)), dict(batches=(1, 16)),
    dict(batches=(2, 1)), dict(contexts=(512, 512)), dict(contexts=(513,)),
    dict(contexts=(0,)), dict(width=0), dict(width=True), dict(width=4), dict(width=9)])
def test_invalid_selection(values):
    with pytest.raises(ValueError):
        BenchmarkSelection(**values)


@pytest.mark.parametrize("values", [
    dict(verification_prefixes=(4, 1, 8)), dict(verification_prefixes=(1, 1, 8)),
    dict(verification_prefixes=(1, 4), batches=(1, 2)), dict(verification_prefixes=(0, 8)),
    dict(verification_prefixes=(1, 9)), dict(verification_prefixes=(True, 8)),
    dict(verification_prefixes=(1, 8), prefill_chunk=4096, batches=(1,)),
    dict(verification_prefixes=(1, 8), speculation_study=True, batches=(1,)),
])
def test_invalid_serving_prefix_matrix(values):
    values = {"batches": (1,), **values}
    with pytest.raises(ValueError, match="verification prefixes"):
        BenchmarkSelection(**values)


def test_plan_retains_explicit_concurrency_and_token_budget():
    plan = replace(_plan(), benchmark_selection=BenchmarkSelection())
    assert plan.concurrency == 32
    assert plan.speculative_token_capacity == 128
    assert plan.to_dict()["benchmark_selection"]["speculative_round_shapes"] == ((1, 8, 8), (2, 8, 16), (4, 8, 32))


@pytest.mark.parametrize("mutator,match", [
    (lambda p: replace(p, parameters=replace(p.parameters, max_concurrency=2)), "max-concurrency"),
    (lambda p: replace(p, parameters=replace(p.parameters, speculation=None)), "speculative width"),
    (lambda p: replace(p, parameters=replace(p.parameters, prompt_logprobs=PromptLogprobsParameters(enabled=True))), "scored or mixed"),
    (lambda p: replace(p, benchmark_selection=BenchmarkSelection(contexts=(262144,))), "prepared context"),
])
def test_plan_rejects_unavailable_geometry(mutator, match):
    base = replace(_plan(), benchmark_selection=BenchmarkSelection())
    with pytest.raises(UsageError, match=match):
        mutator(base)


def test_recipe_config_selects_rounds_without_changing_default_catalog():
    from hyperflare_recipes.recipes.qwen38_dense.bootstrap import PreparationConfig

    normal = PreparationConfig(draft_snapshot=Path("draft"))
    selected = replace(normal, parameters=replace(normal.parameters, max_concurrency=4), benchmark_selection=BenchmarkSelection())
    assert selected.speculative_round_shapes == ((1, 8, 8), (2, 8, 16), (4, 8, 32))
    assert (selected.decode_tokens, selected.decode_samples) == (32, 32)
    assert selected.speculative_token_capacity == 128
    assert (32, 1, 32) in normal.speculative_round_shapes
    assert len(normal.speculative_round_shapes) > len(selected.speculative_round_shapes)


def test_subset_contract_without_decode():
    import torch
    from types import SimpleNamespace
    from ignition.components.state import StateLayout, StateStorage
    from ignition.preparation.execution import GraphSignature
    from hyperflare_recipes.recipes.qwen38_dense.contract import prepare

    layout = StateLayout.from_specs((("consumed_tokens", (1,), torch.int64),))
    storage = StateStorage.allocate(layout, 9, "cpu")
    pool = SimpleNamespace(layout=layout, pages={})
    buffers = {"state_pool": storage.backing, "request_pending_token": torch.zeros(4, dtype=torch.int64)}
    signature = GraphSignature("speculative", 4, 8, speculation_enabled=True, packed_token_capacity=32)
    contract = prepare({"speculative4x8p32": signature}, buffers, pool, vocabulary=128)
    assert [graph["graph"] for graph in contract["graphs"]] == ["speculative4x8p32"]


def test_packed_catalog_retains_logical_sample_capacity():
    from hyperflare_recipes.recipes.qwen38_dense.bootstrap import PreparationConfig

    base = PreparationConfig(draft_snapshot=Path("draft"))
    config = replace(
        base,
        parameters=replace(
            base.parameters, max_concurrency=8,
            speculation=replace(base.parameters.speculation, draft_tokens=7, token_capacity=16),
        ),
    )
    assert (8, 8, 16) in config.speculative_round_shapes
    assert config.decode_tokens == 16
    assert config.decode_samples == 64


@pytest.mark.parametrize("width", [8, 13, 16])
def test_study_geometry_keeps_native_setup_catalog(width):
    from hyperflare_recipes.recipes.qwen38_dense.bootstrap import PreparationConfig

    selection = BenchmarkSelection(batches=(1,), contexts=(8192,), width=width, speculation_study=True)
    plan = replace(_plan(), parameters=replace(_plan().parameters, max_concurrency=1), benchmark_selection=selection)
    assert max(plan.speculation.widths) == 8
    base = PreparationConfig(draft_snapshot=Path("draft"))
    config = replace(base, parameters=replace(base.parameters, max_concurrency=1), benchmark_selection=selection)
    assert config.speculative_round_shapes == ((1, 8, 8),)
    assert config.decode_tokens == config.decode_samples == width


def test_shared_study_catalog_sizes_pool_without_widening_native_catalog():
    from hyperflare_recipes.recipes.qwen38_dense.bootstrap import PreparationConfig

    selection = BenchmarkSelection(batches=(1,), contexts=(8192,), width=16,
                                   speculation_study=True, study_widths=(8, 13, 16))
    base = PreparationConfig(draft_snapshot=Path("draft"))
    config = replace(base, parameters=replace(base.parameters, max_concurrency=1), benchmark_selection=selection)
    assert selection.selected_study_widths == (8, 13, 16)
    assert selection.required_context == 8192 + 32
    assert selection.physical_pages(128) == 66
    assert config.speculative_round_shapes == ((1, 8, 8),)
    assert config.decode_tokens == config.decode_samples == 16
    assert selection.to_dict()["study_widths"] == (8, 13, 16)


@pytest.mark.parametrize("values", [
    dict(width=13, speculation_study=True, study_widths=(8, 16)),
    dict(width=16, speculation_study=True, study_widths=(16, 8)),
    dict(width=16, speculation_study=True, study_widths=(8, 9, 16)),
    dict(width=8, study_widths=(8,)),
])
def test_invalid_shared_study_catalog(values):
    with pytest.raises(ValueError):
        BenchmarkSelection(batches=(1,), **values)


@pytest.mark.parametrize("prefill,drafting", [(False, True), (True, True), (True, False)])
def test_nemotron_single_gpu_benchmark_plan(prefill, drafting):
    model = resolve("nemotron-bolt-24gb")
    gpu = planning.Gpu(0, "NVIDIA GB10", "GPU-test", 128 << 30, 120 << 30, (12, 1))
    plan = planning.plan(
        model.recipe, model.checkpoint, (gpu,), selected=(0,),
        parameters=replace(model.recipe.parameters, max_concurrency=1,
                           speculation=model.recipe.speculation if drafting else None),
        reserve_bytes=0,
    )
    selection = BenchmarkSelection(batches=(1,), contexts=(8192 if prefill else 4096,),
        prefill_chunk=4096 if prefill else 0)
    selected = replace(plan, benchmark_selection=selection)
    assert selected.world_size == selected.concurrency == 1
    assert selected.benchmark_selection is selection
    assert selected.benchmark_selection.speculative_round_shapes == (() if prefill else ((1, 8, 8),))
    assert selected.to_dict()["benchmark_selection"] == selection.to_dict()
    with pytest.raises(UsageError, match="speculation studies require"):
        replace(plan, benchmark_selection=BenchmarkSelection(batches=(1,), speculation_study=True))
