# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed graph-count budgets and their independent preparation settings."""

from dataclasses import replace

import pytest

from ignition.cli import parse_args, preparation
from ignition.cli.config import Configuration
from ignition.errors import UsageError
from ignition.parameters import SpeculationParameters
from hyperflare_recipes.recipes import recipes, resolve


def _declared(alias):
    from ignition.registry import ResolvedModel

    for recipe in recipes():
        for checkpoint in recipe.checkpoints:
            if checkpoint.alias == alias:
                return ResolvedModel(recipe, checkpoint)
    return resolve(alias)


def selected_params(*flags, alias="deepseek-v4.1-flash", configuration=None):
    args = parse_args(["export", alias, *flags], recipes=recipes())
    return preparation.resolve_parameters(_declared(alias), args, configuration)


def selected(*flags, alias="deepseek-v4.1-flash", configuration=None):
    return selected_params(*flags, alias=alias, configuration=configuration).speculation


@pytest.mark.parametrize("command", ["export", "debug-serve"])
def test_graph_budget_flag_parses_independently_of_token_capacity(command):
    args = parse_args([command, "deepseek-v4.1-flash", "--speculative-graph-budget", "48",
                       "--speculative-token-capacity", "96"], recipes=recipes())
    assert (args.speculative_graph_budget, args.speculative_token_capacity) == (48, 96)
    assert parse_args([command, "deepseek-v4.1-flash"], recipes=recipes()).speculative_graph_budget is None


def test_graph_budget_precedence_and_token_capacity_are_independent(tmp_path):
    path = tmp_path / "flare.toml"
    default = resolve("deepseek-v4.1-flash").recipe.speculation
    assert default.graph_budget == 64
    assert selected() == default
    path.write_text('[defaults]\nspeculative_graph_budget=48\n')
    with pytest.raises(UsageError, match="unknown key"):
        Configuration.load(path)
    assert selected("--speculative-graph-budget", "24") == replace(default, graph_budget=24)
    assert selected("--speculative-token-capacity", "192") == replace(default, token_capacity=192)
    assert selected("--speculative-graph-budget", "24", "--speculative-token-capacity", "192") == replace(
        default, graph_budget=24, token_capacity=192,
    )


@pytest.mark.parametrize("alias", ["qwen3.8-flash", "glm-5.3-flash", "step5-preview-nvfp4"])
def test_explicit_graph_budget_rejects_recipes_without_packed_rounds(alias):
    with pytest.raises(UsageError, match="--speculative-graph-budget"):
        selected("--speculative-graph-budget", "32", alias=alias)


@pytest.mark.parametrize("flags,message", [
    (("--no-mtp",), "cannot accompany"),
    (("--mtp-draft-tokens", "0"), "positive draft depth"),
    (("--no-mtp-adaptive-verification",), "requires adaptive verification"),
])
def test_explicit_graph_budget_rejects_incompatible_modes(flags, message):
    with pytest.raises(UsageError, match=message):
        selected(*flags, "--speculative-graph-budget", "32")


@pytest.mark.parametrize("budget", ["0", "-1"])
def test_graph_budget_requires_a_positive_integer(budget):
    with pytest.raises(UsageError, match="positive integer"):
        selected("--speculative-graph-budget", budget)


def test_plan_serializes_graph_budget_only_for_applicable_speculation():
    from tests.cli.test_planning import derive

    packed = selected("--speculative-graph-budget", "48")
    plan = derive(speculation=packed)
    assert plan.to_dict()["speculation"]["graph_budget"] == 48
    assert plan.speculative_token_capacity == 128
    assert "graph_budget" not in derive().to_dict()["speculation"]
    assert derive(speculation=None).to_dict()["speculation"] is None


def test_planning_rejects_graph_budget_below_required_endpoints():
    from tests.cli.test_planning import derive, gpu

    with pytest.raises(UsageError, match="--speculative-graph-budget: .*at least 12"):
        selected("--speculative-graph-budget", "11")
    settings = selected("--speculative-graph-budget", "12")
    devices = (gpu(0, 95), gpu(1, 95))
    with pytest.raises(UsageError, match="--speculative-graph-budget: .*at least 12"):
        derive(gpus=devices, concurrency=32, speculation=replace(settings, graph_budget=11))
    assert derive(gpus=devices, concurrency=32,
                  speculation=settings).speculation.graph_budget == 12


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_speculation_declaration_validates_graph_budget_type_and_range(budget):
    with pytest.raises(ValueError, match="positive integer"):
        SpeculationParameters(5, packed_verification=True, graph_budget=budget)


def test_graph_budget_preserves_cpu_fixed_width_mode():
    declaration = SpeculationParameters(5, adaptive=False, packed_verification=True, graph_budget=64)
    assert declaration.fixed_width == 6 and declaration.draft_prefix_selection == "cpu"
    with pytest.raises(ValueError, match="positive learned draft depth"):
        SpeculationParameters(0, packed_verification=True, graph_budget=64)
    with pytest.raises(ValueError, match="GPU draft prefix selection"):
        SpeculationParameters(5, adaptive=False, packed_verification=True, draft_prefix_selection="gpu", graph_budget=64)


@pytest.mark.parametrize("alias", ["deepseek-v4-flash", "mimo-v2.6-pro-mopd", "mimo-v2.6-flash-rl", "kimi-k3-mxfp4-csf", "qwen3.8-27b", "qwen3.8-27b-qad"])
def test_cpu_packed_recipe_accepts_fixed_width_and_explicit_graph_budget(alias):
    value = selected("--no-mtp-adaptive-verification", "--speculative-graph-budget", "32", alias=alias)
    assert value.packed_verification and value.draft_prefix_selection == "cpu"
    assert not value.adaptive and value.graph_budget == 32


@pytest.mark.parametrize("budget", [12, 32, 48])
def test_rebuild_metadata_preserves_the_selected_physical_graph_catalog(monkeypatch, budget):
    from types import SimpleNamespace
    from google.protobuf.json_format import ParseDict
    from ignition.components.capacity import packed_speculative_shapes
    from ignition.preparation import rebuild_manifest, runtime_core_pb2 as proto
    from tests.cli.test_planning import derive, gpu

    monkeypatch.setattr("ignition.preparation.source_identity.source_identity", lambda: "test-source")
    plan = derive(gpus=(gpu(0, 95), gpu(1, 95)), concurrency=32,
                  speculation=selected("--speculative-graph-budget", str(budget)))
    metadata = rebuild_manifest.from_plan(SimpleNamespace(directory=None), plan, compiled_forwards=True)
    decoded = proto.PreparationManifest.FromString(ParseDict(metadata, proto.PreparationManifest()).SerializeToString())
    values = {p.key: p.value for p in decoded.parameters}
    assert values["speculation.graph_budget"].uint_value == budget
    arguments = (plan.concurrency, plan.speculation.draft_tokens + 1, plan.speculative_token_capacity)
    catalog = packed_speculative_shapes(*arguments, values["speculation.graph_budget"].uint_value)
    assert catalog == packed_speculative_shapes(*arguments, budget)
    assert catalog != packed_speculative_shapes(*arguments, 64)


def test_packed_rebuild_metadata_rejects_an_unresolved_graph_budget(monkeypatch):
    from types import SimpleNamespace
    from ignition.preparation.rebuild_manifest import from_plan
    from tests.cli.test_planning import derive

    monkeypatch.setattr("ignition.preparation.source_identity.source_identity", lambda: "test-source")
    plan = derive(speculation=selected())
    plan = replace(plan, parameters=replace(plan.parameters, speculation=replace(plan.speculation, graph_budget=None)))
    with pytest.raises(ValueError, match="explicit graph budget"):
        from_plan(SimpleNamespace(directory=None), plan, compiled_forwards=True)


def test_vocabulary_parallel_sampling_is_an_explicit_preparation_choice():
    from tests.cli.test_planning import derive, gpu

    params = selected_params("--vocabulary-parallel-sampling", "--draft-head-quantization", "nvfp4",
                             "--markov-head-quantization", "fp8")
    assert params.vocabulary_parallel_sampling
    assert not selected_params().vocabulary_parallel_sampling
    assert derive(model=resolve("deepseek-v4.1-flash").recipe, gpus=tuple(gpu(i, 140) for i in range(4)),
                  parameters=params).to_dict()["speculation"]["vocabulary_parallel_sampling"]
    for flags in (("--no-mtp",), ("--mtp-draft-tokens", "0")):
        with pytest.raises((UsageError, ValueError), match="cannot accompany|positive learned"):
            selected_params(*flags, "--vocabulary-parallel-sampling")
    with pytest.raises(UsageError, match="is not supported"):
        parse_args(["export", "qwen3.8-flash", "--vocabulary-parallel-sampling"], recipes=recipes())


@pytest.mark.parametrize("alias", ["qwen3.8-27b", "qwen3.8-27b-qad"])
def test_qwen_dense_packed_budget_overrides_preserve_token_capacity(alias):
    default = resolve(alias).recipe.speculation
    assert default.packed_verification and default.graph_budget == 64
    assert selected("--speculative-graph-budget", "24", alias=alias) == replace(default, graph_budget=24)
    assert selected("--speculative-graph-budget", "32", "--speculative-token-capacity", "64", alias=alias) == replace(
        default, graph_budget=32, token_capacity=64,
    )
