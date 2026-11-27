# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preparation can omit ordinary graph modes while preserving speculative execution."""

from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

import pytest
from google.protobuf.json_format import ParseDict

from ignition.cli import display, parse_args, preparation
from ignition.components.capacity import batch_capacities, prefill_capacities, serving_modes
from ignition.errors import UsageError
from ignition.parameters import MixedPrefillParameters, SpeculationParameters
from ignition.preparation import rebuild_manifest, runtime_core_pb2 as proto
from hyperflare_recipes.recipes import ResolvedModel, recipes, resolve
from tests.cli.test_planning import derive, gpu


def selected(*flags, command="export"):
    model = resolve("kimi-k3.1-pro")
    args = parse_args([command, model.checkpoint.alias, *flags], recipes=recipes())
    return preparation.resolve_parameters(model, args).speculation


@pytest.mark.parametrize("command", ["export", "debug-serve", "bench"])
def test_mtp_only_is_explicit_and_preserves_speculative_capacities(command):
    default = selected(command=command)
    assert not default.mtp_only
    assert selected("--mtp-only", command=command) == replace(default, mtp_only=True)
    assert selected("--mtp-only", command=command).widths[0] == 1


@pytest.mark.parametrize("flags,message", [
    (("--no-mtp",), "cannot accompany"),
    (("--mtp-draft-tokens", "0"), "positive learned draft depth"),
])
def test_mtp_only_rejects_disabled_speculation(flags, message):
    with pytest.raises(UsageError, match=message):
        selected("--mtp-only", *flags)
    with pytest.raises(ValueError, match="positive learned draft depth"):
        SpeculationParameters(0, mtp_only=True)


def test_mtp_only_rejects_a_recipe_without_speculation():
    model = resolve("kimi-k3.1-pro")
    recipe = replace(model.recipe, parameters=replace(model.recipe.parameters, speculation=None), plain_ranks={})
    with pytest.raises(UsageError, match="is not supported"):
        parse_args(["export", model.checkpoint.alias, "--mtp-only"], recipes=(recipe,))


def test_mtp_only_plan_display_and_rebuild_round_trip(monkeypatch):
    monkeypatch.setattr("ignition.preparation.source_identity.source_identity", lambda: "test-source")
    for mtp_only in (False, True):
        plan = derive(speculation=SpeculationParameters(3, mtp_only=mtp_only))
        assert plan.to_dict()["speculation"]["mtp_only"] is mtp_only
        for brief in (False, True):
            assert ("MTP-only graphs" in display.speculation(plan.speculation, brief=brief)) is mtp_only
        metadata = rebuild_manifest.from_plan(SimpleNamespace(directory=None), plan, compiled_forwards=True)
        encoded = ParseDict(metadata, proto.PreparationManifest())
        decoded = proto.PreparationManifest.FromString(encoded.SerializeToString())
        values = {p.key: p.value for p in decoded.parameters}
        assert values["speculation.mtp_only"].bool_value is mtp_only


@pytest.mark.parametrize("enabled,mtp_only,expected", [
    (False, False, (False,)), (True, False, (True, False)), (True, True, (True,)),
])
def test_serving_modes_select_before_graph_construction(enabled, mtp_only, expected):
    assert serving_modes(enabled, mtp_only) == expected
    with pytest.raises(ValueError, match="enabled learned speculation"):
        serving_modes(False, True)


@pytest.mark.parametrize("recipe", recipes(), ids=lambda recipe: recipe.name)
@pytest.mark.parametrize("mtp_only", [False, True])
def test_recipe_adapters_preserve_mtp_only_and_validate_direct_configs(monkeypatch, tmp_path, recipe, mtp_only):
    import huggingface_hub

    name = {"kimi31": "kimi3", "inkling2_mini": "inkling2"}.get(recipe.name, recipe.name)
    bootstrap = import_module(f"hyperflare_recipes.recipes.{name}.bootstrap")
    world = recipe.tensor_parallel[0]
    devices = tuple(replace(gpu(i, 4096), total_bytes=4096 << 30) for i in range(world))
    plan = derive(model=recipe, gpus=devices, selected=tuple(range(world)),
                  speculation=replace(recipe.speculation, mtp_only=mtp_only))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: str(tmp_path / "draft"))
    original = bootstrap.PreparationConfig
    captured = []

    class ConfigurationCaptured(Exception):
        pass

    def intercept(**kwargs):
        captured.append(original(**kwargs))
        raise ConfigurationCaptured

    monkeypatch.setattr(bootstrap, "PreparationConfig", intercept)
    with pytest.raises(ConfigurationCaptured):
        recipe.prepare(tmp_path, plan, compile_forwards=True, kernel_cache=None, progress=None)
    config, = captured
    assert config.mtp_only is mtp_only
    assert config.speculative_widths == plan.speculation.widths
    if not mtp_only:
        return
    if name in ("kimi3", "gemma4", "nemotron_h"):
        with pytest.raises(ValueError, match="--mtp-only requires enabled learned speculation with positive draft depth"):
            replace(config, draft_checkpoint=None)
    else:
        with pytest.raises(ValueError, match="positive learned draft depth"):
            replace(config.parameters.speculation, draft_tokens=0)


@pytest.mark.parametrize("concurrency,normal,only", [(1, 42, 27), (4, 104, 79)])
def test_kimi_catalog_geometry_keeps_width_one_and_vision(concurrency, normal, only):
    from hyperflare_recipes.components.prompt_logprobs import prefill_modes
    from hyperflare_recipes.recipes.kimi3.bootstrap import PreparationConfig
    from hyperflare_recipes.recipes.kimi3.recipe import RECIPE
    from hyperflare_recipes.recipes.kimi3.vision import PATCH_CAPACITIES

    catalogs = []
    for mtp_only, expected in ((False, normal), (True, only)):
        params = replace(
            RECIPE.parameters,
            max_concurrency=concurrency,
            prefill_chunk=4096,
            speculation=replace(RECIPE.speculation, draft_tokens=7, token_capacity=32, graph_budget=64, mtp_only=mtp_only),
        )
        config = PreparationConfig(parameters=params, draft_checkpoint="fixture")
        batches = batch_capacities(config.max_active)
        catalog = []
        for enabled in serving_modes(config.drafting, config.mtp_only):
            for kind, _, capacities in prefill_modes(prefill_capacities(config.prefill_chunk_size),
                                                   () if enabled else batches, None):
                catalog.extend((kind, capacity, enabled) for capacity in capacities)
            catalog.extend((kind, batch, enabled) for kind in ("init", "restore", "snapshot", "resume")
                           for batch in batches)
        catalog.extend(("speculative", shape, True) for shape in config.speculative_round_shapes)
        catalog.extend(("vision", capacity, None) for capacity in PATCH_CAPACITIES)
        assert len(catalog) == expected
        assert {batch for batch, width, _ in config.speculative_round_shapes if width == 1} == set(batches)
        assert len([graph for graph in catalog if graph[0] == "vision"]) == len(PATCH_CAPACITIES)
        catalogs.append(catalog)
    assert catalogs[1] == [graph for graph in catalogs[0] if graph[2] is not False]


def test_qwen_mixed_shapes_agree_with_the_plan_for_mtp_only():
    from hyperflare_recipes.recipes.qwen38.bootstrap import PreparationConfig

    recipe = resolve("qwen3.8-flash").recipe
    for mtp_only in (False, True):
        settings = replace(recipe.speculation, mtp_only=mtp_only)
        params = replace(
            recipe.parameters,
            max_concurrency=4,
            speculation=settings,
            mixed_prefill=MixedPrefillParameters(enabled=True),
        )
        config = PreparationConfig(parameters=params)
        assert config.mixed_shapes == params.mixed_prefill.shapes(config.max_active, config.prefill_chunk_size, settings)
        assert {shape[0] for shape in config.mixed_shapes} == ({True} if mtp_only else {True, False})


@pytest.mark.parametrize("name", ["kimi3", "qwen38_dense"])
def test_mtp_only_preserves_selected_benchmark_geometry(name):
    from ignition.preparation.selection import BenchmarkSelection

    selection = BenchmarkSelection(batches=(1, 4))
    bootstrap = import_module(f"hyperflare_recipes.recipes.{name}.bootstrap")
    recipe = import_module(f"hyperflare_recipes.recipes.{name}.recipe").RECIPE
    params = replace(
        recipe.parameters,
        speculation=replace(recipe.speculation, mtp_only=True, token_capacity=128),
    )
    if name == "qwen38_dense":
        params = replace(params, vision=replace(params.vision, language_model_only=True))
    options = dict(draft_checkpoint="fixture") if name == "kimi3" else dict(draft_snapshot="fixture")
    config = bootstrap.PreparationConfig(parameters=params, benchmark_selection=selection, **options)
    assert config.speculative_round_shapes == selection.speculative_round_shapes
