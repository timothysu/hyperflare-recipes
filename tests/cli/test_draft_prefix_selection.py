# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe presets and retained CLI controls for packed GPU prefix selection."""

from dataclasses import replace

import pytest

from ignition.cli import display, parse_args, preparation
from ignition.cli.config import Configuration
from ignition.errors import UsageError
from ignition.parameters import SpeculationParameters
from hyperflare_recipes.recipes import recipes, resolve


DEFAULT = SpeculationParameters(
    7, max_draft_tokens=7, token_capacity=128, packed_verification=True, draft_prefix_selection="gpu", graph_budget=64,
)


def settings(*flags, alias="deepseek-v4.1-flash", configuration=None):
    args = parse_args(["export", alias, *flags], recipes=recipes())
    return preparation.resolve_parameters(resolve(alias), args, configuration).speculation


def test_deepseek41_uses_packed_gpu_prefix_selection_without_flags():
    assert settings() == DEFAULT
    assert settings("--mtp-draft-tokens", "3") == replace(DEFAULT, draft_tokens=3)


def test_layout_selection_is_not_a_cli_argument():
    args = parse_args(["export", "deepseek-v4.1-flash"], recipes=recipes())
    assert not hasattr(args, "packed_verification")
    assert not hasattr(args, "draft_prefix_selection")


@pytest.mark.parametrize("command", ("export", "debug-serve"))
@pytest.mark.parametrize("flags", [
    ("--packed-verification",), ("--packed-verification=true",),
    ("--draft-prefix-selection", "cpu"), ("--draft-prefix-selection=gpu",),
])
def test_removed_layout_flags_are_unrecognized(command, flags, capsys):
    with pytest.raises(SystemExit) as error:
        parse_args([command, "deepseek-v4.1-flash", *flags], recipes=recipes())
    assert error.value.code == 2
    assert "unrecognized arguments:" in capsys.readouterr().err


@pytest.mark.parametrize("alias", ("qwen3.8-flash", "glm-5.3-flash", "step5-preview-nvfp4"))
def test_other_recipes_preserve_cpu_and_rectangular_defaults(alias):
    model = resolve(alias)
    assert settings(alias=alias) == model.recipe.speculation
    assert settings(alias=alias).draft_prefix_selection == "cpu"
    assert not settings(alias=alias).packed_verification
    assert not settings("--no-mtp-adaptive-verification", alias=alias).adaptive


def test_gpu_prefix_selection_is_printed():
    assert "GPU prefix selection" in display.speculation(settings())
    assert "GPU prefix selection" in display.speculation(settings(), brief=True)


@pytest.mark.parametrize("flags,message", [
    (("--no-mtp-adaptive-verification",), "requires adaptive verification"),
])
def test_incompatible_selection_fails_before_preparation(flags, message):
    with pytest.raises(UsageError, match=message):
        settings(*flags)


def test_width_one_and_disabled_speculation_preserve_their_meaning():
    assert settings("--no-mtp") is None
    assert settings("--mtp-draft-tokens", "0") == SpeculationParameters(0, max_draft_tokens=7, token_capacity=128)
    assert settings("--mtp-draft-tokens", "0", "--no-mtp-adaptive-verification") == SpeculationParameters(
        0, max_draft_tokens=7, adaptive=False, token_capacity=128,
    )


@pytest.mark.parametrize("flags", [
    ("--mtp-draft-tokens", "0"), ("--speculative-token-capacity", "128"),
    ("--no-mtp-adaptive-verification",),
])
def test_no_mtp_rejects_explicit_speculation_flags(flags):
    with pytest.raises(UsageError, match="cannot accompany"):
        settings("--no-mtp", *flags)


def test_explicit_capacity_flag_overrides_recipe_defaults_and_config_rejects_export_keys(tmp_path):
    path = tmp_path / "flare.toml"
    path.write_text('[defaults]\nspeculative_token_capacity=64\n')
    with pytest.raises(UsageError, match="unknown key"):
        Configuration.load(path)
    path.write_text("")
    configuration = Configuration.load(path)
    assert settings(configuration=configuration) == DEFAULT
    assert settings("--speculative-token-capacity", "192", configuration=configuration) == replace(
        DEFAULT, token_capacity=192,
    )


def test_plan_and_rebuild_metadata_record_prefix_selection(monkeypatch):
    from types import SimpleNamespace
    from tests.cli.test_planning import derive
    from ignition.preparation import rebuild_manifest

    monkeypatch.setattr("ignition.preparation.source_identity.source_identity", lambda: "test-source")
    plan = derive(speculation=settings())
    assert plan.speculative_token_capacity == 128
    assert plan.to_dict()["speculation"]["draft_prefix_selection"] == "gpu"
    encoded = rebuild_manifest.from_plan(SimpleNamespace(directory=None), plan, compiled_forwards=True)
    params = {entry["key"]: entry["value"] for entry in encoded["parameters"]}
    assert params["speculation.draft_prefix_selection"] == {"string_value": "gpu"}
    assert params["speculation.token_capacity"] == {"uint_value": 128}
