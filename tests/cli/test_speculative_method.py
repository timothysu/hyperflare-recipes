# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Learned proposal selection, configuration precedence and physical planning."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from google.protobuf import json_format

from ignition import planning
from ignition.cli import parse_args, preparation
from ignition.cli.config import Configuration
from ignition.errors import UsageError
from ignition.parameters import MixedPrefillParameters
from hyperflare_recipes.recipes import recipes, resolve
from ignition.preparation import rebuild_manifest, runtime_core_pb2 as proto


def selected(*flags, model="qwen3.8-flash", configuration=None):
    args = parse_args(["export", model, *flags], recipes=recipes())
    return preparation.resolve_parameters(resolve(model), args, configuration).speculation


def test_flash_default_mtp_and_explicit_dflash_preserve_requested_capacities():
    default = selected()
    assert default.method == "mtp" and default.draft_tokens == 5
    dflash = selected("--speculative-method", "dflash2", "--mtp-draft-tokens", "7",
                      "--speculative-token-capacity", "128")
    assert dflash.method == "dflash2"
    assert dflash.widths == tuple(range(1, 9)) and dflash.token_capacity == 128
    assert selected("--speculative-method", "mtp") == default
    assert selected("--speculative-method", "dflash2", "--mtp-draft-tokens", "0").widths == (1,)


def test_method_explicit_flags_and_rejection(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[defaults]\nspeculative_method = "dflash2"\n')
    with pytest.raises(UsageError, match="unknown key"):
        Configuration.load(path)
    assert selected("--speculative-method", "dflash2").method == "dflash2"
    assert selected("--speculative-method", "mtp").method == "mtp"
    assert selected("--no-mtp") is None
    with pytest.raises(UsageError, match="cannot accompany"):
        selected("--no-mtp", "--speculative-method", "dflash2")
    with pytest.raises(SystemExit) as error:
        selected("--speculative-method", "mtp", model="qwen3.8-27b")
    assert error.value.code == 2


def test_method_costs_and_rebuild_metadata(monkeypatch):
    model = resolve("qwen3.8-flash")
    gpu = planning.Gpu(0, "GB10", "GPU-test", 128 << 30, 127 << 30, (12, 1))
    def plan(method):
        params = replace(model.recipe.parameters, max_concurrency=4,
                         speculation=selected("--speculative-method", method))
        return planning.plan(model.recipe, model.checkpoint, (gpu,), selected=(0,),
                             parameters=params, reserve_bytes=0)
    mtp, dflash = plan("mtp"), plan("dflash2")
    assert dflash.costs.state_row_bytes == 142_821_120
    assert dflash.costs.page_bytes == 12 * 69_632
    assert mtp.costs.page_bytes == 13 * 69_632
    assert dflash.weights_bytes - mtp.weights_bytes == 1_206_140_008 - model.checkpoint.draft_bytes
    assert replace(dflash, parameters=replace(dflash.parameters, force_moe_decode_a16=True)).parameters.force_moe_decode_a16
    assert dflash.to_dict()["speculation"]["method"] == "dflash2"
    monkeypatch.setattr("ignition.preparation.source_identity.source_identity", lambda: "test-source")
    manifest = rebuild_manifest.from_plan(SimpleNamespace(directory=None), dflash, compiled_forwards=True)
    encoded = json_format.ParseDict(manifest, proto.PreparationManifest())
    decoded = proto.PreparationManifest.FromString(encoded.SerializeToString())
    values = {p.key: p.value for p in decoded.parameters}
    assert values["speculation.method"].string_value == "dflash2"
    assert values["speculation.draft_tokens"].uint_value == 5
    assert values["speculation.token_capacity"].uint_value == 128
    with pytest.raises(UsageError, match="separate prefill"):
        planning.plan(model.recipe, model.checkpoint, (gpu,), selected=(0,),
                      parameters=replace(dflash.parameters, mixed_prefill=MixedPrefillParameters(enabled=True)),
                      reserve_bytes=0)
