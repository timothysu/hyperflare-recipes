# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM graph catalog capacities and rank geometry before loading weights."""

from dataclasses import replace

import pytest

from hyperflare_recipes.recipes.glm53.bootstrap import PreparationConfig
from hyperflare_recipes.recipes.glm53.recipe import RECIPE


def _cfg(**kwargs):
    params = PreparationConfig().parameters
    param_updates = {}
    if "prefill_chunk_size" in kwargs:
        param_updates["prefill_chunk"] = kwargs.pop("prefill_chunk_size")
    if "max_active" in kwargs:
        param_updates["max_concurrency"] = kwargs.pop("max_active")
    if "image_concurrency" in kwargs:
        param_updates["vision"] = replace(params.vision, image_concurrency=kwargs.pop("image_concurrency"))
    if "autotune_effort" in kwargs:
        param_updates["autotune_effort"] = kwargs.pop("autotune_effort")
    if "compile_decoder" in kwargs:
        param_updates["compiled_forwards"] = kwargs.pop("compile_decoder")
    mtp = kwargs.pop("mtp", None)
    fixed_width = kwargs.pop("speculative_fixed_width", None)
    if mtp is not None:
        spec = replace(RECIPE.speculation, token_capacity=None) if mtp else None
        if spec is not None and fixed_width is not None:
            spec = replace(spec, draft_tokens=fixed_width - 1, adaptive=False)
        param_updates["speculation"] = spec
    elif fixed_width is not None:
        param_updates["speculation"] = replace(
            RECIPE.speculation, draft_tokens=fixed_width - 1, adaptive=False, token_capacity=None,
        )
    if param_updates:
        params = replace(params, **param_updates)
    return PreparationConfig(parameters=params, **kwargs)


def test_default_catalog_covers_4096_prefill_and_four_requests():
    config = PreparationConfig()
    assert config.prefill_capacities == (128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096)
    assert config.batch_capacities == (4, 2, 1)
    assert config.page_capacity == 4 * (262144 // 256)
    assert _cfg(prefill_chunk_size=3072).prefill_capacities[-2:] == (2560, 3072)
    assert _cfg(max_active=1, pages=64).batch_capacities == (1,)
    assert _cfg(max_active=3).batch_capacities == (4, 2, 1)
    assert _cfg(max_active=1).image_concurrency == 1
    assert _cfg(max_active=32).image_concurrency == 4
    assert _cfg(max_active=32, image_concurrency=2).image_concurrency == 2


@pytest.mark.parametrize("values", [
    {"devices": ()}, {"devices": (0, 0)}, {"devices": (0, 1, 2)}, {"devices": (-1, 0)},
    {"context": 513}, {"context": 2097152}, {"prefill_chunk_size": 4097},
    {"context": 256, "prefill_chunk_size": 512}, {"max_active": 65}, {"pages": 0},
    {"snapshot_budget_bytes": 0}, {"headroom_bytes": -1}, {"expert_activation": "fp8"},
])
def test_invalid_prepared_geometry_rejects_before_gpu_use(values):
    with pytest.raises((ValueError, TypeError)):
        _cfg(**values)


def test_mtp_catalog_defaults_to_five_drafts_and_reserves_draft_pages():
    config = _cfg(mtp=True)
    assert config.speculative_widths == (1, 2, 3, 4, 5, 6)
    assert config.speculative_shapes == tuple((b, w) for b in (1, 2, 4) for w in range(6, 0, -1) if w == 1 or b * w <= 6)
    assert config.page_capacity == PreparationConfig().page_capacity + 4
    fixed = _cfg(mtp=True, speculative_fixed_width=6)
    assert fixed.speculative_shapes == config.speculative_shapes
    large = _cfg(mtp=True, max_active=64)
    assert len(large.speculative_shapes) == 31
    assert large.speculative_shapes[-1] == (64, 1)


@pytest.mark.parametrize("settings,world,capability,races", [
    ({}, 4, (12, 0), True),
    ({"autotune_effort": "audit"}, 4, (12, 0), True),
    ({"autotune_effort": "off"}, 4, (12, 0), False),
    ({"compile_decoder": False}, 4, (12, 0), False),
    ({}, 2, (12, 0), False),
    ({}, 4, (12, 1), False),
    ({}, 4, (10, 3), False)])
def test_context_transport_race_runs_only_on_the_qualified_cell(monkeypatch, settings, world, capability, races):
    import torch
    from hyperflare_recipes.recipes.glm53.bootstrap import races_context_transport

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    assert races_context_transport(_cfg(**settings), world, "cpu") is races
