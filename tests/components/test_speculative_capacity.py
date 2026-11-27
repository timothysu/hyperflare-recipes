# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The same target-token envelope governs every speculative recipe."""

from dataclasses import replace

import pytest

from ignition.components.capacity import batch_capacities, proposal_shapes, speculative_shapes
from ignition.parameters import SpeculationParameters
from hyperflare_recipes.recipes.qwen38.bootstrap import PreparationConfig as Qwen
from hyperflare_recipes.recipes.glm53.bootstrap import PreparationConfig as Glm
from hyperflare_recipes.recipes.deepseek4.bootstrap import PreparationConfig as DeepSeek4
from hyperflare_recipes.recipes.deepseek41.bootstrap import PreparationConfig as DeepSeek41
from hyperflare_recipes.recipes.mimo_v2.bootstrap import PreparationConfig as Mimo
from hyperflare_recipes.recipes.step5.bootstrap import PreparationConfig as Step5


def _with_spec(config_cls, *, max_active=None, drafts=None, token_capacity=None,
               enabled=True, adaptive=True, packed=False, prefix="cpu", **extra):
    base = config_cls().parameters
    params = base if max_active is None else replace(base, max_concurrency=max_active)
    if not enabled:
        params = replace(params, speculation=None)
    elif drafts is not None:
        packed_active = packed and drafts > 0
        spec = SpeculationParameters(
            draft_tokens=drafts,
            max_draft_tokens=max(drafts, 7),
            adaptive=adaptive,
            token_capacity=token_capacity,
            packed_verification=packed_active,
            draft_prefix_selection=prefix if packed_active else "cpu",
            graph_budget=64 if packed_active else None,
        )
        params = replace(params, speculation=spec)
    return config_cls(parameters=params, **extra)


@pytest.mark.parametrize("concurrency", [1, 3, 4, 5, 8, 16, 33, 64])
@pytest.mark.parametrize("drafts", [0, 1, 3, 5, 7])
@pytest.mark.parametrize("override", [None, 8, 32])
def test_recipes_share_bounded_target_coverage(concurrency, drafts, override):
    widths = tuple(range(1, drafts + 2))
    capacity = max(batch_capacities(concurrency)[-1], drafts + 1) if override is None else override
    expected = {(b, w) for b in batch_capacities(concurrency) for w in widths
                if w == 1 or b * w <= capacity}
    for recipe in (Qwen, Glm):
        config = _with_spec(recipe, max_active=concurrency, drafts=drafts, token_capacity=override)
        assert set(config.speculative_shapes) == expected
    for recipe in (DeepSeek4, Mimo):
        config = _with_spec(recipe, max_active=concurrency, drafts=drafts, token_capacity=override, packed=True)
        assert {(batch, width) for batch, width, packed in config.speculative_round_shapes if width == 1} == {
            (batch, 1) for batch in batch_capacities(concurrency)}
        assert all(width == 1 or (width == drafts + 1 and batch < packed <= min(capacity, batch * width))
                   for batch, width, packed in config.speculative_round_shapes)
    config = _with_spec(DeepSeek41, max_active=concurrency, drafts=drafts, token_capacity=override,
                        packed=True, prefix="gpu")
    assert set(config.speculative_shapes) == {(b, 1) for b in batch_capacities(concurrency)}
    assert all(b < p <= min(capacity, b * (drafts + 1)) for b, p in config.packed_speculative_shapes)
    if drafts <= 3:
        config = _with_spec(Step5, max_active=concurrency, drafts=drafts, token_capacity=override)
        assert set(config.speculative_shapes) == expected
    assert (1, drafts + 1) in expected
    assert all((b, 1) in expected for b in batch_capacities(concurrency))


def test_proposals_keep_full_depth_independently_of_target_coverage():
    assert len(proposal_shapes(4, 7)) == 3
    assert len(speculative_shapes(4, tuple(range(1, 9)), 8)) == 14
    assert set(proposal_shapes(4, 7)) == {(b, 7) for b in (1, 2, 4)}
    assert {w for b, w in speculative_shapes(4, tuple(range(1, 9)), 8) if b == 4} == {1, 2}


@pytest.mark.parametrize("capacity", [0, -1, 7, True, 8.0])
def test_insufficient_or_noninteger_capacity_is_rejected(capacity):
    with pytest.raises(ValueError):
        speculative_shapes(4, tuple(range(1, 9)), capacity)


@pytest.mark.parametrize("recipe", [DeepSeek4, Mimo])
def test_parallel_rounds_bound_physical_rows_independently_of_samples(recipe):
    config = _with_spec(recipe, max_active=2, drafts=3, token_capacity=4, packed=True)
    assert set(config.speculative_round_shapes) == {(1, 1, 1), (2, 1, 2),
                                                   (1, 4, 2), (1, 4, 3), (1, 4, 4),
                                                   (2, 4, 3), (2, 4, 4)}
    assert config.decode_tokens == 4 and config.decode_samples == 8
    fixed = _with_spec(recipe, max_active=2, drafts=2, token_capacity=4, adaptive=False, packed=True)
    assert {width for _, width, _ in fixed.speculative_round_shapes} == {1, 3}
    assert fixed.decode_samples == 6


def test_kimi_parallel_rounds_have_packed_target_capacity():
    from hyperflare_recipes.recipes.kimi3.bootstrap import PreparationConfig

    config = _with_spec(PreparationConfig, max_active=2, drafts=3, token_capacity=4, packed=True,
                        draft_checkpoint="/tmp/dspark")
    assert (2, 4, 3) in config.speculative_round_shapes
    assert config.decode_tokens == 4 and config.decode_samples == 8


def test_shared_catalogs_cover_fifteen_drafts_with_bounded_target_rows():
    from ignition.components.capacity import parallel_speculative_shapes

    widths = tuple(range(1, 17))
    shapes = speculative_shapes(4, widths, 32)
    assert (1, 16) in shapes and (2, 16) in shapes and (4, 16) not in shapes
    assert set(proposal_shapes(4, 15)) == {(1, 15), (2, 15), (4, 15)}
    rounds = parallel_speculative_shapes(4, widths, 32)
    assert (1, 16, 16) in rounds and (2, 16, 32) in rounds and (4, 16, 32) in rounds
    assert all(width == 1 or batch < packed <= min(32, batch * width)
               for batch, width, packed in rounds)
    with pytest.raises(ValueError, match="Int32"):
        speculative_shapes(4, (*widths, 1 << 31), 32)
    with pytest.raises(ValueError, match="proposal depth"):
        proposal_shapes(4, (1 << 31) - 1)


@pytest.mark.parametrize("recipe", [Qwen, Glm, DeepSeek4, DeepSeek41, Mimo, Step5])
def test_recipe_completion_capacity_tracks_enabled_native_catalog(recipe):
    packed = recipe in (DeepSeek4, DeepSeek41, Mimo)
    prefix = "gpu" if recipe is DeepSeek41 else "cpu"
    active = _with_spec(recipe, drafts=3, packed=packed, prefix=prefix)
    assert active.completion_token_capacity == 4
    ordinary = _with_spec(recipe, enabled=False)
    assert ordinary.completion_token_capacity == 1


@pytest.mark.parametrize("width", [1, 8, 13, 16])
def test_qwen_dense_native_capacity_remains_independent_of_recurrent_state_columns(width):
    from pathlib import Path
    from hyperflare_recipes.recipes.qwen38_dense.bootstrap import PreparationConfig

    base = PreparationConfig(draft_snapshot=Path("/tmp/dflash"))
    drafts = width - 1
    spec = SpeculationParameters(
        draft_tokens=drafts, max_draft_tokens=15, token_capacity=128,
        packed_verification=drafts > 0, graph_budget=64 if drafts > 0 else None,
    )
    settings = replace(base, parameters=replace(base.parameters, speculation=spec))
    assert settings.completion_token_capacity == width
    with pytest.raises(ValueError, match="fifteen"):
        replace(
            base,
            parameters=replace(
                base.parameters,
                speculation=SpeculationParameters(draft_tokens=31, max_draft_tokens=31, token_capacity=128,
                                                  packed_verification=True, graph_budget=64),
            ),
        )
