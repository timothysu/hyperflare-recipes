# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prefill capacity and speculative catalog configuration."""

from dataclasses import replace

import pytest

from ignition.parameters import SpeculationParameters
from hyperflare_recipes.recipes.qwen38.bootstrap import PreparationConfig
from hyperflare_recipes.recipes.qwen38.recipe import RECIPE


def _cfg(**kwargs):
    params = PreparationConfig().parameters
    param_updates = {}
    if "shard_hyperconnections" in kwargs:
        param_updates["shard_hyperconnections"] = kwargs.pop("shard_hyperconnections")
    if "prefill_chunk_size" in kwargs:
        param_updates["prefill_chunk"] = kwargs.pop("prefill_chunk_size")
    if "max_active" in kwargs:
        param_updates["max_concurrency"] = kwargs.pop("max_active")
    mtp = kwargs.pop("mtp", None)
    widths = kwargs.pop("speculative_widths", None)
    fixed = kwargs.pop("speculative_fixed_width", None)
    if mtp is not None or widths is not None or fixed is not None:
        if mtp is False:
            param_updates["speculation"] = None
        else:
            draft_tokens = RECIPE.speculation.draft_tokens if widths is None else max(widths) - 1
            adaptive = fixed is None
            if fixed is not None:
                draft_tokens = fixed - 1
            param_updates["speculation"] = SpeculationParameters(
                draft_tokens=draft_tokens,
                max_draft_tokens=max(7, draft_tokens if isinstance(draft_tokens, int) else 7),
                adaptive=adaptive,
                token_capacity=None,
                method="mtp",
                supported_methods=("mtp", "dflash2"),
            )
    if param_updates:
        params = replace(params, **param_updates)
    return PreparationConfig(parameters=params, **kwargs)


def test_hyperconnection_sharding_requires_multiple_ranks():
    assert not PreparationConfig().shard_hyperconnections
    with pytest.raises(ValueError, match="tensor parallelism"):
        _cfg(shard_hyperconnections=True)
    for devices, hosts in (((0, 1), None), ((0, 1, 2, 3), None), ((0, 0), ("a", "b"))):
        config = _cfg(devices=devices, hosts=hosts, shard_hyperconnections=True)
        assert config.shard_hyperconnections


@pytest.mark.parametrize("size, expected", [
    (64, (64,)),
    (512, (128, 256, 512)),
    (3072, (128, 256, 512, 1024, 1536, 2048, 2560, 3072)),
    (4096, (128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096)),
    (8192, (128, 256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192)),
])
def test_prefill_catalog_covers_aligned_limit_and_partial_tails(size, expected):
    config = _cfg(prefill_chunk_size=size)
    assert config.prefill_capacities == expected
    for tokens in range(1, size + 1):
        assert next(capacity for capacity in config.prefill_capacities if capacity >= tokens) <= size


@pytest.mark.parametrize("size", [True, 4096.0, "4096"])
def test_prefill_chunk_requires_integer(size):
    with pytest.raises(TypeError, match="prefill_chunk_size"):
        _cfg(prefill_chunk_size=size)


@pytest.mark.parametrize("size", [0, 16, 65, 4097, 262208])
def test_prefill_chunk_requires_page_alignment_and_context_bound(size):
    with pytest.raises(ValueError, match="prefill chunk size"):
        _cfg(prefill_chunk_size=size)


def test_prefill_chunk_must_fit_explicit_context():
    with pytest.raises(ValueError, match="prefill chunk size"):
        _cfg(context=2048, prefill_chunk_size=4096)
    assert _cfg(context=64, prefill_chunk_size=64).prefill_capacities == (64,)


@pytest.mark.parametrize("draft_tokens", [-1, 8])
def test_speculative_policy_requires_a_prepared_width_catalog(draft_tokens):
    with pytest.raises(ValueError):
        PreparationConfig(
            parameters=replace(
                PreparationConfig().parameters,
                speculation=SpeculationParameters(
                    draft_tokens=draft_tokens,
                    max_draft_tokens=max(7, draft_tokens),
                    token_capacity=128,
                    method="mtp",
                    supported_methods=("mtp", "dflash2"),
                ),
            )
        )


@pytest.mark.parametrize("concurrency", [1, 2, 3, 4, 64])
@pytest.mark.parametrize("maximum", range(1, 9))
def test_capture_covers_every_runtime_reduction(concurrency, maximum):
    from ignition.components.capacity import batch_capacities

    widths = tuple(range(1, maximum + 1))
    config = _cfg(max_active=concurrency, mtp=True, speculative_widths=widths)
    assert set(config.speculative_shapes) == {
        (batch, width) for batch in batch_capacities(concurrency) for width in widths
        if width == 1 or batch * width <= max(batch_capacities(concurrency)[-1], maximum)}
    for active in range(1, concurrency + 1):
        for width in widths:
            assert all((batch, width) in config.speculative_shapes for batch in batch_capacities(active)
                       if width == 1 or batch * width <= max(batch_capacities(concurrency)[-1], maximum))


@pytest.mark.parametrize("drafts", range(8))
def test_fixed_serving_default_does_not_remove_capture_variants(drafts):
    from ignition.preparation.execution import GraphSignature

    config = _cfg(mtp=True, speculative_fixed_width=drafts + 1)
    assert config.speculative_shapes == tuple(
        (b, w) for b in (1, 2, 4) for w in range(drafts + 1, 0, -1)
        if w == 1 or b * w <= max(4, drafts + 1)
    )
    for batch, width in config.speculative_shapes:
        GraphSignature("speculative", batch, width)


def test_ranks_on_separate_hosts_may_share_a_local_device_ordinal():
    config = PreparationConfig(devices=(0, 0), hosts=("boot-a", "boot-b"))
    assert config.devices == (0, 0) and config.hosts == ("boot-a", "boot-b")
    for devices, hosts in (((0, 0), None), ((0, 0), ("boot-a", "boot-a")), ((0, 0), ("boot-a",)),
                           ((0, 0), ["boot-a", "boot-b"]), (None, ("boot-a",))):
        with pytest.raises(ValueError):
            PreparationConfig(devices=devices, hosts=hosts)


@pytest.mark.parametrize("chunk, expected_terminal", [
    (64, ()),
    (256, ()),
    (512, (512,)),
    (2048, (2048,)),
    (3072, (3072,)),
    (4096, (4096,)),
    (8192, (8192,)),
])
def test_terminal_prefill_catalog_split(chunk, expected_terminal):
    from hyperflare_recipes.recipes.qwen38.bootstrap import _terminal_prefill_capacities

    caps = _cfg(prefill_chunk_size=chunk).prefill_capacities
    assert _terminal_prefill_capacities(caps, has_terminal_op=False) == ()
    assert _terminal_prefill_capacities(caps, has_terminal_op=True) == expected_terminal
