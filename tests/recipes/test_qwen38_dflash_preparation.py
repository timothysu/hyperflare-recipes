# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flash Next proposal-state geometry and preparation option boundaries."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hyperflare_recipes.components.dflash2_geometry import DraftAttentionGeometry
from hyperflare_recipes.recipes.qwen38.bootstrap import PreparationConfig
from hyperflare_recipes.recipes.qwen38.recipe import RECIPE
from hyperflare_recipes.recipes.qwen38.state import HybridStateLayout, HybridStatePool, checkpoint_copy_layout


@pytest.mark.parametrize("world", [1, 2, 4])
def test_dflash_rings_belong_to_complete_continuation_rows(world):
    config = SimpleNamespace(layers=5, attention=DraftAttentionGeometry(32, 8, 128, 128, 128, 10000, 2048))
    layout = HybridStateLayout.build(speculative_tokens=7, world_size=world, draft_config=config)
    assert layout.row_bytes == RECIPE.speculative_methods["dflash2"].ranks[world].state_row_bytes
    for layer in range(5):
        for field in ("key", "value"):
            ring = layout.region(f"draft{layer}.{field}")
            assert ring.shape == (2048, 8 // world, 128)
            assert ring.dtype == torch.float8_e4m3fn
    assert not any(region.name.startswith(("draft_", "target_residual")) for region in layout.regions)
    transfers = checkpoint_copy_layout(layout, block=256)
    for region in layout.regions:
        if region.name.startswith("draft"):
            first, last = region.offset // 256, (region.offset + region.nbytes + 255) // 256
            assert not transfers[first:last].any()
    assert layout.region("gdn.0.convolution").shape[-1] == 10
    assert layout.region("ple_convolution").shape[-1] == 16
    pool = HybridStatePool.allocate(3, "meta", speculative_tokens=7, world_size=world, draft_config=config)
    assert pool.reserved_region("draft0.key").shape == (3, 2048, 8 // world, 128)
    mtp = HybridStateLayout.build(speculative_tokens=7, world_size=world)
    assert mtp.region("target_residual").shape == (10240,)
    assert mtp.region("draft_raw_anchor").shape == (1,)


def test_dflash_preparation_options_validate_before_weight_loading():
    from dataclasses import replace
    from ignition.parameters import MixedPrefillParameters

    base_params = PreparationConfig().parameters
    dflash_params = replace(
        base_params,
        speculation=replace(RECIPE.speculation, method="dflash2"),
    )
    config = PreparationConfig(
        parameters=replace(
            dflash_params,
            online_mxfp8=replace(dflash_params.online_mxfp8, enabled=True),
            draft_quantization=replace(dflash_params.draft_quantization, draft_head_quantization="nvfp4"),
            force_moe_decode_a16=True,
        ),
        draft_snapshot=Path("draft"),
    )
    assert config.speculative_method == "dflash2" and config.online_mxfp8
    with pytest.raises(ValueError, match="at least one state row"):
        PreparationConfig(parameters=dflash_params, draft_snapshot=Path("draft"), snapshot_budget_bytes=142_821_119)
    with pytest.raises(ValueError, match="separate prefill"):
        PreparationConfig(
            parameters=replace(dflash_params, mixed_prefill=MixedPrefillParameters(enabled=True)),
            draft_snapshot=Path("draft"),
        )
    mtp_params = replace(base_params, speculation=RECIPE.speculation)
    with pytest.raises(ValueError, match="require DFlash2"):
        PreparationConfig(parameters=replace(mtp_params, online_mxfp8=replace(mtp_params.online_mxfp8, enabled=True)))
    with pytest.raises(ValueError, match="require DFlash2"):
        PreparationConfig(
            parameters=replace(
                mtp_params,
                draft_quantization=replace(mtp_params.draft_quantization, draft_head_quantization="mxfp8"),
            )
        )


def test_prefix_identity_separates_draft_context_and_target_arithmetic():
    from dataclasses import replace
    from hyperflare_recipes.parameters import NativeDenseActivationsParameters
    from hyperflare_recipes.recipes.qwen38.bootstrap import _identity

    manifest = SimpleNamespace(headers_sha256={"model": "header"},
                               metadata_sha256={"config.json": "config"})
    draft = SimpleNamespace(text={"hidden_size": 2560, "sliding_window": 2048})
    ordinary = PreparationConfig()
    mtp = replace(ordinary, parameters=replace(ordinary.parameters, speculation=RECIPE.speculation))
    dflash = replace(
        mtp,
        parameters=replace(mtp.parameters, speculation=replace(RECIPE.speculation, method="dflash2")),
        draft_snapshot=Path("draft"),
    )
    def identity(config, draft_config=None):
        return _identity(manifest, config=config, draft_config=draft_config)
    baseline = identity(dflash, draft)
    variants = [
        identity(ordinary),
        identity(mtp),
        baseline,
        identity(replace(dflash, parameters=replace(dflash.parameters, online_mxfp8=replace(dflash.parameters.online_mxfp8, enabled=True))), draft),
        identity(replace(dflash, parameters=replace(dflash.parameters, force_moe_decode_a16=True)), draft),
        identity(replace(dflash, parameters=replace(dflash.parameters, native_dense_activations=NativeDenseActivationsParameters(enabled=True))), draft),
        identity(dflash, SimpleNamespace(text={"hidden_size": 2560, "sliding_window": 1024})),
    ]
    assert len(set(variants)) == len(variants)
    assert identity(
        replace(dflash, parameters=replace(dflash.parameters, speculation=replace(dflash.parameters.speculation, draft_tokens=0))),
        draft,
    ) == baseline
