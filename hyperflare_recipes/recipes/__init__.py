# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Official model recipes and their checkpoint catalog."""

from pathlib import Path

from ignition.declaration import Checkpoint, Recipe
from ignition.registry import (  # noqa: F401
    ResolvedModel,
    _csf_container,
    _csf_stored_bytes,
    _hub_metadata,
    _replacement,
    _snapshot_cached,
    _stored_bytes,
    aliases as _aliases,
    resolve as _resolve,
    snapshot_directory,
)


def recipes() -> tuple[Recipe, ...]:
    from .deepseek4.recipe import RECIPE as deepseek4
    from .deepseek41.recipe import RECIPE as deepseek41
    from .gemma4.recipe import RECIPE as gemma4
    from .glm53.recipe import RECIPE as glm53
    from .qwen38.recipe import RECIPE as qwen38
    from .qwen38_dense.recipe import RECIPE as qwen38_dense
    from .mimo_v2.recipe import RECIPE as mimo_v2
    from .kimi3.recipe import RECIPE as kimi3
    from .kimi31.recipe import RECIPE as kimi31
    from .step5.recipe import RECIPE as step5
    from .nemotron_h.recipe import RECIPE as nemotron_h
    from .minimax_m3.recipe import RECIPE as minimax_m3
    from .muse_spark.recipe import RECIPE as muse_spark
    from .inkling2.recipe import RECIPE as inkling2
    from .inkling2.recipe import MINI_RECIPE as inkling2_mini
    from .ling32.recipe import RECIPE as ling32

    return qwen38, qwen38_dense, glm53, deepseek41, deepseek4, mimo_v2, kimi3, kimi31, step5, nemotron_h, minimax_m3, gemma4, muse_spark, inkling2_mini, inkling2, ling32


def aliases() -> tuple[str, ...]:
    return _aliases(recipes())


def resolve(model: str, hub_cache: Path | None = None) -> ResolvedModel:
    return _resolve(model, hub_cache, recipes=recipes())
