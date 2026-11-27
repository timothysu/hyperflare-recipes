# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kimi preparation backing follows logical page geometry, independently of serving context."""

from dataclasses import replace

import pytest

from hyperflare_recipes.parameters import ContextParallelParameters
from hyperflare_recipes.recipes.kimi3.bootstrap import PreparationConfig


@pytest.mark.parametrize("context_parallel,page_tokens,pages", [(True, 2048, 10), (False, 128, 40)])
def test_preparation_pages_cover_prefill_and_request_boundaries(context_parallel, page_tokens, pages):
    base = PreparationConfig().parameters
    params = replace(base, context_parallel=ContextParallelParameters(enabled=context_parallel))
    short = PreparationConfig(context=4096, parameters=params)
    long = PreparationConfig(context=1048576, parameters=params)
    assert short.page_tokens == long.page_tokens == page_tokens
    assert short.preparation_pages == long.preparation_pages == pages
    assert (pages - 8) * page_tokens >= short.prefill_chunk_size
    assert (pages - 9) * page_tokens < short.prefill_chunk_size


@pytest.mark.parametrize("chunk,pages", [(128, 9), (2048, 9), (2176, 10), (4096, 10)])
def test_prefill_backing_rounds_up_to_context_parallel_pages(chunk, pages):
    base = PreparationConfig().parameters
    assert PreparationConfig(parameters=replace(base, prefill_chunk=chunk)).preparation_pages == pages
