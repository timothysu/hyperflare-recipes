# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qualified checkpoint, parameters, serving range, and memory costs of the GLM 5.3 recipe."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from ignition.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, SamplingRange
from ignition.parameters import (
    PromptLogprobsParameters,
    RecipeParameters,
    SpeculationParameters,
    VisionParameters,
)
from ignition.preparation.bootstrap import common_preparation_kwargs

from hyperflare_recipes.parameters import (
    ContextParallelParameters,
    NativeDenseActivationsParameters,
    StreamOverlapParameters,
)

# Each sparse-attention layer stores four latent blocks and one index block per
# 256-token page on every rank. The target has eleven such layers; learned drafts
# add the draft layer's.
_PAGE_LAYER_BYTES = 4 * 64 * 528 + 64 * 132
_TARGET_PAGE_LAYERS = 11
# Rough estimate of what a rank holds beyond stored weights, state rows, and
# pages: operator scratch, graphs, collectives, and the CUDA context. It serves
# every rank count and speculation setting.
_GRAPH_BYTES = 6656 << 20


def _costs(state_row_bytes, graph_bytes, *, ranks, speculating):
    layers = _TARGET_PAGE_LAYERS + int(speculating)
    # Decode context parallelism shards each page's latent records over the ranks.
    return RankCosts(state_row_bytes, layers * _PAGE_LAYER_BYTES, graph_bytes,
                     page_tokens=256 * ranks)


@dataclass(frozen=True)
class Glm53Parameters(RecipeParameters):
    """Preparation parameters for GLM-5.3 Flash."""

    native_dense_activations: NativeDenseActivationsParameters = NativeDenseActivationsParameters()
    stream_overlap: StreamOverlapParameters = StreamOverlapParameters()
    context_parallel: ContextParallelParameters = ContextParallelParameters()


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None):
    """Prepare the plan's catalog and speculation settings on process-local GPUs 0 through N-1.

    With `ranks`, rank zero's multi-host rendezvous, each rank uses its own host's GPU 0.
    KV pages fill the free memory measured after graph capture, not the plan's estimate.
    """
    from ignition.preparation.rank_group import prepare_group
    from .bootstrap import PreparationConfig, prepare_rank
    from .frontend import chat_template_asset

    config = PreparationConfig(
        **common_preparation_kwargs(
            plan, compile_forwards=compile_forwards, kernel_cache=kernel_cache, retain_graphs=retain_graphs,
        ),
        devices=tuple(range(plan.world_size)) if ranks is None else (0,) * plan.world_size,
        hosts=None if ranks is None else ranks.hosts,
        minimum_pages=plan.minimum_pages,
        # SM103 runs routed experts through trtllm-gen with NVFP4 activations at every size.
        expert_activation="a4" if any(gpu.capability == (10, 3) for gpu in plan.gpus) else "a16",
        # Expanded scales take 2 x 288 x 4096 x 2048 / 16 bytes per routed layer over the ranks: about
        # 4.9 GB per rank at TP4, which GB10 holds beside its weights, but 9.7 GB at TP2, which it does not.
        expand_expert_scales=plan.world_size >= 4,
    )
    # A refusal or a failed template download is reported here, before any rank starts, so that its guidance
    # reaches the command line at every rank count; a rank that fails during preparation prints a traceback.
    if plan.speculation is not None and any(gpu.capability == (10, 3) for gpu in plan.gpus):
        from ignition.preparation.quantization import quantized_layers
        from .weights import require_draft_experts_supported

        require_draft_experts_supported(
            quantized_layers(json.loads((Path(snapshot) / "config.json").read_text())), sm103=True,
        )
    config = replace(config, chat_template=chat_template_asset(Path(snapshot).resolve(), cache_dir=plan.hub_cache))
    if plan.world_size == 1:
        return prepare_rank(snapshot, config, rank=0, group=None, progress=progress, core_storage=core_storage)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, core_storage=core_storage,
                         ranks=ranks, checkpoint=plan.checkpoint)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="glm53",
    served_model_name="GLM-5.3-Flash",
    prepare=prepare,
    usage=usage,
    # Bounds span HLE, BabyVision, and coding presets: https://huggingface.co/zai-org/GLM-5.3-Flash#footnotes.
    sampling_defaults=SamplingDefaults(
        temperature=1., top_p=.95, top_k=-1,
        ranges={"temperature": SamplingRange(.95, 1.), "top_p": SamplingRange(.95, 1.)},
    ),
    checkpoints=(
        Checkpoint(
            alias="glm-5.3-flash", repository="nvidia/GLM-5.3-Flash-FP4-Local",
            revision="76348ddb2be2e35e39ecfc00e75dc07703f4f3d5", architecture="Glm5NextForConditionalGeneration",
            stored_bytes=177_421_592_008,
            # Language model, draft layer, output head, and replicated vision tower.
            device_bytes=177_421_592_008,
            # Decoder layer 45, the MTP module.
            draft_bytes=4_301_658_112,
        ),
        Checkpoint(
            alias="glm-5.3-flash-nvfp4", repository="local-inference-lab/GLM-5.3-Flash-NVFP4",
            revision="378ca54585c46542bad1f3cb3ed0d73ae51cdb62", architecture="Glm5NextForConditionalGeneration",
            stored_bytes=198_042_331_512, device_bytes=198_042_331_512,
            draft_bytes=7_843_920_512,
            # The draft layer proposes with an NVFP4 copy of the BF16 vocabulary head, made at load: for the
            # 154,880 x 4096 head, 4-bit values, one E4M3 scale per 16 values with each TP4 shard's rows padded
            # to 128 for the swizzle, and an FP32 gain per shard, summed over four shards. Weights estimates add
            # it without speculation too, which overstates them by 89 MB per rank at TP4 with --no-mtp.
            expansion_bytes=356_909_072,
        ),
    ),
    context=1048576,
    page_tokens=256,
    parameters=Glm53Parameters(
        max_concurrency=32,
        max_concurrency_limit=64,
        prefill_chunk=4096,
        speculation=SpeculationParameters(draft_tokens=5, max_draft_tokens=7, token_capacity=128),
        vision=VisionParameters(separable_encoder=True),
        prompt_logprobs=PromptLogprobsParameters(),
    ),
    verification_rows=0,
    max_images=4,
    image_bytes=32 << 20,
    # b12x retains activation scales for A4 prefill launches in W4A16 expert paths when set.
    b12x_environment=MappingProxyType({"B12X_W4A16_A4_PREFILL": "1"}),
    ranks=MappingProxyType({
        # DGX Station can hold GLM on one GPU; TP1 must remain a supported rank count.
        # One rank holds every head: twice TP2's recurrent rows less the replicated part.
        1: _costs(161_267_200, _GRAPH_BYTES, ranks=1, speculating=True),
        2: _costs(81_608_192, _GRAPH_BYTES, ranks=2, speculating=True),
        4: _costs(41_778_688, _GRAPH_BYTES, ranks=4, speculating=True),
    }),
    plain_ranks=MappingProxyType({
        1: _costs(149_230_592, _GRAPH_BYTES, ranks=1, speculating=False),
        2: _costs(75_420_672, _GRAPH_BYTES, ranks=2, speculating=False),
        4: _costs(38_515_712, _GRAPH_BYTES, ranks=4, speculating=False),
    }),
)
