# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qualified checkpoint, parameters, serving range, and memory costs of the Qwen 3.8 recipe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from ignition.declaration import (
    Checkpoint,
    RankCosts,
    RankUsage,
    Recipe,
    SamplingDefaults,
    SamplingRange,
    SpeculativeMethod,
)
from ignition.errors import UsageError
from ignition.parameters import (
    MixedPrefillParameters,
    ParameterContext,
    ParameterScope,
    PromptLogprobsParameters,
    RecipeParameters,
    SpeculationParameters,
    VisionParameters,
    param,
)
from ignition.preparation.bootstrap import common_preparation_kwargs

from hyperflare_recipes.parameters import (
    DraftQuantizationParameters,
    NativeDenseActivationsParameters,
    OnlineMXFP8Parameters,
    StreamOverlapParameters,
)
from .checkpoint_schema import ARCHITECTURE, MODEL_ID, REVISION
from .draft_config import WEIGHT_BYTES

# Twelve target sparse-attention layers each store one page; learned drafts add the draft layer's.
_TARGET_PAGE_LAYERS = 12
# Rough estimates of what a rank holds beyond stored weights, state rows, and
# pages: prepared weight layouts, replicated tensors, operator scratch, graphs, and
# the CUDA context. They serve every speculation setting.
_GRAPH_BYTES = {1: 7168 << 20, 2: 8000 << 20, 4: 8576 << 20}
_MOE_DECODE_A16_INTERMEDIATE_WIDTH = 640


def _costs(state_row_bytes, page_layer_bytes, graph_bytes, *, speculating):
    layers = _TARGET_PAGE_LAYERS + int(speculating)
    return RankCosts(state_row_bytes, layers * page_layer_bytes, graph_bytes)


@dataclass(frozen=True)
class Qwen38Parameters(RecipeParameters):
    """Preparation parameters for Qwen3.8 Flash Next."""

    native_dense_activations: NativeDenseActivationsParameters = NativeDenseActivationsParameters()
    stream_overlap: StreamOverlapParameters = StreamOverlapParameters()
    online_mxfp8: OnlineMXFP8Parameters = OnlineMXFP8Parameters()
    draft_quantization: DraftQuantizationParameters = DraftQuantizationParameters(
        draft_head_choices=("mxfp8", "nvfp4"),
    )
    force_moe_decode_a16: bool = param(
        False,
        flag="--force-moe-decode-a16",
        group="advanced options",
        boolean_optional=True,
        scope=ParameterScope.DEPLOYMENT,
        help="use W4A16 target routed experts for decode and verification, with calibrated A4 prefill;"
             " requires SM120/121 and separate prefill graphs",
        status_label=lambda v: "MoE decode A16 / prefill A4" if v else None,
    )
    shard_hyperconnections: bool = param(
        False,
        flag="--shard-hyperconnections",
        group="advanced options",
        scope=ParameterScope.DEPLOYMENT,
        default_label="replicated",
        help="shard hyperconnection bottlenecks across TP ranks with BF16 gate-logit reduction",
        status_label=lambda v: "BF16 sharded hyperconnections" if v else None,
    )

    def validate(self, context: ParameterContext) -> None:
        super().validate(context)
        if self.force_moe_decode_a16:
            if any(gpu.capability not in ((12, 0), (12, 1)) for gpu in context.gpus):
                raise UsageError("--force-moe-decode-a16 requires SM120/121 packed NVFP4 experts")
            if context.world_size is not None and (_MOE_DECODE_A16_INTERMEDIATE_WIDTH // context.world_size) % 64:
                raise UsageError(
                    "--force-moe-decode-a16 requires rank-local expert intermediate width divisible by 64"
                )
            if self.mixed_prefill is not None and self.mixed_prefill.enabled:
                raise UsageError(
                    "--force-moe-decode-a16 cannot accompany --mixed-prefill: combined rows share one expert precision"
                )
        if self.shard_hyperconnections and context.world_size is not None and context.world_size < 2:
            raise UsageError("--shard-hyperconnections requires tensor parallelism")
        if (
            self.online_mxfp8.enabled
            or self.draft_quantization.draft_head_quantization is not None
        ) and (self.speculation is None or self.speculation.method != "dflash2"):
            raise UsageError("Flash Next online MXFP8 and draft head conversion require --speculative-method dflash2")


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False, core_storage=None,
            ranks=None):
    """Prepare the plan's catalog and speculation settings on process-local GPUs 0 through N-1.

    With `ranks`, rank zero's multi-host rendezvous, each rank uses its own host's GPU 0.
    Preparation backs a bounded KV working set; Helios sizes the serving cache.
    """
    from .bootstrap import PreparationConfig, prepare_model, prepare_rank

    speculation = plan.speculation
    draft_snapshot = None
    if speculation is not None and speculation.method == "dflash2":
        from huggingface_hub import snapshot_download
        from .draft_config import REPOSITORY, REVISION

        draft_snapshot = Path(snapshot_download(
            REPOSITORY, revision=REVISION, cache_dir=plan.hub_cache,
            allow_patterns=("config.json", "model.safetensors"),
        ))
    config = PreparationConfig(
        **common_preparation_kwargs(
            plan, compile_forwards=compile_forwards, kernel_cache=kernel_cache, retain_graphs=retain_graphs,
        ),
        devices=tuple(range(plan.world_size)) if ranks is None else (0,) * plan.world_size,
        hosts=None if ranks is None else ranks.hosts,
        draft_snapshot=draft_snapshot,
    )
    if plan.world_size == 1:
        storage = {} if core_storage is None else {"core_storage": core_storage}
        return prepare_model(snapshot, config, progress=progress, compile_decoder=compile_forwards,
                             **storage)
    from ignition.preparation.rank_group import prepare_group

    return prepare_group(snapshot, config, prepare_rank, progress=progress, core_storage=core_storage,
                         ranks=ranks, checkpoint=plan.checkpoint)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="qwen38",
    served_model_name="Qwen3.8-Flash-Next",
    prepare=prepare,
    usage=usage,
    # Thinking defaults; bounds span thinking/instruct presets: https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices.
    sampling_defaults=SamplingDefaults(
        temperature=1., top_p=.95, top_k=20,
        ranges={"temperature": SamplingRange(.7, 1.), "top_p": SamplingRange(.8, .95),
                "presence_penalty": SamplingRange(0., 2.)},
    ),
    checkpoints=(
        Checkpoint(
            alias="qwen3.8-flash", repository=MODEL_ID, revision=REVISION, architecture=ARCHITECTURE,
            stored_bytes=102_329_644_144,
            # Decoder, vision, and draft tensors; PLE rows are delivered through compact inputs.
            device_bytes=73_529_505_900,
            # The MTP module's tensors.
            draft_bytes=1_596_720_640,
            # Routed-expert block scales expand at load from 4,156,604,564 CSF bytes to
            # 48 layers x 512 experts x 307,200 bytes.
            expansion_bytes=3_393_142_636,
        ),
    ),
    context=262144,
    page_tokens=64,
    parameters=Qwen38Parameters(
        max_concurrency=32,
        max_concurrency_limit=64,
        prefill_chunk=4096,
        speculation=SpeculationParameters(
            draft_tokens=5,
            max_draft_tokens=7,
            token_capacity=128,
            method="mtp",
            supported_methods=("mtp", "dflash2"),
        ),
        vision=VisionParameters(),
        prompt_logprobs=PromptLogprobsParameters(),
        mixed_prefill=MixedPrefillParameters(),
    ),
    verification_rows=0,
    ranks=MappingProxyType({
        1: _costs(122_014_720, 69_632, _GRAPH_BYTES[1], speculating=True),
        2: _costs(61_246_464, 36_864, _GRAPH_BYTES[2], speculating=True),
        4: _costs(31_091_712, 36_864, _GRAPH_BYTES[4], speculating=True),
    }),
    plain_ranks=MappingProxyType({
        1: _costs(116_505_344, 69_632, _GRAPH_BYTES[1], speculating=False),
        2: _costs(58_383_104, 36_864, _GRAPH_BYTES[2], speculating=False),
        4: _costs(29_518_592, 36_864, _GRAPH_BYTES[4], speculating=False),
    }),
    speculative_methods=MappingProxyType({
        "dflash2": SpeculativeMethod(MappingProxyType({
            1: _costs(142_821_120, 69_632, _GRAPH_BYTES[1] + (1 << 30), speculating=False),
            2: _costs(71_632_640, 36_864, _GRAPH_BYTES[2] + (1 << 30), speculating=False),
            4: _costs(36_235_008, 36_864, _GRAPH_BYTES[4] + (1 << 30), speculating=False),
        }), draft_bytes=WEIGHT_BYTES),
    }),
    max_images=4,
    image_bytes=32 << 20,
)
