# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3.1 Pro MXFP4 preparation parameters and serving defaults."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from ignition.declaration import Checkpoint, RankCosts, Recipe, SamplingDefaults, SpeculativeMethod
from ignition.errors import UsageError
from ignition.parameters import (
    ParameterContext,
    PromptLogprobsParameters,
    SpeculationParameters,
    VisionParameters,
)

from hyperflare_recipes.parameters import (
    DraftQuantizationParameters,
    HybridContextParallelParameters,
    OnlineMXFP8Parameters,
)
from hyperflare_recipes.recipes.kimi3.recipe import (
    Kimi3Parameters,
    RECIPE as KIMI3,
    prepare as prepare_kimi,
    usage,
)
from .dspark_config import DSPARK_REPOSITORY, DSPARK_REVISION, WINDOW_REPOSITORY, WINDOW_REVISION


@dataclass(frozen=True)
class Kimi31Parameters(Kimi3Parameters):
    """Preparation parameters for Kimi K3.1 Pro."""

    context_parallel: HybridContextParallelParameters = HybridContextParallelParameters(
        supported_sizes=(4, 8, 16),
        hybrid_topologies=((16, 4), (16, 8)),
        # The DCP-only reduction shrinks from 14 MiB to 512 KiB at C4/W8. Two staging slots with
        # 30 shards release 849,346,560 bytes; let the planner consume at most 64 pages and retain the rest.
        graph_credits=(((16, 8), (849_346_560, 64)),),
    )
    online_mxfp8: OnlineMXFP8Parameters = OnlineMXFP8Parameters(
        default_scope="kda-input",
        supported_scopes=("all", "kda-input"),
    )
    draft_quantization: DraftQuantizationParameters = DraftQuantizationParameters(
        draft_head_choices=("mxfp8", "nvfp4"),
        draft_body_choices=("mxfp8",),
        markov_head_choices=("fp8",),
    )

    def validate(self, context: ParameterContext) -> None:
        super().validate(context)
        if self.speculation is not None:
            method = self.speculation.method
            dq = self.draft_quantization
            if method == "dflash2":
                if dq.markov_head_quantization is not None:
                    raise UsageError("DFlash2 has no Markov transition head")
            elif method != "dspark":
                if dq.markov_head_quantization is not None:
                    raise UsageError("Kimi Markov head conversion requires --speculative-method dspark")
                if dq.draft_head_quantization is not None or dq.draft_body_quantization is not None:
                    raise UsageError("Kimi draft conversion requires --speculative-method dspark or dflash2")


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None):
    params = plan.parameters
    if not params.context_parallel.enabled:
        raise ValueError("Kimi K3.1 preparation requires context parallelism")
    if params.context_parallel.effective_size(plan.world_size) not in (4, 8, 16):
        raise ValueError("Kimi K3.1 TP16 supports DCP4, DCP8 or DCP16")
    if plan.speculation is not None and plan.speculation.method == "dflash2":
        from .dflash_config import REPOSITORY, REVISION

        return prepare_kimi(
            snapshot, plan, compile_forwards=compile_forwards, kernel_cache=kernel_cache,
            progress=progress, retain_graphs=retain_graphs, core_storage=core_storage, ranks=ranks,
            recipe=RECIPE, draft_source=(REPOSITORY, REVISION), draft_directory="dflash2",
        )
    if plan.speculation is not None and plan.speculation.method == "dspark":
        return prepare_kimi(
            snapshot, plan, compile_forwards=compile_forwards, kernel_cache=kernel_cache,
            progress=progress, retain_graphs=retain_graphs, core_storage=core_storage, ranks=ranks,
            recipe=RECIPE, draft_source=(WINDOW_REPOSITORY, WINDOW_REVISION),
        )
    return prepare_kimi(
        snapshot, plan, compile_forwards=compile_forwards, kernel_cache=kernel_cache,
        progress=progress, retain_graphs=retain_graphs, core_storage=core_storage, ranks=ranks,
        recipe=RECIPE, draft_source=(DSPARK_REPOSITORY, DSPARK_REVISION),
    )


RECIPE = Recipe(
    name="kimi31",
    served_model_name="Kimi-K3.1-Pro",
    prepare=prepare,
    usage=usage,
    # https://huggingface.co/moonshotai/Kimi-K3.1-Pro#3-evaluation-results
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1),
    checkpoints=(
        Checkpoint(
            alias="kimi-k3.1-pro",
            repository="moonshotai/Kimi-K3.1-Pro",
            revision="d9e99b0792a5e93b78ea9bea62a68167e36ddd9e",
            architecture="KimiK31ForConditionalGeneration",
            stored_bytes=1_561_217_377_280,
            # K3's matching target layout plus 357,052,416 BF16 indexer bytes replicated on sixteen ranks.
            device_bytes=1_491_026_420_236 + 16 * 357_052_416 + 8_558_149_632,
            draft_bytes=8_558_149_632,
            scale_compressed_device_bytes={"csf": 1_491_026_420_236 + 16 * 357_052_416 + 8_558_149_632},
        ),
    ),
    context=1048576,
    page_tokens=128,
    parameters=Kimi31Parameters(
        max_concurrency=32,
        max_concurrency_limit=64,
        prefill_chunk=4096,
        speculation=SpeculationParameters(
            draft_tokens=7,
            max_draft_tokens=7,
            token_capacity=128,
            packed_verification=True,
            graph_budget=64,
            supported_methods=("dspark", "dflash2"),
        ),
        vision=VisionParameters(separable_encoder=True),
        prompt_logprobs=PromptLogprobsParameters(),
    ),
    verification_rows=0,
    max_images=4,
    image_bytes=32 << 20,
    speculative_methods=MappingProxyType({
        "dflash2": SpeculativeMethod(MappingProxyType({
            16: RankCosts(state_row_bytes=35_450_112, page_bytes=2_174_976, graph_bytes=9 << 29, page_tokens=128 * 16),
        }), draft_bytes=5_569_622_016),
        "dspark": SpeculativeMethod(MappingProxyType({
            16: RankCosts(state_row_bytes=35_122_432, page_bytes=2_174_976, graph_bytes=9 << 29, page_tokens=128 * 16),
        }), draft_bytes=7_625_920_512),
    }),
    benchmark_world_sizes=(16,),
    b12x_environment=MappingProxyType({
        **KIMI3.b12x_environment, "B12X_PCIE_DCP_THREADS": "0", "B12X_PCIE_DCP_BLOCK_LIMIT": "0",
    }),
    # Graph storage is a planning allowance; actual graph memory is reported after preparation.
    ranks=MappingProxyType({16: RankCosts(state_row_bytes=33_238_272, page_bytes=2_912_256,
                                          graph_bytes=9 << 29, page_tokens=128 * 16)}),
    plain_ranks=MappingProxyType({16: RankCosts(state_row_bytes=30_275_328, page_bytes=2_174_976,
                                                graph_bytes=9 << 29, page_tokens=128 * 16)}),
)
