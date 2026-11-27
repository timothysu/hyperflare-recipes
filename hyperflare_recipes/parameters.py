# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Reusable compositional `ParameterGroup` classes shared across official model recipes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, ClassVar

from ignition.errors import UsageError
from ignition.parameters import (
    ParameterContext,
    ParameterGroup,
    ParameterScope,
    param,
)


@dataclass(frozen=True)
class NativeDenseActivationsParameters(ParameterGroup):
    """Quantized activation override for dense projections."""

    default_group: ClassVar[str | None] = "advanced options"

    enabled: bool = param(
        False,
        flag="--native-dense-activations",
        manifest_key="native_dense_activations",
        dict_key="native_dense_activations",
        scope=ParameterScope.DEPLOYMENT,
        help="use quantized activations instead of auto precision for dense projections:"
             " A8 for FP8 weights and A4 for FP4 weights; explicit recipe precision is retained",
        status_label=lambda v: "native dense activations" if v else None,
    )


@dataclass(frozen=True)
class StreamOverlapParameters(ParameterGroup):
    """Auxiliary-stream concurrency for recipes that overlap independent branches."""

    default_group: ClassVar[str | None] = "advanced options"

    enabled: bool = param(
        True,
        negated_flag="--no-stream-overlap",
        manifest_key="stream_overlap",
        dict_key="stream_overlap",
        scope=ParameterScope.DEPLOYMENT,
        default_label="overlap",
        help="run every concurrent branch serially on the main stream, to diagnose scheduling-dependent faults; slower",
        status_label=lambda v: None if v else "branches serialized",
    )


@dataclass(frozen=True)
class ContextParallelParameters(ParameterGroup):
    """Decode context parallelism sharding each KV page's positions across ranks."""

    default_group: ClassVar[str | None] = "advanced options"

    enabled: bool = param(
        True,
        negated_flag="--no-context-parallel",
        enabled_attr="allow_disable",
        manifest_key="context_parallel",
        dict_key="context_parallel",
        scope=ParameterScope.DEPLOYMENT,
        default_label="enabled",
        help="keep each rank's attention context whole instead of sharding it across tensor-parallel ranks;"
             " fewer cached tokens",
    )
    size: int | None = param(
        None,
        cli=False,
        value_type=int,
        manifest_key="context_parallel_size",
        dict_key="context_parallel_size",
        scope=ParameterScope.DEPLOYMENT,
    )
    allow_disable: bool = True

    def _effective_enabled(self, context: ParameterContext | None) -> bool:
        if not self.enabled:
            return False
        if context is None or context.recipe is None or context.world_size is None:
            return self.enabled
        spec = None if context.root is None else context.root.speculation
        costs = context.recipe.costs(context.world_size, spec, context.checkpoint)
        return costs.page_tokens is not None

    def effective_size(self, world_size: int | None) -> int:
        if self.size is not None:
            return self.size
        return 1 if world_size is None else world_size

    def validate(self, context: ParameterContext) -> None:
        super().validate(context)
        if not self.enabled and not self.allow_disable:
            raise UsageError(f"--no-context-parallel: {context.alias} requires context parallelism")
        if self.size is not None and context.world_size is not None:
            if type(self.size) is not int or not 1 <= self.size <= context.world_size or context.world_size % self.size:
                raise UsageError("context parallel size must divide the selected tensor parallel size")

    def adjust_page_tokens(self, page_tokens: int, context: ParameterContext) -> int:
        if not self._effective_enabled(context) or context.world_size is None:
            return page_tokens
        return page_tokens * self.effective_size(context.world_size)

    def manifest_parameters(self, context: ParameterContext) -> list[tuple[str, ParameterScope, Any]]:
        effective = self._effective_enabled(context)
        eff_size = self.effective_size(context.world_size) if effective else 0
        return [
            ("context_parallel", ParameterScope.DEPLOYMENT, effective),
            ("context_parallel_size", ParameterScope.DEPLOYMENT, int(eff_size)),
        ]

    def reproduction_args(self, context: ParameterContext) -> list[str]:
        if not self.enabled and self.allow_disable:
            return ["--no-context-parallel"]
        return []

    def to_dict(self, context: ParameterContext | None = None) -> dict[str, Any]:
        effective = self._effective_enabled(context)
        world = None if context is None else context.world_size
        return {
            "context_parallel": effective,
            "context_parallel_size": self.effective_size(world),
        }


@dataclass(frozen=True)
class HybridContextParallelParameters(ContextParallelParameters):
    """Hybrid decode context parallelism supporting sub-world-size DCP groups."""

    allow_disable: bool = False
    size: int | None = param(
        None,
        flag="--dcp-size",
        group="advanced options",
        metavar="N",
        value_type=int,
        choices_attr="supported_sizes",
        manifest_key="context_parallel_size",
        dict_key="context_parallel_size",
        scope=ParameterScope.DEPLOYMENT,
        default_label="tensor-parallel size",
        help="decode context-parallel ranks per group",
    )
    supported_sizes: tuple[int, ...] = (4, 8, 16)
    hybrid_topologies: tuple[tuple[int, int], ...] = ((16, 4), (16, 8))
    # Picklable tuple of ((world_size, dcp_size), (max_bytes, max_pages)) graph-memory credits.
    graph_credits: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
        ((16, 8), (849_346_560, 64)),
    )

    def validate(self, context: ParameterContext) -> None:
        super().validate(context)
        if self.size is not None and context.world_size is not None:
            if self.size != context.world_size and (context.world_size, self.size) not in self.hybrid_topologies:
                raise UsageError(
                    f"hybrid context parallelism is not supported for {context.alias}"
                    f" at TP{context.world_size} DCP{self.size}"
                )

    def adjust_rank_costs(self, costs: Any, context: ParameterContext) -> Any:
        costs = super().adjust_rank_costs(costs, context)
        if context.world_size is None:
            return costs
        credit_spec = dict(self.graph_credits).get((context.world_size, self.size))
        if credit_spec is not None:
            max_bytes, max_pages = credit_spec
            credit = min(max_bytes, max_pages * costs.page_bytes)
            return replace(costs, graph_bytes=costs.graph_bytes - credit)
        return costs

    def reproduction_args(self, context: ParameterContext) -> list[str]:
        if self.size is not None and context.world_size is not None and self.size != context.world_size:
            return ["--dcp-size", str(self.size)]
        return []


def _compression_choices(group: ExpertScaleCompressionParameters, context: ParameterContext) -> tuple[str, ...]:
    checkpoint = context.checkpoint
    if checkpoint is not None and checkpoint.scale_compressed_device_bytes:
        return (*checkpoint.scale_compressed_device_bytes.keys(), "native")
    return ("csf", "native")


def _has_scale_compression(group: ExpertScaleCompressionParameters, context: ParameterContext) -> bool:
    if context.checkpoint is not None:
        return bool(context.checkpoint.scale_compressed_device_bytes)
    if context.recipe is not None:
        return any(c.scale_compressed_device_bytes for c in context.recipe.checkpoints)
    return True


def _compression_help_extra(group: ExpertScaleCompressionParameters, context: ParameterContext) -> str | None:
    checkpoint = context.checkpoint
    if checkpoint is None or not checkpoint.scale_compressed_device_bytes:
        return None
    notes = []
    for encoding, compressed in checkpoint.scale_compressed_device_bytes.items():
        if checkpoint.device_bytes > 0 and compressed < checkpoint.device_bytes:
            saving = 100 * (1 - compressed / checkpoint.device_bytes)
            notes.append(f"MODE {encoding}: estimated {saving:.0f}% less resident weight storage; actual savings vary")
        else:
            notes.append(f"MODE: {encoding} (with no MODE selects {encoding}; native disables)")
    return "; ".join(notes) if notes else None


@dataclass(frozen=True)
class ExpertScaleCompressionParameters(ParameterGroup):
    """Optional lossless compression of routed-expert block quantization scales."""

    default_group: ClassVar[str | None] = "capacity and compression"

    encoding: str | None = param(
        None,
        flag="--expert-scale-compression",
        metavar="MODE",
        choices_fn=_compression_choices,
        enabled_fn=_has_scale_compression,
        const="csf",
        native_value="native",
        manifest_key="expert_scale_compression",
        dict_key="expert_scale_compression",
        scope=ParameterScope.TARGET_WEIGHT,
        help="losslessly compress routed-expert quantization scales to reduce GPU weight memory",
        help_extra=_compression_help_extra,
        status_label=lambda v: f"expert scales {v or 'native'}",
    )

    def validate(self, context: ParameterContext) -> None:
        if self.encoding is not None:
            checkpoint = context.checkpoint
            if checkpoint is not None and self.encoding not in checkpoint.scale_compressed_device_bytes:
                raise UsageError(
                    f"--expert-scale-compression {self.encoding}: {context.alias} does not prepare this scale encoding"
                )

    def adjust_weights_bytes(self, weights_bytes: int, context: ParameterContext) -> int:
        if self.encoding is None or context.recipe is None or context.checkpoint is None or context.world_size is None:
            return weights_bytes
        compressed_device = context.checkpoint.scale_compressed_device_bytes[self.encoding]
        spec = None if context.root is None else context.root.speculation
        return context.recipe.base_weights_bytes(
            context.checkpoint, context.world_size, spec, device_bytes=compressed_device,
        )

    def reproduction_args(self, context: ParameterContext) -> list[str]:
        if not _has_scale_compression(self, context):
            return []
        defaults_group = self._matching_defaults(context)
        default_encoding = None if defaults_group is None else defaults_group.encoding
        if self.encoding is not None:
            return ["--expert-scale-compression", self.encoding]
        if default_encoding is not None:
            return ["--expert-scale-compression", "native"]
        return []


@dataclass(frozen=True)
class OnlineMXFP8Parameters(ParameterGroup):
    """On-load conversion of eligible BF16 projections to MXFP8."""

    default_group: ClassVar[str | None] = "advanced options"

    enabled: bool = param(
        False,
        flag="--online-mxfp8",
        manifest_key="online_mxfp8",
        dict_key="online_mxfp8",
        scope=ParameterScope.TARGET_WEIGHT,
        default_label="checkpoint precision",
        help="store the recipe's eligible BF16 projections as MXFP8 while loading",
    )
    scope: str | None = param(
        None,
        flag="--online-mxfp8-scope",
        metavar="SCOPE",
        choices_attr="supported_scopes",
        enabled_fn=lambda g, _: len(g.supported_scopes) > 1,
        manifest_key="online_mxfp8_scope",
        dict_key="online_mxfp8_scope",
        scope=ParameterScope.TARGET_WEIGHT,
        help="select the recipe's online MXFP8 conversion group",
    )
    default_scope: str = "all"
    supported_scopes: tuple[str, ...] = ("all",)
    inherit_dflash2_draft_body: bool = True

    def __post_init__(self):
        if self.default_scope not in self.supported_scopes:
            raise ValueError("online MXFP8 default_scope must belong to supported_scopes")

    @property
    def effective_scope(self) -> str:
        if self.scope is not None:
            return self.scope
        return self.default_scope if self.enabled else "all"

    def add_arguments(self, parser, context: ParameterContext, *, include_multi_host: bool = True) -> None:
        if len(self.supported_scopes) <= 1:
            super().add_arguments(parser, context, include_multi_host=include_multi_host)
            return
        # Show the recipe's effective default scope in --help.
        patched = replace(self, scope=self.default_scope)
        super(OnlineMXFP8Parameters, patched).add_arguments(parser, context, include_multi_host=include_multi_host)

    def validate(self, context: ParameterContext) -> None:
        if self.scope is not None:
            if self.scope not in self.supported_scopes:
                raise UsageError(
                    f"online MXFP8 scope: {context.alias} supports {', '.join(self.supported_scopes)}"
                )
            if self.scope != "all" and not self.enabled:
                raise UsageError("online MXFP8 scope requires --online-mxfp8")

    def manifest_parameters(self, context: ParameterContext) -> list[tuple[str, ParameterScope, Any]]:
        eff_scope = self.effective_scope
        entries: list[tuple[str, ParameterScope, Any]] = [
            ("online_mxfp8", ParameterScope.TARGET_WEIGHT, bool(self.enabled)),
            ("online_mxfp8_scope", ParameterScope.TARGET_WEIGHT, eff_scope),
        ]
        root = context.root
        if (
            self.inherit_dflash2_draft_body
            and root is not None
            and root.speculation is not None
            and root.speculation.method == "dflash2"
            and self.enabled
            and eff_scope == "all"
        ):
            draft_quant = getattr(root, "draft_quantization", None)
            explicit_body = None if draft_quant is None else draft_quant.draft_body_quantization
            if explicit_body is None:
                entries.append(("speculation.draft_body_quantization", ParameterScope.DRAFT_WEIGHT, "mxfp8"))
        return entries

    def reproduction_args(self, context: ParameterContext) -> list[str]:
        if not self.enabled:
            return []
        args = ["--online-mxfp8"]
        if len(self.supported_scopes) > 1:
            args.extend(["--online-mxfp8-scope", self.effective_scope])
        return args

    def to_dict(self, context: ParameterContext | None = None) -> dict[str, Any]:
        return {
            "online_mxfp8": self.enabled,
            "online_mxfp8_scope": self.effective_scope,
        }

    def status_parts(self, context: ParameterContext) -> list[str]:
        return [f"online MXFP8 {self.effective_scope}"] if self.enabled else []


@dataclass(frozen=True)
class DraftQuantizationParameters(ParameterGroup):
    """Independent quantization overrides for learned proposal heads and bodies."""

    manifest_prefix: ClassVar[str | None] = "speculation"
    dict_section: ClassVar[str | None] = "speculation"
    default_group: ClassVar[str | None] = "advanced options"

    draft_head_quantization: str | None = param(
        None,
        flag="--draft-head-quantization",
        choices_attr="draft_head_choices",
        enabled_attr="draft_head_choices",
        scope=ParameterScope.DRAFT_WEIGHT,
        requires_speculation=True,
        default_label="checkpoint precision",
        help="prepare a separate quantized vocabulary head for learned draft proposals;"
             " target verification retains its precision",
    )
    draft_body_quantization: str | None = param(
        None,
        flag="--draft-body-quantization",
        choices_attr="draft_body_choices",
        enabled_attr="draft_body_choices",
        scope=ParameterScope.DRAFT_WEIGHT,
        requires_speculation=True,
        default_label="checkpoint precision",
        help="prepare eligible DFlash2 draft-body projections as MXFP8 independently of"
             " target and proposal-head precision",
    )
    markov_head_quantization: str | None = param(
        None,
        flag="--markov-head-quantization",
        choices_attr="markov_head_choices",
        enabled_attr="markov_head_choices",
        scope=ParameterScope.DRAFT_WEIGHT,
        requires_speculation=True,
        default_label="checkpoint precision",
        help="prepare the draft Markov transition projection in a quantized format",
    )
    draft_head_choices: tuple[str, ...] = ()
    draft_body_choices: tuple[str, ...] = ()
    markov_head_choices: tuple[str, ...] = ()

    def validate(self, context: ParameterContext) -> None:
        alias = context.alias
        root = context.root
        spec = None if root is None else root.speculation
        for label, value, choices, kind in (
            ("--draft-head-quantization", self.draft_head_quantization, self.draft_head_choices, "proposal head format"),
            ("--markov-head-quantization", self.markov_head_quantization, self.markov_head_choices, "proposal head format"),
            ("--draft-body-quantization", self.draft_body_quantization, self.draft_body_choices, "draft body format"),
        ):
            if value is None:
                continue
            if value not in choices:
                raise UsageError(f"{label} {value}: {alias} does not prepare this {kind}")
            if spec is None or spec.draft_tokens < 1:
                if label == "--draft-body-quantization":
                    raise UsageError("draft body quantization requires positive learned draft depth")
                raise UsageError("draft head quantization requires positive learned draft depth")
        if self.draft_body_quantization is not None and spec is not None and spec.method != "dflash2":
            raise UsageError("--draft-body-quantization requires --speculative-method dflash2")

    def manifest_parameters(self, context: ParameterContext) -> list[tuple[str, ParameterScope, Any]]:
        if context.root is None or context.root.speculation is None:
            return []
        return super().manifest_parameters(context)

    def reproduction_args(self, context: ParameterContext) -> list[str]:
        if context.root is None or context.root.speculation is None or context.root.speculation.draft_tokens < 1:
            return []
        return super().reproduction_args(context)

    def to_dict(self, context: ParameterContext | None = None) -> dict[str, Any]:
        if context is not None and context.root is not None and context.root.speculation is None:
            return {}
        section: dict[str, Any] = {}
        if self.draft_head_choices:
            section["draft_head_quantization"] = self.draft_head_quantization
        if self.draft_body_choices and self.draft_body_quantization is not None:
            section["draft_body_quantization"] = self.draft_body_quantization
        if self.markov_head_choices:
            section["markov_head_quantization"] = self.markov_head_quantization
        return {"speculation": section} if section else {}

    def status_parts(self, context: ParameterContext) -> list[str]:
        if context.root is None or context.root.speculation is None:
            return []
        parts = []
        if self.draft_body_quantization is not None:
            parts.append(f"draft body {self.draft_body_quantization.upper()}")
        if self.draft_head_quantization is not None:
            parts.append(f"draft head {self.draft_head_quantization.upper()}")
        if self.markov_head_quantization is not None:
            parts.append(f"markov head {self.markov_head_quantization.upper()}")
        return [", ".join(parts)] if parts else []
