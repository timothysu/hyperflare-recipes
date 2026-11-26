# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Direct checkpoint weight loading for ModelOpt MIXED_PRECISION Hy4-flash."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch

from hyperflare_recipes.components.vocabulary import mxfp8_draft_head, nvfp4_draft_head
from hyperflare_recipes.components.weights import DirectWeightLoader, MLPWeights, quantize_bf16_linear_as_mxfp8
from ignition.preparation.persistent import allocated_tensor_bytes, persistent_cat, persistent_copy
from ignition.preparation.rebuild_manifest import weight_container

from .config import (
    ENGRAM_HEADS,
    HC_MULT,
    ModelConfig,
    VOCABULARY,
)


@weight_container(("router", "correction", "down_proj", "up_proj", "experts", "shared", "combine_norm"))
@dataclass(frozen=True)
class LatentMoEWeights:
    router: torch.Tensor
    correction: torch.Tensor
    down_proj: object
    up_proj: object
    experts: object
    shared: MLPWeights
    combine_norm: torch.Tensor
    experts_a4_prefill: bool = False


@weight_container(("mlp", "post_fc2_norm"))
@dataclass(frozen=True)
class DenseMLPWeights:
    mlp: MLPWeights
    post_fc2_norm: torch.Tensor


@weight_container(("attention", "indexer", "mlp", "residual", "engram"))
@dataclass(frozen=True)
class LayerWeights:
    attention: dict
    indexer: dict | None
    mlp: DenseMLPWeights | LatentMoEWeights
    residual: dict
    engram: dict | None


@weight_container(("attention", "mlp", "input_norm", "post_attn_norm"))
@dataclass(frozen=True)
class DFlyLayerWeights:
    attention: dict
    mlp: MLPWeights
    input_norm: torch.Tensor
    post_attn_norm: torch.Tensor


@weight_container((
    "layers", "final_norm", "mtp_enorm", "mtp_hnorm", "mtp_eh_proj", "mtp_gate",
    "hc_hidden_norm", "hc_embed_norm", "hc_gate_up", "hc_down",
    "confidence_weight", "confidence_bias", "draft_head",
))
@dataclass(frozen=True)
class DFlyWeights:
    layers: tuple[DFlyLayerWeights, ...]
    final_norm: torch.Tensor
    mtp_enorm: torch.Tensor
    mtp_hnorm: torch.Tensor
    mtp_eh_proj: object
    mtp_gate: torch.Tensor
    hc_hidden_norm: torch.Tensor
    hc_embed_norm: torch.Tensor
    hc_gate_up: object
    hc_down: object
    confidence_weight: torch.Tensor | None = None
    confidence_bias: torch.Tensor | None = None
    draft_head: object | None = None


@weight_container(("embedding", "norm", "hc_head", "head", "layers", "draft", "vision"))
@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    shard: object
    embedding: torch.Tensor
    norm: torch.Tensor
    hc_head: dict[str, torch.Tensor]
    head: object
    layers: tuple[LayerWeights, ...]
    draft: DFlyWeights | None
    vision: dict[str, torch.Tensor] | None
    allocated_bytes: int
    scale_scratch: tuple[torch.Tensor, ...] = ()

    @classmethod
    def load(
        cls,
        manifest,
        device: torch.device,
        *,
        shard,
        progress=None,
        dfly: bool = True,
        vision: bool = True,
        expert_scale_compression: str | None = "csf",
        draft_head_quantization: str | None = None,
        online_mxfp8: bool = False,
        online_mxfp8_scope: str = "all",
    ) -> ModelWeights:
        if draft_head_quantization not in (None, "nvfp4", "mxfp8"):
            raise ValueError("Hy4-flash draft head quantization must be None, 'nvfp4', or 'mxfp8'")
        if online_mxfp8_scope not in ("all", "draft"):
            raise ValueError("Hy4-flash online MXFP8 scope must be 'all' or 'draft'")
        config = ModelConfig.load(manifest.metadata_path, manifest=manifest)
        if dfly:
            if config.draft is None or not any(k.startswith("model.dfly_layers.") for k in manifest.tensors):
                raise ValueError(
                    "DFly2 speculation was requested, but the checkpoint lacks dfly_num_hidden_layers "
                    "configuration or model.dfly_layers.* tensors"
                )
        if vision:
            if config.vision is None or not any(k.startswith("vit.") for k in manifest.tensors):
                raise ValueError(
                    "Vision preparation was requested, but the checkpoint lacks vision_config or vit.* tensors"
                )
        load_dfly = dfly
        load_vision = vision
        before = allocated_tensor_bytes(device)
        with Hy4WeightLoader(
            manifest,
            device,
            config=config,
            shard=shard,
            progress=progress,
            dfly=load_dfly,
            vision=load_vision,
            expert_scale_compression=expert_scale_compression,
            online_mxfp8=online_mxfp8,
            online_mxfp8_scope=online_mxfp8_scope,
        ) as loader:
            quantize_head = online_mxfp8 and online_mxfp8_scope == "all"
            embedding, norm = loader.load_tensors(("model.embed_tokens.weight", "model.norm.weight"))
            head_bf16 = loader.load_padded_tensor(
                "lm_head.weight",
                rows=shard.padded_partition(VOCABULARY, alignment=64),
                temporary=quantize_head,
            )
            hc_head = {
                "fn": loader.load_tensor("model.hc_head.hc_head_fn", dtype=torch.float32),
                "base": loader.load_tensor("model.hc_head.hc_head_base", dtype=torch.float32),
                "scale": loader.load_tensor("model.hc_head.hc_head_scale", dtype=torch.float32),
            }
            layers = tuple(loader.load_layer(i) for i in range(config.layers))
            draft = loader.load_dfly() if load_dfly else None
            head: object = mxfp8_draft_head(head_bf16) if quantize_head else head_bf16
            if draft is not None and draft_head_quantization is not None:
                if quantize_head and draft_head_quantization == "mxfp8":
                    draft_head = head
                else:
                    converters = {"nvfp4": nvfp4_draft_head, "mxfp8": mxfp8_draft_head}
                    draft_head = converters[draft_head_quantization](head_bf16)
                draft = replace(draft, draft_head=draft_head)
            if quantize_head:
                del head_bf16
            vis = loader.load_vision() if load_vision else None
            scale_scratch = loader.scale_scratch
        return cls(
            config=config,
            shard=shard,
            embedding=embedding,
            norm=norm,
            hc_head=hc_head,
            head=head,
            layers=layers,
            draft=draft,
            vision=vis,
            allocated_bytes=allocated_tensor_bytes(device) - before,
            scale_scratch=scale_scratch,
        )


class Hy4WeightLoader(DirectWeightLoader):
    """Stream ModelOpt MIXED_PRECISION Hy4-flash weights directly into GPU memory."""

    def __init__(
        self,
        manifest,
        device: torch.device,
        *,
        config: ModelConfig,
        shard,
        progress=None,
        dfly: bool = True,
        vision: bool = True,
        expert_scale_compression: str | None = "csf",
        online_mxfp8: bool = False,
        online_mxfp8_scope: str = "all",
    ):
        if shard.world_size != 1:
            raise ValueError("Hy4-flash recipe requires TP=1")
        if expert_scale_compression not in (None, "csf"):
            raise ValueError("Hy4-flash expert scale compression must be None or 'csf'")
        if online_mxfp8_scope not in ("all", "draft"):
            raise ValueError("Hy4-flash online MXFP8 scope must be 'all' or 'draft'")
        self.config = config
        self.shard = shard
        self.dfly = dfly
        self.vision = vision
        self.expert_scale_compression = expert_scale_compression
        self.online_mxfp8 = bool(online_mxfp8)
        self.online_mxfp8_scope = online_mxfp8_scope
        self._expert_scale_scratch: tuple[torch.Tensor, torch.Tensor] | None = None
        super().__init__(manifest, device, progress=progress)

    @property
    def _quantize_target_dense(self) -> bool:
        return self.online_mxfp8 and self.online_mxfp8_scope == "all"

    @property
    def _quantize_draft_dense(self) -> bool:
        return self.online_mxfp8

    def _can_mxfp8_tensor(self, tensor: torch.Tensor) -> bool:
        return (
            isinstance(tensor, torch.Tensor)
            and tensor.dtype == torch.bfloat16
            and tensor.ndim == 2
            and tensor.shape[1] % 32 == 0
            and tensor.shape[0] % 8 == 0
        )

    def _load_opt_mxfp8(self, prefix: str, *, allow: bool):
        if allow and self.linear_format(prefix) == "bfloat16":
            src = self._sources[f"{prefix}.weight"]
            if src.ndim == 2 and src.shape[1] % 32 == 0 and src.shape[0] % 8 == 0:
                weight = self.load_linear(prefix, temporary=True)
                return quantize_bf16_linear_as_mxfp8(weight, retain_scale_rows=False)
        return self.load_linear(prefix)

    def _load_opt_mxfp8_group(self, prefixes: Sequence[str], *, allow: bool):
        if allow and self._same_linear_format(prefixes) and self.linear_format(prefixes[0]) == "bfloat16":
            rows = sum(self._sources[f"{p}.weight"].shape[0] for p in prefixes)
            cols = self._sources[f"{prefixes[0]}.weight"].shape[1]
            if cols % 32 == 0 and rows % 8 == 0:
                weight = self.load_linear_group(tuple(prefixes), temporary=True)
                return quantize_bf16_linear_as_mxfp8(weight, retain_scale_rows=False)
        return self.load_linear_group(tuple(prefixes))

    def _selected_names(self) -> set[str]:
        selected = set()
        for name in self.manifest.tensors:
            # Engram embedding tables remain host/NVMe-backed and never load onto the GPU.
            if ".engram_layer.multi_head_embedding." in name or name.endswith(".engram_layer.lookup_table"):
                continue
            if name.startswith("model.dfly_layers.") and not self.dfly:
                continue
            if name.startswith("vit.") and not self.vision:
                continue
            selected.add(name)
        return selected

    @property
    def scale_scratch(self) -> tuple[torch.Tensor, ...]:
        return () if self._expert_scale_scratch is None else self._expert_scale_scratch

    def _scale_scratch(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.expert_scale_compression is None:
            return None
        if self._expert_scale_scratch is None:
            # Shared expansion buffers for serialized 512 x 1536 x 1536 NVFP4 CSF layers.
            self._expert_scale_scratch = (
                torch.empty((512, 2 * 1536, 1536 // 16), dtype=torch.float8_e4m3fn, device=self.device),
                torch.empty((512, 1536, 1536 // 16), dtype=torch.float8_e4m3fn, device=self.device),
            )
        return self._expert_scale_scratch

    def _same_linear_format(self, prefixes: Sequence[str]) -> bool:
        return len({self.linear_format(p) for p in prefixes}) == 1

    def _load_attention(
        self,
        prefix: str,
        *,
        fuse_gate: bool = False,
        indexer_prefix: str | None = None,
        allow_mxfp8: bool = False,
    ) -> dict:
        group = [f"{prefix}.q_a_proj", f"{prefix}.kv_a_proj_with_mqa"]
        fused_gate = False
        if fuse_gate and self._same_linear_format((*group, f"{prefix}.linear_gate")):
            group.append(f"{prefix}.linear_gate")
            fused_gate = True
        fused_indexer = False
        if indexer_prefix is not None and not allow_mxfp8:
            idx_projs = (
                f"{indexer_prefix}.token_key_projection",
                f"{indexer_prefix}.head_weight_projection",
            )
            if self._same_linear_format((*group, *idx_projs)):
                group.extend(idx_projs)
                fused_indexer = True
        return {
            "qkv_a": self._load_opt_mxfp8_group(tuple(group), allow=allow_mxfp8),
            "q_b": self._load_opt_mxfp8(f"{prefix}.q_b_proj", allow=allow_mxfp8),
            "kv_b": self.load_linear(f"{prefix}.kv_b_proj"),
            "gate": None if fused_gate else self._load_opt_mxfp8(f"{prefix}.linear_gate", allow=allow_mxfp8),
            "fused_gate": fused_gate,
            "fused_indexer_projs": fused_indexer,
            "o_proj": self._load_opt_mxfp8(f"{prefix}.o_proj", allow=allow_mxfp8),
            "q_norm": self.load_tensor(f"{prefix}.q_a_layernorm.weight"),
            "kv_norm": self.load_tensor(f"{prefix}.kv_a_layernorm.weight"),
            "post_proj_norm": self.load_tensor(f"{prefix}.post_proj_norm.weight"),
            "sink": persistent_copy(self.load_tensor(f"{prefix}.learnable_sink_param", temporary=True).float()),
        }

    def _load_indexer(self, prefix: str, *, fused_inputs: bool = False) -> dict:
        kv_w = self.load_tensor(f"{prefix}.key_value_projection.weight", temporary=True).float()
        kg_w = self.load_tensor(f"{prefix}.key_gate_projection.weight", temporary=True).float()
        return {
            "query_projection": self.load_linear(f"{prefix}.query_projection"),
            "head_weight_projection": None if fused_inputs else self.load_linear(f"{prefix}.head_weight_projection"),
            "token_key_projection": None if fused_inputs else self.load_linear(f"{prefix}.token_key_projection"),
            "token_key_norm": self.load_tensor(f"{prefix}.token_key_norm.weight"),
            "key_vg_projection": persistent_cat((kv_w, kg_w), dim=0),
            "intra_group_ape": persistent_copy(
                self.load_tensor(f"{prefix}.intra_group_ape", temporary=True).float()
            ),
            "key_norm": self.load_tensor(f"{prefix}.key_norm.weight"),
        }

    def _load_stacked_nvfp4_experts(self, prefix: str):
        c = self.config.text
        count = int(c["n_routed_experts"])
        lat_dim = int(c["routed_expert_hidden_size"])
        inter_dim = int(c["moe_intermediate_size"])
        limit = float(c["swiglu_limit"])

        # Stacked 3D ModelOpt NVFP4 tensors written by scripts/quantize_hy4_flash_nvfp4.py.
        projections = ("up_proj", "gate_proj", "down_proj")
        if f"{prefix}.up_proj.weight" not in self._sources:
            raise ValueError(
                f"{prefix}: expected stacked NVFP4 checkpoint tensors ({prefix}.{{gate,up,down}}_proj.*); "
                "quantize the BF16 checkpoint with scripts/quantize_hy4_flash_nvfp4.py first"
            )
        vectors: dict[tuple[str, str], torch.Tensor] = {}
        for proj in projections:
            for suffix in ("input_scale", "weight_scale_2"):
                value = self._sources[f"{prefix}.{proj}.{suffix}"]
                if (
                    value.dtype != torch.float32
                    or value.shape != (count,)
                    or not bool((torch.isfinite(value) & (value > 0)).all())
                ):
                    raise ValueError(f"{prefix}.{proj}.{suffix}: expected positive finite FP32 [512] scales")
                vectors[proj, suffix] = value
        for suffix in ("input_scale", "weight_scale_2"):
            if not torch.equal(vectors["up_proj", suffix], vectors["gate_proj", suffix]):
                raise ValueError(f"{prefix}: gate_proj and up_proj {suffix} must match for fused NVFP4")

        aliases = []
        expert_prefixes = tuple(f"{prefix}._e{i}" for i in range(count))
        try:
            for proj in projections:
                for suffix in ("weight", "weight_scale", "weight_scale_2"):
                    tensor = self._sources[f"{prefix}.{proj}.{suffix}"]
                    if tensor.shape[0] != count:
                        raise ValueError(f"{prefix}.{proj}.{suffix}: expected {count} experts")
                    for i, ep in enumerate(expert_prefixes):
                        alias = f"{ep}.{proj}.{suffix}"
                        self._sources[alias] = tensor[i]
                        aliases.append(alias)
            return self.load_fp4_experts(
                expert_prefixes,
                hidden_size=lat_dim,
                intermediate_size=inter_dim,
                format="nvfp4",
                activation_mode="a16",
                a4_prefill=False,
                swiglu_limit=limit,
                swiglu_clamp_after_silu=False,
                scale_compression=self.expert_scale_compression,
                scale_scratch=self._scale_scratch(),
                scale_vectors=(
                    vectors["up_proj", "weight_scale_2"],
                    vectors["down_proj", "weight_scale_2"],
                    vectors["up_proj", "input_scale"].reciprocal(),
                    vectors["down_proj", "input_scale"].reciprocal(),
                ),
            )
        finally:
            for alias in aliases:
                del self._sources[alias]

    def load_layer(self, layer: int) -> LayerWeights:
        p = f"model.layers.{layer}"
        is_idx = self.config.is_indexed(layer)
        allow_dense = self._quantize_target_dense
        attention = self._load_attention(
            f"{p}.self_attn",
            fuse_gate=True,
            indexer_prefix=f"{p}.self_attn.compressed_indexer" if is_idx else None,
            allow_mxfp8=allow_dense,
        )
        indexer = (
            self._load_indexer(
                f"{p}.self_attn.compressed_indexer",
                fused_inputs=attention["fused_indexer_projs"],
            )
            if is_idx
            else None
        )
        if not self.config.is_sparse_mlp(layer):
            mlp: DenseMLPWeights | LatentMoEWeights = DenseMLPWeights(
                mlp=MLPWeights(
                    self._load_opt_mxfp8_group((f"{p}.mlp.gate_proj", f"{p}.mlp.up_proj"), allow=allow_dense),
                    self._load_opt_mxfp8(f"{p}.mlp.down_proj", allow=allow_dense),
                ),
                post_fc2_norm=self.load_tensor(f"{p}.mlp.post_fc2_norm.weight"),
            )
        else:
            mlp = LatentMoEWeights(
                router=self.load_tensor(f"{p}.mlp.gate.weight"),
                correction=persistent_copy(
                    self.load_tensor(f"{p}.mlp.gate.e_score_correction_bias", temporary=True).float()
                ),
                down_proj=self._load_opt_mxfp8(f"{p}.mlp.routed_expert_down_proj", allow=allow_dense),
                up_proj=self._load_opt_mxfp8(f"{p}.mlp.routed_expert_up_proj", allow=allow_dense),
                experts=self._load_stacked_nvfp4_experts(f"{p}.mlp.experts"),
                shared=MLPWeights(
                    self._load_opt_mxfp8_group(
                        (f"{p}.mlp.shared_experts.gate_proj", f"{p}.mlp.shared_experts.up_proj"),
                        allow=allow_dense,
                    ),
                    self._load_opt_mxfp8(f"{p}.mlp.shared_experts.down_proj", allow=allow_dense),
                ),
                combine_norm=self.load_tensor(f"{p}.mlp.combine_output_norm.weight"),
                experts_a4_prefill=False,
            )

        residual = {
            "attn_fn": self.load_tensor(f"{p}.hc_attn_layer.hc_pre.hc_fn", dtype=torch.float32),
            "attn_base": self.load_tensor(f"{p}.hc_attn_layer.hc_pre.hc_base", dtype=torch.float32),
            "attn_scale": self.load_tensor(f"{p}.hc_attn_layer.hc_pre.hc_scale", dtype=torch.float32),
            "mlp_fn": self.load_tensor(f"{p}.hc_mlp_layer.hc_pre.hc_fn", dtype=torch.float32),
            "mlp_base": self.load_tensor(f"{p}.hc_mlp_layer.hc_pre.hc_base", dtype=torch.float32),
            "mlp_scale": self.load_tensor(f"{p}.hc_mlp_layer.hc_pre.hc_scale", dtype=torch.float32),
            "input_norm": self.load_tensor(f"{p}.input_layernorm.weight"),
            "post_attn_norm": self.load_tensor(f"{p}.post_attention_layernorm.weight"),
        }

        engram = None
        if layer in self.config.engram_layers:
            ep = f"{p}.engram_layer"
            kv_prefixes = tuple(f"{ep}.key_projs.{hc}" for hc in range(HC_MULT)) + (f"{ep}.value_proj",)
            engram = {
                "unified_pre_gamma": self.load_tensor(f"{ep}.unified_pre_gamma"),
            }
            if self._same_linear_format(kv_prefixes) and self.linear_format(kv_prefixes[0]) != "block_fp8":
                biases = tuple(self.load_tensor(f"{kp}.bias", temporary=True) for kp in kv_prefixes)
                engram["kv_proj"] = self._load_opt_mxfp8_group(kv_prefixes, allow=allow_dense)
                engram["kv_bias"] = persistent_cat(biases, dim=0)
            else:
                engram["value_proj"] = self._load_opt_mxfp8(f"{ep}.value_proj", allow=allow_dense)
                engram["value_proj.bias"] = self.load_tensor(f"{ep}.value_proj.bias")
                for hc in range(HC_MULT):
                    engram[f"key_projs.{hc}"] = self._load_opt_mxfp8(f"{ep}.key_projs.{hc}", allow=allow_dense)
                    engram[f"key_projs.{hc}.bias"] = self.load_tensor(f"{ep}.key_projs.{hc}.bias")
            for h in range(ENGRAM_HEADS):
                engram[f"lookup_per_head_norms.{h}"] = self.load_tensor(f"{ep}.lookup_per_head_norms.{h}.weight")
            for hc in range(HC_MULT):
                engram[f"norm_key.{hc}"] = self.load_tensor(f"{ep}.norm_key.{hc}.weight")
                engram[f"norm_query.{hc}"] = self.load_tensor(f"{ep}.norm_query.{hc}.weight")
                engram[f"unified_per_hc_scales.{hc}"] = self.load_tensor(f"{ep}.unified_per_hc_scales.{hc}")

        return LayerWeights(attention=attention, indexer=indexer, mlp=mlp, residual=residual, engram=engram)

    def load_dfly(self) -> DFlyWeights:
        assert self.config.draft is not None
        dp = "model.dfly_layers"
        allow = self._quantize_draft_dense
        layers = tuple(
            DFlyLayerWeights(
                attention=self._load_attention(f"{dp}.{i}.self_attn", fuse_gate=True, allow_mxfp8=allow),
                mlp=MLPWeights(
                    self._load_opt_mxfp8_group((f"{dp}.{i}.mlp.gate_proj", f"{dp}.{i}.mlp.up_proj"), allow=allow),
                    self._load_opt_mxfp8(f"{dp}.{i}.mlp.down_proj", allow=allow),
                ),
                input_norm=self.load_tensor(f"{dp}.{i}.input_layernorm.weight"),
                post_attn_norm=self.load_tensor(f"{dp}.{i}.post_attention_layernorm.weight"),
            )
            for i in range(self.config.draft.layers)
        )
        conf_w = (
            self.load_tensor(f"{dp}.confidence_head.proj.weight")
            if f"{dp}.confidence_head.proj.weight" in self._sources
            else None
        )
        conf_b = (
            self.load_tensor(f"{dp}.confidence_head.proj.bias")
            if f"{dp}.confidence_head.proj.bias" in self._sources
            else None
        )
        return DFlyWeights(
            layers=layers,
            final_norm=self.load_tensor(f"{dp}.final_norm.weight"),
            mtp_enorm=self.load_tensor(f"{dp}.mtp_fusion.enorm.weight"),
            mtp_hnorm=self.load_tensor(f"{dp}.mtp_fusion.hnorm.weight"),
            mtp_eh_proj=self._load_opt_mxfp8(f"{dp}.mtp_fusion.eh_proj", allow=allow),
            mtp_gate=self.load_tensor(f"{dp}.mtp_fusion.gate"),
            hc_hidden_norm=self.load_tensor(f"{dp}.hidden_correction.hidden_norm.weight"),
            hc_embed_norm=self.load_tensor(f"{dp}.hidden_correction.embed_norm.weight"),
            hc_gate_up=self._load_opt_mxfp8_group(
                (f"{dp}.hidden_correction.gate_proj", f"{dp}.hidden_correction.up_proj"),
                allow=allow,
            ),
            hc_down=self._load_opt_mxfp8(f"{dp}.hidden_correction.down_proj", allow=allow),
            confidence_weight=conf_w,
            confidence_bias=conf_b,
        )

    def load_vision(self) -> dict[str, object]:
        """Load and pack SigLIP2 + vit.perceive weights once so all vision graphs share them."""
        vcfg = self.config.vision
        assert vcfg is not None
        h = vcfg.hidden_size
        out_h = vcfg.out_hidden_size
        pixel_w = 3 * vcfg.patch_size * vcfg.patch_size
        allow = self._quantize_target_dense
        packed: dict[str, object] = {
            "patch_w": self.load_tensor("vit.embeddings.patch_embedding.weight").reshape(h, pixel_w),
            "patch_b": self.load_tensor("vit.embeddings.patch_embedding.bias"),
            "pos_embed": persistent_copy(
                self.load_tensor("vit.embeddings.position_embedding.weight", temporary=True)[1:].contiguous()
            ),
            "before_rms": self.load_tensor("vit.perceive.norm.weight"),
            "conv0_w": self.load_tensor("vit.perceive.proj.0.weight").reshape(4 * h, 4 * h),
            "conv0_b": self.load_tensor("vit.perceive.proj.0.bias"),
            "conv2_w": self.load_tensor("vit.perceive.proj.2.weight").reshape(out_h, 4 * h),
            "conv2_b": self.load_tensor("vit.perceive.proj.2.bias"),
            "vit.perceive.image_newline": self.load_tensor("vit.perceive.image_newline"),
        }
        blocks = []
        for i in range(vcfg.num_hidden_layers):
            p = f"vit.layers.{i}"
            q_w, k_w, v_w = self.load_tensors(
                (f"{p}.self_attn.q_proj.weight", f"{p}.self_attn.k_proj.weight", f"{p}.self_attn.v_proj.weight"),
                temporary=True,
            )
            q_b, k_b, v_b = self.load_tensors(
                (f"{p}.self_attn.q_proj.bias", f"{p}.self_attn.k_proj.bias", f"{p}.self_attn.v_proj.bias"),
                temporary=True,
            )
            qkv_cat = torch.cat((q_w, k_w, v_w), dim=0) if allow else persistent_cat((q_w, k_w, v_w), dim=0)
            qkv_w = (
                quantize_bf16_linear_as_mxfp8(qkv_cat, retain_scale_rows=False)
                if allow and self._can_mxfp8_tensor(qkv_cat)
                else (persistent_copy(qkv_cat) if allow else qkv_cat)
            )
            blocks.append(
                {
                    "ln1_w": self.load_tensor(f"{p}.input_layernorm.weight"),
                    "ln1_b": self.load_tensor(f"{p}.input_layernorm.bias"),
                    "qkv_w": qkv_w,
                    "qkv_b": persistent_cat((q_b, k_b, v_b), dim=0),
                    "o_w": self._load_opt_mxfp8(f"{p}.self_attn.o_proj", allow=allow),
                    "o_b": self.load_tensor(f"{p}.self_attn.o_proj.bias"),
                    "ln2_w": self.load_tensor(f"{p}.post_attention_layernorm.weight"),
                    "ln2_b": self.load_tensor(f"{p}.post_attention_layernorm.bias"),
                    "fc1_w": self._load_opt_mxfp8(f"{p}.mlp.dense_h_to_4h", allow=allow),
                    "fc1_b": self.load_tensor(f"{p}.mlp.dense_h_to_4h.bias"),
                    "fc2_w": self._load_opt_mxfp8(f"{p}.mlp.dense_4h_to_h", allow=allow),
                    "fc2_b": self.load_tensor(f"{p}.mlp.dense_4h_to_h.bias"),
                }
            )
        packed["blocks"] = tuple(blocks)
        return packed
