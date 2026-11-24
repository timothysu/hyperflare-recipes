# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Direct weight loading for Ling-3.2-Air MXFP8 dense projections and NVFP4-CSF experts."""

from contextlib import nullcontext
from dataclasses import dataclass
from functools import cached_property

import torch

from hyperflare_recipes.components.vocabulary import mxfp8_draft_head, nvfp4_draft_head
from hyperflare_recipes.components.weights import DirectWeightLoader, MLPWeights
from ignition.preparation.persistent import persistent_cat, persistent_copy
from ignition.preparation.quantization import expert_format
from ignition.preparation.rebuild_manifest import weight_container

from .config import ModelConfig
from .vision import VisionBlockWeights, VisionWeights


@weight_container(
    ("qkvfg", "b_proj", "output", "convolution", "o_norm", "A_log", "dt_bias")
)
@dataclass(frozen=True)
class KDAWeights:
    qkvfg: object
    b_proj: torch.Tensor
    output: object
    convolution: torch.Tensor
    o_norm: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor


@weight_container(("q_kv_a", "g_proj", "kv_norm", "key", "value", "output"))
@dataclass(frozen=True)
class MLAWeights:
    q_kv_a: object
    g_proj: torch.Tensor
    kv_norm: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    output: object


@weight_container(
    ("attention", "input_norm", "post_attention_norm", "mlp", "router", "expert_bias", "experts")
)
@dataclass(frozen=True)
class LayerWeights:
    attention: KDAWeights | MLAWeights
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    mlp: MLPWeights
    router: torch.Tensor | None
    expert_bias: torch.Tensor | None
    experts: object | None


@weight_container(("enorm", "hnorm", "eh_proj", "layer", "final_norm", "draft_head"))
@dataclass(frozen=True)
class MTPWeights:
    enorm: torch.Tensor
    hnorm: torch.Tensor
    eh_proj: torch.Tensor
    layer: LayerWeights
    final_norm: torch.Tensor
    draft_head: object | None = None


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    shard: object
    embedding: torch.Tensor
    norm: torch.Tensor
    head: torch.Tensor
    layers: tuple[LayerWeights, ...]
    allocated_bytes: int
    draft: MTPWeights | None = None
    vision: VisionWeights | None = None
    scale_scratch: tuple[torch.Tensor, ...] = ()

    @classmethod
    def load(
        cls,
        manifest,
        device,
        *,
        shard,
        progress=None,
        mtp=True,
        vision=False,
        expert_scale_compression="csf",
        draft_head_quantization=None,
        core_storage=None,
    ):
        before = torch.cuda.memory_allocated(device)
        loading = nullcontext if core_storage is None else core_storage.transient
        with loading(), Ling32WeightLoader(
            manifest,
            device,
            shard=shard,
            progress=progress,
            mtp=mtp,
            vision=vision,
            expert_scale_compression=expert_scale_compression,
            draft_head_quantization=draft_head_quantization,
        ) as loader:
            config = loader.config
            embedding = loader.load_tensor("model.word_embeddings.weight")
            norm = loader.norm("model.norm.weight", config.hidden)
            head = loader.load_tensor("lm_head.weight")
            layers = tuple(loader.load_layer(i) for i in range(config.layers))
            draft = loader.load_mtp(head) if (mtp and config.num_mtp_layers > 0) else None
            vision_weights = loader.load_vision() if vision else None
            scale_scratch = loader.scale_scratch
        return cls(
            config=config,
            shard=shard,
            embedding=embedding,
            norm=norm,
            head=head,
            layers=layers,
            allocated_bytes=torch.cuda.memory_allocated(device) - before,
            draft=draft,
            vision=vision_weights,
            scale_scratch=scale_scratch,
        )


class Ling32WeightLoader(DirectWeightLoader):
    def __init__(
        self,
        manifest,
        device,
        *,
        shard,
        progress=None,
        prefixes=None,
        mtp=True,
        vision=False,
        expert_scale_compression="csf",
        draft_head_quantization=None,
    ):
        super().__init__(manifest, device, progress=progress, prefixes=prefixes)
        self.config = ModelConfig.load(manifest.snapshot)
        self.shard = shard
        self.config.validate_shard(shard)
        self.mtp = bool(mtp)
        self.vision = bool(vision)
        if self.vision and self.config.vision is None:
            raise ValueError("Ling-3.2-Air vision loading requires vision_config in config.json")
        if expert_scale_compression not in (None, "csf"):
            raise ValueError("Ling-3.2-Air expert scale compression must be None or 'csf'")
        if draft_head_quantization not in (None, "mxfp8", "nvfp4"):
            raise ValueError("Ling-3.2-Air draft head quantization must be None, 'mxfp8', or 'nvfp4'")
        self.expert_scale_compression = expert_scale_compression
        self.draft_head_quantization = draft_head_quantization

    @cached_property
    def scale_scratch(self) -> tuple[torch.Tensor, ...]:
        if self.expert_scale_compression is None:
            return ()
        c = self.config.text
        count = c["num_experts"]
        inter = c["moe_intermediate_size"]
        hidden = c["hidden_size"]
        padded = (inter + 127) // 128 * 128
        return tuple(
            torch.empty(shape, dtype=torch.float8_e4m3fn, device=self.device)
            for shape in (
                (count, 2 * padded, hidden // 16),
                (count, hidden, padded // 16),
            )
        )

    def _expert_scale_scratch(self, inter: int):
        if not self.scale_scratch:
            return None
        c = self.config.text
        count = c["num_experts"]
        hidden = c["hidden_size"]
        shapes = (
            (count, (2 * inter + 127) // 128 * 128, hidden // 16),
            (count, hidden, (inter // 16 + 3) // 4 * 4),
        )
        return tuple(
            buf.flatten()[: count * rows * cols].view(count, rows, cols)
            for buf, (count, rows, cols) in zip(self.scale_scratch, shapes, strict=True)
        )

    def _selected_names(self):
        total = self.config.layers + (self.config.num_mtp_layers if self.mtp else 0)

        def selected(name: str) -> bool:
            if name in ("model.word_embeddings.weight", "model.norm.weight", "lm_head.weight"):
                return True
            if self.vision and (name.startswith("model.visual.") or name.startswith("linear_proj.")):
                return True
            parts = name.split(".")
            return parts[:2] == ["model", "layers"] and len(parts) > 2 and int(parts[2]) < total

        return {name for name in super()._selected_names() if selected(name)}

    @cached_property
    def quantization(self):
        return self.config.quantization

    def norm(self, name: str, width: int) -> torch.Tensor:
        source = self._sources[name]
        if source.shape != (width,) or source.dtype != torch.bfloat16:
            raise ValueError(f"{name}: expected BF16 ({width},) normalization weight, got {source.dtype} {source.shape}")
        return self.load_tensor(name)

    def _load_kda(self, prefix: str) -> KDAWeights:
        attn = f"{prefix}.attention"
        qkvfg = self.load_linear_group(
            (f"{attn}.q_proj", f"{attn}.k_proj", f"{attn}.v_proj", f"{attn}.f_proj", f"{attn}.g_proj")
        )
        b_proj = self.load_linear(f"{attn}.b_proj")
        output = self.load_linear(f"{attn}.o_proj")
        convs = self.load_tensors(
            (f"{attn}.q_conv1d.weight", f"{attn}.k_conv1d.weight", f"{attn}.v_conv1d.weight"),
            temporary=True,
        )
        # Checkpoint stores [4096, 1, 4] BF16 per branch; pack into [12288, 4] for short_conv.
        convolution = persistent_cat(tuple(c.squeeze(1) for c in convs), dim=0)
        o_norm = self.norm(f"{attn}.o_norm.weight", 128)
        A_log = self.load_tensor(f"{attn}.A_log")
        dt_bias = self.load_tensor(f"{attn}.dt_bias")
        if A_log.dtype != torch.float32 or A_log.shape != (self.config.heads,):
            raise ValueError(f"{attn}.A_log: expected FP32 ({self.config.heads},)")
        if dt_bias.dtype != torch.float32 or dt_bias.shape != (self.config.heads * 128,):
            raise ValueError(f"{attn}.dt_bias: expected FP32 ({self.config.heads * 128},)")
        return KDAWeights(
            qkvfg=qkvfg,
            b_proj=b_proj,
            output=output,
            convolution=convolution,
            o_norm=o_norm,
            A_log=A_log,
            dt_bias=dt_bias,
        )

    def _load_mla(self, prefix: str) -> MLAWeights:
        attn = f"{prefix}.attention"
        q_kv_a = self.load_linear_group((f"{attn}.q_proj", f"{attn}.kv_a_proj_with_mqa"))
        g_proj = self.load_linear(f"{attn}.g_proj")
        kv_norm = self.norm(f"{attn}.kv_a_layernorm.weight", 512)
        kv_b = self.load_tensor(f"{attn}.kv_b_proj.weight", temporary=True)
        heads = self.config.heads
        if kv_b.dtype != torch.bfloat16 or kv_b.shape != (heads * 256, 512):
            raise ValueError(f"{attn}.kv_b_proj.weight: expected BF16 ({heads * 256}, 512)")
        kv_b_heads = kv_b.view(heads, 256, 512)
        # BatchedUnquantizedProjection(major="n") expects [H, in_dim, out_dim]:
        # absorb: [H, T, 128] @ [H, 128, 512] -> [H, T, 512]
        # expand: [H, T, 512] @ [H, 512, 128] -> [H, T, 128]
        key = persistent_copy(kv_b_heads[:, :128, :].contiguous())
        value = persistent_copy(kv_b_heads[:, 128:, :].transpose(1, 2).contiguous())
        output = self.load_linear(f"{attn}.dense")
        return MLAWeights(
            q_kv_a=q_kv_a,
            g_proj=g_proj,
            kv_norm=kv_norm,
            key=key,
            value=value,
            output=output,
        )

    def _load_mlp(self, prefix: str) -> MLPWeights:
        gate_up = self.load_linear_group((f"{prefix}.gate_proj", f"{prefix}.up_proj"))
        down = self.load_linear(f"{prefix}.down_proj")
        return MLPWeights(gate_up, down)

    def _load_experts(self, prefix: str, limit: float | None):
        c = self.config.text
        count = c["num_experts"]
        inter = c["moe_intermediate_size"]
        hidden = c["hidden_size"]
        projections = ("up_proj", "gate_proj", "down_proj")
        for proj in projections:
            fmt, mode = expert_format(self.quantization, f"{prefix}.{proj}")
            if fmt != "nvfp4" or mode != "a4":
                raise ValueError(
                    f"{prefix}.{proj}: Ling-3.2-Air routed experts require NVFP4 group_size=16 (a4), got {(fmt, mode)!r}"
                )
        vectors = {}
        for proj in projections:
            for suffix in ("input_scale", "weight_scale_2"):
                value = self._sources[f"{prefix}.{proj}.{suffix}"]
                if (
                    value.dtype != torch.float32
                    or value.shape != (count,)
                    or not bool((torch.isfinite(value) & (value > 0)).all())
                ):
                    raise ValueError(f"{prefix}.{proj}.{suffix}: expected positive finite FP32 [{count}]")
                vectors[proj, suffix] = value
        for suffix in ("input_scale", "weight_scale_2"):
            if not torch.equal(vectors["up_proj", suffix], vectors["gate_proj", suffix]):
                raise ValueError(f"{prefix}: gate_proj and up_proj {suffix} must match for W13 fusion")

        aliases = []
        expert_prefixes = tuple(f"{prefix}._slice.{i}" for i in range(count))
        try:
            for proj in projections:
                for suffix in ("weight", "weight_scale", "weight_scale_2"):
                    stacked = self._sources[f"{prefix}.{proj}.{suffix}"]
                    if stacked.shape[0] != count:
                        raise ValueError(f"{prefix}.{proj}.{suffix}: expected leading dim {count}")
                    for i, ep in enumerate(expert_prefixes):
                        alias_name = f"{ep}.{proj}.{suffix}"
                        self._sources[alias_name] = stacked[i]
                        aliases.append(alias_name)
            return self.load_fp4_experts(
                expert_prefixes,
                hidden_size=hidden,
                intermediate_size=inter,
                format="nvfp4",
                activation_mode="a16",
                a4_prefill=True,
                swiglu_limit=limit,
                swiglu_clamp_after_silu=True,
                scale_compression=self.expert_scale_compression,
                scale_scratch=self._expert_scale_scratch(inter),
                scale_vectors=(
                    vectors["up_proj", "weight_scale_2"],
                    vectors["down_proj", "weight_scale_2"],
                    vectors["up_proj", "input_scale"].reciprocal(),
                    vectors["down_proj", "input_scale"].reciprocal(),
                ),
            )
        finally:
            for alias_name in aliases:
                del self._sources[alias_name]

    def load_layer(self, layer: int) -> LayerWeights:
        prefix = f"model.layers.{layer}"
        is_mla = layer >= self.config.layers or self.config.layer_types[layer] == "mla"
        attention = self._load_mla(prefix) if is_mla else self._load_kda(prefix)
        input_norm = self.norm(f"{prefix}.input_layernorm.weight", self.config.hidden)
        post_norm = self.norm(f"{prefix}.post_attention_layernorm.weight", self.config.hidden)
        if self.config.is_moe(layer):
            mlp = self._load_mlp(f"{prefix}.mlp.shared_experts")
            router = self.load_tensor(f"{prefix}.mlp.gate.weight")
            expert_bias = self.load_tensor(f"{prefix}.mlp.gate.expert_bias")
            if router.dtype != torch.float32 or router.shape != (self.config.text["num_experts"], self.config.hidden):
                raise ValueError(f"{prefix}.mlp.gate.weight: expected FP32 router matrix")
            if expert_bias.dtype != torch.float32 or expert_bias.shape != (self.config.text["num_experts"],):
                raise ValueError(f"{prefix}.mlp.gate.expert_bias: expected FP32 router bias")
            experts = self._load_experts(f"{prefix}.mlp.experts", self.config.expert_swiglu_limit(layer))
        else:
            mlp = self._load_mlp(f"{prefix}.mlp")
            router = expert_bias = experts = None
        return LayerWeights(
            attention=attention,
            input_norm=input_norm,
            post_attention_norm=post_norm,
            mlp=mlp,
            router=router,
            expert_bias=expert_bias,
            experts=experts,
        )

    def load_mtp(self, shared_head: torch.Tensor) -> MTPWeights:
        layer_idx = self.config.layers
        prefix = f"model.layers.{layer_idx}"
        enorm = self.norm(f"{prefix}.enorm.weight", self.config.hidden)
        hnorm = self.norm(f"{prefix}.hnorm.weight", self.config.hidden)
        eh_proj = self.load_linear(f"{prefix}.eh_proj")
        final_norm = self.norm(f"{prefix}.final_layernorm.weight", self.config.hidden)
        layer = self.load_layer(layer_idx)
        draft_head = None
        if self.draft_head_quantization == "mxfp8":
            draft_head = mxfp8_draft_head(shared_head)
        elif self.draft_head_quantization == "nvfp4":
            draft_head = nvfp4_draft_head(shared_head)
        return MTPWeights(
            enorm=enorm,
            hnorm=hnorm,
            eh_proj=eh_proj,
            layer=layer,
            final_norm=final_norm,
            draft_head=draft_head,
        )

    def _load_bf16(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        source = self._sources[name]
        if source.dtype != torch.bfloat16 or tuple(source.shape) != shape:
            raise ValueError(f"{name}: expected BF16 {shape}, got {source.dtype} {tuple(source.shape)}")
        return self.load_tensor(name)

    def load_vision(self) -> VisionWeights:
        vc = self.config.vision
        if vc is None:
            raise ValueError("Ling-3.2-Air vision loading requires vision_config")
        h, inter, merged, out = vc.hidden_size, vc.intermediate_size, vc.merged_hidden_size, vc.out_hidden_size
        patch_raw = self.load_tensor("model.visual.patch_embed.proj.weight", temporary=True)
        expected_patch = (h, vc.in_channels, vc.temporal_patch_size, vc.patch_size, vc.patch_size)
        if patch_raw.dtype != torch.bfloat16 or tuple(patch_raw.shape) != expected_patch:
            raise ValueError(f"model.visual.patch_embed.proj.weight: expected BF16 {expected_patch}")
        patch_embed_weight = persistent_copy(patch_raw.reshape(h, vc.patch_dim).contiguous())
        patch_embed_bias = self._load_bf16("model.visual.patch_embed.proj.bias", (h,))
        pos_embed = self._load_bf16("model.visual.pos_embed.weight", (vc.num_position_embeddings, h))

        blocks = []
        for idx in range(vc.depth):
            p = f"model.visual.blocks.{idx}"
            blocks.append(
                VisionBlockWeights(
                    norm1_weight=self._load_bf16(f"{p}.norm1.weight", (h,)),
                    norm1_bias=self._load_bf16(f"{p}.norm1.bias", (h,)),
                    qkv_weight=self._load_bf16(f"{p}.attn.qkv.weight", (3 * h, h)),
                    qkv_bias=self._load_bf16(f"{p}.attn.qkv.bias", (3 * h,)),
                    proj_weight=self._load_bf16(f"{p}.attn.proj.weight", (h, h)),
                    proj_bias=self._load_bf16(f"{p}.attn.proj.bias", (h,)),
                    norm2_weight=self._load_bf16(f"{p}.norm2.weight", (h,)),
                    norm2_bias=self._load_bf16(f"{p}.norm2.bias", (h,)),
                    fc1_weight=self._load_bf16(f"{p}.mlp.linear_fc1.weight", (inter, h)),
                    fc1_bias=self._load_bf16(f"{p}.mlp.linear_fc1.bias", (inter,)),
                    fc2_weight=self._load_bf16(f"{p}.mlp.linear_fc2.weight", (h, inter)),
                    fc2_bias=self._load_bf16(f"{p}.mlp.linear_fc2.bias", (h,)),
                )
            )

        return VisionWeights(
            patch_embed_weight=patch_embed_weight,
            patch_embed_bias=patch_embed_bias,
            pos_embed=pos_embed,
            blocks=tuple(blocks),
            merger_norm_weight=self._load_bf16("model.visual.merger.norm.weight", (h,)),
            merger_norm_bias=self._load_bf16("model.visual.merger.norm.bias", (h,)),
            linear_proj_0_weight=self._load_bf16("linear_proj.0.weight", (out, merged)),
            linear_proj_0_bias=self._load_bf16("linear_proj.0.bias", (out,)),
            linear_proj_2_weight=self._load_bf16("linear_proj.2.weight", (out, out)),
            linear_proj_2_bias=self._load_bf16("linear_proj.2.bias", (out,)),
        )
