# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Canonical Qwen dense layers with prepared projections and persistent GPU caches."""

from copy import copy
from dataclasses import dataclass

import torch
import triton

from hyperflare_recipes.components.embedding import PreparedEmbedding
from hyperflare_recipes.components.dflash2 import PreparedContextCapture
from hyperflare_recipes.components.gated_delta import GDNScratch, PreparedGDN
from hyperflare_recipes.components.head_gate import PreparedHeadGate
from hyperflare_recipes.components.linear import QuantizedProjection, prepare_projection
from hyperflare_recipes.components.gdn_recovery import GDNLayerRecords, PreparedGDNCommit
from hyperflare_recipes.components.paged_attention import PreparedPagedAttention
from hyperflare_recipes.components.paged_batch import rotary_offset_region, rotary_positions
from hyperflare_recipes.components.recurrent import DeltaMetadata, _commit_convolution
from hyperflare_recipes.components.rms_norm import PreparedRMSNorm
from hyperflare_recipes.components.head_norm_rope import qk_head_norm_rope_kv_write
from hyperflare_recipes.components.tensor_fp8 import TensorFP8Projection, TensorFP8Weight
from hyperflare_recipes.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation.quantization import linear_format

from .config import PAGE_TOKENS, rotary_axes, rotary_sections


@dataclass(frozen=True)
class Scratch:
    """Declaration and tuning examples released before forward compilation."""

    activations: torch.Tensor
    hidden: torch.Tensor
    logits: torch.Tensor
    gdn: GDNScratch
    quantized: torch.Tensor

    @classmethod
    def allocate(cls, tokens, rows, config, device):
        def zeros(shape):
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        quantized_width = (max(config.hidden, config.text["num_attention_heads"] * config.text["head_dim"], 6144)
            if any(linear_format(config.quantization, name) == "tensor_fp8" for name in config.quantization) else 0)
        quantized_shape = (tokens * quantized_width,)
        quantized = torch.zeros(quantized_shape, dtype=torch.uint8, device=device)
        return cls(zeros((tokens, config.hidden)), zeros((rows, config.hidden)),
                   zeros((rows, config.vocabulary)),
                   GDNScratch.allocate(tokens, device, hidden_size=config.hidden,
                                       pack_ba=config.pack_gdn_inputs), quantized)

    def rows(self, tokens, rows):
        return Scratch(self.activations[:tokens], self.hidden[:rows], self.logits[:rows], self.gdn.rows(tokens), self.quantized)


class Projection(torch.nn.Module):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, source):
        return self.operation(source)


def rotary_embeddings(positions, inverse_frequency, axes=None, position_axes=None):
    """Cosine and sine tables for token or per-axis rotary positions.

    One-dimensional ``positions`` hold one position per token, and every
    frequency pair rotates at it. Two-dimensional ``positions`` supply one column
    per declared axis, or one column that every axis reuses; ``axes`` names the
    axis of each frequency pair and ``position_axes`` counts the declared axes.
    """
    if positions.ndim == 1:
        frequency = positions.float()[:, None] * inverse_frequency[None, :]
    else:
        columns = positions.expand(-1, position_axes) if positions.shape[1] == 1 else positions
        frequency = columns[:, axes].float() * inverse_frequency[None, :]
    embedding = torch.cat((frequency, frequency), dim=-1)
    return embedding.cos().bfloat16(), embedding.sin().bfloat16()


def _uses_fused_swiglu(activation, down_weight, device):
    from b12x.gemm.blockscaled import NVFP4LinearWeight
    from transformers.activations import SiLUActivation
    from hyperflare_recipes.components import sm103_projection

    canonical_silu = (type(activation) is SiLUActivation
                      or type(activation) is torch.nn.SiLU and not activation.inplace)
    return canonical_silu and isinstance(down_weight, NVFP4LinearWeight) and not sm103_projection.is_sm103(device)


def _allocate_mlp_intermediate(weights, source):
    for index, layer in enumerate(weights.tree.layers):
        prefix = f"layers.{index}.mlp."
        if (prefix + "gate_up" not in weights.projections
                or not _uses_fused_swiglu(layer.mlp.act_fn, weights.projections[prefix + "down_proj"], source.device)):
            shape = (source.shape[0], weights.config.text["intermediate_size"])
            return torch.zeros(shape, dtype=torch.bfloat16, device=source.device)
    return None


class GroupedMLP(torch.nn.Module):
    """Concatenated gate/up projection with the canonical BF16 activation."""

    def __init__(self, canonical, source, gate_up_weight, down_weight, gate_up, intermediate, output,
                 *, session, name, workspace):
        super().__init__()
        self.activation = canonical.act_fn
        self.gate_up = prepare_projection(source, gate_up_weight, gate_up,
            activation_mode="auto", session=session, workspace=workspace, name=name + ".gate_up").functional(session=session)
        self.down_source_is_gate_up = _uses_fused_swiglu(self.activation, down_weight, source.device)
        if self.down_source_is_gate_up:
            self.down = QuantizedProjection(gate_up, down_weight, output,
                source_activation="swiglu_bf16", activation_mode="auto", session=session,
                workspace=workspace, name=name + ".down").functional(session=session)
        else:
            self.down = prepare_projection(intermediate, down_weight, output,
                activation_mode="auto", session=session, workspace=workspace, name=name + ".down").functional(session=session)

    def forward(self, source):
        projected = self.gate_up(source)
        if self.down_source_is_gate_up:
            return self.down(projected)
        gate, up = projected.chunk(2, dim=-1)
        return self.down(self.activation(gate) * up)


class RMSNorm(torch.nn.Module):
    def __init__(self, canonical, source, output, *, session, name, residual=None):
        super().__init__()
        self.operation = PreparedRMSNorm(source, canonical.weight, output, eps=canonical.eps,
            zero_centered=True, session=session, name=name, residual=residual).functional()

    def forward(self, source, residual=None):
        return self.operation(source, residual)


class ResidualNormalizedLayer(torch.nn.Module):
    """Compute attention and MLP branches with prepared residual normalization."""

    def __init__(self, layer, *, cache_only=False):
        super().__init__()
        self.layer = layer
        self.next_input_layernorm = None
        self.cache_only = bool(cache_only)

    def forward(self, hidden_states, normalized, position_embeddings, records=None, *, metadata, delta,
                slots=None, lengths=None):
        layer = self.layer
        if self.cache_only:
            layer.self_attn.write_cache(normalized, position_embeddings=position_embeddings,
                                        metadata=metadata, slots=slots, lengths=lengths)
            return hidden_states, None
        if layer.block_type == "linear_attention":
            attended = layer.linear_attn(normalized, records=records, metadata=delta)
        else:
            attended, _ = layer.self_attn(normalized, position_embeddings=position_embeddings,
                                          metadata=metadata, slots=slots, lengths=lengths)
        normalized = layer.post_attention_layernorm(attended, hidden_states)
        projected = layer.mlp(normalized)
        if self.next_input_layernorm is None:
            return attended + projected, None
        return projected, self.next_input_layernorm(projected, attended)


def module_copy(module):
    result = copy(module)
    result._modules = module._modules.copy()
    return result


class RecurrentAttention(torch.nn.Module):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, hidden_states, records=None, *, metadata, **kwargs):
        return self.operation(hidden_states, records=records, metadata=metadata)


def _attention_query_dtype(source, metadata, mode, head_dim):
    if (mode == "prefill" and source.shape[0] >= 128 and head_dim == 256
            and metadata.page_table.shape[0] == 1
            and torch.cuda.get_device_capability(source.device) in ((12, 0), (12, 1))):
        return torch.float8_e4m3fn
    return torch.bfloat16


class FullAttention(torch.nn.Module):
    def __init__(self, module, weights, source, metadata, pool, layer, *, mode, session, workspace, name, buffers,
                 projection_staging=None, cache_only=False):
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

        self.canonical = module
        self.cache_only = bool(cache_only)
        config = weights.config
        self.heads, self.kv_heads = config.text["num_attention_heads"], config.text["num_key_value_heads"]
        self.dim = config.text["head_dim"]
        q_norm, k_norm = module.q_norm, module.k_norm
        if (not isinstance(q_norm, (Qwen3_5RMSNorm, torch.nn.RMSNorm))
                or type(q_norm) is not type(k_norm)
                or type(q_norm.eps) not in (int, float)
                or not (q_norm.eps >= 0)
                or q_norm.eps != k_norm.eps
                or q_norm.weight.shape != (self.dim,)
                or k_norm.weight.shape != (self.dim,)
                or not q_norm.weight.is_contiguous()
                or not k_norm.weight.is_contiguous()):
            raise ValueError(
                "FullAttention fused Q/K RMSNorm requires matching Qwen3_5RMSNorm or "
                "torch.nn.RMSNorm modules with contiguous head_dim weights and nonnegative eps"
            )
        self._qk_norm_zero_centered = isinstance(q_norm, Qwen3_5RMSNorm)
        tokens, device = source.shape[0], source.device
        self.tokens = tokens

        def empty(field, width, dtype=torch.bfloat16):
            if field not in buffers:
                buffers[field] = torch.zeros((tokens, width), dtype=dtype, device=device)
            return buffers[field][:tokens]

        prefix = f"layers.{layer}.self_attn."
        packed = prefix + "qgkv_proj" in weights.projections
        self.packed = packed
        if packed:
            q_width, kv_width = self.heads * self.dim, self.kv_heads * self.dim
            qgkv = empty("qgkv", 2 * q_width + 2 * kv_width)
            query, gate, key, value = qgkv.split((q_width, q_width, kv_width, kv_width), dim=-1)
            query, gate = (part.view(tokens, self.heads, self.dim) for part in (query, gate))
            inputs = (("qgkv_proj", source, qgkv),)
        else:
            if not self.cache_only:
                qg = empty("qg", 2 * self.heads * self.dim)
                query, gate = qg.view(tokens, self.heads, 2 * self.dim).chunk(2, dim=-1)
            if prefix + "kv_proj" in weights.projections:
                kv = empty("kv", 2 * self.kv_heads * self.dim)
                key, value = kv.chunk(2, dim=-1)
                inputs = (("kv_proj", source, kv),)
            else:
                key, value = (empty(field, self.kv_heads * self.dim) for field in ("k", "v"))
                inputs = (("k_proj", source, key), ("v_proj", source, value))
            if not self.cache_only:
                inputs = (("q_proj", source, qg), *inputs)
        query_dtype = _attention_query_dtype(source, metadata, mode, self.dim)
        self.query_dtype = query_dtype
        def projection(field, inputs, output):
            weight = weights.projections[prefix + field]
            if isinstance(weight, TensorFP8Weight):
                return TensorFP8Projection(inputs, weight, output, session=session, name=name + "." + field,
                                           staging=projection_staging).functional(session=session)
            return prepare_projection(inputs, weight, output, session=session, name=name + "." + field,
                                      workspace=workspace).functional(session=session)

        key_cache, value_cache = pool.attention(layer, config)
        if (key_cache.ndim != 4 or value_cache.shape != key_cache.shape
                or value_cache.stride() != key_cache.stride()
                or value_cache.dtype != key_cache.dtype
                or key_cache.shape[1] != PAGE_TOKENS
                or key_cache.shape[2:] != (self.kv_heads, self.dim)
                or key_cache.stride(3) != 1
                or key_cache.dtype not in (torch.float8_e4m3fn, torch.bfloat16)):
            raise ValueError("FullAttention requires matching 4D [pages, PAGE_TOKENS, kv_heads, head_dim] FP8/BF16 KV caches")
        self.key_cache, self.value_cache = key_cache, value_cache

        if self.cache_only:
            self.input_projections = tuple(field for field, _, _ in inputs)
            self.projections = {
                field: projection(field, proj_in, proj_out)
                for field, proj_in, proj_out in inputs
            }
            self.lengths = None
            self.attention = None
            self.output_gate = None
            return

        q = empty("q", self.heads * self.dim, query_dtype).view(tokens, self.heads, self.dim)
        attended, output = empty("attended", self.heads * self.dim), empty("output", config.hidden)
        operands = (*inputs, ("o_proj", attended, output))
        self.input_projections = tuple(field for field, _, _ in operands[:-1])
        self.projections = {
            field: projection(field, proj_in, proj_out)
            for field, proj_in, proj_out in operands
        }
        self.lengths = torch.zeros_like(metadata.lengths)
        self.attention = PreparedPagedAttention(q, key_cache, value_cache, metadata.page_table,
            self.lengths, metadata.query_start_loc, attended.view(tokens, self.heads, self.dim),
            output_gate=gate if mode == "prefill" else None,
            tuning_page_count=int(pool.memory_capacity[1].item()),
            mode=mode, width=metadata.max_query_tokens if mode == "verify" else 1,
            session=session, workspace=workspace, name=name + ".attention", caller_owned_placeholders=True)
        if query_dtype == torch.float8_e4m3fn and self.attention.tma is None:
            raise ValueError("FP8 Qwen prefill queries require the prepared TMA attention path")
        self.attention = self.attention.functional(session=session, cache_seqlens=None,
                                                    page_table=None, cu_seqlens_q=None)
        self.lengths = None
        self.output_gate = None
        if mode != "prefill":
            values = (attended.view(tokens, self.heads * self.dim, 1) if packed
                      else attended.view(tokens * self.heads, self.dim, 1))
            gates = (gate.view(tokens, self.heads * self.dim) if packed
                     else gate.view(tokens * self.heads, self.dim))
            self.output_gate = PreparedHeadGate(values, gates, session=session, name=name + ".output_gate").functional(session=session)

    def write_cache(self, hidden_states, position_embeddings, metadata, *, slots=None, lengths=None):
        projected = {field: self.projections[field](hidden_states) for field in self.input_projections}
        tokens, heads, kv_heads, dim = self.tokens, self.heads, self.kv_heads, self.dim
        if self.packed:
            query, gate, key, value = projected["qgkv_proj"].split(
                (heads * dim, heads * dim, kv_heads * dim, kv_heads * dim), dim=-1)
            if not self.cache_only:
                query, gate = (part.view(tokens, heads, dim) for part in (query, gate))
        else:
            if not self.cache_only:
                query, gate = projected["q_proj"].view(tokens, heads, 2 * dim).chunk(2, dim=-1)
            if "kv_proj" in projected:
                key, value = projected["kv_proj"].chunk(2, dim=-1)
            else:
                key, value = projected["k_proj"], projected["v_proj"]
        m = metadata
        if slots is None:
            page = m.token_page_table.gather(1, (m.positions.clamp(min=0) // PAGE_TOKENS)[:, None]).squeeze(1)
            slots = torch.where(m.positions >= 0, page.long() * PAGE_TOKENS + m.positions % PAGE_TOKENS, -1)
        cosine, sine = position_embeddings
        if self.cache_only:
            qk_head_norm_rope_kv_write(
                None,
                key.view(tokens, kv_heads, dim),
                value.view(tokens, kv_heads, dim),
                None,
                self.canonical.k_norm.weight,
                cosine,
                sine,
                slots,
                None,
                self.key_cache,
                self.value_cache,
                eps=self.canonical.k_norm.eps,
                zero_centered=self._qk_norm_zero_centered,
            )
            return None, None, None
        q = torch.empty((tokens, heads, dim), dtype=self.query_dtype, device=hidden_states.device)
        qk_head_norm_rope_kv_write(
            query,
            key.view(tokens, kv_heads, dim),
            value.view(tokens, kv_heads, dim),
            self.canonical.q_norm.weight,
            self.canonical.k_norm.weight,
            cosine,
            sine,
            slots,
            q,
            self.key_cache,
            self.value_cache,
            eps=self.canonical.q_norm.eps,
            zero_centered=self._qk_norm_zero_centered,
        )
        if lengths is None:
            lengths = torch.where(m.lengths > 0, m.starts + m.lengths, 0).to(m.lengths.dtype)
        return q, gate, lengths

    def forward(self, hidden_states, position_embeddings, *, metadata, slots=None, lengths=None, **kwargs):
        if self.cache_only:
            self.write_cache(hidden_states, position_embeddings, metadata, slots=slots, lengths=lengths)
            return None, None
        query, gate, lengths = self.write_cache(hidden_states, position_embeddings, metadata,
                                                slots=slots, lengths=lengths)
        attended = self.attention(query, output_gate=gate if self.attention.gated else None,
                                  page_table=metadata.page_table, cu_seqlens_q=metadata.query_start_loc,
                                  cache_seqlens=lengths)
        if not self.attention.gated:
            if self.output_gate is None:
                attended.mul_(gate.reshape_as(attended).sigmoid())
            else:
                values = (attended.view(self.tokens, self.heads * self.dim, 1) if self.packed
                          else attended.view(self.tokens * self.heads, self.dim, 1))
                gates = (gate.view(self.tokens, self.heads * self.dim) if self.packed
                         else gate.view(self.tokens * self.heads, self.dim))
                self.output_gate(values, gates)
        return self.projections["o_proj"](attended.flatten(1)), None


class PreparedDecoder:
    def __init__(self, weights, scratch, metadata, pool, *, mode, vocabulary_runtime,
                 coordinator, session, name, workspace, scoring=None, records=None,
                 vision_state=None, sample=True):
        if not sample and (mode != "prefill" or scoring is not None):
            raise ValueError("non-sampling decoders require unscored prefill mode")
        self.sample = bool(sample)
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        self.vision_state = vision_state if mode == "prefill" else None
        self.embedding = PreparedEmbedding(weights.tree.embed_tokens.weight, metadata.token_ids,
            scratch.activations, num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding").functional(session=session, dynamic_inputs=True)
        self.delta = DeltaMetadata.allocate(metadata.starts.numel(), scratch.activations.device,
            verification_columns=metadata.max_query_tokens if mode == "verify" else 1)
        config, source = weights.config, scratch.activations
        def zeros(shape, dtype=None):
            dtype = source.dtype if dtype is None else dtype
            return torch.zeros(shape, dtype=dtype, device=source.device)
        query_dtype = _attention_query_dtype(source, metadata, mode, config.text["head_dim"])
        attention_buffers = {field: zeros((source.shape[0], width), query_dtype if field == "q" else None)
                             for field, width in (
            ("q", config.text["num_attention_heads"] * config.text["head_dim"]),
            ("attended", config.text["num_attention_heads"] * config.text["head_dim"]),
            ("output", config.hidden))}
        attention_prefixes = tuple(f"layers.{i}.self_attn." for i, kind in enumerate(config.layers)
                                   if kind == "full_attention")
        packed = tuple(prefix + "qgkv_proj" in weights.projections for prefix in attention_prefixes)
        q_width = config.text["num_attention_heads"] * config.text["head_dim"]
        kv_width = config.text["num_key_value_heads"] * config.text["head_dim"]
        if any(packed):
            attention_buffers["qgkv"] = zeros((source.shape[0], 2 * q_width + 2 * kv_width))
        if not all(packed):
            attention_buffers["qg"] = zeros((source.shape[0], 2 * q_width))
        separate = tuple(prefix for prefix, is_packed in zip(attention_prefixes, packed, strict=True) if not is_packed)
        if any(prefix + "kv_proj" in weights.projections for prefix in separate):
            attention_buffers["kv"] = zeros((source.shape[0], 2 * kv_width))
        if any(prefix + "kv_proj" not in weights.projections for prefix in separate):
            for field in ("k", "v"):
                attention_buffers[field] = zeros((source.shape[0], kv_width))
        intermediate = config.text["intermediate_size"]
        def mlp_candidate(width):
            return torch.empty((source.shape[0], width), dtype=torch.bfloat16, device=source.device)
        mlp_buffers = {"down_proj": mlp_candidate(config.hidden)}
        grouped = tuple(f"layers.{i}.mlp.gate_up" in weights.projections for i in range(len(weights.tree.layers)))
        if any(grouped):
            mlp_buffers["gate_up"] = mlp_candidate(2 * intermediate)
        if not all(grouped):
            for field in ("gate_proj", "up_proj"):
                mlp_buffers[field] = mlp_candidate(intermediate)
        mlp_input = _allocate_mlp_intermediate(weights, source)
        normalized = mlp_candidate(config.hidden)
        taps = () if weights.draft is None else tuple(config.draft.target_layers)
        last_index = len(weights.tree.layers) - 1
        # A non-terminal chunk never samples or saves last_hidden. When the final layer
        # is a full-attention layer outside the DFlash2 tap set, only its KV cache update
        # persists; its attention read, o_proj, post-attention norm, and MLP are unused.
        cache_only_last = (
            not self.sample
            and last_index >= 0
            and weights.tree.layers[last_index].block_type == "full_attention"
            and last_index not in taps
        )
        self.layers = []
        previous_attended = None
        for index, original in enumerate(weights.tree.layers):
            layer = module_copy(original)
            prefix = f"layers.{index}."
            is_cache_only = cache_only_last and index == last_index
            layer.input_layernorm = RMSNorm(original.input_layernorm,
                mlp_buffers["down_proj"] if index else source, normalized,
                session=session, name=name + "." + prefix + "input_layernorm",
                residual=previous_attended if index else None)
            if original.block_type == "linear_attention":
                inputs = (("in_proj_qkvzba",) if prefix + "linear_attn.in_proj_qkvzba" in weights.projections
                          else ("in_proj_qkvz", "in_proj_ba") if prefix + "linear_attn.in_proj_qkvz" in weights.projections
                          else ("in_proj_qkv", "in_proj_z", "in_proj_ba"))
                projections = {field: weights.projections[prefix + "linear_attn." + field]
                               for field in (*inputs, "out_proj")}
                layer.linear_attn = RecurrentAttention(PreparedGDN(original.linear_attn, projections, source,
                    scratch.gdn, pool.recurrent(index), self.delta,
                    mode="decode" if mode == "verify" else mode, session=session,
                    records=records[index] if mode == "verify" else None,
                    name=name + "." + prefix + "gdn", workspace=workspace, projection_workspace=workspace,
                    projection_staging=scratch.quantized,
                    gdn_buffers=None).functional(session=session, dynamic_metadata=True))
            else:
                layer.self_attn = FullAttention(original.self_attn, weights, source, metadata, pool, index,
                    mode=mode, session=session, workspace=workspace, name=name + "." + prefix + "attention",
                    buffers=attention_buffers, projection_staging=scratch.quantized,
                    cache_only=is_cache_only)
            if is_cache_only:
                self.layers.append(ResidualNormalizedLayer(layer, cache_only=True))
                continue
            attended = (scratch.gdn.output if original.block_type == "linear_attention"
                        else attention_buffers["output"])
            layer.post_attention_layernorm = RMSNorm(original.post_attention_layernorm,
                attended, normalized, session=session,
                name=name + "." + prefix + "post_attention_layernorm",
                residual=source)
            if grouped[index]:
                layer.mlp = GroupedMLP(original.mlp, source,
                    weights.projections[prefix + "mlp.gate_up"], weights.projections[prefix + "mlp.down_proj"],
                    mlp_buffers["gate_up"], mlp_input, mlp_buffers["down_proj"], session=session,
                    name=name + "." + prefix + "mlp", workspace=workspace)
            else:
                layer.mlp = module_copy(original.mlp)
                for field in ("gate_proj", "up_proj", "down_proj"):
                    weight = weights.projections[prefix + "mlp." + field]
                    inputs = source if field != "down_proj" else mlp_input
                    output = mlp_buffers[field]
                    setattr(layer.mlp, field, Projection(prepare_projection(inputs, weight, output,
                        activation_mode="auto", session=session, workspace=workspace,
                        name=name + "." + prefix + field).functional(session=session)))
            self.layers.append(ResidualNormalizedLayer(layer))
            previous_attended = attended
        for left, right in zip(self.layers, self.layers[1:]):
            left.next_input_layernorm = right.layer.input_layernorm
        rotary = config.text["rope_parameters"]
        dim = int(config.text["head_dim"] * rotary["partial_rotary_factor"])
        from ignition.preparation.persistent import persistent_copy, persistent_empty

        self.inverse_frequency = persistent_copy(1.0 / (rotary["rope_theta"] ** (
            torch.arange(0, dim, 2, device=source.device, dtype=torch.float32) / dim)))
        # Per-axis positions exist only for a pool that declares a rotary offset region;
        # a text-only preparation rotates at token positions and holds no axis table.
        self.rope_axes = self.position_axes = None
        if rotary_offset_region(pool) is not None:
            self.rope_axes = persistent_copy(rotary_axes(rotary, dim // 2).to(source.device))
            self.position_axes = len(rotary_sections(rotary))
        self.scoring = None if scoring is None else scoring.prepare(scratch.activations, weights.head,
            vocabulary=weights.config.text["vocab_size"], session=session, coordinator=coordinator,
            name=name + ".prompt")
        self.head = (
            PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.logits, scratch.logits,
                runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head",
                activation_mode="auto", workspace=workspace).functional(session=session)
            if self.sample else None
        )
        self.context_capture = (PreparedContextCapture(metadata,
            config.hidden * len(config.draft.target_layers), config.draft.attention.window,
            compact_prefill=mode == "prefill").functional() if weights.draft is not None else None)
        self.auxiliary = (None if self.context_capture is None else torch.empty(
            self.context_capture.shape, dtype=torch.bfloat16, device=source.device))
        self.replay = None
        self.record_layers = ()
        if mode == "verify":
            self.record_layers = records.layers
            self.record_rows, self.record_width = records.delta.shape[1], records.delta.shape[3]
            counts = torch.zeros(metadata.starts.numel(), dtype=torch.int32, device=source.device)
            states = {i: pool.region(f"layer{i}.recurrent") for i in records.layers}
            self.replay = PreparedGDNCommit(pool.backing, states, records, self.delta, counts,
                key_heads=16, value_heads=48).functional(session=session, dynamic_metadata=True)
            regions = [pool.layout.region(f"layer{i}.convolution") for i in records.layers]
            self.convolution_offsets = persistent_empty((len(regions),), torch.int64, source.device)
            self.convolution_offsets.copy_(torch.tensor([r.offset // 2 for r in regions], dtype=torch.int64))
            self.convolution_shape = regions[0].shape
        self.verification_columns = metadata.max_query_tokens if mode == "verify" else 1
        self.metadata = self.delta = None

    def release_outputs(self):
        self.scratch = self.auxiliary = None

    def _delta_metadata(self, metadata):
        sequences = metadata.starts.numel()
        active = metadata.lengths > 0
        state_indices = torch.zeros((sequences, self.verification_columns), dtype=torch.int64,
                                    device=metadata.starts.device)
        state_indices[:, 0].copy_(metadata.state_rows)
        return DeltaMetadata(
            metadata.query_start_loc,
            torch.where(active, metadata.state_rows, -1),
            metadata.starts == 0,
            torch.ones((sequences,), dtype=torch.int32, device=metadata.starts.device),
            state_indices,
            torch.where(metadata.starts > 0, metadata.state_rows, 0),
            metadata.state_rows,
            torch.zeros((sequences,), dtype=torch.int64, device=metadata.starts.device),
            torch.zeros((sequences,), dtype=torch.int32, device=metadata.starts.device),
            metadata.live_counts[1:], metadata.live_counts[:1],
        )

    def commit(self, counts, records, metadata=None):
        if metadata is None:
            raise ValueError("dense decoder commit requires metadata")
        counts = counts.to(torch.int32)
        self.replay(counts, records=records, metadata=self._delta_metadata(metadata))
        channels, columns = self.convolution_shape
        torch.library.wrap_triton(_commit_convolution)[
            (self.convolution_offsets.numel(), counts.numel(), triton.cdiv(channels, 128))](
            self.pool.backing.view(torch.bfloat16), self.convolution_offsets, metadata.state_rows, counts,
            self.pool.layout.row_bytes // 2, channels, columns, 128)
        return counts

    def __call__(self, metadata=None):
        if metadata is None:
            raise ValueError("dense decoder requires metadata")
        m, d = metadata, self._delta_metadata(metadata)
        position_embeddings = rotary_embeddings(rotary_positions(m), self.inverse_frequency,
            self.rope_axes, self.position_axes)
        page = m.token_page_table.gather(1, (m.positions.clamp(min=0) // PAGE_TOKENS)[:, None]).squeeze(1)
        slots = torch.where(m.positions >= 0, page.long() * PAGE_TOKENS + m.positions % PAGE_TOKENS, -1)
        lengths = torch.where(m.lengths > 0, m.starts + m.lengths, 0).to(m.lengths.dtype)
        hidden = self.embedding(token_ids=m.token_ids, num_rows=m.live_counts[:1])
        if self.vision_state is not None:
            self.vision_state.inject(m, hidden)
        records = (GDNLayerRecords.allocate(self.record_layers, self.record_rows, hidden.device,
            key_heads=16, value_heads=48, width=self.record_width) if self.record_layers else None)
        if self.context_capture is not None:
            capture = self.context_capture.begin(m)
        taps = () if self.context_capture is None else self.weights.config.draft.target_layers
        normalized = self.layers[0].layer.input_layernorm(hidden)
        for index, layer in enumerate(self.layers):
            layer_records = None if records is None or index not in records.layers else records[index]
            hidden, normalized = layer(hidden, normalized, position_embeddings=position_embeddings, records=layer_records,
                                       metadata=m, delta=d, slots=slots, lengths=lengths)
            if index in taps:
                column = taps.index(index) * hidden.shape[-1]
                self.context_capture(capture, hidden, column)
        if not self.sample:
            return None, None, None if self.context_capture is None else capture[0], records
        if self.scoring is not None:
            normalized_hidden = self.weights.tree.norm(hidden)
            self.scoring(normalized_hidden)
            sampled_hidden = normalized_hidden[m.sample_indices]
        else:
            sampled_hidden = self.weights.tree.norm(hidden[m.sample_indices])
        return sampled_hidden, self.head(sampled_hidden), None if self.context_capture is None else capture[0], records
