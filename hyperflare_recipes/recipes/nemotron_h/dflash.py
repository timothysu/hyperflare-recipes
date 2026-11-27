# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""LiLiCorr context injection and bidirectional draft blocks."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from hyperflare_recipes.components.embedding import PreparedEmbedding
from hyperflare_recipes.components.grouped_convolution import PreparedGroupedConvolution
from hyperflare_recipes.components.linear import QuantizedProjection, UnquantizedProjection
from hyperflare_recipes.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from hyperflare_recipes.components.projection_overlap import ConcurrentProjectionChain
from hyperflare_recipes.components.rms_norm import PreparedRMSNorm
from hyperflare_recipes.components.speculation import SpeculativeRngState
from ignition.components.collectives import PreparedAllReduce
from ignition.components.state import save_hidden
from hyperflare_recipes.components.vocabulary import PreparedVocabularyProjection
from hyperflare_recipes.components.window_attention import stage_windows
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.examples import declaration_tensor

from .config import PAGE_TOKENS
from .decoder import PreparedDecoder, Scratch
from .lilicorr import PreparedLatticeSelector


_DEQUANT_FC_MIN_TOKENS = 1024
_FP4_E2M1_TABLE = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


@triton.jit
def _dequant_nvfp4_swizzled_kernel(
    val_ptr, scale_ptr, gs_ptr, out_ptr, lut_ptr,
    N, K,
    BLOCK_N: tl.constexpr, BLOCK_K_B: tl.constexpr,
):
    pid_n, pid_k = tl.program_id(0), tl.program_id(1)
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    cols_b = pid_k * BLOCK_K_B + tl.arange(0, BLOCK_K_B)
    mask = (rows[:, None] < N) & (cols_b[None, :] < (K // 2))
    packed = tl.load(val_ptr + rows[:, None] * (K // 2) + cols_b[None, :], mask=mask, other=0)
    lo = tl.load(lut_ptr + (packed & 0x0F).to(tl.int32))
    hi = tl.load(lut_ptr + ((packed >> 4) & 0x0F).to(tl.int32))
    c = cols_b // 8
    sc_off = (
        (((rows[:, None] // 128) * (K // 64) + (c[None, :] // 4)) * 32 + (rows[:, None] % 32)) * 4
        + ((rows[:, None] % 128) // 32)
    ) * 4 + (c[None, :] % 4)
    sc = tl.load(scale_ptr + sc_off, mask=mask, other=0.0).to(tl.float32)
    factor = sc * tl.load(gs_ptr)
    out = tl.interleave((lo * factor).to(tl.bfloat16), (hi * factor).to(tl.bfloat16))
    out_cols = pid_k * (BLOCK_K_B * 2) + tl.arange(0, BLOCK_K_B * 2)
    out_mask = (rows[:, None] < N) & (out_cols[None, :] < K)
    tl.store(out_ptr + rows[:, None] * K + out_cols[None, :], out, mask=out_mask)


@torch.library.custom_op("flare::dequant_nvfp4_swizzled", mutates_args=("out",))
def _dequant_nvfp4_swizzled(
    values: torch.Tensor, scale_mma: torch.Tensor, global_scale: torch.Tensor,
    lut: torch.Tensor, out: torch.Tensor,
) -> None:
    n, half_k = values.shape
    k = half_k * 2
    scale_flat = scale_mma.permute(2, 4, 0, 1, 3, 5).reshape(-1)
    grid = (triton.cdiv(n, 16), triton.cdiv(half_k, 128))
    _dequant_nvfp4_swizzled_kernel[grid](
        values, scale_flat, global_scale, out, lut, n, k,
        BLOCK_N=16, BLOCK_K_B=128, num_warps=4,
    )


@_dequant_nvfp4_swizzled.register_fake
def _fake_dequant_nvfp4_swizzled(values, scale_mma, global_scale, lut, out):
    pass


@triton.jit
def _stage_draft_kv_kernel(
    hk_ptr, hv_ptr, ck_ptr, cv_ptr, ok_ptr, ov_ptr,
    starts_ptr, rows_ptr, qsl_ptr, lens_ptr,
    CAND_ROWS, H_ROW_STRIDE: tl.constexpr, WIN: tl.constexpr, CAP: tl.constexpr, WIDTH: tl.constexpr, BLOCK_T: tl.constexpr,
):
    r = tl.program_id(0)
    t0 = tl.program_id(1) * BLOCK_T
    length = tl.load(lens_ptr + r)
    start = tl.load(starts_ptr + r)
    prefix = tl.minimum(start, WIN)
    total = tl.where(length > 0, prefix + length, 0)
    # Pad active prefix+candidates to 32-row TMA stage tiles (covering stage_rows 16 and 32)
    # so paged attention never reads uninitialized FP8 NaN bytes from scratch memory.
    total_pad = tl.minimum(((total + 31) // 32) * 32, CAP)
    if t0 >= total_pad:
        return
    state = tl.load(rows_ptr + r).to(tl.int64)
    q0 = tl.load(qsl_ptr + r).to(tl.int64)
    first = tl.maximum(start - WIN, 0)
    ts = t0 + tl.arange(0, BLOCK_T)
    ws = tl.arange(0, WIDTH)
    valid = (ts < total)[:, None]
    is_hist = (ts < prefix)[:, None]
    in_pad = (ts < total_pad)[:, None]
    ring = (first + ts) % WIN
    h_off = state * tl.full((), H_ROW_STRIDE, tl.int64) + ring[:, None].to(tl.int64) * WIDTH + ws[None, :].to(tl.int64)
    c_idx = tl.maximum(tl.minimum(q0 + ts - prefix, CAND_ROWS - 1), 0)
    c_off = c_idx[:, None] * WIDTH + ws[None, :]
    hk = tl.load(hk_ptr + h_off, mask=valid & is_hist, other=0)
    hv = tl.load(hv_ptr + h_off, mask=valid & is_hist, other=0)
    ck = tl.minimum(tl.maximum(tl.load(ck_ptr + c_off, mask=valid & (~is_hist), other=0.0).to(tl.float32), -448.0), 448.0)
    cv = tl.minimum(tl.maximum(tl.load(cv_ptr + c_off, mask=valid & (~is_hist), other=0.0).to(tl.float32), -448.0), 448.0)
    ck_u8 = ck.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    cv_u8 = cv.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    ok = tl.where(valid, tl.where(is_hist, hk, ck_u8), 0)
    ov = tl.where(valid, tl.where(is_hist, hv, cv_u8), 0)
    o_off = (r.to(tl.int64) * CAP + ts)[:, None] * WIDTH + ws[None, :]
    tl.store(ok_ptr + o_off, ok, mask=in_pad)
    tl.store(ov_ptr + o_off, ov, mask=in_pad)


@torch.library.custom_op("flare::stage_draft_kv", mutates_args=("out_k", "out_v"))
def _stage_draft_kv(
    history_k: torch.Tensor, history_v: torch.Tensor,
    cand_k: torch.Tensor, cand_v: torch.Tensor,
    out_k: torch.Tensor, out_v: torch.Tensor,
    starts: torch.Tensor, state_rows: torch.Tensor,
    query_start_loc: torch.Tensor, lengths: torch.Tensor,
    window: int, capacity: int, width: int,
) -> None:
    if history_k.stride() != history_v.stride() or history_k.stride(1) != width or history_k.stride(-1) != 1:
        raise ValueError("draft KV history requires matching row-strided contiguous token layouts")
    rows = starts.numel()
    grid = (rows, triton.cdiv(capacity, 16))
    _stage_draft_kv_kernel[grid](
        history_k.view(torch.uint8), history_v.view(torch.uint8),
        cand_k, cand_v,
        out_k.view(torch.uint8), out_v.view(torch.uint8),
        starts, state_rows, query_start_loc, lengths,
        cand_k.shape[0], H_ROW_STRIDE=history_k.stride(0), WIN=window, CAP=capacity, WIDTH=width,
        BLOCK_T=16, num_warps=4,
    )


@_stage_draft_kv.register_fake
def _fake_stage_draft_kv(history_k, history_v, cand_k, cand_v, out_k, out_v,
                         starts, state_rows, query_start_loc, lengths, window, capacity, width):
    pass


def _norm(source, weight, epsilon):
    values = source.float()
    return (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon) * weight.float()).to(source.dtype)


class Rotary:
    def __init__(self, config, device):
        c = config.text
        self.inverse = persistent_copy(1 / c["rope_theta"] ** (torch.arange(0, c["head_dim"], 2,
            dtype=torch.float32, device=device) / c["head_dim"]))

    def __call__(self, values, positions):
        angles = positions.clamp_min(0).float()[:, None] * self.inverse[None]
        cosine, sine = angles.cos().to(values.dtype).float()[:, None], angles.sin().to(values.dtype).float()[:, None]
        left, right = values.float().chunk(2, -1)
        return torch.cat((left * cosine - right * sine, right * cosine + left * sine), -1).to(values.dtype)


class PreparedContext:
    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace, examples=None):
        self.session = session
        self.anchor = pool.reserved_region("draft_anchor")
        c, draft = weights.config.draft, weights.draft
        tokens, hidden = auxiliary.shape[0], weights.config.hidden
        self.window, self.epsilon = c.text["sliding_window"], c.text["rms_norm_eps"]
        projected = declaration_tensor(examples, "nemotron.context.projected", (tokens, hidden), torch.bfloat16, auxiliary.device)
        normalized = declaration_tensor(examples, "nemotron.context.normalized", (tokens, hidden), torch.bfloat16, auxiliary.device)
        from b12x.gemm.blockscaled._a16 import NVFP4LinearWeight

        proj_w = draft.projection
        self.dequant_fc = (
            tokens >= _DEQUANT_FC_MIN_TOKENS
            and isinstance(proj_w, NVFP4LinearWeight)
            and proj_w.activation_global_scale is None
            and proj_w.global_scale_kind == "multiplier"
            and proj_w.in_features == proj_w.padded_in_features
            and proj_w.out_features % 128 == 0
            and proj_w.in_features % 64 == 0
            and proj_w.scale_mma.ndim == 6
            and proj_w.scale_mma.permute(2, 4, 0, 1, 3, 5).is_contiguous()
        )
        if self.dequant_fc:
            self.project = None
            self.fc_weight = proj_w
            self.fp4_lut = persistent_copy(torch.tensor(_FP4_E2M1_TABLE, dtype=torch.float32, device=auxiliary.device))
            for tensor in (proj_w.values, proj_w.scale_mma, proj_w.global_scale, self.fp4_lut):
                record_numeric_tensor(tensor)
        else:
            self.fc_weight = self.fp4_lut = None
            self.project = QuantizedProjection(auxiliary, draft.projection, projected,
                session=session, name=name + ".project", workspace=workspace, activation_mode="auto",
                examples=examples, example_role="nemotron.context.project")
        self.norm = PreparedRMSNorm(projected, draft.hidden_norm, normalized,
            eps=self.epsilon, session=session, name=name + ".norm")
        kv = declaration_tensor(examples, "nemotron.context.kv", (tokens, draft.context_kv.shape[0]), torch.bfloat16, auxiliary.device)
        self.project_kv = UnquantizedProjection(normalized, draft.context_kv, kv,
            session=session, name=name + ".kv")
        shape = (tokens, c.local_kv_heads(getattr(weights, "shard", None)), c.text["head_dim"])
        self.key_shape = shape
        key, value = (declaration_tensor(examples, "nemotron.context." + kind, shape, torch.bfloat16, auxiliary.device)
                      for kind in ("key", "value"))
        self.rotary = Rotary(c, auxiliary.device)
        self.caches = []
        self.stages = []
        for index, layer in enumerate(draft.layers):
            key_cache, value_cache = pool.reserved_region(f"draft{index}.key"), pool.reserved_region(f"draft{index}.value")
            writer = PreparedPagedKVWriter(key, value, key_cache, value_cache,
                torch.empty((tokens,), dtype=torch.int64, device=auxiliary.device),
                session=session, name=f"{name}.{index}.write", caller_owned_placeholders=True)
            self.caches.append((key_cache, value_cache))
            self.stages.append((layer["k_norm"], writer))

    def functional(self):
        if self.project is not None:
            self.project.functional(session=self.session)
        self.norm = self.norm.functional()
        self.project_kv.functional(session=self.session)
        self.stages = tuple((weight, writer.functional(k_cache=None, v_cache=None))
                            for weight, writer in self.stages)
        return self

    def __call__(self, auxiliary, counts, metadata):
        m = metadata
        offset, count = m.positions - m.starts[m.requests], counts[m.requests]
        live = (m.positions >= 0) & (offset < count) & (offset >= (count - self.window).clamp_min(0))
        slots = torch.where(live, m.state_rows[m.requests] * self.window + m.positions % self.window, -1)
        if self.dequant_fc:
            w = self.fc_weight
            bf16_w = torch.empty((w.out_features, w.in_features), dtype=torch.bfloat16, device=auxiliary.device)
            _dequant_nvfp4_swizzled(w.values, w.scale_mma, w.global_scale, self.fp4_lut, bf16_w)
            projected = torch.mm(auxiliary, bf16_w.t())
        else:
            projected = self.project(auxiliary)
        normalized = self.norm(projected)
        kv = self.project_kv(normalized)
        _, groups, dim = self.key_shape
        chunks = kv.view(kv.shape[0], len(self.stages), 2, groups, dim)
        for index, (norm, writer) in enumerate(self.stages):
            key = self.rotary(_norm(chunks[:, index, 0], norm, self.epsilon), m.positions)
            value = chunks[:, index, 1].contiguous()
            key_cache, value_cache = self.caches[index]
            writer(key, value, slots, k_cache=key_cache, v_cache=value_cache)
        last = (m.query_start_loc[:-1] + counts - 1).long().clamp(0, normalized.shape[0] - 1)
        save_hidden(normalized[last], self.anchor, m.state_rows, counts > 0)
        return normalized


class DraftAttention:
    def __init__(self, weights, source, metadata, pool, config, index, *, session, name, workspace,
                 shard=None, runtime=None, coordinator=None, examples=None):
        self.weights = weights
        c, tokens = config.text, source.shape[0]
        self.window, self.epsilon = c["sliding_window"], c["rms_norm_eps"]
        qh, kh, dim = config.local_attention_heads(shard), config.local_kv_heads(shard), c["head_dim"]
        projected = declaration_tensor(examples, "nemotron.draft.attention.projected",
            (tokens, (qh + 2 * kh) * dim), torch.bfloat16, source.device)
        self.project = UnquantizedProjection(source, weights["qkv"], projected, session=session, name=name + ".qkv")
        output = declaration_tensor(examples, "nemotron.draft.attention.output", source.shape, torch.bfloat16, source.device)
        partial = (declaration_tensor(examples, "nemotron.draft.attention.partial", source.shape, torch.bfloat16, source.device)
                   if runtime is not None else output)
        self.output = UnquantizedProjection(
            declaration_tensor(examples, "nemotron.draft.attention.attended_input", (tokens, qh * dim), torch.bfloat16, source.device),
            weights["output"], partial, session=session, name=name + ".output")
        self.reduce = None if runtime is None else PreparedAllReduce(
            partial, output, runtime=runtime, coordinator=coordinator, session=session, name=name + ".reduce")
        self.rotary = Rotary(config, source.device)
        self.history = tuple(pool.reserved_region(f"draft{index}.{field}") for field in ("key", "value"))
        rows, pages = metadata.lengths.numel(), (self.window + config.block_size + PAGE_TOKENS - 1) // PAGE_TOKENS
        staged = tuple(declaration_tensor(examples, "nemotron.draft.attention.staged_" + kind,
            (rows * pages, PAGE_TOKENS, kh, dim), history.dtype, source.device)
            for kind, history in zip(("key", "value"), self.history, strict=True))
        query = declaration_tensor(examples, "nemotron.draft.attention.query", (tokens, qh, dim), torch.bfloat16, source.device)
        attended = declaration_tensor(examples, "nemotron.draft.attention.attended", (tokens, qh, dim), torch.bfloat16, source.device)
        self.attention = PreparedPagedAttention(query, *staged,
            torch.arange(rows * pages, device=source.device, dtype=torch.int32).view(rows, pages),
            torch.empty((rows,), dtype=torch.int32, device=source.device),
            metadata.query_start_loc, attended, sink=weights["sink"], mode="verify", width=config.block_size,
            window=self.window - 1, causal=False, session=session, workspace=workspace, name=name + ".attention",
            caller_owned_placeholders=True)
        self.window_pages, self.rows, self.qh, self.kh, self.dim = pages, rows, qh, kh, dim

    def functional(self, *, session):
        self.project.functional(session=session)
        self.output.functional(session=session)
        if self.reduce is not None:
            self.reduce.functional(session=session)
        self.attention = self.attention.functional(session=session, k_cache=None, v_cache=None,
            cache_seqlens=None, page_table=None, cu_seqlens_q=None)
        return self

    def __call__(self, source, metadata):
        m = metadata
        projected = self.project(source)
        query_width, key_width = self.qh * self.dim, self.kh * self.dim
        q, k, v = projected.split((query_width, key_width, key_width), -1)
        query = self.rotary(_norm(q.view(-1, self.qh, self.dim), self.weights["q_norm"], self.epsilon), m.positions)
        key = self.rotary(_norm(k.view(-1, self.kh, self.dim), self.weights["k_norm"], self.epsilon), m.positions)
        value = v.view(-1, self.kh, self.dim)
        staged_key = torch.empty((self.rows * self.window_pages, PAGE_TOKENS, self.kh, self.dim),
                                 dtype=self.history[0].dtype, device=source.device)
        staged_value = torch.empty((self.rows * self.window_pages, PAGE_TOKENS, self.kh, self.dim),
                                   dtype=self.history[1].dtype, device=source.device)
        width = self.kh * self.dim
        if (self.history[0].dtype == torch.float8_e4m3fn
                and self.history[1].dtype == torch.float8_e4m3fn
                and width & (width - 1) == 0):
            _stage_draft_kv(self.history[0], self.history[1], key.contiguous(), value.contiguous(),
                            staged_key, staged_value, m.starts, m.state_rows, m.query_start_loc, m.lengths,
                            self.window, self.window_pages * PAGE_TOKENS, width)
        else:
            stage_windows(self.history[0], key, staged_key, m)
            stage_windows(self.history[1], value, staged_value, m)
        pages = torch.arange(self.rows * self.window_pages, device=source.device, dtype=torch.int32).view(
            self.rows, self.window_pages)
        lengths = torch.where(m.lengths > 0, m.starts.clamp(max=self.window).int() + m.lengths, 0)
        attended = self.attention(query, k_cache=staged_key, v_cache=staged_value,
            page_table=pages, cache_seqlens=lengths, cu_seqlens_q=m.query_start_loc)
        output = self.output(attended.flatten(1))
        return output if self.reduce is None else self.reduce(output)


class DraftMLP:
    def __init__(self, weights, source, *, session, name, workspace, runtime=None, coordinator=None, examples=None):
        gate, up, down = weights
        shape = (source.shape[0], gate.out_features)
        self.gate, self.up = (QuantizedProjection(source, weight,
            declaration_tensor(examples, "nemotron.draft.mlp" + suffix, shape, torch.bfloat16, source.device),
            session=session, name=name + suffix, workspace=workspace, activation_mode="auto",
            examples=examples, example_role="nemotron.draft.mlp" + suffix)
            for weight, suffix in ((gate, ".gate"), (up, ".up")))
        output = declaration_tensor(examples, "nemotron.draft.mlp.output", source.shape, torch.bfloat16, source.device)
        partial = (declaration_tensor(examples, "nemotron.draft.mlp.partial", source.shape, torch.bfloat16, source.device)
                   if runtime is not None else output)
        self.down = QuantizedProjection(declaration_tensor(examples, "nemotron.draft.mlp.intermediate", shape, torch.bfloat16, source.device), down,
            partial, session=session, name=name + ".down", workspace=workspace, activation_mode="auto",
            examples=examples, example_role="nemotron.draft.mlp.down")
        self.reduce = None if runtime is None else PreparedAllReduce(
            partial, output, runtime=runtime, coordinator=coordinator, session=session, name=name + ".reduce")
        self.concurrent_up = (ConcurrentProjectionChain((self.up,), parallel_plans=(self.gate.plan,),
            session=session, workspace=workspace) if workspace.overlap else None)

    def functional(self, *, session):
        if self.concurrent_up is not None:
            self.concurrent_up.functional(session=session)
        else:
            self.up.functional(session=session)
        self.gate.functional(session=session)
        self.down.functional(session=session)
        if self.reduce is not None:
            self.reduce.functional(session=session)
        return self

    def __call__(self, source):
        pending = self.concurrent_up.start(source) if self.concurrent_up is not None else None
        gate = self.gate(source)
        up = self.up(source) if pending is None else self.concurrent_up.finish(pending, gate)
        output = self.down((F.silu(gate.float()) * up.float()).bfloat16())
        return output if self.reduce is None else self.reduce(output)


class PreparedDraft:
    def __init__(self, weights, scratch, metadata, pool, *, vocabulary_runtime, coordinator,
                 session, name, workspace, runtime=None, examples=None):
        self.weights = weights
        c, s = weights.config.draft, scratch
        self.session = session
        self.logit_shape, self.logit_dtype = tuple(s.logits.shape), s.logits.dtype
        self.local_logit_shape, self.local_logit_dtype = tuple(s.local_logits.shape), s.local_logits.dtype
        self.hidden_shape = tuple(s.hidden.shape)
        self._examples = scratch, metadata
        self.embedding = PreparedEmbedding(weights.embedding, metadata.token_ids, s.activations,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.layers, previous = [], None
        for index, w in enumerate(weights.draft.layers):
            prefix = f"{name}.layer{index}"
            norm = PreparedRMSNorm(s.activations, w["input_norm"], s.normalized,
                eps=c.text["rms_norm_eps"], residual=previous, session=session, name=prefix + ".norm")
            convs = []
            for kind in ("attention_conv", "mlp_conv"):
                coefficients = 2 * c.text["dflash_config"]["conv_kernel_size"] * weights.config.hidden // c.text["dflash_config"]["conv_group_size"]
                convs.append(PreparedGroupedConvolution(s.normalized, w[kind], block_size=c.block_size,
                    group_size=c.text["dflash_config"]["conv_group_size"], session=session, name=prefix + "." + kind,
                    buffers={key: declaration_tensor(examples, "nemotron.draft." + kind + "." + key,
                        (s.activations.shape[0], width), torch.bfloat16, s.activations.device)
                        for key, width in (("coefficients", coefficients), ("before", weights.config.hidden), ("after", weights.config.hidden))}))
            attention = DraftAttention(w, s.normalized, metadata, pool, c, index,
                shard=getattr(weights, "shard", None), runtime=runtime, coordinator=coordinator,
                session=session, name=prefix + ".attention", workspace=workspace, examples=examples)
            post = PreparedRMSNorm(s.activations, w["post_norm"], s.normalized,
                eps=c.text["rms_norm_eps"], residual=s.normalized, session=session, name=prefix + ".post_norm")
            mlp = DraftMLP(w["mlp"], s.normalized, runtime=runtime, coordinator=coordinator,
                session=session, name=prefix + ".mlp", workspace=workspace, examples=examples)
            self.layers.append((norm, convs[0], attention, post, convs[1], mlp))
            previous = s.normalized
        self.norm = PreparedRMSNorm(s.activations, weights.draft.norm, s.normalized,
            eps=c.text["rms_norm_eps"], residual=previous, session=session, name=name + ".norm")
        head_weight = weights.draft.head if weights.draft.head is not None else weights.head
        self.head = PreparedVocabularyProjection(s.hidden, head_weight, s.local_logits, s.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head", activation_mode="auto", workspace=workspace)
        self.saved_anchor = pool.reserved_region("draft_anchor")
        self.selector = PreparedLatticeSelector(weights.draft.lattice, c, weights.embedding,
            s.hidden.view(-1, c.block_size, weights.config.hidden), s.logits,
            declaration_tensor(examples, "nemotron.draft.selector.anchor",
                (metadata.lengths.numel(), weights.config.hidden), torch.bfloat16, s.hidden.device),
            session=session, name=name + ".selector", workspace=workspace, examples=examples)

    def release_outputs(self):
        scratch, _ = self._examples
        self.embedding.functional(session=self.session, dynamic_inputs=True)
        self.layers = tuple((norm.functional(), conv.functional(session=self.session),
            attention.functional(session=self.session), post.functional(), mlp_conv.functional(session=self.session),
            mlp.functional(session=self.session))
            for norm, conv, attention, post, mlp_conv, mlp in self.layers)
        self.norm = self.norm.functional()
        self.head.functional(session=self.session)
        self.selector.functional(session=self.session)
        self._examples = None
        return self

    def _forward_with_shortlists(self, metadata):
        m = metadata
        residual, output = self.embedding(token_ids=m.token_ids, num_rows=m.live_counts[:1]), None
        for norm, conv, attention, post, mlp_conv, mlp in self.layers:
            normalized = norm(residual, output)
            before, coefficients = conv.prepare(normalized)
            output = conv.finish(attention(before, m), coefficients)
            normalized = post(residual, output)
            before, coefficients = mlp_conv.prepare(normalized)
            output = mlp_conv.finish(mlp(before), coefficients)
        hidden = self.norm(residual, output)[m.sample_indices]
        logits = self.head(hidden)
        c = self.weights.config.draft
        anchor = self.saved_anchor[m.state_rows]
        selector_features = self.selector.prepare_with_shortlists(
            hidden.view(-1, c.block_size, self.weights.config.hidden),
            logits.view(-1, c.block_size, self.weights.config.vocabulary), anchor,
            (m.starts > 0) & (m.lengths > 0))
        return hidden, logits, selector_features

    def forward(self, metadata):
        hidden, logits, (scores, _, _) = self._forward_with_shortlists(metadata)
        return hidden, logits, scores

    def __call__(self, metadata):
        _, logits, selector_features = self._forward_with_shortlists(metadata)
        return logits, selector_features


@dataclass(frozen=True)
class DraftForwards:
    decoder: object
    context: object
    draft: object
    markov: object = None


class SpeculativePreparation:
    def __init__(self, weights, pool, requests, inputs, *, completion, frame, sequence,
                 session, workspace, coordinator, runtime=None, vocabulary_runtime=None,
                 result_runtime=None, examples=None):
        self.examples = examples
        self.page_tokens = PAGE_TOKENS
        self.weights, self.tables, self.pool = weights, None, pool
        self.requests, self.inputs, self.scratch = requests, inputs, None
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.session, self.workspace = session, workspace
        self.collectives = dict(runtime=runtime, vocabulary_runtime=vocabulary_runtime, coordinator=coordinator)
        self.result_runtime = result_runtime
        self.block_proposal_broadcast = weights.shard.world_size > 1
        self.markov_runtime = None
        self.forwards = DraftForwards(self.decoder, self.context, self.draft)
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.generation, self.offset = torch.zeros_like(requests.generation), torch.zeros_like(requests.generation)
        self.block_size, self.logit_offset = weights.config.draft.block_size, 1
        self.draft_scratch = None

    def begin(self, batch, width, target_token_capacity):
        shard = self.weights.shard
        self.scratch = Scratch.allocate(target_token_capacity, batch * width, self.weights.config,
            self.pool.backing.device, shard=shard, examples=self.examples)
        self.draft_scratch = Scratch.allocate(batch * self.block_size, batch * self.block_size,
            self.weights.config, self.pool.backing.device, shard=shard, examples=self.examples, role="nemotron.draft")

    def release_examples(self):
        self.scratch = self.draft_scratch = None
        self.forwards = self.examples = None

    def decoder(self, weights, tables, scratch, metadata, pool, *, runtime, **kwargs):
        return PreparedDecoder(weights, scratch, metadata, pool, runtime=runtime, examples=self.examples, **kwargs)

    def context(self, *args, **kwargs):
        return PreparedContext(*args, examples=self.examples, **kwargs)

    def draft(self, *args, **kwargs):
        return PreparedDraft(*args, examples=self.examples, **kwargs)
