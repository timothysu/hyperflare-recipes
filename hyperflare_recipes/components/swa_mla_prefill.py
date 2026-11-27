# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Direct-ring tiled-query Triton prefill and ring commit for BF16 sliding-window MLA."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ignition.preparation.triton import warmup_triton

_BLOCK_M = 32
_BLOCK_N = 32
_NUM_WARPS = 4
_NUM_STAGES = 2
_LOG2E = tl.constexpr(1.4426950408889634)
_LN2 = tl.constexpr(0.6931471805599453)


@triton.jit
def _swa_mla_prefill_kernel(
    Q,
    WINDOW_BUF,
    KV_NEW,
    SINK,
    OUT,
    STATE_ROW_PTR,
    START_PTR,
    QSL_PTR,
    LEN_PTR,
    stride_qt,
    stride_qh,
    stride_wr,
    stride_wt,
    stride_kt,
    stride_ot,
    stride_oh,
    stride_sr,
    stride_st,
    stride_qsl,
    stride_len,
    sm_scale,
    TOTAL_TOKENS: tl.constexpr,
    ROWS: tl.constexpr,
    SINGLE_SPAN: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_NOPE: tl.constexpr,
    D_ROPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    q_base = tl.load(QSL_PTR + pid_b * stride_qsl).to(tl.int32)
    if SINGLE_SPAN:
        span_len = tl.maximum(0, TOTAL_TOKENS - q_base)
    else:
        span_end = tl.load(QSL_PTR + (pid_b + 1) * stride_qsl).to(tl.int32)
        span_len = tl.maximum(0, span_end - q_base)

    q_start = pid_m * BLOCK_M
    if q_start >= span_len:
        return

    state_row = tl.load(STATE_ROW_PTR + pid_b * stride_sr).to(tl.int64)
    start_pos = tl.load(START_PTR + pid_b * stride_st).to(tl.int64)
    raw_len = tl.load(LEN_PTR + pid_b * stride_len).to(tl.int32)
    seq_len = tl.where(state_row > 0, tl.minimum(tl.maximum(raw_len, 0), span_len), 0)
    min_valid_q = tl.minimum(seq_len, tl.maximum(0, -start_pos).to(tl.int32))

    arange_m = tl.arange(0, BLOCK_M)
    offs_m = q_start + arange_m
    offs_d_nope = tl.arange(0, D_NOPE)
    if (q_start >= seq_len) | ((q_start + BLOCK_M) <= min_valid_q):
        out_rows_z = q_base + offs_m
        out_mask_z = (offs_m < span_len) & (out_rows_z >= 0) & (out_rows_z < TOTAL_TOKENS)
        out_tile_z = OUT + (q_base + q_start).to(tl.int64) * stride_ot + pid_h * stride_oh
        tl.store(
            out_tile_z + arange_m[:, None] * stride_ot + offs_d_nope[None, :],
            tl.zeros((BLOCK_M, D_NOPE), dtype=tl.bfloat16),
            mask=out_mask_z[:, None],
        )
        return

    hist_len = tl.minimum(tl.maximum(start_pos, 0), WINDOW - 1).to(tl.int32)
    hist_start_mod = (tl.maximum(0, start_pos - hist_len.to(tl.int64)) % WINDOW).to(tl.int32)
    offs_d_rope = tl.arange(0, D_ROPE)
    q_mask = (offs_m >= min_valid_q) & (offs_m < seq_len)

    q_tile_ptr = Q + (q_base + q_start).to(tl.int64) * stride_qt + pid_h * stride_qh
    q_ptrs = q_tile_ptr + arange_m[:, None] * stride_qt
    q_nope = tl.load(q_ptrs + offs_d_nope[None, :], mask=q_mask[:, None], other=0.0)
    q_rope = tl.load(q_ptrs + (D_NOPE + offs_d_rope[None, :]), mask=q_mask[:, None], other=0.0)

    qk_scale = sm_scale * _LOG2E
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D_NOPE), dtype=tl.float32)

    # Loop 1: Attend directly to prior committed circular-ring entries [k_lo, ring_end).
    # Later queries within a boundary tile can mask out every ring entry, leaving m_i == -inf.
    k_lo = tl.maximum(0, hist_len + q_start - WINDOW + 1)
    k_hi = hist_len + tl.minimum(seq_len, q_start + BLOCK_M)
    ring_end = tl.minimum(k_hi, hist_len)
    q_comb = hist_len + offs_m
    w_row_base = WINDOW_BUF + state_row * stride_wr
    arange_n = tl.arange(0, BLOCK_N)
    for k_start in range(k_lo, ring_end, BLOCK_N):
        offs_n = k_start + arange_n
        k_mask = offs_n < ring_end
        ring_i = (hist_start_mod + offs_n) % WINDOW
        w_ptrs = w_row_base + ring_i[:, None] * stride_wt
        k_nope = tl.load(w_ptrs + offs_d_nope[None, :], mask=k_mask[:, None], other=0.0)
        k_rope = tl.load(w_ptrs + (D_NOPE + offs_d_rope[None, :]), mask=k_mask[:, None], other=0.0)
        qk = tl.dot(q_nope, tl.trans(k_nope))
        qk = tl.dot(q_rope, tl.trans(k_rope), qk) * qk_scale
        valid = (
            (offs_n[None, :] <= q_comb[:, None])
            & (offs_n[None, :] > (q_comb[:, None] - WINDOW))
            & q_mask[:, None]
            & k_mask[None, :]
        )
        qk = tl.where(valid, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        safe_m = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        p = tl.where(valid, tl.math.exp2(qk - safe_m[:, None]), 0.0)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.math.exp2(m_i - safe_m))
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), k_nope, acc)
        m_i = m_ij

    # Loop 2: Attend to packed chunk KV records [new_lo, new_hi), split into
    # left window-boundary, unmasked interior, and right causal-boundary tiles.
    new_lo = tl.maximum(min_valid_q, q_start - WINDOW + 1)
    new_hi = tl.minimum(seq_len, q_start + BLOCK_M)
    interior_lo = tl.maximum(new_lo, q_start + BLOCK_M - WINDOW)
    n_left = tl.cdiv(tl.maximum(0, interior_lo - new_lo), BLOCK_N) * BLOCK_N
    mid_lo = tl.minimum(new_hi, new_lo + n_left)
    mid_hi = tl.maximum(mid_lo, tl.minimum(new_hi, q_start))
    mid_hi = mid_lo + ((mid_hi - mid_lo) // BLOCK_N) * BLOCK_N

    for n_start in range(new_lo, mid_lo, BLOCK_N):
        offs_n = n_start + arange_n
        k_mask = offs_n < new_hi
        kv_tile_ptr = KV_NEW + (q_base + n_start).to(tl.int64) * stride_kt
        kv_ptrs = kv_tile_ptr + arange_n[:, None] * stride_kt
        k_nope = tl.load(kv_ptrs + offs_d_nope[None, :], mask=k_mask[:, None], other=0.0)
        k_rope = tl.load(kv_ptrs + (D_NOPE + offs_d_rope[None, :]), mask=k_mask[:, None], other=0.0)
        qk = tl.dot(q_nope, tl.trans(k_nope))
        qk = tl.dot(q_rope, tl.trans(k_rope), qk) * qk_scale
        valid = (
            (offs_n[None, :] <= offs_m[:, None])
            & (offs_n[None, :] > (offs_m[:, None] - WINDOW))
            & q_mask[:, None]
            & k_mask[None, :]
        )
        qk = tl.where(valid, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        safe_m = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        p = tl.where(valid, tl.math.exp2(qk - safe_m[:, None]), 0.0)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.math.exp2(m_i - safe_m))
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), k_nope, acc)
        m_i = m_ij

    for n_start in range(mid_lo, mid_hi, BLOCK_N):
        kv_tile_ptr = KV_NEW + (q_base + n_start).to(tl.int64) * stride_kt
        kv_ptrs = kv_tile_ptr + arange_n[:, None] * stride_kt
        k_nope = tl.load(kv_ptrs + offs_d_nope[None, :])
        k_rope = tl.load(kv_ptrs + (D_NOPE + offs_d_rope[None, :]))
        qk = tl.dot(q_nope, tl.trans(k_nope))
        qk = tl.dot(q_rope, tl.trans(k_rope), qk) * qk_scale
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.math.exp2(m_i - m_ij))
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), k_nope, acc)
        m_i = m_ij

    for n_start in range(mid_hi, new_hi, BLOCK_N):
        offs_n = n_start + arange_n
        k_mask = offs_n < new_hi
        kv_tile_ptr = KV_NEW + (q_base + n_start).to(tl.int64) * stride_kt
        kv_ptrs = kv_tile_ptr + arange_n[:, None] * stride_kt
        k_nope = tl.load(kv_ptrs + offs_d_nope[None, :], mask=k_mask[:, None], other=0.0)
        k_rope = tl.load(kv_ptrs + (D_NOPE + offs_d_rope[None, :]), mask=k_mask[:, None], other=0.0)
        qk = tl.dot(q_nope, tl.trans(k_nope))
        qk = tl.dot(q_rope, tl.trans(k_rope), qk) * qk_scale
        valid = (
            (offs_n[None, :] <= offs_m[:, None])
            & (offs_n[None, :] > (offs_m[:, None] - WINDOW))
            & q_mask[:, None]
            & k_mask[None, :]
        )
        qk = tl.where(valid, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        safe_m = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        p = tl.where(valid, tl.math.exp2(qk - safe_m[:, None]), 0.0)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.math.exp2(m_i - safe_m))
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), k_nope, acc)
        m_i = m_ij

    # Round normalized output to BF16 before applying the FP32 natural-log attention sink.
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    att_bf16 = (acc / safe_l[:, None]).to(tl.bfloat16)
    lse_nat = tl.where(l_i > 0.0, m_i * _LN2 + tl.math.log(safe_l), float("-inf"))
    sink_val = tl.load(SINK + pid_h).to(tl.float32)
    sink_scale = tl.sigmoid(lse_nat - sink_val)
    out = tl.where(
        q_mask[:, None],
        (att_bf16.to(tl.float32) * sink_scale[:, None]).to(tl.bfloat16),
        0.0,
    )
    out_rows = q_base + offs_m
    out_mask = (offs_m < span_len) & (out_rows >= 0) & (out_rows < TOTAL_TOKENS)
    out_tile_ptr = OUT + (q_base + q_start).to(tl.int64) * stride_ot + pid_h * stride_oh
    tl.store(
        out_tile_ptr + arange_m[:, None] * stride_ot + offs_d_nope[None, :],
        out,
        mask=out_mask[:, None],
    )


@triton.jit
def _zero_swa_unowned_rows_kernel(
    OUT,
    QSL_PTR,
    stride_ot,
    stride_oh,
    stride_qsl,
    TOTAL_TOKENS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    D_NOPE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    in_bounds = offs_t < TOTAL_TOKENS
    first = tl.load(QSL_PTR).to(tl.int32)
    last = tl.load(QSL_PTR + ROWS * stride_qsl).to(tl.int32)
    unowned = in_bounds & ((offs_t < first) | (offs_t >= last))
    if tl.sum(unowned.to(tl.int32), axis=0) == 0:
        return
    offs_d = tl.arange(0, D_NOPE)
    tl.store(
        OUT + offs_t[:, None] * stride_ot + pid_h * stride_oh + offs_d[None, :],
        tl.zeros((BLOCK_T, D_NOPE), dtype=tl.bfloat16),
        mask=unowned[:, None],
    )


@triton.jit
def _commit_swa_ring_kernel(
    KV_PTR,
    WIN_PTR,
    STATE_ROW_PTR,
    START_PTR,
    QSL_PTR,
    POS_PTR,
    COUNTS_PTR,
    stride_kt,
    stride_wr,
    stride_wt,
    stride_sr,
    stride_st,
    stride_qsl,
    stride_pos,
    stride_cnt,
    TOTAL_TOKENS: tl.constexpr,
    WINDOW: tl.constexpr,
    RECORD_WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    item = tl.program_id(0)
    req = tl.program_id(1)

    count = tl.load(COUNTS_PTR + req * stride_cnt).to(tl.int64)
    start = tl.load(START_PTR + req * stride_st).to(tl.int64)
    state_row = tl.load(STATE_ROW_PTR + req * stride_sr).to(tl.int64)
    q_base = tl.load(QSL_PTR + req * stride_qsl).to(tl.int64)

    token_off = tl.maximum(0, count - WINDOW) + item
    in_req = (state_row > 0) & (count > 0) & (token_off < count)
    src_row = q_base + token_off
    in_bounds = in_req & (src_row >= 0) & (src_row < TOTAL_TOKENS)
    pos = tl.load(POS_PTR + src_row * stride_pos, mask=in_bounds, other=-1).to(tl.int64)
    end_pos = start + count
    keep = in_bounds & (pos >= 0) & (pos >= end_pos - WINDOW)
    ring_slot = tl.maximum(pos, 0) % WINDOW

    cols = tl.arange(0, BLOCK_D)
    mask = keep & (cols < RECORD_WIDTH)
    vals = tl.load(KV_PTR + src_row * stride_kt + cols, mask=mask, other=0.0)
    tl.store(WIN_PTR + state_row * stride_wr + ring_slot * stride_wt + cols, vals, mask=mask)


def _launch_swa_mla_prefill(
    query: torch.Tensor,
    window: torch.Tensor,
    kv_record: torch.Tensor,
    sink: torch.Tensor,
    out: torch.Tensor,
    state_rows: torch.Tensor,
    starts: torch.Tensor,
    query_start_loc: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    window_size: int,
    max_query_tokens: int,
) -> None:
    tokens, heads, head_dim = query.shape
    d_nope = out.shape[2]
    d_rope = head_dim - d_nope
    rows = starts.shape[0]
    single_span = rows == 1 and max_query_tokens >= tokens
    grid_tokens = tokens if single_span else max_query_tokens
    if not single_span:
        _zero_swa_unowned_rows_kernel[(triton.cdiv(tokens, 64), heads)](
            out,
            query_start_loc,
            out.stride(0),
            out.stride(1),
            query_start_loc.stride(0),
            TOTAL_TOKENS=tokens,
            ROWS=rows,
            BLOCK_T=64,
            D_NOPE=d_nope,
            num_warps=4,
        )
    grid = (triton.cdiv(grid_tokens, _BLOCK_M), heads, rows)
    _swa_mla_prefill_kernel[grid](
        query,
        window,
        kv_record,
        sink,
        out,
        state_rows,
        starts,
        query_start_loc,
        lengths,
        query.stride(0),
        query.stride(1),
        window.stride(0),
        window.stride(1),
        kv_record.stride(0),
        out.stride(0),
        out.stride(1),
        state_rows.stride(0),
        starts.stride(0),
        query_start_loc.stride(0),
        lengths.stride(0),
        float(scale),
        TOTAL_TOKENS=tokens,
        ROWS=rows,
        SINGLE_SPAN=single_span,
        WINDOW=window_size,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        D_NOPE=d_nope,
        D_ROPE=d_rope,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )


@torch.library.custom_op("flare::swa_mla_prefill", mutates_args=("out",))
def swa_mla_prefill_out(
    query: torch.Tensor,
    window: torch.Tensor,
    kv_record: torch.Tensor,
    sink: torch.Tensor,
    out: torch.Tensor,
    state_rows: torch.Tensor,
    starts: torch.Tensor,
    query_start_loc: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    window_size: int,
    max_query_tokens: int,
) -> None:
    _launch_swa_mla_prefill(
        query,
        window,
        kv_record,
        sink,
        out,
        state_rows,
        starts,
        query_start_loc,
        lengths,
        scale,
        window_size,
        max_query_tokens,
    )


@swa_mla_prefill_out.register_fake
def _fake_swa_mla_prefill_out(
    query,
    window,
    kv_record,
    sink,
    out,
    state_rows,
    starts,
    query_start_loc,
    lengths,
    scale,
    window_size,
    max_query_tokens,
):
    pass


@torch.library.custom_op("flare::swa_ring_commit", mutates_args=("window",))
def commit_swa_ring(
    kv_record: torch.Tensor,
    window: torch.Tensor,
    state_rows: torch.Tensor,
    starts: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    counts: torch.Tensor,
    window_size: int,
) -> None:
    tokens, record_width = kv_record.shape
    rows = starts.shape[0]
    if tokens == 0 or rows == 0:
        return
    grid = (min(tokens, window_size), rows)
    _commit_swa_ring_kernel[grid](
        kv_record,
        window,
        state_rows,
        starts,
        query_start_loc,
        positions,
        counts,
        kv_record.stride(0),
        window.stride(0),
        window.stride(1),
        state_rows.stride(0),
        starts.stride(0),
        query_start_loc.stride(0),
        positions.stride(0),
        counts.stride(0),
        TOTAL_TOKENS=tokens,
        WINDOW=window_size,
        RECORD_WIDTH=record_width,
        BLOCK_D=triton.next_power_of_2(record_width),
        num_warps=4,
    )


@commit_swa_ring.register_fake
def _fake_commit_swa_ring(
    kv_record,
    window,
    state_rows,
    starts,
    query_start_loc,
    positions,
    counts,
    window_size,
):
    pass


def _validate_swa_operands(
    query: torch.Tensor,
    window: torch.Tensor,
    kv_record: torch.Tensor,
    sink: torch.Tensor,
    output: torch.Tensor,
    state_rows: torch.Tensor,
    starts: torch.Tensor,
    query_start_loc: torch.Tensor,
    lengths: torch.Tensor,
    *,
    window_size: int,
    max_query_tokens: int,
) -> None:
    if (
        query.device.type != "cuda"
        or query.ndim != 3
        or window.ndim != 3
        or kv_record.ndim != 2
        or sink.ndim != 1
        or output.ndim != 3
    ):
        raise ValueError("SWA MLA prefill requires CUDA 3D query/window/output, 2D KV records, and 1D sink")
    tokens, heads, head_dim = query.shape
    d_nope = output.shape[2]
    d_rope = head_dim - d_nope
    rows = starts.numel()
    if (
        tokens <= 0
        or heads <= 0
        or rows <= 0
        or max_query_tokens <= 0
        or window_size <= 1
        or d_nope <= 0
        or d_rope <= 0
        or (d_nope & (d_nope - 1)) != 0
        or (d_rope & (d_rope - 1)) != 0
    ):
        raise ValueError("SWA MLA prefill requires positive dimensions and power-of-two NoPE/RoPE widths")
    if (
        window.shape[1:] != (window_size, head_dim)
        or kv_record.shape != (tokens, head_dim)
        or sink.shape != (heads,)
        or output.shape != (tokens, heads, d_nope)
    ):
        raise ValueError("SWA MLA prefill tensor shapes do not match the declared MLA geometry")
    if (
        any(t.dtype != torch.bfloat16 for t in (query, window, kv_record, output))
        or sink.dtype != torch.float32
        or state_rows.dtype != torch.int64
        or starts.dtype != torch.int64
        or query_start_loc.dtype != torch.int32
        or lengths.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("SWA MLA prefill requires BF16 activations, FP32 sink, int64 rows/starts, and int32 offsets")
    if (
        state_rows.shape != (rows,)
        or starts.shape != (rows,)
        or query_start_loc.shape != (rows + 1,)
        or lengths.shape != (rows,)
    ):
        raise ValueError("SWA MLA prefill metadata shapes must match the request row count")
    for tensor in (query, window, kv_record, output):
        if tensor.device != query.device or tensor.stride(-1) != 1:
            raise ValueError("SWA MLA prefill operands must reside on one CUDA device with unit trailing stride")
    for tensor in (sink, state_rows, starts, query_start_loc, lengths):
        if tensor.device != query.device:
            raise ValueError("SWA MLA prefill metadata and sink must reside on the query CUDA device")
    if not sink.is_contiguous():
        raise ValueError("SWA MLA prefill sink vector must be contiguous")


class PreparedSwaMlaPrefill:
    """Direct-ring tiled-query BF16 sliding-window MLA prefill with fused FP32 sink scaling.

    Reads committed circular-ring entries directly from ``window[state_row, pos % window_size]``
    in a separate loop from packed chunk KV records, keeping ``BLOCK_M=32`` queries in SM120
    registers without staging intermediate window pages or materializing post-attention LSE.
    """

    def __init__(
        self,
        query: torch.Tensor,
        window: torch.Tensor,
        kv_record: torch.Tensor,
        sink: torch.Tensor,
        output: torch.Tensor,
        metadata,
        *,
        scale: float,
        window_size: int = 512,
        max_query_tokens: int | None = None,
        session=None,
        name: str = "swa_mla_prefill",
    ):
        del session, name
        resolved_max_q = int(metadata.max_query_tokens if max_query_tokens is None else max_query_tokens)
        _validate_swa_operands(
            query,
            window,
            kv_record,
            sink,
            output,
            metadata.state_rows,
            metadata.starts,
            metadata.query_start_loc,
            metadata.lengths,
            window_size=window_size,
            max_query_tokens=resolved_max_q,
        )
        self.query = query
        self.window = window
        self.kv_record = kv_record
        self.sink = sink
        self.output = output
        self.metadata = metadata
        self.scale = float(scale)
        self.window_size = int(window_size)
        self.max_query_tokens = resolved_max_q

        tokens, heads, head_dim = query.shape
        d_nope = output.shape[2]
        d_rope = head_dim - d_nope
        rows = metadata.starts.numel()
        single_span = rows == 1 and resolved_max_q >= tokens
        grid_tokens = tokens if single_span else resolved_max_q

        if not isinstance(query, torch._subclasses.FakeTensor):
            if not single_span:
                compiled_zero = warmup_triton(
                    _zero_swa_unowned_rows_kernel,
                    (
                        output,
                        metadata.query_start_loc,
                        output.stride(0),
                        output.stride(1),
                        metadata.query_start_loc.stride(0),
                        tokens,
                        rows,
                        64,
                        d_nope,
                    ),
                    (triton.cdiv(tokens, 64), heads),
                    num_warps=4,
                )
                compiled_zero._init_handles()
            compiled_main = warmup_triton(
                _swa_mla_prefill_kernel,
                (
                    query,
                    window,
                    kv_record,
                    sink,
                    output,
                    metadata.state_rows,
                    metadata.starts,
                    metadata.query_start_loc,
                    metadata.lengths,
                    query.stride(0),
                    query.stride(1),
                    window.stride(0),
                    window.stride(1),
                    kv_record.stride(0),
                    output.stride(0),
                    output.stride(1),
                    metadata.state_rows.stride(0),
                    metadata.starts.stride(0),
                    metadata.query_start_loc.stride(0),
                    metadata.lengths.stride(0),
                    self.scale,
                    tokens,
                    rows,
                    single_span,
                    self.window_size,
                    _BLOCK_M,
                    _BLOCK_N,
                    d_nope,
                    d_rope,
                ),
                (triton.cdiv(grid_tokens, _BLOCK_M), heads, rows),
                num_warps=_NUM_WARPS,
                num_stages=_NUM_STAGES,
            )
            compiled_main._init_handles()
            if hasattr(metadata, "positions"):
                compiled_commit = warmup_triton(
                    _commit_swa_ring_kernel,
                    (
                        kv_record,
                        window,
                        metadata.state_rows,
                        metadata.starts,
                        metadata.query_start_loc,
                        metadata.positions,
                        metadata.lengths,
                        kv_record.stride(0),
                        window.stride(0),
                        window.stride(1),
                        metadata.state_rows.stride(0),
                        metadata.starts.stride(0),
                        metadata.query_start_loc.stride(0),
                        metadata.positions.stride(0),
                        metadata.lengths.stride(0),
                        tokens,
                        self.window_size,
                        head_dim,
                        triton.next_power_of_2(head_dim),
                    ),
                    (min(tokens, self.window_size), rows),
                    num_warps=4,
                )
                compiled_commit._init_handles()

        self.output_layout = None

    def functional(self, *, session=None):
        """Release declaration operands and allocate attention outputs locally per forward."""
        del session
        if self.output_layout is None:
            self.output_layout = (
                tuple(self.output.shape),
                tuple(self.output.stride()),
                self.output.dtype,
            )
            self.query_shape = tuple(self.query.shape)
            self.kv_shape = tuple(self.kv_record.shape)
            self.rows = self.metadata.starts.numel()
            self.query = self.kv_record = self.output = self.metadata = None
        return self

    def __call__(
        self,
        query: torch.Tensor | None = None,
        kv_record: torch.Tensor | None = None,
        metadata=None,
        *,
        window: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.output_layout is None:
            m = self.metadata if metadata is None else metadata
            out = self.output if output is None else output
            swa_mla_prefill_out(
                self.query if query is None else query,
                self.window if window is None else window,
                self.kv_record if kv_record is None else kv_record,
                self.sink,
                out,
                m.state_rows,
                m.starts,
                m.query_start_loc,
                m.lengths,
                self.scale,
                self.window_size,
                self.max_query_tokens,
            )
            return out

        if query is None or kv_record is None or metadata is None:
            raise ValueError("functional SWA MLA prefill requires query, kv_record, and metadata operands")
        if tuple(query.shape) != self.query_shape or tuple(kv_record.shape) != self.kv_shape:
            raise ValueError("functional SWA MLA prefill operands differ from prepared geometry")
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=query.device)
        swa_mla_prefill_out(
            query,
            self.window if window is None else window,
            kv_record,
            self.sink,
            output,
            metadata.state_rows,
            metadata.starts,
            metadata.query_start_loc,
            metadata.lengths,
            self.scale,
            self.window_size,
            self.max_query_tokens,
        )
        return output
