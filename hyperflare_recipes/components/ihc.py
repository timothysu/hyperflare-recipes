# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Four-stream gated residual connections with affine RMS normalization."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from hyperflare_recipes.components.linear import UnquantizedProjection
from ignition.preparation.tuning import register_finalizer


@triton.jit
def _warp_sum(values, SIZE: tl.constexpr, X: tl.constexpr):
    lane = tl.arange(0, SIZE)
    for level in tl.static_range(X.bit_length() - 1):
        step = X >> (level + 1)
        values += tl.gather(values, lane ^ step, 0)
    return values


@triton.jit
def _chunk(x, WIDTH: tl.constexpr, INDEX: tl.constexpr):
    size: tl.constexpr = x.shape[0]
    for n in tl.static_range((size // WIDTH).bit_length() - 1):
        left, right = tl.split(tl.trans(tl.reshape(x, (2, size >> (n + 1)))))
        if (INDEX * WIDTH) & (size >> (n + 1)):
            x = right
        else:
            x = left
    return x


@triton.jit
def row_sum_fp32(x, D: tl.constexpr, ROWS: tl.constexpr):
    """Sum a zero-padded FP32 row in ATen's vectorized reduction order."""
    Y: tl.constexpr = min(16, 1 << (ROWS.bit_length() - 1))
    X: tl.constexpr = min(512 // Y, 1 << ((D // 4).bit_length() - 1))
    SPLIT_Y: tl.constexpr = D // X >= min(Y * 16, 256)
    LANES: tl.constexpr = X * Y if SPLIT_Y else X
    values = tl.full((LANES * 4,), 0, tl.float32)
    for offset in tl.static_range(triton.cdiv(D, LANES * 4)):
        values += _chunk(x, LANES * 4, offset)
    lane = tl.arange(0, LANES)
    even, odd = tl.split(tl.reshape(values, (LANES * 2, 2)))
    a, c = tl.split(tl.reshape(even, (LANES, 2)))
    b, d = tl.split(tl.reshape(odd, (LANES, 2)))
    sums = ((a + b) + c) + d
    sums = _warp_sum(sums, LANES, X)
    if SPLIT_Y:
        for step in tl.static_range(Y.bit_length() - 1):
            sums += tl.gather(sums, lane ^ ((LANES // 2) >> step), 0)
    return tl.sum(tl.where(lane == 0, sums, 0.), 0)


@triton.jit
def _adjacent_sum(x, SIZE: tl.constexpr):
    for level in tl.static_range(SIZE.bit_length() - 1):
        lo, hi = tl.split(tl.reshape(x, (SIZE >> (level + 1), 2)))
        x = lo + hi
    return tl.sum(x, 0)


@triton.jit
def _grouped_inv(x, D: tl.constexpr, EPS: tl.constexpr):
    # b12x sums adjacent pairs within 16-byte packs. Division rounds before epsilon is added.
    limit: tl.constexpr = 64 if D // 8 <= 64 else (128 if D // 8 <= 160 else 256)
    threads: tl.constexpr = min(limit, x.shape[0] // 8)
    pairs: tl.constexpr = x.shape[0] // 2
    a, b = tl.split(tl.reshape(x, (pairs, 2)))
    squares = a * a + b * b
    even, odd = tl.split(tl.reshape(squares, (pairs // 2, 2)))
    e0, e2 = tl.split(tl.reshape(even, (pairs // 4, 2)))
    e1, e3 = tl.split(tl.reshape(odd, (pairs // 4, 2)))
    sums = tl.full((threads,), 0., tl.float32)
    for part in tl.static_range(triton.cdiv(D, 8 * threads)):
        sums += _chunk(e0, threads, part)
        sums += _chunk(e1, threads, part)
        sums += _chunk(e2, threads, part)
        sums += _chunk(e3, threads, part)
    return tl.rsqrt(tl.div_rn(_adjacent_sum(sums, threads), float(D)) + EPS)


@triton.jit
def _residual(S, Branch, Shared, Post, Gain, Updated, t, T: tl.constexpr,
             D: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr, ENTRY: tl.constexpr,
             ADD: tl.constexpr, BS: tl.constexpr, SS: tl.constexpr):
    col = tl.arange(0, BLOCK)
    idx = tl.arange(0, triton.next_power_of_2(4 * D))
    x = tl.load(S + t * 4 * D + idx, (t < T) & (idx < 4 * D), 0).to(tl.float32)
    if ENTRY:
        b = tl.load(Branch + t * BS + col, (t < T) & (col < D), 0).to(tl.float32)
        if ADD:
            b = (b + tl.load(Shared + t * SS + col, (t < T) & (col < D), 0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        w = tl.load(Gain + col, (t < T) & (col < D), 0).to(tl.float32)
        b = (b * _grouped_inv(b, D, EPS) * w).to(tl.bfloat16).to(tl.float32)
        ps = tl.load(Post + t * 4 + idx // D, (t < T) & (idx < 4 * D), 0)
        x = (x + ps * tl.gather(b, idx % D, 0)).to(tl.bfloat16).to(tl.float32)
        tl.store(Updated + t * 4 * D + idx, x, (t < T) & (idx < 4 * D))
    return x


@triton.jit
def _update(S, Branch, Shared, Post, Gain, Updated, Stats, T: tl.constexpr, D: tl.constexpr,
            BLOCK: tl.constexpr, EPS: tl.constexpr, ENTRY: tl.constexpr, ADD: tl.constexpr,
            REDUCE: tl.constexpr, BS: tl.constexpr, SS: tl.constexpr):
    t = tl.program_id(0)
    row = _residual(S, Branch, Shared, Post, Gain, Updated, t, T, D, BLOCK, EPS, ENTRY, ADD, BS, SS)
    if REDUCE:
        tl.store(Stats + t, row_sum_fp32(row * row, 4 * D, T))


@triton.jit
def _prefill(S, Branch, Shared, Post, Gain, Updated, Fn, Stats, T: tl.constexpr,
             D: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr, ENTRY: tl.constexpr,
             ADD: tl.constexpr, G: tl.constexpr, THREADS: tl.constexpr, PACK: tl.constexpr,
             BT: tl.constexpr, BS: tl.constexpr, SS: tl.constexpr):
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    warps: tl.constexpr = THREADS // 32
    padded: tl.constexpr = triton.next_power_of_2(4 * D)
    x = tl.full((BT, padded), 0, tl.bfloat16)
    for j in tl.static_range(BT):
        t = tl.program_id(0) * BT + j
        row = _residual(S, Branch, Shared, Post, Gain, Updated, t, T,
                        D, BLOCK, EPS, ENTRY, ADD, BS, SS)
        tl.store(Stats + t * (1 + G) * warps, row_sum_fp32(row * row, 4 * D, T), t < T)
        x = tl.where((tl.arange(0, BT) == j)[:, None], row.to(tl.bfloat16)[None, :], x)
    gate = tl.arange(0, G)
    lane = tl.arange(0, THREADS * PACK)
    acc = tl.full((BT, G, THREADS), 0., tl.float32)
    # Reuse each FP32 weight tile across tokens while preserving the selected
    # GEMV lane's FMA sequence. The second launch combines warp totals.
    for i in tl.static_range(triton.cdiv(4 * D, THREADS * PACK)):
        loc = lane + i * THREADS * PACK
        weight = tl.load(Fn + gate[:, None] * (4 * D) + loc[None, :], loc[None, :] < 4 * D, 0.)
        values = x
        for n in tl.static_range((padded // (THREADS * PACK)).bit_length() - 1):
            left, right = tl.split(tl.permute(tl.reshape(values, (BT, 2, padded >> (n + 1))), (0, 2, 1)))
            if (i * THREADS * PACK) & (padded >> (n + 1)):
                values = right
            else:
                values = left
        vals = values.to(tl.float32)
        if PACK == 4:
            ve, vo = tl.split(tl.reshape(vals, (BT, THREADS * 2, 2)))
            v0, v2 = tl.split(tl.reshape(ve, (BT, THREADS, 2)))
            v1, v3 = tl.split(tl.reshape(vo, (BT, THREADS, 2)))
            we, wo = tl.split(tl.reshape(weight, (G, THREADS * 2, 2)))
            w0, w2 = tl.split(tl.reshape(we, (G, THREADS, 2)))
            w1, w3 = tl.split(tl.reshape(wo, (G, THREADS, 2)))
            acc = tl.fma(v0[:, None, :], w0[None, :, :], acc)
            acc = tl.fma(v1[:, None, :], w1[None, :, :], acc)
            acc = tl.fma(v2[:, None, :], w2[None, :, :], acc)
            acc = tl.fma(v3[:, None, :], w3[None, :, :], acc)
        else:
            acc = tl.fma(vals[:, None, :], weight[None, :, :], acc)
    for level in tl.static_range(5):
        lo, hi = tl.split(tl.reshape(acc, (BT, G, THREADS >> (level + 1), 2)))
        acc = lo + hi
    offsets = ((rows[:, None, None] * (G + 1) + gate[None, :, None] + 1) * warps
               + tl.arange(0, warps)[None, None, :])
    tl.store(Stats + offsets, acc, rows[:, None, None] < T)


@triton.jit
def _collapse(S, Stats, Mix, Base, Scale, Gain, Posts, Out, D: tl.constexpr,
              BLOCK: tl.constexpr, G: tl.constexpr, PARTS: tl.constexpr,
              EPS: tl.constexpr, HC_EPS: tl.constexpr, MAG: tl.constexpr):
    t = tl.program_id(0)
    if PARTS:
        lane = tl.arange(0, PARTS)
        stats = Stats + t * (G + 1) * PARTS
        sq = tl.load(stats)
    else:
        sq = tl.load(Stats + t)
    inv = tl.rsqrt(sq * (1. / (4 * D)) + EPS)
    col = tl.arange(0, BLOCK)
    accum = tl.full((BLOCK,), 0, tl.float32)
    for s in tl.static_range(G):
        if PARTS:
            mix = _adjacent_sum(tl.load(stats + (1 + s) * PARTS + lane), PARTS)
        else:
            mix = tl.load(Mix + t * G + s)
        logit = mix * inv * tl.load(Scale + s // 4) + tl.load(Base + s)
        gate = tl.div_rn(1., 1. + libdevice.exp(-logit))
        if s < 4:
            values = tl.load(S + (t * 4 + s) * D + col, col < D, 0).to(tl.float32)
            accum += (gate + HC_EPS) * values
        else:
            tl.store(Posts + t * 4 + s - 4, MAG * gate + HC_EPS)
    x = accum.to(tl.bfloat16).to(tl.float32)
    gain = tl.load(Gain + col, col < D, 0).to(tl.float32)
    normalized = x * _grouped_inv(x, D, EPS) * gain
    tl.store(Out + t * D + col, normalized, col < D)


@torch.library.triton_op("flare::ihc_update", mutates_args=("updated", "stats"))
def _update_out(streams: torch.Tensor, branch: torch.Tensor | None, shared: torch.Tensor | None,
                post: torch.Tensor | None, gain: torch.Tensor | None, fn: torch.Tensor | None,
                updated: torch.Tensor | None, stats: torch.Tensor | None,
                eps: float, gates: int, threads: int, pack: int, batch: int) -> None:
    tokens, _, d = streams.shape
    if not streams.is_contiguous() or streams.shape[1] != 4 or streams.dtype != torch.bfloat16:
        raise ValueError("iHC residual storage requires contiguous BF16 rows of four streams")
    if branch is not None:
        if (branch.shape != (tokens, d) or branch.stride(1) != 1 or branch.dtype != torch.bfloat16
                or post is None or post.shape != (tokens, 4) or post.dtype != torch.float32
                or not post.is_contiguous() or gain is None or gain.shape != (d,) or not gain.is_contiguous()
                or gain.dtype not in (torch.float32, torch.bfloat16)):
            raise ValueError("iHC entry requires a BF16 branch, four FP32 post gates and an affine gain")
        if shared is not None and (shared.shape != branch.shape or shared.stride(1) != 1
                                   or shared.dtype != branch.dtype):
            raise ValueError("both iHC entry branches must have matching BF16 rows")
    elif shared is not None:
        raise ValueError("a shared iHC branch requires its routed branch")
    block = triton.next_power_of_2(d)
    bs = 0 if branch is None else branch.stride(0)
    ss = 0 if shared is None else shared.stride(0)
    if gates:
        torch.library.wrap_triton(_prefill)[(triton.cdiv(tokens, batch),)](
            streams, branch, shared, post, gain, updated, fn, stats, tokens, d, block, eps,
            branch is not None, shared is not None, gates, threads, pack, batch, bs, ss,
            num_warps=8, enable_fp_fusion=False)
    else:
        torch.library.wrap_triton(_update)[(tokens,)](
            streams, branch, shared, post, gain, updated, stats, tokens, d, block, eps,
            branch is not None, shared is not None, stats is not None, bs, ss,
            num_warps=8, enable_fp_fusion=False)


@torch.library.triton_op("flare::ihc_collapse", mutates_args=("posts", "out"))
def _collapse_out(streams: torch.Tensor, stats: torch.Tensor, mixes: torch.Tensor | None,
                  base: torch.Tensor, scale: torch.Tensor, norm: torch.Tensor,
                  posts: torch.Tensor | None, out: torch.Tensor, parts: int,
                  eps: float, hc_eps: float, magnitude: float) -> None:
    tokens, _, d = streams.shape
    torch.library.wrap_triton(_collapse)[(tokens,)](
        streams, stats, mixes, base, scale, norm, posts, out, d, triton.next_power_of_2(d),
        base.numel(), parts, eps, hc_eps, magnitude, enable_fp_fusion=False)


def post_norm(streams, branch, shared, post, gain, eps):
    """Close the residual boundary before an external stream mutation."""
    updated = torch.empty_like(streams)
    _update_out(streams, branch, shared, post, gain, None, updated, None, eps, 0, 32, 1, 1)
    return updated


class PreparedIHC:
    """Prepare stream collapse and the following branch's input normalization."""

    def __init__(self, streams, mixes, fn, base, scale, norm, *, eps, hc_eps, magnitude, session, name):
        t, count, d = streams.shape
        g = fn.shape[0]
        if (count != 4 or d < 256 or d % 8 or streams.dtype != torch.bfloat16 or not streams.is_contiguous()
                or fn.shape != (g, 4 * d) or g not in (4, 8)
                or base.numel() != g or scale.numel() != g // 4 or mixes.shape != (t, g)
                or norm.shape != (d,) or norm.dtype not in (torch.bfloat16, torch.float32)
                or any(not v.is_contiguous() for v in (fn, base, scale, norm))
                or any(v.dtype != torch.float32 for v in (fn, base, scale, mixes))):
            raise ValueError("iHC requires four BF16 streams and four or eight FP32 gate rows")
        self.shape = tuple(streams.shape)
        self.fn, self.base, self.scale, self.norm = fn, base, scale, norm
        self.eps, self.hc_eps, self.magnitude = eps, hc_eps, magnitude
        self.gates = g
        self.threads, self.pack = 32, 1
        self.prefill = t > 64
        self.batch = 4 if t >= 256 else 1
        self.projection = UnquantizedProjection(
            streams.view(t, -1), fn, mixes, session=session, name=name,
        ).functional(session=session)
        if self.prefill:
            register_finalizer(session, self._bind_projection)

    def _bind_projection(self):
        config = self.projection.plan.selection.config
        if config.backend not in ("mixed_vector", "simt"):
            raise ValueError(f"unsupported FP32 iHC projection schedule {config.backend}")
        self.threads = config.block_threads
        self.pack = 4 if config.backend == "mixed_vector" else 1

    def __call__(self, streams, branch=None, shared=None, post=None, gain=None):
        if streams.shape != self.shape:
            raise ValueError("iHC streams must match the prepared token count and feature width")
        t, _, d = streams.shape
        updated = torch.empty_like(streams) if branch is not None else None
        parts = self.threads // 32 if self.prefill else 0
        stats_shape = (t, self.gates + 1, parts) if self.prefill else (t,)
        stats = streams.new_empty(stats_shape, dtype=torch.float32)
        _update_out(streams, branch, shared, post, gain, self.fn, updated, stats, self.eps,
                    self.gates if self.prefill else 0, self.threads, self.pack, self.batch)
        residual = streams if updated is None else updated
        mixes = None if self.prefill else self.projection(residual.view(t, -1))
        posts = streams.new_empty((t, 4), dtype=torch.float32) if self.gates == 8 else None
        out = streams.new_empty((t, d))
        _collapse_out(residual, stats, mixes, self.base, self.scale, self.norm, posts, out,
                      parts, self.eps, self.hc_eps, self.magnitude)
        return residual, posts, out
