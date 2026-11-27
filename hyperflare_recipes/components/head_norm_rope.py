"""Headwise RMS normalization and partial NeoX rotary embeddings into fixed outputs."""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope(source, weight, cosine, sine, output,
               ROWS: tl.constexpr, HEADS: tl.constexpr, WIDTH: tl.constexpr,
               TOKEN_STRIDE: tl.constexpr, HEAD_STRIDE: tl.constexpr,
               ROTARY: tl.constexpr, COS_STRIDE: tl.constexpr, SIN_STRIDE: tl.constexpr,
               EPS: tl.constexpr, BLOCK_ROWS: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    columns = tl.arange(0, WIDTH)
    token, head = rows // HEADS, rows % HEADS
    values = tl.load(source + token[:, None] * TOKEN_STRIDE + head[:, None] * HEAD_STRIDE
                     + columns[None, :], rows[:, None] < ROWS, 0).to(tl.float32)
    gains = tl.load(weight + columns).to(tl.float32) + 1.0
    inverse = tl.rsqrt(tl.sum(values * values, 1) / WIDTH + EPS)
    normalized = (values * inverse[:, None] * gains[None, :]).to(tl.bfloat16).to(tl.float32)
    partner_columns = tl.where(columns < ROTARY // 2, columns + ROTARY // 2,
                               columns - ROTARY // 2)
    partner_columns = tl.where(columns < ROTARY, partner_columns, columns)
    partners = tl.gather(normalized, tl.broadcast_to(partner_columns[None, :],
                                                   (BLOCK_ROWS, WIDTH)), 1)
    partners = tl.where(columns[None, :] < ROTARY // 2, -partners, partners)
    valid = (rows[:, None] < ROWS) & (columns[None, :] < ROTARY)
    cos = tl.load(cosine + token[:, None] * COS_STRIDE + columns[None, :], valid, 0).to(tl.float32)
    sin = tl.load(sine + token[:, None] * SIN_STRIDE + columns[None, :], valid, 0).to(tl.float32)
    left = (normalized * cos).to(tl.bfloat16).to(tl.float32)
    right = (partners * sin).to(tl.bfloat16).to(tl.float32)
    rotated = (left + right).to(tl.bfloat16).to(tl.float32)
    result = tl.where(columns[None, :] < ROTARY, rotated, normalized)
    tl.store(output + rows[:, None] * WIDTH + columns[None, :], result, rows[:, None] < ROWS)


def head_norm_rope(source, weight, cosine, sine, output, *, eps):
    """Write BF16-rounded head normalization and partial rotation without temporary storage."""
    torch.library.wrap_triton(_norm_rope)[(triton.cdiv(source.shape[0] * source.shape[1], 4),)](
        source, weight, cosine, sine, output, source.shape[0] * source.shape[1],
        source.shape[1], source.shape[2], source.stride(0), source.stride(1),
        cosine.shape[-1], cosine.stride(0), sine.stride(0), eps, 4,
        num_warps=4, enable_fp_fusion=False)


@triton.jit
def _qk_head_norm_mrope(QKV, QW, KW, POSITIONS, ROPE_COS, ROPE_SIN, Q_OUT, K_OUT,
                        QKV_ROW_STRIDE, POS_ROW_STRIDE, COS_ROW_STRIDE, SIN_ROW_STRIDE,
                        Q_HEADS: tl.constexpr, KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
                        ROTARY_DIM: tl.constexpr, MROPE_SECTION_0: tl.constexpr,
                        MROPE_SECTION_1: tl.constexpr, EPS: tl.constexpr,
                        ZERO_CENTERED: tl.constexpr):
    pid = tl.program_id(0)
    total_heads = Q_HEADS + KV_HEADS
    token = pid // total_heads
    head = pid % total_heads
    is_q = head < Q_HEADS
    cols = tl.arange(0, HEAD_DIM)
    if is_q:
        src_offset = token.to(tl.int64) * QKV_ROW_STRIDE + head * (2 * HEAD_DIM) + cols
        weight = tl.load(QW + cols).to(tl.float32)
    else:
        k_head = head - Q_HEADS
        src_offset = (
            token.to(tl.int64) * QKV_ROW_STRIDE
            + Q_HEADS * (2 * HEAD_DIM)
            + k_head * HEAD_DIM
            + cols
        )
        weight = tl.load(KW + cols).to(tl.float32)
    if ZERO_CENTERED:
        weight = weight + 1.0
    values = tl.load(QKV + src_offset).to(tl.float32)
    inverse = tl.rsqrt(tl.sum(values * values, axis=0) / HEAD_DIM + EPS)
    normalized = (values * inverse * weight).to(tl.bfloat16).to(tl.float32)

    half_rot = ROTARY_DIM // 2
    in_rot = cols < ROTARY_DIM
    partner_cols = tl.where(cols < half_rot, cols + half_rot, cols - half_rot)
    partner_cols = tl.where(in_rot, partner_cols, cols)
    partner = tl.gather(normalized, partner_cols, 0)
    partner = tl.where(cols < half_rot, -partner, partner)

    pair = cols % half_rot
    is_height = (pair % 3 == 1) & (pair < 3 * MROPE_SECTION_1)
    is_width = (pair % 3 == 2) & (pair < 3 * (half_rot - MROPE_SECTION_0 - MROPE_SECTION_1))
    axis = tl.where(is_height, 1, tl.where(is_width, 2, 0))
    pos = tl.maximum(
        tl.load(POSITIONS + token.to(tl.int64) * POS_ROW_STRIDE + axis, mask=in_rot, other=0).to(tl.int64),
        0,
    )
    cos = tl.load(ROPE_COS + pos * COS_ROW_STRIDE + pair, mask=in_rot, other=1.0).to(tl.bfloat16).to(tl.float32)
    sin = tl.load(ROPE_SIN + pos * SIN_ROW_STRIDE + pair, mask=in_rot, other=0.0).to(tl.bfloat16).to(tl.float32)

    left = (normalized * cos).to(tl.bfloat16).to(tl.float32)
    right = (partner * sin).to(tl.bfloat16).to(tl.float32)
    rotated = (left + right).to(tl.bfloat16).to(tl.float32)
    result = tl.where(in_rot, rotated, normalized).to(tl.bfloat16)

    if is_q:
        tl.store(Q_OUT + (token.to(tl.int64) * Q_HEADS + head) * HEAD_DIM + cols, result)
    else:
        k_head = head - Q_HEADS
        tl.store(K_OUT + (token.to(tl.int64) * KV_HEADS + k_head) * HEAD_DIM + cols, result)


def qk_head_norm_mrope(
    projected_qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_positions: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    *,
    q_heads: int,
    kv_heads: int,
    eps: float,
    head_dim: int = 256,
    rotary_dim: int = 64,
    mrope_section: tuple[int, int, int] = (11, 11, 10),
    zero_centered: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Q/K RMSNorm and 3-axis interleaved MRoPE from packed [Q, gate, K, V]."""
    if (
        type(q_heads) is not int
        or type(kv_heads) is not int
        or q_heads <= 0
        or kv_heads <= 0
        or type(eps) not in (int, float)
        or not (eps >= 0)
    ):
        raise ValueError("qk_head_norm_mrope requires positive integer head counts and nonnegative numeric eps")
    expected_cols = q_heads * 2 * head_dim + kv_heads * 2 * head_dim
    if (
        projected_qkv.ndim != 2
        or projected_qkv.shape[0] <= 0
        or projected_qkv.stride(1) != 1
        or projected_qkv.shape[1] != expected_cols
        or projected_qkv.dtype != torch.bfloat16
    ):
        raise ValueError("packed QKV projection must be non-empty row-contiguous BF16 [tokens, q_heads*2*D + kv_heads*2*D]")
    tokens = projected_qkv.shape[0]
    if (
        head_dim <= 0
        or (head_dim & (head_dim - 1)) != 0
        or rotary_dim <= 0
        or rotary_dim > head_dim
        or rotary_dim % 2 != 0
        or len(mrope_section) != 3
        or any(type(section) is not int or section < 0 for section in mrope_section)
        or sum(mrope_section) != rotary_dim // 2
    ):
        raise ValueError("invalid head_dim, rotary_dim, or 3-axis mrope_section geometry")
    if (
        q_weight.shape != (head_dim,)
        or k_weight.shape != (head_dim,)
        or not q_weight.is_contiguous()
        or not k_weight.is_contiguous()
    ):
        raise ValueError("Q/K RMSNorm weights must be contiguous 1D tensors of length head_dim")
    if (
        rope_positions.shape != (tokens, 3)
        or rope_positions.stride(1) != 1
        or rope_cos.ndim != 2
        or rope_cos.shape[1] != rotary_dim // 2
        or rope_sin.shape != rope_cos.shape
        or rope_cos.stride(1) != 1
        or rope_sin.stride(1) != 1
    ):
        raise ValueError("3-axis MRoPE requires [tokens, 3] positions and [capacity, rotary_dim // 2] cos/sin tables")
    device = projected_qkv.device
    if any(tensor.device != device for tensor in (q_weight, k_weight, rope_positions, rope_cos, rope_sin)):
        raise ValueError("qk_head_norm_mrope operands must reside on the same device")

    q_out = torch.empty((tokens, q_heads, head_dim), dtype=projected_qkv.dtype, device=projected_qkv.device)
    k_out = torch.empty((tokens, kv_heads, head_dim), dtype=projected_qkv.dtype, device=projected_qkv.device)
    torch.library.wrap_triton(_qk_head_norm_mrope)[(tokens * (q_heads + kv_heads),)](
        projected_qkv,
        q_weight,
        k_weight,
        rope_positions,
        rope_cos,
        rope_sin,
        q_out,
        k_out,
        projected_qkv.stride(0),
        rope_positions.stride(0),
        rope_cos.stride(0),
        rope_sin.stride(0),
        Q_HEADS=q_heads,
        KV_HEADS=kv_heads,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        MROPE_SECTION_0=mrope_section[0],
        MROPE_SECTION_1=mrope_section[1],
        EPS=float(eps),
        ZERO_CENTERED=bool(zero_centered),
        num_warps=2 if tokens >= 512 else 4,
        enable_fp_fusion=False,
    )
    return q_out, k_out


@triton.jit
def _qk_head_norm_rope_kv_write(
    Q_SRC, K_SRC, V_SRC, QW, KW, COS, SIN, SLOTS,
    Q_OUT, K_CACHE, V_CACHE,
    Q_TOK_STRIDE, Q_HEAD_STRIDE, K_TOK_STRIDE, K_HEAD_STRIDE,
    V_TOK_STRIDE, V_HEAD_STRIDE,
    COS_STRIDE, SIN_STRIDE,
    CACHE_PAGE_STRIDE, CACHE_TOK_STRIDE, CACHE_HEAD_STRIDE,
    SLOT_CAPACITY,
    Q_ROWS: tl.constexpr, K_ROWS: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    Q_HEADS: tl.constexpr, KV_HEADS: tl.constexpr, WIDTH: tl.constexpr,
    ROTARY: tl.constexpr, PAGE_SIZE: tl.constexpr, EPS: tl.constexpr,
    ZERO_CENTERED: tl.constexpr, BLOCK_ROWS: tl.constexpr,
):
    """Fused Q/K headwise RMSNorm, partial NeoX RoPE, and paged KV cache append."""
    pid = tl.program_id(0)
    cols = tl.arange(0, WIDTH)
    half_rot = ROTARY // 2
    in_rot = cols < ROTARY
    partner_cols = tl.where(cols < half_rot, cols + half_rot, cols - half_rot)
    partner_cols = tl.where(in_rot, partner_cols, cols)

    if pid < Q_BLOCKS:
        rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        row_mask = rows < Q_ROWS
        row_mask_2d = row_mask[:, None]
        token = rows // Q_HEADS
        head = rows % Q_HEADS
        src_ptr = Q_SRC + token[:, None].to(tl.int64) * Q_TOK_STRIDE + head[:, None] * Q_HEAD_STRIDE + cols[None, :]
        values = tl.load(src_ptr, mask=row_mask_2d, other=0.0).to(tl.float32)
        gains = tl.load(QW + cols).to(tl.float32)
        if ZERO_CENTERED:
            gains = gains + 1.0
        inverse = tl.rsqrt(tl.sum(values * values, axis=1) / WIDTH + EPS)
        normalized = (values * inverse[:, None] * gains[None, :]).to(tl.bfloat16).to(tl.float32)
        partners = tl.gather(normalized, tl.broadcast_to(partner_cols[None, :], (BLOCK_ROWS, WIDTH)), 1)
        partners = tl.where(cols[None, :] < half_rot, -partners, partners)
        valid = row_mask_2d & in_rot[None, :]
        cos = tl.load(COS + token[:, None].to(tl.int64) * COS_STRIDE + cols[None, :], mask=valid, other=0.0).to(tl.float32)
        sin = tl.load(SIN + token[:, None].to(tl.int64) * SIN_STRIDE + cols[None, :], mask=valid, other=0.0).to(tl.float32)
        left = (normalized * cos).to(tl.bfloat16).to(tl.float32)
        right = (partners * sin).to(tl.bfloat16).to(tl.float32)
        rotated = (left + right).to(tl.bfloat16).to(tl.float32)
        result = tl.where(in_rot[None, :], rotated, normalized)
        tl.store(Q_OUT + rows[:, None].to(tl.int64) * WIDTH + cols[None, :], result, mask=row_mask_2d)
    else:
        k_pid = pid - Q_BLOCKS
        rows = k_pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        row_mask = rows < K_ROWS
        token = rows // KV_HEADS
        k_head = rows % KV_HEADS
        slots = tl.load(SLOTS + token, mask=row_mask, other=-1).to(tl.int64)
        active = row_mask & (slots >= 0) & (slots < SLOT_CAPACITY)
        active_2d = active[:, None]
        src_ptr = K_SRC + token[:, None].to(tl.int64) * K_TOK_STRIDE + k_head[:, None] * K_HEAD_STRIDE + cols[None, :]
        values = tl.load(src_ptr, mask=active_2d, other=0.0).to(tl.float32)
        gains = tl.load(KW + cols).to(tl.float32)
        if ZERO_CENTERED:
            gains = gains + 1.0
        inverse = tl.rsqrt(tl.sum(values * values, axis=1) / WIDTH + EPS)
        normalized = (values * inverse[:, None] * gains[None, :]).to(tl.bfloat16).to(tl.float32)
        partners = tl.gather(normalized, tl.broadcast_to(partner_cols[None, :], (BLOCK_ROWS, WIDTH)), 1)
        partners = tl.where(cols[None, :] < half_rot, -partners, partners)
        valid = active_2d & in_rot[None, :]
        cos = tl.load(COS + token[:, None].to(tl.int64) * COS_STRIDE + cols[None, :], mask=valid, other=0.0).to(tl.float32)
        sin = tl.load(SIN + token[:, None].to(tl.int64) * SIN_STRIDE + cols[None, :], mask=valid, other=0.0).to(tl.float32)
        left = (normalized * cos).to(tl.bfloat16).to(tl.float32)
        right = (partners * sin).to(tl.bfloat16).to(tl.float32)
        rotated = (left + right).to(tl.bfloat16).to(tl.float32)
        result = tl.where(in_rot[None, :], rotated, normalized)

        page = slots // PAGE_SIZE
        offset = slots - page * PAGE_SIZE
        cache_off = (
            page[:, None] * CACHE_PAGE_STRIDE
            + offset[:, None] * CACHE_TOK_STRIDE
            + k_head[:, None] * CACHE_HEAD_STRIDE
            + cols[None, :]
        )
        tl.store(K_CACHE + cache_off, result.to(tl.bfloat16).to(K_CACHE.dtype.element_ty), mask=active_2d)
        v_ptr = V_SRC + token[:, None].to(tl.int64) * V_TOK_STRIDE + k_head[:, None] * V_HEAD_STRIDE + cols[None, :]
        v_vals = tl.load(v_ptr, mask=active_2d, other=0.0)
        tl.store(V_CACHE + cache_off, v_vals.to(V_CACHE.dtype.element_ty), mask=active_2d)


@triton.jit
def _k_head_norm_rope_kv_write(
    K_SRC, V_SRC, KW, COS, SIN, SLOTS,
    K_CACHE, V_CACHE,
    K_TOK_STRIDE, K_HEAD_STRIDE,
    V_TOK_STRIDE, V_HEAD_STRIDE,
    COS_STRIDE, SIN_STRIDE,
    CACHE_PAGE_STRIDE, CACHE_TOK_STRIDE, CACHE_HEAD_STRIDE,
    SLOT_CAPACITY,
    K_ROWS: tl.constexpr,
    KV_HEADS: tl.constexpr, WIDTH: tl.constexpr,
    ROTARY: tl.constexpr, PAGE_SIZE: tl.constexpr, EPS: tl.constexpr,
    ZERO_CENTERED: tl.constexpr, BLOCK_ROWS: tl.constexpr,
):
    """Cache-only K headwise RMSNorm, partial NeoX RoPE, and paged KV cache append."""
    pid = tl.program_id(0)
    cols = tl.arange(0, WIDTH)
    half_rot = ROTARY // 2
    in_rot = cols < ROTARY
    partner_cols = tl.where(cols < half_rot, cols + half_rot, cols - half_rot)
    partner_cols = tl.where(in_rot, partner_cols, cols)

    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < K_ROWS
    token = rows // KV_HEADS
    k_head = rows % KV_HEADS
    slots = tl.load(SLOTS + token, mask=row_mask, other=-1).to(tl.int64)
    active = row_mask & (slots >= 0) & (slots < SLOT_CAPACITY)
    active_2d = active[:, None]
    src_ptr = K_SRC + token[:, None].to(tl.int64) * K_TOK_STRIDE + k_head[:, None] * K_HEAD_STRIDE + cols[None, :]
    values = tl.load(src_ptr, mask=active_2d, other=0.0).to(tl.float32)
    gains = tl.load(KW + cols).to(tl.float32)
    if ZERO_CENTERED:
        gains = gains + 1.0
    inverse = tl.rsqrt(tl.sum(values * values, axis=1) / WIDTH + EPS)
    normalized = (values * inverse[:, None] * gains[None, :]).to(tl.bfloat16).to(tl.float32)
    partners = tl.gather(normalized, tl.broadcast_to(partner_cols[None, :], (BLOCK_ROWS, WIDTH)), 1)
    partners = tl.where(cols[None, :] < half_rot, -partners, partners)
    valid = active_2d & in_rot[None, :]
    cos = tl.load(COS + token[:, None].to(tl.int64) * COS_STRIDE + cols[None, :], mask=valid, other=0.0).to(tl.float32)
    sin = tl.load(SIN + token[:, None].to(tl.int64) * SIN_STRIDE + cols[None, :], mask=valid, other=0.0).to(tl.float32)
    left = (normalized * cos).to(tl.bfloat16).to(tl.float32)
    right = (partners * sin).to(tl.bfloat16).to(tl.float32)
    rotated = (left + right).to(tl.bfloat16).to(tl.float32)
    result = tl.where(in_rot[None, :], rotated, normalized)

    page = slots // PAGE_SIZE
    offset = slots - page * PAGE_SIZE
    cache_off = (
        page[:, None] * CACHE_PAGE_STRIDE
        + offset[:, None] * CACHE_TOK_STRIDE
        + k_head[:, None] * CACHE_HEAD_STRIDE
        + cols[None, :]
    )
    tl.store(K_CACHE + cache_off, result.to(tl.bfloat16).to(K_CACHE.dtype.element_ty), mask=active_2d)
    v_ptr = V_SRC + token[:, None].to(tl.int64) * V_TOK_STRIDE + k_head[:, None] * V_HEAD_STRIDE + cols[None, :]
    v_vals = tl.load(v_ptr, mask=active_2d, other=0.0)
    tl.store(V_CACHE + cache_off, v_vals.to(V_CACHE.dtype.element_ty), mask=active_2d)


def qk_head_norm_rope_kv_write(
    query: torch.Tensor | None,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor | None,
    k_weight: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    slots: torch.Tensor,
    q_out: torch.Tensor | None,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    eps: float,
    zero_centered: bool = True,
) -> None:
    """Fused Q/K headwise RMSNorm, partial NeoX RoPE, and paged KV cache write in one launch.

    When ``query``, ``q_weight``, and ``q_out`` are all ``None``, only the K normalization,
    rotation, and K/V cache writes execute (cache-only prefill tail layer).
    """
    cache_only = query is None
    if (q_weight is None) != cache_only or (q_out is None) != cache_only:
        raise ValueError("query, q_weight, and q_out must either all be provided or all be None")
    if type(eps) not in (int, float) or not (eps >= 0):
        raise ValueError("qk_head_norm_rope_kv_write requires nonnegative numeric eps")
    if (
        key.ndim != 3
        or value.ndim != 3
        or key.shape != value.shape
        or key.shape[0] <= 0
        or key.shape[1] <= 0
        or key.stride(2) != 1
        or value.stride(2) != 1
        or key.dtype != torch.bfloat16
        or value.dtype != torch.bfloat16
    ):
        raise ValueError("key and value must be non-empty BF16 [tokens, kv_heads, head_dim] with unit inner stride")
    tokens, kv_heads, head_dim = key.shape
    if head_dim <= 0 or (head_dim & (head_dim - 1)) != 0:
        raise ValueError("head_dim must be a positive power of two")
    if (
        k_weight.shape != (head_dim,)
        or not k_weight.is_contiguous()
        or k_weight.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("k_weight must be a contiguous 1D BF16/FP32 tensor of length head_dim")
    if (
        cosine.ndim != 2
        or sine.ndim != 2
        or cosine.shape != sine.shape
        or cosine.shape[0] != tokens
        or not 0 < cosine.shape[1] <= head_dim
        or cosine.shape[1] % 2 != 0
        or cosine.stride(1) != 1
        or sine.stride(1) != 1
        or cosine.dtype not in (torch.bfloat16, torch.float32)
        or sine.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("cosine and sine must be [tokens, rotary_dim] BF16/FP32 tensors with unit inner stride")
    if (
        slots.shape != (tokens,)
        or not slots.is_contiguous()
        or slots.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("slots must be a contiguous 1D int32/int64 tensor of length tokens")
    if (
        k_cache.ndim != 4
        or v_cache.shape != k_cache.shape
        or v_cache.stride() != k_cache.stride()
        or v_cache.dtype != k_cache.dtype
        or k_cache.shape[0] <= 0
        or k_cache.shape[1] <= 0
        or k_cache.shape[2:] != (kv_heads, head_dim)
        or k_cache.stride(3) != 1
        or k_cache.dtype not in (torch.float8_e4m3fn, torch.bfloat16)
    ):
        raise ValueError("k_cache and v_cache must match as 4D [pages, page_size, kv_heads, head_dim] FP8/BF16 caches")
    device = key.device
    operands = [key, value, k_weight, cosine, sine, slots, k_cache, v_cache]
    if not cache_only:
        if (
            query.ndim != 3
            or query.shape[0] != tokens
            or query.shape[1] <= 0
            or query.shape[2] != head_dim
            or query.stride(2) != 1
            or query.dtype != torch.bfloat16
        ):
            raise ValueError("query must be BF16 [tokens, q_heads, head_dim] with unit inner stride")
        if (
            q_weight.shape != (head_dim,)
            or not q_weight.is_contiguous()
            or q_weight.dtype not in (torch.bfloat16, torch.float32)
        ):
            raise ValueError("q_weight must be a contiguous 1D BF16/FP32 tensor of length head_dim")
        if (
            q_out.shape != query.shape
            or not q_out.is_contiguous()
            or q_out.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
        ):
            raise ValueError("q_out must be contiguous BF16/FP8 [tokens, q_heads, head_dim]")
        operands.extend((query, q_weight, q_out))
    if any(tensor.device != device for tensor in operands):
        raise ValueError("qk_head_norm_rope_kv_write operands must reside on the same device")

    rotary_dim = cosine.shape[1]
    page_size = k_cache.shape[1]
    slot_capacity = k_cache.shape[0] * page_size
    block_rows = 4
    k_rows = tokens * kv_heads
    k_blocks = triton.cdiv(k_rows, block_rows)

    if cache_only:
        torch.library.wrap_triton(_k_head_norm_rope_kv_write)[(k_blocks,)](
            key,
            value,
            k_weight,
            cosine,
            sine,
            slots,
            k_cache,
            v_cache,
            key.stride(0),
            key.stride(1),
            value.stride(0),
            value.stride(1),
            cosine.stride(0),
            sine.stride(0),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            slot_capacity,
            K_ROWS=k_rows,
            KV_HEADS=kv_heads,
            WIDTH=head_dim,
            ROTARY=rotary_dim,
            PAGE_SIZE=page_size,
            EPS=float(eps),
            ZERO_CENTERED=bool(zero_centered),
            BLOCK_ROWS=block_rows,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return

    q_heads = query.shape[1]
    q_rows = tokens * q_heads
    q_blocks = triton.cdiv(q_rows, block_rows)
    torch.library.wrap_triton(_qk_head_norm_rope_kv_write)[(q_blocks + k_blocks,)](
        query,
        key,
        value,
        q_weight,
        k_weight,
        cosine,
        sine,
        slots,
        q_out,
        k_cache,
        v_cache,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        cosine.stride(0),
        sine.stride(0),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        slot_capacity,
        Q_ROWS=q_rows,
        K_ROWS=k_rows,
        Q_BLOCKS=q_blocks,
        Q_HEADS=q_heads,
        KV_HEADS=kv_heads,
        WIDTH=head_dim,
        ROTARY=rotary_dim,
        PAGE_SIZE=page_size,
        EPS=float(eps),
        ZERO_CENTERED=bool(zero_centered),
        BLOCK_ROWS=block_rows,
        num_warps=4,
        enable_fp_fusion=False,
    )
