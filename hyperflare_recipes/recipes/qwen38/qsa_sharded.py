# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP-sharded QSA representative scoring and top-k selection over replicated compressed caches.

Both modes keep ``compressed_k_cache`` and ``raw_k_ring`` identically replicated
across ranks (via ``cache_only=True``), so Helios KV cache allocation,
checkpointing, and offload contracts remain unchanged:

- **row** (prefill chunks, ``tokens >= 512``): each rank scores and selects
  interleaved 32-row blocks (``block % world_size == rank`` for causal load
  balance), all-gathers only the ``512`` int32 group IDs per row, and expands
  them directly into ``selected_positions`` with bitwise-identical output.
- **group** (decode / speculative verification, ``tokens <= 128``): each rank
  scores and selects its striped compressed pages (``page % world_size == rank``),
  packs sorted ``(score, global_group_id)`` candidates, all-gathers across ranks,
  and merges to the exact global top-512 groups (higher FP32 score first, then
  lower global group ID).

Every rank constructs an identical ``QsaQuery`` (``dcp_size=1, dcp_rank=0``) and
reads its rank identity from a one-element int64 device tensor so AOT compiled
forwards share one binary across all TP ranks without rank guards.
"""

from __future__ import annotations

import itertools
import math
import weakref

import torch
import triton
import triton.language as tl
from b12x.attention import qsa
from b12x.attention.qsa._contract import _STABLE_TOPK_BLOCK, _run_attention
from b12x.attention.qsa._kernels import (
    launch_prepare_index_query,
    launch_score_representatives_tiled,
    launch_stabilize_topk,
    launch_stage_topk_carry,
    launch_topk_groups,
)
from b12x.preparation import PreparedCall

from ignition.components.collectives import PreparedAllGather
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.scratch import ScratchArena, planned_scratch, prime_owners
from ignition.preparation.tuning import declare, register_finalizer

ROW_BLOCK = 32
GROUP_BUDGET = 512
COMPRESS_RATIO = 4
COMPRESSED_PAGE_SIZE = 16
SELECTION_WIDTH = 2051
# Covers 32,768 groups (128k context) + 512 carry slots in a single pass for 4096 rows.
SM120_PREFILL_SCORE_WORKSPACE_BYTES = (32768 + GROUP_BUDGET) * 4096 * 4

_SHARDED_CONTEXTS = weakref.WeakValueDictionary()
_SHARDED_HANDLES = itertools.count()

_CACHE_MUTABLE_FIELDS = (
    "compressed_k_cache",
    "raw_k_ring",
    "raw_logical_positions",
    "raw_rope_positions",
    "raw_interval_start_positions",
)
_FULL_DYNAMIC_FIELDS = (
    "query",
    "index_query",
    "raw_index_key",
    "request_ids",
    "query_positions",
    "rope_positions",
    "sequence_lengths",
    "query_start_loc",
    "num_accepted_tokens",
    "is_prefilling",
)


def _scratch_view(storage: torch.Tensor, *, offset_bytes: int, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    nbytes = math.prod(shape) * dtype.itemsize
    return storage.narrow(0, int(offset_bytes), int(nbytes)).view(dtype).view(shape)


def _launch(kernel, grid, args, **meta):
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    if torch.compiler.is_compiling() or get_proxy_mode() is not None:
        torch.library.wrap_triton(kernel)[grid](*args, **meta)
    else:
        kernel[grid](*args, **meta)


@triton.jit
def _orderable(scores):
    bits = (scores + 0.0).to(tl.int32, bitcast=True)
    return tl.where(bits < 0, bits ^ 2147483647, bits)


@triton.jit
def _candidate_keys_from_ordered(ord_score, group):
    # Higher score first, then lower global group ID; missing (-1) sorts below every present entry.
    keys = (ord_score.to(tl.int64) << 32) | (2147483647 - group)
    return tl.where(group >= 0, keys, -9223372036854775808)


@triton.jit
def _candidate_keys(score, group):
    return _candidate_keys_from_ordered(_orderable(score), group)


@triton.jit
def _prepare_group_inputs_kernel(
    FullTable, HalfTable, RequestIds, QueryPositions, SeqLengths, Rank,
    LocalPositions, LocalSeqLengths, GlobalEligible,
    full_stride, half_stride,
    HALF_WIDTH: tl.constexpr, TILES: tl.constexpr, BATCH: tl.constexpr, ROWS: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_B: tl.constexpr,
    PAGE_GROUPS: tl.constexpr, WORLD: tl.constexpr,
):
    pid = tl.program_id(0)
    rank = tl.load(Rank).to(tl.int64)
    table_programs = BATCH * TILES
    if pid < table_programs:
        batch = pid // TILES
        tile = pid % TILES
        cols = tile * BLOCK_T + tl.arange(0, BLOCK_T)
        mask = cols < HALF_WIDTH
        src_cols = cols.to(tl.int64) * WORLD + rank
        vals = tl.load(FullTable + batch.to(tl.int64) * full_stride + src_cols, mask=mask, other=-1)
        tl.store(HalfTable + batch.to(tl.int64) * half_stride + cols, vals, mask=mask)
    else:
        round_g = PAGE_GROUPS * WORLD
        r_offs = tl.arange(0, BLOCK_R)
        r_mask = r_offs < ROWS
        req = tl.load(RequestIds + r_offs, mask=r_mask, other=-1).to(tl.int64)
        active = r_mask & (req >= 0)
        pos = tl.load(QueryPositions + r_offs, mask=active, other=0).to(tl.int64)
        seq_len = tl.load(SeqLengths + tl.where(active, req, 0), mask=active, other=0).to(tl.int64)
        g_elig = tl.where(active, tl.minimum(tl.maximum(pos + 1, 0) // 4, tl.maximum(seq_len, 0) // 4), 0)
        rem = g_elig % round_g
        l_elig = (g_elig // round_g) * PAGE_GROUPS + tl.minimum(tl.maximum(rem - rank * PAGE_GROUPS, 0), PAGE_GROUPS)
        local_pos = tl.where(active, tl.where(l_elig > 0, l_elig * 4 - 1, 0), -1)
        tl.store(LocalPositions + r_offs, local_pos, mask=r_mask)
        tl.store(GlobalEligible + r_offs, g_elig.to(tl.int32), mask=r_mask)

        b_offs = tl.arange(0, BLOCK_B)
        b_mask = b_offs < BATCH
        seq_g = tl.maximum(tl.load(SeqLengths + b_offs, mask=b_mask, other=0).to(tl.int64), 0) // 4
        b_rem = seq_g % round_g
        l_seq_g = (seq_g // round_g) * PAGE_GROUPS + tl.minimum(tl.maximum(b_rem - rank * PAGE_GROUPS, 0), PAGE_GROUPS)
        tl.store(LocalSeqLengths + b_offs, (l_seq_g * 4).to(tl.int32), mask=b_mask)


@triton.jit
def _pack_group_candidates_kernel(
    LocalIds, LocalScores, Packed, Rank,
    PAGE_GROUPS: tl.constexpr, WORLD: tl.constexpr, TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    rank = tl.load(Rank).to(tl.int64)
    idx = tl.arange(0, TOPK)
    lg = tl.load(LocalIds + row * TOPK + idx).to(tl.int64)
    sc = tl.load(LocalScores + row * TOPK + idx)
    global_id = tl.where(
        lg >= 0,
        (lg // PAGE_GROUPS) * (PAGE_GROUPS * WORLD) + rank * PAGE_GROUPS + (lg % PAGE_GROUPS),
        -1,
    )
    keys = tl.sort(_candidate_keys(sc, global_id), descending=True)
    present = keys != -9223372036854775808
    # Store the lossless 32-bit orderable score directly so cross-rank merge preserves all FP32 bits.
    out_ord = (keys >> 32).to(tl.int32)
    out_id = (2147483647 - (keys & 4294967295)).to(tl.int32)
    out_base = Packed + row * (TOPK * 2)
    tl.store(out_base + idx, tl.where(present, out_ord, -2147483648))
    tl.store(out_base + TOPK + idx, tl.where(present, out_id, -1))


@triton.jit
def _merge_and_expand_group_candidates_2_kernel(
    Gathered, GlobalEligible, RequestIds, QueryPositions, SelectedPositions,
    selected_stride,
    TOPK: tl.constexpr, COMPRESS_RATIO: tl.constexpr, SELECTION_WIDTH: tl.constexpr,
):
    # Fuses TP2 exact 1-step bitonic merge, ascending group-ID sort, and group-to-position expansion.
    row = tl.program_id(0)
    idx = tl.arange(0, TOPK)
    rev = TOPK - 1 - idx
    base0 = Gathered + row.to(tl.int64) * (4 * TOPK)
    base1 = base0 + 2 * TOPK
    o0 = tl.load(base0 + idx)
    i0 = tl.load(base0 + TOPK + idx).to(tl.int64)
    o1 = tl.load(base1 + rev)
    i1 = tl.load(base1 + TOPK + rev).to(tl.int64)
    keys = tl.maximum(_candidate_keys_from_ordered(o0, i0), _candidate_keys_from_ordered(o1, i1))
    present = keys != -9223372036854775808
    gid = tl.where(present, (2147483647 - (keys & 4294967295)).to(tl.int32), 2147483647)
    sorted_gid = tl.sort(gid, descending=False)
    final_gid = tl.where(sorted_gid < 2147483647, sorted_gid, -1)

    req = tl.load(RequestIds + row).to(tl.int64)
    real = req >= 0
    eligible = tl.where(real, tl.load(GlobalEligible + row), 0)
    selected_groups = tl.minimum(eligible, TOPK)
    expanded_count = selected_groups * COMPRESS_RATIO
    position = tl.load(QueryPositions + row).to(tl.int64)
    tail_start = ((position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_length = tl.where(real, tl.maximum(position + 1 - tail_start, 0), 0)

    lanes = tl.arange(0, COMPRESS_RATIO)[None, :]
    cols = idx[:, None] * COMPRESS_RATIO + lanes
    expanded = final_gid[:, None] * COMPRESS_RATIO + lanes
    tail_col = cols - expanded_count
    in_tail = (tail_col >= 0) & (tail_col < tail_length)
    vals = tl.where(
        cols < expanded_count,
        expanded,
        tl.where(in_tail, (tail_start + tail_col).to(tl.int32), -1),
    )
    out_row = SelectedPositions + row.to(tl.int64) * selected_stride
    tl.store(out_row + cols, vals.to(tl.int32))

    tail_cols = TOPK * COMPRESS_RATIO + tl.arange(0, COMPRESS_RATIO)
    t_col = tail_cols - expanded_count
    t_in = (t_col >= 0) & (t_col < tail_length)
    t_vals = tl.where(t_in, (tail_start + t_col).to(tl.int32), -1)
    tl.store(out_row + tail_cols, t_vals.to(tl.int32), mask=tail_cols < SELECTION_WIDTH)


@triton.jit
def _merge_group_candidates_n_kernel(Gathered, OutIds, WORLD: tl.constexpr, TOPK: tl.constexpr):
    # General WORLD-way exact merge (WORLD=4 -> 2048 candidates), preserving score-descending,
    # group-ID-ascending order to match unsharded b12x stable selection.
    row = tl.program_id(0)
    total: tl.constexpr = WORLD * TOPK
    idx = tl.arange(0, total)
    rank_idx = idx // TOPK
    local_idx = idx % TOPK
    base = Gathered + row * (total * 2) + rank_idx * (2 * TOPK)
    o = tl.load(base + local_idx)
    i = tl.load(base + TOPK + local_idx).to(tl.int64)
    keys = tl.sort(_candidate_keys_from_ordered(o, i), descending=True)
    present = (idx < TOPK) & (keys != -9223372036854775808)
    gid = tl.where(present, (2147483647 - (keys & 4294967295)).to(tl.int32), -1)
    tl.store(OutIds + row * TOPK + idx, gid, mask=idx < TOPK)


@triton.jit
def _expand_groups_kernel(
    TopkIds, GlobalEligible, RequestIds, QueryPositions, SelectedPositions,
    topk_stride, selected_stride,
    GROUP_BUDGET: tl.constexpr, COMPRESS_RATIO: tl.constexpr,
    SELECTION_WIDTH: tl.constexpr, BLOCK_W: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_W)
    col_mask = columns < SELECTION_WIDTH
    req = tl.load(RequestIds + row).to(tl.int64)
    real = req >= 0
    eligible = tl.where(real, tl.load(GlobalEligible + row), 0)
    selected_groups = tl.minimum(eligible, GROUP_BUDGET)
    expanded_count = selected_groups * COMPRESS_RATIO
    group_cols = columns // COMPRESS_RATIO
    group_ids = tl.load(
        TopkIds + row.to(tl.int64) * topk_stride + group_cols,
        mask=col_mask & (columns < expanded_count),
        other=-1,
    )
    expanded = group_ids * COMPRESS_RATIO + columns % COMPRESS_RATIO
    position = tl.load(QueryPositions + row).to(tl.int64)
    tail_start = ((position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_length = tl.where(real, tl.maximum(position + 1 - tail_start, 0), 0)
    tail_col = columns - expanded_count
    in_tail = (tail_col >= 0) & (tail_col < tail_length)
    result = tl.where(
        columns < expanded_count,
        expanded,
        tl.where(in_tail, (tail_start + tail_col).to(tl.int32), -1),
    )
    tl.store(SelectedPositions + row.to(tl.int64) * selected_stride + columns, result, mask=col_mask)


@triton.jit
def _expand_row_split_groups_kernel(
    GatheredIds, RequestIds, QueryPositions, SeqLengths, SelectedPositions,
    gathered_stride, selected_stride,
    WORLD: tl.constexpr, ROW_BLOCK: tl.constexpr,
    GROUP_BUDGET: tl.constexpr, COMPRESS_RATIO: tl.constexpr,
    SELECTION_WIDTH: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # Fuses 32-row interleaved block unshuffle with exact group-to-position expansion.
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_W)
    col_mask = columns < SELECTION_WIDTH
    req = tl.load(RequestIds + row).to(tl.int64)
    real = req >= 0
    position = tl.load(QueryPositions + row).to(tl.int64)
    seq_len = tl.load(SeqLengths + req, mask=real, other=0).to(tl.int64)
    eligible = tl.where(real, tl.minimum(tl.maximum(position + 1, 0) // COMPRESS_RATIO, tl.maximum(seq_len, 0) // COMPRESS_RATIO), 0)
    selected_groups = tl.minimum(eligible, GROUP_BUDGET)
    expanded_count = selected_groups * COMPRESS_RATIO
    block = row // ROW_BLOCK
    in_block = row % ROW_BLOCK
    owner = block % WORLD
    local_row = (block // WORLD) * ROW_BLOCK + in_block
    group_cols = columns // COMPRESS_RATIO
    group_ids = tl.load(
        GatheredIds + local_row.to(tl.int64) * gathered_stride + owner.to(tl.int64) * GROUP_BUDGET + group_cols,
        mask=col_mask & (columns < expanded_count),
        other=-1,
    )
    expanded = group_ids * COMPRESS_RATIO + columns % COMPRESS_RATIO
    tail_start = ((position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_length = tl.where(real, tl.maximum(position + 1 - tail_start, 0), 0)
    tail_col = columns - expanded_count
    in_tail = (tail_col >= 0) & (tail_col < tail_length)
    result = tl.where(
        columns < expanded_count,
        expanded,
        tl.where(in_tail, (tail_start + tail_col).to(tl.int32), -1),
    )
    tl.store(SelectedPositions + row.to(tl.int64) * selected_stride + columns, result, mask=col_mask)


@torch.library.custom_op("flare::qsa_cache_only", mutates_args=("writes", "workspace"))
def _qsa_cache_only_op(
    source: torch.Tensor, reads: list[torch.Tensor], writes: list[torch.Tensor],
    workspace: torch.Tensor, handle: int,
) -> None:
    _SHARDED_CONTEXTS[handle].run_cache_only(reads, writes, workspace)


@_qsa_cache_only_op.register_fake
def _fake_qsa_cache_only(source, reads, writes, workspace, handle):
    pass


@torch.library.custom_op("flare::qsa_select_sharded", mutates_args=("send", "workspace"))
def _qsa_select_sharded_op(
    reads: list[torch.Tensor], send: torch.Tensor, workspace: torch.Tensor, handle: int,
) -> None:
    _SHARDED_CONTEXTS[handle].run_select(reads, send, workspace)


@_qsa_select_sharded_op.register_fake
def _fake_qsa_select_sharded(reads, send, workspace, handle):
    pass


@torch.library.custom_op("flare::qsa_attend_selected", mutates_args=("output", "workspace"))
def _qsa_attend_selected_op(
    reads: list[torch.Tensor], output: torch.Tensor, workspace: torch.Tensor, handle: int,
) -> None:
    _SHARDED_CONTEXTS[handle].run_attend(reads, output, workspace)


@_qsa_attend_selected_op.register_fake
def _fake_qsa_attend_selected(reads, output, workspace, handle):
    pass


@torch.library.custom_op(
    "flare::qsa_group_select_async",
    mutates_args=(
        "cache_writes", "aux_workspace", "aux_buffers",
        "selected_positions", "proj_outputs", "proj_workspaces",
    ),
)
def _qsa_group_select_async_op(
    source: torch.Tensor,
    projected_index_read: torch.Tensor | None,
    cache_reads: list[torch.Tensor],
    cache_writes: list[torch.Tensor],
    select_reads: list[torch.Tensor],
    aux_workspace: torch.Tensor,
    aux_buffers: list[torch.Tensor],
    selected_positions: torch.Tensor,
    proj_values: list[torch.Tensor],
    proj_scales: list[torch.Tensor],
    proj_outputs: list[torch.Tensor],
    proj_workspaces: list[torch.Tensor],
    proj_plans: list[int],
    cache_ws_bytes: int,
    handle: int,
    stream_handle: int,
) -> None:
    _SHARDED_CONTEXTS[handle].run_group_select_async(
        source, projected_index_read, cache_reads, cache_writes, select_reads,
        aux_workspace, aux_buffers, selected_positions,
        proj_values, proj_scales, proj_outputs, proj_workspaces, proj_plans,
        cache_ws_bytes, stream_handle,
    )


@_qsa_group_select_async_op.register_fake
def _fake_qsa_group_select_async(
    source, projected_index_read, cache_reads, cache_writes, select_reads,
    aux_workspace, aux_buffers, selected_positions,
    proj_values, proj_scales, proj_outputs, proj_workspaces, proj_plans,
    cache_ws_bytes, handle, stream_handle,
):
    pass


@torch.library.custom_op(
    "flare::qsa_attend_joined",
    mutates_args=("output", "selected_positions", "workspace"),
)
def _qsa_attend_joined_op(
    reads: list[torch.Tensor],
    keepalive: list[torch.Tensor],
    output: torch.Tensor,
    selected_positions: torch.Tensor,
    workspace: torch.Tensor,
    cache_ws_bytes: int,
    handle: int,
    stream_handle: int,
) -> None:
    _SHARDED_CONTEXTS[handle].run_attend_joined(
        reads, output, selected_positions, workspace, cache_ws_bytes, stream_handle,
    )


@_qsa_attend_joined_op.register_fake
def _fake_qsa_attend_joined(
    reads, keepalive, output, selected_positions, workspace, cache_ws_bytes, handle, stream_handle,
):
    pass


def sharded_score_workspace_bytes(select_rows: int, context: int, device: torch.device) -> int | None:
    """Choose QSA score workspace size for a selection plan."""
    from hyperflare_recipes.components import sm103

    full = select_rows * (context // COMPRESS_RATIO) * 4
    if sm103.is_sm103(device):
        return max(full, 128 << 20)
    if select_rows >= 1024:
        return max(min(full, SM120_PREFILL_SCORE_WORKSPACE_BYTES), 128 << 20)
    return None


class PreparedShardedQSA:
    """TP-sharded QSA selection (row-split for prefill, group-split for decode/verify)."""

    def __init__(
        self,
        attention,
        *,
        mode: str,
        rank: int,
        world_size: int,
        gather_runtime,
        coordinator,
        session,
        name: str,
        workspace: ScratchArena | torch.Tensor | None,
    ) -> None:
        if mode not in ("row", "group"):
            raise ValueError(f"unsupported QSA shard mode: {mode!r}")
        if world_size not in (2, 4) or not 0 <= rank < world_size:
            raise ValueError("sharded QSA selection requires TP2 or TP4")
        self.attention = attention
        self.mode = mode
        self.rank = rank
        self.world_size = world_size
        self.tokens = attention.tokens
        self.q_heads = attention.q_heads
        self.device = attention.scratch.query.device
        self.plan = attention.plan
        self.selection_only = attention.sparse is not None
        self.overlap = (
            mode == "group"
            and isinstance(workspace, ScratchArena)
            and workspace.overlap
        )
        self.stream = workspace.auxiliary_stream if self.overlap else None
        self.stream_handle = self.stream.cuda_stream if self.stream is not None else 0
        self.batch, self.table_width = attention.metadata.main_block_table.shape
        self.page_size = attention.state.main_k_cache.shape[1]
        self.context = self.table_width * self.page_size
        self.rank_tensor = persistent_copy(torch.tensor([rank], dtype=torch.int64, device=self.device))
        self.inactive_raw_slots = persistent_copy(
            torch.full((self.batch,), -1, dtype=torch.int64, device=self.device)
        )

        if mode == "row":
            if self.tokens % (ROW_BLOCK * world_size):
                raise ValueError("row-split QSA selection requires tokens divisible by 32 * world_size")
            self.local_rows = self.tokens // world_size
            self.select_context = self.context
            blocks = torch.arange(self.local_rows // ROW_BLOCK, dtype=torch.int64) * world_size + rank
            indices = (blocks[:, None] * ROW_BLOCK + torch.arange(ROW_BLOCK, dtype=torch.int64)[None, :]).reshape(-1)
            self.row_indices = persistent_copy(indices.to(self.device))
            select_table_width = self.table_width
            send_cols = GROUP_BUDGET * 2  # 512 int32 viewed as 1024 bf16
        else:
            if self.table_width % world_size:
                raise ValueError("group-split QSA selection requires context pages divisible by world_size")
            self.local_rows = self.tokens
            self.select_context = self.context // world_size
            self.row_indices = None
            select_table_width = self.table_width // world_size
            send_cols = GROUP_BUDGET * 4  # 512 fp32 + 512 int32 viewed as 2048 bf16

        # Build selection-only QSA plan with identical Caps/QsaQuery across all ranks.
        score_ws = sharded_score_workspace_bytes(self.local_rows, self.select_context, self.device)
        select_caps = qsa.Caps(
            device=self.device,
            max_batch=self.batch,
            kv_dtype=attention.state.main_k_cache.dtype,
            max_raw_state_slots=attention.state.raw_k_ring.shape[0],
            max_q_rows=self.local_rows,
            max_seq_len=self.select_context,
            num_main_cache_pages=attention.cache_page_capacity,
            num_compressed_cache_pages=attention.cache_page_capacity,
            main_page_size=self.page_size,
            compressed_page_size=self.page_size // COMPRESS_RATIO,
            max_speculative_tokens=attention.plan.query.max_speculative_tokens,
            position_axes=3,
            q_heads=attention.q_heads,
            kv_heads=attention.kv_heads,
            mrope_sections=(11, 11, 10),
            mrope_interleaved=True,
            rms_norm_eps=attention.module.config.rms_norm_eps,
            score_workspace_bytes=score_ws,
        )
        # Preparation examples matching exact runtime shapes & strides of the selection inputs.
        if mode == "row":
            ex_index_q = torch.zeros((self.local_rows, 4, 128), dtype=torch.bfloat16, device=self.device)
            ex_raw_k = torch.zeros((self.local_rows, 128), dtype=torch.bfloat16, device=self.device)
            ex_req = torch.full((self.local_rows,), -1, dtype=torch.int64, device=self.device)
            ex_pos = torch.full((self.local_rows,), -1, dtype=torch.int64, device=self.device)
            ex_rope = torch.full((self.local_rows, 3), -1, dtype=torch.int64, device=self.device)
            ex_table = attention.metadata.compressed_block_table
            ex_main_table = attention.metadata.main_block_table
        else:
            ex_index_q = attention.index_query
            ex_raw_k = attention.index_key
            ex_req = attention.metadata.request_ids
            ex_pos = attention.metadata.query_positions
            ex_rope = attention.metadata.rope_positions
            ex_table = torch.full((self.batch, select_table_width), -1, dtype=torch.int32, device=self.device)
            ex_main_table = ex_table

        select_binding_kwargs = {
            **{field: getattr(attention.state, field) for field in attention.state.__dataclass_fields__},
            "main_block_table": ex_main_table,
            "compressed_block_table": ex_table,
            "raw_state_slot_ids": self.inactive_raw_slots,
            "index_q_norm_weight": attention.binding_kwargs["index_q_norm_weight"],
            "index_k_norm_weight": attention.binding_kwargs["index_k_norm_weight"],
            "rope_cos": attention.binding_kwargs["rope_cos"],
            "rope_sin": attention.binding_kwargs["rope_sin"],
        }
        select_operands = {
            **select_binding_kwargs,
            "request_ids": ex_req,
            "rope_positions": ex_rope,
            "index_query": ex_index_q,
            "raw_index_key": ex_raw_k,
        }
        # Pin sparse_gqa_direct_kv_warps since select_plan never executes sparse attention.
        self.select_plan = qsa.plan(
            select_caps,
            invocation=qsa.invocation_from_tensors(select_caps, **select_operands),
            override=qsa.QsaConfig(backend="cutedsl", sparse_gqa_direct_kv_warps=2),
        )
        # Preparation reuses slices of the transient QSAScratch buffers instead of allocating
        # per-layer persistent dummy tensors; runtime reuses slices of bindings["output"] and
        # bindings["selected_positions"], which run_select never writes.
        select_binding_kwargs.update(
            output=attention.scratch.attended[:self.local_rows],
            selected_positions=attention.scratch.selected_positions[:self.local_rows],
            k_descale=attention.binding_kwargs["k_descale"],
            v_descale=attention.binding_kwargs["v_descale"],
        )
        self.select_binding_kwargs = select_binding_kwargs
        self._select_dynamic_ex = {
            "query": attention.scratch.query[:self.local_rows],
            "index_query": ex_index_q,
            "raw_index_key": ex_raw_k,
            "request_ids": ex_req,
            "query_positions": ex_pos,
            "rope_positions": ex_rope,
            "sequence_lengths": attention.metadata.sequence_lengths,
            "query_start_loc": attention.metadata.query_start_loc,
            "num_accepted_tokens": attention.metadata.num_accepted_tokens,
            "is_prefilling": attention.metadata.is_prefilling,
        }
        self.select_workspace = workspace

        def prepare_select(materialized):
            storage = planned_scratch(materialized.scratch_specs(), workspace)
            binding = materialized.bind_for_preparation(scratch=storage, **self.select_binding_kwargs)
            self.select_workspace = storage
            return PreparedCall(
                run=lambda: materialized.run_for_preparation(binding, **self._select_dynamic_ex),
                owners=prime_owners(workspace, storage, binding),
            )

        declare(session, (self.select_plan.request(name=f"{name}.select_shard", prepare_call=prepare_select),))

        def refresh_select():
            self.select_workspace = planned_scratch(self.select_plan.scratch_specs(), workspace)
            self.select_binding = qsa.bind(self.select_plan, scratch=self.select_workspace, **self.select_binding_kwargs)

        register_finalizer(session, refresh_select, workspace=workspace)

        # Prepare the AllGather collective for the sharded selection payload.
        send_ex = torch.zeros((self.local_rows, send_cols), dtype=torch.bfloat16, device=self.device)
        gathered_ex = torch.zeros((self.local_rows, send_cols * world_size), dtype=torch.bfloat16, device=self.device)
        self.gather = PreparedAllGather(
            send_ex, gathered_ex, runtime=gather_runtime,
            session=session, coordinator=coordinator, name=f"{name}.select_gather",
        )
        self.send_shape = (self.local_rows, send_cols)

        self.cache_read_names = tuple(
            name for name in attention.binding_kwargs if name not in (*_CACHE_MUTABLE_FIELDS, "output", "selected_positions")
        )
        self.handle = next(_SHARDED_HANDLES)
        _SHARDED_CONTEXTS[self.handle] = self

    def rebind_pages_for_preparation(self, state) -> None:
        if self.select_binding_kwargs is None:
            return
        for name in ("main_k_cache", "main_v_cache", "compressed_k_cache"):
            self.select_binding_kwargs[name] = getattr(state, name)
        self.select_binding = qsa.bind(
            self.select_plan, scratch=self.select_workspace, **self.select_binding_kwargs
        )

    def functional(self, *, session):
        self.gather.functional(session=session)
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        self.select_workspace_bytes = self.select_workspace.numel()
        self.select_binding = self.select_binding_kwargs = self._select_dynamic_ex = None
        self.select_workspace = None

    def run_cache_only(self, reads, writes, workspace):
        count = len(self.cache_read_names)
        bindings = dict(zip(self.cache_read_names, reads[:count], strict=True))
        bindings.update(zip(_CACHE_MUTABLE_FIELDS, writes, strict=True))
        bindings["output"] = reads[count]
        bindings["selected_positions"] = reads[count + 1]
        binding = qsa.bind(self.plan, scratch=workspace, **bindings)
        dynamic = dict(zip(_FULL_DYNAMIC_FIELDS, reads[count + 2:], strict=True))
        qsa.run(binding, **dynamic, cache_only=True)

    def run_select(self, reads, send, workspace):
        (
            main_k, main_v, comp_k, raw_k_ring, raw_pos, raw_rope, raw_anchor,
            index_q_norm, index_k_norm, rope_cos, rope_sin, k_descale, v_descale,
            main_table, comp_table, out_buf, sel_buf,
            index_query, request_ids, query_positions, rope_positions, sequence_lengths,
        ) = reads
        binding = qsa.bind(
            self.select_plan,
            scratch=workspace,
            main_k_cache=main_k,
            main_v_cache=main_v,
            compressed_k_cache=comp_k,
            raw_k_ring=raw_k_ring,
            raw_logical_positions=raw_pos,
            raw_rope_positions=raw_rope,
            raw_interval_start_positions=raw_anchor,
            main_block_table=main_table,
            compressed_block_table=comp_table,
            raw_state_slot_ids=self.inactive_raw_slots,
            index_q_norm_weight=index_q_norm,
            index_k_norm_weight=index_k_norm,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            output=out_buf[:self.local_rows],
            selected_positions=sel_buf[:self.local_rows],
            k_descale=k_descale,
            v_descale=v_descale,
        )
        # b12x.attention.qsa exposes no public dcp_size=1 selection-only entry, so
        # validate the active selection operand ABI and scratch bounds before launching only the
        # four selection kernels (prepare_q, score, topk, stabilize) on select_plan.
        state = binding.state
        caps = state.caps
        if state.programs is None:
            raise RuntimeError("sharded QSA selection requires retained native b12x programs")
        support = state.programs.support
        abi_operands = state.abi["operands"]
        for name, tensor in (
            ("index_query", index_query),
            ("request_ids", request_ids),
            ("rope_positions", rope_positions),
        ):
            descriptor = abi_operands[name]
            expected_strides = descriptor["strides"]
            if (
                tensor.device != caps.device
                or tensor.dtype != getattr(torch, descriptor["dtype"])
                or tensor.ndim != len(expected_strides)
                or any(
                    size != 1 and stride != planned
                    for size, stride, planned in zip(tensor.shape, tensor.stride(), expected_strides, strict=True)
                )
            ):
                raise ValueError(
                    f"sharded QSA {name} ABI ({tensor.dtype}, {tensor.stride()}) differs from prepared {descriptor}"
                )
        if (
            query_positions.device != caps.device
            or query_positions.dtype != torch.int64
            or query_positions.shape != (self.local_rows,)
            or not query_positions.is_contiguous()
        ):
            raise ValueError("sharded QSA query_positions must be contiguous int64 [local_rows]")
        if (
            sequence_lengths.device != caps.device
            or sequence_lengths.dtype != torch.int32
            or sequence_lengths.shape != (self.batch,)
            or not sequence_lengths.is_contiguous()
        ):
            raise ValueError("sharded QSA sequence_lengths must be contiguous int32 [batch]")
        work_rows = int(state.workspace_q_rows)
        if work_rows != self.local_rows:
            raise ValueError(
                f"sharded QSA selection requires workspace_q_rows ({work_rows}) to match local_rows ({self.local_rows})"
            )
        group_budget = int(caps.group_budget)
        score_chunk_groups = int(state.score_chunk_groups)
        num_score_chunks = int(state.num_score_chunks)
        stable_blocks = math.ceil(int(state.score_workspace_width) / _STABLE_TOPK_BLOCK)
        stable_offset = int(state._layout.topk_offset_bytes)
        topk_end = stable_offset + int(state._layout.topk_nbytes)
        count_nbytes = work_rows * stable_blocks * torch.int32.itemsize
        stable_nbytes = (
            2 * count_nbytes
            + work_rows * group_budget * (torch.float32.itemsize + torch.int32.itemsize)
            + work_rows * (torch.float32.itemsize + torch.int32.itemsize)
        )
        if stable_offset + stable_nbytes > topk_end or topk_end > workspace.numel():
            raise ValueError("sharded QSA scratch workspace does not cover the prepared stable-topk layout")
        tie_counts = _scratch_view(workspace, offset_bytes=stable_offset, shape=(work_rows, stable_blocks), dtype=torch.int32)
        stable_offset += count_nbytes
        greater_counts = _scratch_view(workspace, offset_bytes=stable_offset, shape=(work_rows, stable_blocks), dtype=torch.int32)
        stable_offset += count_nbytes
        stable_values = _scratch_view(workspace, offset_bytes=stable_offset, shape=(work_rows, group_budget), dtype=torch.float32)
        stable_offset += work_rows * group_budget * torch.float32.itemsize
        stable_ids = _scratch_view(workspace, offset_bytes=stable_offset, shape=(work_rows, group_budget), dtype=torch.int32)
        stable_offset += work_rows * group_budget * torch.int32.itemsize
        thresholds = _scratch_view(workspace, offset_bytes=stable_offset, shape=(work_rows,), dtype=torch.float32)
        stable_offset += work_rows * torch.float32.itemsize
        greater_totals = _scratch_view(workspace, offset_bytes=stable_offset, shape=(work_rows,), dtype=torch.int32)

        launch_prepare_index_query(
            index_query=index_query,
            request_ids=request_ids,
            norm_weight=index_q_norm,
            rope_positions=rope_positions,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            prepared_query=binding.prepared_index_query,
            caps=caps,
            _prepared=support,
        )
        prior_values, prior_ids = binding.topk_values_b, binding.topk_group_ids_b
        final_values, final_ids = prior_values, prior_ids
        for score_chunk in range(num_score_chunks):
            group_offset = score_chunk * score_chunk_groups
            group_count = min(score_chunk_groups, int(caps.max_groups) - group_offset)
            out_values = binding.topk_values if score_chunk % 2 == 0 else binding.topk_values_b
            out_ids = binding.topk_group_ids if score_chunk % 2 == 0 else binding.topk_group_ids_b
            if group_offset:
                launch_stage_topk_carry(
                    prior_values=prior_values,
                    eligible_counts=binding.eligible_group_counts,
                    scores=binding.scores,
                    group_offset=group_offset,
                    group_budget=group_budget,
                    _prepared=support,
                )
            launch_score_representatives_tiled(
                prepared_query=binding.prepared_index_query,
                query_positions=query_positions,
                request_ids=request_ids,
                sequence_lengths=sequence_lengths,
                compressed_cache=comp_k,
                compressed_block_table=comp_table,
                scores=binding.scores,
                eligible_counts=binding.eligible_group_counts,
                merge_lengths=binding.merge_lengths,
                group_offset=group_offset,
                group_count=group_count,
                caps=caps,
                _prepared=support,
            )
            launch_topk_groups(
                scores=binding.scores,
                eligible_counts=binding.merge_lengths,
                topk_values=out_values,
                topk_group_ids=out_ids,
                group_budget=group_budget,
                _prepared=support,
            )
            launch_stabilize_topk(
                scores=binding.scores,
                merge_lengths=binding.merge_lengths,
                prior_ids=prior_ids,
                eligible_counts=binding.eligible_group_counts,
                topk_values=out_values,
                topk_group_ids=out_ids,
                tie_counts=tie_counts,
                greater_counts=greater_counts,
                stable_values=stable_values,
                stable_ids=stable_ids,
                thresholds=thresholds,
                greater_totals=greater_totals,
                group_offset=group_offset,
                group_budget=group_budget,
                _prepared=support,
            )
            prior_values, prior_ids = out_values, out_ids
            final_values, final_ids = out_values, out_ids

        if self.mode == "row":
            send.view(torch.int32).copy_(final_ids[:self.local_rows])
        else:
            _pack_group_candidates_kernel[(self.local_rows,)](
                final_ids, final_values, send.view(torch.int32), self.rank_tensor,
                PAGE_GROUPS=COMPRESSED_PAGE_SIZE, WORLD=self.world_size, TOPK=GROUP_BUDGET,
                num_warps=8,
            )

    def run_attend(self, reads, output, workspace):
        (
            main_k, main_v, comp_k, raw_k_ring, raw_pos, raw_rope, raw_anchor,
            index_q_norm, index_k_norm, rope_cos, rope_sin, k_descale, v_descale,
            main_table, comp_table, raw_slots, selected_positions,
            query, request_ids, query_positions,
        ) = reads
        binding = qsa.bind(
            self.plan,
            scratch=workspace,
            main_k_cache=main_k,
            main_v_cache=main_v,
            compressed_k_cache=comp_k,
            raw_k_ring=raw_k_ring,
            raw_logical_positions=raw_pos,
            raw_rope_positions=raw_rope,
            raw_interval_start_positions=raw_anchor,
            main_block_table=main_table,
            compressed_block_table=comp_table,
            raw_state_slot_ids=raw_slots,
            index_q_norm_weight=index_q_norm,
            index_k_norm_weight=index_k_norm,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            output=output,
            selected_positions=selected_positions,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        _run_attention(binding, query=query, request_ids=request_ids, query_positions=query_positions)

    def run_group_select_async(
        self, source, projected_index_read, cache_reads, cache_writes, select_reads,
        aux_workspace, aux_buffers, selected_positions,
        proj_values, proj_scales, proj_outputs, proj_workspaces, proj_plans,
        cache_ws_bytes, stream_handle,
    ):
        from hyperflare_recipes.components import sm103_projection

        main = torch.cuda.current_stream(aux_workspace.device)
        auxiliary = torch.cuda.ExternalStream(stream_handle, device=aux_workspace.device)
        # Always synchronize the fork after all metadata and temporary allocations exist on main,
        # preventing Inductor from sinking metadata writes or buffer reuses past the fork.
        auxiliary.wait_stream(main)
        (
            main_k, main_v, comp_k, raw_k_ring, raw_pos, raw_rope, raw_anchor,
            index_q_norm, index_k_norm, rope_cos, rope_sin, k_descale, v_descale,
            comp_table, out_buf, request_ids, query_positions,
            rope_positions, sequence_lengths,
        ) = select_reads
        half_table, local_pos, local_seq_len, global_eligible, send, gathered = aux_buffers[:6]
        half_width = self.table_width // self.world_size
        block_t = 256
        tiles = triton.cdiv(half_width, block_t)
        projected_index = proj_outputs[1] if proj_plans else projected_index_read
        index_query = projected_index[:, :512].view(self.tokens, 4, 128)
        raw_index_key = projected_index[:, 512:]
        count = len(self.cache_read_names)
        dummy_query = cache_reads[count]
        full_cache_reads = (
            list(cache_reads[:count])
            + [out_buf, selected_positions, dummy_query, index_query, raw_index_key]
            + list(cache_reads[count + 1:])
        )
        common_constants = [
            main_k, main_v, comp_k, raw_k_ring, raw_pos, raw_rope, raw_anchor,
            index_q_norm, index_k_norm, rope_cos, rope_sin, k_descale, v_descale,
        ]
        with torch.cuda.stream(auxiliary):
            if proj_plans:
                sm103_projection.packed_out_workspace(
                    source, proj_values[1], proj_scales[1], None, None, proj_outputs[1],
                    proj_workspaces[1], "none", None, proj_plans[1], "any",
                )
            self.run_cache_only(full_cache_reads, cache_writes, aux_workspace[:cache_ws_bytes])
            _prepare_group_inputs_kernel[(self.batch * tiles + 1,)](
                comp_table, half_table, request_ids, query_positions, sequence_lengths,
                self.rank_tensor, local_pos, local_seq_len, global_eligible,
                comp_table.stride(0), half_table.stride(0),
                HALF_WIDTH=half_width, TILES=tiles, BATCH=self.batch, ROWS=self.tokens,
                BLOCK_T=block_t, BLOCK_R=triton.next_power_of_2(self.tokens),
                BLOCK_B=triton.next_power_of_2(self.batch),
                PAGE_GROUPS=COMPRESSED_PAGE_SIZE, WORLD=self.world_size, num_warps=4,
            )
            self.run_select(
                common_constants + [
                    half_table, half_table, out_buf, selected_positions,
                    index_query, request_ids, local_pos, rope_positions, local_seq_len,
                ],
                send,
                aux_workspace[:self.select_workspace_bytes],
            )
            self.gather.run_eager(send, gathered)
            if self.world_size == 2:
                _merge_and_expand_group_candidates_2_kernel[(self.tokens,)](
                    gathered.view(torch.int32), global_eligible, request_ids,
                    query_positions, selected_positions, selected_positions.stride(0),
                    TOPK=GROUP_BUDGET, COMPRESS_RATIO=COMPRESS_RATIO, SELECTION_WIDTH=SELECTION_WIDTH,
                    num_warps=8,
                )
            else:
                merged_ids = aux_buffers[6]
                _merge_group_candidates_n_kernel[(self.tokens,)](
                    gathered.view(torch.int32), merged_ids, WORLD=self.world_size, TOPK=GROUP_BUDGET,
                    num_warps=16,
                )
                _expand_groups_kernel[(self.tokens,)](
                    merged_ids, global_eligible, request_ids, query_positions, selected_positions,
                    merged_ids.stride(0), selected_positions.stride(0),
                    GROUP_BUDGET=GROUP_BUDGET, COMPRESS_RATIO=COMPRESS_RATIO,
                    SELECTION_WIDTH=SELECTION_WIDTH, BLOCK_W=triton.next_power_of_2(SELECTION_WIDTH),
                    num_warps=8,
                )
        if proj_plans:
            sm103_projection.packed_out_workspace(
                source, proj_values[0], proj_scales[0], None, None, proj_outputs[0],
                proj_workspaces[0], "none", None, proj_plans[0], "any",
            )

    def run_attend_joined(self, reads, output, selected_positions, workspace, cache_ws_bytes, stream_handle):
        main = torch.cuda.current_stream(output.device)
        main.wait_stream(torch.cuda.ExternalStream(stream_handle, device=output.device))
        if not self.selection_only:
            full_reads = [*reads[:-3], selected_positions, *reads[-3:]]
            self.run_attend(full_reads, output, workspace[:cache_ws_bytes])

    def start_group_selection(
        self, source, projected_index, metadata, constants, full_workspace_bytes,
        *, parallel_projections=None,
    ):
        device = source.device
        attended = torch.empty((self.tokens, self.q_heads, 256), dtype=source.dtype, device=device)
        selected = torch.empty((self.tokens, SELECTION_WIDTH), dtype=torch.int32, device=device)
        # Transient query buffer satisfies b12x cache_only's non-aliasing ABI check without touching qkv_proj.
        dummy_query = torch.empty((self.tokens, self.q_heads, 256), dtype=source.dtype, device=device)
        if parallel_projections is not None:
            proj_outputs, proj_workspaces = parallel_projections.allocate_runtime(source)
            proj_values = parallel_projections.values
            proj_scales = parallel_projections.scales
            proj_plans = parallel_projections.plans
            projected_qkv, projected_index_live = proj_outputs[0], proj_outputs[1]
            projected_index_read = None
        else:
            proj_values = []
            proj_scales = []
            proj_outputs = []
            proj_workspaces = []
            proj_plans = []
            projected_qkv = None
            projected_index_live = projected_index
            projected_index_read = projected_index
        bindings = {
            **constants,
            "main_block_table": metadata.main_block_table,
            "compressed_block_table": metadata.compressed_block_table,
            "raw_state_slot_ids": metadata.raw_state_slot_ids,
        }
        cache_reads = (
            [bindings[name] for name in self.cache_read_names]
            + [
                dummy_query,
                metadata.request_ids,
                metadata.query_positions,
                metadata.rope_positions,
                metadata.sequence_lengths,
                metadata.query_start_loc,
                metadata.num_accepted_tokens,
                metadata.is_prefilling,
            ]
        )
        cache_writes = [bindings[name] for name in _CACHE_MUTABLE_FIELDS]
        common_constants = [
            bindings["main_k_cache"], bindings["main_v_cache"], bindings["compressed_k_cache"],
            bindings["raw_k_ring"], bindings["raw_logical_positions"],
            bindings["raw_rope_positions"], bindings["raw_interval_start_positions"],
            bindings["index_q_norm_weight"], bindings["index_k_norm_weight"],
            bindings["rope_cos"], bindings["rope_sin"],
            bindings["k_descale"], bindings["v_descale"],
        ]
        select_reads = common_constants + [
            metadata.compressed_block_table, attended,
            metadata.request_ids, metadata.query_positions, metadata.rope_positions,
            metadata.sequence_lengths,
        ]
        half_width = self.table_width // self.world_size
        aux_workspace = torch.empty(
            max(full_workspace_bytes, self.select_workspace_bytes), dtype=torch.uint8, device=device
        )
        aux_buffers = [
            torch.empty((self.batch, half_width), dtype=torch.int32, device=device),
            torch.empty_like(metadata.query_positions),
            torch.empty_like(metadata.sequence_lengths),
            torch.empty((self.tokens,), dtype=torch.int32, device=device),
            torch.empty(self.send_shape, dtype=torch.bfloat16, device=device),
            torch.empty((self.local_rows, self.send_shape[1] * self.world_size), dtype=torch.bfloat16, device=device),
        ]
        if self.world_size != 2:
            aux_buffers.append(torch.empty((self.tokens, GROUP_BUDGET), dtype=torch.int32, device=device))
        _qsa_group_select_async_op(
            source, projected_index_read, cache_reads, cache_writes, select_reads,
            aux_workspace, aux_buffers, selected,
            proj_values, proj_scales, proj_outputs, proj_workspaces, proj_plans,
            full_workspace_bytes, self.handle, self.stream_handle,
        )
        keepalive = [
            source,
            projected_index_live,
            dummy_query,
            metadata.rope_positions,
            metadata.sequence_lengths,
            metadata.query_start_loc,
            metadata.num_accepted_tokens,
            metadata.is_prefilling,
            *proj_workspaces,
            *aux_buffers,
        ]
        return projected_qkv, (attended, selected, aux_workspace, common_constants, keepalive)

    def finish_group_selection(self, pending, query, metadata, full_workspace_bytes):
        attended, selected, aux_workspace, common_constants, keepalive = pending
        attend_reads = common_constants + [
            metadata.main_block_table, metadata.compressed_block_table,
            metadata.raw_state_slot_ids,
            query, metadata.request_ids, metadata.query_positions,
        ]
        _qsa_attend_joined_op(
            attend_reads, keepalive, attended, selected, aux_workspace,
            full_workspace_bytes, self.handle, self.stream_handle,
        )
        return attended, selected

    def select_row_runtime(self, source, bindings, dynamic, full_workspace_bytes):
        """Run local row-split cache update and representative selection before AllGather."""
        if self.mode != "row":
            raise ValueError("select_row_runtime requires row-split QSA sharding")
        device = source.device
        cache_workspace = torch.empty(full_workspace_bytes, dtype=torch.uint8, device=device)
        cache_reads = (
            [bindings[name] for name in self.cache_read_names]
            + [bindings["output"], bindings["selected_positions"]]
            + [dynamic[name] for name in _FULL_DYNAMIC_FIELDS]
        )
        cache_writes = [bindings[name] for name in _CACHE_MUTABLE_FIELDS]
        _qsa_cache_only_op(source, cache_reads, cache_writes, cache_workspace, self.handle)
        del cache_workspace

        common_constants = [
            bindings["main_k_cache"], bindings["main_v_cache"], bindings["compressed_k_cache"],
            bindings["raw_k_ring"], bindings["raw_logical_positions"],
            bindings["raw_rope_positions"], bindings["raw_interval_start_positions"],
            bindings["index_q_norm_weight"], bindings["index_k_norm_weight"],
            bindings["rope_cos"], bindings["rope_sin"],
            bindings["k_descale"], bindings["v_descale"],
        ]
        local_index_q = dynamic["index_query"].index_select(0, self.row_indices)
        local_req = dynamic["request_ids"].index_select(0, self.row_indices)
        local_pos = dynamic["query_positions"].index_select(0, self.row_indices)
        local_rope = dynamic["rope_positions"].index_select(0, self.row_indices)
        select_workspace = torch.empty(self.select_workspace_bytes, dtype=torch.uint8, device=device)
        send = torch.empty(self.send_shape, dtype=torch.bfloat16, device=device)
        select_reads = common_constants + [
            bindings["main_block_table"], bindings["compressed_block_table"],
            bindings["output"], bindings["selected_positions"],
            local_index_q, local_req, local_pos, local_rope, dynamic["sequence_lengths"],
        ]
        _qsa_select_sharded_op(select_reads, send, select_workspace, self.handle)
        del select_workspace
        return common_constants, send

    def finish_row_runtime(self, bindings, dynamic, common_constants, send, full_workspace_bytes, *,
                           gather_dependencies=()):
        """Gather row-split group IDs, expand positions, and run sparse attention."""
        if self.mode != "row":
            raise ValueError("finish_row_runtime requires row-split QSA sharding")
        device = send.device
        gathered = self.gather(send, dependencies=gather_dependencies)
        selected_positions = bindings["selected_positions"]
        gathered_ids = gathered.view(torch.int32)
        _launch(
            _expand_row_split_groups_kernel,
            (self.tokens,),
            (
                gathered_ids, dynamic["request_ids"], dynamic["query_positions"],
                dynamic["sequence_lengths"], selected_positions,
                gathered_ids.stride(0), selected_positions.stride(0),
            ),
            WORLD=self.world_size, ROW_BLOCK=ROW_BLOCK,
            GROUP_BUDGET=GROUP_BUDGET, COMPRESS_RATIO=COMPRESS_RATIO,
            SELECTION_WIDTH=SELECTION_WIDTH, BLOCK_W=triton.next_power_of_2(SELECTION_WIDTH),
            num_warps=8,
        )
        if not self.selection_only:
            attend_workspace = torch.empty(full_workspace_bytes, dtype=torch.uint8, device=device)
            attend_reads = common_constants + [
                bindings["main_block_table"], bindings["compressed_block_table"],
                bindings["raw_state_slot_ids"], selected_positions,
                dynamic["query"], dynamic["request_ids"], dynamic["query_positions"],
            ]
            _qsa_attend_selected_op(attend_reads, bindings["output"], attend_workspace, self.handle)

    def __call__(self, source, bindings, dynamic, full_workspace_bytes):
        if self.mode == "row":
            common_constants, send = self.select_row_runtime(source, bindings, dynamic, full_workspace_bytes)
            self.finish_row_runtime(bindings, dynamic, common_constants, send, full_workspace_bytes)
            return

        device = source.device
        # 1) Update replicated compressed_k_cache and raw_k_ring on all ranks.
        # Keep cache_workspace, select_workspace, and attend_workspace disjoint in lifetime
        # so the CUDA graph pool reuses one buffer instead of holding them simultaneously.
        cache_workspace = torch.empty(full_workspace_bytes, dtype=torch.uint8, device=device)
        cache_reads = (
            [bindings[name] for name in self.cache_read_names]
            + [bindings["output"], bindings["selected_positions"]]
            + [dynamic[name] for name in _FULL_DYNAMIC_FIELDS]
        )
        cache_writes = [bindings[name] for name in _CACHE_MUTABLE_FIELDS]
        _qsa_cache_only_op(source, cache_reads, cache_writes, cache_workspace, self.handle)
        del cache_workspace

        # 2) Prepare rank-local selection inputs without rank-dependent Dynamo guards.
        common_constants = [
            bindings["main_k_cache"], bindings["main_v_cache"], bindings["compressed_k_cache"],
            bindings["raw_k_ring"], bindings["raw_logical_positions"],
            bindings["raw_rope_positions"], bindings["raw_interval_start_positions"],
            bindings["index_q_norm_weight"], bindings["index_k_norm_weight"],
            bindings["rope_cos"], bindings["rope_sin"],
            bindings["k_descale"], bindings["v_descale"],
        ]
        half_width = self.table_width // self.world_size
        half_table = torch.empty((self.batch, half_width), dtype=torch.int32, device=device)
        local_pos = torch.empty_like(dynamic["query_positions"])
        local_seq_len = torch.empty_like(dynamic["sequence_lengths"])
        global_eligible = torch.empty((self.tokens,), dtype=torch.int32, device=device)
        block_t = 256
        tiles = triton.cdiv(half_width, block_t)
        _launch(
            _prepare_group_inputs_kernel,
            (self.batch * tiles + 1,),
            (
                bindings["compressed_block_table"], half_table,
                dynamic["request_ids"], dynamic["query_positions"], dynamic["sequence_lengths"],
                self.rank_tensor, local_pos, local_seq_len, global_eligible,
                bindings["compressed_block_table"].stride(0), half_table.stride(0),
            ),
            HALF_WIDTH=half_width, TILES=tiles, BATCH=self.batch, ROWS=self.tokens,
            BLOCK_T=block_t, BLOCK_R=triton.next_power_of_2(self.tokens),
            BLOCK_B=triton.next_power_of_2(self.batch),
            PAGE_GROUPS=COMPRESSED_PAGE_SIZE, WORLD=self.world_size, num_warps=4,
        )
        local_index_q = dynamic["index_query"]
        local_req = dynamic["request_ids"]
        local_rope = dynamic["rope_positions"]
        main_table = half_table
        comp_table = half_table

        # 3) Run rank-local scoring & top-k selection on select_plan.
        select_workspace = torch.empty(self.select_workspace_bytes, dtype=torch.uint8, device=device)
        send = torch.empty(self.send_shape, dtype=torch.bfloat16, device=device)
        select_reads = common_constants + [
            main_table, comp_table, bindings["output"], bindings["selected_positions"],
            local_index_q, local_req, local_pos, local_rope, local_seq_len,
        ]
        _qsa_select_sharded_op(select_reads, send, select_workspace, self.handle)
        del select_workspace

        # 4) Gather sharded selection across TP ranks.
        gathered = self.gather(send)

        # 5) Expand gathered groups into selected_positions [tokens, 2051].
        selected_positions = bindings["selected_positions"]
        if self.world_size == 2:
            _launch(
                _merge_and_expand_group_candidates_2_kernel,
                (self.tokens,),
                (
                    gathered.view(torch.int32), global_eligible, dynamic["request_ids"],
                    dynamic["query_positions"], selected_positions, selected_positions.stride(0),
                ),
                TOPK=GROUP_BUDGET, COMPRESS_RATIO=COMPRESS_RATIO, SELECTION_WIDTH=SELECTION_WIDTH,
                num_warps=8,
            )
        else:
            merged_ids = torch.empty((self.tokens, GROUP_BUDGET), dtype=torch.int32, device=device)
            _launch(
                _merge_group_candidates_n_kernel,
                (self.tokens,),
                (gathered.view(torch.int32), merged_ids),
                WORLD=self.world_size, TOPK=GROUP_BUDGET, num_warps=16,
            )
            _launch(
                _expand_groups_kernel,
                (self.tokens,),
                (
                    merged_ids, global_eligible, dynamic["request_ids"],
                    dynamic["query_positions"], selected_positions,
                    merged_ids.stride(0), selected_positions.stride(0),
                ),
                GROUP_BUDGET=GROUP_BUDGET, COMPRESS_RATIO=COMPRESS_RATIO,
                SELECTION_WIDTH=SELECTION_WIDTH, BLOCK_W=triton.next_power_of_2(SELECTION_WIDTH),
                num_warps=8,
            )

        # 6) Run rank-local sparse attention (skipped when SM103 Triton sparse attention handles it).
        if not self.selection_only:
            attend_workspace = torch.empty(full_workspace_bytes, dtype=torch.uint8, device=device)
            attend_reads = common_constants + [
                bindings["main_block_table"], bindings["compressed_block_table"],
                bindings["raw_state_slot_ids"], selected_positions,
                dynamic["query"], dynamic["request_ids"], dynamic["query_positions"],
            ]
            _qsa_attend_selected_op(attend_reads, bindings["output"], attend_workspace, self.handle)
