# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Combine context-parallel attention across hosts by log-sum-exp, over a RoCE all-gather.

Under decode context parallelism every rank attends all heads over its share of the
context and keeps its own head shard of the combined result. Ranks on one host
exchange shards through b12x's PCIe reduce-scatter. Ranks on separate hosts pack, per
row, the partial outputs and natural-log LSEs of every other rank's head shard, in
rank order without their own, and all-gather the packs. Each rank then merges its own
shard from its local partial and the block each peer packed for it.

A pack row holds (W - 1) * H/W * D BF16 partial values, then (W - 1) * H/W FP32 LSEs
stored as BF16 pairs.
"""

import torch
import triton
import triton.language as tl


def pack_width(world, heads, head_dim):
    """BF16 elements in one row's pack of `heads` total heads."""
    return (world - 1) * (heads // world) * (head_dim + 2)


@triton.jit
def _pack(Partial, Lse, Send, SendLse, Rank, send_stride, send_lse_stride, WORLD: tl.constexpr,
          LOCAL: tl.constexpr, DIM: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    head = tl.program_id(2)
    rank = tl.load(Rank).to(tl.int32)
    # Blocks follow rank order and skip this rank's own shard.
    target = block + (block >= rank).to(tl.int32)
    source = row * WORLD * LOCAL + target * LOCAL + head
    columns = tl.arange(0, DIM)
    tl.store(Send + row * send_stride + (block * LOCAL + head) * DIM + columns,
             tl.load(Partial + source * DIM + columns))
    tl.store(SendLse + row * send_lse_stride + block * LOCAL + head, tl.load(Lse + source))


@triton.jit
def _merge(Received, ReceivedLse, Partial, Lse, Output, Rank, received_stride, received_lse_stride, pack_stride,
           pack_lse_stride, WORLD: tl.constexpr, LOCAL: tl.constexpr, DIM: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    rank = tl.load(Rank).to(tl.int32)
    columns = tl.arange(0, DIM)
    own = row * WORLD * LOCAL + rank * LOCAL + head
    own_lse = tl.load(Lse + own)
    # Each contributor's LSE for this head: this rank's own, else the block that peer packed for it.
    largest = own_lse
    for source in tl.static_range(WORLD):
        block = rank - (rank > source).to(tl.int32)
        lse = tl.load(ReceivedLse + row * received_lse_stride + source * pack_lse_stride + block * LOCAL + head,
                      mask=source != rank, other=-float("inf"))
        largest = tl.maximum(largest, lse)
    # A head no rank attended anything for combines to zero.
    base = tl.where(largest > -float("inf"), largest, 0.0)
    weight = tl.exp(own_lse - base)
    total = weight
    combined = tl.where(weight > 0, weight * tl.load(Partial + own * DIM + columns).to(tl.float32), 0.0)
    for source in tl.static_range(WORLD):
        block = rank - (rank > source).to(tl.int32)
        peer = source != rank
        lse = tl.load(ReceivedLse + row * received_lse_stride + source * pack_lse_stride + block * LOCAL + head,
                      mask=peer, other=-float("inf"))
        weight = tl.exp(lse - base)
        values = tl.load(Received + row * received_stride + source * pack_stride + (block * LOCAL + head) * DIM
                         + columns, mask=(columns < DIM) & peer, other=0.0)
        total += weight
        combined += tl.where(weight > 0, weight * values.to(tl.float32), 0.0)
    result = tl.where(total > 0, combined / total, 0.0)
    tl.store(Output + (row * LOCAL + head) * DIM + columns, result.to(Output.dtype.element_ty))


def _launch(kernel, grid, args, **meta):
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    if torch.compiler.is_compiling() or get_proxy_mode() is not None:
        torch.library.wrap_triton(kernel)[grid](*args, **meta)
    else:
        kernel[grid](*args, **meta)


def _lse_view(packs, world, heads, head_dim):
    """FP32 view of the LSE tails of [rows, packs * width] BF16 packs; row and pack strides in FP32 units."""
    values = (world - 1) * (heads // world) * head_dim
    return packs.view(torch.float32)[:, values // 2:]


def pack(partial, lse, send, rank, world):
    """Write every other rank's head shard of `partial` [rows, H, D] and `lse` [rows, H] into `send` [rows, width].

    `rank` is this rank as a one-element device tensor, so compiled forwards serve every rank.
    """
    rows, heads, dim = partial.shape
    send_lse = _lse_view(send, world, heads, dim)
    _launch(_pack, (rows, world - 1, heads // world), (partial, lse, send, send_lse, rank, send.stride(0),
            send.stride(0) // 2), WORLD=world, LOCAL=heads // world, DIM=dim, num_warps=4)


def merge(received, partial, lse, output, rank, world):
    """Combine this rank's head shard into `output` [rows, H/W, D] from its own partial and the gathered packs."""
    rows, heads, dim = partial.shape
    width = pack_width(world, heads, dim)
    received_lse = _lse_view(received, world, heads, dim)
    _launch(_merge, (rows, heads // world), (received, received_lse, partial, lse, output, rank,
            received.stride(0), received.stride(0) // 2, width, width // 2), WORLD=world, LOCAL=heads // world,
            DIM=dim, num_warps=4)


def create_head_exchange(group, device, *, max_rows, heads, head_dim, query_head_dim=None, ipc=None):
    """Allocate b12x staging for all-gathering head shards and LSE-combining partial attention.

    Queries may be wider than the combined values: latent attention gathers absorbed queries of
    `query_head_dim` and combines `head_dim` outputs.
    """
    import torch.distributed as dist
    from b12x.comm.pcie import DcpAllToAll

    world_size = dist.get_world_size(group)
    query_head_dim = head_dim if query_head_dim is None else query_head_dim
    if world_size < 2 or max_rows <= 0 or heads % world_size or head_dim % 8 or query_head_dim % 8:
        raise ValueError("head exchanges require multiple equal head shards and positive capacity")
    return DcpAllToAll.from_exchange_group(exchange_group=group, device=device, max_batch_size=max_rows,
        total_heads=heads, head_dim=head_dim, query_head_dim=query_head_dim, stream_affine=False, ipc=ipc)


import itertools as _itertools
import weakref as _weakref

_OWNERS = _weakref.WeakValueDictionary()
_HANDLES = _itertools.count(1)


@torch.library.custom_op("flare::pcie_lse_reduce_scatter", mutates_args=("output",))
def _lse_reduce_scatter(partial: torch.Tensor, lse: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    owner.runtime.lse_reduce_scatter(partial, lse, out=output, plan=owner.plan, is_lse_base_on_e=True)


@_lse_reduce_scatter.register_fake
def _fake_lse_reduce_scatter(partial, lse, output, handle):
    pass


class PreparedLseReduceScatter:
    """Combine every rank's partial attention by natural-log LSE and keep this rank's heads.

    Partial outputs are [rows, heads, dim] over all heads with FP32 [rows, heads] LSE;
    the output holds this rank's contiguous head shard. Operations sharing the
    channel must execute serially on every rank. Over RoCEnante, ranks all-gather
    the shards each peer keeps through `staging` (send [rows, width] and received
    [rows, W * width] BF16, from head_exchange.pack_width) and merge locally;
    `rank` is this rank as a one-element device tensor.
    """

    def __init__(self, partial, lse, output, *, runtime, session, coordinator, name, staging=None, rank=None,
                 threads=None, block_limit=None):
        import torch.distributed as dist
        from b12x.comm import pcie
        from b12x.preparation import CollectiveRequirement, PreparedCall
        from ignition.components.collectives import PreparedAllGather, _roce
        from ignition.preparation.tuning import declare

        rows = partial.shape[0]
        self.roce = _roce(runtime)
        if self.roce:
            heads, dim = partial.shape[1:]
            world = runtime.world_size
            width = pack_width(world, heads, dim)
            send, received = (None, None) if staging is None else staging
            if (rank is None or send is None or heads % world or partial.dtype != torch.bfloat16
                    or lse.shape != (rows, heads) or lse.dtype != torch.float32
                    or output.shape != (rows, heads // world, dim) or output.dtype != torch.bfloat16
                    or send.shape[1:] != (width,) or received.shape[1:] != (world * width,)
                    or min(send.shape[0], received.shape[0]) < rows):
                raise ValueError("RoCE head combination requires BF16 partials, FP32 LSE, staging and the rank")
            self.output, self.rank, self.world = output, rank, world
            self.send, self.received = send[:rows], received[:rows]
            self.gather = PreparedAllGather(self.send, self.received, runtime=runtime, session=session,
                                            coordinator=coordinator, name=f"{name}.gather")
            return
        # Partials and outputs have the channel's value width, which a latent query may exceed.
        if (partial.shape != (rows, runtime.total_heads, runtime.head_dim) or partial.dtype != torch.bfloat16
                or lse.shape != (rows, runtime.total_heads) or lse.dtype != torch.float32
                or output.shape != (rows, runtime.heads_per_rank, runtime.head_dim)
                or output.dtype != torch.bfloat16 or not 0 < rows <= runtime.max_batch_size):
            raise ValueError("LSE reduce-scatter requires BF16 partials, FP32 LSE and a head shard of its channel")
        self.runtime, self.output = runtime, output
        call = {"partial_output": partial, "partial_lse": lse, "out": output, "is_lse_base_on_e": True}
        geometry = {key: value for key, value in (("threads", threads), ("block_limit", block_limit))
                    if value is not None}
        query = pcie.query_from_runtime(runtime, surface="DcpAllToAll.lse_reduce_scatter", call={**call, **geometry})
        self.plan = pcie.plan(query, runtime=runtime)
        ranks = tuple(dist.get_process_group_ranks(runtime.exchange_group))
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: state.run(**call), output=output,
                                                    owners=(runtime, partial, lse, output)),
            collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def functional(self, *, session):
        self.output_layout = tuple(self.output.shape), tuple(self.output.stride()), self.output.dtype
        self.output = None
        if self.roce:
            self.send_layout = tuple(self.send.shape), self.send.dtype
            self.send = self.received = None
            self.gather.functional(session=session)
        return self

    def __call__(self, partial, lse):
        output = self.output
        if output is None:
            output = torch.empty_strided(self.output_layout[0], self.output_layout[1], dtype=self.output_layout[2], device=partial.device)
        if self.roce:
            send = self.send
            if send is None:
                send = torch.zeros(self.send_layout[0], dtype=self.send_layout[1], device=partial.device)
            pack(partial, lse, send, self.rank, self.world)
            received = self.gather(send)
            merge(received, partial, lse, output, self.rank, self.world)
            return output
        _lse_reduce_scatter(partial, lse, output, self.handle)
        return output
