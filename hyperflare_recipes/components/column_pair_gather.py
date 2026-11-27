# SPDX-License-Identifier: Apache-2.0

"""Gather BF16 column partitions together with bit-preserved FP32 columns."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ignition.components.collectives import (
    ConcurrentAllGather,
    ConcurrentGatherShards,
    PreparedAllGather,
    PreparedGatherShards,
    _island_row_shard,
)


def _is_dma_runtime(runtime):
    if runtime is None:
        return False
    from b12x.comm import pcie

    return isinstance(runtime, pcie.DmaAllReduce)


def _is_island_gather_runtime(runtime):
    return runtime is not None and _island_row_shard(runtime)


@triton.jit
def _pack_column_pair_kernel(
    First, Second, Send,
    first_stride, second_stride, send_stride,
    FIRST_W: tl.constexpr, SECOND_BF16_W: tl.constexpr, TOTAL_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m_first = cols < FIRST_W
    v_first = tl.load(First + row * first_stride + cols, mask=m_first, other=0)
    tl.store(Send + row * send_stride + cols, v_first, mask=m_first)
    m_second = cols < SECOND_BF16_W
    v_second = tl.load(Second + row * second_stride + cols, mask=m_second, other=0)
    tl.store(Send + row * send_stride + FIRST_W + cols, v_second, mask=m_second)


@triton.jit
def _unpack_dma_column_pair_kernel(
    Received, OutFirst, OutSecond,
    rec_rank_stride, rec_row_stride,
    out1_row_stride, out2_row_stride,
    ROWS,
    FIRST_W: tl.constexpr, SECOND_BF16_W: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
):
    row_tile = tl.program_id(0).to(tl.int64)
    rank = tl.program_id(1).to(tl.int64)
    rows = row_tile * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_C)
    m_row = rows < ROWS
    rec_base = Received + rank * rec_rank_stride + rows[:, None] * rec_row_stride
    m1 = m_row[:, None] & (cols[None, :] < FIRST_W)
    v1 = tl.load(rec_base + cols[None, :], mask=m1, other=0)
    tl.store(OutFirst + rows[:, None] * out1_row_stride + rank * FIRST_W + cols[None, :], v1, mask=m1)
    m2 = m_row[:, None] & (cols[None, :] < SECOND_BF16_W)
    v2 = tl.load(rec_base + FIRST_W + cols[None, :], mask=m2, other=0)
    tl.store(OutSecond + rows[:, None] * out2_row_stride + rank * SECOND_BF16_W + cols[None, :], v2, mask=m2)


def _pack_columns(first, second, send, first_width):
    if first.is_cuda:
        second_bf16 = second.view(torch.bfloat16)
        second_bf16_w = second_bf16.shape[1]
        total_w = send.shape[1]
        block = triton.next_power_of_2(max(first_width, second_bf16_w))
        torch.library.wrap_triton(_pack_column_pair_kernel)[(first.shape[0],)](
            first, second_bf16, send,
            first.stride(0), second_bf16.stride(0), send.stride(0),
            first_width, second_bf16_w, total_w, block,
            num_warps=4,
        )
        return
    send[:, :first_width].copy_(first)
    send[:, first_width:].copy_(second.view(torch.bfloat16))


def _unpack_dma_columns(received, out_first, out_second, world, first_width, second_width):
    count = out_first.shape[0]
    if received.is_cuda:
        out_second_bf16 = out_second.view(torch.bfloat16)
        second_bf16_w = 2 * second_width
        block_r = 4
        block_c = triton.next_power_of_2(max(first_width, second_bf16_w))
        grid = (triton.cdiv(count, block_r), world)
        torch.library.wrap_triton(_unpack_dma_column_pair_kernel)[grid](
            received, out_first, out_second_bf16,
            received.stride(0), received.stride(1),
            out_first.stride(0), out_second_bf16.stride(0),
            count, first_width, second_bf16_w, block_r, block_c,
            num_warps=4,
        )
        return
    rec = received.transpose(0, 1)
    out_first.view(count, world, first_width).copy_(rec[:, :, :first_width])
    out_second.view(count, world, second_width).view(torch.bfloat16).copy_(rec[:, :, first_width:])


@dataclass(frozen=True)
class PendingColumnPairGather:
    chunks: tuple

    @property
    def output(self):
        return self.chunks[-1][2].output


def _validate_storage(first, second, **destinations):
    # Storage identity and byte offsets also describe aliases during metadata-only tracing.
    fields = (("first", first), ("second", second), *destinations.items())
    ranges = [(name, value.untyped_storage()._cdata, value.storage_offset() * value.element_size(),
               (value.storage_offset() + value.numel()) * value.element_size()) for name, value in fields]
    for index, (name, storage, begin, end) in enumerate(ranges):
        for other, other_storage, other_begin, other_end in ranges[max(index + 1, 2):]:
            if storage == other_storage and max(begin, other_begin) < min(end, other_end):
                raise ValueError(f"paired-column live storage overlaps: {name} and {other}")


def validate_columns(first, second, output_first, output_second, *, world, columns, max_rows):
    if type(world) is not int or world < 2 or type(max_rows) is not int or max_rows <= 0:
        raise ValueError("paired column gathering requires multiple ranks and positive row capacity")
    if any(value.ndim != 2 for value in (first, second, output_first, output_second)):
        raise ValueError("paired column gathering requires matrices")
    rows, first_width = first.shape
    second_width = second.shape[1]
    if (min(rows, first_width, second_width) <= 0 or second.shape[0] != rows
            or output_first.shape != (rows, world * first_width)
            or output_second.shape != (rows, world * second_width)
            or columns != first_width + 2 * second_width or columns % 8):
        raise ValueError("paired column dimensions differ from their exchange channel")
    if first.dtype != torch.bfloat16 or output_first.dtype != torch.bfloat16:
        raise TypeError("first paired columns require BF16 storage")
    if second.dtype != torch.float32 or output_second.dtype != torch.float32:
        raise TypeError("second paired columns require FP32 storage")
    if any(value.device != first.device or not value.is_contiguous()
           for value in (first, second, output_first, output_second)):
        raise ValueError("paired columns require contiguous storage on one device")
    _validate_storage(first, second, output_first=output_first, output_second=output_second)


class PreparedColumnPairGather:
    """Pack two column shards into bounded caller-owned send/receive buffers.

    FP32 columns use two BF16 transport lanes without numerical conversion.
    Each chunk is consumed before another overwrites the shared transport buffers.
    Outputs and transport buffers must be disjoint from each other and both inputs.
    """

    def __init__(self, first, second, output_first, output_second, *, runtime,
                 session, coordinator, name, buffers, allocator=None, concurrent=False):
        world = runtime.world_size
        self.dma = _is_dma_runtime(runtime)
        self.island = _is_island_gather_runtime(runtime)
        self.rank_major = self.dma or self.island
        if self.dma:
            columns = first.shape[1] + 2 * second.shape[1]
            max_rows = runtime.shard_capacity // (columns * 2) if columns > 0 else 0
        elif self.island:
            columns = first.shape[1] + 2 * second.shape[1]
            max_rows = (runtime._island_rs.max_elements // world) // columns if columns > 0 else 0
        else:
            columns = runtime.heads_per_rank * runtime.query_head_dim
            max_rows = runtime.max_batch_size
        validate_columns(first, second, output_first, output_second, world=world,
                         columns=columns, max_rows=max_rows)
        self.first, self.second = first, second
        self.output_first, self.output_second = output_first, output_second
        self.world, self.columns = world, columns
        self.first_width, self.second_width = first.shape[1], second.shape[1]
        chunk_rows = min(first.shape[0], max_rows)

        def allocate(field, width):
            value = buffers.get(field)
            if value is not None and (value.ndim != 2 or value.shape[1] != width
                    or value.dtype != torch.bfloat16 or value.device != first.device or not value.is_contiguous()):
                raise ValueError("shared paired-column buffers have incompatible geometry")
            if value is None or value.shape[0] < chunk_rows:
                shape = (chunk_rows, width)
                value = (torch.zeros(shape, dtype=torch.bfloat16, device=first.device) if allocator is None
                         else allocator.zeros(shape, torch.bfloat16))
                buffers[field] = value
            if (value.shape[1:] != (width,) or value.dtype != torch.bfloat16
                    or value.device != first.device or not value.is_contiguous()):
                raise ValueError("shared paired-column buffers have incompatible geometry")
            return value[:chunk_rows]

        self.send = allocate("send", columns)
        if self.rank_major:
            value = buffers.get("received_dma")
            if value is not None and (value.ndim != 3 or value.shape[0] != world or value.shape[2] != columns
                    or value.dtype != torch.bfloat16 or value.device != first.device or not value.is_contiguous()):
                raise ValueError("shared paired-column buffers have incompatible geometry")
            if value is None or value.shape[1] < chunk_rows:
                shape = (world, chunk_rows, columns)
                value = (torch.zeros(shape, dtype=torch.bfloat16, device=first.device) if allocator is None
                         else allocator.zeros(shape, torch.bfloat16))
                buffers["received_dma"] = value
            self.received = value.view(-1)[:world * chunk_rows * columns].view(world, chunk_rows, columns)
        else:
            self.received = allocate("received", world * columns)
        _validate_storage(first, second, output_first=output_first, output_second=output_second,
                          send=self.send, received=self.received)
        self.bounds = tuple((begin, min(begin + chunk_rows, first.shape[0]))
                            for begin in range(0, first.shape[0], chunk_rows))
        if self.dma:
            self.parts = tuple(
                PreparedAllGather(
                    self.send[:end - begin],
                    self.received.view(-1)[:world * (end - begin) * columns].view(world, end - begin, columns),
                    runtime=runtime, session=session, coordinator=coordinator, name=f"{name}.{index}",
                )
                for index, (begin, end) in enumerate(self.bounds)
            )
        elif self.island:
            self.parts = tuple(
                PreparedGatherShards(
                    self.send[:end - begin],
                    self.received.view(-1)[:world * (end - begin) * columns].view(world * (end - begin), columns),
                    runtime=runtime, session=session, coordinator=coordinator, name=f"{name}.{index}",
                    concurrent=concurrent,
                )
                for index, (begin, end) in enumerate(self.bounds)
            )
        else:
            self.parts = tuple(PreparedAllGather(self.send[:end - begin], self.received[:end - begin],
                runtime=runtime, session=session, coordinator=coordinator, name=f"{name}.{index}")
                for index, (begin, end) in enumerate(self.bounds))
        self.concurrent = None

    def enable_concurrent(self, *, workspace, session, parallel_plans, send_buffer=None):
        if send_buffer is not None and len(self.parts) != 1:
            raise ValueError("shared send_buffer requires a single paired-gather chunk")
        wrapper = ConcurrentGatherShards if self.island else ConcurrentAllGather
        self.concurrent = tuple(
            wrapper(
                part, workspace=workspace, session=session,
                parallel_plans=parallel_plans, send_buffer=send_buffer,
            )
            for part in self.parts
        )
        return self

    def functional(self, *, session):
        self.rows = self.first.shape[0]
        self.chunk_rows = self.send.shape[0]
        for part in self.parts:
            part.functional(session=session)
        if self.concurrent is not None:
            for item in self.concurrent:
                item.functional(session=session)
        self.first = self.second = self.output_first = self.output_second = self.send = self.received = None
        return self

    def start(self, first, second, *, dependencies=()):
        if self.concurrent is None:
            raise ValueError("paired gather concurrency was not enabled")
        pending = []
        deps = tuple(dependencies)
        for (begin, end), item in zip(self.bounds, self.concurrent, strict=True):
            count = end - begin
            send = torch.empty((count, self.columns), dtype=torch.bfloat16, device=first.device)
            _pack_columns(first[begin:end], second[begin:end], send, self.first_width)
            if self.island:
                token = item.start(send, dependencies=deps, clone=False)
            else:
                token = item.start(send, dependencies=deps)
            pending.append((begin, end, token))
            deps = (token.output,)
        return PendingColumnPairGather(tuple(pending))

    def finish(self, pending, dependency):
        output_first = torch.empty((self.rows, self.world * self.first_width),
                                   dtype=torch.bfloat16, device=dependency.device)
        output_second = torch.empty((self.rows, self.world * self.second_width),
                                    dtype=torch.float32, device=dependency.device)
        dep = dependency
        for (begin, end, token), item in zip(pending.chunks, self.concurrent, strict=True):
            count = end - begin
            raw = item.finish(token, dep)
            if self.rank_major:
                _unpack_dma_columns(
                    raw.view(self.world, count, self.columns),
                    output_first[begin:end], output_second[begin:end],
                    self.world, self.first_width, self.second_width,
                )
            else:
                received = raw.view(count, self.world, self.columns)
                output_first[begin:end].view(count, self.world, self.first_width).copy_(
                    received[:, :, :self.first_width])
                output_second[begin:end].view(count, self.world, self.second_width).view(torch.bfloat16).copy_(
                    received[:, :, self.first_width:])
            dep = raw
        return output_first, output_second

    def __call__(self, first=None, second=None, *, dependencies=()):
        local = self.first is None
        if local:
            if first is None or second is None:
                raise ValueError("functional paired gather requires both column shards")
            output_first = torch.empty((self.rows, self.world * self.first_width), dtype=first.dtype, device=first.device)
            output_second = torch.empty((self.rows, self.world * self.second_width), dtype=second.dtype, device=second.device)
        else:
            first, second = self.first, self.second
            output_first, output_second = self.output_first, self.output_second
        deps = tuple(dependencies)
        for (begin, end), gather in zip(self.bounds, self.parts, strict=True):
            count = end - begin
            send = (torch.empty((count, self.columns), dtype=torch.bfloat16, device=first.device)
                    if local else self.send[:count])
            _pack_columns(first[begin:end], second[begin:end], send, self.first_width)
            raw = gather(send, dependencies=deps) if deps else gather(send)
            deps = (raw,) if deps else ()
            if self.rank_major:
                _unpack_dma_columns(
                    raw.view(self.world, count, self.columns),
                    output_first[begin:end], output_second[begin:end],
                    self.world, self.first_width, self.second_width,
                )
            else:
                received = raw.view(count, self.world, self.columns)
                output_first[begin:end].view(count, self.world, self.first_width).copy_(received[:, :, :self.first_width])
                output_second[begin:end].view(count, self.world, self.second_width).view(torch.bfloat16).copy_(
                    received[:, :, self.first_width:])
        return output_first, output_second
