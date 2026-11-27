# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""File-backed FP8 n-gram tables, compact GPU inputs, and residual injection."""

from dataclasses import dataclass
from functools import cached_property

import torch
from b12x.norm import hyperconnection
from b12x.preparation import PreparedCall
from b12x.sequence import engram

from ignition.components.engram import EngramInputs, PreparedEngramLookup, PreparedEngramRequests
from ignition.components.readiness import ReadinessTimings
from hyperflare_recipes.components.embedding_partition import EmbeddingHeadPartition
from hyperflare_recipes.components.linear import QuantizedProjection
from ignition.components.collectives import PreparedAllGather, PreparedChunkedAllReduce
from ignition.components.state import gather_rows, save_hidden
from ignition.preparation.runtime_core import FileSource
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.tuning import declare
from ignition.progress import phase


@dataclass(frozen=True)
class Tables:
    geometry: object
    token_map: tuple[int, ...]
    sources: dict
    inputs: dict
    mailboxes: dict
    sequence: torch.Tensor
    shard: object
    row_mapping: dict
    columns: torch.Tensor
    readiness_runtime: object = None

    @classmethod
    def prepare(cls, manifest, config, *, tokens, device, sequence, frame, host_allocator, progress=None, shard=None):
        from transformers import PreTrainedTokenizerFast
        from hyperflare_recipes.components.parallel import TensorParallelShard

        shard = shard or TensorParallelShard(0, 1)

        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(manifest.metadata_path / "tokenizer.json"))
        token_map, size = engram.build_compressed_token_map(tokenizer)
        text = config.text
        if size != text["engram_compressed_vocab_size"]:
            raise ValueError(f"Engram compressed vocabulary differs: {size}")
        geometry = engram.build_geometry(layer_ids=tuple(text["engram_layer_ids"]),
            base_table_size=text["engram_vocab_size"], compressed_vocab_size=size)
        if geometry.num_embeddings != tuple(text["engram_num_embeddings"]):
            raise ValueError("Engram hash geometry differs from checkpoint tables")
        sources, inputs, mailboxes, row_mapping = {}, {}, {}, {}
        timings = ReadinessTimings.allocate(frame, len(geometry.layer_ids))
        with phase(progress, "Prepare embedding tables") as stage:
            for index, layer in enumerate(geometry.layer_ids):
                rows = geometry.num_embeddings[index]
                pair = []
                for suffix, columns, dtype in (("weight", 256, "F8_E4M3"), ("scale", 8, "F8_E8M0")):
                    name = f"layers.{layer}.engram.embed.{suffix}"
                    metadata = manifest.tensors[name]
                    if metadata.shape != (rows, columns) or metadata.dtype != dtype:
                        raise ValueError(f"{name}: invalid Engram table geometry or dtype")
                    pair.append(FileSource(manifest.snapshot / metadata.shard, metadata.offset, rows * columns))
                sources[layer] = tuple(pair)
                partition = EmbeddingHeadPartition.for_rank(geometry.offsets[index], rows,
                                                            shard.rank, shard.world_size)
                row_mapping[layer] = torch.tensor(partition.row_mapping, dtype=torch.int64, device=device)
                inputs[layer] = EngramInputs.allocate(tokens, device, timings=timings, interval=index)
                mailboxes[layer] = host_allocator(f"engram_{layer}_requests", (tokens * 24 + 1,), torch.int64)
                mailboxes[layer].host_view.zero_()
            stage.update(f"{sum(source.nbytes for pair in sources.values() for source in pair) / (1 << 30):.1f} GiB in checkpoint spans")
        columns = persistent_copy(torch.tensor([head * 256 + column for head in range(shard.rank, 24, shard.world_size)
                                for column in range(256)], dtype=torch.int64, device=device))
        return cls(geometry, tuple(token_map), sources, inputs, mailboxes, sequence, shard, row_mapping, columns)

    @cached_property
    def device_token_map(self):
        device = next(iter(self.inputs.values())).hashes.device
        return persistent_copy(torch.tensor(self.token_map, dtype=torch.int64, device=device))

    def buffers(self):
        return {**{name: tensor for layer, inputs in self.inputs.items() for name, tensor in inputs.buffers(layer).items()},
                **{f"engram_{layer}_requests": mailbox.device_view for layer, mailbox in self.mailboxes.items()},
                **next(iter(self.inputs.values())).timings.buffers()}

    def events(self):
        return {f"engram_{layer}_ready": inputs.ready.cuda_event for layer, inputs in self.inputs.items()}

    def resident_catalog(self):
        """Source spans and native bindings for shared RAM delivery without core export."""
        return {"vocabulary": len(self.token_map), "tables": [
            {"planes": [{"segments": [
                            {"path": str(source.path), "offset": source.offset + first * row_bytes,
                             "nbytes": (last - first) * row_bytes}
                            for first, last in partition.source_rows], "row_bytes": row_bytes}
                        for source, row_bytes in zip(self.sources[layer], (256, 8), strict=True)],
             "heads": 24, "weights": f"engram_{layer}_weights", "scales": f"engram_{layer}_scales",
             "hashes": f"engram_{layer}_hashes", "event": f"engram_{layer}_ready",
             "request": f"engram_{layer}_requests"}
            for ordinal, layer in enumerate(self.geometry.layer_ids)
            for partition in (self.partition(ordinal, self.shard.rank),)]}

    def partition(self, ordinal, rank):
        return EmbeddingHeadPartition.for_rank(self.geometry.offsets[ordinal],
            self.geometry.num_embeddings[ordinal], rank, self.shard.world_size)

    def export_metadata(self):
        """Describe table geometry and checkpoint spans for native publication."""
        tables = []
        for ordinal, layer in enumerate(self.geometry.layer_ids):
            table = dict(layer=layer, rows=self.geometry.num_embeddings[ordinal],
                embedding_width=256, scale_block=32, encoding="FP8_E4M3_E8M0",
                weight_buffer=f"engram_{layer}_weights", scale_buffer=f"engram_{layer}_scales",
                hash_buffer=f"engram_{layer}_hashes", ready_event=f"engram_{layer}_ready",
                request_buffer=f"engram_{layer}_requests")
            offsets = [0] * 24
            first = 0
            shards = []
            for rank in range(self.shard.world_size):
                partition = self.partition(ordinal, rank)
                for head in partition.heads:
                    offsets[head] = first + partition.row_mapping[head][2]
                planes = dict(rank=rank, first_row=first, rows=partition.rows)
                for name, source, width in zip(("weights", "scales"), self.sources[layer], (256, 8), strict=True):
                    planes[name] = [dict(path=str(source.path), offset=source.offset + start * width,
                                         nbytes=(end - start) * width)
                                    for start, end in partition.source_rows]
                if self.shard.world_size == 1:
                    table.update(weights=planes["weights"], scales=planes["scales"])
                else:
                    shards.append(planes)
                first += partition.rows
            table["hash"] = dict(method="XOR_MODULO", orders=[order for order in (2, 3, 4) for _ in range(8)],
                multipliers=list(self.geometry.multipliers[ordinal]), primes=list(self.geometry.primes[ordinal]),
                offsets=offsets)
            if shards:
                table["shards"] = shards
            tables.append(table)
        return dict(token_map=list(self.token_map), pad_token_id=2,
                    compressed_vocabulary=self.geometry.compressed_vocab_size, tables=tables)


class PreparedTableRequests:
    def __init__(self, tables, metadata, pool, *, session, name):
        self.pool = pool
        history = torch.zeros((metadata.starts.numel(), 3), dtype=torch.int64, device=metadata.starts.device)
        self.token_map = tables.device_token_map
        self.operations = tuple(PreparedEngramRequests(inputs, metadata.token_ids, metadata.query_start_loc,
            history, metadata.live_counts, tables.mailboxes[layer].device_view, tables.sequence,
            layer=layer, token_map=tables.token_map, geometry=tables.geometry, session=session,
            name=f"{name}.{layer}", row_mapping=tables.row_mapping[layer]).functional()
            for layer, inputs in tables.inputs.items())

    def __call__(self, metadata):
        m = metadata
        raw_history = torch.zeros((m.starts.numel(), 3), dtype=torch.int64, device=m.starts.device)
        gather_rows(self.pool.region("engram_history"), m.state_rows, raw_history)
        columns = torch.arange(3, device=m.starts.device)
        raw = torch.where(columns >= 3 - m.starts[:, None], raw_history, 2)
        # Image token types stop n-grams at image boundaries, including across prefill chunks.
        text = (raw >= 0) & (raw < self.token_map.numel())
        history = torch.where(text, self.token_map[raw.clamp(0, self.token_map.numel() - 1)], -1)
        for operation in self.operations:
            operation(m.token_ids, m.query_start_loc, history, m.live_counts)
        return raw_history

    def commit(self, counts, raw_history, metadata):
        m = metadata
        columns = torch.arange(3, device=m.starts.device)
        relative = counts[:, None].long() + columns - 3
        indices = (m.query_start_loc[:-1, None] + relative).clamp(0, m.token_ids.numel() - 1)
        history = torch.where(relative >= 0, m.token_ids[indices],
                              raw_history.gather(1, (relative + 3).clamp(0, 2)))
        save_hidden(history, self.pool.region("engram_history"), m.state_rows, counts > 0)


@dataclass(frozen=True)
class EngramScratch:
    """Intermediate lookup and projection storage for serialized Engram layers."""

    mask: torch.Tensor
    gathered: torch.Tensor
    local: torch.Tensor
    partial: torch.Tensor
    projected: torch.Tensor

    @classmethod
    def allocate(cls, tokens, device, *, world=1):
        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device)
        gathered = zeros((tokens, 6144), torch.bfloat16)
        local = gathered if world == 1 else zeros((tokens, 6144 // world), torch.bfloat16)
        partial = zeros((tokens, 25600), torch.bfloat16)
        projected = partial if world == 1 else zeros((tokens, 25600), torch.bfloat16)
        return cls(zeros(tokens, torch.bool), gathered, local, partial, projected)


class PreparedEngram:
    def __init__(self, layer, weights, tables, source, output, metadata, pool, *, scratch, session, name, workspace,
                 runtime=None, coordinator=None, examples=None, defer_reduce_functional=False, row_shard=None,
                 min_gather_chunks: int = 1):
        self.pool = pool
        tokens, device = metadata.token_ids.numel(), source.device
        self.mask, self.gathered, self.projected = scratch.mask, scratch.gathered, scratch.projected
        self.local, self.partial = scratch.local, scratch.partial
        self.columns = tables.columns
        self.sharded = tables.shard.world_size > 1
        if self.sharded and runtime is None:
            raise ValueError("sharded Engram projection requires a collective runtime")
        for tensor, shape, dtype in ((self.mask, (tokens,), torch.bool),
                (self.gathered, (tokens, 6144), torch.bfloat16),
                (self.local, (tokens, 6144 // tables.shard.world_size), torch.bfloat16),
                (self.partial, (tokens, 25600), torch.bfloat16),
                (self.projected, (tokens, 25600), torch.bfloat16)):
            if tensor.shape != shape or tensor.dtype != dtype or tensor.device != device or not tensor.is_contiguous():
                raise ValueError("Engram scratch must match the contiguous lookup and projection geometry")
        self.lookup = PreparedEngramLookup(tables.inputs[layer], self.gathered, metadata.live_counts[:1],
            layer=layer, token_map=tables.token_map, geometry=tables.geometry,
            session=session, name=name + ".lookup", runtime=tables.readiness_runtime, coordinator=coordinator).functional(session=session)
        from b12x.comm import pcie
        # For prefills >= 512 tokens on local PCIe DMA runtimes, gather the 6144-col
        # Engram activations first and run one full projection (`projection_full`).
        # Note: the full projection accumulates across all 24 heads in FP32 before a
        # single BF16 round, whereas the sharded <512-token path rounds each rank's
        # partial sum to BF16 before the all-reduce. The two paths share identical
        # weights/precision but are not bit-identical across the 512-token boundary.
        self.use_all_gather = (
            self.sharded
            and isinstance(runtime, pcie.DmaAllReduce)
            and "projection_full" in weights
            and tokens >= 512
        )
        self.row_shard = True if (row_shard is not None and self.use_all_gather) else None
        if self.row_shard is not None:
            rank, half_tokens, shard_rows = row_shard
            if 2 * half_tokens != tokens or shard_rows <= 0 or rank < 0 or (rank + 1) * shard_rows > half_tokens:
                raise ValueError("Engram row shard must partition both microbatch halves within the chunk")
            s0 = rank * shard_rows
            self.row_shard_indices = persistent_copy(torch.cat((
                torch.arange(s0, s0 + shard_rows, dtype=torch.int64, device=device),
                torch.arange(half_tokens + s0, half_tokens + s0 + shard_rows, dtype=torch.int64, device=device),
            )))
            mix_tokens = 2 * shard_rows
        else:
            self.row_shard_indices = None
            mix_tokens = tokens
        self.gather_parts = ()
        self.gather_bounds = ()
        if self.use_all_gather:
            if type(min_gather_chunks) is not int or min_gather_chunks < 1:
                raise ValueError("Engram min_gather_chunks must be a positive integer")
            world = tables.shard.world_size
            local_cols = 6144 // world
            max_rows = runtime.shard_capacity // (local_cols * 2)
            if max_rows < 1:
                raise ValueError("DMA runtime shard capacity is too small for Engram all-gather")
            chunks = max(min_gather_chunks, (tokens + max_rows - 1) // max_rows)
            step = min(max_rows, (tokens + chunks - 1) // chunks)
            self.gather_bounds = tuple((start, min(start + step, tokens)) for start in range(0, tokens, step))
            self.gather_parts = tuple(
                PreparedAllGather(
                    self.local[begin:end],
                    self.gathered[begin:end].view(world, end - begin, local_cols),
                    runtime=runtime, session=session, coordinator=coordinator,
                    name=name + ".gather" if len(self.gather_bounds) == 1 else f"{name}.gather.{index}",
                )
                for index, (begin, end) in enumerate(self.gather_bounds)
            )
            if not defer_reduce_functional:
                for part in self.gather_parts:
                    part.functional(session=session)
            self.projection = QuantizedProjection(
                self.gathered[:mix_tokens], weights["projection_full"], self.projected[:mix_tokens],
                session=session, name=name + ".projection", workspace=workspace, activation_mode="auto",
                examples=examples, example_role="engram.projection").functional(session=session)
            self.reduce = None
        else:
            self.projection = QuantizedProjection(self.local, weights["projection"], self.partial,
                session=session, name=name + ".projection", workspace=workspace, activation_mode="auto",
                examples=examples, example_role="engram.projection").functional(session=session)
            self.reduce = (PreparedChunkedAllReduce(self.partial.view(-1, 5120), self.projected.view(-1, 5120), runtime=runtime,
                session=session, coordinator=coordinator, name=name + ".reduce") if self.sharded else None)
            if self.reduce is not None and not defer_reduce_functional:
                self.reduce.functional(session=session)
        self.norm_weight = weights["weight"].view(-1)
        self.eps = pool.config.text["rms_norm_eps"]
        self.mix_plan = hyperconnection.plan(hyperconnection.Caps(device=device, max_tokens=mix_tokens, hidden_size=5120),
            invocation={"operation": "engram_mix", "eps": self.eps, "token_mask": True})
        projected, mask = self.projected[:mix_tokens], self.mask[:mix_tokens]
        declare(session, (self.mix_plan.request(name=name + ".mix", prepare_call=lambda state: PreparedCall(
            run=lambda: state.launch(source[:mix_tokens].view(mix_tokens, -1), projected, self.norm_weight,
                                      mask, output[:mix_tokens].view(mix_tokens, -1), eps=self.eps))),))
        self.mask = self.gathered = self.projected = self.local = self.partial = None

    def lookup_local(self, source, metadata):
        gathered = self.lookup(source, metadata.live_counts[:1])
        return torch.index_select(gathered, 1, self.columns) if self.sharded else gathered

    def project_gathered(self, gathered_chunks):
        reordered = [chunk.permute(1, 0, 2).reshape(chunk.shape[1], 6144) for chunk in gathered_chunks]
        full = reordered[0].contiguous() if len(reordered) == 1 else torch.cat(reordered, dim=0)
        if self.row_shard_indices is not None:
            full = torch.index_select(full, 0, self.row_shard_indices)
        return self.projection(full)

    def project(self, source, metadata):
        local = self.lookup_local(source, metadata)
        return self.projection(local)

    def mix(self, source, projected, metadata):
        mask = (metadata.positions >= 0) & (metadata.token_ids < 129280)
        if self.row_shard_indices is not None:
            mask = torch.index_select(mask, 0, self.row_shard_indices)
        output = torch.empty_like(source)
        hyperconnection.run_engram_mix(source.view(source.shape[0], -1), projected, self.norm_weight,
            eps=self.eps, plan=self.mix_plan, out=output.view(source.shape[0], -1), token_mask=mask)
        return output

    def __call__(self, source, metadata):
        if self.use_all_gather:
            local = self.lookup_local(source, metadata)
            chunks = [part(local[begin:end]) for (begin, end), part in zip(self.gather_bounds, self.gather_parts, strict=True)]
            projected = self.project_gathered(chunks)
            return self.mix(source, projected, metadata)
        projected = self.project(source, metadata)
        if self.sharded:
            projected = self.reduce(projected.view(-1, 5120)).view(source.shape[0], 25600)
        return self.mix(source, projected, metadata)
