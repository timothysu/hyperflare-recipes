# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Windowed MLA, source-shared compression, and two-level sparse selection."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from b12x.quantization import mla_norm_mxfp8, mxfp8
import triton
import triton.language as tl
from b12x.attention import mla_compress
from b12x.preparation import PreparedCall

from hyperflare_recipes.components import sm103_projection
from hyperflare_recipes.components.compressed_mla import PreparedCompressedCacheWriter, PreparedCompressedMLA
from hyperflare_recipes.components.attention_output import PreparedGroupedOutputProjection
from hyperflare_recipes.components.linear import QuantizedProjection, UnquantizedProjection
from hyperflare_recipes.components.mla_norm_quant import PreparedMLANormQuant
from hyperflare_recipes.components.mxfp4_indexer import PreparedMXFP4Indexer
from hyperflare_recipes.components.projection_overlap import (
    ConcurrentAttentionAux,
    register_tc_gemv_rope,
    run_mxfp8_projection_rope,
)
from hyperflare_recipes.components.rms_norm import PreparedRMSNorm
from ignition.components.state import gather_rows, gather_rows_three, save_hidden
from hyperflare_recipes.components.window_indices import PreparedWindowIndices, WindowIndices
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare, register_finalizer
from ignition.preparation.scratch import ScratchArena

from hyperflare_recipes.components.rotary import CompressedRotary as Rotary, PreparedRotaryAngles
from hyperflare_recipes.components.parallel import TensorParallelShard

from .pool import PAGE_TOKENS


# Bound indexer workspace while amortizing prefill selection launches.
INDEXER_QUERY_ROWS = 512


@dataclass(frozen=True)
class ShardAttentionInputs:
    """Rank-local slices of shared encoder attention geometry across microbatch halves."""

    window_indices: tuple[torch.Tensor, torch.Tensor]
    window_lengths: tuple[torch.Tensor, torch.Tensor]
    angles: dict[tuple[bool, int], tuple[torch.Tensor, torch.Tensor]]
    index_lengths: tuple[torch.Tensor, torch.Tensor] | None = None
    compressed_lengths: tuple[torch.Tensor, torch.Tensor] | None = None
    page_table: tuple[torch.Tensor, torch.Tensor] | None = None
    active_width_a: torch.Tensor | None = None


@dataclass(frozen=True)
class CompressionGeometry:
    slots: torch.Tensor
    lengths: torch.Tensor
    index_lengths: torch.Tensor
    active_width: torch.Tensor
    position: torch.Tensor


@dataclass(frozen=True)
class AttentionGeometry(WindowIndices):
    position: torch.Tensor
    ratios: dict
    angles: dict = None

    def compressed(self, ratio):
        return self.ratios[ratio]


class AttentionIndices(PreparedWindowIndices):
    """Window, compression, and shared rotary angles for layers with identical metadata."""

    def __init__(self, metadata, *, mode, tokens, page_size, width, ratios=(), visible=None,
                 replay_start=None):
        super().__init__(metadata, mode=mode, tokens=tokens, page_size=page_size, width=width,
                         visible=visible, replay_start=replay_start)
        self.ratio_ids = tuple(sorted(set(ratios)))
        if any(type(ratio) is not int or ratio not in (1, 2) for ratio in self.ratio_ids):
            raise ValueError("DeepSeek V4.1 compression ratios must be one or two")
        self.position = torch.zeros_like(metadata.positions)
        self.ratios = {ratio: CompressionGeometry(
            slots=torch.full_like(metadata.positions, -1),
            lengths=torch.zeros(tokens, dtype=torch.int32, device=metadata.positions.device),
            index_lengths=torch.zeros(tokens, dtype=torch.int32, device=metadata.positions.device),
            active_width=torch.zeros(1, dtype=torch.int32, device=metadata.positions.device),
            position=torch.zeros_like(metadata.positions)) for ratio in self.ratio_ids}
        self.angles = {}
        self.token_offsets = persistent_copy(torch.arange(tokens, dtype=torch.int64, device=metadata.positions.device))
        record_numeric_tensor(self.token_offsets)

    def rotary_angles(self, rotary, *, compressed, ratio=0):
        if rotary is None:
            return None
        key = (bool(compressed), int(ratio))
        if key not in self.angles:
            positions = self.position if ratio == 0 else self.compressed(ratio).position
            self.angles[key] = PreparedRotaryAngles(positions, rotary.frequencies[bool(compressed)], split=True)
        return self.angles[key]

    def compressed(self, ratio):
        if ratio not in self.ratios:
            raise ValueError("compression geometry must be declared before preparing attention")
        return self.ratios[ratio]

    def functional(self):
        super().functional()
        self.position = self.ratios = None
        for operation in self.angles.values():
            operation.functional()
        return self

    def __call__(self, metadata, visible=None, replay_start=None):
        window = super().__call__(metadata, visible=visible, replay_start=replay_start)
        m = metadata
        live, position = m.positions >= 0, m.positions.clamp_min(0)
        ratios = {}
        if self.ratio_ids:
            pages = m.page_table[m.requests, (position // PAGE_TOKENS).clamp(max=m.page_table.shape[1] - 1)]
        for ratio in self.ratio_ids:
            lengths = torch.where(live, (position + 1) // ratio, 0).to(torch.int32)
            ratios[ratio] = CompressionGeometry(lengths=lengths, index_lengths=lengths.clamp(max=512),
                active_width=lengths.amax().view(1),
                slots=torch.where(live, pages.long() * (PAGE_TOKENS // ratio) + (position % PAGE_TOKENS) // ratio, -1),
                position=(position + 1) // ratio * ratio - ratio)
        angles = {key: operation(position if key[1] == 0 else ratios[key[1]].position)
                  for key, operation in self.angles.items()}
        return AttentionGeometry(window.window_slots, window.window_indices, window.window_lengths,
                                 window.window_page_size, position, ratios, angles)


def normalize(source, weight, out, eps):
    value = source.float()
    out.copy_((value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)) * weight.float())


@triton.jit
def _q_kv_norm_rope_kernel(X, QW, KW, ANGLES, Q_OUT, KV_OUT, EPS: tl.constexpr):
    """Fused Q RMSNorm (1280) + KV RMSNorm (512) + KV RoPE (last 64), bit-exact BF16 seam."""
    row = tl.program_id(0).to(tl.int64)
    task = tl.program_id(1)
    if task == 0:
        qc = tl.arange(0, 2048)
        qm = qc < 1280
        qx = tl.load(X + row * 1792 + qc, mask=qm, other=0.0).to(tl.float32)
        qw = tl.load(QW + qc, mask=qm, other=0.0).to(tl.float32)
        qinv = tl.rsqrt(tl.sum(qx * qx, 0) / 1280.0 + EPS)
        tl.store(Q_OUT + row * 1280 + qc, (qx * qinv * qw).to(tl.bfloat16), mask=qm)
    else:
        kc = tl.arange(0, 512)
        kx = tl.load(X + row * 1792 + 1280 + kc).to(tl.float32)
        kw = tl.load(KW + kc).to(tl.float32)
        kinv = tl.rsqrt(tl.sum(kx * kx, 0) / 512.0 + EPS)
        tl.store(KV_OUT + row * 512 + kc, (kx * kinv * kw).to(tl.bfloat16), mask=kc < 448)
        pair = tl.arange(0, 32)
        xa = tl.load(X + row * 1792 + 1280 + 448 + 2 * pair).to(tl.float32)
        xb = tl.load(X + row * 1792 + 1280 + 448 + 2 * pair + 1).to(tl.float32)
        wa = tl.load(KW + 448 + 2 * pair).to(tl.float32)
        wb = tl.load(KW + 448 + 2 * pair + 1).to(tl.float32)
        a = (xa * kinv * wa).to(tl.bfloat16).to(tl.float32)
        b = (xb * kinv * wb).to(tl.bfloat16).to(tl.float32)
        cos = tl.load(ANGLES + row * 64 + pair)
        sin = tl.load(ANGLES + row * 64 + 32 + pair)
        tl.store(KV_OUT + row * 512 + 448 + 2 * pair, (a * cos - b * sin).to(tl.bfloat16))
        tl.store(KV_OUT + row * 512 + 448 + 2 * pair + 1, (a * sin + b * cos).to(tl.bfloat16))


def _q_kv_norm_rope(projected, qw, kw, angles, eps):
    rows = projected.shape[0]
    query_low = torch.empty((rows, 1280), dtype=torch.bfloat16, device=projected.device)
    window_values = torch.empty((rows, 512), dtype=torch.bfloat16, device=projected.device)
    torch.library.wrap_triton(_q_kv_norm_rope_kernel)[(rows, 2)](
        projected, qw, kw, angles, query_low, window_values, eps, num_warps=4)
    return query_low, window_values


@triton.jit
def _index_norm_and_rotate_kernel(KEYS_IN, KW, ANGLES, KEYS_OUT, LATENT, EPS: tl.constexpr):
    """Fused index_k RMSNorm (128) + index_k RoPE (last 64) + latent RoPE (last 64)."""
    row = tl.program_id(0).to(tl.int64)
    task = tl.program_id(1)
    pair = tl.arange(0, 32)
    cos = tl.load(ANGLES + row * 64 + pair)
    sin = tl.load(ANGLES + row * 64 + 32 + pair)
    if task == 0:
        kc = tl.arange(0, 128)
        kx = tl.load(KEYS_IN + row * 128 + kc).to(tl.float32)
        kw = tl.load(KW + kc).to(tl.float32)
        kinv = tl.rsqrt(tl.sum(kx * kx, 0) / 128.0 + EPS)
        tl.store(KEYS_OUT + row * 128 + kc, (kx * kinv * kw).to(tl.bfloat16), mask=kc < 64)
        xa = tl.load(KEYS_IN + row * 128 + 64 + 2 * pair).to(tl.float32)
        xb = tl.load(KEYS_IN + row * 128 + 64 + 2 * pair + 1).to(tl.float32)
        wa = tl.load(KW + 64 + 2 * pair).to(tl.float32)
        wb = tl.load(KW + 64 + 2 * pair + 1).to(tl.float32)
        ka = (xa * kinv * wa).to(tl.bfloat16).to(tl.float32)
        kb = (xb * kinv * wb).to(tl.bfloat16).to(tl.float32)
        tl.store(KEYS_OUT + row * 128 + 64 + 2 * pair, (ka * cos - kb * sin).to(tl.bfloat16))
        tl.store(KEYS_OUT + row * 128 + 64 + 2 * pair + 1, (ka * sin + kb * cos).to(tl.bfloat16))
    else:
        base = LATENT + row * 512 + 448 + 2 * pair
        la = tl.load(base).to(tl.float32)
        lb = tl.load(base + 1).to(tl.float32)
        tl.store(base, (la * cos - lb * sin).to(tl.bfloat16))
        tl.store(base + 1, (la * sin + lb * cos).to(tl.bfloat16))


def _index_norm_and_rotate(keys_raw, kw, latent, angles, eps):
    rows = keys_raw.shape[0]
    keys = torch.empty((rows, 128), dtype=torch.bfloat16, device=keys_raw.device)
    torch.library.wrap_triton(_index_norm_and_rotate_kernel)[(rows, 2)](
        keys_raw, kw, angles, keys, latent, eps, num_warps=4)
    return keys


@dataclass(frozen=True)
class AttentionScratch:
    projected: torch.Tensor
    query_low: torch.Tensor
    query: torch.Tensor
    window_values: torch.Tensor
    attended: torch.Tensor
    output: torch.Tensor
    selected: torch.Tensor
    candidates: torch.Tensor
    candidate_counts: torch.Tensor
    index_query: torch.Tensor
    index_weights: torch.Tensor
    index_keys: torch.Tensor
    normalized_keys: torch.Tensor

    @classmethod
    def allocate(cls, tokens, world, device, *, candidate_tokens=None):
        candidate_tokens = tokens if candidate_tokens is None else candidate_tokens
        if type(candidate_tokens) is not int or not 0 < candidate_tokens <= tokens:
            raise ValueError("candidate row capacity must fit the attention token capacity")
        def zeros(shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)
        first, last = TensorParallelShard(0, world).padded_partition(8)
        heads = (last - first) * 8
        return cls(zeros((tokens, 1792)),
            zeros((tokens, 1280)), zeros((tokens, heads, 512)),
            zeros((tokens, 512)), zeros((tokens, heads, 512)),
            zeros((tokens, 5120)), zeros((tokens, 512), torch.int32).fill_(-1),
            zeros((candidate_tokens, 16384), torch.int32).fill_(-1), zeros(candidate_tokens, torch.int32),
            zeros((tokens, 32, 128)), zeros((tokens, 32)), zeros((tokens, 128)), zeros((tokens, 128)))


class PreparedCompressor:
    def __init__(self, layer, weights, source, metadata, pool, slots, *, session, name,
                 deferred=False, examples=None, proj_tokens=None, compress_tokens=None):
        self.pool, self.layer = pool, layer
        self.weight = weights["norm"]
        self.deferred = deferred
        ratio = pool.config.text["compress_ratios"][layer]
        self.ratio = ratio
        total_tokens, rows = source.shape[0], metadata.starts.numel()
        c_tokens = total_tokens if compress_tokens is None else compress_tokens
        p_tokens = c_tokens if proj_tokens is None else proj_tokens
        if not 0 < p_tokens <= c_tokens <= total_tokens:
            raise ValueError("compressor projection and compression capacities must fit the source rows")
        device = source.device
        def zeros(shape, dtype):
            return torch.zeros(shape, device=device, dtype=dtype)
        values_dtype = torch.float32 if ratio == 2 else torch.bfloat16
        self.values = declaration_tensor(examples, f"compressor.values.r{ratio}",
            (c_tokens, 512), values_dtype, device)
        self.project_values = UnquantizedProjection(source[:p_tokens], weights["values"], self.values[:p_tokens],
            session=session, name=f"{name}.values").functional(session=session)
        self.output = declaration_tensor(examples, "compressor.output", (c_tokens, 512), torch.bfloat16, device)
        self.emitted = declaration_tensor(examples, "compressor.emitted", (c_tokens,), torch.bool, device)
        self.slots = declaration_tensor(examples, "compressor.slots", (c_tokens,), torch.int64, device)
        if examples is None:
            self.slots.fill_(-1)
        self.state_ids = persistent_copy(torch.arange(rows, dtype=torch.int64, device=device))
        record_numeric_tensor(self.state_ids)
        self.arguments = dict(values=self.values, weight=weights["norm"], query_start_loc=metadata.query_start_loc,
            positions=metadata.starts, state_ids=self.state_ids, destination_slots=slots[:c_tokens],
            live_counts=metadata.live_counts, out=self.output, emitted=self.emitted, emitted_slots=self.slots)
        if ratio == 2:
            self.gates = declaration_tensor(examples, "compressor.gates",
                self.values.shape, self.values.dtype, device)
            self.project_gates = UnquantizedProjection(source[:p_tokens], weights["gates"], self.gates[:p_tokens],
                session=session, name=f"{name}.gates").functional(session=session)
            self.pending = {key: zeros((rows, 512) if key != "pending_position" else (rows,),
                torch.float32 if key != "pending_position" else torch.int64)
                for key in ("pending_values", "pending_gates", "pending_position")}
            self.arguments.update(gates=self.gates, **self.pending)
        self.plan = mla_compress.plan(mla_compress.Caps(device=device, max_tokens=c_tokens,
            max_requests=rows, max_states=rows, ratio=ratio))

        declaration = self.arguments

        def prepare(state):
            arguments = {key: value if key in ("weight", "state_ids") else torch.zeros_like(value)
                         for key, value in declaration.items()}
            arguments["query_start_loc"].copy_(torch.arange(rows + 1, device=device) * c_tokens // rows)
            arguments["live_counts"].copy_(torch.tensor([c_tokens, rows], dtype=torch.int32, device=device))
            arguments["destination_slots"].copy_(torch.arange(c_tokens, device=device))
            if ratio == 2:
                arguments["pending_position"].fill_(-1)
            binding = state.bind(**arguments)
            return PreparedCall(run=lambda: state.run(binding))

        declare(session, (self.plan.request(name=name, prepare_call=prepare),))
    def functional(self, *, session):
        register_finalizer(session, self.release_examples)
        return self

    def release_examples(self):
        self.values = self.output = self.emitted = self.slots = self.arguments = None
        if self.ratio == 2:
            self.gates = self.pending = None

    def project_shard(self, source_shard):
        values = self.project_values(source_shard)
        gates = self.project_gates(source_shard) if self.ratio == 2 else None
        return values, gates

    def init_pending(self, metadata, device):
        if self.ratio != 2:
            return None
        rows = metadata.starts.numel()
        pending = {
            key: torch.empty(
                (rows, 512) if key != "pending_position" else (rows,),
                dtype=torch.float32 if key != "pending_position" else torch.int64,
                device=device,
            )
            for key in ("pending_values", "pending_gates", "pending_position")
        }
        gather_rows_three(
            tuple(self.pool.region(f"layer{self.layer}.{key}") for key in pending),
            metadata.state_rows,
            tuple(pending.values()),
        )
        return pending

    def compress_half(self, values_half, gates_half, metadata, slots_half, *, half_idx, pending=None):
        from b12x.attention.mla_compress._impl import _run_ratio1, _run_ratio2

        m = metadata
        h = values_half.shape[0]
        offset = 0 if half_idx == 0 else h
        nt = (m.live_counts[:1] - offset).clamp(0, h).to(torch.int32)
        qsl = torch.cat((m.query_start_loc[:1], nt), dim=0)
        counts = torch.cat((nt, m.live_counts[1:2]), dim=0)
        starts = m.starts if half_idx == 0 else m.starts + h
        output = torch.empty((h, 512), dtype=torch.bfloat16, device=values_half.device)
        emitted = torch.empty(h, dtype=torch.bool, device=values_half.device)
        emitted_slots = torch.empty((h,), dtype=torch.int64, device=values_half.device)
        if self.ratio == 2:
            if pending is None:
                raise ValueError("ratio 2 half-chunk compression requires pending carry tensors")
            _run_ratio2(self.plan.handle, values_half, gates_half, self.weight, qsl, starts,
                self.state_ids, slots_half, counts, *pending.values(), output, emitted, emitted_slots)
            if half_idx == 1 and not self.deferred:
                active = m.lengths > 0
                has_carry = active & (pending["pending_position"] >= 0)
                for key in ("pending_values", "pending_gates"):
                    save_hidden(pending[key].view(pending[key].shape[0], -1),
                                self.pool.region(f"layer{self.layer}.{key}"), m.state_rows, has_carry)
                save_hidden(pending["pending_position"].view(pending["pending_position"].shape[0], -1),
                            self.pool.region(f"layer{self.layer}.pending_position"), m.state_rows, active)
        else:
            _run_ratio1(self.plan.handle, values_half, self.weight, qsl, starts,
                self.state_ids, slots_half, counts, output, emitted, emitted_slots)
        return output, emitted_slots

    def __call__(self, source, metadata, slots):
        from b12x.attention.mla_compress._impl import _run_ratio1, _run_ratio2

        m = metadata
        output = torch.empty((source.shape[0], 512), dtype=torch.bfloat16, device=source.device)
        emitted = torch.empty(source.shape[0], dtype=torch.bool, device=source.device)
        emitted_slots = torch.empty((source.shape[0],), dtype=torch.int64, device=source.device)
        state_ids = self.state_ids
        gates = None
        if self.ratio == 2:
            pending = {key: torch.empty((m.starts.numel(), 512) if key != "pending_position" else (m.starts.numel(),),
                dtype=torch.float32 if key != "pending_position" else torch.int64, device=source.device)
                for key in ("pending_values", "pending_gates", "pending_position")}
            gather_rows_three(tuple(self.pool.region(f"layer{self.layer}.{key}") for key in pending),
                              m.state_rows, tuple(pending.values()))
            gates = self.project_gates(source)
            values = self.project_values(source)
            _run_ratio2(self.plan.handle, values, gates, self.weight, m.query_start_loc, m.starts,
                state_ids, slots, m.live_counts, *pending.values(), output, emitted, emitted_slots)
            if not self.deferred:
                for key, value in pending.items():
                    save_hidden(value.view(value.shape[0], -1), self.pool.region(f"layer{self.layer}.{key}"),
                                m.state_rows, m.lengths > 0)
        else:
            values = self.project_values(source)
            _run_ratio1(self.plan.handle, values, self.weight, m.query_start_loc, m.starts,
                       state_ids, slots, m.live_counts, output, emitted, emitted_slots)
        return output, emitted_slots, (values, gates)

    def commit(self, counts, records, metadata):
        if self.ratio != 2:
            return
        m = metadata
        values, gates = records
        last = (m.query_start_loc[:-1] + counts - 1).long().clamp(0, values.shape[0] - 1)
        position = m.starts + counts - 1
        pending = (counts > 0) & (position % 2 == 0)
        for key, source in (("pending_values", values), ("pending_gates", gates)):
            save_hidden(source[last], self.pool.region(f"layer{self.layer}.{key}"), m.state_rows, pending)
        tags = torch.where(position % 2 == 0, position, -1).view(-1, 1)
        save_hidden(tags, self.pool.region(f"layer{self.layer}.pending_position"), m.state_rows, counts > 0)


class PreparedGlobalKV:
    """Project and store every source row before decoder queries are packed."""

    def __init__(self, layer, weights, source, scratch, metadata, pool, rotary, *,
                 session, name, indices, deferred=False, examples=None,
                 proj_tokens=None, compress_tokens=None):
        self.rotary = rotary
        self.ratio = pool.config.text["compress_ratios"][layer]
        if indices.metadata is not metadata or indices.position.shape != (source.shape[0],):
            raise ValueError("global KV indices must describe every source row")
        c_tokens = source.shape[0] if compress_tokens is None else compress_tokens
        slots = indices.compressed(self.ratio).slots
        self.angles = indices.rotary_angles(rotary, compressed=True, ratio=self.ratio)
        self.compressor = PreparedCompressor(layer, weights["compressor"], source, metadata,
            pool, slots, session=session, name=f"{name}.compress", deferred=deferred,
            examples=examples, proj_tokens=proj_tokens, compress_tokens=compress_tokens)
        self.index_key = UnquantizedProjection(self.compressor.output, weights["index_k"],
            scratch.index_keys[:c_tokens], session=session, name=f"{name}.index_key").functional(session=session)
        self.eps = pool.config.text["rms_norm_eps"]
        self.index_norm_weight = weights["index_norm"]
        record_numeric_tensor(self.index_norm_weight)
        self.index_norm = (None if self.angles.split else PreparedRMSNorm(
            scratch.index_keys[:c_tokens], weights["index_norm"], scratch.normalized_keys[:c_tokens],
            eps=self.eps, session=session, name=name + ".index_norm").functional())
        self.writer = PreparedCompressedCacheWriter(self.compressor.output,
            pool.pages[f"layer{layer}.latent_page"], self.compressor.slots,
            page_size=PAGE_TOKENS // self.ratio, kind="indexed", session=session,
            name=f"{name}.compressed_write").functional(session=session)
        self.compressor.functional(session=session)

    def plans(self):
        result = [self.compressor.project_values.plan, self.compressor.plan,
                  self.index_key.plan, self.writer.plan]
        if self.index_norm is not None:
            result.append(self.index_norm.plan)
        if hasattr(self.compressor, "project_gates"):
            result.append(self.compressor.project_gates.plan)
        return tuple(p for p in result if p is not None)

    def run_half(self, values_half, gates_half, indexer, metadata, geometry, *,
                 half_idx, pending=None, angles=None):
        h = values_half.shape[0]
        sl = slice(0, h) if half_idx == 0 else slice(h, 2 * h)
        latent, slots = self.compressor.compress_half(
            values_half, gates_half, metadata, geometry.slots[sl],
            half_idx=half_idx, pending=pending)
        keys_raw = self.index_key(latent)
        half_angles = None if angles is None else angles[sl]
        if half_angles is not None and self.angles.split:
            keys = _index_norm_and_rotate(keys_raw, self.index_norm_weight, latent, half_angles, self.eps)
            indexer.write_keys(keys, slots)
        elif half_angles is not None:
            keys = self.index_norm(keys_raw)
            self.angles.apply(keys, half_angles)
            indexer.write_keys(keys, slots)
            self.angles.apply(latent, half_angles)
        else:
            keys = torch.empty_like(keys_raw) if self.index_norm is None else self.index_norm(keys_raw)
            if self.index_norm is None:
                normalize(keys_raw, self.index_norm_weight, keys, self.eps)
            self.rotary.apply(keys, geometry.position[sl], compressed=True)
            indexer.write_keys(keys, slots)
            self.rotary.apply(latent, geometry.position[sl], compressed=True)
        self.writer(latent, slots)
        return latent

    def __call__(self, source, indexer, metadata, geometry, angles=None):
        latent, slots, records = self.compressor(source, metadata, geometry.slots)
        keys_raw = self.index_key(latent)
        if angles is not None and self.angles.split:
            keys = _index_norm_and_rotate(keys_raw, self.index_norm_weight, latent, angles, self.eps)
            indexer.write_keys(keys, slots)
        elif angles is not None:
            keys = self.index_norm(keys_raw)
            self.angles.apply(keys, angles)
            indexer.write_keys(keys, slots)
            self.angles.apply(latent, angles)
        else:
            keys = torch.empty_like(keys_raw) if self.index_norm is None else self.index_norm(keys_raw)
            if self.index_norm is None:
                normalize(keys_raw, self.index_norm_weight, keys, self.eps)
            self.rotary.apply(keys, geometry.position, compressed=True)
            indexer.write_keys(keys, slots)
            self.rotary.apply(latent, geometry.position, compressed=True)
        self.writer(latent, slots)
        return records


@triton.jit
def _commit_window(Staged, Window, State, Starts, Lengths, STRIDE: tl.constexpr,
                   STAGED_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    item = tl.program_id(0)
    request = tl.program_id(1)
    length, start, row = tl.load(Lengths + request), tl.load(Starts + request), tl.load(State + request)
    token = tl.maximum(0, length - 128) + item
    offset = tl.arange(0, BLOCK)
    valid = (token < length) & (row > 0) & (offset < 528)
    value = tl.load(Staged + request * STAGED_STRIDE + (128 + token) * 528 + offset, valid, other=0)
    tl.store(Window + row * STRIDE + ((start + token) % 128) * 528 + offset, value, valid)


class PreparedAttention:
    def __init__(self, layer, weights, source, scratch, metadata, pool, rotary, *, mode,
                 world, session, name, workspace, replay_start=None, window_staging=None,
                 mla_compute_mode="auto", window_metadata=None, examples=None, pipelined_mode=None,
                 qkv_encoded=None, shard_tokens=None, replicated_weights=None):
        self.layer, self.weights, self.scratch, self.metadata = layer, weights, scratch, metadata
        self.pool, self.rotary, self.mode = pool, rotary, mode
        self.replay_start = replay_start
        if pipelined_mode not in (None, "full", "mla_wo"):
            raise ValueError(f"unsupported pipelined attention mode: {pipelined_mode!r}")
        if qkv_encoded is not None and pipelined_mode != "full":
            raise ValueError("preencoded qkv_a requires full pipelined attention")
        if pipelined_mode == "full" and ("compressor" in weights or "index_q" in weights):
            raise ValueError("full pipelined attention requires a non-source, non-indexer layer")
        if pipelined_mode == "mla_wo" and "compressor" not in weights and "index_q" not in weights:
            raise ValueError("mla_wo pipelined attention requires a source or indexer layer")
        if (shard_tokens is None) != (replicated_weights is None):
            raise ValueError("row-sharded attention requires both shard_tokens and replicated_weights")
        self.row_shard_attention = shard_tokens is not None
        if self.row_shard_attention:
            if mode != "prefill" or pipelined_mode not in ("full", "mla_wo"):
                raise ValueError("row-sharded attention requires prefill mode with pipelined attention")
            if qkv_encoded is not None:
                raise ValueError("row-sharded attention owns shard-local qkv_a quantization and rejects preencoded qkv_a")
            if "index_q" in weights and layer >= pool.config.text["candidate_source_layer_id"]:
                raise ValueError("row-sharded attention indexer requires an encoder layer before candidate selection")
        self.pipelined_mode = pipelined_mode
        config = pool.config
        draft = mode == "draft"
        self.ratio = 0 if draft else config.text["compress_ratios"][layer]
        self.source = None if draft else config.source(layer)
        self.eps = config.text["rms_norm_eps"]
        tokens, device = source.shape[0], source.device
        if pipelined_mode is not None and (tokens < 2 or tokens % 2 != 0):
            raise ValueError("pipelined attention requires an even token capacity >= 2")
        self.half_tokens = tokens // 2
        if self.row_shard_attention:
            if type(shard_tokens) is not int or shard_tokens <= 0 or self.half_tokens % shard_tokens != 0:
                raise ValueError("row-sharded attention shard_tokens must be a positive divisor of half_tokens")
            if (not isinstance(replicated_weights, Mapping)
                    or any(replicated_weights.get(k) is None for k in ("q_b_full", "output_full", "sink_full"))):
                raise ValueError("row-sharded attention requires q_b_full, output_full, and sink_full weights")
            num_heads = config.text.get("num_attention_heads", 64)
            head_dim = config.text.get("head_dim", 512)
            o_groups = config.text.get("o_groups", 8)
            q_b_weight = replicated_weights["q_b_full"]
            output_weight = replicated_weights["output_full"]
            sink_weight = replicated_weights["sink_full"]
            q_b_out = getattr(q_b_weight, "out_features", None)
            out_groups = getattr(output_weight, "groups", None)
            if ((q_b_out is not None and q_b_out != num_heads * head_dim)
                    or (out_groups is not None and out_groups != o_groups)
                    or not isinstance(sink_weight, torch.Tensor)
                    or sink_weight.shape != (num_heads,)
                    or sink_weight.dtype != torch.float32
                    or sink_weight.device != device):
                raise ValueError("row-sharded attention replicated weights must span all heads and output groups")
            self.shard_tokens = shard_tokens
            prefix_tokens = shard_tokens
            mla_tokens = shard_tokens
            window_tokens = self.half_tokens
        else:
            self.shard_tokens = 0
            prefix_tokens = self.half_tokens if pipelined_mode == "full" else tokens
            mla_tokens = self.half_tokens if pipelined_mode in ("full", "mla_wo") else tokens
            window_tokens = prefix_tokens
        rows = metadata.starts.numel()
        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device)
        self.window_page_size = 128 if mode == "decode" else 128 + metadata.max_query_tokens
        self.window = pool.window(layer, draft=draft)
        if window_staging is not None:
            if (mode != "verify" or window_staging.shape != (rows, self.window_page_size * 528)
                    or window_staging.dtype != torch.uint8 or window_staging.device != device
                    or not window_staging.is_contiguous()):
                raise ValueError("verification window staging must match the contiguous cache geometry")
            self.window_cache = window_staging
        else:
            self.window_cache = self.window if mode == "decode" else declaration_tensor(
                examples, "attention.window_cache", (rows, self.window_page_size * 528), torch.uint8, device)
        window_width = self.window_page_size if draft else 128
        self.owns_window_metadata = window_metadata is None
        if window_metadata is None:
            window_metadata = AttentionIndices(metadata, mode=mode, tokens=tokens,
                page_size=self.window_page_size, width=window_width, replay_start=replay_start,
                ratios=(self.ratio,) if self.ratio else ())
        elif (window_metadata.metadata is not metadata or window_metadata.mode != mode
                or window_metadata.window_page_size != self.window_page_size
                or window_metadata.window_indices.shape != (tokens, window_width)
                or window_metadata.replay_start is not replay_start or window_metadata.visible is not None
                or not isinstance(window_metadata, AttentionIndices)):
            raise ValueError("shared window indices must match the attention metadata and cache geometry")
        self.window_metadata = window_metadata
        self.window_slots = window_metadata.window_slots
        self.window_indices = window_metadata.window_indices
        self.window_lengths = window_metadata.window_lengths
        self.token_offsets = window_metadata.token_offsets
        self.angles = window_metadata.rotary_angles(rotary, compressed=bool(self.ratio), ratio=0)
        self.index_angles = (window_metadata.rotary_angles(rotary, compressed=True, ratio=0)
                             if "index_q" in weights else None)
        geometry = window_metadata.compressed(self.ratio) if self.ratio else None
        self.lengths = zeros(tokens, torch.int32) if geometry is None else geometry.lengths
        self.active_width = zeros(1, torch.int32) if geometry is None else geometry.active_width
        self.index_lengths = zeros(tokens, torch.int32) if geometry is None else geometry.index_lengths
        self.page_table = metadata.token_page_table
        self.projections = {}
        if self.row_shard_attention:
            query_buf = declaration_tensor(
                examples, "row_shard.attn_query", (prefix_tokens, num_heads, head_dim), torch.bfloat16, device)
            attended_buf = declaration_tensor(
                examples, "row_shard.attn_attended", (mla_tokens, num_heads, head_dim), torch.bfloat16, device)
        else:
            q_b_weight = weights["q_b"]
            output_weight = weights["output"]
            sink_weight = weights["sink"]
            query_buf = scratch.query[:prefix_tokens]
            attended_buf = scratch.attended[:mla_tokens]
        use_b12x_mxfp8 = not sm103_projection.is_sm103(device)
        paired_mxfp8 = use_b12x_mxfp8 and all(
            isinstance(getattr(w, "weight", None), mxfp8.MXFP8Rows)
            for w in (weights["qkv_a"], q_b_weight))
        use_a16_qkv = (paired_mxfp8 and prefix_tokens <= 28
                       and not getattr(session, "native_dense_activations", False))
        self.norm_quant = (PreparedMLANormQuant(scratch.projected[:prefix_tokens],
            (weights["q_norm"], weights["kv_norm"]), scratch.query_low[:prefix_tokens],
            scratch.window_values[:prefix_tokens],
            eps=self.eps, session=session, name=name + ".qkv_norm",
            release_dependents=prefix_tokens <= 8, examples=examples)
            if paired_mxfp8 and not use_a16_qkv and self.eps == 1e-20 and mla_norm_mxfp8.is_supported(device) else None)
        for key, w, values, out in (
            ("qkv_a", weights["qkv_a"], source[:prefix_tokens], scratch.projected[:prefix_tokens]),
            ("q_b", q_b_weight, scratch.query_low[:prefix_tokens], query_buf.view(prefix_tokens, -1)),
        ):
            packed_mxfp8 = use_b12x_mxfp8 and isinstance(
                getattr(w, "weight", None), mxfp8.MXFP8Rows)
            use_quantized_seam = packed_mxfp8 and not use_a16_qkv
            qkv_preencoded = key == "qkv_a" and qkv_encoded is not None
            if qkv_preencoded and not use_quantized_seam:
                raise ValueError("preencoded qkv_a requires an MXFP8 quantized attention seam")
            self.projections[key] = QuantizedProjection(values, w, out,
                session=session, name=f"{name}.{key}", workspace=workspace, activation_mode="auto",
                mxfp8_min_amax=(1e-4 if use_quantized_seam and not qkv_preencoded
                                and (key != "q_b" or self.norm_quant is None) else None),
                encoded=(qkv_encoded if qkv_preencoded else
                         (self.norm_quant.encoded if key == "q_b" and self.norm_quant is not None else None)),
                rhs_prefetch_ready=use_quantized_seam,
                allow_bf16_atomic_reduction=not packed_mxfp8,
                examples=examples, example_role="attention." + key)
        self.fused_qkv_norm_rope = (self.norm_quant is None and self.angles is not None and self.angles.split)
        if self.fused_qkv_norm_rope:
            self.q_norm_weight = weights["q_norm"]
            self.kv_norm_weight = weights["kv_norm"]
            record_numeric_tensor(self.q_norm_weight)
            record_numeric_tensor(self.kv_norm_weight)
            self.query_norm = self.window_norm = None
        else:
            self.q_norm_weight = self.kv_norm_weight = None
            self.query_norm = (PreparedRMSNorm(scratch.projected[:prefix_tokens, :1280], weights["q_norm"],
                scratch.query_low[:prefix_tokens], eps=self.eps, session=session, name=name + ".q_norm")
                if self.norm_quant is None else None)
            self.window_norm = (PreparedRMSNorm(scratch.projected[:prefix_tokens, 1280:], weights["kv_norm"],
                scratch.window_values[:prefix_tokens], eps=self.eps, session=session, name=name + ".kv_norm")
                if self.norm_quant is None else None)
        self.output_projection = PreparedGroupedOutputProjection(
            attended_buf, output_weight, scratch.output[:mla_tokens],
            session=session, name=name + ".output", workspace=workspace,
            inv_rope=self.angles is not None,
            heads_per_group=attended_buf.shape[1] // getattr(output_weight, "groups", 1),
            nope_dim=448, rope_dim=64, positions=self.token_offsets[:mla_tokens],
            cos_sin_cache=None if self.angles is None else self.angles.values[:mla_tokens])
        self.window_writer = PreparedCompressedCacheWriter(scratch.window_values[:window_tokens], self.window_cache,
            self.window_slots[:window_tokens], page_size=self.window_page_size, kind="swa", session=session,
            name=f"{name}.window_write",
            history=self.window if mode in ("verify", "draft") else None,
            history_rows=metadata.state_rows if mode in ("verify", "draft") else None,
            snapshot_tokens=128 if mode in ("verify", "draft") else 0)
        self.global_kv = None
        self.selectors = []
        if "compressor" in weights:
            self.global_kv = PreparedGlobalKV(layer, weights, source, scratch, metadata, pool, rotary,
                session=session, name=name, indices=window_metadata, deferred=mode == "verify",
                examples=examples,
                proj_tokens=self.shard_tokens if self.row_shard_attention else None,
                compress_tokens=self.half_tokens if self.row_shard_attention else None)
        self.kv_pack_cols = 512 + (0 if self.global_kv is None else (2048 if self.ratio == 2 else 512))
        if "index_q" in weights:
            idx_tokens = self.shard_tokens if self.row_shard_attention else tokens
            self.index_query = QuantizedProjection(scratch.query_low[:idx_tokens], weights["index_q"],
                scratch.index_query[:idx_tokens].view(idx_tokens, -1), session=session, name=f"{name}.index_query",
                workspace=workspace, activation_mode="auto",
                examples=examples, example_role="attention.index_query")
            self.index_weight = UnquantizedProjection(source[:idx_tokens], weights["index_weights"],
                scratch.index_weights[:idx_tokens], session=session, name=f"{name}.index_weight")
            candidate_source = config.text["candidate_source_layer_id"]
            if layer >= candidate_source and scratch.candidates.shape[0] < idx_tokens:
                raise ValueError("hierarchical selection exceeds candidate row capacity")
            for first in range(0, idx_tokens, INDEXER_QUERY_ROWS):
                last = min(idx_tokens, first + INDEXER_QUERY_ROWS)
                candidates = {}
                if layer == candidate_source:
                    candidates = dict(candidates_out=scratch.candidates[first:last], counts_out=scratch.candidate_counts[first:last])
                elif layer > candidate_source:
                    candidates = dict(candidates=scratch.candidates[first:last], counts=scratch.candidate_counts[first:last])
                selector = PreparedMXFP4Indexer(scratch.index_query[first:last], scratch.index_weights[first:last],
                    pool.pages[f"layer{self.source.layer}.index_page"],
                    self.page_table[first:last],
                    self.lengths[first:last], self.active_width, scratch.selected[first:last],
                    page_size=128 // self.ratio, session=session, name=f"{name}.index{first}", workspace=workspace,
                    mode="prefill" if mode == "prefill" else "decode", **candidates)
                self.selectors.append(selector)
        indexed = {} if not self.ratio else dict(indexed=pool.pages[f"layer{self.source.layer}.latent_page"],
            indexed_indices=scratch.selected[:mla_tokens], indexed_lengths=self.index_lengths[:mla_tokens],
            page_table=self.page_table[:mla_tokens],
            indexed_page_size=128 // self.ratio)
        self.attention = PreparedCompressedMLA(query_buf[:mla_tokens], self.window_cache,
            self.window_indices[:mla_tokens], self.window_lengths[:mla_tokens], sink_weight,
            attended_buf,
            window_page_size=self.window_page_size,
            # Speculative rows use explicit indices into staged caches. Their
            # prepared row bound belongs to the decode contract, including
            # multi-request verification and noncausal draft blocks.
            mode="decode" if mode in ("decode", "verify", "draft") else "extend",
            decode_row_capacity=tokens if mode in ("decode", "verify", "draft") else None,
            session=session, name=f"{name}.attention",
            workspace=workspace, compute_mode=mla_compute_mode, **indexed)
        self.concurrent_query = None
        overlap_indexer = False
        if (mode in ("decode", "verify", "draft") and not (mode == "decode" and tokens == 1)
                and isinstance(workspace, ScratchArena) and workspace.overlap
                and self.angles is not None and self.angles.split
                and not (self.projections["q_b"].encoded is not None
                         and self.projections["q_b"].quantizer_plan is not None)):
            overlap_indexer = (self.global_kv is not None and bool(self.selectors)
                               and self.index_angles is not None and self.index_angles.split
                               and self.index_query.encoded is None)
            parallel_plans = [self.window_writer.plan]
            if self.window_norm is not None:
                parallel_plans.append(self.window_norm.plan)
            if self.global_kv is not None:
                global_kv = self.global_kv
                compressor = global_kv.compressor
                parallel_plans.extend((compressor.project_values.plan, compressor.plan,
                    global_kv.index_key.plan, global_kv.writer.plan))
                if global_kv.index_norm is not None:
                    parallel_plans.append(global_kv.index_norm.plan)
                if hasattr(compressor, "project_gates"):
                    parallel_plans.append(compressor.project_gates.plan)
            if self.selectors and not overlap_indexer:
                parallel_plans.extend((self.index_query.plan, self.index_weight.plan))
                parallel_plans.extend(selector.plan for selector in self.selectors)
            self.concurrent_query = ConcurrentAttentionAux(
                self.projections["q_b"], parallel_plans=parallel_plans, session=session,
                workspace=workspace,
                index_query=self.index_query if overlap_indexer else None,
                index_weight=self.index_weight if overlap_indexer else None,
                selectors=self.selectors if overlap_indexer else (),
                selector_rows=INDEXER_QUERY_ROWS)

        self.query_shape = (prefix_tokens, query_buf.shape[1], query_buf.shape[2])
        self.candidate_source = config.text["candidate_source_layer_id"]
        self.q_b_fused_rope = False
        self.index_query_fused_rope = False
        self.projections["qkv_a"].functional(session=session)
        if self.concurrent_query is None:
            self.projections["q_b"].functional(session=session)
        else:
            self.concurrent_query.functional(session=session)
        if self.norm_quant is not None:
            self.norm_quant.functional(session=session)
        elif not self.fused_qkv_norm_rope:
            self.query_norm = self.query_norm.functional()
            self.window_norm = self.window_norm.functional()
        self.output_projection.functional(session=session)
        self.window_writer.functional(session=session, cache=None)
        self.attention.functional(session=session, window=None)
        if self.selectors and not overlap_indexer:
            self.index_query.functional(session=session)
            self.index_weight.functional(session=session)
            for selector in self.selectors:
                selector.functional(session=session)
        if self.owns_window_metadata:
            self.window_metadata.functional()
        register_finalizer(session, self._finalize_standalone_rope)
        register_finalizer(session, self.release_examples)

    def _finalize_standalone_rope(self):
        if (not self.row_shard_attention and self.concurrent_query is None
                and self.angles is not None and self.angles.split
                and self.projections["q_b"].encoded is None
                and self.projections["q_b"].encoded_layout is None):
            self.q_b_fused_rope = register_tc_gemv_rope(
                self.projections["q_b"].plan, head_dim=512, rope_dim=64)
        if (not self.row_shard_attention and self.selectors
                and (self.concurrent_query is None or not self.concurrent_query.has_indexer)
                and self.index_angles is not None and self.index_angles.split
                and self.index_query.encoded is None and self.index_query.encoded_layout is None):
            self.index_query_fused_rope = register_tc_gemv_rope(
                self.index_query.plan, head_dim=128, rope_dim=64)

    def plans(self):
        result = []
        for proj in self.projections.values():
            result.append(proj.plan)
            if getattr(proj, "quantizer_plan", None) is not None:
                result.append(proj.quantizer_plan)
        if self.norm_quant is not None:
            result.append(self.norm_quant.plan)
        else:
            if self.query_norm is not None:
                result.append(self.query_norm.plan)
            if self.window_norm is not None:
                result.append(self.window_norm.plan)
        result.append(self.window_writer.plan)
        if self.global_kv is not None:
            global_kv = self.global_kv
            compressor = global_kv.compressor
            result.extend((compressor.project_values.plan, compressor.plan,
                global_kv.index_key.plan, global_kv.writer.plan))
            if global_kv.index_norm is not None:
                result.append(global_kv.index_norm.plan)
            if hasattr(compressor, "project_gates"):
                result.append(compressor.project_gates.plan)
        if self.selectors:
            result.extend((self.index_query.plan, self.index_weight.plan))
            if getattr(self.index_query, "quantizer_plan", None) is not None:
                result.append(self.index_query.quantizer_plan)
            result.extend(selector.plan for selector in self.selectors)
        result.extend((self.attention.plan, self.output_projection.plan))
        return tuple(plan for plan in result if plan is not None)

    def release_examples(self):
        self.scratch = self.metadata = self.replay_start = self.window_cache = None
        self.window_slots = self.window_indices = self.window_lengths = None
        self.lengths = self.active_width = self.index_lengths = self.page_table = None

    def init_window_cache(self, metadata, device):
        window_cache = torch.empty(
            (metadata.starts.numel(), self.window_page_size * 528), dtype=torch.uint8, device=device)
        if self.mode == "prefill":
            gather_rows(self.window, metadata.state_rows, window_cache[:, :128 * 528])
        return window_cache

    def _run_shard_qkv_core(self, source_shard, *, indices, shard_rows, half_idx=None, shard_inputs=None):
        qkv = self.projections["qkv_a"]
        qkv_encoded_bases = (None if qkv.encoded_layout is None else tuple(
            torch.empty(shape, dtype=dtype, device=source_shard.device) for shape, dtype in qkv.encoded_layout))
        projected = qkv(source_shard, encoded_bases=qkv_encoded_bases)
        encoded_bases = None
        angle_key = (bool(self.ratio), 0)
        layer_angles = (
            shard_inputs.angles[angle_key][half_idx]
            if shard_inputs is not None and half_idx is not None and angle_key in shard_inputs.angles
            else torch.index_select(indices.angles[angle_key], 0, shard_rows)
        )
        if self.fused_qkv_norm_rope:
            query_low, window_values = _q_kv_norm_rope(
                projected, self.q_norm_weight, self.kv_norm_weight, layer_angles, self.eps)
        elif self.norm_quant is None:
            query_low = self.query_norm(projected[:, :1280])
            window_values = self.window_norm(projected[:, 1280:])
            self.angles.apply(window_values, layer_angles)
        else:
            query_low, window_values, encoded_bases = self.norm_quant(projected)
            self.angles.apply(window_values, layer_angles)
        query = self.projections["q_b"](query_low, encoded_bases=encoded_bases).view(self.query_shape)
        self.angles.apply(query, layer_angles)
        return query, query_low, window_values, layer_angles

    def run_shard_qkv_q(self, source_shard, *, indices, shard_rows, half_idx=None, shard_inputs=None):
        """Run rank-local [st, 5120] through qkv_a, norm_quant, KV RoPE, q_b_full, and Q RoPE."""
        if not getattr(self, "row_shard_attention", False):
            raise ValueError("run_shard_qkv_q requires row-sharded attention")
        query, _, window_values, layer_angles = self._run_shard_qkv_core(
            source_shard, indices=indices, shard_rows=shard_rows, half_idx=half_idx, shard_inputs=shard_inputs)
        return query, window_values, layer_angles

    def run_shard_mla_wo_local(self, source_shard, *, indices, shard_rows, half_idx=None, shard_inputs=None):
        """Run rank-local QKV, compressor projections, and indexer query/weight projections."""
        if not getattr(self, "row_shard_attention", False):
            raise ValueError("run_shard_mla_wo_local requires row-sharded attention")
        query, query_low, window_values, layer_angles = self._run_shard_qkv_core(
            source_shard, indices=indices, shard_rows=shard_rows, half_idx=half_idx, shard_inputs=shard_inputs)
        if self.global_kv is not None:
            values_sh, gates_sh = self.global_kv.compressor.project_shard(source_shard)
            if self.ratio == 2:
                kv_pack_sh = torch.cat((window_values, values_sh.view(torch.bfloat16), gates_sh.view(torch.bfloat16)), dim=1)
            else:
                kv_pack_sh = torch.cat((window_values, values_sh), dim=1)
        else:
            kv_pack_sh = window_values
        if self.selectors:
            iq_sh = self.index_query(query_low).view(self.shard_tokens, 32, 128)
            idx_angles = (
                shard_inputs.angles[(True, 0)][half_idx]
                if shard_inputs is not None and half_idx is not None and (True, 0) in shard_inputs.angles
                else torch.index_select(indices.angles[(True, 0)], 0, shard_rows)
            )
            self.index_angles.apply(iq_sh, idx_angles)
            iw_sh = self.index_weight(source_shard)
            iw_sh.mul_(128 ** -.5 * 32 ** -.5)
        else:
            iq_sh = iw_sh = None
        return query, kv_pack_sh, iq_sh, iw_sh, layer_angles

    def run_shard_mla_wo(self, query_shard, window_values_full, metadata, selection, *,
                         indices, window_cache, half_idx, shard_rows, layer_angles, shard_inputs=None):
        """Write full half-chunk SWA KV cache [h, 512], then run rank-local [st, 64, 512] MLA + Wo."""
        if not getattr(self, "row_shard_attention", False):
            raise ValueError("run_shard_mla_wo requires row-sharded attention")
        m = metadata
        h = self.half_tokens
        sl = slice(0, h) if half_idx == 0 else slice(h, 2 * h)
        self.window_writer(window_values_full, indices.window_slots[sl], cache=window_cache)
        selected, _, _ = selection
        geometry = indices.compressed(self.ratio) if self.ratio else None
        if shard_inputs is not None:
            win_idx = shard_inputs.window_indices[half_idx]
            win_len = shard_inputs.window_lengths[half_idx]
        else:
            win_idx = torch.index_select(indices.window_indices, 0, shard_rows)
            win_len = torch.index_select(indices.window_lengths, 0, shard_rows)
        if self.ratio:
            if isinstance(selected, tuple):
                idx_sel = selected[half_idx]
            elif selected.shape[0] == self.shard_tokens:
                idx_sel = selected
            elif selected.shape[0] == 2 * self.shard_tokens:
                idx_sel = selected[:self.shard_tokens] if half_idx == 0 else selected[self.shard_tokens:]
            else:
                idx_sel = torch.index_select(selected, 0, shard_rows)
            idx_len = (
                shard_inputs.index_lengths[half_idx]
                if shard_inputs is not None and shard_inputs.index_lengths is not None
                else torch.index_select(geometry.index_lengths, 0, shard_rows)
            )
            pt_sh = (
                shard_inputs.page_table[half_idx]
                if shard_inputs is not None and shard_inputs.page_table is not None
                else torch.index_select(m.token_page_table, 0, shard_rows)
            )
        else:
            idx_sel = idx_len = pt_sh = None
        attended = self.attention(
            query_shard, win_idx, win_len,
            indexed_indices=idx_sel,
            indexed_lengths=idx_len,
            page_table=pt_sh,
            window=window_cache,
        )
        return self.output_projection(
            attended, positions=self.token_offsets[:self.shard_tokens], cos_sin_cache=layer_angles)

    def run_shard_mla_wo_source(self, query_shard, kv_pack_full, iq_sh, iw_sh, metadata, selection, *,
                                indices, window_cache, half_idx, shard_rows, layer_angles,
                                pending_compress=None, shard_inputs=None):
        """Compress half-chunk global KV, run rank-local MXFP4 selection, then run MLA + Wo."""
        if not getattr(self, "row_shard_attention", False):
            raise ValueError("run_shard_mla_wo_source requires row-sharded attention")
        if self.global_kv is not None:
            window_values_full = kv_pack_full[:, :512].contiguous()
            geometry = indices.compressed(self.ratio)
            if self.ratio == 2:
                values_half = kv_pack_full[:, 512:1536].contiguous().view(torch.float32)
                gates_half = kv_pack_full[:, 1536:2560].contiguous().view(torch.float32)
            else:
                values_half = kv_pack_full[:, 512:1024].contiguous()
                gates_half = None
            self.global_kv.run_half(
                values_half, gates_half, self.selectors[0], metadata, geometry,
                half_idx=half_idx, pending=pending_compress, angles=indices.angles[(True, self.ratio)])
        else:
            window_values_full = kv_pack_full
        if self.selectors:
            geometry = indices.compressed(self.ratio)
            pt_sh = (
                shard_inputs.page_table[half_idx]
                if shard_inputs is not None and shard_inputs.page_table is not None
                else torch.index_select(metadata.token_page_table, 0, shard_rows)
            )
            len_sh = (
                shard_inputs.compressed_lengths[half_idx]
                if shard_inputs is not None and shard_inputs.compressed_lengths is not None
                else torch.index_select(geometry.lengths, 0, shard_rows)
            )
            if half_idx == 0:
                active = (
                    shard_inputs.active_width_a
                    if shard_inputs is not None and shard_inputs.active_width_a is not None
                    else geometry.lengths[:self.half_tokens].amax().view(1)
                )
            else:
                active = geometry.active_width
            outputs = []
            for ordinal, selector in enumerate(self.selectors):
                first = ordinal * INDEXER_QUERY_ROWS
                last = min(self.shard_tokens, first + INDEXER_QUERY_ROWS)
                outputs.append(selector(iq_sh[first:last], iw_sh[first:last],
                                        pt_sh[first:last], len_sh[first:last], active))
            sel_sh = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
            half_selection = (sel_sh, selection[1], selection[2])
        else:
            sel_sh = None
            half_selection = selection
        att_sh = self.run_shard_mla_wo(
            query_shard, window_values_full, metadata, half_selection,
            indices=indices, window_cache=window_cache, half_idx=half_idx,
            shard_rows=shard_rows, layer_angles=layer_angles, shard_inputs=shard_inputs)
        return att_sh, sel_sh

    def run_half(self, source_half, metadata, selection, *, indices, window_cache, half_idx,
                 qkv_encoded_bases=None):
        """Run one half-chunk [h, 5120] through QKV, SWA cache write, MLA, and Wo."""
        m = metadata
        h = self.half_tokens
        sl = slice(0, h) if half_idx == 0 else slice(h, 2 * h)
        qkv = self.projections["qkv_a"]
        if qkv_encoded_bases is None and qkv.quantizer_plan is not None and qkv.encoded_layout is not None:
            qkv_encoded_bases = tuple(
                torch.empty(shape, dtype=dtype, device=source_half.device) for shape, dtype in qkv.encoded_layout)
        projected = qkv(source_half, encoded_bases=qkv_encoded_bases)
        encoded_bases = None
        layer_angles = indices.angles[(bool(self.ratio), 0)][sl]
        if self.fused_qkv_norm_rope:
            query_low, window_values = _q_kv_norm_rope(
                projected, self.q_norm_weight, self.kv_norm_weight, layer_angles, self.eps)
        elif self.norm_quant is None:
            query_low = self.query_norm(projected[:, :1280])
            window_values = self.window_norm(projected[:, 1280:])
            self.angles.apply(window_values, layer_angles)
        else:
            query_low, window_values, encoded_bases = self.norm_quant(projected)
            self.angles.apply(window_values, layer_angles)
        query = self.projections["q_b"](query_low, encoded_bases=encoded_bases).view(self.query_shape)
        self.angles.apply(query, layer_angles)
        self.window_writer(window_values, indices.window_slots[sl], cache=window_cache)
        selected, _, _ = selection
        geometry = indices.compressed(self.ratio) if self.ratio else None
        attended = self.attention(query, indices.window_indices[sl], indices.window_lengths[sl],
            indexed_indices=selected[sl] if self.ratio else None,
            indexed_lengths=geometry.index_lengths[sl] if self.ratio else None,
            page_table=m.token_page_table[sl] if self.ratio else None, window=window_cache)
        return self.output_projection(attended, positions=self.token_offsets[:h], cos_sin_cache=layer_angles)

    def run_prefix(self, source, metadata, selection, *, indices, window_cache=None):
        """Run full-chunk QKV, window write, global KV compression, and indexer before half-MLA."""
        m = metadata
        if window_cache is None:
            window_cache = self.init_window_cache(m, source.device)
        qkv = self.projections["qkv_a"]
        qkv_encoded_bases = (None if qkv.encoded_layout is None else tuple(
            torch.empty(shape, dtype=dtype, device=source.device) for shape, dtype in qkv.encoded_layout))
        projected = qkv(source, encoded_bases=qkv_encoded_bases)
        encoded_bases = None
        layer_angles = indices.angles[(bool(self.ratio), 0)]
        if self.fused_qkv_norm_rope:
            query_low, window_values = _q_kv_norm_rope(
                projected, self.q_norm_weight, self.kv_norm_weight, layer_angles, self.eps)
        elif self.norm_quant is None:
            query_low = self.query_norm(projected[:, :1280])
            window_values = self.window_norm(projected[:, 1280:])
            self.angles.apply(window_values, layer_angles)
        else:
            query_low, window_values, encoded_bases = self.norm_quant(projected)
            self.angles.apply(window_values, layer_angles)
        query = self.projections["q_b"](query_low, encoded_bases=encoded_bases).view(self.query_shape)
        self.angles.apply(query, layer_angles)
        self.window_writer(window_values, indices.window_slots, cache=window_cache)
        selected, candidates, candidate_counts = selection
        records = None
        geometry = None
        if self.ratio:
            geometry = indices.compressed(self.ratio)
            if self.global_kv is not None:
                records = self.global_kv(source, self.selectors[0], m, geometry,
                    angles=indices.angles[(True, self.ratio)])
            if self.selectors:
                index_query = self.index_query(query_low).view(source.shape[0], 32, 128)
                self.index_angles.apply(index_query, indices.angles[(True, 0)])
                index_weights = self.index_weight(source)
                index_weights.mul_(128 ** -.5 * 32 ** -.5)
                outputs, candidate_outputs, count_outputs = [], [], []
                for ordinal, selector in enumerate(self.selectors):
                    first = ordinal * INDEXER_QUERY_ROWS
                    last = min(source.shape[0], first + INDEXER_QUERY_ROWS)
                    result = selector(index_query[first:last], index_weights[first:last],
                        m.token_page_table[first:last], geometry.lengths[first:last], geometry.active_width,
                        candidates=None if self.layer <= self.candidate_source else candidates[first:last],
                        counts=None if self.layer <= self.candidate_source else candidate_counts[first:last])
                    if self.layer == self.candidate_source:
                        result, next_candidates, next_counts = result
                        candidate_outputs.append(next_candidates)
                        count_outputs.append(next_counts)
                    outputs.append(result)
                selected = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
                if self.layer == self.candidate_source:
                    candidates = candidate_outputs[0] if len(candidate_outputs) == 1 else torch.cat(candidate_outputs, dim=0)
                    candidate_counts = count_outputs[0] if len(count_outputs) == 1 else torch.cat(count_outputs, dim=0)
        return (query, window_cache, (selected, candidates, candidate_counts), geometry, layer_angles, records)

    def run_mla_wo_half(self, prefix_state, metadata, *, indices, half_idx):
        """Run one half-chunk [h] through MLA and Wo after run_prefix."""
        query, window_cache, selection, geometry, layer_angles, _ = prefix_state
        selected, _, _ = selection
        h = self.half_tokens
        sl = slice(0, h) if half_idx == 0 else slice(h, 2 * h)
        attended = self.attention(query[sl], indices.window_indices[sl], indices.window_lengths[sl],
            indexed_indices=selected[sl] if self.ratio else None,
            indexed_lengths=geometry.index_lengths[sl] if self.ratio else None,
            page_table=metadata.token_page_table[sl] if self.ratio else None, window=window_cache)
        return self.output_projection(attended, positions=self.token_offsets[:h], cos_sin_cache=layer_angles[sl])

    def __call__(self, source, metadata, selection, *, indices=None, replay_start=None, window_cache=None):
        output, record, selection, _ = self._run(source, metadata, selection, indices=indices,
                                               replay_start=replay_start, window_cache=window_cache)
        return output, record, selection

    def _run(self, source, metadata, selection, *, indices=None, replay_start=None, window_cache=None):
        m = metadata
        indices = (self.window_metadata(m, replay_start=replay_start)
                   if self.owns_window_metadata else indices)
        position = indices.position
        if window_cache is None:
            window_cache = self.window if self.mode == "decode" else torch.empty(
                (m.starts.numel(), self.window_page_size * 528), dtype=torch.uint8, device=source.device)
        if self.mode == "prefill":
            gather_rows(self.window, m.state_rows, window_cache[:, :128 * 528])
        qkv = self.projections["qkv_a"]
        qkv_encoded_bases = (None if qkv.encoded_layout is None else tuple(
            torch.empty(shape, dtype=dtype, device=source.device) for shape, dtype in qkv.encoded_layout))
        projected = qkv(source, encoded_bases=qkv_encoded_bases)
        encoded_bases = None
        layer_angles = indices.angles[(bool(self.ratio), 0)]
        if self.fused_qkv_norm_rope:
            query_low, window_values = _q_kv_norm_rope(
                projected, self.q_norm_weight, self.kv_norm_weight, layer_angles, self.eps)
        elif self.norm_quant is None:
            query_low = self.query_norm(projected[:, :1280])
        else:
            query_low, window_values, encoded_bases = self.norm_quant(projected)
        pending = None
        if self.concurrent_query is not None:
            pending = self.concurrent_query.start(
                query_low, source, layer_angles,
                index_angles=indices.angles[(True, 0)] if self.concurrent_query.has_indexer else None,
                encoded_bases=encoded_bases)
        if not self.fused_qkv_norm_rope:
            if self.window_norm is not None:
                window_values = self.window_norm(projected[:, 1280:])
            self.angles.apply(window_values, layer_angles)
        if self.concurrent_query is None:
            if self.q_b_fused_rope:
                query = run_mxfp8_projection_rope(
                    self.projections["q_b"], query_low, layer_angles, head_dim=512).view(self.query_shape)
            else:
                query = self.projections["q_b"](query_low, encoded_bases=encoded_bases).view(self.query_shape)
                self.angles.apply(query, layer_angles)
        self.window_writer(window_values, indices.window_slots,
                           history_rows=m.state_rows if self.mode in ("verify", "draft") else None, cache=window_cache)
        dependency = window_cache
        selected, candidates, candidate_counts = selection
        records = None
        joined_query = False
        if self.ratio:
            geometry = indices.compressed(self.ratio)
            global_dep = None
            if self.global_kv is not None:
                records = self.global_kv(source, self.selectors[0], m, geometry,
                    angles=indices.angles[(True, self.ratio)])
                global_dep = self.selectors[0].cache
            if self.selectors:
                quantized_list = None
                if self.concurrent_query is not None and self.concurrent_query.has_indexer:
                    q_flat, iq_flat, index_weights, quantized_list = self.concurrent_query.finish(
                        pending, window_cache, global_dependency=global_dep)
                    query = q_flat.view(self.query_shape)
                    index_query = iq_flat.view(source.shape[0], 32, 128)
                    joined_query = True
                else:
                    if self.index_query_fused_rope:
                        index_query = run_mxfp8_projection_rope(
                            self.index_query, query_low, indices.angles[(True, 0)], head_dim=128
                        ).view(source.shape[0], 32, 128)
                    else:
                        index_query = self.index_query(query_low).view(source.shape[0], 32, 128)
                        self.index_angles.apply(index_query, indices.angles[(True, 0)])
                    index_weights = self.index_weight(source)
                    index_weights.mul_(128 ** -.5 * 32 ** -.5)
                outputs, candidate_outputs, count_outputs = [], [], []
                for ordinal, selector in enumerate(self.selectors):
                    first = ordinal * INDEXER_QUERY_ROWS
                    last = min(source.shape[0], first + INDEXER_QUERY_ROWS)
                    result = selector(index_query[first:last], index_weights[first:last],
                        m.token_page_table[first:last], geometry.lengths[first:last], geometry.active_width,
                        candidates=None if self.layer <= self.candidate_source else candidates[first:last],
                        counts=None if self.layer <= self.candidate_source else candidate_counts[first:last],
                        quantized=None if quantized_list is None else quantized_list[ordinal])
                    if self.layer == self.candidate_source:
                        result, next_candidates, next_counts = result
                        candidate_outputs.append(next_candidates)
                        count_outputs.append(next_counts)
                    outputs.append(result)
                selected = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
                if self.layer == self.candidate_source:
                    candidates = candidate_outputs[0] if len(candidate_outputs) == 1 else torch.cat(candidate_outputs, dim=0)
                    candidate_counts = count_outputs[0] if len(count_outputs) == 1 else torch.cat(count_outputs, dim=0)
                dependency = selected
        if self.concurrent_query is not None and not joined_query:
            query = self.concurrent_query.finish(pending, dependency).view(self.query_shape)
        attended = self.attention(query, indices.window_indices, indices.window_lengths,
            indexed_indices=selected if self.ratio else None,
            indexed_lengths=geometry.index_lengths if self.ratio else None,
            page_table=m.token_page_table if self.ratio else None, window=window_cache)
        output = self.output_projection(attended, positions=self.token_offsets, cos_sin_cache=layer_angles)
        record = window_cache, records
        if self.mode == "prefill":
            self.commit(m.lengths, record, m)
        observations = dict(projected=projected, query_low=query_low, query=query,
                            window_values=window_values, qkv_encoded_bases=qkv_encoded_bases,
                            query_encoded_bases=encoded_bases, selected=selected, attended=attended, output=output)
        return output, record, (selected, candidates, candidate_counts), observations

    def commit(self, counts, records, metadata):
        window_cache, _ = records
        torch.library.wrap_triton(_commit_window)[(128, metadata.starts.numel())](window_cache, self.window,
            metadata.state_rows, metadata.starts, counts, self.window.stride(0), window_cache.stride(0), 1024)
        self.commit_pending(counts, records, metadata)

    def commit_pending(self, counts, records, metadata):
        if self.mode == "verify" and self.global_kv is not None:
            self.global_kv.compressor.commit(counts, records[1], metadata)
