# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Host-backed FP8_E4M3_E8M0 / NVFP4_E4M3_GLOBAL n-gram tables, GPU hashing, and 4-stream Engram injection.

Each delivered row contains one 96-element head. N-gram hashing runs in PyTorch (``PreparedTableRequests``) because
``ignition.components.engram.PreparedEngramRequests`` targets order-4 24-head hashing with a
3-token history window, whereas Hy4-flash hashes order-3 (2-gram and 3-gram) 16-head tables
over a 2-token continuation history.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from hyperflare_recipes.components.ihc import row_sum_fp32
from hyperflare_recipes.components.linear import prepare_projection
from ignition.components.engram import EngramInputs
from ignition.components.ngram_delivery import publish_requests
from ignition.components.readiness import PreparedReadinessWait, ReadinessTimings, wait_for_readiness
from ignition.components.state import gather_rows, save_hidden
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.runtime_core import FileSource
from ignition.preparation.triton import warmup_triton
from ignition.progress import phase

from .config import (
    ENGRAM_HEADS,
    ENGRAM_HEADS_PER_NGRAM,
    ENGRAM_HEAD_DIM,
    ENGRAM_MAX_NGRAM,
    ENGRAM_WIDTH,
    HC_MULT,
    HIDDEN_SIZE,
    IMAGE_NEWLINE_TOKEN,
    IMAGE_TOKEN,
    VIDEO_TOKEN,
    VOCABULARY,
)

ENGRAM_FP8_SCALE_BLOCK = 32
ENGRAM_FP8_SCALE_BYTES = ENGRAM_HEAD_DIM // ENGRAM_FP8_SCALE_BLOCK  # 3
ENGRAM_NVFP4_ROW_BYTES = ENGRAM_HEAD_DIM // 2  # 48
ENGRAM_NVFP4_SCALE_BLOCK = 16
ENGRAM_NVFP4_SCALE_BYTES = ENGRAM_HEAD_DIM // ENGRAM_NVFP4_SCALE_BLOCK  # 6
_FP4_SIGNED_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def dequantize_nvfp4_engram(
    packed: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    lut: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize packed ``uint8 [..., 48]`` E2M1 + ``uint8 [..., 6]`` E4M3 rows to ``[..., 96]``."""
    prefix = packed.shape[:-1]
    raw = packed.view(*prefix, ENGRAM_NVFP4_SCALE_BYTES, ENGRAM_NVFP4_SCALE_BLOCK // 2)
    low = (raw & 0x0F).long()
    high = ((raw >> 4) & 0x0F).long()
    codes = torch.stack((low, high), dim=-1).reshape(*prefix, ENGRAM_NVFP4_SCALE_BYTES, ENGRAM_NVFP4_SCALE_BLOCK)
    vals = lut[codes]
    s = scales.view(torch.float8_e4m3fn).view(*prefix, ENGRAM_NVFP4_SCALE_BYTES, 1).float() * global_scale
    return (vals * s).reshape(*prefix, ENGRAM_HEAD_DIM).to(dtype)


def _draw_hash_multipliers(layer_id_1based: int, compressed_vocab_size: int, seed: int) -> tuple[int, ...]:
    """Match ``HYV4FlashEngramLayer.draw_hash_multipliers`` bit-for-bit."""
    half_bound = max(1, np.iinfo(np.int64).max // compressed_vocab_size // 2)
    generator = np.random.default_rng(seed + 10007 * layer_id_1based)
    drawn = generator.integers(0, half_bound, size=ENGRAM_MAX_NGRAM, dtype=np.int64)
    return tuple(int(v) * 2 + 1 for v in drawn.tolist())


def _head_offsets(primes: tuple[int, ...]) -> tuple[int, ...]:
    offsets = []
    acc = 0
    for p in primes:
        offsets.append(acc)
        acc += p
    return tuple(offsets)


@dataclass(frozen=True)
class LayerHashSpec:
    layer: int  # 0-based decoder layer index
    rows: int
    primes: tuple[int, ...]
    offsets: tuple[int, ...]
    multipliers: tuple[int, ...]
    encoding: str = "FP8_E4M3_E8M0"
    row_bytes: int = ENGRAM_HEAD_DIM
    scale_bytes: int = ENGRAM_FP8_SCALE_BYTES
    scale_block: int = ENGRAM_FP8_SCALE_BLOCK
    global_scale: float | None = None


@dataclass(frozen=True)
class Tables:
    """Host-backed FP8_E4M3_E8M0 or NVFP4_E4M3_GLOBAL Engram tables and device delivery buffers."""

    specs: tuple[LayerHashSpec, ...]
    token_map: tuple[int, ...]
    pad_token_id: int
    compressed_vocabulary: int
    sources: dict[int, tuple[FileSource, FileSource]]
    inputs: dict[int, EngramInputs]
    mailboxes: dict[int, object]
    sequence: torch.Tensor
    readiness_runtime: object = None

    @classmethod
    def prepare(cls, manifest, config, *, tokens: int, device: torch.device, sequence: torch.Tensor,
                frame: torch.Tensor, host_allocator, progress=None) -> Tables:
        from safetensors import safe_open

        text = config.text
        pad_id = int(text["engram_pad_id"])
        seed = int(text["engram_seed"])
        if not config.engram_layers:
            raise ValueError("Hy4-flash requires declared Engram layers")

        # Load the checkpoint's frozen normalized vocabulary map from the first Engram layer.
        first_layer = config.engram_layers[0]
        lt_name = f"model.layers.{first_layer}.engram_layer.lookup_table"
        lt_meta = manifest.tensors[lt_name]
        with safe_open(str(manifest.snapshot / lt_meta.shard), framework="pt", device="cpu") as h:
            lookup = h.get_tensor(lt_name).to(torch.int64).tolist()
        if len(lookup) > VOCABULARY:
            raise ValueError(f"{lt_name}: lookup_table length {len(lookup)} exceeds vocabulary {VOCABULARY}")
        pad_mapped = int(lookup[pad_id])
        # Extend to full vocabulary with pad_mapped; align IMAGE_NEWLINE_TOKEN to IMAGE_TOKEN
        # so n-grams crossing image boundaries match Transformers bit-for-bit.
        token_map = list(lookup) + [pad_mapped] * (VOCABULARY - len(lookup))
        token_map[IMAGE_NEWLINE_TOKEN] = token_map[IMAGE_TOKEN]
        compressed_vocab = max(lookup) + 1

        specs = []
        sources: dict[int, tuple[FileSource, FileSource]] = {}
        inputs: dict[int, EngramInputs] = {}
        mailboxes: dict[int, object] = {}
        timings = ReadinessTimings.allocate(frame, len(config.engram_layers))

        with phase(progress, "Prepare Engram tables") as stage:
            encodings_seen: set[str] = set()
            for index, layer in enumerate(config.engram_layers):
                primes = config.engram_head_vocab_sizes[layer]
                offsets = _head_offsets(primes)
                rows = sum(primes)
                multipliers = _draw_hash_multipliers(layer + 1, compressed_vocab, seed)

                w_name = f"model.layers.{layer}.engram_layer.multi_head_embedding.weight"
                s_name = f"model.layers.{layer}.engram_layer.multi_head_embedding.scale"
                if w_name not in manifest.tensors or s_name not in manifest.tensors:
                    raise ValueError(
                        f"{w_name} / {s_name} missing; convert BF16 Engram tables with scripts/quantize_hy4_flash_nvfp4.py"
                    )
                w_meta = manifest.tensors[w_name]
                s_meta = manifest.tensors[s_name]
                if w_meta.dtype == "F8_E4M3":
                    encoding = "FP8_E4M3_E8M0"
                    row_bytes = ENGRAM_HEAD_DIM
                    scale_bytes = ENGRAM_FP8_SCALE_BYTES
                    scale_block = ENGRAM_FP8_SCALE_BLOCK
                    global_scale = None
                    if w_meta.shape != (rows, row_bytes) or s_meta.dtype != "F8_E8M0" or s_meta.shape != (rows, scale_bytes):
                        raise ValueError(
                            f"{w_name}: expected F8_E4M3 ({rows}, {row_bytes}) + F8_E8M0 ({rows}, {scale_bytes}), "
                            f"got {w_meta.dtype} {w_meta.shape} + {s_meta.dtype} {s_meta.shape}"
                        )
                elif w_meta.dtype == "U8":
                    encoding = "NVFP4_E4M3_GLOBAL"
                    row_bytes = ENGRAM_NVFP4_ROW_BYTES
                    scale_bytes = ENGRAM_NVFP4_SCALE_BYTES
                    scale_block = ENGRAM_NVFP4_SCALE_BLOCK
                    if w_meta.shape != (rows, row_bytes) or s_meta.dtype != "F8_E4M3" or s_meta.shape != (rows, scale_bytes):
                        raise ValueError(
                            f"{w_name}: expected U8 ({rows}, {row_bytes}) + F8_E4M3 ({rows}, {scale_bytes}), "
                            f"got {w_meta.dtype} {w_meta.shape} + {s_meta.dtype} {s_meta.shape}"
                        )
                    gs_name = f"model.layers.{layer}.engram_layer.multi_head_embedding.weight_scale_2"
                    if gs_name not in manifest.tensors:
                        raise ValueError(f"{gs_name} missing for NVFP4_E4M3_GLOBAL Engram table")
                    gs_meta = manifest.tensors[gs_name]
                    with safe_open(str(manifest.snapshot / gs_meta.shard), framework="pt", device="cpu") as h:
                        gs_t = h.get_tensor(gs_name)
                    if gs_t.dtype != torch.float32 or gs_t.numel() != 1:
                        raise ValueError(f"{gs_name}: expected FP32 scalar, got {gs_t.dtype} {tuple(gs_t.shape)}")
                    global_scale = float(gs_t.item())
                    if not (np.isfinite(global_scale) and global_scale > 0.0):
                        raise ValueError(f"{gs_name}: global scale must be positive finite, got {global_scale}")
                else:
                    raise ValueError(
                        f"{w_name}: unsupported Engram table dtype {w_meta.dtype}; expected F8_E4M3 or U8"
                    )

                encodings_seen.add(encoding)
                specs.append(
                    LayerHashSpec(
                        layer,
                        rows,
                        primes,
                        offsets,
                        multipliers,
                        encoding=encoding,
                        row_bytes=row_bytes,
                        scale_bytes=scale_bytes,
                        scale_block=scale_block,
                        global_scale=global_scale,
                    )
                )
                sources[layer] = (
                    FileSource(manifest.snapshot / w_meta.shard, w_meta.offset, rows * row_bytes),
                    FileSource(manifest.snapshot / s_meta.shard, s_meta.offset, rows * scale_bytes),
                )
                inputs[layer] = EngramInputs.allocate(
                    tokens,
                    device,
                    timings=timings,
                    interval=index,
                    heads=ENGRAM_HEADS,
                    row_bytes=row_bytes,
                    scale_bytes=scale_bytes,
                )
                mailboxes[layer] = host_allocator(
                    f"engram_{layer}_requests", (tokens * ENGRAM_HEADS + 1,), torch.int64
                )
                mailboxes[layer].host_view.zero_()
            total_gib = sum(src.nbytes for pair in sources.values() for src in pair) / (1 << 30)
            enc_label = "/".join(sorted(encodings_seen))
            stage.update(f"{total_gib:.2f} GiB across {len(specs)} {enc_label} tables")

        return cls(
            specs=tuple(specs),
            token_map=tuple(token_map),
            pad_token_id=pad_id,
            compressed_vocabulary=compressed_vocab,
            sources=sources,
            inputs=inputs,
            mailboxes=mailboxes,
            sequence=sequence,
        )

    @cached_property
    def device_token_map(self) -> torch.Tensor:
        device = next(iter(self.inputs.values())).hashes.device
        return persistent_copy(torch.tensor(self.token_map, dtype=torch.int64, device=device))

    def buffers(self) -> dict[str, torch.Tensor]:
        return {
            **{name: tensor for layer, inp in self.inputs.items() for name, tensor in inp.buffers(layer).items()},
            **{f"engram_{layer}_requests": mb.device_view for layer, mb in self.mailboxes.items()},
            **next(iter(self.inputs.values())).timings.buffers(),
        }

    def events(self) -> dict[str, int]:
        return {f"engram_{layer}_ready": inp.ready.cuda_event for layer, inp in self.inputs.items()}

    def resident_catalog(self) -> dict:
        """Native catalog for host-RAM/disk delivery without core export."""
        return {
            "vocabulary": len(self.token_map),
            "tables": [
                {
                    "planes": [
                        {
                            "segments": [{"path": str(source.path), "offset": source.offset, "nbytes": source.nbytes}],
                            "row_bytes": row_bytes,
                        }
                        for source, row_bytes in zip(
                            self.sources[spec.layer], (spec.row_bytes, spec.scale_bytes), strict=True
                        )
                    ],
                    "heads": ENGRAM_HEADS,
                    "weights": f"engram_{spec.layer}_weights",
                    "scales": f"engram_{spec.layer}_scales",
                    "hashes": f"engram_{spec.layer}_hashes",
                    "event": f"engram_{spec.layer}_ready",
                    "request": f"engram_{spec.layer}_requests",
                }
                for spec in self.specs
            ],
        }

    def export_metadata(self) -> dict:
        """Runtime-core protobuf metadata describing the 4 FP8_E4M3_E8M0 or NVFP4_E4M3_GLOBAL Engram tables."""
        tables = []
        orders = [order for order in range(2, ENGRAM_MAX_NGRAM + 1) for _ in range(ENGRAM_HEADS_PER_NGRAM)]
        for spec in self.specs:
            w_src, s_src = self.sources[spec.layer]
            entry = dict(
                layer=spec.layer,
                rows=spec.rows,
                embedding_width=ENGRAM_HEAD_DIM,
                scale_block=spec.scale_block,
                encoding=spec.encoding,
                weight_buffer=f"engram_{spec.layer}_weights",
                scale_buffer=f"engram_{spec.layer}_scales",
                hash_buffer=f"engram_{spec.layer}_hashes",
                ready_event=f"engram_{spec.layer}_ready",
                request_buffer=f"engram_{spec.layer}_requests",
                weights=[dict(path=str(w_src.path), offset=w_src.offset, nbytes=w_src.nbytes)],
                scales=[dict(path=str(s_src.path), offset=s_src.offset, nbytes=s_src.nbytes)],
                hash=dict(
                    method="XOR_MODULO",
                    orders=orders,
                    multipliers=list(spec.multipliers),
                    primes=list(spec.primes),
                    offsets=list(spec.offsets),
                ),
            )
            if spec.global_scale is not None:
                entry["global_scale"] = spec.global_scale
            tables.append(entry)
        return dict(
            token_map=list(self.token_map),
            pad_token_id=self.pad_token_id,
            compressed_vocabulary=self.compressed_vocabulary,
            tables=tables,
        )


class PreparedTableRequests:
    """Compute order-3 16-head XOR-modulo hashes in PyTorch and publish delivery mailboxes."""

    def __init__(self, tables: Tables, metadata, pool):
        self.pool = pool
        self.tables = tables
        device = metadata.starts.device
        tokens = metadata.token_ids.numel()
        self.token_map = tables.device_token_map
        self.pad_id = tables.pad_token_id
        self.pad_compressed = tables.token_map[tables.pad_token_id]
        self.layer_params = tuple(
            (
                spec.layer,
                tables.inputs[spec.layer].hashes[:tokens],
                tables.mailboxes[spec.layer].device_view,
                persistent_copy(torch.tensor(spec.multipliers, dtype=torch.int64, device=device)),
                persistent_copy(torch.tensor(spec.primes, dtype=torch.int64, device=device)),
                persistent_copy(torch.tensor(spec.offsets, dtype=torch.int64, device=device)),
            )
            for spec in tables.specs
        )
        self.sequence = tables.sequence

    def __call__(self, metadata) -> torch.Tensor:
        m = metadata
        device = m.token_ids.device
        tokens = m.token_ids.numel()
        rows = m.starts.numel()
        raw_history = torch.zeros((rows, 2), dtype=torch.int64, device=device)
        gather_rows(self.pool.region("engram_history"), m.state_rows, raw_history)

        # Build the 3-token window [t, t-1, t-2] for every packed token, pulling from
        # raw_history for the first 2 tokens of each request and padding before position 0.
        t_idx = torch.arange(tokens, dtype=torch.int64, device=device)
        req = m.requests.long()
        req_begin = m.query_start_loc[:-1].long()[req]
        local_off = t_idx - req_begin
        logical_pos = m.positions  # -1 for inactive rows
        live = logical_pos >= 0

        def shifted_token(lag: int) -> torch.Tensor:
            if lag == 0:
                return torch.where(live, m.token_ids, self.pad_id)
            in_chunk = local_off >= lag
            chunk_tok = m.token_ids[(t_idx - lag).clamp(0, tokens - 1)]
            # History column 0 holds start-2, column 1 holds start-1.
            hist_col = (2 - lag + local_off).clamp(0, 1)
            hist_tok = raw_history[req, hist_col]
            tok = torch.where(in_chunk, chunk_tok, hist_tok)
            return torch.where(live & (logical_pos >= lag), tok, self.pad_id)

        t0 = shifted_token(0)
        t1 = shifted_token(1)
        t2 = shifted_token(2)
        window = torch.stack((t0, t1, t2), dim=-1)  # [T, 3]
        in_range = (window >= 0) & (window < self.token_map.numel())
        comp = torch.where(
            in_range,
            self.token_map[window.clamp(0, self.token_map.numel() - 1)],
            self.pad_compressed,
        )

        # Mask out inactive, image, video, and pad tokens so host skips unnecessary disk I/O.
        active_engram = (
            live
            & (m.token_ids != IMAGE_TOKEN)
            & (m.token_ids != IMAGE_NEWLINE_TOKEN)
            & (m.token_ids != VIDEO_TOKEN)
            & (m.token_ids != self.pad_id)
        )

        for _, hashes_buf, mailbox, multipliers, primes, offsets in self.layer_params:
            mixed = comp * multipliers[None, :]  # [T, 3]
            h2 = torch.bitwise_xor(mixed[:, 0], mixed[:, 1])  # order-2 bigram
            h3 = torch.bitwise_xor(h2, mixed[:, 2])  # order-3 trigram
            ids2 = (h2[:, None] % primes[None, :8]) + offsets[None, :8]
            ids3 = (h3[:, None] % primes[None, 8:]) + offsets[None, 8:]
            all_ids = torch.cat((ids2, ids3), dim=-1)  # [T, 16]
            all_ids = torch.where(active_engram[:, None], all_ids, -1)
            hashes_buf.copy_(all_ids)
            publish_requests(hashes_buf, mailbox, self.sequence)

        return raw_history

    def commit(self, counts: torch.Tensor, raw_history: torch.Tensor, metadata) -> None:
        """Persist the last two accepted token IDs per request into continuation state."""
        m = metadata
        cols = torch.arange(2, dtype=torch.int64, device=m.token_ids.device)
        relative = counts[:, None].long() + cols - 2
        indices = (m.query_start_loc[:-1, None].long() + relative).clamp(0, m.token_ids.numel() - 1)
        history = torch.where(
            relative >= 0,
            m.token_ids[indices],
            raw_history.gather(1, (relative + 2).clamp(0, 1)),
        )
        save_hidden(history, self.pool.region("engram_history"), m.state_rows, counts > 0)


@dataclass(frozen=True)
class EngramScratch:
    """Intermediate projection storage shared across serialized Engram layers."""

    embeddings: torch.Tensor
    kv_proj: torch.Tensor
    key_proj: torch.Tensor
    value_proj: torch.Tensor

    @classmethod
    def allocate(cls, tokens: int, device: torch.device) -> EngramScratch:
        def zeros(*shape):
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)

        kv_proj = zeros(tokens, (HC_MULT + 1) * HIDDEN_SIZE)
        return cls(
            zeros(tokens, ENGRAM_WIDTH),
            kv_proj,
            kv_proj.view(-1)[: tokens * HIDDEN_SIZE].view(tokens, HIDDEN_SIZE),
            kv_proj.view(-1)[tokens * HIDDEN_SIZE : 2 * tokens * HIDDEN_SIZE].view(tokens, HIDDEN_SIZE),
        )

    def rows(self, tokens: int) -> EngramScratch:
        return EngramScratch(
            self.embeddings[:tokens],
            self.kv_proj[:tokens],
            self.kv_proj.view(-1)[: tokens * HIDDEN_SIZE].view(tokens, HIDDEN_SIZE),
            self.kv_proj.view(-1)[tokens * HIDDEN_SIZE : 2 * tokens * HIDDEN_SIZE].view(tokens, HIDDEN_SIZE),
        )


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Compute RMSNorm in FP32 and round to ``x.dtype`` before weight multiply, matching ``nn.RMSNorm``."""
    xf = x.float()
    inv_rms = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    out = (xf * inv_rms).to(x.dtype)
    return out if weight is None else out * weight


@torch.library.custom_op(
    "flare::engram_wait_ready",
    mutates_args=("timings", "barrier_output", "weights", "scales"),
)
def _wait_ready_op(
    source: torch.Tensor,
    timings: torch.Tensor,
    frame: torch.Tensor,
    barrier_input: torch.Tensor,
    barrier_output: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    interval: int,
    barrier: int,
    event: int,
) -> None:
    del weights, scales
    wait_for_readiness(source, event, timings, frame, interval, barrier_input, barrier_output, barrier)


@_wait_ready_op.register_fake
def _wait_ready_fake(
    source, timings, frame, barrier_input, barrier_output, weights, scales, interval, barrier, event
):
    pass


@triton.jit
def _head_kernel(W, Scale, Hash, Gain, Lut, Out, GLOBAL: tl.constexpr, NVFP4: tl.constexpr,
                 D: tl.constexpr, HEADS: tl.constexpr, ROWS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, 128)
    valid = col[None, :] < D
    if NVFP4:
        byte = tl.load(W + row[:, None] * (D // 2) + col[None, :] // 2, valid, 0)
        idx = (byte >> ((col[None, :] % 2) * 4)) & 15
        value = tl.load(Lut + idx)
        block = tl.load(Scale + row[:, None] * (D // 16) + col[None, :] // 16, valid, 0)
        scale = block.to(tl.float8e4nv, bitcast=True).to(tl.float32) * GLOBAL
    else:
        byte = tl.load(W + row[:, None] * D + col[None, :], valid, 0)
        value = byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        exp = tl.load(Scale + row[:, None] * (D // 32) + col[None, :] // 32, valid, 127)
        # The smallest stored exponent represents the FP32 subnormal 2**-127.
        scale = tl.where(exp == 0, 1 << 22, exp.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    x = tl.where(valid & (tl.load(Hash + row) >= 0)[:, None], value * scale, 0.).to(tl.bfloat16).to(tl.float32)
    lo, hi = tl.split(tl.permute(tl.reshape(x * x, (ROWS, 2, 64)), (0, 2, 1)))
    first, second = tl.split(tl.permute(tl.reshape(lo, (ROWS, 2, 32)), (0, 2, 1)))
    third, _ = tl.split(tl.permute(tl.reshape(hi, (ROWS, 2, 32)), (0, 2, 1)))
    squares = (first + second) + third
    for level in tl.static_range(5):
        lo, hi = tl.split(tl.permute(tl.reshape(squares, (ROWS, 2, 16 >> level)), (0, 2, 1)))
        squares = lo + hi
    inv = tl.rsqrt(squares * (1. / D) + 0.0078125)
    normalized = (x * inv).to(tl.bfloat16).to(tl.float32)
    gain = tl.load(Gain + (row[:, None] % HEADS) * D + col[None, :], valid, 0).to(tl.float32)
    tl.store(Out + row[:, None] * D + col[None, :], normalized * gain, valid)


@torch.library.custom_op("flare::hy4_engram_heads", mutates_args=())
def _decode_heads_op(weights: torch.Tensor, scales: torch.Tensor, hashes: torch.Tensor,
                  gains: torch.Tensor, lut: torch.Tensor | None, global_scale: float, rows: int) -> torch.Tensor:
    tokens, heads = hashes.shape
    width = gains.shape[1]
    out = torch.empty((tokens, heads * width), dtype=torch.bfloat16, device=weights.device)
    _dispatch_heads(weights, scales, hashes, gains, lut, out, global_scale, rows)
    return out


@_decode_heads_op.register_fake
def _decode_heads_fake(weights, scales, hashes, gains, lut, global_scale, rows):
    return weights.new_empty((hashes.shape[0], hashes.shape[1] * gains.shape[1]), dtype=torch.bfloat16)


def _dispatch_heads(weights, scales, hashes, gains, lut, out, global_scale, rows, *, warmup=False):
    tokens, heads = hashes.shape
    grid = (tokens * heads // rows,)
    args = (weights, scales, hashes, gains, lut, out, global_scale, lut is not None, gains.shape[1], heads, rows)
    if warmup:
        warmup_triton(_head_kernel, args, grid, enable_fp_fusion=False)._init_handles()
    else:
        _head_kernel[grid](*args, enable_fp_fusion=False)


def _decode_heads(weights, scales, hashes, gains, lut, global_scale):
    rows = ENGRAM_HEADS if hashes.shape[0] > 64 else 1
    return _decode_heads_op(weights, scales, hashes, gains, lut, global_scale, rows)


@triton.jit
def _bf16(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _unit(x, D: tl.constexpr, ROWS: tl.constexpr):
    return _bf16(x * tl.rsqrt(row_sum_fp32(x * x, D, ROWS) * (1. / D) + 0.0078125))


@triton.jit
def _mix_kernel(S, K, V, KG, QG, Gamma, Scale, Pos, Id, Out, KS: tl.constexpr,
                VS: tl.constexpr, PS: tl.constexpr, IDS: tl.constexpr, D: tl.constexpr,
                BLOCK: tl.constexpr, PAD: tl.constexpr, IMAGE: tl.constexpr,
                NEWLINE: tl.constexpr, VIDEO: tl.constexpr, ROWS: tl.constexpr):
    t, lane = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, BLOCK)
    valid = col < D
    offset = lane * D + col
    x = tl.load(S + t * 4 * D + offset, valid, 0).to(tl.float32)
    k = tl.load(K + t * KS + offset, valid, 0).to(tl.float32)
    v = tl.load(V + t * VS + col, valid, 0).to(tl.float32)
    kg = tl.load(KG + offset, valid, 0).to(tl.float32)
    qg = tl.load(QG + offset, valid, 0).to(tl.float32)
    key = _bf16(_unit(k, D, ROWS * 4) * kg)
    query = _bf16(_unit(x, D, ROWS * 4) * qg)
    dot = row_sum_fp32(key * query, D, ROWS * 4) * (D ** -0.5)
    signed = tl.sqrt(tl.maximum(tl.abs(dot), 1.e-6)) * ((dot > 0).to(tl.int32) - (dot < 0).to(tl.int32))
    gate = _bf16(tl.div_rn(1., 1. + libdevice.exp(-signed)))
    main = _bf16(_unit(v, D, ROWS) * tl.load(Gamma + col, valid, 0).to(tl.float32))
    branch = _bf16(main * tl.load(Scale + offset, valid, 0).to(tl.float32))
    delta = _bf16(_bf16(gate * main) + branch)
    token = tl.load(Id + t * IDS)
    live = (tl.load(Pos + t * PS) >= 0) & (token != PAD) & (token != IMAGE) & (token != NEWLINE) & (token != VIDEO)
    tl.store(Out + t * 4 * D + offset, x + tl.where(live, delta, 0.), valid)


@torch.library.custom_op("flare::hy4_engram_mix", mutates_args=())
def _mix_streams(streams: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                 k_gain: torch.Tensor, q_gain: torch.Tensor, gamma: torch.Tensor, scales: torch.Tensor,
                 positions: torch.Tensor, tokens: torch.Tensor, pad: int) -> torch.Tensor:
    out = torch.empty_like(streams)
    _dispatch_mix(streams, keys, values, k_gain, q_gain, gamma, scales, positions, tokens, out, pad)
    return out


@_mix_streams.register_fake
def _mix_fake(streams, keys, values, k_gain, q_gain, gamma, scales, positions, tokens, pad):
    return torch.empty_like(streams)


def _dispatch_mix(streams, keys, values, k_gain, q_gain, gamma, scales, positions, tokens, out, pad, *, warmup=False):
    rows, lanes, d = streams.shape
    args = (streams, keys, values, k_gain, q_gain, gamma, scales, positions, tokens, out, keys.stride(0),
            values.stride(0), positions.stride(0), tokens.stride(0), d, triton.next_power_of_2(d),
            pad, IMAGE_TOKEN, IMAGE_NEWLINE_TOKEN, VIDEO_TOKEN, rows)
    if warmup:
        warmup_triton(_mix_kernel, args, (rows, lanes), enable_fp_fusion=False)._init_handles()
    else:
        _mix_kernel[(rows, lanes)](*args, enable_fp_fusion=False)


class PreparedEngram:
    """Wait for host-staged FP8_E4M3_E8M0 or NVFP4_E4M3_GLOBAL rows, dequantize, normalize, project, and gate."""

    def __init__(self, layer: int, weights: dict, tables: Tables, source: torch.Tensor, metadata, *,
                 scratch: EngramScratch, session, name: str, workspace, coordinator=None):
        tokens = metadata.token_ids.numel()
        self.pad_id = tables.pad_token_id
        spec = next(s for s in tables.specs if s.layer == layer)
        self.encoding = spec.encoding
        self.global_scale = spec.global_scale
        self.fp4_lut = (
            persistent_copy(torch.tensor(_FP4_SIGNED_VALUES, dtype=torch.float32, device=source.device))
            if self.encoding == "NVFP4_E4M3_GLOBAL"
            else None
        )
        inp = tables.inputs[layer]
        self.weights_buf = inp.weights[: tokens * ENGRAM_HEADS]
        self.scales_buf = inp.scales[: tokens * ENGRAM_HEADS]
        self.hashes_buf = inp.hashes[:tokens]
        self.event_handle = inp.ready.cuda_event
        self.wait = PreparedReadinessWait(
            inp.timings,
            inp.interval,
            runtime=tables.readiness_runtime,
            session=session,
            coordinator=coordinator,
            name=f"{name}.readiness",
        ).functional(session=session)

        self.head_norms = persistent_copy(
            torch.stack([weights[f"lookup_per_head_norms.{h}"] for h in range(ENGRAM_HEADS)], dim=0)
        )  # [16, 96]
        if "kv_proj" in weights:
            self.kv_proj = prepare_projection(
                scratch.embeddings,
                weights["kv_proj"],
                scratch.kv_proj,
                bias=weights["kv_bias"],
                session=session,
                name=f"{name}.kv_proj",
                workspace=workspace,
                activation_mode="auto",
            ).functional(session=session)
            self.key_projs = None
            self.value_proj = None
        else:
            self.kv_proj = None
            self.key_projs = tuple(
                prepare_projection(
                    scratch.embeddings,
                    weights[f"key_projs.{hc}"],
                    scratch.key_proj,
                    bias=weights[f"key_projs.{hc}.bias"],
                    session=session,
                    name=f"{name}.key_proj{hc}",
                    workspace=workspace,
                    activation_mode="auto",
                ).functional(session=session)
                for hc in range(HC_MULT)
            )
            self.value_proj = prepare_projection(
                scratch.embeddings,
                weights["value_proj"],
                scratch.value_proj,
                bias=weights["value_proj.bias"],
                session=session,
                name=f"{name}.value_proj",
                workspace=workspace,
                activation_mode="auto",
            ).functional(session=session)
        self.norm_keys = persistent_copy(
            torch.stack([weights[f"norm_key.{hc}"] for hc in range(HC_MULT)], dim=0)
        )  # [4, 3072]
        self.norm_queries = persistent_copy(
            torch.stack([weights[f"norm_query.{hc}"] for hc in range(HC_MULT)], dim=0)
        )  # [4, 3072]
        self.unified_pre_gamma = weights["unified_pre_gamma"]
        self.branch_scales = persistent_copy(
            torch.stack([weights[f"unified_per_hc_scales.{hc}"] for hc in range(HC_MULT)], dim=0)
        )  # [4, 3072]

        if not isinstance(source, torch._subclasses.FakeTensor):
            _dispatch_heads(self.weights_buf, self.scales_buf, self.hashes_buf, self.head_norms, self.fp4_lut,
                            scratch.embeddings, self.global_scale or 1.0, ENGRAM_HEADS if tokens > 64 else 1,
                            warmup=True)
            if self.kv_proj is None:
                keys, values = source, scratch.value_proj
            else:
                kv = scratch.kv_proj.view(tokens, HC_MULT + 1, HIDDEN_SIZE)
                keys, values = kv[:, :HC_MULT], kv[:, HC_MULT]
            _dispatch_mix(source, keys, values, self.norm_keys, self.norm_queries, self.unified_pre_gamma,
                          self.branch_scales, metadata.positions, metadata.token_ids, source, self.pad_id, warmup=True)

    def __call__(self, streams: torch.Tensor, metadata) -> torch.Tensor:
        tokens = streams.shape[0]
        barrier_in, barrier_out = self.wait.operands(streams)
        _wait_ready_op(
            streams,
            self.wait.timings.timestamps,
            self.wait.timings.frame,
            barrier_in,
            barrier_out,
            self.weights_buf,
            self.scales_buf,
            self.wait.interval,
            self.wait.handle,
            self.event_handle,
        )

        flat_emb = _decode_heads(self.weights_buf, self.scales_buf, self.hashes_buf, self.head_norms,
                                 self.fp4_lut, self.global_scale or 1.0)

        if self.kv_proj is not None:
            kv = self.kv_proj(flat_emb).view(tokens, HC_MULT + 1, HIDDEN_SIZE)
            keys_raw = kv[:, :HC_MULT, :]
            val_raw = kv[:, HC_MULT, :]
        else:
            keys_raw = torch.stack([proj(flat_emb) for proj in self.key_projs], dim=1)
            val_raw = self.value_proj(flat_emb)

        return _mix_streams(streams, keys_raw, val_raw, self.norm_keys, self.norm_queries,
                            self.unified_pre_gamma, self.branch_scales, metadata.positions,
                            metadata.token_ids, self.pad_id)
