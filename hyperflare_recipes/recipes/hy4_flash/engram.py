# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Host-backed FP8_E4M3_E8M0 / NVFP4_E4M3_GLOBAL n-gram tables, GPU hashing, and 4-stream Engram injection.

Dequantization runs in PyTorch rather than ``flare::staged_engram``
(``ignition/components/engram.py``, backed by ``b12x.sequence.engram._kernels.lookup_op``)
because that fused lookup kernel supports only 24-head width-256 FP8 tables, whereas
Hy4-flash uses 16-head width-96 tables in either FP8_E4M3_E8M0 or NVFP4_E4M3_GLOBAL format.
N-gram hashing likewise runs in PyTorch (``PreparedTableRequests``) because
``ignition.components.engram.PreparedEngramRequests`` targets order-4 24-head hashing with a
3-token history window, whereas Hy4-flash hashes order-3 (2-gram and 3-gram) 16-head tables
over a 2-token continuation history.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch

from ignition.components.engram import EngramInputs
from hyperflare_recipes.components.linear import prepare_projection
from ignition.components.ngram_delivery import publish_requests
from ignition.components.readiness import PreparedReadinessWait, ReadinessTimings, wait_for_readiness
from ignition.components.state import gather_rows, save_hidden
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.runtime_core import FileSource
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

        # Dequantize host-staged compact FP8_E4M3_E8M0 or NVFP4_E4M3_GLOBAL Engram rows.
        if self.encoding == "NVFP4_E4M3_GLOBAL":
            assert self.global_scale is not None and self.fp4_lut is not None
            deq = dequantize_nvfp4_engram(
                self.weights_buf.view(tokens, ENGRAM_HEADS, ENGRAM_NVFP4_ROW_BYTES),
                self.scales_buf.view(tokens, ENGRAM_HEADS, ENGRAM_NVFP4_SCALE_BYTES),
                self.global_scale,
                self.fp4_lut,
                dtype=streams.dtype,
            )
        else:
            q = self.weights_buf.view(torch.float8_e4m3fn).view(
                tokens, ENGRAM_HEADS, ENGRAM_FP8_SCALE_BYTES, ENGRAM_FP8_SCALE_BLOCK
            )
            exp = self.scales_buf.view(tokens, ENGRAM_HEADS, ENGRAM_FP8_SCALE_BYTES, 1).to(torch.float32) - 127.0
            deq = (q.float() * torch.exp2(exp)).reshape(tokens, ENGRAM_HEADS, ENGRAM_HEAD_DIM).to(streams.dtype)
        deq = torch.where(self.hashes_buf[:, :, None] >= 0, deq, 0)

        # Per-head RMSNorm matching PyTorch nn.RMSNorm(96, eps=None).
        eps_bf16 = torch.finfo(streams.dtype).eps
        inv_rms = torch.rsqrt(deq.float().square().mean(dim=-1, keepdim=True) + eps_bf16)
        normed = (deq.float() * inv_rms).to(streams.dtype) * self.head_norms[None, :, :]
        flat_emb = normed.reshape(tokens, ENGRAM_WIDTH).contiguous()

        if self.kv_proj is not None:
            kv = self.kv_proj(flat_emb).view(tokens, HC_MULT + 1, HIDDEN_SIZE)
            keys_raw = kv[:, :HC_MULT, :]
            val_raw = kv[:, HC_MULT, :]
        else:
            keys_raw = torch.stack([proj(flat_emb) for proj in self.key_projs], dim=1)
            val_raw = self.value_proj(flat_emb)

        # Vectorized 4-stream RMSNorm + dot-product gate retaining ATen rounding.
        k_norm = _rmsnorm(keys_raw, self.norm_keys[None, :, :], eps_bf16)
        q_norm = _rmsnorm(streams, self.norm_queries[None, :, :], eps_bf16)
        dot = (k_norm.float() * q_norm.float()).sum(dim=-1) * (HIDDEN_SIZE ** -0.5)  # [T, 4]
        g = dot.abs().clamp_min(1e-6).sqrt() * dot.sign()
        gates_t = torch.sigmoid(g).to(streams.dtype).unsqueeze(-1)  # [T, 4, 1]

        main_signal = self.unified_pre_gamma * _rmsnorm(val_raw, None, eps_bf16)  # [T, 3072]
        branch = main_signal[:, None, :] * self.branch_scales[None, :, :]  # [T, 4, 3072]
        delta = gates_t * main_signal[:, None, :] + branch

        mask = (
            (metadata.positions >= 0)
            & (metadata.token_ids != IMAGE_TOKEN)
            & (metadata.token_ids != IMAGE_NEWLINE_TOKEN)
            & (metadata.token_ids != VIDEO_TOKEN)
            & (metadata.token_ids != self.pad_id)
        )
        delta = torch.where(mask[:, None, None], delta, 0)
        return streams + delta
