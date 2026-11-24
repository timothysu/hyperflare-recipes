# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Streaming calibrated NVFP4 + MXFP8 + FP8/NVFP4-Engram quantizer for Tencent Hy4-flash.

Converts a BF16 ``tencent/Hy4-flash`` (``model_type == "hy_v4_flash"``) checkpoint
layer-by-layer into a ModelOpt ``MIXED_PRECISION`` checkpoint sized to fit on a
single 96 GiB Blackwell (SM120) GPU with NVMe/RAM-backed Engram tables:

- Routed experts (layers 1..41, 512 experts per layer): calibrated NVFP4
  (``group_size=16``, shared ``gate_proj``/``up_proj`` ``weight_scale_2`` and
  ``input_scale``) using Four-Over-Six + activation-weighted MSE block-scale
  selection, stored as stacked 3D tensors
  (``model.layers.{i}.mlp.experts.{gate_proj,up_proj,down_proj}.*``).
- Dense linears (``--dense-policy``): default ``selective`` quantizes the three
  largest backbone attention projections (``q_b_proj``, ``linear_gate``,
  ``o_proj``) to MXFP8 (``group_size=32``, ``F8_E4M3`` weights with ``F8_E8M0``
  block scales) and keeps smaller backbone and DFly2 linears in BF16; ``all``
  quantizes all backbone attention, layer-0 dense FFN, LatentMoE down/up,
  shared-expert, Engram dense, and DFly2 linears to MXFP8.
- Engram n-gram tables (4 layers): streamed chunk-by-chunk into either
  ``FP8_E4M3_E8M0`` (``scale_block=32``, ``F8_E4M3 [rows, 96]`` + ``F8_E8M0 [rows, 3]``)
  or ``NVFP4_E4M3_GLOBAL`` (``scale_block=16``, ``U8 [rows, 48]`` + ``F8_E4M3 [rows, 6]``
  + FP32 scalar ``weight_scale_2``), or repacked from an existing checkpoint via
  ``--repack-engram-nvfp4``.
- Embeddings, ``lm_head``, routers, compressed indexers, iHC parameters, norms,
  sinks, biases, DFly2 confidence/fusion gates, and SigLIP2 vision weights:
  preserved verbatim at checkpoint precision.

Streams one decoder layer at a time and reads only the unique Engram rows hit by
calibration tokens to keep host and device memory bounded.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import math
import os
import re
import shutil
import struct
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0
FP4_POSITIVE_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# Candidate scale multipliers relative to (block_amax / 6.0); includes 4-max (1.5x)
# and nearby points spanning clipping vs. resolution trade-offs.
DEFAULT_MSE_FACTORS = (
    0.65,
    0.72,
    0.80,
    0.88,
    0.94,
    1.00,
    1.06,
    1.14,
    1.25,
    1.38,
    1.50,  # exact Four-Over-Six (block_amax / 4.0)
    1.65,
)

# Built-in diverse calibration excerpts spanning code, math, reasoning, tool-use
# JSON, English prose, and Chinese text when no external corpus is supplied.
_BUILTIN_CALIB_TEXTS = (
    """def binary_search_left(arr: list[int], target: int) -> int:
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.order = {}
    def get(self, key: int) -> int:
        if key not in self.order:
            return -1
        val = self.order.pop(key)
        self.order[key] = val
        return val
    def put(self, key: int, value: int) -> None:
        if key in self.order:
            self.order.pop(key)
        elif len(self.order) >= self.capacity:
            self.order.pop(next(iter(self.order)))
        self.order[key] = value
""",
    """Theorem (Cauchy-Schwarz). For any vectors u, v in an inner product space,
|<u, v>| <= ||u|| * ||v||, with equality if and only if u and v are linearly dependent.
Proof: If v = 0 both sides vanish. Otherwise let t = <u, v> / <v, v> and expand
0 <= ||u - t v||^2 = <u, u> - |<u, v>|^2 / <v, v>. Rearranging gives the bound.
Application to variance reduction: for a randomized estimator X with control variate Y
having known expectation E[Y], the optimal linear combination X - c*(Y - E[Y]) minimizes
Var(X - c*Y) at c* = Cov(X, Y) / Var(Y), reducing variance by factor (1 - rho^2).
""",
    """<think:opensource>
The user asks for the current weather in Tokyo and Paris and wants a comparison in Celsius.
Let's emit two parallel tool calls inside <tool_calls:opensource> using the declared schema.
</think:opensource><tool_calls:opensource><tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Tokyo</arg_value:opensource><arg_key:opensource>units</arg_key:opensource><arg_value:opensource>metric</arg_value:opensource></tool_call:opensource><tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Paris</arg_value:opensource><arg_key:opensource>units</arg_key:opensource><arg_value:opensource>metric</arg_value:opensource></tool_call:opensource></tool_calls:opensource>
""",
    """在长上下文推理系统中，内存带宽和键值缓存容量共同决定了单卡吞吐上限。
多头潜在注意力（MLA）通过低秩投影将键和值压缩为共享潜在向量，显著减少了
显存占用；滑动窗口注意力与稀疏索引层的混合结构进一步把全局计算复杂度限制
在固定预算以内。与此同时，基于哈希查表的 n-gram 记忆层可将高频词法知识
卸载至主机内存与固态硬盘，仅在每次前向传播时传输当前批次命中的行。
""",
    """fn parse_header(buf: &[u8]) -> Result<(u64, &[u8]), &'static str> {
    if buf.len() < 8 {
        return Err("truncated safetensors header length");
    }
    let len = u64::from_le_bytes(buf[..8].try_into().unwrap()) as usize;
    if 8usize.checked_add(len).is_none_or(|end| end > buf.len()) {
        return Err("header length exceeds buffer");
    }
    Ok((len as u64, &buf[8..8 + len]))
}
""",
    """Let G = (V, E) be a directed acyclic graph with edge weights w: E -> R.
Shortest paths from source s can be computed in O(|V| + |E|) time by relaxing edges
in topological order: initialize d[s] = 0 and d[v] = +inf for v != s, then for each
vertex u in topological order and each outgoing edge (u, v), set d[v] = min(d[v], d[u] + w(u, v)).
Because every path visits vertices in strictly increasing topological rank, induction on
the rank shows d[v] equals the true shortest-path distance upon completion.
""",
    """{"schema_version": 2, "pipeline": {"stage": "calibration", "batch_size": 4,
"metrics": {"kl_divergence": 0.0142, "top1_agreement": 0.941, "perplexity_ratio": 1.008},
"hardware": {"gpu": "RTX PRO 6000 Blackwell", "vram_gib": 95.59, "host_ram_gib": 62.0},
"notes": ["Block size 16 for NVFP4 routed experts", "Block size 32 for MXFP8 dense projections"]}}
""",
    """To diagnose a numerical mismatch between eager PyTorch and a captured CUDA graph:
1. Verify all persistent scratch buffers are zeroed before replay so stale NaNs or
   previous-step activations cannot leak into inactive rows.
2. Check reduction order and accumulator precision (FP32 vs BF16) on RMSNorm, softmax,
   and residual stream additions.
3. Confirm RoPE layout (interleaved pairs vs split halves) and attention scaling factor.
4. Compare layer-by-layer cosine similarity and max absolute error against the reference.
""",
)


# ---------------------------------------------------------------------------
# Safetensors streaming I/O
# ---------------------------------------------------------------------------


class ShardedSafetensorsReader:
    """Lazy reader over a sharded (or single-file) safetensors checkpoint."""

    def __init__(self, checkpoint_dir: Path):
        self.root = checkpoint_dir
        index_path = checkpoint_dir / "model.safetensors.index.json"
        if index_path.is_file():
            self.weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]
        elif (checkpoint_dir / "model.safetensors").is_file():
            with safe_open(str(checkpoint_dir / "model.safetensors"), framework="pt", device="cpu") as h:
                self.weight_map = {k: "model.safetensors" for k in h.keys()}
        else:
            raise FileNotFoundError(f"No safetensors index or model.safetensors in {checkpoint_dir}")
        self._handles: dict[str, object] = {}

    def _open(self, shard: str):
        handle = self._handles.get(shard)
        if handle is None:
            # Keep open handle count bounded so we do not exhaust file descriptors.
            if len(self._handles) >= 16:
                self._handles.clear()
            handle = safe_open(str(self.root / shard), framework="pt", device="cpu")
            self._handles[shard] = handle
        return handle

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def keys(self) -> list[str]:
        return list(self.weight_map.keys())

    def get_tensor(self, name: str, device: torch.device | str = "cpu") -> torch.Tensor:
        shard = self.weight_map[name]
        tensor = self._open(shard).get_tensor(name)
        if str(device) != "cpu":
            tensor = tensor.to(device, non_blocking=True)
        return tensor

    def get_slice(self, name: str):
        shard = self.weight_map[name]
        return self._open(shard).get_slice(name)

    def raw_tensor_info(self, name: str) -> tuple[Path, str, list[int], int, int]:
        """Return (shard_path, dtype_str, shape, file_byte_offset, nbytes) without loading."""
        shard = self.weight_map[name]
        path = self.root / shard
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
        meta = header[name]
        start, end = meta["data_offsets"]
        return path, meta["dtype"], list(meta["shape"]), 8 + hlen + start, end - start

    def close(self) -> None:
        self._handles.clear()


_ST_TO_TORCH_DTYPE = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
}


class HfRangeSafetensorsReader:
    """Zero-disk tensor reader over a sharded Hugging Face safetensors repo via HTTP Range."""

    def __init__(
        self,
        repo_id: str,
        *,
        revision: str = "main",
        token: str | None = None,
        metadata_dir: Path | None = None,
        workers: int = 12,
    ):
        self.repo_id = repo_id
        self.revision = revision
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.workers = max(1, workers)
        self._owns_meta_dir = metadata_dir is None
        self.root = Path(tempfile.mkdtemp(prefix="hy4_hf_meta_")) if metadata_dir is None else metadata_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self._headers = {"User-Agent": "flare-hy4-quantizer/1.0"}
        if self.token:
            self._headers["Authorization"] = f"Bearer {self.token}"
        self._temp_files: dict[str, Path] = {}
        self._tensor_meta: dict[str, tuple[str, str, list[int], int, int]] = {}
        self._local = threading.local()

        # Fetch non-weight metadata files and model.safetensors.index.json.
        self._download_metadata_files()
        index_path = self.root / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"model.safetensors.index.json not found in HF repo {repo_id}@{revision}")
        self.weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]
        self._fetch_all_shard_headers()

    def _get_session(self):
        sess = getattr(self._local, "session", None)
        if sess is None:
            try:
                import requests
                from requests.adapters import HTTPAdapter

                sess = requests.Session()
                adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32)
                sess.mount("https://", adapter)
                sess.mount("http://", adapter)
                sess.headers.update(self._headers)
                self._local.session = sess
            except ImportError:
                self._local.session = False
                sess = False
        return sess

    def _resolve_url(self, filename: str) -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/{self.revision}/{filename}"

    @staticmethod
    def _validate_content_range(
        content_range: str | None,
        byte_range: tuple[int, int],
        *,
        allow_short: bool,
        url: str,
    ) -> int:
        start, end = byte_range
        if not content_range:
            raise RuntimeError(f"Missing Content-Range header for ranged response from {url}")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range.strip())
        if match is None:
            raise RuntimeError(f"Malformed Content-Range {content_range!r} for {url}")
        c_start = int(match.group(1))
        c_end = int(match.group(2))
        c_total = None if match.group(3) == "*" else int(match.group(3))
        if c_start != start or c_end < c_start or c_end > end:
            raise RuntimeError(
                f"Mismatched Content-Range {content_range!r} for requested bytes={start}-{end} from {url}"
            )
        if c_end < end:
            if not (allow_short and start == 0 and c_total is not None and c_end + 1 == c_total):
                raise RuntimeError(
                    f"Truncated Content-Range {content_range!r} for requested bytes={start}-{end} from {url}"
                )
        return c_end - c_start + 1

    def _http_get_bytes(
        self,
        url: str,
        *,
        byte_range: tuple[int, int] | None = None,
        allow_short: bool = False,
        retries: int = 6,
        timeout: float = 120.0,
    ) -> bytes:
        hdrs = dict(self._headers)
        if byte_range is not None:
            start, end = byte_range
            hdrs["Range"] = f"bytes={start}-{end}"

        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                sess = self._get_session()
                if sess:
                    resp = sess.get(
                        url,
                        headers=hdrs,
                        timeout=timeout,
                        allow_redirects=True,
                        stream=byte_range is not None,
                    )
                    try:
                        status = resp.status_code
                        if byte_range is not None:
                            if status != 206:
                                raise RuntimeError(f"Expected HTTP 206 for Range request, got HTTP {status} for {url}")
                            expected_body = self._validate_content_range(
                                resp.headers.get("Content-Range"),
                                byte_range,
                                allow_short=allow_short,
                                url=url,
                            )
                            buf = bytearray()
                            for chunk in resp.iter_content(chunk_size=min(65536, expected_body + 1)):
                                if not chunk:
                                    continue
                                buf.extend(chunk)
                                if len(buf) > expected_body:
                                    raise RuntimeError(
                                        f"Response body exceeded Content-Range length {expected_body} for {url}"
                                    )
                            if len(buf) != expected_body:
                                raise RuntimeError(f"Short read {len(buf)} != {expected_body} for {url}")
                            return bytes(buf)
                        if status != 200:
                            raise RuntimeError(f"HTTP {status} for {url}")
                        return resp.content
                    finally:
                        resp.close()
                else:
                    req = urllib.request.Request(url, headers=hdrs)
                    with urllib.request.urlopen(req, timeout=timeout) as r:
                        status = getattr(r, "status", 200)
                        if byte_range is not None:
                            if status != 206:
                                raise RuntimeError(f"Expected HTTP 206 for Range request, got HTTP {status} for {url}")
                            expected_body = self._validate_content_range(
                                r.headers.get("Content-Range"),
                                byte_range,
                                allow_short=allow_short,
                                url=url,
                            )
                            data = r.read(expected_body + 1)
                            if len(data) != expected_body:
                                raise RuntimeError(f"Short read {len(data)} != {expected_body} for {url}")
                            return data
                        return r.read()
            except Exception as exc:
                last_err = exc
                if attempt + 1 < retries:
                    time.sleep(min(2.0 ** attempt, 20.0))
        raise RuntimeError(f"Failed to fetch {url} after {retries} retries: {last_err}") from last_err

    def _download_metadata_files(self) -> None:
        api_url = f"https://huggingface.co/api/models/{self.repo_id}/tree/{self.revision}"
        entries = json.loads(self._http_get_bytes(api_url).decode("utf-8"))
        for item in entries:
            if item.get("type") != "file":
                continue
            name = item["path"]
            if "/" in name or name.endswith(".safetensors"):
                continue
            dst = self.root / name
            if not dst.is_file():
                dst.write_bytes(self._http_get_bytes(self._resolve_url(name)))

    def _fetch_shard_header(self, shard: str) -> tuple[str, int, dict]:
        url = self._resolve_url(shard)
        probe = self._http_get_bytes(url, byte_range=(0, 65535), allow_short=True)
        hlen = struct.unpack("<Q", probe[:8])[0]
        header_end = 8 + hlen
        if header_end > len(probe):
            rest = self._http_get_bytes(url, byte_range=(len(probe), header_end - 1))
            probe = probe + rest
        hdr = json.loads(probe[8:header_end].decode("utf-8"))
        return shard, header_end, hdr

    def _fetch_all_shard_headers(self) -> None:
        shards = sorted(set(self.weight_map.values()))
        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(shards))) as pool:
            for shard, base_off, hdr in pool.map(self._fetch_shard_header, shards):
                for name, meta in hdr.items():
                    if name == "__metadata__":
                        continue
                    s0, s1 = meta["data_offsets"]
                    self._tensor_meta[name] = (
                        shard,
                        meta["dtype"],
                        list(meta["shape"]),
                        base_off + s0,
                        s1 - s0,
                    )
        print(
            f"[hf] Indexed {len(self._tensor_meta)} tensors across {len(shards)} shards "
            f"via HTTP Range in {time.monotonic() - t0:.1f}s"
        )

    def has(self, name: str) -> bool:
        return name in self._tensor_meta

    def keys(self) -> list[str]:
        return list(self._tensor_meta.keys())

    def _fetch_single_tensor_cpu(self, name: str, *, chunk_bytes: int = 64 * (1 << 20)) -> torch.Tensor:
        shard, st_dtype, shape, base_off, nbytes = self._tensor_meta[name]
        dtype = _ST_TO_TORCH_DTYPE[st_dtype]
        if nbytes == 0:
            return torch.empty(shape, dtype=dtype)
        url = self._resolve_url(shard)
        if nbytes <= chunk_bytes:
            raw = bytearray(self._http_get_bytes(url, byte_range=(base_off, base_off + nbytes - 1)))
        else:
            raw = bytearray(nbytes)
            spans = [
                (off, min(off + chunk_bytes, nbytes) - 1)
                for off in range(0, nbytes, chunk_bytes)
            ]

            def _get_span(span: tuple[int, int]) -> tuple[int, bytes]:
                s0, s1 = span
                return s0, self._http_get_bytes(url, byte_range=(base_off + s0, base_off + s1))

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.workers, len(spans))) as pool:
                for s0, chunk in pool.map(_get_span, spans):
                    raw[s0 : s0 + len(chunk)] = chunk
        return torch.frombuffer(raw, dtype=torch.uint8).view(dtype).view(shape).clone()

    def fetch_tensors(self, names: list[str]) -> dict[str, torch.Tensor]:
        """Download a layer's tensors in parallel via HTTP Range into CPU RAM."""
        if not names:
            return {}
        # Split giant tensors (>64 MiB) across chunk workers and fetch small tensors in parallel.
        big = [n for n in names if self._tensor_meta[n][4] > 64 * (1 << 20)]
        small = [n for n in names if self._tensor_meta[n][4] <= 64 * (1 << 20)]
        out: dict[str, torch.Tensor] = {}
        if small:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.workers, len(small))) as pool:
                futs = {pool.submit(self._fetch_single_tensor_cpu, n): n for n in small}
                for fut in concurrent.futures.as_completed(futs):
                    out[futs[fut]] = fut.result()
        for n in big:
            out[n] = self._fetch_single_tensor_cpu(n)
        return out

    def get_tensor(self, name: str, device: torch.device | str = "cpu") -> torch.Tensor:
        tensor = self._fetch_single_tensor_cpu(name)
        if str(device) != "cpu":
            tensor = tensor.to(device, non_blocking=True)
        return tensor

    def materialize_temp_shard(self, tensor_name: str, temp_dir: Path | None = None) -> Path:
        """Download the dedicated shard for a huge table (e.g. Engram) to a temporary local file."""
        shard, _, _, base_off, nbytes = self._tensor_meta[tensor_name]
        if shard in self._temp_files and self._temp_files[shard].is_file():
            return self._temp_files[shard]
        dst_dir = self.root if temp_dir is None else temp_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / f"tmp_{shard}"
        total_bytes = base_off + nbytes
        url = self._resolve_url(shard)
        t0 = time.monotonic()
        chunk_bytes = 64 * (1 << 20)
        # Pre-allocate file and fill 64 MiB chunks in parallel for multi-GB/s cloud download speed.
        with open(dst_path, "wb") as f:
            f.truncate(total_bytes)
        spans = [
            (off, min(off + chunk_bytes, total_bytes) - 1)
            for off in range(0, total_bytes, chunk_bytes)
        ]
        write_lock = threading.Lock()
        with open(dst_path, "r+b") as fout:
            def _fetch_and_write(span: tuple[int, int]) -> None:
                s0, s1 = span
                data = self._http_get_bytes(url, byte_range=(s0, s1))
                with write_lock:
                    fout.seek(s0)
                    fout.write(data)

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
                list(pool.map(_fetch_and_write, spans))
        dt = time.monotonic() - t0
        print(f"[hf] Downloaded temporary shard {shard} ({total_bytes / (1 << 30):.2f} GiB) in {dt:.1f}s ({total_bytes / (1 << 20) / max(dt, 1e-3):.0f} MiB/s)")
        self._temp_files[shard] = dst_path
        return dst_path

    def release_temp_shard(self, tensor_name: str) -> None:
        shard = self._tensor_meta[tensor_name][0]
        path = self._temp_files.pop(shard, None)
        if path is not None and path.exists():
            path.unlink()

    def raw_tensor_info(self, name: str) -> tuple[Path, str, list[int], int, int]:
        shard, st_dtype, shape, base_off, nbytes = self._tensor_meta[name]
        if shard not in self._temp_files:
            self.materialize_temp_shard(name)
        return self._temp_files[shard], st_dtype, shape, base_off, nbytes

    def close(self) -> None:
        for path in list(self._temp_files.values()):
            if path.exists():
                path.unlink()
        self._temp_files.clear()
        if self._owns_meta_dir and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)


_TORCH_TO_ST_DTYPE = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.float8_e8m0fnu: "F8_E8M0",
}


def validate_sync_paths(output_dir: Path, sync_dir: Path | None) -> None:
    """Reject sync_dir paths that alias or nest with output_dir."""
    if sync_dir is None:
        return
    out_r = output_dir.resolve()
    sync_r = sync_dir.resolve()
    if out_r == sync_r or out_r.is_relative_to(sync_r) or sync_r.is_relative_to(out_r):
        raise ValueError(
            f"sync_dir ({sync_dir}) must not resolve to the same directory as or nest with output_dir ({output_dir})"
        )


def _safetensors_header_bytes(path: Path) -> tuple[dict, int]:
    """Read and validate a safetensors header, returning ``(header_dict, total_tensor_bytes)``."""
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"Truncated safetensors header length in {path}")
        hl = struct.unpack("<Q", raw_len)[0]
        raw_hdr = f.read(hl)
        if len(raw_hdr) != hl:
            raise ValueError(f"Truncated safetensors header payload in {path}")
    hdr = json.loads(raw_hdr.decode("utf-8"))
    nbytes = sum(
        int(v["data_offsets"][1]) - int(v["data_offsets"][0])
        for k, v in hdr.items()
        if k != "__metadata__"
    )
    return hdr, nbytes


class ShardedSafetensorsWriter:
    """Writes tensors into size-bounded safetensors shards and builds the index.

    When ``sync_dir`` is set, each completed shard is mirrored asynchronously to
    ``sync_dir`` (e.g. remote or slower storage) on a background thread so slow
    remote filesystems do not stall GPU quantization. With ``move_to_sync_dir``
    enabled, the local copy is unlinked after each verified transfer to bound
    ephemeral local disk usage.
    """

    def __init__(
        self,
        output_dir: Path,
        max_shard_bytes: int = 4 * (1 << 30),
        *,
        sync_dir: Path | None = None,
        move_to_sync_dir: bool = False,
    ):
        validate_sync_paths(output_dir, sync_dir)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_shard_bytes = max_shard_bytes
        self.sync_dir = sync_dir
        self.move_to_sync_dir = bool(move_to_sync_dir and sync_dir is not None)
        if self.sync_dir is not None:
            self.sync_dir.mkdir(parents=True, exist_ok=True)
        self._sync_pool = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1)
            if self.sync_dir is not None
            else None
        )
        self._sync_futures: list[concurrent.futures.Future] = []
        self.buffer: dict[str, torch.Tensor] = {}
        self.buffer_bytes = 0
        self.shard_files: list[str] = []
        self.weight_map: dict[str, str] = {}
        self.total_bytes = 0

    def _enqueue_sync(self, shard_name: str) -> None:
        if self._sync_pool is None or self.sync_dir is None:
            return
        src = self.output_dir / shard_name
        dst = self.sync_dir / shard_name
        move = self.move_to_sync_dir

        def _copy_one() -> None:
            tmp = dst.with_suffix(".safetensors.part")
            shutil.copyfile(src, tmp)
            if tmp.stat().st_size != src.stat().st_size:
                raise IOError(f"Size mismatch syncing {shard_name} to {dst}")
            os.replace(tmp, dst)
            if move and src.exists():
                src.unlink()

        self._sync_futures.append(self._sync_pool.submit(_copy_one))

    def _drain_sync(self) -> None:
        if self._sync_pool is None:
            return
        for fut in self._sync_futures:
            fut.result()
        self._sync_futures.clear()

    def add(self, name: str, tensor: torch.Tensor) -> None:
        cpu_tensor = tensor.detach().to("cpu").contiguous().clone()
        nbytes = cpu_tensor.numel() * cpu_tensor.element_size()
        if self.buffer and self.buffer_bytes + nbytes > self.max_shard_bytes:
            self.flush()
        self.buffer[name] = cpu_tensor
        self.buffer_bytes += nbytes

    def flush(self) -> None:
        if not self.buffer:
            return
        idx = len(self.shard_files) + 1
        tmp_name = f"shard-{idx:05d}.safetensors"
        save_file(self.buffer, str(self.output_dir / tmp_name))
        self.shard_files.append(tmp_name)
        for name, t in self.buffer.items():
            self.weight_map[name] = tmp_name
            self.total_bytes += t.numel() * t.element_size()
        self.buffer.clear()
        self.buffer_bytes = 0
        self._enqueue_sync(tmp_name)

    def register_external_shard(
        self,
        tmp_name: str,
        entries: dict[str, int],
    ) -> None:
        """Register a shard written directly by the streaming binary writer."""
        self.shard_files.append(tmp_name)
        for name, nbytes in entries.items():
            self.weight_map[name] = tmp_name
            self.total_bytes += nbytes
        self._enqueue_sync(tmp_name)

    def finalize(self) -> dict:
        self.flush()
        self._drain_sync()
        if self._sync_pool is not None:
            self._sync_pool.shutdown(wait=True)
            self._sync_pool = None
        total = len(self.shard_files)
        rename_map: dict[str, str] = {}
        for i, old_name in enumerate(self.shard_files, start=1):
            new_name = f"model-{i:05d}-of-{total:05d}.safetensors"
            if old_name != new_name:
                if (self.output_dir / old_name).exists():
                    os.replace(self.output_dir / old_name, self.output_dir / new_name)
                if self.sync_dir is not None and (self.sync_dir / old_name).exists():
                    os.replace(self.sync_dir / old_name, self.sync_dir / new_name)
            rename_map[old_name] = new_name
        final_map = {k: rename_map[v] for k, v in sorted(self.weight_map.items())}
        expected_shards = sorted(set(final_map.values()))
        target_dir = self.sync_dir if (self.move_to_sync_dir and self.sync_dir is not None) else self.output_dir
        missing = [name for name in expected_shards if not (target_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Cannot publish index: missing shard(s) in {target_dir}: {missing[:5]}"
            )
        if self.sync_dir is not None and not self.move_to_sync_dir:
            missing_sync = [name for name in expected_shards if not (self.sync_dir / name).is_file()]
            if missing_sync:
                raise FileNotFoundError(
                    f"Cannot publish index: missing synced shard(s) in {self.sync_dir}: {missing_sync[:5]}"
                )
        index = {
            "metadata": {"total_size": self.total_bytes},
            "weight_map": final_map,
        }
        index_text = json.dumps(index, indent=2) + "\n"
        if not self.move_to_sync_dir:
            out_tmp = self.output_dir / "model.safetensors.index.json.part"
            out_tmp.write_text(index_text)
            os.replace(out_tmp, self.output_dir / "model.safetensors.index.json")
        if self.sync_dir is not None:
            sync_tmp = self.sync_dir / "model.safetensors.index.json.part"
            sync_tmp.write_text(index_text)
            os.replace(sync_tmp, self.sync_dir / "model.safetensors.index.json")
        return index


def write_streaming_safetensors(
    path: Path,
    specs: list[tuple[str, str, list[int], int]],
    data_writer,
) -> None:
    """Write a safetensors file with a precomputed header and streamed tensor payloads.

    ``specs`` is a list of ``(name, st_dtype, shape, nbytes)`` written strictly in
    order. ``data_writer(file_handle)`` must write exact ``sum(nbytes)`` bytes.
    """
    header: dict[str, object] = {}
    offset = 0
    for name, st_dtype, shape, nbytes in specs:
        header[name] = {
            "dtype": st_dtype,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(header_bytes)) % 8
    header_bytes += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        written = data_writer(f)
    if written != offset:
        raise RuntimeError(
            f"Streaming safetensors {path.name} wrote {written} bytes, expected {offset}"
        )


# ---------------------------------------------------------------------------
# Quantization primitives (NVFP4 4/6 + MSE, MXFP8, FP8_E4M3_E8M0)
# ---------------------------------------------------------------------------


_FP4_POS_GRID_CACHE: dict[torch.device, torch.Tensor] = {}
_FP4_SIGNED_GRID_CACHE: dict[torch.device, torch.Tensor] = {}


def _fp4_pos_grid(device: torch.device) -> torch.Tensor:
    t = _FP4_POS_GRID_CACHE.get(device)
    if t is None:
        t = torch.tensor(FP4_POSITIVE_GRID, dtype=torch.float32, device=device)
        _FP4_POS_GRID_CACHE[device] = t
    return t


def _fp4_signed_grid(device: torch.device) -> torch.Tensor:
    t = _FP4_SIGNED_GRID_CACHE.get(device)
    if t is None:
        t = torch.tensor(
            [
                0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
            ],
            dtype=torch.float32,
            device=device,
        )
        _FP4_SIGNED_GRID_CACHE[device] = t
    return t


def _quantize_e2m1_rtne(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact round-to-nearest-even FP4 E2M1 quantization and 4-bit code encoding.

    Matches Blackwell ``cvt.rn.satfinite.e2m1x2.f32`` and ``flashinfer.fp4_quantize``:
    magnitudes in ``{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`` with ties rounded to even
    code index ``{0, 2, 4, 6}`` (i.e. ``{0.0, 1.0, 2.0, 4.0}``).
    """
    neg = x < 0
    sign_bit = neg.to(torch.uint8) << 3
    ax = x.abs()
    # Tie-to-even boundaries for positive E2M1 codes 0..7:
    # 0.25 -> 0 (even), 0.75 -> 2 (even), 1.25 -> 2 (even), 1.75 -> 4 (even),
    # 2.50 -> 4 (even), 3.50 -> 6 (even), 5.00 -> 6 (even).
    code = (
        (ax > 0.25).to(torch.uint8)
        + (ax >= 0.75).to(torch.uint8)
        + (ax > 1.25).to(torch.uint8)
        + (ax >= 1.75).to(torch.uint8)
        + (ax > 2.5).to(torch.uint8)
        + (ax >= 3.5).to(torch.uint8)
        + (ax > 5.0).to(torch.uint8)
    )
    deq = _fp4_pos_grid(x.device)[code.long()]
    deq = torch.where(neg, -deq, deq)
    return (code | sign_bit), deq


def pack_e2m1_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """Pack ``[..., K]`` uint8 E2M1 codes (0..15) into ``[..., K // 2]`` uint8 bytes."""
    low = codes[..., 0::2] & 0x0F
    high = (codes[..., 1::2] & 0x0F) << 4
    return (low | high).contiguous()


@torch.no_grad()
def quantize_nvfp4_mse_block16(
    weight: torch.Tensor,
    weight_scale_2: torch.Tensor,
    hessian_diag: torch.Tensor | None = None,
    mse_factors: tuple[float, ...] = DEFAULT_MSE_FACTORS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D ``[N, K]`` weight matrix to NVFP4 (group_size=16) via
    Four-Over-Six + activation-weighted MSE block-scale search.

    Returns:
      - ``packed``: ``uint8 [N, K // 2]``
      - ``weight_scale``: ``float8_e4m3fn [N, K // 16]`` (unswizzled row-major)
    """
    assert weight.ndim == 2 and weight.shape[1] % 16 == 0
    n, k = weight.shape
    num_blocks = k // 16
    w = weight.float().view(n, num_blocks, 16)
    gs2 = float(weight_scale_2.item())
    if not (math.isfinite(gs2) and gs2 > 0.0):
        raise ValueError(f"weight_scale_2 must be positive finite, got {gs2}")

    if hessian_diag is not None:
        h = hessian_diag.to(device=weight.device, dtype=torch.float32).view(1, num_blocks, 16)
        h = h.clamp_min(1e-12)
    else:
        h = None

    block_amax = w.abs().amax(dim=-1)  # [N, K//16]
    base_scale = block_amax / (FP4_E2M1_MAX * gs2)  # exact 6-max FP8 scale target

    best_err = torch.full((n, num_blocks), float("inf"), dtype=torch.float32, device=weight.device)
    best_scale_fp8 = torch.zeros((n, num_blocks), dtype=torch.float8_e4m3fn, device=weight.device)
    best_codes = torch.zeros((n, num_blocks, 16), dtype=torch.uint8, device=weight.device)

    min_fp8 = float(torch.finfo(torch.float8_e4m3fn).tiny)
    for factor in mse_factors:
        cand_fp8 = (base_scale * factor).clamp(0.0, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        cand_f32 = cand_fp8.float()
        # Effective block multiplier in original weight coordinates:
        eff_scale = cand_f32 * gs2
        inv_eff = torch.where(eff_scale > 0.0, eff_scale.reciprocal(), torch.zeros_like(eff_scale))
        scaled = w * inv_eff.unsqueeze(-1)
        codes, deq_q = _quantize_e2m1_rtne(scaled)
        recon = deq_q * eff_scale.unsqueeze(-1)
        diff_sq = (w - recon).square()
        if h is not None:
            err = (diff_sq * h).sum(dim=-1)
        else:
            err = diff_sq.sum(dim=-1)
        # Preserve zero blocks cleanly with zero scale.
        zero_mask = block_amax == 0.0
        err = torch.where(zero_mask & (cand_f32 > min_fp8), torch.full_like(err, float("inf")), err)
        improved = err < best_err
        best_err = torch.where(improved, err, best_err)
        best_scale_fp8 = torch.where(improved, cand_fp8, best_scale_fp8)
        best_codes = torch.where(improved.unsqueeze(-1), codes, best_codes)

    packed = pack_e2m1_nibbles(best_codes.view(n, k))
    return packed, best_scale_fp8.contiguous()


@torch.no_grad()
def dequantize_nvfp4(
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
) -> torch.Tensor:
    """Exact dequantization of NVFP4 ``(packed, weight_scale, weight_scale_2)`` to FP32."""
    n, half_k = packed.shape
    k = half_k * 2
    low = (packed & 0x0F).long()
    high = ((packed >> 4) & 0x0F).long()
    codes = torch.stack((low, high), dim=-1).view(n, k)
    vals = _fp4_signed_grid(packed.device)[codes].view(n, k // 16, 16)
    scales = weight_scale.float().unsqueeze(-1) * float(weight_scale_2.item())
    return (vals * scales).view(n, k)


@torch.no_grad()
def quantize_nvfp4_gptq_batched(
    weights: torch.Tensor,
    weight_scale_2: torch.Tensor,
    hessians: torch.Tensor,
    counts: torch.Tensor,
    *,
    mse_factors: tuple[float, ...] = DEFAULT_MSE_FACTORS,
    base_damp: float = 0.01,
    min_gptq_count: int = 256,
    blocksize: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched Block-16 GPTQ + Four-Over-Six NVFP4 quantization for ``[B, N, K]`` experts.

    Runs full ``K x K`` Cholesky inverse per expert in batch, selects optimal FP8
    block-16 scales via activation-weighted Four-Over-Six / MSE search on the
    error-compensated weights, and propagates quantization error across columns
    within each ``blocksize`` superblock and lazily to remaining columns via ``bmm``.
    Experts with ``count < min_gptq_count`` or singular Hessians fall back cleanly to
    diagonal-Hessian MSE quantization.
    """
    bsz, n, k = weights.shape
    assert k % 16 == 0 and blocksize % 16 == 0
    device = weights.device

    packed_out = torch.empty((bsz, n, k // 2), dtype=torch.uint8, device=device)
    scales_out = torch.empty((bsz, n, k // 16), dtype=torch.float8_e4m3fn, device=device)

    W = weights.float().clone()
    gs2 = weight_scale_2.float()  # [B]
    cnt_list = counts.tolist()

    # Build Cholesky inverse Hinv [B_ok, K, K] for experts with sufficient calibration tokens.
    ok_indices: list[int] = []
    hinv_list: list[torch.Tensor] = []
    hdiag_list: list[torch.Tensor] = []
    idx_k = torch.arange(k, device=device)

    for bi in range(bsz):
        c = int(cnt_list[bi])
        if c < min_gptq_count:
            h_d = (torch.diagonal(hessians[bi]) / max(c, 1)).float() if c > 0 else None
            packed_out[bi], scales_out[bi] = quantize_nvfp4_mse_block16(
                weights[bi], gs2[bi], h_d, mse_factors
            )
            continue
        Hf = (hessians[bi] / float(c)).float().clone()
        diag = torch.diagonal(Hf).clone()
        dead = diag <= 0
        if dead.any():
            Hf[dead, dead] = 1.0
            W[bi, :, dead] = 0.0
        mean_d = diag[~dead].mean().clamp_min(1e-8) if (~dead).any() else torch.tensor(1.0, device=device)
        damp_ratio = max(base_damp, min(0.25, 0.05 * (k / max(float(c), 1.0))))
        Hf[idx_k, idx_k] += damp_ratio * mean_d
        try:
            L = torch.linalg.cholesky(Hf)
            Hinv = torch.cholesky_inverse(L)
            Hinv = torch.linalg.cholesky(Hinv, upper=True)
        except RuntimeError:
            packed_out[bi], scales_out[bi] = quantize_nvfp4_mse_block16(
                weights[bi], gs2[bi], diag.clamp_min(1e-12), mse_factors
            )
            continue
        ok_indices.append(bi)
        hinv_list.append(Hinv)
        hdiag_list.append(diag.clamp_min(1e-12))

    if not ok_indices:
        return packed_out, scales_out

    ok_idx_t = torch.tensor(ok_indices, dtype=torch.long, device=device)
    Wb = W[ok_idx_t]  # [Bok, N, K]
    Hinv_b = torch.stack(hinv_list, dim=0)  # [Bok, K, K]
    hdiag_b = torch.stack(hdiag_list, dim=0)  # [Bok, K]
    gs2_b = gs2[ok_idx_t]  # [Bok]
    bok = Wb.shape[0]

    codes_b = torch.empty((bok, n, k), dtype=torch.uint8, device=device)
    scales_b = torch.empty((bok, n, k // 16), dtype=torch.float8_e4m3fn, device=device)
    min_fp8 = float(torch.finfo(torch.float8_e4m3fn).tiny)

    for b0 in range(0, k, blocksize):
        b1 = min(b0 + blocksize, k)
        blk_w = b1 - b0
        W_blk = Wb[:, :, b0:b1].clone()
        Err_blk = torch.zeros_like(W_blk)
        Hinv_blk = Hinv_b[:, b0:b1, b0:b1]  # [Bok, blk_w, blk_w]

        for g0 in range(0, blk_w, 16):
            g1 = g0 + 16
            grp_idx = (b0 + g0) // 16
            sub = W_blk[:, :, g0:g1]  # [Bok, N, 16]
            sub_h = hdiag_b[:, b0 + g0 : b0 + g1].unsqueeze(1)  # [Bok, 1, 16]
            b_amax = sub.abs().amax(dim=-1)  # [Bok, N]
            base_s = b_amax / (FP4_E2M1_MAX * gs2_b[:, None])  # [Bok, N]

            best_err = torch.full((bok, n), float("inf"), dtype=torch.float32, device=device)
            best_s_fp8 = torch.zeros((bok, n), dtype=torch.float8_e4m3fn, device=device)
            zero_mask = b_amax == 0.0
            for fac in mse_factors:
                c_fp8 = (base_s * fac).clamp(0.0, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
                c_f32 = c_fp8.float()
                eff = c_f32 * gs2_b[:, None]
                inv_e = torch.where(eff > 0.0, eff.reciprocal(), torch.zeros_like(eff))
                _, dq = _quantize_e2m1_rtne(sub * inv_e.unsqueeze(-1))
                err = ((sub - dq * eff.unsqueeze(-1)).square() * sub_h).sum(dim=-1)
                err = torch.where(zero_mask & (c_f32 > min_fp8), torch.full_like(err, float("inf")), err)
                imp = err < best_err
                best_err = torch.where(imp, err, best_err)
                best_s_fp8 = torch.where(imp, c_fp8, best_s_fp8)

            scales_b[:, :, grp_idx] = best_s_fp8
            eff = best_s_fp8.float() * gs2_b[:, None]  # [Bok, N]
            inv_e = torch.where(eff > 0.0, eff.reciprocal(), torch.zeros_like(eff))

            for c in range(16):
                col = g0 + c
                w_col = W_blk[:, :, col]  # [Bok, N]
                code_c, dq_c = _quantize_e2m1_rtne(w_col * inv_e)
                q_col = dq_c * eff
                codes_b[:, :, b0 + col] = code_c
                d_inv = Hinv_blk[:, col, col].unsqueeze(1)  # [Bok, 1]
                err_c = (w_col - q_col) / d_inv  # [Bok, N]
                Err_blk[:, :, col] = err_c
                if col + 1 < blk_w:
                    W_blk[:, :, col + 1 :].sub_(
                        err_c.unsqueeze(-1) * Hinv_blk[:, col, col + 1 :].unsqueeze(1)
                    )

        if b1 < k:
            Wb[:, :, b1:].baddbmm_(Err_blk, Hinv_b[:, b0:b1, b1:], beta=1.0, alpha=-1.0)

    packed_out[ok_idx_t] = pack_e2m1_nibbles(codes_b)
    scales_out[ok_idx_t] = scales_b
    return packed_out, scales_out


@torch.no_grad()
def quantize_mxfp8_block32(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D ``[N, K]`` tensor to MXFP8 (group_size=32, E4M3 + E8M0 scales).

    Evaluates both the non-clipping power-of-two exponent ``exp_hi = ceil(log2(amax / 448))``
    and ``exp_hi - 1`` per 32-element block, selecting whichever yields lower squared
    reconstruction error (avoiding half-bit resolution loss when block amax barely
    crosses a power-of-two boundary).

    Returns:
      - ``q``: ``float8_e4m3fn [N, K]``
      - ``scale``: ``float8_e8m0fnu [N, K // 32]``
    """
    assert weight.ndim == 2 and weight.shape[1] % 32 == 0
    n, k = weight.shape
    blocks = weight.float().view(n, k // 32, 32)
    amax = blocks.abs().amax(dim=-1)  # [N, K//32]
    ratio = (amax / FP8_E4M3_MAX).clamp_min(2.0 ** -127)
    exp_hi = torch.ceil(torch.log2(ratio)).clamp(-127.0, 127.0)
    exp_hi = torch.where(amax > 0.0, exp_hi, torch.zeros_like(exp_hi))
    exp_lo = (exp_hi - 1.0).clamp(-127.0, 127.0)

    def _eval(exp_cand: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_s = torch.where(amax > 0.0, torch.exp2(-exp_cand), torch.zeros_like(exp_cand))
        s = torch.where(amax > 0.0, torch.exp2(exp_cand), torch.zeros_like(exp_cand))
        q_c = (blocks * inv_s.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        err = (blocks - q_c.float() * s.unsqueeze(-1)).square().sum(dim=-1)
        return q_c, err

    q_hi, err_hi = _eval(exp_hi)
    q_lo, err_lo = _eval(exp_lo)
    use_lo = (amax > 0.0) & (err_lo < err_hi)
    best_exp = torch.where(use_lo, exp_lo, exp_hi)
    best_q = torch.where(use_lo.unsqueeze(-1), q_lo, q_hi)
    scale_u8 = (best_exp.to(torch.int32) + 127).to(torch.uint8)
    return best_q.view(n, k).contiguous(), scale_u8.view(torch.float8_e8m0fnu).contiguous()


@torch.no_grad()
def dequantize_mxfp8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Exact dequantization of MXFP8 ``(F8_E4M3, F8_E8M0)`` to FP32."""
    n, k = q.shape
    exp = scale.view(torch.uint8).to(torch.float32) - 127.0
    factor = torch.exp2(exp).unsqueeze(-1)
    return (q.float().view(n, k // 32, 32) * factor).view(n, k)


# ---------------------------------------------------------------------------
# Reference Hy4-flash layer forward for streaming calibration & verification
# ---------------------------------------------------------------------------


def _rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    dtype = x.dtype
    xf = x.float()
    out = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        return weight * out.to(dtype)
    return out.to(dtype)


def _pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    """Match ``torch.nn.RMSNorm(..., eps=None)`` used in ``HYV4FlashEngramLayer``."""
    return F.rms_norm(x, (x.shape[-1],), weight, eps=None)


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _build_1d_rope(seq_len: int, head_dim: int, theta: float, device: torch.device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = pos[:, None] * inv_freq[None, :]
    cos = freqs.cos().to(torch.bfloat16).repeat_interleave(2, dim=-1)
    sin = freqs.sin().to(torch.bfloat16).repeat_interleave(2, dim=-1)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, S, H, D], cos/sin: [S, D]
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    out = (x.float() * c) + (_rotate_half_interleaved(x).float() * s)
    return out.to(x.dtype)


def _draw_engram_multipliers(layer_id_1based: int, compressed_vocab_size: int, max_ngram: int, seed: int) -> torch.Tensor:
    half_bound = max(1, np.iinfo(np.int64).max // compressed_vocab_size // 2)
    gen = np.random.default_rng(seed + 10007 * layer_id_1based)
    drawn = gen.integers(0, half_bound, size=max_ngram, dtype=np.int64)
    return torch.tensor((drawn * 2 + 1).tolist(), dtype=torch.int64)


def _build_ngram_tokens(input_ids: torch.Tensor, max_ngram: int, pad_id: int) -> torch.Tensor:
    bsz, seqlen = input_ids.shape
    pos = torch.arange(seqlen, device=input_ids.device)[None, :]
    shifted = [input_ids]
    for offset in range(1, max_ngram):
        prev = F.pad(input_ids, (offset, 0), value=pad_id)[:, :seqlen]
        shifted.append(torch.where(pos < offset, pad_id, prev))
    return torch.stack(shifted, dim=-1)


def _hash_engram_rows(
    ngram_tokens: torch.Tensor,
    lookup_table: torch.Tensor,
    multipliers: torch.Tensor,
    head_vocab_sizes: list[int],
    pad_id: int,
    max_ngram: int,
    n_head_per_ngram: int,
) -> torch.Tensor:
    """Return flat table row indices ``[B, S, num_heads]`` (including head offsets)."""
    table_size = lookup_table.shape[0]
    compressed = lookup_table[ngram_tokens.clamp_max(table_size - 1)]
    compressed = torch.where(ngram_tokens < table_size, compressed, lookup_table[pad_id])
    mixed = compressed.long() * multipliers.to(ngram_tokens.device)
    moduli_all = torch.tensor(head_vocab_sizes, dtype=torch.int64, device=ngram_tokens.device)
    offsets = torch.tensor([0] + head_vocab_sizes[:-1], dtype=torch.int64, device=ngram_tokens.device).cumsum(0)
    parts = []
    for ngram_size in range(2, max_ngram + 1):
        mix = mixed[..., 0]
        for k in range(1, ngram_size):
            mix = torch.bitwise_xor(mix, mixed[..., k])
        s = (ngram_size - 2) * n_head_per_ngram
        m = moduli_all[s : s + n_head_per_ngram]
        parts.append(mix.unsqueeze(-1) % m)
    return torch.cat(parts, dim=-1) + offsets


def _gather_engram_rows_from_file(
    reader: ShardedSafetensorsReader | HfRangeSafetensorsReader,
    tensor_name: str,
    flat_row_ids: torch.Tensor,
    *,
    scale_name: str | None = None,
    global_scale_name: str | None = None,
) -> torch.Tensor:
    """Read only the unique Engram rows hit by calibration sequences from disk.

    Works for BF16 source tables, FP8_E4M3_E8M0 tables, and NVFP4_E4M3_GLOBAL
    tables without mapping or loading the full 8-30 GiB table into RAM.
    """
    unique_rows, inv = torch.unique(flat_row_ids.cpu().reshape(-1), return_inverse=True)
    unique_list = unique_rows.tolist()

    w_path, w_dtype, w_shape, w_base, _ = reader.raw_tensor_info(tensor_name)
    cols = w_shape[1]
    if scale_name is None:
        assert w_dtype == "BF16"
        row_bytes = cols * 2
        buf = bytearray(len(unique_list) * row_bytes)
        with open(w_path, "rb") as f:
            for i, row in enumerate(unique_list):
                f.seek(w_base + row * row_bytes)
                buf[i * row_bytes : (i + 1) * row_bytes] = f.read(row_bytes)
        table_sub = torch.frombuffer(buf, dtype=torch.uint8).view(torch.bfloat16).view(len(unique_list), cols).clone()
        out_cols = cols
    elif w_dtype == "F8_E4M3":
        s_path, s_dtype, s_shape, s_base, _ = reader.raw_tensor_info(scale_name)
        assert s_dtype == "F8_E8M0"
        s_cols = s_shape[1]
        w_buf = bytearray(len(unique_list) * cols)
        s_buf = bytearray(len(unique_list) * s_cols)
        with open(w_path, "rb") as fw, open(s_path, "rb") as fs:
            for i, row in enumerate(unique_list):
                fw.seek(w_base + row * cols)
                w_buf[i * cols : (i + 1) * cols] = fw.read(cols)
                fs.seek(s_base + row * s_cols)
                s_buf[i * s_cols : (i + 1) * s_cols] = fs.read(s_cols)
        q = torch.frombuffer(w_buf, dtype=torch.uint8).view(torch.float8_e4m3fn).view(len(unique_list), cols)
        s = torch.frombuffer(s_buf, dtype=torch.uint8).view(torch.float8_e8m0fnu).view(len(unique_list), s_cols)
        table_sub = dequantize_mxfp8(q, s).to(torch.bfloat16).clone()
        out_cols = cols
    elif w_dtype == "U8":
        assert global_scale_name is not None
        s_path, s_dtype, s_shape, s_base, _ = reader.raw_tensor_info(scale_name)
        assert s_dtype == "F8_E4M3"
        s_cols = s_shape[1]
        gs = reader.get_tensor(global_scale_name, device="cpu").float()
        w_buf = bytearray(len(unique_list) * cols)
        s_buf = bytearray(len(unique_list) * s_cols)
        with open(w_path, "rb") as fw, open(s_path, "rb") as fs:
            for i, row in enumerate(unique_list):
                fw.seek(w_base + row * cols)
                w_buf[i * cols : (i + 1) * cols] = fw.read(cols)
                fs.seek(s_base + row * s_cols)
                s_buf[i * s_cols : (i + 1) * s_cols] = fs.read(s_cols)
        q = torch.frombuffer(w_buf, dtype=torch.uint8).view(len(unique_list), cols)
        s = torch.frombuffer(s_buf, dtype=torch.uint8).view(torch.float8_e4m3fn).view(len(unique_list), s_cols)
        table_sub = dequantize_nvfp4(q, s, gs).to(torch.bfloat16).clone()
        out_cols = cols * 2
    else:
        raise ValueError(f"Unsupported Engram table dtype {w_dtype} for {tensor_name}")

    gathered = table_sub[inv].view(*flat_row_ids.shape, out_cols)
    return gathered


def _fp8_round_trip(x: torch.Tensor) -> torch.Tensor:
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / FP8_E4M3_MAX
    return (x / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn).to(x.dtype) * scale


@dataclass
class ExpertCalibStats:
    """Per-expert calibration statistics accumulated over all calibration micro-batches."""

    x1_amax: torch.Tensor  # [E] float32
    x1_sq_sum: torch.Tensor  # [E, D_lat] float64
    x2_amax: torch.Tensor  # [E] float32
    x2_sq_sum: torch.Tensor  # [E, D_inter] float64
    counts: torch.Tensor  # [E] int64
    H1: torch.Tensor | None = None  # [E, D_lat, D_lat] float32 (for GPTQ)
    H2: torch.Tensor | None = None  # [E, D_inter, D_inter] float32 (for GPTQ)
    global_x1_amax: float = 0.0
    global_x2_amax: float = 0.0

    @classmethod
    def create(
        cls,
        num_experts: int,
        latent_dim: int,
        inter_dim: int,
        device: torch.device,
        *,
        use_gptq: bool = False,
    ) -> ExpertCalibStats:
        return cls(
            x1_amax=torch.zeros(num_experts, dtype=torch.float32, device=device),
            x1_sq_sum=torch.zeros((num_experts, latent_dim), dtype=torch.float64, device=device),
            x2_amax=torch.zeros(num_experts, dtype=torch.float32, device=device),
            x2_sq_sum=torch.zeros((num_experts, inter_dim), dtype=torch.float64, device=device),
            counts=torch.zeros(num_experts, dtype=torch.int64, device=device),
            H1=torch.zeros((num_experts, latent_dim, latent_dim), dtype=torch.float32, device=device) if use_gptq else None,
            H2=torch.zeros((num_experts, inter_dim, inter_dim), dtype=torch.float32, device=device) if use_gptq else None,
        )


def _ihc_forward(
    streams: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    hc_magnitude: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    hc_mult = streams.shape[2]
    flat = streams.flatten(2).float()
    inv_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + rms_eps)
    mixes = F.linear(flat, fn.float()) * inv_rms
    pre_b, post_b = base.float().split(hc_mult, dim=-1)
    pre_s, post_s = scale.float().unbind(0)
    pre_l, post_l = mixes.split(hc_mult, dim=-1)
    pre = torch.sigmoid(pre_l * pre_s + pre_b) + hc_eps
    post = hc_magnitude * torch.sigmoid(post_l * post_s + post_b) + hc_eps
    out = (pre.unsqueeze(-1) * streams.float()).sum(dim=2).to(streams.dtype)
    return post, out


def _ihc_head(
    streams: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    flat = streams.flatten(2).float()
    inv_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + rms_eps)
    mixes = F.linear(flat, fn.float()) * inv_rms
    pre = torch.sigmoid(mixes * scale.float() + base.float()) + hc_eps
    return (pre.unsqueeze(-1) * streams.float()).sum(dim=2).to(streams.dtype)


def _layer_config_info(text_cfg: dict, layer_idx: int) -> dict:
    layer_types = text_cfg.get("layer_types")
    if layer_types is None:
        is_indexed = layer_idx < 2 or (layer_idx - 1) % 4 == 0
    else:
        lt = layer_types[layer_idx]
        is_indexed = lt in ("indexed_attention", "deepseek_sparse_attention")
    if is_indexed:
        num_heads = int(text_cfg["num_attention_heads"])
        qk_nope = int(text_cfg["qk_nope_head_dim"])
        qk_rope = int(text_cfg["qk_rope_head_dim"])
        v_dim = int(text_cfg["v_head_dim"])
        rope_theta = 10000000.0
    else:
        num_heads = int(text_cfg.get("swa_num_attention_heads", 64))
        qk_nope = int(text_cfg.get("swa_qk_nope_head_dim", text_cfg["qk_nope_head_dim"]))
        qk_rope = int(text_cfg.get("swa_qk_rope_head_dim", text_cfg["qk_rope_head_dim"]))
        v_dim = int(text_cfg.get("swa_v_head_dim", text_cfg["v_head_dim"]))
        rope_theta = 10000.0
    rp = text_cfg.get("rope_parameters") or {}
    if isinstance(rp, dict):
        key = "indexed_attention" if is_indexed else "sliding_attention"
        alt = "deepseek_sparse_attention" if is_indexed else "sliding_attention"
        entry = rp.get(key) or rp.get(alt)
        if isinstance(entry, dict) and "rope_theta" in entry:
            rope_theta = float(entry["rope_theta"])
    mlp_types = text_cfg.get("mlp_layer_types")
    is_sparse_mlp = (layer_idx > 0) if mlp_types is None else (mlp_types[layer_idx] == "sparse")
    engram_ids = text_cfg.get("engram_layer_ids") or []
    has_engram = (layer_idx + 1) in engram_ids
    return dict(
        is_indexed=is_indexed,
        num_heads=num_heads,
        qk_nope=qk_nope,
        qk_rope=qk_rope,
        v_dim=v_dim,
        rope_theta=rope_theta,
        is_sparse_mlp=is_sparse_mlp,
        has_engram=has_engram,
    )


def _load_weight_for_forward(
    weights: dict[str, torch.Tensor],
    prefix: str,
    *,
    dequantize: bool,
) -> torch.Tensor:
    w = weights[f"{prefix}.weight"]
    if not dequantize or w.dtype in (torch.bfloat16, torch.float32, torch.float16):
        return w
    if w.dtype == torch.float8_e4m3fn:
        return dequantize_mxfp8(w, weights[f"{prefix}.weight_scale"]).to(torch.bfloat16)
    raise ValueError(f"Unsupported linear dtype {w.dtype} for {prefix}")


@torch.no_grad()
def _forward_decoder_layer(
    hidden_streams: torch.Tensor,
    layer_idx: int,
    text_cfg: dict,
    weights: dict[str, torch.Tensor],
    *,
    engram_gathered: torch.Tensor | None = None,
    engram_mask: torch.Tensor | None = None,
    stats: ExpertCalibStats | None = None,
    quantized_experts: dict[str, torch.Tensor] | None = None,
    dequantize_mxfp8_linears: bool = False,
) -> torch.Tensor:
    """Exact single-layer Hy4-flash forward (BF16 or simulated-quantized)."""
    bsz, seqlen, hc_mult, hidden_size = hidden_streams.shape
    device = hidden_streams.device
    info = _layer_config_info(text_cfg, layer_idx)
    p = f"model.layers.{layer_idx}"
    rms_eps = float(text_cfg.get("rms_norm_eps", 1e-5))
    hc_eps = float(text_cfg.get("hc_eps", 1e-6))
    hc_mag = float(text_cfg.get("hc_magnitude", 2.0))
    linear_w = lambda name: _load_weight_for_forward(weights, name, dequantize=dequantize_mxfp8_linears)

    # 1. Optional Engram memory injection before attention
    if info["has_engram"] and engram_gathered is not None:
        ep = f"{p}.engram_layer"
        emb = engram_gathered.to(device=device, dtype=hidden_streams.dtype)
        if text_cfg.get("engram_fp8_embedding", True) and not dequantize_mxfp8_linears:
            emb = _fp8_round_trip(emb)
        num_hash_heads = emb.shape[2]
        emb = torch.stack(
            [
                _pytorch_rmsnorm(emb[:, :, h], weights[f"{ep}.lookup_per_head_norms.{h}.weight"])
                for h in range(num_hash_heads)
            ],
            dim=2,
        ).flatten(start_dim=-2)
        gates = []
        for hci in range(hc_mult):
            k_proj = F.linear(emb, linear_w(f"{ep}.key_projs.{hci}"), weights[f"{ep}.key_projs.{hci}.bias"])
            key = _pytorch_rmsnorm(k_proj, weights[f"{ep}.norm_key.{hci}.weight"])
            query = _pytorch_rmsnorm(hidden_streams[:, :, hci], weights[f"{ep}.norm_query.{hci}.weight"])
            gate = (key * query).sum(dim=-1) * (hidden_size ** -0.5)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gates.append(gate.sigmoid())
        gates_t = torch.stack(gates, dim=2).unsqueeze(-1)
        val_proj = F.linear(emb, linear_w(f"{ep}.value_proj"), weights[f"{ep}.value_proj.bias"])
        main_signal = weights[f"{ep}.unified_pre_gamma"] * _pytorch_rmsnorm(val_proj, None)
        branch = torch.stack(
            [weights[f"{ep}.unified_per_hc_scales.{hci}"] * main_signal for hci in range(hc_mult)],
            dim=2,
        )
        engram_out = gates_t * main_signal.unsqueeze(2) + branch
        if engram_mask is not None:
            engram_out = engram_out.masked_fill(~engram_mask.to(device)[:, :, None, None], 0.0)
        hidden_streams = hidden_streams + engram_out

    # 2. Attention iHC + MLA (sparse IndexPool or SWA-512) with sinks & output gate
    residual = hidden_streams
    post_attn, x_attn = _ihc_forward(
        hidden_streams,
        weights[f"{p}.hc_attn_layer.hc_pre.hc_fn"],
        weights[f"{p}.hc_attn_layer.hc_pre.hc_base"],
        weights[f"{p}.hc_attn_layer.hc_pre.hc_scale"],
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        hc_magnitude=hc_mag,
    )
    x_attn = _rms_norm(x_attn, weights[f"{p}.input_layernorm.weight"], rms_eps)

    ap = f"{p}.self_attn"
    num_heads = info["num_heads"]
    qk_nope = info["qk_nope"]
    qk_rope = info["qk_rope"]
    qk_head_dim = qk_nope + qk_rope
    v_dim = info["v_dim"]
    kv_rank = int(text_cfg["kv_lora_rank"])

    gate_states = F.linear(x_attn, linear_w(f"{ap}.linear_gate")).view(bsz, seqlen, num_heads, v_dim)
    q_resid = _rms_norm(F.linear(x_attn, linear_w(f"{ap}.q_a_proj")), weights[f"{ap}.q_a_layernorm.weight"], rms_eps)
    q_states = F.linear(q_resid, linear_w(f"{ap}.q_b_proj")).view(bsz, seqlen, num_heads, qk_head_dim)
    q_pass, q_rot = q_states.split([qk_nope, qk_rope], dim=-1)

    comp_kv = F.linear(x_attn, linear_w(f"{ap}.kv_a_proj_with_mqa"))
    kv_pass_raw, k_rot = comp_kv.split([kv_rank, qk_rope], dim=-1)
    k_pass = _rms_norm(kv_pass_raw, weights[f"{ap}.kv_a_layernorm.weight"], rms_eps)
    k_rot = k_rot.unsqueeze(2)  # [B, S, 1, 64]

    cos, sin = _build_1d_rope(seqlen, qk_rope, info["rope_theta"], device)
    q_rot = _apply_rope(q_rot, cos, sin)
    k_rot = _apply_rope(k_rot, cos, sin)

    query = torch.cat((q_pass, q_rot), dim=-1).transpose(1, 2)  # [B, H, S, 256]
    kv_exp = F.linear(k_pass, linear_w(f"{ap}.kv_b_proj")).view(bsz, seqlen, num_heads, qk_nope + v_dim).transpose(1, 2)
    k_nope, v_states = kv_exp.split([qk_nope, v_dim], dim=-1)
    k_states = torch.cat((k_nope, k_rot.transpose(1, 2).expand(-1, num_heads, -1, -1)), dim=-1)

    # Exact causal / SWA-512 / compressed-indexer mask
    idx = torch.arange(seqlen, device=device)
    if info["is_indexed"]:
        valid = idx[:, None] >= idx[None, :]
        index_topk = int(text_cfg.get("index_topk", 2048))
        if seqlen > index_topk:
            cp = f"{ap}.compressed_indexer"
            ratio = int(text_cfg.get("compressed_indexer_compress_ratio", text_cfg.get("index_kpool", 4)))
            idx_heads = int(text_cfg.get("compressed_indexer_num_attention_heads", text_cfg.get("index_n_heads", 16)))
            idx_dim = int(text_cfg.get("compressed_indexer_head_dim", text_cfg.get("index_head_dim", 128)))
            tkeys = _rms_norm(F.linear(x_attn, weights[f"{cp}.token_key_projection.weight"]), weights[f"{cp}.token_key_norm.weight"], rms_eps)
            ng = seqlen // ratio
            grouped = tkeys[:, : ng * ratio].reshape(bsz, ng, ratio, idx_dim).float()
            vals = F.linear(grouped, weights[f"{cp}.key_value_projection.weight"].float())
            gts = F.linear(grouped, weights[f"{cp}.key_gate_projection.weight"].float()) + weights[f"{cp}.intra_group_ape"].float()
            gkeys = _rms_norm((vals * gts.softmax(dim=2)).sum(dim=2), weights[f"{cp}.key_norm.weight"], rms_eps).to(x_attn.dtype)
            iq = F.linear(q_resid, weights[f"{cp}.query_projection.weight"]).view(bsz, seqlen, idx_heads, idx_dim)
            nope_d = idx_dim - qk_rope
            iq_p, iq_r = iq.split([nope_d, qk_rope], dim=-1)
            iq = torch.cat((iq_p, _apply_rope(iq_r, cos, sin)), dim=-1)
            gcos, gsin = cos[: ng * ratio : ratio], sin[: ng * ratio : ratio]
            gk_p, gk_r = gkeys.unsqueeze(2).split([nope_d, qk_rope], dim=-1)
            gkeys = torch.cat((gk_p, _apply_rope(gk_r, gcos, gsin)), dim=-1).squeeze(2)
            hw = F.linear(x_attn, weights[f"{cp}.head_weight_projection.weight"]).float() * (idx_heads ** -0.5 * idx_dim ** -0.5)
            sc = F.relu(torch.matmul(iq.float(), gkeys.float().unsqueeze(1).transpose(-1, -2)))
            lg = torch.matmul(hw.unsqueeze(-2), sc).squeeze(-2)
            q_grp = idx // ratio
            g_ids = torch.arange(ng, device=device)
            lg = lg.masked_fill(g_ids[None, None, :] >= q_grp[None, :, None], float("-inf"))
            topk_g = min(index_topk // ratio - 1, ng)
            rec_g = torch.cat((lg.topk(topk_g, dim=-1).indices, q_grp.view(1, seqlen, 1).expand(bsz, -1, -1)), dim=-1)
            offs = torch.arange(ratio, device=device)
            rec = (rec_g.unsqueeze(-1) * ratio + offs).flatten(-2).clamp(max=seqlen - 1)
            sel = torch.zeros((bsz, seqlen, seqlen), dtype=torch.bool, device=device)
            sel.scatter_(-1, rec, True)
            valid = valid.unsqueeze(0) & sel
        else:
            valid = valid.unsqueeze(0)
    else:
        win = int(text_cfg.get("sliding_window", 512))
        valid = ((idx[:, None] >= idx[None, :]) & ((idx[:, None] - idx[None, :]) < win)).unsqueeze(0)

    # Exact attention with learnable sinks (computed per-sequence in float32 softmax)
    scale = qk_head_dim ** -0.5
    sinks = weights[f"{ap}.learnable_sink_param"].float().view(1, num_heads, 1, 1)
    attn_scores = torch.matmul(query.float(), k_states.float().transpose(-1, -2)) * scale
    attn_scores = attn_scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    attn_scores = torch.cat((attn_scores, sinks.expand(bsz, -1, seqlen, -1)), dim=-1)
    probs = F.softmax(attn_scores, dim=-1)[..., :-1].to(v_states.dtype)
    attn_out = torch.matmul(probs, v_states).transpose(1, 2).contiguous()  # [B, S, H, 256]
    attn_out = (attn_out * torch.sigmoid(gate_states)).reshape(bsz, seqlen, -1)
    attn_out = _rms_norm(F.linear(attn_out, linear_w(f"{ap}.o_proj")), weights[f"{ap}.post_proj_norm.weight"], rms_eps)
    hidden_streams = (post_attn.float().unsqueeze(-1) * attn_out.float().unsqueeze(-2) + residual.float()).to(hidden_streams.dtype)

    # 3. FFN / LatentMoE iHC
    residual = hidden_streams
    post_ffn, x_ffn = _ihc_forward(
        hidden_streams,
        weights[f"{p}.hc_mlp_layer.hc_pre.hc_fn"],
        weights[f"{p}.hc_mlp_layer.hc_pre.hc_base"],
        weights[f"{p}.hc_mlp_layer.hc_pre.hc_scale"],
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        hc_magnitude=hc_mag,
    )
    x_ffn = _rms_norm(x_ffn, weights[f"{p}.post_attention_layernorm.weight"], rms_eps)

    mp = f"{p}.mlp"
    if not info["is_sparse_mlp"]:
        gate = F.linear(x_ffn, linear_w(f"{mp}.gate_proj"))
        up = F.linear(x_ffn, linear_w(f"{mp}.up_proj"))
        mlp_out = F.linear(F.silu(gate) * up, linear_w(f"{mp}.down_proj"))
        mlp_out = _rms_norm(mlp_out, weights[f"{mp}.post_fc2_norm.weight"], rms_eps)
    else:
        flat_x = x_ffn.reshape(-1, hidden_size)
        router_w = weights[f"{mp}.gate.weight"].float()
        router_bias = weights[f"{mp}.gate.e_score_correction_bias"].float()
        router_logits = F.linear(flat_x.float(), router_w)
        scores = router_logits.sigmoid()
        choice = scores + router_bias
        top_k = int(text_cfg["num_experts_per_tok"])
        topk_idx = torch.topk(choice, k=top_k, dim=-1, sorted=False).indices
        topk_w = scores.gather(1, topk_idx)
        if text_cfg.get("norm_topk_prob", True):
            topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
        topk_w = topk_w * float(text_cfg["routed_scaling_factor"])

        routed_in = F.linear(flat_x, linear_w(f"{mp}.routed_expert_down_proj"))
        if stats is not None:
            stats.global_x1_amax = max(stats.global_x1_amax, float(routed_in.abs().amax().item()))

        num_experts = int(text_cfg["n_routed_experts"])
        swiglu_limit = float(text_cfg.get("swiglu_limit", 10.0))
        routed_acc = torch.zeros_like(routed_in)

        # Group tokens by expert ID using a single sort (no 512-way one-hot allocation).
        flat_exp = topk_idx.reshape(-1)
        flat_tok = torch.arange(flat_x.shape[0], device=device).repeat_interleave(top_k)
        flat_wt = topk_w.reshape(-1)
        order = torch.argsort(flat_exp)
        sorted_exp = flat_exp[order]
        sorted_tok = flat_tok[order]
        sorted_wt = flat_wt[order]
        counts = torch.bincount(sorted_exp, minlength=num_experts)
        offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
        offsets[1:] = counts.cumsum(0)
        counts_cpu = counts.tolist()
        offsets_cpu = offsets.tolist()

        for e in range(num_experts):
            c = counts_cpu[e]
            if c == 0:
                continue
            s0, s1 = offsets_cpu[e], offsets_cpu[e + 1]
            toks = sorted_tok[s0:s1]
            wts = sorted_wt[s0:s1]
            xe = routed_in[toks]  # [c, 1536]
            if stats is not None:
                xf = xe.float()
                stats.x1_amax[e] = torch.maximum(stats.x1_amax[e], xf.abs().amax())
                stats.x1_sq_sum[e] += xf.square().sum(dim=0, dtype=torch.float64)
                if stats.H1 is not None:
                    stats.H1[e].addmm_(xf.T, xf)
                stats.counts[e] += c

            if quantized_experts is not None:
                if "deq_gate" in quantized_experts:
                    wg = quantized_experts["deq_gate"][e]
                    wu = quantized_experts["deq_up"][e]
                    wd = quantized_experts["deq_down"][e]
                else:
                    wg = dequantize_nvfp4(
                        quantized_experts["gate_w"][e],
                        quantized_experts["gate_s"][e],
                        quantized_experts["gate_s2"][e],
                    ).to(xe.dtype)
                    wu = dequantize_nvfp4(
                        quantized_experts["up_w"][e],
                        quantized_experts["up_s"][e],
                        quantized_experts["up_s2"][e],
                    ).to(xe.dtype)
                    wd = dequantize_nvfp4(
                        quantized_experts["down_w"][e],
                        quantized_experts["down_s"][e],
                        quantized_experts["down_s2"][e],
                    ).to(xe.dtype)
                gate = F.linear(xe, wg)
                up = F.linear(xe, wu)
            else:
                w_gu = weights[f"{mp}.experts.gate_up_proj"][e]
                gu = F.linear(xe, w_gu)
                gate, up = gu.chunk(2, dim=-1)
                wd = weights[f"{mp}.experts.down_proj"][e]

            if swiglu_limit > 0.0:
                gate = gate.clamp(max=swiglu_limit)
                up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
            inter = F.silu(gate) * up
            if stats is not None:
                inf = inter.float()
                amax2 = inf.abs().amax()
                stats.x2_amax[e] = torch.maximum(stats.x2_amax[e], amax2)
                stats.x2_sq_sum[e] += inf.square().sum(dim=0, dtype=torch.float64)
                if stats.H2 is not None:
                    stats.H2[e].addmm_(inf.T, inf)
                stats.global_x2_amax = max(stats.global_x2_amax, float(amax2.item()))

            out_e = F.linear(inter, wd) * wts.to(inter.dtype).unsqueeze(-1)
            routed_acc.index_add_(0, toks, out_e.to(routed_acc.dtype))

        routed_out = F.linear(routed_acc, linear_w(f"{mp}.routed_expert_up_proj"))
        sh_gate = F.linear(flat_x, linear_w(f"{mp}.shared_experts.gate_proj"))
        sh_up = F.linear(flat_x, linear_w(f"{mp}.shared_experts.up_proj"))
        sh_out = F.linear(F.silu(sh_gate) * sh_up, linear_w(f"{mp}.shared_experts.down_proj"))
        mlp_out = _rms_norm((routed_out + sh_out).view(bsz, seqlen, hidden_size), weights[f"{mp}.combine_output_norm.weight"], rms_eps)

    hidden_streams = (post_ffn.float().unsqueeze(-1) * mlp_out.float().unsqueeze(-2) + residual.float()).to(hidden_streams.dtype)
    return hidden_streams


# ---------------------------------------------------------------------------
# Calibration token loading
# ---------------------------------------------------------------------------


def build_calibration_tokens(
    input_dir: Path,
    num_samples: int,
    seq_len: int,
    *,
    calib_file: Path | None = None,
    seed: int = 42,
) -> torch.Tensor:
    """Return ``int64 [num_samples, seq_len]`` calibration token IDs."""
    from tokenizers import Tokenizer

    tok_path = input_dir / "tokenizer.json"
    if not tok_path.is_file():
        raise FileNotFoundError(f"Expected tokenizer.json in {input_dir}")
    tokenizer = Tokenizer.from_file(str(tok_path))

    texts: list[str] = []
    if calib_file is not None:
        raw = calib_file.read_text(encoding="utf-8")
        if calib_file.suffix == ".jsonl":
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, str):
                    texts.append(obj)
                elif isinstance(obj, dict):
                    for field in ("text", "content", "prompt"):
                        if isinstance(obj.get(field), str):
                            texts.append(obj[field])
                            break
        else:
            texts = [chunk for chunk in raw.split("\n\n") if chunk.strip()]
    else:
        texts = list(_BUILTIN_CALIB_TEXTS)

    if not texts:
        raise ValueError("Calibration text source is empty")

    # Concatenate tokenized excerpts with varied shuffles so every sequence covers
    # multiple domains (code, math, reasoning, tool calls, Chinese, JSON).
    rng = np.random.default_rng(seed)
    encoded_docs = [tokenizer.encode(t).ids for t in texts if t.strip()]
    encoded_docs = [ids for ids in encoded_docs if ids]
    total_needed = num_samples * seq_len
    stream: list[int] = []
    while len(stream) < total_needed:
        perm = rng.permutation(len(encoded_docs))
        for idx in perm:
            stream.extend(encoded_docs[int(idx)])
            if len(stream) >= total_needed:
                break
    return torch.tensor(stream[:total_needed], dtype=torch.int64).view(num_samples, seq_len)


# ---------------------------------------------------------------------------
# Streaming Engram table FP8_E4M3_E8M0 conversion
# ---------------------------------------------------------------------------


def stream_quantize_engram_table(
    reader: ShardedSafetensorsReader,
    writer: ShardedSafetensorsWriter,
    layer_idx: int,
    device: torch.device,
    *,
    chunk_rows: int = 262144,
) -> tuple[int, int]:
    """Stream one BF16 Engram table from disk -> GPU -> FP8_E4M3_E8M0 shard on disk."""
    src_name = f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.embedding.weight"
    dst_weight = f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.weight"
    dst_scale = f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.scale"

    src_path, src_dtype, shape, base_offset, _ = reader.raw_tensor_info(src_name)
    if src_dtype != "BF16" or len(shape) != 2 or shape[1] % 32 != 0:
        raise ValueError(f"{src_name}: expected BF16 [rows, multiple_of_32], got {src_dtype} {shape}")
    rows, cols = shape
    scale_cols = cols // 32

    shard_idx = len(writer.shard_files) + 1
    shard_name = f"shard-{shard_idx:05d}.safetensors"
    shard_path = writer.output_dir / shard_name

    # Hold scales (~475 MiB for 165M x 3) in a flat CPU buffer so weights stream
    # sequentially into the first tensor span, followed immediately by scales.
    all_scales = bytearray(rows * scale_cols)
    weight_nbytes = rows * cols
    scale_nbytes = rows * scale_cols

    def write_payload(fout) -> int:
        written = 0
        with open(src_path, "rb") as fin:
            fin.seek(base_offset)
            row_bytes_in = cols * 2
            for start in range(0, rows, chunk_rows):
                count = min(chunk_rows, rows - start)
                raw = fin.read(count * row_bytes_in)
                chunk = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(count, cols).to(device)
                q, s = quantize_mxfp8_block32(chunk)
                q_bytes = q.view(torch.uint8).cpu().numpy().tobytes()
                s_bytes = s.view(torch.uint8).cpu().numpy().tobytes()
                fout.write(q_bytes)
                written += len(q_bytes)
                all_scales[start * scale_cols : (start + count) * scale_cols] = s_bytes
        fout.write(all_scales)
        written += len(all_scales)
        return written

    specs = [
        (dst_weight, "F8_E4M3", [rows, cols], weight_nbytes),
        (dst_scale, "F8_E8M0", [rows, scale_cols], scale_nbytes),
    ]
    write_streaming_safetensors(shard_path, specs, write_payload)
    writer.register_external_shard(shard_name, {dst_weight: weight_nbytes, dst_scale: scale_nbytes})
    return weight_nbytes, scale_nbytes


ENGRAM_NVFP4_MSE_FACTORS = (0.88, 1.00, 1.25, 1.50)


def _compute_engram_nvfp4_global_scale(
    reader: ShardedSafetensorsReader | HfRangeSafetensorsReader,
    layer_idx: int,
    device: torch.device,
    *,
    chunk_rows: int = 262144,
) -> float:
    """Compute exact table-wide NVFP4 weight_scale_2 leaving 1.5x headroom for Four-Over-Six."""
    src_name = f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.embedding.weight"
    src_path, src_dtype, shape, base_offset, _ = reader.raw_tensor_info(src_name)
    if src_dtype != "BF16" or len(shape) != 2 or shape[1] % 16 != 0:
        raise ValueError(f"{src_name}: expected BF16 [rows, multiple_of_16], got {src_dtype} {shape}")
    rows, cols = shape
    row_bytes_in = cols * 2
    max_abs = 1e-6
    with open(src_path, "rb") as fin:
        fin.seek(base_offset)
        for start in range(0, rows, chunk_rows):
            count = min(chunk_rows, rows - start)
            raw = fin.read(count * row_bytes_in)
            chunk = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(torch.bfloat16).view(count, cols).to(device)
            max_abs = max(max_abs, float(chunk.abs().amax().item()))
    return max_abs / (FP8_E4M3_MAX * 4.0)


def _round_trip_engram_rows(
    rows: torch.Tensor,
    *,
    engram_format: str,
    global_scale: float | None = None,
    mse_factors: tuple[float, ...] = ENGRAM_NVFP4_MSE_FACTORS,
) -> torch.Tensor:
    """Round-trip gathered Engram BF16 rows through the exact writer encoding and table scale."""
    flat = rows.reshape(-1, rows.shape[-1])
    if engram_format == "nvfp4":
        if global_scale is None:
            raise ValueError("NVFP4 Engram round-trip requires the table-wide global_scale")
        gs_t = torch.tensor(float(global_scale), dtype=torch.float32, device=flat.device)
        p, s = quantize_nvfp4_mse_block16(flat, gs_t, None, mse_factors)
        out = dequantize_nvfp4(p, s, gs_t).to(torch.bfloat16).view_as(rows)
        del p, s, gs_t
        return out
    eq, es = quantize_mxfp8_block32(flat)
    out = dequantize_mxfp8(eq, es).to(torch.bfloat16).view_as(rows)
    del eq, es
    return out


def stream_quantize_engram_table_nvfp4(
    reader: ShardedSafetensorsReader | HfRangeSafetensorsReader,
    writer: ShardedSafetensorsWriter | None,
    layer_idx: int,
    device: torch.device,
    *,
    chunk_rows: int = 262144,
    shard_path_override: Path | None = None,
    mse_factors: tuple[float, ...] = ENGRAM_NVFP4_MSE_FACTORS,
    global_scale: float | None = None,
) -> tuple[int, int, float]:
    """Stream one BF16 Engram table from disk -> GPU -> NVFP4_E4M3_GLOBAL shard (48B + 6B/row)."""
    src_name = f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.embedding.weight"
    dst_weight = f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.weight"
    dst_scale = f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.scale"
    dst_gs = f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.weight_scale_2"

    src_path, src_dtype, shape, base_offset, _ = reader.raw_tensor_info(src_name)
    if src_dtype != "BF16" or len(shape) != 2 or shape[1] % 16 != 0:
        raise ValueError(f"{src_name}: expected BF16 [rows, multiple_of_16], got {src_dtype} {shape}")
    rows, cols = shape
    packed_cols = cols // 2
    scale_cols = cols // 16
    row_bytes_in = cols * 2

    if global_scale is None:
        global_scale = _compute_engram_nvfp4_global_scale(reader, layer_idx, device, chunk_rows=chunk_rows)
    gs_t = torch.tensor([global_scale], dtype=torch.float32, device=device)

    if shard_path_override is not None:
        shard_path = shard_path_override
        shard_name = shard_path.name
    else:
        assert writer is not None
        shard_idx = len(writer.shard_files) + 1
        shard_name = f"shard-{shard_idx:05d}.safetensors"
        shard_path = writer.output_dir / shard_name

    all_scales = bytearray(rows * scale_cols)
    weight_nbytes = rows * packed_cols
    scale_nbytes = rows * scale_cols
    gs_bytes = struct.pack("<f", global_scale)

    def write_payload(fout) -> int:
        written = 0
        with open(src_path, "rb") as fin:
            fin.seek(base_offset)
            for start in range(0, rows, chunk_rows):
                count = min(chunk_rows, rows - start)
                raw = fin.read(count * row_bytes_in)
                chunk = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(torch.bfloat16).view(count, cols).to(device)
                p, s = quantize_nvfp4_mse_block16(chunk, gs_t[0], None, mse_factors)
                p_bytes = p.cpu().numpy().tobytes()
                s_bytes = s.view(torch.uint8).cpu().numpy().tobytes()
                fout.write(p_bytes)
                written += len(p_bytes)
                all_scales[start * scale_cols : (start + count) * scale_cols] = s_bytes
        fout.write(all_scales)
        written += len(all_scales)
        fout.write(gs_bytes)
        written += len(gs_bytes)
        return written

    specs = [
        (dst_weight, "U8", [rows, packed_cols], weight_nbytes),
        (dst_scale, "F8_E4M3", [rows, scale_cols], scale_nbytes),
        (dst_gs, "F32", [1], 4),
    ]
    write_streaming_safetensors(shard_path, specs, write_payload)
    if writer is not None:
        writer.register_external_shard(
            shard_name, {dst_weight: weight_nbytes, dst_scale: scale_nbytes, dst_gs: 4}
        )
    return weight_nbytes, scale_nbytes, global_scale


@torch.no_grad()
def repack_engram_nvfp4(
    source_input: Path | None,
    quant_dir: Path,
    *,
    device: torch.device,
    input_hf: str | None = None,
    hf_revision: str = "main",
    hf_token: str | None = None,
    hf_workers: int = 20,
    temp_dir: Path | None = None,
) -> dict:
    """Replace the 4 Engram shards in an existing quantized checkpoint with NVFP4_E4M3_GLOBAL."""
    t0 = time.monotonic()
    index_path = quant_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    cfg = json.loads((quant_dir / "config.json").read_text())
    text_cfg = cfg.get("text_config", cfg)
    engram_layers = [int(i) - 1 for i in (text_cfg.get("engram_layer_ids") or [])]
    if not engram_layers:
        raise ValueError(f"No engram_layer_ids found in {quant_dir / 'config.json'}")

    reader: ShardedSafetensorsReader | HfRangeSafetensorsReader = (
        HfRangeSafetensorsReader(input_hf, revision=hf_revision, token=hf_token, workers=hf_workers)
        if input_hf is not None
        else ShardedSafetensorsReader(source_input)  # type: ignore[arg-type]
    )
    wm: dict[str, str] = dict(index["weight_map"])
    stage_dir = Path(tempfile.mkdtemp(prefix="hy4_engram_nvfp4_")) if temp_dir is None else temp_dir
    stage_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{os.getpid()}_{int(time.time())}"
    created_shards: list[Path] = []
    superseded_shards: set[Path] = set()
    published = False
    try:
        for layer_idx in engram_layers:
            lt0 = time.monotonic()
            src_key = f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.embedding.weight"
            dst_w = f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.weight"
            dst_s = f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.scale"
            dst_gs = f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.weight_scale_2"

            old_shard = wm.get(dst_w)
            if old_shard is None:
                raise KeyError(f"{dst_w} missing from {index_path}")
            old_path = quant_dir / old_shard
            # Verify the shard is dedicated to this Engram table before retiring it after atomic publish.
            co_keys = [k for k, s in wm.items() if s == old_shard and k not in (dst_w, dst_s, dst_gs)]
            if co_keys:
                raise ValueError(f"Engram shard {old_shard} contains non-Engram tensors: {co_keys[:3]}")

            local_tmp = stage_dir / f"nvfp4_layer{layer_idx}.safetensors"
            wb, sb, gs = stream_quantize_engram_table_nvfp4(
                reader, None, layer_idx, device, shard_path_override=local_tmp
            )
            if isinstance(reader, HfRangeSafetensorsReader):
                reader.release_temp_shard(src_key)

            # Write to a fresh shard filename so existing shards remain intact if repacking aborts mid-run.
            new_shard = f"engram-nvfp4-layer{layer_idx:02d}-{run_tag}.safetensors"
            new_path = quant_dir / new_shard
            dst_part = quant_dir / f"{new_shard}.part"
            shutil.copyfile(local_tmp, dst_part)
            os.replace(dst_part, new_path)
            local_tmp.unlink()
            created_shards.append(new_path)

            hdr, written_bytes = _safetensors_header_bytes(new_path)
            expected_keys = {dst_w, dst_s, dst_gs}
            if not expected_keys.issubset(hdr.keys()) or written_bytes != wb + sb + 4:
                raise RuntimeError(
                    f"Verified header mismatch in {new_path}: expected {wb + sb + 4} bytes, got {written_bytes}"
                )

            wm[dst_w] = new_shard
            wm[dst_s] = new_shard
            wm[dst_gs] = new_shard
            if old_path != new_path:
                superseded_shards.add(old_path)
            print(
                f"[engram-nvfp4] layer {layer_idx:02d}: {(wb + sb + 4) / (1 << 30):.2f} GiB "
                f"(global_scale={gs:.6g}) in {time.monotonic() - lt0:.1f}s",
                flush=True,
            )

        # Recompute total_size from every referenced shard header so retries never drift.
        total_size = 0
        for shard_name in sorted(set(wm.values())):
            shard_path = quant_dir / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"Referenced shard missing before index publish: {shard_path}")
            _, shard_bytes = _safetensors_header_bytes(shard_path)
            total_size += shard_bytes

        new_index = {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(wm.items()))}
        index_part = quant_dir / "model.safetensors.index.json.part"
        index_part.write_text(json.dumps(new_index, indent=2) + "\n")
        os.replace(index_part, index_path)
        published = True
    finally:
        reader.close()
        if not published:
            for orphan in created_shards:
                if orphan.exists():
                    orphan.unlink()
        if temp_dir is None and stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)

    for old_path in sorted(superseded_shards):
        if old_path.exists():
            old_path.unlink()
    print(
        f"[engram-nvfp4] Done in {time.monotonic() - t0:.1f}s; checkpoint total_size = {total_size / (1 << 30):.2f} GiB",
        flush=True,
    )
    return new_index


# ---------------------------------------------------------------------------
# Tensor classification
# ---------------------------------------------------------------------------


def _is_mxfp8_linear(name: str, *, dense_policy: str = "selective") -> bool:
    if not name.endswith(".weight"):
        return False
    prefix = name[:-7]
    if not prefix.startswith("model."):
        # Preserve SigLIP2 vision encoder (vit.*) at checkpoint BF16 precision.
        return False
    if dense_policy == "selective":
        # Quantize only the three largest backbone attention projections (linear_gate,
        # o_proj, q_b_proj) to MXFP8; retain smaller critical-path dense linears,
        # kv_b_proj (single 64-head SWA BMM), layer-0 dense FFN, Engram dense projs,
        # and all 5 DFly2 speculative draft layers in exact BF16.
        if prefix.startswith("model.layers.") and any(
            prefix.endswith(f".self_attn.{proj}")
            for proj in ("q_b_proj", "linear_gate", "o_proj")
        ):
            return True
        return False
    if dense_policy != "all":
        raise ValueError(f"Unsupported dense_policy {dense_policy!r}; expected 'selective' or 'all'")
    # Full 'all' policy: all backbone + DFly2 dense linears in MXFP8.
    if any(
        prefix.endswith(f".self_attn.{proj}")
        for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "linear_gate", "o_proj")
    ):
        return True
    if (prefix.startswith("model.layers.0.mlp.") or prefix.startswith("model.dfly_layers.")) and any(
        prefix.endswith(f".{proj}") for proj in ("gate_proj", "up_proj", "down_proj")
    ):
        return True
    if any(
        prefix.endswith(suffix)
        for suffix in (
            ".mlp.routed_expert_down_proj",
            ".mlp.routed_expert_up_proj",
            ".mlp.shared_experts.gate_proj",
            ".mlp.shared_experts.up_proj",
            ".mlp.shared_experts.down_proj",
        )
    ):
        return True
    if ".engram_layer." in prefix and (
        prefix.endswith(".value_proj") or ".key_projs." in prefix
    ):
        return True
    if prefix in (
        "model.dfly_layers.mtp_fusion.eh_proj",
        "model.dfly_layers.hidden_correction.gate_proj",
        "model.dfly_layers.hidden_correction.up_proj",
        "model.dfly_layers.hidden_correction.down_proj",
    ):
        return True
    return False


def _reuse_engram_table(
    reuse_reader: ShardedSafetensorsReader,
    writer: ShardedSafetensorsWriter,
    layer_idx: int,
    *,
    expected_shape: tuple[int, ...] | None = None,
) -> tuple[int, int]:
    """Hardlink (or copy) a previously quantized FP8_E4M3_E8M0 Engram shard into output_dir."""
    dst_weight = f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.weight"
    dst_scale = f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.scale"
    w_path, w_dtype, w_shape, _, w_nbytes = reuse_reader.raw_tensor_info(dst_weight)
    s_path, s_dtype, s_shape, _, s_nbytes = reuse_reader.raw_tensor_info(dst_scale)
    if w_path != s_path:
        raise ValueError(f"Expected {dst_weight} and {dst_scale} to share one dedicated shard")
    with open(w_path, "rb") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        shard_header = json.loads(fh.read(hlen))
    shard_tensors = {k for k in shard_header.keys() if k != "__metadata__"}
    expected_tensors = {dst_weight, dst_scale}
    if shard_tensors != expected_tensors:
        raise ValueError(
            f"Reused Engram shard {w_path.name} must contain only {sorted(expected_tensors)}, "
            f"got {sorted(shard_tensors)}"
        )
    if w_dtype != "F8_E4M3" or s_dtype != "F8_E8M0" or w_shape[0] != s_shape[0] or w_shape[1] != s_shape[1] * 32:
        raise ValueError(
            f"Incompatible reused Engram table for layer {layer_idx}: "
            f"weight=({w_dtype}, {w_shape}), scale=({s_dtype}, {s_shape})"
        )
    if expected_shape is not None and tuple(w_shape) != tuple(expected_shape):
        raise ValueError(
            f"Reused Engram table shape mismatch for layer {layer_idx}: "
            f"reused {tuple(w_shape)} != source {tuple(expected_shape)}"
        )
    idx = len(writer.shard_files) + 1
    shard_name = f"shard-{idx:05d}.safetensors"
    dst_path = writer.output_dir / shard_name
    if dst_path.exists():
        dst_path.unlink()
    try:
        os.link(w_path, dst_path)
    except OSError:
        shutil.copy2(w_path, dst_path)
    writer.register_external_shard(shard_name, {dst_weight: w_nbytes, dst_scale: s_nbytes})
    return w_nbytes, s_nbytes


def _load_quantized_engram_rows(
    *,
    layer_idx: int,
    row_ids: torch.Tensor,
    bf16_rows: torch.Tensor,
    engram_format: str,
    global_scale: float | None,
    reuse_reader: ShardedSafetensorsReader | None,
    device: torch.device,
) -> torch.Tensor:
    """Return dequantized Engram rows matching the exact table shipped in the output checkpoint."""
    if reuse_reader is not None:
        if engram_format != "fp8":
            raise ValueError(
                f"--reuse-engram-from only supports --engram-format fp8 (got --engram-format {engram_format!r})"
            )
        dst_weight = f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.weight"
        dst_scale = f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.scale"
        return _gather_engram_rows_from_file(
            reuse_reader,
            dst_weight,
            row_ids,
            scale_name=dst_scale,
        ).to(device=device)
    return _round_trip_engram_rows(
        bf16_rows.to(device=device),
        engram_format=engram_format,
        global_scale=global_scale,
    )


# ---------------------------------------------------------------------------
# Main streaming quantization driver
# ---------------------------------------------------------------------------


@torch.no_grad()
def quantize_checkpoint(
    input_dir: Path | None,
    output_dir: Path,
    *,
    device: torch.device,
    input_hf: str | None = None,
    hf_revision: str = "main",
    hf_token: str | None = None,
    hf_workers: int = 12,
    num_calib_samples: int = 128,
    calib_seq_len: int = 1024,
    micro_batch_size: int = 4,
    calib_file: Path | None = None,
    mse_factors: tuple[float, ...] = DEFAULT_MSE_FACTORS,
    propagate_quantized: bool = True,
    skip_engram_tables: bool = False,
    engram_format: str = "fp8",
    reuse_engram_from: Path | None = None,
    dense_policy: str = "selective",
    use_gptq: bool = True,
    gptq_down_proj: bool = True,
    gptq_batch_size: int = 64,
    shard_bytes: int = 4 * (1 << 30),
    sync_dir: Path | None = None,
    move_to_sync_dir: bool = False,
    inline_verify: bool = False,
    verify_samples: int = 4,
    verify_seq_len: int = 256,
    verify_file: Path | None = None,
    verify_seed: int = 999,
    seed: int = 42,
) -> dict:
    """Run full streaming calibration and quantization of a BF16 Hy4-flash checkpoint."""
    if reuse_engram_from is not None and engram_format != "fp8":
        raise ValueError(
            f"--reuse-engram-from only supports --engram-format fp8 (got --engram-format {engram_format!r})"
        )
    t0 = time.monotonic()
    if input_hf is not None:
        reader: ShardedSafetensorsReader | HfRangeSafetensorsReader = HfRangeSafetensorsReader(
            input_hf, revision=hf_revision, token=hf_token, workers=hf_workers
        )
        meta_root = reader.root
    elif input_dir is not None:
        reader = ShardedSafetensorsReader(input_dir)
        meta_root = input_dir
    else:
        raise ValueError("Either input_dir or input_hf must be provided")

    config = json.loads((meta_root / "config.json").read_text())
    if config.get("model_type") != "hy_v4_flash":
        raise ValueError(f"Expected model_type='hy_v4_flash', got {config.get('model_type')!r}")
    text_cfg = config.get("text_config", config)
    num_layers = int(text_cfg["num_hidden_layers"])
    hc_mult = int(text_cfg.get("hc_mult", 4))
    rms_eps = float(text_cfg.get("rms_norm_eps", 1e-5))
    hc_eps = float(text_cfg.get("hc_eps", 1e-6))
    engram_pad_id = int(text_cfg.get("engram_pad_id", 120002))
    engram_max_ngram = int(text_cfg.get("engram_max_ngram_size", 3))
    engram_n_head = int(text_cfg.get("engram_n_head_per_ngram", 8))
    engram_seed = int(text_cfg.get("engram_seed", 0))
    image_token_id = int(config.get("image_token_id", 120120))
    video_token_id = int(config.get("video_token_id", 120124))

    reuse_reader = ShardedSafetensorsReader(reuse_engram_from) if reuse_engram_from is not None else None
    writer = ShardedSafetensorsWriter(
        output_dir,
        max_shard_bytes=shard_bytes,
        sync_dir=sync_dir,
        move_to_sync_dir=move_to_sync_dir,
    )
    quantized_layers_cfg: dict[str, dict] = {}

    # Copy non-weight metadata files (tokenizer, template, processor, etc.).
    write_local_meta = not (move_to_sync_dir and sync_dir is not None)
    for item in sorted(meta_root.iterdir()):
        if item.is_file() and not item.name.endswith(".safetensors") and item.name not in (
            "model.safetensors.index.json",
            "config.json",
        ):
            if write_local_meta:
                shutil.copy2(item, output_dir / item.name)
            if sync_dir is not None:
                shutil.copy2(item, sync_dir / item.name)

    # 1. Build calibration token IDs and embed them into 4-stream CPU BF16 activations.
    print(
        f"[calib] Preparing {num_calib_samples} x {calib_seq_len} calibration tokens "
        f"(dense_policy={dense_policy}, gptq={use_gptq}, gptq_down={gptq_down_proj}, inline_verify={inline_verify})..."
    )
    calib_ids = build_calibration_tokens(
        meta_root, num_calib_samples, calib_seq_len, calib_file=calib_file, seed=seed
    )
    ngram_tokens = _build_ngram_tokens(calib_ids, engram_max_ngram, engram_pad_id)
    engram_mask = (
        (calib_ids != image_token_id)
        & (calib_ids != video_token_id)
        & (calib_ids != engram_pad_id)
    )

    embed_w = reader.get_tensor("model.embed_tokens.weight", device="cpu")
    writer.add("model.embed_tokens.weight", embed_w)
    # hidden_streams lives on CPU RAM: [N, S, 4, 3072] BF16.
    hidden_streams_cpu = F.embedding(calib_ids, embed_w).unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous()

    eval_ids = eval_ngram_tokens = eval_engram_mask = None
    eval_bf16_streams = eval_quant_streams = None
    if inline_verify:
        eval_ids = build_calibration_tokens(
            meta_root, verify_samples, verify_seq_len, calib_file=verify_file, seed=verify_seed
        )
        eval_ngram_tokens = _build_ngram_tokens(eval_ids, engram_max_ngram, engram_pad_id)
        eval_engram_mask = (
            (eval_ids != image_token_id)
            & (eval_ids != video_token_id)
            & (eval_ids != engram_pad_id)
        )
        eval_bf16_streams = (
            F.embedding(eval_ids, embed_w).unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous().to(device)
        )
        eval_quant_streams = eval_bf16_streams.clone()

    del embed_w
    gc.collect()

    # Partition tensor names by layer vs non-layer.
    layer_tensors: dict[int, list[str]] = defaultdict(list)
    non_layer_tensors: list[str] = []
    for name in sorted(reader.keys()):
        if name == "model.embed_tokens.weight":
            continue
        if name.startswith("model.layers."):
            layer_idx = int(name.split(".")[2])
            layer_tensors[layer_idx].append(name)
        else:
            non_layer_tensors.append(name)

    is_hf = isinstance(reader, HfRangeSafetensorsReader)
    prefetch_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1) if is_hf else None

    def _layer_non_engram_names(idx: int) -> list[str]:
        ekey = f"model.layers.{idx}.engram_layer.multi_head_embedding.embedding.weight"
        return [n for n in layer_tensors[idx] if n != ekey]

    next_layer_fut = (
        prefetch_pool.submit(reader.fetch_tensors, _layer_non_engram_names(0))
        if prefetch_pool is not None and num_layers > 0
        else None
    )

    # 2. Stream layers 0..num_layers-1: calibrate -> quantize -> write -> drop.
    for layer_idx in range(num_layers):
        lt0 = time.monotonic()
        info = _layer_config_info(text_cfg, layer_idx)
        names = layer_tensors[layer_idx]
        engram_table_key = f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.embedding.weight"

        # Load non-Engram-table weights for this layer to GPU (overlapping Layer L+1 HF download).
        weights: dict[str, torch.Tensor] = {}
        if prefetch_pool is not None and next_layer_fut is not None:
            cpu_weights = next_layer_fut.result()
            if layer_idx + 1 < num_layers:
                next_layer_fut = prefetch_pool.submit(
                    reader.fetch_tensors, _layer_non_engram_names(layer_idx + 1)
                )
            else:
                next_layer_fut = None
            for name, t_cpu in cpu_weights.items():
                weights[name] = t_cpu.to(device, non_blocking=True)
            del cpu_weights
        else:
            for name in names:
                if name == engram_table_key:
                    continue
                weights[name] = reader.get_tensor(name, device=device)

        # If this layer has an Engram table, pre-gather only the unique rows hit by calib_ids (and eval_ids).
        engram_calib_gathered = None
        engram_eval_bf16 = None
        engram_calib_row_ids = None
        engram_eval_row_ids = None
        src_engram_shape: tuple[int, ...] | None = None
        if info["has_engram"] and reader.has(engram_table_key):
            layer_id_1based = layer_idx + 1
            src_engram_shape = tuple(reader.raw_tensor_info(engram_table_key)[2])
            lt_tensor = weights[f"model.layers.{layer_idx}.engram_layer.lookup_table"].cpu()
            comp_vocab = int(lt_tensor.max().item()) + 1
            multipliers = _draw_engram_multipliers(layer_id_1based, comp_vocab, engram_max_ngram, engram_seed)
            head_vocab_sizes = [
                sz for order in text_cfg["engram_layer_head_vocab_sizes"][str(layer_id_1based)] for sz in order
            ]
            engram_calib_row_ids = _hash_engram_rows(
                ngram_tokens, lt_tensor, multipliers, head_vocab_sizes, engram_pad_id, engram_max_ngram, engram_n_head
            )
            engram_calib_gathered = _gather_engram_rows_from_file(reader, engram_table_key, engram_calib_row_ids)
            if inline_verify and eval_ngram_tokens is not None:
                engram_eval_row_ids = _hash_engram_rows(
                    eval_ngram_tokens, lt_tensor, multipliers, head_vocab_sizes, engram_pad_id, engram_max_ngram, engram_n_head
                )
                engram_eval_bf16 = _gather_engram_rows_from_file(reader, engram_table_key, engram_eval_row_ids)

        if inline_verify and eval_bf16_streams is not None:
            eval_bf16_streams = _forward_decoder_layer(
                eval_bf16_streams,
                layer_idx,
                text_cfg,
                weights,
                engram_gathered=engram_eval_bf16,
                engram_mask=eval_engram_mask,
                dequantize_mxfp8_linears=False,
            )

        stats = None
        if info["is_sparse_mlp"]:
            stats = ExpertCalibStats.create(
                int(text_cfg["n_routed_experts"]),
                int(text_cfg["routed_expert_hidden_size"]),
                int(text_cfg["moe_intermediate_size"]),
                device,
                use_gptq=use_gptq,
            )

        # Pass A: Run BF16 layer forward over calibration micro-batches to gather stats.
        next_streams_cpu = torch.empty_like(hidden_streams_cpu) if not propagate_quantized else None
        for b0 in range(0, num_calib_samples, micro_batch_size):
            b1 = min(num_calib_samples, b0 + micro_batch_size)
            batch_in = hidden_streams_cpu[b0:b1].to(device, non_blocking=True)
            eg = None if engram_calib_gathered is None else engram_calib_gathered[b0:b1]
            em = None if engram_calib_gathered is None else engram_mask[b0:b1]
            batch_out = _forward_decoder_layer(
                batch_in,
                layer_idx,
                text_cfg,
                weights,
                engram_gathered=eg,
                engram_mask=em,
                stats=stats,
            )
            if next_streams_cpu is not None:
                next_streams_cpu[b0:b1].copy_(batch_out.cpu())
            del batch_in, batch_out

        # Quantize this layer's weights.
        quantized_experts_gpu = None
        need_deq_experts = propagate_quantized or inline_verify
        if info["is_sparse_mlp"]:
            assert stats is not None
            mp = f"model.layers.{layer_idx}.mlp"
            w_gu = weights.pop(f"{mp}.experts.gate_up_proj")  # [E, 2*I, H_lat]
            w_dn = weights.pop(f"{mp}.experts.down_proj")  # [E, H_lat, I]
            num_experts = w_gu.shape[0]
            inter_dim = w_gu.shape[1] // 2
            lat_dim = w_gu.shape[2]

            gate_w = torch.empty((num_experts, inter_dim, lat_dim // 2), dtype=torch.uint8, device=device)
            gate_s = torch.empty((num_experts, inter_dim, lat_dim // 16), dtype=torch.float8_e4m3fn, device=device)
            up_w = torch.empty_like(gate_w)
            up_s = torch.empty_like(gate_s)
            down_w = torch.empty((num_experts, lat_dim, inter_dim // 2), dtype=torch.uint8, device=device)
            down_s = torch.empty((num_experts, lat_dim, inter_dim // 16), dtype=torch.float8_e4m3fn, device=device)

            w13_s2 = torch.empty((num_experts,), dtype=torch.float32, device=device)
            w2_s2 = torch.empty((num_experts,), dtype=torch.float32, device=device)
            a1_scale = torch.empty((num_experts,), dtype=torch.float32, device=device)
            a2_scale = torch.empty((num_experts,), dtype=torch.float32, device=device)

            fallback_x1 = max(stats.global_x1_amax, 1e-4)
            fallback_x2 = max(stats.global_x2_amax, 1e-4)
            counts_list = stats.counts.tolist()
            zero_hit_experts = 0

            # Leave 1.5x headroom in weight_scale_2 (divide by 448*4 instead of 448*6)
            # so Four-Over-Six (1.5x scale) is never clamped on the largest-magnitude blocks.
            for e in range(num_experts):
                wg = w_gu[e, :inter_dim]
                wu = w_gu[e, inter_dim:]
                wd = w_dn[e]
                amax_13 = max(float(wg.abs().amax().item()), float(wu.abs().amax().item()), 1e-6)
                amax_2 = max(float(wd.abs().amax().item()), 1e-6)
                w13_s2[e] = amax_13 / (FP8_E4M3_MAX * 4.0)
                w2_s2[e] = amax_2 / (FP8_E4M3_MAX * 4.0)

                c = counts_list[e]
                if c > 0:
                    x1_am = max(float(stats.x1_amax[e].item()), 1e-4)
                    x2_am = max(float(stats.x2_amax[e].item()), 1e-4)
                else:
                    zero_hit_experts += 1
                    x1_am = fallback_x1
                    x2_am = fallback_x2
                a1_scale[e] = x1_am / (FP8_E4M3_MAX * FP4_E2M1_MAX)
                a2_scale[e] = x2_am / (FP8_E4M3_MAX * FP4_E2M1_MAX)

            if use_gptq and stats.H1 is not None and stats.H2 is not None:
                for e0 in range(0, num_experts, gptq_batch_size):
                    e1 = min(num_experts, e0 + gptq_batch_size)
                    cnt_sl = stats.counts[e0:e1]
                    gate_w[e0:e1], gate_s[e0:e1] = quantize_nvfp4_gptq_batched(
                        w_gu[e0:e1, :inter_dim], w13_s2[e0:e1], stats.H1[e0:e1], cnt_sl,
                        mse_factors=mse_factors, base_damp=0.01,
                    )
                    up_w[e0:e1], up_s[e0:e1] = quantize_nvfp4_gptq_batched(
                        w_gu[e0:e1, inter_dim:], w13_s2[e0:e1], stats.H1[e0:e1], cnt_sl,
                        mse_factors=mse_factors, base_damp=0.01,
                    )
                    if gptq_down_proj:
                        down_w[e0:e1], down_s[e0:e1] = quantize_nvfp4_gptq_batched(
                            w_dn[e0:e1], w2_s2[e0:e1], stats.H2[e0:e1], cnt_sl,
                            mse_factors=mse_factors, base_damp=0.02,
                        )
                    else:
                        for e in range(e0, e1):
                            c = counts_list[e]
                            h2 = (stats.x2_sq_sum[e] / c).float() if c > 0 else None
                            down_w[e], down_s[e] = quantize_nvfp4_mse_block16(w_dn[e], w2_s2[e], h2, mse_factors)
            else:
                for e in range(num_experts):
                    c = counts_list[e]
                    h1 = (stats.x1_sq_sum[e] / c).float() if c > 0 else None
                    h2 = (stats.x2_sq_sum[e] / c).float() if c > 0 else None
                    gate_w[e], gate_s[e] = quantize_nvfp4_mse_block16(w_gu[e, :inter_dim], w13_s2[e], h1, mse_factors)
                    up_w[e], up_s[e] = quantize_nvfp4_mse_block16(w_gu[e, inter_dim:], w13_s2[e], h1, mse_factors)
                    down_w[e], down_s[e] = quantize_nvfp4_mse_block16(w_dn[e], w2_s2[e], h2, mse_factors)

            del w_gu, w_dn
            if stats.H1 is not None:
                stats.H1 = None
                stats.H2 = None
            for proj_name, pw, ps, ps2, pis in (
                ("gate_proj", gate_w, gate_s, w13_s2, a1_scale),
                ("up_proj", up_w, up_s, w13_s2, a1_scale),
                ("down_proj", down_w, down_s, w2_s2, a2_scale),
            ):
                eprefix = f"{mp}.experts.{proj_name}"
                writer.add(f"{eprefix}.weight", pw)
                writer.add(f"{eprefix}.weight_scale", ps)
                writer.add(f"{eprefix}.weight_scale_2", ps2)
                writer.add(f"{eprefix}.input_scale", pis)
                quantized_layers_cfg[eprefix] = {"quant_algo": "NVFP4", "group_size": 16}

            if need_deq_experts:
                # Pre-dequantize this layer's NVFP4 experts once for Pass B / inline verify.
                deq_gate = torch.empty((num_experts, inter_dim, lat_dim), dtype=torch.bfloat16, device=device)
                deq_up = torch.empty_like(deq_gate)
                deq_down = torch.empty((num_experts, lat_dim, inter_dim), dtype=torch.bfloat16, device=device)
                for e in range(num_experts):
                    deq_gate[e].copy_(dequantize_nvfp4(gate_w[e], gate_s[e], w13_s2[e]).to(torch.bfloat16))
                    deq_up[e].copy_(dequantize_nvfp4(up_w[e], up_s[e], w13_s2[e]).to(torch.bfloat16))
                    deq_down[e].copy_(dequantize_nvfp4(down_w[e], down_s[e], w2_s2[e]).to(torch.bfloat16))
                del gate_w, gate_s, up_w, up_s, down_w, down_s, w13_s2, w2_s2, a1_scale, a2_scale
                quantized_experts_gpu = dict(deq_gate=deq_gate, deq_up=deq_up, deq_down=deq_down)
            else:
                del gate_w, gate_s, up_w, up_s, down_w, down_s, w13_s2, w2_s2, a1_scale, a2_scale
        else:
            zero_hit_experts = 0

        # Quantize selected MXFP8 linears and preserve remaining tensors verbatim in BF16.
        for name in list(weights.keys()):
            tensor = weights[name]
            if _is_mxfp8_linear(name, dense_policy=dense_policy):
                prefix = name[:-7]
                q, s = quantize_mxfp8_block32(tensor)
                writer.add(f"{prefix}.weight", q)
                writer.add(f"{prefix}.weight_scale", s)
                quantized_layers_cfg[prefix] = {"quant_algo": "MXFP8", "group_size": 32}
                if propagate_quantized or inline_verify:
                    weights[f"{prefix}.weight"] = q
                    weights[f"{prefix}.weight_scale"] = s
                else:
                    del q, s
            else:
                writer.add(name, tensor)

        # Stream-convert or reuse this layer's Engram table before inline verification and
        # Pass B error propagation so both use the exact table shipped in the output checkpoint.
        engram_msg = ""
        engram_global_scale: float | None = None
        active_reuse_reader = None
        if info["has_engram"] and reader.has(engram_table_key):
            if not skip_engram_tables:
                writer.flush()
                if reuse_reader is not None:
                    wb, sb = _reuse_engram_table(
                        reuse_reader, writer, layer_idx, expected_shape=src_engram_shape
                    )
                    active_reuse_reader = reuse_reader
                    engram_msg = f" + engram={(wb + sb) / (1 << 30):.2f} GiB (reused)"
                elif engram_format == "nvfp4":
                    wb, sb, engram_global_scale = stream_quantize_engram_table_nvfp4(reader, writer, layer_idx, device)
                    engram_msg = f" + engram_nvfp4={(wb + sb + 4) / (1 << 30):.2f} GiB"
                else:
                    wb, sb = stream_quantize_engram_table(reader, writer, layer_idx, device)
                    engram_msg = f" + engram_fp8={(wb + sb) / (1 << 30):.2f} GiB"
            elif engram_format == "nvfp4":
                engram_global_scale = _compute_engram_nvfp4_global_scale(reader, layer_idx, device)
            if isinstance(reader, HfRangeSafetensorsReader):
                reader.release_temp_shard(engram_table_key)

        if inline_verify and eval_quant_streams is not None:
            eg_eval_q = None
            if engram_eval_bf16 is not None and engram_eval_row_ids is not None:
                eg_eval_q = _load_quantized_engram_rows(
                    layer_idx=layer_idx,
                    row_ids=engram_eval_row_ids,
                    bf16_rows=engram_eval_bf16,
                    engram_format=engram_format,
                    global_scale=engram_global_scale,
                    reuse_reader=active_reuse_reader,
                    device=device,
                )
            eval_quant_streams = _forward_decoder_layer(
                eval_quant_streams,
                layer_idx,
                text_cfg,
                weights,
                engram_gathered=eg_eval_q,
                engram_mask=eval_engram_mask,
                quantized_experts=quantized_experts_gpu,
                dequantize_mxfp8_linears=True,
            )
            del eg_eval_q

        # Pass B (sequential error-propagated calibration): recompute layer output using
        # quantized weights and exact shipped Engram activations so layer i+1
        # calibrates on realistic quantized activations.
        if propagate_quantized:
            engram_calib_reused_cpu = (
                _load_quantized_engram_rows(
                    layer_idx=layer_idx,
                    row_ids=engram_calib_row_ids,
                    bf16_rows=engram_calib_gathered,
                    engram_format=engram_format,
                    global_scale=None,
                    reuse_reader=active_reuse_reader,
                    device=torch.device("cpu"),
                )
                if active_reuse_reader is not None
                and engram_calib_gathered is not None
                and engram_calib_row_ids is not None
                else None
            )
            next_streams_cpu = torch.empty_like(hidden_streams_cpu)
            for b0 in range(0, num_calib_samples, micro_batch_size):
                b1 = min(num_calib_samples, b0 + micro_batch_size)
                batch_in = hidden_streams_cpu[b0:b1].to(device, non_blocking=True)
                if engram_calib_gathered is None or engram_calib_row_ids is None:
                    eg = None
                elif engram_calib_reused_cpu is not None:
                    eg = engram_calib_reused_cpu[b0:b1].to(device, non_blocking=True)
                else:
                    eg = _load_quantized_engram_rows(
                        layer_idx=layer_idx,
                        row_ids=engram_calib_row_ids[b0:b1],
                        bf16_rows=engram_calib_gathered[b0:b1],
                        engram_format=engram_format,
                        global_scale=engram_global_scale,
                        reuse_reader=None,
                        device=device,
                    )
                em = None if engram_calib_gathered is None else engram_mask[b0:b1]
                batch_out = _forward_decoder_layer(
                    batch_in,
                    layer_idx,
                    text_cfg,
                    weights,
                    engram_gathered=eg,
                    engram_mask=em,
                    stats=None,
                    quantized_experts=quantized_experts_gpu,
                    dequantize_mxfp8_linears=True,
                )
                next_streams_cpu[b0:b1].copy_(batch_out.cpu())
                del batch_in, batch_out, eg
            del engram_calib_reused_cpu
        if quantized_experts_gpu is not None:
            del quantized_experts_gpu

        assert next_streams_cpu is not None
        hidden_streams_cpu = next_streams_cpu
        del weights, engram_calib_gathered, engram_eval_bf16, stats
        torch.cuda.empty_cache()

        elapsed = time.monotonic() - lt0
        print(
            f"[layer {layer_idx:02d}/{num_layers - 1}] "
            f"{'indexed' if info['is_indexed'] else 'swa    '} "
            f"{'moe' if info['is_sparse_mlp'] else 'dns'} "
            f"(zero_hit_experts={zero_hit_experts}) {elapsed:.1f}s{engram_msg}",
            flush=True,
        )

    if prefetch_pool is not None:
        prefetch_pool.shutdown(wait=True)

    del hidden_streams_cpu, calib_ids, ngram_tokens, engram_mask
    gc.collect()

    # 3. Process non-layer tensors (DFly2 stays BF16 under 'selective' policy; lm_head/norm/hc_head/ViT verbatim).
    print(f"[tail] Quantizing/copying {len(non_layer_tensors)} non-layer tensors (DFly2, lm_head, ViT)...", flush=True)
    tail_needed_for_verify = {
        "model.hc_head.hc_head_fn",
        "model.hc_head.hc_head_base",
        "model.hc_head.hc_head_scale",
        "model.norm.weight",
        "lm_head.weight",
    }
    verify_head_weights: dict[str, torch.Tensor] = {}

    if isinstance(reader, HfRangeSafetensorsReader):
        # Fetch non-layer tensors in bounded groups (~32 tensors at a time) so CPU RAM stays bounded.
        for g0 in range(0, len(non_layer_tensors), 32):
            batch_names = non_layer_tensors[g0 : g0 + 32]
            fetched = reader.fetch_tensors(batch_names)
            for name in batch_names:
                tensor_cpu = fetched.pop(name)
                if inline_verify and name in tail_needed_for_verify:
                    verify_head_weights[name] = tensor_cpu.to(device)
                if _is_mxfp8_linear(name, dense_policy=dense_policy):
                    prefix = name[:-7]
                    q, s = quantize_mxfp8_block32(tensor_cpu.to(device))
                    writer.add(f"{prefix}.weight", q)
                    writer.add(f"{prefix}.weight_scale", s)
                    quantized_layers_cfg[prefix] = {"quant_algo": "MXFP8", "group_size": 32}
                    del q, s
                else:
                    writer.add(name, tensor_cpu)
                del tensor_cpu
    else:
        for name in non_layer_tensors:
            if _is_mxfp8_linear(name, dense_policy=dense_policy):
                prefix = name[:-7]
                tensor = reader.get_tensor(name, device=device)
                q, s = quantize_mxfp8_block32(tensor)
                writer.add(f"{prefix}.weight", q)
                writer.add(f"{prefix}.weight_scale", s)
                quantized_layers_cfg[prefix] = {"quant_algo": "MXFP8", "group_size": 32}
                del tensor, q, s
            else:
                tensor = reader.get_tensor(name, device="cpu")
                if inline_verify and name in tail_needed_for_verify:
                    verify_head_weights[name] = tensor.to(device)
                writer.add(name, tensor)
                del tensor
    torch.cuda.empty_cache()

    index = writer.finalize()
    reader.close()
    if reuse_reader is not None:
        reuse_reader.close()

    # 4. Write ModelOpt MIXED_PRECISION config.json.
    out_config = dict(config)
    out_config["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "kv_cache_quant_algo": None,
        "quantized_layers": dict(sorted(quantized_layers_cfg.items())),
    }
    config_text = json.dumps(out_config, indent=2) + "\n"
    if write_local_meta:
        (output_dir / "config.json").write_text(config_text)
    if sync_dir is not None:
        (sync_dir / "config.json").write_text(config_text)

    if inline_verify and eval_bf16_streams is not None and eval_quant_streams is not None and eval_ids is not None:
        fn = verify_head_weights["model.hc_head.hc_head_fn"]
        base = verify_head_weights["model.hc_head.hc_head_base"]
        scale = verify_head_weights["model.hc_head.hc_head_scale"]
        norm_w = verify_head_weights["model.norm.weight"]
        lm_w = verify_head_weights["lm_head.weight"].float()
        ref_h = _rms_norm(_ihc_head(eval_bf16_streams, fn, base, scale, rms_eps=rms_eps, hc_eps=hc_eps), norm_w, rms_eps)
        q_h = _rms_norm(_ihc_head(eval_quant_streams, fn, base, scale, rms_eps=rms_eps, hc_eps=hc_eps), norm_w, rms_eps)
        ref_logits = F.linear(ref_h.float(), lm_w).cpu()
        q_logits = F.linear(q_h.float(), lm_w).cpu()
        ref_lp = F.log_softmax(ref_logits, dim=-1)
        q_lp = F.log_softmax(q_logits, dim=-1)
        kl = float((ref_lp.exp() * (ref_lp - q_lp)).sum(dim=-1).mean().item())
        top1 = float((ref_logits.argmax(dim=-1) == q_logits.argmax(dim=-1)).float().mean().item())
        targets = eval_ids[:, 1:].reshape(-1)
        ref_nll = float(F.cross_entropy(ref_logits[:, :-1].reshape(-1, ref_logits.shape[-1]), targets).item())
        q_nll = float(F.cross_entropy(q_logits[:, :-1].reshape(-1, q_logits.shape[-1]), targets).item())
        metrics = {
            "kl_divergence": kl,
            "top1_agreement": top1,
            "bf16_ppl": math.exp(ref_nll),
            "quant_ppl": math.exp(q_nll),
            "ppl_ratio": math.exp(q_nll - ref_nll),
        }
        print(f"[verify-inline] {json.dumps(metrics, indent=2)}", flush=True)
        metrics_text = json.dumps(metrics, indent=2) + "\n"
        if write_local_meta:
            (output_dir / "verify_metrics.json").write_text(metrics_text)
        if sync_dir is not None:
            (sync_dir / "verify_metrics.json").write_text(metrics_text)
        del verify_head_weights, eval_bf16_streams, eval_quant_streams

    total_time = time.monotonic() - t0
    print(
        f"[done] Wrote {len(index['weight_map'])} tensors "
        f"({index['metadata']['total_size'] / (1 << 30):.2f} GiB across "
        f"{len(set(index['weight_map'].values()))} shards) in {total_time:.1f}s",
        flush=True,
    )
    return index


# ---------------------------------------------------------------------------
# Streaming end-to-end KL / PPL verification against BF16 reference
# ---------------------------------------------------------------------------


@torch.no_grad()
def _run_streaming_logits(
    ckpt_dir: Path,
    token_ids: torch.Tensor,
    device: torch.device,
    *,
    is_quantized: bool,
) -> torch.Tensor:
    """Stream all 42 layers of a BF16 or quantized checkpoint to compute FP32 logits."""
    config = json.loads((ckpt_dir / "config.json").read_text())
    text_cfg = config.get("text_config", config)
    num_layers = int(text_cfg["num_hidden_layers"])
    hc_mult = int(text_cfg.get("hc_mult", 4))
    rms_eps = float(text_cfg.get("rms_norm_eps", 1e-5))
    hc_eps = float(text_cfg.get("hc_eps", 1e-6))
    engram_pad_id = int(text_cfg.get("engram_pad_id", 120002))
    engram_max_ngram = int(text_cfg.get("engram_max_ngram_size", 3))
    engram_n_head = int(text_cfg.get("engram_n_head_per_ngram", 8))
    engram_seed = int(text_cfg.get("engram_seed", 0))
    image_token_id = int(config.get("image_token_id", 120120))
    video_token_id = int(config.get("video_token_id", 120124))

    reader = ShardedSafetensorsReader(ckpt_dir)
    ngram_tokens = _build_ngram_tokens(token_ids, engram_max_ngram, engram_pad_id)
    engram_mask = (
        (token_ids != image_token_id)
        & (token_ids != video_token_id)
        & (token_ids != engram_pad_id)
    )
    embed_w = reader.get_tensor("model.embed_tokens.weight", device="cpu")
    streams = F.embedding(token_ids, embed_w).unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous().to(device)
    del embed_w

    layer_keys: dict[int, list[str]] = defaultdict(list)
    for k in reader.keys():
        if k.startswith("model.layers."):
            layer_keys[int(k.split(".")[2])].append(k)

    for layer_idx in range(num_layers):
        info = _layer_config_info(text_cfg, layer_idx)
        engram_bf16_key = f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.embedding.weight"
        engram_q_w = f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.weight"
        engram_q_s = f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.scale"
        engram_q_gs = f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.weight_scale_2"
        skip = {engram_bf16_key, engram_q_w, engram_q_s, engram_q_gs}

        weights: dict[str, torch.Tensor] = {}
        for k in layer_keys[layer_idx]:
            if k in skip:
                continue
            weights[k] = reader.get_tensor(k, device=device)

        engram_gathered = None
        if info["has_engram"]:
            layer_id_1based = layer_idx + 1
            lt = weights[f"model.layers.{layer_idx}.engram_layer.lookup_table"].cpu()
            comp_vocab = int(lt.max().item()) + 1
            mults = _draw_engram_multipliers(layer_id_1based, comp_vocab, engram_max_ngram, engram_seed)
            hvs = [sz for order in text_cfg["engram_layer_head_vocab_sizes"][str(layer_id_1based)] for sz in order]
            row_ids = _hash_engram_rows(ngram_tokens, lt, mults, hvs, engram_pad_id, engram_max_ngram, engram_n_head)
            if reader.has(engram_q_w) and reader.has(engram_q_s):
                engram_gathered = _gather_engram_rows_from_file(
                    reader,
                    engram_q_w,
                    row_ids,
                    scale_name=engram_q_s,
                    global_scale_name=engram_q_gs if reader.has(engram_q_gs) else None,
                )
            elif reader.has(engram_bf16_key):
                engram_gathered = _gather_engram_rows_from_file(reader, engram_bf16_key, row_ids)

        q_experts = None
        if is_quantized and info["is_sparse_mlp"]:
            mp = f"model.layers.{layer_idx}.mlp.experts"
            q_experts = dict(
                gate_w=weights[f"{mp}.gate_proj.weight"],
                gate_s=weights[f"{mp}.gate_proj.weight_scale"],
                gate_s2=weights[f"{mp}.gate_proj.weight_scale_2"],
                up_w=weights[f"{mp}.up_proj.weight"],
                up_s=weights[f"{mp}.up_proj.weight_scale"],
                up_s2=weights[f"{mp}.up_proj.weight_scale_2"],
                down_w=weights[f"{mp}.down_proj.weight"],
                down_s=weights[f"{mp}.down_proj.weight_scale"],
                down_s2=weights[f"{mp}.down_proj.weight_scale_2"],
            )

        streams = _forward_decoder_layer(
            streams,
            layer_idx,
            text_cfg,
            weights,
            engram_gathered=engram_gathered,
            engram_mask=engram_mask,
            quantized_experts=q_experts,
            dequantize_mxfp8_linears=is_quantized,
        )
        del weights, q_experts, engram_gathered
        torch.cuda.empty_cache()

    collapsed = _ihc_head(
        streams,
        reader.get_tensor("model.hc_head.hc_head_fn", device=device),
        reader.get_tensor("model.hc_head.hc_head_base", device=device),
        reader.get_tensor("model.hc_head.hc_head_scale", device=device),
        rms_eps=rms_eps,
        hc_eps=hc_eps,
    )
    hidden = _rms_norm(collapsed, reader.get_tensor("model.norm.weight", device=device), rms_eps)
    lm_head = reader.get_tensor("lm_head.weight", device=device)
    logits = F.linear(hidden.float(), lm_head.float()).cpu()
    reader.close()
    return logits


@torch.no_grad()
def verify_quality(
    bf16_dir: Path,
    quant_dir: Path,
    *,
    device: torch.device,
    num_samples: int = 4,
    seq_len: int = 512,
    verify_file: Path | None = None,
    seed: int = 999,
) -> dict[str, float]:
    """Measure KL(BF16 || Quant), top-1 agreement, and perplexity delta on held-out text."""
    eval_ids = build_calibration_tokens(bf16_dir, num_samples, seq_len, calib_file=verify_file, seed=seed)
    ref_logits = _run_streaming_logits(bf16_dir, eval_ids, device, is_quantized=False)
    q_logits = _run_streaming_logits(quant_dir, eval_ids, device, is_quantized=True)

    ref_lp = F.log_softmax(ref_logits, dim=-1)
    q_lp = F.log_softmax(q_logits, dim=-1)
    kl = float((ref_lp.exp() * (ref_lp - q_lp)).sum(dim=-1).mean().item())
    top1 = float((ref_logits.argmax(dim=-1) == q_logits.argmax(dim=-1)).float().mean().item())

    targets = eval_ids[:, 1:].reshape(-1)
    ref_nll = float(F.cross_entropy(ref_logits[:, :-1].reshape(-1, ref_logits.shape[-1]), targets).item())
    q_nll = float(F.cross_entropy(q_logits[:, :-1].reshape(-1, q_logits.shape[-1]), targets).item())
    metrics = {
        "kl_divergence": kl,
        "top1_agreement": top1,
        "bf16_ppl": math.exp(ref_nll),
        "quant_ppl": math.exp(q_nll),
        "ppl_ratio": math.exp(q_nll - ref_nll),
    }
    print(f"[verify] {json.dumps(metrics, indent=2)}")
    return metrics


# ---------------------------------------------------------------------------
# Self-test on a synthetic mini Hy4-flash checkpoint (runs on CPU or CUDA)
# ---------------------------------------------------------------------------


def _run_self_test(device: torch.device) -> None:
    """Verify NVFP4 packing, MXFP8 scales, streaming Engram FP8, and ModelOpt schema."""
    import tempfile
    from ignition.preparation.quantization import linear_format, quantized_layers

    print(f"[self-test] Running synthetic Hy4-flash quantization test on {device}...")
    torch.manual_seed(0)

    # 1. Verify NVFP4 MSE quantization beats or matches naive 6-max quantization.
    w = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    h = torch.rand((128,), dtype=torch.float32, device=device) + 0.1
    gs2 = torch.tensor(float(w.abs().amax().item()) / (FP8_E4M3_MAX * FP4_E2M1_MAX), device=device)
    p_mse, s_mse = quantize_nvfp4_mse_block16(w, gs2, h, DEFAULT_MSE_FACTORS)
    p_std, s_std = quantize_nvfp4_mse_block16(w, gs2, h, (1.0,))
    err_mse = float(((w.float() - dequantize_nvfp4(p_mse, s_mse, gs2)).square() * h).mean().item())
    err_std = float(((w.float() - dequantize_nvfp4(p_std, s_std, gs2)).square() * h).mean().item())
    assert err_mse <= err_std + 1e-9, f"MSE search worse than 6-max: {err_mse} > {err_std}"
    print(f"[self-test] NVFP4 activation-weighted MSE error: {err_mse:.6f} vs naive 6-max {err_std:.6f}")

    # 2. Verify exact E2M1 nibble values for all 16 representable codes.
    test_vals = torch.tensor(
        [[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]],
        dtype=torch.float32,
        device=device,
    )
    codes, deq = _quantize_e2m1_rtne(test_vals)
    expected_codes = [0, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15]
    assert codes[0].tolist() == expected_codes, f"unexpected E2M1 codes: {codes[0].tolist()}"
    assert torch.allclose(deq, test_vals), f"E2M1 roundtrip mismatch: {deq}"

    # 3. Verify MXFP8 roundtrip accuracy.
    wm = torch.randn((32, 64), dtype=torch.bfloat16, device=device)
    qm, sm = quantize_mxfp8_block32(wm)
    assert qm.dtype == torch.float8_e4m3fn and sm.dtype == torch.float8_e8m0fnu
    rel_err = float((wm.float() - dequantize_mxfp8(qm, sm)).abs().max().item() / wm.abs().max().item())
    assert rel_err < 0.08, f"MXFP8 relative error too high: {rel_err}"

    # 4. End-to-end mini Hy4-flash checkpoint quantize + verify + schema validation.
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "bf16"
        dst = Path(tmp) / "nvfp4"
        src.mkdir()

        # Minimal valid HuggingFace ByteLevel-like WordLevel tokenizer JSON.
        vocab = {f"t{i}": i for i in range(64)}
        tok_json = {
            "version": "1.0",
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "t0"},
            "pre_tokenizer": {"type": "Whitespace"},
        }
        (src / "tokenizer.json").write_text(json.dumps(tok_json))

        # Mini 2-layer Hy4-flash config: layer 0 dense indexed, layer 1 sparse SWA + Engram.
        head_primes = [[13, 17], [19, 23]]
        engram_rows = sum(sum(x) for x in head_primes)
        cfg = {
            "model_type": "hy_v4_flash",
            "image_token_id": 60,
            "video_token_id": 61,
            "dfly_num_hidden_layers": 1,
            "dfly_mask_token_id": 62,
            "text_config": {
                "model_type": "hy_v4_flash_text",
                "vocab_size": 64,
                "hidden_size": 64,
                "intermediate_size": 64,
                "moe_intermediate_size": 32,
                "routed_expert_hidden_size": 32,
                "n_routed_experts": 4,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "routed_scaling_factor": 2.0,
                "norm_topk_prob": True,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "swa_num_attention_heads": 2,
                "q_lora_rank": 32,
                "kv_lora_rank": 32,
                "qk_nope_head_dim": 16,
                "qk_rope_head_dim": 16,
                "v_head_dim": 16,
                "swa_qk_nope_head_dim": 16,
                "swa_qk_rope_head_dim": 16,
                "swa_v_head_dim": 16,
                "sliding_window": 16,
                "index_topk": 16,
                "index_kpool": 4,
                "index_n_heads": 2,
                "index_head_dim": 32,
                "hc_mult": 4,
                "hc_magnitude": 2.0,
                "hc_eps": 1e-6,
                "rms_norm_eps": 1e-5,
                "swiglu_limit": 10.0,
                "layer_types": ["indexed_attention", "sliding_attention"],
                "mlp_layer_types": ["dense", "sparse"],
                "engram_layer_ids": [2],
                "engram_max_ngram_size": 3,
                "engram_n_head_per_ngram": 2,
                "engram_n_embed_per_ngram": 64,
                "engram_layer_head_vocab_sizes": {"2": head_primes},
                "engram_pad_id": 63,
                "engram_fp8_embedding": True,
                "engram_seed": 0,
                "engram_lookup_table_size": 64,
            },
        }
        (src / "config.json").write_text(json.dumps(cfg))

        bf16 = lambda *s: (torch.randn(*s, dtype=torch.float32) * 0.05).to(torch.bfloat16)
        fp32 = lambda *s: torch.randn(*s, dtype=torch.float32) * 0.05
        ones = lambda *s: torch.ones(*s, dtype=torch.bfloat16)
        tensors: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": bf16(64, 64),
            "lm_head.weight": bf16(64, 64),
            "model.norm.weight": ones(64),
            "model.hc_head.hc_head_fn": fp32(4, 256),
            "model.hc_head.hc_head_base": fp32(4),
            "model.hc_head.hc_head_scale": fp32(1),
        }
        for l in range(2):
            p = f"model.layers.{l}"
            tensors.update({
                f"{p}.hc_attn_layer.hc_pre.hc_fn": fp32(8, 256),
                f"{p}.hc_attn_layer.hc_pre.hc_base": fp32(8),
                f"{p}.hc_attn_layer.hc_pre.hc_scale": fp32(2),
                f"{p}.hc_mlp_layer.hc_pre.hc_fn": fp32(8, 256),
                f"{p}.hc_mlp_layer.hc_pre.hc_base": fp32(8),
                f"{p}.hc_mlp_layer.hc_pre.hc_scale": fp32(2),
                f"{p}.input_layernorm.weight": ones(64),
                f"{p}.post_attention_layernorm.weight": ones(64),
                f"{p}.self_attn.q_a_proj.weight": bf16(32, 64),
                f"{p}.self_attn.q_a_layernorm.weight": ones(32),
                f"{p}.self_attn.q_b_proj.weight": bf16(64, 32),
                f"{p}.self_attn.kv_a_proj_with_mqa.weight": bf16(48, 64),
                f"{p}.self_attn.kv_a_layernorm.weight": ones(32),
                f"{p}.self_attn.kv_b_proj.weight": bf16(64, 32),
                f"{p}.self_attn.linear_gate.weight": bf16(32, 64),
                f"{p}.self_attn.o_proj.weight": bf16(64, 32),
                f"{p}.self_attn.post_proj_norm.weight": ones(64),
                f"{p}.self_attn.learnable_sink_param": fp32(2),
            })
            if l == 0:
                tensors.update({
                    f"{p}.self_attn.compressed_indexer.head_weight_projection.weight": bf16(2, 64),
                    f"{p}.self_attn.compressed_indexer.intra_group_ape": fp32(4, 32),
                    f"{p}.self_attn.compressed_indexer.key_gate_projection.weight": fp32(32, 32),
                    f"{p}.self_attn.compressed_indexer.key_norm.weight": ones(32),
                    f"{p}.self_attn.compressed_indexer.key_value_projection.weight": fp32(32, 32),
                    f"{p}.self_attn.compressed_indexer.query_projection.weight": bf16(64, 32),
                    f"{p}.self_attn.compressed_indexer.token_key_norm.weight": ones(32),
                    f"{p}.self_attn.compressed_indexer.token_key_projection.weight": bf16(32, 64),
                    f"{p}.mlp.gate_proj.weight": bf16(64, 64),
                    f"{p}.mlp.up_proj.weight": bf16(64, 64),
                    f"{p}.mlp.down_proj.weight": bf16(64, 64),
                    f"{p}.mlp.post_fc2_norm.weight": ones(64),
                })
            else:
                tensors.update({
                    f"{p}.mlp.gate.weight": bf16(4, 64),
                    f"{p}.mlp.gate.e_score_correction_bias": fp32(4),
                    f"{p}.mlp.routed_expert_down_proj.weight": bf16(32, 64),
                    f"{p}.mlp.routed_expert_up_proj.weight": bf16(64, 32),
                    f"{p}.mlp.experts.gate_up_proj": bf16(4, 64, 32),
                    f"{p}.mlp.experts.down_proj": bf16(4, 32, 32),
                    f"{p}.mlp.shared_experts.gate_proj.weight": bf16(32, 64),
                    f"{p}.mlp.shared_experts.up_proj.weight": bf16(32, 64),
                    f"{p}.mlp.shared_experts.down_proj.weight": bf16(64, 32),
                    f"{p}.mlp.combine_output_norm.weight": ones(64),
                    f"{p}.engram_layer.lookup_table": torch.arange(64, dtype=torch.int32),
                    f"{p}.engram_layer.multi_head_embedding.embedding.weight": bf16(engram_rows, 32),
                    f"{p}.engram_layer.value_proj.weight": bf16(64, 128),
                    f"{p}.engram_layer.value_proj.bias": bf16(64),
                    f"{p}.engram_layer.unified_pre_gamma": ones(64),
                })
                for h_i in range(4):
                    tensors[f"{p}.engram_layer.lookup_per_head_norms.{h_i}.weight"] = ones(32)
                    tensors[f"{p}.engram_layer.key_projs.{h_i}.weight"] = bf16(64, 128)
                    tensors[f"{p}.engram_layer.key_projs.{h_i}.bias"] = bf16(64)
                    tensors[f"{p}.engram_layer.norm_key.{h_i}.weight"] = ones(64)
                    tensors[f"{p}.engram_layer.norm_query.{h_i}.weight"] = ones(64)
                    tensors[f"{p}.engram_layer.unified_per_hc_scales.{h_i}"] = ones(64)
        # Minimal DFly2 + ViT tensors to verify non-layer classification & quantization
        tensors.update({
            "model.dfly_layers.0.self_attn.q_a_proj.weight": bf16(32, 64),
            "model.dfly_layers.0.mlp.gate_proj.weight": bf16(64, 64),
            "model.dfly_layers.mtp_fusion.eh_proj.weight": bf16(64, 128),
            "model.dfly_layers.mtp_fusion.gate": bf16(64),
            "model.dfly_layers.hidden_correction.gate_proj.weight": bf16(64, 128),
            "model.dfly_layers.confidence_head.proj.weight": bf16(1, 64),
            "model.dfly_layers.confidence_head.proj.bias": bf16(1),
            "vit.embeddings.patch_embedding.weight": bf16(32, 3, 4, 4),
            "vit.perceive.image_newline": bf16(64),
        })

        save_file(tensors, str(src / "model.safetensors"))
        calib_txt = Path(tmp) / "calib.txt"
        calib_txt.write_text(" ".join(f"t{i % 59}" for i in range(512)))

        quantize_checkpoint(
            src,
            dst,
            device=device,
            num_calib_samples=4,
            calib_seq_len=32,
            micro_batch_size=2,
            calib_file=calib_txt,
            dense_policy="selective",
            use_gptq=True,
            gptq_batch_size=4,
        )

        out_cfg = json.loads((dst / "config.json").read_text())
        qlayers = quantized_layers(out_cfg)
        assert linear_format(qlayers, "model.layers.0.self_attn.q_b_proj") == "mxfp8"
        assert linear_format(qlayers, "model.layers.0.self_attn.linear_gate") == "mxfp8"
        assert linear_format(qlayers, "model.layers.0.self_attn.o_proj") == "mxfp8"
        assert linear_format(qlayers, "model.layers.1.mlp.experts.gate_proj") == "nvfp4"
        assert "model.layers.0.self_attn.q_a_proj" not in qlayers
        assert "model.layers.1.mlp.routed_expert_down_proj" not in qlayers
        assert "model.layers.1.engram_layer.value_proj" not in qlayers
        assert "model.dfly_layers.0.self_attn.q_a_proj" not in qlayers
        assert "model.dfly_layers.mtp_fusion.eh_proj" not in qlayers
        assert "model.dfly_layers.hidden_correction.gate_proj" not in qlayers
        assert "model.dfly_layers.confidence_head.proj" not in qlayers
        assert "vit.embeddings.patch_embedding" not in qlayers

        q_reader = ShardedSafetensorsReader(dst)
        assert q_reader.has("model.layers.1.engram_layer.multi_head_embedding.weight")
        assert q_reader.has("model.layers.1.engram_layer.multi_head_embedding.scale")
        assert q_reader.get_tensor("model.layers.1.engram_layer.multi_head_embedding.scale").dtype == torch.float8_e8m0fnu
        assert torch.equal(
            q_reader.get_tensor("model.layers.1.mlp.experts.gate_proj.weight_scale_2"),
            q_reader.get_tensor("model.layers.1.mlp.experts.up_proj.weight_scale_2"),
        )
        assert torch.equal(
            q_reader.get_tensor("model.layers.1.mlp.experts.gate_proj.input_scale"),
            q_reader.get_tensor("model.layers.1.mlp.experts.up_proj.input_scale"),
        )
        q_reader.close()

        # Verify batched GPTQ beats diag-H MSE on correlated synthetic activations.
        A = torch.randn((512, 128), dtype=torch.float32, device=device)
        M_corr = torch.randn((128, 128), dtype=torch.float32, device=device) * 0.25 + torch.eye(128, device=device)
        X_corr = A @ M_corr
        H_corr = (X_corr.T @ X_corr).unsqueeze(0)
        W_corr = torch.randn((1, 64, 128), dtype=torch.bfloat16, device=device) * 0.05
        s2_c = torch.tensor([float(W_corr.abs().amax().item()) / (FP8_E4M3_MAX * 4.0)], device=device)
        cnt_c = torch.tensor([512], dtype=torch.int64, device=device)
        pg, sg = quantize_nvfp4_gptq_batched(W_corr, s2_c, H_corr, cnt_c, base_damp=0.01)
        pm, sm = quantize_nvfp4_mse_block16(W_corr[0], s2_c[0], torch.diagonal(H_corr[0]) / 512.0)
        yg = X_corr @ dequantize_nvfp4(pg[0], sg[0], s2_c[0]).T
        ym = X_corr @ dequantize_nvfp4(pm, sm, s2_c[0]).T
        yr = X_corr @ W_corr[0].float().T
        assert (yg - yr).square().mean().item() < (ym - yr).square().mean().item()

        metrics = verify_quality(src, dst, device=device, num_samples=2, seq_len=32)
        assert metrics["kl_divergence"] < 0.05, f"KL too high in self-test: {metrics}"
        print("[self-test] PASSED!")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="Source local BF16 Hy4-flash checkpoint directory")
    parser.add_argument("--input-hf", type=str, default=None, help="Source Hugging Face repo ID streamed via HTTP Range (e.g. tencent/Hy4-flash)")
    parser.add_argument("--hf-revision", type=str, default="main", help="Hugging Face revision for --input-hf")
    parser.add_argument("--hf-token", type=str, default=None, help="Optional Hugging Face auth token")
    parser.add_argument("--hf-workers", type=int, default=12, help="Parallel HTTP Range fetch threads for --input-hf")
    parser.add_argument("--output", type=Path, help="Destination quantized checkpoint directory")
    parser.add_argument("--sync-dir", type=Path, default=None, help="Asynchronously mirror finished shards and metadata to a secondary directory")
    parser.add_argument("--move-to-sync-dir", action="store_true", help="Delete each local shard after verified copy into --sync-dir")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-calib-samples", type=int, default=384)
    parser.add_argument("--calib-seq-len", type=int, default=1024)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--calib-file", type=Path, default=None)
    parser.add_argument("--dense-policy", choices=("selective", "all"), default="selective")
    parser.add_argument("--no-gptq", action="store_true", help="Use diagonal-H MSE instead of batched Cholesky GPTQ")
    parser.add_argument("--no-gptq-down-proj", action="store_true", help="Keep diagonal-H MSE on expert down_proj while running GPTQ on gate_proj/up_proj")
    parser.add_argument("--gptq-batch-size", type=int, default=64)
    parser.add_argument("--no-propagate-quantized", action="store_true")
    parser.add_argument("--skip-engram-tables", action="store_true")
    parser.add_argument("--engram-format", choices=("fp8", "nvfp4"), default="fp8", help="Engram table format: fp8 (60.8 GiB) or nvfp4 (33.2 GiB, fits in 64 GiB host RAM)")
    parser.add_argument("--repack-engram-nvfp4", action="store_true", help="Replace only the 4 Engram shards in --output with NVFP4_E4M3_GLOBAL from --input/--input-hf")
    parser.add_argument("--reuse-engram-from", type=Path, default=None, help="Hardlink FP8 Engram shards from an existing quantized checkpoint")
    parser.add_argument("--verify", action="store_true", help="Run KL/PPL quality verification after quantizing (runs inline when --input-hf is used)")
    parser.add_argument("--inline-verify", action="store_true", help="Run single-pass lockstep BF16-vs-quant verification during quantization")
    parser.add_argument("--verify-only", action="store_true", help="Only run verification on --input vs --output")
    parser.add_argument("--verify-samples", type=int, default=4)
    parser.add_argument("--verify-seq-len", type=int, default=512)
    parser.add_argument("--verify-file", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true", help="Run synthetic mini-model verification and exit")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.self_test:
        _run_self_test(device)
        return
    if (args.input is None and args.input_hf is None) or args.output is None:
        parser.error("(--input or --input-hf) and --output are required unless --self-test is set")

    if args.repack_engram_nvfp4:
        repack_engram_nvfp4(
            args.input,
            args.output,
            device=device,
            input_hf=args.input_hf,
            hf_revision=args.hf_revision,
            hf_token=args.hf_token,
            hf_workers=args.hf_workers,
        )
        return

    use_inline_verify = args.inline_verify or (args.verify and args.input_hf is not None)
    if not args.verify_only:
        quantize_checkpoint(
            args.input,
            args.output,
            device=device,
            input_hf=args.input_hf,
            hf_revision=args.hf_revision,
            hf_token=args.hf_token,
            hf_workers=args.hf_workers,
            num_calib_samples=args.num_calib_samples,
            calib_seq_len=args.calib_seq_len,
            micro_batch_size=args.micro_batch_size,
            calib_file=args.calib_file,
            propagate_quantized=not args.no_propagate_quantized,
            skip_engram_tables=args.skip_engram_tables,
            engram_format=args.engram_format,
            reuse_engram_from=args.reuse_engram_from,
            dense_policy=args.dense_policy,
            use_gptq=not args.no_gptq,
            gptq_down_proj=not args.no_gptq_down_proj,
            gptq_batch_size=args.gptq_batch_size,
            sync_dir=args.sync_dir,
            move_to_sync_dir=args.move_to_sync_dir,
            inline_verify=use_inline_verify,
            verify_samples=args.verify_samples,
            verify_seq_len=args.verify_seq_len,
            verify_file=args.verify_file,
        )
    if (args.verify and not use_inline_verify) or args.verify_only:
        if args.input is None:
            parser.error("--verify-only requires local --input directory")
        verify_dir = (
            args.sync_dir if (not args.verify_only and args.move_to_sync_dir and args.sync_dir is not None) else args.output
        )
        verify_quality(
            args.input,
            verify_dir,
            device=device,
            num_samples=args.verify_samples,
            seq_len=args.verify_seq_len,
            verify_file=args.verify_file,
        )


if __name__ == "__main__":
    main()
