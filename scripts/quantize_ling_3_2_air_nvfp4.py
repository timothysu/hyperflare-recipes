# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Streaming calibrated NVFP4 + MXFP8 quantizer for inclusionAI/Ling-3.2-Air.

Converts a block-128 FP8 ``inclusionAI/Ling-3.2-Air`` (``model_type ==
"bailing_moe_v3_vl"``) checkpoint layer-by-layer into a ModelOpt
``MIXED_PRECISION`` checkpoint sized for a single 96 GiB Blackwell (SM120) GPU:

- Routed MoE experts (target layers 2..41 and MTP layer 42, 512 experts per
  layer): calibrated NVFP4 (``group_size=16``, shared ``gate_proj``/``up_proj``
  ``weight_scale_2`` and ``input_scale``) using batched block-16 Cholesky GPTQ
  with Four-Over-Six (``1.5x`` headroom) and activation-weighted MSE block-scale
  search, written as stacked 3D tensors
  (``model.layers.{i}.mlp.experts.{gate_proj,up_proj,down_proj}.*``).
- KDA input/output projections (``q_proj``, ``k_proj``, ``v_proj``, ``f_proj``,
  ``g_proj``, ``o_proj``), MLA query/latent/output projections (``q_proj``,
  ``kv_a_proj_with_mqa``, ``dense``), dense MLP layers 0..1, and shared experts
  (layers 2..42): MXFP8 (``group_size=32``, ``F8_E4M3`` weights with optimal
  dual-exponent ``F8_E8M0`` block scales).
- MLA ``kv_b_proj`` (all 8 MLA layers) and MTP ``attention.g_proj`` (layer 42):
  dequantized from source block-128 FP8 to BF16 so batched MLA key/value
  absorption and head-wise attention gating remain exact without per-layer
  format branches.
- Embeddings, ``lm_head``, KDA ``b_proj``/convolutions/``A_log``/``dt_bias``,
  MLA ``g_proj`` (layers 5..41), routers (``mlp.gate.weight`` and
  ``expert_bias``), MTP ``eh_proj``, all normalization weights, and vision
  weights (``model.visual.*``, ``linear_proj.*``): preserved verbatim.

The quantizer streams one source safetensors shard at a time (evicting each
shard as soon as its last referencing layer finishes) and runs an inline
lockstep FP8-reference vs. quantized verification pass across all 42 target
layers and the layer-42 MTP predictor.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0
FP4_POSITIVE_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
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

# Diverse fallback calibration passages spanning code, math, tool-calling XML,
# reasoning tags, English prose, and Chinese text.
_BUILTIN_CALIB_TEXTS = (
    """<role>SYSTEM</role>detailed thinking on<|role_end|><role>HUMAN</role>Implement a lock-free ring buffer and a balanced binary search tree in Rust and Python, explaining memory ordering guarantees.<|role_end|><role>ASSISTANT</role>
<think>We need a single-producer single-consumer ring buffer using Acquire/Release atomics on head and tail indices, padded to separate cache lines to avoid false sharing.</think>```python
def binary_search_left(arr: list[int], target: int) -> int:
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo
```<|role_end|>""",
    """<role>SYSTEM</role>detailed thinking on<|role_end|><role>HUMAN</role>Prove the Cauchy-Schwarz inequality and derive the optimal control variate coefficient for Monte Carlo variance reduction.<|role_end|><role>ASSISTANT</role>
<think>Let u and v be vectors in a real inner product space. Expand ||u - t v||^2 >= 0 and minimize over real t = <u, v> / <v, v>.</think>Theorem (Cauchy-Schwarz). For any vectors u, v in an inner product space, |<u, v>| <= ||u|| * ||v||, with equality iff u and v are linearly dependent. For a Monte Carlo estimator X with zero-mean control variate Z, Var(X - c Z) = Var(X) - 2 c Cov(X, Z) + c^2 Var(Z) is minimized at c* = Cov(X, Z) / Var(Z), achieving variance (1 - rho^2) Var(X).<|role_end|>""",
    """<role>SYSTEM</role># Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "units": {"type": "string"}}, "required": ["city"]}}}
</tools>

If you need to use a function, output the function name and arguments within XML tags.
detailed thinking on<|role_end|><role>HUMAN</role>Compare the current weather in Tokyo and Zurich in Celsius.<|role_end|><role>ASSISTANT</role>
<think>Call get_weather for Tokyo and Zurich with units=metric.</think><tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Tokyo</arg_value>
</tool_call>
<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Zurich</arg_value>
</tool_call><|role_end|><role>OBSERVATION</role>
<tool_response>
{"city": "Tokyo", "temp_c": 19.4, "condition": "clear"}
</tool_response>
<tool_response>
{"city": "Zurich", "temp_c": 11.2, "condition": "overcast"}
</tool_response><|role_end|><role>ASSISTANT</role>
<think>Tokyo is 8.2 degrees warmer than Zurich.</think>Tokyo is currently 19.4 C (clear), which is 8.2 C warmer than Zurich at 11.2 C (overcast).<|role_end|>""",
    """<role>SYSTEM</role>detailed thinking on<|role_end|><role>HUMAN</role>请详细解释混合专家模型（MoE）中分组路由（group-limited top-k）与无辅助损失负载均衡（expert bias）的协同机制，并给出显存带宽与计算密度的定量分析。<|role_end|><role>ASSISTANT</role>
<think>分组路由将512个专家分为8组，每组64个专家，计算每组前2名得分之和并选出前4组，再在候选的256个专家中选择Top-8。这样既限制了跨节点通信扇出，又配合动态偏置维持专家负载均衡。</think>在细粒度MoE架构中，门控层首先在FP32精度下计算所有专家的sigmoid得分，加上无梯度更新的专家偏置后执行分组筛选。对于512选8的配置，每个token仅激活约1.56%的专家参数，因此在单请求解码阶段性能严格受限于HBM/GDDR显存带宽，而将路由专家量化为4-bit NVFP4可将权重读取量减半。<|role_end|>""",
    """<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>Write a Linux epoll event loop in C with edge-triggered non-blocking sockets, proper EAGAIN draining, and clean shutdown on SIGTERM.<|role_end|><role>ASSISTANT</role>
<think></think>```c
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/epoll.h>
#include <unistd.h>

static volatile sig_atomic_t g_stop = 0;
static void on_term(int sig) { (void)sig; g_stop = 1; }

int set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    return flags < 0 ? -1 : fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}
```<|role_end|>""",
    """<role>SYSTEM</role>detailed thinking on<|role_end|><role>HUMAN</role>Given a 2D matrix of integers, design an O(M N) algorithm to find the largest rectangle containing only 1s and verify it on a 4x5 example.<|role_end|><role>ASSISTANT</role>
<think>Maintain a histogram height array per row and apply a monotonic increasing stack to compute the largest rectangle in each row histogram in O(N) time.</think>For each row, update heights[j] = heights[j] + 1 if grid[i][j] == 1 else 0. Sweep a monotonic stack of column indices with sentinel height 0 at the end; whenever heights[j] < heights[stack[-1]], pop index h_idx and update ans = max(ans, heights[h_idx] * (j - stack[-1] - 1)). Total time is O(M N) and auxiliary space is O(N).<|role_end|>""",
)


# ---------------------------------------------------------------------------
# Source checkpoint readers (local directory or sliding-cache HF download)
# ---------------------------------------------------------------------------


class LocalSafetensorsReader:
    """Read tensors from a local sharded safetensors directory."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        index_path = self.root / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"model.safetensors.index.json not found in {self.root}")
        self.weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]
        self._handles: dict[str, object] = {}

    def _open(self, shard: str):
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(str(self.root / shard), framework="pt", device="cpu")
            self._handles[shard] = handle
        return handle

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def keys(self) -> list[str]:
        return list(self.weight_map.keys())

    def get_tensor(self, name: str, device: torch.device | str = "cpu") -> torch.Tensor:
        t = self._open(self.weight_map[name]).get_tensor(name)
        return t.to(device, non_blocking=True) if str(device) != "cpu" else t

    def release_shards_except(self, keep_shards: set[str]) -> None:
        for shard in list(self._handles):
            if shard not in keep_shards:
                self._handles.pop(shard, None)

    def close(self) -> None:
        self._handles.clear()

    def __enter__(self) -> "LocalSafetensorsReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")


def _default_hub_token() -> str | None:
    from huggingface_hub.utils import get_token

    return get_token()


def _paths_overlap(a: Path, b: Path) -> bool:
    ra, rb = a.resolve(), b.resolve()
    return ra == rb or ra in rb.parents or rb in ra.parents


def validate_output_paths(
    source_dir: Path | None,
    output_dir: Path,
    *,
    sync_dir: Path | None = None,
    work_dir: Path | None = None,
) -> None:
    """Reject directory aliases and refuse to overwrite an existing checkpoint directory."""
    out = output_dir.resolve()
    sync = sync_dir.resolve() if sync_dir is not None else None
    work = work_dir.resolve() if work_dir is not None else None
    src = source_dir.resolve() if source_dir is not None else None

    if src is not None:
        if _paths_overlap(src, out):
            raise ValueError(f"--output ({out}) must not alias or overlap --source ({src})")
        if sync is not None and _paths_overlap(src, sync):
            raise ValueError(f"--sync-dir ({sync}) must not alias or overlap --source ({src})")
        if work is not None and _paths_overlap(src, work):
            raise ValueError(f"--work-dir ({work}) must not alias or overlap --source ({src})")
    if sync is not None and _paths_overlap(out, sync):
        raise ValueError(f"--sync-dir ({sync}) must not alias or overlap --output ({out})")
    if work is not None and _paths_overlap(out, work):
        raise ValueError(f"--work-dir ({work}) must not alias or overlap --output ({out})")
    if sync is not None and work is not None and _paths_overlap(sync, work):
        raise ValueError(f"--work-dir ({work}) must not alias or overlap --sync-dir ({sync})")

    for label, target in (("--output", out), ("--sync-dir", sync)):
        if target is None or not target.exists():
            continue
        existing = [p.name for p in target.iterdir()]
        if existing:
            raise FileExistsError(
                f"{label} ({target}) is not empty ({', '.join(sorted(existing)[:3])}); "
                "use a fresh empty directory"
            )


class SlidingHfShardReader:
    """Download HF safetensors shards on demand via parallel range chunks and evict finished shards."""

    def __init__(
        self,
        repo_id: str,
        revision: str,
        *,
        work_dir: Path,
        token: str | None = None,
        workers: int = 16,
    ):
        self.repo_id = repo_id
        self.requested_revision = revision
        self.workers = workers
        self.token = token if token is not None else _default_hub_token()
        self._headers = {"User-Agent": "flare-ling32-quantizer/1.0"}
        if self.token:
            self._headers["Authorization"] = f"Bearer {self.token}"
        self._local = threading.local()
        self._handles: dict[str, object] = {}
        self._shard_sizes: dict[str, int] = {}

        work_base = Path(work_dir).resolve()
        work_base.mkdir(parents=True, exist_ok=True)
        self.revision = self._resolve_commit_sha(revision)
        repo_key = hashlib.sha256(f"{repo_id}@{self.revision}".encode("utf-8")).hexdigest()[:16]
        self.root = work_base / "meta" / f"{repo_key}-{self.revision[:12]}"
        self.root.mkdir(parents=True, exist_ok=True)
        # Run-scoped temporary shard directory so concurrent conversions never collide or evict each other.
        self.shard_dir = Path(tempfile.mkdtemp(prefix=f"shards-{self.revision[:12]}-", dir=work_base))

        self._download_metadata_files()
        index_path = self.root / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"model.safetensors.index.json not found in {repo_id}@{self.revision}")
        self.weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]

    def _resolve_commit_sha(self, revision: str) -> str:
        if _HEX40_RE.match(revision):
            return revision
        info_url = f"https://huggingface.co/api/models/{self.repo_id}/revision/{revision}"
        payload = json.loads(self._http_get(info_url).decode("utf-8"))
        sha = payload.get("sha")
        if not isinstance(sha, str) or not _HEX40_RE.match(sha):
            raise RuntimeError(f"Could not resolve immutable commit SHA for {self.repo_id}@{revision}")
        return sha

    def _get_session(self):
        sess = getattr(self._local, "session", None)
        if sess is None:
            import requests
            from requests.adapters import HTTPAdapter

            sess = requests.Session()
            adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32)
            sess.mount("https://", adapter)
            sess.mount("http://", adapter)
            sess.headers.update(self._headers)
            self._local.session = sess
        return sess

    def _resolve_url(self, filename: str) -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/{self.revision}/{filename}"

    def _http_get(
        self,
        url: str,
        *,
        byte_range: tuple[int, int] | None = None,
        retries: int = 6,
        timeout: float = 120.0,
    ) -> bytes:
        hdrs = dict(self._headers)
        expected = None
        if byte_range is not None:
            s0, s1 = byte_range
            hdrs["Range"] = f"bytes={s0}-{s1}"
            expected = s1 - s0 + 1
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                sess = self._get_session()
                resp = sess.get(url, headers=hdrs, timeout=timeout, allow_redirects=True)
                if resp.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {resp.status_code} for {url}")
                data = resp.content
                if expected is not None and len(data) != expected:
                    raise RuntimeError(f"Short read {len(data)} != {expected} for {url}")
                return data
            except Exception as exc:
                last_err = exc
                if attempt + 1 < retries:
                    time.sleep(min(2.0 ** attempt, 20.0))
        raise RuntimeError(f"Failed to fetch {url} after {retries} retries: {last_err}") from last_err

    def _download_metadata_files(self) -> None:
        api_url = f"https://huggingface.co/api/models/{self.repo_id}/tree/{self.revision}"
        entries = json.loads(self._http_get(api_url).decode("utf-8"))
        for item in entries:
            if item.get("type") != "file":
                continue
            name = item["path"]
            if name.endswith(".safetensors"):
                self._shard_sizes[name] = int(item["size"])
                continue
            if "/" in name:
                continue
            expected_size = int(item["size"]) if "size" in item and item["size"] is not None else None
            dst = self.root / name
            if dst.is_file() and (expected_size is None or dst.stat().st_size == expected_size):
                continue
            data = self._http_get(self._resolve_url(name))
            if expected_size is not None and len(data) != expected_size:
                raise RuntimeError(
                    f"Metadata size mismatch for {name}: got {len(data)} bytes, expected {expected_size}"
                )
            fd, tmp_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".part", dir=self.root)
            tmp_path = Path(tmp_name)
            try:
                with os.fdopen(fd, "wb") as fout:
                    fout.write(data)
                os.replace(tmp_path, dst)
            except BaseException:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise

    def _ensure_shard(self, shard: str) -> Path:
        dst = self.shard_dir / shard
        size = self._shard_sizes[shard]
        if dst.is_file() and dst.stat().st_size == size:
            return dst
        url = self._resolve_url(shard)
        t0 = time.monotonic()
        chunk_bytes = 64 * (1 << 20)
        tmp = self.shard_dir / f"{shard}.part"
        with open(tmp, "wb") as f:
            f.truncate(size)
        spans = [(off, min(off + chunk_bytes, size) - 1) for off in range(0, size, chunk_bytes)]
        lock = threading.Lock()
        with open(tmp, "r+b") as fout:
            def _fetch(span: tuple[int, int]) -> None:
                s0, s1 = span
                data = self._http_get(url, byte_range=(s0, s1))
                with lock:
                    fout.seek(s0)
                    fout.write(data)

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
                list(pool.map(_fetch, spans))
        os.replace(tmp, dst)
        dt = time.monotonic() - t0
        print(
            f"[hf] Downloaded {shard} ({size / (1 << 30):.2f} GiB) in {dt:.1f}s "
            f"({size / (1 << 20) / max(dt, 1e-3):.0f} MiB/s)",
            flush=True,
        )
        return dst

    def _open(self, shard: str):
        handle = self._handles.get(shard)
        if handle is None:
            path = self._ensure_shard(shard)
            handle = safe_open(str(path), framework="pt", device="cpu")
            self._handles[shard] = handle
        return handle

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def keys(self) -> list[str]:
        return list(self.weight_map.keys())

    def get_tensor(self, name: str, device: torch.device | str = "cpu") -> torch.Tensor:
        t = self._open(self.weight_map[name]).get_tensor(name)
        return t.to(device, non_blocking=True) if str(device) != "cpu" else t

    def release_shards_except(self, keep_shards: set[str]) -> None:
        for shard in list(self._handles):
            if shard not in keep_shards:
                self._handles.pop(shard, None)
                path = self.shard_dir / shard
                if path.exists():
                    path.unlink()

    def close(self) -> None:
        self._handles.clear()
        shutil.rmtree(self.shard_dir, ignore_errors=True)

    def __enter__(self) -> "SlidingHfShardReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Sharded safetensors writer with optional asynchronous mirror
# ---------------------------------------------------------------------------


RESERVED_OUTPUT_FILES = frozenset(
    {"config.json", "model.safetensors.index.json", "quantization_metadata.json"}
)


def collect_non_weight_assets(source_root: Path) -> list[Path]:
    """Return non-weight, non-generated source files to copy into the output checkpoint."""
    return [
        item
        for item in sorted(source_root.iterdir())
        if item.is_file()
        and item.name not in RESERVED_OUTPUT_FILES
        and not item.name.endswith(".safetensors")
    ]


def _reserve_empty_destination(target: Path) -> tuple[Path, Path]:
    """Exclusively reserve an empty destination directory and create a run-owned staging dir."""
    target.mkdir(parents=True, exist_ok=True)
    lock_path = target / ".quantize.lock"
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"Destination {target} is already reserved by another quantizer run") from exc
    try:
        try:
            os.write(fd, f"{os.getpid()}\n".encode("ascii"))
        finally:
            os.close(fd)
        others = [p.name for p in target.iterdir() if p.name != ".quantize.lock"]
        if others:
            raise FileExistsError(
                f"Destination {target} is not empty ({', '.join(sorted(others)[:3])}); use a fresh empty directory"
            )
        stage_dir = Path(tempfile.mkdtemp(prefix=".stage-", dir=target))
    except BaseException:
        lock_path.unlink(missing_ok=True)
        raise
    return lock_path, stage_dir


class ShardedSafetensorsWriter:
    """Stages safetensors shards in a run-owned directory and publishes a complete checkpoint atomically."""

    def __init__(
        self,
        output_dir: Path,
        *,
        max_shard_bytes: int = 4 * (1 << 30),
        sync_dir: Path | None = None,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.sync_dir = Path(sync_dir).resolve() if sync_dir is not None else None
        if self.sync_dir is not None and _paths_overlap(self.output_dir, self.sync_dir):
            raise ValueError(f"--sync-dir ({self.sync_dir}) must not alias or overlap --output ({self.output_dir})")
        self.max_shard_bytes = max_shard_bytes
        self._closed = False
        self._sync_pool = None
        self._sync_futs: list[concurrent.futures.Future] = []
        self._out_lock, self.stage_dir = _reserve_empty_destination(self.output_dir)
        self._sync_lock: Path | None = None
        self.sync_stage_dir: Path | None = None
        if self.sync_dir is not None:
            try:
                self._sync_lock, self.sync_stage_dir = _reserve_empty_destination(self.sync_dir)
            except BaseException:
                self.abort()
                raise
        self._buffer: dict[str, torch.Tensor] = {}
        self._buffer_bytes = 0
        self.weight_map: dict[str, str] = {}
        self.shard_files: list[str] = []
        self.total_bytes = 0
        if self.sync_stage_dir is not None:
            self._sync_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def _enqueue_sync(self, filename: str) -> None:
        if self._sync_pool is None or self.sync_stage_dir is None:
            return
        src = self.stage_dir / filename
        dst = self.sync_stage_dir / filename
        self._sync_futs.append(self._sync_pool.submit(shutil.copy2, src, dst))

    def add(self, name: str, tensor: torch.Tensor) -> None:
        cpu = tensor.detach().to("cpu", non_blocking=False).contiguous().clone()
        nbytes = cpu.numel() * cpu.element_size()
        if self._buffer and self._buffer_bytes + nbytes > self.max_shard_bytes:
            self.flush()
        self._buffer[name] = cpu
        self._buffer_bytes += nbytes

    def flush(self) -> None:
        if not self._buffer:
            return
        idx = len(self.shard_files) + 1
        tmp_name = f"model-{idx:05d}-pending.safetensors"
        save_file(self._buffer, str(self.stage_dir / tmp_name))
        self.shard_files.append(tmp_name)
        for name, t in self._buffer.items():
            self.weight_map[name] = tmp_name
            self.total_bytes += t.numel() * t.element_size()
        self._buffer.clear()
        self._buffer_bytes = 0
        self._enqueue_sync(tmp_name)

    def _shutdown_sync(self) -> None:
        for fut in self._sync_futs:
            fut.result()
        self._sync_futs.clear()
        if self._sync_pool is not None:
            self._sync_pool.shutdown(wait=True)
            self._sync_pool = None

    def abort(self) -> None:
        """Clean up staging directories and release destination locks after an aborted run."""
        if self._closed:
            return
        self._closed = True
        if self._sync_pool is not None:
            self._sync_pool.shutdown(wait=True, cancel_futures=True)
            self._sync_pool = None
        for d in (self.stage_dir, self.sync_stage_dir):
            if d is not None and d.exists():
                shutil.rmtree(d, ignore_errors=True)
        for lock in (self._out_lock, self._sync_lock):
            if lock is not None:
                lock.unlink(missing_ok=True)

    def __enter__(self) -> "ShardedSafetensorsWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.abort()

    def publish_checkpoint(
        self,
        *,
        config_text: str,
        quant_meta_text: str,
        asset_sources: list[Path] | tuple[Path, ...] = (),
        copy_fn=shutil.copy2,
    ) -> dict:
        """Stage all shards, config, metadata, and assets before publishing index.json last."""
        promoted: list[Path] = []
        try:
            seen_assets: set[str] = set()
            for src in asset_sources:
                name = src.name
                if name in RESERVED_OUTPUT_FILES or name.endswith(".safetensors"):
                    raise ValueError(f"Asset source collides with generated checkpoint file: {name}")
                if name in seen_assets:
                    raise ValueError(f"Duplicate asset filename in checkpoint publication: {name}")
                seen_assets.add(name)
            self.flush()
            self._shutdown_sync()
            total = len(self.shard_files)
            rename_map: dict[str, str] = {}
            for i, old_name in enumerate(self.shard_files, start=1):
                new_name = f"model-{i:05d}-of-{total:05d}.safetensors"
                if old_name != new_name:
                    os.replace(self.stage_dir / old_name, self.stage_dir / new_name)
                    if self.sync_stage_dir is not None:
                        os.replace(self.sync_stage_dir / old_name, self.sync_stage_dir / new_name)
                rename_map[old_name] = new_name

            final_map = {k: rename_map[v] for k, v in sorted(self.weight_map.items())}
            expected_shards = sorted(set(final_map.values()))
            for shard_name in expected_shards:
                if not (self.stage_dir / shard_name).is_file():
                    raise FileNotFoundError(f"Missing staged shard: {self.stage_dir / shard_name}")
                if self.sync_stage_dir is not None and not (self.sync_stage_dir / shard_name).is_file():
                    raise FileNotFoundError(f"Missing mirrored staged shard: {self.sync_stage_dir / shard_name}")

            # Stage config, conversion metadata, and all non-weight assets completely before promoting anything.
            staged_files = list(expected_shards)
            for stage in (s for s in (self.stage_dir, self.sync_stage_dir) if s is not None):
                (stage / "config.json").write_text(config_text)
                (stage / "quantization_metadata.json").write_text(quant_meta_text)
                for src in asset_sources:
                    copy_fn(src, stage / src.name)
            staged_files.extend(["config.json", "quantization_metadata.json", *(p.name for p in asset_sources)])

            index = {"metadata": {"total_size": self.total_bytes}, "weight_map": final_map}
            index_text = json.dumps(index, indent=2) + "\n"

            # Promote every shard and asset into each destination, then publish model.safetensors.index.json LAST.
            targets = [(self.stage_dir, self.output_dir)]
            if self.sync_stage_dir is not None and self.sync_dir is not None:
                targets.append((self.sync_stage_dir, self.sync_dir))
            for stage, dst in targets:
                for fname in staged_files:
                    dst_file = dst / fname
                    os.replace(stage / fname, dst_file)
                    promoted.append(dst_file)
                tmp_idx = stage / "model.safetensors.index.json.tmp"
                tmp_idx.write_text(index_text)
                dst_idx = dst / "model.safetensors.index.json"
                os.replace(tmp_idx, dst_idx)
                promoted.append(dst_idx)

            for stage in (s for s in (self.stage_dir, self.sync_stage_dir) if s is not None):
                shutil.rmtree(stage, ignore_errors=True)
            for lock in (self._out_lock, self._sync_lock):
                if lock is not None:
                    lock.unlink(missing_ok=True)
            self._closed = True
            return index
        except BaseException:
            for path in promoted:
                path.unlink(missing_ok=True)
            self.abort()
            raise

    def finalize(self) -> dict:
        return self.publish_checkpoint(config_text="{}\n", quant_meta_text="{}\n")


# ---------------------------------------------------------------------------
# Quantization and dequantization primitives
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


def dequantize_block_fp8(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Dequantize a block-128 FP8 E4M3 matrix ``[N, K]`` to FP32."""
    n, k = weight.shape
    s = scale_inv.float()
    s_exp = (
        s.repeat_interleave(block_size, dim=0)[:n]
        .repeat_interleave(block_size, dim=1)[:, :k]
    )
    return weight.float() * s_exp


def load_dequantized_linear(
    reader,
    prefix: str,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Load a linear weight, dequantizing block-128 FP8 when ``weight_scale_inv`` exists."""
    w = reader.get_tensor(f"{prefix}.weight", device=device)
    scale_key = f"{prefix}.weight_scale_inv"
    if reader.has(scale_key):
        s = reader.get_tensor(scale_key, device=device)
        out = dequantize_block_fp8(w, s, block_size=128)
        return out.to(dtype)
    return w.to(dtype)


def _quantize_e2m1_rtne(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact round-to-nearest-even FP4 E2M1 quantization and 4-bit code encoding."""
    neg = x < 0
    sign_bit = neg.to(torch.uint8) << 3
    ax = x.abs()
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
    """Quantize ``[N, K]`` weight to NVFP4 (group_size=16) via Four-Over-Six + MSE search."""
    assert weight.ndim == 2 and weight.shape[1] % 16 == 0
    n, k = weight.shape
    num_blocks = k // 16
    w = weight.float().view(n, num_blocks, 16)
    gs2 = float(weight_scale_2.item())
    h = (
        hessian_diag.to(device=weight.device, dtype=torch.float32).view(1, num_blocks, 16).clamp_min(1e-12)
        if hessian_diag is not None
        else None
    )
    block_amax = w.abs().amax(dim=-1)
    base_scale = block_amax / (FP4_E2M1_MAX * gs2)
    best_err = torch.full((n, num_blocks), float("inf"), dtype=torch.float32, device=weight.device)
    best_scale_fp8 = torch.zeros((n, num_blocks), dtype=torch.float8_e4m3fn, device=weight.device)
    best_codes = torch.zeros((n, num_blocks, 16), dtype=torch.uint8, device=weight.device)
    min_fp8 = float(torch.finfo(torch.float8_e4m3fn).tiny)
    zero_mask = block_amax == 0.0

    for factor in mse_factors:
        cand_fp8 = (base_scale * factor).clamp(0.0, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        cand_f32 = cand_fp8.float()
        eff = cand_f32 * gs2
        inv_eff = torch.where(eff > 0.0, eff.reciprocal(), torch.zeros_like(eff))
        codes, deq_q = _quantize_e2m1_rtne(w * inv_eff.unsqueeze(-1))
        diff_sq = (w - deq_q * eff.unsqueeze(-1)).square()
        err = (diff_sq * h).sum(dim=-1) if h is not None else diff_sq.sum(dim=-1)
        err = torch.where(zero_mask & (cand_f32 > min_fp8), torch.full_like(err, float("inf")), err)
        improved = err < best_err
        best_err = torch.where(improved, err, best_err)
        best_scale_fp8 = torch.where(improved, cand_fp8, best_scale_fp8)
        best_codes = torch.where(improved.unsqueeze(-1), codes, best_codes)

    return pack_e2m1_nibbles(best_codes.view(n, k)), best_scale_fp8.contiguous()


@torch.no_grad()
def quantize_nvfp4_gptq_batched(
    weights: torch.Tensor,
    weight_scale_2: torch.Tensor,
    hessians: torch.Tensor,
    counts: torch.Tensor,
    *,
    mse_factors: tuple[float, ...] = DEFAULT_MSE_FACTORS,
    base_damp: float = 0.01,
    min_gptq_count: int = 128,
    blocksize: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched block-16 Cholesky GPTQ + Four-Over-Six NVFP4 quantization for ``[B, N, K]``.

    Returns ``(packed, scales, gptq_succeeded)`` where ``gptq_succeeded`` is a boolean
    mask indicating which experts actually completed Cholesky GPTQ (versus MSE/RTN).
    """
    bsz, n, k = weights.shape
    assert k % 16 == 0 and blocksize % 16 == 0
    device = weights.device

    packed_out = torch.empty((bsz, n, k // 2), dtype=torch.uint8, device=device)
    scales_out = torch.empty((bsz, n, k // 16), dtype=torch.float8_e4m3fn, device=device)
    gptq_mask = torch.zeros((bsz,), dtype=torch.bool, device=device)

    W = weights.float().clone()
    gs2 = weight_scale_2.float()
    cnt_list = counts.tolist()

    ok_idx: list[int] = []
    Hinv_list: list[torch.Tensor] = []
    hdiag_list: list[torch.Tensor] = []
    diag_idx = torch.arange(k, device=device)

    for i in range(bsz):
        cnt = cnt_list[i]
        H = hessians[i].float()
        hdiag = H.diagonal().clone()
        if cnt < min_gptq_count or float(hdiag.mean().item()) <= 1e-10:
            p_i, s_i = quantize_nvfp4_mse_block16(
                W[i], gs2[i], hessian_diag=hdiag if cnt > 0 else None, mse_factors=mse_factors
            )
            packed_out[i] = p_i
            scales_out[i] = s_i
            continue

        # Normalize H by sample count and invert via Cholesky with adaptive damping.
        Hn = H / float(cnt)
        mean_diag = float(Hn.diagonal().mean().item())
        dead = Hn.diagonal() <= 0.0
        if dead.any():
            Hn[dead, dead] = 1.0
        Hinv = None
        for mult in (1.0, 5.0, 25.0, 100.0):
            damp = max(base_damp * mult * mean_diag, 1e-6)
            Hd = Hn.clone()
            Hd[diag_idx, diag_idx] += damp
            L, info = torch.linalg.cholesky_ex(Hd)
            if int(info.item()) == 0:
                Hinv_cand = torch.cholesky_inverse(L)
                U, info2 = torch.linalg.cholesky_ex(Hinv_cand, upper=True)
                if int(info2.item()) == 0 and torch.isfinite(U).all():
                    Hinv = U
                    break
        if Hinv is None:
            p_i, s_i = quantize_nvfp4_mse_block16(
                W[i], gs2[i], hessian_diag=hdiag, mse_factors=mse_factors
            )
            packed_out[i] = p_i
            scales_out[i] = s_i
            continue

        ok_idx.append(i)
        Hinv_list.append(Hinv)
        hdiag_list.append((hdiag / float(cnt)).clamp_min(1e-12))

    if not ok_idx:
        return packed_out, scales_out, gptq_mask

    ok_t = torch.tensor(ok_idx, dtype=torch.long, device=device)
    Wb = W[ok_t]
    gs2_b = gs2[ok_t]
    Hinv_b = torch.stack(Hinv_list, dim=0)
    hdiag_b = torch.stack(hdiag_list, dim=0)
    bok = len(ok_idx)

    codes_b = torch.empty((bok, n, k), dtype=torch.uint8, device=device)
    scales_b = torch.empty((bok, n, k // 16), dtype=torch.float8_e4m3fn, device=device)
    min_fp8 = float(torch.finfo(torch.float8_e4m3fn).tiny)

    for b0 in range(0, k, blocksize):
        b1 = min(b0 + blocksize, k)
        blk_w = b1 - b0
        W_blk = Wb[:, :, b0:b1].clone()
        Err_blk = torch.zeros_like(W_blk)
        Hinv_blk = Hinv_b[:, b0:b1, b0:b1]

        for g0 in range(0, blk_w, 16):
            g1 = g0 + 16
            grp_idx = (b0 + g0) // 16
            sub = W_blk[:, :, g0:g1]
            sub_h = hdiag_b[:, b0 + g0 : b0 + g1].unsqueeze(1)
            b_amax = sub.abs().amax(dim=-1)
            base_s = b_amax / (FP4_E2M1_MAX * gs2_b[:, None])

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
            eff = best_s_fp8.float() * gs2_b[:, None]
            inv_e = torch.where(eff > 0.0, eff.reciprocal(), torch.zeros_like(eff))

            for c in range(16):
                col = g0 + c
                w_col = W_blk[:, :, col]
                code_c, dq_c = _quantize_e2m1_rtne(w_col * inv_e)
                q_col = dq_c * eff
                codes_b[:, :, b0 + col] = code_c
                d_inv = Hinv_blk[:, col, col].unsqueeze(1)
                err_c = (w_col - q_col) / d_inv
                Err_blk[:, :, col] = err_c
                if col + 1 < blk_w:
                    W_blk[:, :, col + 1 :].sub_(
                        err_c.unsqueeze(-1) * Hinv_blk[:, col, col + 1 :].unsqueeze(1)
                    )

        if b1 < k:
            Wb[:, :, b1:].baddbmm_(Err_blk, Hinv_b[:, b0:b1, b1:], beta=1.0, alpha=-1.0)

    packed_out[ok_t] = pack_e2m1_nibbles(codes_b)
    scales_out[ok_t] = scales_b
    gptq_mask[ok_t] = True
    return packed_out, scales_out, gptq_mask


@torch.no_grad()
def dequantize_nvfp4(
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
) -> torch.Tensor:
    n, half_k = packed.shape
    k = half_k * 2
    low = (packed & 0x0F).long()
    high = ((packed >> 4) & 0x0F).long()
    codes = torch.stack((low, high), dim=-1).view(n, k)
    vals = _fp4_signed_grid(packed.device)[codes].view(n, k // 16, 16)
    scales = weight_scale.float().unsqueeze(-1) * float(weight_scale_2.item())
    return (vals * scales).view(n, k)


@torch.no_grad()
def quantize_mxfp8_block32(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[N, K]`` weight to MXFP8 (group_size=32, E4M3 + E8M0) via dual-exponent MSE."""
    assert weight.ndim == 2 and weight.shape[1] % 32 == 0
    n, k = weight.shape
    blocks = weight.float().view(n, k // 32, 32)
    amax = blocks.abs().amax(dim=-1)
    ratio = (amax / FP8_E4M3_MAX).clamp_min(2.0 ** -127)
    exp_hi = torch.ceil(torch.log2(ratio)).clamp(-127.0, 127.0)
    exp_hi = torch.where(amax > 0.0, exp_hi, torch.zeros_like(exp_hi))
    exp_lo = (exp_hi - 1.0).clamp(-127.0, 127.0)

    def _eval(exp_c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_s = torch.where(amax > 0.0, torch.exp2(-exp_c), torch.zeros_like(exp_c))
        s = torch.where(amax > 0.0, torch.exp2(exp_c), torch.zeros_like(exp_c))
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
    n, k = q.shape
    exp = scale.view(torch.uint8).to(torch.int32) - 127
    s = torch.exp2(exp.float()).unsqueeze(-1)
    return (q.float().view(n, k // 32, 32) * s).view(n, k)


def compute_weight_scale_2(amax: float, *, headroom: float = 1.5) -> float:
    """Global NVFP4 scale with optional Four-Over-Six headroom (1.5x)."""
    if amax <= 0.0:
        return 1.0
    val = (amax * headroom) / (FP8_E4M3_MAX * FP4_E2M1_MAX)
    return max(val, float(torch.finfo(torch.float32).tiny))


# ---------------------------------------------------------------------------
# Ling-3.2-Air exact reference ops for layer-by-layer calibration & verification
# ---------------------------------------------------------------------------


def rms_norm_t5(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Exact ``BailingMoeV3RMSNorm``: FP32 variance/rsqrt, round to input dtype before weight."""
    dtype = x.dtype
    xf = x.float()
    normed = (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)).to(dtype)
    return normed * weight.to(dtype)


def swiglu_bailing(
    gate: torch.Tensor,
    up: torch.Tensor,
    limit: float | None,
) -> torch.Tensor:
    """Exact ``BailingMoeV3MLP`` SwiGLU with post-SiLU clamping."""
    g = F.silu(gate)
    if limit is not None:
        g = g.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return g * up


def route_bailing(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    expert_bias: torch.Tensor,
    *,
    n_group: int = 8,
    topk_group: int = 4,
    top_k: int = 8,
    routed_scaling_factor: float = 2.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact ``BailingMoeV3Gate`` group-limited sigmoid+bias routing in FP32."""
    num_tokens = x.shape[0]
    num_experts = gate_weight.shape[0]
    logits = F.linear(x.float(), gate_weight.float())
    scores = torch.sigmoid(logits)
    scores_for_routing = scores + expert_bias.float()
    group_scores = scores_for_routing.view(num_tokens, n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, n_group, num_experts // n_group)
        .reshape(num_tokens, -1)
    )
    masked_scores = scores_for_routing.masked_fill(~score_mask.bool(), float("-inf"))
    _, topk_idx = torch.topk(masked_scores, k=top_k, dim=-1)
    gathered = torch.gather(scores, dim=1, index=topk_idx)
    topk_weight = gathered / (gathered.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weight = topk_weight * routed_scaling_factor
    return topk_idx, topk_weight


def build_rope_cos_sin(seq_len: int, device: torch.device, *, rope_theta: float = 6_000_000.0, rope_dim: int = 64):
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim))
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv_freq)  # [S, 32]
    return freqs.cos(), freqs.sin()


def apply_rope_interleave(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved pair rotation on ``[..., S, 64]`` matching ``apply_rotary_pos_emb_interleave``."""
    dtype = x.dtype
    xf = x.float()
    pairs = xf.view(*xf.shape[:-1], xf.shape[-1] // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    c = cos.view(*([1] * (even.ndim - 2)), *cos.shape)
    s = sin.view(*([1] * (even.ndim - 2)), *sin.shape)
    out = torch.stack((even * c - odd * s, odd * c + even * s), dim=-1).flatten(-2)
    return out.to(dtype)


def _causal_depthwise_conv1d_silu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Exact ``ShortConvolution(..., kernel_size=4, activation='silu')`` on ``[B, S, C]``."""
    bsz, seqlen, ch = x.shape
    w = weight.view(ch, 1, -1).to(dtype=x.dtype, device=x.device)
    k = w.shape[-1]
    xt = x.transpose(1, 2)  # [B, C, S]
    xt = F.pad(xt, (k - 1, 0))
    y = F.conv1d(xt, w, groups=ch)
    return F.silu(y.transpose(1, 2))


class _B12xKdaPrefillRunner:
    """Cache prepared ``b12x.sequence.kda_prefill`` plans keyed by ``(batch, seqlen)``."""

    def __init__(self, device: torch.device, *, heads: int = 32, dim: int = 128, lower_bound: float = -5.0):
        self.device = device
        self.heads = heads
        self.dim = dim
        self.lower_bound = lower_bound
        self._plans: dict[tuple[int, int], tuple[object, object, dict]] = {}

    def _get_plan(self, bsz: int, seqlen: int):
        key = (bsz, seqlen)
        cached = self._plans.get(key)
        if cached is not None:
            return cached
        from b12x.sequence import kda_prefill
        from hyperflare_recipes.components.recurrent import prepare_sequence_binding
        from ignition.preparation.scratch import ScratchArena
        from ignition.preparation.tuning import RecordedPreparationSession

        tokens = bsz * seqlen
        width = self.heads * self.dim
        dev = self.device
        qkv = torch.zeros((tokens, 3 * width), dtype=torch.bfloat16, device=dev)
        q, k, v = (x.view(tokens, self.heads, self.dim) for x in qkv.split(width, dim=-1))
        raw_g = torch.zeros((tokens, self.heads, self.dim), dtype=torch.bfloat16, device=dev)
        raw_beta = torch.zeros((tokens, self.heads), dtype=torch.bfloat16, device=dev)
        A_log = torch.zeros((self.heads,), dtype=torch.float32, device=dev)
        dt_bias = torch.zeros((self.heads, self.dim), dtype=torch.float32, device=dev)
        state = torch.zeros((bsz + 1, self.heads, self.dim, self.dim), dtype=torch.float32, device=dev)
        cu = torch.arange(0, tokens + 1, seqlen, dtype=torch.int32, device=dev)
        init_idx = torch.zeros((bsz,), dtype=torch.int32, device=dev)
        final_idx = torch.arange(1, bsz + 1, dtype=torch.int32, device=dev)
        ckpt_idx = torch.zeros((bsz,), dtype=torch.int32, device=dev)
        ckpt_off = torch.zeros((bsz,), dtype=torch.int32, device=dev)
        nseq = torch.tensor([bsz], dtype=torch.int32, device=dev)
        ntok = torch.tensor([tokens], dtype=torch.int32, device=dev)
        out = torch.empty((tokens, self.heads, self.dim), dtype=torch.bfloat16, device=dev)
        args = dict(
            q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta, A_log=A_log, dt_bias=dt_bias,
            recurrent_state=state, cu_seqlens=cu, initial_state_indices=init_idx,
            final_state_indices=final_idx, checkpoint_state_indices=ckpt_idx,
            checkpoint_offsets=ckpt_off, num_seqs=nseq, num_tokens=ntok, output=out,
        )
        caps = kda_prefill.Caps(
            device=dev, max_tokens=tokens, max_seqs=bsz, max_state_slots=bsz + 1,
            heads=self.heads, null_state_index=0,
        )
        arena = ScratchArena(dev)
        holder: dict[str, object] = {}
        with RecordedPreparationSession(
            device=dev, checkpoint=f"ling32_calib_kda_{bsz}_{seqlen}",
            workload_schema="ling32_calib", numerical_policy="default",
            compile_workers=1, autotune_effort="off",
        ) as sess:
            sess.register_scratch(arena)
            plan = kda_prefill.plan(caps, invocation=kda_prefill.invocation_from_tensors(**args))
            prepare_sequence_binding(
                plan, args, session=sess, name=f"kda_{bsz}_{seqlen}", mode="prefill",
                public_bind=kda_prefill.bind, bind_method="bind", workspace=arena,
                run_options={"lower_bound": self.lower_bound},
                set_binding=lambda b: holder.__setitem__("binding", b),
            )
            sess.tune()
        scratch = torch.empty(plan.scratch_specs()[0].nbytes, dtype=torch.uint8, device=dev)
        entry = (plan, scratch, dict(
            qkv=qkv, cu=cu, init_idx=init_idx, final_idx=final_idx,
            ckpt_idx=ckpt_idx, ckpt_off=ckpt_off, nseq=nseq, ntok=ntok, state=state, out=out,
        ))
        self._plans[key] = entry
        return entry

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_g: torch.Tensor,
        raw_beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
    ) -> torch.Tensor:
        from b12x.sequence import kda_prefill

        bsz, seqlen, heads, dim = q.shape
        tokens = bsz * seqlen
        width = heads * dim
        plan, scratch, bufs = self._get_plan(bsz, seqlen)
        qkv = bufs["qkv"]
        qkv[:, :width].copy_(q.reshape(tokens, width))
        qkv[:, width : 2 * width].copy_(k.reshape(tokens, width))
        qkv[:, 2 * width :].copy_(v.reshape(tokens, width))
        qp, kp, vp = (x.view(tokens, heads, dim) for x in qkv.split(width, dim=-1))
        bufs["state"].zero_()
        binding = kda_prefill.bind(
            plan,
            scratch=scratch,
            q=qp,
            k=kp,
            v=vp,
            raw_g=raw_g.reshape(tokens, heads, dim).contiguous(),
            raw_beta=raw_beta.reshape(tokens, heads).contiguous(),
            A_log=A_log.contiguous(),
            dt_bias=dt_bias.view(heads, dim).contiguous(),
            recurrent_state=bufs["state"],
            cu_seqlens=bufs["cu"],
            initial_state_indices=bufs["init_idx"],
            final_state_indices=bufs["final_idx"],
            checkpoint_state_indices=bufs["ckpt_idx"],
            checkpoint_offsets=bufs["ckpt_off"],
            num_seqs=bufs["nseq"],
            num_tokens=bufs["ntok"],
            output=bufs["out"],
        )
        kda_prefill.run(binding, lower_bound=self.lower_bound)
        return bufs["out"].view(bsz, seqlen, heads, dim).clone()


def kda_forward_batch(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    *,
    eps: float = 1e-6,
    lower_bound: float = -5.0,
    kda_runner: _B12xKdaPrefillRunner,
) -> torch.Tensor:
    """Exact ``BailingMoeV3KimiDeltaAttention`` forward on ``[B, S, H]`` BF16 via b12x."""
    del lower_bound
    bsz, seqlen, _ = x.shape
    heads, dim = 32, 128
    q = _causal_depthwise_conv1d_silu(F.linear(x, w["q_proj"]), w["q_conv1d"]).view(bsz, seqlen, heads, dim)
    k = _causal_depthwise_conv1d_silu(F.linear(x, w["k_proj"]), w["k_conv1d"]).view(bsz, seqlen, heads, dim)
    v = _causal_depthwise_conv1d_silu(F.linear(x, w["v_proj"]), w["v_conv1d"]).view(bsz, seqlen, heads, dim)
    raw_g = F.linear(x, w["f_proj"]).view(bsz, seqlen, heads, dim)
    raw_beta = F.linear(x, w["b_proj"])
    z = F.linear(x, w["g_proj"]).view(bsz, seqlen, heads, dim)
    rec = kda_runner(q, k, v, raw_g, raw_beta, w["A_log"], w["dt_bias"])
    # FusedRMSNormGated(activation='sigmoid')
    rf = rec.float()
    normed = rf * torch.rsqrt(rf.pow(2).mean(dim=-1, keepdim=True) + eps)
    gated = (normed * w["o_norm"].float() * torch.sigmoid(z.float())).to(torch.bfloat16)
    return F.linear(gated.reshape(bsz, seqlen, heads * dim), w["o_proj"])


def mla_forward_batch(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Exact ``BailingMoeV3MultiLatentAttention`` forward on ``[B, S, H]`` BF16."""
    bsz, seqlen, _ = x.shape
    heads = 32
    q = F.linear(x, w["q_proj"]).view(bsz, seqlen, heads, 192).transpose(1, 2)  # [B, H, S, 192]
    q_nope, q_rot = q.split((128, 64), dim=-1)
    kv_a = F.linear(x, w["kv_a_proj_with_mqa"])
    k_pass, k_rot = kv_a.split((512, 64), dim=-1)
    kv_norm = rms_norm_t5(k_pass, w["kv_a_layernorm"], eps=eps)
    kv = F.linear(kv_norm, w["kv_b_proj"]).view(bsz, seqlen, heads, 256).transpose(1, 2)
    k_nope, v = kv.split((128, 128), dim=-1)
    k_rot = k_rot.view(bsz, 1, seqlen, 64)
    q_rot = apply_rope_interleave(q_rot, cos, sin)
    k_rot = apply_rope_interleave(k_rot, cos, sin).expand(bsz, heads, seqlen, 64)
    q_full = torch.cat((q_nope, q_rot), dim=-1)
    k_full = torch.cat((k_nope, k_rot), dim=-1)
    attn = F.scaled_dot_product_attention(q_full, k_full, v, is_causal=True, scale=192.0 ** -0.5)
    attn = attn.transpose(1, 2).contiguous()  # [B, S, H, 128]
    gate = torch.sigmoid(F.linear(x, w["g_proj"]).float()).to(x.dtype)  # [B, S, H]
    attn = attn * gate.unsqueeze(-1)
    return F.linear(attn.reshape(bsz, seqlen, heads * 128), w["dense"])


# ---------------------------------------------------------------------------
# Calibration token builder
# ---------------------------------------------------------------------------


def split_disjoint_token_streams(
    encode_fn,
    records: list[str],
    *,
    num_samples: int,
    verify_seqs: int,
    seq_len: int,
    prefix_calib_records: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Split distinct source records into calibration and held-out verification tensors.

    Records are partitioned strictly before tokenization so no record can appear
    in both calibration and verification. Raises ``ValueError`` when the corpus
    does not supply enough distinct tokens without repetition.
    """
    if num_samples <= 0 or verify_seqs <= 0 or seq_len <= 1:
        raise ValueError("num_samples, verify_seqs must be positive and seq_len > 1")
    need_calib = num_samples * seq_len
    need_verify = verify_seqs * seq_len

    # Fill the held-out verification slice from the tail of source records first so it
    # never overlaps calibration records (and never includes prefix_calib_records).
    verify_ids: list[int] = []
    split_idx = len(records)
    while split_idx > 0 and len(verify_ids) < need_verify:
        split_idx -= 1
        verify_ids.extend(encode_fn(records[split_idx]))
    if len(verify_ids) < need_verify:
        raise ValueError(
            f"Corpus produced only {len(verify_ids)} held-out verification tokens; need {need_verify} "
            f"({verify_seqs} x {seq_len}) without repeating records"
        )

    calib_records = list(prefix_calib_records or ()) + records[:split_idx]
    calib_ids: list[int] = []
    used_calib_records = 0
    for rec in calib_records:
        calib_ids.extend(encode_fn(rec))
        used_calib_records += 1
        if len(calib_ids) >= need_calib:
            break
    if len(calib_ids) < need_calib:
        raise ValueError(
            f"Calibration partition produced only {len(calib_ids)} tokens from {len(calib_records)} "
            f"disjoint records; need {need_calib} ({num_samples} x {seq_len}) without repeating records"
        )

    calib = torch.tensor(calib_ids[:need_calib], dtype=torch.long).view(num_samples, seq_len)
    verify = torch.tensor(verify_ids[:need_verify], dtype=torch.long).view(verify_seqs, seq_len)
    meta = {
        "total_source_records": len(records),
        "prefix_calib_records": len(prefix_calib_records or ()),
        "calib_source_records_available": split_idx,
        "calib_records_consumed": used_calib_records,
        "verify_source_records": len(records) - split_idx,
        "calib_tokens": need_calib,
        "verify_tokens": need_verify,
        "seq_len": seq_len,
    }
    return calib, verify, meta


def build_calibration_tokens(
    tokenizer_path: Path,
    *,
    num_samples: int,
    verify_seqs: int,
    seq_len: int,
    dataset_name: str,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Stream the requested dataset and build strictly disjoint calibration and verification splits."""
    from datasets import load_dataset
    from tokenizers import Tokenizer

    if not dataset_name:
        raise ValueError("--calib-dataset must name a non-empty Hugging Face dataset")

    tok = Tokenizer.from_file(str(tokenizer_path))
    texts: list[str] = []

    print(f"[calib] Streaming calibration corpus from {dataset_name}...", flush=True)
    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        field = "text"
    elif dataset_name == "fineweb":
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        field = "text"
    else:
        ds = load_dataset(dataset_name, split="train", streaming=True)
        field = None

    target_records = num_samples + verify_seqs + 32
    buf = ""
    for row in ds:
        if field is None:
            for cand in ("text", "content", "prompt", "code"):
                if cand in row and isinstance(row[cand], str):
                    field = cand
                    break
        txt = row.get(field or "text", "")
        if not txt or len(txt) < 64:
            continue
        buf += txt + "\n\n"
        if len(buf) >= 2400:
            texts.append(buf)
            buf = ""
            if len(texts) >= target_records:
                break
    if buf:
        texts.append(buf)
    print(f"[calib] Collected {len(texts)} corpus records from {dataset_name}", flush=True)

    calib, verify, split_meta = split_disjoint_token_streams(
        lambda text: tok.encode(text, add_special_tokens=False).ids,
        texts,
        num_samples=num_samples,
        verify_seqs=verify_seqs,
        seq_len=seq_len,
        prefix_calib_records=list(_BUILTIN_CALIB_TEXTS),
    )
    split_meta["dataset"] = dataset_name
    return calib, verify, split_meta


# ---------------------------------------------------------------------------
# Main layer-by-layer streaming pipeline
# ---------------------------------------------------------------------------


def _layer_type(layer_idx: int, num_layers: int, group_size: int) -> str:
    if layer_idx >= num_layers:
        return "mla"  # MTP layer 42 uses Gated MLA
    if (layer_idx + 1) % group_size == 0 or layer_idx >= (num_layers // group_size) * group_size:
        return "mla"
    return "kda"


def _compute_shard_last_layer(weight_map: dict[str, str], total_layers: int) -> dict[str, int]:
    """Return the highest layer index referencing each shard (or total_layers for top-level)."""
    last: dict[str, int] = {}
    for name, shard in weight_map.items():
        if name.startswith("model.layers."):
            idx = int(name.split(".")[2])
        elif name.startswith(("model.visual.", "linear_proj.")):
            idx = total_layers + 1  # copied at the end
        elif name in ("lm_head.weight", "model.norm.weight"):
            idx = total_layers + 1
        else:
            idx = 0
        if idx > last.get(shard, -1):
            last[shard] = idx
    return last


def assert_nonzero_expert_counts(counts: torch.Tensor, prefix: str) -> None:
    """Fail explicitly if any routed expert received zero calibration tokens."""
    zero_ids = (counts == 0).nonzero(as_tuple=False).reshape(-1).tolist()
    if zero_ids:
        raise RuntimeError(
            f"{prefix}.mlp.experts.{zero_ids[0]} received 0 calibration tokens "
            f"({len(zero_ids)} zero-token expert(s)); increase --calib-samples so every expert "
            "has a measured activation scale"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize inclusionAI/Ling-3.2-Air to calibrated NVFP4 + MXFP8.")
    parser.add_argument("--source", default="inclusionAI/Ling-3.2-Air", help="Local directory or HF repo ID")
    parser.add_argument("--revision", default="1613b1a4dcc6aa0d8f346606628cd9a92061a525")
    parser.add_argument("--output", required=True, help="Output quantized checkpoint directory")
    parser.add_argument("--sync-dir", default=None, help="Optional secondary mirror directory (e.g., Google Drive)")
    parser.add_argument("--work-dir", default="/tmp/ling32_quant_work", help="Temp directory for sliding shard cache")
    parser.add_argument("--calib-samples", type=int, default=384, help="Number of calibration sequences")
    parser.add_argument("--calib-seq-len", type=int, default=512, help="Tokens per calibration sequence")
    parser.add_argument("--calib-dataset", default="wikitext", help="HF dataset name (wikitext, fineweb, or empty)")
    parser.add_argument("--calib-batch-size", type=int, default=16, help="Micro-batch size for layer forward")
    parser.add_argument("--gptq-batch", type=int, default=64, help="Experts quantized in parallel per Cholesky batch")
    parser.add_argument("--min-gptq-count", type=int, default=128, help="Min routed tokens for full Cholesky GPTQ")
    parser.add_argument("--verify-seqs", type=int, default=4, help="Disjoint sequences for lockstep KL/top1/PPL check")
    parser.add_argument("--workers", type=int, default=16, help="Parallel chunk download threads")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    t_start = time.monotonic()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output_dir = Path(args.output).resolve()
    sync_dir = Path(args.sync_dir).resolve() if args.sync_dir else None
    src_path = Path(args.source)
    is_local_source = src_path.is_dir()
    work_dir = None if is_local_source else Path(args.work_dir).resolve()
    validate_output_paths(
        src_path if is_local_source else None,
        output_dir,
        sync_dir=sync_dir,
        work_dir=work_dir,
    )

    if is_local_source:
        reader = LocalSafetensorsReader(src_path)
    else:
        assert work_dir is not None
        reader = SlidingHfShardReader(
            args.source,
            args.revision,
            work_dir=work_dir,
            workers=args.workers,
        )
    with reader:
        _run_quantization(args, device=device, reader=reader, output_dir=output_dir, sync_dir=sync_dir, t_start=t_start)


def _run_quantization(
    args: argparse.Namespace,
    *,
    device: torch.device,
    reader: LocalSafetensorsReader | SlidingHfShardReader,
    output_dir: Path,
    sync_dir: Path | None,
    t_start: float,
) -> None:
    config = json.loads((reader.root / "config.json").read_text())
    tc = config["text_config"]
    num_layers = int(tc["num_hidden_layers"])
    num_mtp = int(tc.get("num_nextn_predict_layers", 0))
    total_layers = num_layers + num_mtp
    hidden_size = int(tc["hidden_size"])
    num_experts = int(tc["num_experts"])
    moe_inter = int(tc["moe_intermediate_size"])
    first_dense = int(tc["first_k_dense_replace"])
    group_size = int(tc["layer_group_size"])
    eps = float(tc["rms_norm_eps"])
    rope_theta = float(tc["rope_theta"])
    n_group = int(tc["n_group"])
    topk_group = int(tc["topk_group"])
    top_k = int(tc["num_experts_per_tok"])
    routed_scale = float(tc["routed_scaling_factor"])
    kda_lower = float(tc.get("kda_lower_bound", -5.0))
    expert_limits = tc.get("expert_swiglu_limit_list", [])
    shared_limits = tc.get("share_expert_swiglu_limit_list", [])

    import b12x.sequence.kda_prefill  # noqa: F401 - required for KDA calibration forward

    kda_runner = _B12xKdaPrefillRunner(device, heads=32, dim=128, lower_bound=kda_lower)
    print(
        f"[init] Ling-3.2-Air: layers={num_layers}+{num_mtp}MTP, hidden={hidden_size}, "
        f"experts={num_experts}x{moe_inter}",
        flush=True,
    )

    calib_ids, verify_ids, corpus_metadata = build_calibration_tokens(
        reader.root / "tokenizer.json",
        num_samples=args.calib_samples,
        verify_seqs=args.verify_seqs,
        seq_len=args.calib_seq_len,
        dataset_name=args.calib_dataset,
    )
    print(
        f"[calib] calib={tuple(calib_ids.shape)} ({calib_ids.numel()} tokens), "
        f"verify={tuple(verify_ids.shape)} ({verify_ids.numel()} tokens)",
        flush=True,
    )

    shard_last = _compute_shard_last_layer(reader.weight_map, total_layers)
    with ShardedSafetensorsWriter(output_dir, max_shard_bytes=4 * (1 << 30), sync_dir=sync_dir) as writer:
        _quantize_and_publish(
            args=args,
            device=device,
            reader=reader,
            writer=writer,
            config=config,
            calib_ids=calib_ids,
            verify_ids=verify_ids,
            corpus_metadata=corpus_metadata,
            kda_runner=kda_runner,
            shard_last=shard_last,
            num_layers=num_layers,
            total_layers=total_layers,
            hidden_size=hidden_size,
            num_experts=num_experts,
            moe_inter=moe_inter,
            first_dense=first_dense,
            group_size=group_size,
            eps=eps,
            rope_theta=rope_theta,
            n_group=n_group,
            topk_group=topk_group,
            top_k=top_k,
            routed_scale=routed_scale,
            kda_lower=kda_lower,
            expert_limits=expert_limits,
            shared_limits=shared_limits,
            output_dir=output_dir,
            t_start=t_start,
        )


def _quantize_and_publish(
    *,
    args: argparse.Namespace,
    device: torch.device,
    reader: LocalSafetensorsReader | SlidingHfShardReader,
    writer: ShardedSafetensorsWriter,
    config: dict,
    calib_ids: torch.Tensor,
    verify_ids: torch.Tensor,
    corpus_metadata: dict,
    kda_runner: _B12xKdaPrefillRunner,
    shard_last: dict[str, int],
    num_layers: int,
    total_layers: int,
    hidden_size: int,
    num_experts: int,
    moe_inter: int,
    first_dense: int,
    group_size: int,
    eps: float,
    rope_theta: float,
    n_group: int,
    topk_group: int,
    top_k: int,
    routed_scale: float,
    kda_lower: float,
    expert_limits: list,
    shared_limits: list,
    output_dir: Path,
    t_start: float,
) -> None:
    quantized_layers: dict[str, dict] = {}

    # 1. Word embeddings -> initial hidden states on CPU pinned/paged RAM.
    embed_w = reader.get_tensor("model.word_embeddings.weight", device=device).to(torch.bfloat16)
    writer.add("model.word_embeddings.weight", embed_w)
    calib_h = F.embedding(calib_ids.to(device), embed_w).cpu()  # [N, S, H] BF16 on CPU
    ver_h_ref = F.embedding(verify_ids.to(device), embed_w)  # [V, S, H] BF16 on GPU
    ver_h_q = ver_h_ref.clone()
    # Shifted token embeddings for Layer-42 MTP calibration & verification.
    mtp_calib_ids = torch.roll(calib_ids, shifts=-1, dims=-1)
    mtp_calib_ids[:, -1] = 0
    mtp_ver_ids = torch.roll(verify_ids, shifts=-1, dims=-1)
    mtp_ver_ids[:, -1] = 0
    del embed_w
    torch.cuda.empty_cache()

    cos_c, sin_c = build_rope_cos_sin(args.calib_seq_len, device, rope_theta=rope_theta, rope_dim=64)
    total_gptq_experts = 0
    total_mse_experts = 0

    def quant_and_write_mxfp8(prefix: str, w_f32: torch.Tensor) -> torch.Tensor:
        q, sc = quantize_mxfp8_block32(w_f32)
        writer.add(f"{prefix}.weight", q)
        writer.add(f"{prefix}.weight_scale", sc)
        quantized_layers[prefix] = {"quant_algo": "MXFP8", "group_size": 32}
        return dequantize_mxfp8(q, sc).to(torch.bfloat16)

    # 2. Stream target layers 0..num_layers-1, finalize target norm/lm_head + target
    #    verification independently of num_mtp, then stream optional MTP layer(s).
    final_norm_w: torch.Tensor | None = None
    lm_head_w: torch.Tensor | None = None
    target_metrics: dict[str, float] | None = None
    mtp_metrics: dict[str, float] | None = None
    for layer_idx in range(total_layers):
        t_layer = time.monotonic()
        is_mtp = layer_idx >= num_layers
        prefix = f"model.layers.{layer_idx}"
        kind = _layer_type(layer_idx, num_layers, group_size)
        is_moe = layer_idx >= first_dense

        if is_mtp:
            assert final_norm_w is not None and lm_head_w is not None
            # Prepare MTP inputs: eh_proj(cat(enorm(embed(roll(ids))), hnorm(norm(h))))
            enorm_w = reader.get_tensor(f"{prefix}.enorm.weight", device=device).to(torch.bfloat16)
            hnorm_w = reader.get_tensor(f"{prefix}.hnorm.weight", device=device).to(torch.bfloat16)
            eh_w = load_dequantized_linear(reader, f"{prefix}.eh_proj", device, dtype=torch.bfloat16)
            final_mtp_norm = reader.get_tensor(f"{prefix}.final_layernorm.weight", device=device).to(torch.bfloat16)
            writer.add(f"{prefix}.enorm.weight", enorm_w)
            writer.add(f"{prefix}.hnorm.weight", hnorm_w)
            writer.add(f"{prefix}.eh_proj.weight", eh_w)
            writer.add(f"{prefix}.final_layernorm.weight", final_mtp_norm)

            embed_w = reader.get_tensor("model.word_embeddings.weight", device=device).to(torch.bfloat16)
            # Transform calib_h through model.norm and eh_proj in micro-batches.
            new_calib = torch.empty_like(calib_h)
            for b0 in range(0, calib_h.shape[0], args.calib_batch_size):
                b1 = min(b0 + args.calib_batch_size, calib_h.shape[0])
                h_b = rms_norm_t5(calib_h[b0:b1].to(device), final_norm_w, eps=eps)
                e_b = F.embedding(mtp_calib_ids[b0:b1].to(device), embed_w)
                joined = torch.cat((rms_norm_t5(e_b, enorm_w, eps=eps), rms_norm_t5(h_b, hnorm_w, eps=eps)), dim=-1)
                new_calib[b0:b1] = F.linear(joined, eh_w).cpu()
            calib_h = new_calib

            e_ver = F.embedding(mtp_ver_ids.to(device), embed_w)
            ver_h_ref = F.linear(
                torch.cat((rms_norm_t5(e_ver, enorm_w, eps=eps), rms_norm_t5(ver_h_ref, hnorm_w, eps=eps)), dim=-1),
                eh_w,
            )
            ver_h_q = F.linear(
                torch.cat((rms_norm_t5(e_ver, enorm_w, eps=eps), rms_norm_t5(ver_h_q, hnorm_w, eps=eps)), dim=-1),
                eh_w,
            )
            del embed_w, enorm_w, hnorm_w, eh_w

        # --- Attention block ---
        in_norm = reader.get_tensor(f"{prefix}.input_layernorm.weight", device=device).to(torch.bfloat16)
        post_norm = reader.get_tensor(f"{prefix}.post_attention_layernorm.weight", device=device).to(torch.bfloat16)
        writer.add(f"{prefix}.input_layernorm.weight", in_norm)
        writer.add(f"{prefix}.post_attention_layernorm.weight", post_norm)

        attn_ref: dict[str, torch.Tensor] = {}
        attn_q: dict[str, torch.Tensor] = {}
        if kind == "kda":
            for proj in ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj", "o_proj"):
                wf = load_dequantized_linear(reader, f"{prefix}.attention.{proj}", device, dtype=torch.float32)
                attn_ref[proj] = wf.to(torch.bfloat16)
                attn_q[proj] = quant_and_write_mxfp8(f"{prefix}.attention.{proj}", wf)
                del wf
            for name, key in (
                ("b_proj", "b_proj.weight"),
                ("q_conv1d", "q_conv1d.weight"),
                ("k_conv1d", "k_conv1d.weight"),
                ("v_conv1d", "v_conv1d.weight"),
                ("o_norm", "o_norm.weight"),
            ):
                t = reader.get_tensor(f"{prefix}.attention.{key}", device=device).to(torch.bfloat16)
                writer.add(f"{prefix}.attention.{key}", t)
                attn_ref[name] = attn_q[name] = t
            for name in ("A_log", "dt_bias"):
                t = reader.get_tensor(f"{prefix}.attention.{name}", device=device).to(torch.float32)
                writer.add(f"{prefix}.attention.{name}", t)
                attn_ref[name] = attn_q[name] = t
        else:
            for proj in ("q_proj", "kv_a_proj_with_mqa", "dense"):
                wf = load_dequantized_linear(reader, f"{prefix}.attention.{proj}", device, dtype=torch.float32)
                attn_ref[proj] = wf.to(torch.bfloat16)
                attn_q[proj] = quant_and_write_mxfp8(f"{prefix}.attention.{proj}", wf)
                del wf
            # Keep kv_b_proj and g_proj exact BF16 across all MLA layers (including MTP layer 42).
            for proj in ("kv_b_proj", "g_proj"):
                tb = load_dequantized_linear(reader, f"{prefix}.attention.{proj}", device, dtype=torch.bfloat16)
                writer.add(f"{prefix}.attention.{proj}.weight", tb)
                attn_ref[proj] = attn_q[proj] = tb
            kv_ln = reader.get_tensor(f"{prefix}.attention.kv_a_layernorm.weight", device=device).to(torch.bfloat16)
            writer.add(f"{prefix}.attention.kv_a_layernorm.weight", kv_ln)
            attn_ref["kv_a_layernorm"] = attn_q["kv_a_layernorm"] = kv_ln

        # Run attention on calibration sequences (using quantized attention weights so
        # downstream MoE Hessians see actual post-quantization activations!) + lockstep verify.
        mlp_in_calib = torch.empty_like(calib_h)
        for b0 in range(0, calib_h.shape[0], args.calib_batch_size):
            b1 = min(b0 + args.calib_batch_size, calib_h.shape[0])
            hb = calib_h[b0:b1].to(device)
            normed = rms_norm_t5(hb, in_norm, eps=eps)
            if kind == "kda":
                ao = kda_forward_batch(normed, attn_q, eps=eps, lower_bound=kda_lower, kda_runner=kda_runner)
            else:
                ao = mla_forward_batch(normed, attn_q, cos_c, sin_c, eps=eps)
            hb = hb + ao
            calib_h[b0:b1] = hb.cpu()
            mlp_in_calib[b0:b1] = rms_norm_t5(hb, post_norm, eps=eps).cpu()

        # Verification attention step (ref vs quant).
        vr_norm = rms_norm_t5(ver_h_ref, in_norm, eps=eps)
        vq_norm = rms_norm_t5(ver_h_q, in_norm, eps=eps)
        if kind == "kda":
            ver_h_ref = ver_h_ref + kda_forward_batch(vr_norm, attn_ref, eps=eps, lower_bound=kda_lower, kda_runner=kda_runner)
            ver_h_q = ver_h_q + kda_forward_batch(vq_norm, attn_q, eps=eps, lower_bound=kda_lower, kda_runner=kda_runner)
        else:
            ver_h_ref = ver_h_ref + mla_forward_batch(vr_norm, attn_ref, cos_c, sin_c, eps=eps)
            ver_h_q = ver_h_q + mla_forward_batch(vq_norm, attn_q, cos_c, sin_c, eps=eps)
        del attn_ref, attn_q, vr_norm, vq_norm

        # --- MLP / MoE block ---
        vr_mlp_in = rms_norm_t5(ver_h_ref, post_norm, eps=eps)
        vq_mlp_in = rms_norm_t5(ver_h_q, post_norm, eps=eps)

        if not is_moe:
            mlp_ref, mlp_q = {}, {}
            for proj in ("gate_proj", "up_proj", "down_proj"):
                wf = load_dequantized_linear(reader, f"{prefix}.mlp.{proj}", device, dtype=torch.float32)
                mlp_ref[proj] = wf.to(torch.bfloat16)
                mlp_q[proj] = quant_and_write_mxfp8(f"{prefix}.mlp.{proj}", wf)
                del wf
            for b0 in range(0, calib_h.shape[0], args.calib_batch_size):
                b1 = min(b0 + args.calib_batch_size, calib_h.shape[0])
                xb = mlp_in_calib[b0:b1].to(device)
                mo = F.linear(swiglu_bailing(F.linear(xb, mlp_q["gate_proj"]), F.linear(xb, mlp_q["up_proj"]), None), mlp_q["down_proj"])
                calib_h[b0:b1] = (calib_h[b0:b1].to(device) + mo).cpu()
            ver_h_ref = ver_h_ref + F.linear(swiglu_bailing(F.linear(vr_mlp_in, mlp_ref["gate_proj"]), F.linear(vr_mlp_in, mlp_ref["up_proj"]), None), mlp_ref["down_proj"])
            ver_h_q = ver_h_q + F.linear(swiglu_bailing(F.linear(vq_mlp_in, mlp_q["gate_proj"]), F.linear(vq_mlp_in, mlp_q["up_proj"]), None), mlp_q["down_proj"])
            del mlp_ref, mlp_q
            print(f"[layer {layer_idx:2d}/{total_layers-1}] {kind.upper()} + dense MLP -> MXFP8 ({time.monotonic()-t_layer:.1f}s)", flush=True)
        else:
            exp_limit = (expert_limits[layer_idx] or None) if layer_idx < len(expert_limits) else None
            sh_limit = (shared_limits[layer_idx] or None) if layer_idx < len(shared_limits) else None

            # Router (FP32 verbatim)
            gate_w = reader.get_tensor(f"{prefix}.mlp.gate.weight", device=device).to(torch.float32)
            exp_bias = reader.get_tensor(f"{prefix}.mlp.gate.expert_bias", device=device).to(torch.float32)
            writer.add(f"{prefix}.mlp.gate.weight", gate_w)
            writer.add(f"{prefix}.mlp.gate.expert_bias", exp_bias)

            # Shared expert -> MXFP8
            sh_ref, sh_q = {}, {}
            for proj in ("gate_proj", "up_proj", "down_proj"):
                wf = load_dequantized_linear(reader, f"{prefix}.mlp.shared_experts.{proj}", device, dtype=torch.float32)
                sh_ref[proj] = wf.to(torch.bfloat16)
                sh_q[proj] = quant_and_write_mxfp8(f"{prefix}.mlp.shared_experts.{proj}", wf)
                del wf

            # Route all calibration tokens once and accumulate per-expert token lists.
            flat_x = mlp_in_calib.view(-1, hidden_size)
            n_tok = flat_x.shape[0]
            all_topk_idx = torch.empty((n_tok, top_k), dtype=torch.int32)
            all_topk_w = torch.empty((n_tok, top_k), dtype=torch.float32)
            for t0 in range(0, n_tok, 8192):
                t1 = min(t0 + 8192, n_tok)
                ti, tw = route_bailing(
                    flat_x[t0:t1].to(device),
                    gate_w,
                    exp_bias,
                    n_group=n_group,
                    topk_group=topk_group,
                    top_k=top_k,
                    routed_scaling_factor=routed_scale,
                )
                all_topk_idx[t0:t1] = ti.to(torch.int32).cpu()
                all_topk_w[t0:t1] = tw.cpu()

            # Sort routed assignments by expert id for contiguous slicing.
            flat_eids = all_topk_idx.reshape(-1)
            flat_toks = torch.arange(n_tok, dtype=torch.int32).unsqueeze(1).expand(-1, top_k).reshape(-1)
            flat_wts = all_topk_w.reshape(-1)
            order = torch.argsort(flat_eids, stable=True)
            sorted_eids = flat_eids[order]
            sorted_toks = flat_toks[order]
            sorted_wts = flat_wts[order]
            counts = torch.bincount(sorted_eids, minlength=num_experts)
            assert_nonzero_expert_counts(counts, prefix)
            offsets = torch.zeros(num_experts + 1, dtype=torch.int64)
            offsets[1:] = counts.cumsum(0)

            # Route verification tokens (ref and quant).
            vr_flat = vr_mlp_in.reshape(-1, hidden_size)
            vq_flat = vq_mlp_in.reshape(-1, hidden_size)
            vr_ti, vr_tw = route_bailing(vr_flat, gate_w, exp_bias, n_group=n_group, topk_group=topk_group, top_k=top_k, routed_scaling_factor=routed_scale)
            vq_ti, vq_tw = route_bailing(vq_flat, gate_w, exp_bias, n_group=n_group, topk_group=topk_group, top_k=top_k, routed_scaling_factor=routed_scale)
            vr_moe_out = torch.zeros_like(vr_flat, dtype=torch.float32)
            vq_moe_out = torch.zeros_like(vq_flat, dtype=torch.float32)

            # Accumulate next-layer calib_h update (starting from shared expert).
            calib_moe_out = torch.zeros((n_tok, hidden_size), dtype=torch.float32)
            for t0 in range(0, n_tok, 8192):
                t1 = min(t0 + 8192, n_tok)
                xb = flat_x[t0:t1].to(device)
                so = F.linear(swiglu_bailing(F.linear(xb, sh_q["gate_proj"]), F.linear(xb, sh_q["up_proj"]), sh_limit), sh_q["down_proj"])
                calib_moe_out[t0:t1] = so.float().cpu()

            # Preallocate stacked 3D NVFP4 output tensors on CPU.
            g_pack = torch.empty((num_experts, moe_inter, hidden_size // 2), dtype=torch.uint8)
            g_sc = torch.empty((num_experts, moe_inter, hidden_size // 16), dtype=torch.float8_e4m3fn)
            u_pack = torch.empty((num_experts, moe_inter, hidden_size // 2), dtype=torch.uint8)
            u_sc = torch.empty((num_experts, moe_inter, hidden_size // 16), dtype=torch.float8_e4m3fn)
            d_pack = torch.empty((num_experts, hidden_size, moe_inter // 2), dtype=torch.uint8)
            d_sc = torch.empty((num_experts, hidden_size, moe_inter // 16), dtype=torch.float8_e4m3fn)
            gu_gs2 = torch.empty((num_experts,), dtype=torch.float32)
            d_gs2 = torch.empty((num_experts,), dtype=torch.float32)
            gu_in_sc = torch.empty((num_experts,), dtype=torch.float32)
            d_in_sc = torch.empty((num_experts,), dtype=torch.float32)

            # Quantize 512 experts in batches of args.gptq_batch.
            layer_gptq = 0
            layer_mse = 0
            for e0 in range(0, num_experts, args.gptq_batch):
                e1 = min(e0 + args.gptq_batch, num_experts)
                bsz_e = e1 - e0
                # Load and dequantize batch of experts from source FP8.
                w_gate = torch.empty((bsz_e, moe_inter, hidden_size), dtype=torch.float32, device=device)
                w_up = torch.empty((bsz_e, moe_inter, hidden_size), dtype=torch.float32, device=device)
                w_down = torch.empty((bsz_e, hidden_size, moe_inter), dtype=torch.float32, device=device)
                for j, e in enumerate(range(e0, e1)):
                    ep = f"{prefix}.mlp.experts.{e}"
                    w_gate[j] = load_dequantized_linear(reader, f"{ep}.gate_proj", device, dtype=torch.float32)
                    w_up[j] = load_dequantized_linear(reader, f"{ep}.up_proj", device, dtype=torch.float32)
                    w_down[j] = load_dequantized_linear(reader, f"{ep}.down_proj", device, dtype=torch.float32)

                # Build Hessians & input_scales for gate+up (shared H_x) and down (H_mid).
                H_x = torch.zeros((bsz_e, hidden_size, hidden_size), dtype=torch.float32, device=device)
                H_mid = torch.zeros((bsz_e, moe_inter, moe_inter), dtype=torch.float32, device=device)
                cnt_b = torch.zeros((bsz_e,), dtype=torch.int64, device=device)
                gu_amax_x = torch.full((bsz_e,), 1e-4, dtype=torch.float32, device=device)
                d_amax_mid = torch.full((bsz_e,), 1e-4, dtype=torch.float32, device=device)

                for j, e in enumerate(range(e0, e1)):
                    s0, s1 = int(offsets[e].item()), int(offsets[e + 1].item())
                    c_e = s1 - s0
                    cnt_b[j] = c_e
                    if c_e == 0:
                        raise RuntimeError(
                            f"{prefix}.mlp.experts.{e} received 0 calibration tokens; "
                            "increase --calib-samples so every expert has a measured activation scale"
                        )
                    tok_idx = sorted_toks[s0:s1].long()
                    xe = flat_x[tok_idx].to(device)  # [c_e, H] BF16
                    xf = xe.float()
                    gu_amax_x[j] = xf.abs().amax().clamp_min(1e-4)
                    H_x[j] = xf.T @ xf
                    # Compute exact SwiGLU intermediate with ref weights for H_mid and down input_scale.
                    wg_bf = w_gate[j].to(torch.bfloat16)
                    wu_bf = w_up[j].to(torch.bfloat16)
                    mid = swiglu_bailing(F.linear(xe, wg_bf), F.linear(xe, wu_bf), exp_limit).float()
                    d_amax_mid[j] = mid.abs().amax().clamp_min(1e-4)
                    H_mid[j] = mid.T @ mid

                # Shared weight_scale_2 for gate+up (required by b12x W13 fusion), 1.5x Four-Over-Six headroom.
                gu_w_amax = torch.maximum(w_gate.abs().amax(dim=(1, 2)), w_up.abs().amax(dim=(1, 2)))
                d_w_amax = w_down.abs().amax(dim=(1, 2))
                gu_s2_b = torch.tensor(
                    [compute_weight_scale_2(float(v), headroom=1.5) for v in gu_w_amax.tolist()],
                    dtype=torch.float32,
                    device=device,
                )
                d_s2_b = torch.tensor(
                    [compute_weight_scale_2(float(v), headroom=1.5) for v in d_w_amax.tolist()],
                    dtype=torch.float32,
                    device=device,
                )

                # Quantize gate + up together by stacking along N ([bsz_e, 2*moe_inter, H]) so Cholesky runs ONCE!
                w_gu = torch.cat((w_gate, w_up), dim=1)
                gu_p_b, gu_s_b, gu_gptq_ok = quantize_nvfp4_gptq_batched(
                    w_gu, gu_s2_b, H_x, cnt_b, min_gptq_count=args.min_gptq_count
                )
                d_p_b, d_s_b, d_gptq_ok = quantize_nvfp4_gptq_batched(
                    w_down, d_s2_b, H_mid, cnt_b, min_gptq_count=args.min_gptq_count
                )
                n_ok = int((gu_gptq_ok & d_gptq_ok).sum().item())
                layer_gptq += n_ok
                layer_mse += bsz_e - n_ok

                gp_b = gu_p_b[:, :moe_inter]
                gs_b = gu_s_b[:, :moe_inter]
                up_b = gu_p_b[:, moe_inter:]
                us_b = gu_s_b[:, moe_inter:]

                # Save stacked outputs to CPU.
                g_pack[e0:e1] = gp_b.cpu()
                g_sc[e0:e1] = gs_b.cpu()
                u_pack[e0:e1] = up_b.cpu()
                u_sc[e0:e1] = us_b.cpu()
                d_pack[e0:e1] = d_p_b.cpu()
                d_sc[e0:e1] = d_s_b.cpu()
                gu_gs2[e0:e1] = gu_s2_b.cpu()
                d_gs2[e0:e1] = d_s2_b.cpu()
                gu_in_sc[e0:e1] = (gu_amax_x / (FP8_E4M3_MAX * FP4_E2M1_MAX)).clamp_min(1e-8).cpu()
                d_in_sc[e0:e1] = (d_amax_mid / (FP8_E4M3_MAX * FP4_E2M1_MAX)).clamp_min(1e-8).cpu()

                # Propagate quantized experts onto calib_moe_out AND lockstep ref/quant onto verify!
                for j, e in enumerate(range(e0, e1)):
                    wg_q = dequantize_nvfp4(gp_b[j], gs_b[j], gu_s2_b[j]).to(torch.bfloat16)
                    wu_q = dequantize_nvfp4(up_b[j], us_b[j], gu_s2_b[j]).to(torch.bfloat16)
                    wd_q = dequantize_nvfp4(d_p_b[j], d_s_b[j], d_s2_b[j]).to(torch.bfloat16)

                    s0, s1 = int(offsets[e].item()), int(offsets[e + 1].item())
                    if s1 > s0:
                        tok_idx = sorted_toks[s0:s1].long()
                        wt_e = sorted_wts[s0:s1].to(device).unsqueeze(-1)
                        xe = flat_x[tok_idx].to(device)
                        mid_q = swiglu_bailing(F.linear(xe, wg_q), F.linear(xe, wu_q), exp_limit)
                        out_q = F.linear(mid_q, wd_q).float() * wt_e
                        calib_moe_out.index_add_(0, tok_idx, out_q.cpu())

                    # Verification ref
                    mr = vr_ti == e
                    if mr.any():
                        t_idx, k_idx = mr.nonzero(as_tuple=True)
                        xr = vr_flat[t_idx]
                        wr = vr_tw[t_idx, k_idx].unsqueeze(-1)
                        mid_r = swiglu_bailing(
                            F.linear(xr, w_gate[j].to(torch.bfloat16)),
                            F.linear(xr, w_up[j].to(torch.bfloat16)),
                            exp_limit,
                        )
                        vr_moe_out.index_add_(0, t_idx, F.linear(mid_r, w_down[j].to(torch.bfloat16)).float() * wr)
                    # Verification quant
                    mq = vq_ti == e
                    if mq.any():
                        t_idx, k_idx = mq.nonzero(as_tuple=True)
                        xq = vq_flat[t_idx]
                        wq = vq_tw[t_idx, k_idx].unsqueeze(-1)
                        mid_qv = swiglu_bailing(F.linear(xq, wg_q), F.linear(xq, wu_q), exp_limit)
                        vq_moe_out.index_add_(0, t_idx, F.linear(mid_qv, wd_q).float() * wq)

                del w_gate, w_up, w_down, w_gu, H_x, H_mid, gu_p_b, gu_s_b, d_p_b, d_s_b

            total_gptq_experts += layer_gptq
            total_mse_experts += layer_mse

            # Write stacked 3D NVFP4 expert tensors.
            ep_base = f"{prefix}.mlp.experts"
            for pname, p_t, s_t, s2_t, in_t in (
                ("gate_proj", g_pack, g_sc, gu_gs2, gu_in_sc),
                ("up_proj", u_pack, u_sc, gu_gs2, gu_in_sc),
                ("down_proj", d_pack, d_sc, d_gs2, d_in_sc),
            ):
                full_p = f"{ep_base}.{pname}"
                writer.add(f"{full_p}.weight", p_t)
                writer.add(f"{full_p}.weight_scale", s_t)
                writer.add(f"{full_p}.weight_scale_2", s2_t)
                writer.add(f"{full_p}.input_scale", in_t)
                quantized_layers[full_p] = {"quant_algo": "NVFP4", "group_size": 16}

            # Add MoE + shared residual to calib_h and verify streams.
            calib_h = (calib_h.view(n_tok, hidden_size).float() + calib_moe_out).to(torch.bfloat16).view_as(calib_h)
            vr_sh = F.linear(swiglu_bailing(F.linear(vr_mlp_in, sh_ref["gate_proj"]), F.linear(vr_mlp_in, sh_ref["up_proj"]), sh_limit), sh_ref["down_proj"])
            vq_sh = F.linear(swiglu_bailing(F.linear(vq_mlp_in, sh_q["gate_proj"]), F.linear(vq_mlp_in, sh_q["up_proj"]), sh_limit), sh_q["down_proj"])
            ver_h_ref = ver_h_ref + vr_moe_out.view_as(ver_h_ref).to(torch.bfloat16) + vr_sh
            ver_h_q = ver_h_q + vq_moe_out.view_as(ver_h_q).to(torch.bfloat16) + vq_sh

            min_cnt = int(counts.min().item())
            med_cnt = int(counts.median().item())
            cos_sim = F.cosine_similarity(ver_h_ref.float().flatten(0, 1), ver_h_q.float().flatten(0, 1), dim=-1).mean().item()
            print(
                f"[layer {layer_idx:2d}/{total_layers-1}] {kind.upper()} + MoE -> NVFP4 "
                f"(gptq={layer_gptq}, mse={layer_mse}, tok/exp min={min_cnt} med={med_cnt}, "
                f"ver_cos={cos_sim:.5f}, lim=({exp_limit},{sh_limit})) in {time.monotonic()-t_layer:.1f}s",
                flush=True,
            )
            del g_pack, g_sc, u_pack, u_sc, d_pack, d_sc, sh_ref, sh_q, calib_moe_out

        if layer_idx == num_layers - 1:
            # Always write model.norm and lm_head and run target verification independently of num_mtp.
            final_norm_w = reader.get_tensor("model.norm.weight", device=device).to(torch.bfloat16)
            writer.add("model.norm.weight", final_norm_w)
            lm_head_w = reader.get_tensor("lm_head.weight", device=device).to(torch.bfloat16)
            writer.add("lm_head.weight", lm_head_w)

            ver_h_ref = rms_norm_t5(ver_h_ref, final_norm_w, eps=eps)
            ver_h_q = rms_norm_t5(ver_h_q, final_norm_w, eps=eps)
            target_metrics = _eval_logits_metrics(ver_h_ref, ver_h_q, lm_head_w, verify_ids.to(device))
            print(
                f"\n[verify:target-{num_layers}L] positions={target_metrics['scored_positions']} | "
                f"KL={target_metrics['kl']:.5f} | top1={target_metrics['top1']*100:.2f}% | "
                f"reference_top1_in_quantized_top5={target_metrics['reference_top1_in_quantized_top5']*100:.2f}% | "
                f"PPL_ref={target_metrics['ppl_ref']:.3f} vs PPL_q={target_metrics['ppl_q']:.3f} "
                f"({target_metrics['ppl_q']/target_metrics['ppl_ref']:.4f}x)\n",
                flush=True,
            )

        if is_mtp:
            assert lm_head_w is not None
            ver_mtp_ref = rms_norm_t5(ver_h_ref, final_mtp_norm, eps=eps)
            ver_mtp_q = rms_norm_t5(ver_h_q, final_mtp_norm, eps=eps)
            # MTP predicts token t+2 from hidden at position t (ignoring last 2 positions).
            mtp_targets = torch.roll(verify_ids.to(device), shifts=-1, dims=-1)
            mtp_metrics = _eval_logits_metrics(
                ver_mtp_ref[:, :-1], ver_mtp_q[:, :-1], lm_head_w, mtp_targets
            )
            print(
                f"\n[verify:MTP-L{layer_idx}] positions={mtp_metrics['scored_positions']} | "
                f"KL={mtp_metrics['kl']:.5f} | top1={mtp_metrics['top1']*100:.2f}% | "
                f"reference_top1_in_quantized_top5={mtp_metrics['reference_top1_in_quantized_top5']*100:.2f}%\n",
                flush=True,
            )

        # Evict finished source shards to keep peak disk minimal.
        keep = {s for s, last_l in shard_last.items() if last_l > layer_idx}
        reader.release_shards_except(keep)
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Copy remaining vision weights (model.visual.*, linear_proj.*) verbatim.
    vision_keys = sorted(k for k in reader.keys() if k.startswith(("model.visual.", "linear_proj.")))
    if vision_keys:
        t_vis = time.monotonic()
        for k in vision_keys:
            writer.add(k, reader.get_tensor(k, device="cpu"))
        print(f"[vision] Preserved {len(vision_keys)} vision tensors verbatim in {time.monotonic()-t_vis:.1f}s", flush=True)

    # 4. Stage config.json, conversion metadata, and non-weight assets before publishing index.json last.
    config["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": dict(sorted(quantized_layers.items())),
    }
    cfg_text = json.dumps(config, indent=2) + "\n"
    quant_meta = json.dumps(
        {
            "source": args.source,
            "revision": getattr(reader, "revision", args.revision),
            "corpus": corpus_metadata,
            "gptq_experts": total_gptq_experts,
            "mse_experts": total_mse_experts,
            "target_metrics": target_metrics,
            "mtp_metrics": mtp_metrics,
        },
        indent=2,
    ) + "\n"
    asset_sources = collect_non_weight_assets(reader.root)
    index = writer.publish_checkpoint(
        config_text=cfg_text,
        quant_meta_text=quant_meta,
        asset_sources=asset_sources,
    )

    dt = time.monotonic() - t_start
    print(
        f"[done] Wrote {output_dir} ({index['metadata']['total_size'] / (1 << 30):.2f} GiB across "
        f"{len(set(index['weight_map'].values()))} shards, {len(index['weight_map'])} tensors) "
        f"in {dt/60:.1f} min (gptq_experts={total_gptq_experts}, mse_experts={total_mse_experts})",
        flush=True,
    )


@torch.no_grad()
def _eval_logits_metrics(
    h_ref: torch.Tensor,
    h_q: torch.Tensor,
    lm_head: torch.Tensor,
    input_ids: torch.Tensor,
) -> dict[str, float]:
    """Compute exact next-token KL(P_ref || P_q), top-1/top-5 agreement, and PPL."""
    # Next-token prediction: positions :-1 predict input_ids[:, 1:]
    bsz, seqlen, _ = h_ref.shape
    targets = input_ids[:, 1 : seqlen].reshape(-1)
    n_eval = targets.numel()
    kl_sum = 0.0
    nll_ref_sum = 0.0
    nll_q_sum = 0.0
    top1_ok = 0
    top5_ok = 0
    flat_ref = h_ref[:, :-1].reshape(-1, h_ref.shape[-1])
    flat_q = h_q[:, :-1].reshape(-1, h_q.shape[-1])
    for i0 in range(0, n_eval, 256):
        i1 = min(i0 + 256, n_eval)
        lr = F.linear(flat_ref[i0:i1], lm_head).float()
        lq = F.linear(flat_q[i0:i1], lm_head).float()
        log_pr = F.log_softmax(lr, dim=-1)
        log_pq = F.log_softmax(lq, dim=-1)
        pr = log_pr.exp()
        kl_sum += float((pr * (log_pr - log_pq)).sum().item())
        tgt = targets[i0:i1]
        nll_ref_sum += float(F.nll_loss(log_pr, tgt, reduction="sum").item())
        nll_q_sum += float(F.nll_loss(log_pq, tgt, reduction="sum").item())
        top1_r = lr.argmax(dim=-1)
        top1_q = lq.argmax(dim=-1)
        top1_ok += int((top1_r == top1_q).sum().item())
        top5_q = lq.topk(5, dim=-1).indices
        top5_ok += int((top5_q == top1_r.unsqueeze(-1)).any(dim=-1).sum().item())
    return {
        "scored_positions": int(n_eval),
        "kl": kl_sum / n_eval,
        "top1": top1_ok / n_eval,
        "reference_top1_in_quantized_top5": top5_ok / n_eval,
        "ppl_ref": math.exp(nll_ref_sum / n_eval),
        "ppl_q": math.exp(nll_q_sum / n_eval),
    }


if __name__ == "__main__":
    main()
