# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for scripts/quantize_ling_3_2_air_nvfp4.py safety, atomicity, and quantization accounting."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


def _load_script():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts/quantize_ling_3_2_air_nvfp4.py"
    spec = importlib.util.spec_from_file_location("quantize_ling_3_2_air_nvfp4", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_output_paths_rejects_aliases_symlinks_and_nonempty_directories(tmp_path):
    q = _load_script()
    src = tmp_path / "src"
    src.mkdir()
    src_cfg = src / "config.json"
    src_idx = src / "model.safetensors.index.json"
    cfg_bytes = b'{"model_type":"bailing_moe_v3_vl"}\n'
    idx_bytes = b'{"weight_map":{}}\n'
    src_cfg.write_bytes(cfg_bytes)
    src_idx.write_bytes(idx_bytes)

    alias = tmp_path / "src_link"
    alias.symlink_to(src)
    fresh_out = tmp_path / "out"
    fresh_sync = tmp_path / "sync"

    with pytest.raises(ValueError, match="--output"):
        q.validate_output_paths(src, src)
    with pytest.raises(ValueError, match="--output"):
        q.validate_output_paths(src, alias)
    with pytest.raises(ValueError, match="--sync-dir"):
        q.validate_output_paths(src, fresh_out, sync_dir=alias)
    with pytest.raises(ValueError, match="--sync-dir"):
        q.validate_output_paths(src, fresh_out, sync_dir=fresh_out)

    # Rejects both checkpoint directories and directories with unrelated pre-existing files.
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "notes.txt").write_text("do not overwrite")
    with pytest.raises(FileExistsError, match="not empty"):
        q.validate_output_paths(src, existing)

    q.validate_output_paths(src, fresh_out, sync_dir=fresh_sync, work_dir=tmp_path / "work")
    assert src_cfg.read_bytes() == cfg_bytes
    assert src_idx.read_bytes() == idx_bytes


def test_sliding_hf_reader_isolates_revisions_and_concurrent_shard_dirs(tmp_path, monkeypatch):
    q = _load_script()
    sha_a = "a" * 40
    sha_b = "b" * 40

    def _payloads(rev: str):
        idx = json.dumps({"weight_map": {"w": f"{rev[:4]}.safetensors"}}).encode()
        cfg = json.dumps({"revision_marker": rev}).encode()
        return idx, cfg

    def fake_http(self, url, *, byte_range=None, retries=6, timeout=120.0):
        idx, cfg = _payloads(self.revision)
        if "/api/models/" in url and "/tree/" in url:
            return json.dumps(
                [
                    {"type": "file", "path": "model.safetensors.index.json", "size": len(idx)},
                    {"type": "file", "path": "config.json", "size": len(cfg)},
                ]
            ).encode()
        if url.endswith("model.safetensors.index.json"):
            return idx
        if url.endswith("config.json"):
            return cfg
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(q.SlidingHfShardReader, "_http_get", fake_http)
    work = tmp_path / "work"
    r1 = q.SlidingHfShardReader("org/model", sha_a, work_dir=work, token="")
    r2 = q.SlidingHfShardReader("org/model", sha_b, work_dir=work, token="")
    r1_concurrent = q.SlidingHfShardReader("org/model", sha_a, work_dir=work, token="")

    try:
        assert r1.root != r2.root
        assert json.loads((r1.root / "config.json").read_text())["revision_marker"] == sha_a
        assert json.loads((r2.root / "config.json").read_text())["revision_marker"] == sha_b
        assert r1.weight_map["w"] != r2.weight_map["w"]

        assert r1.shard_dir != r1_concurrent.shard_dir
        marker = r1_concurrent.shard_dir / "keep.bin"
        marker.write_bytes(b"live")
        r1.close()
        assert marker.is_file()
        assert not r1.shard_dir.exists()
    finally:
        r2.close()
        r1_concurrent.close()


def test_metadata_download_is_atomic_and_rejects_interrupted_or_truncated_writes(tmp_path, monkeypatch):
    q = _load_script()
    sha = "c" * 40
    idx = json.dumps({"weight_map": {"w": "s.safetensors"}}).encode()
    cfg = json.dumps({"ok": True}).encode()
    mode = {"fail_cfg": "truncated"}

    def fake_http(self, url, *, byte_range=None, retries=6, timeout=120.0):
        if "/api/models/" in url and "/tree/" in url:
            return json.dumps(
                [
                    {"type": "file", "path": "model.safetensors.index.json", "size": len(idx)},
                    {"type": "file", "path": "config.json", "size": len(cfg)},
                ]
            ).encode()
        if url.endswith("model.safetensors.index.json"):
            return idx
        if url.endswith("config.json"):
            if mode["fail_cfg"] == "truncated":
                return cfg[:3]
            if mode["fail_cfg"] == "boom":
                raise RuntimeError("interrupted connection")
            return cfg
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(q.SlidingHfShardReader, "_http_get", fake_http)
    work = tmp_path / "work"

    # Truncated metadata download must raise and leave no truncated config.json or leftover .part file.
    with pytest.raises(RuntimeError, match="Metadata size mismatch"):
        q.SlidingHfShardReader("org/model", sha, work_dir=work, token="")
    meta_dirs = list((work / "meta").iterdir())
    assert len(meta_dirs) == 1
    assert not (meta_dirs[0] / "config.json").exists()
    assert not list(meta_dirs[0].glob("*.part"))

    # Interrupted download also leaves no partial file, and a clean retry succeeds atomically.
    mode["fail_cfg"] = "boom"
    with pytest.raises(RuntimeError, match="interrupted connection"):
        q.SlidingHfShardReader("org/model", sha, work_dir=work, token="")
    assert not (meta_dirs[0] / "config.json").exists()

    mode["fail_cfg"] = "ok"
    reader = q.SlidingHfShardReader("org/model", sha, work_dir=work, token="")
    try:
        assert (reader.root / "config.json").read_bytes() == cfg
        assert not list(reader.root.glob("*.part"))
    finally:
        reader.close()


def test_competing_writers_on_same_destination_are_rejected(tmp_path):
    q = _load_script()
    out = tmp_path / "out"
    sync = tmp_path / "sync"
    w1 = q.ShardedSafetensorsWriter(out, sync_dir=sync)
    try:
        with pytest.raises(FileExistsError, match="already reserved"):
            q.ShardedSafetensorsWriter(out)
        with pytest.raises(FileExistsError, match="already reserved"):
            q.ShardedSafetensorsWriter(tmp_path / "other_out", sync_dir=sync)
    finally:
        w1.abort()
    # After abort, locks and staging dirs are cleaned up so a fresh writer succeeds.
    w2 = q.ShardedSafetensorsWriter(out, sync_dir=sync)
    w2.add("w", torch.zeros((4, 4), dtype=torch.bfloat16))
    w2.publish_checkpoint(config_text='{"ok":true}\n', quant_meta_text='{"ok":true}\n')
    assert (out / "model.safetensors.index.json").is_file()
    assert (sync / "model.safetensors.index.json").is_file()


def test_injected_asset_publication_failure_leaves_no_published_index_and_allows_retry(tmp_path):
    q = _load_script()
    out = tmp_path / "out"
    sync = tmp_path / "sync"
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    tok = asset_dir / "tokenizer.json"
    tok.write_text('{"version":"1.0"}\n')

    w1 = q.ShardedSafetensorsWriter(out, sync_dir=sync)
    w1.add("w", torch.zeros((4, 4), dtype=torch.bfloat16))

    def _boom(src, dst):
        raise OSError("simulated disk full during tokenizer copy")

    with pytest.raises(OSError, match="simulated disk full"):
        w1.publish_checkpoint(
            config_text='{"ok":true}\n',
            quant_meta_text='{"ok":true}\n',
            asset_sources=[tok],
            copy_fn=_boom,
        )
    # Neither destination has a published index, partial shard, or stale lock/staging dir.
    assert list(out.iterdir()) == []
    assert list(sync.iterdir()) == []

    # Clean retry publishes complete checkpoint (shards + config + metadata + tokenizer + index) in both dirs.
    w2 = q.ShardedSafetensorsWriter(out, sync_dir=sync)
    w2.add("w", torch.zeros((4, 4), dtype=torch.bfloat16))
    idx = w2.publish_checkpoint(
        config_text='{"ok":true}\n',
        quant_meta_text='{"ok":true}\n',
        asset_sources=[tok],
    )
    for target in (out, sync):
        assert (target / "model.safetensors.index.json").is_file()
        assert (target / "config.json").is_file()
        assert (target / "quantization_metadata.json").is_file()
        assert (target / "tokenizer.json").read_text() == '{"version":"1.0"}\n'
        assert (target / idx["weight_map"]["w"]).is_file()
        assert not (target / ".quantize.lock").exists()


def test_disjoint_token_split_never_repeats_and_rejects_short_corpus():
    q = _load_script()
    records = [f"rec{i}" for i in range(10)]
    encode = lambda s: [int(s[3:]) * 100 + k for k in range(8)]

    calib, verify, meta = q.split_disjoint_token_streams(
        encode, records, num_samples=4, verify_seqs=2, seq_len=8
    )
    assert calib.shape == (4, 8)
    assert verify.shape == (2, 8)
    assert set(calib.flatten().tolist()).isdisjoint(set(verify.flatten().tolist()))
    assert meta["calib_tokens"] == 32
    assert meta["verify_tokens"] == 16

    with pytest.raises(ValueError, match="without repeating records"):
        q.split_disjoint_token_streams(encode, records[:3], num_samples=4, verify_seqs=2, seq_len=8)


def test_eval_logits_metrics_reports_scored_positions_and_top1_in_top5():
    q = _load_script()
    torch.manual_seed(0)
    h_ref = torch.randn(4, 10, 16)
    h_q = h_ref.clone()
    lm_head = torch.randn(32, 16)
    ids = torch.randint(0, 32, (4, 10))
    metrics = q._eval_logits_metrics(h_ref, h_q, lm_head, ids)
    assert metrics["scored_positions"] == 4 * 9
    assert metrics["top1"] == 1.0
    assert metrics["reference_top1_in_quantized_top5"] == 1.0
    assert "top5" not in metrics


def test_zero_token_expert_raises():
    q = _load_script()
    q.assert_nonzero_expert_counts(torch.tensor([12, 5, 1]), "model.layers.2")
    with pytest.raises(RuntimeError, match="model.layers.2.mlp.experts.1 received 0 calibration tokens"):
        q.assert_nonzero_expert_counts(torch.tensor([12, 0, 5]), "model.layers.2")


def test_gptq_batched_reports_cholesky_vs_rtn_fallback_counts():
    q = _load_script()
    torch.manual_seed(42)
    bsz, n, k = 3, 16, 32
    W = torch.randn(bsz, n, k, dtype=torch.float32) * 0.5
    gs2 = torch.tensor(
        [q.compute_weight_scale_2(float(W[i].abs().amax().item()), headroom=1.5) for i in range(bsz)],
        dtype=torch.float32,
    )
    # Expert 0: well-conditioned Hessian with sufficient count -> Cholesky GPTQ.
    # Expert 1: count below min_gptq_count -> RTN/MSE fallback.
    # Expert 2: zero Hessian -> RTN/MSE fallback despite high count.
    X = torch.randn(64, k, dtype=torch.float32)
    H = torch.zeros(bsz, k, k, dtype=torch.float32)
    H[0] = X.T @ X
    H[1] = X[:2].T @ X[:2]
    H[2].zero_()
    counts = torch.tensor([64, 2, 64], dtype=torch.int64)

    packed, scales, gptq_mask = q.quantize_nvfp4_gptq_batched(
        W, gs2, H, counts, min_gptq_count=8, mse_factors=(1.0, 1.25)
    )
    assert packed.shape == (bsz, n, k // 2)
    assert scales.shape == (bsz, n, k // 16)
    assert gptq_mask.tolist() == [True, False, False]
    assert int(gptq_mask.sum().item()) == 1
    assert bsz - int(gptq_mask.sum().item()) == 2


def test_pre_publication_failure_after_flush_releases_locks_and_allows_retry(tmp_path, monkeypatch):
    q = _load_script()
    out = tmp_path / "out"
    sync = tmp_path / "sync"

    # A failure after flushing a shard (before publish_checkpoint) must release both locks and
    # remove staging directories so an immediate retry on the same destinations succeeds.
    with pytest.raises(RuntimeError, match="zero-token expert"):
        with q.ShardedSafetensorsWriter(out, max_shard_bytes=1, sync_dir=sync) as w1:
            w1.add("layer0.weight", torch.ones((4, 4), dtype=torch.bfloat16))
            w1.flush()
            raise RuntimeError("zero-token expert")

    assert list(out.iterdir()) == []
    assert list(sync.iterdir()) == []

    with q.ShardedSafetensorsWriter(out, sync_dir=sync) as w2:
        w2.add("layer0.weight", torch.ones((4, 4), dtype=torch.bfloat16))
        idx = w2.publish_checkpoint(config_text='{"ok":true}\n', quant_meta_text='{"run":2}\n')
    for target in (out, sync):
        assert (target / "model.safetensors.index.json").is_file()
        assert (target / idx["weight_map"]["layer0.weight"]).is_file()
        assert not (target / ".quantize.lock").exists()

    # If staging-directory creation itself fails inside _reserve_empty_destination, release the lock.
    blocked = tmp_path / "mkdtemp_fail"
    monkeypatch.setattr(q.tempfile, "mkdtemp", lambda **kwargs: (_ for _ in ()).throw(OSError("no space")))
    with pytest.raises(OSError, match="no space"):
        q._reserve_empty_destination(blocked)
    assert not (blocked / ".quantize.lock").exists()
    monkeypatch.undo()

    # If PID write fails right after O_EXCL acquisition, close fd, release .quantize.lock, and allow immediate retry.
    pid_fail = tmp_path / "pid_write_fail"
    real_write = q.os.write
    monkeypatch.setattr(q.os, "write", lambda fd, data: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        q._reserve_empty_destination(pid_fail)
    assert not (pid_fail / ".quantize.lock").exists()
    monkeypatch.setattr(q.os, "write", real_write)
    lock_path, stage_dir = q._reserve_empty_destination(pid_fail)
    assert lock_path.is_file() and stage_dir.is_dir()
    stage_dir.rmdir()
    lock_path.unlink()


def test_source_with_prior_quantization_metadata_publishes_current_metrics(tmp_path):
    q = _load_script()
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text('{"model_type":"bailing_moe_v3_vl"}\n')
    (src / "model.safetensors.index.json").write_text('{"weight_map":{}}\n')
    (src / "model-00001-of-00001.safetensors").write_bytes(b"placeholder")
    (src / "tokenizer.json").write_text('{"version":"1.0"}\n')
    prior_meta = src / "quantization_metadata.json"
    prior_meta.write_text('{"source":"stale-prior-run","target_metrics":{"kl":9.99}}\n')

    assets = q.collect_non_weight_assets(src)
    assert [p.name for p in assets] == ["tokenizer.json"]

    out = tmp_path / "out"
    sync = tmp_path / "sync"
    current_meta = '{"source":"current-run","target_metrics":{"kl":0.01}}\n'
    with q.ShardedSafetensorsWriter(out, sync_dir=sync) as w:
        w.add("w", torch.zeros((4, 4), dtype=torch.bfloat16))
        w.publish_checkpoint(
            config_text='{"ok":true}\n',
            quant_meta_text=current_meta,
            asset_sources=assets,
        )
    for target in (out, sync):
        assert (target / "quantization_metadata.json").read_text() == current_meta
        assert (target / "tokenizer.json").read_text() == '{"version":"1.0"}\n'
        assert (target / "model.safetensors.index.json").is_file()

    # Passing a reserved filename directly to publish_checkpoint must fail and clean up locks.
    out_bad = tmp_path / "out_bad"
    with pytest.raises(ValueError, match="collides with generated checkpoint file"):
        with q.ShardedSafetensorsWriter(out_bad) as w_bad:
            w_bad.add("w", torch.zeros((4, 4), dtype=torch.bfloat16))
            w_bad.publish_checkpoint(
                config_text='{"ok":true}\n',
                quant_meta_text=current_meta,
                asset_sources=[prior_meta],
            )
    assert list(out_bad.iterdir()) == []
