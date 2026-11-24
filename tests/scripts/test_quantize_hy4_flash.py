# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU safety, HTTP Range validation, NVFP4 Engram parity, and atomicity checks for scripts/quantize_hy4_flash_nvfp4.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts/quantize_hy4_flash_nvfp4.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("quantize_hy4_flash_nvfp4", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_sync_paths_rejects_same_symlink_and_nested_dirs_and_move_publishes_only_in_sync(tmp_path):
    mod = _load_module()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    alias = tmp_path / "out_link"
    alias.symlink_to(out_dir)

    with pytest.raises(ValueError, match="same directory"):
        mod.ShardedSafetensorsWriter(out_dir, sync_dir=out_dir)
    with pytest.raises(ValueError, match="same directory"):
        mod.ShardedSafetensorsWriter(out_dir, sync_dir=alias, move_to_sync_dir=True)
    with pytest.raises(ValueError, match="nest"):
        mod.ShardedSafetensorsWriter(out_dir, sync_dir=out_dir / "nested")
    with pytest.raises(ValueError, match="nest"):
        mod.ShardedSafetensorsWriter(out_dir, sync_dir=tmp_path)

    other = tmp_path / "sync"
    writer = mod.ShardedSafetensorsWriter(out_dir, sync_dir=other, move_to_sync_dir=True)
    writer.add("w", torch.zeros((4, 4), dtype=torch.bfloat16))
    index = writer.finalize()
    shard = index["weight_map"]["w"]
    assert (other / shard).is_file()
    assert (other / "model.safetensors.index.json").is_file()
    assert not (out_dir / shard).exists()
    # Move mode must not leave an unusable local index pointing at unlinked shards.
    assert not (out_dir / "model.safetensors.index.json").exists()


def test_http_get_bytes_validates_status_content_range_bounds_and_eof():
    mod = _load_module()
    reader = object.__new__(mod.HfRangeSafetensorsReader)
    reader._headers = {}

    class _Resp:
        def __init__(self, status: int, headers: dict[str, str], chunks: list[bytes]):
            self.status_code = status
            self.headers = headers
            self._chunks = chunks
            self.iter_called = False
            self.closed = False

        def iter_content(self, chunk_size=65536):
            self.iter_called = True
            yield from self._chunks

        def close(self):
            self.closed = True

    def _set_resp(resp):
        class _Sess:
            def get(self, *args, **kwargs):
                assert kwargs.get("stream") is True
                return resp

        reader._get_session = lambda: _Sess()

    # 1. Server ignores Range and replies HTTP 200 -> rejected before reading body.
    r200 = _Resp(200, {}, [b"giant-body"])
    _set_resp(r200)
    with pytest.raises(RuntimeError, match="Expected HTTP 206"):
        reader._http_get_bytes("http://example.invalid/s", byte_range=(0, 9), retries=1)
    assert not r200.iter_called and r200.closed

    # 2. Same-length 206 response from wrong offset -> rejected.
    r_wrong = _Resp(206, {"Content-Range": "bytes 10-19/100"}, [b"0123456789"])
    _set_resp(r_wrong)
    with pytest.raises(RuntimeError, match="Mismatched Content-Range"):
        reader._http_get_bytes("http://example.invalid/s", byte_range=(0, 9), retries=1)
    assert not r_wrong.iter_called and r_wrong.closed

    # 3. Short 206 header probe without EOF (c_end + 1 < total) -> rejected even with allow_short=True.
    r_trunc = _Resp(206, {"Content-Range": "bytes 0-2/100"}, [b"abc"])
    _set_resp(r_trunc)
    with pytest.raises(RuntimeError, match="Truncated Content-Range"):
        reader._http_get_bytes("http://example.invalid/s", byte_range=(0, 65535), allow_short=True, retries=1)

    # 4. Short 206 header probe proving EOF (c_end + 1 == total) -> accepted with allow_short=True.
    r_eof = _Resp(206, {"Content-Range": "bytes 0-2/3"}, [b"abc"])
    _set_resp(r_eof)
    assert reader._http_get_bytes("http://example.invalid/s", byte_range=(0, 65535), allow_short=True, retries=1) == b"abc"
    assert r_eof.closed

    # 5. Exact ranged payload with matching Content-Range -> accepted; truncated body rejected.
    r_ok = _Resp(206, {"Content-Range": "bytes 4-7/100"}, [b"wx", b"yz"])
    _set_resp(r_ok)
    assert reader._http_get_bytes("http://example.invalid/s", byte_range=(4, 7), retries=1) == b"wxyz"
    r_short_body = _Resp(206, {"Content-Range": "bytes 4-7/100"}, [b"wx"])
    _set_resp(r_short_body)
    with pytest.raises(RuntimeError, match="Short read"):
        reader._http_get_bytes("http://example.invalid/s", byte_range=(4, 7), retries=1)


def test_inline_nvfp4_engram_roundtrip_matches_written_shard(tmp_path):
    """Verify inline NVFP4 Engram round-trip matches reopening and dequantizing the written NVFP4 shard."""
    mod = _load_module()
    torch.manual_seed(7)
    src_dir = tmp_path / "src"
    out_dir = tmp_path / "out"
    src_dir.mkdir()
    table = (torch.randn((32, 96), dtype=torch.bfloat16) * 1.5).contiguous()
    src_key = "model.layers.1.engram_layer.multi_head_embedding.embedding.weight"
    save_file({src_key: table}, str(src_dir / "src.safetensors"))
    (src_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": table.numel() * 2}, "weight_map": {src_key: "src.safetensors"}})
    )

    reader = mod.ShardedSafetensorsReader(src_dir)
    writer = mod.ShardedSafetensorsWriter(out_dir)
    _, _, gs = mod.stream_quantize_engram_table_nvfp4(reader, writer, 1, torch.device("cpu"))
    writer.finalize()
    reader.close()

    row_ids = torch.tensor([[[0, 5, 17, 31], [3, 12, 20, 29]]], dtype=torch.int64)
    bf16_rows = table[row_ids]
    inline_nvfp4 = mod._round_trip_engram_rows(bf16_rows, engram_format="nvfp4", global_scale=gs)

    out_reader = mod.ShardedSafetensorsReader(out_dir)
    reopened_nvfp4 = mod._gather_engram_rows_from_file(
        out_reader,
        "model.layers.1.engram_layer.multi_head_embedding.weight",
        row_ids,
        scale_name="model.layers.1.engram_layer.multi_head_embedding.scale",
        global_scale_name="model.layers.1.engram_layer.multi_head_embedding.weight_scale_2",
    )
    out_reader.close()
    assert torch.equal(inline_nvfp4, reopened_nvfp4)
    # Confirm NVFP4 round-trip actually differs from MXFP8 round-trip on non-trivial rows.
    inline_fp8 = mod._round_trip_engram_rows(bf16_rows, engram_format="fp8")
    assert not torch.equal(inline_nvfp4, inline_fp8)


def test_repack_engram_nvfp4_is_atomic_on_mid_run_failure_and_recomputes_total_size(tmp_path, monkeypatch):
    mod = _load_module()
    src_dir = tmp_path / "src"
    quant_dir = tmp_path / "quant"
    src_dir.mkdir()
    quant_dir.mkdir()

    # Minimal source checkpoint for ShardedSafetensorsReader.
    save_file({"dummy": torch.zeros(4, dtype=torch.bfloat16)}, str(src_dir / "src.safetensors"))
    (src_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 8}, "weight_map": {"dummy": "src.safetensors"}})
    )

    # Quantized checkpoint with 1 backbone shard and 2 dedicated FP8 Engram shards (layers 1 and 11).
    save_file({"model.norm.weight": torch.ones(8, dtype=torch.bfloat16)}, str(quant_dir / "backbone.safetensors"))
    w1_key = "model.layers.1.engram_layer.multi_head_embedding.weight"
    s1_key = "model.layers.1.engram_layer.multi_head_embedding.scale"
    w11_key = "model.layers.11.engram_layer.multi_head_embedding.weight"
    s11_key = "model.layers.11.engram_layer.multi_head_embedding.scale"
    save_file(
        {w1_key: torch.zeros((4, 96), dtype=torch.uint8), s1_key: torch.zeros((4, 3), dtype=torch.uint8)},
        str(quant_dir / "engram1.safetensors"),
    )
    save_file(
        {w11_key: torch.zeros((4, 96), dtype=torch.uint8), s11_key: torch.zeros((4, 3), dtype=torch.uint8)},
        str(quant_dir / "engram11.safetensors"),
    )
    orig_index = {
        "metadata": {"total_size": 16 + 396 + 396},
        "weight_map": {
            "model.norm.weight": "backbone.safetensors",
            w1_key: "engram1.safetensors",
            s1_key: "engram1.safetensors",
            w11_key: "engram11.safetensors",
            s11_key: "engram11.safetensors",
        },
    }
    (quant_dir / "model.safetensors.index.json").write_text(json.dumps(orig_index))
    (quant_dir / "config.json").write_text(json.dumps({"text_config": {"engram_layer_ids": [2, 12]}}))

    def _write_fake_nvfp4(layer_idx: int, out_path: Path):
        w = torch.zeros((4, 48), dtype=torch.uint8)
        s = torch.zeros((4, 6), dtype=torch.uint8)
        gs = torch.tensor(1.25, dtype=torch.float32)
        save_file(
            {
                f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.weight": w,
                f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.scale": s,
                f"model.layers.{layer_idx}.engram_layer.multi_head_embedding.weight_scale_2": gs,
            },
            str(out_path),
        )
        return w.numel(), s.numel(), 1.25

    # 1. Fail on the second layer: old shards and original index must remain untouched, no orphan shards left.
    def _fail_second(reader, writer, layer_idx, device, shard_path_override=None):
        if layer_idx == 11:
            raise RuntimeError("simulated disk full on layer 11")
        return _write_fake_nvfp4(layer_idx, shard_path_override)

    monkeypatch.setattr(mod, "stream_quantize_engram_table_nvfp4", _fail_second)
    with pytest.raises(RuntimeError, match="simulated disk full"):
        mod.repack_engram_nvfp4(src_dir, quant_dir, device=torch.device("cpu"))
    assert json.loads((quant_dir / "model.safetensors.index.json").read_text()) == orig_index
    assert (quant_dir / "engram1.safetensors").is_file()
    assert (quant_dir / "engram11.safetensors").is_file()
    assert not list(quant_dir.glob("engram-nvfp4-*"))

    # 2. Retry succeeds: new index atomically swaps in fresh shards, old shards removed, exact total_size.
    monkeypatch.setattr(
        mod,
        "stream_quantize_engram_table_nvfp4",
        lambda reader, writer, layer_idx, device, shard_path_override=None: _write_fake_nvfp4(
            layer_idx, shard_path_override
        ),
    )
    new_index = mod.repack_engram_nvfp4(src_dir, quant_dir, device=torch.device("cpu"))
    assert not (quant_dir / "engram1.safetensors").exists()
    assert not (quant_dir / "engram11.safetensors").exists()
    assert new_index["metadata"]["total_size"] == 16 + 2 * (4 * 48 + 4 * 6 + 4)
    for key in (
        w1_key,
        s1_key,
        "model.layers.1.engram_layer.multi_head_embedding.weight_scale_2",
        w11_key,
        s11_key,
        "model.layers.11.engram_layer.multi_head_embedding.weight_scale_2",
    ):
        assert (quant_dir / new_index["weight_map"][key]).is_file()


def test_main_move_to_sync_dir_verify_uses_sync_dir(tmp_path, monkeypatch):
    """Verify --move-to-sync-dir --verify passes the authoritative sync_dir to verify_quality."""
    mod = _load_module()
    src = tmp_path / "src"
    out = tmp_path / "out"
    sync = tmp_path / "sync"
    src.mkdir()
    out.mkdir()
    sync.mkdir()
    verified_dirs: list[tuple[Path, Path]] = []

    monkeypatch.setattr(mod, "quantize_checkpoint", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        mod,
        "verify_quality",
        lambda bf16_dir, quant_dir, **kwargs: verified_dirs.append((bf16_dir, quant_dir)) or {},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quantize_hy4_flash_nvfp4.py",
            "--input",
            str(src),
            "--output",
            str(out),
            "--sync-dir",
            str(sync),
            "--move-to-sync-dir",
            "--verify",
            "--device",
            "cpu",
        ],
    )
    mod.main()
    assert verified_dirs == [(src, sync)]


def test_reuse_engram_rejects_nvfp4_and_shape_mismatch_and_uses_reused_shard_values(tmp_path):
    """Verify --reuse-engram-from rejects NVFP4 and mismatched shapes, and reads the actual reused FP8 rows."""
    mod = _load_module()
    torch.manual_seed(19)

    src_table = (torch.randn((32, 96), dtype=torch.bfloat16) * 0.5).contiguous()
    # Deliberately construct a reused FP8 Engram shard with completely distinct values from src_table.
    reused_source = (-src_table * 2.5 + 1.0).contiguous()
    reused_q, reused_s = mod.quantize_mxfp8_block32(reused_source.float())
    reused_deq = mod.dequantize_mxfp8(reused_q, reused_s).to(torch.bfloat16)

    reuse_dir = tmp_path / "reuse_fp8"
    reuse_dir.mkdir()
    w_key = "model.layers.1.engram_layer.multi_head_embedding.weight"
    s_key = "model.layers.1.engram_layer.multi_head_embedding.scale"
    save_file({w_key: reused_q, s_key: reused_s}, str(reuse_dir / "engram1.safetensors"))
    (reuse_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": reused_q.numel() + reused_s.numel()},
                "weight_map": {w_key: "engram1.safetensors", s_key: "engram1.safetensors"},
            }
        )
    )

    # 1. Incompatible --reuse-engram-from + --engram-format nvfp4 must be rejected explicitly.
    with pytest.raises(ValueError, match="only supports --engram-format fp8"):
        mod.quantize_checkpoint(
            tmp_path / "missing_src",
            tmp_path / "out_bad_fmt",
            device=torch.device("cpu"),
            engram_format="nvfp4",
            reuse_engram_from=reuse_dir,
        )

    reuse_reader = mod.ShardedSafetensorsReader(reuse_dir)
    try:
        # 2. Reused shard shape mismatch against source BF16 table must be rejected.
        out_bad_shape = tmp_path / "out_bad_shape"
        w_bad = mod.ShardedSafetensorsWriter(out_bad_shape)
        with pytest.raises(ValueError, match="shape mismatch"):
            mod._reuse_engram_table(reuse_reader, w_bad, 1, expected_shape=(16, 96))

        # 3. A reused shard containing an unrelated third tensor must be rejected before linking.
        extra_dir = tmp_path / "reuse_extra"
        extra_dir.mkdir()
        extra_t = torch.zeros((4, 4), dtype=torch.bfloat16)
        save_file(
            {w_key: reused_q, s_key: reused_s, "model.norm.weight": extra_t},
            str(extra_dir / "engram1_mixed.safetensors"),
        )
        (extra_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": reused_q.numel() + reused_s.numel() + extra_t.nbytes},
                    "weight_map": {
                        w_key: "engram1_mixed.safetensors",
                        s_key: "engram1_mixed.safetensors",
                        "model.norm.weight": "engram1_mixed.safetensors",
                    },
                }
            )
        )
        extra_reader = mod.ShardedSafetensorsReader(extra_dir)
        try:
            out_bad_extra = tmp_path / "out_bad_extra"
            w_extra = mod.ShardedSafetensorsWriter(out_bad_extra)
            with pytest.raises(ValueError, match="must contain only"):
                mod._reuse_engram_table(extra_reader, w_extra, 1, expected_shape=(32, 96))
        finally:
            extra_reader.close()

        # 4. Valid two-tensor shard links cleanly, passes Ignition's indexed-checkpoint validation,
        #    and calibration/inline-verify rows come from the reused shard itself.
        out_ok = tmp_path / "out_ok"
        w_ok = mod.ShardedSafetensorsWriter(out_ok)
        mod._reuse_engram_table(reuse_reader, w_ok, 1, expected_shape=(32, 96))
        w_ok.finalize()

        from ignition.preparation.checkpoint import inspect_indexed_checkpoint

        inspected = inspect_indexed_checkpoint(out_ok)
        assert set(inspected.tensors.keys()) == {w_key, s_key}

        row_ids = torch.tensor([[[0, 4, 15, 31], [2, 9, 21, 30]]], dtype=torch.int64)
        bf16_rows = src_table[row_ids]
        loaded = mod._load_quantized_engram_rows(
            layer_idx=1,
            row_ids=row_ids,
            bf16_rows=bf16_rows,
            engram_format="fp8",
            global_scale=None,
            reuse_reader=reuse_reader,
            device=torch.device("cpu"),
        )
        fresh_from_src = mod._round_trip_engram_rows(bf16_rows, engram_format="fp8")
        assert torch.equal(loaded, reused_deq[row_ids])
        assert not torch.equal(loaded, fresh_from_src)
    finally:
        reuse_reader.close()
