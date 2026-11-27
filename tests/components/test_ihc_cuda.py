# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires CUDA qualification")


@pytest.fixture(scope="module", autouse=True)
def _artifact_directory(tmp_path_factory):
    from ignition.preparation import compilation
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(compilation, "DEFAULT_CACHE_ROOT", tmp_path_factory.mktemp("ihc_artifacts"))
        yield


def _check_capture(function, inputs, *, standalone=False):
    from ignition.preparation.compilation import _compile_options, compile_forward

    if standalone:
        replay = compile_forward(lambda: function(*inputs))
    else:
        compiled = torch.compile(function, fullgraph=True, dynamic=False, options=_compile_options())
        def replay():
            return compiled(*inputs)
    replay()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = replay()
    original = inputs[0].clone()
    inputs[0].neg_()
    allocated = torch.cuda.memory_allocated()
    graph.replay()
    assert torch.cuda.memory_allocated() == allocated
    expected = function(*inputs)
    for actual, ref in zip(captured, expected, strict=True):
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
    inputs[0].copy_(original)


@pytest.mark.parametrize("tokens", [1, 4, 32, 48, 128, 257, 512, 4096])
def test_stream_boundaries(tokens):
    from b12x.preparation import PreparationSession
    from hyperflare_recipes.components.ihc import PreparedIHC, post_norm
    from hyperflare_recipes.components.rms_norm import PreparedRMSNorm
    from hyperflare_recipes.recipes.hy4_flash.ihc import ihc_head, ihc_post, ihc_pre

    rng = torch.Generator(device="cuda").manual_seed(529)
    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(shape, device="cuda", generator=rng, dtype=dtype)

    torch.compiler.reset()
    d, eps, hc_eps, magnitude = 3072, 1e-6, 1e-6, 2.
    s = rand(tokens, 4, d)
    a = rand(tokens, d + 8)[:, :d] if tokens == 257 else rand(tokens, d)
    b = rand(tokens, d)
    w, norm = rand(d), rand(d)
    prev = rand(tokens, 4, dtype=torch.float32)
    session = PreparationSession()
    def rms(weight, name):
        return PreparedRMSNorm(a, weight, torch.empty_like(a), eps=eps, session=session,
                               name=name).functional()

    entry, exit_norm = rms(w, "entry"), rms(norm, "exit")
    for gates in (8, 4):
        fn = rand(gates, 4 * d, dtype=torch.float32) * .01
        base, scale = rand(gates, dtype=torch.float32), rand(gates // 4, dtype=torch.float32)
        mix = torch.empty((tokens, gates), device="cuda", dtype=torch.float32)
        fused = PreparedIHC(s, mix, fn, base, scale, norm, eps=eps, hc_eps=hc_eps,
                            magnitude=magnitude, session=session, name=f"ihc{gates}")
        for branch, shared in ((None, None), (a, None), (a, b)):
            residual = s if branch is None else ihc_post(s, entry(a if shared is None else a + b), prev)
            if branch is not None:
                torch.testing.assert_close(post_norm(s, a, shared, prev, w, eps), residual, rtol=0, atol=0)
            mixes = fused.projection(residual.flatten(1))
            if gates == 4:
                collapsed = ihc_head(residual, fn, base, scale, rms_eps=eps, hc_eps=hc_eps, mixes=mixes)
                post = None
            else:
                post, collapsed = ihc_pre(residual, fn, base, scale, rms_eps=eps, hc_eps=hc_eps,
                                          hc_magnitude=magnitude, mixes=mixes)
            updated, actual_post, actual = fused(s, branch, shared, prev, w)
            torch.testing.assert_close(updated, residual, rtol=0, atol=0)
            torch.testing.assert_close(actual, exit_norm(collapsed), rtol=0, atol=0)
            if post is not None:
                torch.testing.assert_close(actual_post, post, rtol=0, atol=0)
            if shared is not None and gates == 8:
                _check_capture(lambda x, y, z, p: fused(x, y, z, p, w), (s, a, b, prev),
                               standalone=tokens in (4, 128, 4096))
            if gates == 4 and tokens in (4, 128) and shared is not None:
                _check_capture(lambda x, y, z, p: (fused(x, y, z, p, w)[2],), (s, a, b, prev))


@pytest.mark.parametrize("nvfp4", [False, True])
@pytest.mark.parametrize("tokens", [4, 128, 4096])
def test_engram_heads(tokens, nvfp4):
    from hyperflare_recipes.recipes.hy4_flash.engram import (
        _decode_heads, _FP4_SIGNED_VALUES, _rmsnorm, dequantize_nvfp4_engram,
    )
    d, heads = 96, 16
    device = torch.device("cuda")
    rng = torch.Generator(device=device).manual_seed(361)
    shape = (tokens, heads, d)
    hashes = torch.randint(-1, 3, shape[:2], generator=rng, device=device)
    gain = torch.randn((heads, d), generator=rng, device=device, dtype=torch.bfloat16)
    if nvfp4:
        w = torch.randint(256, (*shape[:2], d // 2), dtype=torch.uint8, device=device, generator=rng)
        scales = torch.rand((*shape[:2], 6), device=device, generator=rng).to(torch.float8_e4m3fn).view(torch.uint8)
        lut = torch.tensor(_FP4_SIGNED_VALUES, device=device)
        deq = dequantize_nvfp4_engram(w, scales, .031, lut, dtype=gain.dtype)
    else:
        w = torch.randn(shape, device=device, generator=rng).to(torch.float8_e4m3fn).view(torch.uint8)
        scales = torch.randint(111, 129, (*shape[:2], 3), dtype=torch.uint8, device=device, generator=rng)
        hashes[0, :2] = torch.tensor([0, -1], device=device)
        scales[0, 0, :2] = torch.tensor([0, 1], device=device)
        w[0, 0].fill_(126)
        w[0, 1].fill_(127)
        lut = None
        deq = (w.view(torch.float8_e4m3fn).float() *
               torch.exp2(scales.float() - 127).repeat_interleave(32, -1)).bfloat16()
    ref = _rmsnorm(torch.where(hashes[:, :, None] >= 0, deq, 0), gain, torch.finfo(gain.dtype).eps)
    actual = _decode_heads(w.flatten(0, 1), scales.flatten(0, 1), hashes, gain, lut, .031)
    torch.testing.assert_close(actual.view_as(ref), ref, atol=0, rtol=0)
    if tokens == 128:
        hashes[0, 1] = 0
        w[0, 1].zero_()
        _check_capture(lambda ids: (_decode_heads(w.flatten(0, 1), scales.flatten(0, 1),
                                                  ids, gain, lut, .031),), (hashes,), standalone=True)


@pytest.mark.parametrize("rows", [8, 128, 4096])
def test_engram_gated_mix(rows):
    from hyperflare_recipes.recipes.hy4_flash.engram import (
        IMAGE_NEWLINE_TOKEN, IMAGE_TOKEN, VIDEO_TOKEN, _mix_streams, _rmsnorm,
    )
    d = 3072
    rng = torch.Generator(device="cuda").manual_seed(825)
    def rand(*shape):
        return torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=rng)
    streams, kv = rand(rows, 4, d), rand(rows, 5, d)
    k, v = kv[:, :4], kv[:, 4]
    kg, qg, gamma, scale = rand(4, d), rand(4, d), rand(d), rand(4, d)
    ids = torch.arange(rows, device="cuda") + 1
    ids[:4] = torch.tensor([0, IMAGE_TOKEN, IMAGE_NEWLINE_TOKEN, VIDEO_TOKEN], device="cuda")
    pos = torch.arange(rows, device="cuda")
    pos[4] = -1
    eps = torch.finfo(streams.dtype).eps
    dot = (_rmsnorm(k, kg, eps).float() * _rmsnorm(streams, qg, eps).float()).sum(-1) * (d ** -.5)
    gate = (dot.abs().clamp_min(1e-6).sqrt() * dot.sign()).sigmoid().bfloat16().unsqueeze(-1)
    main = (_rmsnorm(v, None, eps) * gamma).unsqueeze(1)
    delta = gate * main + main * scale
    live = (ids != 0) & (ids != IMAGE_TOKEN) & (ids != IMAGE_NEWLINE_TOKEN) & (ids != VIDEO_TOKEN) & (pos >= 0)
    ref = streams + torch.where(live[:, None, None], delta, 0)
    actual = _mix_streams(streams, k, v, kg, qg, gamma, scale, pos, ids, 0)
    torch.testing.assert_close(actual, ref, atol=0, rtol=0)
    if rows == 128:
        _check_capture(lambda x: (_mix_streams(x, k, v, kg, qg, gamma, scale, pos, ids, 0),), (streams,))


def test_dfly_pools_selected_before_commit():
    from types import SimpleNamespace
    from hyperflare_recipes.recipes.hy4_flash.decoder import PreparedDecoder

    device = torch.device("cuda")
    rows = torch.tensor([1, 0, 2], device=device, dtype=torch.int32)
    counts = torch.tensor([2, 0, 3], device=device, dtype=torch.int32)
    meta = SimpleNamespace(state_rows=rows, starts=torch.tensor([0, 0, 4], device=device),
                           query_start_loc=torch.tensor([0, 2, 3, 6], device=device),
                           positions=torch.tensor([0, 1, -1, 4, 5, 6], device=device))
    saved = torch.full((3, 2, 2048), -10, device=device, dtype=torch.int32)
    dec = object.__new__(PreparedDecoder)
    dec.pool = SimpleNamespace(region=lambda name: saved)
    dec.dfly_index_layers, dec.dfly_topk = (1, 3), 2048
    pools = {layer: torch.randint(-1, 4, (6, 512), device=device, dtype=torch.int32)
             for layer in dec.dfly_index_layers}
    dec._save_dfly_indices(counts, pools, meta)
    reference = saved.clone()
    saved.fill_(-10)
    selected = dec._last_tokens(counts, meta)
    dec._save_dfly_indices(counts, {key: value.index_select(0, selected) for key, value in pools.items()},
                           meta, selected)
    torch.testing.assert_close(saved, reference, rtol=0, atol=0)
    assert (saved[0] == -10).all()


def test_checkpoint_boundary_against_fp32_linear():
    import json
    from pathlib import Path
    from b12x.preparation import PreparationSession
    from safetensors import safe_open
    from hyperflare_recipes.components.ihc import PreparedIHC
    from hyperflare_recipes.components.rms_norm import PreparedRMSNorm
    from hyperflare_recipes.recipes.hy4_flash.ihc import ihc_post, ihc_pre

    snapshot = os.environ.get("FLARE_HY4_FLASH_SNAPSHOT")
    if not snapshot:
        pytest.skip("requires Hy4-flash checkpoint weights")
    root = Path(snapshot)
    shards = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    def load(name):
        with safe_open(root / shards[name], framework="pt") as shard:
            return shard.get_tensor(name).cuda()

    entry_weight = load("model.layers.1.mlp.combine_output_norm.weight")
    exit_weight = load("model.layers.2.input_layernorm.weight")
    fn, base, scale = [load("model.layers.2.hc_attn_layer.hc_pre.hc_" + key) for key in ("fn", "base", "scale")]
    tokens, d = 4096, 3072
    rng = torch.Generator(device="cuda").manual_seed(92)
    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(shape, device="cuda", generator=rng, dtype=dtype)
    streams, routed, shared = rand(tokens, 4, d), rand(tokens, d), rand(tokens, d)
    post = rand(tokens, 4, dtype=torch.float32)
    mixes = post.new_empty((tokens, 8))
    eps = 1e-5
    session = PreparationSession()
    fused = PreparedIHC(streams, mixes, fn, base, scale, exit_weight, eps=eps, hc_eps=1e-6,
                        magnitude=2., session=session, name="checkpoint_ihc")
    entry, exit_norm = [PreparedRMSNorm(routed, w, torch.empty_like(routed), eps=eps, session=session,
                                        name=name).functional() for w, name in
                        ((entry_weight, "checkpoint_entry"), (exit_weight, "checkpoint_exit"))]
    updated = ihc_post(streams, entry(routed + shared), post)
    def normalize(projected):
        _, collapsed = ihc_pre(updated, fn, base, scale, rms_eps=eps, hc_eps=1e-6,
                               hc_magnitude=2., mixes=projected)
        return exit_norm(collapsed)
    eager = normalize(fused.projection(updated.flatten(1)))
    linear = normalize(None)
    _, _, actual = fused(streams, routed, shared, post, entry_weight)
    torch.testing.assert_close(actual, eager, rtol=0, atol=0)
    peak = linear.abs().max()
    peak_ulp = (torch.nextafter(peak, torch.full_like(peak, float("inf"))) - peak).float()
    error = actual.float() - linear.float()
    assert error.abs().max() <= peak_ulp
    assert error.norm() / linear.float().norm() < 1e-4


class _EngramBoundaryForward:
    def __init__(self, tokens, nvfp4):
        from types import SimpleNamespace
        from b12x.preparation import PreparationSession
        from ignition.components.engram import EngramInputs
        from ignition.components.readiness import ReadinessTimings
        from hyperflare_recipes.components.ihc import PreparedIHC
        from hyperflare_recipes.recipes.hy4_flash.engram import EngramScratch, PreparedEngram

        device = torch.device("cuda", 0)
        def ones(*shape, dtype=torch.bfloat16):
            return torch.ones(shape, device=device, dtype=dtype)

        d, heads = 3072, 16
        session = PreparationSession()
        self.streams = ones(tokens, 4, d)
        self.routed, self.shared = ones(tokens, d), ones(tokens, d)
        self.post, self.gain = ones(tokens, 4, dtype=torch.float32), ones(d)
        self.boundary = PreparedIHC(
            self.streams, ones(tokens, 8, dtype=torch.float32), ones(8, 4*d, dtype=torch.float32),
            ones(8, dtype=torch.float32), ones(2, dtype=torch.float32), self.gain,
            eps=1e-5, hc_eps=1e-6, magnitude=2., session=session, name="boundary",
        )
        ids = torch.arange(tokens, device=device)
        self.meta = SimpleNamespace(token_ids=ids, positions=ids.clone())
        frame = torch.zeros(1, device=device, dtype=torch.int32)
        delivery = EngramInputs.allocate(tokens, device, timings=ReadinessTimings.allocate(frame, 1),
                                         heads=heads, row_bytes=48 if nvfp4 else 96, scale_bytes=6 if nvfp4 else 3)
        self.event = delivery.ready
        delivery.weights.fill_(17)
        delivery.scales.fill_(56 if nvfp4 else 127)
        delivery.hashes.zero_()
        spec = SimpleNamespace(layer=1, encoding="NVFP4_E4M3_GLOBAL" if nvfp4 else "FP8_E4M3_E8M0", global_scale=1.)
        tables = SimpleNamespace(pad_token_id=-1, specs=[spec], inputs={1: delivery}, readiness_runtime=None)
        weights = {f"lookup_per_head_norms.{i}": ones(96) for i in range(heads)}
        weights.update({f"{key}.{i}": ones(d) for i in range(4)
                        for key in ("norm_key", "norm_query", "unified_per_hc_scales")})
        weights.update(unified_pre_gamma=ones(d), kv_proj=ones(5*d, heads*96), kv_bias=ones(5*d))
        self.engram = PreparedEngram(1, weights, tables, self.streams, self.meta,
                                     scratch=EngramScratch.allocate(tokens, device),
                                     session=session, name="engram", workspace=None)

    def __call__(self):
        from hyperflare_recipes.components.ihc import post_norm
        streams = post_norm(self.streams, self.routed, self.shared, self.post, self.gain, 1e-5)
        streams = self.engram(streams, self.meta)
        return self.boundary(streams, self.routed, self.shared, self.post, self.gain)[2]


def _export_boundaries(root):
    from pathlib import Path
    from ignition.preparation.aot import ForwardCache
    from ignition.preparation.core_file import read_sources
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.runtime_core import Allocation, RuntimeCore, capture_graph, instantiate_graph
    from ignition.progress import StartupProgress
    from triton import knobs

    root = Path(root)
    symbols = {"_update", "_prefill", "_collapse", "_head_kernel", "_mix_kernel"}
    loads = []
    def loaded(module, function, name, metadata, kernel_hash):
        if name in symbols:
            loads.append(name)

    with ModuleImages() as images, pytest.MonkeyPatch.context() as patch:
        patch.setattr(knobs.runtime, "kernel_load_end_hook", loaded)
        torch.cuda.set_device(0)
        forwards = {"decode": _EngramBoundaryForward(4, False), "prefill": _EngramBoundaryForward(128, True)}
        assert set(loads) == symbols
        loads.clear()
        signatures = {name: GraphSignature(name, fn.streams.shape[0], 1) for name, fn in forwards.items()}
        cache = ForwardCache(namespace={"component": "engram-ihc"}, device="cuda:0", root=root, compile_workers=1)
        compiled = cache.prepare_many(forwards, signatures, progress=StartupProgress())
        graphs = {}
        for name, bound in compiled.items():
            graph = capture_graph(retain_graphs=True)
            with torch.cuda.graph(graph):
                output = bound()
            instantiate_graph(graph)
            graph.replay()
            torch.testing.assert_close(output, forwards[name](), atol=0, rtol=0)
            graphs[name] = graph

        assert not loads, f"Triton modules first loaded after preparation: {loads}"
        allocations = [Allocation(f"segment-{i}", seg["address"], seg["total_size"], initialize="zero")
                       for i, seg in enumerate(torch.cuda.memory_snapshot())]
        package = RuntimeCore(allocations=allocations, modules={}, kernels=(), recorder=images.native,
                              owners=(forwards, compiled), events={name: f.event.cuda_event for name, f in forwards.items()})
        path = root / "boundary-rank.json"
        package.write(path, graphs, sources=True)
        index, locate = read_sources(path)
        rank, = index.ranks
        files = {m.name: m.file for m in rank.modules}
        for symbol in symbols:
            modules = {f.module for f in rank.functions if f.symbol == symbol}
            assert modules, (symbol, [f.symbol for f in rank.functions])
            for module in modules:
                span = rank.resources[files[module]]
                (source, offset, _), *_ = locate(span)
                with source.open("rb") as image:
                    image.seek(offset)
                    assert image.read(4) == b"\x7fELF"


def test_export_compiled_engram_boundaries(tmp_path):
    # CUPTI must subscribe before Triton loads these modules in the process.
    import subprocess
    import sys
    subprocess.run([sys.executable, "-c", "from tests.components.test_ihc_cuda import _export_boundaries; "
                    "import sys; _export_boundaries(sys.argv[1])", str(tmp_path)], check=True)
