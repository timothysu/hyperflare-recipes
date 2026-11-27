# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic support forwards must not expand the compiled graph catalog."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from ignition.preparation import capture


def test_support_forward_is_neither_compiled_nor_captured(monkeypatch):
    calls = []
    selected = lambda: calls.append("speculative")
    selected.prepare_capture = lambda: 0
    selected.retire_after_capture = lambda: None
    selected.capture_allocation_sizes = ()
    support = lambda: calls.append("support")
    stream = SimpleNamespace(wait_stream=lambda value: None, synchronize=lambda: None)

    class Cache:
        def __init__(self, **kwargs):
            pass

        def prepare_many(self, functions, signatures, **kwargs):
            assert functions == {"speculative": selected}
            return functions

    monkeypatch.setattr(capture, "ForwardCache", Cache)
    monkeypatch.setattr(capture.torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(capture, "_warmup_stream", lambda device: stream)
    monkeypatch.setattr(capture, "_seed_warmup_workspaces", lambda stream, device: None)

    monkeypatch.setattr(capture.torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(capture.torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(capture.torch.cuda, "CUDAGraph", lambda: "graph")
    monkeypatch.setattr(capture.torch.cuda, "graph", lambda *args, **kwargs: nullcontext())
    graphs, compiled = capture.capture_forwards(
        {"speculative": selected, "support": support},
        {"speculative": SimpleNamespace(kind="speculative"), "support": SimpleNamespace(kind="prefill")},
        session=SimpleNamespace(namespace=SimpleNamespace(to_dict=lambda: {}), capture=nullcontext),
        device="cuda:0", stream=stream, pool=None, compile_kinds=("speculative", "prefill"),
        capture_names=("speculative",))
    assert graphs == {"speculative": "graph"}
    assert compiled == {"speculative": selected}
    assert calls == ["speculative"]


def test_trial_inputs_are_restored_before_warmup_and_outside_graph_capture(monkeypatch):
    calls, state = [], {"value": 0, "capturing": False}
    stream = SimpleNamespace(wait_stream=lambda value: None, synchronize=lambda: None)

    @contextmanager
    def graph_scope(*args, **kwargs):
        state["capturing"] = True
        try:
            yield
        finally:
            state["capturing"] = False

    def restore(name):
        assert name in ("trial", "other") and not state["capturing"]
        state["value"] += 1
        calls.append(("restore", state["value"]))

    def forward():
        calls.append(("run", state["value"], state["capturing"]))

    monkeypatch.setattr(capture.torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(capture, "_warmup_stream", lambda device: stream)
    monkeypatch.setattr(capture, "_seed_warmup_workspaces", lambda stream, device: None)

    monkeypatch.setattr(capture.torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(capture.torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(capture.torch.cuda, "CUDAGraph", lambda: "graph")
    monkeypatch.setattr(capture.torch.cuda, "graph", graph_scope)
    capture.capture_forwards(dict.fromkeys(("trial", "other"), forward),
        {name: SimpleNamespace(kind="speculative") for name in ("trial", "other")},
        session=SimpleNamespace(namespace=SimpleNamespace(to_dict=lambda: {}), capture=nullcontext),
        device="cuda:0", stream=stream, pool=None, compile_decoder=False, before_run=restore)
    assert calls == [("restore", 1), ("run", 1, False), ("restore", 2), ("run", 2, False),
                     ("restore", 3), ("run", 3, True), ("restore", 4), ("run", 4, True)]


@pytest.mark.parametrize("selection", [("missing",), ("speculative", "speculative")])
def test_invalid_selection_rejected_before_cuda(selection):
    with pytest.raises(ValueError, match="capture selection"):
        capture.capture_forwards({"speculative": lambda: None}, {}, session=None,
            device="cuda:0", stream=None, pool=None, capture_names=selection)


def test_width_one_speculation_can_be_selected_without_verification():
    from hyperflare_recipes.recipes.minimax_m3.bootstrap import _verification_variants

    shapes = ((8, 1), (8, 6), (1, 1))
    assert list(_verification_variants(shapes, {"speculative8x1"})) == [
        ("speculative8x1", "speculative", 8, 1)]
    assert list(_verification_variants(shapes, {"speculative8x6"})) == [
        ("speculative8x6", "speculative", 8, 6)]
    assert [name for name, *_ in _verification_variants(shapes, None)] == [
        "speculative8x6", "speculative8x1", "speculative1x1"]
