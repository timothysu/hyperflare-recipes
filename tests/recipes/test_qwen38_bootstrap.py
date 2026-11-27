# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU ownership tests for prepared-model handoff and confirmed-drain cleanup."""

from __future__ import annotations

import gc
import sys
import weakref
from threading import Event, Thread
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.execution import GraphSignature
from ignition.preparation.model import PreparedModel


class Resource:
    def __init__(self, name, events):
        self.name, self.events = name, events

    def close(self):
        self.events.append(self.name)

    def reset(self):
        self.events.append(self.name)


def model_fixture(monkeypatch, *, synchronize=None, server=None):
    events = []
    weights = Resource("weights", events)
    weights_ref = weakref.ref(weights)
    model = PreparedModel(
        device=0,
        graphs={"decode": Resource("graph", events)},
        buffers={"frame_index": object(), "submission_sequence": object(),
                 "completion_records": torch.empty((2, 4, 32), dtype=torch.int64)},
        signatures={"decode": GraphSignature("decode", 1, 1)},
        resources=(weights,),
        server_config={},
        memory_report={},
        session=Resource("session", events),
        close_resources=(Resource("table", events),),
    )
    monkeypatch.setattr(model, "manifest", lambda: {"test_manifest": True})
    monkeypatch.setattr("torch.cuda.synchronize", synchronize or (lambda device: events.append("synchronize")))
    monkeypatch.setitem(sys.modules, "flare_heliopause", SimpleNamespace(Server=server))
    return model, events, weights_ref


def thread_call(call, *, entered=None):
    errors = []

    def run():
        if entered is not None:
            entered.set()
        try:
            call()
        except BaseException as error:  # noqa: BLE001 - Propagate worker failures to the test thread.
            errors.append(error)

    thread = Thread(target=run, daemon=True)
    thread.start()
    return thread, errors


def join(thread):
    thread.join(timeout=3)
    assert not thread.is_alive(), "prepared-model handoff deadlocked"


def test_close_excludes_concurrent_native_start(monkeypatch):
    closing, release, attempted, constructed = Event(), Event(), Event(), Event()

    def synchronize(device):
        closing.set()
        assert release.wait(3)

    def server(*args):
        constructed.set()
        raise AssertionError("native server must not start while or after the owner closes")

    model, events, weights = model_fixture(monkeypatch, synchronize=synchronize, server=server)

    def start():
        assert closing.wait(3)
        attempted.set()
        model.start_server()

    def release_close():
        assert attempted.wait(3)
        try:
            assert not constructed.wait(0.05)
            assert not model.closed and not events and weights() is not None
        finally:
            release.set()

    starter, start_errors = thread_call(start)
    observer, observer_errors = thread_call(release_close)
    model.close()
    join(observer)
    join(starter)
    assert not observer_errors
    assert len(start_errors) == 1 and "closed" in str(start_errors[0])
    assert not constructed.is_set()
    assert events == ["graph", "session", "table"]
    assert model.closed and not model.resources and not model.buffers and not model.graphs
    gc.collect()
    assert weights() is None


def test_native_start_excludes_close_until_owner_handoff_finishes(monkeypatch):
    constructing, release, attempted, drained = Event(), Event(), Event(), Event()

    class Runtime:
        drained = False

        def close(self):
            self.drained = True
            drained.set()

    def server(*args):
        constructing.set()
        assert release.wait(3)
        return Runtime()

    model, events, _ = model_fixture(monkeypatch, server=server)
    starter, start_errors = thread_call(model.start_server)
    assert constructing.wait(3)

    def release_startup():
        assert attempted.wait(3)
        try:
            assert not drained.wait(0.05)
            assert events == ["synchronize"]
        finally:
            release.set()

    observer, observer_errors = thread_call(release_startup)
    attempted.set()
    model.close()
    join(starter)
    join(observer)
    assert not start_errors and not observer_errors
    assert drained.is_set() and model.closed
    assert events == ["synchronize", "synchronize", "graph", "session", "table"]
    model.close()
    assert events.count("graph") == 1


def test_wrong_thread_close_preserves_runtime_and_all_owners(monkeypatch):
    class Runtime:
        drained = False

        def close(self):
            raise AssertionError("wrong-thread cleanup must not stop native serving")

    model, events, weights = model_fixture(monkeypatch)
    runtime = Runtime()
    model.runtime, model._consumed = runtime, True
    closer, errors = thread_call(model.close)
    join(closer)
    assert len(errors) == 1 and "preparation thread" in str(errors[0])
    assert model.runtime is runtime and not model.closed
    assert model.graphs and model.buffers and model.resources and weights() is not None
    assert not events


@pytest.mark.parametrize("raises", [False, True])
def test_unconfirmed_native_drain_retains_every_prepared_resource(monkeypatch, raises):
    class Runtime:
        drained = False

        def close(self):
            if raises:
                raise RuntimeError("CUDA stream drain failed")

    model, events, weights = model_fixture(monkeypatch)
    runtime = Runtime()
    model.runtime, model._consumed = runtime, True
    with pytest.raises(RuntimeError, match="drain"):
        model.close()
    gc.collect()
    assert not model.closed and model.runtime is runtime
    assert model.graphs and model.buffers and model.resources
    assert weights() is not None and not events


def test_failed_startup_retention_blocks_cleanup_and_second_handoff(monkeypatch):
    error = RuntimeError("native startup failed without confirmed drain")
    error.resources_retained = True

    def server(*args):
        raise error

    model, events, weights = model_fixture(monkeypatch, server=server)
    with pytest.raises(RuntimeError) as raised:
        model.start_server()
    assert raised.value is error and model.consumed
    with pytest.raises(RuntimeError, match="remain retained"):
        model.close()
    with pytest.raises(RuntimeError, match="consumed or closed"):
        model.start_server()
    assert not model.closed and weights() is not None
    assert model.graphs and model.buffers and model.resources
    assert events == ["synchronize"]


def test_confirmed_drain_releases_resources_even_when_runtime_reports_error(monkeypatch):
    failure = RuntimeError("model worker reported a device completion error")

    class Runtime:
        drained = True

        def close(self):
            raise failure

    model, events, weights = model_fixture(monkeypatch)
    model.runtime, model._consumed = Runtime(), True
    with pytest.raises(RuntimeError) as raised:
        model.close()
    assert raised.value is failure
    assert model.closed and model.runtime is None
    assert not model.resources and not model.buffers and not model.graphs
    assert events == ["synchronize", "graph", "session", "table"]
    gc.collect()
    assert weights() is None
    model.close()
    assert events.count("graph") == 1


def test_context_manager_preserves_primary_failure_when_cleanup_is_undrained(monkeypatch):
    model, events, weights = model_fixture(monkeypatch)
    model._startup_undrained = True
    primary = ValueError("preparation consumer failed")
    with pytest.raises(ValueError) as raised, model:
        raise primary
    assert raised.value is primary
    assert any("cleanup failed" in note and "remain retained" in note for note in primary.__notes__)
    assert not events and weights() is not None and not model.closed


def test_drained_owner_can_retry_incomplete_resource_cleanup(monkeypatch):
    class Runtime:
        drained = True

        def close(self):
            pass

    class Session(Resource):
        calls = 0

        def close(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("preparation resource cleanup failed once")
            super().close()

    model, events, weights = model_fixture(monkeypatch)
    model.session = Session("session", events)
    model.runtime, model._consumed = Runtime(), True
    with pytest.raises(RuntimeError, match="failed once"):
        model.close()
    assert not model.closed and model.resources and weights() is not None
    assert "table" not in events
    model.close()
    assert model.closed and model.runtime is None
    assert events.count("table") == 1
    assert not model.resources and not model.graphs and not model.buffers
    gc.collect()
    assert weights() is None


@pytest.mark.parametrize("fails", [False, True])
def test_peer_resources_release_only_after_native_rank_drains(monkeypatch, fails):
    def serve_rank(owner, socket, *, rank, world):
        owner._consumed = True
        assert (socket, rank, world) == ("peer socket", 1, 2)
        if fails:
            raise RuntimeError("peer drain failed")

    monkeypatch.setattr("ignition.preparation.execution.PreparedExecution.serve_rank", serve_rank)
    model, events, weights = model_fixture(monkeypatch)
    if fails:
        with pytest.raises(RuntimeError, match="peer drain failed"):
            model.serve_rank("peer socket", rank=1, world=2)
        with pytest.raises(RuntimeError, match="must retain"):
            model.close()
        assert not events and not model.closed and weights() is not None
    else:
        model.serve_rank("peer socket", rank=1, world=2)
        model.close()
        assert model.closed and events == ["synchronize", "graph", "session", "table"]
        gc.collect()
        assert weights() is None


def test_snapshot_budget_covers_the_selected_speculative_layout():
    from dataclasses import replace
    from hyperflare_recipes.recipes.qwen38.bootstrap import PreparationConfig
    from hyperflare_recipes.recipes.qwen38.recipe import RECIPE
    from hyperflare_recipes.recipes.qwen38.state import HybridStateLayout

    ordinary = HybridStateLayout.build().row_bytes
    speculative = HybridStateLayout.build(speculative_tokens=7).row_bytes
    assert speculative > ordinary
    PreparationConfig(snapshot_budget_bytes=ordinary)
    mtp_params = replace(PreparationConfig().parameters, speculation=RECIPE.speculation)
    with pytest.raises(ValueError, match="at least one state row"):
        PreparationConfig(parameters=mtp_params, snapshot_budget_bytes=ordinary)
    config = PreparationConfig(parameters=mtp_params, snapshot_budget_bytes=speculative)
    assert config.snapshot_budget_bytes == speculative


def test_moe_decode_a16_requires_separate_graph_roles_and_aligned_experts():
    from dataclasses import replace
    from ignition.parameters import MixedPrefillParameters
    from hyperflare_recipes.recipes.qwen38.bootstrap import PreparationConfig

    base = PreparationConfig().parameters
    a16 = replace(base, force_moe_decode_a16=True)
    assert PreparationConfig(parameters=a16).force_moe_decode_a16
    assert PreparationConfig(devices=(0, 1), parameters=a16).force_moe_decode_a16
    with pytest.raises(ValueError, match="separate prefill and decode graphs"):
        PreparationConfig(parameters=replace(a16, mixed_prefill=MixedPrefillParameters(enabled=True)))
    with pytest.raises(ValueError, match="intermediate width divisible by 64"):
        PreparationConfig(devices=(0, 1, 2, 3), parameters=a16)


def test_prepared_model_retains_persistent_storage_until_confirmed_drain(monkeypatch):
    from ignition.preparation.persistent import _ACTIVE_STORAGE

    storage = Resource("persistent storage", [])
    token = _ACTIVE_STORAGE.set(storage)
    try:
        model, events, _ = model_fixture(monkeypatch)
    finally:
        _ACTIVE_STORAGE.reset(token)
    storage.events = events
    model._startup_undrained = True
    with pytest.raises(RuntimeError, match="resources remain retained"):
        model.close()
    assert not events
    model._startup_undrained = False
    model.close()
    assert events == ["synchronize", "graph", "session", "table", "persistent storage"]
    model.close()
    assert events.count("persistent storage") == 1


def test_qwen38_decoder_routes_ple_barrier_to_barrier_runtime(monkeypatch):
    """PreparedDecoder keeps PLE's 16-byte readiness barrier on barrier_runtime when layer reductions use DMA."""
    import hyperflare_recipes.components.embedding as embedding_mod
    import hyperflare_recipes.components.gated_delta as gdn_mod
    import hyperflare_recipes.components.vocabulary as vocab_mod
    import hyperflare_recipes.recipes.qwen38.attention as attention_mod
    import hyperflare_recipes.recipes.qwen38.decoder as decoder_mod
    import hyperflare_recipes.recipes.qwen38.hyperconnection as hc_mod
    import hyperflare_recipes.recipes.qwen38.moe as moe_mod
    import hyperflare_recipes.recipes.qwen38.ple as ple_mod
    import ignition.components.collectives as collectives_mod

    seen = {"hc": [], "reduce": [], "ple": [], "attn": []}

    class DummyOp:
        def __init__(self, *args, **kwargs):
            self.scratch = SimpleNamespace(output=torch.zeros((1, 1)))
            self.output = torch.zeros((1, 1))

        def prepare_combine_norm(self, *args, **kwargs):
            pass

    def fake_hc(*args, **kwargs):
        seen["hc"].append(kwargs.get("runtime"))
        return DummyOp()

    def fake_reduce(*args, **kwargs):
        seen["reduce"].append(kwargs.get("runtime"))
        return DummyOp()

    def fake_ple(*args, **kwargs):
        seen["ple"].append(kwargs.get("runtime"))
        return DummyOp()

    def fake_attn(*args, **kwargs):
        seen["attn"].append((kwargs.get("qsa_gather_runtime"), kwargs.get("qsa_shard_mode")))
        return DummyOp()

    monkeypatch.setattr(embedding_mod, "PreparedEmbedding", DummyOp)
    monkeypatch.setattr(hc_mod, "PreparedHyperconnection", fake_hc)
    monkeypatch.setattr(attention_mod, "PreparedAttention", fake_attn)
    monkeypatch.setattr(gdn_mod, "PreparedGDN", DummyOp)
    monkeypatch.setattr(moe_mod, "PreparedQwenMoe", DummyOp)
    monkeypatch.setattr(ple_mod, "PreparedPLE", fake_ple)
    monkeypatch.setattr(collectives_mod, "PreparedAllReduce", fake_reduce)
    monkeypatch.setattr(vocab_mod, "PreparedVocabularyProjection", DummyOp)

    def make_layer():
        ple = SimpleNamespace(
            key_proj=SimpleNamespace(weight=None),
            value_proj=SimpleNamespace(weight=None),
        )
        return SimpleNamespace(
            attn_hyper_connection=SimpleNamespace(
                input_mix_weight_down=SimpleNamespace(weight=torch.empty((320, 10240)))
            ),
            mlp_hyper_connection=object(),
            self_attn=object(),
            linear_attn=object(),
            ple=ple,
        )

    layers = [make_layer() for _ in range(48)]
    language = SimpleNamespace(
        embed_tokens=SimpleNamespace(weight=None),
        layers=layers,
        hyper_connection_mixer=object(),
    )
    weights = SimpleNamespace(
        tree=SimpleNamespace(model=SimpleNamespace(language_model=language), lm_head=SimpleNamespace(weight=None)),
        layers=[SimpleNamespace(attention={}, moe=None) for _ in range(48)],
    )
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class DummyHCScratch:
        block_input: torch.Tensor

    scratch = SimpleNamespace(
        embedding=torch.zeros((4, 2560)),
        residual=torch.zeros((4, 10240)),
        hidden=torch.zeros((4, 2560)),
        sample_hidden=torch.zeros((1, 2560)),
        logits=torch.zeros((1, 16)),
        local_logits=torch.zeros((1, 8)),
        hyperconnection=DummyHCScratch(block_input=torch.zeros((4, 2560))),
        gdn=object(),
        qsa=object(),
        ple=object(),
        moe=object(),
    )
    state = SimpleNamespace(
        tokens=4,
        rows=1,
        mode="prefill",
        ple=SimpleNamespace(token_ids=None),
        gdn=SimpleNamespace(num_tokens=None),
        qsa=None,
        pool=SimpleNamespace(
            layout=SimpleNamespace(speculative_tokens=0),
            gdn={i: None for i in range(48) if i % 4 != 3},
            ple_convolution=None,
        ),
        prepare_hidden=lambda *args: None,
        prepare_residual=lambda *args: None,
    )
    qsa_states = {i: None for i in range(3, 48, 4)}
    dma_rt = object()
    pcie_rt = object()
    vocab_rt = object()
    qsa_gather_rt = object()

    decoder_mod.PreparedDecoder(
        weights, scratch, state, qsa_states, None, None, None,
        None, None, None, session=None, name="prefill1024", verification=True,
        runtime=dma_rt, barrier_runtime=pcie_rt, vocabulary_runtime=vocab_rt,
        qsa_gather_runtime=qsa_gather_rt, qsa_shard_mode="row",
    )
    assert seen["ple"] == [pcie_rt]
    assert len(seen["reduce"]) == 96 and all(rt is dma_rt for rt in seen["reduce"])
    assert len(seen["hc"]) == 97 and all(rt is dma_rt for rt in seen["hc"])
    assert seen["attn"] == [(qsa_gather_rt, "row")] * 12

    for bucket in seen.values():
        bucket.clear()
    decoder_mod.PreparedDecoder(
        weights, scratch, state, qsa_states, None, None, None,
        None, None, None, session=None, name="prefill512", verification=True,
        runtime=pcie_rt, vocabulary_runtime=vocab_rt,
    )
    assert seen["ple"] == [pcie_rt]
    assert len(seen["reduce"]) == 96 and all(rt is pcie_rt for rt in seen["reduce"])
    assert seen["attn"] == [(None, None)] * 12


def _build_stub_decoder_env(monkeypatch, *, tokens=4096, rows=1, mode="prefill", hc_lowrank=320,
                            world_size=2, gdn_partitions=None):
    from dataclasses import dataclass
    import hyperflare_recipes.components.embedding as embedding_mod
    import hyperflare_recipes.components.gated_delta as gdn_mod
    import hyperflare_recipes.components.linear as linear_mod
    import hyperflare_recipes.components.vocabulary as vocab_mod
    import hyperflare_recipes.recipes.qwen38.attention as attention_mod
    import hyperflare_recipes.recipes.qwen38.decoder as decoder_mod
    import hyperflare_recipes.recipes.qwen38.hyperconnection as hc_mod
    import hyperflare_recipes.recipes.qwen38.moe as moe_mod
    import hyperflare_recipes.recipes.qwen38.ple as ple_mod
    import ignition.components.collectives as collectives_mod
    import ignition.components.sampling as sampling_mod

    events = {"combine_norm": [], "completion": 0, "sampling": 0, "rope_delta": 0, "hidden": 0, "residual": 0}

    class DummyOp:
        def __init__(self, *args, **kwargs):
            self.scratch = SimpleNamespace(output=torch.zeros((1, 1)))
            self.output = torch.zeros((1, 1))
            self.plan = SimpleNamespace(name=kwargs.get("name", "op"))
            self.plans = {"combine": self.plan}
            self.down = SimpleNamespace(plan=self.plan)
            self.gate = SimpleNamespace(plan=self.plan)
            self.projections = {}
            self.writer = self.plan
            self.sharded = None
            self.router_projection = SimpleNamespace(plan=self.plan)
            self.routing = SimpleNamespace(plan=self.plan)
            self.experts = SimpleNamespace(plan=self.plan)
            self.shared = SimpleNamespace(
                up_projection=SimpleNamespace(plan=self.plan),
                activation_plan=self.plan,
                down_projection=SimpleNamespace(plan=self.plan),
                gate_projection=SimpleNamespace(plan=self.plan),
            )
            self.reduce = SimpleNamespace(plan=self.plan)

        def prepare_combine_norm(self, *args, **kwargs):
            name = kwargs.get("name", "")
            events["combine_norm"].append(name)
            self.plans["combine_norm"] = SimpleNamespace(name=name)

    class DummyCompletion:
        def __init__(self, *args, **kwargs):
            self.prepared = False

        def prepare(self):
            self.prepared = True
            events["completion"] += 1

    class DummySampling:
        def __init__(self, *args, **kwargs):
            pass

        def prepare(self):
            events["sampling"] += 1

        def prepare_broadcast(self, *args, **kwargs):
            pass

    monkeypatch.setattr(embedding_mod, "PreparedEmbedding", DummyOp)
    monkeypatch.setattr(hc_mod, "PreparedHyperconnection", DummyOp)
    monkeypatch.setattr(attention_mod, "PreparedAttention", DummyOp)
    monkeypatch.setattr(gdn_mod, "PreparedGDN", DummyOp)
    monkeypatch.setattr(moe_mod, "PreparedQwenMoe", DummyOp)
    monkeypatch.setattr(ple_mod, "PreparedPLE", DummyOp)
    monkeypatch.setattr(collectives_mod, "PreparedAllReduce", DummyOp)
    monkeypatch.setattr(collectives_mod, "ConcurrentAllReduce", DummyOp)
    monkeypatch.setattr(vocab_mod, "PreparedVocabularyProjection", DummyOp)
    monkeypatch.setattr(linear_mod, "UnquantizedProjection", DummyOp)
    monkeypatch.setattr(sampling_mod, "PreparedCompletion", DummyCompletion)
    monkeypatch.setattr(sampling_mod, "PreparedSampling", DummySampling)

    def make_layer():
        ple = SimpleNamespace(key_proj=SimpleNamespace(weight=None), value_proj=SimpleNamespace(weight=None))
        return SimpleNamespace(
            attn_hyper_connection=SimpleNamespace(
                input_mix_weight_down=SimpleNamespace(weight=torch.empty((hc_lowrank, 10240)))
            ),
            mlp_hyper_connection=object(),
            self_attn=object(),
            linear_attn=object(),
            ple=ple,
        )

    layers = [make_layer() for _ in range(48)]
    language = SimpleNamespace(
        embed_tokens=SimpleNamespace(weight=None),
        layers=layers,
        hyper_connection_mixer=object(),
    )
    weights = SimpleNamespace(
        tree=SimpleNamespace(model=SimpleNamespace(language_model=language), lm_head=SimpleNamespace(weight=None)),
        layers=[SimpleNamespace(attention={}, moe=None) for _ in range(48)],
        shard=SimpleNamespace(rank=0, world_size=world_size),
    )

    @dataclass(frozen=True)
    class DummyHCScratch:
        block_input: torch.Tensor

    class DummyScratch:
        def __init__(self, n):
            self.embedding = torch.zeros((n, 2560))
            self.residual = torch.zeros((n, 10240))
            self.hidden = torch.zeros((n, 2560))
            self.sample_hidden = torch.zeros((rows, 2560))
            self.logits = torch.zeros((rows, 16))
            self.local_logits = torch.zeros((rows, 8))
            self.hyperconnection = DummyHCScratch(block_input=torch.zeros((n, 2560)))
            self.gdn = object()
            self.qsa = object()
            self.ple = object()
            self.moe = object()

        def rows(self, n, seqs):
            return DummyScratch(n)

    scratch = DummyScratch(tokens)
    inputs = SimpleNamespace(
        row_slot_ids=torch.zeros((4,), dtype=torch.int32),
        row_generations=torch.zeros((4,), dtype=torch.int64),
        live_rows=torch.zeros((1,), dtype=torch.int32),
        sample_enabled=torch.zeros((4,), dtype=torch.int32),
    )
    state = SimpleNamespace(
        tokens=tokens,
        rows=rows,
        mode=mode,
        inputs=inputs,
        requests=object(),
        ple=SimpleNamespace(token_ids=None),
        gdn=SimpleNamespace(num_tokens=None),
        qsa=SimpleNamespace(main_block_table=torch.full((1, 2), -1, dtype=torch.int32)),
        gdn_partitions=gdn_partitions,
        pool=SimpleNamespace(
            layout=SimpleNamespace(speculative_tokens=0),
            gdn={i: None for i in range(48) if i % 4 != 3},
            ple_convolution=None,
        ),
        prepare_hidden=lambda *args: events.__setitem__("hidden", events["hidden"] + 1),
        prepare_rope_delta=lambda *args: events.__setitem__("rope_delta", events["rope_delta"] + 1),
        prepare_residual=lambda *args: events.__setitem__("residual", events["residual"] + 1),
    )
    qsa_states = {i: None for i in range(3, 48, 4)}
    return decoder_mod, weights, scratch, state, qsa_states, events


def test_qwen38_decoder_non_sampling_prefill_omits_tail_and_validates_mode(monkeypatch):
    """PreparedDecoder(sample=False) omits final mixer, lm_head, and sampling while keeping rope_delta and residual state."""
    decoder_mod, weights, scratch, state, qsa_states, events = _build_stub_decoder_env(
        monkeypatch, tokens=512, rows=1, mode="prefill"
    )
    nonterm = decoder_mod.PreparedDecoder(
        weights, scratch, state, qsa_states, None, None, None,
        None, None, None, session=None, name="prefill512", sample=False,
    )
    assert nonterm.sample is False
    assert nonterm.final_mixer is None
    assert nonterm.lm_head is None
    assert nonterm.sampler is None
    assert nonterm.completion_recorder is not None and nonterm.completion_recorder.prepared
    assert events["completion"] == 1 and events["sampling"] == 0
    assert events["rope_delta"] == 1 and events["hidden"] == 0 and events["residual"] == 1
    assert "prefill512.layer47.mlp_combine_norm" not in events["combine_norm"]
    assert "prefill512.layer46.mlp_combine_norm" in events["combine_norm"]

    # capture_context remains attachable on sample=False decoders (used by DFlash2 prefill absorption).
    draft_cfg = SimpleNamespace(target_layers=(5, 15, 25, 35, 45), attention=SimpleNamespace(window=256))
    cap_meta = SimpleNamespace(
        positions=torch.zeros((512,), dtype=torch.int64),
        starts=torch.zeros((1,), dtype=torch.int64),
    )
    nonterm.capture_context(cap_meta, draft_cfg, compact_prefill=True)
    assert nonterm.context_layers == (5, 15, 25, 35, 45)
    assert nonterm.auxiliary.shape == (256, 5 * 2560)

    # Invalid sample=False combinations must raise ValueError.
    for bad_kwargs in (
        dict(verification=True),
        dict(scoring=object()),
    ):
        with pytest.raises(ValueError, match="non-sampling"):
            decoder_mod.PreparedDecoder(
                weights, scratch, state, qsa_states, None, None, None,
                None, None, None, session=None, name="bad", sample=False, **bad_kwargs,
            )
    decode_state = SimpleNamespace(**{**vars(state), "mode": "decode"})
    with pytest.raises(ValueError, match="non-sampling"):
        decoder_mod.PreparedDecoder(
            weights, scratch, decode_state, qsa_states, None, None, None,
            None, None, None, session=None, name="bad_decode", sample=False,
        )


def test_qwen38_decoder_pipelined_prefill_gating(monkeypatch):
    """PreparedDecoder enables 2-microbatch DMA pipelining only when every hardware and shape guard holds."""
    from ignition.preparation.scratch import ScratchArena

    class FakeArena(ScratchArena):
        def __init__(self, overlap):
            self.overlap = overlap
            self.parallel_pairs = []

        def register_parallel(self, a, b):
            self.parallel_pairs.append((a, b))

    class FakeDMA:
        def __init__(self, eligible=True):
            self.eligible = eligible

        def should_allreduce(self, tensor):
            return self.eligible

    def check(*, tokens=4096, rows=1, hc_lowrank=320, world_size=2, overlap=True, is_dma=True,
              eligible=True, qsa_shard_mode="row", gdn_records=None, gdn_partitions=None):
        decoder_mod, weights, scratch, state, qsa_states, _ = _build_stub_decoder_env(
            monkeypatch, tokens=tokens, rows=rows, hc_lowrank=hc_lowrank,
            world_size=world_size, gdn_partitions=gdn_partitions,
        )
        monkeypatch.setattr(decoder_mod, "_is_dma_runtime", lambda rt: isinstance(rt, FakeDMA))
        arena = FakeArena(overlap=overlap)
        runtime = FakeDMA(eligible=eligible) if is_dma else object()
        dec = decoder_mod.PreparedDecoder(
            weights, scratch, state, qsa_states, None, None, arena,
            None, None, None, session=None, name=f"prefill{tokens}", verification=True,
            runtime=runtime, qsa_shard_mode=qsa_shard_mode, gdn_records=gdn_records,
        )
        return dec, arena

    dec_tp2, arena_tp2 = check(tokens=4096, world_size=2, qsa_shard_mode="row")
    assert dec_tp2.pipelined_prefill is True and dec_tp2.half_tokens == 2048
    assert len(dec_tp2.pipelined_reductions) == 48 and not dec_tp2.reductions
    assert arena_tp2.parallel_pairs

    dec_tp4, _ = check(tokens=4096, world_size=4, qsa_shard_mode="row")
    assert dec_tp4.pipelined_prefill is True and dec_tp4.half_tokens == 2048

    dec_unsharded, _ = check(tokens=4096, world_size=2, qsa_shard_mode=None)
    assert dec_unsharded.pipelined_prefill is True

    # Each disabled guard forces serial execution.
    class DummyRecords:
        layers = (0,)
        delta = torch.empty((1, 1, 1, 1))
        keys = torch.empty((1, 1, 1, 1))

        def __getitem__(self, index):
            return None

    dummy_records = DummyRecords()
    for label, overrides in (
        ("below_min_tokens", dict(tokens=2048)),
        ("sharded_hc", dict(hc_lowrank=160)),
        ("no_overlap", dict(overlap=False)),
        ("non_dma", dict(is_dma=False)),
        ("dma_ineligible", dict(eligible=False)),
        ("tp4_misaligned_row_qsa", dict(tokens=4224, world_size=4, qsa_shard_mode="row")),
        ("gdn_records", dict(gdn_records=dummy_records)),
        ("gdn_partitions", dict(gdn_partitions=object())),
    ):
        dec, _ = check(**overrides)
        assert dec.pipelined_prefill is False, f"expected pipelined_prefill=False for {label}"
        assert dec.half_tokens == 0 and len(dec.reductions) == 48 and not dec.pipelined_reductions


@pytest.mark.parametrize("chunk, has_terminal", [(4096, True), (512, True), (4096, False)])
def test_qualify_graph_execution_scenario_dispatches_terminal_prefill(tmp_path, chunk, has_terminal):
    """qualify.graph_execution.Scenario covers both non-terminal prefill and terminal_prefill graphs."""
    from dataclasses import replace
    from hyperflare_recipes.recipes.qwen38.bootstrap import PreparationConfig
    from qualify.graph_execution import Scenario

    caps = PreparationConfig(parameters=replace(PreparationConfig().parameters, prefill_chunk=chunk)).prefill_capacities
    signatures = {}
    for cap in caps:
        signatures[f"prefill{cap}"] = GraphSignature("prefill", 1, cap, ())
    if has_terminal:
        signatures[f"terminal_prefill{chunk}"] = GraphSignature("terminal_prefill", 1, chunk, ())
    for batch in (1, 2, 4):
        for kind in ("init", "restore", "snapshot", "resume", "decode"):
            signatures[f"{kind}{batch}"] = GraphSignature(kind, batch, 1, ())
    for patch_cap in (256, 1024, 4096, 16384):
        signatures[f"vision{patch_cap}"] = GraphSignature("vision", 1, patch_cap, ())

    max_tokens = chunk
    from ignition.components.sampling import AdmissionBuffers

    admission = AdmissionBuffers.allocate(torch.device("cpu"), vocab_size=248320, capacity=4)
    buffers = {
        **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
        "row_slot_ids": torch.full((4,), -1, dtype=torch.int32),
        "row_generations": torch.zeros((4,), dtype=torch.int64),
        "live_rows": torch.zeros((1,), dtype=torch.int32),
        "query_lengths": torch.zeros((4,), dtype=torch.int32),
        "sample_enabled": torch.zeros((4,), dtype=torch.int32),
        "token_ids": torch.zeros((max_tokens,), dtype=torch.int64),
        "prefill_positions": torch.zeros((max_tokens, 3), dtype=torch.int64),
        "page_tables": torch.full((4, 256), -1, dtype=torch.int32),
        "vision_token_indices": torch.full((max_tokens,), -1, dtype=torch.int32),
        "vision_feature_slot": torch.full((1,), -1, dtype=torch.int32),
        "vision_slot": torch.full((1,), -1, dtype=torch.int32),
        "vision_generation": torch.zeros((1,), dtype=torch.int64),
        "vision_feature_offset": torch.zeros((1,), dtype=torch.int32),
        "state_num_ops": torch.zeros((1,), dtype=torch.int32),
        "state_source_slots": torch.zeros((4,), dtype=torch.int64),
        "state_destination_slots": torch.zeros((4,), dtype=torch.int64),
        "state_qsa_slots": torch.full((4,), -1, dtype=torch.int64),
        "state_operation": torch.zeros((4,), dtype=torch.int32),
        "state_accepted_tokens": torch.zeros((4,), dtype=torch.int32),
        "state_expected_consumed": torch.full((4,), -1, dtype=torch.int64),
    }

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(range(10, 10 + max(len(text), 8192)))

    model = SimpleNamespace(
        buffers=buffers,
        signatures=signatures,
        serving_graphs=tuple(signatures),
        memory_report={"pages": 8192},
    )
    image = {
        "prompt": list(range(100, 140)),
        "mrope": torch.arange(40, dtype=torch.int64)[:, None].expand(-1, 3),
        "token_indices": torch.full((40,), -1, dtype=torch.int32),
        "vision_inputs": {},
    }
    (tmp_path / "uploads").mkdir()
    scenario = Scenario(model, FakeTokenizer(), image, tmp_path).build()
    commands = [cmd for group in scenario.groups for cmd in group]
    executed = {cmd["graph"] for cmd in commands}
    assert executed == set(signatures)
    if has_terminal:
        term_cmds = [cmd for cmd in commands if cmd["graph"] == f"terminal_prefill{chunk}"]
        assert len(term_cmds) >= 4
        assert all(next(iter(cmd["expected_records"].values()))["generated"] == 1 for cmd in term_cmds)
        nonterm_cmds = [cmd for cmd in commands if cmd["graph"] == f"prefill{chunk}"]
        assert nonterm_cmds
        assert all(next(iter(cmd["expected_records"].values()))["token"] == -1 for cmd in nonterm_cmds)


def test_qualify_replay_floor_prefill_graph_name():
    """qualify.replay_floor.prefill_graph_name selects terminal_prefill only on final chunks when present."""
    from qualify.replay_floor import prefill_graph_name

    split_graphs = {"prefill128", "prefill512", "prefill4096", "terminal_prefill4096"}
    assert prefill_graph_name(split_graphs, 4096, is_final=False) == "prefill4096"
    assert prefill_graph_name(split_graphs, 4096, is_final=True) == "terminal_prefill4096"
    assert prefill_graph_name(split_graphs, 128, is_final=True) == "prefill128"
    assert prefill_graph_name({"prefill512", "terminal_prefill512"}, 512, is_final=True) == "terminal_prefill512"
    assert prefill_graph_name({"prefill4096"}, 4096, is_final=True) == "prefill4096"


def _tp2_dma_and_barrier_worker(rank, world, port, error_queue):
    import os
    import torch.distributed as dist
    from ignition.components.collectives import PreparedAllReduce
    from ignition.components.readiness import PreparedReadinessWait, wait_for_readiness
    from ignition.preparation.collectives import (
        CollectivePreparation,
        create_all_reduce_runtime,
        create_dma_all_reduce_runtime,
    )
    from ignition.preparation.tuning import RecordedPreparationSession

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    group = dist.group.WORLD
    pcie_rt = dma_rt = session = None
    try:
        coordinator = CollectivePreparation(group)
        pcie_rt = create_all_reduce_runtime(
            group, device, max_tokens=1024, hidden_size=2560
        )
        dma_rt = create_dma_all_reduce_runtime(group, device, max_tokens=1024, hidden_size=2560)
        session = RecordedPreparationSession(
            device=device,
            checkpoint="test_qwen38_dma_and_barrier",
            workload_schema="test_qwen38_dma_and_barrier",
            numerical_policy="default",
            coordinator=coordinator,
            autotune_effort="off",
        )
        session.__enter__()

        from ignition.components.readiness import ReadinessTimings

        frame = torch.zeros(1, dtype=torch.int32, device=device)
        timings = ReadinessTimings.allocate(frame, 1)
        # Verify DmaAllReduce rejects the 16-byte PLE readiness barrier while PCIeAllReduce accepts it.
        with pytest.raises(ValueError, match="alignment"):
            PreparedReadinessWait(timings, 0, runtime=dma_rt, session=session, coordinator=coordinator, name="bad")
        wait = PreparedReadinessWait(
            timings, 0, runtime=pcie_rt, session=session, coordinator=coordinator, name="ple.readiness"
        )
        source = torch.empty((1024, 2560), dtype=torch.bfloat16, device=device)
        output = torch.empty_like(source)
        reduce = PreparedAllReduce(
            source, output, runtime=dma_rt, session=session, coordinator=coordinator, name="layer.reduce"
        )
        session.tune(show_progress=False)

        torch.manual_seed(700 + rank)
        source.copy_(torch.randn_like(source))
        local_f32 = source.float().cpu()
        gathered = [torch.empty_like(local_f32) for _ in range(world)]
        dist.all_gather(gathered, local_f32, group=group)
        expected = sum(gathered).bfloat16().to(device)

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        stream = torch.cuda.Stream(device=device)
        b_in, b_out = wait.operands(source)
        with torch.cuda.stream(stream):
            wait_for_readiness(source, event.cuda_event, timings.timestamps, frame, 0, b_in, b_out, wait.handle)
            reduce(source)
            stream.synchronize()
            dist.barrier(group)
            graph = torch.cuda.CUDAGraph()
            with pcie_rt.capture(stream=stream), torch.cuda.graph(graph, stream=stream):
                wait_for_readiness(source, event.cuda_event, timings.timestamps, frame, 0, b_in, b_out, wait.handle)
                reduce(source)
            stream.synchronize()
            output.zero_()
            dist.barrier(group)
            graph.replay()
            stream.synchronize()
            graph.reset()
        assert torch.equal(output, expected)
    except BaseException as exc:  # noqa: BLE001
        error_queue.put((rank, repr(exc)))
        raise
    finally:
        if session is not None:
            session.close()
        if dma_rt is not None:
            dma_rt.close()
        if pcie_rt is not None:
            pcie_rt.close()
        dist.destroy_process_group()


def test_qwen38_prefill_dma_all_reduce_coexists_with_ple_readiness_barrier_tp2():
    """TP2 CUDA graph combines a PCIeAllReduce PLE readiness barrier with DmaAllReduce prefill reductions."""
    import os
    import socket
    import torch.multiprocessing as mp

    if os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1":
        pytest.skip("requires FLARE_RUN_CUDA_TESTS=1 and FLARE_RUN_TP_CUDA_TESTS=1")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least two CUDA devices")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    errors = ctx.Queue()
    mp.spawn(_tp2_dma_and_barrier_worker, args=(2, port, errors), nprocs=2, join=True)
    assert errors.empty(), errors.get()
