# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DFlash2 proposal scratch for attention widths distinct from model hidden widths."""

from dataclasses import replace
from types import SimpleNamespace
import copy

import pytest
import torch

from hyperflare_recipes.components.dflash2 import DraftScratch, _draft_templates
from hyperflare_recipes.components.dflash2_geometry import DraftAttentionGeometry
from hyperflare_recipes.components.parallel import TensorParallelShard


@pytest.mark.parametrize('world', [1, 2, 4, 8])
def test_non_square_attention_scratch_and_proposal_views(world):
    config = SimpleNamespace(block_size=8,
        text=dict(hidden_size=5120, intermediate_size=12288, vocab_size=128),
        attention=DraftAttentionGeometry(32, 8, 128, 128, 128, 10000000., 2048))
    shard = TensorParallelShard(0, world)
    scratch = DraftScratch.allocate(config, shard, 32, 'meta')
    templates = _draft_templates(config, shard, scratch, 'meta')
    assert scratch.logit_shape == (32, 128)
    assert templates.embedding.shape == (32, 5120)
    assert templates.local_embedding.shape == (32, 5120 // world)
    assert templates.query.shape == templates.attended.shape == (32, 32 // world, 128)
    assert templates.key.shape == templates.value.shape == (32, 8 // world, 128)
    assert templates.qkv.shape == (32, 6144 // world)
    assert templates.partial.shape == templates.output.shape == (32, 5120)
    assert templates.window_key.shape == (68, 128, 8 // world, 128)
    assert templates.window_value.shape == (68, 128, 8 // world, 128)
    assert templates.local_logits.shape == (32, 128 if world == 1 else 64)
    assert templates.local_logits.dtype == torch.bfloat16
    view = scratch.rows(8, 8)
    view_templates = _draft_templates(config, shard, view, 'meta')
    assert view.logit_shape == (8, 128)
    assert view_templates.query.shape == (8, 32 // world, 128)
    assert view_templates.embedding.shape == (8, 5120)
    assert view_templates.window_key.shape == (17, 128, 8 // world, 128)
    assert view_templates.convolution["before"].shape == (8, 5120)
    assert scratch.scratch_tensors() == view.scratch_tensors() == ()
    with pytest.raises(ValueError, match='complete proposal blocks'):
        scratch.rows(7, 7)


@pytest.mark.parametrize("width", [8, 13, 16])
def test_effective_block_preserves_checkpoint_and_scratch_fields(width):
    config = SimpleNamespace(block_size=8,
        text=dict(hidden_size=5120, intermediate_size=17408, vocab_size=128),
        attention=DraftAttentionGeometry(32, 8, 128, 128, 128, 10000000., 2048))
    shard = TensorParallelShard(0, 1)
    scratch = DraftScratch.allocate(config, shard, width, "meta", block_size=width)
    view = scratch.rows(width, width)
    templates = _draft_templates(config, shard, view, "meta")
    assert config.block_size == 8
    assert view.block_size == width
    assert templates.query.shape == (width, 32, 128)
    assert templates.window_key.shape[0] == 17
    assert templates.mlp.intermediate.shape == (width, 17408)
    assert templates.local_logits.shape == (width, 128)
    assert templates.gathered_logits.dtype == torch.bfloat16
    assert view.scratch_tensors() == ()


@pytest.mark.parametrize("model", ["dense", "flash"])
@pytest.mark.parametrize("world", [1, 2, 4])
def test_checkpoint_geometry_drives_shared_context_storage(model, world):
    from hyperflare_recipes.components.dflash2_geometry import DFlash2Config, draft_context_specs
    from hyperflare_recipes.recipes.qwen38.draft_config import EXPECTED
    from hyperflare_recipes.recipes.qwen38_dense.draft_config import PROFILE

    profile = EXPECTED if model == "flash" else PROFILE
    text = dict(model_type="qwen3", architectures=["DFlash2DraftModel"], hidden_act="silu",
        dtype="bfloat16", attention_bias=False, attention_dropout=0., tie_word_embeddings=False,
        use_sliding_window=True, is_causal=False, rms_norm_eps=1e-6, sliding_window=2048,
        layer_types=["sliding_attention"] * 5) | copy.deepcopy(profile)
    hidden, layers = (2560, 48) if model == "flash" else (5120, 64)
    config = DFlash2Config.from_dict(text, target_hidden=hidden, target_vocabulary=248320,
        target_layers=layers, expected=profile)
    shard = TensorParallelShard(0, world)
    config.validate_shard(shard)
    regions = dict((name, (shape, dtype)) for name, shape, dtype in draft_context_specs(config, shard))
    assert set(regions) == {f"draft{i}.{part}" for i in range(5) for part in ("key", "value")}
    assert set(regions.values()) == {((2048, 8 // world, 128), torch.float8_e4m3fn)}
    scratch = DraftScratch.allocate(config, shard, 8, "meta")
    templates = _draft_templates(config, shard, scratch, "meta")
    assert scratch.hidden_size == hidden
    assert templates.embedding.shape == (8, hidden)
    assert templates.query.shape == (8, 32 // world, 128)
    assert templates.local_embedding.shape == (8, hidden // world)
    assert templates.qkv.shape == (8, 6144 // world)
    with pytest.raises(ValueError, match="does not match the target"):
        DFlash2Config.from_dict(text, target_hidden=hidden, target_vocabulary=248320,
            target_layers=layers + 1, expected=profile)


KEY, VALUE, WINDOW = 8, 4, 2048


def _context(compact):
    """A draft context whose projections are identities and whose writer records its inputs."""
    from hyperflare_recipes.components.dflash2 import PreparedContext

    written = []
    context = object.__new__(PreparedContext)
    context.compact, context.window = compact, WINDOW
    context.kv_heads, context.key_dim, context.value_dim = 1, KEY, VALUE
    context.project = context.gather = context.norm = context.project_kv = lambda value: value
    context.inverse = 1.0 / (10000 ** (torch.arange(0, KEY, 2, dtype=torch.float32) / KEY))
    context.layers = [(lambda value: value, lambda key, value, slots: written.append((key, value, slots)))]
    return context, written


def _chunk_metadata(chunk, three_axis):
    from hyperflare_recipes.components.paged_batch import Metadata

    metadata = Metadata.allocate(chunk, 1, 1, "cpu")
    metadata.set(torch.zeros(1, dtype=torch.int64), torch.full((1,), chunk, dtype=torch.int32),
        torch.ones(1, dtype=torch.int64), torch.zeros((1, 1), dtype=torch.int32),
        torch.arange(chunk, dtype=torch.int64).reshape(1, chunk), torch.ones(1, dtype=torch.int32))
    if three_axis:
        positions = metadata.positions
        return replace(metadata, rotary_positions=torch.stack(
            (positions + 7, positions + 100, positions + 200), dim=1))
    return metadata


@pytest.mark.parametrize("chunk", [1024, 2560, 4096])
@pytest.mark.parametrize("metadata_type", ["paged", "plain", "three_axis"])
def test_context_rotates_and_places_retained_tokens(chunk, metadata_type):
    """Non-compact chunks write every token; compact chunks keep the trailing window.

    Each written row rotates at its own position: the token index for plain or
    paged metadata (even when a recipe-specific ``rope_positions`` attribute is
    present) and the temporal column when ``rotary_positions`` is supplied.
    """
    from hyperflare_recipes.components.dflash2 import _rotary

    compact = chunk > WINDOW
    rows = WINDOW if compact else chunk
    three_axis = metadata_type == "three_axis"
    metadata = _chunk_metadata(chunk, three_axis)
    if metadata_type == "plain":
        metadata = SimpleNamespace(positions=metadata.positions, starts=metadata.starts,
            requests=metadata.requests, state_rows=metadata.state_rows,
            query_start_loc=metadata.query_start_loc, rope_positions="recipe-specific")
    counts = torch.tensor([chunk], dtype=torch.int32)
    auxiliary = torch.randn((rows, KEY + VALUE), generator=torch.Generator().manual_seed(2)).bfloat16()
    context, written = _context(compact)
    context(auxiliary, counts, metadata)
    (key, value, slots), = written
    retained = torch.arange(chunk - rows, chunk)
    assert slots.tolist() == (WINDOW + retained.remainder(WINDOW)).tolist()
    source = auxiliary[:, :KEY].view(rows, 1, KEY)
    assert torch.equal(key, _rotary(source, retained + 7 if three_axis else retained, context.inverse))
    if compact:
        assert not torch.equal(key, _rotary(source, torch.arange(rows), context.inverse))
    assert torch.equal(value, auxiliary[:, KEY:].view(rows, 1, VALUE))


class _TensorWriter:
    def __init__(self, key, value, slots):
        self.key, self.value, self.slots = key, value, slots

    def __call__(self, key, value, slots):
        self.key.copy_(key)
        self.value.copy_(value)
        self.slots.copy_(slots)


class _ContextForward:
    def __init__(self, context, auxiliary, counts, metadata):
        self.context, self.auxiliary, self.counts, self.metadata = context, auxiliary, counts, metadata

    def __call__(self):
        self.context(self.auxiliary, self.counts, self.metadata)


def _execute_forward(forward):
    forward()


@pytest.mark.parametrize("chunk", [1024, 4096])
def test_context_aot_guards_match_across_tp_ranks(chunk):
    """AOT-compiled PreparedContext guards must not depend on TensorParallelShard.rank."""
    from torch._dynamo.aot_compile_types import GraphModuleSerializableCallable
    from hyperflare_recipes.components.dflash2 import PreparedContext

    compact = chunk > WINDOW
    rows = WINDOW if compact else chunk
    geometry = DraftAttentionGeometry(4, 2, KEY, VALUE, KEY, 10000., WINDOW)

    def build(rank):
        shard = TensorParallelShard(rank, 2)
        _, kv_heads = geometry.local(shard)
        out_key = torch.zeros((rows, kv_heads, KEY), dtype=torch.bfloat16)
        out_value = torch.zeros((rows, kv_heads, VALUE), dtype=torch.bfloat16)
        out_slots = torch.full((rows,), -1, dtype=torch.int64)
        context = object.__new__(PreparedContext)
        context.compact, context.window = compact, WINDOW
        context.kv_heads, context.key_dim, context.value_dim = kv_heads, KEY, VALUE
        context.weights = SimpleNamespace(shard=shard, config=SimpleNamespace(draft=SimpleNamespace(attention=geometry)))
        context.project = context.gather = context.norm = context.project_kv = lambda value: value
        context.inverse = 1.0 / (10000 ** (torch.arange(0, KEY, 2, dtype=torch.float32) / KEY))
        context.layers = [(lambda value: value, _TensorWriter(out_key, out_value, out_slots))]
        auxiliary = torch.randn((rows, kv_heads * (KEY + VALUE)),
                                generator=torch.Generator().manual_seed(11 + rank)).bfloat16()
        counts = torch.tensor([chunk], dtype=torch.int32)
        metadata = _chunk_metadata(chunk, three_axis=False)
        return _ContextForward(context, auxiliary, counts, metadata), (out_key, out_value, out_slots)

    rank0_forward, _ = build(0)
    rank1_forward, (out_key, out_value, out_slots) = build(1)
    compiled = torch.compile(
        _execute_forward,
        backend=lambda graph, inputs: GraphModuleSerializableCallable(graph),
        fullgraph=True,
        dynamic=False,
    ).aot_compile(((rank0_forward,), {}))
    assert compiled.guard_check(rank0_forward)
    assert compiled.guard_check(rank1_forward)

    rank1_forward()
    expected_key, expected_value, expected_slots = out_key.clone(), out_value.clone(), out_slots.clone()
    out_key.zero_()
    out_value.zero_()
    out_slots.fill_(-1)

    compiled(rank1_forward)
    assert torch.equal(out_key, expected_key)
    assert torch.equal(out_value, expected_value)
    assert torch.equal(out_slots, expected_slots)


class _FixedGather:
    def __init__(self, output):
        self.output = output

    def __call__(self, packet):
        return self.output


class _ShortlistForward:
    def __init__(self, draft, local_logits, out_u, out_i):
        self.draft, self.local_logits, self.out_u, self.out_i = draft, local_logits, out_u, out_i

    def __call__(self):
        u, i = self.draft._shortlist_from_local_logits(self.local_logits)
        self.out_u.copy_(u)
        self.out_i.copy_(i)


@pytest.mark.parametrize("vocab,local_vocab", [(256, 128), (250, 128)])
@pytest.mark.parametrize("softcap", [None, 20.0])
def test_distributed_shortlist_matches_full_topk_and_aot_guards_match_across_tp_ranks(vocab, local_vocab, softcap):
    """Distributed local top-k shortlist matches full-vocab top-k and shares AOT guards across TP ranks."""
    from torch._dynamo.aot_compile_types import GraphModuleSerializableCallable
    from hyperflare_recipes.components.dflash2 import PreparedDraft

    batch, block, top_k, world = 2, 4, 16, 2
    tokens = batch * block
    gen = torch.Generator().manual_seed(173 + vocab)
    full_bf16 = torch.randn((tokens, vocab), generator=gen).bfloat16()
    padded = torch.zeros((tokens, local_vocab * world), dtype=torch.bfloat16)
    padded[:, :vocab] = full_bf16
    shards = padded.chunk(world, dim=-1)
    words_per_rank = ((top_k * 4 + 63) // 64) * 64 // 2

    def local_packet(rank):
        draft = object.__new__(PreparedDraft)
        draft.shortlist_output = True
        draft.top_k, draft.world_size = top_k, world
        draft.block_size, draft.vocabulary_size = block, vocab
        draft.multiplier, draft.softcap = 0.5, softcap
        draft.vocab_offset = torch.tensor(rank * local_vocab, dtype=torch.int32)
        draft.shortlist_cols = words_per_rank * 2
        captured = []
        draft.shortlist_gather = lambda pkt: (captured.append(pkt.clone()), torch.cat((pkt, pkt), dim=-1))[1]
        draft._shortlist_from_local_logits(shards[rank])
        return captured[0]

    packets = [local_packet(r) for r in range(world)]
    gathered_packet = torch.cat(packets, dim=-1)

    def build_forward(rank):
        draft = object.__new__(PreparedDraft)
        draft.shortlist_output = True
        draft.top_k, draft.world_size = top_k, world
        draft.block_size, draft.vocabulary_size = block, vocab
        draft.multiplier, draft.softcap = 0.5, softcap
        draft.vocab_offset = torch.tensor(rank * local_vocab, dtype=torch.int32)
        draft.shortlist_cols = words_per_rank * 2
        draft.shortlist_gather = _FixedGather(gathered_packet)
        out_u = torch.empty((batch, block, top_k), dtype=torch.float32)
        out_i = torch.empty((batch, block, top_k), dtype=torch.int64)
        return _ShortlistForward(draft, shards[rank].contiguous(), out_u, out_i), (out_u, out_i)

    f0, (u0, i0) = build_forward(0)
    f1, (u1, i1) = build_forward(1)
    compiled = torch.compile(
        _execute_forward,
        backend=lambda graph, inputs: GraphModuleSerializableCallable(graph),
        fullgraph=True,
        dynamic=False,
    ).aot_compile(((f0,), {}))
    assert compiled.guard_check(f0)
    assert compiled.guard_check(f1)
    compiled(f0)
    compiled(f1)
    assert torch.equal(u0, u1) and torch.equal(i0, i1)

    ref_scores = full_bf16.float() * 0.5
    if softcap is not None:
        ref_scores = (ref_scores / softcap).tanh() * softcap
    ref_u, _ = ref_scores.topk(top_k, dim=-1, sorted=True)
    got_u, order = u0.view(tokens, top_k).sort(dim=-1, descending=True)
    got_i = i0.view(tokens, top_k).gather(-1, order)
    assert torch.equal(got_u, ref_u)
    assert torch.equal(ref_scores.gather(-1, got_i), got_u)
    # Every selected token ID within a row must be distinct and in-bounds.
    for row in range(tokens):
        assert len(set(got_i[row].tolist())) == top_k
        assert int(got_i[row].min()) >= 0 and int(got_i[row].max()) < vocab


@pytest.mark.parametrize("multiplier,softcap,vocab,local_vocab", [
    (-0.5, None, 250, 128),
    (-0.5, 20.0, 250, 128),
    (0.0, None, 250, 128),
    (1.0, 5.0, 132, 128),
])
def test_distributed_shortlist_masks_padded_columns_for_nonmonotonic_and_softcap(
        multiplier, softcap, vocab, local_vocab):
    """Padded out-of-vocabulary columns never win top-k under negative/zero scaling or softcap."""
    from hyperflare_recipes.components.dflash2 import PreparedDraft

    batch, block, top_k, world = 2, 4, 16, 2
    tokens = batch * block
    gen = torch.Generator().manual_seed(293 + vocab)
    # Shift logits negative so softcapped padded -inf (-softcap) would beat real logits if unmasked.
    full_bf16 = (torch.randn((tokens, vocab), generator=gen) - 20.0).bfloat16()
    padded = torch.full((tokens, local_vocab * world), 10.0, dtype=torch.bfloat16)
    padded[:, :vocab] = full_bf16
    shards = padded.chunk(world, dim=-1)
    words_per_rank = ((top_k * 4 + 63) // 64) * 64 // 2

    def make_draft(rank, gather):
        draft = object.__new__(PreparedDraft)
        draft.shortlist_output = True
        draft.top_k, draft.world_size = top_k, world
        draft.block_size, draft.vocabulary_size = block, vocab
        draft.multiplier, draft.softcap = multiplier, softcap
        draft.vocab_offset = torch.tensor(rank * local_vocab, dtype=torch.int32)
        draft.shortlist_cols = words_per_rank * 2
        draft.shortlist_gather = gather
        return draft

    packets = []
    for rank in range(world):
        captured = []
        draft = make_draft(rank, lambda pkt, c=captured: (c.append(pkt.clone()), torch.cat((pkt, pkt), dim=-1))[1])
        draft._shortlist_from_local_logits(shards[rank])
        packets.append(captured[0])
    gathered_packet = torch.cat(packets, dim=-1)
    got_u, got_i = make_draft(0, _FixedGather(gathered_packet))._shortlist_from_local_logits(shards[0])

    ref_scores = full_bf16.float() * multiplier
    if softcap is not None:
        ref_scores = (ref_scores / softcap).tanh() * softcap
    ref_u, _ = ref_scores.topk(top_k, dim=-1, sorted=True)
    sorted_u, order = got_u.view(tokens, top_k).sort(dim=-1, descending=True)
    sorted_i = got_i.view(tokens, top_k).gather(-1, order)
    assert torch.equal(sorted_u, ref_u)
    for row in range(tokens):
        assert int(sorted_i[row].min()) >= 0 and int(sorted_i[row].max()) < vocab
        assert len(set(sorted_i[row].tolist())) == top_k
    assert torch.equal(ref_scores.gather(-1, sorted_i), sorted_u)


@pytest.mark.parametrize("selector", [
    None,
    SimpleNamespace(top_k=65, score_shortlist=lambda *args: None),
    SimpleNamespace(top_k=16),
])
def test_parallel_draft_rejects_shortlist_output_without_direct_shortlist(selector):
    """shortlist_only draft output fails during preparation when direct shortlist scoring is unavailable."""
    from hyperflare_recipes.components.parallel_draft import PreparedParallelDraftRound
    from ignition.components.sampling import DeviceRequestState

    device = torch.device("cpu")
    state = DeviceRequestState.allocate(device, vocab_size=128, capacity=2)
    round_op = object.__new__(PreparedParallelDraftRound)
    round_op.operations = None
    round_op.vocabulary_parallel = False
    round_op.proposal_only = True
    round_op.generate_proposals = True
    round_op.batch, round_op.width = 1, 2
    round_op.name = "test"
    round_op.draft = SimpleNamespace(shortlist_output=True)
    round_op.selector = selector
    round_op.shortlisted_selector = selector
    round_op.markov = None
    round_op.draft_inputs = torch.zeros((1, 2), dtype=torch.int64, device=device)
    round_op.drafts = torch.zeros((1, 1), dtype=torch.int64, device=device)
    round_op.logits = torch.zeros((1, 1, 128), dtype=torch.float32, device=device)
    round_op.seed = torch.zeros(1, dtype=torch.int64, device=device)
    round_op.offset = torch.zeros(1, dtype=torch.int64, device=device)
    round_op.sampled = torch.zeros(1, dtype=torch.int64, device=device)
    round_op.owner = SimpleNamespace(
        requests=state,
        weights=SimpleNamespace(draft=SimpleNamespace()),
        inputs=SimpleNamespace(
            row_slot_ids=torch.zeros(2, dtype=torch.int32, device=device),
            row_generations=torch.zeros(2, dtype=torch.int64, device=device),
            live_rows=torch.zeros(1, dtype=torch.int32, device=device),
            query_lengths=torch.ones(2, dtype=torch.int32, device=device),
        ),
    )
    with pytest.raises(ValueError, match="direct shortlist proposal scoring"):
        round_op.prepare()


def test_score_shortlist_clamps_padded_out_of_bounds_ids():
    """Padded out-of-vocabulary shortlist IDs do not index past successor weights and score as -inf."""
    from hyperflare_recipes.components.candidate_selector import PreparedCandidateSelector
    from hyperflare_recipes.components.dflash2 import PreparedDraft

    batch, block, top_k, vocab, local_vocab, world, rank_dim = 2, 2, 16, 100, 64, 2, 8
    tokens = batch * block
    gen = torch.Generator().manual_seed(411)
    # Row 0 has only 4 finite real logits and row 1 is entirely -inf, so padded columns (id >= vocab)
    # tie at -inf and enter the merged top-k shortlist.
    padded = torch.full((tokens, local_vocab * world), -torch.inf, dtype=torch.bfloat16)
    padded[0, :4] = torch.tensor([1.5, -0.5, 2.0, 0.25], dtype=torch.bfloat16)
    padded[2, :2] = torch.tensor([0.75, -1.25], dtype=torch.bfloat16)
    shards = padded.chunk(world, dim=-1)
    words_per_rank = ((top_k * 4 + 63) // 64) * 64 // 2

    def make_draft(rank, gather):
        draft = object.__new__(PreparedDraft)
        draft.shortlist_output = True
        draft.top_k, draft.world_size = top_k, world
        draft.block_size, draft.vocabulary_size = block, vocab
        draft.multiplier, draft.softcap = 1.0, None
        draft.vocab_offset = torch.tensor(rank * local_vocab, dtype=torch.int32)
        draft.shortlist_cols = words_per_rank * 2
        draft.shortlist_gather = gather
        return draft

    packets = []
    for rank in range(world):
        captured = []
        draft = make_draft(rank, lambda pkt, c=captured: (c.append(pkt.clone()), torch.cat((pkt, pkt), dim=-1))[1])
        draft._shortlist_from_local_logits(shards[rank])
        packets.append(captured[0])
    unary, ids = make_draft(0, _FixedGather(torch.cat(packets, dim=-1)))._shortlist_from_local_logits(shards[0])
    assert bool((ids >= vocab).any())

    selector = object.__new__(PreparedCandidateSelector)
    selector.top_k = top_k
    selector.weights = {
        "predecessor": torch.randn((vocab, rank_dim), generator=gen, dtype=torch.bfloat16),
        "successor": torch.randn((vocab, rank_dim), generator=gen, dtype=torch.bfloat16),
    }
    features = torch.randn((batch, block, rank_dim), generator=gen, dtype=torch.bfloat16)
    predecessor = torch.tensor([3, 7], dtype=torch.int64)
    # Include an explicit negative ID as well as padded IDs >= vocab.
    ids_with_neg = ids.clone()
    ids_with_neg[1, 1, 0] = -1
    got = selector.score_shortlist(predecessor, 1, features, unary, ids_with_neg)

    row_ids = ids_with_neg[:, 1]
    in_bounds = (row_ids >= 0) & (row_ids < vocab)
    assert bool((~in_bounds).any())
    assert torch.all(got[~in_bounds] == -torch.inf)
    coded = selector.weights["predecessor"][predecessor] * features[:, 1]
    for b_idx in range(batch):
        for k_idx in range(top_k):
            if in_bounds[b_idx, k_idx]:
                tok = int(row_ids[b_idx, k_idx])
                expected = unary[b_idx, 1, k_idx] + torch.dot(
                    selector.weights["successor"][tok], coded[b_idx]
                ).float()
                assert torch.equal(got[b_idx, k_idx], expected)


def test_parallel_draft_prepare_binds_2d_shortlist_scores_example(monkeypatch):
    """Direct shortlist preparation binds a contiguous 2D score example matching score_shortlist output."""
    from hyperflare_recipes.components import draft_sampling, parallel_draft
    from ignition.components.sampling import DeviceRequestState

    device = torch.device("cpu")
    batch, width, block, top_k, vocab = 2, 3, 8, 12, 128
    state = DeviceRequestState.allocate(device, vocab_size=vocab, capacity=2)
    bound = []

    class FakeProposal:
        def __init__(self, *args, **kwargs):
            pass

        def bind_shortlist_scores(self, scores, ids):
            bound.append((tuple(scores.shape), tuple(scores.stride()), tuple(ids.shape), tuple(ids.stride())))

        def prepare(self):
            pass

        def functional(self, **kwargs):
            return self

    monkeypatch.setattr(parallel_draft, "PreparedDraftDistribution", FakeProposal)
    monkeypatch.setattr(parallel_draft, "prepare_sampling_broadcasts", lambda *args, **kwargs: None)

    round_op = object.__new__(parallel_draft.PreparedParallelDraftRound)
    round_op.operations = None
    round_op.vocabulary_parallel = False
    round_op.proposal_only = True
    round_op.generate_proposals = True
    round_op.batch, round_op.width = batch, width
    round_op.name = "test"
    round_op.draft = SimpleNamespace(shortlist_output=True)
    selector = SimpleNamespace(top_k=top_k, score_shortlist=lambda *args: None)
    round_op.selector = round_op.shortlisted_selector = selector
    round_op.markov = None
    round_op.context = None
    round_op.full_proposal = None
    round_op.proposal_block = None
    round_op.validator = SimpleNamespace(functional=lambda: None)
    round_op.validation = SimpleNamespace(functional=lambda **kwargs: None)
    round_op.draft.release_outputs = lambda: None
    round_op.draft_inputs = torch.zeros((batch, block), dtype=torch.int64, device=device)
    round_op.drafts = torch.zeros((batch, width - 1), dtype=torch.int64, device=device)
    round_op.logits = torch.zeros((batch, 1, vocab), dtype=torch.float32, device=device)
    round_op.seed = torch.zeros(batch, dtype=torch.int64, device=device)
    round_op.offset = torch.zeros(batch, dtype=torch.int64, device=device)
    round_op.sampled = torch.zeros(batch, dtype=torch.int64, device=device)
    round_op.owner = SimpleNamespace(
        requests=state,
        session=None,
        logit_offset=1,
        collectives={"runtime": None, "vocabulary_runtime": None, "coordinator": None},
        weights=SimpleNamespace(draft=SimpleNamespace()),
        inputs=SimpleNamespace(
            row_slot_ids=torch.zeros(batch, dtype=torch.int32, device=device),
            row_generations=torch.zeros(batch, dtype=torch.int64, device=device),
            live_rows=torch.zeros(1, dtype=torch.int32, device=device),
            query_lengths=torch.full((batch,), width, dtype=torch.int32, device=device),
        ),
    )
    round_op.prepare()
    assert len(bound) == width - 1
    for score_shape, score_stride, id_shape, id_stride in bound:
        assert score_shape == (batch, top_k)
        assert score_stride == (top_k, 1)
        assert id_shape == (batch, top_k)
        assert id_stride == (block * top_k, 1)
    kernel = draft_sampling._draft_shortlist_score_candidates
    assert {"candidate_stride", "score_stride", "id_stride"}.issubset(set(kernel.do_not_specialize))
