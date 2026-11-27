# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct shortlist-scored draft proposals and token-limited target distributions."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (os.environ.get("FLARE_RUN_CUDA_TESTS") and torch.cuda.is_available()),
    reason="set FLARE_RUN_CUDA_TESTS=1 on a CUDA host",
)


@pytest.mark.parametrize("batch,vocab,prefix", [(1, 248320, 7), (4, 10000, 3), (8, 257, 0)])
@pytest.mark.parametrize("top_k", [0, 1, 8, 20])
@pytest.mark.parametrize("write_dense", [True, False])
@torch.no_grad()
def test_shortlist_scores_match_dense_reference(batch, vocab, prefix, top_k, write_dense):
    from hyperflare_recipes.components.draft_sampling import (
        DRAFT_CANDIDATES,
        PreparedDraftBlockBroadcast,
        PreparedDraftDistribution,
    )
    from ignition.components.sampling import DeviceRequestState, RUNNING

    device = torch.device("cuda:0")
    generator = torch.Generator().manual_seed(917 + prefix + batch)
    state = DeviceRequestState.allocate(device, vocab_size=vocab, capacity=max(4, batch))
    state.generation.fill_(7)
    state.status.fill_(RUNNING)
    state.histogram.copy_(torch.randint(0, 4, state.histogram.shape, generator=generator, dtype=torch.int32))
    state.parameters.copy_(torch.tensor([1., .9, .02, 1., 0., 0.], device=device))
    state.top_k.fill_(top_k)
    state.grammar_mask.fill_(-1)
    state.grammar_enabled.zero_()

    slots = torch.arange(batch, dtype=torch.int32, device=device)
    epochs = torch.full((batch,), 7, dtype=torch.int64, device=device)
    live = torch.tensor([batch], dtype=torch.int32, device=device)
    prefix_ids = torch.full((batch, max(1, prefix)), 7, dtype=torch.int64, device=device)
    support = min(16, vocab)
    ids = torch.empty((batch, support), dtype=torch.int64, device=device)
    for row in range(batch):
        ids[row].copy_(torch.cat((torch.tensor([0, 1]),
                                  torch.randperm(vocab - 2, generator=generator)[:support - 2] + 2)))
    values = torch.randn((batch, support), generator=generator, dtype=torch.float32).to(device)
    initial_ids, initial_values = ids.clone(), values.clone()
    logits = torch.empty((batch, 1, vocab), dtype=torch.float32, device=device)
    seed = torch.arange(batch, dtype=torch.int64, device=device) + 331
    offset = torch.zeros_like(seed)

    ref_sampled = torch.zeros_like(seed)
    ref = PreparedDraftDistribution(logits, state, slots, epochs, live, prefix_ids,
        seed=seed, offset=offset, sampled=ref_sampled, prefix_offset=prefix)
    ref.prepare()

    cand_sampled = torch.zeros_like(seed)
    cand = PreparedDraftDistribution(logits, state, slots, epochs, live, prefix_ids,
        seed=seed, offset=offset, sampled=cand_sampled, prefix_offset=prefix)
    cand.bind_shortlist_scores(values, ids)
    cand.prepare()
    block = None
    if not write_dense:
        drafts_ex = torch.zeros((batch, 1), dtype=torch.int64, device=device)
        probs_ex = torch.zeros((batch, 1, vocab), dtype=torch.float32, device=device)
        valid_ex = torch.zeros((batch, 1), dtype=torch.bool, device=device)
        block = PreparedDraftBlockBroadcast((cand,), drafts_ex, probs_ex, valid_ex, cand_sampled,
            runtime=None, session=None, coordinator=None, name="shortlist_block")
    cand.functional()

    modes = ("neutral", "penalties", "grammar", "ties", "signed_zero", "tiny_temperature",
             "nonfinite", "all_masked", "invalid_prefix", "inactive",
             "duplicate_or_stale", "token_zero", "mutable_ids")
    for phase, mode in enumerate(modes):
        values.copy_(initial_values)
        ids.copy_(initial_ids)
        state.status.fill_(RUNNING)
        state.parameters.copy_(torch.tensor([1., .9, .02, 1., 0., 0.], device=device))
        state.grammar_enabled.zero_()
        slots.copy_(torch.arange(batch, dtype=torch.int32, device=device))
        epochs.fill_(7)
        live.fill_(batch)
        prefix_ids.fill_(7)
        offset.fill_(phase * 4)
        if mode == "penalties":
            state.parameters.copy_(torch.tensor([.7, .8, .1, 1.3, -.2, -.1], device=device))
        elif mode in ("grammar", "all_masked"):
            state.grammar_enabled.fill_(1)
            state.grammar_mask.copy_(torch.randint(-2**31, 2**31 - 1, state.grammar_mask.shape,
                generator=generator, dtype=torch.int64).to(torch.int32))
            if mode == "all_masked":
                state.grammar_mask.zero_()
        elif mode in ("ties", "signed_zero"):
            values.zero_()
            values[:, 1::2] = -0. if mode == "signed_zero" else 1.
            if mode == "signed_zero":
                state.parameters[:, 4:] = -0.
        elif mode == "tiny_temperature":
            state.parameters[:, 0] = torch.finfo(torch.float32).tiny
        elif mode == "nonfinite":
            values[:, :3] = torch.tensor([float("nan"), float("inf"), -float("inf")], device=device)
        elif mode == "invalid_prefix":
            prefix_ids[:, 0] = -1
        elif mode == "inactive":
            live.fill_(batch // 2)
        elif mode == "duplicate_or_stale":
            epochs[-1] = 99
            if batch > 1:
                slots[-1] = slots[0]
        elif mode == "token_zero":
            values.fill_(-10.)
            values[:, 0] = 2.0
            values[:, 1] = 1.0
        elif mode == "mutable_ids":
            ids.copy_(initial_ids.roll(5, dims=1))

        logits.fill_(-torch.inf)
        logits[:, 0].scatter_(1, ids, values)
        ref()

        scratch = {k: torch.empty(sh, dtype=dt, device=device) for k, (sh, dt) in cand.layout.items()}
        if block is not None:
            bstate = block.begin(device)
            scratch["support_ids"] = bstate["support_ids"][0]
            scratch["support_probabilities"] = bstate["support_probabilities"][0]
        got_probs = cand.run(scratch, None, shortlist_scores=values, shortlist_ids=ids,
                             candidates=prefix_ids, seed=seed, offset=offset, sampled=cand_sampled)
        if block is not None:
            drafts_out = torch.zeros((batch, 1), dtype=torch.int64, device=device)
            probs_out = torch.zeros((batch, 1, vocab), dtype=torch.float32, device=device)
            valid_out = torch.zeros((batch, 1), dtype=torch.bool, device=device)
            block.retain(bstate, 0, scratch, cand_sampled)
            block.finish(bstate, drafts_out, probs_out, valid_out, cand_sampled,
                         torch.ones(batch, dtype=torch.bool, device=device),
                         torch.full((batch,), 2, dtype=torch.int32, device=device))
            got_probs = probs_out
            got_valid = valid_out
        else:
            got_valid = scratch["valid"]
        torch.cuda.synchronize()
        assert torch.equal(cand_sampled, ref_sampled), mode
        assert torch.equal(got_valid, ref.valid), mode
        assert torch.equal(got_probs.view(torch.int32), ref.probabilities.view(torch.int32)), mode


@pytest.mark.parametrize("batch,width,vocab", [(4, 8, 257), (8, 8, 248320)])
@torch.no_grad()
def test_target_distributions_mask_unused_columns_beyond_token_limits(batch, width, vocab):
    from hyperflare_recipes.components.speculation import (
        PreparedSpeculativeDistributions,
        PreparedStochasticVerification,
        SpeculativeRngState,
    )
    from ignition.components.sampling import DeviceRequestState, RUNNING

    device = torch.device("cuda:0")
    state = DeviceRequestState.allocate(device, vocab_size=vocab, capacity=batch)
    state.generation.fill_(7)
    state.status.fill_(RUNNING)
    state.parameters.copy_(torch.tensor([0.7, 0.9, 0.0, 1.0, 0.0, 0.0], device=device))
    state.top_k.zero_()
    slots = torch.arange(batch, dtype=torch.int32, device=device)
    epochs = torch.full((batch,), 7, dtype=torch.int64, device=device)
    live = torch.tensor([batch], dtype=torch.int32, device=device)
    drafts = torch.randint(0, vocab, (batch, width - 1), dtype=torch.int64, device=device)
    limits = torch.tensor([max(1, width - (row % width)) for row in range(batch)], dtype=torch.int32, device=device)

    torch.manual_seed(11)
    logits = torch.randn((batch, width, vocab), device=device, dtype=torch.bfloat16) * 5.0
    for row in range(batch):
        logits[row, int(limits[row]):, :] = 0.0

    unmasked = PreparedSpeculativeDistributions(logits, state, slots, epochs, live, drafts)
    unmasked.prepare()
    masked = PreparedSpeculativeDistributions(logits, state, slots, epochs, live, drafts, token_limits=limits)
    masked.prepare()
    unmasked()
    masked()
    torch.cuda.synchronize()

    for row in range(batch):
        lim = int(limits[row])
        assert torch.equal(masked.valid[row, :lim], unmasked.valid[row, :lim])
        assert torch.equal(masked.probabilities[row, :lim].view(torch.int32),
                           unmasked.probabilities[row, :lim].view(torch.int32))
        assert not masked.valid[row, lim:].any()
        if lim < width:
            expected_dummy = (torch.arange(vocab, device=device) == 0).float().expand(width - lim, -1)
            assert torch.equal(masked.probabilities[row, lim:], expected_dummy)

    draft_probs = torch.zeros((batch, width - 1, vocab), device=device, dtype=torch.float32)
    draft_probs.scatter_(2, drafts.unsqueeze(-1), 1.0)
    draft_valid = torch.ones((batch, width - 1), dtype=torch.bool, device=device)
    v_ref = PreparedStochasticVerification(draft_probs, drafts, unmasked.probabilities, state, slots, epochs, live,
        rng=SpeculativeRngState.allocate(device, capacity=batch), draft_valid=draft_valid,
        target_valid=unmasked.valid, token_limits=limits)
    v_new = PreparedStochasticVerification(draft_probs, drafts, masked.probabilities, state, slots, epochs, live,
        rng=SpeculativeRngState.allocate(device, capacity=batch), draft_valid=draft_valid,
        target_valid=masked.valid, token_limits=limits)
    v_ref.prepare()
    v_new.prepare()
    v_ref()
    v_new()
    torch.cuda.synchronize()
    for field in ("token_ids", "proposed_count", "accepted_drafts", "offered_drafts", "valid", "rng_increment", "survival"):
        actual, expected = getattr(v_new.output, field), getattr(v_ref.output, field)
        if actual.dtype.is_floating_point:
            assert torch.equal(actual.view(torch.int32), expected.view(torch.int32)), field
        else:
            assert torch.equal(actual, expected), field


@pytest.mark.parametrize("batch,vocab", [(1, 257), (4, 10000)])
@torch.no_grad()
def test_block_broadcast_without_dense_write_matches_bind_logits_reference(batch, vocab):
    """PreparedDraftBlockBroadcast with write_dense=False matches dense bind_logits proposals."""
    from hyperflare_recipes.components.draft_sampling import (
        PreparedDraftBlockBroadcast,
        PreparedDraftDistribution,
    )
    from ignition.components.sampling import DeviceRequestState, RUNNING

    device = torch.device("cuda:0")
    generator = torch.Generator().manual_seed(613 + batch)
    state = DeviceRequestState.allocate(device, vocab_size=vocab, capacity=max(4, batch))
    state.generation.fill_(7)
    state.status.fill_(RUNNING)
    state.histogram.copy_(torch.randint(0, 3, state.histogram.shape, generator=generator, dtype=torch.int32))
    state.parameters.copy_(torch.tensor([0.8, 0.9, 0.02, 1.15, 0.1, 0.05], device=device))
    state.top_k.fill_(17)
    state.grammar_mask.fill_(-1)
    state.grammar_enabled.zero_()

    slots = torch.arange(batch, dtype=torch.int32, device=device)
    epochs = torch.full((batch,), 7, dtype=torch.int64, device=device)
    live = torch.tensor([batch], dtype=torch.int32, device=device)
    prefix_ids = torch.zeros((batch, 2), dtype=torch.int64, device=device)
    source_logits = torch.randn((batch, vocab), generator=generator, dtype=torch.bfloat16).to(device)
    source_bias = torch.randn((batch, vocab), generator=generator, dtype=torch.float32).to(device) * 0.25
    staging = torch.empty((batch, 1, vocab), dtype=torch.float32, device=device)
    seed = torch.arange(batch, dtype=torch.int64, device=device) + 409
    offset = torch.zeros_like(seed)

    ref_sampled = torch.zeros_like(seed)
    ref = PreparedDraftDistribution(staging, state, slots, epochs, live, prefix_ids,
        seed=seed, offset=offset, sampled=ref_sampled, prefix_offset=1)
    ref.bind_logits(source_logits, source_bias)
    ref.prepare()

    cand_sampled = torch.zeros_like(seed)
    cand = PreparedDraftDistribution(staging, state, slots, epochs, live, prefix_ids,
        seed=seed, offset=offset, sampled=cand_sampled, prefix_offset=1)
    cand.bind_logits(source_logits, source_bias)
    cand.prepare()
    drafts_ex = torch.zeros((batch, 1), dtype=torch.int64, device=device)
    probs_ex = torch.zeros((batch, 1, vocab), dtype=torch.float32, device=device)
    valid_ex = torch.zeros((batch, 1), dtype=torch.bool, device=device)
    block = PreparedDraftBlockBroadcast((cand,), drafts_ex, probs_ex, valid_ex, cand_sampled,
        runtime=None, session=None, coordinator=None, name="bind_logits_block")
    cand.functional()

    for mode in ("default", "ties", "token_zero", "inactive", "invalid"):
        state.status.fill_(0 if mode == "invalid" else RUNNING)
        live.fill_(batch // 2 if mode == "inactive" else batch)
        if mode == "ties":
            source_logits.zero_()
            source_bias.zero_()
        elif mode == "token_zero":
            source_logits.fill_(-10.0)
            source_logits[:, 0] = 2.0
            source_logits[:, 3] = 1.0
            source_bias.zero_()
        ref()
        scratch = {k: torch.empty(sh, dtype=dt, device=device) for k, (sh, dt) in cand.layout.items()}
        bstate = block.begin(device)
        scratch["support_ids"] = bstate["support_ids"][0]
        scratch["support_probabilities"] = bstate["support_probabilities"][0]
        cand.run(scratch, staging, source_logits=source_logits, source_bias=source_bias,
                 candidates=prefix_ids, seed=seed, offset=offset, sampled=cand_sampled)
        drafts_out = torch.zeros((batch, 1), dtype=torch.int64, device=device)
        probs_out = torch.zeros((batch, 1, vocab), dtype=torch.float32, device=device)
        valid_out = torch.zeros((batch, 1), dtype=torch.bool, device=device)
        block.retain(bstate, 0, scratch, cand_sampled)
        block.finish(bstate, drafts_out, probs_out, valid_out, cand_sampled,
                     torch.ones(batch, dtype=torch.bool, device=device),
                     torch.full((batch,), 2, dtype=torch.int32, device=device))
        torch.cuda.synchronize()
        assert torch.equal(cand_sampled, ref_sampled), mode
        assert torch.equal(valid_out, ref.valid), mode
        assert torch.equal(probs_out.view(torch.int32), ref.probabilities.view(torch.int32)), mode
