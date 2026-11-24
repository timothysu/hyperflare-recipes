# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for causal-round operand wiring with substituted GPU operators."""

from types import SimpleNamespace

import pytest
import torch

from hyperflare_recipes.components import causal_draft


@pytest.mark.parametrize("batch", (1, 2))
@pytest.mark.parametrize("width", (1, 3))
def test_verification_receives_complete_probabilities_and_validity(monkeypatch, batch, width):
    vocabulary, hidden, capacity = 7, 4, batch + 2
    slots = torch.arange(batch - 1, -1, -1, dtype=torch.int32)
    operation = causal_draft.PreparedCausalDraftRound.__new__(causal_draft.PreparedCausalDraftRound)
    operation.batch, operation.width, operation.blocks, operation.hidden = batch, width, 1, hidden
    state_rows = torch.zeros((capacity, 8), dtype=torch.int64)
    state_rows[:, 0] = torch.arange(1, capacity + 1)
    operation.inputs = SimpleNamespace(row_slot_ids=slots, query_lengths=torch.full((batch,), width, dtype=torch.int32),
        request_state_rows=state_rows, page_tables=torch.ones((capacity, 1), dtype=torch.int32),
        live_rows=torch.tensor([batch], dtype=torch.int32))
    operation.requests = SimpleNamespace(capacity=capacity, vocab_size=vocabulary,
        consumed_tokens=torch.arange(capacity) + 10, pending_token=torch.arange(capacity),
        generation=torch.ones(capacity, dtype=torch.int64), seed=torch.arange(capacity))
    operation.draft_offset = torch.zeros(capacity, dtype=torch.int64)
    operation.draft_generation = torch.zeros_like(operation.draft_offset)
    operation.pool = SimpleNamespace(last_hidden=torch.zeros((capacity + 1, hidden)))
    operation.validator = SimpleNamespace(validate=lambda: (torch.zeros(1), torch.ones(capacity)),
        begin=lambda value, *, present: SimpleNamespace(lengths=torch.ones(batch, dtype=torch.int32)))
    operation.validation = lambda value: value
    operation.sampling_layout = {
        "target": {"probabilities": ((batch, width, vocabulary), torch.float32),
                   "valid": ((batch, width), torch.bool)},
        "proposal": {"probabilities": ((batch, 1, vocabulary), torch.float32),
                     "valid": ((batch, 1), torch.bool)},
        "verify": {},
    }
    observed = {}
    auxiliary, records, acceptance = object(), object(), object()
    target_hidden = torch.arange(batch * width * hidden).reshape(batch * width, hidden).float()
    target_logits = torch.arange(batch * width * vocabulary).reshape(batch * width, vocabulary).float()

    class Decoder:
        def __call__(self, metadata):
            observed["metadata"] = metadata
            return target_hidden, target_logits, auxiliary, records

        def commit(self, counts, passed_records, metadata):
            assert passed_records is records and metadata is observed["metadata"]
            observed["decoder_commit"] = counts

    operation.decoder = Decoder()
    operation.draft = SimpleNamespace(begin=lambda states: states,
        step=lambda state, index, *args: (torch.zeros((batch, vocabulary)), state),
        finish=lambda *args: None)

    def propose(scratch, logits, *, sampled, **kwargs):
        sampled.copy_(torch.arange(batch) + 3)
        scratch["probabilities"].fill_(1 / vocabulary)
        scratch["valid"].fill_(True)

    def target(scratch, logits, *, candidates):
        scratch["probabilities"].copy_(logits.softmax(-1))
        scratch["valid"].fill_(True)
        scratch["valid"][-1, -1] = False
        observed["probabilities"], observed["valid"] = scratch["probabilities"], scratch["valid"]
        return scratch["probabilities"]

    def verify(scratch, draft_probabilities, target_probabilities, target_valid, **kwargs):
        assert target_probabilities is observed["probabilities"]
        assert target_probabilities.shape == (batch, width, vocabulary)
        assert target_valid is observed["valid"] and target_valid.dtype == torch.bool
        assert target_valid.shape == (batch, width)
        torch.testing.assert_close(target_probabilities, target_logits.view(batch, width, -1).softmax(-1))
        assert not target_valid[-1, -1]
        return acceptance

    def commit(logits, *, acceptance, inputs):
        assert acceptance is observed["acceptance"]
        torch.testing.assert_close(inputs[:, 0], operation.requests.pending_token[slots.long()])
        return SimpleNamespace(emitted_count=torch.ones(batch, dtype=torch.int64))

    observed["acceptance"] = acceptance
    operation.operations = (SimpleNamespace(run=target),
        tuple(SimpleNamespace(run=propose) for _ in range(width - 1)), SimpleNamespace(run=verify), commit)

    def context(passed_auxiliary, counts, metadata):
        assert passed_auxiliary is auxiliary and metadata is observed["metadata"]
        observed["context_counts"] = counts

    operation.context = context

    def save_hidden(value, destination, rows, active):
        destination.index_copy_(0, rows[active], value[active])
        observed["saved_hidden"] = value

    monkeypatch.setattr(causal_draft, "save_hidden", save_hidden)
    metadata, passed_acceptance, committed = operation()
    assert metadata is observed["metadata"] and passed_acceptance is acceptance
    assert observed["context_counts"] is committed.emitted_count
    torch.testing.assert_close(observed["saved_hidden"], target_hidden.view(batch, width, hidden)[:, 0])
    if width > 1:
        assert observed["decoder_commit"] is committed.emitted_count
    assert metadata.rotary_positions is None


def test_causal_draft_propagates_mixed_restored_rotary_offsets_and_skips_inactive_rows(monkeypatch):
    """Restored nonzero offsets per request feed packed verification while inactive rows stay -1."""
    batch, width, vocabulary, hidden, capacity = 3, 4, 7, 4, 5
    slots = torch.tensor([2, 0, 4], dtype=torch.int32)
    operation = causal_draft.PreparedCausalDraftRound.__new__(causal_draft.PreparedCausalDraftRound)
    operation.batch, operation.width, operation.blocks, operation.hidden = batch, width, 1, hidden
    state_rows = torch.zeros((capacity, 8), dtype=torch.int64)
    state_rows[:, 0] = torch.arange(1, capacity + 1)
    # Row 0 (slot 2): active, consumed=20, restored rotary_offset=-15 -> rotary starts at 5.
    # Row 1 (slot 0): inactive (length 0 after validation) -> contributes no packed positions.
    # Row 2 (slot 4): active, consumed=12, restored rotary_offset=-7, query_limit=2 -> rotary 5, 6.
    consumed = torch.tensor([100, 100, 20, 100, 12], dtype=torch.int64)
    validated_lengths = torch.tensor([1, 0, 1], dtype=torch.int32)
    validated_query_start_loc = torch.tensor([0, 1, 1, 2], dtype=torch.int32)
    validated_starts = torch.tensor([20, 100, 12], dtype=torch.int64)
    validated_positions = torch.tensor([20, 12, -1], dtype=torch.int64)
    validated_rotary = torch.tensor([5, 5, -1], dtype=torch.int64)[:, None]

    operation.inputs = SimpleNamespace(
        row_slot_ids=slots,
        query_lengths=torch.tensor([4, 4, 2], dtype=torch.int32),
        request_state_rows=state_rows,
        page_tables=torch.ones((capacity, 1), dtype=torch.int32),
        live_rows=torch.tensor([2], dtype=torch.int32),
    )
    operation.requests = SimpleNamespace(
        capacity=capacity,
        vocab_size=vocabulary,
        consumed_tokens=consumed,
        pending_token=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64),
        generation=torch.ones(capacity, dtype=torch.int64),
        seed=torch.arange(capacity),
    )
    operation.draft_offset = torch.zeros(capacity, dtype=torch.int64)
    operation.draft_generation = torch.zeros_like(operation.draft_offset)
    operation.pool = SimpleNamespace(last_hidden=torch.zeros((capacity + 1, hidden)))
    operation.validator = SimpleNamespace(
        validate=lambda: (torch.zeros(1), torch.ones(capacity)),
        begin=lambda value, *, present: SimpleNamespace(
            lengths=validated_lengths,
            query_start_loc=validated_query_start_loc,
            starts=validated_starts,
            positions=validated_positions,
            rotary_positions=validated_rotary,
        ),
    )
    operation.validation = lambda value: value
    operation.sampling_layout = {
        "target": {
            "probabilities": ((batch, width, vocabulary), torch.float32),
            "valid": ((batch, width), torch.bool),
        },
        "proposal": {
            "probabilities": ((batch, 1, vocabulary), torch.float32),
            "valid": ((batch, 1), torch.bool),
        },
        "verify": {},
    }
    observed = {}
    target_hidden = torch.arange(batch * width * hidden).reshape(batch * width, hidden).float()
    target_logits = torch.arange(batch * width * vocabulary).reshape(batch * width, vocabulary).float()

    class Decoder:
        def __call__(self, metadata):
            observed["metadata"] = metadata
            return target_hidden, target_logits, "aux", "rec"

        def commit(self, counts, passed_records, metadata):
            observed["commit"] = (counts.clone(), passed_records, metadata)

    operation.decoder = Decoder()
    operation.draft = SimpleNamespace(
        begin=lambda states: states,
        step=lambda state, index, *args: (torch.zeros((batch, vocabulary)), state),
        finish=lambda *args: None,
    )

    def propose(scratch, logits, *, sampled, **kwargs):
        sampled.fill_(2)
        scratch["probabilities"].fill_(1 / vocabulary)
        scratch["valid"].fill_(True)

    def target(scratch, logits, *, candidates):
        scratch["probabilities"].copy_(logits.softmax(-1))
        scratch["valid"].fill_(True)
        return scratch["probabilities"]

    def verify(scratch, *args, **kwargs):
        return "acc"

    # Partial acceptance: row 0 accepts 2 of 4, inactive row 1 accepts 0, row 2 accepts 1 of 2.
    emitted = torch.tensor([2, 0, 1], dtype=torch.int64)
    operation.operations = (
        SimpleNamespace(run=target),
        tuple(SimpleNamespace(run=propose) for _ in range(width - 1)),
        SimpleNamespace(run=verify),
        lambda logits, *, acceptance, inputs: SimpleNamespace(emitted_count=emitted),
    )
    operation.context = lambda aux, counts, meta: observed.__setitem__("context", (aux, counts.clone(), meta))

    def save_hidden(value, destination, rows, active):
        destination.index_copy_(0, rows[active], value[active])
        observed["saved_active"] = active.clone()

    monkeypatch.setattr(causal_draft, "save_hidden", save_hidden)
    metadata, _, committed = operation()

    # Packed tokens: 4 from row 0 (logical 20..23, rotary 5..8), 0 from row 1,
    # 2 from row 2 (logical 12..13, rotary 5..6), remaining 6 inactive (-1).
    expected_pos = torch.tensor([20, 21, 22, 23, 12, 13, -1, -1, -1, -1, -1, -1], dtype=torch.int64)
    expected_rotary = torch.tensor([5, 6, 7, 8, 5, 6, -1, -1, -1, -1, -1, -1], dtype=torch.int64)[:, None]
    assert torch.equal(metadata.positions, expected_pos)
    assert torch.equal(metadata.rotary_positions, expected_rotary)
    assert torch.equal(observed["commit"][0], emitted)
    assert observed["context"][2] is metadata
    assert torch.equal(observed["saved_active"], torch.tensor([True, False, True]))
    assert torch.equal(operation.pool.last_hidden[state_rows[0, 0]], torch.zeros(hidden))
    assert committed.emitted_count is emitted


def test_rotary_batch_advance_axis_controls_post_prefill_offset():
    """RotaryBatch advances from all axes by default or from one explicit axis when configured."""
    from hyperflare_recipes.components.paged_batch import Metadata, RotaryBatch

    def _run(advance_axis):
        region = torch.zeros((4, 1), dtype=torch.int64)
        batch = RotaryBatch.__new__(RotaryBatch)
        batch.advance_axis = advance_axis
        batch.rotary_offset = region
        metadata = Metadata(
            token_ids=torch.zeros(6, dtype=torch.int64),
            sample_indices=torch.tensor([5], dtype=torch.int64),
            query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
            state_rows=torch.tensor([2], dtype=torch.int64),
            starts=torch.tensor([0], dtype=torch.int64),
            lengths=torch.tensor([6], dtype=torch.int32),
            requests=torch.zeros(6, dtype=torch.int64),
            positions=torch.arange(6, dtype=torch.int64),
            page_table=torch.zeros((1, 1), dtype=torch.int32),
            token_page_table=torch.zeros((6, 1), dtype=torch.int32),
            live_counts=torch.tensor([6, 1], dtype=torch.int32),
            max_query_tokens=6,
            rotary_positions=torch.tensor(
                [[0, 0, 0], [1, 1, 1], [2, 10, 12], [2, 10, 14], [3, 3, 3], [4, 4, 4]],
                dtype=torch.int64,
            ),
        )
        batch.commit_rotary(metadata)
        return int(region[2, 0].item())

    # Default (all axes): max is 14 -> next rotary 15, offset = 15 - 6 = 9.
    assert _run(None) == 9
    # Axis 0 (temporal axis): max is 4 -> next rotary 5, offset = 5 - 6 = -1.
    assert _run(0) == -1
    with pytest.raises(ValueError, match="advance_axis"):
        RotaryBatch(
            SimpleNamespace(page_tables=torch.zeros((1, 1)), row_slot_ids=torch.zeros(1),
                            row_generations=torch.zeros(1), live_rows=torch.zeros(1)),
            SimpleNamespace(capacity=1, vocab_size=8),
            Metadata.allocate(1, 1, 1, torch.device("cpu")),
            mode="decode",
            shard=SimpleNamespace(world_size=1),
            pool=SimpleNamespace(),
            advance_axis=3,
        )
