# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""In-graph asynchronous conditional grammar mask exchange for learned speculation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_grammar_drafts(
    Drafts,
    Slots,
    Active,
    Mailbox,
    CAPACITY: tl.constexpr,
    MAX_K: tl.constexpr,
    K: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(Active + row) != 0:
        slot = tl.load(Slots + row).to(tl.int64)
        if (slot >= 0) & (slot < CAPACITY):
            for k in tl.static_range(K):
                tl.store(Mailbox + slot * MAX_K + k, tl.load(Drafts + row * K + k))


@triton.jit
def _publish_grammar_sequence(Mailbox, Sequence, FLAG: tl.constexpr):
    value = tl.load(Sequence)
    tl.atomic_xchg(Mailbox + FLAG, value, sem="release", scope="sys")


@torch.library.custom_op(
    "flare::publish_grammar_drafts",
    mutates_args=("mailbox", "candidates"),
)
def publish_grammar_drafts(
    drafts: torch.Tensor,
    slots: torch.Tensor,
    active: torch.Tensor,
    mailbox: torch.Tensor,
    sequence: torch.Tensor,
    candidates: torch.Tensor,
    capacity: int,
    max_k: int,
) -> None:
    b, k = drafts.shape
    if k > 0:
        _copy_grammar_drafts[(b,)](
            drafts,
            slots,
            active,
            mailbox,
            CAPACITY=capacity,
            MAX_K=max_k,
            K=k,
            num_warps=1,
        )
    _publish_grammar_sequence[(1,)](mailbox, sequence, FLAG=mailbox.numel() - 1, num_warps=1)


@publish_grammar_drafts.register_fake
def _publish_grammar_drafts_fake(
    drafts, slots, active, mailbox, sequence, candidates, capacity, max_k
):
    pass


@torch.library.custom_op(
    "flare::wait_grammar_ready",
    mutates_args=("masks", "mask_count", "verify_extent", "emit_cap"),
)
def wait_grammar_ready(
    dependency: torch.Tensor,
    mailbox: torch.Tensor,
    masks: torch.Tensor,
    mask_count: torch.Tensor,
    verify_extent: torch.Tensor,
    emit_cap: torch.Tensor,
    event: int,
) -> None:
    from cuda.bindings import driver

    stream = torch.cuda.current_stream(dependency.device)
    flags = int(torch.cuda.is_current_stream_capturing())
    (status,) = driver.cuStreamWaitEvent(
        driver.CUstream(stream.cuda_stream), driver.CUevent(event), flags
    )
    if int(status):
        raise RuntimeError(f"grammar readiness wait failed: {status}")


@wait_grammar_ready.register_fake
def _wait_grammar_ready_fake(
    dependency, mailbox, masks, mask_count, verify_extent, emit_cap, event
):
    pass


class GrammarExchange:
    """Device and mapped-host storage for per-round conditional grammar masks."""

    def __init__(
        self,
        *,
        capacity: int,
        verification_width: int,
        vocab_size: int,
        mailbox,
        ready: torch.cuda.Event,
        masks: torch.Tensor,
        mask_count: torch.Tensor,
        verify_extent: torch.Tensor,
        emit_cap: torch.Tensor,
    ):
        self.capacity = capacity
        self.verification_width = verification_width
        self.max_drafts = verification_width - 1
        self.vocab_size = vocab_size
        self.mailbox = mailbox
        self.mailbox_view = mailbox.device_view
        self.ready = ready
        self.handle = int(ready.cuda_event)
        self.masks = masks
        self.mask_count = mask_count
        self.verify_extent = verify_extent
        self.emit_cap = emit_cap

    @classmethod
    def allocate(
        cls,
        capacity: int,
        *,
        verification_width: int,
        vocab_size: int,
        device,
        host_allocator=None,
    ):
        from b12x.sequence._shared.disk_table import MappedHostAllocation

        if verification_width < 2:
            raise ValueError("grammar exchange requires verification_width >= 2")
        device = torch.device(device)
        max_drafts = verification_width - 1
        words = (vocab_size + 31) // 32
        shape = (capacity * max_drafts + 1,)
        mailbox = (
            MappedHostAllocation(shape, torch.int64, device)
            if host_allocator is None
            else host_allocator("grammar_mailbox", shape, torch.int64)
        )
        try:
            mailbox.host_view.zero_()
            ready = torch.cuda.Event(external=True)
            ready.record(torch.cuda.current_stream(device))
            masks = torch.zeros(
                (capacity, verification_width, words), dtype=torch.int32, device=device
            )
            mask_count = torch.zeros((capacity,), dtype=torch.int32, device=device)
            verify_extent = torch.zeros((capacity,), dtype=torch.int32, device=device)
            emit_cap = torch.zeros((capacity,), dtype=torch.int32, device=device)
            return cls(
                capacity=capacity,
                verification_width=verification_width,
                vocab_size=vocab_size,
                mailbox=mailbox,
                ready=ready,
                masks=masks,
                mask_count=mask_count,
                verify_extent=verify_extent,
                emit_cap=emit_cap,
            )
        except BaseException:
            mailbox.close()
            raise

    def publish(
        self,
        drafts: torch.Tensor,
        slots: torch.Tensor,
        active: torch.Tensor,
        sequence: torch.Tensor,
        candidates: torch.Tensor,
    ) -> None:
        publish_grammar_drafts(
            drafts,
            slots,
            active,
            self.mailbox_view,
            sequence,
            candidates,
            self.capacity,
            self.max_drafts,
        )

    def wait(self, dependency: torch.Tensor) -> None:
        wait_grammar_ready(
            dependency,
            self.mailbox_view,
            self.masks,
            self.mask_count,
            self.verify_extent,
            self.emit_cap,
            self.handle,
        )

    def buffers(self):
        return dict(
            grammar_mailbox=self.mailbox.device_view,
            grammar_exchange_masks=self.masks,
            grammar_exchange_mask_count=self.mask_count,
            grammar_exchange_verify_extent=self.verify_extent,
            grammar_exchange_emit_cap=self.emit_cap,
        )

    def events(self):
        return {"grammar_ready": self.ready.cuda_event}

    def zero_ranges(self):
        return tuple((t.data_ptr(), t.nbytes) for t in self.buffers().values())

    def close(self):
        self.mailbox.close()
