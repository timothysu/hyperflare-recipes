# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Ling-3.2-Air Layer-42 MTP causal proposals with KDA target verification commits."""

from hyperflare_recipes.components.causal_draft import PreparedCausalDraftRound

from .decoder import PreparedDecoder
from .mtp import PreparedAcceptedContext, PreparedProposals


class PreparedRound(PreparedCausalDraftRound):
    """Compose Layer-42 MTP drafting, fused KDA verification, KDA commit, and MTP KV absorb."""

    def __init__(self, owner, batch: int, width: int, name: str):
        def decoder_factory(weights, scratch, metadata, pool, *, mode, name, overlap, **common):
            return PreparedDecoder(
                weights,
                scratch,
                metadata,
                pool,
                mode=mode,
                name=name,
                overlap=overlap,
                kda_records=owner.kda_records if mode == "verify" else None,
                examples=owner.examples,
                **common,
            )

        def context_factory(weights, metadata, pool, **common):
            return PreparedAcceptedContext(
                weights,
                metadata,
                pool,
                draft_tables=owner.draft_tables,
                inputs=owner.inputs,
                examples=owner.examples,
                name=name + ".absorb",
                is_prefill=False,
                **common,
            )

        def proposal_factory(weights, pool, draft_scratch, batch_size, depth, *, name, **common):
            del draft_scratch
            return PreparedProposals(
                weights,
                pool,
                batch_size,
                depth,
                draft_tables=owner.draft_tables,
                inputs=owner.inputs,
                requests=owner.requests,
                examples=owner.examples,
                name=name,
                **common,
            )

        super().__init__(
            owner,
            batch,
            width,
            name,
            decoder_factory=decoder_factory,
            context_factory=context_factory,
            proposal_factory=proposal_factory,
            draft_context_lag=1,
        )
