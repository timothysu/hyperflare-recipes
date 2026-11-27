# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unrotated grouped-query attention with paged BF16 or FP8 history."""

import torch

from hyperflare_recipes.components.linear import prepare_projection
from hyperflare_recipes.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from ignition.components.collectives import PreparedAllReduce

from ignition.preparation.examples import declaration_tensor

from .config import PAGE_TOKENS


class PreparedAttention:
    def __init__(self, weights, source, metadata, key_cache, value_cache, config, *, mode,
                 session, name, workspace, shard=None, runtime=None, coordinator=None,
                 key_scale=None, value_scale=None, tuning_page_count=None, examples=None):
        c, tokens = config.text, source.shape[0]
        heads = config.local_attention_heads(shard)
        groups, dim = config.local_kv_heads(shard), c["head_dim"]

        def empty(role, shape, dtype=torch.bfloat16):
            return declaration_tensor(examples, "nemotron.target.attention." + role, shape, dtype, source.device)

        projected = empty("projected", (tokens, (heads + 2 * groups) * dim))
        self.projection = prepare_projection(source, weights["qkv"], projected, session=session,
            name=name + ".qkv", activation_mode="auto", workspace=workspace)
        self.query_shape = (tokens, heads, dim)
        query = projected[:, :heads * dim].view(tokens, heads, dim)
        key, value = (tensor.view(tokens, groups, dim)
                      for tensor in projected[:, heads * dim:].split(groups * dim, -1))
        attended, output = empty("attended", self.query_shape), empty("output", (tokens, config.hidden))
        partial = empty("partial", (tokens, config.hidden)) if runtime is not None else output
        columns = (metadata.positions.clamp_min(0) // PAGE_TOKENS).clamp(max=metadata.token_page_table.shape[1] - 1)
        token_rows = torch.arange(metadata.positions.numel(), device=source.device)
        page_ids = metadata.token_page_table[token_rows, columns]
        slots = torch.where((metadata.positions >= 0) & (page_ids >= 0),
            page_ids.long() * PAGE_TOKENS + metadata.positions.remainder(PAGE_TOKENS), -1)
        self.writer = PreparedPagedKVWriter(key, value, key_cache, value_cache, slots,
            session=session, name=name + ".cache_write", k_scale=key_scale, v_scale=value_scale,
            caller_owned_placeholders=True)
        self.attention = PreparedPagedAttention(query, key_cache, value_cache, metadata.page_table,
            metadata.lengths, metadata.query_start_loc, attended, mode=mode, tuning_page_count=tuning_page_count,
            width=1 if mode == "prefill" else metadata.max_query_tokens,
            session=session, name=name + ".attention", workspace=workspace,
            k_descale=key_scale, v_descale=value_scale, caller_owned_placeholders=True)
        self.output = prepare_projection(attended.flatten(1), weights["output"], partial, session=session,
            name=name + ".output", activation_mode="auto", workspace=workspace)
        self.reduce = None if runtime is None else PreparedAllReduce(
            partial, output, runtime=runtime, coordinator=coordinator, session=session, name=name + ".reduce")
        self.groups, self.heads, self.dim, self.hidden = groups, heads, dim, config.hidden

    def functional(self, *, session):
        self.projection.functional(session=session)
        self.output.functional(session=session)
        if self.reduce is not None:
            self.reduce.functional(session=session)
        self.writer = self.writer.functional(k_cache=None, v_cache=None)
        self.attention = self.attention.functional(session=session, k_cache=None, v_cache=None,
            cache_seqlens=None, page_table=None, cu_seqlens_q=None)
        return self

    def __call__(self, source, metadata, key_cache, value_cache):
        projected = self.projection(source)
        tokens = projected.shape[0]
        query_width = self.heads * self.dim
        query = projected[:, :query_width].view(tokens, self.heads, self.dim)
        key, value = (tensor.view(tokens, self.groups, self.dim)
                      for tensor in projected[:, query_width:].split(self.groups * self.dim, -1))
        positions = metadata.positions
        columns = (positions.clamp_min(0) // PAGE_TOKENS).clamp(max=metadata.token_page_table.shape[1] - 1)
        rows = torch.arange(positions.numel(), device=positions.device)
        pages = metadata.token_page_table[rows, columns]
        slots = torch.where((positions >= 0) & (pages >= 0),
            pages.long() * PAGE_TOKENS + positions.remainder(PAGE_TOKENS), -1)
        lengths = torch.where(metadata.lengths > 0, (metadata.starts + metadata.lengths).int(), 0)
        self.writer(key, value, slots, k_cache=key_cache, v_cache=value_cache)
        attended = self.attention(query, k_cache=key_cache, v_cache=value_cache,
            cache_seqlens=lengths, page_table=metadata.page_table, cu_seqlens_q=metadata.query_start_loc)
        output = self.output(attended.flatten(1))
        return output if self.reduce is None else self.reduce(output)
