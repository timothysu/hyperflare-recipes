# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Recipe gating, Triton kernel parity, and multi-rank equivalence for Kimi row-sharded prefill."""

from dataclasses import replace
import threading
from types import SimpleNamespace

import pytest
import torch

from hyperflare_recipes.recipes.kimi3 import decoder as kimi_decoder
from hyperflare_recipes.recipes.kimi3 import pointwise as kimi_pointwise
from hyperflare_recipes.recipes.kimi3 import residual as kimi_residual
from hyperflare_recipes.recipes.kimi3.bootstrap import PreparationConfig
from hyperflare_recipes.recipes.kimi3.decoder import PreparedDecoder, PreparedLatentMoE, PreparedMLP
from hyperflare_recipes.recipes.kimi3.recipe import RECIPE as KIMI3_RECIPE
from hyperflare_recipes.recipes.kimi31.recipe import RECIPE as KIMI31_RECIPE
from ignition.errors import UsageError
from ignition.planning import Gpu, plan as derive_plan


def _gpus(count=16, capability=(12, 0)):
    return tuple(
        Gpu(
            index=i,
            name="RTX PRO 6000" if capability == (12, 0) else "B300",
            uuid=f"GPU-{i}",
            free_bytes=102 << 30,
            total_bytes=102 << 30,
            capability=capability,
        )
        for i in range(count)
    )


def test_kimi_recipes_gate_row_shard_prefill_to_tp16_sm120():
    assert KIMI3_RECIPE.parameters.row_shard_prefill_world_sizes == (16,)
    assert KIMI31_RECIPE.parameters.row_shard_prefill_world_sizes == (16,)

    checkpoint = KIMI31_RECIPE.checkpoints[0]
    row_shard_params = replace(
        KIMI31_RECIPE.parameters,
        max_concurrency=1,
        speculation=None,
        row_shard_prefill=True,
        online_mxfp8=replace(KIMI31_RECIPE.parameters.online_mxfp8, enabled=True),
        vision=replace(KIMI31_RECIPE.parameters.vision, language_model_only=True),
    )
    p = derive_plan(
        KIMI31_RECIPE,
        checkpoint,
        _gpus(16, (12, 0)),
        selected=None,
        parameters=row_shard_params,
        reserve_bytes=0,
    )
    assert p.parameters.row_shard_prefill is True
    assert PreparationConfig().row_shard_prefill is False
    assert PreparationConfig(parameters=row_shard_params).row_shard_prefill is True

    with pytest.raises(UsageError, match="SM103"):
        derive_plan(
            KIMI31_RECIPE,
            checkpoint,
            _gpus(16, (10, 3)),
            selected=None,
            parameters=row_shard_params,
            reserve_bytes=0,
        )

    with pytest.raises(TypeError, match="row_shard_prefill must be a boolean"):
        replace(KIMI31_RECIPE.parameters, row_shard_prefill=1)


def test_rms_norm_rows_and_scale_columns_match_rms_norm_columns_bit_exact():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device("cuda:0")
    g = torch.Generator(device=device).manual_seed(77)
    rows, width, world = 128, 3584, 16
    c = width // world
    eps = 1e-6
    x = torch.randn((rows, width), dtype=torch.bfloat16, device=device, generator=g)
    full_weight = torch.randn((width,), dtype=torch.bfloat16, device=device, generator=g)

    normed = torch.empty_like(x)
    kimi_pointwise.rms_norm_rows(x, normed, eps)

    for rank in range(world):
        begin = torch.tensor([rank * c], dtype=torch.int32, device=device)
        w_shard = full_weight[rank * c : (rank + 1) * c].contiguous()
        ref = torch.empty((rows, c), dtype=torch.bfloat16, device=device)
        kimi_pointwise.rms_norm_columns(x, w_shard, ref, eps, begin=begin)

        actual = torch.empty((rows, c), dtype=torch.bfloat16, device=device)
        kimi_pointwise.scale_columns(normed[:, rank * c : (rank + 1) * c].contiguous(), w_shard, actual)
        assert torch.equal(actual, ref), f"mismatch on rank {rank}"


def _cpu_mix(self, delta, mode, *, sources, write, score, norm, eps, output_eps,
             mixture=None, output=None, reduction_choice=None, addend=None):
    assert reduction_choice is None and addend is None
    if mode == kimi_residual.REPLACE:
        current = delta.clone()
    elif mode == kimi_residual.ADD:
        current = (self.prefix.float() + delta.float()).to(self.prefix.dtype)
    else:
        current = self.prefix.clone()
    if mode != kimi_residual.KEEP:
        self.prefix.copy_(current)
    if write >= 0:
        self.blocks[:, write].copy_(current)
    if sources == 0:
        mixed = current.float()
    else:
        stack = torch.cat((self.blocks[:, :sources].float(), current.float().unsqueeze(1)), dim=1)
        scale = torch.rsqrt((stack * stack).mean(dim=-1, keepdim=True) + eps)
        scores = (stack * score[None, None, :]).sum(dim=-1) * scale.squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        mixed = (weights.unsqueeze(-1) * stack).sum(dim=1).to(self.prefix.dtype).float()
    if mixture is not None:
        mixture.copy_(mixed.to(mixture.dtype))
    normalized = (mixed * torch.rsqrt((mixed * mixed).mean(dim=-1, keepdim=True) + output_eps)).to(self.prefix.dtype)
    result = (norm.float() * normalized.float()).to(self.prefix.dtype)
    out = self.output if output is None else output
    out.copy_(result)
    return out


def _cpu_rms_norm_columns(source, weight, out, eps, begin=None):
    val = source.float()
    normed = (val * torch.rsqrt((val * val).mean(dim=-1, keepdim=True) + eps)).to(out.dtype)
    offset = 0 if begin is None else int(begin.item())
    cols = normed[:, offset : offset + out.shape[1]]
    out.copy_((cols.float() * weight.float()).to(out.dtype))
    return out


def _cpu_rms_norm_rows(source, out, eps):
    val = source.float()
    out.copy_((val * torch.rsqrt((val * val).mean(dim=-1, keepdim=True) + eps)).to(out.dtype))
    return out


def _cpu_scale_columns(source, weight, out):
    out.copy_((source.float() * weight.float()).to(out.dtype))
    return out


@pytest.mark.parametrize("subchunks", [1, 2])
@pytest.mark.parametrize("aux_stream", ["attn_res", "prefix"])
@pytest.mark.parametrize("split_moe,split_kda", [(False, False), (True, False), (True, True)])
def test_row_sharded_prefill_matches_serial_decoder_bit_exact_across_all_ranks(
    monkeypatch, subchunks, aux_stream, split_moe, split_kda
):
    """Verify exact bit-for-bit parity of row-sharded prefill (M=2, subchunks=1,2) vs serial decoder."""
    monkeypatch.setattr(kimi_residual.ResidualStream, "mix", _cpu_mix)
    monkeypatch.setattr(kimi_decoder, "rms_norm_columns", _cpu_rms_norm_columns)
    monkeypatch.setattr(kimi_decoder, "rms_norm_rows", _cpu_rms_norm_rows)
    monkeypatch.setattr(kimi_decoder, "scale_columns", _cpu_scale_columns)

    torch.manual_seed(1234)
    world = 4  # Four ranks exercise row sharding, reduce-scatter, all-gather, and all-to-all.
    reduce_step = 8  # 2 rows per rank per subchunk
    h = reduce_step * subchunks
    tokens = 2 * h
    hidden = 16
    latent = 8
    c_local = latent // world
    eps = 1e-5
    block_size = 2
    layer_kinds = ("kda", "kda", "kda", "dsa", "kda")  # Dense L0 + KDA->KDA->DSA->KDA MoE transitions
    aux_layers = (1, 3, 4)

    # Deterministic input embedding shared across ranks.
    embedding_full = torch.randn((tokens, hidden), dtype=torch.bfloat16)
    sample_indices = torch.tensor([0, tokens // 2 - 1, tokens // 2, tokens - 1], dtype=torch.int64)

    # Shared layer weights.
    layer_weights = []
    for idx in range(len(layer_kinds)):
        w = {
            "attention_res_score": torch.randn((hidden,), dtype=torch.float32),
            "mlp_res_score": torch.randn((hidden,), dtype=torch.float32),
            "input_layernorm": torch.randn((hidden,), dtype=torch.bfloat16),
            "post_attention_layernorm": torch.randn((hidden,), dtype=torch.bfloat16),
            "attn_in_w": torch.randn((hidden, hidden), dtype=torch.bfloat16) * 0.1,
            "attn_out_w": torch.randn((hidden, hidden), dtype=torch.bfloat16) * 0.1,
            "mlp_w": torch.randn((hidden, hidden), dtype=torch.bfloat16) * 0.1,
            "router_w": torch.randn((hidden, 4), dtype=torch.bfloat16) * 0.1,
            "down_w": torch.randn((hidden, c_local), dtype=torch.bfloat16) * 0.1,
            "shared_w": torch.randn((hidden, hidden), dtype=torch.bfloat16) * 0.1,
            "exp_w": torch.randn((latent, latent), dtype=torch.bfloat16) * 0.1,
            "latent_norm_full": torch.randn((latent,), dtype=torch.bfloat16),
            "up_w_per_rank": [torch.randn((c_local, hidden), dtype=torch.bfloat16) * 0.1 for _ in range(world)],
        }
        layer_weights.append(w)

    output_res_score = torch.randn((hidden,), dtype=torch.float32)
    final_norm = torch.randn((hidden,), dtype=torch.bfloat16)
    head_w = torch.randn((hidden, 12), dtype=torch.bfloat16) * 0.1

    class FakeAttn:
        def __init__(self, w, kind):
            self.w, self.kind = w, kind

        def project_input_chunk(self, x):
            return (x.float() @ self.w["attn_in_w"].float()).to(torch.bfloat16)

        def run_core(self, *args, **kwargs):
            if self.kind == "kda":
                proj = args[0]
                return (proj.float() * 0.5 + 0.1).to(torch.bfloat16)
            hidden_full, proj = args[0], args[1]
            return (proj.float() * 0.5).to(torch.bfloat16), (hidden_full.float() * 0.25).to(torch.bfloat16)

        def project_output_chunk(self, gated):
            return (gated.float() @ self.w["attn_out_w"].float()).to(torch.bfloat16)

        def finish_chunk(self, core, gate):
            return ((core.float() + gate.float()) @ self.w["attn_out_w"].float()).to(torch.bfloat16)

        def __call__(self, x, *args, **kwargs):
            proj = self.project_input_chunk(x)
            if self.kind == "kda":
                return self.project_output_chunk(self.run_core(proj))
            core, gate = self.run_core(x, proj)
            return self.finish_chunk(core, gate)

    class FakeDenseMLP(PreparedMLP):
        def __init__(self, w):
            self.w = w

        def __call__(self, x):
            return (x.float() @ self.w["mlp_w"].float()).to(torch.bfloat16)

    # Execute all ranks concurrently across threads with a shared barrier so every
    # cross-rank collective exchanges tensors between ranks without NCCL or CUDA.
    def run_cluster(row_shard: bool):
        barrier = threading.Barrier(world)
        mailbox = [None] * world

        def cluster_all_reduce(rank, tensor):
            mailbox[rank] = tensor.clone()
            barrier.wait()
            res = sum(t.float() for t in mailbox).to(torch.bfloat16)
            barrier.wait()
            return res

        def cluster_rs(rank, tensor):
            mailbox[rank] = tensor.clone()
            barrier.wait()
            full = sum(t.float() for t in mailbox).to(torch.bfloat16)
            st = tensor.shape[0] // world
            res = full[rank * st : (rank + 1) * st].clone()
            barrier.wait()
            return res

        def cluster_ag(rank, shard):
            mailbox[rank] = shard.clone()
            barrier.wait()
            res = torch.cat(list(mailbox), dim=0)
            barrier.wait()
            return res

        def cluster_a2a(rank, shard):
            mailbox[rank] = shard.clone()
            barrier.wait()
            c = shard.shape[1] // world
            res = torch.cat([t[:, rank * c : (rank + 1) * c] for t in mailbox], dim=0)
            barrier.wait()
            return res

        def cluster_rg(rank, down, logits):
            mailbox[rank] = (down.clone(), logits.clone())
            barrier.wait()
            lat = torch.cat([item[0] for item in mailbox], dim=1)
            log = torch.cat([item[1] for item in mailbox], dim=1)
            barrier.wait()
            return lat, log

        results = [None] * world
        errors = [None] * world

        class _FakeShared:
            concurrent = False

            def __init__(self, weight):
                self.weight = weight

            def __call__(self, x):
                return (x.float() @ self.weight.float()).to(torch.bfloat16)

        def rank_fn(rank):
            try:
                _run_rank(rank)
            except BaseException as exc:
                errors[rank] = exc
                barrier.abort()

        def _run_rank(rank):
            dec = object.__new__(PreparedDecoder)
            dec.weights = SimpleNamespace(
                config=SimpleNamespace(
                    text={"attn_res_block_size": block_size, "rms_norm_eps": eps},
                    layers=layer_kinds,
                ),
                shard=SimpleNamespace(rank=rank, world_size=world),
                output_res_score=output_res_score,
                norm=final_norm,
            )
            dec.mode = "prefill"
            dec.vision_state = None
            dec.tokens = tokens
            dec.hidden = hidden
            dec.sequences = 1
            dec.verification_columns = 1
            dec.shared_index_type = None
            dec.index_shards = None
            dec.replay = None
            dec.scoring = None
            dec.compact_head = False
            dec.row_mixes = None
            dec.residual_blocks = -(-len(layer_kinds) // block_size)
            dec.auxiliary_layers = aux_layers
            dec.auxiliary_stream = aux_stream
            dec.embedding = SimpleNamespace(
                lookup=lambda ids, num_rows: embedding_full,
                gather=lambda local: local,
            )
            dec.head = lambda x: (x.float() @ head_w.float()).to(torch.bfloat16)
            dec.pipeline_chunk = h if row_shard else None
            dec.reduce_step = reduce_step if row_shard else None
            dec.subchunks = subchunks if row_shard else 0
            dec.row_shard_prefill = row_shard
            dec.split_kda = bool(row_shard and split_kda)

            sub_st = reduce_step // world
            if row_shard:
                dec.shard_rows = tuple(
                    torch.cat([
                        torch.arange(
                            half_idx * h + j * reduce_step + rank * sub_st,
                            half_idx * h + j * reduce_step + (rank + 1) * sub_st,
                            dtype=torch.int64,
                        )
                        for j in range(subchunks)
                    ])
                    for half_idx in range(2)
                )
                dec.final_gather = lambda sub, dependencies=(): cluster_ag(rank, sub)
            else:
                dec.shard_rows = None
                dec.final_gather = None

            class FakeConcurrent:
                def __init__(self, fn):
                    self.fn = fn

                def start(self, x, *, dependencies=(), clone=True):
                    out = self.fn(x)
                    return SimpleNamespace(output=out)

                def finish(self, pending, dep):
                    return pending.output

                def __call__(self, x, *, dependencies=()):
                    return self.fn(x)

            class FakeRG:
                def start(self, down, logits, *, dependencies=()):
                    return cluster_rg(rank, down, logits)

                def finish(self, pending, dep):
                    return pending

                def __call__(self, down, logits, *, dependencies=()):
                    return cluster_rg(rank, down, logits)

            dec.layers = []
            for idx, (w, kind) in enumerate(zip(layer_weights, layer_kinds, strict=True)):
                attn = FakeAttn(w, kind)
                if idx == 0:
                    mlp = FakeDenseMLP(w)
                else:
                    mlp = object.__new__(PreparedLatentMoE)
                    mlp.world = world
                    mlp.latent_width = latent
                    mlp.eps = eps
                    mlp.latent_norm = w["latent_norm_full"][rank * c_local : (rank + 1) * c_local]
                    mlp.latent_begin = torch.tensor([rank * c_local], dtype=torch.int32)
                    mlp.row_norm = None
                    mlp.dual_down_router = None
                    mlp.concurrent_down = None
                    mlp.route_gather = None
                    mlp.prefill_route_gather = FakeRG()
                    mlp.router_projection = lambda x, _w=w: (x.float() @ _w["router_w"].float())
                    mlp.down = lambda x, _w=w: (x.float() @ _w["down_w"].float()).to(torch.bfloat16)
                    mlp.shared = _FakeShared(w["shared_w"])
                    mlp.router = lambda log: SimpleNamespace(ids=torch.zeros((log.shape[0], 2), dtype=torch.int32), weights=torch.ones((log.shape[0], 2)))
                    expert_rows = []
                    mlp.expert_rows = expert_rows

                    def _run_experts(lat, ids, rw, _w=w, _rows=expert_rows):
                        _rows.append(lat.shape[0])
                        return (lat.float() @ _w["exp_w"].float() * (1.0 + 0.05 * rank)).to(torch.bfloat16)

                    mlp.experts = _run_experts
                    mlp.up = lambda x, _w=w: (x.float() @ _w["up_w_per_rank"][rank].float()).to(torch.bfloat16)
                    mlp.expert_reduce = lambda x: cluster_all_reduce(rank, x)
                    mlp.expert_rs = lambda x, dependencies=(): cluster_rs(rank, x)
                    mlp.expert_a2a = lambda x, dependencies=(): cluster_a2a(rank, x)
                    mlp.concurrent_expert_rs = FakeConcurrent(lambda x: cluster_rs(rank, x))
                    mlp.split_moe = bool(row_shard and split_moe)
                if row_shard:
                    n_sub = 2 * subchunks
                    rs_parts = tuple(FakeConcurrent(lambda x: cluster_rs(rank, x)) for _ in range(n_sub))
                    ag_parts = tuple(FakeConcurrent(lambda x: cluster_ag(rank, x)) for _ in range(n_sub))
                    dec.layers.append((w, attn, mlp, None, None, (rs_parts, ag_parts, rs_parts, ag_parts)))
                else:
                    dec.layers.append((w, attn, mlp, lambda x: cluster_all_reduce(rank, x), None, None))

            meta = SimpleNamespace(
                token_ids=torch.zeros(tokens, dtype=torch.int32),
                live_counts=torch.tensor([tokens, 1], dtype=torch.int32),
                starts=torch.zeros(1, dtype=torch.int32),
                lengths=torch.tensor([tokens], dtype=torch.int32),
                query_start_loc=torch.tensor([0, tokens], dtype=torch.int32),
                state_rows=torch.zeros(1, dtype=torch.int32),
                sample_indices=sample_indices,
            )
            results[rank] = dec(meta)
            expected_rows = [h, h] if (row_shard and split_moe) else [tokens]
            for _, _, mlp_obj, _, _, _ in dec.layers:
                if isinstance(mlp_obj, PreparedLatentMoE):
                    assert mlp_obj.expert_rows == expected_rows

        threads = [threading.Thread(target=rank_fn, args=(r,)) for r in range(world)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for err in errors:
            if err is not None:
                raise err
        return results

    serial_res = run_cluster(row_shard=False)
    sharded_res = run_cluster(row_shard=True)

    for rank in range(world):
        s_sampled, s_logits, s_aux, _ = serial_res[rank]
        r_sampled, r_logits, r_aux, _ = sharded_res[rank]
        assert torch.equal(r_sampled, s_sampled), f"sampled hidden mismatch on rank {rank}"
        assert torch.equal(r_logits, s_logits), f"logits mismatch on rank {rank}"
        assert torch.equal(r_aux, s_aux), f"auxiliary mismatch on rank {rank}"
