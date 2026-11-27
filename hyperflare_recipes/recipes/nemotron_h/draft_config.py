# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LiLiCorr draft backbone, candidate lattice, and target feature geometry."""

from dataclasses import dataclass
import json
from pathlib import Path

REPOSITORY = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr"
REVISION = "abd478065678ee83a93821c4b3f512da24e4cecb"
SUPER_REPOSITORY = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr"
SUPER_REVISION = "292bf1efc06ba39567da7351f2ad7a522afa3959"

DRAFT_SOURCES = {
    "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB": (REPOSITORY, REVISION),
    "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4": (SUPER_REPOSITORY, SUPER_REVISION),
}


def draft_source(repository: str, snapshot: Path | None = None) -> tuple[str, str]:
    if repository in DRAFT_SOURCES:
        return DRAFT_SOURCES[repository]
    if snapshot is not None:
        raw = json.loads((Path(snapshot) / "config.json").read_text())
        model_type = raw.get("llm_config", {}).get("model_type")
        if model_type == "nemotron_h":
            return SUPER_REPOSITORY, SUPER_REVISION
        if model_type == "nemotron_h_puzzle":
            return REPOSITORY, REVISION
    raise ValueError(f"no pinned LiLiCorr draft checkpoint for {repository}")


@dataclass(frozen=True)
class DraftConfig:
    text: dict

    @classmethod
    def load(cls, snapshot, target):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()), target)

    @classmethod
    def from_dict(cls, c, target):
        expected = dict(architectures=["LiLiCorrDraftModel"], model_type="qwen3", block_size=8,
            hidden_size=target.hidden, vocab_size=target.vocabulary, hidden_act="silu",
            attention_bias=False, tie_word_embeddings=False, rope_scaling=None,
            num_attention_heads=32, head_dim=128, num_hidden_layers=5,
            intermediate_size=12288, use_sliding_window=True, layer_types=["sliding_attention"] * 5)
        for field, value in expected.items():
            if c.get(field) != value:
                raise ValueError(f"unsupported LiLiCorr {field}: {c.get(field)!r}")
        kv_heads = c.get("num_key_value_heads")
        window = c.get("sliding_window")
        if (type(kv_heads) is not int or kv_heads <= 0 or c["num_attention_heads"] % kv_heads
                or type(window) is not int or window <= 0):
            raise ValueError("LiLiCorr requires positive KV head count dividing query heads and positive window")
        d = c["dflash_config"]
        expected = dict(use_swa=True, swa_window_size=window, causal=False,
            projector_type="lilicorr", lilicorr_enabled=True, lilicorr_candidate_topk=8,
            lilicorr_hidden_size=1024, lilicorr_num_layers=2, lilicorr_num_heads=8,
            lilicorr_mlp_ratio=2., lilicorr_factor_dim=1024, conv_kernel_size=2, conv_group_size=16)
        for field, value in expected.items():
            if d.get(field) != value:
                raise ValueError(f"unsupported LiLiCorr {field}: {d.get(field)!r}")
        sink_top = c.get("attention_sink_bias", False)
        sink_dflash = d.get("attention_sink_bias", False)
        if type(sink_top) is not bool or type(sink_dflash) is not bool or sink_top != sink_dflash:
            raise ValueError("LiLiCorr attention_sink_bias declarations must be matching booleans")
        own_head = c.get("has_own_lm_head", False)
        if type(own_head) is not bool:
            raise ValueError(f"unsupported LiLiCorr has_own_lm_head: {own_head!r}")
        if own_head and c.get("draft_vocab_size", target.vocabulary) != target.vocabulary:
            raise ValueError("LiLiCorr draft_vocab_size must match target vocabulary")
        taps = d["target_layer_ids"]
        if not taps or sorted(set(taps)) != taps or not all(type(i) is int and 0 <= i < len(target.layers) for i in taps):
            raise ValueError("LiLiCorr feature taps must be ordered target decoder indices")
        if not 0 <= d["mask_token_id"] < target.vocabulary:
            raise ValueError("LiLiCorr mask token must belong to the target vocabulary")
        if min(c["rope_theta"], c["rms_norm_eps"], d["lilicorr_vector_eps"], d["lilicorr_logit_scale"]) <= 0:
            raise ValueError("LiLiCorr rotary base, normalization epsilons, and logit scale must be positive")
        return cls(c)

    @property
    def block_size(self):
        return self.text["block_size"]

    @property
    def noise_token(self):
        return self.text["dflash_config"]["mask_token_id"]

    @property
    def target_layers(self):
        return tuple(self.text["dflash_config"]["target_layer_ids"])

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    @property
    def has_sink_bias(self):
        return bool(self.text.get("attention_sink_bias", False))

    @property
    def has_own_lm_head(self):
        return bool(self.text.get("has_own_lm_head", False))

    def local_attention_heads(self, shard=None):
        world = 1 if shard is None else shard.world_size
        return self.text["num_attention_heads"] // world

    def local_kv_heads(self, shard=None):
        world = 1 if shard is None else shard.world_size
        return max(1, self.text["num_key_value_heads"] // world)

    def local_intermediate(self, shard=None):
        world = 1 if shard is None else shard.world_size
        return self.text["intermediate_size"] // world

    def validate_shard(self, shard):
        world = shard.world_size
        c = self.text
        if (world < 1 or c["num_attention_heads"] % world
                or (c["num_key_value_heads"] % world and world % c["num_key_value_heads"])
                or c["intermediate_size"] % (world * 16)
                or c["vocab_size"] % world):
            raise ValueError(f"LiLiCorr draft geometry does not divide evenly across {world} ranks")
