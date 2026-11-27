# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-H heterogeneous block geometry and execution semantics."""

from dataclasses import dataclass
import json
from pathlib import Path

PAGE_TOKENS = 128


@dataclass(frozen=True)
class ModelConfig:
    text: dict
    vision: dict
    image_token: int
    eos_token: int
    draft: object = None

    @classmethod
    def load(cls, snapshot):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, raw):
        if (raw.get("model_type") != "nemotron_h_omni"
                or raw.get("architectures") != ["NemotronH_Omni_Reasoning_V3"]):
            raise ValueError("Nemotron-H requires NemotronH_Omni_Reasoning_V3")
        text = raw["llm_config"]
        if text.get("model_type") not in ("nemotron_h_puzzle", "nemotron_h"):
            raise ValueError(f"unsupported Nemotron-H model_type: {text.get('model_type')!r}")
        required = dict(mamba_hidden_act="silu", mlp_hidden_act="relu2",
            use_bias=False, mlp_bias=False, attention_bias=False,
            use_conv_bias=True, conv_kernel=4, n_group=1, topk_group=1,
            norm_topk_prob=True, residual_in_fp32=False,
            tie_word_embeddings=False, sliding_window=None)
        for key, expected in required.items():
            if text.get(key) != expected:
                raise ValueError(f"unsupported Nemotron-H {key}: {text.get(key)!r}")
        if text.get("use_qk_norm", False) is not False:
            raise ValueError(f"unsupported Nemotron-H use_qk_norm: {text.get('use_qk_norm')!r}")
        if text.get("scoring_func", "sigmoid") != "sigmoid":
            raise ValueError(f"unsupported Nemotron-H scoring_func: {text.get('scoring_func')!r}")
        if text.get("num_nextn_predict_layers", 0) not in (0, 1):
            raise ValueError(
                f"unsupported Nemotron-H num_nextn_predict_layers: {text.get('num_nextn_predict_layers')!r}"
            )
        for key in ("hidden_size", "vocab_size", "mamba_num_heads", "mamba_head_dim", "ssm_state_size",
                    "n_groups", "num_attention_heads", "num_key_value_heads", "head_dim", "moe_latent_size",
                    "moe_shared_expert_intermediate_size", "max_position_embeddings"):
            if type(text.get(key)) is not int or text[key] <= 0:
                raise ValueError(f"Nemotron-H requires positive {key}")
        if (text["mamba_head_dim"] not in (64, 128) or text["ssm_state_size"] not in (64, 128)
                or text["mamba_num_heads"] % text["n_groups"]
                or text["num_attention_heads"] % text["num_key_value_heads"]):
            raise ValueError("unsupported Nemotron-H attention or state geometry")
        limits = text.get("time_step_limit")
        if limits is not None and limits != [0.0, {"__float__": "Infinity"}] and limits != [0.0, float("inf")]:
            raise ValueError("Nemotron-H requires an unclipped softplus time step")
        layers = text.get("layers_block_type")
        if not layers or any(kind not in ("mamba", "attention", "moe") for kind in layers):
            raise ValueError("Nemotron-H requires a nonempty layer sequence of mamba, attention, and moe blocks")
        blocks = text.get("block_configs")
        if blocks is not None:
            if not blocks or [b["block_type"] for b in blocks] != layers:
                raise ValueError("Nemotron-H block configurations must agree with the layer sequence")
            moe_blocks = [block for block in blocks if block["block_type"] == "moe"]
        else:
            moe_blocks = [text] if "moe" in layers else []
        for block in moe_blocks:
            n, k, width = (block.get(key) for key in
                ("n_routed_experts", "num_experts_per_tok", "moe_intermediate_size"))
            if (any(type(v) is not int for v in (n, k, width)) or not 0 < k <= n <= 1024
                    or width <= 0 or width % 128):
                raise ValueError("Nemotron-H experts require top_k <= experts <= 1024 and 128-aligned widths")
        return cls(text, raw["vision_config"], raw["img_context_token_id"], raw["eos_token_id"])

    def block_config(self, index):
        blocks = self.text.get("block_configs")
        if blocks is not None:
            return blocks[index]
        kind = self.layers[index]
        if kind != "moe":
            return {"block_type": kind}
        return {
            "block_type": kind,
            "n_routed_experts": self.text["n_routed_experts"],
            "num_experts_per_tok": self.text["num_experts_per_tok"],
            "moe_intermediate_size": self.text["moe_intermediate_size"],
        }

    @property
    def hidden(self):
        return self.text["hidden_size"]

    @property
    def vocabulary(self):
        return self.text["vocab_size"]

    @property
    def layers(self):
        return tuple(self.text["layers_block_type"])

    @property
    def mamba_width(self):
        return self.text["mamba_num_heads"] * self.text["mamba_head_dim"]

    @property
    def convolution_width(self):
        return self.mamba_width + 2 * self.text["n_groups"] * self.text["ssm_state_size"]

    def local_mamba_heads(self, shard=None):
        world = 1 if shard is None else shard.world_size
        return self.text["mamba_num_heads"] // world

    def local_n_groups(self, shard=None):
        world = 1 if shard is None else shard.world_size
        return self.text["n_groups"] // world

    def local_mamba_width(self, shard=None):
        return self.local_mamba_heads(shard) * self.text["mamba_head_dim"]

    def local_convolution_width(self, shard=None):
        return self.local_mamba_width(shard) + 2 * self.local_n_groups(shard) * self.text["ssm_state_size"]

    def local_attention_heads(self, shard=None):
        world = 1 if shard is None else shard.world_size
        return self.text["num_attention_heads"] // world

    def local_kv_heads(self, shard=None):
        world = 1 if shard is None else shard.world_size
        return max(1, self.text["num_key_value_heads"] // world)

    def local_shared_intermediate(self, shard=None):
        world = 1 if shard is None else shard.world_size
        return self.text["moe_shared_expert_intermediate_size"] // world

    def local_vocabulary(self, shard=None):
        world = 1 if shard is None else shard.world_size
        return self.vocabulary // world

    def validate_shard(self, shard):
        world = shard.world_size
        c = self.text
        if (world < 1 or self.vocabulary % world
                or c["mamba_num_heads"] % world or c["n_groups"] % world
                or c["num_attention_heads"] % world
                or (c["num_key_value_heads"] % world and world % c["num_key_value_heads"])
                or c["moe_shared_expert_intermediate_size"] % (world * 16)):
            raise ValueError(f"Nemotron-H geometry does not divide evenly across {world} ranks")
        for index, kind in enumerate(self.layers):
            if kind == "moe":
                width = self.block_config(index)["moe_intermediate_size"]
                if width % world or (width // world) % 64:
                    raise ValueError(
                        f"Nemotron-H layer {index} expert width {width} does not divide into 64-aligned shards for {world} ranks"
                    )
        if self.draft is not None:
            self.draft.validate_shard(shard)
