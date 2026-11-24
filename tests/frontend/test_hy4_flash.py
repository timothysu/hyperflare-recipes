# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Frontend contract, chat-template rendering, and native Helios parity checks for Hy4-flash."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ignition.preparation import runtime_core_pb2 as p
from hyperflare_recipes.recipes.hy4_flash.bootstrap import ASSETS, chat_template_asset
from hyperflare_recipes.recipes.hy4_flash.config import (
    IMAGE_END_TOKEN,
    IMAGE_NEWLINE_TOKEN,
    IMAGE_START_TOKEN,
    IMAGE_TOKEN,
    ModelConfig,
    VIDEO_TOKEN,
)
from hyperflare_recipes.recipes.hy4_flash.contract import frontend

ROOT = Path(__file__).resolve().parents[2]


def _hy4_snapshot() -> Path:
    raw = os.environ.get("FLARE_HY4_FLASH_SNAPSHOT")
    if not raw:
        from huggingface_hub.constants import HF_HUB_CACHE
        from hyperflare_recipes.recipes import resolve

        snapshot = Path(resolve("hy4-flash-nvfp4").snapshot(Path(HF_HUB_CACHE)))
    else:
        snapshot = Path(raw)
    required = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json")
    for name in required:
        if not (snapshot / name).is_file():
            pytest.skip(f"Hy4-flash frontend asset {name} not present at {snapshot} (set FLARE_HY4_FLASH_SNAPSHOT)")
    return snapshot


def _find_test_frontend_binary() -> Path | None:
    env_bin = os.environ.get("FLARE_TEST_FRONTEND_BIN")
    if env_bin:
        candidate = Path(env_bin)
        return candidate if candidate.is_file() else None
    for rel in ("helios/target/release/test_frontend", "helios/target/debug/test_frontend"):
        candidate = ROOT / rel
        if candidate.is_file():
            return candidate
    return None


def _build_hy4_cases(snapshot: Path):
    from tokenizers import Tokenizer
    from transformers.utils.chat_template_utils import _compile_jinja_template

    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    raw_template = (snapshot / "chat_template.jinja").read_text()
    tpl = chat_template_asset(snapshot)
    annotated = tpl["text"]
    assert annotated.replace("{%- mark -%}", "") == raw_template

    template = _compile_jinja_template(raw_template)
    marked_template = _compile_jinja_template(annotated.replace("{%- mark -%}", "{{- message_position() -}}"))

    user = {"role": "user", "content": "Hello café 世界"}
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Find a value",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}, "count": {"type": "integer"}},
                    "required": ["q"],
                },
            },
        }
    ]
    tool_history = [
        {"role": "system", "content": "Be precise."},
        user,
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Search first.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": {"q": "café", "count": 2}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Result value"},
        {"role": "assistant", "reasoning_content": "Summarize.", "content": "Final answer."},
        {"role": "user", "content": "Follow-up"},
    ]
    image_turn = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "fixture://image"}},
                {"type": "text", "text": "Describe the image."},
            ],
        }
    ]

    cases = []

    def add(name: str, messages: list[dict], **options):
        rendered = template.render(messages=messages, **options)
        marker = "\0message-position\0"
        marked = marked_template.render(messages=messages, **options, message_position=lambda: marker)
        chunks = marked.split(marker)
        assert "".join(chunks) == rendered
        positions, pos = [], 0
        for chunk in chunks[:-1]:
            pos += len(chunk)
            positions.append(pos)
        encoding = tokenizer.encode(rendered, add_special_tokens=False)
        prefix_ends = [0]
        for _, end in encoding.offsets:
            prefix_ends.append(max(prefix_ends[-1], end))
        suffix_starts = [len(rendered)] * (len(encoding.ids) + 1)
        for index in range(len(encoding.ids) - 1, -1, -1):
            suffix_starts[index] = min(suffix_starts[index + 1], encoding.offsets[index][0])
        boundaries = []
        for target in positions:
            matched = [i for i in range(len(encoding.ids) + 1) if prefix_ends[i] <= target <= suffix_starts[i]]
            assert matched, f"{name}: mark offset {target} did not align to a token boundary"
            if 0 < matched[0] < len(encoding.ids) and (not boundaries or boundaries[-1] != matched[0]):
                boundaries.append(matched[0])
        cases.append(
            {
                "name": name,
                "messages": messages,
                "options": options,
                "rendered": rendered,
                "token_ids": list(encoding.ids),
                "message_boundaries": boundaries,
            }
        )

    add("thinking_disabled", [user], reasoning_effort="no_think", add_generation_prompt=True)
    add("thinking_enabled", [user], reasoning_effort="high", add_generation_prompt=True)
    add("default", [user], reasoning_effort="high", add_generation_prompt=False)
    add("tools_thinking_enabled", tool_history, tools=tools, reasoning_effort="high", add_generation_prompt=True)
    add("tools_thinking_disabled", tool_history, tools=tools, reasoning_effort="no_think", add_generation_prompt=True)
    add("image_placeholder", image_turn, reasoning_effort="high", add_generation_prompt=True)
    return tpl, cases


def test_hy4_frontend_contract_declarations_for_multimodal_and_text_only():
    vision_spec = frontend(vision=True)
    assert vision_spec["tools"]["kind"] == p.HY_ARGUMENT_TAGS
    assert vision_spec["reasoning"]["kind"] == p.HY_THINK
    assert vision_spec["reasoning"]["open"] == "<think:opensource>"
    assert vision_spec["reasoning"]["close"] == "</think:opensource>"
    assert vision_spec["position_dimensions"] == 3
    assert vision_spec["image_preparation"] == p.ROW_NEWLINE_IMAGE_PREPARATION
    assert vision_spec["image_layout"] == p.ROW_NEWLINE_IMAGE_LAYOUT
    assert vision_spec["image_tokens"] == {"feature_token": IMAGE_TOKEN, "span": None}
    assert vision_spec["rejected_tokens"] == [VIDEO_TOKEN]

    text_spec = frontend(vision=False)
    assert text_spec["tools"]["kind"] == p.HY_ARGUMENT_TAGS
    assert text_spec["reasoning"]["kind"] == p.HY_THINK
    assert text_spec["position_dimensions"] == 3
    assert text_spec["image_preparation"] is None
    assert text_spec["image_layout"] is None
    assert set(text_spec["rejected_tokens"]) == {
        IMAGE_START_TOKEN,
        IMAGE_END_TOKEN,
        IMAGE_TOKEN,
        IMAGE_NEWLINE_TOKEN,
        VIDEO_TOKEN,
    }


def test_hy4_chat_template_patch_metadata_is_well_formed():
    patch = json.loads((ASSETS / "chat_template.patch.json").read_text())
    assert patch["source"]["repository"] == "tencent/Hy4-flash"
    assert patch["source"]["filename"] == "chat_template.jinja"
    for key in ("source_sha256", "result_sha256"):
        assert isinstance(patch[key], str) and len(patch[key]) == 64
        int(patch[key], 16)
    insertions = patch["insertions"]
    assert isinstance(insertions, list) and len(insertions) > 0
    assert all(type(off) is int and off >= 0 for off in insertions)
    assert insertions == sorted(insertions)


def test_hy4_frontend_snapshot_patch_rendering_and_row_newline_settings():
    snapshot = _hy4_snapshot()
    from ignition.preparation.frontend import prepare_frontend

    cfg = ModelConfig.load(snapshot)
    tpl, cases = _build_hy4_cases(snapshot)
    patch = json.loads((ASSETS / "chat_template.patch.json").read_text())
    assert tpl["sha256"] == patch["result_sha256"]
    fe = prepare_frontend(
        snapshot,
        eos_token_ids=cfg.text["eos_token_id"],
        image_profile="row_newline",
        preprocessor=snapshot / "processor_config.json",
    )
    assert fe["image"]["row_newline"]["image_token_id"] == IMAGE_TOKEN
    assert fe["image"]["row_newline"]["newline_token_id"] == IMAGE_NEWLINE_TOKEN
    assert fe["image"]["row_newline"]["video_token_id"] == VIDEO_TOKEN

    by_name = {c["name"]: c for c in cases}
    assert "<think:opensource></think:opensource>" in by_name["thinking_disabled"]["rendered"]
    assert by_name["thinking_enabled"]["rendered"].endswith("<think:opensource>")
    assert "<tool_call:opensource>lookup" in by_name["tools_thinking_enabled"]["rendered"]
    assert "<arg_key:opensource>q</arg_key:opensource>" in by_name["tools_thinking_enabled"]["rendered"]
    assert IMAGE_TOKEN in by_name["image_placeholder"]["token_ids"]
    assert len(by_name["tools_thinking_enabled"]["message_boundaries"]) >= 5


def test_native_hy4_chat_matches_helios_frontend(tmp_path):
    """Verify byte-for-byte, token-for-token, and boundary parity against Helios test_frontend."""
    binary = _find_test_frontend_binary()
    if binary is None:
        pytest.skip("requires Helios test_frontend binary (set FLARE_TEST_FRONTEND_BIN)")
    snapshot = _hy4_snapshot()
    tpl, cases = _build_hy4_cases(snapshot)

    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(snapshot / name, tmp_path / name)
    (tmp_path / "chat_template.jinja").write_text(tpl["text"])

    assets = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir() if p.is_file()}
    fixtures = {
        "assets": assets,
        "template_defaults": {"reasoning_effort": "no_think", "add_generation_prompt": True},
        "default_overrides": [
            ["thinking_disabled", {}],
            ["thinking_enabled", {"reasoning_effort": "high"}],
            ["default", {"reasoning_effort": "high", "add_generation_prompt": False}],
        ],
        "cases": cases,
        "encode_cases": [
            {
                "name": "hy_special_tokens",
                "text": "<think:opensource><tool_call:opensource>",
                "add_special_tokens": False,
                "token_ids": __import__("tokenizers")
                .Tokenizer.from_file(str(snapshot / "tokenizer.json"))
                .encode("<think:opensource><tool_call:opensource>", add_special_tokens=False)
                .ids,
            }
        ],
        "decode_cases": [
            {
                "name": "utf8_roundtrip",
                "token_ids": cases[0]["token_ids"],
                "skip_special_tokens": False,
                "text": cases[0]["rendered"],
            }
        ],
    }
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(fixtures, ensure_ascii=False))
    result = subprocess.run([str(binary), str(tmp_path), str(cases_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    counts = json.loads(result.stdout)
    assert counts["cases"] == len(cases)
    assert counts["default_override_cases"] == 3
