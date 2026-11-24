# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native chat rendering, tool/think syntax, and Qwen-profile image settings checks for Ling-3.2-Air."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ignition.preparation.frontend import QwenPositionScheme, QwenResizeKernel
from hyperflare_recipes.recipes.ling32.config import IMAGE_TOKEN_ID, ModelConfig, VIDEO_TOKEN_ID
from hyperflare_recipes.recipes.ling32.frontend import (
    BAILING_QWEN_IMAGE_PARAMS,
    prepare_frontend,
    runtime_assets,
)

ROOT = Path(__file__).resolve().parents[2]
PATCH_PATH = ROOT / "hyperflare_recipes/recipes/ling32/assets/chat_template.patch.json"


def _ling32_snapshot() -> Path:
    raw = os.environ.get("FLARE_LING32_SNAPSHOT")
    if raw:
        snapshot = Path(raw)
    else:
        from huggingface_hub.constants import HF_HUB_CACHE
        from hyperflare_recipes.recipes import resolve

        snapshot = Path(resolve("ling-3.2-air-nvfp4").snapshot(Path(HF_HUB_CACHE)))
    required = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json")
    for name in required:
        if not (snapshot / name).is_file():
            pytest.skip(
                f"requires Ling-3.2-Air checkpoint frontend asset {name} at {snapshot} (set FLARE_LING32_SNAPSHOT)"
            )
    return snapshot


def _find_test_frontend_binary() -> Path | None:
    env_bin = os.environ.get("FLARE_TEST_FRONTEND_BIN")
    if env_bin:
        candidate = Path(env_bin)
        return candidate if candidate.is_file() else None
    core_root = os.environ.get("FLARE_CORE_ROOT")
    if core_root:
        for rel in ("helios/target/release/test_frontend", "helios/target/debug/test_frontend"):
            candidate = Path(core_root) / rel
            if candidate.is_file():
                return candidate
    for rel in ("helios/target/release/test_frontend", "helios/target/debug/test_frontend"):
        candidate = ROOT / rel
        if candidate.is_file():
            return candidate
    return None


def _build_ling32_cases(snapshot: Path):
    from tokenizers import Tokenizer
    from transformers.utils.chat_template_utils import _compile_jinja_template

    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    raw_template = (snapshot / "chat_template.jinja").read_text()
    assets = runtime_assets(snapshot)
    annotated = assets["chat_template"]["text"]
    assert annotated.replace("{%- mark -%}", "") == raw_template

    template = _compile_jinja_template(raw_template)
    marked_template = _compile_jinja_template(annotated.replace("{%- mark -%}", "{{- message_position() -}}"))
    special = prepare_frontend(snapshot, eos_token_ids=156895, vision=True)["tokenizer"]

    cases = []

    def add(name: str, messages: list[dict], **options):
        rendered = template.render(messages=messages, **special, **options)
        marker = "\0message-position\0"
        marked = marked_template.render(messages=messages, **special, **options, message_position=lambda: marker)
        chunks = marked.split(marker)
        assert "".join(chunks) == rendered
        positions, position = [], 0
        for chunk in chunks[:-1]:
            position += len(chunk)
            positions.append(position)
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
            dict(
                name=name,
                messages=messages,
                options=options,
                rendered=rendered,
                token_ids=list(encoding.ids),
                message_boundaries=boundaries,
            )
        )

    user = dict(role="user", content="Hello café 世界")
    add("thinking_disabled", [user], enable_thinking=False, add_generation_prompt=True)
    add("thinking_enabled", [user], enable_thinking=True, add_generation_prompt=True)
    add("default", [user], enable_thinking=True, add_generation_prompt=False)
    add(
        "multi_turn_thinking",
        [
            dict(role="system", content="Be concise."),
            user,
            dict(role="assistant", content="Hi.", reasoning_content="Greet politely."),
            dict(role="user", content="Next question"),
        ],
        enable_thinking=True,
        add_generation_prompt=True,
    )
    tools = [
        dict(
            type="function",
            function=dict(
                name="lookup",
                description="Search records.",
                parameters=dict(
                    type="object",
                    properties=dict(query=dict(type="string"), limit=dict(type="integer")),
                    required=["query", "limit"],
                ),
            ),
        )
    ]
    add("tool_schema", [user], tools=tools, enable_thinking=True, add_generation_prompt=True)
    add(
        "tool_call_history",
        [
            user,
            dict(
                role="assistant",
                content="",
                tool_calls=[
                    dict(
                        id="call_1",
                        type="function",
                        function=dict(name="lookup", arguments=dict(query="weather", limit=3)),
                    )
                ],
            ),
            dict(role="tool", tool_call_id="call_1", content='{"temp": 20}'),
        ],
        tools=tools,
        enable_thinking=True,
        add_generation_prompt=True,
    )
    add(
        "image_and_followup_turn",
        [
            dict(
                role="user",
                content=[
                    dict(type="image"),
                    dict(type="text", text="Describe the image."),
                ],
            ),
            dict(role="assistant", content="A square."),
            dict(role="user", content="What color?"),
        ],
        enable_thinking=True,
        add_generation_prompt=True,
    )
    return annotated, cases


def test_ling32_template_patch_metadata_and_image_params():
    """Verify patch metadata and declared Qwen-profile image parameters for BailingMM2."""
    patch = json.loads(PATCH_PATH.read_text())
    assert len(patch["source_sha256"]) == 64
    assert len(patch["result_sha256"]) == 64
    assert patch["insertions"] == sorted(patch["insertions"])
    assert len(patch["insertions"]) == 2

    assert BAILING_QWEN_IMAGE_PARAMS["qwen_position_scheme"] is QwenPositionScheme.CENTERED_STEP
    assert BAILING_QWEN_IMAGE_PARAMS["qwen_resize_kernel"] is QwenResizeKernel.PILLOW_BICUBIC
    assert BAILING_QWEN_IMAGE_PARAMS["qwen_channel_normalization"] is True
    assert set(BAILING_QWEN_IMAGE_PARAMS["qwen_tokens"]) == {"image_token", "video_token"}


def test_ling32_snapshot_config_frontend_and_chat_rendering():
    """Validate pinned checkpoint ModelConfig, Qwen image frontend settings, and Jinja boundaries."""
    snapshot = _ling32_snapshot()
    cfg = ModelConfig.load(snapshot)
    assert cfg.mrope_section == (8, 12, 12)
    assert cfg.image_token_id == IMAGE_TOKEN_ID
    assert cfg.video_token_id == VIDEO_TOKEN_ID

    fe = prepare_frontend(snapshot, eos_token_ids=cfg.text["eos_token_id"], vision=True)
    qwen = fe["image"]["qwen"]
    assert qwen["position_scheme"] == int(QwenPositionScheme.CENTERED_STEP)
    assert qwen["resize_kernel"] == int(QwenResizeKernel.PILLOW_BICUBIC)
    assert qwen["image_token_id"] == IMAGE_TOKEN_ID
    assert qwen["video_token_id"] == VIDEO_TOKEN_ID

    _, cases = _build_ling32_cases(snapshot)
    by_name = {c["name"]: c for c in cases}
    assert by_name["thinking_disabled"]["rendered"].endswith("\n<think></think>")
    assert by_name["thinking_enabled"]["rendered"].endswith("\n<think>")
    assert "<tool_call>lookup\n<arg_key>query</arg_key>\n<arg_value>weather</arg_value>" in by_name["tool_call_history"]["rendered"]
    assert IMAGE_TOKEN_ID in by_name["image_and_followup_turn"]["token_ids"]


def test_native_ling32_chat_matches_helios_frontend(tmp_path):
    """Compare patched Jinja rendering, prefix marks, and default overrides against Helios test_frontend."""
    binary = _find_test_frontend_binary()
    if binary is None:
        pytest.skip("requires Helios test_frontend binary (set FLARE_TEST_FRONTEND_BIN)")
    snapshot = _ling32_snapshot()
    annotated, cases = _build_ling32_cases(snapshot)

    runtime_snapshot = tmp_path / "runtime"
    runtime_snapshot.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(snapshot / name, runtime_snapshot / name)
    (runtime_snapshot / "chat_template.jinja").write_text(annotated)
    asset_hashes = {
        name: hashlib.sha256((runtime_snapshot / name).read_bytes()).hexdigest()
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    }
    fixture = tmp_path / "chat.json"
    fixture.write_text(
        json.dumps(
            dict(
                assets=asset_hashes,
                template_defaults={"enable_thinking": False, "add_generation_prompt": True},
                default_overrides=[
                    ["thinking_disabled", {}],
                    ["thinking_enabled", {"enable_thinking": True}],
                    ["default", {"enable_thinking": True, "add_generation_prompt": False}],
                ],
                cases=cases,
                encode_cases=[],
                decode_cases=[],
            ),
            ensure_ascii=False,
        )
    )
    result = subprocess.run([str(binary), str(runtime_snapshot), str(fixture)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    counts = json.loads(result.stdout)
    assert counts["cases"] == len(cases)
    assert counts["default_override_cases"] == 3
