# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Recipe discovery and preparation syntax without GPU initialization."""

import subprocess
import sys
from pathlib import Path

import pytest

from ignition.cli import main as _cli_main, parse_args
from ignition.errors import UsageError
from hyperflare_recipes.recipes import aliases, recipes, resolve


def main(argv=None):
    return _cli_main(argv, recipes=recipes())


def parse(argv):
    return parse_args(argv, recipes=recipes())


@pytest.mark.parametrize('flags', [['--output', 'model'], ['--form', 'packed'], ['--form', 'directory']])
def test_standalone_destinations_belong_to_convert(flags, capsys):
    with pytest.raises(SystemExit) as error:
        parse(['export', 'qwen3.8-flash', *flags])
    assert error.value.code == 2
    assert 'unrecognized arguments' in capsys.readouterr().err
    args = parse(['convert', 'cached-entry', *flags])
    assert args.core == Path('cached-entry')


@pytest.mark.parametrize('command', [
    ['export', 'qwen3.8-flash'],
    ['convert', 'core'],
    ['sign', 'core', '--output', 'signed-core'],
])
def test_core_commands_reject_shard_size(command, capsys):
    with pytest.raises(SystemExit) as error:
        parse([*command, '--shard-size', '32G'])
    assert error.value.code == 2
    assert 'unrecognized arguments: --shard-size 32G' in capsys.readouterr().err


@pytest.mark.parametrize('flags', [['--history-drafts'], ['--no-history-drafts'], ['--mtp-proposal-source', 'copy']])
def test_export_rejects_history_preferences_and_copy_only_preparation(flags):
    with pytest.raises(SystemExit) as error:
        parse(['export', 'qwen3.8-flash', *flags])
    assert error.value.code == 2


@pytest.mark.parametrize("command", ["export", "debug-serve"])
def test_mla_arithmetic_flag_defaults_and_override(command):
    arguments = [command, "deepseek-v4.1-flash"]
    default = parse(arguments)
    explicit = parse([*arguments, "--mla-compute-mode", "bf16"])
    assert default.mla_compute_mode is None
    assert explicit.mla_compute_mode == "bf16"


@pytest.mark.parametrize("model", ["deepseek-v4.1-flash", "deepseek-v4-flash"])
@pytest.mark.parametrize("flags", [["--draft-head-quantization", "nvfp4"],
                                   ["--markov-head-quantization", "nvfp4"],
                                   ["--draft-head-quantization", "nvfp4", "--markov-head-quantization", "nvfp4"]])
def test_independent_draft_precision_flags_require_learned_proposals(model, flags):
    from ignition.cli.preparation import resolve_parameters

    resolved = resolve(model)
    params = resolve_parameters(resolved, parse(["export", model, *flags]))
    assert bool(params.draft_quantization.draft_head_quantization) == ("--draft-head-quantization" in flags)
    assert bool(params.draft_quantization.markov_head_quantization) == ("--markov-head-quantization" in flags)
    with pytest.raises(UsageError, match="disables speculation"):
        resolve_parameters(resolved, parse(["export", model, "--no-mtp", *flags]))
    with pytest.raises(UsageError, match="positive learned draft depth"):
        resolve_parameters(resolved, parse(["export", model, "--mtp-draft-tokens", "0", *flags]))


@pytest.mark.parametrize("command", ["export", "debug-serve"])
def test_mxfp8_draft_precision_reaches_deepseek41_plan(command):
    from dataclasses import replace
    from ignition import planning
    from ignition.cli.preparation import resolve_parameters

    default = parse([command, "deepseek-v4.1-flash"])
    assert default.draft_head_quantization is None
    args = parse([command, "deepseek-v4.1-flash", "--draft-head-quantization", "mxfp8"])
    model = resolve(args.model)
    params = replace(resolve_parameters(model, args), max_concurrency=1)
    gpus = tuple(planning.Gpu(i, "SM120", f"GPU-{i}", 96 << 30, 95 << 30, (12, 0)) for i in range(4))
    plan = planning.fit(model.recipe, model.checkpoint, gpus, parameters=params, reserve_bytes=0)
    assert plan.parameters.draft_quantization.draft_head_quantization == "mxfp8"
    assert plan.parameters.draft_quantization.markov_head_quantization is None
    unsupported = resolve("deepseek-v4-flash")
    bad_params = replace(
        unsupported.recipe.parameters,
        max_concurrency=1,
        draft_quantization=replace(unsupported.recipe.parameters.draft_quantization, draft_head_quantization="mxfp8"),
    )
    with pytest.raises(UsageError, match="proposal head format"):
        planning.fit(unsupported.recipe, unsupported.checkpoint, gpus, parameters=bad_params, reserve_bytes=0)


@pytest.mark.parametrize("command", ["export", "debug-serve"])
def test_fp8_markov_precision_reaches_only_deepseek41_plan(command):
    from dataclasses import replace
    from ignition import planning
    from ignition.cli.preparation import resolve_parameters

    default = parse([command, "deepseek-v4.1-flash"])
    assert default.markov_head_quantization is None
    args = parse([command, "deepseek-v4.1-flash", "--markov-head-quantization", "fp8"])
    model = resolve(args.model)
    params = replace(resolve_parameters(model, args), max_concurrency=1)
    gpus = tuple(planning.Gpu(i, "SM120", f"GPU-{i}", 96 << 30, 95 << 30, (12, 0)) for i in range(4))
    plan = planning.fit(model.recipe, model.checkpoint, gpus, parameters=params, reserve_bytes=0)
    assert plan.parameters.draft_quantization.markov_head_quantization == "fp8"
    assert plan.parameters.draft_quantization.draft_head_quantization is None
    unsupported = resolve("deepseek-v4-flash")
    bad_params = replace(
        unsupported.recipe.parameters,
        max_concurrency=1,
        draft_quantization=replace(unsupported.recipe.parameters.draft_quantization, markov_head_quantization="fp8"),
    )
    with pytest.raises(UsageError, match="proposal head format"):
        planning.fit(unsupported.recipe, unsupported.checkpoint, gpus, parameters=bad_params, reserve_bytes=0)


@pytest.mark.parametrize("command", [[], ["export"], ["debug-serve"]])
def test_help_lists_every_recipe(command, capsys):
    with pytest.raises(SystemExit) as error:
        main([*command, "--help"])
    assert error.value.code == 0
    captured = capsys.readouterr()
    output = captured.out
    assert not captured.err
    for alias in aliases():
        assert alias in output
    assert "Drafts (default/max)" in output
    assert "--packed-verification" not in output
    assert "--draft-prefix-selection" not in output
    if command:
        assert "--mla-compute-mode" not in output
        assert "--force-moe-decode-a16" not in output
        assert "--dcp-size" not in output
    if command == ["debug-serve"]:
        normalized = " ".join(output.split())
        assert "not for actual serving or performance comparisons against other runtimes" in normalized
        assert "ignition export and flare run" in normalized


@pytest.mark.parametrize("command", ["export", "debug-serve"])
def test_model_help_explains_step5_limits_and_compression(command, capsys):
    with pytest.raises(SystemExit) as error:
        main([command, "step5-preview-nvfp4", "--help"])
    assert error.value.code == 0
    captured = capsys.readouterr()
    output = " ".join(captured.out.split())
    assert "--gpus: select 4, 8, 16 GPUs" in output
    assert "0..3; default: 3" in output
    assert "1..64; default: 32" in output
    assert "--expert-scale-compression [MODE]" in output
    assert "estimated 6% less resident weight storage" in output
    assert "choices: csf, native; default: native" in output
    assert "batch * width <= N" in output
    assert "--mixed-prefill" not in output
    assert "--speculative-graph-budget" not in output
    assert not captured.err


def test_help_for_all_aliases_and_repositories_needs_no_model_dependencies():
    script = """
import contextlib
import io
import sys
from ignition.cli import main
from hyperflare_recipes.recipes import recipes
with contextlib.redirect_stdout(io.StringIO()):
    for recipe in recipes():
        for checkpoint in recipe.checkpoints:
            for model in (checkpoint.alias, checkpoint.repository):
                try:
                    main(['export', model, '--help'], recipes=recipes())
                except SystemExit as error:
                    assert error.code == 0
assert not {'torch', 'transformers', 'flare_heliopause', 'huggingface_hub', 'pynvml'} & set(sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("command", ["export", "debug-serve"])
def test_deepseek41_help_reports_packed_gpu_recipe_defaults(command, capsys):
    with pytest.raises(SystemExit) as error:
        main([command, "deepseek-v4.1-flash", "--help"])
    assert error.value.code == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "--max-concurrency N" in output and "1..64; default: 32" in output
    assert "--prefill-chunk-size N" in output and "64..1,048,576; multiple of 64; default: 4096" in output
    assert "--mtp-draft-tokens N" in output and "0..7; default: 7" in output
    assert "--image-concurrency N" in output and "default: min(4, max concurrency) = 4" in output
    assert "--mla-compute-mode {auto,bf16}" in output and "default: bf16" in output
    assert "--expert-scale-compression [MODE]" in output and "choices: csf, native; default: csf" in output
    assert "--speculative-token-capacity N" in output and "default: 128" in output
    assert "--speculative-graph-budget N" in output and "default: 64" in output
    assert "excluding mandatory width-one and ordinary graphs" in output
    assert "Prepared verification: packed rows with GPU prefix selection." in output
    assert "--packed-verification" not in output
    assert "--draft-prefix-selection" not in output
    assert "packed rows shared across retained request prefixes" in output


@pytest.mark.parametrize("command", ["export", "debug-serve"])
def test_qwen_help_reports_concurrency_and_target_row_defaults(command, capsys):
    with pytest.raises(SystemExit) as error:
        main([command, "qwen3.8-flash", "--help"])
    assert error.value.code == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "1..64; default: 32" in output
    assert "--speculative-token-capacity N" in output and "default: 128" in output
    assert "Prepared verification: rectangular rows with CPU prefix selection." in output
    assert "--mixed-prefill" in output
    assert "--force-moe-decode-a16" in output
    assert "--shard-hyperconnections" in output
    assert "--mla-compute-mode" not in output


def test_optional_features_follow_the_selected_recipe(capsys):
    for recipe in recipes():
        checkpoint = recipe.checkpoints[0]
        with pytest.raises(SystemExit):
            main(["export", checkpoint.alias, "--help"])
        output = capsys.readouterr().out
        assert ("--mixed-prefill" in output) == (recipe.parameters.mixed_prefill is not None)
        assert "--history-drafts" not in output
        assert ("--no-stream-overlap" in output) == (getattr(recipe.parameters, "stream_overlap", None) is not None)
        assert ("--expert-scale-compression" in output) == bool(checkpoint.scale_compressed_device_bytes)


@pytest.mark.parametrize("command", ["export", "debug-serve"])
def test_all_recipe_help_reports_concurrency_and_speculative_capacity_defaults(command, capsys):
    for recipe in recipes():
        for checkpoint in recipe.checkpoints:
            with pytest.raises(SystemExit) as error:
                main([command, checkpoint.alias, "--help"])
            assert error.value.code == 0
            output = " ".join(capsys.readouterr().out.split())
            assert f"1..{recipe.max_concurrency}; default: {recipe.concurrency}" in output
            if recipe.speculation is None:
                assert "Multi-token speculation: unavailable." in output
            elif recipe.speculation.token_capacity is None:
                assert "default: larger of padded concurrency and drafts + 1" in output
            else:
                assert f"default: {recipe.speculation.token_capacity}" in output


@pytest.mark.parametrize("flag", [["--expert-scale-compression"], ["--expert-scale-compression", "csf"],
                                  ["--expert-scale-compression=csf"]])
def test_export_command_accepts_bare_and_explicit_compression(flag):
    args = parse([
        "export", "step5-preview-nvfp4", *flag, "--prefill-chunk-size", "4096",
        "--mtp-draft-tokens", "3", "--no-mtp-adaptive-verification",
        "--speculative-token-capacity", "16", "--max-concurrency", "16"])
    assert args.expert_scale_compression == "csf"
    assert (args.prefill_chunk_size, args.mtp_draft_tokens, args.speculative_token_capacity,
            args.max_concurrency) == (4096, 3, 16, 16)
    assert parse(["export", "step5-preview-nvfp4"]).expert_scale_compression is None


def test_invalid_compression_is_rejected_with_the_supported_encoding(capsys):
    with pytest.raises(SystemExit) as error:
        main(["export", "step5-preview-nvfp4", "--expert-scale-compression", "zip"])
    assert error.value.code == 2
    assert "invalid choice: 'zip'" in capsys.readouterr().err


def test_mistyped_alias_suggests_step5():
    with pytest.raises(UsageError, match="did you mean 'step5-preview-nvfp4'"):
        resolve("step4-preview-nvfp4")


@pytest.mark.parametrize("command", ["export", "debug-serve"])
@pytest.mark.parametrize("effort", ["off", "default", "audit"])
def test_autotune_effort_flag(command, effort):
    args = [command, "deepseek-v4.1-flash"]
    assert parse(args).autotune_effort is None
    assert parse([*args, "--autotune-effort", effort]).autotune_effort == effort
    for legacy in ("exhaustive", "medium"):
        with pytest.raises(SystemExit):
            parse([*args, "--autotune-effort", legacy])
