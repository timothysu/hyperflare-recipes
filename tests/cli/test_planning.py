# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Plan sizing, GPU selection, and capacity errors from declared costs and injected free memory."""

import json
from dataclasses import dataclass, replace
import re
from types import MappingProxyType

import pytest

from ignition import planning
from ignition.cli import display, parse_args, preparation
from ignition.errors import CapacityError, UsageError
from ignition.declaration import Checkpoint, RankCosts, Recipe, SamplingDefaults, SamplingRange
from ignition.parameters import (
    MixedPrefillParameters,
    PromptLogprobsParameters,
    RecipeParameters,
    SpeculationParameters,
    VisionParameters,
)
from hyperflare_recipes.parameters import (
    ContextParallelParameters,
    DraftQuantizationParameters,
    ExpertScaleCompressionParameters,
    NativeDenseActivationsParameters,
    OnlineMXFP8Parameters,
    StreamOverlapParameters,
)
from hyperflare_recipes.recipes import ResolvedModel, recipes, resolve

GIB, MIB = 1 << 30, 1 << 20


DRAFTING = SpeculationParameters(draft_tokens=3, max_draft_tokens=7)


def recipe(name, device_gib, ranks, *, speculation=None, mapped_gib=0, draft_gib=0, parameters=None):
    checkpoint = Checkpoint(name, f"lab/{name}", "ab12cd34" + "0" * 32, "DemoForCausalLM",
                            stored_bytes=(device_gib + mapped_gib) * GIB, device_bytes=device_gib * GIB,
                            mapped_host_bytes=mapped_gib * GIB, draft_bytes=draft_gib * GIB)
    speculating = speculation is not None
    params = parameters if parameters is not None else RecipeParameters(
        max_concurrency=4, max_concurrency_limit=64, prefill_chunk=4096, speculation=speculation,
    )
    return Recipe(name=name, served_model_name=name, checkpoints=(checkpoint,),
                  context=262144, page_tokens=64,
                  sampling_defaults=SamplingDefaults(temperature=1., top_p=1., top_k=-1),
                  parameters=params, prepare=None,
                  usage=None, mapped_host_name="lookup table",
                  ranks=MappingProxyType({size: RankCosts(100 * MIB // size, 2 * MIB // size, 4 * GIB)
                                          for size in ranks}),
                  plain_ranks=MappingProxyType({size: RankCosts(80 * MIB // size, 3 * MIB // (2 * size), 3 * GIB)
                                                for size in ranks} if speculating else {}))


# While drafting, one rank holds 60 GiB of weights, 2 GiB of them the draft model, 4 GiB of graphs,
# 100 MiB per state row, and 2 MiB per page, a quarter of it the draft model's. At concurrency 4 it
# keeps 17 execution rows, and its minimum of 512 pages costs 1 GiB. Without drafting it holds 3 GiB of
# graphs, 80 MiB per state row, and 1.5 MiB per page.
SOLO = recipe("solo", 60, (1, 2, 4), speculation=DRAFTING, mapped_gib=4, draft_gib=2)
PAIR = recipe("pair", 160, (2, 4))


@pytest.fixture(autouse=True)
def headroom(monkeypatch):
    monkeypatch.setattr(planning, "HEADROOM_BYTES", 4 * GIB)


def gpu(index, free_gib, capability=(12, 0)):
    return planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index:04d}", 96 * GIB, int(free_gib * GIB),
                        capability)


def spark(index, free_gib):
    return planning.Gpu(index, "NVIDIA GB10", f"GPU-{index:04d}", 121 * GIB, int(free_gib * GIB), (12, 1))


DEFAULT = object()


def derive(model=SOLO, gpus=(gpu(0, 95),), *, selected=None, concurrency=4, reserve=0, prefill_chunk=None,
           speculation=DEFAULT, parameters=None):
    if parameters is None:
        updates = {"max_concurrency": concurrency}
        if prefill_chunk is not None:
            updates["prefill_chunk"] = prefill_chunk
        if speculation is not DEFAULT:
            updates["speculation"] = speculation
        parameters = replace(model.parameters, **updates)
    return planning.plan(model, model.checkpoints[0], tuple(gpus), selected=selected,
                         parameters=parameters, reserve_bytes=int(reserve * GIB))


def test_pages_take_all_remaining_supply():
    limited = derive()
    assert limited.fits and limited.cache_states == limited.cache_target == 32
    assert limited.supply_bytes == 91 * GIB and limited.live_bytes == 17 * 100 * MIB
    # 91 GiB of supply less weights, graphs, 17 execution rows leaves 12,974 pages.
    assert limited.pages == (91 * GIB - 64 * GIB - 17 * 100 * MIB) // (2 * MIB) == 12974
    assert limited.demand_pages == 16384 and limited.minimum_pages == 512
    assert limited.context == 262144 and limited.guaranteed_tokens == 12974 // 4 * 64
    assert limited.snapshots == 32

    assert derive(gpus=(gpu(0, 95), gpu(1, 95))).world_size == 1

    # One request needs 4,096 pages at full context; the rest of the supply holds cached prefixes.
    whole = derive(concurrency=1)
    assert whole.demand_pages == 4096 and whole.pages == 13574 and whole.guaranteed_tokens == 262144
    assert whole.supply_bytes - whole.page_bytes - whole.fixed_bytes < 2 * MIB


def test_vocabulary_sampling_rejects_tp3_before_weight_loading():
    from hyperflare_recipes.recipes.deepseek41.recipe import RECIPE

    params = replace(RECIPE.parameters, max_concurrency=1, vocabulary_parallel_sampling=True)
    with pytest.raises(UsageError, match="no TP3 gather"):
        planning.fit(RECIPE, RECIPE.checkpoints[0], tuple(gpu(i, 95) for i in range(3)),
                     parameters=params, reserve_bytes=0)


def test_vocabulary_sampling_rejects_unsupported_transition_storage():
    from hyperflare_recipes.recipes.deepseek41.recipe import DeepSeek41Parameters

    with pytest.raises(ValueError, match="BF16 or row-scaled FP8"):
        DeepSeek41Parameters(
            speculation=DRAFTING,
            vocabulary_parallel_sampling=True,
            draft_quantization=DraftQuantizationParameters(
                markov_head_quantization="nvfp4",
                draft_head_choices=("nvfp4", "mxfp8"),
                markov_head_choices=("nvfp4", "fp8"),
            ),
        )


def test_vocabulary_sampling_selects_tp4_and_rejects_tp3_hosts():
    from hyperflare_recipes.recipes.deepseek41.recipe import RECIPE

    params = replace(RECIPE.parameters, max_concurrency=4, vocabulary_parallel_sampling=True)
    gpus = tuple(gpu(i, 140) for i in range(4))
    assert derive(model=RECIPE, gpus=gpus, parameters=params).world_size == 4
    with pytest.raises(UsageError, match="no TP3 gather"):
        derive(model=RECIPE, gpus=gpus, selected=(0, 1, 2), parameters=params)
    with pytest.raises(CapacityError, match="needs 4"):
        derive(model=RECIPE, gpus=gpus[:3], parameters=params)


@pytest.mark.parametrize("filtering", [{"ranges": {"top_p": SamplingRange(.95, 1.)}}, {"top_p": .95},
                                       {"top_k": 20}, {"min_p": .05}])
def test_vocabulary_sampling_rejects_declared_token_filtering(filtering):
    from hyperflare_recipes.recipes.deepseek41.recipe import RECIPE

    params = replace(RECIPE.parameters, max_concurrency=4, vocabulary_parallel_sampling=True)
    gpus = tuple(gpu(i, 140) for i in range(4))
    filtered = replace(RECIPE, sampling_defaults=replace(RECIPE.sampling_defaults, **filtering))
    with pytest.raises(UsageError, match="without top-k, top-p, or min-p filtering"):
        derive(model=filtered, gpus=gpus, parameters=params)


@pytest.mark.parametrize("draft,markov", [(None, None), ("nvfp4", None), (None, "nvfp4"),
                                         ("nvfp4", "nvfp4"), ("mxfp8", None), ("nvfp4", "fp8")])
def test_proposal_precision_is_independent_plan_metadata(draft, markov, monkeypatch):
    from types import SimpleNamespace
    from ignition.preparation import rebuild_manifest
    from google.protobuf import json_format
    from ignition.preparation import runtime_core_pb2 as proto

    # Source identity needs the native export extension; the record's precision keys do not.
    monkeypatch.setattr("ignition.preparation.source_identity.source_identity", lambda: "test-source")

    @dataclass(frozen=True)
    class QuantParams(RecipeParameters):
        draft_quantization: DraftQuantizationParameters = DraftQuantizationParameters(
            draft_head_choices=("nvfp4", "mxfp8"),
            markov_head_choices=("nvfp4", "fp8"),
        )

    base_params = QuantParams(max_concurrency=4, max_concurrency_limit=64, prefill_chunk=4096, speculation=DRAFTING)
    model = replace(SOLO, parameters=base_params)
    params = replace(
        base_params,
        draft_quantization=replace(
            base_params.draft_quantization,
            draft_head_quantization=draft,
            markov_head_quantization=markov,
        ),
    )
    result = derive(model=model, parameters=params)
    assert result.to_dict()["speculation"]["draft_head_quantization"] == draft
    assert result.to_dict()["speculation"]["markov_head_quantization"] == markov
    status = " · ".join(result.parameters.status_parts(result.parameter_context))
    assert (f"draft head {(draft or '').upper()}" in status) == bool(draft)
    assert (f"markov head {(markov or '').upper()}" in status) == bool(markov)

    manifest = rebuild_manifest.from_plan(SimpleNamespace(directory=None), result, compiled_forwards=True)
    encoded = proto.PreparationManifest()
    json_format.ParseDict(manifest, encoded)
    values = {entry.key: entry.value.string_value for entry in encoded.parameters if entry.value.HasField("string_value")}
    assert ("speculation.draft_head_quantization" in values) == bool(draft)
    assert ("speculation.markov_head_quantization" in values) == bool(markov)
    assert values.get("speculation.draft_head_quantization", "") == (draft or "")
    assert values.get("speculation.markov_head_quantization", "") == (markov or "")
    if draft or markov:
        with pytest.raises(UsageError, match="positive learned"):
            derive(model=model, parameters=replace(params, speculation=replace(DRAFTING, draft_tokens=0)))


def test_mla_arithmetic_is_explicit_plan_metadata_without_changing_capacity():
    from hyperflare_recipes.recipes.deepseek41.recipe import RECIPE

    gpus = tuple(gpu(i, 140) for i in range(4))
    auto = planning.fit(
        RECIPE, RECIPE.checkpoints[0], gpus,
        parameters=replace(RECIPE.parameters, max_concurrency=4, mla_compute_mode="auto"), reserve_bytes=0,
    )
    bf16 = planning.fit(
        RECIPE, RECIPE.checkpoints[0], gpus,
        parameters=replace(RECIPE.parameters, max_concurrency=4, mla_compute_mode="bf16"), reserve_bytes=0,
    )
    assert auto.parameters.mla_compute_mode == "auto" and bf16.parameters.mla_compute_mode == "bf16"
    assert bf16.to_dict()["mla_compute_mode"] == "bf16"
    assert (auto.pages, auto.cache_states, auto.fixed_bytes) == (bf16.pages, bf16.cache_states, bf16.fixed_bytes)
    with pytest.raises(UsageError, match="is not supported"):
        parse_args(["export", "qwen3.8-flash", "--mla-compute-mode", "bf16"], recipes=recipes())


def test_image_concurrency_defaults_and_explicit_capacity():
    vision = replace(SOLO, max_images=4, parameters=replace(SOLO.parameters, vision=VisionParameters()))

    def image_plan(value=None, model=vision, concurrency=32):
        params = replace(model.parameters, max_concurrency=concurrency, speculation=None)
        if value is not None:
            if params.vision is None:
                raise UsageError("--image-concurrency requires a vision model")
            params = replace(params, vision=replace(params.vision, image_concurrency=value))
        return planning.plan(model, model.checkpoints[0], (gpu(0, 95),), selected=None,
                             parameters=params, reserve_bytes=0)

    assert image_plan().image_concurrency == 4
    assert image_plan(concurrency=2).image_concurrency == 2
    assert image_plan(1).image_concurrency == 1
    assert image_plan(32).to_dict()["image_concurrency"] == 32
    assert image_plan(model=SOLO).image_concurrency == 0
    for invalid in (0, -1, 33, True):
        with pytest.raises(UsageError, match="image-concurrency"):
            image_plan(invalid)
    with pytest.raises(UsageError, match="vision model"):
        image_plan(1, model=SOLO)


def test_dense_default_plan_prepares_image_slots_and_language_model_only_drops_them():
    from hyperflare_recipes.recipes.qwen38_dense import contract
    from hyperflare_recipes.recipes.qwen38_dense.recipe import RECIPE, VISION_BYTES, VISION_SLOT_BYTES

    # Default preparation uses the shared vision-recipe rule: min(4, concurrency) image slots.
    image = planning.plan(RECIPE, RECIPE.checkpoints[0], (gpu(0, 48),), selected=None,
                          parameters=replace(RECIPE.parameters, max_concurrency=4, speculation=None),
                          reserve_bytes=0)
    assert (image.image_concurrency, image.max_images, image.language_model_only) == (4, 4, False)
    assert image.costs.state_row_bytes == 158_149_120
    assert image.graph_bytes == (4 << 30) + VISION_BYTES + 4 * VISION_SLOT_BYTES
    assert contract.frontend(vision=True)["image_tokens"] == dict(feature_token=248056, span=None)
    # Concurrency clamps the default image slot count.
    narrow = planning.plan(RECIPE, RECIPE.checkpoints[0], (gpu(0, 48),), selected=None,
                           parameters=replace(RECIPE.parameters, max_concurrency=2, speculation=None),
                           reserve_bytes=0)
    assert narrow.image_concurrency == 2
    assert image.graph_bytes - narrow.graph_bytes == 2 * VISION_SLOT_BYTES
    # --language-model-only prepares text graphs only with the text-only continuation row.
    text = planning.plan(
        RECIPE, RECIPE.checkpoints[0], (gpu(0, 30),), selected=None,
        parameters=replace(
            RECIPE.parameters, max_concurrency=4, speculation=None,
            vision=replace(RECIPE.parameters.vision, language_model_only=True),
        ),
        reserve_bytes=0,
    )
    assert (text.image_concurrency, text.max_images, text.language_model_only) == (0, 0, True)
    assert text.costs.state_row_bytes == 158_148_864
    assert text.weights_bytes == 20_458_440_840
    assert text.graph_bytes == 4 << 30
    assert text.fits
    assert contract.frontend(vision=False)["image_tokens"] is None
    drafting = planning.plan(
        RECIPE, RECIPE.checkpoints[0], (gpu(0, 36),), selected=None,
        parameters=replace(
            RECIPE.parameters, max_concurrency=4,
            vision=replace(RECIPE.parameters.vision, language_model_only=True),
        ),
        reserve_bytes=0,
    )
    assert drafting.image_concurrency == 0 and drafting.costs.state_row_bytes == 193_865_984
    assert drafting.graph_bytes == 4 << 30


def test_dense_derive_uses_standard_image_settings_and_rejects_conflicting_flags(tmp_path):
    from ignition.cli.config import Configuration
    from hyperflare_recipes.recipes.qwen38_dense.recipe import RECIPE

    model = ResolvedModel(RECIPE, RECIPE.checkpoints[0])
    host = preparation.Host((gpu(0, 95),), None)
    path = tmp_path / "config.toml"
    path.write_text("")
    configuration = Configuration.load(path)

    def planned(*flags):
        args = parse_args(["export", "qwen3.8-27b", "--gpus", "0", "--max-concurrency", "8", "--no-mtp", *flags],
                          recipes=recipes())
        params = preparation.resolve_parameters(model, args, configuration)
        return preparation.derive(configuration, model, host, gpus="0", parameters=params)

    assert planned().image_concurrency == 4
    assert planned("--image-concurrency", "3").image_concurrency == 3
    plan = planned("--language-model-only")
    assert (plan.language_model_only, plan.image_concurrency, plan.max_images) == (True, 0, 0)
    with pytest.raises(UsageError, match="--image-concurrency requires the vision encoder"):
        planned("--language-model-only", "--image-concurrency", "2")

    glm = resolve("glm-5.3-flash")
    glm_host = preparation.Host(tuple(replace(gpu(index, 140), total_bytes=140 * GIB) for index in range(4)), None)
    glm_args = parse_args(["export", "glm-5.3-flash", "--gpus", "0-3", "--max-concurrency", "4",
                           "--no-mtp", "--language-model-only"], recipes=recipes())
    glm_text = preparation.derive(configuration, glm, glm_host, gpus="0-3",
                                  parameters=preparation.resolve_parameters(glm, glm_args, configuration))
    assert (glm_text.language_model_only, glm_text.image_concurrency, glm_text.max_images) == (True, 0, 0)
    with pytest.raises(UsageError, match="--image-concurrency requires the vision encoder"):
        bad_args = parse_args(["export", "glm-5.3-flash", "--gpus", "0-3", "--max-concurrency", "4",
                               "--no-mtp", "--language-model-only", "--image-concurrency", "2"], recipes=recipes())
        preparation.derive(configuration, glm, glm_host, gpus="0-3",
                           parameters=preparation.resolve_parameters(glm, bad_args, configuration))


def test_flash_recipe_keeps_the_shared_image_default():
    from hyperflare_recipes.recipes.qwen38.recipe import RECIPE

    pair = (gpu(0, 95), gpu(1, 95))
    def planned(concurrency):
        return planning.plan(RECIPE, RECIPE.checkpoints[0], pair, selected=(0, 1),
                             parameters=replace(RECIPE.parameters, max_concurrency=concurrency, speculation=None),
                             reserve_bytes=0)
    assert planned(4).image_concurrency == 4
    assert planned(2).image_concurrency == 2
    assert planned(4).max_images == 4


def test_expert_scale_compression_changes_weight_budget_and_preserves_native_default():
    import pickle

    @dataclass(frozen=True)
    class CompressedParams(RecipeParameters):
        expert_scale_compression: ExpertScaleCompressionParameters = ExpertScaleCompressionParameters()

    base_params = CompressedParams(max_concurrency=4, max_concurrency_limit=64, prefill_chunk=4096, speculation=DRAFTING)
    compressed_solo = replace(SOLO, parameters=base_params)
    checkpoint = replace(SOLO.checkpoints[0], scale_compressed_device_bytes={"csf": 56 * GIB})
    assert pickle.loads(pickle.dumps(checkpoint)) == checkpoint
    native = planning.fit(compressed_solo, checkpoint, (gpu(0, 95),), parameters=base_params, reserve_bytes=0)
    compressed = planning.fit(
        compressed_solo, checkpoint, (gpu(0, 95),),
        parameters=replace(base_params, expert_scale_compression=ExpertScaleCompressionParameters(encoding="csf")),
        reserve_bytes=0,
    )
    assert native.weights_bytes == 60 * GIB and native.parameters.expert_scale_compression.encoding is None
    assert compressed.weights_bytes == 56 * GIB and compressed.pages > native.pages
    assert compressed.to_dict()["expert_scale_compression"] == "csf"
    plain = planning.fit(
        compressed_solo, checkpoint, (gpu(0, 95),),
        parameters=replace(base_params, speculation=None,
                           expert_scale_compression=ExpertScaleCompressionParameters(encoding="csf")),
        reserve_bytes=0,
    )
    assert plain.weights_bytes == 54 * GIB
    with pytest.raises(UsageError, match="does not prepare this scale encoding"):
        planning.fit(
            compressed_solo, SOLO.checkpoints[0], (gpu(0, 95),),
            parameters=replace(base_params, expert_scale_compression=ExpertScaleCompressionParameters(encoding="csf")),
            reserve_bytes=0,
        )


def test_live_state_counts_only_null_live_and_speculative_rows():
    assert SOLO.live_rows(4, DRAFTING) == 1 + 4 * 4
    assert SOLO.live_rows(5, DRAFTING) == 1 + 8 * 4
    assert SOLO.live_rows(4, None) == PAIR.live_rows(4, None) == 1 + 4


def test_snapshot_slots_share_the_page_budget():
    assert derive(concurrency=1).cache_states == 8
    assert derive(concurrency=2).cache_states == 16
    capped = derive(reserve=12)
    assert capped.cache_states == 32
    assert capped.pages == (capped.supply_bytes - capped.fixed_bytes) // capped.costs.page_bytes
    devices = (gpu(0, 95),)
    small = planning.fit(SOLO, SOLO.checkpoints[0], devices, parameters=SOLO.parameters, reserve_bytes=0, cache_states=1)
    large = planning.fit(SOLO, SOLO.checkpoints[0], devices, parameters=SOLO.parameters, reserve_bytes=0, cache_states=64)
    assert small.pages == large.pages and small.live_rows == large.live_rows
    assert (small.snapshots, large.snapshots) == (1, 64)


def test_speculation_settings_select_weights_rows_and_pages():
    drafting = derive()
    assert drafting.speculation == DRAFTING and drafting.live_rows == 17
    assert (drafting.weights_bytes, drafting.graph_bytes, drafting.costs.page_bytes) == (60 * GIB, 4 * GIB, 2 * MIB)
    # Without drafting the draft model is not loaded, rows shrink, and each request keeps no verification rows.
    plain = derive(speculation=None)
    assert (plain.weights_bytes, plain.graph_bytes, plain.live_rows) == (58 * GIB, 3 * GIB, 5)
    assert plain.live_bytes == 5 * 80 * MIB and plain.cache_states == 32
    assert plain.pages == (91 * GIB - 61 * GIB - 5 * 80 * MIB) * 2 // (3 * MIB) == 20213
    assert plain.to_dict()["speculation"] is None
    split = derive(gpus=(gpu(0, 50), gpu(1, 50)), speculation=None)
    assert split.world_size == 2 and split.weights_bytes == 29 * GIB and split.costs.state_row_bytes == 40 * MIB


def test_speculation_settings_are_validated_and_described():
    assert DRAFTING.widths == (1, 2, 3, 4) and DRAFTING.fixed_width is None
    fixed = SpeculationParameters(2, adaptive=False)
    assert fixed.widths == (1, 2, 3) and fixed.fixed_width == 3
    assert SpeculationParameters(0).widths == (1,) and SpeculationParameters(0, adaptive=False).fixed_width == 1
    record = derive(speculation=SpeculationParameters(5)).to_dict()["speculation"]
    assert record == {"draft_tokens": 5, "adaptive": True, "mtp_only": False,
                      "packed_verification": False, "draft_prefix_selection": "cpu"}
    assert derive(speculation=SpeculationParameters(5, packed_verification=True, graph_budget=64)).to_dict()["speculation"]["packed_verification"]
    assert display.speculation(None) == "off"
    assert display.speculation(DRAFTING) == "up to 3 draft tokens per round · MTP module"
    assert display.speculation(fixed) == "exactly 2 draft tokens per round · MTP module"
    assert display.speculation(None, brief=True) == "speculation off"
    assert display.speculation(DRAFTING, brief=True) == "up to 3 draft tokens"
    assert display.speculation(fixed, brief=True) == "exactly 2 draft tokens"


def test_headroom_grows_with_prefill_chunk_tokens_past_4096():
    # The fixture sets four GiB of headroom; each 4096 chunk tokens past 4096 add half a GiB.
    assert derive().headroom_bytes == derive(prefill_chunk=1024).headroom_bytes == 4 * GIB
    assert derive(prefill_chunk=8192).headroom_bytes == 4 * GIB + GIB // 2
    assert derive(prefill_chunk=8192).supply_bytes == derive().supply_bytes - GIB // 2
    assert derive(prefill_chunk=8192).to_dict()["headroom_bytes"] == 4 * GIB + GIB // 2


def test_prefill_chunk_defaults_to_the_recipe_and_accepts_an_override():
    assert derive().prefill_chunk == 4096
    assert derive(prefill_chunk=1024).prefill_chunk == 1024
    # A whole-context chunk's activations need about 32 GiB more headroom.
    assert derive(gpus=(gpu(0, 140),), prefill_chunk=262144).prefill_chunk == 262144
    for chunk in (0, 100, 262208):
        with pytest.raises(UsageError, match="multiple of 64 from 64 through 262,144"):
            derive(prefill_chunk=chunk)
    # 16 GiB of reserve and the chunk's 11.5 GiB headroom leave pages for 60,288 tokens, the longest
    # sequence this plan allows.
    with pytest.raises(UsageError, match="65,536 exceeds the 60,288-token context"):
        derive(reserve=16, prefill_chunk=65536)


@pytest.mark.parametrize("model,world_size,packed", [
    pytest.param(ResolvedModel(recipe, checkpoint),
                 recipe.checkpoint_tensor_parallel(checkpoint)[-1] if recipe.name == "mimo_v2" else recipe.tensor_parallel[0],
                 recipe.speculation.packed_verification, id=checkpoint.alias)
    for recipe in recipes() if recipe.speculation is not None for checkpoint in recipe.checkpoints
])
@pytest.mark.parametrize("flags,configuration_text,expected", [
    ((), "", None),
    ((), '[defaults]\nmax_concurrency=16\n', (16, None, None)),
    (("--max-concurrency", "8", "--prefill-chunk-size", "1024", "--speculative-token-capacity", "192"),
     '[models."{alias}"]\nmax_concurrency=16\n', (8, 1024, 192)),
])
def test_cli_plan_resolves_recipe_configuration_and_flag_defaults(tmp_path, model, world_size, packed,
                                                                 flags, configuration_text, expected):
    from ignition.cli.config import Configuration

    path = tmp_path / "flare.toml"
    path.write_text(configuration_text.format(alias=model.checkpoint.alias))
    configuration = Configuration.load(path)
    default_tuple = (
        (4, 4096, 6) if model.recipe.name == "gemma4" else
        (4, 4096, 32) if model.recipe.name == "muse_spark" else
        (8, 8192, 64) if model.recipe.name == "inkling2_mini" else
        (8, 4096, 48) if model.recipe.name == "hy4_flash" else
        (8, 4096, 128) if model.recipe.name == "inkling2" else (32, 4096, 128)
    )
    if expected is None:
        expected = default_tuple
    else:
        exp_conc, exp_chunk, exp_cap = expected
        expected = (
            exp_conc,
            default_tuple[1] if exp_chunk is None else exp_chunk,
            (16 if model.recipe.name == "gemma4" else default_tuple[2]) if exp_cap is None else exp_cap,
        )
    args = parse_args(["export", model.checkpoint.alias, *flags], recipes=recipes())
    devices = tuple(gpu(index, 95) for index in range(world_size))
    if model.recipe.name != "mimo_v2":
        devices = tuple(replace(device, total_bytes=256 * GIB, free_bytes=255 * GIB) for device in devices)
    host = preparation.Host(devices, None)
    if expected[0] > model.recipe.max_concurrency:
        with pytest.raises(UsageError, match="concurrency"):
            params = preparation.resolve_parameters(model, args, configuration)
            preparation.derive(configuration, model, host, gpus=args.gpus, parameters=params)
        return
    params = preparation.resolve_parameters(model, args, configuration)
    plan = preparation.derive(configuration, model, host, gpus=args.gpus, parameters=params)
    assert plan.world_size == world_size
    assert (plan.concurrency, plan.prefill_chunk, plan.speculative_token_capacity) == expected
    assert plan.speculation.draft_tokens == model.recipe.speculation.draft_tokens and plan.speculation.adaptive
    assert plan.speculation.packed_verification is packed
    assert plan.speculation.draft_prefix_selection == model.recipe.speculation.draft_prefix_selection
    record = plan.to_dict()
    assert record["speculative_token_capacity"] == expected[2]
    assert record["speculation"]["packed_verification"] is packed
    assert record["speculation"]["draft_prefix_selection"] == model.recipe.speculation.draft_prefix_selection


@pytest.mark.parametrize("model", [
    pytest.param(ResolvedModel(recipe, checkpoint), id=checkpoint.alias)
    for recipe in recipes() if recipe.name in ("kimi3", "kimi31") for checkpoint in recipe.checkpoints
])
@pytest.mark.parametrize("compression", [("--expert-scale-compression",), ("--expert-scale-compression", "csf")],
                         ids=["bare-csf", "explicit-csf"])
@pytest.mark.parametrize("drafting", [False, True])
def test_kimi_expert_scale_compression_preserves_cli_plan_capacity(tmp_path, model, compression, drafting):
    from ignition.cli.config import Configuration

    path = tmp_path / "flare.toml"
    path.write_text("")
    configuration = Configuration.load(path)
    alias = model.checkpoint.alias
    flags = ["--speculative-token-capacity", "64"] if drafting else ["--no-mtp"]
    devices = tuple(replace(gpu(i, 95), total_bytes=256 * GIB, free_bytes=255 * GIB) for i in range(16))
    host = preparation.Host(devices, None)

    base_args = parse_args(["export", alias, "--gpus", "0-15", "--max-concurrency", "8",
                            "--prefill-chunk-size", "4096", *flags], recipes=recipes())
    default = preparation.derive(configuration, model, host, gpus=base_args.gpus,
                                 parameters=preparation.resolve_parameters(model, base_args, configuration))

    comp_args = parse_args(["export", alias, "--gpus", "0-15", "--max-concurrency", "8",
                            "--prefill-chunk-size", "4096", *flags, *compression], recipes=recipes())
    plan = preparation.derive(configuration, model, host, gpus=comp_args.gpus,
                              parameters=preparation.resolve_parameters(model, comp_args, configuration))

    assert plan.parameters.expert_scale_compression.encoding == plan.to_dict()["expert_scale_compression"] == "csf"
    assert (plan.world_size, plan.concurrency, plan.prefill_chunk) == (16, 8, 4096)
    assert (plan.speculation is not None) is drafting
    if drafting:
        assert plan.speculative_token_capacity == 64
    assert (plan.weights_bytes, plan.pages, plan.cache_states) == (
        default.weights_bytes, default.pages, default.cache_states)


@pytest.mark.parametrize("alias,scope", [("kimi-k3", "all"), ("kimi-k3.1-pro", "kda-input")])
@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("drafting", [False, True])
def test_kimi_online_mxfp8_recipe_default_reaches_cli_plan_and_metadata(tmp_path, monkeypatch, alias, online, scope, drafting):
    from types import SimpleNamespace
    from google.protobuf import json_format
    from ignition.cli.config import Configuration
    from ignition.preparation import rebuild_manifest, runtime_core_pb2 as proto

    monkeypatch.setattr("ignition.preparation.source_identity.source_identity", lambda: "test-source")
    path = tmp_path / "flare.toml"
    path.write_text("")
    configuration = Configuration.load(path)
    model = resolve(alias)
    recipe = model.recipe
    expected_scope = scope if online else "all"
    flags = ["--online-mxfp8"] if online else []
    flags += ["--mtp-draft-tokens", "7"] if drafting else ["--no-mtp"]
    args = parse_args(["export", alias, "--gpus", "0-15", "--max-concurrency", "4",
                       "--image-concurrency", "4", "--prefill-chunk-size", "4096", *flags], recipes=recipes())
    # Inject ample device memory to test configuration, independently of physical fit.
    devices = tuple(replace(gpu(i, 95), total_bytes=256 * GIB, free_bytes=255 * GIB) for i in range(16))
    params = preparation.resolve_parameters(model, args, configuration)
    plan = preparation.derive(configuration, model, preparation.Host(devices, None), gpus=args.gpus, parameters=params)

    assert args.online_mxfp8 is plan.parameters.online_mxfp8.enabled is online
    assert plan.parameters.online_mxfp8.effective_scope == expected_scope
    assert plan.to_dict()["online_mxfp8_scope"] == expected_scope
    assert (plan.world_size, plan.concurrency, plan.prefill_chunk, plan.image_concurrency) == (16, 4, 4096, 4)
    assert not plan.language_model_only and plan.max_images == recipe.max_images
    assert plan.parameters.expert_scale_compression.encoding is None
    assert (plan.speculation is not None) is drafting
    if drafting:
        assert plan.speculation.draft_tokens == 7
        draft_quant = getattr(plan.parameters, "draft_quantization", None)
        if draft_quant is not None:
            assert draft_quant.draft_head_quantization is None
            assert draft_quant.markov_head_quantization is None
    assert plan.to_dict()["online_mxfp8"] is online
    manifest = rebuild_manifest.from_plan(SimpleNamespace(directory=None), plan, compiled_forwards=True)
    encoded = proto.PreparationManifest()
    json_format.ParseDict(manifest, encoded)
    values = {p.key: p.value for p in encoded.parameters}
    assert values["online_mxfp8"].bool_value is online
    assert values["online_mxfp8_scope"].string_value == expected_scope
    assert values["image_concurrency"].uint_value == 4 and not values["language_model_only"].bool_value
    assert values["prefill_chunk"].uint_value == 4096
    assert ("speculation.has_draft" in values) is drafting


def test_online_mxfp8_scope_requires_conversion():
    from hyperflare_recipes.recipes.kimi31.recipe import RECIPE

    params = replace(
        RECIPE.parameters, max_concurrency=1, speculation=None,
        online_mxfp8=replace(RECIPE.parameters.online_mxfp8, enabled=False, scope="kda-input"),
    )
    with pytest.raises(UsageError, match="requires --online-mxfp8"):
        planning.plan(RECIPE, RECIPE.checkpoints[0], (), selected=None, parameters=params, reserve_bytes=0)


def test_online_mxfp8_scope_must_be_declared_by_recipe():
    from hyperflare_recipes.recipes.kimi31.recipe import RECIPE

    broad_params = replace(
        RECIPE.parameters, max_concurrency=1, speculation=None,
        online_mxfp8=OnlineMXFP8Parameters(enabled=True, scope="kda-input", default_scope="all", supported_scopes=("all",)),
    )
    broad_only = replace(RECIPE, parameters=replace(broad_params, online_mxfp8=OnlineMXFP8Parameters()))
    with pytest.raises(UsageError, match="supports all"):
        planning.plan(broad_only, broad_only.checkpoints[0], (), selected=None,
                      parameters=broad_params, reserve_bytes=0)


@pytest.mark.parametrize("command", ["export", "debug-serve", "bench"])
def test_online_mxfp8_scope_option(command):
    default = parse_args([command, "kimi-k3.1-pro", "--online-mxfp8"], recipes=recipes())
    args = parse_args([command, "kimi-k3.1-pro", "--online-mxfp8", "--online-mxfp8-scope", "all"], recipes=recipes())
    assert default.online_mxfp8_scope is None
    assert args.online_mxfp8_scope == "all"


@pytest.mark.parametrize("command", ["export", "debug-serve", "bench"])
def test_draft_body_quantization_option(command):
    default = parse_args([command, "kimi-k3.1-pro", "--speculative-method", "dflash2"], recipes=recipes())
    args = parse_args([command, "kimi-k3.1-pro", "--speculative-method", "dflash2",
                       "--draft-body-quantization", "mxfp8"], recipes=recipes())
    assert default.draft_body_quantization is None
    assert args.draft_body_quantization == "mxfp8"


def test_draft_body_quantization_requires_dflash2_and_positive_depth():
    from hyperflare_recipes.recipes.kimi31.recipe import RECIPE

    devices = tuple(replace(gpu(i, 95), total_bytes=256 * GIB, free_bytes=255 * GIB) for i in range(16))
    with pytest.raises(UsageError, match="requires --speculative-method dflash2"):
        params = replace(
            RECIPE.parameters, max_concurrency=1,
            draft_quantization=replace(RECIPE.parameters.draft_quantization, draft_body_quantization="mxfp8"),
        )
        planning.fit(RECIPE, RECIPE.checkpoints[0], devices, parameters=params, reserve_bytes=0)
    with pytest.raises(UsageError, match="positive learned draft depth"):
        params = replace(
            RECIPE.parameters, max_concurrency=1,
            speculation=replace(RECIPE.speculation, draft_tokens=0, method="dflash2", graph_budget=None),
            draft_quantization=replace(RECIPE.parameters.draft_quantization, draft_body_quantization="mxfp8"),
        )
        planning.fit(RECIPE, RECIPE.checkpoints[0], devices, parameters=params, reserve_bytes=0)


@pytest.mark.parametrize("alias,default", [("kimi-k3", "all"), ("kimi-k3.1-pro", "kda-input")])
@pytest.mark.parametrize("online,scope", [(False, None), (False, "all"), (True, None),
                                        (True, "all"), (True, "kda-input")])
def test_kimi_plan_construction_preserves_explicit_conversion_scope(alias, default, online, scope):
    model = resolve(alias)
    if scope is not None and scope not in model.recipe.parameters.online_mxfp8.supported_scopes:
        return
    devices = tuple(replace(gpu(i, 95), total_bytes=256 * GIB, free_bytes=255 * GIB) for i in range(16))
    params = replace(
        model.recipe.parameters, max_concurrency=4, speculation=None,
        online_mxfp8=replace(model.recipe.parameters.online_mxfp8, enabled=online, scope=scope),
    )
    fitted = planning.fit(model.recipe, model.checkpoint, devices, parameters=params, reserve_bytes=0)
    selected = planning.plan(model.recipe, model.checkpoint, devices, selected=None, parameters=params, reserve_bytes=0)
    expected = scope if scope is not None else default if online else "all"
    assert fitted.parameters.online_mxfp8.effective_scope == selected.parameters.online_mxfp8.effective_scope == expected


def test_context_parallel_pages_hold_every_rank_share_unless_disabled():
    @dataclass(frozen=True)
    class CpParams(RecipeParameters):
        context_parallel: ContextParallelParameters = ContextParallelParameters()

    base = CpParams(max_concurrency=4, max_concurrency_limit=64, prefill_chunk=4096, speculation=None)
    sharded = replace(
        PAIR,
        parameters=base,
        ranks=MappingProxyType({size: replace(costs, page_tokens=64 * size) for size, costs in PAIR.ranks.items()}),
    )
    pair = (gpu(0, 95), gpu(1, 95))
    shared = planning.plan(sharded, sharded.checkpoints[0], pair, selected=None, parameters=base, reserve_bytes=0)
    whole = planning.plan(
        sharded, sharded.checkpoints[0], pair, selected=None,
        parameters=replace(base, context_parallel=ContextParallelParameters(enabled=False)), reserve_bytes=0,
    )
    assert (shared.parameters.context_parallel.enabled, shared.page_tokens) == (True, 128)
    assert (whole.parameters.context_parallel.enabled, whole.page_tokens) == (False, 64)
    assert shared.pages == whole.pages and shared.tokens == 2 * whole.tokens
    assert not derive().to_dict().get("context_parallel", False)


def test_context_parallelism_follows_the_flag():
    glm = resolve("glm-5.3-flash")
    default_args = parse_args(["export", "glm-5.3-flash"], recipes=recipes())
    no_cp_args = parse_args(["export", "glm-5.3-flash", "--no-context-parallel"], recipes=recipes())
    assert preparation.resolve_parameters(glm, default_args).context_parallel.enabled is True
    assert preparation.resolve_parameters(glm, no_cp_args).context_parallel.enabled is False
    with pytest.raises(UsageError, match="is not supported"):
        parse_args(["export", "qwen3.8-flash", "--no-context-parallel"], recipes=recipes())


def test_serialized_branches_are_planned_and_recorded_without_changing_memory():
    @dataclass(frozen=True)
    class OverlapParams(RecipeParameters):
        stream_overlap: StreamOverlapParameters = StreamOverlapParameters()

    base = OverlapParams(max_concurrency=4, max_concurrency_limit=64, prefill_chunk=4096, speculation=None)
    overlapping = replace(PAIR, parameters=base)
    pair = (gpu(0, 95), gpu(1, 95))

    overlapped = planning.plan(overlapping, overlapping.checkpoints[0], pair, selected=None, parameters=base, reserve_bytes=0)
    serial = planning.plan(
        overlapping, overlapping.checkpoints[0], pair, selected=None,
        parameters=replace(base, stream_overlap=StreamOverlapParameters(enabled=False)), reserve_bytes=0,
    )
    assert overlapped.parameters.stream_overlap.enabled and not serial.parameters.stream_overlap.enabled
    assert overlapped.to_dict()["stream_overlap"] is True
    assert json.loads(json.dumps(serial.to_dict()))["stream_overlap"] is False
    assert (serial.pages, serial.cache_states, serial.fixed_bytes) == (
        overlapped.pages, overlapped.cache_states, overlapped.fixed_bytes,
    )


def test_stream_overlap_flag_applies_only_to_recipes_that_overlap_branches():
    glm = resolve("glm-5.3-flash")
    assert preparation.resolve_parameters(glm, parse_args(["export", "glm-5.3-flash"], recipes=recipes())).stream_overlap.enabled
    assert not preparation.resolve_parameters(
        glm, parse_args(["export", "glm-5.3-flash", "--no-stream-overlap"], recipes=recipes()),
    ).stream_overlap.enabled
    with pytest.raises(UsageError, match="is not supported"):
        parse_args(["export", "mimo-v2.6-pro-mopd", "--no-stream-overlap"], recipes=recipes())


def test_language_model_only_plans_no_images_for_recipes_that_declare_it():
    pair = (gpu(0, 95), gpu(1, 95))
    seeing = replace(
        PAIR, max_images=4, image_bytes=32 * MIB,
        parameters=replace(PAIR.parameters, vision=VisionParameters(separable_encoder=True)),
    )

    def planned(model, *, language_model_only=False, image_concurrency=None):
        params = model.parameters
        if language_model_only or image_concurrency is not None:
            params = replace(
                params,
                vision=replace(
                    params.vision or VisionParameters(separable_encoder=False),
                    language_model_only=language_model_only,
                    image_concurrency=image_concurrency,
                ),
            )
        return planning.plan(model, model.checkpoints[0], pair, selected=None, parameters=params, reserve_bytes=0)

    text = planned(seeing, language_model_only=True)
    assert planned(seeing).max_images == 4 and text.max_images == 0
    # Image feature slots follow the vision encoder.
    assert planned(seeing).image_concurrency == 4 and text.image_concurrency == 0
    with pytest.raises(UsageError, match="--image-concurrency requires the vision encoder"):
        planned(seeing, language_model_only=True, image_concurrency=2)
    assert text.to_dict()["language_model_only"] is True and planned(seeing).to_dict()["language_model_only"] is False
    with pytest.raises(UsageError, match="--language-model-only: pair prepares no separable vision encoder"):
        planned(PAIR, language_model_only=True)


def test_first_fitting_gpus_and_smallest_supported_tensor_parallel_size():
    assert [g.index for g in derive(gpus=(gpu(0, 10), gpu(1, 95), gpu(2, 95))).gpus] == [1]
    assert [g.index for g in derive(gpus=(gpu(0, 95, (9, 0)), gpu(1, 95))).gpus] == [1]
    assert [g.index for g in derive(PAIR, (gpu(0, 95), gpu(1, 20), gpu(2, 95))).gpus] == [0, 2]
    split = derive(gpus=(gpu(0, 50), gpu(1, 50)))
    assert split.world_size == 2 and split.weights_bytes == 30 * GIB
    assert derive(PAIR, tuple(gpu(index, 60) for index in range(4))).world_size == 4


def test_explicit_gpus_set_the_tensor_parallel_size():
    gpus = tuple(gpu(index, 95) for index in range(4))
    assert derive(gpus=gpus, selected=(2, 3)).world_size == 2
    with pytest.raises(CapacityError, match=r"pair needs 2 or 4 GPUs; 1 selected; use --gpus 0-1"):
        derive(PAIR, gpus, selected=(0,))
    with pytest.raises(CapacityError, match=r"solo needs 1 or 2 or 4 GPUs; 3 selected; use --gpus 0-3"):
        derive(gpus=gpus, selected=(0, 1, 2))
    with pytest.raises(UsageError, match="names a GPU this host does not have"):
        derive(gpus=gpus, selected=(4,))
    with pytest.raises(CapacityError, match="not an SM103, SM120, or SM121 device"):
        derive(gpus=(gpu(0, 95, (9, 0)),), selected=(0,))
    with pytest.raises(CapacityError, match="pair needs 2 or 4 SM103, SM120, or SM121 GPUs; this host has 1"):
        derive(PAIR, (gpu(0, 95),))


@pytest.mark.parametrize("limit", [0, 65])
def test_concurrency_outside_the_recipe_range_is_a_usage_error(limit):
    with pytest.raises(UsageError, match="must be 1 through 64"):
        derive(concurrency=limit)


def option_arguments(option, arguments):
    """Apply one printed option to the arguments of `derive`."""
    arguments = dict(arguments)
    if match := re.fullmatch(r'\[memory\] reserve = "(\d+)G?"', option):
        arguments["reserve"] = int(match[1])
    elif match := re.fullmatch(r"--max-concurrency (\d+)", option):
        arguments["concurrency"] = int(match[1])
    elif match := re.fullmatch(r"--gpus (\S+)", option):
        arguments["selected"] = planning.parse_gpus(match[1])
    elif option == "--no-mtp":
        arguments["speculation"] = None
    else:
        raise AssertionError(option)
    return arguments


@pytest.mark.parametrize("arguments, expected", [
    (dict(gpus=(gpu(0, 95),), selected=(0,), reserve=25),
     ['[memory] reserve = "24G"', "--max-concurrency 2", "--no-mtp"]),
    (dict(gpus=(gpu(0, 95), gpu(1, 95)), selected=(0,), reserve=25),
     ['[memory] reserve = "24G"', "--max-concurrency 2", "--no-mtp", "--gpus 0-1"]),
    (dict(gpus=(gpu(0, 70.5),)), ["--max-concurrency 3", "--no-mtp"]),
    (dict(gpus=(gpu(0, 72),), concurrency=8), ["--max-concurrency 4", "--no-mtp"]),
    (dict(gpus=(gpu(0, 80), gpu(1, 80)), reserve=41),
     ['[memory] reserve = "40G"', "--max-concurrency 2", "--no-mtp"]),
    # Without speculation the plan has no drafting to give up.
    (dict(gpus=(gpu(0, 95),), selected=(0,), reserve=30, speculation=None), ['[memory] reserve = "28G"']),
])
def test_every_capacity_error_option_fits_on_its_own(arguments, expected):
    with pytest.raises(CapacityError) as error:
        derive(**arguments)
    message = str(error.value)
    assert message.startswith("needs ") and "minimum pages" in message and "\nhas " in message
    options = re.search(r"Options: (.*)\.$", message)[1].split(", or ")
    assert options == expected
    for option in options:
        assert derive(**option_arguments(option, arguments)).fits, option


def test_capacity_error_states_requirement_and_supply():
    with pytest.raises(CapacityError) as error:
        derive(gpus=(gpu(0, 95),), selected=(0,), reserve=25)
    assert str(error.value).splitlines()[:3] == [
        "needs 60.0 GiB weights + 4.0 GiB graphs + 1.7 GiB live state",
        "+ 1.0 GiB minimum pages (4 × 8,192 tokens) = 66.7 GiB;",
        "has 95.0 GiB free − 4.0 GiB headroom − 25.0 GiB reserve = 66.0 GiB.",
    ]


def test_capacity_error_without_a_working_setting_says_so():
    with pytest.raises(CapacityError, match="No setting makes solo fit on this host"):
        derive(gpus=(gpu(0, 40),))
    with pytest.raises(CapacityError, match="No setting makes solo fit on GPU 0"):
        derive(gpus=(gpu(0, 40),), selected=(0,))


def test_automatic_selection_explains_the_closest_tensor_parallel_size():
    with pytest.raises(CapacityError) as error:
        derive(gpus=(gpu(0, 80), gpu(1, 80)), reserve=41)
    assert "= 35.3 GiB per GPU;" in str(error.value)


def test_estimate_record():
    plan = derive(reserve=2)
    record = json.loads(json.dumps(plan.to_dict()))
    assert record["pages"] == plan.pages == 11950 and record["snapshots"] == 32
    assert record["reserve_bytes"] == 2 * GIB and "measured" not in record


def test_gpu_lists_and_restrictions():
    assert planning.parse_gpus("0") == (0,)
    assert planning.parse_gpus("0,2") == (0, 2)
    assert planning.parse_gpus("0-3") == (0, 1, 2, 3)
    assert planning.format_gpus((0, 1, 2)) == "0-2" and planning.format_gpus((0, 2)) == "0,2"
    for text in ("", "a", "0-", "1,1", "3-1", "0,,1"):
        with pytest.raises(UsageError):
            planning.parse_gpus(text)
    gpus = tuple(gpu(index, 95) for index in range(3))
    restricted = planning.restrict_gpus(gpus, "2,GPU-0000")
    assert [(g.index, g.uuid) for g in restricted] == [(0, "GPU-0002"), (1, "GPU-0000")]
    assert planning.restrict_gpus(gpus, "") == ()
    for text in ("7", "GPU-", "1,1", "GPU-9"):
        with pytest.raises(UsageError):
            planning.restrict_gpus(gpus, text)


def test_unified_memory_gpus_plan_from_their_reported_free_memory():
    # The PLE table reaches the GPU through compact row delivery, so no mapped table is subtracted.
    discrete, unified = derive(gpus=(gpu(0, 95),)), derive(gpus=(spark(0, 95),))
    assert discrete.free_bytes == unified.free_bytes == 95 * GIB


def test_tuning_devices_keep_serving_capacity_and_tensor_parallel_size():
    from types import SimpleNamespace
    from ignition.cli.preparation import tuning_devices

    devices = tuple(gpu(index, 95) for index in range(6))
    host = SimpleNamespace(gpus=devices)
    for model, count in ((SOLO, 1), (PAIR, 2)):
        plan = derive(model, devices, selected=tuple(range(count)))
        assert tuning_devices(plan, host, None) is plan
        assert tuning_devices(plan, host, ','.join(str(i) for i in range(count))) == plan
        expanded = tuning_devices(plan, host, '0-5')
        assert expanded.world_size == count
        assert expanded.tuning_gpus == devices
        assert replace(expanded, tuning_gpus=()) == plan
        assert expanded.to_dict()['tuning_gpus'] == list(range(6))


@pytest.mark.parametrize('selection,message', [
    ('1,0', 'start with the serving GPUs'),
    ('0-2', 'complete TP replicas'),
    ('0,1,6,7', 'outside the visible device list'),
])
def test_tuning_devices_reject_invalid_replica_groups(selection, message):
    from types import SimpleNamespace
    from ignition.cli.preparation import tuning_devices

    devices = tuple(gpu(index, 95) for index in range(4))
    with pytest.raises(UsageError, match=message):
        tuning_devices(derive(PAIR, devices), SimpleNamespace(gpus=devices), selection)


@pytest.mark.parametrize('helper,message', [
    (gpu(1, 95, (12, 1)), 'model and architecture'),
    (replace(gpu(1, 95), name='another GPU'), 'model and architecture'),
    (gpu(1, 68), 'needs at least'),
])
def test_tuning_devices_require_matching_hardware_and_memory(helper, message):
    from types import SimpleNamespace
    from ignition.cli.preparation import tuning_devices

    plan = derive(reserve=4)
    with pytest.raises(UsageError, match=message):
        tuning_devices(plan, SimpleNamespace(gpus=(plan.gpus[0], helper)), '0,1')


@pytest.mark.parametrize('model,workers,selection,message', [
    (PAIR, 1, None, 'local serving rank'),
    (SOLO, 1, '0,1', 'local tuning rank'),
])
def test_compiler_budget_rejects_more_local_ranks_than_workers(model, workers, selection, message):
    from types import SimpleNamespace
    from ignition.cli.preparation import tuning_devices

    devices = (gpu(0, 95), gpu(1, 95))
    base = derive(model, devices)
    plan = replace(base, parameters=replace(base.parameters, compile_workers=workers))
    with pytest.raises(UsageError, match=message):
        tuning_devices(plan, SimpleNamespace(gpus=devices), selection)


def test_recipe_b12x_settings_name_b12x_variables():
    from hyperflare_recipes.recipes.kimi3.recipe import RECIPE

    with pytest.raises(ValueError, match="B12X_"):
        replace(RECIPE, b12x_environment={"CUDA_VISIBLE_DEVICES": "0"})
    with pytest.raises(ValueError, match="B12X_"):
        replace(RECIPE, b12x_environment={"B12X_PCIE_DCP_THREADS": 512})


def test_sharded_hyperconnections_require_qwen_tp_and_survive_rebuild_serialization(monkeypatch):
    from types import SimpleNamespace
    from google.protobuf import json_format
    from ignition.preparation import rebuild_manifest, runtime_core_pb2 as proto
    from hyperflare_recipes.recipes.qwen38.recipe import RECIPE as QWEN

    monkeypatch.setattr("ignition.preparation.source_identity.source_identity", lambda: "test-source")
    pair = (gpu(0, 95), gpu(1, 95))
    ordinary = derive(model=QWEN, gpus=pair, selected=(0, 1))
    assert not ordinary.parameters.shard_hyperconnections
    sharded = replace(ordinary, parameters=replace(ordinary.parameters, shard_hyperconnections=True))
    assert sharded.to_dict()["shard_hyperconnections"] is True
    assert replace(sharded, parameters=replace(sharded.parameters, shard_hyperconnections=False)) == ordinary
    for plan in (ordinary, sharded):
        manifest = rebuild_manifest.from_plan(SimpleNamespace(directory=None), plan, compiled_forwards=True)
        encoded = json_format.ParseDict(manifest, proto.PreparationManifest())
        decoded = proto.PreparationManifest.FromString(encoded.SerializeToString())
        values = {entry.key: entry.value.bool_value for entry in decoded.parameters if entry.value.HasField("bool_value")}
        assert values["shard_hyperconnections"] is plan.parameters.shard_hyperconnections
    with pytest.raises(UsageError, match="is not supported"):
        parse_args(["export", "glm-5.3-flash", "--shard-hyperconnections"], recipes=recipes())
    with pytest.raises(UsageError, match="tensor parallelism"):
        derive(model=QWEN, gpus=(gpu(0, 95),), selected=(0,),
               parameters=replace(QWEN.parameters, max_concurrency=4, shard_hyperconnections=True))


def test_moe_decode_a16_geometry_and_rebuild_metadata(monkeypatch):
    from types import SimpleNamespace
    from google.protobuf import json_format
    from ignition.preparation import rebuild_manifest, runtime_core_pb2 as proto
    from hyperflare_recipes.recipes.qwen38.recipe import RECIPE as QWEN

    monkeypatch.setattr("ignition.preparation.source_identity.source_identity", lambda: "test-source")
    ordinary = derive(model=QWEN, gpus=(gpu(0, 95),), selected=(0,))
    for enabled in (False, True):
        plan = replace(ordinary, parameters=replace(ordinary.parameters, force_moe_decode_a16=enabled))
        assert plan.to_dict()["force_moe_decode_a16"] is enabled
        manifest = rebuild_manifest.from_plan(SimpleNamespace(directory=None), plan, compiled_forwards=True)
        encoded = json_format.ParseDict(manifest, proto.PreparationManifest())
        decoded = proto.PreparationManifest.FromString(encoded.SerializeToString())
        values = {entry.key: entry.value.bool_value for entry in decoded.parameters if entry.value.HasField("bool_value")}
        assert values["force_moe_decode_a16"] is enabled
    with pytest.raises(UsageError, match="is not supported"):
        parse_args(["export", "glm-5.3-flash", "--force-moe-decode-a16"], recipes=recipes())
    with pytest.raises(UsageError, match="SM120/121"):
        derive(model=QWEN, gpus=(gpu(0, 95, (10, 3)),), selected=(0,),
               parameters=replace(QWEN.parameters, max_concurrency=4, force_moe_decode_a16=True))
    with pytest.raises(UsageError, match="intermediate width divisible by 64"):
        derive(model=QWEN, gpus=tuple(gpu(i, 95) for i in range(4)), selected=(0, 1, 2, 3),
               parameters=replace(QWEN.parameters, max_concurrency=4, force_moe_decode_a16=True))
    with pytest.raises(UsageError, match="mixed-prefill"):
        derive(model=QWEN, gpus=(gpu(0, 95),), selected=(0,),
               parameters=replace(QWEN.parameters, max_concurrency=4,
                                  mixed_prefill=MixedPrefillParameters(enabled=True),
                                  force_moe_decode_a16=True))


def test_glm53_declares_w4a16_a4_prefill_in_b12x_environment():
    from hyperflare_recipes.recipes.glm53.recipe import RECIPE

    assert RECIPE.b12x_environment.get("B12X_W4A16_A4_PREFILL") == "1"


def test_recipe_prepare_forwards_plan_context_headroom_and_options(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import sys

    def stub_bootstrap(recipe_name):
        module_name = f"hyperflare_recipes.recipes.{recipe_name}.bootstrap"
        captured = {}

        class CapturingConfig:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                raise RuntimeError("captured")

        fake = SimpleNamespace(PreparationConfig=CapturingConfig, prepare=lambda *a, **k: None, prepare_rank=lambda *a, **k: None)
        monkeypatch.setitem(sys.modules, module_name, fake)
        return captured

    from ignition.planning import HEADROOM_BYTES
    from hyperflare_recipes.recipes.gemma4.recipe import RECIPE as GEMMA4
    from hyperflare_recipes.recipes.minimax_m3.recipe import RECIPE as MINIMAX
    from hyperflare_recipes.recipes.deepseek4.recipe import RECIPE as DS4
    from hyperflare_recipes.recipes.deepseek41.recipe import RECIPE as DS41
    from hyperflare_recipes.recipes.mimo_v2.recipe import RECIPE as MIMO

    # Gemma4 forwards autotune_effort, roce_chain, and plan.recipe.context.
    gemma_plan = derive(
        model=GEMMA4,
        gpus=tuple(replace(gpu(i, 200), total_bytes=200 * GIB) for i in range(2)),
        selected=(0, 1),
        parameters=replace(GEMMA4.parameters, max_concurrency=4, speculation=None,
                           autotune_effort="off", roce_chain=True),
    )
    captured = stub_bootstrap("gemma4")
    with pytest.raises(RuntimeError, match="captured"):
        GEMMA4.prepare(tmp_path, gemma_plan, compile_forwards=True, kernel_cache=None, progress=None,
                       ranks=SimpleNamespace(hosts=("a", "b")))
    assert captured["parameters"].autotune_effort == "off"
    assert captured["parameters"].roce_chain is True
    assert captured["context"] == GEMMA4.context

    # MiniMax M3 does not accept --prompt-logprobs and forwards roce_chain + recipe context.
    with pytest.raises(UsageError, match="is not supported"):
        parse_args(["export", "minimax-m3-1-flash", "--prompt-logprobs"], recipes=recipes())

    minimax_gpus = tuple(replace(gpu(i, 256), total_bytes=256 * GIB) for i in range(4))
    minimax_chain_plan = derive(
        model=MINIMAX, gpus=minimax_gpus, selected=(0, 1, 2, 3),
        parameters=replace(MINIMAX.parameters, max_concurrency=4, speculation=None, roce_chain=True),
    )
    captured = stub_bootstrap("minimax_m3")
    with pytest.raises(RuntimeError, match="captured"):
        MINIMAX.prepare(tmp_path, minimax_chain_plan, compile_forwards=True, kernel_cache=None, progress=None,
                        ranks=SimpleNamespace(hosts=("a", "b", "c", "d")))
    assert captured["parameters"].roce_chain is True
    assert captured["context"] == MINIMAX.context

    # DeepSeek V4, V4.1, and MiMo V2 honor plan.headroom_bytes and plan.recipe.context.
    for recipe, world in ((DS4, 4), (DS41, 4), (MIMO, 4)):
        devices = tuple(replace(gpu(i, 256), total_bytes=256 * GIB) for i in range(world))
        plan = derive(model=recipe, gpus=devices, selected=tuple(range(world)),
                      speculation=None, prefill_chunk=8192)
        assert plan.headroom_bytes > HEADROOM_BYTES
        captured = stub_bootstrap(recipe.name)
        with pytest.raises(RuntimeError, match="captured"):
            recipe.prepare(tmp_path, plan, compile_forwards=True, kernel_cache=None, progress=None)
        assert captured["headroom_bytes"] == plan.headroom_bytes + plan.reserve_bytes
        assert captured["context"] == recipe.context


@pytest.mark.parametrize("recipe", recipes(), ids=lambda recipe: recipe.name)
def test_every_recipe_inherits_base_config_and_forwards_shared_plan_fields(monkeypatch, tmp_path, recipe):
    from importlib import import_module
    from ignition.preparation.bootstrap import BasePreparationConfig

    bootstrap_name = {"kimi31": "kimi3", "inkling2_mini": "inkling2"}.get(recipe.name, recipe.name)
    bootstrap = import_module(f"hyperflare_recipes.recipes.{bootstrap_name}.bootstrap")
    assert issubclass(bootstrap.PreparationConfig, BasePreparationConfig)

    world = min(recipe.tensor_parallel)
    devices = tuple(replace(gpu(i, 4096), total_bytes=4096 * GIB) for i in range(world))
    params = replace(
        recipe.parameters,
        max_concurrency=4,
        speculation=None,
        prefill_chunk=8192,
        autotune_effort="audit",
        compile_workers=3,
        lock_gpu_clocks=True,
        native_dense_activations=NativeDenseActivationsParameters(enabled=True),
    )
    plan = derive(model=recipe, gpus=devices, selected=tuple(range(world)), parameters=params)
    captured = {}

    class Captured(Exception):
        pass

    def capture_config(**kwargs):
        captured.update(kwargs)
        raise Captured

    monkeypatch.setattr(bootstrap, "PreparationConfig", capture_config)
    with pytest.raises(Captured):
        recipe.prepare(tmp_path, plan, compile_forwards=True, kernel_cache=None, progress=None)

    assert captured["context"] == recipe.context
    assert captured["headroom_bytes"] == plan.headroom_bytes + plan.reserve_bytes
    forwarded = captured["parameters"]
    assert forwarded.autotune_effort == "audit"
    assert forwarded.compile_workers == 3
    assert forwarded.native_dense_activations.enabled is True
    assert forwarded.lock_gpu_clocks is True
    assert forwarded.max_concurrency == plan.concurrency
    assert forwarded.prefill_chunk == 8192
