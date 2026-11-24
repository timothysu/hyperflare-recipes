> **Coding agents:** skip this README and read [AGENTS.md](AGENTS.md) directly.

# Hyperflare Recipes

Official model recipes, reusable PyTorch and b12x model components, and
qualification drivers for preparing runtime cores served by
[Hyperflare](https://github.com/NVIDIA/hyperflare-inference) (Flare / Helios).

## Design principles

- **Embrace PyTorch and FlashInfer/b12x at build time; serve from Rust.**
  Recipes load checkpoints, autotune b12x operators, compile complete forwards,
  and capture CUDA graphs into a runtime core exported by `ignition` and
  `flare_heliopause`. Helios in `NVIDIA/hyperflare-inference` serves that core
  without Python.
- **Build on the model ecosystem.** Reuse canonical Transformers forwards where
  their dataflow fits, and replace only the smallest coherent modules needed for
  layout, state, or execution control.
- **Share reusable execution mechanisms without a rigid framework.** Keep
  model-specific composition in `hyperflare_recipes/recipes/` and reusable
  projections, MoE, attention, MLA, GDN, Mamba, head exchange, and speculation
  blocks in `hyperflare_recipes/components/`.
- **Keep core Flare small and recipe-free.** Core `ignition` in
  `NVIDIA/hyperflare-inference` owns preparation orchestration, planning,
  export, and the 13 Helios-ABI component interfaces; Helios owns scheduling,
  cache storage, and request handling.

## Repository layout

- `hyperflare_recipes/recipes/` — official model recipes (`deepseek4`,
  `deepseek41`, `gemma4`, `glm53`, `inkling2` including `inkling2_mini`,
  `kimi3`, `kimi31`, `ling32`, `mimo_v2`, `minimax_m3`, `muse_spark`,
  `nemotron_h`, `qwen38`, `qwen38_dense`, `step5`).
- `hyperflare_recipes/components/` — reusable b12x and Triton model building
  blocks.
- `hyperflare_recipes/fixtures/` — operator-tuning benchmark fixtures.
- `hyperflare_recipes/reference/` — numerical reference implementations.
- `hyperflare_recipes/cli.py` — `ignition` CLI entrypoint wrapping
  `ignition.cli.main(...)` from `NVIDIA/hyperflare-inference`.
- `qualify/`, `tests/`, `scripts/`, `tools/` — model qualification drivers,
  benchmarks, and recipe test suites.
- `docs/` — [model preparation and serving](docs/model-preparation.md),
  [model development](docs/model-development.md),
  [preparation development](docs/preparation-development.md),
  [GPU performance work](docs/performance-work.md),
  [direct execution](docs/direct-execution.md),
  [compressed scale layout](docs/tp-checkpoint-layout.md), and
  [commit design review](docs/design-review.md).

## Usage

Install dependencies (including `ignition[prepare]` and `flare-heliopause` from
`NVIDIA/hyperflare-inference`) and export a model core:

```sh
uv sync --extra test
.venv/bin/ignition export MODEL --gpus 0
```

Serve the exported core with the native `flare` executable from
`NVIDIA/hyperflare-inference`:

```sh
flare run CORE
```

All core preparation and publishing subcommands (`export`, `debug-serve`,
`bench`, `join`, `keygen`, `key-password`, `sign`, `edit`, `combine`,
`convert`) are provided by `ignition.cli` and inherited automatically. See
[model preparation and serving](docs/model-preparation.md) for hardware setup
and per-model commands.

## Building an independent recipe repository

External recipe repositories do not need to merge into Flare or
`hyperflare-recipes`. Depend on `ignition` from `NVIDIA/hyperflare-inference`
(and optionally `hyperflare-recipes` to reuse b12x model components) and expose
a CLI entrypoint calling `ignition.cli.main`:

```python
from pathlib import Path
from ignition.cli import main as ignition_main
from .recipes import RECIPES

def main(argv=None) -> int:
    return ignition_main(
        argv,
        recipes=RECIPES,
        prog="my-recipes",
        source_roots=(("my_recipes", Path(__file__).resolve().parent),),
    )
```

## Licensing and contributions

Project code is licensed under [Apache 2.0](LICENSE). Contributions require the
Developer Certificate of Origin sign-off described in
[CONTRIBUTING.md](CONTRIBUTING.md) and commit design review in
[docs/design-review.md](docs/design-review.md). Third-party attributions are
listed in [NOTICE](NOTICE); upstream terms remain applicable to third-party code
and assets.

The DeepSeek recipe assets retain their
[MIT license](hyperflare_recipes/recipes/deepseek41/assets/LICENSE). Upstream
chat templates are fetched with the model checkpoints and annotated locally
using bundled insertion patches. Gemma uses pinned Google sources, and Kimi K3 /
K3.1 segmented renderers are included as project code. See
[sampling and frontend metadata](docs/model-development.md#sampling-and-frontend-metadata).
