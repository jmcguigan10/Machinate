# Machinator

Machinator is a prompt-first control-plane CLI for ML workspaces and pipeline repos.

The intended UX is:

```bash
macht workspace init
macht guide beginner
macht grab data
macht legate report --data --dataset demo-dataset
macht init pipeline --name demo-pipeline
macht build configs --pipeline demo-pipeline
macht task list --pipeline demo-pipeline
macht run train --pipeline demo-pipeline
macht check
macht doctor
```

This repo is the source of truth for the installed `macht` command. Homebrew is only the delivery layer.

If you want the operator walkthrough rather than the reference overview, start with [docs/beginner-guide.md](docs/beginner-guide.md).

## Status

The first scaffold in this repo supports:

- `macht workspace init`
- `macht guide beginner`
- `macht check`
- `macht test`
- `macht workspace show`
- `macht init pipeline`
- `macht build configs`
- `macht new pipeline` for manual scaffolding
- `macht grab data`
- `macht collate pipeline`
- `macht legate report --data`
- `macht model validate`
- `macht model edit`
- `macht model diff`
- `macht model migrate`
- `macht model compile`
- `macht task list`
- `macht run <task>`
- `macht doctor`

The generated pipeline repo is intentionally lightweight, but it is now native to Machinator. It gives you a recipe-first `machinate.toml` pipeline key, a `config-ref.toml`, a `config/` directory, a `data/` directory, and a workspace registration manifest. It does not generate or rely on a `Makefile`, and the runtime tasks are executed from Machinator’s installed packages rather than repo-local bootstrap Python.

## Install For Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The default install now includes `questionary`, so interactive commands use the richer prompt UI automatically when running in a real TTY.

## Quick Start

Create a workspace:

```bash
macht workspace init
```

Read the built-in operator guide at any time:

```bash
macht guide list
macht guide beginner
macht guide workflow
```

Stage a dataset into the workspace:

```bash
macht grab data
macht grab data --src https://example.com/data.csv --name remote-dataset
```

Delegate a structured data report to Codex CLI:

```bash
macht legate report --data --dataset demo-dataset
macht legate report --data --dataset demo-dataset --notes "This came from a churn export and may contain leakage."
macht legate report --data --dataset demo-dataset --notes-prompt
```

The `legate` flow is intentionally native to Machinator:

- it resolves the dataset from the workspace asset registry or a direct path
- it runs `codex exec` non-interactively with a JSON schema contract
- it stores prompt, raw response, and final JSON artifact under `outputs/reports/legate/`
- it prints the agent's plain-English summary back to the terminal after completion

Initialize a thin pipeline directory from the delegated report:

```bash
macht init pipeline --name demo-pipeline
cd pipelines/demo-pipeline
macht build configs
macht task list
macht run train
```

`macht init pipeline` is now the preferred dataset-first path. It creates a thin pipeline directory with `machinate.toml`, `config-ref.toml`, `data/<dataset>`, and `data/reports/<chosen-report>.json`. The pipeline directory is for references, configs, logs, plots, and artifacts, not repo-local bootstrap Python.

`macht build configs` turns the chosen JSON data report into generated config files under `config/`, and those generated files are the inputs referenced by `config-ref.toml` and consumed by the built-in package-managed runtime tasks.

`macht new pipeline` still exists, but it is now the manual/advanced escape hatch when you intentionally want to scaffold a pipeline before report-driven collation.

The current compiler paths support:

- `tabular_mlp`
- `binary_classification`
- `tabular` modality
- `vision_cnn`
- `binary_classification`
- `vision` modality
- `vision_resnet`
- `binary_classification`
- `vision` modality
- `transformer_encoder`
- `binary_classification`
- `text` modality

The compiler writes deterministic build artifacts under `outputs/compiled_model/`, including a generated Python module and a `param_store_manifest.json`.

`macht run train` now consumes the generated `config/model.yaml` and `config/training.yaml` files referenced by `config-ref.toml`. The built-in package-managed task compiles the model, writes the compiled artifact paths into the run summary, and emits an initialized checkpoint when `torch` is available in the active runtime.

`macht model diff` and `macht model migrate` are the first editability layer for the IR. Diff produces a migration plan, and migrate can preserve compatible weights from an existing checkpoint when `torch` is installed.

Check install and workspace health:

```bash
macht doctor
```

Run contributor verification through Machinator itself:

```bash
macht check
macht check --fast
macht test unit
macht test integration
macht test rust
```

## Workspace Model

Machinator separates three layers:

1. Global CLI install
   The `macht` command installed from this repo.
2. Workspace state
   A `.machinator/` directory inside each managed workspace.
3. Pipeline runtime environments
   Repo-local environments created per pipeline, not shared globally.

Pipelines are configured by `machinate.toml` and executed through `macht`, not through generated Make targets.

The repo now also ships a checked-in recipe gallery workspace under `examples/recipe-gallery-workspace/`. That gallery downloads curated public datasets, collates five starter pipelines, and shows the same recipe-first flow across:

- `tabular.binary.basic`
- `tabular.binary.deep`
- `text.binary.transformer`
- `vision.binary.cnn`
- `vision.binary.resnet`

The long-term architecture now has two layers:

- pipeline repos stay thin and mostly hold specs/configs
- `Machinator` owns the typed IR, validators, compilers, and migration logic

The first IR foundation is a Rust crate at `rust/machinator-ir/`. It now validates specs and computes diff/migration plans, while the initial compiler path remains in Python so the feature is usable immediately.

The current workspace scaffold creates:

- `.machinator/workspace.json`
- `.machinator/pipelines/`
- `.machinator/assets/`
- `.envs/venvs/`
- `data/staged/`
- `outputs/`
- `pipelines/`

## Homebrew Shape

The intended release flow is:

1. Tag a release in the `Machinator` repo.
2. Update the Homebrew formula in your tap repo.
3. Install with:

```bash
brew tap jmcguigan10/tap
brew install machinator
```

See [docs/homebrew-release.md](docs/homebrew-release.md) and [packaging/homebrew/machinator.rb](packaging/homebrew/machinator.rb) for the initial formula template and release process.
The Homebrew install tracks the latest tagged release, not unreleased changes on `main`.

## Local Homebrew Smoke Test

You can also test the Homebrew install flow locally before pushing anything to GitHub:

1. Build a local release tarball:

```bash
./scripts/build_release_artifact.sh
./scripts/render_homebrew_formula.py --owner jmcguigan10
```

2. Update the formula in your local tap repo, for example:

```text
/Users/johnny/Projects/homebrew-tap/Formula/machinator.rb
```

3. Register the local tap and install from it:

```bash
brew tap jmcguigan10/tap /Users/johnny/Projects/homebrew-tap
brew install jmcguigan10/tap/machinator
```

4. Verify the brewed command:

```bash
$(brew --prefix)/bin/macht --help
```

## Repo Layout

```text
scripts/
  render_homebrew_formula.py
src/machinator/
  cli.py
  core.py
  modeling.py
  ui.py
  commands/
rust/machinator-ir/
docs/
packaging/homebrew/
tests/
```
