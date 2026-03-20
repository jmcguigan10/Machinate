# Machinator

Machinator is a prompt-first control-plane CLI for ML workspaces and pipeline repos.

The intended UX is:

```bash
macht workspace init
macht guide beginner
macht grab data
macht legate report --data --dataset demo-dataset
macht collate pipeline --create
macht model validate
macht model edit --set 'hidden_dims=[256,64]' --output model.v2.toml
macht model diff --new model.v2.toml
macht model compile
macht task list
macht run train --experiment baseline
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

The generated pipeline repo is intentionally lightweight, but it is now native to Machinator. It gives you a recipe-first `machinate.toml` pipeline key, a `config/` directory, a `data/` directory, a starter Python task module, and a workspace registration manifest. It does not generate or rely on a `Makefile`. The new model system starts moving more pipeline definition into spec files such as `dataset_facts.toml`, `model.toml`, and `training.toml`.

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

Collate a spec-first model pipeline from a delegated data report:

```bash
macht collate pipeline --create

# scripted use can still pin an explicit report path
macht collate pipeline --create --report /path/to/outputs/reports/legate/report.json
cd pipelines/demo-pipeline
macht task list
macht model validate
macht model edit --set 'hidden_dims=[256,64]' --output model.v2.toml
macht model diff --new model.v2.toml
macht model compile
macht run train --experiment baseline --dataset /path/to/data.csv
```

`macht collate pipeline --create` is now the preferred dataset-first path. If `--report` is omitted, Machinator uses the latest compatible delegated data report from the active workspace and prompts only when multiple candidates exist. The report-driven create path now derives the scaffold directly from the report facts: pipeline type comes from modality, the starter recipe defaults from the inferred task/modality, the template is the native Python scaffold, and the standard starter tasks are included automatically.

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

`macht run train` now consumes `model.toml` and `training.toml` directly when those specs are present. The generated starter task compiles the model, writes the compiled artifact paths into the run summary, and emits an initialized checkpoint when `torch` is available in the active runtime.

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
