# Machinator

Machinator is a prompt-first control-plane CLI for machine-learning workspaces, report-driven pipeline scaffolds, and reproducible starter builds. It gives a workspace a typed structure instead of leaving data, configs, and tasks to drift across notebooks, shell snippets, and stale README commands.

The primary command is `machinator`. The shorter `macht` alias is kept for convenience and backward compatibility.

## Why

Most small ML repos start as an experiment and end as archaeology. Machinator exists to make workspace state inspectable, pipeline setup repeatable, and starter model and task flows easy to regenerate from structured dataset facts.

## Install

For local development:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
machinator --help
```

For Homebrew installs:

```bash
brew tap jmcguigan10/tap
brew install machinator
machinator --help
```

## Quickstart

Create a workspace:

```bash
machinator workspace init --path demo-workspace --name demo-workspace
cd demo-workspace
```

Inspect the built-in guides and environment checks:

```bash
machinator guide beginner
machinator doctor
```

Run the report-driven pipeline flow:

```bash
machinator legate report --data --dataset demo-dataset
machinator init pipeline --name demo-pipeline
machinator build configs --pipeline demo-pipeline
machinator task list --pipeline demo-pipeline
machinator run train --pipeline demo-pipeline
```

For contributors, the smallest verification loop is:

```bash
machinator check --fast
python3 -m pytest -q
```

## Expected Output

`machinator --help` should show top-level commands including `workspace`, `guide`, `init`, `build`, `collate`, `model`, `task`, and `run`.

`machinator workspace init --path demo-workspace --name demo-workspace` should print the initialized workspace path, its `.machinator/workspace.json` manifest, and the global config path.

## Repository Layout

```text
src/machinator/          CLI, workspace management, IR helpers, and task runtime
tests/                   Unit and integration tests
examples/                Checked-in example workspace and starter pipelines
docs/                    Operator and release notes
packaging/homebrew/      Homebrew formula template
rust/machinator-ir/      Rust-side IR validation and migration groundwork
scripts/                 Release and gallery tooling
```

## Included Example

`examples/recipe-gallery-workspace/` is a portable sample workspace with staged datasets, delegated report fixtures, and starter pipelines for tabular, text, and vision flows. It is intended as a concrete repo artifact, not just a screenshot prop.

## Current Scope

- Workspace initialization and health checks
- Dataset-first pipeline scaffolding
- Config generation from structured report JSON
- Native package-managed pipeline tasks
- Model-spec validation, diffing, migration, and compilation
- Homebrew packaging support

## Limitations

- The `legate` flow expects a compatible non-interactive Codex CLI environment.
- The current starter model families are intentionally limited to tabular MLPs, transformer encoders, CNNs, and small ResNet-style image models.
- Some paths are still geared toward starter workflows rather than full production MLOps systems.

## License

MIT
