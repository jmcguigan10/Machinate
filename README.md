# Machinate

Machinate is a prompt-first control-plane CLI for ML workspaces and pipeline repos.

The intended UX is:

```bash
macht workspace init
macht new pipeline
macht grab data
macht task list
macht run train --experiment baseline
macht doctor
```

This repo is the source of truth for the installed `macht` command. Homebrew is only the delivery layer.

## Status

The first scaffold in this repo supports:

- `macht workspace init`
- `macht workspace show`
- `macht new pipeline`
- `macht grab data`
- `macht legate report --data`
- `macht task list`
- `macht run <task>`
- `macht doctor`

The generated pipeline repo is intentionally lightweight, but it is now native to Machinate. It gives you a `machinate.toml` pipeline config, TOML experiment config, a starter Python task module, and a workspace registration manifest. It does not generate or rely on a `Makefile`.

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

Create a placeholder pipeline repo inside that workspace:

```bash
macht new pipeline
```

Stage a dataset into the workspace:

```bash
macht grab data
macht grab data --src https://example.com/data.csv --name remote-dataset
```

Inspect native tasks and run one:

```bash
cd pipelines/demo-pipeline
macht task list
macht run validate --experiment baseline
macht run train --experiment baseline
```

Delegate a structured data report to Codex CLI:

```bash
macht legate report --data --dataset demo-dataset
macht legate report --data --dataset demo-dataset --notes "This came from a churn export and may contain leakage."
```

The `legate` flow is intentionally native to Machinate:

- it resolves the dataset from the workspace asset registry or a direct path
- it runs `codex exec` non-interactively with a JSON schema contract
- it stores prompt, raw response, and final JSON artifact under `outputs/reports/legate/`
- it prints the agent's plain-English summary back to the terminal after completion

Check install and workspace health:

```bash
macht doctor
```

## Workspace Model

Machinate separates three layers:

1. Global CLI install
   The `macht` command installed from this repo.
2. Workspace state
   A `.machinate/` directory inside each managed workspace.
3. Pipeline runtime environments
   Repo-local environments created per pipeline, not shared globally.

Pipelines are configured by `machinate.toml` and executed through `macht`, not through generated Make targets.

The current workspace scaffold creates:

- `.machinate/workspace.json`
- `.machinate/pipelines/`
- `.machinate/assets/`
- `.envs/venvs/`
- `data/staged/`
- `outputs/`
- `pipelines/`

## Homebrew Shape

The intended release flow is:

1. Tag a release in the `Machinate` repo.
2. Update the Homebrew formula in your tap repo.
3. Install with:

```bash
brew tap jmcguigan10/tap
brew install machinate
```

See [docs/homebrew-release.md](docs/homebrew-release.md) and [packaging/homebrew/machinate.rb](packaging/homebrew/machinate.rb) for the initial formula template and release process.
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
/Users/johnny/Projects/homebrew-tap/Formula/machinate.rb
```

3. Register the local tap and install from it:

```bash
brew tap jmcguigan10/tap /Users/johnny/Projects/homebrew-tap
brew install jmcguigan10/tap/machinate
```

4. Verify the brewed command:

```bash
$(brew --prefix)/bin/macht --help
```

## Repo Layout

```text
scripts/
  render_homebrew_formula.py
src/machinate/
  cli.py
  core.py
  ui.py
  commands/
docs/
packaging/homebrew/
tests/
```
