# Machinate

Machinate is a prompt-first control-plane CLI for ML workspaces and pipeline repos.

The intended UX is:

```bash
macht workspace init
macht new pipeline
macht grab data
macht doctor
```

This repo is the source of truth for the installed `macht` command. Homebrew is only the delivery layer.

## Status

The first scaffold in this repo supports:

- `macht workspace init`
- `macht workspace show`
- `macht new pipeline`
- `macht grab data`
- `macht doctor`

The generated pipeline repo is intentionally skeletal. It gives you a placeholder repo with a `Makefile`, baseline config, package directory, and registration manifest, but it does not ship real training code yet.

## Install For Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you want richer interactive prompts, install the optional prompt extra:

```bash
pip install -e ".[prompts]"
```

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
```

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
