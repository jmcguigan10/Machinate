# Machinator

A prompt-first control-plane CLI for machine-learning workspaces, pipelines, and reproducible experiment scaffolds.

## Overview

Machinator is a command-line system for creating, validating, migrating, and operating ML pipeline repositories. It treats a pipeline workspace as a structured object rather than a pile of scripts held together by optimism and coffee residue.

The project combines:

- workspace initialization
- typed pipeline specifications
- validation and migration logic
- deterministic build artifacts
- task execution
- model/pipeline compilation paths
- optional AI-assisted report and code-edit flows
- packaging support, including a Homebrew tap

## Why this exists

ML projects often decay into ad hoc notebooks, shell scripts, stale config files, and README commands that stopped working sometime around the Bronze Age. Machinator is an attempt to impose structure on that mess by making the workspace itself inspectable, validated, and reproducible.

## Features

- Initialize pipeline workspaces from templates.
- Validate model and pipeline specs.
- Compile supported model families into build artifacts.
- Run task lists from a consistent control plane.
- Show diffs and migration plans between workspace states.
- Generate beginner-oriented guidance for a workspace.
- Produce pipeline reports from a structured JSON contract.
- Support tabular MLPs, CNNs, ResNets, transformer encoders, and related templates.

## Installation

```bash
git clone https://github.com/jmcguigan10/Machinator.git
cd Machinator
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If using Homebrew packaging:

```bash
brew tap jmcguigan10/tap
brew install machinator
```

## Quickstart

Create a workspace:

```bash
machinator workspace init my-workspace
cd my-workspace
```

Inspect available commands:

```bash
machinator --help
machinator doctor
```

Validate a model or pipeline spec:

```bash
machinator model validate path/to/model.yaml
machinator check
```

Compile a supported pipeline configuration:

```bash
machinator compile path/to/spec.yaml
```

## Example workflow

```bash
machinator workspace init demo-ml-workspace
cd demo-ml-workspace
machinator init pipeline tabular_mlp
machinator check
machinator compile configs/model.yaml
machinator task list
machinator task run train-smoke
```

Replace the commands above with the exact command sequence that works in the current release.

## Architecture

Machinator is organized around a typed intermediate representation for ML workspaces.

Suggested diagram to add:

```text
user prompt / CLI
      ↓
workspace spec
      ↓
validation + migration
      ↓
compiler / task runner
      ↓
generated pipeline artifacts
```

## Repository layout

```text
src/machinator/          Python CLI and orchestration code
tests/                   test suite
docs/                    documentation
examples/                example workspaces and recipes
rust/machinator-ir/      Rust validation / IR components
packaging/homebrew/      Homebrew packaging support
scripts/                 development utilities
```

## What reviewers should notice

This project demonstrates:

- CLI design
- typed configuration and validation
- reproducible ML workflow design
- packaging and release management
- Python/Rust mixed tooling
- practical experiment infrastructure

## Current status

Machinator is actively evolving. Some compiler paths and workspace recipes may be experimental. The stable surface should be listed here by release version.

## Roadmap

- Add a screenshot or terminal recording of the CLI.
- Add one fully reproducible example from init to training.
- Add a release table with versioned features.
- Separate stable commands from experimental commands.
- Add architecture docs for the IR and migration system.

## License

MIT
