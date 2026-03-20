# Machinator Beginner Guide

This guide is for the first time you sit down with `macht` and want to understand what you are supposed to do, in what order, and why.

You can read this guide from the CLI too:

```bash
macht guide beginner
macht guide workflow
```

Machinator is not a normal ML repo template generator. It is a control plane for ML work:

- you install one global CLI: `macht`
- you create a workspace
- inside that workspace, you create or manage pipeline repos
- pipelines stay relatively thin
- model and training structure increasingly live in specs such as `dataset_facts.toml`, `model.toml`, and `training.toml`

If that feels unusual, that is normal. The easiest way to use Machinator is to think in terms of a guided workflow rather than a library API.

## The Mental Model

Machinator has three layers:

1. Global CLI
   The `macht` command installed through Homebrew or from source.
2. Workspace
   A directory that holds shared state, staged data, outputs, and registered pipelines.
3. Pipeline repo
   A specific project under `pipelines/` with its own `machinate.toml`, `config/`, `data/`, specs, and optional custom code.

The important point is that Machinator does not expect every pipeline repo to reinvent orchestration. The workspace and the CLI own that.

## What You Usually Do

The normal beginner flow is:

```bash
macht workspace init
macht grab data
macht legate report --data
macht collate pipeline --create
macht model validate
macht run train --experiment baseline
```

That sequence means:

1. create a workspace
2. stage a dataset
3. ask an agent to inspect the dataset
4. create the pipeline from the report
5. validate the model spec
6. run the pipeline

## Your First Session

### 1. Create a workspace

```bash
mkdir my-workspace
cd my-workspace
macht workspace init
```

This creates the shared control-plane layout:

- `.machinator/`
- `data/staged/`
- `outputs/`
- `pipelines/`
- `.envs/venvs/`

### 2. Stage data

```bash
macht grab data
```

Or explicitly:

```bash
macht grab data --src /path/to/data.csv --name customer-churn
macht grab data --src https://example.com/data.csv --name remote-data
```

Machinator stages the data into the workspace and records it in the asset registry.

### 3. Ask Machinator to inspect the data

```bash
macht legate report --data --dataset customer-churn
```

This uses Codex in non-interactive mode to inspect the dataset and write a structured report. By default it continues without stopping for notes. You can also add plain-English notes:

```bash
macht legate report --data --dataset customer-churn --notes "This came from a churn export. The label may be noisy."
macht legate report --data --dataset customer-churn --notes-prompt
```

The result is a report artifact under `outputs/reports/legate/`.

### 4. Create the pipeline from the report

This is the preferred path:

```bash
macht collate pipeline --create
```

Machinator will:

- read the delegated report
- infer dataset facts
- prompt for missing task intent when needed
- choose a recipe
- create the pipeline scaffold
- write:
  - `dataset_facts.toml`
  - `model.toml`
  - `training.toml`
- append collation metadata into `machinate.toml`

After that, move into the created repo if you want:

```bash
cd pipelines/my-pipeline
```

### 5. Validate the model

```bash
macht model validate
```

This checks that the generated `model.toml` is internally valid. If Rust validation is available, Machinator uses that backend automatically.

### 6. Train

```bash
macht run train --experiment baseline --dataset customer-churn
```

If `model.toml` and `training.toml` exist, the generated starter task uses them directly. It compiles the model and writes run artifacts under `outputs/`.

## What The Important Files Mean

### Workspace-level files

- `.machinator/workspace.json`
  Marks the workspace root and stores workspace metadata.
- `.machinator/pipelines/*.json`
  Registry entries for known pipelines.
- `.machinator/assets/*.json`
  Registry entries for staged datasets.

### Pipeline-level files

- `machinate.toml`
  The main pipeline configuration and task declarations.
- `config/*.toml`
  Experiment settings such as baseline training values.
- `dataset_facts.toml`
  Normalized facts derived from the delegated data report.
- `model.toml`
  The editable architecture spec.
- `training.toml`
  The editable training spec.
- `src/<pipeline_package>/tasks.py`
  Pipeline task entrypoints. Keep this small unless you truly need custom logic.

## Interactive Vs Scripted Use

Machinator supports both modes.

Interactive mode is best when you are exploring:

- `macht grab data`
- `macht collate pipeline`
- `macht run`

Scripted mode is best when you know the exact values:

```bash
macht collate pipeline \
  --pipeline-path /abs/path/to/pipeline \
  --report /abs/path/to/report.json \
  --intent-task binary_classification \
  --recipe tabular.binary.basic
```

Rule of thumb:

- use prompts when you are deciding
- use flags when you are repeating

## The Manual Escape Hatch

If you intentionally want a blank pipeline before you have a delegated report, you can still do that:

```bash
macht new pipeline
```

Treat that as the advanced/manual path, not the default beginner path.

## How To Think About Specs

The two most important editable files are:

- `model.toml`
- `training.toml`

Machinator is moving toward a spec-first workflow. That means:

- you do not always write raw model code first
- you edit the architecture and training intent through specs
- Machinator validates, diffs, compiles, and later migrates those specs

For example, after collation you can change the hidden sizes:

```bash
macht model edit --set 'hidden_dims=[256,64]' --output model.v2.toml
macht model diff --new model.v2.toml
```

That gives you a new spec plus a migration plan.

## The Commands You Will Use Most

### `macht workspace init`

Create a new workspace scaffold.

### `macht new pipeline`

Create a new native Machinator pipeline repo.

### `macht grab data`

Stage a local or remote dataset into the workspace.

### `macht legate report --data`

Ask Codex to inspect a dataset and produce a structured report.

### `macht collate pipeline`

Turn the delegated report into pipeline specs, and preferably create the pipeline from the report with `--create`.

### `macht model validate`

Check whether the model spec is valid.

### `macht model edit`

Apply controlled edits to the model spec.

### `macht model diff`

Compare two specs and generate a migration plan.

### `macht model migrate`

Preserve compatible checkpoint weights across spec changes.

### `macht task list`

Show the tasks declared by the current pipeline.

### `macht run <task>`

Run a native pipeline task such as `validate`, `audit`, `train`, or `smoke`.

### `macht doctor`

Check install, prompt backend, Codex availability, and workspace health.

### `macht check`

Run contributor verification from the Machinator repo.

### `macht test`

Run unit, integration, Python, or Rust test suites from the Machinator repo.

## Common Beginner Mistakes

### Running from the wrong directory

If Machinator cannot find the workspace or pipeline, either:

- `cd` into the workspace or pipeline first, or
- pass `--workspace` or `--pipeline-path`

### Treating the workspace like the pipeline repo

The workspace is not the same thing as the pipeline. The workspace holds many pipelines and shared assets.

### Expecting `legate` to train the model

`legate report --data` is for understanding the data, not for building the full training code by itself. The normal flow is:

- report
- collate
- validate
- run

### Forgetting that specs are now the source of truth

If `model.toml` exists, use the model commands. Do not immediately jump to stuffing everything into `tasks.py`.

## How To Recover When You Are Lost

If you are unsure what state you are in:

```bash
macht workspace show
macht doctor
macht task list
```

Then inspect:

- `machinate.toml`
- `dataset_facts.toml`
- `model.toml`
- `training.toml`
- `outputs/`

If a pipeline feels broken after edits:

```bash
macht model validate
macht model diff --new model.v2.toml
```

Work from the spec files first.

## A Good Beginner Habit

When starting a new project, keep your first session simple:

1. create workspace
2. create pipeline
3. stage one small dataset
4. generate one report
5. collate one recipe
6. validate
7. run one baseline task

Do not try to solve advanced architecture customization on the first pass. Get the control-plane flow working first.

## Where This Is Going

Machinator is moving toward:

- thinner pipeline repos
- richer model and training specs
- Rust-backed IR validation and migration planning
- more editable architecture workflows
- more delegation through `legate`

So the beginner mindset should be:

- start with a workspace
- let Machinator generate and validate structure
- edit the specs deliberately
- only drop into custom code when you truly need it

If you are comfortable with that model, the rest of the tool becomes much easier to reason about.
