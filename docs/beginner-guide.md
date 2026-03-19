# Machinate Beginner Guide

This guide is for the first time you sit down with `macht` and want to understand what you are supposed to do, in what order, and why.

You can read this guide from the CLI too:

```bash
macht guide beginner
macht guide workflow
```

Machinate is not a normal ML repo template generator. It is a control plane for ML work:

- you install one global CLI: `macht`
- you create a workspace
- inside that workspace, you create or manage pipeline repos
- pipelines stay relatively thin
- model and training structure increasingly live in specs such as `dataset_facts.toml`, `model.toml`, and `training.toml`

If that feels unusual, that is normal. The easiest way to use Machinate is to think in terms of a guided workflow rather than a library API.

## The Mental Model

Machinate has three layers:

1. Global CLI
   The `macht` command installed through Homebrew or from source.
2. Workspace
   A directory that holds shared state, staged data, outputs, and registered pipelines.
3. Pipeline repo
   A specific project under `pipelines/` with its own `machinate.toml`, experiment configs, specs, and optional custom code.

The important point is that Machinate does not expect every pipeline repo to reinvent orchestration. The workspace and the CLI own that.

## What You Usually Do

The normal beginner flow is:

```bash
macht workspace init
macht new pipeline
macht grab data
macht legate report --data
macht collate pipeline
macht model validate
macht run train --experiment baseline
```

That sequence means:

1. create a workspace
2. create a pipeline repo
3. stage a dataset
4. ask an agent to inspect the dataset
5. convert the report into specs
6. validate the model spec
7. run the pipeline

## Your First Session

### 1. Create a workspace

```bash
mkdir my-workspace
cd my-workspace
macht workspace init
```

This creates the shared control-plane layout:

- `.machinate/`
- `data/staged/`
- `outputs/`
- `pipelines/`
- `.envs/venvs/`

### 2. Create a pipeline

```bash
macht new pipeline
```

In interactive mode, Machinate will ask for:

- pipeline name
- pipeline type
- template
- starter tasks

This creates a pipeline repo under `pipelines/<name>/`.

### 3. Stage data

```bash
macht grab data
```

Or explicitly:

```bash
macht grab data --src /path/to/data.csv --name customer-churn
macht grab data --src https://example.com/data.csv --name remote-data
```

Machinate stages the data into the workspace and records it in the asset registry.

### 4. Ask Machinate to inspect the data

```bash
macht legate report --data --dataset customer-churn
```

This uses Codex in non-interactive mode to inspect the dataset and write a structured report. You can also add plain-English notes:

```bash
macht legate report --data --dataset customer-churn --notes "This came from a churn export. The label may be noisy."
```

The result is a report artifact under `outputs/reports/legate/`.

### 5. Collate the report into specs

Move into the pipeline repo, or point at it explicitly:

```bash
cd pipelines/my-pipeline
macht collate pipeline --report /absolute/path/to/report.json
```

Machinate will:

- read the delegated report
- infer dataset facts
- prompt for missing task intent when needed
- choose a recipe
- write:
  - `dataset_facts.toml`
  - `model.toml`
  - `training.toml`
- append collation metadata into `machinate.toml`

### 6. Validate the model

```bash
macht model validate
```

This checks that the generated `model.toml` is internally valid. If Rust validation is available, Machinate uses that backend automatically.

### 7. Train

```bash
macht run train --experiment baseline --dataset customer-churn
```

If `model.toml` and `training.toml` exist, the generated starter task uses them directly. It compiles the model and writes run artifacts under `outputs/`.

## What The Important Files Mean

### Workspace-level files

- `.machinate/workspace.json`
  Marks the workspace root and stores workspace metadata.
- `.machinate/pipelines/*.json`
  Registry entries for known pipelines.
- `.machinate/assets/*.json`
  Registry entries for staged datasets.

### Pipeline-level files

- `machinate.toml`
  The main pipeline configuration and task declarations.
- `configs/experiments/*.toml`
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

Machinate supports both modes.

Interactive mode is best when you are exploring:

- `macht new pipeline`
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

## How To Think About Specs

The two most important editable files are:

- `model.toml`
- `training.toml`

Machinate is moving toward a spec-first workflow. That means:

- you do not always write raw model code first
- you edit the architecture and training intent through specs
- Machinate validates, diffs, compiles, and later migrates those specs

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

Create a new native Machinate pipeline repo.

### `macht grab data`

Stage a local or remote dataset into the workspace.

### `macht legate report --data`

Ask Codex to inspect a dataset and produce a structured report.

### `macht collate pipeline`

Turn the delegated report into pipeline specs.

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

## Common Beginner Mistakes

### Running from the wrong directory

If Machinate cannot find the workspace or pipeline, either:

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

Machinate is moving toward:

- thinner pipeline repos
- richer model and training specs
- Rust-backed IR validation and migration planning
- more editable architecture workflows
- more delegation through `legate`

So the beginner mindset should be:

- start with a workspace
- let Machinate generate and validate structure
- edit the specs deliberately
- only drop into custom code when you truly need it

If you are comfortable with that model, the rest of the tool becomes much easier to reason about.
