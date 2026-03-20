# Machinator Beginner Guide

This guide is for the first time you sit down with `macht` and want to understand what you are supposed to do, in what order, and why.

Machinator is not a normal ML repo template generator. It is a control plane for ML work:

- you install one global CLI: `macht`
- you create a workspace
- inside that workspace, you create or manage pipeline repos
- pipelines stay relatively thin
- the default pipeline flow builds generated configs under `config/` from delegated data reports

If that feels unusual, that is normal. The easiest way to use Machinator is to think in terms of a guided workflow rather than a library API.

## The Mental Model

Machinator has three layers:

1. Global CLI
   The `macht` command installed through Homebrew or from source.
2. Workspace
   A directory that holds shared state, staged data, outputs, and registered pipelines.
3. Pipeline repo
   A specific project under `pipelines/` with its own `machinate.toml`, `config-ref.toml`, `config/`, `data/`, and `outputs/`.

The important point is that Machinator does not expect every pipeline repo to reinvent orchestration. The workspace and the CLI own that.

## What You Usually Do

The normal beginner flow is:

```bash
macht workspace init
macht grab data
macht legate report --data
macht init pipeline
macht build configs
macht run train
```

That sequence means:

1. create a workspace
2. stage a dataset
3. ask an agent to inspect the dataset
4. create the pipeline references from the report
5. build generated configs
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

### 4. Initialize the pipeline from the report

This is the preferred path:

```bash
macht init pipeline --name customer-churn-pipeline
```

Machinator will:

- pick the delegated report
- create a thin pipeline directory
- write `machinate.toml`
- write `config-ref.toml`
- place the chosen dataset under `data/`
- place the chosen delegated report under `data/reports/`

After that, move into the created repo if you want:

```bash
cd pipelines/my-pipeline
```

### 5. Build generated configs

```bash
macht build configs
```

This turns the chosen JSON report into generated config files under `config/`, including the dataset, model, and training config set.

### 6. Train

```bash
macht run train
```

The pipeline runtime is package-managed by Machinator. The pipeline directory is just holding references, generated configs, and artifacts.

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
- `config-ref.toml`
  The report-driven reference file that points at the selected dataset, report, and generated configs.
- `config/dataset.yaml`
  Generated dataset facts for the active pipeline.
- `config/model.yaml`
  Generated model configuration derived from the delegated report and chosen recipe.
- `config/training.yaml`
  Generated training configuration used by the built-in runtime tasks.
- `data/reports/*.json`
  The delegated report artifact chosen for this pipeline.

## Interactive Vs Scripted Use

Machinator supports both modes.

Interactive mode is best when you are exploring:

- `macht grab data`
- `macht init pipeline`
- `macht build configs`
- `macht run`

Scripted mode is best when you want to pin the exact values:

```bash
macht init pipeline --report /abs/path/to/report.json --name customer-churn-pipeline
macht build configs --pipeline customer-churn-pipeline
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

The two most important default config files are:

- `config/model.yaml`
- `config/training.yaml`

Machinator is moving toward a spec-first workflow. That means:

- you do not always write raw model code first
- you derive an initial model and training setup from the data report
- Machinator validates, diffs, compiles, and later migrates those specs

For example, after config generation you can inspect or revise the model setup:

```bash
cat config/model.yaml
cat config/training.yaml
```

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
