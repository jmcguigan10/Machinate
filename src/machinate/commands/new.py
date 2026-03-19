from __future__ import annotations

import argparse
from pathlib import Path

from machinate.core import (
    clean_optional,
    now_utc,
    require_workspace_root,
    slugify,
    workspace_paths,
    write_json,
)
from machinate.ui import MenuChoice, can_prompt_interactively, prompt_multiselect, prompt_select, prompt_text


PIPELINE_TYPES = [
    MenuChoice("tabular", "tabular"),
    MenuChoice("vision", "vision"),
    MenuChoice("nlp", "nlp"),
    MenuChoice("custom", "custom"),
]

PIPELINE_TEMPLATES = [
    MenuChoice("native-python", "native-python"),
    MenuChoice("minimal", "minimal"),
]

STARTER_TASKS = ["validate", "audit", "train", "smoke"]
TASK_DESCRIPTIONS = {
    "validate": "Validate the experiment configuration",
    "audit": "Audit the dataset and write a JSON summary",
    "train": "Run a demo training task",
    "smoke": "Run validate, audit, and train in sequence",
}


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    new_parser = subparsers.add_parser("new", help="Create new Machinate-managed objects")
    new_subparsers = new_parser.add_subparsers(dest="new_command", required=True)

    new_pipeline = new_subparsers.add_parser("pipeline", help="Create a placeholder pipeline repo")
    new_pipeline.add_argument("--workspace")
    new_pipeline.add_argument("--name")
    new_pipeline.add_argument("--type")
    new_pipeline.add_argument("--template")
    new_pipeline.add_argument("--path")
    new_pipeline.add_argument("--task", action="append", default=[])
    new_pipeline.set_defaults(func=cmd_new_pipeline)


def pipeline_config_toml(
    *,
    pipeline_name: str,
    pipeline_slug: str,
    pipeline_type: str,
    template: str,
    package_slug: str,
    tasks: list[str],
) -> str:
    lines = [
        "[pipeline]",
        f'name = "{pipeline_name}"',
        f'slug = "{pipeline_slug}"',
        f'type = "{pipeline_type}"',
        f'template = "{template}"',
        f'package = "{package_slug}"',
        "",
        "[paths]",
        'source_root = "src"',
        'experiments = "configs/experiments"',
        'outputs = "outputs"',
        "",
        "[specs]",
        'dataset_facts = "dataset_facts.toml"',
        'model = "model.toml"',
        'training = "training.toml"',
        "",
        "[dataset]",
        'kind = "csv"',
        'target_column = "target"',
        "",
    ]
    for task_name in tasks:
        lines.extend(
            [
                f"[tasks.{task_name}]",
                f'entry = "{package_slug}.tasks:{task_name}"',
                f'description = "{TASK_DESCRIPTIONS[task_name]}"',
                f"requires_dataset = {'true' if task_name in {'audit', 'smoke'} else 'false'}",
                f"requires_experiment = {'true' if task_name in {'validate', 'audit', 'train', 'smoke'} else 'false'}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def pipeline_readme(name: str, pipeline_type: str, template: str) -> str:
    return f"""# {name}

This placeholder pipeline was created by `macht new pipeline`.

## Metadata

- pipeline_type: {pipeline_type}
- template: {template}
- pipeline_config: `machinate.toml`

## Native Usage

```bash
macht task list
macht run validate --experiment baseline
macht run train --experiment baseline
```

## Next Steps

1. Stage data and run `macht legate report --data ...` to inspect the dataset.
2. Run `macht collate pipeline --report <report.json>` to materialize `dataset_facts.toml`, `model.toml`, and `training.toml`.
3. Replace the starter task functions in `src/<package>/tasks.py` with real project logic or custom blocks.
4. Add runtime dependencies to `requirements.txt` or your preferred env manager.
"""


def baseline_config(name: str, pipeline_type: str) -> str:
    return """[pipeline]
name = "%s"
type = "%s"

[dataset]
kind = "csv"
target_column = "target"

[training]
epochs = 1
learning_rate = 0.01
""" % (name, pipeline_type)


def starter_tasks_module() -> str:
    return '''from __future__ import annotations

import csv
from collections import Counter
import importlib.util
from pathlib import Path

from machinate.modeling import ModelSpecError, prepare_training_runtime


def _resolve_csv(path):
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, got {path}")
        return path

    candidates = sorted(candidate for candidate in path.rglob("*.csv") if candidate.is_file())
    if not candidates:
        raise ValueError(f"No CSV files were found under {path}")
    if len(candidates) > 1:
        raise ValueError(f"Expected exactly one CSV file under {path}, found {len(candidates)}")
    return candidates[0]


def _load_rows(csv_path):
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return reader.fieldnames or [], rows


def _load_compiled_model_class(module_path, class_name):
    module_name = f"machinate_pipeline_compiled_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load compiled model module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _maybe_prepare_runtime(context):
    try:
        return prepare_training_runtime(context.pipeline_root, context.pipeline_config)
    except ModelSpecError:
        return None


def _maybe_write_initialized_checkpoint(context, runtime):
    if runtime is None:
        return None
    try:
        import torch
    except ModuleNotFoundError:
        return None

    compile_artifacts = runtime["compile_artifacts"]
    model_class = _load_compiled_model_class(
        Path(compile_artifacts["module_path"]),
        str(compile_artifacts["class_name"]),
    )
    model = model_class()
    checkpoint_dir = context.output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{context.experiment_name or 'default'}_initialized.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_name": compile_artifacts["class_name"],
            "family": runtime["model_spec"].family,
            "task": runtime["model_spec"].task,
        },
        checkpoint_path,
    )
    return checkpoint_path


def validate(context):
    runtime = _maybe_prepare_runtime(context)
    config = context.experiment_config
    if config:
        missing = [section for section in ("dataset", "training") if section not in config]
        if missing:
            raise ValueError(f"Missing required config sections: {', '.join(missing)}")
        training = config.get("training", {})
        epochs = int(training.get("epochs", 0))
        if epochs <= 0:
            raise ValueError("training.epochs must be positive")

    if runtime is not None:
        print(
            f"validated model spec `{runtime['model_spec'].family}` "
            f"with {runtime['compile_artifacts']['estimated_parameter_count']} estimated parameters"
        )
    elif config:
        print(f"validated experiment `{context.experiment_name or 'default'}`")
    else:
        raise ValueError("No experiment config or model/training spec is available to validate")
    return 0


def audit(context):
    dataset_path = context.require_dataset_path()
    csv_path = _resolve_csv(dataset_path)
    columns, rows = _load_rows(csv_path)
    config = context.require_experiment_config()
    target_column = str(config.get("dataset", {}).get("target_column", "target"))
    target_support = {}
    if target_column in columns:
        target_support = dict(sorted(Counter(str(row.get(target_column, "")) for row in rows).items()))

    artifact_path = context.write_json_artifact(
        "audits",
        f"{context.experiment_name or 'default'}_audit",
        {
            "task": "audit",
            "pipeline_root": str(context.pipeline_root),
            "dataset_path": str(dataset_path),
            "csv_path": str(csv_path),
            "row_count": len(rows),
            "column_count": len(columns),
            "columns": columns,
            "target_column": target_column,
            "target_support": target_support,
        },
    )
    print(f"audit written: {artifact_path}")
    return 0


def train(context):
    runtime = _maybe_prepare_runtime(context)
    config = context.experiment_config
    training = config.get("training", {}) if config else {}
    checkpoint_path = _maybe_write_initialized_checkpoint(context, runtime)

    if runtime is not None:
        training_spec = runtime["training_spec"]
        model_spec = runtime["model_spec"]
        payload = {
            "task": "train",
            "pipeline_root": str(context.pipeline_root),
            "experiment_name": context.experiment_name or "",
            "experiment_config_path": str(context.experiment_config_path) if context.experiment_config_path else "",
            "dataset_path": str(context.dataset_path) if context.dataset_path else "",
            "epochs": training_spec.epochs,
            "learning_rate": training_spec.learning_rate,
            "batch_size": training_spec.batch_size,
            "optimizer": training_spec.optimizer,
            "weight_decay": training_spec.weight_decay,
            "primary_metric": training_spec.primary_metric,
            "model_family": model_spec.family,
            "model_task": model_spec.task,
            "estimated_parameter_count": runtime["compile_artifacts"]["estimated_parameter_count"],
            "compiled_module_path": str(runtime["compile_artifacts"]["module_path"]),
            "param_store_manifest_path": str(runtime["compile_artifacts"]["param_store_manifest_path"]),
            "model_spec_path": str(runtime["spec_paths"]["model"]),
            "training_spec_path": str(runtime["spec_paths"]["training"]),
            "initialized_checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        }
    else:
        if not config:
            raise ValueError("No experiment config or model/training spec is available to train")
        payload = {
            "task": "train",
            "pipeline_root": str(context.pipeline_root),
            "experiment_name": context.experiment_name or "",
            "experiment_config_path": str(context.experiment_config_path) if context.experiment_config_path else "",
            "dataset_path": str(context.dataset_path) if context.dataset_path else "",
            "epochs": int(training.get("epochs", 0)),
            "learning_rate": float(training.get("learning_rate", 0.0)),
        }

    artifact_path = context.write_json_artifact(
        "runs",
        f"{context.experiment_name or 'default'}_train",
        payload,
    )
    print(f"run summary written: {artifact_path}")
    return 0


def smoke(context):
    validate(context)
    if context.dataset_path is not None:
        audit(context)
    train(context)
    print("smoke task completed")
    return 0
'''


def selected_tasks(raw_tasks: list[str]) -> list[str]:
    if raw_tasks:
        deduped: list[str] = []
        for task_name in raw_tasks:
            if task_name not in STARTER_TASKS:
                raise SystemExit(f"unsupported starter task `{task_name}`")
            if task_name not in deduped:
                deduped.append(task_name)
        return deduped

    if can_prompt_interactively():
        chosen = prompt_multiselect("Select starter pipeline tasks", STARTER_TASKS)
        deduped = [task_name for task_name in STARTER_TASKS if task_name in chosen]
        if deduped:
            return deduped
    return list(STARTER_TASKS)


def cmd_new_pipeline(args: argparse.Namespace) -> int:
    workspace_root = require_workspace_root(args.workspace)
    paths = workspace_paths(workspace_root)

    raw_name = clean_optional(args.name)
    if raw_name is None and can_prompt_interactively():
        raw_name = prompt_text("Pipeline name")
    if raw_name is None:
        raise SystemExit("pipeline name is required; pass --name or run interactively")

    pipeline_name = raw_name
    pipeline_slug = slugify(pipeline_name, fallback="pipeline")

    pipeline_type = clean_optional(args.type)
    if pipeline_type is None:
        if can_prompt_interactively():
            pipeline_type = prompt_select("Pipeline type", PIPELINE_TYPES, default="tabular")
        else:
            pipeline_type = "tabular"

    template = clean_optional(args.template)
    if template is None:
        if can_prompt_interactively():
            template = prompt_select("Pipeline template", PIPELINE_TEMPLATES, default="minimal")
        else:
            template = "native-python"

    default_repo_path = paths.pipeline_root / pipeline_slug
    repo_path_text = clean_optional(args.path)
    if repo_path_text is None and can_prompt_interactively():
        repo_path_text = prompt_text("Pipeline repo path", default=str(default_repo_path))
    repo_path = Path(repo_path_text or default_repo_path).expanduser().resolve()

    if repo_path.exists() and any(repo_path.iterdir()):
        raise SystemExit(f"pipeline path already exists and is not empty: {repo_path}")

    tasks = selected_tasks(args.task)
    package_slug = pipeline_slug.replace("-", "_").replace(".", "_")
    config_path = repo_path / "machinate.toml"
    manifest_path = paths.pipeline_registry_root / f"{pipeline_slug}.json"
    if manifest_path.exists():
        raise SystemExit(f"pipeline registry entry already exists: {manifest_path}")

    (repo_path / "configs" / "experiments").mkdir(parents=True, exist_ok=True)
    (repo_path / "src" / package_slug).mkdir(parents=True, exist_ok=True)
    (repo_path / "outputs").mkdir(parents=True, exist_ok=True)

    config_path.write_text(
        pipeline_config_toml(
            pipeline_name=pipeline_name,
            pipeline_slug=pipeline_slug,
            pipeline_type=pipeline_type,
            template=template,
            package_slug=package_slug,
            tasks=tasks,
        )
    )
    (repo_path / "README.md").write_text(pipeline_readme(pipeline_name, pipeline_type, template))
    (repo_path / ".gitignore").write_text("__pycache__/\n*.py[cod]\n.venv/\noutputs/\n")
    (repo_path / "requirements.txt").write_text("# Add runtime dependencies here.\n")
    (repo_path / "src" / package_slug / "__init__.py").write_text(f'"""Pipeline package for {pipeline_name}."""\n')
    (repo_path / "src" / package_slug / "tasks.py").write_text(starter_tasks_module())
    (repo_path / "configs" / "experiments" / "baseline.toml").write_text(
        baseline_config(pipeline_name, pipeline_type)
    )

    write_json(
        manifest_path,
        {
            "pipeline_name": pipeline_name,
            "pipeline_slug": pipeline_slug,
            "repo_path": str(repo_path),
            "pipeline_type": pipeline_type,
            "template": template,
            "created_at": now_utc(),
            "pipeline_config_path": str(config_path),
            "supported_tasks": tasks,
        },
    )

    print(f"pipeline created: {pipeline_name}")
    print(f"repo path: {repo_path}")
    print(f"registry manifest: {manifest_path}")
    return 0
