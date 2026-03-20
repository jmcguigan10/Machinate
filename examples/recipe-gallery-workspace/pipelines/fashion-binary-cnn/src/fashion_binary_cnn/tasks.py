from __future__ import annotations

import csv
from collections import Counter
import importlib.util
from pathlib import Path

from machinator.modeling_compile import prepare_training_runtime
from machinator.modeling_types import ModelSpecError


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
    module_name = f"machinator_pipeline_compiled_{module_path.stem}"
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
