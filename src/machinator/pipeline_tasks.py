from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from machinator.modeling_compile import compile_architecture_spec, load_compiled_model_class
from machinator.modeling_specs import validate_training_spec
from machinator.pipeline_refs import (
    CONFIG_REF_FILENAME,
    generated_config_paths,
    load_architecture_spec_yaml,
    load_config_ref,
    load_training_spec_yaml,
    referenced_dataset_path,
)


def _resolve_csv(path: Path) -> Path:
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


def _load_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return reader.fieldnames or [], rows


def _config_ref_path(pipeline_root: Path, pipeline_config: dict[str, object]) -> Path:
    refs = pipeline_config.get("refs", {})
    if not isinstance(refs, dict):
        refs = {}
    return pipeline_root / str(refs.get("config_ref", CONFIG_REF_FILENAME))


def _load_pipeline_refs(context) -> tuple[dict[str, object], dict[str, Path]]:
    config_ref = load_config_ref(_config_ref_path(context.pipeline_root, context.pipeline_config))
    return config_ref, generated_config_paths(context.pipeline_root, config_ref)


def _load_runtime_specs(context):
    config_ref, generated_paths = _load_pipeline_refs(context)
    model_path = generated_paths["model"]
    training_path = generated_paths["training"]
    if not model_path.exists() or not training_path.exists():
        raise ValueError("generated configs are missing; run `macht build configs` first")

    model_spec = load_architecture_spec_yaml(model_path)
    training_spec = load_training_spec_yaml(training_path)
    validate_training_spec(training_spec)
    compile_artifacts = compile_architecture_spec(model_spec, context.output_root / "compiled_model")
    return config_ref, generated_paths, model_spec, training_spec, compile_artifacts


def _maybe_write_initialized_checkpoint(context, model_spec, compile_artifacts):
    try:
        import torch
    except ModuleNotFoundError:
        return None

    model_class = load_compiled_model_class(Path(compile_artifacts["module_path"]), str(compile_artifacts["class_name"]))
    model = model_class()
    checkpoint_dir = context.output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{context.experiment_name or 'default'}_initialized.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_name": compile_artifacts["class_name"],
            "family": model_spec.family,
            "task": model_spec.task,
        },
        checkpoint_path,
    )
    return checkpoint_path


def validate(context):
    config_ref, generated_paths = _load_pipeline_refs(context)
    dataset_path = referenced_dataset_path(context.pipeline_root, config_ref)
    if not dataset_path.exists():
        raise ValueError(f"referenced dataset path is missing: {dataset_path}")

    if generated_paths["model"].exists() and generated_paths["training"].exists():
        _config_ref, _generated_paths, model_spec, _training_spec, compile_artifacts = _load_runtime_specs(context)
        print(
            f"validated generated config set `{model_spec.family}` "
            f"with {compile_artifacts['estimated_parameter_count']} estimated parameters"
        )
        return 0

    print(f"validated pipeline references; generated configs are not built yet ({CONFIG_REF_FILENAME})")
    return 0


def audit(context):
    config_ref, _generated_paths = _load_pipeline_refs(context)
    dataset_path = referenced_dataset_path(context.pipeline_root, config_ref)
    csv_path = _resolve_csv(dataset_path)
    columns, rows = _load_rows(csv_path)

    pipeline_section = config_ref.get("pipeline", {})
    if not isinstance(pipeline_section, dict):
        pipeline_section = {}
    target_column = str(pipeline_section.get("target_column", "target"))
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
    config_ref, generated_paths, model_spec, training_spec, compile_artifacts = _load_runtime_specs(context)
    checkpoint_path = _maybe_write_initialized_checkpoint(context, model_spec, compile_artifacts)

    artifact_path = context.write_json_artifact(
        "runs",
        f"{context.experiment_name or 'default'}_train",
        {
            "task": "train",
            "pipeline_root": str(context.pipeline_root),
            "experiment_name": context.experiment_name or "",
            "dataset_path": str(referenced_dataset_path(context.pipeline_root, config_ref)),
            "config_ref_path": str(_config_ref_path(context.pipeline_root, context.pipeline_config)),
            "dataset_config_path": str(generated_paths["dataset"]),
            "model_config_path": str(generated_paths["model"]),
            "training_config_path": str(generated_paths["training"]),
            "epochs": training_spec.epochs,
            "learning_rate": training_spec.learning_rate,
            "batch_size": training_spec.batch_size,
            "optimizer": training_spec.optimizer,
            "weight_decay": training_spec.weight_decay,
            "primary_metric": training_spec.primary_metric,
            "model_family": model_spec.family,
            "model_task": model_spec.task,
            "estimated_parameter_count": compile_artifacts["estimated_parameter_count"],
            "compiled_module_path": str(compile_artifacts["module_path"]),
            "param_store_manifest_path": str(compile_artifacts["param_store_manifest_path"]),
            "initialized_checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        },
    )
    print(f"run summary written: {artifact_path}")
    return 0


def smoke(context):
    validate(context)
    audit(context)
    train(context)
    print("smoke task completed")
    return 0
