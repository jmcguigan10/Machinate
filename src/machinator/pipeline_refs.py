from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tomllib
from typing import Any

from machinator.core import now_utc
from machinator.modeling_specs import parse_architecture_spec_payload, parse_training_spec_payload
from machinator.modeling_types import ArchitectureSpec, TrainingSpec


CONFIG_REF_FILENAME = "config-ref.toml"


def render_config_ref_toml(
    *,
    pipeline_name: str,
    pipeline_slug: str,
    dataset_name: str,
    modality: str,
    intent_task: str,
    recipe_name: str,
    target_column: str,
    dataset_ref_path: str,
    report_ref_path: str,
) -> str:
    return (
        "[pipeline]\n"
        f'name = "{pipeline_name}"\n'
        f'slug = "{pipeline_slug}"\n'
        f'modality = "{modality}"\n'
        f'intent_task = "{intent_task}"\n'
        f'recipe = "{recipe_name}"\n'
        f'dataset_name = "{dataset_name}"\n'
        f'target_column = "{target_column}"\n'
        "\n"
        "[refs]\n"
        f'dataset = "{dataset_ref_path}"\n'
        f'report = "{report_ref_path}"\n'
        "\n"
        "[generated]\n"
        'dataset = "config/dataset.yaml"\n'
        'model = "config/model.yaml"\n'
        'training = "config/training.yaml"\n'
        "\n"
        "[build]\n"
        f'initialized_at = "{now_utc()}"\n'
    )


def load_config_ref(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"config ref is invalid: {path}")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON is a strict subset of YAML, so these generated `.yaml` files stay
    # valid YAML while keeping the runtime dependency surface minimal.
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_yaml(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a mapping in YAML file: {path}")
    return loaded


def dataset_yaml_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return payload


def load_architecture_spec_yaml(path: Path) -> ArchitectureSpec:
    return parse_architecture_spec_payload(load_yaml(path), fallback_name=path.stem)


def load_training_spec_yaml(path: Path) -> TrainingSpec:
    return parse_training_spec_payload(load_yaml(path))


def generated_config_paths(pipeline_root: Path, config_ref: dict[str, Any]) -> dict[str, Path]:
    generated = config_ref.get("generated", {})
    if not isinstance(generated, dict):
        generated = {}
    return {
        "dataset": pipeline_root / str(generated.get("dataset", "config/dataset.yaml")),
        "model": pipeline_root / str(generated.get("model", "config/model.yaml")),
        "training": pipeline_root / str(generated.get("training", "config/training.yaml")),
    }


def referenced_dataset_path(pipeline_root: Path, config_ref: dict[str, Any]) -> Path:
    refs = config_ref.get("refs", {})
    if not isinstance(refs, dict):
        raise ValueError("config ref is missing a valid [refs] section")
    return pipeline_root / str(refs.get("dataset", "data"))


def referenced_report_path(pipeline_root: Path, config_ref: dict[str, Any]) -> Path:
    refs = config_ref.get("refs", {})
    if not isinstance(refs, dict):
        raise ValueError("config ref is missing a valid [refs] section")
    return pipeline_root / str(refs.get("report", "data/reports/report.json"))


def update_build_metadata(config_ref: dict[str, Any], *, model_family: str) -> dict[str, Any]:
    build_section = config_ref.get("build", {})
    if not isinstance(build_section, dict):
        build_section = {}
    updated = dict(config_ref)
    updated["build"] = {
        **build_section,
        "generated_at": now_utc(),
        "model_family": model_family,
    }
    return updated


def replace_reference(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() or destination_path.is_symlink():
        if destination_path.is_dir() and not destination_path.is_symlink():
            shutil.rmtree(destination_path)
        else:
            destination_path.unlink()

    try:
        if source_path.is_dir():
            destination_path.symlink_to(source_path, target_is_directory=True)
        else:
            destination_path.symlink_to(source_path)
    except OSError:
        if source_path.is_dir():
            shutil.copytree(source_path, destination_path)
        else:
            shutil.copy2(source_path, destination_path)


def model_yaml_payload(spec: ArchitectureSpec) -> dict[str, Any]:
    # The generated YAML mirrors the logical sections of model.toml so it stays
    # readable while still being easy to round-trip back into typed specs.
    return {
        "model": {
            "name": spec.name,
            "family": spec.family,
            "task": spec.task,
            "modality": spec.modality,
        },
        "input": {
            "kind": spec.input_kind,
            "feature_names": list(spec.feature_names),
            "feature_count": spec.feature_count,
            "token_vocab_size": spec.token_vocab_size,
            "max_sequence_length": spec.max_sequence_length,
            "image_channels": spec.image_channels,
            "image_height": spec.image_height,
            "image_width": spec.image_width,
        },
        "target": {
            "column": spec.target_column,
            "kind": spec.target_kind,
        },
        "backbone": {
            "hidden_dims": list(spec.hidden_dims),
            "channels": list(spec.conv_channels),
            "model_dim": spec.model_dim,
            "num_heads": spec.num_heads,
            "num_layers": spec.num_layers,
            "ffn_dim": spec.ffn_dim,
            "activation": spec.activation,
            "normalization": spec.normalization,
            "dropout": spec.dropout,
        },
        "head": {
            "pooling": spec.pooling,
            "output_dim": spec.head_output_dim,
        },
        "procedures": {
            "forward": {
                "input": spec.forward_input,
                "output": spec.forward_output,
            },
            "loss": {
                "kind": spec.loss_kind,
            },
        },
        "param_store": {
            "format": spec.param_store_format,
            "root_key": spec.param_store_root_key,
        },
    }


def training_yaml_payload(spec: TrainingSpec) -> dict[str, Any]:
    return {"training": asdict(spec)}
