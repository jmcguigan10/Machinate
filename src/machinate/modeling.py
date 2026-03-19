from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import importlib.util
import json
import re
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from machinate.core import clean_optional, load_json, now_utc, slugify, write_json


VALID_MODEL_FAMILIES = {"tabular_mlp", "transformer_encoder"}
VALID_MODALITIES = {"tabular", "text"}
VALID_INPUT_KINDS = {"dense_features", "token_ids"}
VALID_TASKS = {"binary_classification"}
VALID_TARGET_KINDS = {"binary"}
VALID_ACTIVATIONS = {"relu", "gelu", "silu", "tanh"}
VALID_NORMALIZATIONS = {"none", "batchnorm", "layernorm"}
VALID_LOSSES = {"bce_with_logits"}
VALID_POOLING = {"mean", "cls"}


class ModelSpecError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetFacts:
    dataset_name: str
    dataset_path: Path
    modality: str
    suspected_problem_type: str
    row_count_estimate: int | None
    column_names: list[str]
    feature_names: list[str]
    target_column: str
    target_candidates: list[str]
    id_candidates: list[str]
    time_candidates: list[str]
    source_report_path: Path


@dataclass(frozen=True)
class TrainingSpec:
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    weight_decay: float
    primary_metric: str


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    family: str
    task: str
    modality: str
    input_kind: str
    feature_names: list[str]
    feature_count: int
    token_vocab_size: int | None
    max_sequence_length: int | None
    target_column: str
    target_kind: str
    hidden_dims: list[int]
    model_dim: int | None
    num_heads: int | None
    num_layers: int | None
    ffn_dim: int | None
    activation: str
    normalization: str
    dropout: float
    pooling: str | None
    head_output_dim: int
    forward_input: str
    forward_output: str
    loss_kind: str
    param_store_format: str
    param_store_root_key: str

    @property
    def class_name(self) -> str:
        tokens = re.split(r"[^a-zA-Z0-9]+", self.name)
        cleaned = "".join(token.capitalize() for token in tokens if token)
        return cleaned or "CompiledModel"


def rust_ir_manifest_path() -> Path:
    return Path(__file__).resolve().parents[2] / "rust" / "machinate-ir" / "Cargo.toml"


def rust_ir_available() -> bool:
    return shutil.which("cargo") is not None and rust_ir_manifest_path().exists()


def run_rust_ir_cli(*args: str) -> dict[str, Any] | None:
    if not rust_ir_available():
        return None
    command = [
        shutil.which("cargo") or "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        str(rust_ir_manifest_path()),
        "--",
        *args,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ModelSpecError(result.stderr.strip() or f"Rust IR CLI exited with code {result.returncode}")
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ModelSpecError("Rust IR CLI returned a non-object payload")
    if payload.get("ok") is False:
        raise ModelSpecError(str(payload.get("error", "Rust IR CLI reported an unknown error")))
    return payload


def rust_validate_spec_file(path: Path) -> dict[str, Any] | None:
    return run_rust_ir_cli("validate", str(path))


def rust_diff_spec_files(old_path: Path, new_path: Path) -> dict[str, Any] | None:
    return run_rust_ir_cli("diff", str(old_path), str(new_path))


def rust_migration_plan_spec_files(old_path: Path, new_path: Path) -> dict[str, Any] | None:
    return run_rust_ir_cli("migration-plan", str(old_path), str(new_path))


def _json_string(value: str) -> str:
    return json.dumps(value)


def _json_string_array(values: list[str]) -> str:
    return json.dumps(values)


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _require_section(payload: dict[str, Any], key: str) -> dict[str, Any]:
    section = payload.get(key)
    if not isinstance(section, dict):
        raise ModelSpecError(f"model spec is missing a valid [{key}] section")
    return section


def _clean_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    for value in values:
        item = clean_optional(str(value))
        if item is not None:
            cleaned.append(item)
    return cleaned


def _first_nonempty(values: list[str]) -> str | None:
    for value in values:
        if value:
            return value
    return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def load_training_spec(path: Path) -> TrainingSpec:
    payload = _read_toml(path)
    training = payload.get("training")
    if not isinstance(training, dict):
        raise ModelSpecError(f"training spec is missing a valid [training] section: {path}")
    return TrainingSpec(
        epochs=int(training.get("epochs", 0)),
        batch_size=int(training.get("batch_size", 0)),
        learning_rate=float(training.get("learning_rate", 0.0)),
        optimizer=str(training.get("optimizer", "")).strip(),
        weight_decay=float(training.get("weight_decay", 0.0)),
        primary_metric=str(training.get("primary_metric", "")).strip(),
    )


def validate_training_spec(spec: TrainingSpec) -> None:
    if spec.epochs <= 0:
        raise ModelSpecError("training.epochs must be positive")
    if spec.batch_size <= 0:
        raise ModelSpecError("training.batch_size must be positive")
    if spec.learning_rate <= 0.0:
        raise ModelSpecError("training.learning_rate must be positive")
    if not spec.optimizer:
        raise ModelSpecError("training.optimizer is required")
    if not spec.primary_metric:
        raise ModelSpecError("training.primary_metric is required")


def load_architecture_spec(path: Path) -> ArchitectureSpec:
    payload = _read_toml(path)
    model = _require_section(payload, "model")
    input_section = _require_section(payload, "input")
    target = _require_section(payload, "target")
    backbone = _require_section(payload, "backbone")
    head = _require_section(payload, "head")
    procedures = _require_section(payload, "procedures")
    forward = _require_section(procedures, "forward")
    loss = _require_section(procedures, "loss")
    param_store = _require_section(payload, "param_store")

    feature_names = _clean_list(input_section.get("feature_names", []))
    input_kind = str(input_section.get("kind", "dense_features")).strip() or "dense_features"
    feature_count = int(input_section.get("feature_count", len(feature_names) or 0))
    hidden_dims = [int(value) for value in backbone.get("hidden_dims", [])]

    spec = ArchitectureSpec(
        name=str(model.get("name", path.stem)).strip() or path.stem,
        family=str(model.get("family", "")).strip(),
        task=str(model.get("task", "")).strip(),
        modality=str(model.get("modality", "")).strip(),
        input_kind=input_kind,
        feature_names=feature_names,
        feature_count=feature_count,
        token_vocab_size=_int_or_none(input_section.get("token_vocab_size")),
        max_sequence_length=_int_or_none(input_section.get("max_sequence_length")),
        target_column=str(target.get("column", "")).strip(),
        target_kind=str(target.get("kind", "")).strip(),
        hidden_dims=hidden_dims,
        model_dim=_int_or_none(backbone.get("model_dim")),
        num_heads=_int_or_none(backbone.get("num_heads")),
        num_layers=_int_or_none(backbone.get("num_layers")),
        ffn_dim=_int_or_none(backbone.get("ffn_dim")),
        activation=str(backbone.get("activation", "")).strip(),
        normalization=str(backbone.get("normalization", "")).strip(),
        dropout=float(backbone.get("dropout", 0.0)),
        pooling=clean_optional(str(head.get("pooling", ""))),
        head_output_dim=int(head.get("output_dim", 0)),
        forward_input=str(forward.get("input", "")).strip(),
        forward_output=str(forward.get("output", "")).strip(),
        loss_kind=str(loss.get("kind", "")).strip(),
        param_store_format=str(param_store.get("format", "")).strip(),
        param_store_root_key=str(param_store.get("root_key", "")).strip(),
    )
    validate_architecture_spec(spec)
    return spec


def validate_architecture_spec(spec: ArchitectureSpec) -> None:
    if spec.family not in VALID_MODEL_FAMILIES:
        raise ModelSpecError(f"unsupported model family `{spec.family}`")
    if spec.modality not in VALID_MODALITIES:
        raise ModelSpecError(f"unsupported modality `{spec.modality}`")
    if spec.input_kind not in VALID_INPUT_KINDS:
        raise ModelSpecError(f"unsupported input kind `{spec.input_kind}`")
    if spec.task not in VALID_TASKS:
        raise ModelSpecError(f"unsupported task `{spec.task}`")
    if spec.target_kind not in VALID_TARGET_KINDS:
        raise ModelSpecError(f"unsupported target kind `{spec.target_kind}`")
    if not spec.target_column:
        raise ModelSpecError("target.column is required")
    if spec.activation not in VALID_ACTIVATIONS:
        raise ModelSpecError(f"unsupported activation `{spec.activation}`")
    if spec.normalization not in VALID_NORMALIZATIONS:
        raise ModelSpecError(f"unsupported normalization `{spec.normalization}`")
    if spec.dropout < 0.0 or spec.dropout >= 1.0:
        raise ModelSpecError("backbone.dropout must be in the range [0.0, 1.0)")
    if spec.head_output_dim != 1:
        raise ModelSpecError("head.output_dim must be 1 for binary classification")
    if spec.forward_input != "features":
        raise ModelSpecError("procedures.forward.input must be `features`")
    if spec.forward_output != "logits":
        raise ModelSpecError("procedures.forward.output must be `logits`")
    if spec.loss_kind not in VALID_LOSSES:
        raise ModelSpecError(f"unsupported loss `{spec.loss_kind}`")
    if spec.param_store_format != "safetensors":
        raise ModelSpecError("param_store.format must be `safetensors`")
    if not spec.param_store_root_key:
        raise ModelSpecError("param_store.root_key is required")

    if spec.family == "tabular_mlp":
        if spec.modality != "tabular":
            raise ModelSpecError("tabular_mlp requires modality `tabular`")
        if spec.input_kind != "dense_features":
            raise ModelSpecError("tabular_mlp requires input.kind `dense_features`")
        if spec.feature_count <= 0:
            raise ModelSpecError("input.feature_count must be positive for tabular_mlp")
        if spec.feature_names and len(spec.feature_names) != spec.feature_count:
            raise ModelSpecError("input.feature_names must match input.feature_count")
        if not spec.hidden_dims or any(value <= 0 for value in spec.hidden_dims):
            raise ModelSpecError("backbone.hidden_dims must contain one or more positive integers")
        return

    if spec.family == "transformer_encoder":
        if spec.modality != "text":
            raise ModelSpecError("transformer_encoder requires modality `text`")
        if spec.input_kind != "token_ids":
            raise ModelSpecError("transformer_encoder requires input.kind `token_ids`")
        if spec.token_vocab_size is None or spec.token_vocab_size <= 0:
            raise ModelSpecError("input.token_vocab_size must be positive for transformer_encoder")
        if spec.max_sequence_length is None or spec.max_sequence_length <= 0:
            raise ModelSpecError("input.max_sequence_length must be positive for transformer_encoder")
        if spec.model_dim is None or spec.model_dim <= 0:
            raise ModelSpecError("backbone.model_dim must be positive for transformer_encoder")
        if spec.num_heads is None or spec.num_heads <= 0:
            raise ModelSpecError("backbone.num_heads must be positive for transformer_encoder")
        if spec.num_layers is None or spec.num_layers <= 0:
            raise ModelSpecError("backbone.num_layers must be positive for transformer_encoder")
        if spec.ffn_dim is None or spec.ffn_dim <= 0:
            raise ModelSpecError("backbone.ffn_dim must be positive for transformer_encoder")
        if spec.model_dim % spec.num_heads != 0:
            raise ModelSpecError("backbone.model_dim must be divisible by backbone.num_heads")
        pooling = spec.pooling or "mean"
        if pooling not in VALID_POOLING:
            raise ModelSpecError(f"unsupported pooling `{pooling}` for transformer_encoder")
        return


def validate_spec_file(path: Path) -> dict[str, Any]:
    rust_payload = rust_validate_spec_file(path)
    if rust_payload is not None:
        return {
            "backend": "rust",
            "family": str(rust_payload["family"]),
            "task": str(rust_payload["task"]),
            "estimated_parameters": int(rust_payload["parameter_count"]),
            "param_store_manifest": rust_payload["param_store_manifest"],
        }

    spec = load_architecture_spec(path)
    return {
        "backend": "python",
        "family": spec.family,
        "task": spec.task,
        "estimated_parameters": parameter_count(spec),
        "param_store_manifest": build_param_store_manifest(spec),
    }


def parameter_count(spec: ArchitectureSpec) -> int:
    if spec.family == "tabular_mlp":
        total = 0
        input_dim = spec.feature_count
        for hidden_dim in spec.hidden_dims:
            total += input_dim * hidden_dim
            total += hidden_dim
            if spec.normalization != "none":
                total += hidden_dim * 2
            input_dim = hidden_dim
        total += input_dim * spec.head_output_dim
        total += spec.head_output_dim
        return total

    if spec.family == "transformer_encoder":
        assert spec.token_vocab_size is not None
        assert spec.max_sequence_length is not None
        assert spec.model_dim is not None
        assert spec.ffn_dim is not None
        assert spec.num_layers is not None
        total = spec.token_vocab_size * spec.model_dim
        total += spec.max_sequence_length * spec.model_dim
        for _layer_index in range(spec.num_layers):
            total += 3 * spec.model_dim * spec.model_dim
            total += 3 * spec.model_dim
            total += spec.model_dim * spec.model_dim
            total += spec.model_dim
            total += spec.model_dim * spec.ffn_dim
            total += spec.ffn_dim
            total += spec.ffn_dim * spec.model_dim
            total += spec.model_dim
            total += spec.model_dim * 4
        total += spec.model_dim * spec.head_output_dim
        total += spec.head_output_dim
        return total

    raise ModelSpecError(f"unsupported model family `{spec.family}`")


def build_param_store_manifest(spec: ArchitectureSpec) -> dict[str, Any]:
    parameters: list[dict[str, Any]] = []

    if spec.family == "tabular_mlp":
        input_dim = spec.feature_count
        for layer_index, hidden_dim in enumerate(spec.hidden_dims):
            parameters.append(
                {
                    "tensor_key": f"layers.{layer_index}.weight",
                    "owner_id": f"backbone.layer.{layer_index}",
                    "shape": [hidden_dim, input_dim],
                    "dtype": "float32",
                }
            )
            parameters.append(
                {
                    "tensor_key": f"layers.{layer_index}.bias",
                    "owner_id": f"backbone.layer.{layer_index}",
                    "shape": [hidden_dim],
                    "dtype": "float32",
                }
            )
            if spec.normalization != "none":
                parameters.append(
                    {
                        "tensor_key": f"norms.{layer_index}.weight",
                        "owner_id": f"backbone.norm.{layer_index}",
                        "shape": [hidden_dim],
                        "dtype": "float32",
                    }
                )
                parameters.append(
                    {
                        "tensor_key": f"norms.{layer_index}.bias",
                        "owner_id": f"backbone.norm.{layer_index}",
                        "shape": [hidden_dim],
                        "dtype": "float32",
                    }
                )
            input_dim = hidden_dim
        parameters.append(
            {
                "tensor_key": "head.weight",
                "owner_id": "head",
                "shape": [spec.head_output_dim, input_dim],
                "dtype": "float32",
            }
        )
        parameters.append(
            {
                "tensor_key": "head.bias",
                "owner_id": "head",
                "shape": [spec.head_output_dim],
                "dtype": "float32",
            }
        )
    elif spec.family == "transformer_encoder":
        assert spec.token_vocab_size is not None
        assert spec.max_sequence_length is not None
        assert spec.model_dim is not None
        assert spec.ffn_dim is not None
        assert spec.num_layers is not None
        parameters.extend(
            [
                {
                    "tensor_key": "token_embedding.weight",
                    "owner_id": "token_embedding",
                    "shape": [spec.token_vocab_size, spec.model_dim],
                    "dtype": "float32",
                },
                {
                    "tensor_key": "position_embedding.weight",
                    "owner_id": "position_embedding",
                    "shape": [spec.max_sequence_length, spec.model_dim],
                    "dtype": "float32",
                },
            ]
        )
        for layer_index in range(spec.num_layers):
            prefix = f"encoder.layers.{layer_index}"
            parameters.extend(
                [
                    {
                        "tensor_key": f"{prefix}.self_attn.in_proj_weight",
                        "owner_id": f"encoder.layer.{layer_index}.self_attn",
                        "shape": [spec.model_dim * 3, spec.model_dim],
                        "dtype": "float32",
                    },
                    {
                        "tensor_key": f"{prefix}.self_attn.in_proj_bias",
                        "owner_id": f"encoder.layer.{layer_index}.self_attn",
                        "shape": [spec.model_dim * 3],
                        "dtype": "float32",
                    },
                    {
                        "tensor_key": f"{prefix}.self_attn.out_proj.weight",
                        "owner_id": f"encoder.layer.{layer_index}.self_attn.out_proj",
                        "shape": [spec.model_dim, spec.model_dim],
                        "dtype": "float32",
                    },
                    {
                        "tensor_key": f"{prefix}.self_attn.out_proj.bias",
                        "owner_id": f"encoder.layer.{layer_index}.self_attn.out_proj",
                        "shape": [spec.model_dim],
                        "dtype": "float32",
                    },
                    {
                        "tensor_key": f"{prefix}.linear1.weight",
                        "owner_id": f"encoder.layer.{layer_index}.linear1",
                        "shape": [spec.ffn_dim, spec.model_dim],
                        "dtype": "float32",
                    },
                    {
                        "tensor_key": f"{prefix}.linear1.bias",
                        "owner_id": f"encoder.layer.{layer_index}.linear1",
                        "shape": [spec.ffn_dim],
                        "dtype": "float32",
                    },
                    {
                        "tensor_key": f"{prefix}.linear2.weight",
                        "owner_id": f"encoder.layer.{layer_index}.linear2",
                        "shape": [spec.model_dim, spec.ffn_dim],
                        "dtype": "float32",
                    },
                    {
                        "tensor_key": f"{prefix}.linear2.bias",
                        "owner_id": f"encoder.layer.{layer_index}.linear2",
                        "shape": [spec.model_dim],
                        "dtype": "float32",
                    },
                    {
                        "tensor_key": f"{prefix}.norm1.weight",
                        "owner_id": f"encoder.layer.{layer_index}.norm1",
                        "shape": [spec.model_dim],
                        "dtype": "float32",
                    },
                    {
                        "tensor_key": f"{prefix}.norm1.bias",
                        "owner_id": f"encoder.layer.{layer_index}.norm1",
                        "shape": [spec.model_dim],
                        "dtype": "float32",
                    },
                    {
                        "tensor_key": f"{prefix}.norm2.weight",
                        "owner_id": f"encoder.layer.{layer_index}.norm2",
                        "shape": [spec.model_dim],
                        "dtype": "float32",
                    },
                    {
                        "tensor_key": f"{prefix}.norm2.bias",
                        "owner_id": f"encoder.layer.{layer_index}.norm2",
                        "shape": [spec.model_dim],
                        "dtype": "float32",
                    },
                ]
            )
        parameters.append(
            {
                "tensor_key": "head.weight",
                "owner_id": "head",
                "shape": [spec.head_output_dim, spec.model_dim],
                "dtype": "float32",
            }
        )
        parameters.append(
            {
                "tensor_key": "head.bias",
                "owner_id": "head",
                "shape": [spec.head_output_dim],
                "dtype": "float32",
            }
        )
    else:
        raise ModelSpecError(f"unsupported model family `{spec.family}`")

    return {
        "schema_version": 1,
        "format": spec.param_store_format,
        "root_key": spec.param_store_root_key,
        "parameter_count": parameter_count(spec),
        "parameters": parameters,
    }


def diff_architecture_specs(old_spec: ArchitectureSpec, new_spec: ArchitectureSpec) -> dict[str, Any]:
    tracked_fields = {
        "family": (old_spec.family, new_spec.family),
        "task": (old_spec.task, new_spec.task),
        "modality": (old_spec.modality, new_spec.modality),
        "input_kind": (old_spec.input_kind, new_spec.input_kind),
        "feature_count": (old_spec.feature_count, new_spec.feature_count),
        "feature_names": (old_spec.feature_names, new_spec.feature_names),
        "token_vocab_size": (old_spec.token_vocab_size, new_spec.token_vocab_size),
        "max_sequence_length": (old_spec.max_sequence_length, new_spec.max_sequence_length),
        "target_column": (old_spec.target_column, new_spec.target_column),
        "hidden_dims": (old_spec.hidden_dims, new_spec.hidden_dims),
        "model_dim": (old_spec.model_dim, new_spec.model_dim),
        "num_heads": (old_spec.num_heads, new_spec.num_heads),
        "num_layers": (old_spec.num_layers, new_spec.num_layers),
        "ffn_dim": (old_spec.ffn_dim, new_spec.ffn_dim),
        "activation": (old_spec.activation, new_spec.activation),
        "normalization": (old_spec.normalization, new_spec.normalization),
        "dropout": (old_spec.dropout, new_spec.dropout),
        "pooling": (old_spec.pooling, new_spec.pooling),
    }
    changes = [
        {"field": field, "old_value": old_value, "new_value": new_value}
        for field, (old_value, new_value) in tracked_fields.items()
        if old_value != new_value
    ]
    compatible = (
        old_spec.task == new_spec.task
        and old_spec.modality == new_spec.modality
        and old_spec.target_kind == new_spec.target_kind
    )
    migration_plan = build_migration_plan(old_spec, new_spec)
    return {
        "compatible": compatible,
        "old_family": old_spec.family,
        "new_family": new_spec.family,
        "parameter_delta": parameter_count(new_spec) - parameter_count(old_spec),
        "changes": changes,
        "migration_plan": migration_plan,
    }


def build_migration_plan(old_spec: ArchitectureSpec, new_spec: ArchitectureSpec) -> dict[str, Any]:
    old_manifest = build_param_store_manifest(old_spec)
    new_manifest = build_param_store_manifest(new_spec)
    old_map = {item["tensor_key"]: item for item in old_manifest["parameters"]}
    actions: list[dict[str, Any]] = []
    exact_copy_count = 0
    partial_copy_count = 0
    reinitialize_count = 0

    for new_param in new_manifest["parameters"]:
        old_param = old_map.get(new_param["tensor_key"])
        if old_param is None:
            reinitialize_count += 1
            actions.append(
                {
                    "action": "reinitialize",
                    "target_tensor_key": new_param["tensor_key"],
                    "source_tensor_key": None,
                    "source_shape": None,
                    "target_shape": new_param["shape"],
                    "overlap_shape": None,
                    "reason": "tensor key is new in the updated spec",
                }
            )
            continue

        old_shape = [int(value) for value in old_param["shape"]]
        new_shape = [int(value) for value in new_param["shape"]]
        if old_shape == new_shape:
            exact_copy_count += 1
            actions.append(
                {
                    "action": "exact_copy",
                    "target_tensor_key": new_param["tensor_key"],
                    "source_tensor_key": old_param["tensor_key"],
                    "source_shape": old_shape,
                    "target_shape": new_shape,
                    "overlap_shape": new_shape,
                    "reason": "matching tensor key and shape",
                }
            )
            continue

        if len(old_shape) == len(new_shape):
            overlap_shape = [min(old_dim, new_dim) for old_dim, new_dim in zip(old_shape, new_shape)]
            if all(value > 0 for value in overlap_shape):
                partial_copy_count += 1
                actions.append(
                    {
                        "action": "partial_copy",
                        "target_tensor_key": new_param["tensor_key"],
                        "source_tensor_key": old_param["tensor_key"],
                        "source_shape": old_shape,
                        "target_shape": new_shape,
                        "overlap_shape": overlap_shape,
                        "reason": "tensor key matches but shape changed; preserve the overlapping slice",
                    }
                )
                continue

        reinitialize_count += 1
        actions.append(
            {
                "action": "reinitialize",
                "target_tensor_key": new_param["tensor_key"],
                "source_tensor_key": old_param["tensor_key"],
                "source_shape": old_shape,
                "target_shape": new_shape,
                "overlap_shape": None,
                "reason": "tensor rank changed or there is no meaningful overlap to preserve",
            }
        )

    return {
        "exact_copy_count": exact_copy_count,
        "partial_copy_count": partial_copy_count,
        "reinitialize_count": reinitialize_count,
        "actions": actions,
    }


def diff_spec_files(old_path: Path, new_path: Path) -> dict[str, Any]:
    rust_payload = rust_diff_spec_files(old_path, new_path)
    if rust_payload is not None:
        diff_payload = rust_payload["diff"]
        if not isinstance(diff_payload, dict):
            raise ModelSpecError("Rust IR diff payload is malformed")
        return diff_payload
    old_spec = load_architecture_spec(old_path)
    new_spec = load_architecture_spec(new_path)
    return diff_architecture_specs(old_spec, new_spec)


def migration_plan_spec_files(old_path: Path, new_path: Path) -> dict[str, Any]:
    rust_payload = rust_migration_plan_spec_files(old_path, new_path)
    if rust_payload is not None:
        plan_payload = rust_payload["migration_plan"]
        if not isinstance(plan_payload, dict):
            raise ModelSpecError("Rust IR migration plan payload is malformed")
        return plan_payload
    old_spec = load_architecture_spec(old_path)
    new_spec = load_architecture_spec(new_path)
    return build_migration_plan(old_spec, new_spec)


EDITABLE_SPEC_FIELDS = {
    "name": "name",
    "model.name": "name",
    "family": "family",
    "model.family": "family",
    "task": "task",
    "model.task": "task",
    "modality": "modality",
    "model.modality": "modality",
    "input_kind": "input_kind",
    "input.kind": "input_kind",
    "feature_names": "feature_names",
    "input.feature_names": "feature_names",
    "feature_count": "feature_count",
    "input.feature_count": "feature_count",
    "token_vocab_size": "token_vocab_size",
    "input.token_vocab_size": "token_vocab_size",
    "max_sequence_length": "max_sequence_length",
    "input.max_sequence_length": "max_sequence_length",
    "target_column": "target_column",
    "target.column": "target_column",
    "target_kind": "target_kind",
    "target.kind": "target_kind",
    "hidden_dims": "hidden_dims",
    "backbone.hidden_dims": "hidden_dims",
    "model_dim": "model_dim",
    "backbone.model_dim": "model_dim",
    "num_heads": "num_heads",
    "backbone.num_heads": "num_heads",
    "num_layers": "num_layers",
    "backbone.num_layers": "num_layers",
    "ffn_dim": "ffn_dim",
    "backbone.ffn_dim": "ffn_dim",
    "activation": "activation",
    "backbone.activation": "activation",
    "normalization": "normalization",
    "backbone.normalization": "normalization",
    "dropout": "dropout",
    "backbone.dropout": "dropout",
    "pooling": "pooling",
    "head.pooling": "pooling",
    "head_output_dim": "head_output_dim",
    "head.output_dim": "head_output_dim",
    "loss_kind": "loss_kind",
    "procedures.loss.kind": "loss_kind",
    "param_store_root_key": "param_store_root_key",
    "param_store.root_key": "param_store_root_key",
}


def _parse_assignment_value(field: str, raw_value: str) -> Any:
    text = raw_value.strip()
    if text == "":
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    canonical = EDITABLE_SPEC_FIELDS.get(field, field)
    if canonical in {
        "feature_names",
        "hidden_dims",
    }:
        items = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
        if canonical == "hidden_dims":
            return [int(item) for item in items]
        return items
    if canonical in {
        "feature_count",
        "token_vocab_size",
        "max_sequence_length",
        "model_dim",
        "num_heads",
        "num_layers",
        "ffn_dim",
        "head_output_dim",
    }:
        return int(text)
    if canonical == "dropout":
        return float(text)
    if text.lower() == "none":
        return None
    return text


def _normalize_edited_value(field_name: str, value: Any) -> Any:
    if field_name == "feature_names":
        if not isinstance(value, list):
            raise ModelSpecError("feature_names must be a JSON array or comma-separated list")
        return [str(item).strip() for item in value if str(item).strip()]
    if field_name == "hidden_dims":
        if not isinstance(value, list):
            raise ModelSpecError("hidden_dims must be a JSON array or comma-separated list")
        return [int(item) for item in value]
    if field_name in {
        "feature_count",
        "token_vocab_size",
        "max_sequence_length",
        "model_dim",
        "num_heads",
        "num_layers",
        "ffn_dim",
        "head_output_dim",
    }:
        return None if value is None else int(value)
    if field_name == "dropout":
        return float(value)
    if field_name in {
        "name",
        "family",
        "task",
        "modality",
        "input_kind",
        "target_column",
        "target_kind",
        "activation",
        "normalization",
        "pooling",
        "loss_kind",
        "param_store_root_key",
    }:
        return None if value is None else str(value).strip()
    return value


def edit_architecture_spec(spec: ArchitectureSpec, assignments: list[str]) -> ArchitectureSpec:
    if not assignments:
        raise ModelSpecError("at least one --set assignment is required")

    updates: dict[str, Any] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise ModelSpecError(f"invalid assignment `{assignment}`; expected KEY=VALUE")
        raw_field, raw_value = assignment.split("=", 1)
        field = clean_optional(raw_field)
        if field is None:
            raise ModelSpecError(f"invalid assignment `{assignment}`; field name is empty")
        field_name = EDITABLE_SPEC_FIELDS.get(field)
        if field_name is None:
            supported = ", ".join(sorted(EDITABLE_SPEC_FIELDS))
            raise ModelSpecError(f"unsupported editable field `{field}`; supported keys: {supported}")
        updates[field_name] = _normalize_edited_value(field_name, _parse_assignment_value(field, raw_value))

    if "feature_names" in updates and "feature_count" not in updates:
        updates["feature_count"] = len(updates["feature_names"])

    edited = replace(spec, **updates)
    validate_architecture_spec(edited)
    return edited


def write_edited_architecture_spec(
    spec_path: Path,
    assignments: list[str],
    *,
    output_path: Path | None = None,
    inplace: bool = False,
) -> dict[str, Any]:
    spec = load_architecture_spec(spec_path)
    edited = edit_architecture_spec(spec, assignments)
    rendered = render_model_spec_toml(edited)

    if inplace:
        backup_path = spec_path.with_name(f"{spec_path.stem}.{slugify(now_utc(), fallback='backup')}.bak{spec_path.suffix}")
        backup_path.write_text(spec_path.read_text())
        spec_path.write_text(rendered)
        target_path = spec_path
    else:
        backup_path = None
        target_path = output_path or spec_path.with_name(f"{spec_path.stem}.edited{spec_path.suffix}")
        if target_path.exists():
            raise ModelSpecError(f"refusing to overwrite existing edited spec without --in-place: {target_path}")
        target_path.write_text(rendered)

    return {
        "target_path": target_path,
        "backup_path": backup_path,
        "edited_spec": edited,
        "diff": diff_architecture_specs(spec, edited),
    }


def render_model_spec_toml(spec: ArchitectureSpec) -> str:
    lines = [
        "[model]",
        f"name = {_json_string(spec.name)}",
        f"family = {_json_string(spec.family)}",
        f"task = {_json_string(spec.task)}",
        f"modality = {_json_string(spec.modality)}",
        "",
        "[input]",
        f"kind = {_json_string(spec.input_kind)}",
    ]
    if spec.input_kind == "dense_features":
        lines.extend(
            [
                f"feature_names = {_json_string_array(spec.feature_names)}",
                f"feature_count = {spec.feature_count}",
            ]
        )
    else:
        lines.extend(
            [
                f"token_vocab_size = {spec.token_vocab_size or 0}",
                f"max_sequence_length = {spec.max_sequence_length or 0}",
            ]
        )
    lines.extend(
        [
            "",
            "[target]",
            f"column = {_json_string(spec.target_column)}",
            f"kind = {_json_string(spec.target_kind)}",
            "",
            "[backbone]",
        ]
    )
    if spec.family == "tabular_mlp":
        lines.append(f"hidden_dims = {json.dumps(spec.hidden_dims)}")
    else:
        lines.extend(
            [
                f"model_dim = {spec.model_dim or 0}",
                f"num_heads = {spec.num_heads or 0}",
                f"num_layers = {spec.num_layers or 0}",
                f"ffn_dim = {spec.ffn_dim or 0}",
            ]
        )
    lines.extend(
        [
            f"activation = {_json_string(spec.activation)}",
            f"normalization = {_json_string(spec.normalization)}",
            f"dropout = {spec.dropout}",
            "",
            "[head]",
            f"output_dim = {spec.head_output_dim}",
        ]
    )
    if spec.pooling is not None:
        lines.append(f"pooling = {_json_string(spec.pooling)}")
    lines.extend(
        [
            "",
            "[procedures.forward]",
            f"input = {_json_string(spec.forward_input)}",
            f"output = {_json_string(spec.forward_output)}",
            "",
            "[procedures.loss]",
            f"kind = {_json_string(spec.loss_kind)}",
            'prediction = "logits"',
            'target = "target"',
            "",
            "[param_store]",
            f"format = {_json_string(spec.param_store_format)}",
            f"root_key = {_json_string(spec.param_store_root_key)}",
            "",
        ]
    )
    return "\n".join(lines)


def render_training_spec_toml(spec: TrainingSpec) -> str:
    return (
        "[training]\n"
        f"epochs = {spec.epochs}\n"
        f"batch_size = {spec.batch_size}\n"
        f"learning_rate = {spec.learning_rate}\n"
        f"optimizer = {_json_string(spec.optimizer)}\n"
        f"weight_decay = {spec.weight_decay}\n"
        f"primary_metric = {_json_string(spec.primary_metric)}\n"
    )


def render_dataset_facts_toml(facts: DatasetFacts) -> str:
    row_count = 0 if facts.row_count_estimate is None else facts.row_count_estimate
    return (
        "[dataset]\n"
        f"name = {_json_string(facts.dataset_name)}\n"
        f"path = {_json_string(str(facts.dataset_path))}\n"
        f"modality = {_json_string(facts.modality)}\n"
        f"suspected_problem_type = {_json_string(facts.suspected_problem_type)}\n"
        f"row_count_estimate = {row_count}\n"
        f"row_count_estimate_known = {json.dumps(facts.row_count_estimate is not None).lower()}\n"
        "\n"
        "[columns]\n"
        f"names = {_json_string_array(facts.column_names)}\n"
        f"feature_names = {_json_string_array(facts.feature_names)}\n"
        f"target_column = {_json_string(facts.target_column)}\n"
        f"target_candidates = {_json_string_array(facts.target_candidates)}\n"
        f"id_candidates = {_json_string_array(facts.id_candidates)}\n"
        f"time_candidates = {_json_string_array(facts.time_candidates)}\n"
        "\n"
        "[source]\n"
        f"report_path = {_json_string(str(facts.source_report_path))}\n"
    )


def default_training_spec(facts: DatasetFacts, *, family: str = "tabular_mlp") -> TrainingSpec:
    batch_size = 32
    if facts.row_count_estimate is not None and facts.row_count_estimate > 0:
        batch_size = min(32, max(4, facts.row_count_estimate))
    learning_rate = 0.001
    if family == "transformer_encoder":
        learning_rate = 0.0003
        batch_size = min(batch_size, 16)
    return TrainingSpec(
        epochs=25,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer="adamw",
        weight_decay=0.0001,
        primary_metric="accuracy",
    )


def architecture_spec_from_dataset_facts(
    *,
    facts: DatasetFacts,
    pipeline_name: str,
    recipe_name: str,
) -> ArchitectureSpec:
    if recipe_name == "tabular.binary.basic":
        if "binary" not in facts.suspected_problem_type.lower():
            raise ModelSpecError(
                f"unsupported problem type for recipe `{recipe_name}`: `{facts.suspected_problem_type}`"
            )
        if facts.modality != "tabular":
            raise ModelSpecError(f"recipe `{recipe_name}` requires tabular dataset facts")
        return ArchitectureSpec(
            name=f"{pipeline_name} binary tabular model",
            family="tabular_mlp",
            task="binary_classification",
            modality="tabular",
            input_kind="dense_features",
            feature_names=facts.feature_names,
            feature_count=len(facts.feature_names),
            token_vocab_size=None,
            max_sequence_length=None,
            target_column=facts.target_column,
            target_kind="binary",
            hidden_dims=[128, 64],
            model_dim=None,
            num_heads=None,
            num_layers=None,
            ffn_dim=None,
            activation="relu",
            normalization="layernorm",
            dropout=0.1,
            pooling=None,
            head_output_dim=1,
            forward_input="features",
            forward_output="logits",
            loss_kind="bce_with_logits",
            param_store_format="safetensors",
            param_store_root_key=slugify(pipeline_name, fallback="model").replace("-", "_"),
        )
    if recipe_name == "text.binary.transformer":
        if "binary" not in facts.suspected_problem_type.lower():
            raise ModelSpecError(
                f"unsupported problem type for recipe `{recipe_name}`: `{facts.suspected_problem_type}`"
            )
        if facts.modality != "text":
            raise ModelSpecError(f"recipe `{recipe_name}` requires text dataset facts")
        return ArchitectureSpec(
            name=f"{pipeline_name} binary text transformer",
            family="transformer_encoder",
            task="binary_classification",
            modality="text",
            input_kind="token_ids",
            feature_names=[],
            feature_count=0,
            token_vocab_size=32000,
            max_sequence_length=256,
            target_column=facts.target_column,
            target_kind="binary",
            hidden_dims=[],
            model_dim=128,
            num_heads=4,
            num_layers=2,
            ffn_dim=256,
            activation="gelu",
            normalization="layernorm",
            dropout=0.1,
            pooling="mean",
            head_output_dim=1,
            forward_input="features",
            forward_output="logits",
            loss_kind="bce_with_logits",
            param_store_format="safetensors",
            param_store_root_key=slugify(pipeline_name, fallback="model").replace("-", "_"),
        )
    raise ModelSpecError(f"unsupported recipe `{recipe_name}`")


def dataset_facts_from_report_path(report_path: Path) -> DatasetFacts:
    payload = load_json(report_path)
    report = payload.get("report", payload.get("data_report", payload))
    if not isinstance(report, dict):
        raise ModelSpecError(f"report payload is malformed: {report_path}")

    structure = report.get("structure", {})
    if not isinstance(structure, dict):
        raise ModelSpecError(f"report structure is malformed: {report_path}")

    column_payloads = structure.get("columns", [])
    if not isinstance(column_payloads, list):
        column_payloads = []
    column_names = [
        clean_optional(str(column.get("name", "")))
        for column in column_payloads
        if isinstance(column, dict)
    ]
    column_names = [name for name in column_names if name is not None]

    target_candidates = _clean_list(structure.get("target_candidates", []))
    role_guess_target = [
        str(column.get("name", "")).strip()
        for column in column_payloads
        if isinstance(column, dict) and str(column.get("role_guess", "")).strip() == "target"
    ]
    target_column = _first_nonempty(target_candidates + role_guess_target)
    if target_column is None:
        fallback = next((name for name in column_names if name.lower() in {"target", "label", "y"}), None)
        if fallback is None:
            raise ModelSpecError("could not infer a target column from the delegated report")
        target_column = fallback

    feature_names = [name for name in column_names if name != target_column]
    if not feature_names:
        raise ModelSpecError("no feature columns remain after removing the target column")

    row_count_estimate_raw = structure.get("row_count_estimate")
    row_count_estimate = int(row_count_estimate_raw) if isinstance(row_count_estimate_raw, int) else None

    dataset_name = clean_optional(str(report.get("dataset_name", ""))) or report_path.stem
    dataset_path_text = clean_optional(str(report.get("dataset_path", "")))
    if dataset_path_text is None:
        raise ModelSpecError("delegated report is missing dataset_path")

    suspected_domain = clean_optional(str(report.get("suspected_domain", ""))) or ""
    modality = "text" if "text" in suspected_domain.lower() or "nlp" in suspected_domain.lower() else "tabular"

    return DatasetFacts(
        dataset_name=dataset_name,
        dataset_path=Path(dataset_path_text).expanduser().resolve(),
        modality=modality,
        suspected_problem_type=clean_optional(str(report.get("suspected_problem_type", "")))
        or "binary classification",
        row_count_estimate=row_count_estimate,
        column_names=column_names,
        feature_names=feature_names,
        target_column=target_column,
        target_candidates=target_candidates,
        id_candidates=_clean_list(structure.get("id_candidates", [])),
        time_candidates=_clean_list(structure.get("time_candidates", [])),
        source_report_path=report_path.resolve(),
    )


def _activation_expr(name: str) -> str:
    mapping = {
        "relu": "torch.relu",
        "gelu": "F.gelu",
        "silu": "F.silu",
        "tanh": "torch.tanh",
    }
    return mapping[name]


def render_compiled_model_python(spec: ArchitectureSpec) -> str:
    if spec.family == "tabular_mlp":
        norm_factory = {
            "none": "nn.Identity()",
            "batchnorm": "nn.BatchNorm1d(hidden_dim)",
            "layernorm": "nn.LayerNorm(hidden_dim)",
        }[spec.normalization]
        dropout_block = ""
        if spec.dropout > 0.0:
            dropout_block = "            x = F.dropout(x, p=self.dropout, training=self.training)\n"
        return f'''from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class {spec.class_name}(nn.Module):
    family = {json.dumps(spec.family)}
    task = {json.dumps(spec.task)}
    target_column = {json.dumps(spec.target_column)}

    def __init__(self) -> None:
        super().__init__()
        self.feature_names = {json.dumps(spec.feature_names)}
        self.input_dim = {spec.feature_count}
        self.hidden_dims = {json.dumps(spec.hidden_dims)}
        self.dropout = {spec.dropout}
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.norms.append({norm_factory})
            in_dim = hidden_dim
        self.head = nn.Linear(in_dim, {spec.head_output_dim})

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            x = norm(x)
            x = {_activation_expr(spec.activation)}(x)
{dropout_block}        logits = self.head(x)
        return logits


__all__ = [{json.dumps(spec.class_name)}]
'''

    if spec.family == "transformer_encoder":
        pooling = spec.pooling or "mean"
        pooling_block = "encoded[:, 0, :]" if pooling == "cls" else "encoded.mean(dim=1)"
        activation_expr = _activation_expr(spec.activation)
        assert spec.token_vocab_size is not None
        assert spec.max_sequence_length is not None
        assert spec.model_dim is not None
        assert spec.num_heads is not None
        assert spec.num_layers is not None
        assert spec.ffn_dim is not None
        return f'''from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class {spec.class_name}(nn.Module):
    family = {json.dumps(spec.family)}
    task = {json.dumps(spec.task)}
    target_column = {json.dumps(spec.target_column)}

    def __init__(self) -> None:
        super().__init__()
        self.max_sequence_length = {spec.max_sequence_length}
        self.token_embedding = nn.Embedding({spec.token_vocab_size}, {spec.model_dim})
        self.position_embedding = nn.Embedding({spec.max_sequence_length}, {spec.model_dim})
        encoder_layer = nn.TransformerEncoderLayer(
            d_model={spec.model_dim},
            nhead={spec.num_heads},
            dim_feedforward={spec.ffn_dim},
            dropout={spec.dropout},
            activation={activation_expr},
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers={spec.num_layers})
        self.head = nn.Linear({spec.model_dim}, {spec.head_output_dim})

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(features.size(1), device=features.device).unsqueeze(0)
        positions = positions.clamp(max=self.max_sequence_length - 1)
        embedded = self.token_embedding(features) + self.position_embedding(positions)
        encoded = self.encoder(embedded)
        pooled = {pooling_block}
        logits = self.head(pooled)
        return logits


__all__ = [{json.dumps(spec.class_name)}]
'''

    raise ModelSpecError(f"unsupported model family `{spec.family}`")


def compile_architecture_spec(spec: ArchitectureSpec, output_dir: Path) -> dict[str, Any]:
    validate_architecture_spec(spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    module_path = output_dir / "compiled_model.py"
    manifest_path = output_dir / "compile_manifest.json"
    param_store_path = output_dir / "param_store_manifest.json"
    spec_snapshot_path = output_dir / "model_spec.snapshot.json"

    module_path.write_text(render_compiled_model_python(spec))
    param_store_manifest = build_param_store_manifest(spec)
    write_json(param_store_path, param_store_manifest)

    spec_snapshot = asdict(spec)
    spec_snapshot["compiled_at"] = now_utc()
    spec_snapshot["class_name"] = spec.class_name
    write_json(spec_snapshot_path, spec_snapshot)

    manifest = {
        "schema_version": 1,
        "compiled_at": now_utc(),
        "family": spec.family,
        "task": spec.task,
        "class_name": spec.class_name,
        "module_path": str(module_path),
        "param_store_manifest_path": str(param_store_path),
        "spec_snapshot_path": str(spec_snapshot_path),
        "estimated_parameter_count": parameter_count(spec),
    }
    write_json(manifest_path, manifest)
    return {
        "module_path": module_path,
        "manifest_path": manifest_path,
        "param_store_manifest_path": param_store_path,
        "spec_snapshot_path": spec_snapshot_path,
        "class_name": spec.class_name,
        "estimated_parameter_count": parameter_count(spec),
    }


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ModelSpecError("checkpoint migration requires `torch` in the active Python environment") from exc
    return torch


def _load_compiled_model_class(module_path: Path, class_name: str) -> type[Any]:
    module_name = f"machinate_compiled_{slugify(module_path.stem, fallback='compiled')}_{abs(hash(str(module_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ModelSpecError(f"could not load compiled model module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class = getattr(module, class_name, None)
    if model_class is None:
        raise ModelSpecError(f"compiled model class `{class_name}` is missing from {module_path}")
    return model_class


def _fresh_state_dict_for_spec(spec: ArchitectureSpec) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="machinate-compile-") as tempdir:
        output_dir = Path(tempdir)
        artifacts = compile_architecture_spec(spec, output_dir)
        model_class = _load_compiled_model_class(Path(artifacts["module_path"]), str(artifacts["class_name"]))
        model = model_class()
        return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _load_state_dict_payload(state_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = _require_torch()
    payload = torch.load(state_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload, dict(payload["state_dict"])
    if isinstance(payload, dict):
        return {"state_dict": payload}, dict(payload)
    raise ModelSpecError(f"unsupported checkpoint payload at {state_path}; expected a dict-like torch payload")


def _copy_tensor_overlap(*, source: Any, target: Any, overlap_shape: list[int] | None) -> Any:
    migrated = target.clone()
    if overlap_shape is None:
        return migrated
    slices = tuple(slice(0, int(dim)) for dim in overlap_shape)
    migrated[slices] = source[slices].to(dtype=target.dtype)
    return migrated


def migrate_checkpoint(
    *,
    old_spec: ArchitectureSpec,
    new_spec: ArchitectureSpec,
    source_state_path: Path,
    output_state_path: Path,
    plan_path: Path | None = None,
) -> dict[str, Any]:
    torch = _require_torch()
    migration_plan = build_migration_plan(old_spec, new_spec)
    source_payload, source_state = _load_state_dict_payload(source_state_path)
    target_state = _fresh_state_dict_for_spec(new_spec)

    preserved = 0
    for action in migration_plan["actions"]:
        target_key = str(action["target_tensor_key"])
        source_key = clean_optional(str(action.get("source_tensor_key", "")))
        if source_key is None or source_key not in source_state or target_key not in target_state:
            continue
        if action["action"] == "exact_copy":
            target_state[target_key] = source_state[source_key].detach().clone().to(dtype=target_state[target_key].dtype)
            preserved += 1
            continue
        if action["action"] == "partial_copy":
            overlap_shape = action.get("overlap_shape")
            if isinstance(overlap_shape, list):
                target_state[target_key] = _copy_tensor_overlap(
                    source=source_state[source_key],
                    target=target_state[target_key],
                    overlap_shape=[int(value) for value in overlap_shape],
                )
                preserved += 1

    output_state_path.parent.mkdir(parents=True, exist_ok=True)
    migrated_payload = dict(source_payload)
    migrated_payload["state_dict"] = target_state
    migrated_payload["migration_plan"] = migration_plan
    migrated_payload["source_spec_family"] = old_spec.family
    migrated_payload["target_spec_family"] = new_spec.family
    torch.save(migrated_payload, output_state_path)

    if plan_path is not None:
        write_json(
            plan_path,
            {
                "schema_version": 1,
                "generated_at": now_utc(),
                "source_state_path": str(source_state_path),
                "output_state_path": str(output_state_path),
                "preserved_tensor_count": preserved,
                "migration_plan": migration_plan,
            },
        )

    return {
        "output_state_path": output_state_path,
        "plan_path": plan_path,
        "preserved_tensor_count": preserved,
        "migration_plan": migration_plan,
    }


def resolve_pipeline_spec_paths(
    pipeline_root: Path,
    pipeline_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    raw_specs = {}
    if pipeline_config is not None:
        candidate = pipeline_config.get("specs", {})
        if isinstance(candidate, dict):
            raw_specs = candidate
    return {
        "dataset_facts": pipeline_root / str(raw_specs.get("dataset_facts", "dataset_facts.toml")),
        "model": pipeline_root / str(raw_specs.get("model", "model.toml")),
        "training": pipeline_root / str(raw_specs.get("training", "training.toml")),
    }


def prepare_training_runtime(
    pipeline_root: Path,
    pipeline_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec_paths = resolve_pipeline_spec_paths(pipeline_root, pipeline_config)
    model_path = spec_paths["model"]
    training_path = spec_paths["training"]
    if not model_path.exists():
        raise ModelSpecError(f"model spec is missing: {model_path}")
    if not training_path.exists():
        raise ModelSpecError(f"training spec is missing: {training_path}")

    model_spec = load_architecture_spec(model_path)
    training_spec = load_training_spec(training_path)
    validate_training_spec(training_spec)
    compile_artifacts = compile_architecture_spec(model_spec, pipeline_root / "outputs" / "compiled_model")
    return {
        "model_spec": model_spec,
        "training_spec": training_spec,
        "compile_artifacts": compile_artifacts,
        "spec_paths": spec_paths,
    }
