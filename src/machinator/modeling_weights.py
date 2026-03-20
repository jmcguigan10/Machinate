from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

from machinator.core import clean_optional, now_utc, slugify, write_json
from machinator.modeling_rust import rust_diff_spec_files, rust_migration_plan_spec_files
from machinator.modeling_specs import load_architecture_spec, parameter_count
from machinator.modeling_types import ArchitectureSpec, ModelSpecError


def build_param_store_manifest(spec: ArchitectureSpec) -> dict[str, Any]:
    # The architecture remains the source of truth for required parameter slots,
    # but the manifest is materialized as a separate artifact so it can later be
    # analyzed, migrated, clustered, or transformed independently of the code.
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
            layer_prefix = f"encoder.layers.{layer_index}"
            attention_owner = f"backbone.layer.{layer_index}.attention"
            parameters.extend(
                [
                    {
                        "tensor_key": f"{layer_prefix}.self_attn.in_proj_weight",
                        "owner_id": attention_owner,
                        "shape": [spec.model_dim * 3, spec.model_dim],
                        "dtype": "float32",
                    },
                    {
                        "tensor_key": f"{layer_prefix}.self_attn.in_proj_bias",
                        "owner_id": attention_owner,
                        "shape": [spec.model_dim * 3],
                        "dtype": "float32",
                    },
                    {
                        "tensor_key": f"{layer_prefix}.self_attn.out_proj.weight",
                        "owner_id": attention_owner,
                        "shape": [spec.model_dim, spec.model_dim],
                        "dtype": "float32",
                    },
                    {
                        "tensor_key": f"{layer_prefix}.self_attn.out_proj.bias",
                        "owner_id": attention_owner,
                        "shape": [spec.model_dim],
                        "dtype": "float32",
                    },
                    {
                        "tensor_key": f"{layer_prefix}.linear1.weight",
                        "owner_id": f"backbone.layer.{layer_index}.ffn",
                        "shape": [spec.ffn_dim, spec.model_dim],
                        "dtype": "float32",
                    },
                    {
                        "tensor_key": f"{layer_prefix}.linear1.bias",
                        "owner_id": f"backbone.layer.{layer_index}.ffn",
                        "shape": [spec.ffn_dim],
                        "dtype": "float32",
                    },
                    {
                        "tensor_key": f"{layer_prefix}.linear2.weight",
                        "owner_id": f"backbone.layer.{layer_index}.ffn",
                        "shape": [spec.model_dim, spec.ffn_dim],
                        "dtype": "float32",
                    },
                    {
                        "tensor_key": f"{layer_prefix}.linear2.bias",
                        "owner_id": f"backbone.layer.{layer_index}.ffn",
                        "shape": [spec.model_dim],
                        "dtype": "float32",
                    },
                    {
                        "tensor_key": f"{layer_prefix}.norm1.weight",
                        "owner_id": f"backbone.layer.{layer_index}.norm1",
                        "shape": [spec.model_dim],
                        "dtype": "float32",
                    },
                    {
                        "tensor_key": f"{layer_prefix}.norm1.bias",
                        "owner_id": f"backbone.layer.{layer_index}.norm1",
                        "shape": [spec.model_dim],
                        "dtype": "float32",
                    },
                    {
                        "tensor_key": f"{layer_prefix}.norm2.weight",
                        "owner_id": f"backbone.layer.{layer_index}.norm2",
                        "shape": [spec.model_dim],
                        "dtype": "float32",
                    },
                    {
                        "tensor_key": f"{layer_prefix}.norm2.bias",
                        "owner_id": f"backbone.layer.{layer_index}.norm2",
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


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ModelSpecError("checkpoint migration requires `torch` in the active Python environment") from exc
    return torch


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


def _fresh_state_dict_for_spec(spec: ArchitectureSpec) -> dict[str, Any]:
    from machinator.modeling_compile import compile_architecture_spec, load_compiled_model_class

    with tempfile.TemporaryDirectory(prefix="machinator-compile-") as tempdir:
        output_dir = Path(tempdir)
        artifacts = compile_architecture_spec(spec, output_dir)
        model_class = load_compiled_model_class(Path(artifacts["module_path"]), str(artifacts["class_name"]))
        model = model_class()
        return {key: value.detach().clone() for key, value in model.state_dict().items()}


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
