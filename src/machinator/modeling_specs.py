from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tomllib
from typing import Any

from machinator.core import clean_optional, now_utc, slugify
from machinator.modeling_rust import rust_validate_spec_file
from machinator.modeling_types import (
    ArchitectureSpec,
    ModelSpecError,
    TrainingSpec,
    VALID_ACTIVATIONS,
    VALID_INPUT_KINDS,
    VALID_LOSSES,
    VALID_MODEL_FAMILIES,
    VALID_MODALITIES,
    VALID_NORMALIZATIONS,
    VALID_POOLING,
    VALID_TARGET_KINDS,
    VALID_TASKS,
)

# Spec parsing/rendering/editing lives here. This is the file to inspect if you
# want to see how `model.toml` and `training.toml` are read, validated, or
# rewritten without yet looking at weight migration or code generation.


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
    conv_channels = [int(value) for value in backbone.get("channels", [])]

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
        image_channels=_int_or_none(input_section.get("image_channels")),
        image_height=_int_or_none(input_section.get("image_height")),
        image_width=_int_or_none(input_section.get("image_width")),
        target_column=str(target.get("column", "")).strip(),
        target_kind=str(target.get("kind", "")).strip(),
        hidden_dims=hidden_dims,
        conv_channels=conv_channels,
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

    if spec.family == "vision_cnn":
        if spec.modality != "vision":
            raise ModelSpecError("vision_cnn requires modality `vision`")
        if spec.input_kind != "image_tensor":
            raise ModelSpecError("vision_cnn requires input.kind `image_tensor`")
        if spec.image_channels is None or spec.image_channels <= 0:
            raise ModelSpecError("input.image_channels must be positive for vision_cnn")
        if spec.image_height is None or spec.image_height <= 0:
            raise ModelSpecError("input.image_height must be positive for vision_cnn")
        if spec.image_width is None or spec.image_width <= 0:
            raise ModelSpecError("input.image_width must be positive for vision_cnn")
        if not spec.conv_channels or any(value <= 0 for value in spec.conv_channels):
            raise ModelSpecError("backbone.channels must contain one or more positive integers for vision_cnn")
        if spec.normalization not in {"none", "batchnorm"}:
            raise ModelSpecError("vision_cnn supports only `none` or `batchnorm` normalization")
        pooling = spec.pooling or "avg"
        if pooling != "avg":
            raise ModelSpecError("vision_cnn currently supports only `avg` pooling")
        return

    if spec.family == "vision_resnet":
        if spec.modality != "vision":
            raise ModelSpecError("vision_resnet requires modality `vision`")
        if spec.input_kind != "image_tensor":
            raise ModelSpecError("vision_resnet requires input.kind `image_tensor`")
        if spec.image_channels is None or spec.image_channels <= 0:
            raise ModelSpecError("input.image_channels must be positive for vision_resnet")
        if spec.image_height is None or spec.image_height <= 0:
            raise ModelSpecError("input.image_height must be positive for vision_resnet")
        if spec.image_width is None or spec.image_width <= 0:
            raise ModelSpecError("input.image_width must be positive for vision_resnet")
        if not spec.conv_channels or any(value <= 0 for value in spec.conv_channels):
            raise ModelSpecError("backbone.channels must contain one or more positive integers for vision_resnet")
        if spec.num_layers is None or spec.num_layers <= 0:
            raise ModelSpecError("backbone.num_layers must be positive for vision_resnet")
        if spec.normalization not in {"none", "batchnorm"}:
            raise ModelSpecError("vision_resnet supports only `none` or `batchnorm` normalization")
        pooling = spec.pooling or "avg"
        if pooling != "avg":
            raise ModelSpecError("vision_resnet currently supports only `avg` pooling")
        return


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

    if spec.family == "vision_cnn":
        assert spec.image_channels is not None
        total = 0
        in_channels = spec.image_channels
        for out_channels in spec.conv_channels:
            total += out_channels * in_channels * 3 * 3
            total += out_channels
            if spec.normalization != "none":
                total += out_channels * 2
            in_channels = out_channels
        total += in_channels * spec.head_output_dim
        total += spec.head_output_dim
        return total

    if spec.family == "vision_resnet":
        assert spec.image_channels is not None
        assert spec.num_layers is not None
        total = 0
        in_channels = spec.image_channels
        stem_channels = spec.conv_channels[0]
        total += stem_channels * in_channels * 3 * 3
        total += stem_channels
        if spec.normalization != "none":
            total += stem_channels * 2
        in_channels = stem_channels
        for stage_channels in spec.conv_channels:
            for block_index in range(spec.num_layers):
                stride_projection = block_index == 0 and in_channels != stage_channels
                total += stage_channels * in_channels * 3 * 3
                total += stage_channels
                total += stage_channels * stage_channels * 3 * 3
                total += stage_channels
                if spec.normalization != "none":
                    total += stage_channels * 4
                if stride_projection:
                    total += stage_channels * in_channels
                    total += stage_channels
                    if spec.normalization != "none":
                        total += stage_channels * 2
                in_channels = stage_channels
        total += in_channels * spec.head_output_dim
        total += spec.head_output_dim
        return total

    raise ModelSpecError(f"unsupported model family `{spec.family}`")


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

    from machinator.modeling_weights import build_param_store_manifest

    spec = load_architecture_spec(path)
    return {
        "backend": "python",
        "family": spec.family,
        "task": spec.task,
        "estimated_parameters": parameter_count(spec),
        "param_store_manifest": build_param_store_manifest(spec),
    }


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
    "image_channels": "image_channels",
    "input.image_channels": "image_channels",
    "image_height": "image_height",
    "input.image_height": "image_height",
    "image_width": "image_width",
    "input.image_width": "image_width",
    "target_column": "target_column",
    "target.column": "target_column",
    "target_kind": "target_kind",
    "target.kind": "target_kind",
    "hidden_dims": "hidden_dims",
    "backbone.hidden_dims": "hidden_dims",
    "conv_channels": "conv_channels",
    "backbone.channels": "conv_channels",
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
    if canonical in {"feature_names", "hidden_dims", "conv_channels"}:
        items = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
        if canonical in {"hidden_dims", "conv_channels"}:
            return [int(item) for item in items]
        return items
    if canonical in {
        "feature_count",
        "token_vocab_size",
        "max_sequence_length",
        "image_channels",
        "image_height",
        "image_width",
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
    if field_name in {"hidden_dims", "conv_channels"}:
        if not isinstance(value, list):
            raise ModelSpecError(f"{field_name} must be a JSON array or comma-separated list")
        return [int(item) for item in value]
    if field_name in {
        "feature_count",
        "token_vocab_size",
        "max_sequence_length",
        "image_channels",
        "image_height",
        "image_width",
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
    elif spec.input_kind == "token_ids":
        lines.extend(
            [
                f"token_vocab_size = {spec.token_vocab_size or 0}",
                f"max_sequence_length = {spec.max_sequence_length or 0}",
            ]
        )
    else:
        lines.extend(
            [
                f"image_channels = {spec.image_channels or 0}",
                f"image_height = {spec.image_height or 0}",
                f"image_width = {spec.image_width or 0}",
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
    elif spec.family == "transformer_encoder":
        lines.extend(
            [
                f"model_dim = {spec.model_dim or 0}",
                f"num_heads = {spec.num_heads or 0}",
                f"num_layers = {spec.num_layers or 0}",
                f"ffn_dim = {spec.ffn_dim or 0}",
            ]
        )
    else:
        lines.append(f"channels = {json.dumps(spec.conv_channels)}")
        if spec.family == "vision_resnet":
            lines.append(f"num_layers = {spec.num_layers or 0}")
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


def write_edited_architecture_spec(
    spec_path: Path,
    assignments: list[str],
    *,
    output_path: Path | None = None,
    inplace: bool = False,
) -> dict[str, Any]:
    from machinator.modeling_weights import diff_architecture_specs

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
