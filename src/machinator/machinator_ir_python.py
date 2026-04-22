from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


# This module is a readable Python mirror of `rust/machinator-ir/src/lib.rs`.
# It is not the primary runtime path; it exists so the IR can be inspected
# without hopping between several Python runtime helpers and the Rust crate.

VALID_PARAM_STORE_FORMAT = "safetensors"
VALID_FAMILIES = {"tabular_mlp", "transformer_encoder", "vision_cnn", "vision_resnet"}
VALID_MODALITIES = {"tabular", "text", "vision"}
VALID_INPUT_KINDS = {"dense_features", "token_ids", "image_tensor"}
VALID_ACTIVATIONS = {"relu", "gelu", "silu", "tanh"}
VALID_NORMALIZATIONS = {"none", "batchnorm", "layernorm"}


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    family: str
    task: str
    modality: str


@dataclass(frozen=True)
class InputSpec:
    kind: str
    feature_names: list[str]
    feature_count: int
    token_vocab_size: int | None
    max_sequence_length: int | None
    image_channels: int | None
    image_height: int | None
    image_width: int | None


@dataclass(frozen=True)
class TargetSpec:
    column: str
    kind: str


@dataclass(frozen=True)
class BackboneSpec:
    hidden_dims: list[int]
    channels: list[int]
    model_dim: int | None
    num_heads: int | None
    num_layers: int | None
    ffn_dim: int | None
    activation: str
    normalization: str
    dropout: float


@dataclass(frozen=True)
class HeadSpec:
    output_dim: int
    pooling: str | None


@dataclass(frozen=True)
class ForwardProcedure:
    input: str
    output: str


@dataclass(frozen=True)
class LossProcedure:
    kind: str
    prediction: str
    target: str


@dataclass(frozen=True)
class ProcedureSpec:
    forward: ForwardProcedure
    loss: LossProcedure


@dataclass(frozen=True)
class ParamStoreSpec:
    format: str
    root_key: str


@dataclass(frozen=True)
class ParameterBinding:
    tensor_key: str
    owner_id: str
    shape: list[int]
    dtype: str


@dataclass(frozen=True)
class ParamStoreManifest:
    schema_version: int
    format: str
    root_key: str
    parameter_count: int
    parameters: list[ParameterBinding]


@dataclass(frozen=True)
class DiffChange:
    field: str
    old_value: Any
    new_value: Any


@dataclass(frozen=True)
class MigrationAction:
    action: str
    target_tensor_key: str
    source_tensor_key: str | None
    source_shape: list[int] | None
    target_shape: list[int]
    overlap_shape: list[int] | None
    reason: str


@dataclass(frozen=True)
class MigrationPlan:
    exact_copy_count: int
    partial_copy_count: int
    reinitialize_count: int
    actions: list[MigrationAction]


@dataclass(frozen=True)
class ModelDiff:
    compatible: bool
    old_family: str
    new_family: str
    parameter_delta: int
    changes: list[DiffChange]
    migration_plan: MigrationPlan


def _require_section(payload: dict[str, Any], name: str) -> dict[str, Any]:
    section = payload.get(name)
    if not isinstance(section, dict):
        raise ValidationError(f"missing a valid [{name}] section")
    return section


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _int_list(values: Any) -> list[int]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValidationError("expected a list of integers")
    return [int(value) for value in values]


def _string_list(values: Any) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValidationError("expected a list of strings")
    return [str(value) for value in values]


def _track_change(changes: list[DiffChange], field: str, old_value: Any, new_value: Any) -> None:
    if old_value != new_value:
        changes.append(DiffChange(field=field, old_value=old_value, new_value=new_value))


@dataclass(frozen=True)
class ArchitectureSpec:
    model: ModelMetadata
    input: InputSpec
    target: TargetSpec
    backbone: BackboneSpec
    head: HeadSpec
    procedures: ProcedureSpec
    param_store: ParamStoreSpec

    @classmethod
    def from_toml_str(cls, text: str) -> "ArchitectureSpec":
        return cls.from_payload(tomllib.loads(text))

    @classmethod
    def from_toml_file(cls, path: Path) -> "ArchitectureSpec":
        return cls.from_toml_str(path.read_text())

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ArchitectureSpec":
        model = _require_section(payload, "model")
        input_section = _require_section(payload, "input")
        target = _require_section(payload, "target")
        backbone = _require_section(payload, "backbone")
        head = _require_section(payload, "head")
        procedures = _require_section(payload, "procedures")
        forward = _require_section(procedures, "forward")
        loss = _require_section(procedures, "loss")
        param_store = _require_section(payload, "param_store")

        return cls(
            model=ModelMetadata(
                name=str(model.get("name", "")).strip(),
                family=str(model.get("family", "")).strip(),
                task=str(model.get("task", "")).strip(),
                modality=str(model.get("modality", "")).strip(),
            ),
            input=InputSpec(
                kind=str(input_section.get("kind", "")).strip(),
                feature_names=_string_list(input_section.get("feature_names")),
                feature_count=int(input_section.get("feature_count", 0)),
                token_vocab_size=_optional_int(input_section.get("token_vocab_size")),
                max_sequence_length=_optional_int(input_section.get("max_sequence_length")),
                image_channels=_optional_int(input_section.get("image_channels")),
                image_height=_optional_int(input_section.get("image_height")),
                image_width=_optional_int(input_section.get("image_width")),
            ),
            target=TargetSpec(
                column=str(target.get("column", "")).strip(),
                kind=str(target.get("kind", "")).strip(),
            ),
            backbone=BackboneSpec(
                hidden_dims=_int_list(backbone.get("hidden_dims")),
                channels=_int_list(backbone.get("channels")),
                model_dim=_optional_int(backbone.get("model_dim")),
                num_heads=_optional_int(backbone.get("num_heads")),
                num_layers=_optional_int(backbone.get("num_layers")),
                ffn_dim=_optional_int(backbone.get("ffn_dim")),
                activation=str(backbone.get("activation", "")).strip(),
                normalization=str(backbone.get("normalization", "")).strip(),
                dropout=float(backbone.get("dropout", 0.0)),
            ),
            head=HeadSpec(
                output_dim=int(head.get("output_dim", 0)),
                pooling=str(head.get("pooling")).strip() if head.get("pooling") is not None else None,
            ),
            procedures=ProcedureSpec(
                forward=ForwardProcedure(
                    input=str(forward.get("input", "")).strip(),
                    output=str(forward.get("output", "")).strip(),
                ),
                loss=LossProcedure(
                    kind=str(loss.get("kind", "")).strip(),
                    prediction=str(loss.get("prediction", "")).strip(),
                    target=str(loss.get("target", "")).strip(),
                ),
            ),
            param_store=ParamStoreSpec(
                format=str(param_store.get("format", "")).strip(),
                root_key=str(param_store.get("root_key", "")).strip(),
            ),
        )

    def validate(self) -> None:
        if self.model.family not in VALID_FAMILIES:
            raise ValidationError(f"unsupported model family `{self.model.family}`")
        if self.model.modality not in VALID_MODALITIES:
            raise ValidationError(f"unsupported modality `{self.model.modality}`")
        if self.input.kind not in VALID_INPUT_KINDS:
            raise ValidationError(f"unsupported input kind `{self.input.kind}`")
        if self.model.task != "binary_classification":
            raise ValidationError(f"unsupported task `{self.model.task}`")
        if self.target.kind != "binary":
            raise ValidationError(f"unsupported target kind `{self.target.kind}`")
        if not self.target.column.strip():
            raise ValidationError("target.column is required")
        if self.backbone.activation not in VALID_ACTIVATIONS:
            raise ValidationError(f"unsupported activation `{self.backbone.activation}`")
        if self.backbone.normalization not in VALID_NORMALIZATIONS:
            raise ValidationError(f"unsupported normalization `{self.backbone.normalization}`")
        if not (0.0 <= self.backbone.dropout < 1.0):
            raise ValidationError("backbone.dropout must be in the range [0.0, 1.0)")
        if self.head.output_dim != 1:
            raise ValidationError("head.output_dim must be 1 for binary classification")
        if self.procedures.forward.input != "features":
            raise ValidationError("procedures.forward.input must be `features`")
        if self.procedures.forward.output != "logits":
            raise ValidationError("procedures.forward.output must be `logits`")
        if self.procedures.loss.kind != "bce_with_logits":
            raise ValidationError(f"unsupported loss `{self.procedures.loss.kind}`")
        if self.procedures.loss.prediction != "logits":
            raise ValidationError("procedures.loss.prediction must be `logits`")
        if self.procedures.loss.target != "target":
            raise ValidationError("procedures.loss.target must be `target`")
        if self.param_store.format != VALID_PARAM_STORE_FORMAT:
            raise ValidationError("param_store.format must be `safetensors`")
        if not self.param_store.root_key.strip():
            raise ValidationError("param_store.root_key is required")

        if self.model.family == "tabular_mlp":
            if self.model.modality != "tabular":
                raise ValidationError("tabular_mlp requires modality `tabular`")
            if self.input.kind != "dense_features":
                raise ValidationError("tabular_mlp requires input.kind `dense_features`")
            if self.input.feature_count <= 0:
                raise ValidationError("input.feature_count must be positive for tabular_mlp")
            if self.input.feature_names and len(self.input.feature_names) != self.input.feature_count:
                raise ValidationError("input.feature_names must match input.feature_count")
            if not self.backbone.hidden_dims or any(value <= 0 for value in self.backbone.hidden_dims):
                raise ValidationError(
                    "backbone.hidden_dims must contain one or more positive integers"
                )
            return

        if self.model.family == "transformer_encoder":
            if self.model.modality != "text":
                raise ValidationError("transformer_encoder requires modality `text`")
            if self.input.kind != "token_ids":
                raise ValidationError("transformer_encoder requires input.kind `token_ids`")
            if (self.input.token_vocab_size or 0) <= 0:
                raise ValidationError(
                    "input.token_vocab_size must be positive for transformer_encoder"
                )
            if (self.input.max_sequence_length or 0) <= 0:
                raise ValidationError(
                    "input.max_sequence_length must be positive for transformer_encoder"
                )
            if (self.backbone.model_dim or 0) <= 0:
                raise ValidationError("backbone.model_dim must be positive for transformer_encoder")
            if (self.backbone.num_heads or 0) <= 0:
                raise ValidationError("backbone.num_heads must be positive for transformer_encoder")
            if (self.backbone.num_layers or 0) <= 0:
                raise ValidationError("backbone.num_layers must be positive for transformer_encoder")
            if (self.backbone.ffn_dim or 0) <= 0:
                raise ValidationError("backbone.ffn_dim must be positive for transformer_encoder")
            if self.backbone.model_dim % self.backbone.num_heads != 0:
                raise ValidationError(
                    "backbone.model_dim must be divisible by backbone.num_heads"
                )
            pooling = self.head.pooling or "mean"
            if pooling not in {"mean", "cls"}:
                raise ValidationError(f"unsupported pooling `{pooling}` for transformer_encoder")
            return

        if self.model.family == "vision_cnn":
            if self.model.modality != "vision":
                raise ValidationError("vision_cnn requires modality `vision`")
            if self.input.kind != "image_tensor":
                raise ValidationError("vision_cnn requires input.kind `image_tensor`")
            if (self.input.image_channels or 0) <= 0:
                raise ValidationError("input.image_channels must be positive for vision_cnn")
            if (self.input.image_height or 0) <= 0:
                raise ValidationError("input.image_height must be positive for vision_cnn")
            if (self.input.image_width or 0) <= 0:
                raise ValidationError("input.image_width must be positive for vision_cnn")
            if not self.backbone.channels or any(value <= 0 for value in self.backbone.channels):
                raise ValidationError(
                    "backbone.channels must contain one or more positive integers for vision_cnn"
                )
            if self.backbone.normalization not in {"none", "batchnorm"}:
                raise ValidationError(
                    "vision_cnn supports only `none` or `batchnorm` normalization"
                )
            if (self.head.pooling or "avg") != "avg":
                raise ValidationError("vision_cnn currently supports only `avg` pooling")
            return

        if self.model.family == "vision_resnet":
            if self.model.modality != "vision":
                raise ValidationError("vision_resnet requires modality `vision`")
            if self.input.kind != "image_tensor":
                raise ValidationError("vision_resnet requires input.kind `image_tensor`")
            if (self.input.image_channels or 0) <= 0:
                raise ValidationError("input.image_channels must be positive for vision_resnet")
            if (self.input.image_height or 0) <= 0:
                raise ValidationError("input.image_height must be positive for vision_resnet")
            if (self.input.image_width or 0) <= 0:
                raise ValidationError("input.image_width must be positive for vision_resnet")
            if not self.backbone.channels or any(value <= 0 for value in self.backbone.channels):
                raise ValidationError(
                    "backbone.channels must contain one or more positive integers for vision_resnet"
                )
            if (self.backbone.num_layers or 0) <= 0:
                raise ValidationError("backbone.num_layers must be positive for vision_resnet")
            if self.backbone.normalization not in {"none", "batchnorm"}:
                raise ValidationError(
                    "vision_resnet supports only `none` or `batchnorm` normalization"
                )
            if (self.head.pooling or "avg") != "avg":
                raise ValidationError("vision_resnet currently supports only `avg` pooling")

    def parameter_count(self) -> int:
        if self.model.family == "tabular_mlp":
            total = 0
            input_dim = self.input.feature_count
            for hidden_dim in self.backbone.hidden_dims:
                total += input_dim * hidden_dim
                total += hidden_dim
                if self.backbone.normalization != "none":
                    total += hidden_dim * 2
                input_dim = hidden_dim
            total += input_dim * self.head.output_dim
            total += self.head.output_dim
            return total

        if self.model.family == "transformer_encoder":
            token_vocab_size = self.input.token_vocab_size or 0
            max_sequence_length = self.input.max_sequence_length or 0
            model_dim = self.backbone.model_dim or 0
            ffn_dim = self.backbone.ffn_dim or 0
            num_layers = self.backbone.num_layers or 0

            total = token_vocab_size * model_dim
            total += max_sequence_length * model_dim
            for _ in range(num_layers):
                total += 3 * model_dim * model_dim
                total += 3 * model_dim
                total += model_dim * model_dim
                total += model_dim
                total += model_dim * ffn_dim
                total += ffn_dim
                total += ffn_dim * model_dim
                total += model_dim
                total += model_dim * 4
            total += model_dim * self.head.output_dim
            total += self.head.output_dim
            return total

        if self.model.family == "vision_cnn":
            total = 0
            in_channels = self.input.image_channels or 0
            for out_channels in self.backbone.channels:
                total += out_channels * in_channels * 3 * 3
                total += out_channels
                if self.backbone.normalization != "none":
                    total += out_channels * 2
                in_channels = out_channels
            total += in_channels * self.head.output_dim
            total += self.head.output_dim
            return total

        if self.model.family == "vision_resnet":
            total = 0
            in_channels = self.input.image_channels or 0
            blocks_per_stage = self.backbone.num_layers or 0
            stem_channels = self.backbone.channels[0] if self.backbone.channels else 0
            total += stem_channels * in_channels * 3 * 3
            total += stem_channels
            if self.backbone.normalization != "none":
                total += stem_channels * 2
            in_channels = stem_channels
            for stage_channels in self.backbone.channels:
                for _ in range(blocks_per_stage):
                    projection = in_channels != stage_channels
                    total += stage_channels * in_channels * 3 * 3
                    total += stage_channels
                    total += stage_channels * stage_channels * 3 * 3
                    total += stage_channels
                    if self.backbone.normalization != "none":
                        total += stage_channels * 4
                    if projection:
                        total += stage_channels * in_channels
                        total += stage_channels
                        if self.backbone.normalization != "none":
                            total += stage_channels * 2
                    in_channels = stage_channels
            total += in_channels * self.head.output_dim
            total += self.head.output_dim
            return total

        return 0

    def param_store_manifest(self) -> ParamStoreManifest:
        parameters: list[ParameterBinding] = []

        if self.model.family == "tabular_mlp":
            input_dim = self.input.feature_count
            for layer_index, hidden_dim in enumerate(self.backbone.hidden_dims):
                parameters.append(
                    ParameterBinding(
                        tensor_key=f"layers.{layer_index}.weight",
                        owner_id=f"backbone.layer.{layer_index}",
                        shape=[hidden_dim, input_dim],
                        dtype="float32",
                    )
                )
                parameters.append(
                    ParameterBinding(
                        tensor_key=f"layers.{layer_index}.bias",
                        owner_id=f"backbone.layer.{layer_index}",
                        shape=[hidden_dim],
                        dtype="float32",
                    )
                )
                if self.backbone.normalization != "none":
                    parameters.append(
                        ParameterBinding(
                            tensor_key=f"norms.{layer_index}.weight",
                            owner_id=f"backbone.norm.{layer_index}",
                            shape=[hidden_dim],
                            dtype="float32",
                        )
                    )
                    parameters.append(
                        ParameterBinding(
                            tensor_key=f"norms.{layer_index}.bias",
                            owner_id=f"backbone.norm.{layer_index}",
                            shape=[hidden_dim],
                            dtype="float32",
                        )
                    )
                input_dim = hidden_dim
            parameters.append(
                ParameterBinding(
                    tensor_key="head.weight",
                    owner_id="head",
                    shape=[self.head.output_dim, input_dim],
                    dtype="float32",
                )
            )
            parameters.append(
                ParameterBinding(
                    tensor_key="head.bias",
                    owner_id="head",
                    shape=[self.head.output_dim],
                    dtype="float32",
                )
            )

        if self.model.family == "transformer_encoder":
            token_vocab_size = self.input.token_vocab_size or 0
            max_sequence_length = self.input.max_sequence_length or 0
            model_dim = self.backbone.model_dim or 0
            ffn_dim = self.backbone.ffn_dim or 0
            num_layers = self.backbone.num_layers or 0

            parameters.extend(
                [
                    ParameterBinding(
                        tensor_key="token_embedding.weight",
                        owner_id="token_embedding",
                        shape=[token_vocab_size, model_dim],
                        dtype="float32",
                    ),
                    ParameterBinding(
                        tensor_key="position_embedding.weight",
                        owner_id="position_embedding",
                        shape=[max_sequence_length, model_dim],
                        dtype="float32",
                    ),
                ]
            )
            for layer_index in range(num_layers):
                prefix = f"encoder.layers.{layer_index}"
                owner_prefix = f"encoder.layer.{layer_index}"
                parameters.extend(
                    [
                        ParameterBinding(
                            tensor_key=f"{prefix}.self_attn.in_proj_weight",
                            owner_id=f"{owner_prefix}.self_attn",
                            shape=[model_dim * 3, model_dim],
                            dtype="float32",
                        ),
                        ParameterBinding(
                            tensor_key=f"{prefix}.self_attn.in_proj_bias",
                            owner_id=f"{owner_prefix}.self_attn",
                            shape=[model_dim * 3],
                            dtype="float32",
                        ),
                        ParameterBinding(
                            tensor_key=f"{prefix}.self_attn.out_proj.weight",
                            owner_id=f"{owner_prefix}.self_attn.out_proj",
                            shape=[model_dim, model_dim],
                            dtype="float32",
                        ),
                        ParameterBinding(
                            tensor_key=f"{prefix}.self_attn.out_proj.bias",
                            owner_id=f"{owner_prefix}.self_attn.out_proj",
                            shape=[model_dim],
                            dtype="float32",
                        ),
                        ParameterBinding(
                            tensor_key=f"{prefix}.linear1.weight",
                            owner_id=f"{owner_prefix}.linear1",
                            shape=[ffn_dim, model_dim],
                            dtype="float32",
                        ),
                        ParameterBinding(
                            tensor_key=f"{prefix}.linear1.bias",
                            owner_id=f"{owner_prefix}.linear1",
                            shape=[ffn_dim],
                            dtype="float32",
                        ),
                        ParameterBinding(
                            tensor_key=f"{prefix}.linear2.weight",
                            owner_id=f"{owner_prefix}.linear2",
                            shape=[model_dim, ffn_dim],
                            dtype="float32",
                        ),
                        ParameterBinding(
                            tensor_key=f"{prefix}.linear2.bias",
                            owner_id=f"{owner_prefix}.linear2",
                            shape=[model_dim],
                            dtype="float32",
                        ),
                        ParameterBinding(
                            tensor_key=f"{prefix}.norm1.weight",
                            owner_id=f"{owner_prefix}.norm1",
                            shape=[model_dim],
                            dtype="float32",
                        ),
                        ParameterBinding(
                            tensor_key=f"{prefix}.norm1.bias",
                            owner_id=f"{owner_prefix}.norm1",
                            shape=[model_dim],
                            dtype="float32",
                        ),
                        ParameterBinding(
                            tensor_key=f"{prefix}.norm2.weight",
                            owner_id=f"{owner_prefix}.norm2",
                            shape=[model_dim],
                            dtype="float32",
                        ),
                        ParameterBinding(
                            tensor_key=f"{prefix}.norm2.bias",
                            owner_id=f"{owner_prefix}.norm2",
                            shape=[model_dim],
                            dtype="float32",
                        ),
                    ]
                )
            parameters.append(
                ParameterBinding(
                    tensor_key="head.weight",
                    owner_id="head",
                    shape=[self.head.output_dim, model_dim],
                    dtype="float32",
                )
            )
            parameters.append(
                ParameterBinding(
                    tensor_key="head.bias",
                    owner_id="head",
                    shape=[self.head.output_dim],
                    dtype="float32",
                )
            )

        if self.model.family == "vision_cnn":
            in_channels = self.input.image_channels or 0
            for layer_index, out_channels in enumerate(self.backbone.channels):
                parameters.append(
                    ParameterBinding(
                        tensor_key=f"convs.{layer_index}.weight",
                        owner_id=f"backbone.conv.{layer_index}",
                        shape=[out_channels, in_channels, 3, 3],
                        dtype="float32",
                    )
                )
                parameters.append(
                    ParameterBinding(
                        tensor_key=f"convs.{layer_index}.bias",
                        owner_id=f"backbone.conv.{layer_index}",
                        shape=[out_channels],
                        dtype="float32",
                    )
                )
                if self.backbone.normalization != "none":
                    parameters.append(
                        ParameterBinding(
                            tensor_key=f"norms.{layer_index}.weight",
                            owner_id=f"backbone.norm.{layer_index}",
                            shape=[out_channels],
                            dtype="float32",
                        )
                    )
                    parameters.append(
                        ParameterBinding(
                            tensor_key=f"norms.{layer_index}.bias",
                            owner_id=f"backbone.norm.{layer_index}",
                            shape=[out_channels],
                            dtype="float32",
                        )
                    )
                in_channels = out_channels
            parameters.append(
                ParameterBinding(
                    tensor_key="head.weight",
                    owner_id="head",
                    shape=[self.head.output_dim, in_channels],
                    dtype="float32",
                )
            )
            parameters.append(
                ParameterBinding(
                    tensor_key="head.bias",
                    owner_id="head",
                    shape=[self.head.output_dim],
                    dtype="float32",
                )
            )

        if self.model.family == "vision_resnet":
            in_channels = self.input.image_channels or 0
            stem_channels = self.backbone.channels[0] if self.backbone.channels else 0
            blocks_per_stage = self.backbone.num_layers or 0
            parameters.append(
                ParameterBinding(
                    tensor_key="stem.weight",
                    owner_id="stem",
                    shape=[stem_channels, in_channels, 3, 3],
                    dtype="float32",
                )
            )
            parameters.append(
                ParameterBinding(
                    tensor_key="stem.bias",
                    owner_id="stem",
                    shape=[stem_channels],
                    dtype="float32",
                )
            )
            if self.backbone.normalization != "none":
                parameters.extend(
                    [
                        ParameterBinding(
                            tensor_key="stem_norm.weight",
                            owner_id="stem_norm",
                            shape=[stem_channels],
                            dtype="float32",
                        ),
                        ParameterBinding(
                            tensor_key="stem_norm.bias",
                            owner_id="stem_norm",
                            shape=[stem_channels],
                            dtype="float32",
                        ),
                    ]
                )
            in_channels = stem_channels
            for stage_index, stage_channels in enumerate(self.backbone.channels):
                for block_index in range(blocks_per_stage):
                    prefix = f"stages.{stage_index}.{block_index}"
                    owner_prefix = f"backbone.stage.{stage_index}.block.{block_index}"
                    parameters.extend(
                        [
                            ParameterBinding(
                                tensor_key=f"{prefix}.conv1.weight",
                                owner_id=f"{owner_prefix}.conv1",
                                shape=[stage_channels, in_channels, 3, 3],
                                dtype="float32",
                            ),
                            ParameterBinding(
                                tensor_key=f"{prefix}.conv1.bias",
                                owner_id=f"{owner_prefix}.conv1",
                                shape=[stage_channels],
                                dtype="float32",
                            ),
                            ParameterBinding(
                                tensor_key=f"{prefix}.conv2.weight",
                                owner_id=f"{owner_prefix}.conv2",
                                shape=[stage_channels, stage_channels, 3, 3],
                                dtype="float32",
                            ),
                            ParameterBinding(
                                tensor_key=f"{prefix}.conv2.bias",
                                owner_id=f"{owner_prefix}.conv2",
                                shape=[stage_channels],
                                dtype="float32",
                            ),
                        ]
                    )
                    if self.backbone.normalization != "none":
                        parameters.extend(
                            [
                                ParameterBinding(
                                    tensor_key=f"{prefix}.norm1.weight",
                                    owner_id=f"{owner_prefix}.norm1",
                                    shape=[stage_channels],
                                    dtype="float32",
                                ),
                                ParameterBinding(
                                    tensor_key=f"{prefix}.norm1.bias",
                                    owner_id=f"{owner_prefix}.norm1",
                                    shape=[stage_channels],
                                    dtype="float32",
                                ),
                                ParameterBinding(
                                    tensor_key=f"{prefix}.norm2.weight",
                                    owner_id=f"{owner_prefix}.norm2",
                                    shape=[stage_channels],
                                    dtype="float32",
                                ),
                                ParameterBinding(
                                    tensor_key=f"{prefix}.norm2.bias",
                                    owner_id=f"{owner_prefix}.norm2",
                                    shape=[stage_channels],
                                    dtype="float32",
                                ),
                            ]
                        )
                    if in_channels != stage_channels:
                        parameters.extend(
                            [
                                ParameterBinding(
                                    tensor_key=f"{prefix}.proj.weight",
                                    owner_id=f"{owner_prefix}.proj",
                                    shape=[stage_channels, in_channels, 1, 1],
                                    dtype="float32",
                                ),
                                ParameterBinding(
                                    tensor_key=f"{prefix}.proj.bias",
                                    owner_id=f"{owner_prefix}.proj",
                                    shape=[stage_channels],
                                    dtype="float32",
                                ),
                            ]
                        )
                        if self.backbone.normalization != "none":
                            parameters.extend(
                                [
                                    ParameterBinding(
                                        tensor_key=f"{prefix}.proj_norm.weight",
                                        owner_id=f"{owner_prefix}.proj_norm",
                                        shape=[stage_channels],
                                        dtype="float32",
                                    ),
                                    ParameterBinding(
                                        tensor_key=f"{prefix}.proj_norm.bias",
                                        owner_id=f"{owner_prefix}.proj_norm",
                                        shape=[stage_channels],
                                        dtype="float32",
                                    ),
                                ]
                            )
                    in_channels = stage_channels
            parameters.append(
                ParameterBinding(
                    tensor_key="head.weight",
                    owner_id="head",
                    shape=[self.head.output_dim, in_channels],
                    dtype="float32",
                )
            )
            parameters.append(
                ParameterBinding(
                    tensor_key="head.bias",
                    owner_id="head",
                    shape=[self.head.output_dim],
                    dtype="float32",
                )
            )

        return ParamStoreManifest(
            schema_version=1,
            format=self.param_store.format,
            root_key=self.param_store.root_key,
            parameter_count=self.parameter_count(),
            parameters=parameters,
        )

    def diff(self, other: "ArchitectureSpec") -> ModelDiff:
        changes: list[DiffChange] = []
        _track_change(changes, "family", self.model.family, other.model.family)
        _track_change(changes, "task", self.model.task, other.model.task)
        _track_change(changes, "modality", self.model.modality, other.model.modality)
        _track_change(changes, "input_kind", self.input.kind, other.input.kind)
        _track_change(changes, "feature_count", self.input.feature_count, other.input.feature_count)
        _track_change(changes, "feature_names", self.input.feature_names, other.input.feature_names)
        _track_change(
            changes,
            "token_vocab_size",
            self.input.token_vocab_size,
            other.input.token_vocab_size,
        )
        _track_change(
            changes,
            "max_sequence_length",
            self.input.max_sequence_length,
            other.input.max_sequence_length,
        )
        _track_change(changes, "image_channels", self.input.image_channels, other.input.image_channels)
        _track_change(changes, "image_height", self.input.image_height, other.input.image_height)
        _track_change(changes, "image_width", self.input.image_width, other.input.image_width)
        _track_change(changes, "target_column", self.target.column, other.target.column)
        _track_change(changes, "hidden_dims", self.backbone.hidden_dims, other.backbone.hidden_dims)
        _track_change(changes, "conv_channels", self.backbone.channels, other.backbone.channels)
        _track_change(changes, "model_dim", self.backbone.model_dim, other.backbone.model_dim)
        _track_change(changes, "num_heads", self.backbone.num_heads, other.backbone.num_heads)
        _track_change(changes, "num_layers", self.backbone.num_layers, other.backbone.num_layers)
        _track_change(changes, "ffn_dim", self.backbone.ffn_dim, other.backbone.ffn_dim)
        _track_change(changes, "activation", self.backbone.activation, other.backbone.activation)
        _track_change(
            changes,
            "normalization",
            self.backbone.normalization,
            other.backbone.normalization,
        )
        _track_change(changes, "dropout", self.backbone.dropout, other.backbone.dropout)
        _track_change(changes, "pooling", self.head.pooling, other.head.pooling)

        return ModelDiff(
            compatible=(
                self.model.task == other.model.task
                and self.model.modality == other.model.modality
                and self.target.kind == other.target.kind
            ),
            old_family=self.model.family,
            new_family=other.model.family,
            parameter_delta=other.parameter_count() - self.parameter_count(),
            changes=changes,
            migration_plan=build_migration_plan(self, other),
        )


def build_migration_plan(old_spec: ArchitectureSpec, new_spec: ArchitectureSpec) -> MigrationPlan:
    old_manifest = old_spec.param_store_manifest()
    new_manifest = new_spec.param_store_manifest()
    old_map = {binding.tensor_key: binding for binding in old_manifest.parameters}

    exact_copy_count = 0
    partial_copy_count = 0
    reinitialize_count = 0
    actions: list[MigrationAction] = []

    for new_param in new_manifest.parameters:
        old_param = old_map.get(new_param.tensor_key)
        if old_param is None:
            reinitialize_count += 1
            actions.append(
                MigrationAction(
                    action="reinitialize",
                    target_tensor_key=new_param.tensor_key,
                    source_tensor_key=None,
                    source_shape=None,
                    target_shape=new_param.shape,
                    overlap_shape=None,
                    reason="tensor key is new in the updated spec",
                )
            )
            continue

        if old_param.shape == new_param.shape:
            exact_copy_count += 1
            actions.append(
                MigrationAction(
                    action="exact_copy",
                    target_tensor_key=new_param.tensor_key,
                    source_tensor_key=old_param.tensor_key,
                    source_shape=old_param.shape,
                    target_shape=new_param.shape,
                    overlap_shape=new_param.shape,
                    reason="matching tensor key and shape",
                )
            )
            continue

        if len(old_param.shape) == len(new_param.shape):
            overlap_shape = [min(old_dim, new_dim) for old_dim, new_dim in zip(old_param.shape, new_param.shape)]
            if all(value > 0 for value in overlap_shape):
                partial_copy_count += 1
                actions.append(
                    MigrationAction(
                        action="partial_copy",
                        target_tensor_key=new_param.tensor_key,
                        source_tensor_key=old_param.tensor_key,
                        source_shape=old_param.shape,
                        target_shape=new_param.shape,
                        overlap_shape=overlap_shape,
                        reason="tensor key matches but shape changed; preserve the overlapping slice",
                    )
                )
                continue

        reinitialize_count += 1
        actions.append(
            MigrationAction(
                action="reinitialize",
                target_tensor_key=new_param.tensor_key,
                source_tensor_key=old_param.tensor_key,
                source_shape=old_param.shape,
                target_shape=new_param.shape,
                overlap_shape=None,
                reason="tensor rank changed or there is no meaningful overlap to preserve",
            )
        )

    return MigrationPlan(
        exact_copy_count=exact_copy_count,
        partial_copy_count=partial_copy_count,
        reinitialize_count=reinitialize_count,
        actions=actions,
    )
