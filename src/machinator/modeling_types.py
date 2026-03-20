from __future__ import annotations

from dataclasses import dataclass, field
import re
from pathlib import Path

# These constants describe the currently supported surface. Keeping them here
# makes the allowed architecture/method vocabulary easy to inspect in one file.
VALID_MODEL_FAMILIES = {"tabular_mlp", "transformer_encoder", "vision_cnn", "vision_resnet"}
VALID_MODALITIES = {"tabular", "text", "vision"}
VALID_INPUT_KINDS = {"dense_features", "token_ids", "image_tensor"}
VALID_TASKS = {"binary_classification"}
VALID_TARGET_KINDS = {"binary"}
VALID_ACTIVATIONS = {"relu", "gelu", "silu", "tanh"}
VALID_NORMALIZATIONS = {"none", "batchnorm", "layernorm"}
VALID_LOSSES = {"bce_with_logits"}
VALID_POOLING = {"mean", "cls", "avg"}


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
    image_channels: int | None = None
    image_height: int | None = None
    image_width: int | None = None
    class_names: list[str] = field(default_factory=list)


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
    # This is still the Python runtime view of the model specification. The
    # longer-term goal is for Rust to be the canonical authoring/validation
    # layer and for Python to consume that validated spec.
    name: str
    family: str
    task: str
    modality: str
    input_kind: str
    feature_names: list[str]
    feature_count: int
    token_vocab_size: int | None
    max_sequence_length: int | None
    image_channels: int | None
    image_height: int | None
    image_width: int | None
    target_column: str
    target_kind: str
    hidden_dims: list[int]
    conv_channels: list[int]
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
