from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from typing import Any

from machinator.core import now_utc, slugify, write_json
from machinator.modeling_specs import (
    load_architecture_spec,
    load_training_spec,
    parameter_count,
    validate_architecture_spec,
    validate_training_spec,
)
from machinator.modeling_types import ArchitectureSpec, ModelSpecError
from machinator.modeling_weights import build_param_store_manifest


def _activation_expr(name: str) -> str:
    mapping = {
        "relu": "torch.relu",
        "gelu": "F.gelu",
        "silu": "F.silu",
        "tanh": "torch.tanh",
    }
    return mapping[name]


def render_compiled_model_python(spec: ArchitectureSpec) -> str:
    # This renderer is intentionally dumb and deterministic. The compiled file
    # is a runtime artifact derived from the spec, not a hand-maintained source
    # file that users are expected to edit directly.
    if spec.family == "tabular_mlp":
        norm_factory = {
            "none": "nn.Identity()",
            "batchnorm": "nn.BatchNorm1d(hidden_dim)",
            "layernorm": "nn.LayerNorm(hidden_dim)",
        }[spec.normalization]
        dropout_block = ""
        if spec.dropout > 0.0:
            dropout_block = "            x = F.dropout(x, p=self.dropout, training=self.training)\\n"
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


def load_compiled_model_class(module_path: Path, class_name: str) -> type[Any]:
    module_name = f"machinator_compiled_{slugify(module_path.stem, fallback='compiled')}_{abs(hash(str(module_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ModelSpecError(f"could not load compiled model module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class = getattr(module, class_name, None)
    if model_class is None:
        raise ModelSpecError(f"compiled model class `{class_name}` is missing from {module_path}")
    return model_class


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
