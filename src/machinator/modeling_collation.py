from __future__ import annotations

import json
from pathlib import Path

from machinator.core import clean_optional, load_json, slugify
from machinator.modeling_types import ArchitectureSpec, DatasetFacts, ModelSpecError, TrainingSpec

# This module is the dataset-first planner. It turns delegated report facts into
# normalized `DatasetFacts`, picks defaults, and materializes starter specs.


def _json_string(value: str) -> str:
    return json.dumps(value)


def _json_string_array(values: list[str]) -> str:
    return json.dumps(values)


def _clean_list(values: object) -> list[str]:
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
