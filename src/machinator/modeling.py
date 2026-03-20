from __future__ import annotations

# Compatibility facade for the older `machinator.modeling` import path.
# The real responsibilities now live in the focused modules below:
# - `modeling_types`: architecture/method/data dataclasses and constants
# - `modeling_specs`: spec loading, validation, rendering, editing
# - `modeling_collation`: dataset facts + recipe-driven spec creation
# - `modeling_weights`: parameter manifests, diffs, migration plans/checkpoints
# - `modeling_compile`: deterministic Python module generation + runtime prep

from machinator.modeling_collation import (
    architecture_spec_from_dataset_facts,
    dataset_facts_from_report_path,
    default_training_spec,
    render_dataset_facts_toml,
)
from machinator.modeling_compile import (
    compile_architecture_spec,
    load_compiled_model_class,
    prepare_training_runtime,
    render_compiled_model_python,
    resolve_pipeline_spec_paths,
)
from machinator.modeling_rust import (
    run_rust_ir_cli,
    rust_diff_spec_files,
    rust_ir_available,
    rust_ir_manifest_path,
    rust_migration_plan_spec_files,
    rust_validate_spec_file,
)
from machinator.modeling_specs import (
    EDITABLE_SPEC_FIELDS,
    edit_architecture_spec,
    load_architecture_spec,
    load_training_spec,
    parameter_count,
    render_model_spec_toml,
    render_training_spec_toml,
    validate_architecture_spec,
    validate_spec_file,
    validate_training_spec,
    write_edited_architecture_spec,
)
from machinator.modeling_types import (
    ArchitectureSpec,
    DatasetFacts,
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
from machinator.modeling_weights import (
    build_migration_plan,
    build_param_store_manifest,
    diff_architecture_specs,
    diff_spec_files,
    migrate_checkpoint,
    migration_plan_spec_files,
)

__all__ = [
    "ArchitectureSpec",
    "DatasetFacts",
    "EDITABLE_SPEC_FIELDS",
    "ModelSpecError",
    "TrainingSpec",
    "VALID_ACTIVATIONS",
    "VALID_INPUT_KINDS",
    "VALID_LOSSES",
    "VALID_MODEL_FAMILIES",
    "VALID_MODALITIES",
    "VALID_NORMALIZATIONS",
    "VALID_POOLING",
    "VALID_TARGET_KINDS",
    "VALID_TASKS",
    "architecture_spec_from_dataset_facts",
    "build_migration_plan",
    "build_param_store_manifest",
    "compile_architecture_spec",
    "dataset_facts_from_report_path",
    "default_training_spec",
    "diff_architecture_specs",
    "diff_spec_files",
    "edit_architecture_spec",
    "load_architecture_spec",
    "load_compiled_model_class",
    "load_training_spec",
    "migrate_checkpoint",
    "migration_plan_spec_files",
    "parameter_count",
    "prepare_training_runtime",
    "render_compiled_model_python",
    "render_dataset_facts_toml",
    "render_model_spec_toml",
    "render_training_spec_toml",
    "resolve_pipeline_spec_paths",
    "run_rust_ir_cli",
    "rust_diff_spec_files",
    "rust_ir_available",
    "rust_ir_manifest_path",
    "rust_migration_plan_spec_files",
    "rust_validate_spec_file",
    "validate_architecture_spec",
    "validate_spec_file",
    "validate_training_spec",
    "write_edited_architecture_spec",
]
