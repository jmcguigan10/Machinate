from __future__ import annotations

import argparse
import json
from pathlib import Path

from machinate.commands.task import resolve_selected_pipeline
from machinate.modeling import (
    compile_architecture_spec,
    diff_spec_files,
    load_architecture_spec,
    migrate_checkpoint,
    validate_spec_file,
    write_edited_architecture_spec,
)
from machinate.ui import can_prompt_interactively, prompt_text


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    model_parser = subparsers.add_parser("model", help="Validate, edit, diff, compile, and migrate Machinate model specs")
    model_subparsers = model_parser.add_subparsers(dest="model_command", required=True)

    validate_parser = model_subparsers.add_parser("validate", help="Validate model.toml for the current pipeline")
    add_pipeline_selection_args(validate_parser)
    validate_parser.add_argument("--spec")
    validate_parser.set_defaults(func=cmd_model_validate)

    compile_parser = model_subparsers.add_parser(
        "compile",
        help="Compile model.toml into a deterministic Python module",
    )
    add_pipeline_selection_args(compile_parser)
    compile_parser.add_argument("--spec")
    compile_parser.add_argument("--output-dir")
    compile_parser.set_defaults(func=cmd_model_compile)

    edit_parser = model_subparsers.add_parser("edit", help="Apply field edits to a model spec")
    add_pipeline_selection_args(edit_parser)
    edit_parser.add_argument("--spec")
    edit_parser.add_argument("--set", action="append", default=[])
    edit_parser.add_argument("--output")
    edit_parser.add_argument("--in-place", action="store_true")
    edit_parser.set_defaults(func=cmd_model_edit)

    diff_parser = model_subparsers.add_parser("diff", help="Diff two model specs and generate a migration plan")
    add_pipeline_selection_args(diff_parser)
    diff_parser.add_argument("--old")
    diff_parser.add_argument("--new")
    diff_parser.add_argument("--json-out")
    diff_parser.set_defaults(func=cmd_model_diff)

    migrate_parser = model_subparsers.add_parser(
        "migrate",
        help="Preserve compatible weights when migrating a checkpoint between specs",
    )
    add_pipeline_selection_args(migrate_parser)
    migrate_parser.add_argument("--old")
    migrate_parser.add_argument("--new")
    migrate_parser.add_argument("--source-state")
    migrate_parser.add_argument("--output-state")
    migrate_parser.add_argument("--plan-out")
    migrate_parser.set_defaults(func=cmd_model_migrate)


def add_pipeline_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace")
    parser.add_argument("--pipeline")
    parser.add_argument("--pipeline-path")


def _prompted_path(label: str, default: Path | None = None) -> Path:
    default_text = str(default) if default is not None else None
    path_text = prompt_text(label, default=default_text)
    return Path(path_text).expanduser().resolve()


def _resolve_current_spec_path(args: argparse.Namespace) -> tuple[Path, Path | None]:
    try:
        pipeline_root, _workspace_root = resolve_selected_pipeline(args)
    except SystemExit:
        if can_prompt_interactively():
            spec_path = _prompted_path("Path to model spec", Path.cwd() / "model.toml")
            if not spec_path.exists():
                raise SystemExit(f"model spec does not exist: {spec_path}")
            return spec_path, spec_path.parent
        raise

    spec_path = pipeline_root / "model.toml"
    if not spec_path.exists():
        raise SystemExit(f"model spec not found at the default pipeline path: {spec_path}")
    return spec_path, pipeline_root


def resolve_spec_path(args: argparse.Namespace, attribute: str = "spec") -> tuple[Path, Path | None]:
    spec_text = getattr(args, attribute, None)
    if spec_text:
        spec_path = Path(spec_text).expanduser().resolve()
        if not spec_path.exists():
            raise SystemExit(f"model spec does not exist: {spec_path}")
        return spec_path, spec_path.parent
    return _resolve_current_spec_path(args)


def resolve_named_spec_path(
    args: argparse.Namespace,
    *,
    attribute: str,
    label: str,
    default_to_current: bool = False,
) -> Path:
    spec_text = getattr(args, attribute, None)
    if spec_text:
        spec_path = Path(spec_text).expanduser().resolve()
        if not spec_path.exists():
            raise SystemExit(f"model spec does not exist: {spec_path}")
        return spec_path

    if default_to_current:
        return resolve_spec_path(args)[0]

    if can_prompt_interactively():
        spec_path = _prompted_path(label)
        if not spec_path.exists():
            raise SystemExit(f"model spec does not exist: {spec_path}")
        return spec_path

    raise SystemExit(f"{attribute.replace('_', '-')} is required")


def cmd_model_validate(args: argparse.Namespace) -> int:
    spec_path, _pipeline_root = resolve_spec_path(args)
    summary = validate_spec_file(spec_path)

    spec = load_architecture_spec(spec_path)
    print("model spec OK")
    print(f"spec: {spec_path}")
    print(f"backend: {summary['backend']}")
    print(f"family: {summary['family']}")
    print(f"task: {summary['task']}")
    print(f"features: {spec.feature_count}")
    print(f"target: {spec.target_column}")
    print(f"estimated_parameters: {summary['estimated_parameters']}")
    return 0


def cmd_model_compile(args: argparse.Namespace) -> int:
    spec_path, pipeline_root = resolve_spec_path(args)
    spec = load_architecture_spec(spec_path)
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    elif pipeline_root is not None:
        output_dir = (pipeline_root / "outputs" / "compiled_model").resolve()
    else:
        output_dir = (spec_path.parent / "outputs" / "compiled_model").resolve()

    artifacts = compile_architecture_spec(spec, output_dir)
    print("model compiled")
    print(f"spec: {spec_path}")
    print(f"module: {artifacts['module_path']}")
    print(f"manifest: {artifacts['manifest_path']}")
    print(f"param store: {artifacts['param_store_manifest_path']}")
    print(f"estimated_parameters: {artifacts['estimated_parameter_count']}")
    return 0


def cmd_model_edit(args: argparse.Namespace) -> int:
    spec_path, _pipeline_root = resolve_spec_path(args)
    if not args.set:
        raise SystemExit("at least one --set KEY=VALUE assignment is required")

    output_path = Path(args.output).expanduser().resolve() if args.output else None
    result = write_edited_architecture_spec(
        spec_path,
        list(args.set),
        output_path=output_path,
        inplace=bool(args.in_place),
    )
    diff_payload = result["diff"]

    print("model spec updated")
    print(f"source: {spec_path}")
    print(f"target: {result['target_path']}")
    if result["backup_path"] is not None:
        print(f"backup: {result['backup_path']}")
    print(f"change_count: {len(diff_payload['changes'])}")
    print(f"parameter_delta: {diff_payload['parameter_delta']}")
    return 0


def cmd_model_diff(args: argparse.Namespace) -> int:
    old_path = resolve_named_spec_path(args, attribute="old", label="Path to old model spec", default_to_current=True)
    new_path = resolve_named_spec_path(args, attribute="new", label="Path to new model spec")
    diff_payload = diff_spec_files(old_path, new_path)

    if args.json_out:
        json_out = Path(args.json_out).expanduser().resolve()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(diff_payload, indent=2) + "\n")
    else:
        json_out = None

    print("model diff ready")
    print(f"old spec: {old_path}")
    print(f"new spec: {new_path}")
    print(f"compatible: {str(bool(diff_payload['compatible'])).lower()}")
    print(f"change_count: {len(diff_payload['changes'])}")
    print(f"parameter_delta: {diff_payload['parameter_delta']}")
    plan = diff_payload["migration_plan"]
    print(f"migration.exact_copy: {plan['exact_copy_count']}")
    print(f"migration.partial_copy: {plan['partial_copy_count']}")
    print(f"migration.reinitialize: {plan['reinitialize_count']}")
    if json_out is not None:
        print(f"json: {json_out}")
    return 0


def cmd_model_migrate(args: argparse.Namespace) -> int:
    old_path = resolve_named_spec_path(args, attribute="old", label="Path to source model spec", default_to_current=True)
    new_path = resolve_named_spec_path(args, attribute="new", label="Path to target model spec")

    if args.source_state:
        source_state_path = Path(args.source_state).expanduser().resolve()
    elif can_prompt_interactively():
        source_state_path = _prompted_path("Path to source checkpoint")
    else:
        raise SystemExit("source-state is required")
    if not source_state_path.exists():
        raise SystemExit(f"source checkpoint does not exist: {source_state_path}")

    if args.output_state:
        output_state_path = Path(args.output_state).expanduser().resolve()
    else:
        output_state_path = new_path.parent / "outputs" / "checkpoints" / f"{new_path.stem}.migrated.pt"

    if args.plan_out:
        plan_out = Path(args.plan_out).expanduser().resolve()
    else:
        plan_out = output_state_path.with_suffix(".migration-plan.json")

    old_spec = load_architecture_spec(old_path)
    new_spec = load_architecture_spec(new_path)
    result = migrate_checkpoint(
        old_spec=old_spec,
        new_spec=new_spec,
        source_state_path=source_state_path,
        output_state_path=output_state_path.resolve(),
        plan_path=plan_out.resolve(),
    )

    print("checkpoint migrated")
    print(f"old spec: {old_path}")
    print(f"new spec: {new_path}")
    print(f"source checkpoint: {source_state_path}")
    print(f"output checkpoint: {result['output_state_path']}")
    print(f"preserved_tensor_count: {result['preserved_tensor_count']}")
    if result["plan_path"] is not None:
        print(f"migration plan: {result['plan_path']}")
    return 0
