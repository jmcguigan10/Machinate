from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

from machinator.commands.collate import candidate_recipes, infer_intent_task
from machinator.commands.task import resolve_selected_pipeline
from machinator.core import load_pipeline_config
from machinator.modeling_collation import architecture_spec_from_dataset_facts, dataset_facts_from_report_path, default_training_spec, render_dataset_facts_toml
from machinator.pipeline_refs import (
    CONFIG_REF_FILENAME,
    dataset_yaml_payload,
    generated_config_paths,
    load_config_ref,
    model_yaml_payload,
    referenced_report_path,
    training_yaml_payload,
    write_yaml,
)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    build_parser = subparsers.add_parser("build", help="Build report-derived pipeline artifacts")
    build_subparsers = build_parser.add_subparsers(dest="build_command", required=True)

    configs_parser = build_subparsers.add_parser(
        "configs",
        help="Generate YAML configs from the selected delegated JSON data report",
    )
    configs_parser.add_argument("--workspace")
    configs_parser.add_argument("--pipeline")
    configs_parser.add_argument("--pipeline-path")
    configs_parser.add_argument("--force", action="store_true")
    configs_parser.set_defaults(func=cmd_build_configs)


def _write_if_allowed(path: Path, *, payload: dict[str, object], force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"refusing to overwrite existing generated config without --force: {path}")
    write_yaml(path, payload)


def cmd_build_configs(args: argparse.Namespace) -> int:
    pipeline_root, _workspace_root = resolve_selected_pipeline(args)
    pipeline_config = load_pipeline_config(pipeline_root)
    config_ref_path = pipeline_root / str(pipeline_config.get("refs", {}).get("config_ref", CONFIG_REF_FILENAME))
    config_ref = load_config_ref(config_ref_path)

    report_path = referenced_report_path(pipeline_root, config_ref)
    facts = dataset_facts_from_report_path(report_path)

    pipeline_section = config_ref.get("pipeline", {})
    if not isinstance(pipeline_section, dict):
        pipeline_section = {}
    pipeline_name = str(pipeline_section.get("name", pipeline_root.name))
    intent_task = str(pipeline_section.get("intent_task", "")).strip() or infer_intent_task(facts.suspected_problem_type) or ""
    if not intent_task:
        raise SystemExit("could not infer a supported intent task; update config-ref.toml")

    recipe_name = str(pipeline_section.get("recipe", "")).strip()
    if not recipe_name:
        recipes = candidate_recipes(modality=facts.modality, intent_task=intent_task)
        if not recipes:
            raise SystemExit(f"no supported recipe for modality `{facts.modality}` and task `{intent_task}`")
        recipe_name = recipes[0].value

    model_spec = architecture_spec_from_dataset_facts(
        facts=facts,
        pipeline_name=pipeline_name,
        recipe_name=recipe_name,
    )
    training_spec = default_training_spec(facts, family=model_spec.family)
    config_paths = generated_config_paths(pipeline_root, config_ref)

    _write_if_allowed(
        config_paths["dataset"],
        payload=dataset_yaml_payload(tomllib.loads(render_dataset_facts_toml(facts))),
        force=args.force,
    )
    _write_if_allowed(config_paths["model"], payload=model_yaml_payload(model_spec), force=args.force)
    _write_if_allowed(config_paths["training"], payload=training_yaml_payload(training_spec), force=args.force)

    print("generated configs written")
    print(f"config ref: {config_ref_path}")
    print(f"report: {report_path}")
    print(f"dataset config: {config_paths['dataset']}")
    print(f"model config: {config_paths['model']}")
    print(f"training config: {config_paths['training']}")
    return 0
