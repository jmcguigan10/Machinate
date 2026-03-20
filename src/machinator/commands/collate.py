from __future__ import annotations

import argparse
from pathlib import Path

from machinator.commands.task import resolve_selected_pipeline
from machinator.commands.new import (
    PIPELINE_TEMPLATES,
    PIPELINE_TYPES,
    create_pipeline_scaffold,
    selected_tasks,
)
from machinator.core import clean_optional, load_pipeline_config, now_utc, require_workspace_root, slugify
from machinator.modeling_collation import (
    architecture_spec_from_dataset_facts,
    dataset_facts_from_report_path,
    default_training_spec,
    render_dataset_facts_toml,
)
from machinator.modeling_specs import (
    render_model_spec_toml,
    render_training_spec_toml,
)
from machinator.ui import MenuChoice, can_prompt_interactively, prompt_select, prompt_text


COLLATION_BEGIN = "# BEGIN MACHINATE COLLATION"
COLLATION_END = "# END MACHINATE COLLATION"
INTENT_TASK_CHOICES = [MenuChoice("binary classification", "binary_classification")]


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    collate_parser = subparsers.add_parser("collate", help="Materialize pipeline specs from delegated facts")
    collate_subparsers = collate_parser.add_subparsers(dest="collate_command", required=True)

    pipeline_parser = collate_subparsers.add_parser(
        "pipeline",
        help="Generate dataset/model/training specs for an existing or newly created pipeline",
    )
    pipeline_parser.add_argument("--workspace")
    pipeline_parser.add_argument("--pipeline")
    pipeline_parser.add_argument("--pipeline-path")
    pipeline_parser.add_argument("--report")
    pipeline_parser.add_argument("--create", action="store_true", help="Create the pipeline scaffold from the report first")
    pipeline_parser.add_argument("--name", help="Pipeline name to use when creating a new pipeline")
    pipeline_parser.add_argument("--type", help="Pipeline type to use when creating a new pipeline")
    pipeline_parser.add_argument("--template", help="Template to use when creating a new pipeline")
    pipeline_parser.add_argument("--task", action="append", default=[], help="Starter task to include when creating")
    pipeline_parser.add_argument("--intent-task")
    pipeline_parser.add_argument("--recipe")
    pipeline_parser.add_argument("--force", action="store_true")
    pipeline_parser.set_defaults(func=cmd_collate_pipeline)


def resolve_report_path(report_text: str | None) -> Path:
    if report_text:
        report_path = Path(report_text).expanduser().resolve()
        if not report_path.exists():
            raise SystemExit(f"delegated report does not exist: {report_path}")
        return report_path

    if can_prompt_interactively():
        report_path = Path(prompt_text("Path to delegated report JSON")).expanduser().resolve()
        if not report_path.exists():
            raise SystemExit(f"delegated report does not exist: {report_path}")
        return report_path

    raise SystemExit("delegated report is required; pass --report")


def write_if_allowed(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"refusing to overwrite existing file without --force: {path}")
    path.write_text(content)


def infer_intent_task(problem_type: str) -> str | None:
    lowered = problem_type.lower()
    if "binary" in lowered:
        return "binary_classification"
    return None


def resolve_intent_task(args: argparse.Namespace, problem_type: str) -> str:
    explicit = clean_optional(args.intent_task)
    if explicit is not None:
        return explicit

    inferred = infer_intent_task(problem_type)
    if can_prompt_interactively():
        return prompt_select(
            "Select the intended training task",
            INTENT_TASK_CHOICES,
            default=inferred or "binary_classification",
        )
    if inferred is None:
        raise SystemExit("could not infer a supported intent task; pass --intent-task")
    return inferred


def candidate_recipes(*, modality: str, intent_task: str) -> list[MenuChoice]:
    recipes: list[MenuChoice] = []
    if intent_task == "binary_classification" and modality == "tabular":
        recipes.append(MenuChoice("tabular.binary.basic", "tabular.binary.basic"))
    if intent_task == "binary_classification" and modality == "text":
        recipes.append(MenuChoice("text.binary.transformer", "text.binary.transformer"))
    return recipes


def resolve_recipe(args: argparse.Namespace, *, modality: str, intent_task: str) -> str:
    explicit = clean_optional(args.recipe)
    if explicit is not None:
        return explicit

    recipes = candidate_recipes(modality=modality, intent_task=intent_task)
    if not recipes:
        raise SystemExit(f"no supported recipes for modality `{modality}` and intent `{intent_task}`")
    if can_prompt_interactively():
        return prompt_select(
            "Select a pipeline recipe",
            recipes,
            default=recipes[0].value,
        )
    return recipes[0].value


def default_pipeline_type_for_modality(modality: str) -> str:
    return {
        "tabular": "tabular",
        "text": "nlp",
    }.get(modality, "custom")


def resolve_pipeline_creation_name(args: argparse.Namespace, dataset_name: str) -> str:
    explicit = clean_optional(args.name) or clean_optional(args.pipeline)
    if explicit is not None:
        return explicit
    default_name = f"{slugify(dataset_name, fallback='dataset')}-pipeline"
    if can_prompt_interactively():
        return prompt_text("Pipeline name", default=default_name)
    return default_name


def resolve_pipeline_creation_type(args: argparse.Namespace, modality: str) -> str:
    explicit = clean_optional(args.type)
    default_type = default_pipeline_type_for_modality(modality)
    if explicit is not None:
        return explicit
    if can_prompt_interactively():
        return prompt_select("Pipeline type", PIPELINE_TYPES, default=default_type)
    return default_type


def resolve_pipeline_creation_template(args: argparse.Namespace) -> str:
    explicit = clean_optional(args.template)
    if explicit is not None:
        return explicit
    if can_prompt_interactively():
        return prompt_select("Pipeline template", PIPELINE_TEMPLATES, default="native-python")
    return "native-python"


def resolve_or_create_pipeline(args: argparse.Namespace, *, dataset_name: str, modality: str) -> tuple[Path, Path | None]:
    if not args.create:
        return resolve_selected_pipeline(args)

    workspace_root = require_workspace_root(args.workspace)
    pipeline_name = resolve_pipeline_creation_name(args, dataset_name)
    pipeline_type = resolve_pipeline_creation_type(args, modality)
    template = resolve_pipeline_creation_template(args)
    tasks = selected_tasks(args.task)
    repo_path = Path(args.pipeline_path).expanduser().resolve() if clean_optional(args.pipeline_path) else None

    scaffold = create_pipeline_scaffold(
        workspace_root=workspace_root,
        pipeline_name=pipeline_name,
        pipeline_type=pipeline_type,
        template=template,
        repo_path=repo_path,
        tasks=tasks,
    )
    return Path(scaffold["repo_path"]), workspace_root


def render_collation_block(
    *,
    report_path: Path,
    dataset_name: str,
    intent_task: str,
    recipe_name: str,
    model_family: str,
) -> str:
    return (
        f"{COLLATION_BEGIN}\n"
        "[collation]\n"
        f'generated_at = "{now_utc()}"\n'
        f'source_report = "{report_path}"\n'
        f'source_dataset = "{dataset_name}"\n'
        f'intent_task = "{intent_task}"\n'
        f'recipe = "{recipe_name}"\n'
        f'model_family = "{model_family}"\n'
        f"{COLLATION_END}\n"
    )


def upsert_collation_block(
    *,
    config_path: Path,
    report_path: Path,
    dataset_name: str,
    intent_task: str,
    recipe_name: str,
    model_family: str,
) -> None:
    block = render_collation_block(
        report_path=report_path,
        dataset_name=dataset_name,
        intent_task=intent_task,
        recipe_name=recipe_name,
        model_family=model_family,
    )
    existing = config_path.read_text()
    if COLLATION_BEGIN in existing and COLLATION_END in existing:
        start = existing.index(COLLATION_BEGIN)
        end = existing.index(COLLATION_END) + len(COLLATION_END)
        updated = f"{existing[:start].rstrip()}\n\n{block}\n{existing[end:].lstrip()}"
    else:
        updated = existing.rstrip() + "\n\n" + block
    config_path.write_text(updated.rstrip() + "\n")


def cmd_collate_pipeline(args: argparse.Namespace) -> int:
    report_path = resolve_report_path(clean_optional(args.report))
    facts = dataset_facts_from_report_path(report_path)
    pipeline_root, _workspace_root = resolve_or_create_pipeline(
        args,
        dataset_name=facts.dataset_name,
        modality=facts.modality,
    )
    pipeline_config = load_pipeline_config(pipeline_root)
    pipeline_name = str(pipeline_config.get("pipeline", {}).get("name", pipeline_root.name))
    intent_task = resolve_intent_task(args, facts.suspected_problem_type)
    recipe_name = resolve_recipe(args, modality=facts.modality, intent_task=intent_task)
    model_spec = architecture_spec_from_dataset_facts(
        facts=facts,
        pipeline_name=pipeline_name,
        recipe_name=recipe_name,
    )
    training_spec = default_training_spec(facts, family=model_spec.family)

    dataset_facts_path = pipeline_root / "dataset_facts.toml"
    model_path = pipeline_root / "model.toml"
    training_path = pipeline_root / "training.toml"

    write_if_allowed(dataset_facts_path, render_dataset_facts_toml(facts), force=args.force)
    write_if_allowed(model_path, render_model_spec_toml(model_spec), force=args.force)
    write_if_allowed(training_path, render_training_spec_toml(training_spec), force=args.force)
    upsert_collation_block(
        config_path=pipeline_root / "machinator.toml",
        report_path=report_path,
        dataset_name=facts.dataset_name,
        intent_task=intent_task,
        recipe_name=recipe_name,
        model_family=model_spec.family,
    )

    print("pipeline specs collated")
    print(f"pipeline: {pipeline_root}")
    print(f"report: {report_path}")
    print(f"intent_task: {intent_task}")
    print(f"recipe: {recipe_name}")
    print(f"dataset facts: {dataset_facts_path}")
    print(f"model spec: {model_path}")
    print(f"training spec: {training_path}")
    return 0
