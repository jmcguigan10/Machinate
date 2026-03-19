from __future__ import annotations

import argparse
from pathlib import Path

from machinate.commands.task import resolve_selected_pipeline
from machinate.core import clean_optional, load_pipeline_config, now_utc
from machinate.modeling import (
    architecture_spec_from_dataset_facts,
    dataset_facts_from_report_path,
    default_training_spec,
    render_dataset_facts_toml,
    render_model_spec_toml,
    render_training_spec_toml,
)
from machinate.ui import MenuChoice, can_prompt_interactively, prompt_select, prompt_text


COLLATION_BEGIN = "# BEGIN MACHINATE COLLATION"
COLLATION_END = "# END MACHINATE COLLATION"
INTENT_TASK_CHOICES = [MenuChoice("binary classification", "binary_classification")]


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    collate_parser = subparsers.add_parser("collate", help="Materialize pipeline specs from delegated facts")
    collate_subparsers = collate_parser.add_subparsers(dest="collate_command", required=True)

    pipeline_parser = collate_subparsers.add_parser(
        "pipeline",
        help="Generate dataset/model/training specs for the current pipeline",
    )
    pipeline_parser.add_argument("--workspace")
    pipeline_parser.add_argument("--pipeline")
    pipeline_parser.add_argument("--pipeline-path")
    pipeline_parser.add_argument("--report")
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
    pipeline_root, _workspace_root = resolve_selected_pipeline(args)
    report_path = resolve_report_path(clean_optional(args.report))
    pipeline_config = load_pipeline_config(pipeline_root)
    pipeline_name = str(pipeline_config.get("pipeline", {}).get("name", pipeline_root.name))

    facts = dataset_facts_from_report_path(report_path)
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
        config_path=pipeline_root / "machinate.toml",
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
