from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from machinator.commands.task import resolve_selected_pipeline
from machinator.commands.new import (
    RECIPE_BEGIN,
    RECIPE_END,
    STARTER_TASKS,
    create_pipeline_scaffold,
)
from machinator.core import (
    clean_optional,
    load_json,
    load_workspace_pipeline_manifest,
    load_pipeline_config,
    now_utc,
    require_workspace_root,
    slugify,
    workspace_paths,
)
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
REPORT_DRIVEN_TEMPLATE = "native-python"


@dataclass(frozen=True)
class ReportCandidate:
    path: Path
    dataset_name: str
    generated_at: str

    @property
    def label(self) -> str:
        stamp = self.generated_at or "unknown time"
        return f"{self.dataset_name} ({stamp}) [{self.path.name}]"


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
    pipeline_parser.add_argument("--intent-task")
    pipeline_parser.add_argument("--recipe")
    pipeline_parser.add_argument("--force", action="store_true")
    pipeline_parser.set_defaults(func=cmd_collate_pipeline)


def _report_candidate_from_payload(path: Path, payload: dict[str, Any]) -> ReportCandidate | None:
    # Only completed delegated data-report artifacts should be auto-discovered.
    if str(payload.get("delegate_kind", "")) != "report":
        return None
    if str(payload.get("report_kind", "")) != "data":
        return None

    report_payload = payload.get("report")
    if not isinstance(report_payload, dict):
        return None

    dataset_name = clean_optional(str(report_payload.get("dataset_name", ""))) or path.stem
    generated_at = clean_optional(str(payload.get("generated_at", ""))) or ""
    return ReportCandidate(path=path, dataset_name=dataset_name, generated_at=generated_at)


def discover_report_candidates(workspace_root: Path) -> list[ReportCandidate]:
    report_root = workspace_paths(workspace_root).output_root / "reports" / "legate"
    if not report_root.exists():
        return []

    candidates: list[ReportCandidate] = []
    for report_path in report_root.glob("*.json"):
        try:
            payload = load_json(report_path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        candidate = _report_candidate_from_payload(report_path.resolve(), payload)
        if candidate is not None:
            candidates.append(candidate)

    return sorted(candidates, key=lambda item: (item.generated_at, item.path.name), reverse=True)


def resolve_report_path(report_text: str | None, *, workspace_root: Path) -> Path:
    if report_text:
        report_path = Path(report_text).expanduser().resolve()
        if not report_path.exists():
            raise SystemExit(f"delegated report does not exist: {report_path}")
        return report_path

    candidates = discover_report_candidates(workspace_root)
    if not candidates:
        if can_prompt_interactively():
            report_path = Path(prompt_text("Path to delegated report JSON")).expanduser().resolve()
            if not report_path.exists():
                raise SystemExit(f"delegated report does not exist: {report_path}")
            return report_path
        raise SystemExit(
            "no compatible delegated data reports found in this workspace; "
            "run `macht legate report --data` or pass --report"
        )

    if len(candidates) == 1 or not can_prompt_interactively():
        return candidates[0].path

    # When there are several reports, let the operator choose instead of
    # forcing them to paste a path back into the CLI.
    selected = prompt_select(
        "Select the delegated report to collate",
        [MenuChoice(candidate.label, str(candidate.path)) for candidate in candidates],
        default=str(candidates[0].path),
        use_search_filter=len(candidates) > 8,
    )
    return Path(selected)


def write_if_allowed(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"refusing to overwrite existing file without --force: {path}")
    path.write_text(content)


def render_baseline_experiment(*, dataset_kind: str, target_column: str) -> str:
    return (
        "[dataset]\n"
        f'kind = "{dataset_kind}"\n'
        f'target_column = "{target_column}"\n'
        "\n"
        "[training]\n"
        "epochs = 1\n"
        "learning_rate = 0.001\n"
    )


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
    if inferred is None:
        raise SystemExit("could not infer a supported intent task; pass --intent-task")
    return inferred


def candidate_recipes(*, modality: str, intent_task: str) -> list[MenuChoice]:
    recipes: list[MenuChoice] = []
    if intent_task == "binary_classification" and modality == "tabular":
        recipes.append(MenuChoice("tabular.binary.basic", "tabular.binary.basic"))
        recipes.append(MenuChoice("tabular.binary.deep", "tabular.binary.deep"))
    if intent_task == "binary_classification" and modality == "text":
        recipes.append(MenuChoice("text.binary.transformer", "text.binary.transformer"))
    if intent_task == "binary_classification" and modality == "vision":
        recipes.append(MenuChoice("vision.binary.cnn", "vision.binary.cnn"))
        recipes.append(MenuChoice("vision.binary.resnet", "vision.binary.resnet"))
    return recipes


def resolve_recipe(args: argparse.Namespace, *, modality: str, intent_task: str) -> str:
    explicit = clean_optional(args.recipe)
    if explicit is not None:
        return explicit

    recipes = candidate_recipes(modality=modality, intent_task=intent_task)
    if not recipes:
        raise SystemExit(f"no supported recipes for modality `{modality}` and intent `{intent_task}`")
    # The dataset-first path is intentionally recipe-centric and deterministic for
    # now. We choose the first supported starter recipe for the inferred facts
    # instead of asking legacy scaffolding questions during creation.
    return recipes[0].value


def default_pipeline_type_for_modality(modality: str) -> str:
    return {
        "tabular": "tabular",
        "text": "nlp",
        "vision": "vision",
    }.get(modality, "custom")


def default_dataset_kind_for_modality(modality: str) -> str:
    return {
        "tabular": "csv",
        "text": "text_table",
        "vision": "image_folder",
    }.get(modality, "data")


def resolve_pipeline_creation_name(args: argparse.Namespace, dataset_name: str) -> str:
    explicit = clean_optional(args.name) or clean_optional(args.pipeline)
    if explicit is not None:
        return explicit
    return f"{slugify(dataset_name, fallback='dataset')}-pipeline"


def resolve_or_create_pipeline(args: argparse.Namespace, *, dataset_name: str, modality: str) -> tuple[Path, Path | None]:
    if not args.create:
        return resolve_selected_pipeline(args)

    workspace_root = require_workspace_root(args.workspace)
    pipeline_name = resolve_pipeline_creation_name(args, dataset_name)
    pipeline_slug = slugify(pipeline_name, fallback="pipeline")
    default_repo_path = workspace_paths(workspace_root).pipeline_root / pipeline_slug

    # Re-running report-driven collation against an existing scaffold should use
    # that scaffold rather than failing on the default workspace path.
    manifest_path = workspace_paths(workspace_root).pipeline_registry_root / f"{pipeline_slug}.json"
    if manifest_path.exists():
        manifest = load_workspace_pipeline_manifest(workspace_root, pipeline_slug)
        return Path(str(manifest["repo_path"])).expanduser().resolve(), workspace_root
    if (default_repo_path / "machinate.toml").exists():
        return default_repo_path.resolve(), workspace_root

    scaffold = create_pipeline_scaffold(
        workspace_root=workspace_root,
        pipeline_name=pipeline_name,
        pipeline_type=default_pipeline_type_for_modality(modality),
        template=REPORT_DRIVEN_TEMPLATE,
        repo_path=default_repo_path,
        tasks=list(STARTER_TASKS),
    )
    return Path(scaffold["repo_path"]), workspace_root


def recipe_variant(recipe_name: str) -> str:
    if recipe_name.endswith(".deep"):
        return "deep"
    return "vanilla"


def render_recipe_block(*, recipe_name: str, model_family: str, modality: str, intent_task: str) -> str:
    return (
        f"{RECIPE_BEGIN}\n"
        "[recipe]\n"
        f'name = "{recipe_name}"\n'
        f'family = "{model_family}"\n'
        f'variant = "{recipe_variant(recipe_name)}"\n'
        f'modality = "{modality}"\n'
        f'task = "{intent_task}"\n'
        f"{RECIPE_END}\n"
    )


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


def upsert_generated_block(*, existing: str, begin_marker: str, end_marker: str, rendered_block: str) -> str:
    if begin_marker in existing and end_marker in existing:
        start = existing.index(begin_marker)
        end = existing.index(end_marker) + len(end_marker)
        updated = f"{existing[:start].rstrip()}\n\n{rendered_block}\n{existing[end:].lstrip()}"
    else:
        updated = existing.rstrip() + "\n\n" + rendered_block
    return updated.rstrip() + "\n"


def upsert_collation_block(
    *,
    config_path: Path,
    report_path: Path,
    dataset_name: str,
    intent_task: str,
    recipe_name: str,
    model_family: str,
    modality: str,
) -> None:
    recipe_block = render_recipe_block(
        recipe_name=recipe_name,
        model_family=model_family,
        modality=modality,
        intent_task=intent_task,
    )
    collation_block = render_collation_block(
        report_path=report_path,
        dataset_name=dataset_name,
        intent_task=intent_task,
        recipe_name=recipe_name,
        model_family=model_family,
    )
    existing = config_path.read_text()
    updated = upsert_generated_block(
        existing=existing,
        begin_marker=RECIPE_BEGIN,
        end_marker=RECIPE_END,
        rendered_block=recipe_block,
    )
    updated = upsert_generated_block(
        existing=updated,
        begin_marker=COLLATION_BEGIN,
        end_marker=COLLATION_END,
        rendered_block=collation_block,
    )
    config_path.write_text(updated)


def cmd_collate_pipeline(args: argparse.Namespace) -> int:
    workspace_root = require_workspace_root(args.workspace)
    report_path = resolve_report_path(clean_optional(args.report), workspace_root=workspace_root)
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
    baseline_config_path = pipeline_root / "config" / "baseline.toml"

    write_if_allowed(dataset_facts_path, render_dataset_facts_toml(facts), force=args.force)
    write_if_allowed(model_path, render_model_spec_toml(model_spec), force=args.force)
    write_if_allowed(training_path, render_training_spec_toml(training_spec), force=args.force)
    if args.create or args.force or not baseline_config_path.exists():
        baseline_config_path.write_text(
            render_baseline_experiment(
                dataset_kind=default_dataset_kind_for_modality(facts.modality),
                target_column=facts.target_column,
            )
        )
    upsert_collation_block(
        config_path=pipeline_root / "machinate.toml",
        report_path=report_path,
        dataset_name=facts.dataset_name,
        intent_task=intent_task,
        recipe_name=recipe_name,
        model_family=model_spec.family,
        modality=facts.modality,
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
