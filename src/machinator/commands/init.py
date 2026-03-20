from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from machinator.commands.collate import candidate_recipes, infer_intent_task, resolve_report_path
from machinator.core import clean_optional, now_utc, require_workspace_root, slugify, workspace_paths, write_json
from machinator.modeling_collation import dataset_facts_from_report_path
from machinator.pipeline_refs import CONFIG_REF_FILENAME, render_config_ref_toml, replace_reference


STARTER_TASK_ENTRIES = {
    "validate": "machinator.pipeline_tasks:validate",
    "audit": "machinator.pipeline_tasks:audit",
    "train": "machinator.pipeline_tasks:train",
    "smoke": "machinator.pipeline_tasks:smoke",
}

# `init pipeline` is the thin-pipeline path, so refreshing it should clean away
# legacy scaffold files from older repo-local pipeline shapes without touching
# outputs that may contain logs, plots, or run artifacts the operator wants.
RESETTABLE_PIPELINE_ENTRIES = [
    Path("README.md"),
    Path("requirements.txt"),
    Path("src"),
    Path("config"),
    Path("data"),
    Path("dataset_facts.toml"),
    Path("model.toml"),
    Path("training.toml"),
]


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    init_parser = subparsers.add_parser("init", help="Initialize report-driven Machinator objects")
    init_subparsers = init_parser.add_subparsers(dest="init_command", required=True)

    pipeline_parser = init_subparsers.add_parser(
        "pipeline",
        help="Create a thin pipeline directory from a delegated data report",
    )
    pipeline_parser.add_argument("--workspace")
    pipeline_parser.add_argument("--report")
    pipeline_parser.add_argument("--name")
    pipeline_parser.add_argument("--force", action="store_true")
    pipeline_parser.set_defaults(func=cmd_init_pipeline)


def _default_recipe_for_facts(problem_type: str, modality: str) -> tuple[str, str]:
    intent_task = infer_intent_task(problem_type)
    if intent_task is None:
        raise SystemExit("could not infer a supported intent task from the delegated report")
    recipes = candidate_recipes(modality=modality, intent_task=intent_task)
    if not recipes:
        raise SystemExit(f"no supported starter recipe for modality `{modality}` and task `{intent_task}`")
    return intent_task, recipes[0].value


def _pipeline_type_for_modality(modality: str) -> str:
    return {
        "tabular": "tabular",
        "text": "nlp",
        "vision": "vision",
    }.get(modality, "custom")


def _pipeline_config_text(*, pipeline_name: str, pipeline_slug: str, pipeline_type: str) -> str:
    lines = [
        "[pipeline]",
        f'name = "{pipeline_name}"',
        f'slug = "{pipeline_slug}"',
        f'type = "{pipeline_type}"',
        'mode = "report-driven"',
        "",
        "[paths]",
        'data_root = "data"',
        'config_root = "config"',
        'experiments = "config"',
        'outputs = "outputs"',
        "",
        "[refs]",
        f'config_ref = "{CONFIG_REF_FILENAME}"',
        "",
    ]
    for task_name, entry in STARTER_TASK_ENTRIES.items():
        lines.extend(
            [
                f"[tasks.{task_name}]",
                f'entry = "{entry}"',
                f'description = "Package-managed {task_name} task"',
                "requires_dataset = false",
                "requires_experiment = false",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def _reset_report_driven_layout(pipeline_root: Path) -> None:
    for relative_path in RESETTABLE_PIPELINE_ENTRIES:
        candidate = pipeline_root / relative_path
        if candidate.exists() or candidate.is_symlink():
            _remove_path(candidate)


def cmd_init_pipeline(args: argparse.Namespace) -> int:
    workspace_root = require_workspace_root(args.workspace)
    report_path = resolve_report_path(clean_optional(args.report), workspace_root=workspace_root)
    facts = dataset_facts_from_report_path(report_path)
    intent_task, recipe_name = _default_recipe_for_facts(facts.suspected_problem_type, facts.modality)

    pipeline_name = clean_optional(args.name) or f"{slugify(facts.dataset_name, fallback='dataset')}-pipeline"
    pipeline_slug = slugify(pipeline_name, fallback="pipeline")
    paths = workspace_paths(workspace_root)
    pipeline_root = paths.pipeline_root / pipeline_slug
    manifest_path = paths.pipeline_registry_root / f"{pipeline_slug}.json"

    if (pipeline_root.exists() and any(pipeline_root.iterdir()) or manifest_path.exists()) and not args.force:
        raise SystemExit(f"pipeline `{pipeline_slug}` already exists; pass --force to refresh the report-driven scaffold")

    if args.force and pipeline_root.exists():
        _reset_report_driven_layout(pipeline_root)

    pipeline_root.mkdir(parents=True, exist_ok=True)
    (pipeline_root / "config").mkdir(parents=True, exist_ok=True)
    (pipeline_root / "outputs").mkdir(parents=True, exist_ok=True)
    (pipeline_root / "data" / "reports").mkdir(parents=True, exist_ok=True)

    dataset_source = facts.dataset_path.resolve()
    dataset_slug = slugify(facts.dataset_name, fallback="dataset")
    dataset_ref_root = pipeline_root / "data" / dataset_slug
    if dataset_source.is_dir():
        replace_reference(dataset_source, dataset_ref_root)
    else:
        dataset_ref_root.mkdir(parents=True, exist_ok=True)
        replace_reference(dataset_source, dataset_ref_root / dataset_source.name)

    report_ref_path = pipeline_root / "data" / "reports" / report_path.name
    replace_reference(report_path.resolve(), report_ref_path)

    (pipeline_root / "machinate.toml").write_text(
        _pipeline_config_text(
            pipeline_name=pipeline_name,
            pipeline_slug=pipeline_slug,
            pipeline_type=_pipeline_type_for_modality(facts.modality),
        )
    )
    (pipeline_root / CONFIG_REF_FILENAME).write_text(
        render_config_ref_toml(
            pipeline_name=pipeline_name,
            pipeline_slug=pipeline_slug,
            dataset_name=facts.dataset_name,
            modality=facts.modality,
            intent_task=intent_task,
            recipe_name=recipe_name,
            target_column=facts.target_column,
            dataset_ref_path=str(dataset_ref_root.relative_to(pipeline_root)),
            report_ref_path=str(report_ref_path.relative_to(pipeline_root)),
        )
    )
    (pipeline_root / ".gitignore").write_text("__pycache__/\n*.py[cod]\noutputs/\n")

    write_json(
        manifest_path,
        {
            "pipeline_name": pipeline_name,
            "pipeline_slug": pipeline_slug,
            "repo_path": str(pipeline_root),
            "pipeline_type": _pipeline_type_for_modality(facts.modality),
            "created_at": now_utc(),
            "pipeline_config_path": str(pipeline_root / "machinate.toml"),
            "supported_tasks": list(STARTER_TASK_ENTRIES),
            "mode": "report-driven",
        },
    )

    print("pipeline initialized")
    print(f"pipeline: {pipeline_root}")
    print(f"report: {report_ref_path}")
    print(f"dataset: {dataset_ref_root}")
    print(f"config ref: {pipeline_root / CONFIG_REF_FILENAME}")
    return 0
