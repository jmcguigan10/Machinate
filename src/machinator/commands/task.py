from __future__ import annotations

import argparse
from pathlib import Path

from machinator.core import (
    find_workspace_root,
    load_pipeline_config,
    pipeline_tasks,
    registered_pipeline_manifests,
    resolve_pipeline_root,
)
from machinator.ui import MenuChoice, can_prompt_interactively, prompt_select


def prompt_pipeline_name(workspace_root: Path) -> str:
    manifests = registered_pipeline_manifests(workspace_root)
    if not manifests:
        raise SystemExit(f"no pipelines are registered in workspace `{workspace_root}`")
    choices = [
        MenuChoice(
            f"{manifest['pipeline_slug']} ({manifest['repo_path']})",
            str(manifest["pipeline_slug"]),
        )
        for manifest in manifests
    ]
    return prompt_select(
        "Select a pipeline",
        choices,
        default=str(manifests[0]["pipeline_slug"]),
        use_search_filter=len(choices) > 8,
    )


def resolve_selected_pipeline(args: argparse.Namespace) -> tuple[Path, Path | None]:
    workspace_root = find_workspace_root(Path(args.workspace).expanduser().resolve()) if args.workspace else find_workspace_root()
    pipeline_name = args.pipeline
    if pipeline_name is None and args.pipeline_path is None and workspace_root is not None and can_prompt_interactively():
        if registered_pipeline_manifests(workspace_root) and Path.cwd().resolve() == workspace_root:
            pipeline_name = prompt_pipeline_name(workspace_root)
    return resolve_pipeline_root(
        workspace_root=workspace_root,
        pipeline_name=pipeline_name,
        pipeline_path=args.pipeline_path,
    )


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    task_parser = subparsers.add_parser("task", help="Inspect native Machinator pipeline tasks")
    task_subparsers = task_parser.add_subparsers(dest="task_command", required=True)

    task_list = task_subparsers.add_parser("list", help="List the tasks declared in machinate.toml")
    task_list.add_argument("--workspace")
    task_list.add_argument("--pipeline")
    task_list.add_argument("--pipeline-path")
    task_list.set_defaults(func=cmd_task_list)


def cmd_task_list(args: argparse.Namespace) -> int:
    pipeline_root, _workspace_root = resolve_selected_pipeline(args)
    pipeline_config = load_pipeline_config(pipeline_root)
    tasks = pipeline_tasks(pipeline_config)
    pipeline_name = str(pipeline_config.get("pipeline", {}).get("name", pipeline_root.name))

    print(f"pipeline `{pipeline_name}` tasks")
    for task_name, task_config in tasks.items():
        description = str(task_config.get("description", "")).strip()
        entry = str(task_config.get("entry", "")).strip()
        if description:
            print(f"  - {task_name}: {description}")
        else:
            print(f"  - {task_name}")
        if entry:
            print(f"    entry: {entry}")
    return 0
