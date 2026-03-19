from __future__ import annotations

import argparse

from machinate.core import build_task_context, load_pipeline_config, load_task_callable, pipeline_tasks
from machinate.ui import MenuChoice, can_prompt_interactively, prompt_select

from machinate.commands.task import resolve_selected_pipeline


def prompt_task_name(task_names: list[str]) -> str:
    choices = [MenuChoice(task_name, task_name) for task_name in task_names]
    default = "train" if "train" in task_names else choices[0].value
    return prompt_select(
        "Select a pipeline task",
        choices,
        default=default,
        use_search_filter=len(choices) > 8,
    )


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    run_parser = subparsers.add_parser("run", help="Run a native Machinate pipeline task")
    run_parser.add_argument("task_name", nargs="?")
    run_parser.add_argument("--workspace")
    run_parser.add_argument("--pipeline")
    run_parser.add_argument("--pipeline-path")
    run_parser.add_argument("--experiment")
    run_parser.add_argument("--dataset")
    run_parser.set_defaults(func=cmd_run)


def cmd_run(args: argparse.Namespace) -> int:
    pipeline_root, workspace_root = resolve_selected_pipeline(args)
    pipeline_config = load_pipeline_config(pipeline_root)
    tasks = pipeline_tasks(pipeline_config)

    task_name = args.task_name
    if task_name is None:
        if can_prompt_interactively():
            task_name = prompt_task_name(list(tasks))
        else:
            raise SystemExit("task name is required; run `macht run <task>`")
    if task_name not in tasks:
        raise SystemExit(f"unknown pipeline task `{task_name}`")

    task_callable, _task_config = load_task_callable(pipeline_root, pipeline_config, task_name)
    context = build_task_context(
        workspace_root=workspace_root,
        pipeline_root=pipeline_root,
        pipeline_config=pipeline_config,
        task_name=task_name,
        experiment_name=args.experiment,
        dataset_ref=args.dataset,
    )
    result = task_callable(context)
    if isinstance(result, int):
        return result
    return 0
