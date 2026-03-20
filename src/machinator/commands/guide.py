from __future__ import annotations

import argparse
import importlib.resources


GUIDES = {
    "beginner": "beginner-guide.md",
    "workflow": "beginner-guide.md",
}


def guide_path(name: str):
    filename = GUIDES.get(name)
    if filename is None:
        raise SystemExit(f"unknown guide `{name}`")
    return importlib.resources.files("machinator").joinpath("guides").joinpath(filename)


def guide_text(name: str) -> str:
    return guide_path(name).read_text(encoding="utf-8")


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    guide_parser = subparsers.add_parser("guide", help="Read built-in operator guides")
    guide_subparsers = guide_parser.add_subparsers(dest="guide_command", required=True)

    list_parser = guide_subparsers.add_parser("list", help="List the available guides")
    list_parser.set_defaults(func=cmd_guide_list)

    beginner_parser = guide_subparsers.add_parser("beginner", help="Read the beginner guide")
    beginner_parser.add_argument("--path", action="store_true", help="Print the packaged guide path instead of its contents")
    beginner_parser.set_defaults(func=cmd_guide_beginner)

    workflow_parser = guide_subparsers.add_parser("workflow", help="Read the default operating workflow guide")
    workflow_parser.add_argument("--path", action="store_true", help="Print the packaged guide path instead of its contents")
    workflow_parser.set_defaults(func=cmd_guide_workflow)


def cmd_guide_list(_args: argparse.Namespace) -> int:
    print("available guides")
    print("  - beginner")
    print("  - workflow")
    return 0


def _emit_guide(name: str, *, path_only: bool) -> int:
    path = guide_path(name)
    if path_only:
        print(path)
        return 0
    print(guide_text(name).rstrip())
    return 0


def cmd_guide_beginner(args: argparse.Namespace) -> int:
    return _emit_guide("beginner", path_only=bool(args.path))


def cmd_guide_workflow(args: argparse.Namespace) -> int:
    return _emit_guide("workflow", path_only=bool(args.path))
