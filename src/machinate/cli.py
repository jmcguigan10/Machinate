from __future__ import annotations

import argparse

from machinate import __version__
from machinate.commands import doctor, grab, legate, new, run, task, workspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="macht",
        description="Machinate: prompt-first control-plane CLI for ML workspaces and pipelines",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor.register(subparsers)
    workspace.register(subparsers)
    new.register(subparsers)
    grab.register(subparsers)
    legate.register(subparsers)
    task.register(subparsers)
    run.register(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
