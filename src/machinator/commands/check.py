from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from machinator.commands.test import python_env, run_test_target
from machinator.core import require_project_root


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    check_parser = subparsers.add_parser("check", help="Run Machinator contributor verification")
    check_parser.add_argument("--root", help="Path inside the Machinator source repo")
    check_parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip integration tests and run the quicker contributor checks",
    )
    check_parser.set_defaults(func=cmd_check)


def run_step(*, cwd: Path, command: list[str], env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(command))
    result = subprocess.run(command, cwd=cwd, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def cmd_check(args: argparse.Namespace) -> int:
    project_root = require_project_root(args.root)
    env = python_env(project_root)

    run_step(
        cwd=project_root,
        env=env,
        command=[sys.executable, "-m", "compileall", "src", "tests"],
    )
    run_test_target(project_root, "unit")
    if not args.fast:
        run_test_target(project_root, "integration")
    run_test_target(project_root, "rust")
    return 0
