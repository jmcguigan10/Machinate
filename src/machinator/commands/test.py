from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from machinator.core import require_project_root


VALID_TEST_TARGETS = ("all", "python", "unit", "integration", "rust")


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    test_parser = subparsers.add_parser("test", help="Run Machinator contributor test suites")
    test_parser.add_argument("target", nargs="?", default="all", choices=VALID_TEST_TARGETS)
    test_parser.add_argument("--root", help="Path inside the Machinator source repo")
    test_parser.set_defaults(func=cmd_test)


def python_env(project_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    source_root = str((project_root / "src").resolve())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = source_root if not existing else f"{source_root}{os.pathsep}{existing}"
    return env


def run_step(*, cwd: Path, command: list[str], env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(command))
    result = subprocess.run(command, cwd=cwd, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def run_test_target(project_root: Path, target: str) -> None:
    env = python_env(project_root)
    if target == "unit":
        run_step(
            cwd=project_root,
            env=env,
            command=[
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests/unit",
                "-t",
                ".",
                "-p",
                "test_*.py",
            ],
        )
        return
    if target == "integration":
        run_step(
            cwd=project_root,
            env=env,
            command=[
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests/integration",
                "-t",
                ".",
                "-p",
                "test_*.py",
            ],
        )
        return
    if target == "python":
        run_test_target(project_root, "unit")
        run_test_target(project_root, "integration")
        return
    if target == "rust":
        run_step(
            cwd=project_root,
            command=[
                "cargo",
                "test",
                "--manifest-path",
                str((project_root / "rust" / "machinator-ir" / "Cargo.toml").resolve()),
            ],
        )
        return
    if target == "all":
        run_test_target(project_root, "python")
        run_test_target(project_root, "rust")
        return
    raise SystemExit(f"unsupported test target `{target}`")


def cmd_test(args: argparse.Namespace) -> int:
    project_root = require_project_root(args.root)
    run_test_target(project_root, str(args.target))
    return 0
