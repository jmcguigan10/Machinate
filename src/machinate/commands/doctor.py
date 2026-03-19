from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from machinate.core import WORKSPACE_SENTINEL, app_paths, find_workspace_root, load_json
from machinate.ui import QUESTIONARY


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    doctor_parser = subparsers.add_parser("doctor", help="Check the Machinate install and workspace health")
    doctor_parser.add_argument("--workspace")
    doctor_parser.set_defaults(func=cmd_doctor)


def cmd_doctor(args: argparse.Namespace) -> int:
    failures: list[str] = []
    notes: list[str] = []

    python_bin = shutil.which("python3") or shutil.which("python")
    if python_bin:
        notes.append(f"python: {python_bin}")
    else:
        failures.append("python is not available on PATH")

    brew_bin = shutil.which("brew")
    if brew_bin:
        notes.append(f"homebrew: {brew_bin}")
    else:
        notes.append("homebrew: not found on PATH")

    node_bin = shutil.which("node")
    if node_bin:
        notes.append(f"node: {node_bin}")
    else:
        notes.append("node: not found on PATH; URL dataset downloads will be unavailable")

    codex_bin = shutil.which("codex")
    if codex_bin:
        notes.append(f"codex: {codex_bin}")
    else:
        notes.append("codex: not found on PATH; `macht legate` will be unavailable")

    config_path = app_paths().config_path
    if config_path.exists():
        notes.append(f"global config: {config_path}")
    else:
        notes.append(f"global config not created yet: {config_path}")

    if QUESTIONARY is not None:
        notes.append("prompt backend: questionary")
    else:
        notes.append("prompt backend: plain terminal input")

    workspace_root = find_workspace_root(Path(args.workspace).expanduser().resolve()) if args.workspace else find_workspace_root()

    if workspace_root is None:
        notes.append("workspace: not detected from the current directory")
    else:
        sentinel_path = workspace_root / WORKSPACE_SENTINEL
        if not sentinel_path.exists():
            failures.append(f"workspace marker missing: {sentinel_path}")
        else:
            manifest = load_json(sentinel_path)
            notes.append(f"workspace root: {workspace_root}")
            notes.append(f"workspace name: {manifest.get('workspace_name', '')}")

        for required_dir in (
            workspace_root / ".machinate" / "pipelines",
            workspace_root / ".machinate" / "assets",
            workspace_root / ".envs" / "venvs",
            workspace_root / "data" / "staged",
            workspace_root / "outputs",
            workspace_root / "pipelines",
        ):
            if not required_dir.exists():
                failures.append(f"missing workspace directory: {required_dir}")

    if failures:
        print("doctor found issues")
        for failure in failures:
            print(f"  - {failure}")
        if notes:
            print("notes")
            for note in notes:
                print(f"  - {note}")
        return 1

    print("doctor OK")
    for note in notes:
        print(f"  - {note}")
    return 0
