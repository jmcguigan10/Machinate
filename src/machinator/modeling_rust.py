from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from machinator.modeling_types import ModelSpecError

# Thin bridge to the Rust IR crate. Python still shells out today, but this file
# isolates that boundary so we can swap it for a tighter binding later.


def rust_ir_manifest_path() -> Path:
    return Path(__file__).resolve().parents[2] / "rust" / "machinator-ir" / "Cargo.toml"


def rust_ir_available() -> bool:
    return shutil.which("cargo") is not None and rust_ir_manifest_path().exists()


def run_rust_ir_cli(*args: str) -> dict[str, Any] | None:
    if not rust_ir_available():
        return None

    command = [
        shutil.which("cargo") or "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        str(rust_ir_manifest_path()),
        "--",
        *args,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ModelSpecError(result.stderr.strip() or f"Rust IR CLI exited with code {result.returncode}")
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ModelSpecError("Rust IR CLI returned a non-object payload")
    if payload.get("ok") is False:
        raise ModelSpecError(str(payload.get("error", "Rust IR CLI reported an unknown error")))
    return payload


def rust_validate_spec_file(path: Path) -> dict[str, Any] | None:
    return run_rust_ir_cli("validate", str(path))


def rust_diff_spec_files(old_path: Path, new_path: Path) -> dict[str, Any] | None:
    return run_rust_ir_cli("diff", str(old_path), str(new_path))


def rust_migration_plan_spec_files(old_path: Path, new_path: Path) -> dict[str, Any] | None:
    return run_rust_ir_cli("migration-plan", str(old_path), str(new_path))
