from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
import platform
import re
import shutil
from pathlib import Path


WORKSPACE_SENTINEL = Path(".machinate/workspace.json")


@dataclass(frozen=True)
class AppPaths:
    home: Path = Path.home()

    @property
    def config_root(self) -> Path:
        xdg_home = os.environ.get("XDG_CONFIG_HOME")
        if xdg_home:
            return Path(xdg_home).expanduser().resolve() / "machinate"
        return self.home / ".config" / "machinate"

    @property
    def config_path(self) -> Path:
        return self.config_root / "config.json"


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path

    @property
    def metadata_root(self) -> Path:
        return self.root / ".machinate"

    @property
    def workspace_manifest(self) -> Path:
        return self.metadata_root / "workspace.json"

    @property
    def pipeline_registry_root(self) -> Path:
        return self.metadata_root / "pipelines"

    @property
    def asset_registry_root(self) -> Path:
        return self.metadata_root / "assets"

    @property
    def env_root(self) -> Path:
        return self.root / ".envs" / "venvs"

    @property
    def data_staging_root(self) -> Path:
        return self.root / "data" / "staged"

    @property
    def output_root(self) -> Path:
        return self.root / "outputs"

    @property
    def pipeline_root(self) -> Path:
        return self.root / "pipelines"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def slugify(raw: str, fallback: str = "item") -> str:
    cleaned = raw.strip().lower()
    slug = re.sub(r"[^a-z0-9._-]+", "-", cleaned).strip("-")
    return slug or fallback


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def app_paths() -> AppPaths:
    return AppPaths()


def ensure_global_config() -> Path:
    paths = app_paths()
    paths.config_root.mkdir(parents=True, exist_ok=True)
    if not paths.config_path.exists():
        write_json(
            paths.config_path,
            {
                "created_at": now_utc(),
                "default_python": shutil.which("python3") or "",
                "platform": platform.platform(),
            },
        )
    return paths.config_path


def workspace_paths(root: Path) -> WorkspacePaths:
    return WorkspacePaths(root=root.resolve())


def find_workspace_root(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).expanduser().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / WORKSPACE_SENTINEL).exists():
            return candidate
    return None


def require_workspace_root(explicit_root: str | None = None) -> Path:
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
        detected = find_workspace_root(root)
        if detected is None:
            raise SystemExit(f"workspace marker missing from path or its parents: {root}")
        return detected

    detected = find_workspace_root()
    if detected is None:
        raise SystemExit("no Machinate workspace found from the current directory; run `macht workspace init` first")
    return detected


def ensure_workspace_layout(root: Path, workspace_name: str) -> WorkspacePaths:
    paths = workspace_paths(root)
    for directory in (
        paths.metadata_root,
        paths.pipeline_registry_root,
        paths.asset_registry_root,
        paths.env_root,
        paths.data_staging_root,
        paths.output_root / "reports",
        paths.output_root / "runs",
        paths.pipeline_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    write_json(
        paths.workspace_manifest,
        {
            "schema_version": 1,
            "workspace_name": workspace_name,
            "workspace_root": str(paths.root),
            "created_at": now_utc(),
            "machine_name": platform.node(),
            "default_python": shutil.which("python3") or "",
        },
    )
    return paths


def parse_supported_targets(makefile_path: Path) -> list[str]:
    targets: list[str] = []
    for raw_line in makefile_path.read_text().splitlines():
        line = raw_line.strip()
        if not line.startswith(".PHONY:"):
            continue
        for token in line.split(":", 1)[1].split():
            if token not in targets:
                targets.append(token)
    return targets


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_path(path: Path) -> dict[str, object]:
    if path.is_file():
        return {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    digest = sha256()
    total_size = 0
    files = sorted(file_path for file_path in path.rglob("*") if file_path.is_file())
    for file_path in files:
        rel = file_path.relative_to(path).as_posix()
        file_hash = sha256_file(file_path)
        digest.update(rel.encode("utf-8"))
        digest.update(file_hash.encode("utf-8"))
        total_size += file_path.stat().st_size
    return {
        "sha256": digest.hexdigest(),
        "size_bytes": total_size,
    }


def derive_name(raw: str, fallback: str) -> str:
    name = Path(raw.rstrip("/")).name
    if not name:
        return fallback
    return slugify(name, fallback=fallback)


def materialize_source(src: str, destination_root: Path, mode: str) -> tuple[Path, str]:
    source_path = Path(src).expanduser().resolve()
    if not source_path.exists():
        raise SystemExit(f"source path does not exist: {source_path}")

    normalized_mode = mode.lower()
    if normalized_mode not in {"copy", "symlink", "hardlink"}:
        raise SystemExit("MODE must be one of `copy`, `symlink`, or `hardlink`")

    if destination_root.exists():
        raise SystemExit(f"destination already exists: {destination_root}")

    destination_root.parent.mkdir(parents=True, exist_ok=True)

    if source_path.is_dir():
        if normalized_mode == "symlink":
            destination_root.symlink_to(source_path, target_is_directory=True)
            return destination_root, "symlink"
        if normalized_mode == "hardlink":
            shutil.copytree(source_path, destination_root, copy_function=os.link)
            return destination_root, "hardlink"
        shutil.copytree(source_path, destination_root)
        return destination_root, "copy"

    destination_root.mkdir(parents=True, exist_ok=True)
    destination_path = destination_root / source_path.name
    if normalized_mode == "symlink":
        destination_path.symlink_to(source_path)
        return destination_path, "symlink"
    if normalized_mode == "hardlink":
        os.link(source_path, destination_path)
        return destination_path, "hardlink"
    shutil.copy2(source_path, destination_path)
    return destination_path, "copy"
