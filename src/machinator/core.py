from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import importlib.resources
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


WORKSPACE_SENTINEL = Path(".machinator/workspace.json")
PRIMARY_PIPELINE_SENTINEL = Path("machinate.toml")
LEGACY_PIPELINE_SENTINEL = Path("machinator.toml")
PIPELINE_SENTINELS = (PRIMARY_PIPELINE_SENTINEL, LEGACY_PIPELINE_SENTINEL)
PROJECT_SENTINEL = Path("pyproject.toml")


@dataclass(frozen=True)
class AppPaths:
    home: Path = Path.home()

    @property
    def config_root(self) -> Path:
        xdg_home = os.environ.get("XDG_CONFIG_HOME")
        if xdg_home:
            return Path(xdg_home).expanduser().resolve() / "machinator"
        return self.home / ".config" / "machinator"

    @property
    def config_path(self) -> Path:
        return self.config_root / "config.json"


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path

    @property
    def metadata_root(self) -> Path:
        return self.root / ".machinator"

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


@dataclass(frozen=True)
class PipelineTaskContext:
    workspace_root: Path | None
    pipeline_root: Path
    pipeline_config_path: Path
    pipeline_config: dict[str, Any]
    task_name: str
    experiment_name: str | None
    experiment_config_path: Path | None
    experiment_config: dict[str, Any]
    dataset_ref: str | None
    dataset_path: Path | None
    output_root: Path

    def require_dataset_path(self) -> Path:
        if self.dataset_path is None:
            raise ValueError("This task requires a dataset, but none was provided.")
        return self.dataset_path

    def require_experiment_config(self) -> dict[str, Any]:
        if not self.experiment_config:
            raise ValueError("This task requires an experiment configuration, but none was resolved.")
        return self.experiment_config

    def write_json_artifact(self, category: str, stem: str, payload: dict[str, object]) -> Path:
        category_dir = self.output_root / category
        category_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        artifact_path = category_dir / f"{stamp}_{stem}.json"
        write_json(artifact_path, payload)
        return artifact_path


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


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


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


def find_pipeline_root(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).expanduser().resolve()
    for candidate in [current, *current.parents]:
        if any((candidate / sentinel).exists() for sentinel in PIPELINE_SENTINELS):
            return candidate
    return None


def find_project_root(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).expanduser().resolve()
    for candidate in [current, *current.parents]:
        if not (candidate / PROJECT_SENTINEL).exists():
            continue
        if (candidate / "src" / "machinator" / "cli.py").exists():
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
        raise SystemExit("no Machinator workspace found from the current directory; run `macht workspace init` first")
    return detected


def require_pipeline_root(explicit_root: str | None = None) -> Path:
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
        detected = find_pipeline_root(root)
        if detected is None:
            raise SystemExit(f"pipeline marker missing from path or its parents: {root}")
        return detected

    detected = find_pipeline_root()
    if detected is None:
        raise SystemExit(
            "no Machinator pipeline found from the current directory; run inside a pipeline repo with `machinate.toml` "
            "or pass --pipeline-path"
        )
    return detected


def require_project_root(explicit_root: str | None = None) -> Path:
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
        detected = find_project_root(root)
        if detected is None:
            raise SystemExit(f"Machinator project root missing from path or its parents: {root}")
        return detected

    detected = find_project_root()
    if detected is None:
        raise SystemExit("no Machinator project root found from the current directory; run inside the Machinator repo or pass --root")
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


def registered_pipeline_manifests(workspace_root: Path) -> list[dict[str, object]]:
    root = workspace_paths(workspace_root).pipeline_registry_root
    manifests: list[dict[str, object]] = []
    for manifest_path in sorted(root.glob("*.json")):
        manifests.append(load_json(manifest_path))
    return manifests


def registered_asset_manifests(workspace_root: Path) -> list[dict[str, object]]:
    root = workspace_paths(workspace_root).asset_registry_root
    manifests: list[dict[str, object]] = []
    for manifest_path in sorted(root.glob("*.json")):
        manifests.append(load_json(manifest_path))
    return manifests


def workspace_pipeline_manifest_path(workspace_root: Path, pipeline_slug: str) -> Path:
    return workspace_paths(workspace_root).pipeline_registry_root / f"{pipeline_slug}.json"


def load_workspace_pipeline_manifest(workspace_root: Path, pipeline_slug: str) -> dict[str, object]:
    manifest_path = workspace_pipeline_manifest_path(workspace_root, pipeline_slug)
    if not manifest_path.exists():
        raise SystemExit(f"pipeline `{pipeline_slug}` is not registered in workspace `{workspace_root}`")
    return load_json(manifest_path)


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


def is_url(src: str) -> bool:
    parsed = urlparse(src)
    return parsed.scheme in {"http", "https"}


def default_download_filename(url: str, fallback: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if not name:
        return fallback
    return name


def js_fetch_helper_path() -> Path:
    return Path(str(importlib.resources.files("machinator").joinpath("js/fetch_url.mjs")))


def download_url_with_node(url: str, destination_root: Path, fallback_name: str) -> Path:
    node_bin = shutil.which("node")
    if node_bin is None:
        raise SystemExit("URL downloads require `node` on PATH")

    destination_root.mkdir(parents=True, exist_ok=True)
    destination_path = destination_root / default_download_filename(url, fallback_name)
    helper_path = js_fetch_helper_path()
    result = subprocess.run(
        [node_bin, str(helper_path), url, str(destination_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    output_path = destination_path
    stdout = result.stdout.strip()
    if stdout:
        try:
            payload = json.loads(stdout)
            candidate = clean_optional(str(payload.get("output_path", "")))
            if candidate:
                output_path = Path(candidate)
        except json.JSONDecodeError:
            pass
    if not output_path.exists():
        raise SystemExit(f"download helper did not produce the expected file: {output_path}")
    return output_path


def materialize_source(src: str, destination_root: Path, mode: str) -> tuple[Path, str]:
    if is_url(src):
        if destination_root.exists():
            raise SystemExit(f"destination already exists: {destination_root}")
        downloaded_path = download_url_with_node(src, destination_root, "downloaded_asset")
        return downloaded_path, "download"

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


def pipeline_config_path(pipeline_root: Path) -> Path:
    # `machinate.toml` is now the canonical recipe key for a usable pipeline, but
    # we still fall back to the legacy filename so older local test fixtures do
    # not break while the repo moves over.
    for sentinel in PIPELINE_SENTINELS:
        candidate = pipeline_root / sentinel
        if candidate.exists():
            return candidate
    return pipeline_root / PRIMARY_PIPELINE_SENTINEL


def load_pipeline_config(pipeline_root: Path) -> dict[str, Any]:
    config_path = pipeline_config_path(pipeline_root)
    if not config_path.exists():
        raise SystemExit(f"pipeline config missing: {config_path}")
    return load_toml(config_path)


def resolve_pipeline_root(
    *,
    workspace_root: Path | None,
    pipeline_name: str | None,
    pipeline_path: str | None,
) -> tuple[Path, Path | None]:
    if pipeline_path:
        resolved = require_pipeline_root(pipeline_path)
        detected_workspace = find_workspace_root(resolved)
        return resolved, detected_workspace

    detected_pipeline = find_pipeline_root()
    if detected_pipeline is not None:
        return detected_pipeline, find_workspace_root(detected_pipeline)

    if pipeline_name:
        if workspace_root is None:
            workspace_root = require_workspace_root()
        manifest = load_workspace_pipeline_manifest(workspace_root, pipeline_name)
        return require_pipeline_root(str(manifest["repo_path"])), workspace_root

    raise SystemExit("no pipeline selected; run inside a pipeline repo or pass --pipeline/--pipeline-path")


def pipeline_paths_from_config(pipeline_root: Path, pipeline_config: dict[str, Any]) -> dict[str, Path]:
    raw_paths = pipeline_config.get("paths", {})
    if not isinstance(raw_paths, dict):
        raw_paths = {}
    source_root = pipeline_root / str(raw_paths.get("source_root", "src"))
    data_root = pipeline_root / str(raw_paths.get("data_root", "data"))
    config_root = pipeline_root / str(raw_paths.get("config_root", "config"))
    experiment_root = pipeline_root / str(raw_paths.get("experiments", "config"))
    output_root = pipeline_root / str(raw_paths.get("outputs", "outputs"))
    return {
        "source_root": source_root,
        "data_root": data_root,
        "config_root": config_root,
        "experiment_root": experiment_root,
        "output_root": output_root,
    }


def pipeline_tasks(pipeline_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_tasks = pipeline_config.get("tasks", {})
    if not isinstance(raw_tasks, dict):
        raise SystemExit("pipeline config is missing a valid [tasks] section")
    tasks: dict[str, dict[str, Any]] = {}
    for name, payload in raw_tasks.items():
        if isinstance(payload, dict):
            tasks[str(name)] = dict(payload)
    return tasks


def discover_experiment_configs(pipeline_root: Path, pipeline_config: dict[str, Any]) -> list[Path]:
    paths = pipeline_paths_from_config(pipeline_root, pipeline_config)
    experiment_root = paths["experiment_root"]
    if not experiment_root.exists():
        return []
    configs: list[Path] = []
    for config_path in sorted(experiment_root.glob("*.toml")):
        if config_path.is_file():
            configs.append(config_path)
    return configs


def resolve_experiment_config(
    pipeline_root: Path,
    pipeline_config: dict[str, Any],
    experiment_name: str | None,
) -> tuple[str | None, Path | None, dict[str, Any]]:
    available = discover_experiment_configs(pipeline_root, pipeline_config)
    if experiment_name:
        target = next((path for path in available if path.stem == experiment_name), None)
        if target is None:
            raise SystemExit(f"unknown experiment `{experiment_name}`")
        return experiment_name, target, load_toml(target)

    if not available:
        return None, None, {}

    default_config = next((path for path in available if path.stem == "baseline"), available[0])
    return default_config.stem, default_config, load_toml(default_config)


def resolve_dataset_path(workspace_root: Path | None, dataset_ref: str | None) -> tuple[str | None, Path | None]:
    cleaned = clean_optional(dataset_ref)
    if cleaned:
        candidate = Path(cleaned).expanduser().resolve()
        if candidate.exists():
            return cleaned, candidate
        if workspace_root is not None:
            asset_manifest = workspace_paths(workspace_root).asset_registry_root / f"{slugify(cleaned)}.json"
            if asset_manifest.exists():
                manifest = load_json(asset_manifest)
                return cleaned, Path(str(manifest["local_stored_path"]))
        raise SystemExit(f"unknown dataset reference `{cleaned}`")

    return None, None


def load_task_callable(
    pipeline_root: Path,
    pipeline_config: dict[str, Any],
    task_name: str,
) -> tuple[Callable[[PipelineTaskContext], object], dict[str, Any]]:
    tasks = pipeline_tasks(pipeline_config)
    task_config = tasks.get(task_name)
    if task_config is None:
        raise SystemExit(f"unknown pipeline task `{task_name}`")

    entry = clean_optional(str(task_config.get("entry", "")))
    if not entry or ":" not in entry:
        raise SystemExit(f"task `{task_name}` has an invalid entrypoint")

    module_name, func_name = entry.split(":", 1)
    source_root = pipeline_paths_from_config(pipeline_root, pipeline_config)["source_root"]
    if source_root.exists() and str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    module = importlib.import_module(module_name)
    func = getattr(module, func_name, None)
    if func is None or not callable(func):
        raise SystemExit(f"task entrypoint `{entry}` is not callable")
    return func, task_config


def build_task_context(
    *,
    workspace_root: Path | None,
    pipeline_root: Path,
    pipeline_config: dict[str, Any],
    task_name: str,
    experiment_name: str | None,
    dataset_ref: str | None,
) -> PipelineTaskContext:
    resolved_experiment_name, experiment_config_path, experiment_config = resolve_experiment_config(
        pipeline_root, pipeline_config, experiment_name
    )
    resolved_dataset_ref, dataset_path = resolve_dataset_path(workspace_root, dataset_ref)
    resolved_paths = pipeline_paths_from_config(pipeline_root, pipeline_config)
    if dataset_path is None:
        # Recipe-centric pipelines keep a local `data/` directory. When the user
        # has not explicitly selected a workspace asset, default to that local
        # dataset root so example pipelines are runnable in place.
        local_data_root = resolved_paths["data_root"]
        if local_data_root.exists() and any(local_data_root.iterdir()):
            resolved_dataset_ref = "pipeline-local-data"
            dataset_path = local_data_root
    output_root = resolved_paths["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    return PipelineTaskContext(
        workspace_root=workspace_root,
        pipeline_root=pipeline_root,
        pipeline_config_path=pipeline_config_path(pipeline_root),
        pipeline_config=pipeline_config,
        task_name=task_name,
        experiment_name=resolved_experiment_name,
        experiment_config_path=experiment_config_path,
        experiment_config=experiment_config,
        dataset_ref=resolved_dataset_ref,
        dataset_path=dataset_path,
        output_root=output_root,
    )
