from __future__ import annotations

import argparse
from pathlib import Path

from machinate.core import (
    clean_optional,
    now_utc,
    parse_supported_targets,
    require_workspace_root,
    slugify,
    workspace_paths,
    write_json,
)
from machinate.ui import MenuChoice, can_prompt_interactively, prompt_select, prompt_text


PIPELINE_TYPES = [
    MenuChoice("tabular", "tabular"),
    MenuChoice("vision", "vision"),
    MenuChoice("nlp", "nlp"),
    MenuChoice("custom", "custom"),
]

PIPELINE_TEMPLATES = [
    MenuChoice("minimal", "minimal"),
    MenuChoice("makefile", "makefile"),
]


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    new_parser = subparsers.add_parser("new", help="Create new Machinate-managed objects")
    new_subparsers = new_parser.add_subparsers(dest="new_command", required=True)

    new_pipeline = new_subparsers.add_parser("pipeline", help="Create a placeholder pipeline repo")
    new_pipeline.add_argument("--workspace")
    new_pipeline.add_argument("--name")
    new_pipeline.add_argument("--type")
    new_pipeline.add_argument("--template")
    new_pipeline.add_argument("--path")
    new_pipeline.set_defaults(func=cmd_new_pipeline)


def pipeline_makefile() -> str:
    return """SHELL := /bin/bash

PYTHON ?= python3
EXPERIMENT ?= baseline

.DEFAULT_GOAL := help

.PHONY: help config.validate data.audit dev.smoke run.train

help:
\t@echo "Local pipeline targets"
\t@echo "  make config.validate EXPERIMENT=$(EXPERIMENT)"
\t@echo "  make data.audit EXPERIMENT=$(EXPERIMENT)"
\t@echo "  make dev.smoke EXPERIMENT=$(EXPERIMENT)"
\t@echo "  make run.train EXPERIMENT=$(EXPERIMENT)"

config.validate:
\t@echo "config.validate is not implemented yet for $(EXPERIMENT)"

data.audit:
\t@echo "data.audit is not implemented yet for $(EXPERIMENT)"

dev.smoke:
\t@echo "dev.smoke is not implemented yet for $(EXPERIMENT)"

run.train:
\t@echo "run.train is not implemented yet for $(EXPERIMENT)"
"""


def pipeline_readme(name: str, pipeline_type: str, template: str) -> str:
    return f"""# {name}

This placeholder pipeline was created by `macht new pipeline`.

## Metadata

- pipeline_type: {pipeline_type}
- template: {template}

## Next Steps

1. Replace the stub Makefile targets with real project logic.
2. Add runtime dependencies to `requirements.txt` or your preferred env manager.
3. Define your dataset contract and experiment configs.
4. Decide whether the repo will stay Makefile-driven or move to a native CLI.
"""


def baseline_config(name: str, pipeline_type: str) -> str:
    return """{
  "pipeline_name": "%s",
  "pipeline_type": "%s",
  "dataset_contract": {
    "kind": "csv",
    "target_column": "target"
  },
  "training": {
    "epochs": 1
  }
}
""" % (name, pipeline_type)


def cmd_new_pipeline(args: argparse.Namespace) -> int:
    workspace_root = require_workspace_root(args.workspace)
    paths = workspace_paths(workspace_root)

    raw_name = clean_optional(args.name)
    if raw_name is None and can_prompt_interactively():
        raw_name = prompt_text("Pipeline name")
    if raw_name is None:
        raise SystemExit("pipeline name is required; pass --name or run interactively")

    pipeline_name = raw_name
    pipeline_slug = slugify(pipeline_name, fallback="pipeline")

    pipeline_type = clean_optional(args.type)
    if pipeline_type is None:
        if can_prompt_interactively():
            pipeline_type = prompt_select("Pipeline type", PIPELINE_TYPES, default="tabular")
        else:
            pipeline_type = "tabular"

    template = clean_optional(args.template)
    if template is None:
        if can_prompt_interactively():
            template = prompt_select("Pipeline template", PIPELINE_TEMPLATES, default="minimal")
        else:
            template = "minimal"

    default_repo_path = paths.pipeline_root / pipeline_slug
    repo_path_text = clean_optional(args.path)
    if repo_path_text is None and can_prompt_interactively():
        repo_path_text = prompt_text("Pipeline repo path", default=str(default_repo_path))
    repo_path = Path(repo_path_text or default_repo_path).expanduser().resolve()

    if repo_path.exists() and any(repo_path.iterdir()):
        raise SystemExit(f"pipeline path already exists and is not empty: {repo_path}")

    package_slug = pipeline_slug.replace("-", "_").replace(".", "_")
    makefile_path = repo_path / "Makefile"
    manifest_path = paths.pipeline_registry_root / f"{pipeline_slug}.json"
    if manifest_path.exists():
        raise SystemExit(f"pipeline registry entry already exists: {manifest_path}")

    (repo_path / "configs" / "experiments").mkdir(parents=True, exist_ok=True)
    (repo_path / "src" / package_slug).mkdir(parents=True, exist_ok=True)

    makefile_path.write_text(pipeline_makefile())
    (repo_path / "README.md").write_text(pipeline_readme(pipeline_name, pipeline_type, template))
    (repo_path / ".gitignore").write_text("__pycache__/\n*.py[cod]\n.venv/\noutputs/\n")
    (repo_path / "requirements.txt").write_text("# Add runtime dependencies here.\n")
    (repo_path / "src" / package_slug / "__init__.py").write_text(f'"""Pipeline package for {pipeline_name}."""\n')
    (repo_path / "configs" / "experiments" / "baseline.json").write_text(
        baseline_config(pipeline_name, pipeline_type)
    )

    supported_commands = parse_supported_targets(makefile_path)
    write_json(
        manifest_path,
        {
            "pipeline_name": pipeline_name,
            "pipeline_slug": pipeline_slug,
            "repo_path": str(repo_path),
            "pipeline_type": pipeline_type,
            "template": template,
            "created_at": now_utc(),
            "supported_commands": supported_commands,
        },
    )

    print(f"pipeline created: {pipeline_name}")
    print(f"repo path: {repo_path}")
    print(f"registry manifest: {manifest_path}")
    return 0
