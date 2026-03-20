from __future__ import annotations

import argparse
from pathlib import Path

from machinator.core import ensure_global_config, ensure_workspace_layout, find_workspace_root, load_json
from machinator.ui import can_prompt_interactively, prompt_text


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    workspace_parser = subparsers.add_parser("workspace", help="Manage Machinator workspaces")
    workspace_subparsers = workspace_parser.add_subparsers(dest="workspace_command", required=True)

    workspace_init = workspace_subparsers.add_parser("init", help="Create a Machinator workspace scaffold")
    workspace_init.add_argument("--path")
    workspace_init.add_argument("--name")
    workspace_init.set_defaults(func=cmd_workspace_init)

    workspace_show = workspace_subparsers.add_parser("show", help="Show the detected workspace")
    workspace_show.add_argument("--path")
    workspace_show.set_defaults(func=cmd_workspace_show)


def cmd_workspace_init(args: argparse.Namespace) -> int:
    root = Path(args.path or ".").expanduser().resolve()
    name = args.name
    if not name and can_prompt_interactively():
        name = prompt_text("Workspace name", default=root.name)
    name = name or root.name

    config_path = ensure_global_config()
    paths = ensure_workspace_layout(root, name)

    print(f"workspace initialized: {paths.root}")
    print(f"workspace manifest: {paths.workspace_manifest}")
    print(f"global config: {config_path}")
    return 0


def cmd_workspace_show(args: argparse.Namespace) -> int:
    root = find_workspace_root(Path(args.path).expanduser().resolve()) if args.path else find_workspace_root()
    if root is None:
        raise SystemExit("no Machinator workspace detected")

    manifest_path = root / ".machinator" / "workspace.json"
    if not manifest_path.exists():
        raise SystemExit(f"workspace manifest missing: {manifest_path}")

    manifest = load_json(manifest_path)
    print("Machinator workspace")
    print(f"  name: {manifest.get('workspace_name', '')}")
    print(f"  root: {manifest.get('workspace_root', '')}")
    print(f"  created_at: {manifest.get('created_at', '')}")
    print(f"  machine_name: {manifest.get('machine_name', '')}")
    return 0
