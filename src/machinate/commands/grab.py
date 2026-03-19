from __future__ import annotations

import argparse
import platform

from machinate.core import (
    clean_optional,
    derive_name,
    fingerprint_path,
    is_url,
    materialize_source,
    now_utc,
    require_workspace_root,
    slugify,
    workspace_paths,
    write_json,
)
from machinate.ui import MenuChoice, can_prompt_interactively, prompt_select, prompt_text


GRAB_MODES = [
    MenuChoice("copy", "copy"),
    MenuChoice("symlink", "symlink"),
    MenuChoice("hardlink", "hardlink"),
]


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    grab_parser = subparsers.add_parser("grab", help="Stage assets into a Machinate workspace")
    grab_subparsers = grab_parser.add_subparsers(dest="grab_command", required=True)

    grab_data = grab_subparsers.add_parser("data", help="Stage a dataset into the workspace")
    grab_data.add_argument("--workspace")
    grab_data.add_argument("--src")
    grab_data.add_argument("--name")
    grab_data.add_argument("--mode")
    grab_data.set_defaults(func=cmd_grab_data)


def cmd_grab_data(args: argparse.Namespace) -> int:
    workspace_root = require_workspace_root(args.workspace)
    paths = workspace_paths(workspace_root)

    src = clean_optional(args.src)
    if src is None and can_prompt_interactively():
        src = prompt_text("Path to the dataset")
    if src is None:
        raise SystemExit("dataset source path is required; pass --src or run interactively")

    asset_name = clean_optional(args.name)
    if asset_name is None and can_prompt_interactively():
        asset_name = prompt_text("Asset name", default=derive_name(src, "dataset"))
    asset_name = slugify(asset_name or derive_name(src, "dataset"), fallback="dataset")

    mode = clean_optional(args.mode)
    if is_url(src):
        if mode is not None and mode != "download":
            raise SystemExit("URL sources only support MODE=download")
        mode = "download"
    else:
        if mode is None:
            if can_prompt_interactively():
                mode = prompt_select("Materialization mode", GRAB_MODES, default="copy")
            else:
                mode = "copy"

    destination_root = paths.data_staging_root / asset_name
    stored_path, stored_mode = materialize_source(src, destination_root, mode)
    fingerprint = fingerprint_path(stored_path)
    manifest_path = paths.asset_registry_root / f"{asset_name}.json"
    write_json(
        manifest_path,
        {
            "asset_id": asset_name,
            "asset_kind": "data",
            "source_uri_or_path": src,
            "acquisition_mode": stored_mode,
            "local_stored_path": str(stored_path),
            "sha256": fingerprint["sha256"],
            "size_bytes": fingerprint["size_bytes"],
            "machine_name": platform.node(),
            "created_at": now_utc(),
        },
    )

    print(f"data asset staged: {asset_name}")
    print(f"stored path: {stored_path}")
    print(f"manifest: {manifest_path}")
    return 0
