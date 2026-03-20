from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from machinator.core import (
    clean_optional,
    now_utc,
    registered_asset_manifests,
    require_workspace_root,
    resolve_dataset_path,
    slugify,
    workspace_paths,
    write_json,
)
from machinator.ui import (
    MenuChoice,
    can_prompt_interactively,
    prompt_multiline,
    prompt_select,
    prompt_text,
)


DATASET_MANUAL_ENTRY = "__manual__"
REPORT_KIND_DATA = "data"
SUPPORTED_REPORT_KINDS = [MenuChoice("data", REPORT_KIND_DATA)]


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    legate_parser = subparsers.add_parser(
        "legate",
        help="Delegate structured tasks to a non-interactive agent",
    )
    legate_subparsers = legate_parser.add_subparsers(dest="legate_command", required=True)

    report_parser = legate_subparsers.add_parser(
        "report",
        help="Generate a structured report through a delegated agent task",
    )
    report_parser.add_argument("--workspace")
    report_parser.add_argument("--provider", default="codex", choices=["codex"])
    report_parser.add_argument("--data", action="store_true", help="Generate a data report")
    report_parser.add_argument("--dataset", help="Dataset asset id or local path")
    report_parser.add_argument("--notes", help="Plain-English notes to hand to the agent")
    report_parser.add_argument("--notes-file", help="Read notes from a local text or markdown file")
    report_parser.add_argument("--model", help="Optional Codex model override")
    report_parser.add_argument(
        "--sandbox",
        default="read-only",
        choices=["read-only", "workspace-write", "danger-full-access"],
        help="Codex sandbox mode for the delegated task",
    )
    report_parser.add_argument("--name", help="Optional output stem override")
    report_parser.set_defaults(func=cmd_report)


def resolve_report_kind(args: argparse.Namespace) -> str:
    if args.data:
        return REPORT_KIND_DATA
    if can_prompt_interactively():
        return prompt_select("Select a report kind", SUPPORTED_REPORT_KINDS, default=REPORT_KIND_DATA)
    return REPORT_KIND_DATA


def resolve_dataset_argument(args: argparse.Namespace, workspace_root: Path) -> str:
    explicit = clean_optional(args.dataset)
    if explicit is not None:
        return explicit

    if not can_prompt_interactively():
        raise SystemExit("dataset is required for `macht legate report --data`; pass --dataset")

    manifests = [
        manifest
        for manifest in registered_asset_manifests(workspace_root)
        if str(manifest.get("asset_kind", "")) == "data"
    ]
    if manifests:
        choices = [
            MenuChoice(
                f"{manifest['asset_id']} ({manifest['local_stored_path']})",
                str(manifest["asset_id"]),
            )
            for manifest in manifests
        ]
        choices.append(MenuChoice("Enter a local path manually", DATASET_MANUAL_ENTRY))
        selection = prompt_select(
            "Select the dataset for this report",
            choices,
            default=str(manifests[0]["asset_id"]),
            use_search_filter=len(choices) > 8,
        )
        if selection != DATASET_MANUAL_ENTRY:
            return selection

    return prompt_text("Dataset asset name or local path")


def resolve_notes(args: argparse.Namespace) -> str:
    inline_notes = clean_optional(args.notes)
    notes_file = clean_optional(args.notes_file)
    if inline_notes is not None and notes_file is not None:
        raise SystemExit("pass either --notes or --notes-file, not both")

    if notes_file is not None:
        notes_path = Path(notes_file).expanduser().resolve()
        if not notes_path.exists():
            raise SystemExit(f"notes file does not exist: {notes_path}")
        return notes_path.read_text().strip()

    if inline_notes is not None:
        return inline_notes

    if can_prompt_interactively():
        return prompt_multiline("Describe everything you know about this data.")

    return ""


def legate_report_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def data_report_schema() -> dict[str, object]:
    string_list = {"type": "array", "items": {"type": "string"}}
    nullable_integer = {"type": ["integer", "null"]}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["plain_summary", "data_report"],
        "properties": {
            "plain_summary": {"type": "string"},
            "data_report": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "report_kind",
                    "dataset_name",
                    "dataset_path",
                    "user_context",
                    "overview",
                    "suspected_domain",
                    "suspected_problem_type",
                    "files_reviewed",
                    "structure",
                    "quality",
                    "recommended_checks",
                    "recommended_next_steps",
                    "confidence",
                ],
                "properties": {
                    "report_kind": {"type": "string", "enum": ["data"]},
                    "dataset_name": {"type": "string"},
                    "dataset_path": {"type": "string"},
                    "user_context": {"type": "string"},
                    "overview": {"type": "string"},
                    "suspected_domain": {"type": "string"},
                    "suspected_problem_type": {"type": "string"},
                    "files_reviewed": string_list,
                    "structure": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "format",
                            "file_count",
                            "primary_file",
                            "row_count_estimate",
                            "column_count",
                            "columns",
                            "target_candidates",
                            "id_candidates",
                            "time_candidates",
                        ],
                        "properties": {
                            "format": {"type": "string"},
                            "file_count": {"type": "integer"},
                            "primary_file": {"type": "string"},
                            "row_count_estimate": nullable_integer,
                            "column_count": nullable_integer,
                            "columns": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["name", "dtype_guess", "role_guess", "notes"],
                                    "properties": {
                                        "name": {"type": "string"},
                                        "dtype_guess": {"type": "string"},
                                        "role_guess": {"type": "string"},
                                        "notes": {"type": "string"},
                                    },
                                },
                            },
                            "target_candidates": string_list,
                            "id_candidates": string_list,
                            "time_candidates": string_list,
                            "image": {
                                "type": ["object", "null"],
                                "additionalProperties": False,
                                "required": ["channels", "height", "width", "class_names"],
                                "properties": {
                                    "channels": nullable_integer,
                                    "height": nullable_integer,
                                    "width": nullable_integer,
                                    "class_names": string_list,
                                },
                            },
                        },
                    },
                    "quality": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "missingness_risks",
                            "leakage_risks",
                            "label_risks",
                            "privacy_risks",
                            "general_risks",
                        ],
                        "properties": {
                            "missingness_risks": string_list,
                            "leakage_risks": string_list,
                            "label_risks": string_list,
                            "privacy_risks": string_list,
                            "general_risks": string_list,
                        },
                    },
                    "recommended_checks": string_list,
                    "recommended_next_steps": string_list,
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                },
            },
        },
    }


def build_data_report_prompt(
    *,
    workspace_root: Path,
    dataset_ref: str,
    dataset_path: Path,
    user_notes: str,
) -> str:
    notes_block = user_notes.strip() or "(no user notes provided)"
    return f"""You are acting as a Machinator delegated reporting agent running through Codex CLI in non-interactive exec mode.

Objective:
- Inspect the dataset at `{dataset_path}`.
- Use local shell and Python commands as needed to understand the dataset.
- Produce a structured JSON data report for Machinator.

Hard requirements:
- Do not modify any files.
- Use the user notes only as hints; verify against the observed data.
- Be conservative when uncertain and state uncertainty explicitly.
- Use absolute paths in `files_reviewed`.
- Return only JSON that matches the provided schema.

Context:
- workspace_root: {workspace_root}
- dataset_reference: {dataset_ref}
- dataset_path: {dataset_path}
- user_notes:
{notes_block}

Report expectations:
- Summarize what the dataset appears to be for in plain English.
- Identify likely target, ID, and time-related fields when possible.
- For image datasets, populate `structure.image` with channels, height, width, and class names when you can infer them.
- Call out data quality, leakage, privacy, and labeling risks.
- Recommend the next checks needed before building a pipeline.
- Keep `plain_summary` concise enough to print directly in a terminal window.
"""


def codex_working_root(workspace_root: Path, dataset_path: Path) -> Path:
    if dataset_path.is_relative_to(workspace_root):
        return workspace_root
    return dataset_path if dataset_path.is_dir() else dataset_path.parent


def parse_last_message(path: Path) -> dict[str, Any]:
    raw = path.read_text().strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SystemExit("Codex returned a non-object payload for the structured report")
    return payload


def run_codex_structured(
    *,
    prompt: str,
    schema_path: Path,
    last_message_path: Path,
    working_root: Path,
    model: str | None,
    sandbox: str,
) -> None:
    codex_bin = shutil.which("codex")
    if codex_bin is None:
        raise SystemExit("`codex` is not available on PATH; `macht legate` currently requires Codex CLI")

    command = [
        codex_bin,
        "exec",
        "-",
        "-C",
        str(working_root),
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox",
        sandbox,
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(last_message_path),
    ]
    if model:
        command.extend(["--model", model])

    try:
        subprocess.run(command, input=prompt, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Codex delegated task failed with exit code {exc.returncode}") from exc


def cmd_report(args: argparse.Namespace) -> int:
    workspace_root = require_workspace_root(args.workspace)
    report_kind = resolve_report_kind(args)
    if report_kind != REPORT_KIND_DATA:
        raise SystemExit(f"unsupported report kind `{report_kind}`")

    dataset_input = resolve_dataset_argument(args, workspace_root)
    dataset_ref, dataset_path = resolve_dataset_path(workspace_root, dataset_input)
    if dataset_ref is None or dataset_path is None:
        raise SystemExit("could not resolve the dataset reference for the delegated report")

    user_notes = resolve_notes(args)
    output_root = workspace_paths(workspace_root).output_root / "reports" / "legate"
    output_root.mkdir(parents=True, exist_ok=True)

    base_name = clean_optional(args.name) or f"data-report-{dataset_ref}"
    artifact_stem = f"{legate_report_stamp()}_{slugify(base_name, fallback='data-report')}"
    prompt_path = output_root / f"{artifact_stem}.prompt.txt"
    raw_response_path = output_root / f"{artifact_stem}.raw.json"
    artifact_path = output_root / f"{artifact_stem}.json"

    prompt = build_data_report_prompt(
        workspace_root=workspace_root,
        dataset_ref=dataset_ref,
        dataset_path=dataset_path,
        user_notes=user_notes,
    )
    prompt_path.write_text(prompt)

    with tempfile.TemporaryDirectory(prefix="machinator-legate-") as tempdir:
        temp_root = Path(tempdir)
        schema_path = temp_root / "data-report.schema.json"
        last_message_path = temp_root / "codex-last-message.json"
        schema_path.write_text(json.dumps(data_report_schema(), indent=2) + "\n")

        run_codex_structured(
            prompt=prompt,
            schema_path=schema_path,
            last_message_path=last_message_path,
            working_root=codex_working_root(workspace_root, dataset_path),
            model=clean_optional(args.model),
            sandbox=args.sandbox,
        )

        raw_response = last_message_path.read_text()
        raw_response_path.write_text(raw_response)
        structured = parse_last_message(last_message_path)

    artifact_payload = {
        "schema_version": 1,
        "generated_at": now_utc(),
        "delegate_provider": args.provider,
        "delegate_kind": "report",
        "report_kind": report_kind,
        "workspace_root": str(workspace_root),
        "dataset_ref": dataset_ref,
        "dataset_path": str(dataset_path),
        "model": clean_optional(args.model) or "",
        "sandbox": args.sandbox,
        "user_notes": user_notes,
        "plain_summary": structured["plain_summary"],
        "report": structured["data_report"],
        "prompt_path": str(prompt_path),
        "raw_response_path": str(raw_response_path),
    }
    write_json(artifact_path, artifact_payload)

    print("legate report ready")
    print(f"provider: {args.provider}")
    print(f"dataset: {dataset_ref}")
    print(f"report: {artifact_path}")
    print(f"prompt: {prompt_path}")
    print(f"raw response: {raw_response_path}")
    print("")
    print(str(structured["plain_summary"]).strip())

    recommended_next_steps = structured["data_report"].get("recommended_next_steps", [])
    if isinstance(recommended_next_steps, list) and recommended_next_steps:
        print("")
        print("recommended next steps")
        for item in recommended_next_steps:
            print(f"  - {item}")
    return 0
