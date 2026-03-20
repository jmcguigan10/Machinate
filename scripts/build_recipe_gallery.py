#!/usr/bin/env python3

from __future__ import annotations

import csv
import gzip
import json
import shutil
import struct
import sys
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = PROJECT_ROOT / "examples" / "recipe-gallery-workspace"
DOWNLOAD_ROOT = PROJECT_ROOT / ".tmp" / "recipe-downloads"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from machinator.cli import main as mach_main  # noqa: E402


TITANIC_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
PENGUINS_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
SMS_SPAM_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
FASHION_IMAGES_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz"
FASHION_LABELS_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz"


def download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "Machinator recipe gallery builder"})
    with urlopen(request) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return destination


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def reset_workspace() -> None:
    if EXAMPLES_ROOT.exists():
        shutil.rmtree(EXAMPLES_ROOT)
    EXAMPLES_ROOT.parent.mkdir(parents=True, exist_ok=True)
    result = mach_main(["workspace", "init", "--path", str(EXAMPLES_ROOT), "--name", "recipe-gallery-workspace"])
    if result != 0:
        raise SystemExit(f"workspace init failed with exit code {result}")


def stage_url_asset(asset_name: str, url: str, filename: str) -> Path:
    destination = EXAMPLES_ROOT / "data" / "staged" / asset_name / filename
    print(f"downloading: {url}")
    return download(url, destination)


def build_titanic_dataset() -> tuple[Path, Path]:
    dataset_path = stage_url_asset("titanic", TITANIC_URL, "titanic.csv")
    with dataset_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        rows = list(reader)
    report_path = EXAMPLES_ROOT / "outputs" / "reports" / "fixtures" / "titanic-report.json"
    write_json(
        report_path,
        {
            "report": {
                "dataset_name": "titanic",
                "dataset_path": str(dataset_path),
                "suspected_domain": "tabular binary classification",
                "suspected_problem_type": "binary classification",
                "structure": {
                    "format": "csv",
                    "file_count": 1,
                    "primary_file": str(dataset_path),
                    "row_count_estimate": len(rows),
                    "column_count": len(headers),
                    "columns": [
                        {
                            "name": header,
                            "dtype_guess": "string",
                            "role_guess": "target" if header == "survived" else "feature",
                            "notes": "",
                        }
                        for header in headers
                    ],
                    "target_candidates": ["survived"],
                    "id_candidates": [],
                    "time_candidates": [],
                },
            }
        },
    )
    return dataset_path, report_path


def build_penguins_dataset() -> tuple[Path, Path]:
    dataset_path = stage_url_asset("penguins", PENGUINS_URL, "penguins.csv")
    with dataset_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        rows = [row for row in reader if (row.get("sex") or "").strip()]
    filtered_path = dataset_path.parent / "penguins_binary.csv"
    with filtered_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    report_path = EXAMPLES_ROOT / "outputs" / "reports" / "fixtures" / "penguins-report.json"
    write_json(
        report_path,
        {
            "report": {
                "dataset_name": "penguins-sex",
                "dataset_path": str(filtered_path),
                "suspected_domain": "tabular binary classification",
                "suspected_problem_type": "binary classification",
                "structure": {
                    "format": "csv",
                    "file_count": 1,
                    "primary_file": str(filtered_path),
                    "row_count_estimate": len(rows),
                    "column_count": len(headers),
                    "columns": [
                        {
                            "name": header,
                            "dtype_guess": "string",
                            "role_guess": "target" if header == "sex" else "feature",
                            "notes": "",
                        }
                        for header in headers
                    ],
                    "target_candidates": ["sex"],
                    "id_candidates": [],
                    "time_candidates": [],
                },
            }
        },
    )
    return filtered_path, report_path


def build_sms_dataset() -> tuple[Path, Path]:
    archive_path = DOWNLOAD_ROOT / "sms-spam-collection.zip"
    download(SMS_SPAM_URL, archive_path)
    dataset_dir = EXAMPLES_ROOT / "data" / "staged" / "sms-spam"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "sms_spam.csv"
    with zipfile.ZipFile(archive_path) as archive, dataset_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "text"])
        writer.writeheader()
        with archive.open("SMSSpamCollection") as source:
            for raw_line in source.read().decode("utf-8").splitlines():
                if not raw_line.strip():
                    continue
                label, text = raw_line.split("\t", 1)
                writer.writerow({"label": label, "text": text})
    with dataset_path.open(newline="") as handle:
        row_count = sum(1 for _ in handle) - 1
    report_path = EXAMPLES_ROOT / "outputs" / "reports" / "fixtures" / "sms-spam-report.json"
    write_json(
        report_path,
        {
            "report": {
                "dataset_name": "sms-spam",
                "dataset_path": str(dataset_path),
                "suspected_domain": "text binary classification",
                "suspected_problem_type": "binary classification",
                "structure": {
                    "format": "csv",
                    "file_count": 1,
                    "primary_file": str(dataset_path),
                    "row_count_estimate": row_count,
                    "column_count": 2,
                    "columns": [
                        {"name": "text", "dtype_guess": "string", "role_guess": "feature", "notes": ""},
                        {"name": "label", "dtype_guess": "string", "role_guess": "target", "notes": ""},
                    ],
                    "target_candidates": ["label"],
                    "id_candidates": [],
                    "time_candidates": [],
                },
            }
        },
    )
    return dataset_path, report_path


def _fashion_samples() -> list[tuple[int, bytes]]:
    image_archive = DOWNLOAD_ROOT / "fashion-images.gz"
    label_archive = DOWNLOAD_ROOT / "fashion-labels.gz"
    download(FASHION_IMAGES_URL, image_archive)
    download(FASHION_LABELS_URL, label_archive)
    with gzip.open(image_archive, "rb") as image_handle:
        image_magic, image_count, rows, cols = struct.unpack(">IIII", image_handle.read(16))
        if image_magic != 2051:
            raise SystemExit("unexpected Fashion-MNIST image archive header")
        image_bytes = image_handle.read()
    with gzip.open(label_archive, "rb") as label_handle:
        label_magic, label_count = struct.unpack(">II", label_handle.read(8))
        if label_magic != 2049:
            raise SystemExit("unexpected Fashion-MNIST label archive header")
        labels = label_handle.read()
    if image_count != label_count:
        raise SystemExit("Fashion-MNIST image and label counts do not match")
    sample_size = rows * cols
    selected: list[tuple[int, bytes]] = []
    counts = {0: 0, 1: 0}
    for index, label in enumerate(labels):
        if label not in counts or counts[label] >= 8:
            continue
        offset = index * sample_size
        selected.append((label, image_bytes[offset : offset + sample_size]))
        counts[label] += 1
        if all(count >= 8 for count in counts.values()):
            break
    return selected


def write_pgm(path: Path, payload: bytes, *, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        handle.write(payload)


def build_fashion_binary_dataset() -> tuple[Path, Path]:
    dataset_path = EXAMPLES_ROOT / "data" / "staged" / "fashion-top-vs-trouser"
    dataset_path.mkdir(parents=True, exist_ok=True)
    class_names = {0: "tshirt_top", 1: "trouser"}
    for sample_index, (label, payload) in enumerate(_fashion_samples()):
        class_root = dataset_path / class_names[label]
        write_pgm(class_root / f"sample_{sample_index:03d}.pgm", payload, width=28, height=28)
    report_path = EXAMPLES_ROOT / "outputs" / "reports" / "fixtures" / "fashion-report.json"
    write_json(
        report_path,
        {
            "report": {
                "dataset_name": "fashion-top-vs-trouser",
                "dataset_path": str(dataset_path),
                "suspected_domain": "vision image classification",
                "suspected_problem_type": "binary classification",
                "structure": {
                    "format": "image_folder",
                    "file_count": 16,
                    "primary_file": str(dataset_path),
                    "row_count_estimate": 16,
                    "column_count": 2,
                    "columns": [
                        {"name": "image_path", "dtype_guess": "path", "role_guess": "feature", "notes": ""},
                        {"name": "label", "dtype_guess": "string", "role_guess": "target", "notes": ""},
                    ],
                    "target_candidates": ["label"],
                    "id_candidates": [],
                    "time_candidates": [],
                    "image": {
                        "channels": 1,
                        "height": 28,
                        "width": 28,
                        "class_names": list(class_names.values()),
                    },
                },
            }
        },
    )
    return dataset_path, report_path


def copy_dataset_into_pipeline(pipeline_root: Path, dataset_path: Path) -> None:
    pipeline_data_root = pipeline_root / "data"
    if dataset_path.is_file():
        shutil.copy2(dataset_path, pipeline_data_root / dataset_path.name)
        return
    shutil.copytree(dataset_path, pipeline_data_root, dirs_exist_ok=True)


def collate_pipeline(*, pipeline_name: str, report_path: Path, recipe_name: str) -> Path:
    result = mach_main(
        [
            "collate",
            "pipeline",
            "--workspace",
            str(EXAMPLES_ROOT),
            "--create",
            "--report",
            str(report_path),
            "--name",
            pipeline_name,
            "--recipe",
            recipe_name,
            "--force",
        ]
    )
    if result != 0:
        raise SystemExit(f"collate pipeline failed for `{pipeline_name}`")
    return EXAMPLES_ROOT / "pipelines" / pipeline_name


def verify_pipeline(pipeline_root: Path) -> None:
    for command in (
        ["model", "validate", "--pipeline-path", str(pipeline_root)],
        ["model", "compile", "--pipeline-path", str(pipeline_root)],
        ["run", "train", "--pipeline-path", str(pipeline_root)],
    ):
        result = mach_main(command)
        if result != 0:
            raise SystemExit(f"verification command failed: {' '.join(command)}")


def write_workspace_readme(pipelines: list[dict[str, str]]) -> None:
    lines = [
        "# Recipe Gallery Workspace",
        "",
        "This workspace is generated by `scripts/build_recipe_gallery.py`.",
        "",
        "It shows the same recipe-first collator creating usable starter pipelines across several families.",
        "",
        "## Pipelines",
        "",
    ]
    for pipeline in pipelines:
        lines.extend(
            [
                f"- `{pipeline['name']}`",
                f"  - recipe: `{pipeline['recipe']}`",
                f"  - dataset: `{pipeline['dataset']}`",
                f"  - source: `{pipeline['source']}`",
            ]
        )
    (EXAMPLES_ROOT / "README.md").write_text("\n".join(lines) + "\n")


def scrub_workspace_noise() -> None:
    for pyc_path in EXAMPLES_ROOT.rglob("*.pyc"):
        pyc_path.unlink()
    for cache_dir in sorted(EXAMPLES_ROOT.rglob("__pycache__"), reverse=True):
        cache_dir.rmdir()


def main() -> int:
    reset_workspace()

    titanic_path, titanic_report = build_titanic_dataset()
    penguins_path, penguins_report = build_penguins_dataset()
    sms_path, sms_report = build_sms_dataset()
    fashion_path, fashion_report = build_fashion_binary_dataset()

    pipeline_specs = [
        {
            "name": "titanic-vanilla-mlp",
            "report": titanic_report,
            "recipe": "tabular.binary.basic",
            "dataset_path": titanic_path,
            "dataset": "titanic",
            "source": TITANIC_URL,
        },
        {
            "name": "penguins-deep-mlp",
            "report": penguins_report,
            "recipe": "tabular.binary.deep",
            "dataset_path": penguins_path,
            "dataset": "penguins-sex",
            "source": PENGUINS_URL,
        },
        {
            "name": "sms-spam-transformer",
            "report": sms_report,
            "recipe": "text.binary.transformer",
            "dataset_path": sms_path,
            "dataset": "sms-spam",
            "source": SMS_SPAM_URL,
        },
        {
            "name": "fashion-binary-cnn",
            "report": fashion_report,
            "recipe": "vision.binary.cnn",
            "dataset_path": fashion_path,
            "dataset": "fashion-top-vs-trouser",
            "source": FASHION_IMAGES_URL,
        },
        {
            "name": "fashion-binary-resnet",
            "report": fashion_report,
            "recipe": "vision.binary.resnet",
            "dataset_path": fashion_path,
            "dataset": "fashion-top-vs-trouser",
            "source": FASHION_IMAGES_URL,
        },
    ]

    for pipeline in pipeline_specs:
        pipeline_root = collate_pipeline(
            pipeline_name=pipeline["name"],
            report_path=Path(pipeline["report"]),
            recipe_name=pipeline["recipe"],
        )
        copy_dataset_into_pipeline(pipeline_root, Path(pipeline["dataset_path"]))
        verify_pipeline(pipeline_root)

    write_workspace_readme(
        [
            {
                "name": pipeline["name"],
                "recipe": pipeline["recipe"],
                "dataset": pipeline["dataset"],
                "source": pipeline["source"],
            }
            for pipeline in pipeline_specs
        ]
    )
    scrub_workspace_noise()
    shutil.rmtree(DOWNLOAD_ROOT.parent, ignore_errors=True)
    print(f"recipe gallery ready: {EXAMPLES_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
