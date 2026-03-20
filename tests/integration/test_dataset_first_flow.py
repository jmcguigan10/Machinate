from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from machinator.cli import main


class DatasetFirstFlowTests(unittest.TestCase):
    def test_collate_create_builds_pipeline_from_report(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            workspace = root / "workspace"
            dataset_dir = workspace / "data" / "staged" / "demo"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = dataset_dir / "demo.csv"
            dataset_path.write_text("feature_a,feature_b,label\n1.0,0.5,0\n2.0,1.5,1\n")

            report_dir = workspace / "outputs" / "reports" / "legate"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-03-20T12:00:00Z",
                        "delegate_kind": "report",
                        "report_kind": "data",
                        "report": {
                            "dataset_name": "demo-dataset",
                            "dataset_path": str(dataset_path),
                            "suspected_problem_type": "binary classification",
                            "structure": {
                                "row_count_estimate": 2,
                                "target_candidates": ["label"],
                                "id_candidates": [],
                                "time_candidates": [],
                                "columns": [
                                    {"name": "feature_a", "role_guess": "feature"},
                                    {"name": "feature_b", "role_guess": "feature"},
                                    {"name": "label", "role_guess": "target"},
                                ],
                            },
                        }
                    },
                    indent=2,
                )
                + "\n"
            )

            self.assertEqual(
                main(["workspace", "init", "--path", str(workspace), "--name", "demo-workspace"]),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "collate",
                        "pipeline",
                        "--workspace",
                        str(workspace),
                        "--create",
                        "--name",
                        "demo-pipeline",
                    ]
                ),
                0,
            )

            pipeline_root = workspace / "pipelines" / "demo-pipeline"
            self.assertTrue((pipeline_root / "machinate.toml").exists())
            self.assertTrue((pipeline_root / "dataset_facts.toml").exists())
            self.assertTrue((pipeline_root / "model.toml").exists())
            self.assertTrue((pipeline_root / "training.toml").exists())

            config_text = (pipeline_root / "machinate.toml").read_text()
            self.assertIn("[collation]", config_text)
            self.assertIn('recipe = "tabular.binary.basic"', config_text)

            self.assertEqual(
                main(
                    [
                        "run",
                        "train",
                        "--pipeline-path",
                        str(pipeline_root),
                        "--experiment",
                        "baseline",
                        "--dataset",
                        str(dataset_path),
                    ]
                ),
                0,
            )
            run_artifacts = sorted((pipeline_root / "outputs" / "runs").glob("*.json"))
            self.assertTrue(run_artifacts)


if __name__ == "__main__":
    unittest.main()
