from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from machinate.cli import build_parser
from machinate.commands.legate import data_report_schema
from machinate.core import is_url, slugify, write_json
from machinate.modeling import (
    DatasetFacts,
    architecture_spec_from_dataset_facts,
    compile_architecture_spec,
    dataset_facts_from_report_path,
    diff_architecture_specs,
    edit_architecture_spec,
    migration_plan_spec_files,
    parameter_count,
    render_model_spec_toml,
    validate_spec_file,
)


class CliSmokeTests(unittest.TestCase):
    def test_slugify(self) -> None:
        self.assertEqual(slugify("Demo Pipeline"), "demo-pipeline")

    def test_is_url(self) -> None:
        self.assertTrue(is_url("https://example.com/data.csv"))
        self.assertFalse(is_url("/tmp/data.csv"))

    def test_workspace_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["workspace", "init"])
        self.assertEqual(args.command, "workspace")
        self.assertEqual(args.workspace_command, "init")

    def test_guide_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["guide", "beginner"])
        self.assertEqual(args.command, "guide")
        self.assertEqual(args.guide_command, "beginner")

    def test_run_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "train", "--experiment", "baseline"])
        self.assertEqual(args.command, "run")
        self.assertEqual(args.task_name, "train")

    def test_legate_report_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["legate", "report", "--data", "--dataset", "demo-dataset"])
        self.assertEqual(args.command, "legate")
        self.assertEqual(args.legate_command, "report")
        self.assertTrue(args.data)
        self.assertEqual(args.dataset, "demo-dataset")

    def test_data_report_schema_shape(self) -> None:
        schema = data_report_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("plain_summary", schema["properties"])
        self.assertIn("data_report", schema["properties"])

    def test_model_validate_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["model", "validate", "--spec", "model.toml"])
        self.assertEqual(args.command, "model")
        self.assertEqual(args.model_command, "validate")
        self.assertEqual(args.spec, "model.toml")

    def test_model_edit_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["model", "edit", "--spec", "model.toml", "--set", "hidden_dims=[256,64]"])
        self.assertEqual(args.command, "model")
        self.assertEqual(args.model_command, "edit")
        self.assertEqual(args.set, ["hidden_dims=[256,64]"])

    def test_model_diff_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["model", "diff", "--old", "old.toml", "--new", "new.toml"])
        self.assertEqual(args.command, "model")
        self.assertEqual(args.model_command, "diff")
        self.assertEqual(args.old, "old.toml")
        self.assertEqual(args.new, "new.toml")

    def test_model_migrate_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["model", "migrate", "--old", "old.toml", "--new", "new.toml", "--source-state", "old.pt"]
        )
        self.assertEqual(args.command, "model")
        self.assertEqual(args.model_command, "migrate")
        self.assertEqual(args.source_state, "old.pt")

    def test_collate_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["collate", "pipeline", "--report", "report.json", "--intent-task", "binary_classification"]
        )
        self.assertEqual(args.command, "collate")
        self.assertEqual(args.collate_command, "pipeline")
        self.assertEqual(args.report, "report.json")
        self.assertEqual(args.intent_task, "binary_classification")

    def test_dataset_facts_and_tabular_compile(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            report_path = root / "report.json"
            write_json(
                report_path,
                {
                    "report": {
                        "dataset_name": "demo-dataset",
                        "dataset_path": str(root / "demo.csv"),
                        "suspected_problem_type": "binary classification",
                        "structure": {
                            "row_count_estimate": 5,
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
            )
            facts = dataset_facts_from_report_path(report_path)
            self.assertEqual(facts.target_column, "label")
            self.assertEqual(facts.feature_names, ["feature_a", "feature_b"])

            spec = architecture_spec_from_dataset_facts(
                facts=facts,
                pipeline_name="demo",
                recipe_name="tabular.binary.basic",
            )
            self.assertEqual(spec.family, "tabular_mlp")
            self.assertEqual(parameter_count(spec), 9089)

            artifacts = compile_architecture_spec(spec, root / "compiled")
            self.assertTrue(Path(artifacts["module_path"]).exists())
            self.assertTrue(Path(artifacts["manifest_path"]).exists())
            self.assertTrue(Path(artifacts["param_store_manifest_path"]).exists())

    def test_transformer_recipe_compile(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            facts = DatasetFacts(
                dataset_name="demo-text",
                dataset_path=root / "dataset.txt",
                modality="text",
                suspected_problem_type="binary classification",
                row_count_estimate=20,
                column_names=["text", "label"],
                feature_names=["text"],
                target_column="label",
                target_candidates=["label"],
                id_candidates=[],
                time_candidates=[],
                source_report_path=root / "report.json",
            )
            spec = architecture_spec_from_dataset_facts(
                facts=facts,
                pipeline_name="demo-text",
                recipe_name="text.binary.transformer",
            )
            self.assertEqual(spec.family, "transformer_encoder")
            artifacts = compile_architecture_spec(spec, root / "compiled_transformer")
            self.assertTrue(Path(artifacts["module_path"]).exists())

    def test_edit_and_diff_migration(self) -> None:
        facts = DatasetFacts(
            dataset_name="demo",
            dataset_path=Path("/tmp/demo.csv"),
            modality="tabular",
            suspected_problem_type="binary classification",
            row_count_estimate=10,
            column_names=["a", "b", "label"],
            feature_names=["a", "b"],
            target_column="label",
            target_candidates=["label"],
            id_candidates=[],
            time_candidates=[],
            source_report_path=Path("/tmp/report.json"),
        )
        spec = architecture_spec_from_dataset_facts(
            facts=facts,
            pipeline_name="demo",
            recipe_name="tabular.binary.basic",
        )
        edited = edit_architecture_spec(spec, ["hidden_dims=[256,64]", "dropout=0.2"])
        diff_payload = diff_architecture_specs(spec, edited)
        self.assertEqual(diff_payload["parameter_delta"], 8832)
        self.assertGreater(diff_payload["migration_plan"]["partial_copy_count"], 0)

    def test_migration_plan_from_files(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            facts = DatasetFacts(
                dataset_name="demo",
                dataset_path=root / "demo.csv",
                modality="tabular",
                suspected_problem_type="binary classification",
                row_count_estimate=10,
                column_names=["a", "b", "label"],
                feature_names=["a", "b"],
                target_column="label",
                target_candidates=["label"],
                id_candidates=[],
                time_candidates=[],
                source_report_path=root / "report.json",
            )
            original = architecture_spec_from_dataset_facts(
                facts=facts,
                pipeline_name="demo",
                recipe_name="tabular.binary.basic",
            )
            edited = edit_architecture_spec(original, ["hidden_dims=[256,64]"])
            old_path = root / "old.toml"
            new_path = root / "new.toml"
            old_path.write_text(render_model_spec_toml(original))
            new_path.write_text(render_model_spec_toml(edited))

            validation = validate_spec_file(old_path)
            self.assertIn(validation["backend"], {"python", "rust"})

            migration_plan = migration_plan_spec_files(old_path, new_path)
            self.assertGreater(migration_plan["partial_copy_count"], 0)


if __name__ == "__main__":
    unittest.main()
