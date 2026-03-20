from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from machinator.cli import build_parser
from machinator.commands.collate import discover_report_candidates
from machinator.commands.legate import data_report_schema, resolve_notes
from machinator.core import is_url, slugify, write_json
from machinator.modeling import (
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

    def test_test_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["test", "integration"])
        self.assertEqual(args.command, "test")
        self.assertEqual(args.target, "integration")

    def test_check_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["check", "--fast"])
        self.assertEqual(args.command, "check")
        self.assertTrue(args.fast)

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
        self.assertFalse(args.notes_prompt)

    def test_data_report_schema_shape(self) -> None:
        schema = data_report_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("plain_summary", schema["properties"])
        self.assertIn("data_report", schema["properties"])
        structure = schema["properties"]["data_report"]["properties"]["structure"]
        self.assertIn("image", structure["required"])

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
            [
                "collate",
                "pipeline",
                "--report",
                "report.json",
                "--intent-task",
                "binary_classification",
                "--create",
                "--name",
                "demo",
            ]
        )
        self.assertEqual(args.command, "collate")
        self.assertEqual(args.collate_command, "pipeline")
        self.assertEqual(args.report, "report.json")
        self.assertEqual(args.intent_task, "binary_classification")
        self.assertTrue(args.create)

    def test_notes_prompt_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["legate", "report", "--data", "--dataset", "demo", "--notes-prompt"])
        self.assertTrue(args.notes_prompt)

    def test_resolve_notes_defaults_empty_without_prompt(self) -> None:
        args = build_parser().parse_args(["legate", "report", "--data", "--dataset", "demo"])
        self.assertEqual(resolve_notes(args), "")

    def test_discover_report_candidates_filters_completed_data_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            report_root = root / "outputs" / "reports" / "legate"
            report_root.mkdir(parents=True, exist_ok=True)

            write_json(
                report_root / "20260101T000000Z_old.json",
                {
                    "generated_at": "2026-01-01T00:00:00Z",
                    "delegate_kind": "report",
                    "report_kind": "data",
                    "report": {"dataset_name": "old-dataset"},
                },
            )
            write_json(report_root / "ignore.raw.json", {"plain_summary": "not a wrapped artifact"})
            write_json(
                report_root / "20260102T000000Z_new.json",
                {
                    "generated_at": "2026-01-02T00:00:00Z",
                    "delegate_kind": "report",
                    "report_kind": "data",
                    "report": {"dataset_name": "new-dataset"},
                },
            )

            candidates = discover_report_candidates(root)
            self.assertEqual([candidate.dataset_name for candidate in candidates], ["new-dataset", "old-dataset"])

    def test_resolve_notes_prompt_uses_multiline_mode_when_requested(self) -> None:
        args = build_parser().parse_args(["legate", "report", "--data", "--dataset", "demo", "--notes-prompt"])
        with mock.patch("machinator.commands.legate.can_prompt_interactively", return_value=True):
            with mock.patch("machinator.commands.legate.prompt_multiline", return_value="operator hints") as prompt:
                self.assertEqual(resolve_notes(args), "operator hints")
        prompt.assert_called_once()

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

    def test_vision_recipes_compile(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            facts = DatasetFacts(
                dataset_name="demo-vision",
                dataset_path=root / "fashion",
                modality="vision",
                suspected_problem_type="binary classification",
                row_count_estimate=16,
                column_names=["image_path", "label"],
                feature_names=["image_path"],
                target_column="label",
                target_candidates=["label"],
                id_candidates=[],
                time_candidates=[],
                source_report_path=root / "report.json",
                image_channels=1,
                image_height=28,
                image_width=28,
                class_names=["tshirt_top", "trouser"],
            )
            cnn_spec = architecture_spec_from_dataset_facts(
                facts=facts,
                pipeline_name="demo-cnn",
                recipe_name="vision.binary.cnn",
            )
            self.assertEqual(cnn_spec.family, "vision_cnn")
            cnn_artifacts = compile_architecture_spec(cnn_spec, root / "compiled_cnn")
            self.assertTrue(Path(cnn_artifacts["module_path"]).exists())

            resnet_spec = architecture_spec_from_dataset_facts(
                facts=facts,
                pipeline_name="demo-resnet",
                recipe_name="vision.binary.resnet",
            )
            self.assertEqual(resnet_spec.family, "vision_resnet")
            self.assertGreater(parameter_count(resnet_spec), parameter_count(cnn_spec))

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
