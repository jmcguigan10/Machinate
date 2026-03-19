from __future__ import annotations

import unittest

from machinate.cli import build_parser
from machinate.core import is_url, slugify
from machinate.commands.legate import data_report_schema


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


if __name__ == "__main__":
    unittest.main()
