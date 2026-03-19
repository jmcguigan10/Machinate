from __future__ import annotations

import unittest

from machinate.cli import build_parser
from machinate.core import slugify


class CliSmokeTests(unittest.TestCase):
    def test_slugify(self) -> None:
        self.assertEqual(slugify("Demo Pipeline"), "demo-pipeline")

    def test_workspace_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["workspace", "init"])
        self.assertEqual(args.command, "workspace")
        self.assertEqual(args.workspace_command, "init")


if __name__ == "__main__":
    unittest.main()
