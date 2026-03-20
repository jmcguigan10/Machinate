#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


DEFAULT_OWNER = "jmcguigan10"
DEFAULT_APP_REPO = "Machinator"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_version(pyproject_path: Path) -> str:
    text = pyproject_path.read_text()
    match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
    if match is None:
        raise SystemExit(f"Unable to find version in {pyproject_path}")
    return match.group(1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def formula_text(*, owner: str, app_repo: str, version: str, sha256: str) -> str:
    url = f"https://github.com/{owner}/{app_repo}/releases/download/v{version}/machinator-{version}.tar.gz"
    homepage = f"https://github.com/{owner}/{app_repo}"
    return f"""class Machinator < Formula
  include Language::Python::Virtualenv

  desc "Prompt-first control-plane CLI for ML workspaces and pipelines"
  homepage "{homepage}"
  url "{url}"
  sha256 "{sha256}"
  license "MIT"

  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "Machinator", shell_output("#{{bin}}/macht --help")
  end
end
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a Homebrew formula for Machinator")
    parser.add_argument("--owner", default=DEFAULT_OWNER, help="GitHub owner or org")
    parser.add_argument("--app-repo", default=DEFAULT_APP_REPO, help="GitHub app repository name")
    parser.add_argument(
        "--tap-formula",
        default=str(Path("/Users/johnny/Projects/homebrew-tap/Formula/machinator.rb")),
        help="Path to the tap formula file to write",
    )
    args = parser.parse_args(argv)

    root = project_root()
    pyproject_path = root / "pyproject.toml"
    version = read_version(pyproject_path)
    artifact_path = root / "dist" / f"machinator-{version}.tar.gz"
    if not artifact_path.exists():
        raise SystemExit(f"Release artifact is missing: {artifact_path}")

    sha256 = sha256_file(artifact_path)
    rendered = formula_text(owner=args.owner, app_repo=args.app_repo, version=version, sha256=sha256)

    app_formula_path = root / "packaging" / "homebrew" / "machinator.rb"
    app_formula_path.write_text(rendered)

    tap_formula_path = Path(args.tap_formula).expanduser().resolve()
    tap_formula_path.parent.mkdir(parents=True, exist_ok=True)
    tap_formula_path.write_text(rendered)

    print(f"version: {version}")
    print(f"sha256: {sha256}")
    print(f"wrote: {app_formula_path}")
    print(f"wrote: {tap_formula_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

