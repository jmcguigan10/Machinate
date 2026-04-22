#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(python3 - <<'PY'
from pathlib import Path
import re

text = Path("pyproject.toml").read_text()
match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
if not match:
    raise SystemExit("Unable to find version in pyproject.toml")
print(match.group(1))
PY
)"
DIST_DIR="${ROOT}/dist"
ARTIFACT="${DIST_DIR}/machinator-${VERSION}.tar.gz"
FILELIST="$(mktemp)"

cleanup() {
  rm -f "${FILELIST}"
}

trap cleanup EXIT

mkdir -p "${DIST_DIR}"
rm -f "${ARTIFACT}"

git -C "${ROOT}" ls-files > "${FILELIST}"

tar \
  -czf "${ARTIFACT}" \
  -C "${ROOT}" \
  -T "${FILELIST}"

echo "artifact: ${ARTIFACT}"
shasum -a 256 "${ARTIFACT}"
