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
ARTIFACT="${DIST_DIR}/machinate-${VERSION}.tar.gz"

mkdir -p "${DIST_DIR}"
rm -f "${ARTIFACT}"

tar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='dist' \
  --exclude='__pycache__' \
  -czf "${ARTIFACT}" \
  -C "${ROOT}" \
  .

echo "artifact: ${ARTIFACT}"
shasum -a 256 "${ARTIFACT}"

