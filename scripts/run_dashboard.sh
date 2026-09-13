#!/usr/bin/env bash
set -euo pipefail

PYTHON="/Users/admin/Desktop/KSL_Projects/R&D/Smart-Scan/venv/bin/python"

if [ ! -x "$PYTHON" ]; then
  PYTHON="/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
fi

if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

exec "$PYTHON" -m nid_ocr_lab.dashboard.server "$@"
