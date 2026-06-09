#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_PATH="${1:-$ROOT/dist/MokuPredictor.app}"
RESOURCES_DIR="$APP_PATH/Contents/Resources"
PYTHON="$RESOURCES_DIR/python/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  printf 'Bundled Python not found at %s\n' "$PYTHON" >&2
  exit 1
fi

PYTHONPATH="$RESOURCES_DIR/worker/src" "$PYTHON" - <<'PY'
from worker.api import handle
health = handle("GET", "/health")
print(health)
assert health["ok"] is True
assert health["capabilities"]["sqlalchemy"] is True
assert health["capabilities"]["sqlglot"] is True
assert health["capabilities"]["pandas"] is True
assert health["capabilities"]["sklearn"] is True
assert health["capabilities"]["tabpfn"] is True
PY
