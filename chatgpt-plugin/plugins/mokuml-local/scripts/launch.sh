#!/bin/bash
set -euo pipefail
umask 077
PLUGIN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASK_STATE="${MOKUML_HOME:-$HOME/Library/Application Support/MokuML Local}"
TASK_PYTHON="$TASK_STATE/runtime/bin/python"
if [ ! -x "$TASK_PYTHON" ]; then
  echo "MokuML needs its local runtime. Run the plugin's scripts/setup.sh and retry." >&2
  exit 1
fi
export MOKUML_HOME="$TASK_STATE"
export PYTHONPATH="$PLUGIN_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
exec "$TASK_PYTHON" -m mokuml_local
