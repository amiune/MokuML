#!/bin/bash
set -euo pipefail
umask 077
PLUGIN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASK_STATE="${MOKUML_HOME:-$HOME/Library/Application Support/MokuML Local}"
if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv from https://docs.astral.sh/uv/getting-started/installation/ and rerun setup." >&2
  exit 1
fi
mkdir -p "$TASK_STATE"
chmod 700 "$TASK_STATE"
uv venv --python 3.11 "$TASK_STATE/runtime"
uv pip sync "$PLUGIN_ROOT/requirements.lock" --python "$TASK_STATE/runtime/bin/python" --require-hashes
printf '%s\n' "MokuML runtime installed. Model weights are downloaded separately from the private local window."
