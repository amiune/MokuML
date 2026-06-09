# MokuML

MokuML is a local-first macOS app for small businesses that want to turn a database and a plain-language prediction request into a safe tabular machine-learning workflow.

The privacy boundary is the core product rule:

- The LLM receives the user prediction prompt and schema metadata only.
- The LLM never receives table rows, cell values, previews, API keys, or connection passwords.
- SQL is validated before local execution.
- Query execution and model inference happen on the user machine.

## Current MVP

This repository contains:

- A SwiftUI macOS app shell in `app/MokuPredictor`.
- A local Python worker in `worker/src/worker`.
- Postgres connection, schema introspection, SQL validation, local preview execution, and ML runner endpoints.
- LLM adapters for Ollama-compatible local models and OpenAI-compatible APIs.
- Metadata-only run audit records in `~/Library/Application Support/MokuML/runs/`.
- Dependency-light fallback behavior so safety/privacy tests can run before the full Python environment is installed.

## Run The App

```bash
swift run MokuPredictor
```

The app starts a local worker at `127.0.0.1:8765` using:

```bash
PYTHONPATH=worker/src python3 -m worker.main --host 127.0.0.1 --port 8765
```

For a production-style worker environment, use Python 3.11+:

```bash
cd worker
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
python -m worker.main
```

The macOS app automatically prefers `worker/.venv/bin/python` when that environment exists. Without this environment, it falls back to system Python, which may not have the Postgres dependencies needed for `/connect/test`.

## Worker Endpoints

- `GET /health`
- `POST /connect/test`
- `POST /schema/introspect`
- `POST /llm/generate-sql`
- `POST /sql/validate`
- `POST /sql/execute-preview`
- `POST /ml/run-tabpfn`

## Local LLM

The default UI points at Ollama on `http://127.0.0.1:11434` with a SQLCoder-style model name. You can switch to the offline template provider for a no-network dry run, or configure an OpenAI-compatible endpoint. API mode still sends schema metadata only.

## Tests

Core dependency-light tests:

```bash
PYTHONPATH=worker/src python3 -m unittest discover worker/tests
```

Full SQL AST validation, Postgres access, and TabPFN inference require the worker dependencies from `worker/pyproject.toml`.

## Package A DMG

Create a self-contained macOS DMG with bundled Python and worker dependencies:

```bash
scripts/package_app.sh
```

Output:

```text
dist/MokuPredictor.app
dist/MokuPredictor-0.1.0.dmg
```

See `docs/PACKAGING.md` for local signing, Developer ID signing, and notarization.

## Important Limitations

The app cannot invent future outcomes. Supervised predictions require a historical target column or a safe target derivation already present in the database. If the target cannot be inferred from schema alone, the app asks the user to choose or clarify it.

Before commercial use, verify the current TabPFN-3 license and deployment terms.
