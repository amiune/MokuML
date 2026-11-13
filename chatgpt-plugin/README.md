# MokuML Predictions for ChatGPT desktop

MokuML turns a conversational prediction request into a reproducible local ML task. It supports classification, regression and future outcomes. ChatGPT plans and explains; the plugin reads data and runs TILE/Qwen with TabICL, KumoTabular-small or TabPFN.

## Setup and installation

Targets macOS on Apple silicon with [uv](https://docs.astral.sh/uv/getting-started/installation/). Setup creates a hash-locked Python 3.11 environment:

    bash scripts/setup.sh

Register this workspace as a local marketplace:

    codex plugin marketplace add /absolute/path/to/chatgpt-plugin

Open ChatGPT desktop's Plugins directory, choose the local marketplace and install **MokuML Predictions**. Start a new local chat and mention the plugin. The stable identifier is mokuml-local. For details, see [OpenAI's local marketplace instructions](https://developers.openai.com/plugins/build/plugins#install-a-local-plugin-manually) and [desktop plugin architecture](https://developers.openai.com/plugins/deploy/desktop-only).

The local MCP server launches on demand. Its private window opens directly in your system browser. Chat tools never receive the capability URL.

Default runtime/model state is under ~/Library/Application Support/MokuML Local. Set MOKUML_HOME to choose another private directory. Keep state outside the ChatGPT project and source control.

## Use

1. Ask MokuML to connect data. In the private window, import CSV/Excel files or connect PostgreSQL, MySQL or SQLite. Related files become separate relations; Excel worksheets are imported together. Mark files containing new cases as scoring data.
2. Describe the desired outcome and prediction unit. ChatGPT inspects the catalog and aggregate profiles, clarifies ambiguous target meaning and prepares a typed, read-only query.
3. Prepare the three default engines in the private window. Text features additionally require the approximately 1.2 GB Qwen checkpoint. Downloads never read your input records.
4. ChatGPT starts the prepared plan, compares eligible models and explains the returned Augmented Prediction Result.
5. Inspect complete individual outputs or save CSV/report exports locally. For one new case, use the prepared plan's private input form, then ask ChatGPT to run that plan.

Examples: estimate house prices, classify labeled cases, predict future event occurrence, or forecast a future total. A supervised task needs historical outcomes or an explicit outcome definition. If all outcomes are known and no new inputs are supplied, results are clearly labeled historical evaluation.

## Data and query behavior

- CSV: UTF-8/BOM with comma, semicolon or tab separators. Excel: .xlsx with unique nonempty column names.
- Limits: 50 MB per file, 100,000 rows per imported relation/materialized task, 2,000 columns and 32 file relations. Overflow fails instead of truncating.
- Catalog discovery includes accessible schemas, tables/views, column types/descriptions, keys, indexes, constraints and foreign keys. Column defaults, check predicates, enumerated values, credentials and input samples are excluded.
- Profiles use at most 10,000 rows, label partial profiles as estimates and suppress detailed statistics below ten non-null observations. Category values are never listed.
- Typed recipes support equijoins, filters, arithmetic, conditional expressions, aggregation and derived outcomes. SQLGlot is mandatory; only compiler-generated read-only queries execute.
- Local file queries run through DuckDB with external access and extension loading disabled. Database reads have 60-second statement/read limits, read-only transactions and cancellation.
- Identifiers, targets and explicitly unavailable columns cannot enter features. Declare when source fields became available: SQL checks cannot establish undocumented domain semantics.
- Separate scoring files use explicit column mappings or exact-name matches. Joined/aggregated scoring inputs should contain the prepared feature output names; map those output names to scoring column IDs when names differ.
- Future recipes build historical/scoring anchors, past-only aggregates and complete label windows. Units are day/week/month; rolling and next-calendar-period alignment are supported. The observation watermark is an explicit task assumption. Calendar windows use UTC; naive source timestamps are assumed to be UTC.
- Use group splitting for independent new entities, or chronological splitting for future outcomes of known entities. Simultaneously requesting both ordinary group and time fields is rejected for clarification.

## Evaluation and explanations

Automatic selection compares installed compatible engines on three folds inside the training partition. Classification selects by log loss; regression selects by MAE. Classification reserves 20% for untouched testing. Regression reserves 20% for calibration and 20% for testing. Grouped and temporal splits respect those units; overlapping future-label windows are purged. Invalid splits fail rather than becoming random splits.

Preprocessing and target encoding follow training partitions. The final model uses the training partition only, so its test measurements and calibration refer to the same fitted model. Baselines use class priors, median outcomes or matching historical lag features. Candidate failures and checkpoint-limit exclusions remain visible.

Regression can return a 90% split-conformal interval with sufficient independent calibration examples. Temporal coverage is empirical. Classification probabilities are explicitly uncalibrated.

Global explanations perturb transformed blocks for up to ten original features, three times on at most 128 test cases. These are approximate associations because TILE embeddings contain cross-cell context. Local explanations replace up to five features with training references, returning measured prediction/probability changes without revealing input values. These sensitivities are not causal effects.

Up to twenty requested individual prediction outputs can enter chat. Larger batches have aggregate summaries and complete private exports. Input records and identifying details never enter the result contract.

## Models, licensing and privacy

Default packages: tabicl 2.2.0, structured-data-models 0.2.1 and tabpfn 9.0.0. The default TabPFN checkpoint is v2; package and checkpoint versions are separately identified. Model revisions, SHA-256 hashes and license files are pinned in model-manifest.json. Newer TabPFN v3.5 is available only through explicit local opt-in with appropriate access/license; its token is used only by the setup subprocess.

TILE's vendored embedding/feature modules remain byte-identical to upstream revision f765ef8a35af9228f69589e94aab072051477e07. The split-aware feature adapter extends the recipe outside those pinned files.

Inference runs in a subprocess with Hugging Face offline mode, telemetry disabled, blocked network sockets and discarded stdout/stderr. Checkpoint loading uses verified local assets. Database reads happen before that process. There is no paid API call, cloud embedding path or automatic inference download.

The authenticated loopback window validates Host/Origin, uses a random session capability and restrictive CSP, and renders data as text. CSV exports escape spreadsheet formulas. Release/shutdown cancels work and removes session files. Exports saved by the owner remain until deleted. Abnormal termination can leave private session directories; remove stale session directories only after all MokuML sessions stop.

This is a plugin I/O boundary. ChatGPT's separate file/terminal/browser tools remain outside it; plugin skills explicitly forbid using them to inspect owner records or private exports.

## Development

Use an isolated ignored state directory:

    export MOKUML_HOME="$PWD/.local"
    bash scripts/setup.sh
    export PYTHONPATH="$PWD/plugins/mokuml-local/src"
    "$MOKUML_HOME/runtime/bin/python" -m pytest plugins/mokuml-local/tests -m 'not models and not database' -q

Prepare all models and run real offline smoke tests:

    "$MOKUML_HOME/runtime/bin/python" -m mokuml_local.models --component all --text
    MOKUML_RUN_MODELS=1 MOKUML_ESTIMATORS=1 "$MOKUML_HOME/runtime/bin/python" -m pytest plugins/mokuml-local/tests -m models -q

Disposable database checks:

    "$MOKUML_HOME/runtime/bin/python" plugins/mokuml-local/scripts/test_databases.py

The Docker helper creates and removes isolated PostgreSQL/MySQL containers. With an existing PostgreSQL installation, scripts/test_postgres.py runs a temporary private cluster; set MOKUML_POSTGRES_BIN if needed. Neither helper uses owner databases. See tests/test_databases.py for explicit disposable-fixture configuration.

Build the portable marketplace ZIP:

    python3 scripts/package_plugin.py

See [verification](VERIFICATION.md) for actual results and environment limitations. Third-party notices are included in the plugin package.
