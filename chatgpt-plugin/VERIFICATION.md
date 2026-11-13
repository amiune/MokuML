# Verification — 13 November 2026

Verified on macOS arm64, Python 3.11.9. The native application/legacy worker were removed, including app build artifacts. The plugin is the sole product.

| Check | Observed result |
|---|---|
| Unit, privacy, HTTP, planning, leakage, lifecycle and stdio packaging | **88 passed** in a fresh runtime installed from the SHA-256 dependency lock |
| Real offline model suite | **14 passed, 1 skipped**: numeric/text classification and regression for all three default engines, automatic comparison, private house-case entry and future sales; MPS unavailable |
| Final automatic-selection and UTC future-sales checks | **2 passed** in the clean runtime |
| Live PostgreSQL | **1 passed** against a private disposable PostgreSQL 14.5 cluster, stopped and removed afterward |
| Extracted distributable | **4 passed**: marketplace paths, pinned source, stdio contract and copied-install launch |
| Local browser verification | Synthetic house case showed model setup, prepared SQL/case form, completed prediction, interval, validation and export controls |
| Static checks | Ruff, JavaScript syntax, shell syntax and both skill validators passed |
| Vendored TILE source | Original Git blob hashes verified by the packaging tests |

The unit suite includes hand-checked joins/derived outcomes, separate/precomputed scoring inputs, complete/empty future windows, leap/calendar boundaries, UTC offsets, SQLite CTE-name collisions, read-only transactions, unsafe AST rejection, immutable/stale plans, no entity overlap, purged temporal labels, scoped target encoding, test-independent selection, final-fit candidate failure, private errors/inputs/exports, cancellation during materialization and copied-install stdio launch.

Real model tests use one estimator for test speed; normal inference defaults to eight. Inference subprocesses block networking. All test data is synthetic; these checks establish implementation behavior, not real-world prediction quality.

## Reproduce

From the workspace, set MOKUML_HOME to an isolated ignored state directory and follow README setup. Use its runtime Python and PYTHONPATH pointing to plugins/mokuml-local/src.

    python -m pytest plugins/mokuml-local/tests -m 'not models and not database' -q
    MOKUML_RUN_MODELS=1 MOKUML_ESTIMATORS=1 python -m pytest plugins/mokuml-local/tests/test_models.py -m models -q
    python plugins/mokuml-local/scripts/test_postgres.py
    python scripts/package_plugin.py

The PostgreSQL helper accepts MOKUML_POSTGRES_BIN and creates its own private disposable cluster. The Docker helper can exercise PostgreSQL/MySQL together when Docker is available.

## Verification limits

- Native ChatGPT installation/conversation could not be automated: the computer-use tool blocks access to the ChatGPT application. The stdio/copied-package tests and local browser checks passed; the plugin was not registered in the owner's settings.
- A live MySQL service and Docker Desktop were unavailable. MySQL transaction/error contracts and SQL compilation were checked; live MySQL integration remains to be run with the supplied disposable-fixture helper.
- MPS was unavailable to the test process. CPU inference and tested same-model retry behavior passed.
- The optional licensed TabPFN v3.5 path is gated and pinned, but its real weights were not downloaded or tested.
- TestClient and pandas emit upstream deprecation warnings. Vendored TILE files remain unchanged and pinned.
- Calendar windows are UTC, and naive datetime inputs are assumed UTC. Source timezone/label-completeness semantics must be established when preparing a task.

A recovery archive of legacy source was written to /tmp/mokuml-legacy-source-before-removal.tar.gz before removal. It is excluded from the distributable.
