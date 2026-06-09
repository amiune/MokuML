# specs.md — macOS Local Text-to-SQL + TabPFN-3 Prediction App

## 1. Product Summary

Build a native macOS app that lets a user connect to a database, describe a prediction task in natural language, and receive machine-learning predictions generated locally.

The app must:

1. Accept a database connection string.
2. Accept a natural-language prediction prompt, for example:
   - “Predict whether each customer will churn next month.”
   - “Predict the expected order value for each open quote.”
   - “Predict which invoices are likely to be paid late.”
3. Inspect the database schema locally.
4. Send only schema metadata, table names, columns, types, primary keys, foreign keys, indexes, and relationships to an LLM.
5. Never send real database rows or cell values to the LLM.
6. Use the LLM to create a safe SQL `SELECT` query that extracts a flat ML-ready table.
7. Validate that the generated SQL is read-only and safe.
8. Execute the generated SQL locally against the user’s database.
9. Convert the resulting table into a standard tabular ML problem.
10. Run TabPFN-3 locally for classification or regression.
11. Show the user:
    - final predictions,
    - probabilities or confidence scores when available,
    - target column inference/explanation,
    - generated SQL,
    - dataset summary,
    - model/task diagnostics,
    - warnings and limitations.

Primary privacy rule: **the LLM may see schema metadata, but never raw data.**

---

## 2. Target Platform

- Platform: macOS desktop app.
- Recommended UI stack: SwiftUI.
- Minimum OS target: macOS 14+.
- Architecture: Apple Silicon first, Intel optional.
- Local ML runtime: Python 3.11+ environment managed by the app.
- Local LLM runtime: either Ollama, llama.cpp server, or embedded llama.cpp wrapper.
- Optional remote LLM provider: OpenAI-compatible API.

The app should be designed as a local-first product. API LLM mode is optional and must preserve the same privacy boundary: only schema metadata and the user’s prompt may be sent externally.

---

## 3. High-Level Architecture

```text
macOS SwiftUI App
    |
    |-- Connection Manager
    |-- Schema Introspection Engine
    |-- Prompt Builder
    |-- LLM SQL Generator
    |       |-- Local LLM provider
    |       |-- Optional API LLM provider
    |
    |-- SQL Safety Validator
    |-- Local SQL Executor
    |-- ML Task Builder
    |-- Python TabPFN Worker
    |-- Results Renderer
    |-- Audit Log / Run History
```

Recommended implementation split:

```text
App shell: Swift + SwiftUI
Backend worker: Python FastAPI or stdio worker process
Local ML: Python package using tabpfn, pandas, scikit-learn
Local SQL: Python SQLAlchemy + database-specific drivers
LLM adapter: Swift or Python, depending on whether using Ollama/llama.cpp HTTP or direct API
```

Preferred MVP architecture:

- SwiftUI frontend starts and communicates with a local Python worker process.
- Python worker exposes a local-only HTTP API on `127.0.0.1` or communicates through stdio.
- Python worker handles:
  - database connection,
  - schema introspection,
  - SQL validation,
  - query execution,
  - TabPFN inference.
- SwiftUI handles:
  - input forms,
  - progress states,
  - generated SQL preview,
  - prediction table,
  - warnings,
  - settings.

---

## 4. Supported Database Types for v1

Implement Postgres first.

Required v1:

- PostgreSQL connection strings:
  - `postgresql://user:password@host:port/database`
  - `postgresql+psycopg://user:password@host:port/database`

Optional v1.1:

- SQLite local file.
- MySQL / MariaDB.
- DuckDB.

Database access must be read-only from the app’s point of view, even if the connection string technically has write permissions.

For Postgres, the app should set the transaction to read-only before executing generated SQL:

```sql
BEGIN READ ONLY;
-- generated SELECT here
ROLLBACK;
```

If supported by the driver, also enforce:

```sql
SET statement_timeout = '60s';
SET idle_in_transaction_session_timeout = '60s';
SET default_transaction_read_only = on;
```

---

## 5. Security and Privacy Requirements

### 5.1 Hard Privacy Boundary

The LLM must never receive:

- table rows,
- column values,
- samples,
- distinct values,
- min/max values,
- example records,
- user data,
- query result data.

The LLM may receive only:

- database dialect,
- table names,
- column names,
- column data types,
- nullability,
- primary keys,
- foreign keys,
- indexes,
- approximate row counts only if computed from metadata without scanning sensitive data,
- user prediction prompt,
- allowed output format instructions.

Do not send `SELECT * LIMIT 5` samples to the LLM.

### 5.2 SQL Safety Rules

The generated SQL must be validated before execution.

Allowed:

- `SELECT`
- `WITH ... SELECT`
- joins
- subqueries
- aggregate functions
- window functions
- `CASE WHEN`
- aliases
- `WHERE`
- `GROUP BY`
- `ORDER BY`
- `LIMIT`

Forbidden:

- `DELETE`
- `INSERT`
- `UPDATE`
- `UPSERT`
- `MERGE`
- `DROP`
- `ALTER`
- `CREATE`
- `TRUNCATE`
- `GRANT`
- `REVOKE`
- `COPY`
- `CALL`
- `EXECUTE`
- `DO`
- stored procedures
- temporary table creation
- extension creation
- multiple SQL statements in one response
- comments that hide additional SQL
- semicolon-separated statements
- database-specific filesystem/network functions
- any mutation, DDL, permission, admin, or side-effect command

The SQL validator must parse SQL using an AST parser, not just regex.

Recommended Python libraries:

- `sqlglot` for SQL parsing and dialect-aware AST validation.
- `sqlparse` only as a secondary formatting tool, not as the main safety gate.

Validation requirements:

1. Parse the generated SQL.
2. Reject if parsing fails.
3. Reject if there is more than one SQL expression/statement.
4. Reject if root node is not `SELECT` or `WITH` whose final expression is `SELECT`.
5. Walk the AST and reject forbidden expression types.
6. Reject SQL containing comments.
7. Reject SQL containing semicolon except optional trailing semicolon after a single statement.
8. Reject references to schemas outside the allowlist.
9. Reject references to tables not present in the introspected schema.
10. Apply an automatic outer `LIMIT` unless disabled by the user.
11. Execute inside a read-only transaction.

### 5.3 Connection Security

- Do not persist connection strings by default.
- If the user chooses to save a connection, store credentials in macOS Keychain.
- Mask passwords in UI and logs.
- Never write raw connection strings to logs.
- Provide a “Forget all saved connections” command.

### 5.4 API LLM Mode

When using a remote API LLM:

- Show a clear warning that schema metadata and the natural-language prompt will be sent to the selected provider.
- Never send raw data.
- Require explicit opt-in.
- Provide a “local-only mode” switch.
- Store API keys in Keychain.

---

## 6. Local LLM Requirements

### 6.1 LLM Purpose

The LLM is used only to produce a SQL `SELECT` query from:

- user prediction prompt,
- database dialect,
- schema metadata,
- relationship graph,
- strict output contract.

The LLM must not perform ML, inspect data, or see query results.

### 6.2 Recommended Local Model Options

The app should support configurable local LLM providers.

Recommended defaults:

1. **SQLCoder 7B / SQLCoder family**
   - Good default for text-to-SQL.
   - Smaller and practical on Apple Silicon when quantized.

2. **Qwen Coder family, quantized GGUF**
   - Good fallback for general code and SQL generation.
   - Prefer smaller quantized models for base M-series Macs.

3. **Any Ollama-compatible or llama.cpp-compatible model**
   - The user can configure model name, context length, temperature, and endpoint.

Use deterministic settings:

```text
temperature: 0.0
top_p: 1.0
repeat_penalty: provider default
max_tokens: 2048 to 4096
```

The local LLM runtime should support Apple Silicon acceleration through Metal when available.

### 6.3 Optional API LLM Provider

Support OpenAI-compatible chat completion APIs.

Provider config:

```json
{
  "provider": "openai_compatible",
  "base_url": "https://api.openai.com/v1",
  "model": "gpt-4.1-mini or user-selected",
  "api_key_keychain_ref": "...",
  "temperature": 0.0
}
```

Do not hardcode one provider. Implement a provider interface.

---

## 7. Schema Introspection

### 7.1 Postgres Introspection Query

The app must execute local metadata queries to collect:

- schemas,
- tables,
- columns,
- data types,
- nullable flags,
- primary keys,
- foreign keys,
- indexes.

For Postgres, use `information_schema` and `pg_catalog`.

Do not query table rows.

Example metadata query categories:

```sql
-- Tables
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_type = 'BASE TABLE'
  AND table_schema NOT IN ('pg_catalog', 'information_schema');

-- Columns
SELECT table_schema, table_name, column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
ORDER BY table_schema, table_name, ordinal_position;

-- Foreign keys
SELECT
  tc.table_schema,
  tc.table_name,
  kcu.column_name,
  ccu.table_schema AS foreign_table_schema,
  ccu.table_name AS foreign_table_name,
  ccu.column_name AS foreign_column_name
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
  ON tc.constraint_name = kcu.constraint_name
 AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage AS ccu
  ON ccu.constraint_name = tc.constraint_name
 AND ccu.table_schema = tc.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY';
```

### 7.2 Schema JSON Format Sent to LLM

The LLM receives schema metadata in compact JSON.

Example:

```json
{
  "dialect": "postgres",
  "schemas": [
    {
      "name": "public",
      "tables": [
        {
          "name": "customers",
          "columns": [
            {"name": "id", "type": "uuid", "nullable": false, "primary_key": true},
            {"name": "created_at", "type": "timestamp", "nullable": false},
            {"name": "country", "type": "text", "nullable": true}
          ],
          "foreign_keys": []
        },
        {
          "name": "orders",
          "columns": [
            {"name": "id", "type": "uuid", "nullable": false, "primary_key": true},
            {"name": "customer_id", "type": "uuid", "nullable": false},
            {"name": "total", "type": "numeric", "nullable": false},
            {"name": "created_at", "type": "timestamp", "nullable": false}
          ],
          "foreign_keys": [
            {
              "column": "customer_id",
              "references_table": "customers",
              "references_column": "id"
            }
          ]
        }
      ]
    }
  ]
}
```

### 7.3 Schema Size Control

If the schema is too large for the LLM context window:

1. Build a schema graph.
2. Rank tables by semantic match against the prompt using table/column names only.
3. Include the top relevant tables plus directly connected foreign-key neighbors.
4. Show the user which tables were included and excluded.
5. Allow manual table selection.

No data values may be used for ranking.

---

## 8. LLM Prompt Contract

### 8.1 System Prompt

Use a strict system prompt similar to this:

```text
You are a SQL generation engine for a local machine-learning app.

You receive:
- a user prediction request,
- database dialect,
- database schema metadata with tables, columns, keys, and relationships.

You must generate exactly one safe read-only SQL query that extracts a flat machine-learning dataset for the requested prediction task.

Rules:
- Return JSON only.
- The SQL must be a single SELECT query or WITH ... SELECT query.
- Do not use DELETE, INSERT, UPDATE, MERGE, DROP, ALTER, CREATE, TRUNCATE, GRANT, REVOKE, CALL, EXECUTE, COPY, DO, temp tables, stored procedures, or any command with side effects.
- Do not use multiple statements.
- Do not include comments.
- Do not invent tables or columns.
- Use only tables and columns from the provided schema.
- Prefer explicit column lists; do not use SELECT *.
- Include a target column if it already exists in the database and the request implies supervised learning.
- Include an entity id column if possible.
- Include feature columns useful for the prediction.
- Avoid leaking future information into features.
- If the task is impossible because the target is not present or cannot be derived safely from schema alone, explain why in the JSON.

Return JSON with this exact shape:
{
  "status": "ok" | "needs_clarification" | "impossible",
  "task_type": "classification" | "regression" | "unknown",
  "target_column": "string or null",
  "entity_id_columns": ["string"],
  "sql": "single SELECT query or null",
  "reasoning_summary": "short non-sensitive explanation",
  "leakage_warnings": ["string"],
  "assumptions": ["string"]
}
```

### 8.2 User Prompt Payload

The LLM user message should contain:

```json
{
  "prediction_request": "Predict whether each customer will churn next month",
  "dialect": "postgres",
  "schema": { "...": "schema metadata only" },
  "constraints": {
    "must_be_single_select": true,
    "no_data_samples_provided": true,
    "no_mutations_allowed": true,
    "prefer_entity_id": true,
    "prefer_explicit_target_column": true
  }
}
```

### 8.3 LLM Output Handling

After receiving output:

1. Parse JSON.
2. Reject non-JSON output.
3. Reject missing fields.
4. If `status != ok`, show the issue to the user.
5. Validate SQL using the SQL Safety Validator.
6. Display generated SQL to the user.
7. Optionally require user approval before execution in v1.
8. Execute locally only after validation.

---

## 9. ML Data Table Contract

The generated SQL should return a flat table.

Recommended columns:

```text
entity_id | feature_1 | feature_2 | ... | target
```

Rules:

- The result must include one target column for supervised learning.
- The target column must be known before training.
- Entity ID columns are excluded from training features but included in final output.
- Date/time columns may be included but must be transformed before TabPFN if necessary.
- Text columns should be handled conservatively in local mode.
- High-cardinality IDs should not be used as features by default.
- If no target column can be inferred, the app must ask the user to choose the target column or modify the prompt.

### Important Limitation

The app cannot magically predict future outcomes unless the database contains historical labeled examples or a target column that can be derived safely. The app must detect and explain this.

Examples:

- Good: “Predict whether invoices will be paid late” if historical invoices include `paid_at`, `due_date`, and current open invoices exist.
- Bad: “Predict which customers will buy next month” if there is no historical purchase data or target derivation strategy.

---

## 10. TabPFN-3 Integration

### 10.1 Runtime

Use Python package:

```bash
pip install tabpfn pandas scikit-learn sqlalchemy psycopg[binary] sqlglot fastapi uvicorn pydantic
```

The app must support local TabPFN-3 model files and first-run model download/authentication where required.

### 10.2 Task Detection

The app should infer task type from:

1. LLM output `task_type`,
2. target column dtype,
3. number of unique target values,
4. user confirmation if ambiguous.

Classification when:

- target is boolean,
- target is categorical/string,
- target has small number of discrete classes.

Regression when:

- target is continuous numeric,
- prompt asks for amount, price, quantity, duration, or score.

### 10.3 Preprocessing

TabPFN can handle many tabular realities, but the app still needs robust table preparation.

Implement:

- split entity id columns from features,
- split target from features,
- remove rows where target is null for training,
- identify rows where target is null as prediction rows when possible,
- convert unsupported dtypes,
- encode dates into numeric features:
  - year,
  - month,
  - day,
  - day of week,
  - days since epoch,
- preserve categorical columns as categorical when supported,
- drop columns with all nulls,
- warn on high-cardinality columns,
- warn on very wide datasets,
- cap dataset size according to TabPFN limits or ask the user to sample.

### 10.4 Prediction Modes

Support two modes:

#### Mode A — Predict rows with missing target

If SQL returns both historical labeled rows and rows with null target:

- train/fit TabPFN on rows with non-null target,
- predict rows where target is null,
- show entity IDs and predictions.

#### Mode B — Train/evaluate on historical data

If all rows have targets:

- perform train/test split,
- fit TabPFN,
- show validation metrics,
- optionally allow user to upload or query separate prediction rows later.

### 10.5 Classification Output

For classification, show:

- predicted class,
- probability per class,
- top class probability,
- row/entity ID,
- class labels,
- validation accuracy if test split exists,
- ROC AUC for binary classification if possible,
- confusion matrix if validation exists.

### 10.6 Regression Output

For regression, show:

- predicted value,
- row/entity ID,
- validation MAE if test split exists,
- validation RMSE if test split exists,
- R² if test split exists,
- warning that per-row probability is not available for plain regression unless uncertainty estimation is implemented.

### 10.7 Useful TabPFN Metadata to Surface

Show:

- model/checkpoint used,
- task type,
- number of training rows,
- number of prediction rows,
- number of features,
- ignored columns,
- dropped columns,
- warnings,
- validation metrics,
- prediction probabilities for classifiers,
- model/runtime errors.

---

## 11. Python Worker API

Implement a local worker with these endpoints or equivalent stdio commands.

### 11.1 `POST /connect/test`

Input:

```json
{
  "connection_string": "postgresql://..."
}
```

Output:

```json
{
  "ok": true,
  "database_type": "postgres",
  "server_version": "...",
  "message": "Connection successful"
}
```

### 11.2 `POST /schema/introspect`

Input:

```json
{
  "connection_string": "postgresql://...",
  "include_schemas": ["public"],
  "exclude_tables": []
}
```

Output:

```json
{
  "ok": true,
  "schema": { }
}
```

### 11.3 `POST /llm/generate-sql`

Input:

```json
{
  "prediction_prompt": "Predict late invoices",
  "schema": { },
  "llm_provider": {
    "type": "local_ollama",
    "endpoint": "http://127.0.0.1:11434",
    "model": "sqlcoder"
  }
}
```

Output:

```json
{
  "ok": true,
  "task_type": "classification",
  "target_column": "is_late",
  "sql": "WITH ... SELECT ...",
  "reasoning_summary": "...",
  "leakage_warnings": [],
  "assumptions": []
}
```

### 11.4 `POST /sql/validate`

Input:

```json
{
  "sql": "SELECT ...",
  "dialect": "postgres",
  "schema": { }
}
```

Output:

```json
{
  "ok": true,
  "is_safe": true,
  "errors": [],
  "normalized_sql": "SELECT ... LIMIT 100000"
}
```

### 11.5 `POST /sql/execute-preview`

This endpoint executes the SQL locally and returns only metadata and a small preview to the app UI. The preview is for the user only and must not be sent to the LLM.

Input:

```json
{
  "connection_string": "postgresql://...",
  "sql": "SELECT ...",
  "preview_rows": 20
}
```

Output:

```json
{
  "ok": true,
  "columns": [
    {"name": "customer_id", "dtype": "string"},
    {"name": "total_orders", "dtype": "number"},
    {"name": "churned", "dtype": "boolean"}
  ],
  "row_count": 12345,
  "preview": [ ],
  "warnings": []
}
```

### 11.6 `POST /ml/run-tabpfn`

Input:

```json
{
  "connection_string": "postgresql://...",
  "sql": "SELECT ...",
  "target_column": "churned",
  "entity_id_columns": ["customer_id"],
  "task_type": "classification",
  "mode": "auto",
  "max_rows": 100000,
  "test_size": 0.2,
  "random_seed": 42
}
```

Output:

```json
{
  "ok": true,
  "task_type": "classification",
  "target_column": "churned",
  "entity_id_columns": ["customer_id"],
  "model": {
    "name": "TabPFNClassifier",
    "checkpoint": "default TabPFN-3"
  },
  "dataset": {
    "training_rows": 10000,
    "prediction_rows": 250,
    "feature_count": 42,
    "dropped_columns": [],
    "warnings": []
  },
  "metrics": {
    "accuracy": 0.87,
    "roc_auc": 0.91
  },
  "predictions": [
    {
      "entity_id": {"customer_id": "..."},
      "prediction": true,
      "probability": 0.82,
      "class_probabilities": {"false": 0.18, "true": 0.82}
    }
  ]
}
```

---

## 12. UI Requirements

### 12.1 Main Flow

Create a step-by-step UI:

1. **Connect Database**
   - connection string input,
   - test connection button,
   - save connection checkbox.

2. **Describe Prediction**
   - prompt text area,
   - examples dropdown,
   - local/API LLM selector.

3. **Inspect Schema**
   - show tables and relationships discovered,
   - allow table inclusion/exclusion,
   - show privacy notice: “Only schema metadata is sent to the LLM.”

4. **Generate SQL**
   - show generated SQL,
   - show assumptions,
   - show leakage warnings,
   - validate SQL status.

5. **Preview ML Table**
   - show local-only preview rows,
   - show columns,
   - choose/confirm target column,
   - choose/confirm ID columns,
   - choose classification/regression if ambiguous.

6. **Run Prediction**
   - progress indicator,
   - model/checkpoint info,
   - validation metrics,
   - final predictions table.

7. **Export**
   - export predictions to CSV,
   - copy SQL,
   - save run report as Markdown.

### 12.2 Results View

Results table columns:

- entity ID columns,
- prediction,
- probability/confidence where available,
- class probabilities for classification,
- optional source features selected by user.

Provide sorting and filtering by:

- highest probability,
- predicted class,
- entity ID,
- confidence threshold.

### 12.3 Warnings

Show warnings prominently for:

- target column not found,
- possible data leakage,
- SQL query too broad,
- too many rows/features for TabPFN,
- remote API LLM mode enabled,
- saved credentials enabled,
- regression has no probability output,
- no prediction rows found,
- all rows have target values only,
- insufficient training rows.

---

## 13. Run History and Auditability

Each run should save a local audit record, excluding passwords and raw data.

Store:

- timestamp,
- database type,
- masked host/database name,
- user prompt,
- schema hash,
- selected tables,
- LLM provider type and model name,
- generated SQL,
- SQL validation result,
- target column,
- task type,
- dataset shape,
- model/checkpoint,
- metrics,
- warnings.

Do not store:

- raw rows,
- connection password,
- API keys,
- full connection string.

Use local app storage:

```text
~/Library/Application Support/<AppName>/runs/
~/Library/Application Support/<AppName>/settings.json
~/Library/Caches/<AppName>/
```

Credentials must go to Keychain only.

---

## 14. Error Handling

Handle:

- invalid connection string,
- database unreachable,
- unsupported database dialect,
- missing database driver,
- schema too large,
- LLM unavailable,
- LLM invalid JSON,
- LLM unsafe SQL,
- SQL execution timeout,
- query returns no rows,
- query returns no target column,
- target has only one class,
- too few labeled rows,
- TabPFN model unavailable,
- TabPFN license/auth/model download issue,
- out-of-memory on Apple Silicon,
- unsupported dtypes.

Every error should include:

- user-friendly message,
- technical details collapsible section,
- suggested fix.

---

## 15. Implementation Milestones

### Milestone 1 — macOS Shell

- SwiftUI app with sidebar/stepper UI.
- Settings screen.
- Local Python worker startup/shutdown.
- Health check endpoint.

### Milestone 2 — Database Connection and Introspection

- Postgres connection test.
- Schema introspection.
- Schema graph rendering.
- No data leaves local machine.

### Milestone 3 — LLM SQL Generation

- Local Ollama provider.
- OpenAI-compatible API provider.
- Prompt contract.
- JSON output parser.
- SQL display.

### Milestone 4 — SQL Safety Layer

- AST validation using `sqlglot`.
- Read-only transaction.
- Statement timeout.
- Automatic limit.
- Rejection test suite for dangerous SQL.

### Milestone 5 — ML Table Execution

- Execute safe SQL locally.
- Dataframe creation.
- Target/ID selection.
- Dataset summary and preview.

### Milestone 6 — TabPFN-3 Integration

- Classification.
- Regression.
- Missing-target prediction mode.
- Train/test validation mode.
- Metrics and probabilities.

### Milestone 7 — Results and Export

- Prediction table.
- Probability sorting/filtering.
- CSV export.
- Markdown run report.
- Local run history.

### Milestone 8 — Packaging

- Bundle Python environment or create first-run installer.
- Code signing/notarization preparation.
- Keychain integration.
- App sandbox review.

---

## 16. Testing Requirements

### 16.1 Unit Tests

Test:

- connection string masking,
- schema introspection parsing,
- prompt creation,
- LLM JSON parser,
- SQL validator,
- forbidden SQL rejection,
- target detection,
- preprocessing,
- classification output formatting,
- regression output formatting.

### 16.2 SQL Safety Tests

The following must be rejected:

```sql
DELETE FROM customers;
INSERT INTO x VALUES (1);
UPDATE customers SET name = 'x';
DROP TABLE customers;
ALTER TABLE customers ADD COLUMN x int;
CREATE TABLE temp_x AS SELECT * FROM customers;
TRUNCATE customers;
SELECT * FROM customers; DELETE FROM customers;
COPY customers TO '/tmp/customers.csv';
CALL dangerous_proc();
DO $$ BEGIN DELETE FROM customers; END $$;
SELECT pg_read_file('/etc/passwd');
```

The following should be allowed if tables/columns exist:

```sql
SELECT customer_id, COUNT(*) AS order_count
FROM orders
GROUP BY customer_id;

WITH invoice_features AS (
  SELECT customer_id, COUNT(*) AS invoice_count
  FROM invoices
  GROUP BY customer_id
)
SELECT * FROM invoice_features;
```

### 16.3 Privacy Tests

Assert that LLM requests contain:

- prompt,
- dialect,
- schema metadata.

Assert that LLM requests do not contain:

- query preview rows,
- table cell values,
- raw dataframes,
- connection passwords,
- API keys.

### 16.4 ML Tests

Use synthetic fixtures:

- binary classification,
- multiclass classification,
- regression,
- missing-target prediction rows,
- all-labeled evaluation-only dataset,
- too few target labels,
- high-cardinality ID columns.

---

## 17. Suggested Project Structure

```text
macos-tabpfn-sql-predictor/
  specs.md
  README.md
  app/
    MacPredictor.xcodeproj
    MacPredictor/
      App.swift
      Views/
      Models/
      Services/
      Keychain/
  worker/
    pyproject.toml
    src/
      worker/
        main.py
        api.py
        config.py
        db/
          connection.py
          introspection_postgres.py
          execute.py
        llm/
          providers.py
          prompt_builder.py
          output_parser.py
        sql/
          validator.py
          normalize.py
        ml/
          prepare_table.py
          tabpfn_runner.py
          metrics.py
        security/
          masking.py
          privacy.py
    tests/
      test_sql_validator.py
      test_privacy.py
      test_ml.py
  fixtures/
    postgres_demo_schema.sql
    synthetic_churn.csv
```

---

## 18. Example User Flow

User enters:

```text
Connection string:
postgresql://user:password@localhost:5432/shop

Prompt:
Predict which customers are likely to churn next month.
```

App does:

1. Connects locally to Postgres.
2. Reads schema metadata.
3. Sends prompt + schema metadata to local LLM.
4. Receives JSON with SQL.
5. Validates SQL is single read-only `SELECT`.
6. Executes SQL locally.
7. Finds target column, for example `churned_next_month`.
8. Fits TabPFN classifier on rows with known target.
9. Predicts rows where target is null or evaluates historical split.
10. Shows predictions with probabilities.

---

## 19. Non-Goals for v1

Do not implement in v1:

- automatic database mutation,
- writing predictions back to the database,
- training custom neural networks,
- multi-user collaboration,
- cloud data upload,
- sending raw data to an LLM,
- BI dashboard generation,
- full AutoML comparison with many models,
- fine-tuning TabPFN,
- arbitrary Python code generated by LLM.

---

## 20. Future Versions

### v1.1

- SQLite and DuckDB support.
- Manual target/feature selection UI.
- Better date cutoff controls to avoid leakage.
- SQL query refinement loop.
- Export run as reproducible Python notebook.

### v1.2

- MySQL support.
- Relationship-aware table recommendation.
- SHAP/explainability support if feasible with selected TabPFN version.
- Background task queue.
- Scheduled local runs.

### v2

- Relational prediction without forcing full flattening if TabPFN relational capabilities become locally accessible and production-licensed.
- Optional TabPFN API mode.
- Write-back connector for predictions, still requiring explicit user approval.
- Natural-language report generation from local prediction results, with privacy controls.

---

## 21. Important Licensing Note

Before commercial or production use, verify the current TabPFN-3 license and deployment terms. The app should include a configuration layer that can switch between:

- local TabPFN-3 weights where legally allowed,
- licensed enterprise deployment,
- TabPFN API/client if required for production use.

Do not assume that research/evaluation model weights are automatically allowed for commercial production use.

---

## 22. Definition of Done

The v1 implementation is complete when:

1. The app runs on Apple Silicon macOS.
2. A user can connect to a local or remote Postgres database.
3. The app introspects schema locally without reading data rows.
4. The app can call a local LLM to generate SQL from prompt + schema.
5. The app can optionally call an API LLM with the same privacy boundary.
6. Generated SQL is validated as a single safe `SELECT` query.
7. Unsafe SQL is reliably rejected.
8. Safe SQL executes locally in a read-only transaction.
9. The resulting table is converted to a TabPFN-ready dataframe.
10. TabPFN-3 runs locally for classification and regression.
11. The UI shows predictions and probabilities where available.
12. The UI shows generated SQL, assumptions, warnings, and metrics.
13. No raw data is sent to any LLM provider.
14. Credentials and API keys are stored only in Keychain when saved.
15. Tests cover SQL safety, privacy, and ML execution.
