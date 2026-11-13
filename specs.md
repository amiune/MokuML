# MokuML 0.2 architecture

The sole product is the ChatGPT desktop plugin in chatgpt-plugin. It packages a Python stdio MCP server, setup/prediction skills, an authenticated local data window and hash-locked dependencies.

## Flow and interfaces

The eight MCP tools are open_local_window, inspect_source, profile_data, prepare_prediction, start_prediction, check_prediction, cancel_prediction and release_dataset. Input records, credentials and export retrieval are private HTTP routes, never MCP tools/resources.

PredictionTask defines the prediction question, classification/regression task, typed relational recipe, target/features, prediction-unit identifiers, unavailable fields, grouping/time semantics, units, scoring relation/mapping, optional future recipe and individual/summary output preference.

prepare_prediction compiles this recipe into SQL for the active dialect and validates the AST. PreparedPredictionPlan contains an opaque ID, source revision, generated SQL, feature/output structure, split strategy and findings. Stored task/plan copies prevent caller mutation. A changed source invalidates the plan.

start_prediction returns a run ID immediately. Materialization happens in a cancellable background thread using read-only source access. The table snapshot is handed to a separate network-blocked inference process. Progress, selection, evaluation, prediction and explanation are observable through check_prediction. Cancellation/release prevents late materialization from starting inference.

## Source layer

A session owns one database connection or a collection of file relations. Database catalogs cover accessible schemas unless restricted. SQLite reads the original file using mode=ro and native WAL behavior. File frames are registered in DuckDB with external access disabled.

Catalog IDs are opaque and stable for the session. Revision hashes cover schema structure, not input records. Catalog pages expose at most fifty tables; profiles page selected columns. Profiles contain counts/missingness/cardinality and permitted numerical/date summaries. They never contain category samples and suppress detailed distributions below ten non-null observations.

Typed expressions and join pairs reference catalog IDs. A deterministic SQLGlot compiler quotes physical identifiers and emits only supported operations. An AST guard checks statements, relations and functions without a regex fallback. Results are bounded at 100,001 rows so overflow is detected rather than silently sampled.

A future recipe defines event time/entity, observation range, horizon, unit, alignment and lookbacks. Historical/scoring anchors use the same formulas. Features precede each cutoff. Only fully observed future labels enter training; other historical incomplete windows are excluded. The observation-end anchor is the scoring population.

## Inference and evaluation

The default adapters are TabICLv2, KumoTabular-small and TabPFN v2. Newer licensed TabPFN is explicitly enabled locally. Checkpoint-specific limits exclude incompatible candidates, with fixed diagnostic reasons. Explicit engine requests never substitute another engine. Supported accelerator failures retry the same engine on CPU.

Qwen3-0.6B-Base/16 provides TILE cell embeddings. Numerical/date-only tasks skip Qwen. Embeddings contain selected feature context, excluding targets and identifiers. Each fold fits its own TILE reductions, vectorization, category handling and scoped target encoding. Group encoding excludes the same entity; temporal encoding uses only past matured labels and fixed neutral values when no past labels exist.

Outer partitions reserve 20% for test; regression also reserves 20% for calibration. Three training-only folds select minimum log loss or MAE. Stratified, random, grouped and temporal strategies are explicit. Equal timestamps stay together and overlapping label windows are purged. Insufficient history/classes fail before encoding.

The selected learner is fit on the training partition only. Test/calibration data never enter model fitting or selection. Baselines, calibration, evaluation and explanation use the same fitted learner. Regression intervals use finite-sample 90% split-conformal residual quantiles when sufficient independent calibration examples exist; grouped calibration uses group-maximum residuals.

## Augmented result

AugmentedPredictionResult includes outcome/units/horizon, exact model/checkpoint provenance, candidate comparisons, split/counts, test metrics, baselines, calibrated-interval availability, probability-calibration status, bounded outputs, aggregate summary, measured influences, data quality and limitations.

Individual chat outputs contain opaque handles and predictions only. Input identifiers remain in private results/CSV exports. At most twenty outputs enter chat. Global effects are grouped permutations of transformed feature blocks; local effects are prediction/probability changes under a training-reference perturbation. The feature names and measured deltas are public, while the underlying input values remain local.

ChatGPT's skill explains the returned evidence in domain context. It distinguishes probabilities, intervals, evaluation, prediction, association and causation, and reports missing evidence explicitly.

## Runtime and distribution

The portable package retains mokuml-local as its identifier at version 0.2.0. Model/setup secrets are never in manifests, logs, tool responses or archives. The ZIP includes the marketplace, source, launch/setup scripts, dependency lock, skills and notices; it excludes runtime/model/session caches.

Verification is documented in chatgpt-plugin/VERIFICATION.md. Native ChatGPT installation requires the host plugin interface; a copied package can also be verified directly through the stdio MCP contract.
