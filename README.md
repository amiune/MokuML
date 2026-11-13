# MokuML

A local ChatGPT desktop plugin for classification, regression and future outcomes from files or databases. The repository contains the plugin and its supporting documentation; the native macOS app and legacy worker have been retired.

Describe a question such as “Predict prices for these houses”, “Classify these new cases”, or “Predict next month’s sales”. ChatGPT discovers the source structure, prepares a typed query, compares compatible local engines, and explains an **Augmented Prediction Result** with validation, baselines, uncertainty and measured influences.

Input records and identifying details remain local. ChatGPT receives source metadata, permitted aggregate profiles and requested prediction outputs. A private local window handles imports, credentials, individual case inputs, explicit model downloads and complete exports.

## Install

Open [the plugin workspace](chatgpt-plugin/README.md) for setup and ChatGPT desktop installation instructions.

Requires macOS, Python 3.11 and uv. From this repository:

    bash chatgpt-plugin/scripts/setup.sh
    codex plugin marketplace add ./chatgpt-plugin

In ChatGPT desktop, install **MokuML Predictions** from the local marketplace and start a new local chat. No additional OpenAI API key is needed. Model weights are prepared separately in the plugin's private window.

## Engines and method

- TabICLv2, KumoTabular-small and TabPFN v2, selected using training-only validation folds.
- Qwen3-0.6B-Base's first sixteen layers implement the [TILE contextual cell-embedding method](https://github.com/soda-inria/tile) for string features. Numeric-only tasks skip Qwen.
- PostgreSQL, MySQL, SQLite, CSV and Excel. Imported file relations are queried with DuckDB; database connections use read-only transactions.
- General supervised tabular prediction requires historical outcomes or a defensible label definition. Dataset validation does not establish clinical, causal or population-wide validity.

See the [architecture](specs.md), [verification record](chatgpt-plugin/VERIFICATION.md) and [plugin instructions](chatgpt-plugin/README.md).

## Package

    python3 chatgpt-plugin/scripts/package_plugin.py

This creates an installable marketplace ZIP under chatgpt-plugin/dist, excluding runtime/model caches, private data, credentials and test outputs.
