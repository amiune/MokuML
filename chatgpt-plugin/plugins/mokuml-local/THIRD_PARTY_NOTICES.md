# Third-party notices

TiLE's `features.py` and `embeddings.py` are included unmodified from [soda-inria/tile](https://github.com/soda-inria/tile/tree/f765ef8a35af9228f69589e94aab072051477e07), revision `f765ef8a35af9228f69589e94aab072051477e07`. Their BSD-3-Clause license is included at `src/mokuml_local/vendor/tile/LICENSE`.

Citation: Lefebvre et al., *TILE: Bridging LLMs and Table Foundation Models for Tabular Learning with Strings*, arXiv:2610.17065 (2026).

TabICLv2 implementation and checkpoint licenses are BSD-3-Clause; the checkpoint notice is included under `src/mokuml_local/vendor/tabicl/LICENSE`. Source: [soda-inria/tabicl](https://github.com/soda-inria/tabicl) and [jingang/TabICL](https://huggingface.co/jingang/TabICL).

Qwen3-0.6B-Base weights and tokenizer are Apache-2.0 licensed. Its LICENSE is downloaded and hash-verified with the model. Source: [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base).

Python dependencies retain their own distribution license files in the installed runtime. This package does not redistribute their wheels or the model weights inside the plugin archive.

## Added prediction engines

KumoTabular uses NVIDIA structured-data-models 0.2.1 (Apache-2.0 code) and nvidia/Kumo-Tabular small-v1.0.1 weights (OpenMDW-1.1). See https://github.com/NVIDIA/structured-data-models and https://huggingface.co/nvidia/Kumo-Tabular.

TabPFN uses PriorLabs/tabpfn 9.0.0 (Apache-2.0 code). Default v2 classifier/regressor weights use the Prior Labs License with attribution requirements. Newer v3.5 weights have separate terms and require explicit user opt-in with appropriate access/license. See https://github.com/PriorLabs/TabPFN#license.

Every installed default model includes its pinned upstream license file. Model revisions and SHA-256 hashes are recorded in src/mokuml_local/model-manifest.json. The split-aware TILE feature adapter extends the BSD-3-Clause upstream recipe outside the byte-pinned vendored files.

DuckDB is MIT licensed (https://github.com/duckdb/duckdb). SQLGlot is MIT licensed (https://github.com/tobymao/sqlglot). Dependencies retain their own license metadata in the installed runtime.
