"""Private subprocess entrypoint. Raw records and model output never reach MCP stdio."""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
import warnings
from pathlib import Path

from .errors import SafeError
from .models import load_encoder


def deny_network():
    """Installed only in the isolated inference child, after source data was loaded."""
    for key in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "DO_NOT_TRACK",
        "TOKENIZERS_PARALLELISM",
    ):
        os.environ[key] = "false" if key == "TOKENIZERS_PARALLELISM" else "1"

    def audit(event, args):
        if event in {"socket.connect", "socket.sendto", "socket.getaddrinfo"}:
            raise PermissionError("Inference networking is disabled")

    sys.addaudithook(audit)


def progress(path: Path | None, stage: str):
    if path:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"stage": stage}))
        temporary.replace(path)


def feature_frame(frame, features, training_positions=None):
    import pandas as pd
    from pandas.api.types import is_numeric_dtype

    result = frame[features].copy()
    for name in features:
        if not is_numeric_dtype(result[name]):
            import numbers

            values = result[name]
            training = (
                values
                if training_positions is None
                else values.iloc[training_positions]
            )
            present = training.dropna().astype(str)
            if (
                len(training.dropna())
                and training.dropna().map(lambda v: isinstance(v, numbers.Number)).all()
            ):
                result[name] = pd.to_numeric(values, errors="coerce")
                continue
            if (
                len(present)
                and present.str.match(r"^\d{4}-\d{2}-\d{2}(?:[T ].*)?$").all()
            ):
                parsed = pd.to_datetime(values, errors="coerce", utc=True)
                parsed_training = (
                    parsed
                    if training_positions is None
                    else parsed.iloc[training_positions]
                )
                if parsed_training[training.notna()].notna().all():
                    result[name] = parsed.dt.tz_localize(None)
                    continue
            result[name] = values.map(lambda x: None if pd.isna(x) else str(x)).astype(
                object
            )
    return result


def embeddings_for(X, device, status=None):
    import numpy as np

    from .vendor.tile.embeddings import Embeddings, pool_cells, serialize
    from .vendor.tile.features import string_columns

    strings = string_columns(X)
    if not strings:
        return Embeddings(np.empty((0, len(X), 0), dtype=np.float16), []), "none"
    progress(status, "embedding")
    # Bound the retained embedding array before loading the encoder or serializing rows.
    if (
        len(strings) * len(X) * 1024 * 2
        > int(os.environ.get("MOKUML_MAX_EMBED_MB", "512")) * 1024**2
    ):
        raise SafeError("resource_limit")
    model, tokenizer = load_encoder(device)
    texts, spans = serialize(X)
    budget = 4096
    for offset in range(0, len(texts), 128):
        tokens = tokenizer(texts[offset : offset + 128], truncation=False)["input_ids"]
        if any(
            len(row) > min(budget, model.config.max_position_embeddings)
            for row in tokens
        ):
            raise SafeError("text_too_long")
    # All features supply row context; only string-cell states need to be retained.
    selected_spans = spans[:, [X.columns.get_loc(col) for col in strings], :]
    values = pool_cells(model, tokenizer, texts, selected_spans, max_tokens=budget)
    del model, tokenizer, texts, spans
    gc.collect()
    return Embeddings(values, strings), "Qwen3-0.6B-Base/16"


def run_prediction(job, status=None):
    from .engine import run

    return run(job, status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    warnings.filterwarnings("ignore")
    os.environ["MPLCONFIGDIR"] = str(args.output.parent / "mpl-cache")
    os.environ["XDG_CACHE_HOME"] = str(args.output.parent / "cache")
    os.environ["TABPFN_NO_BROWSER"] = "1"
    deny_network()
    try:
        with args.job.open("rb") as stream:
            job = pickle.load(
                stream
            )  # App-created file in a private session directory.
        args.job.unlink(missing_ok=True)
        result = {"ok": True, "result": run_prediction(job, args.status)}
    except SafeError as exc:
        result = {"ok": False, "code": exc.code}
        if hasattr(exc, "candidates"):
            from .contracts import CandidateResult

            result["candidates"] = [
                CandidateResult.model_validate(c).model_dump(exclude_none=True)
                for c in exc.candidates
            ]
    except Exception:
        result = {"ok": False, "code": "inference_failed"}
    temporary = args.output.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(result, stream)
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
