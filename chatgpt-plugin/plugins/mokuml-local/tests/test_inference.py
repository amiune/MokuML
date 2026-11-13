import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from mokuml_local import inference, models
from mokuml_local.errors import SafeError
from mokuml_local.vendor.tile.embeddings import Embeddings
from mokuml_local.vendor.tile.features import TargetEncoder, TILEFeatures


def test_pca_is_fitted_on_training_embeddings_only():
    rng = np.random.default_rng(42)
    train = pd.DataFrame({"text": ["repeated"] * 12, "amount": range(12)})
    values = rng.normal(size=(1, 12, 8))
    features = TILEFeatures(window=(2, 4), native_categories=True)
    embeddings = Embeddings(values, ["text"])
    features.fit_transform(
        train.iloc[:8],
        np.array([0, 1] * 4),
        embeddings.take(range(8)),
        "classification",
    )
    mean = features.reducers_["text"].steps[0][1].mean_.copy()
    np.testing.assert_allclose(mean, values[0, :8].mean(axis=0))
    changed = values.copy()
    changed[0, 8:] += 100000
    features.transform(train.iloc[8:], Embeddings(changed, ["text"]).take(range(8, 12)))
    np.testing.assert_allclose(mean, features.reducers_["text"].steps[0][1].mean_)


def test_target_encoding_excludes_each_rows_own_outcome():
    X = pd.DataFrame({"category": ["shared"] * 8})
    y = np.arange(8, dtype=float)
    baseline = TargetEncoder().fit_transform(X, y, "regression")
    for i in range(len(y)):
        changed = y.copy()
        changed[i] += 100000
        actual = TargetEncoder().fit_transform(X, changed, "regression")
        assert actual[i, 0] == baseline[i, 0]


def test_date_type_is_decided_using_training_records_only():
    frame = pd.DataFrame({"date": ["2026-01-01", "2026-01-02", "private invalid date"]})
    transformed = inference.feature_frame(frame, ["date"], [0, 1])
    assert pd.api.types.is_datetime64_any_dtype(transformed["date"])
    assert pd.isna(transformed.iloc[2]["date"])


def test_numeric_input_does_not_load_or_download_qwen(monkeypatch):
    monkeypatch.setattr(
        inference, "load_encoder", lambda *args: pytest.fail("Qwen must be skipped")
    )
    embeddings, encoder = inference.embeddings_for(
        pd.DataFrame({"amount": [1, 2, 3]}), "cpu"
    )
    assert encoder == "none" and embeddings.columns == []


def test_text_embedding_memory_preflight(monkeypatch):
    monkeypatch.setenv("MOKUML_MAX_EMBED_MB", "0")
    monkeypatch.setattr(
        inference, "load_encoder", lambda *args: pytest.fail("Preflight must run first")
    )
    with pytest.raises(SafeError) as exc:
        inference.embeddings_for(pd.DataFrame({"note": ["private"] * 2}), "cpu")
    assert exc.value.code == "resource_limit"


def test_runtime_network_guard_blocks_dns_and_sockets():
    code = "from mokuml_local.inference import deny_network; import socket; deny_network();\ntry: socket.getaddrinfo('example.com',443)\nexcept PermissionError: print('blocked')\nelse: raise SystemExit(1)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env=os.environ.copy(),
    )
    assert result.returncode == 0 and result.stdout.strip() == "blocked"


def test_checksum_verification_rejects_corrupt_models(monkeypatch, tmp_path):
    expected = hashlib.sha256(b"expected").hexdigest()
    monkeypatch.setattr(
        models, "manifest", lambda: {"qwen": {"files": {"weights": expected}}}
    )
    directory = tmp_path / "qwen"
    directory.mkdir()
    (directory / "weights").write_bytes(b"PRIVATE_CORRUPT_CONTENT")
    with pytest.raises(SafeError) as exc:
        models.verify("qwen", tmp_path)
    assert exc.value.code == "models_invalid" and "PRIVATE" not in str(exc.value)


def test_insufficient_data_is_rejected_before_encoding(monkeypatch):
    from mokuml_local import engine

    monkeypatch.setattr(
        engine, "embeddings_for", lambda *args: pytest.fail("No encoding expected")
    )
    with pytest.raises(SafeError):
        engine._run(
            {
                "frame": pd.DataFrame(
                    {"x": [1, 2], "outcome": [1, 2], "__score": [False, False]}
                ),
                "task": "regression",
                "target": "outcome",
                "features": ["x"],
                "split": "random",
                "identifiers": [],
            },
            None,
            "cpu",
        )
