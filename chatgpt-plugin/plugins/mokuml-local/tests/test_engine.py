from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mokuml_local import engine
from mokuml_local.splitting import partitions
from mokuml_local.vendor.tile.embeddings import Embeddings


class SyntheticEstimator:
    fits = []

    def __init__(self, name, task, device, encoding_folds):
        self.engine, self.task, self.device = name, task, device
        self.features = SimpleNamespace(blocks_={"x": [0]})

    def fit(self, X, y, embeddings):
        self.fits.append((self.engine, set(X.index)))
        self.mean = float(np.mean(y))
        return self

    def transform(self, X, embeddings):
        return X.to_numpy(dtype=float)

    def predict_z(self, Z):
        return Z[:, 0] + ({"tabicl": 0, "kumo": 2, "tabpfn": 4}[self.engine])

    def proba_z(self, Z):
        p = np.where((Z[:, 0] % 2) == 1, 0.8, 0.2)
        return np.column_stack([1 - p, p])

    def provenance(self, encoder):
        return dict(
            engine=self.engine,
            version="test",
            checkpoint_revision="test",
            encoder=encoder,
            device=self.device,
        )


def fake_embeddings(X, device, status=None):
    return Embeddings(np.empty((0, len(X), 0)), []), "none"


def job(task="regression", missing=2):
    n = 100
    frame = pd.DataFrame(
        {
            "x": np.arange(n, dtype=float),
            "outcome": np.arange(n, dtype=float)
            if task == "regression"
            else np.arange(n) % 2,
            "__score": False,
        }
    )
    if missing:
        frame.loc[n - missing :, "__score"] = True
        frame.loc[n - missing :, "outcome"] = np.nan
    return dict(
        frame=frame,
        features=["x"],
        target="outcome",
        task=task,
        split="random" if task == "regression" else "stratified",
        identifiers=[],
        model="auto",
    )


@pytest.fixture
def fake_models(monkeypatch):
    SyntheticEstimator.fits = []
    monkeypatch.setattr(engine, "Estimator", SyntheticEstimator)
    monkeypatch.setattr(engine, "embeddings_for", fake_embeddings)
    monkeypatch.setattr(engine, "available", lambda name: True)
    monkeypatch.setenv("MOKUML_DEVICE", "cpu")


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("missing", [0, 2, 25])
def test_selection_independent_partitions_and_output_cap(fake_models, task, missing):
    definition = job(task, missing)
    frame = definition["frame"].copy()
    frame["__target"] = frame.outcome
    known = np.flatnonzero(~frame.__score)
    train, calibration, test = partitions(frame, known, task, definition["split"])
    result = engine._run(definition, None, "cpu")["augmented"]
    assert result["result_kind"] == ("predictions" if missing else "evaluation")
    assert len(result["predictions"]) <= 20
    assert result["prediction_rows"] == (missing or len(test))
    assert result["model"]["engine"] == "tabicl"
    assert all(
        not fitted & set(test) and not fitted & set(calibration)
        for _, fitted in SyntheticEstimator.fits
    )
    assert all(fitted <= set(train) for _, fitted in SyntheticEstimator.fits)
    assert result["feature_influences"]
    if task == "regression":
        assert result["interval_method"] == "split_conformal_90"


def test_test_outcomes_cannot_change_model_selection_or_fits(fake_models):
    definition = job()
    first = engine._run(definition, None, "cpu")["augmented"]
    frame = definition["frame"].copy()
    frame["__target"] = frame.outcome
    _, _, test = partitions(
        frame, np.flatnonzero(~frame.__score), "regression", "random"
    )
    changed = job()
    changed["frame"].loc[test, "outcome"] += 100000
    second = engine._run(changed, None, "cpu")["augmented"]
    assert first["candidates"] == second["candidates"]
    assert first["predictions"] == second["predictions"]
    assert first["metrics"] != second["metrics"]


def test_explicit_failed_model_does_not_substitute(fake_models, monkeypatch):
    class Failing(SyntheticEstimator):
        def fit(self, *args):
            raise RuntimeError("PRIVATE_EXCEPTION")

    monkeypatch.setattr(engine, "Estimator", Failing)
    definition = job()
    definition["model"] = "kumo"
    with pytest.raises(engine.SelectionError) as exc:
        engine._run(definition, None, "cpu")
    assert exc.value.candidates == [
        dict(model="kumo", status="failed", reason="local_candidate_failed")
    ]


def test_same_model_accelerator_failure_retries_cpu(fake_models, monkeypatch):
    seen = []

    class DeviceEstimator(SyntheticEstimator):
        def fit(self, *args):
            seen.append((self.engine, self.device))
            if self.device == "mps":
                raise RuntimeError("PRIVATE_EXCEPTION")
            return super().fit(*args)

    monkeypatch.setattr(engine, "Estimator", DeviceEstimator)
    definition = job()
    definition["model"] = "tabicl"
    result = engine._run(definition, None, "mps")["augmented"]
    assert seen[0] == ("tabicl", "mps") and all(name == "tabicl" for name, _ in seen)
    assert result["model"]["device"] == "cpu"
    assert result["candidates"][0]["reason"] == "same_model_cpu_retry"


def test_auto_reports_final_fit_failure_and_uses_next_cv_candidate(
    fake_models, monkeypatch
):
    class CapacityEstimator(SyntheticEstimator):
        def fit(self, X, *args):
            if self.engine == "tabicl" and len(X) > 50:
                raise MemoryError("synthetic capacity limit")
            return super().fit(X, *args)

    monkeypatch.setattr(engine, "Estimator", CapacityEstimator)
    result = engine._run(job(), None, "cpu")["augmented"]
    assert result["model"]["engine"] == "kumo"
    failed = next(c for c in result["candidates"] if c["model"] == "tabicl")
    assert failed["status"] == "failed" and failed["reason"] == "local_candidate_failed"
