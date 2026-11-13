import json

import numpy as np
import pandas as pd
import pytest
from conftest import task_for, wait_run

from mokuml_local.models import available


@pytest.mark.models
@pytest.mark.parametrize("model", ["tabicl", "kumo", "tabpfn"])
@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("text", [False, True])
def test_real_offline_models_and_augmented_outputs(
    service, monkeypatch, model, task, text
):
    assert available(model) and (available("qwen") or not text)
    monkeypatch.setenv("MOKUML_DEVICE", "cpu")
    monkeypatch.setenv("MOKUML_ESTIMATORS", "1")
    n = 102
    frame = pd.DataFrame(
        {
            "record_id": ["PRIVATE_ID_" + str(i) for i in range(n)],
            "x": np.arange(n, dtype=float),
            "visits": np.arange(n) % 7,
            "outcome": np.arange(n, dtype=float) * 1.5
            if task == "regression"
            else np.arange(n) % 2,
        }
    )
    frame.loc[100:, "outcome"] = np.nan
    if text:
        frame["description"] = [
            "RAW_INPUT_PRIVATE local shop" if i % 2 else "RAW_INPUT_PRIVATE bakery"
            for i in range(n)
        ]
    service.publish(frame, "csv")
    definition = task_for(
        service, features=["x", "visits"] + (["description"] if text else []), task=task
    )
    definition["identifier_column_ids"] = [
        next(
            c.id
            for t in service.catalog.tables.values()
            for c in t.columns
            if c.name == "record_id"
        )
    ]
    prepared = service.invoke("prepare_prediction", task=definition)
    assert prepared["ok"], prepared
    result = wait_run(
        service,
        service.invoke("start_prediction", plan_id=prepared["plan"]["id"], model=model),
        timeout=600,
    )
    assert result["ok"], result
    assert result["run"]["status"] == "completed", result
    a = result["run"]["result"]
    assert a["model"]["engine"] == model
    assert a["prediction_rows"] == 2 and len(a["predictions"]) == 2
    assert a["model"]["encoder"] == ("Qwen3-0.6B-Base/16" if text else "none")
    assert a["feature_influences"]
    assert "PRIVATE_ID" not in json.dumps(
        result
    ) and "RAW_INPUT_PRIVATE" not in json.dumps(result)
    assert not service.runs[result["run"]["id"]].directory.exists()
    assert all(np.isfinite(v) for v in a["metrics"].values())


@pytest.mark.models
def test_real_automatic_selection_and_private_house_case(service, monkeypatch):
    monkeypatch.setenv("MOKUML_DEVICE", "cpu")
    monkeypatch.setenv("MOKUML_ESTIMATORS", "1")
    service.publish(
        pd.DataFrame(
            {
                "x": np.arange(100, dtype=float),
                "outcome": np.arange(100) * 1000 + 100000,
            }
        ),
        "csv",
    )
    prepared = service.prepare_prediction(task_for(service))["plan"]
    service.submit_case(prepared["id"], {"x": 70})
    result = wait_run(service, service.start_prediction(prepared["id"]), timeout=600)
    assert result["ok"], result
    a = result["run"]["result"]
    assert (
        len([c for c in a["candidates"] if c["status"] in {"selected", "evaluated"}])
        == 3
    )
    assert a["prediction_rows"] == 1 and isinstance(
        a["predictions"][0]["prediction"], float
    )


@pytest.mark.models
def test_real_mps_or_same_model_cpu_retry(service, monkeypatch):
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    monkeypatch.delenv("MOKUML_DEVICE", raising=False)
    monkeypatch.setenv("MOKUML_ESTIMATORS", "1")
    service.publish(pd.DataFrame({"x": range(100), "outcome": [0, 1] * 50}), "csv")
    prepared = service.prepare_prediction(task_for(service, task="classification"))[
        "plan"
    ]
    result = wait_run(
        service, service.start_prediction(prepared["id"], model="tabicl"), timeout=600
    )
    assert result["ok"], result
    assert result["run"]["device"] in {"mps", "cpu"}


@pytest.mark.models
def test_real_future_sales(service, monkeypatch):
    monkeypatch.setenv("MOKUML_DEVICE", "cpu")
    monkeypatch.setenv("MOKUML_ESTIMATORS", "1")
    dates = pd.date_range("2017-01-01", periods=102, freq="MS")
    service.publish(
        pd.DataFrame({"date": dates, "amount": 100 + np.arange(102) * 2}), "csv"
    )
    table = next(iter(service.catalog.tables.values()))
    columns = {c.name: c.id for c in table.columns}
    agg = {
        "kind": "aggregate",
        "function": "sum",
        "args": [{"kind": "column", "column_id": columns["amount"]}],
    }
    task = dict(
        question="Predict next month sales",
        task="regression",
        query={"table_id": table.id},
        target={"name": "sales", "expression": agg},
        features=[{"name": "sales_history", "expression": agg}],
        future=dict(
            event_time_column_id=columns["date"],
            start="2017-05-01T00:00:00",
            observed_through=dates[-1].isoformat(),
            lookback_periods=[1, 3],
        ),
    )
    prepared = service.prepare_prediction(task)["plan"]
    result = wait_run(
        service, service.start_prediction(prepared["id"], model="tabicl"), timeout=600
    )
    assert result["ok"], result
    assert result["run"]["result"]["split"] == "temporal"
    assert result["run"]["result"]["prediction_rows"] == 1
