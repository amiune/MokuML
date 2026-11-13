import os
import time

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def private_home(request, monkeypatch, tmp_path):
    if request.node.get_closest_marker("models"):
        if os.environ.get("MOKUML_RUN_MODELS") != "1":
            pytest.skip("Set MOKUML_RUN_MODELS=1 after preparing the real local models")
    else:
        monkeypatch.setenv("MOKUML_HOME", str(tmp_path / "state"))


@pytest.fixture
def service():
    from mokuml_local.service import Service

    instance = Service()
    yield instance
    instance.close()


@pytest.fixture
def business_frame():
    return pd.DataFrame(
        {
            "record_id": ["001", "002", "003", "004", "005", "006"],
            "amount": [10, 20, 30, 40, 50, 60],
            "note": ["PRIVATE_CELL_SENTINEL"] * 6,
            "outcome": ["PRIVATE_LABEL_A", "PRIVATE_LABEL_B"] * 3,
        }
    )


def task_for(service, target="outcome", features=None, task="regression"):
    table = next(iter(service.catalog.tables.values()))
    columns = {c.name: c.id for c in table.columns}

    def col(name):
        return {"kind": "column", "column_id": columns[name]}

    return dict(
        question="Predict the outcome",
        task=task,
        query={"table_id": table.id},
        target={"name": "outcome", "expression": col(target)},
        features=[
            {"name": name, "expression": col(name)} for name in (features or ["x"])
        ],
    )


def wait_run(service, started, timeout=300):
    assert started["ok"], started
    run_id = started["run"]["id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = service.invoke("get_run", run_id=run_id)
        if result["run"]["status"] != "running":
            return result
        time.sleep(0.1)
    service.cancel_run(run_id)
    pytest.fail("Prediction exceeded the synthetic test budget")
