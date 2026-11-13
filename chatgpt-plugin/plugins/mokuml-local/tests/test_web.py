from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from test_privacy import completed_result

from mokuml_local.service import Run
from mokuml_local.web import create_app


@pytest.fixture
def client(service):
    app = create_app(service, "test-capability", "127.0.0.1:12345")
    with TestClient(
        app,
        base_url="http://127.0.0.1:12345",
        headers={
            "Authorization": "Bearer test-capability",
            "Origin": "http://127.0.0.1:12345",
        },
    ) as client:
        yield client


def test_authentication_origin_and_host(client):
    assert client.get("/api/state", headers={"Authorization": ""}).status_code == 401
    assert (
        client.get(
            "/api/state", headers={"Origin": "https://attacker.example"}
        ).status_code
        == 403
    )
    assert (
        client.get("/api/state", headers={"Host": "attacker.example"}).status_code
        == 403
    )
    assert client.post("/api/release", headers={"Origin": "null"}).status_code == 403
    response = client.get("/")
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert response.headers["Cache-Control"] == "no-store"
    assert "test-capability" not in response.text


def test_private_preview_stays_in_authenticated_routes(service, client):
    content = b"id,note,outcome\n001,PRIVATE_CELL_SENTINEL,PRIVATE_LABEL_A\n002,<img src=x onerror=alert(1)>,PRIVATE_LABEL_B"
    response = client.post(
        "/api/import", files={"file": ("PRIVATE_PATH.csv", content, "text/csv")}
    )
    assert response.status_code == 200
    assert "PRIVATE_CELL" not in response.text and "PRIVATE_PATH" not in response.text
    assert "PRIVATE_CELL_SENTINEL" in client.get("/api/state").text
    assert "PRIVATE_CELL_SENTINEL" not in str(service.invoke("describe_dataset"))
    assert "PRIVATE_CELL_SENTINEL" not in client.get("/").text


def test_validation_errors_never_echo_passwords(client):
    response = client.post("/api/predict", json={"password": "PRIVATE_PASSWORD"})
    assert response.status_code == 422 and "PRIVATE_PASSWORD" not in response.text
    response = client.post(
        "/api/database/connect",
        json={"password": "PRIVATE_PASSWORD", "kind": "unsupported"},
    )
    assert response.status_code == 400 and "PRIVATE_PASSWORD" not in response.text


def test_csv_and_report_are_private_and_complete(service, client, tmp_path):
    value = completed_result()
    value["predictions"] = [
        {
            "row_number": i,
            "identifiers": {"id": "=PRIVATE_FORMULA()"},
            "prediction": "PRIVATE_LABEL_A",
            "class_probabilities": {"PRIVATE_LABEL_A": 0.8},
        }
        for i in range(120)
    ]
    service.runs["opaque"] = Run(
        "opaque",
        "classification",
        tmp_path / "unused",
        Mock(),
        status="completed",
        result=value,
    )
    response = client.get("/api/runs/opaque/csv")
    assert response.status_code == 200 and len(response.text.splitlines()) == 121
    assert "'=PRIVATE_FORMULA()" in response.text
    assert "PRIVATE_LABEL_A" in response.text
    report = client.get("/api/runs/opaque/report").json()
    assert "predictions" not in report and report["result_kind"] == "predictions"
    assert "PRIVATE" not in str(service.invoke("get_run", run_id="opaque"))


def test_release_hides_preview_and_invalidates_results(client):
    client.post("/api/import", files={"file": ("x.csv", b"x,y\n1,2", "text/csv")})
    assert client.post("/api/release").json()["ok"]
    state = client.get("/api/state").json()
    assert state["dataset"] is None and state["preview"] == []


def test_case_inputs_are_private_and_validated(service, client):
    import pandas as pd
    from conftest import task_for

    service.publish(pd.DataFrame({"x": range(30), "outcome": range(30)}), "csv")
    plan = service.prepare_prediction(task_for(service))["plan"]
    response = client.post("/api/cases/" + plan["id"], json={"x": "PRIVATE_CASE_INPUT"})
    assert response.status_code == 200 and "PRIVATE_CASE_INPUT" not in response.text
    assert "PRIVATE_CASE_INPUT" not in str(service.inspect_source())
    invalid = client.post(
        "/api/cases/" + plan["id"], json={"password": "PRIVATE_PASSWORD"}
    )
    assert invalid.status_code == 400 and "PRIVATE_PASSWORD" not in invalid.text
