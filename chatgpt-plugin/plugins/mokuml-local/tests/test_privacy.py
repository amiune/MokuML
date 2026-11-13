import json
import pickle
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from mokuml_local import service as implementation
from mokuml_local.public import Metrics, Reply


def test_public_contract_rejects_private_fields_and_nonfinite_metrics():
    with pytest.raises(ValidationError):
        Reply(ok=True, code="ready", message="ready", predictions=["secret"])
    with pytest.raises(ValidationError):
        Metrics(mae=float("nan"))
    with pytest.raises(ValidationError):
        Metrics(accuracy=2.0)


def test_only_metadata_enters_mcp_and_release_invalidates_ids(service, business_frame):
    service.publish(business_frame, "csv")
    result = service.invoke("describe_dataset")
    assert result["dataset"]["row_count"] == 6
    assert "PRIVATE_" not in json.dumps(result)
    old_id = service.dataset.metadata.id
    service.publish(business_frame, "sqlite")
    result = service.invoke(
        "start_prediction",
        dataset_id=old_id,
        target_column_id="bad",
        feature_column_ids=[],
        identifier_column_ids=[],
        task="classification",
    )
    assert not result["ok"]
    assert service.invoke("release_dataset")["ok"]
    assert not service.invoke("describe_dataset")["ok"]


def test_unhandled_errors_are_fixed_messages(service, monkeypatch):
    monkeypatch.setattr(
        service,
        "describe_dataset",
        Mock(
            side_effect=RuntimeError(
                "PRIVATE_CELL_SENTINEL postgres://owner:password@server/private"
            )
        ),
    )
    result = service.invoke("describe_dataset")
    assert result["code"] == "internal_error"
    assert "PRIVATE" not in json.dumps(result)
    assert "password" not in json.dumps(result)


def test_task_cannot_use_outcome_or_identifiers_as_features(service, business_frame):
    from conftest import task_for

    service.publish(business_frame, "csv")
    task = task_for(service, features=["amount"], task="classification")
    task["features"][0]["expression"] = task["target"]["expression"]
    assert service.invoke("prepare_prediction", task=task)["code"] == "invalid_plan"


def completed_result():
    return {
        "augmented": dict(
            task="regression",
            result_kind="predictions",
            target="outcome",
            model=dict(
                engine="tabicl",
                version="2.2.0",
                checkpoint_revision="pinned",
                encoder="none",
                device="cpu",
            ),
            selection_metric="mae",
            candidates=[],
            split="random",
            training_rows=60,
            calibration_rows=20,
            test_rows=20,
            prediction_rows=1,
            metrics={"mae": 1.0},
            baseline_metrics={"mae": 2.0},
            probability_calibration="not_applicable",
            interval_method="unavailable",
            predictions=[{"case_id": "case-1", "prediction": 12.0}],
            summary={"count": 1, "mean": 12.0},
            feature_influences=[],
            data_quality=[],
            limitations=[],
        ),
        "predictions": [
            {"row_number": 1, "identifiers": {"id": "PRIVATE_ID"}, "prediction": 12.0}
        ],
        "private_extra": "PRIVATE_CELL_SENTINEL",
    }


def test_completed_output_is_projected_and_private_job_files_removed(service, tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    with (directory / "output.pickle").open("wb") as stream:
        pickle.dump({"ok": True, "result": completed_result()}, stream)
    (directory / "job.pickle").write_bytes(b"PRIVATE_CELL_SENTINEL")
    process = Mock()
    process.poll.return_value = 0
    run = implementation.Run("opaque-id", "classification", directory, process)
    service.runs[run.id] = run
    result = service.invoke("get_run", run_id=run.id)
    assert result["run"]["metrics"] == {"mae": 1.0}
    assert "PRIVATE" not in json.dumps(result)
    assert "query_results" not in json.dumps(result)
    assert not directory.exists()
    assert service.private_result(run.id)["predictions"][0]["prediction"] == 12.0


def test_worker_error_cannot_be_used_to_exfiltrate_values(service, tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    with (directory / "output.pickle").open("wb") as stream:
        pickle.dump({"ok": False, "code": "PRIVATE_CELL_SENTINEL"}, stream)
    process = Mock()
    process.poll.return_value = 1
    service.runs["run"] = implementation.Run("run", "regression", directory, process)
    result = service.invoke("get_run", run_id="run")
    assert result["code"] == "inference_failed" and "PRIVATE" not in json.dumps(result)


def test_cancel_kills_worker_and_removes_data(service, tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    process = Mock()
    process.poll.return_value = None
    service.runs["run"] = implementation.Run("run", "regression", directory, process)
    result = service.invoke("cancel_run", run_id="run")
    assert result["run"]["status"] == "cancelled"
    process.terminate.assert_called_once()
    assert not directory.exists()


def test_private_session_permissions_and_shutdown(service):
    directory = Path(service.temporary.name)
    assert directory.stat().st_mode & 0o777 == 0o700
    service.close()
    assert not directory.exists()
