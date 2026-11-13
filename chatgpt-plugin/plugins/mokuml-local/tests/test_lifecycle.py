import threading
from unittest.mock import Mock

import pandas as pd
from conftest import task_for


def test_cancel_during_materialization_does_not_launch_inference(service, monkeypatch):
    service.publish(pd.DataFrame({"x": range(30), "outcome": range(30)}), "csv")
    plan = service.prepare_prediction(task_for(service))["plan"]
    entered, finished = threading.Event(), threading.Event()

    def read(sql, cancel_event=None):
        entered.set()
        cancel_event.wait(5)
        finished.set()
        return pd.DataFrame({"x": [1], "outcome": [2]})

    monkeypatch.setattr(service.source, "execute_query", read)
    child = Mock(side_effect=AssertionError("Cancelled run cannot launch"))
    monkeypatch.setattr("mokuml_local.service.subprocess.Popen", child)
    run = service.start_prediction(plan["id"])["run"]
    assert entered.wait(2)
    service.cancel_run(run["id"])
    assert finished.wait(2)
    assert service.get_run(run["id"])["run"]["status"] == "cancelled"
    child.assert_not_called()
    assert not service.runs[run["id"]].directory.exists()


def test_newer_tabpfn_requires_explicit_local_opt_in(service, monkeypatch):
    import pytest

    from mokuml_local.errors import SafeError

    child = Mock()
    monkeypatch.setattr("mokuml_local.service.subprocess.Popen", child)
    with pytest.raises(SafeError):
        service.setup_models(component="tabpfn35", token="PRIVATE_TOKEN")
    child.assert_not_called()
    assert "PRIVATE_TOKEN" not in str(service.inspect_source())
