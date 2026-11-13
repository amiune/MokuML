"""Private source/plan/run lifecycle and strictly typed MCP projections."""

from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .catalog import Catalog, FileSource
from .contracts import AugmentedPredictionResult, CandidateResult, PredictionTask
from .data import MAX_COLUMNS, MAX_ROWS, Dataset, read_spreadsheet
from .databases import DatabaseConfig, SQLiteSource, SQLSource
from .errors import MESSAGES, SafeError
from .models import active_tabpfn_component, available, state_directory
from .planning import Compiler
from .public import ModelAvailability, RunSummary, metric_values, reply


@dataclass(repr=False)
class Run:
    id: str
    task: str
    directory: Path
    process: subprocess.Popen | None = None
    status: str = "running"
    result: dict | None = None
    error: str | None = None
    candidates: list = field(default_factory=list)
    cancelled: threading.Event = field(default_factory=threading.Event)


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


class Service:
    def __init__(self):
        self.lock = threading.RLock()
        self.dataset = self.source = self.catalog = self.window = None
        self.plans, self.runs, self.cases = {}, {}, {}
        self.setup_process, self.setup_error = None, False
        self.temporary = tempfile.TemporaryDirectory(
            prefix="session-", dir=state_directory()
        )
        Path(self.temporary.name).chmod(0o700)

    def _clear_dataset(self):
        for run in self.runs.values():
            run.cancelled.set()
            stop(run.process)
            shutil.rmtree(run.directory, ignore_errors=True)
        self.runs.clear()
        self.plans.clear()
        self.cases.clear()
        self.dataset = None

    def _close_source(self):
        if self.source:
            self.source.close()
        self.source = self.catalog = None

    def publish(self, frame, kind, primary_keys=(), disconnect=True):
        """Private convenience API for local imports and synthetic verification."""
        with self.lock:
            if disconnect or not isinstance(self.source, FileSource):
                self._clear_dataset()
                self._close_source()
                self.source = FileSource()
            self.source.add(frame, "table_" + str(len(self.source.tables) + 1))
            self.catalog = Catalog(self.source)
            self.dataset = Dataset.create(frame, kind, primary_keys)
            return reply(dataset=self.dataset.metadata, catalog=self.catalog.page())

    def import_file(self, content, filename, worksheet=None, scoring=False):
        frame, sheets = read_spreadsheet(content, filename, worksheet)
        frames = (
            [(name, read_spreadsheet(content, filename, name)[0]) for name in sheets]
            if frame is None
            else [(worksheet or "file_relation", frame)]
        )
        with self.lock:
            self._clear_dataset()
            if not isinstance(self.source, FileSource):
                self._close_source()
                self.source = FileSource()
            for name, data in frames:
                if name == "file_relation":
                    name = f"file_{len(self.source.tables) + 1}"
                self.source.add(data, name, scoring=scoring)
            self.catalog = Catalog(self.source)
            self.dataset = Dataset.create(
                frames[-1][1], "xlsx" if filename.lower().endswith(".xlsx") else "csv"
            )
            return reply(dataset=self.dataset.metadata, catalog=self.catalog.page())

    def _connect(self, source):
        with self.lock:
            try:
                catalog = Catalog(source)
            except Exception:
                source.close()
                raise
            self._clear_dataset()
            self._close_source()
            self.source, self.catalog = source, catalog
            return {
                "ok": True,
                "tables": [
                    {
                        "id": t.id,
                        "name": f"{t.schema_name + '.' if t.schema_name else ''}{t.name}",
                    }
                    for t in catalog.tables.values()
                ],
            }

    def connect_database(self, config):
        return self._connect(SQLSource(DatabaseConfig.model_validate(config)))

    def connect_sqlite(self, path):
        return self._connect(SQLiteSource(path))

    def load_table(self, table_id):
        with self.lock:
            if not self.catalog or table_id not in self.catalog.tables:
                raise SafeError("invalid_selection")
            table = self.catalog.tables[table_id]
            frame = self.source.sample(table_id, [c.name for c in table.columns]).head(
                30
            )
            self.dataset = Dataset.create(
                frame,
                self.source.kind,
                [c.name for c in table.columns if c.primary_key],
            )
            return reply(dataset=self.dataset.metadata)

    def inspect_source(self, cursor=0, limit=20):
        with self.lock:
            if not self.catalog:
                return reply("no_source", ok=False, models=self.models_status())
            self.catalog.refresh()
            return reply(
                catalog=self.catalog.page(cursor, limit), models=self.models_status()
            )

    def profile_data(self, table_id, column_ids=None, cursor=0, limit=20):
        with self.lock:
            if not self.catalog:
                raise SafeError("no_source")
            return reply(
                profile=self.catalog.profile(table_id, column_ids, cursor, limit)
            )

    def prepare_prediction(self, task):
        with self.lock:
            if not self.catalog:
                raise SafeError("no_source")
            self.catalog.refresh()
            definition, plan = Compiler(self.catalog).compile(task)
            self.plans[plan.id] = (
                definition.model_dump(mode="json"),
                plan.model_copy(deep=True),
            )
            while len(self.plans) > 20:
                key = next(iter(self.plans))
                self.plans.pop(key)
                self.cases.pop(key, None)
            return reply(plan=plan)

    def submit_case(self, plan_id, values):
        """Private UI only: records cannot be supplied through MCP."""
        with self.lock:
            if plan_id not in self.plans:
                raise SafeError("invalid_plan")
            _, plan = self.plans[plan_id]
            if not isinstance(values, dict) or set(values) != set(plan.feature_names):
                raise SafeError("scoring_mismatch")
            if any(
                not isinstance(v, (str, int, float, bool, type(None)))
                or len(str(v)) > 4096
                for v in values.values()
            ):
                raise SafeError("scoring_mismatch")
            self.cases[plan_id] = pd.DataFrame([values])
            return {
                "ok": True,
                "message": "Private case saved. Ask ChatGPT to start this prepared plan.",
            }

    def _materialize(self, source, task, plan, run, model, cases):
        try:
            frame = source.execute_query(plan.sql, cancel_event=run.cancelled)
            if run.cancelled.is_set():
                return
            if task.future:
                times = pd.to_datetime(frame["__as_of"], errors="coerce", utc=True)
                observed = pd.Timestamp(task.future.observed_through)
                observed = (
                    observed.tz_localize("UTC")
                    if observed.tzinfo is None
                    else observed.tz_convert("UTC")
                )
                frame["__score"] = times == observed
                frame["calendar_month"], frame["calendar_weekday"] = (
                    times.dt.month,
                    times.dt.weekday,
                )
            else:
                frame["__score"] = frame[plan.target].isna()
            if plan.scoring_sql:
                scoring = source.execute_query(
                    plan.scoring_sql, cancel_event=run.cancelled
                )
                scoring["__score"] = True
                frame = pd.concat([frame, scoring], ignore_index=True)
            if cases is not None:
                if task.future:
                    raise SafeError("scoring_mismatch")
                scoring = cases.copy()
                scoring[plan.target], scoring["__score"] = None, True
                frame = pd.concat([frame, scoring], ignore_index=True)
            if len(frame) > MAX_ROWS or len(frame.columns) > MAX_COLUMNS:
                raise SafeError("resource_limit")
            if not len(frame):
                raise SafeError("empty_data")
            grain = (
                ["__as_of"] + plan.identifier_names
                if task.future
                else plan.identifier_names + (["__as_of"] if "__as_of" in frame else [])
            )
            if grain:
                for _, rows in frame.groupby("__score"):
                    if rows[grain].dropna().duplicated().any():
                        raise SafeError("ambiguous_grain")
            baseline_feature = None
            if task.future and task.future.horizon in task.future.lookback_periods:
                for feature in task.features:
                    if feature.expression == task.target.expression:
                        baseline_feature = f"{feature.name}_past_{task.future.horizon}_{task.future.unit}"
                        break
            job = dict(
                frame=frame,
                target=plan.target,
                features=plan.feature_names,
                identifiers=plan.identifier_names,
                task=task.task,
                split=plan.split,
                model=model,
                units=plan.units,
                horizon=plan.horizon,
                output=task.output,
                baseline_feature=baseline_feature,
            )
            if task.future:
                row = frame.loc[frame["__score"]].iloc[0]
                for column, name in [
                    ("__as_of", "as_of"),
                    ("__label_start", "prediction_window_start"),
                    ("__label_end", "prediction_window_end"),
                ]:
                    value = pd.Timestamp(row[column])
                    value = (
                        value.tz_localize("UTC")
                        if value.tzinfo is None
                        else value.tz_convert("UTC")
                    )
                    job[name] = value.isoformat()
            with self.lock:
                if run.cancelled.is_set() or run.id not in self.runs:
                    return
                if task.future and task.future.entity_column_id:
                    job["identifier_labels"] = {
                        "__entity": self.catalog.columns[task.future.entity_column_id][
                            1
                        ].name
                    }
                else:
                    job["identifier_labels"] = {
                        f"__id_{i}": self.catalog.columns[c][1].name
                        for i, c in enumerate(task.identifier_column_ids)
                    }
                with (run.directory / "job.pickle").open("wb") as stream:
                    pickle.dump(job, stream)
                run.process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "mokuml_local.inference",
                        "--job",
                        str(run.directory / "job.pickle"),
                        "--output",
                        str(run.directory / "output.pickle"),
                        "--status",
                        str(run.directory / "status.json"),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )
        except Exception as exc:
            with self.lock:
                if run.id in self.runs and not run.cancelled.is_set():
                    run.status, run.error = (
                        "failed",
                        exc.code if isinstance(exc, SafeError) else "invalid_plan",
                    )
                    shutil.rmtree(run.directory, ignore_errors=True)

    def start_prediction(self, plan_id, model="auto"):
        with self.lock:
            if (
                model not in {"auto", "tabicl", "kumo", "tabpfn"}
                or plan_id not in self.plans
            ):
                raise SafeError("invalid_plan")
            if not self.catalog:
                raise SafeError("no_source")
            self.catalog.refresh()
            raw_task, plan = self.plans[plan_id]
            if (
                plan.source_id != self.catalog.id
                or plan.source_revision != self.catalog.revision
            ):
                raise SafeError("stale_plan")
            self._collect()
            if any(r.status == "running" for r in self.runs.values()):
                raise SafeError("busy")
            run_id = uuid.uuid4().hex
            directory = Path(tempfile.mkdtemp(prefix="run-", dir=self.temporary.name))
            (directory / "status.json").write_text(
                json.dumps({"stage": "materializing"})
            )
            run = Run(run_id, plan.task, directory)
            self.runs[run_id] = run
            while len(self.runs) > 10:
                self.runs.pop(next(iter(self.runs)))
            threading.Thread(
                target=self._materialize,
                args=(
                    self.source,
                    PredictionTask.model_validate(raw_task),
                    plan.model_copy(deep=True),
                    run,
                    model,
                    self.cases.get(plan_id),
                ),
                daemon=True,
            ).start()
            return reply(run=self._summary(run))

    def _collect(self):
        for run in self.runs.values():
            if (
                run.status != "running"
                or run.process is None
                or run.process.poll() is None
            ):
                continue
            try:
                with (run.directory / "output.pickle").open("rb") as stream:
                    output = pickle.load(stream)
                if output.get("ok"):
                    run.result, run.status = output["result"], "completed"
                    self._summary(run)
                else:
                    run.status = "failed"
                    run.error = (
                        output.get("code")
                        if output.get("code") in MESSAGES
                        else "inference_failed"
                    )
                    run.candidates = [
                        CandidateResult.model_validate(c)
                        for c in output.get("candidates", [])
                    ]
            except Exception:
                run.status, run.result, run.error = "failed", None, "inference_failed"
            finally:
                shutil.rmtree(run.directory, ignore_errors=True)

    def _summary(self, run):
        fields = dict(
            id=run.id,
            status=run.status,
            stage="finished",
            task=run.task,
            candidates=run.candidates,
        )
        if run.status == "running":
            try:
                stage = json.loads((run.directory / "status.json").read_text())["stage"]
                fields["stage"] = (
                    stage
                    if stage
                    in {
                        "preparing",
                        "materializing",
                        "embedding",
                        "selecting",
                        "evaluating",
                        "predicting",
                        "explaining",
                    }
                    else "preparing"
                )
            except Exception:
                fields["stage"] = "preparing"
        if run.status == "completed":
            result = AugmentedPredictionResult.model_validate(run.result["augmented"])
            fields.update(
                result=result,
                result_kind=result.result_kind,
                model=result.model.engine,
                model_version=result.model.version,
                encoder=result.model.encoder,
                device=result.model.device,
                validation_rows=result.test_rows,
                prediction_rows=result.prediction_rows,
                metrics=metric_values(result.metrics),
                baseline_metrics=metric_values(result.baseline_metrics),
            )
        return RunSummary(**fields)

    def get_run(self, run_id):
        with self.lock:
            self._collect()
            if run_id not in self.runs:
                raise SafeError("unknown_run")
            run = self.runs[run_id]
            return reply(
                run.error or "ready", ok=run.error is None, run=self._summary(run)
            )

    def cancel_run(self, run_id):
        with self.lock:
            if run_id not in self.runs:
                raise SafeError("unknown_run")
            run = self.runs[run_id]
            if run.status == "running":
                run.cancelled.set()
                stop(run.process)
                run.status = "cancelled"
                shutil.rmtree(run.directory, ignore_errors=True)
            return reply("cancelled", run=self._summary(run))

    def release_dataset(self):
        with self.lock:
            self._clear_dataset()
            self._close_source()
            return reply("released")

    def models_status(self):
        if self.setup_process is not None and self.setup_process.poll() is not None:
            self.setup_error = self.setup_process.returncode != 0
            self.setup_process = None
        return ModelAvailability(
            tabicl=available("tabicl"),
            qwen=available("qwen"),
            tabpfn=available(active_tabpfn_component()),
            kumo=available("kumo"),
            downloading=self.setup_process is not None,
        )

    def setup_models(
        self, text=False, component="all", licensed_opt_in=False, token=None
    ):
        with self.lock:
            if component not in {
                "all",
                "tabicl",
                "kumo",
                "tabpfn",
                "tabpfn35",
                "qwen",
            } or (component == "tabpfn35" and not licensed_opt_in):
                raise SafeError("invalid_selection")
            if self.models_status().downloading:
                return reply("setup_running", models=self.models_status())
            command = [
                sys.executable,
                "-m",
                "mokuml_local.models",
                "--component",
                component,
            ]
            if text:
                command.append("--text")
            if licensed_opt_in:
                command.append("--licensed-opt-in")
            env = {**os.environ}
            if token:
                env["MOKUML_SETUP_TOKEN"] = token
            self.setup_error = False
            self.setup_process = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
            return reply("setup_running", models=self.models_status())

    def describe_dataset(self):
        return reply(
            "ready" if self.dataset else "no_dataset",
            ok=self.dataset is not None,
            dataset=self.dataset.metadata if self.dataset else None,
            models=self.models_status(),
        )

    def invoke(self, operation, **kwargs):
        try:
            if operation not in {
                "inspect_source",
                "profile_data",
                "prepare_prediction",
                "describe_dataset",
                "start_prediction",
                "get_run",
                "cancel_run",
                "release_dataset",
            }:
                raise SafeError("invalid_selection")
            return getattr(self, operation)(**kwargs)
        except SafeError as exc:
            return reply(exc.code, ok=False)
        except Exception:
            return reply("internal_error", ok=False)

    def local_snapshot(self):
        with self.lock:
            self._collect()
            preview = (
                json.loads(
                    self.dataset.frame.head(30).to_json(
                        orient="records", date_format="iso"
                    )
                )
                if self.dataset
                else []
            )
            return dict(
                dataset=self.dataset.metadata.model_dump() if self.dataset else None,
                catalog=self.catalog.page().model_dump() if self.catalog else None,
                preview=preview,
                runs=[self.get_run(k) for k in reversed(self.runs)],
                plans=[p.model_dump() for _, p in self.plans.values()],
                models=self.models_status().model_dump(),
                setup_error=self.setup_error,
            )

    def private_result(self, run_id):
        with self.lock:
            self._collect()
            run = self.runs.get(run_id)
            if not run or run.status != "completed":
                raise SafeError("unknown_run")
            return run.result

    def close(self):
        with self.lock:
            stop(self.setup_process)
            self._clear_dataset()
            self._close_source()
            self.temporary.cleanup()
