"""Authenticated loopback-only UI. These routes are private and are never MCP tools."""

from __future__ import annotations

import csv
import hmac
import json
import secrets
import socket
import threading
import time
import webbrowser
from io import StringIO
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict

from .data import MAX_BYTES
from .databases import choose_sqlite_file
from .errors import SafeError
from .public import reply


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan_id: str
    model: str = "auto"


def create_app(service, token: str, authority: str) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    origin = "http://" + authority
    assets = Path(__file__).with_name("assets")

    @app.middleware("http")
    async def privacy_boundary(request: Request, call_next):
        if request.headers.get("host") != authority:
            return JSONResponse({"message": "Invalid local host."}, status_code=403)
        if request.headers.get("origin") not in {None, origin}:
            return JSONResponse(
                {"message": "Cross-origin requests are disabled."}, status_code=403
            )
        if request.url.path.startswith("/api/"):
            if not hmac.compare_digest(
                request.headers.get("authorization", ""), "Bearer " + token
            ):
                return JSONResponse(
                    {"message": "Open the local window again from MokuML."},
                    status_code=401,
                )
            if request.method != "GET" and request.headers.get("origin") != origin:
                return JSONResponse(
                    {"message": "Use the local window for this operation."},
                    status_code=403,
                )
            try:
                if (
                    int(request.headers.get("content-length", "0"))
                    > MAX_BYTES + 1024 * 1024
                ):
                    return JSONResponse(
                        reply("file_too_large", ok=False), status_code=413
                    )
            except ValueError:
                return JSONResponse(reply("internal_error", ok=False), status_code=400)
        try:
            response = await call_next(request)
        except Exception:
            response = JSONResponse(reply("internal_error", ok=False), status_code=500)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        )
        return response

    @app.exception_handler(SafeError)
    async def known_error(request, exc):
        return JSONResponse(reply(exc.code, ok=False), status_code=400)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(reply("invalid_selection", ok=False), status_code=422)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (assets / "index.html").read_text()

    @app.get("/assets/{name}")
    def asset(name: str):
        if name not in {"app.js", "style.css"}:
            return Response(status_code=404)
        return Response(
            (assets / name).read_bytes(),
            media_type="text/javascript" if name.endswith(".js") else "text/css",
        )

    @app.get("/api/state")
    def state():
        return service.local_snapshot()

    @app.post("/api/import")
    async def import_file(
        file: UploadFile = File(...),
        worksheet: str | None = Form(None),
        scoring: bool = Form(False),
    ):
        try:
            content = await file.read(MAX_BYTES + 1)
            return service.import_file(
                content, file.filename or "", worksheet, scoring=scoring
            )
        finally:
            await file.close()

    @app.post("/api/database/connect")
    def connect(config: dict):
        try:
            return service.connect_database(config)
        except SafeError:
            raise
        except Exception:
            raise SafeError("database_failed") from None

    @app.post("/api/sqlite/open")
    def sqlite_open():
        return service.connect_sqlite(choose_sqlite_file())

    @app.post("/api/database/load")
    def load(body: dict):
        return service.load_table(body.get("table_id"))

    @app.post("/api/predict")
    def predict(body: PredictionRequest):
        return service.invoke("start_prediction", **body.model_dump())

    @app.post("/api/models")
    def setup_models(body: dict):
        return service.setup_models(
            text=body.get("text") is True,
            component=body.get("component", "all"),
            licensed_opt_in=body.get("licensed_opt_in") is True,
            token=body.get("token"),
        )

    @app.post("/api/cases/{plan_id}")
    def submit_case(plan_id: str, body: dict):
        return service.submit_case(plan_id, body)

    @app.post("/api/release")
    def release():
        return service.invoke("release_dataset")

    @app.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: str):
        return service.invoke("cancel_run", run_id=run_id)

    @app.get("/api/runs/{run_id}")
    def result(run_id: str):
        return service.private_result(run_id)

    @app.get("/api/runs/{run_id}/report")
    def report(run_id: str):
        value = {
            key: value
            for key, value in service.private_result(run_id)["augmented"].items()
            if key != "predictions"
        }
        return Response(
            json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2),
            media_type="application/json",
            headers={
                "Content-Disposition": 'attachment; filename="mokuml-validation-report.json"'
            },
        )

    @app.get("/api/runs/{run_id}/csv")
    def export(run_id: str):
        predictions = service.private_result(run_id)["predictions"]
        rows = []
        for prediction in predictions:
            row = {
                "record": prediction["row_number"],
                **{
                    "id:" + key: value
                    for key, value in prediction["identifiers"].items()
                },
                "prediction": prediction["prediction"],
            }
            if "actual_outcome" in prediction:
                row["actual_outcome"] = prediction["actual_outcome"]
            if "lower" in prediction:
                row.update(
                    lower=prediction["lower"],
                    upper=prediction["upper"],
                    interval_level=prediction["interval_level"],
                )
            row.update(
                {
                    "probability:" + key: value
                    for key, value in prediction.get(
                        "probabilities", prediction.get("class_probabilities", {})
                    ).items()
                }
            )
            # Prevent spreadsheet formula execution in exported text identifiers and labels.
            rows.append(
                {
                    key: "'" + value
                    if isinstance(value, str)
                    and value.startswith(("=", "+", "-", "@", "\t", "\r"))
                    else value
                    for key, value in row.items()
                }
            )
        stream = StringIO()
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]) if rows else ["record", "prediction"]
        )
        writer.writeheader()
        writer.writerows(rows)
        return Response(
            stream.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": 'attachment; filename="mokuml-predictions.csv"'
            },
        )

    return app


class LocalWindow:
    def __init__(self, service):
        self.token = secrets.token_urlsafe(32)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(128)
        self.authority = "127.0.0.1:" + str(self.socket.getsockname()[1])
        app = create_app(service, self.token, self.authority)
        self.server = uvicorn.Server(
            uvicorn.Config(app, access_log=False, log_config=None, log_level="critical")
        )
        self.thread = threading.Thread(
            target=lambda: self.server.run(sockets=[self.socket]), daemon=True
        )
        self.thread.start()
        deadline = time.monotonic() + 10
        while (
            not self.server.started
            and self.thread.is_alive()
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        if not self.server.started:
            self.close()
            raise SafeError("window_failed")

    def open(self):
        # The capability URL is passed directly to the system browser, never to MCP.
        opened = webbrowser.open("http://" + self.authority + "/#" + self.token)
        return reply("window_opened" if opened else "window_failed", ok=bool(opened))

    def close(self):
        self.server.should_exit = True
        self.thread.join(timeout=5)
        self.socket.close()
