from __future__ import annotations

from typing import Any, Dict

from fastapi import Body, FastAPI

from worker.api import handle


app = FastAPI(
    title="MokuML Local Worker",
    version="0.1.0",
    description="Local-only API for schema introspection, safe SQL, and tabular predictions.",
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return handle("GET", "/health")


@app.post("/connect/test")
def connect_test(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return handle("POST", "/connect/test", payload)


@app.post("/schema/introspect")
def schema_introspect(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return handle("POST", "/schema/introspect", payload)


@app.post("/llm/generate-sql")
def generate_sql(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return handle("POST", "/llm/generate-sql", payload)


@app.post("/llm/ollama-status")
def ollama_status(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return handle("POST", "/llm/ollama-status", payload)


@app.post("/llm/build-prompt")
def build_llm_prompt(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return handle("POST", "/llm/build-prompt", payload)


@app.post("/llm/parse-sql")
def parse_llm_sql(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return handle("POST", "/llm/parse-sql", payload)


@app.post("/sql/validate")
def validate_sql(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return handle("POST", "/sql/validate", payload)


@app.post("/sql/execute-preview")
def execute_preview(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return handle("POST", "/sql/execute-preview", payload)


@app.post("/ml/run-tabpfn")
def run_tabpfn(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return handle("POST", "/ml/run-tabpfn", payload)
