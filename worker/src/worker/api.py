from __future__ import annotations

import importlib.util
import json
import sys
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from worker import __version__
from worker.db.connection import test_connection
from worker.db.execute import execute_preview, execute_records
from worker.db.introspection_postgres import introspect_postgres
from worker.llm.output_parser import parse_llm_sql_response
from worker.llm.prompt_builder import build_chat_messages, build_user_payload
from worker.llm.providers import generate_sql_with_provider
from worker.security.masking import database_type_from_url
from worker.security.privacy import assert_llm_payload_is_private
from worker.sql.validator import validate_sql


def handle_health(_: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "ok": True,
        "version": __version__,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "port_hint": "The Swift app chooses the port and passes it to worker.main.",
        "capabilities": {
            "fastapi": _module_available("fastapi"),
            "sqlglot": _module_available("sqlglot"),
            "sqlalchemy": _module_available("sqlalchemy"),
            "pandas": _module_available("pandas"),
            "sklearn": _module_available("sklearn"),
            "tabpfn": _module_available("tabpfn"),
        },
        "warnings": _dependency_warnings(),
    }


def handle_connect_test(payload: Dict[str, Any]) -> Dict[str, Any]:
    connection_string = payload.get("connection_string", "")
    if not connection_string:
        return {"ok": False, "message": "Connection string is required."}
    return test_connection(connection_string)


def handle_schema_introspect(payload: Dict[str, Any]) -> Dict[str, Any]:
    connection_string = payload.get("connection_string", "")
    if database_type_from_url(connection_string) != "postgres":
        return {"ok": False, "message": "Only PostgreSQL introspection is supported in v1."}
    return introspect_postgres(
        connection_string,
        include_schemas=payload.get("include_schemas") or ["public"],
        exclude_tables=payload.get("exclude_tables") or [],
    )


def handle_generate_sql(payload: Dict[str, Any]) -> Dict[str, Any]:
    prediction_prompt = payload.get("prediction_prompt", "")
    schema = payload.get("schema") or {}
    dialect = schema.get("dialect", "postgres")
    provider = payload.get("llm_provider") or {"type": "local_ollama"}
    user_payload = build_user_payload(prediction_prompt, schema, dialect=dialect)
    try:
        assert_llm_payload_is_private(user_payload)
        parsed = generate_sql_with_provider(user_payload, provider)
        return {"ok": True, **parsed}
    except Exception as exc:
        return {
            "ok": False,
            "message": "SQL generation failed.",
            "technical_details": str(exc),
            "suggested_fix": "Check the LLM provider settings or switch to local offline template mode for a dry run.",
        }


def handle_build_llm_prompt(payload: Dict[str, Any]) -> Dict[str, Any]:
    prediction_prompt = payload.get("prediction_prompt", "")
    schema = payload.get("schema") or {}
    dialect = schema.get("dialect", "postgres")
    user_payload = build_user_payload(prediction_prompt, schema, dialect=dialect)
    try:
        assert_llm_payload_is_private(user_payload)
        return {"ok": True, "messages": build_chat_messages(user_payload)}
    except Exception as exc:
        return {
            "ok": False,
            "message": "Could not build LLM prompt.",
            "technical_details": str(exc),
        }


def handle_parse_llm_sql(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw_content = payload.get("raw_content", "")
    try:
        parsed = parse_llm_sql_response(raw_content)
        return {"ok": True, **parsed}
    except Exception as exc:
        return {
            "ok": False,
            "message": "SQL generation failed.",
            "technical_details": str(exc),
            "suggested_fix": "The LLM returned output that was not valid SQL-generation JSON.",
        }


def handle_ollama_status(payload: Dict[str, Any]) -> Dict[str, Any]:
    endpoint = str(payload.get("endpoint") or "http://127.0.0.1:11434").rstrip("/")
    model = str(payload.get("model") or "sqlcoder")
    try:
        request = Request(f"{endpoint}/api/tags", method="GET")
        with urlopen(request, timeout=3) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return {
            "ok": False,
            "server_running": False,
            "model_available": False,
            "message": f"Ollama returned HTTP {exc.code}.",
            "technical_details": exc.read().decode("utf-8", errors="replace"),
        }
    except URLError as exc:
        return {
            "ok": False,
            "server_running": False,
            "model_available": False,
            "message": "Ollama is not running at the configured endpoint.",
            "technical_details": str(exc.reason),
            "suggested_fix": "Start Ollama, then pull the configured model.",
        }
    except Exception as exc:
        return {
            "ok": False,
            "server_running": False,
            "model_available": False,
            "message": "Could not check Ollama status.",
            "technical_details": str(exc),
        }

    models = body.get("models") if isinstance(body, dict) else []
    model_names = sorted(
        name
        for item in models
        if isinstance(item, dict)
        for name in [item.get("name"), item.get("model")]
        if isinstance(name, str)
    )
    model_available = any(name == model or name.split(":")[0] == model for name in model_names)
    if model_available:
        message = f"Ollama is ready with {model}."
    elif model_names:
        message = f"Ollama is running, but {model} is not installed."
    else:
        message = "Ollama is running, but no models are installed."
    return {
        "ok": model_available,
        "server_running": True,
        "model_available": model_available,
        "message": message,
        "installed_models": model_names,
        "suggested_fix": None if model_available else f"Run: ollama pull {model}",
    }


def handle_sql_validate(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = validate_sql(
        payload.get("sql", ""),
        dialect=payload.get("dialect", "postgres"),
        schema=payload.get("schema"),
        max_rows=int(payload.get("max_rows", 100000)),
        allow_schemas=payload.get("allow_schemas"),
    )
    return result.to_dict()


def handle_execute_preview(payload: Dict[str, Any]) -> Dict[str, Any]:
    connection_string = payload.get("connection_string", "")
    sql = payload.get("sql", "")
    return execute_preview(
        connection_string,
        sql,
        dialect=payload.get("dialect", "postgres"),
        schema=payload.get("schema"),
        preview_rows=int(payload.get("preview_rows", 20)),
        max_rows=int(payload.get("max_rows", 100000)),
    )


def handle_run_tabpfn(payload: Dict[str, Any]) -> Dict[str, Any]:
    from worker.ml.tabpfn_runner import run_tabular_predictions

    connection_string = payload.get("connection_string", "")
    sql = payload.get("sql", "")
    schema = payload.get("schema")
    target_column = payload.get("target_column")
    entity_id_columns = payload.get("entity_id_columns") or []
    if not target_column:
        return {"ok": False, "message": "target_column is required."}
    validation = validate_sql(
        sql,
        dialect=payload.get("dialect", "postgres"),
        schema=schema,
        max_rows=int(payload.get("max_rows", 100000)),
    )
    if not validation.is_safe or not validation.normalized_sql:
        return {
            "ok": False,
            "message": "SQL did not pass safety validation.",
            "errors": validation.errors,
            "warnings": validation.warnings,
        }

    records_result = execute_records(
        connection_string,
        validation.normalized_sql,
        max_rows=int(payload.get("max_rows", 100000)),
    )
    if not records_result.get("ok"):
        return records_result

    return run_tabular_predictions(
        records_result["records"],
        target_column=target_column,
        entity_id_columns=entity_id_columns,
        task_type=payload.get("task_type", "unknown"),
        mode=payload.get("mode", "auto"),
        test_size=float(payload.get("test_size", 0.2)),
        random_seed=int(payload.get("random_seed", 42)),
    )


ROUTES = {
    "GET /health": handle_health,
    "POST /connect/test": handle_connect_test,
    "POST /schema/introspect": handle_schema_introspect,
    "POST /llm/ollama-status": handle_ollama_status,
    "POST /llm/build-prompt": handle_build_llm_prompt,
    "POST /llm/parse-sql": handle_parse_llm_sql,
    "POST /llm/generate-sql": handle_generate_sql,
    "POST /sql/validate": handle_sql_validate,
    "POST /sql/execute-preview": handle_execute_preview,
    "POST /ml/run-tabpfn": handle_run_tabpfn,
}


def handle(method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    route = f"{method.upper()} {path}"
    handler = ROUTES.get(route)
    if not handler:
        return {"ok": False, "message": f"Unknown endpoint: {route}"}
    return handler(payload or {})


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _dependency_warnings() -> list[str]:
    warnings = []
    if not _module_available("sqlglot"):
        warnings.append("sqlglot is not installed; SQL validation will use a conservative fallback.")
    if not _module_available("sqlalchemy"):
        warnings.append("SQLAlchemy/psycopg are not installed; Postgres connection and execution endpoints are unavailable.")
    if not _module_available("tabpfn"):
        warnings.append("TabPFN is not installed; ML endpoint will use sklearn or baseline fallback if possible.")
    return warnings
