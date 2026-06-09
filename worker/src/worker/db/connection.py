from __future__ import annotations

from typing import Any, Dict
from urllib.parse import urlsplit

from worker.security.masking import database_type_from_url, mask_connection_string
from worker.db.urls import normalize_sqlalchemy_connection_string


def test_connection(connection_string: str) -> Dict[str, Any]:
    database_type = database_type_from_url(connection_string)
    if database_type != "postgres":
        return {
            "ok": False,
            "database_type": database_type,
            "message": "Only PostgreSQL connection strings are supported in v1.",
    }

    try:
        from sqlalchemy import create_engine, text
    except Exception as exc:
        return {
            "ok": False,
            "database_type": "postgres",
            "message": "PostgreSQL dependencies are not installed in the worker environment.",
            "technical_details": _safe_error(exc, connection_string),
            "suggested_fix": "Create the worker environment with `cd worker && uv venv --python 3.11 && uv pip install -e .`, then restart the app.",
        }

    engine = None
    try:
        engine = create_engine(normalize_sqlalchemy_connection_string(connection_string), pool_pre_ping=True)
        with engine.connect() as connection:
            connection.execute(text("SET statement_timeout = '10s'"))
            version = connection.execute(text("SHOW server_version")).scalar_one()
        return {
            "ok": True,
            "database_type": "postgres",
            "server_version": str(version),
            "message": "Connection successful",
            "masked_connection": mask_connection_string(connection_string),
        }
    except Exception as exc:
        return {
            "ok": False,
            "database_type": "postgres",
            "message": "Database connection failed.",
            "technical_details": _safe_error(exc, connection_string),
            "suggested_fix": "Check host, port, database name, credentials, VPN, and database firewall rules.",
            "masked_connection": mask_connection_string(connection_string),
        }
    finally:
        if engine is not None:
            engine.dispose()


def _safe_error(exc: Exception, connection_string: str) -> str:
    raw = str(exc)
    masked = mask_connection_string(connection_string)
    if connection_string:
        raw = raw.replace(connection_string, masked)
    try:
        parsed = urlsplit(connection_string)
    except ValueError:
        return raw
    if parsed.password:
        raw = raw.replace(parsed.password, "****")
    return raw
