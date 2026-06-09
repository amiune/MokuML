from __future__ import annotations

import hashlib
from typing import Any, Dict
from urllib.parse import SplitResult, urlsplit, urlunsplit


def mask_connection_string(connection_string: str) -> str:
    """Mask credentials while preserving enough host/db context for audit records."""
    if not connection_string:
        return ""

    try:
        parsed = urlsplit(connection_string)
    except ValueError:
        return "<invalid connection string>"

    if not parsed.netloc:
        return connection_string

    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    user = parsed.username or ""
    auth = f"{user}:****@" if user else ""
    masked = SplitResult(
        scheme=parsed.scheme,
        netloc=f"{auth}{host}{port}",
        path=parsed.path,
        query="",
        fragment="",
    )
    return urlunsplit(masked)


def database_type_from_url(connection_string: str) -> str:
    scheme = urlsplit(connection_string).scheme.lower()
    if scheme in {"postgresql", "postgresql+psycopg", "postgres"}:
        return "postgres"
    if scheme in {"sqlite", "sqlite3"}:
        return "sqlite"
    return "unsupported"


def stable_schema_hash(schema: Dict[str, Any]) -> str:
    normalized = repr(_sort_for_hash(schema)).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _sort_for_hash(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sort_for_hash(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sort_for_hash(item) for item in value]
    return value

