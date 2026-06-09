from __future__ import annotations

from typing import Any, Dict, Optional

from .validator import validate_sql


def normalize_safe_sql(
    sql: str,
    dialect: str = "postgres",
    schema: Optional[Dict[str, Any]] = None,
    max_rows: int = 100000,
) -> str:
    result = validate_sql(sql, dialect=dialect, schema=schema, max_rows=max_rows)
    if not result.is_safe or not result.normalized_sql:
        raise ValueError("; ".join(result.errors) or "SQL is unsafe")
    return result.normalized_sql

