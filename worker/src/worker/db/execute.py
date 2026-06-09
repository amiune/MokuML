from __future__ import annotations

from typing import Any, Dict, List, Optional

from worker.db.urls import normalize_sqlalchemy_connection_string
from worker.sql.validator import validate_sql


def execute_preview(
    connection_string: str,
    sql: str,
    dialect: str = "postgres",
    schema: Optional[Dict[str, Any]] = None,
    preview_rows: int = 20,
    max_rows: int = 100000,
) -> Dict[str, Any]:
    validation = validate_sql(sql, dialect=dialect, schema=schema, max_rows=max_rows)
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
        preview_rows=preview_rows,
        max_rows=max_rows,
    )
    if not records_result.get("ok"):
        return records_result
    preview = records_result["records"][:preview_rows]
    return {
        "ok": True,
        "columns": records_result["columns"],
        "row_count": records_result["row_count"],
        "preview": preview,
        "warnings": validation.warnings + records_result.get("warnings", []),
        "normalized_sql": validation.normalized_sql,
    }


def execute_records(
    connection_string: str,
    sql: str,
    preview_rows: int = 20,
    max_rows: int = 100000,
) -> Dict[str, Any]:
    try:
        from sqlalchemy import create_engine, text
    except Exception as exc:
        return {
            "ok": False,
            "message": "Database execution dependencies are not installed.",
            "technical_details": str(exc),
            "suggested_fix": "Install SQLAlchemy and psycopg in the worker environment.",
        }

    engine = None
    try:
        engine = create_engine(normalize_sqlalchemy_connection_string(connection_string), pool_pre_ping=True)
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                connection.execute(text("SET LOCAL statement_timeout = '60s'"))
                connection.execute(text("SET LOCAL idle_in_transaction_session_timeout = '60s'"))
                connection.execute(text("SET LOCAL default_transaction_read_only = on"))
                connection.execute(text("SET TRANSACTION READ ONLY"))
                result = connection.execute(text(sql))
                rows = result.fetchmany(max_rows + 1)
                columns = [
                    {"name": key, "dtype": "unknown"}
                    for key in result.keys()
                ]
                records = [dict(row._mapping) for row in rows[:max_rows]]
                warnings: List[str] = []
                if len(rows) > max_rows:
                    warnings.append(f"Result exceeded max_rows={max_rows}; additional rows were not loaded.")
                transaction.rollback()
            except Exception:
                transaction.rollback()
                raise
        return {
            "ok": True,
            "columns": _infer_column_types(columns, records[:preview_rows]),
            "row_count": len(records),
            "records": records,
            "warnings": warnings,
        }
    except Exception as exc:
        return {
            "ok": False,
            "message": "SQL execution failed.",
            "technical_details": str(exc),
            "suggested_fix": "Check the generated SQL, table permissions, query timeout, and database connectivity.",
        }
    finally:
        if engine is not None:
            engine.dispose()


def _infer_column_types(columns: List[Dict[str, str]], records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    inferred = []
    for column in columns:
        name = column["name"]
        dtype = "unknown"
        for record in records:
            value = record.get(name)
            if value is None:
                continue
            if isinstance(value, bool):
                dtype = "boolean"
            elif isinstance(value, (int, float)):
                dtype = "number"
            else:
                dtype = "string"
            break
        inferred.append({"name": name, "dtype": dtype})
    return inferred
