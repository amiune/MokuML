from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from worker.db.urls import normalize_sqlalchemy_connection_string


TABLES_SQL = """
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_type = 'BASE TABLE'
  AND table_schema NOT IN ('pg_catalog', 'information_schema')
ORDER BY table_schema, table_name
"""

COLUMNS_SQL = """
SELECT table_schema, table_name, column_name, data_type, is_nullable, ordinal_position
FROM information_schema.columns
WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
ORDER BY table_schema, table_name, ordinal_position
"""

PRIMARY_KEYS_SQL = """
SELECT tc.table_schema, tc.table_name, kcu.column_name
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
  ON tc.constraint_name = kcu.constraint_name
 AND tc.table_schema = kcu.table_schema
WHERE tc.constraint_type = 'PRIMARY KEY'
  AND tc.table_schema NOT IN ('pg_catalog', 'information_schema')
"""

FOREIGN_KEYS_SQL = """
SELECT
  tc.table_schema,
  tc.table_name,
  kcu.column_name,
  ccu.table_schema AS foreign_table_schema,
  ccu.table_name AS foreign_table_name,
  ccu.column_name AS foreign_column_name
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
  ON tc.constraint_name = kcu.constraint_name
 AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage AS ccu
  ON ccu.constraint_name = tc.constraint_name
 AND ccu.table_schema = tc.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY'
  AND tc.table_schema NOT IN ('pg_catalog', 'information_schema')
"""

INDEXES_SQL = """
SELECT schemaname AS table_schema, tablename AS table_name, indexname AS index_name, indexdef
FROM pg_indexes
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY schemaname, tablename, indexname
"""


def introspect_postgres(
    connection_string: str,
    include_schemas: Optional[Iterable[str]] = None,
    exclude_tables: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    try:
        from sqlalchemy import create_engine, text
    except Exception as exc:
        return {
            "ok": False,
            "message": "PostgreSQL dependencies are not installed in the worker environment.",
            "technical_details": str(exc),
            "suggested_fix": "Install worker dependencies with `pip install -e .` inside the worker directory.",
        }

    include_schema_set = set(include_schemas or [])
    exclude_table_set = {_qualified_name(table) for table in (exclude_tables or [])}

    engine = None
    try:
        engine = create_engine(normalize_sqlalchemy_connection_string(connection_string), pool_pre_ping=True)
        with engine.connect() as connection:
            connection.execute(text("SET statement_timeout = '60s'"))
            table_rows = connection.execute(text(TABLES_SQL)).mappings().all()
            column_rows = connection.execute(text(COLUMNS_SQL)).mappings().all()
            pk_rows = connection.execute(text(PRIMARY_KEYS_SQL)).mappings().all()
            fk_rows = connection.execute(text(FOREIGN_KEYS_SQL)).mappings().all()
            index_rows = connection.execute(text(INDEXES_SQL)).mappings().all()

        schema = _build_schema(
            table_rows,
            column_rows,
            pk_rows,
            fk_rows,
            index_rows,
            include_schema_set,
            exclude_table_set,
        )
        return {"ok": True, "schema": schema}
    except Exception as exc:
        return {
            "ok": False,
            "message": "Schema introspection failed.",
            "technical_details": str(exc),
            "suggested_fix": "Confirm the user can read information_schema and pg_catalog metadata.",
        }
    finally:
        if engine is not None:
            engine.dispose()


def _build_schema(
    table_rows: List[Any],
    column_rows: List[Any],
    pk_rows: List[Any],
    fk_rows: List[Any],
    index_rows: List[Any],
    include_schemas: Set[str],
    exclude_tables: Set[str],
) -> Dict[str, Any]:
    tables: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in table_rows:
        schema_name = row["table_schema"]
        table_name = row["table_name"]
        if include_schemas and schema_name not in include_schemas:
            continue
        if _qualified_name(schema_name, table_name) in exclude_tables:
            continue
        tables[(schema_name, table_name)] = {
            "name": table_name,
            "columns": [],
            "foreign_keys": [],
            "indexes": [],
        }

    primary_keys = defaultdict(set)
    for row in pk_rows:
        primary_keys[(row["table_schema"], row["table_name"])].add(row["column_name"])

    for row in column_rows:
        key = (row["table_schema"], row["table_name"])
        if key not in tables:
            continue
        tables[key]["columns"].append(
            {
                "name": row["column_name"],
                "type": row["data_type"],
                "nullable": row["is_nullable"] == "YES",
                "primary_key": row["column_name"] in primary_keys[key],
            }
        )

    for row in fk_rows:
        key = (row["table_schema"], row["table_name"])
        if key not in tables:
            continue
        tables[key]["foreign_keys"].append(
            {
                "column": row["column_name"],
                "references_schema": row["foreign_table_schema"],
                "references_table": row["foreign_table_name"],
                "references_column": row["foreign_column_name"],
            }
        )

    for row in index_rows:
        key = (row["table_schema"], row["table_name"])
        if key not in tables:
            continue
        tables[key]["indexes"].append(
            {
                "name": row["index_name"],
                "definition": row["indexdef"],
            }
        )

    schemas: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for (schema_name, _), table in sorted(tables.items()):
        schemas[schema_name].append(table)

    return {
        "dialect": "postgres",
        "schemas": [
            {"name": schema_name, "tables": schema_tables}
            for schema_name, schema_tables in sorted(schemas.items())
        ],
    }


def _qualified_name(*parts: str) -> str:
    if len(parts) == 1:
        return str(parts[0]).lower()
    return ".".join(part for part in parts if part).lower()
