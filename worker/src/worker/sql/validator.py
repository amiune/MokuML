from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


FORBIDDEN_KEYWORDS = {
    "delete",
    "insert",
    "update",
    "upsert",
    "merge",
    "drop",
    "alter",
    "create",
    "truncate",
    "grant",
    "revoke",
    "copy",
    "call",
    "execute",
    "do",
    "vacuum",
    "analyze",
    "listen",
    "notify",
}

FORBIDDEN_FUNCTIONS = {
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "pg_stat_file",
    "lo_import",
    "lo_export",
    "dblink",
    "postgres_fdw_handler",
}


@dataclass
class ValidationResult:
    ok: bool
    is_safe: bool
    errors: List[str] = field(default_factory=list)
    normalized_sql: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    validator: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "is_safe": self.is_safe,
            "errors": self.errors,
            "normalized_sql": self.normalized_sql,
            "warnings": self.warnings,
            "validator": self.validator,
        }


def validate_sql(
    sql: str,
    dialect: str = "postgres",
    schema: Optional[Dict[str, Any]] = None,
    max_rows: int = 100000,
    allow_schemas: Optional[Iterable[str]] = None,
) -> ValidationResult:
    if not sql or not sql.strip():
        return _reject("SQL is empty")

    base_errors = _base_text_errors(sql)
    if base_errors:
        return ValidationResult(ok=True, is_safe=False, errors=base_errors, validator="base")

    try:
        import sqlglot
        from sqlglot import exp
    except Exception:
        return _fallback_validate(sql, schema=schema, max_rows=max_rows, allow_schemas=allow_schemas)

    stripped = _strip_optional_trailing_semicolon(sql)
    try:
        expressions = sqlglot.parse(stripped, read=dialect)
    except Exception as exc:
        return _reject(f"SQL parse failed: {exc}", validator="sqlglot")

    if len(expressions) != 1:
        return _reject("SQL must contain exactly one statement", validator="sqlglot")

    root = expressions[0]
    if not isinstance(root, exp.Select):
        return _reject("SQL root must be SELECT or WITH ... SELECT", validator="sqlglot")

    forbidden_nodes = {
        "Delete",
        "Insert",
        "Update",
        "Merge",
        "Drop",
        "Alter",
        "Create",
        "TruncateTable",
        "Command",
        "Transaction",
    }
    errors: List[str] = []
    for node in root.walk():
        name = node.__class__.__name__
        if name in forbidden_nodes:
            errors.append(f"Forbidden SQL expression: {name}")
        if isinstance(node, exp.Anonymous) and node.name.lower() in FORBIDDEN_FUNCTIONS:
            errors.append(f"Forbidden database function: {node.name}")

    table_errors = _table_reference_errors_sqlglot(root, schema, allow_schemas)
    errors.extend(table_errors)
    if errors:
        return ValidationResult(ok=True, is_safe=False, errors=sorted(set(errors)), validator="sqlglot")

    normalized = root.sql(dialect=dialect, pretty=True)
    normalized = _apply_outer_limit(normalized, max_rows)
    return ValidationResult(
        ok=True,
        is_safe=True,
        normalized_sql=normalized,
        warnings=[],
        validator="sqlglot",
    )


def _base_text_errors(sql: str) -> List[str]:
    errors: List[str] = []
    if "--" in sql or "/*" in sql or "*/" in sql:
        errors.append("SQL comments are forbidden")

    stripped = sql.strip()
    without_final = stripped[:-1] if stripped.endswith(";") else stripped
    if ";" in without_final:
        errors.append("Multiple SQL statements are forbidden")

    lowered = re.sub(r"\s+", " ", without_final.lower())
    if not (lowered.startswith("select ") or lowered.startswith("with ")):
        errors.append("SQL must start with SELECT or WITH")

    tokens = set(re.findall(r"\b[a-z_][a-z0-9_]*\b", lowered))
    forbidden = sorted(tokens.intersection(FORBIDDEN_KEYWORDS))
    if forbidden:
        errors.append(f"Forbidden SQL keyword(s): {', '.join(forbidden)}")

    for function_name in FORBIDDEN_FUNCTIONS:
        if re.search(rf"\b{re.escape(function_name)}\s*\(", lowered):
            errors.append(f"Forbidden database function: {function_name}")
    return errors


def _fallback_validate(
    sql: str,
    schema: Optional[Dict[str, Any]],
    max_rows: int,
    allow_schemas: Optional[Iterable[str]],
) -> ValidationResult:
    stripped = _strip_optional_trailing_semicolon(sql)
    warnings = [
        "sqlglot is not installed; using a conservative fallback validator. Install worker dependencies for AST validation."
    ]
    table_errors = _table_reference_errors_fallback(stripped, schema, allow_schemas)
    if table_errors:
        return ValidationResult(
            ok=True,
            is_safe=False,
            errors=table_errors,
            warnings=warnings,
            validator="fallback",
        )
    return ValidationResult(
        ok=True,
        is_safe=True,
        normalized_sql=_apply_outer_limit(stripped, max_rows),
        warnings=warnings,
        validator="fallback",
    )


def _table_reference_errors_sqlglot(
    root: Any,
    schema: Optional[Dict[str, Any]],
    allow_schemas: Optional[Iterable[str]],
) -> List[str]:
    try:
        from sqlglot import exp
    except Exception:
        return []

    allowed_tables, allowed_schema_names = _allowed_tables(schema)
    if allow_schemas is not None:
        allowed_schema_names = set(allow_schemas)

    if not allowed_tables and not allowed_schema_names:
        return []

    errors: List[str] = []
    cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
    for table in root.find_all(exp.Table):
        table_name = _clean_identifier(table.name)
        db_name = _clean_identifier(table.db or "")
        if table_name.lower() in cte_names:
            continue
        if db_name and db_name not in allowed_schema_names:
            errors.append(f"Schema `{db_name}` is not allowed")
        candidates = {table_name}
        if db_name:
            candidates.add(f"{db_name}.{table_name}")
        if allowed_tables and not any(candidate in allowed_tables for candidate in candidates):
            errors.append(f"Table `{db_name + '.' if db_name else ''}{table_name}` is not in introspected schema")
    return errors


def _table_reference_errors_fallback(
    sql: str,
    schema: Optional[Dict[str, Any]],
    allow_schemas: Optional[Iterable[str]],
) -> List[str]:
    allowed_tables, allowed_schema_names = _allowed_tables(schema)
    if allow_schemas is not None:
        allowed_schema_names = set(allow_schemas)
    if not allowed_tables and not allowed_schema_names:
        return []

    errors: List[str] = []
    cte_names = {
        _clean_identifier(match.group(1)).lower()
        for match in re.finditer(r"\bwith\s+([\"a-zA-Z_][\"a-zA-Z0-9_]*)\s+as\b|,\s*([\"a-zA-Z_][\"a-zA-Z0-9_]*)\s+as\b", sql, re.I)
        if match.group(1)
    }
    for schema_name, table_name in _extract_table_refs_fallback(sql):
        if table_name.lower() in cte_names:
            continue
        if schema_name and schema_name not in allowed_schema_names:
            errors.append(f"Schema `{schema_name}` is not allowed")
        candidates = {table_name}
        if schema_name:
            candidates.add(f"{schema_name}.{table_name}")
        if allowed_tables and not any(candidate in allowed_tables for candidate in candidates):
            errors.append(f"Table `{schema_name + '.' if schema_name else ''}{table_name}` is not in introspected schema")
    return errors


def _extract_table_refs_fallback(sql: str) -> List[Tuple[str, str]]:
    refs: List[Tuple[str, str]] = []
    pattern = re.compile(r"\b(?:from|join)\s+((?:\"?[a-zA-Z_][a-zA-Z0-9_]*\"?\.)?\"?[a-zA-Z_][a-zA-Z0-9_]*\"?)", re.I)
    for match in pattern.finditer(sql):
        raw = match.group(1)
        if raw.startswith("("):
            continue
        parts = [_clean_identifier(part) for part in raw.split(".")]
        if len(parts) == 2:
            refs.append((parts[0], parts[1]))
        elif len(parts) == 1:
            refs.append(("", parts[0]))
    return refs


def _allowed_tables(schema: Optional[Dict[str, Any]]) -> Tuple[Set[str], Set[str]]:
    allowed_tables: Set[str] = set()
    allowed_schemas: Set[str] = set()
    if not schema:
        return allowed_tables, allowed_schemas
    for schema_entry in schema.get("schemas", []):
        schema_name = _clean_identifier(schema_entry.get("name", ""))
        if schema_name:
            allowed_schemas.add(schema_name)
        for table in schema_entry.get("tables", []):
            table_name = _clean_identifier(table.get("name", ""))
            if table_name:
                allowed_tables.add(table_name)
                if schema_name:
                    allowed_tables.add(f"{schema_name}.{table_name}")
    return allowed_tables, allowed_schemas


def _apply_outer_limit(sql: str, max_rows: int) -> str:
    stripped = _strip_optional_trailing_semicolon(sql)
    if re.search(r"\blimit\s+\d+\s*$", stripped, re.I):
        return stripped
    return f"SELECT * FROM (\n{stripped}\n) AS moku_safe_query LIMIT {int(max_rows)}"


def _strip_optional_trailing_semicolon(sql: str) -> str:
    stripped = sql.strip()
    if stripped.endswith(";"):
        return stripped[:-1].strip()
    return stripped


def _clean_identifier(identifier: str) -> str:
    return identifier.strip().strip('"').strip().lower()


def _reject(error: str, validator: str = "base") -> ValidationResult:
    return ValidationResult(ok=True, is_safe=False, errors=[error], validator=validator)

