"""Read-only adapters. No SQL text, connection URL, or database path is exposed through MCP."""

from __future__ import annotations

import sqlite3
import ssl
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
import sqlalchemy as sa
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from sqlalchemy.pool import NullPool
from sqlalchemy.sql import quoted_name

from .cancellation import interruptible
from .catalog import opaque
from .contracts import CatalogColumn, CatalogTable, Constraint, Relationship
from .data import MAX_COLUMNS, MAX_ROWS, checked_frame
from .errors import SafeError


def _kind(sql_type):
    name = str(sql_type).lower()
    if any(x in name for x in ("bool", "bit")):
        return "boolean"
    if any(x in name for x in ("date", "time")):
        return "datetime"
    if any(x in name for x in ("int", "float", "double", "numeric", "decimal", "real")):
        return "number"
    return "string"


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["postgresql", "mysql"]
    host: str = Field(min_length=1, max_length=253)
    port: int = Field(ge=1, le=65535)
    database: str = Field(min_length=1, max_length=255)
    username: str = Field(min_length=1, max_length=255)
    password: SecretStr
    schema_name: str | None = None
    require_tls: bool = True


class SQLSource:
    def __init__(self, config: DatabaseConfig):
        self.id = uuid.uuid4().hex
        self.kind = config.kind
        self.dialect = "postgres" if config.kind == "postgresql" else "mysql"
        self.selected_schema = config.schema_name
        self.schema = config.schema_name or (
            "public" if self.kind == "postgresql" else config.database
        )
        driver = "postgresql+psycopg" if self.kind == "postgresql" else "mysql+pymysql"
        url = sa.URL.create(
            driver,
            username=config.username,
            password=config.password.get_secret_value(),
            host=config.host,
            port=config.port,
            database=config.database,
        )
        if self.kind == "postgresql":
            connect_args = {
                "connect_timeout": 10,
                "sslmode": "require" if config.require_tls else "disable",
            }
        else:
            connect_args = {
                "connect_timeout": 10,
                "read_timeout": 60,
                "write_timeout": 10,
            }
            if config.require_tls:
                connect_args["ssl"] = {
                    "check_hostname": True,
                    "verify_mode": ssl.CERT_REQUIRED,
                }
        self.engine = sa.create_engine(
            url,
            connect_args=connect_args,
            poolclass=NullPool,
            hide_parameters=True,
            echo=False,
        )
        self.tables = {}
        self.locations = {}

    def table_location(self, table_id):
        return self.locations[table_id]

    def catalog_tables(self):
        try:
            with self.connection() as connection:
                inspector = sa.inspect(connection)
                schemas = (
                    [self.selected_schema]
                    if self.selected_schema
                    else inspector.get_schema_names()
                )
                schemas = [
                    s
                    for s in schemas
                    if s
                    not in {
                        "information_schema",
                        "pg_catalog",
                        "mysql",
                        "sys",
                        "performance_schema",
                    }
                    and not s.startswith(("pg_toast", "pg_temp"))
                ]
                result, foreign = [], []
                for schema in sorted(schemas):
                    for name in sorted(
                        set(
                            inspector.get_table_names(schema=schema)
                            + inspector.get_view_names(schema=schema)
                        )
                    ):
                        key = opaque(self.id, schema, name)
                        primary = inspector.get_pk_constraint(name, schema=schema).get(
                            "constrained_columns", []
                        )
                        columns = inspector.get_columns(name, schema=schema)
                        table = CatalogTable(
                            id=key,
                            name=name,
                            schema_name=schema,
                            kind="view"
                            if name in inspector.get_view_names(schema=schema)
                            else "table",
                            columns=[
                                CatalogColumn(
                                    id=opaque(key, c["name"]),
                                    name=c["name"],
                                    type=_kind(c["type"]),
                                    database_type=str(c["type"])[:200],
                                    nullable=bool(c.get("nullable", True)),
                                    primary_key=c["name"] in primary,
                                    description=str(c["comment"])[:1000]
                                    if c.get("comment")
                                    else None,
                                )
                                for c in columns
                            ],
                        )
                        try:
                            table.description = (
                                inspector.get_table_comment(name, schema=schema).get(
                                    "text"
                                )
                                or ""
                            )[:1000] or None
                        except NotImplementedError:
                            pass
                        if primary:
                            table.constraints.append(
                                Constraint(
                                    kind="primary_key",
                                    column_ids=[opaque(key, c) for c in primary],
                                    unique=True,
                                )
                            )
                        for constraint in inspector.get_unique_constraints(
                            name, schema=schema
                        ):
                            table.constraints.append(
                                Constraint(
                                    kind="unique",
                                    column_ids=[
                                        opaque(key, c)
                                        for c in constraint["column_names"]
                                        if c
                                    ],
                                    unique=True,
                                )
                            )
                        for index in inspector.get_indexes(name, schema=schema):
                            table.constraints.append(
                                Constraint(
                                    kind="index",
                                    column_ids=[
                                        opaque(key, c)
                                        for c in index["column_names"]
                                        if c
                                    ],
                                    unique=bool(index.get("unique")),
                                )
                            )
                        # Do not expose check predicates, enum members, or defaults containing data literals.
                        for _ in inspector.get_check_constraints(name, schema=schema):
                            table.constraints.append(
                                Constraint(kind="check", column_ids=[])
                            )
                        foreign.extend(
                            (table, fk)
                            for fk in inspector.get_foreign_keys(name, schema=schema)
                        )
                        self.locations[key] = schema, name
                        result.append(table)
                known = {t.id for t in result}
                for table, fk in foreign:
                    other = opaque(
                        self.id,
                        fk.get("referred_schema") or table.schema_name,
                        fk["referred_table"],
                    )
                    if other in known:
                        table.relationships.append(
                            Relationship(
                                column_ids=[
                                    opaque(table.id, c)
                                    for c in fk["constrained_columns"]
                                ],
                                referenced_table_id=other,
                                referenced_column_ids=[
                                    opaque(other, c) for c in fk["referred_columns"]
                                ],
                            )
                        )
                return result
        except Exception:
            raise SafeError("database_failed") from None

    def execute_query(self, sql, cancel_event=None):
        try:
            with self.connection() as connection:
                raw = connection.connection.driver_connection
                interrupt = raw.cancel if self.kind == "postgresql" else raw.close
                with interruptible(cancel_event, interrupt):
                    result = connection.execution_options(
                        stream_results=True
                    ).exec_driver_sql(sql)
                    rows = result.fetchmany(MAX_ROWS + 1)
                if len(rows) > MAX_ROWS:
                    raise SafeError("too_many_rows")
                return pd.DataFrame.from_records(rows, columns=list(result.keys()))
        except SafeError:
            raise
        except Exception:
            raise SafeError("invalid_plan") from None

    def sample(self, table_id, names):
        schema, name = self.locations[table_id]
        table = sa.table(
            quoted_name(name, True),
            *[sa.column(quoted_name(c, True)) for c in names],
            schema=quoted_name(schema, True),
        )
        with self.connection() as connection:
            result = connection.execute(sa.select(table).limit(10001))
            return pd.DataFrame.from_records(result.fetchmany(10001), columns=names)

    @contextmanager
    def connection(self):
        with self.engine.connect() as connection:
            try:
                if self.kind == "postgresql":
                    connection.exec_driver_sql("BEGIN READ ONLY")
                    connection.exec_driver_sql("SET LOCAL TIME ZONE 'UTC'")
                    connection.exec_driver_sql("SET LOCAL statement_timeout = '60s'")
                else:
                    connection.exec_driver_sql("START TRANSACTION READ ONLY")
                    connection.exec_driver_sql("SET SESSION time_zone = '+00:00'")
                    connection.exec_driver_sql("SET SESSION max_execution_time = 60000")
                yield connection
            finally:
                connection.rollback()

    def list_tables(self) -> list[dict]:
        try:
            with self.connection() as connection:
                inspector = sa.inspect(connection)
                names = sorted(
                    set(
                        inspector.get_table_names(schema=self.schema)
                        + inspector.get_view_names(schema=self.schema)
                    )
                )
            self.tables = {uuid.uuid4().hex: name for name in names}
            return [{"id": key, "name": value} for key, value in self.tables.items()]
        except Exception:
            raise SafeError("database_failed") from None

    def load_table(self, table_id: str) -> tuple[pd.DataFrame, list[str]]:
        if table_id not in self.tables:
            raise SafeError("invalid_selection")
        try:
            name = self.tables[table_id]
            with self.connection() as connection:
                inspector = sa.inspect(connection)
                columns = inspector.get_columns(name, schema=self.schema)
                if len(columns) > MAX_COLUMNS:
                    raise SafeError("too_many_columns")
                table = sa.table(
                    quoted_name(name, True),
                    *[sa.column(quoted_name(c["name"], True)) for c in columns],
                    schema=quoted_name(self.schema, True),
                )
                result = connection.execution_options(stream_results=True).execute(
                    sa.select(table).limit(MAX_ROWS + 1)
                )
                rows = result.fetchmany(MAX_ROWS + 1)
                keys = list(result.keys())
                primary_keys = inspector.get_pk_constraint(
                    name, schema=self.schema
                ).get("constrained_columns", [])
            return checked_frame(
                pd.DataFrame.from_records(rows, columns=keys)
            ), primary_keys
        except SafeError:
            raise
        except Exception:
            raise SafeError("database_failed") from None

    def close(self):
        self.tables.clear()
        self.engine.dispose()


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class SQLiteSource:
    kind = "sqlite"
    dialect = "sqlite"

    def __init__(self, path: Path, timeout: float = 60):
        self.id = uuid.uuid4().hex
        self.path = path.resolve(strict=True)
        self.timeout = timeout
        self.tables = {}
        self.locations = {}

    def table_location(self, table_id):
        return None, self.locations[table_id]

    def catalog_tables(self):
        try:
            with self.connection() as connection:
                names = list(
                    connection.execute(
                        "SELECT name,type FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
                    )
                )
                result = []
                for name, kind in names:
                    key = opaque(self.id, name)
                    self.locations[key] = name
                    rows = list(connection.execute(f"PRAGMA table_info({quote(name)})"))
                    table = CatalogTable(
                        id=key,
                        name=name,
                        kind=kind,
                        columns=[
                            CatalogColumn(
                                id=opaque(key, r[1]),
                                name=r[1],
                                type=_kind(r[2]),
                                database_type=r[2],
                                nullable=not bool(r[3] or r[5]),
                                primary_key=bool(r[5]),
                            )
                            for r in rows
                        ],
                    )
                    pk = [opaque(key, r[1]) for r in rows if r[5]]
                    if pk:
                        table.constraints.append(
                            Constraint(kind="primary_key", column_ids=pk, unique=True)
                        )
                    for index in connection.execute(
                        f"PRAGMA index_list({quote(name)})"
                    ).fetchall():
                        cols = [
                            opaque(key, r[2])
                            for r in connection.execute(
                                f"PRAGMA index_info({quote(index[1])})"
                            )
                            if r[2]
                        ]
                        table.constraints.append(
                            Constraint(
                                kind="index", column_ids=cols, unique=bool(index[2])
                            )
                        )
                    fk_groups = {}
                    for row in connection.execute(
                        f"PRAGMA foreign_key_list({quote(name)})"
                    ):
                        fk_groups.setdefault(row[0], []).append(row)
                    for group in fk_groups.values():
                        group.sort(key=lambda r: r[1])
                        other = opaque(self.id, group[0][2])
                        target_rows = list(
                            connection.execute(
                                f"PRAGMA table_info({quote(group[0][2])})"
                            )
                        )
                        target_pk = [
                            r[1]
                            for r in sorted(target_rows, key=lambda r: r[5] or 9999)
                            if r[5]
                        ]
                        referenced = [
                            r[4] or (target_pk[i] if i < len(target_pk) else "")
                            for i, r in enumerate(group)
                        ]
                        table.relationships.append(
                            Relationship(
                                column_ids=[opaque(key, r[3]) for r in group],
                                referenced_table_id=other,
                                referenced_column_ids=[
                                    opaque(other, c) for c in referenced
                                ],
                            )
                        )
                    result.append(table)
                return result
        except Exception:
            raise SafeError("database_failed") from None

    def execute_query(self, sql, cancel_event=None):
        try:
            with self.connection() as connection:
                with interruptible(cancel_event, connection.interrupt):
                    result = connection.execute(sql)
                    rows = result.fetchmany(MAX_ROWS + 1)
                if len(rows) > MAX_ROWS:
                    raise SafeError("too_many_rows")
                return pd.DataFrame.from_records(
                    rows, columns=[c[0] for c in result.description]
                )
        except SafeError:
            raise
        except Exception:
            raise SafeError("invalid_plan") from None

    def sample(self, table_id, names):
        with self.connection() as connection:
            result = connection.execute(
                f"SELECT {', '.join(quote(n) for n in names)} FROM {quote(self.locations[table_id])} LIMIT 10001"
            )
            return pd.DataFrame.from_records(result.fetchall(), columns=names)

    @contextmanager
    def connection(self):
        # Do not use immutable=1 or copy a live database without its WAL.
        connection = sqlite3.connect(
            self.path.as_uri() + "?mode=ro", uri=True, timeout=min(self.timeout, 5)
        )
        connection.enable_load_extension(False)
        connection.execute("PRAGMA query_only=ON")
        started = time.monotonic()
        connection.set_progress_handler(
            lambda: int(time.monotonic() - started > self.timeout), 1000
        )
        denied = {
            sqlite3.SQLITE_ATTACH,
            sqlite3.SQLITE_DETACH,
            sqlite3.SQLITE_INSERT,
            sqlite3.SQLITE_UPDATE,
            sqlite3.SQLITE_DELETE,
            sqlite3.SQLITE_CREATE_TABLE,
            sqlite3.SQLITE_DROP_TABLE,
        }

        def authorize(action, arg1, arg2, database, trigger):
            if action in denied or (
                action == sqlite3.SQLITE_FUNCTION
                and str(arg2).lower() in {"load_extension", "readfile", "writefile"}
            ):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorize)
        try:
            connection.execute("BEGIN")
            yield connection
        finally:
            connection.rollback()
            connection.close()

    def list_tables(self) -> list[dict]:
        try:
            with self.connection() as connection:
                names = [
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
                    )
                ]
            self.tables = {uuid.uuid4().hex: name for name in names}
            return [{"id": key, "name": value} for key, value in self.tables.items()]
        except Exception:
            raise SafeError("database_failed") from None

    def load_table(self, table_id: str) -> tuple[pd.DataFrame, list[str]]:
        if table_id not in self.tables:
            raise SafeError("invalid_selection")
        try:
            name = quote(self.tables[table_id])
            with self.connection() as connection:
                columns = list(connection.execute(f"PRAGMA table_info({name})"))
                if len(columns) > MAX_COLUMNS:
                    raise SafeError("too_many_columns")
                result = connection.execute(
                    f"SELECT * FROM {name} LIMIT {MAX_ROWS + 1}"
                )
                rows = result.fetchmany(MAX_ROWS + 1)
                keys = [item[0] for item in result.description]
            return checked_frame(pd.DataFrame.from_records(rows, columns=keys)), [
                row[1] for row in columns if row[5]
            ]
        except SafeError:
            raise
        except Exception:
            raise SafeError("database_failed") from None

    def close(self):
        self.tables.clear()


def choose_sqlite_file() -> Path:
    try:
        result = subprocess.run(
            [
                "osascript",
                "-e",
                'POSIX path of (choose file with prompt "Choose a SQLite database for local predictions")',
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=True,
        )
        path = Path(result.stdout.strip())
        if not path.is_file():
            raise ValueError
        return path
    except Exception:
        raise SafeError("picker_failed") from None
