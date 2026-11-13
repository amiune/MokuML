"""Integration config is explicit JSON for disposable databases, never production.

MOKUML_TEST_POSTGRES='{"kind":"postgresql","host":"127.0.0.1","port":55439,"database":"mokuml_test","username":"tester","password":"test","require_tls":false}'
MOKUML_TEST_MYSQL='{"kind":"mysql","host":"127.0.0.1","port":53309,"database":"mokuml_test","username":"tester","password":"test","require_tls":false}'
"""

import json
import os
import sqlite3
import uuid
from unittest.mock import patch

import pytest
import sqlalchemy as sa

from mokuml_local.databases import DatabaseConfig, SQLiteSource, SQLSource
from mokuml_local.errors import SafeError


def test_sqlite_reads_original_wal_and_quotes_names(tmp_path):
    path = tmp_path / "private.sqlite"
    writer = sqlite3.connect(path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute(
        'CREATE TABLE "odd"" table" (id INTEGER PRIMARY KEY, note TEXT, amount REAL)'
    )
    writer.execute(
        'INSERT INTO "odd"" table" VALUES (1, ?, 4.5)', ("PRIVATE_CELL_SENTINEL",)
    )
    writer.commit()
    assert path.with_name(path.name + "-wal").exists()
    source = SQLiteSource(path)
    tables = source.list_tables()
    frame, keys = source.load_table(tables[0]["id"])
    assert frame["note"].tolist() == ["PRIVATE_CELL_SENTINEL"] and keys == ["id"]
    with source.connection() as connection:
        for sql in [
            'INSERT INTO "odd"" table" VALUES (2, "bad", 0)',
            "CREATE TABLE mutation (x)",
            "ATTACH DATABASE ':memory:' AS other",
            "SELECT load_extension('private')",
        ]:
            with pytest.raises(sqlite3.DatabaseError):
                connection.execute(sql)
    assert writer.execute('SELECT count(*) FROM "odd"" table"').fetchone()[0] == 1
    source.close()
    writer.close()


def test_sqlite_lock_failure_is_generic(tmp_path):
    path = tmp_path / "PRIVATE_PATH.sqlite"
    writer = sqlite3.connect(path)
    writer.execute("CREATE TABLE demo (x)")
    writer.commit()
    writer.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(SafeError) as exc:
            SQLiteSource(path, timeout=0.01).list_tables()
        assert "PRIVATE_PATH" not in str(exc.value)
    finally:
        writer.rollback()
        writer.close()


@pytest.mark.parametrize("kind", ["postgresql", "mysql"])
def test_adapter_enforces_readonly_transaction_and_timeout(kind):
    config = DatabaseConfig(
        kind=kind,
        host="localhost",
        port=5432 if kind == "postgresql" else 3306,
        database="test",
        username="tester",
        password="PRIVATE_PASSWORD",
        require_tls=False,
    )
    with patch("mokuml_local.databases.sa.create_engine") as create:
        source = SQLSource(config)
        connection = create.return_value.connect.return_value.__enter__.return_value
        with source.connection():
            pass
        commands = [call.args[0] for call in connection.exec_driver_sql.call_args_list]
        assert any("READ ONLY" in sql for sql in commands)
        assert any("timeout" in sql or "max_execution_time" in sql for sql in commands)
        connection.rollback.assert_called_once()
        assert create.call_args.kwargs["hide_parameters"] is True


@pytest.mark.parametrize("kind", ["postgresql", "mysql"])
def test_sql_adapter_does_not_return_connection_errors(kind):
    config = DatabaseConfig(
        kind=kind,
        host="localhost",
        port=1234,
        database="private",
        username="owner",
        password="PRIVATE_PASSWORD",
    )
    with patch("mokuml_local.databases.sa.create_engine") as create:
        create.return_value.connect.side_effect = RuntimeError(
            "PRIVATE_PASSWORD PRIVATE_CELL_SENTINEL"
        )
        source = SQLSource(config)
        with pytest.raises(SafeError) as exc:
            source.list_tables()
        assert "PRIVATE" not in str(exc.value)


def test_sqlite_row_overflow_fails_without_truncation(tmp_path, monkeypatch):
    from mokuml_local import data, databases

    monkeypatch.setattr(databases, "MAX_ROWS", 2)
    monkeypatch.setattr(data, "MAX_ROWS", 2)
    path = tmp_path / "data.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE demo (x)")
        connection.executemany("INSERT INTO demo VALUES (?)", [(1,), (2,), (3,)])
    source = SQLiteSource(path)
    with pytest.raises(SafeError) as exc:
        source.load_table(source.list_tables()[0]["id"])
    assert exc.value.code == "too_many_rows"


@pytest.mark.database
@pytest.mark.parametrize("variable", ["MOKUML_TEST_POSTGRES", "MOKUML_TEST_MYSQL"])
def test_live_disposable_database(variable):
    if variable not in os.environ:
        pytest.skip("A disposable database fixture was not configured")
    config = DatabaseConfig.model_validate(json.loads(os.environ[variable]))
    if "test" not in config.database.lower():
        pytest.fail("Fixture database name must contain 'test'")
    source = SQLSource(config)
    table = sa.Table(
        "mokuml_plugin_fixture_" + uuid.uuid4().hex[:12],
        sa.MetaData(),
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("amount", sa.Float),
        sa.Column("note", sa.String(100)),
    )
    created = False
    try:
        with source.engine.begin() as connection:
            table.create(connection, checkfirst=False)
            created = True
            connection.execute(
                table.insert(),
                [{"id": 1, "amount": 4.5, "note": "PRIVATE_CELL_SENTINEL"}],
            )
        tables = source.list_tables()
        identifier = next(item["id"] for item in tables if item["name"] == table.name)
        frame, keys = source.load_table(identifier)
        assert frame.iloc[0]["note"] == "PRIVATE_CELL_SENTINEL" and keys == ["id"]
        from mokuml_local.catalog import Catalog
        from mokuml_local.planning import Compiler

        catalog = Catalog(source)
        discovered = next(t for t in catalog.tables.values() if t.name == table.name)
        cols = {c.name: c.id for c in discovered.columns}
        definition = dict(
            question="Predict an amount",
            task="regression",
            query={"table_id": discovered.id},
            target={
                "name": "amount",
                "expression": {"kind": "column", "column_id": cols["amount"]},
            },
            features=[
                {
                    "name": "note",
                    "expression": {"kind": "column", "column_id": cols["note"]},
                }
            ],
            identifier_column_ids=[cols["id"]],
        )
        _, plan = Compiler(catalog).compile(definition)
        assert source.execute_query(plan.sql).iloc[0]["amount"] == 4.5
        assert (
            "PRIVATE_CELL_SENTINEL"
            not in catalog.profile(discovered.id).model_dump_json()
        )
        with source.connection() as connection:
            with pytest.raises(sa.exc.DBAPIError):
                connection.execute(table.insert().values(id=2))
    finally:
        if created:
            with source.engine.begin() as connection:
                table.drop(connection, checkfirst=True)
        source.close()
