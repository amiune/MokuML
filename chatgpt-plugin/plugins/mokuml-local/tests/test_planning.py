import json
import sqlite3

import pandas as pd
import pytest
from conftest import task_for

from mokuml_local.catalog import Catalog, FileSource
from mokuml_local.databases import SQLiteSource
from mokuml_local.errors import SafeError
from mokuml_local.planning import Compiler, validate_sql


def test_catalog_profiles_pagination_suppression_and_filename_privacy(service):
    service.import_file(b"x,outcome\n1,2\n2,3\n", "/PRIVATE_PATH/case.csv")
    source = service.inspect_source()["catalog"]
    assert "PRIVATE_PATH" not in json.dumps(source)
    table = source["tables"][0]
    profile = service.profile_data(table["id"])["profile"]
    assert all(p["details_suppressed"] for p in profile["profiles"])
    assert all("minimum" not in p for p in profile["profiles"])
    for i in range(22):
        service.source.add(pd.DataFrame({"x": range(12)}), f"relation_{i}")
    first = service.inspect_source(limit=10)["catalog"]
    assert first["table_count"] == 23 and first["next_cursor"] == 10
    second = service.inspect_source(cursor=10, limit=10)["catalog"]
    assert not {t["id"] for t in first["tables"]} & {t["id"] for t in second["tables"]}


def test_profiles_are_estimates_and_never_return_categorical_values():
    source = FileSource()
    key = source.add(
        pd.DataFrame({"x": range(10002), "note": ["PRIVATE_CELL"] * 10002}), "records"
    )
    catalog = Catalog(source)
    result = catalog.profile(key).model_dump_json()
    assert "PRIVATE_CELL" not in result
    profile = catalog.profile(key).profiles[0]
    assert profile.estimated and profile.sampled_rows == 10000
    assert profile.quantiles == [2499.75, 4999.5, 7499.25]


def test_sqlite_catalog_compound_keys_and_readonly(tmp_path):
    path = tmp_path / "fixture.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(
            "CREATE TABLE parent (a INTEGER,b INTEGER,PRIMARY KEY(a,b)); CREATE TABLE child (x INTEGER,y INTEGER,FOREIGN KEY(x,y) REFERENCES parent(a,b));"
        )
    source = SQLiteSource(path)
    tables = {t.name: t for t in source.catalog_tables()}
    relation = tables["child"].relationships[0]
    assert relation.referenced_table_id == tables["parent"].id
    assert len(relation.column_ids) == len(relation.referenced_column_ids) == 2
    with pytest.raises(SafeError):
        source.execute_query("DELETE FROM parent")


@pytest.mark.parametrize("dialect", ["duckdb", "postgres", "mysql", "sqlite"])
def test_portable_simple_query_and_target_exclusion(service, dialect):
    service.publish(pd.DataFrame({"x": [1, 2], "outcome": [2, 4]}), "csv")
    service.source.dialect = dialect
    task = task_for(service)
    _, plan = Compiler(service.catalog).compile(task)
    assert "LIMIT 100001" in plan.sql
    task["features"][0]["expression"] = task["target"]["expression"]
    with pytest.raises(SafeError):
        Compiler(service.catalog).compile(task)


def test_joins_aggregate_and_derived_outcome():
    source = FileSource()
    source.add(pd.DataFrame({"order_id": [1, 2], "customer": [10, 20]}), "orders")
    source.add(
        pd.DataFrame({"order": [1, 1, 2], "quantity": [2, 1, 3], "price": [5, 20, 10]}),
        "items",
    )
    catalog = Catalog(source)
    a, b = catalog.tables.values()
    ca = {c.name: c.id for c in a.columns}
    cb = {c.name: c.id for c in b.columns}

    def col(c):
        return {"kind": "column", "column_id": c}

    def agg(function, arg):
        return {"kind": "aggregate", "function": function, "args": [arg]}

    revenue = {
        "kind": "operation",
        "operator": "multiply",
        "args": [col(cb["quantity"]), col(cb["price"])],
    }
    task = dict(
        question="Predict customer spend",
        task="regression",
        query=dict(
            table_id=a.id,
            joins=[dict(table_id=b.id, column_pairs=[(ca["order_id"], cb["order"])])],
            group_by_column_ids=[ca["customer"]],
        ),
        target=dict(name="spend", expression=agg("sum", revenue)),
        features=[dict(name="lines", expression=agg("count", col(cb["order"])))],
        identifier_column_ids=[ca["customer"]],
    )
    _, plan = Compiler(catalog).compile(task)
    frame = source.execute_query(plan.sql).sort_values("__id_0")
    assert frame.spend.tolist() == [30, 30]
    assert frame.lines.tolist() == [2, 1]


@pytest.mark.parametrize(
    "unit,start,end",
    [
        ("month", "2024-01-31T00:00:00", "2024-03-31T00:00:00"),
        ("week", "2024-01-01T00:00:00", "2024-04-01T00:00:00"),
        ("day", "2024-01-01T00:00:00", "2024-02-01T00:00:00"),
    ],
)
def test_future_labels_are_complete_and_features_use_only_past(unit, start, end):
    source = FileSource()
    source.add(
        pd.DataFrame(
            {"date": pd.date_range("2023-01-01", "2024-05-01", freq="D"), "amount": 1}
        ),
        "events",
    )
    catalog = Catalog(source)
    table = next(iter(catalog.tables.values()))
    ids = {c.name: c.id for c in table.columns}
    agg = {
        "kind": "aggregate",
        "function": "sum",
        "args": [{"kind": "column", "column_id": ids["amount"]}],
    }
    task = dict(
        question="Predict a future total",
        task="regression",
        query={"table_id": table.id},
        target={"name": "total", "expression": agg},
        features=[{"name": "history", "expression": agg}],
        future=dict(
            event_time_column_id=ids["date"],
            start=start,
            observed_through=end,
            unit=unit,
            lookback_periods=[1],
        ),
    )
    _, plan = Compiler(catalog).compile(task)
    frame = source.execute_query(plan.sql)
    score = frame.loc[pd.to_datetime(frame.__as_of) == pd.Timestamp(end)].iloc[0]
    assert pd.isna(score.total)
    assert score[f"history_past_1_{unit}"] > 0
    mature = frame[pd.to_datetime(frame.__label_end) <= pd.Timestamp(end)]
    assert mature.total.notna().all()
    # A large sentinel after the observation end cannot affect any historical feature.
    source.frames[table.id].loc[
        source.frames[table.id]["date"] >= pd.Timestamp(end), "amount"
    ] = 99999
    changed = source.execute_query(plan.sql).sort_values("__as_of")
    pd.testing.assert_series_equal(
        frame.sort_values("__as_of")[f"history_past_1_{unit}"].reset_index(drop=True),
        changed[f"history_past_1_{unit}"].reset_index(drop=True),
    )


def test_calendar_next_month_and_zero_complete_windows():
    source = FileSource()
    source.add(pd.DataFrame({"date": ["2024-01-01"], "value": [10]}), "events")
    catalog = Catalog(source)
    table = next(iter(catalog.tables.values()))
    ids = {c.name: c.id for c in table.columns}
    agg = {
        "kind": "aggregate",
        "function": "sum",
        "args": [{"kind": "column", "column_id": ids["value"]}],
    }
    task = dict(
        question="Next calendar month",
        task="regression",
        query={"table_id": table.id},
        target={"name": "total", "expression": agg},
        features=[{"name": "history", "expression": agg}],
        future=dict(
            event_time_column_id=ids["date"],
            start="2024-01-13T00:00:00",
            observed_through="2024-04-13T00:00:00",
            alignment="calendar_next",
            lookback_periods=[1],
        ),
    )
    _, plan = Compiler(catalog).compile(task)
    frame = source.execute_query(plan.sql)
    score = frame.loc[pd.to_datetime(frame.__as_of) == pd.Timestamp("2024-04-13")].iloc[
        0
    ]
    assert pd.Timestamp(score.__label_start) == pd.Timestamp("2024-05-01")
    assert pd.Timestamp(score.__label_end) == pd.Timestamp("2024-06-01")
    assert (
        frame.loc[
            pd.to_datetime(frame.__label_end) <= pd.Timestamp("2024-04-13"), "total"
        ]
        .eq(0)
        .all()
    )


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM allowed",
        "SELECT 1; SELECT 2",
        "SELECT read_csv('/tmp/private')",
        "SELECT pg_read_file('/tmp/private')",
        "SELECT * INTO new_table FROM allowed",
        "SELECT * FROM hidden",
        "SELECT * FROM allowed FOR UPDATE",
        "WITH x AS (DELETE FROM allowed RETURNING *) SELECT * FROM x",
    ],
)
def test_ast_rejects_unsafe_queries(sql):
    with pytest.raises(SafeError):
        validate_sql(sql, "postgres", {(None, "allowed")})


def test_private_new_scoring_file_mapping(service):
    service.publish(pd.DataFrame({"x": [1, 2], "outcome": [2, 4]}), "csv")
    source = service.source
    scoring_id = source.add(pd.DataFrame({"new_x": [8, 9]}), "new_cases", scoring=True)
    service.catalog.refresh()
    task = task_for(service)
    original = next(
        c.id
        for t in service.catalog.tables.values()
        for c in t.columns
        if c.name == "x"
    )
    dest = service.catalog.tables[scoring_id].columns[0].id
    task.update(scoring_table_id=scoring_id, scoring_column_map={original: dest})
    _, plan = Compiler(service.catalog).compile(task)
    scored = source.execute_query(plan.scoring_sql)
    assert scored.x.tolist() == [8, 9] and scored.outcome.isna().all()


def test_prepared_definition_is_immutable_and_old_source_rejected(service):
    service.publish(pd.DataFrame({"x": range(20), "outcome": range(20)}), "csv")
    task = task_for(service)
    prepared = service.prepare_prediction(task)["plan"]
    task["target"]["name"] = "tampered"
    assert service.plans[prepared["id"]][0]["target"]["name"] == "outcome"
    service.source.add(pd.DataFrame({"other": [1]}), "new")
    assert (
        service.invoke("start_prediction", plan_id=prepared["id"])["code"]
        == "stale_plan"
    )
