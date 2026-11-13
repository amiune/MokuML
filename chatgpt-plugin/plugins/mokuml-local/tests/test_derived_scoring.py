import sqlite3

import pandas as pd
import pytest

from mokuml_local.catalog import Catalog, FileSource
from mokuml_local.databases import SQLiteSource
from mokuml_local.planning import Compiler


@pytest.mark.parametrize("name", ["events", "__moku_events", "__moku_anchors"])
def test_future_event_counts_and_primary_key_count_features_in_sqlite(tmp_path, name):
    path = tmp_path / "events.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(f'''CREATE TABLE "{name}" (id INTEGER PRIMARY KEY, at TEXT);
            INSERT INTO "{name}" VALUES (1,'2024-01-05'),(2,'2024-01-15'),(3,'2024-02-05');''')
    source = SQLiteSource(path)
    catalog = Catalog(source)
    table = next(iter(catalog.tables.values()))
    columns = {c.name: c.id for c in table.columns}
    task = dict(
        question="Predict event counts",
        task="regression",
        query={"table_id": table.id},
        target={
            "name": "events",
            "expression": {"kind": "aggregate", "function": "count"},
        },
        features=[
            {
                "name": "history",
                "expression": {
                    "kind": "aggregate",
                    "function": "count",
                    "args": [{"kind": "column", "column_id": columns["id"]}],
                },
            }
        ],
        future=dict(
            event_time_column_id=columns["at"],
            start="2024-01-01T00:00:00",
            observed_through="2024-03-01T00:00:00",
            lookback_periods=[1],
        ),
    )
    _, plan = Compiler(catalog).compile(task)
    frame = source.execute_query(plan.sql).sort_values("__as_of")
    assert frame["events"].iloc[:2].tolist() == [2, 1]
    assert pd.isna(frame["events"].iloc[2])
    assert frame.history_past_1_month.tolist() == [0, 2, 1]


def test_precomputed_scoring_table_for_aggregate_recipe():
    source = FileSource()
    history = source.add(
        pd.DataFrame({"customer": [1, 1, 2], "price": [10, 30, 40], "area": [2, 4, 8]}),
        "history",
    )
    scoring = source.add(pd.DataFrame({"size": [12]}), "new", scoring=True)
    catalog = Catalog(source)
    columns = {c.name: c.id for c in catalog.tables[history].columns}

    def mean(name):
        return dict(
            kind="aggregate",
            function="mean",
            args=[dict(kind="column", column_id=columns[name])],
        )

    task = dict(
        question="Predict mean price",
        task="regression",
        query={"table_id": history, "group_by_column_ids": [columns["customer"]]},
        target={"name": "price", "expression": mean("price")},
        features=[{"name": "size", "expression": mean("area")}],
        identifier_column_ids=[columns["customer"]],
        scoring_table_id=scoring,
    )
    _, plan = Compiler(catalog).compile(task)
    frame = source.execute_query(plan.scoring_sql)
    assert frame["size"].tolist() == [12] and frame["price"].isna().all()


def test_offset_aware_events_obey_utc_calendar_boundaries():
    source = FileSource()
    source.add(
        pd.DataFrame({"date": ["2024-01-31T23:30:00-01:00"], "value": [1]}), "events"
    )
    catalog = Catalog(source)
    table = next(iter(catalog.tables.values()))
    columns = {c.name: c.id for c in table.columns}
    agg = {
        "kind": "aggregate",
        "function": "sum",
        "args": [{"kind": "column", "column_id": columns["value"]}],
    }
    task = dict(
        question="Future UTC total",
        task="regression",
        query={"table_id": table.id},
        target={"name": "total", "expression": agg},
        features=[{"name": "history", "expression": agg}],
        future=dict(
            event_time_column_id=columns["date"],
            start="2024-01-01T00:00:00Z",
            observed_through="2024-03-01T00:00:00Z",
            lookback_periods=[1],
        ),
    )
    _, plan = Compiler(catalog).compile(task)
    frame = source.execute_query(plan.sql).sort_values("__as_of")
    assert frame.total.iloc[:2].tolist() == [0, 1]
    assert frame.history_past_1_month.tolist() == [0, 0, 1]
