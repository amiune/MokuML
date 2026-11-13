"""Complete structural discovery and bounded, aggregate-only profiling."""

from __future__ import annotations

import hashlib
import json
import uuid

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from .cancellation import interruptible
from .contracts import (
    CatalogColumn,
    CatalogTable,
    ColumnProfile,
    DataProfile,
    SourceCatalog,
)
from .data import MAX_COLUMNS, MAX_ROWS, checked_frame
from .errors import SafeError


def opaque(*parts):
    return hashlib.sha256("\0".join(str(p) for p in parts).encode()).hexdigest()[:32]


def data_type(values):
    if is_bool_dtype(values):
        return "boolean"
    if is_numeric_dtype(values):
        return "number"
    if is_datetime64_any_dtype(values):
        return "datetime"
    present = values.dropna().astype(str)
    if len(present) and present.str.match(r"^\d{4}-\d{2}-\d{2}(?:[ T].*)?$").all():
        return "datetime"
    return "string"


class FileSource:
    kind = "files"
    dialect = "duckdb"

    def __init__(self):
        self.id = uuid.uuid4().hex
        self.frames = {}
        self.tables = {}
        self.scoring_ids = set()

    def add(self, frame, name, scoring=False):
        frame = checked_frame(frame)
        if len(self.frames) >= 32:
            raise SafeError("resource_limit")
        table_id = uuid.uuid4().hex
        columns = [
            CatalogColumn(
                id=opaque(table_id, col),
                name=col,
                type=data_type(frame[col]),
                database_type=str(frame[col].dtype),
                nullable=bool(frame[col].isna().any()),
            )
            for col in frame.columns
        ]
        self.frames[table_id] = frame
        self.tables[table_id] = CatalogTable(
            id=table_id,
            name=name[:256],
            kind="file",
            purpose="scoring" if scoring else "history",
            row_count=len(frame),
            row_count_estimated=False,
            columns=columns,
        )
        if scoring:
            self.scoring_ids.add(table_id)
        return table_id

    def catalog_tables(self):
        return list(self.tables.values())

    def table_location(self, table_id):
        return None, "r_" + table_id

    def execute_query(self, sql, cancel_event=None):
        import duckdb

        try:
            with duckdb.connect(
                config={
                    "enable_external_access": "false",
                    "autoinstall_known_extensions": "false",
                    "autoload_known_extensions": "false",
                    "memory_limit": "512MB",
                }
            ) as connection:
                connection.execute("SET TimeZone='UTC'")
                for key, frame in list(self.frames.items()):
                    connection.register("r_" + key, frame)
                # Results are bounded by the compiler at MAX_ROWS+1, never silently truncated.
                with interruptible(cancel_event, connection.interrupt):
                    result = connection.execute(sql).fetch_df()
                if len(result) > MAX_ROWS:
                    raise SafeError("too_many_rows")
                if len(result.columns) > MAX_COLUMNS:
                    raise SafeError("too_many_columns")
                return result
        except SafeError:
            raise
        except Exception:
            raise SafeError("invalid_plan") from None

    def sample(self, table_id, names):
        return self.frames[table_id][names].head(10001).copy()

    def close(self):
        self.frames.clear()
        self.tables.clear()


class Catalog:
    def __init__(self, source):
        self.source = source
        self.id = getattr(source, "id", uuid.uuid4().hex)
        self.refresh()

    def refresh(self):
        self.tables = {t.id: t for t in self.source.catalog_tables()}
        self.columns = {
            c.id: (t.id, c) for t in self.tables.values() for c in t.columns
        }
        serialized = [t.model_dump(mode="json") for t in self.tables.values()]
        self.revision = hashlib.sha256(
            json.dumps(serialized, sort_keys=True).encode()
        ).hexdigest()

    def page(self, cursor=0, limit=20):
        if cursor < 0 or not 1 <= limit <= 50:
            raise SafeError("invalid_selection")
        tables = list(self.tables.values())
        return SourceCatalog(
            id=self.id,
            revision=self.revision,
            kind=self.source.kind,
            table_count=len(tables),
            tables=tables[cursor : cursor + limit],
            next_cursor=cursor + limit if cursor + limit < len(tables) else None,
        )

    def profile(self, table_id, column_ids=None, cursor=0, limit=20):
        if table_id not in self.tables or cursor < 0 or not 1 <= limit <= 50:
            raise SafeError("invalid_selection")
        columns = self.tables[table_id].columns
        if column_ids:
            if not set(column_ids) <= {c.id for c in columns}:
                raise SafeError("invalid_selection")
            columns = [c for c in columns if c.id in column_ids]
        selected = columns[cursor : cursor + limit]
        if not selected:
            raise SafeError("invalid_selection")
        frame = self.source.sample(table_id, [c.name for c in selected])
        estimated = len(frame) > 10000
        frame = frame.head(10000)
        profiles = []
        for column in selected:
            values = frame[column.name]
            present = values.dropna()
            suppressed = len(present) < 10
            fields = dict(
                column_id=column.id,
                sampled_rows=len(frame),
                estimated=estimated,
                null_fraction=float(values.isna().mean()) if len(frame) else 0.0,
                distinct_count=int(present.nunique()),
                details_suppressed=suppressed,
            )
            if not suppressed and column.type == "number":
                import numpy as np

                numeric = pd.to_numeric(present, errors="coerce")
                numeric = numeric[np.isfinite(numeric)]
                if len(numeric) >= 10:
                    fields.update(
                        minimum=float(numeric.min()),
                        maximum=float(numeric.max()),
                        mean=float(numeric.mean()),
                        quantiles=[
                            float(x) for x in numeric.quantile([0.25, 0.5, 0.75])
                        ],
                    )
            elif not suppressed and column.type == "datetime":
                parsed = pd.to_datetime(present, errors="coerce", utc=True).dropna()
                if len(parsed) >= 10:
                    fields.update(
                        minimum=parsed.min().isoformat(),
                        maximum=parsed.max().isoformat(),
                    )
            profiles.append(ColumnProfile(**fields))
        return DataProfile(
            source_id=self.id,
            revision=self.revision,
            table_id=table_id,
            profiles=profiles,
            next_cursor=cursor + limit if cursor + limit < len(columns) else None,
        )
