"""Local spreadsheet ingestion and safe structural metadata."""

from __future__ import annotations

import csv
import uuid
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from .errors import SafeError
from .public import Column, DatasetMetadata

MAX_BYTES = 50 * 1024 * 1024
MAX_ROWS = 100000
MAX_COLUMNS = 2000


def headers(values) -> list[str]:
    result = [str(value).strip() if value is not None else "" for value in values]
    if (
        not result
        or any(not value for value in result)
        or len(set(x.casefold() for x in result)) != len(result)
    ):
        raise SafeError("file_invalid")
    if len(result) > MAX_COLUMNS:
        raise SafeError("too_many_columns")
    return result


def checked_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) > MAX_ROWS:
        raise SafeError("too_many_rows")
    if frame.empty:
        raise SafeError("empty_data")
    frame = frame.copy()
    frame.columns = headers(frame.columns)
    for name in frame.columns:
        values = frame[name]
        if is_numeric_dtype(values) or is_datetime64_any_dtype(values):
            continue
        values = values.map(
            lambda x: None if x is None or (isinstance(x, str) and not x.strip()) else x
        )
        present = values.dropna()
        numeric = pd.to_numeric(values, errors="coerce")
        leading_zero = any(
            isinstance(x, str) and len(x) > 1 and x[0] == "0" and x[1].isdigit()
            for x in present
        )
        if len(present) and numeric[values.notna()].notna().all() and not leading_zero:
            frame[name] = numeric
        else:
            # TiLE's string discovery expects object columns; handle pandas 3 string dtypes too.
            frame[name] = values.map(lambda x: None if pd.isna(x) else str(x)).astype(
                object
            )
    return frame.reset_index(drop=True)


def read_spreadsheet(
    content: bytes, filename: str, worksheet: str | None = None
) -> tuple[pd.DataFrame | None, list[str]]:
    if len(content) > MAX_BYTES:
        raise SafeError("file_too_large")
    kind = Path(filename).suffix.lower()
    try:
        if kind == ".csv":
            text = content.decode("utf-8-sig")
            try:
                dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel
            reader = csv.reader(StringIO(text), dialect)
            names = headers(next(reader, []))
            rows = []
            for row in reader:
                if not row or all(not x.strip() for x in row):
                    continue
                if len(row) != len(names):
                    raise SafeError("file_invalid")
                rows.append(row)
                if len(rows) > MAX_ROWS:
                    raise SafeError("too_many_rows")
            return checked_frame(pd.DataFrame(rows, columns=names)), []
        if kind != ".xlsx":
            raise SafeError("file_invalid")
        # Reject compressed workbooks that would expand without a reasonable bound.
        with ZipFile(BytesIO(content)) as archive:
            if sum(info.file_size for info in archive.infolist()) > 512 * 1024 * 1024:
                raise SafeError("file_too_large")
        from openpyxl import load_workbook

        workbook = load_workbook(BytesIO(content), read_only=True, data_only=True)
        try:
            sheets = workbook.sheetnames
            if worksheet is None and len(sheets) > 1:
                return None, sheets
            if worksheet is not None and worksheet not in sheets:
                raise SafeError("file_invalid")
            iterator = workbook[worksheet or sheets[0]].iter_rows(values_only=True)
            names = headers(next(iterator, []))
            rows = []
            for row in iterator:
                if all(x is None for x in row):
                    continue
                if len(row) != len(names):
                    raise SafeError("file_invalid")
                rows.append(row)
                if len(rows) > MAX_ROWS:
                    raise SafeError("too_many_rows")
            return checked_frame(pd.DataFrame(rows, columns=names)), []
        finally:
            workbook.close()
    except SafeError:
        raise
    except Exception:
        raise SafeError("file_invalid") from None


@dataclass(repr=False)
class Dataset:
    frame: pd.DataFrame
    metadata: DatasetMetadata
    columns: dict[str, str]

    @classmethod
    def create(cls, frame: pd.DataFrame, source_kind: str, primary_keys=()):
        frame = checked_frame(frame)
        columns = {}
        public_columns = []
        for name in frame.columns:
            identifier = uuid.uuid4().hex
            columns[identifier] = name
            values = frame[name]
            kind = (
                "boolean"
                if is_bool_dtype(values)
                else "datetime"
                if is_datetime64_any_dtype(values)
                else "number"
                if is_numeric_dtype(values)
                else "string"
            )
            public_columns.append(
                Column(
                    id=identifier,
                    name=name,
                    type=kind,
                    nullable=bool(values.isna().any()),
                    primary_key=name in primary_keys,
                )
            )
        metadata = DatasetMetadata(
            id=uuid.uuid4().hex,
            source_kind=source_kind,
            row_count=len(frame),
            columns=public_columns,
        )
        return cls(frame, metadata, columns)
