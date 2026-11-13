from io import BytesIO

import pandas as pd
import pytest

from mokuml_local import data
from mokuml_local.errors import SafeError


@pytest.mark.parametrize("delimiter", [",", ";", "\t"])
def test_csv_formats_and_leading_zeroes(delimiter):
    content = (
        "\ufeff"
        + delimiter.join(["id", "amount", "outcome"])
        + "\n"
        + delimiter.join(["001", "2.5", "yes"])
        + "\n"
        + delimiter.join(["002", "3.5", ""])
    )
    frame, sheets = data.read_spreadsheet(content.encode(), "private.csv")
    assert frame["id"].tolist() == ["001", "002"]
    assert frame["amount"].tolist() == [2.5, 3.5]
    assert pd.isna(frame.iloc[1]["outcome"])
    assert sheets == []


@pytest.mark.parametrize(
    "content", [b"a,a\n1,2", b"a,\n1,2", b"a,b\n1,2,3", b"a,b\n", b"\xff\xfe"]
)
def test_bad_csv_is_safely_rejected(content):
    with pytest.raises(SafeError) as exc:
        data.read_spreadsheet(content, "private.csv")
    assert "private" not in str(exc.value)


def test_excel_worksheet_selection():
    from openpyxl import Workbook

    workbook = Workbook()
    workbook.active.title = "PRIVATE_SHEET"
    workbook.active.append(["id", "outcome"])
    workbook.active.append([1, "yes"])
    workbook.create_sheet("Other")
    stream = BytesIO()
    workbook.save(stream)
    content = stream.getvalue()
    frame, sheets = data.read_spreadsheet(content, "private.xlsx")
    assert frame is None and sheets == ["PRIVATE_SHEET", "Other"]
    frame, _ = data.read_spreadsheet(content, "private.xlsx", "PRIVATE_SHEET")
    assert frame.iloc[0]["outcome"] == "yes"


def test_limits_fail_without_truncation(monkeypatch):
    monkeypatch.setattr(data, "MAX_ROWS", 2)
    with pytest.raises(SafeError, match="100,000"):
        data.read_spreadsheet(b"a\n1\n2\n3", "a.csv")
    monkeypatch.setattr(data, "MAX_BYTES", 2)
    with pytest.raises(SafeError) as exc:
        data.read_spreadsheet(b"a\n1", "a.csv")
    assert exc.value.code == "file_too_large"
    monkeypatch.setattr(data, "MAX_COLUMNS", 1)
    with pytest.raises(SafeError) as exc:
        data.headers(["one", "two"])
    assert exc.value.code == "too_many_columns"


def test_metadata_types_do_not_include_values(business_frame):
    dataset = data.Dataset.create(business_frame, "mysql")
    value = dataset.metadata.model_dump_json()
    assert "PRIVATE_" not in value
    assert dataset.metadata.source_kind == "mysql"
    assert dataset.metadata.columns[1].type == "number"
