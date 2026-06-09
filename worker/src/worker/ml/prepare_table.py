from __future__ import annotations

from collections import Counter
from datetime import date, datetime
import json
from typing import Any, Dict, List, Optional, Sequence


def infer_task_type(values: Sequence[Any], requested: str = "unknown") -> str:
    if requested in {"classification", "regression"}:
        return requested
    non_null = [value for value in values if value is not None]
    if not non_null:
        return "unknown"
    if all(isinstance(value, bool) for value in non_null):
        return "classification"
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in non_null):
        unique_count = len(set(non_null))
        if unique_count <= min(20, max(2, int(len(non_null) * 0.1))):
            return "classification"
        return "regression"
    return "classification"


def prepare_records(
    records: List[Dict[str, Any]],
    target_column: str,
    entity_id_columns: Optional[List[str]] = None,
    task_type: str = "unknown",
) -> Dict[str, Any]:
    entity_id_columns = entity_id_columns or []
    warnings: List[str] = []
    if not records:
        return {
            "ok": False,
            "message": "Query returned no rows.",
            "warnings": ["No data was available for model preparation."],
        }
    if target_column not in records[0]:
        return {
            "ok": False,
            "message": f"Target column `{target_column}` was not present in the result table.",
            "warnings": [],
        }

    all_columns = list(records[0].keys())
    ignored = set(entity_id_columns + [target_column])
    feature_columns = [column for column in all_columns if column not in ignored]
    if not feature_columns:
        warnings.append("No feature columns were available after excluding target and entity ID columns.")

    prepared_records = [_expand_dates(record) for record in records]
    expanded_feature_columns = [
        column
        for column in prepared_records[0].keys()
        if column not in ignored and not column.startswith("__")
    ]

    all_null_columns = []
    for column in expanded_feature_columns:
        if all(record.get(column) is None for record in prepared_records):
            all_null_columns.append(column)
    if all_null_columns:
        warnings.append(f"Dropped all-null feature columns: {', '.join(all_null_columns)}")
    final_feature_columns = [column for column in expanded_feature_columns if column not in all_null_columns]

    high_cardinality = []
    for column in final_feature_columns:
        values = [record.get(column) for record in prepared_records if record.get(column) is not None]
        unique_values = {_hashable_value(value) for value in values}
        if values and len(unique_values) > max(50, int(len(values) * 0.8)):
            high_cardinality.append(column)
    if high_cardinality:
        warnings.append(f"High-cardinality feature columns may behave like IDs: {', '.join(high_cardinality)}")

    labeled = [record for record in prepared_records if record.get(target_column) is not None]
    prediction_rows = [record for record in prepared_records if record.get(target_column) is None]
    inferred_task = infer_task_type([record.get(target_column) for record in records], task_type)

    if len(labeled) < 5:
        warnings.append("Fewer than 5 labeled rows are available; predictions may be unreliable.")
    if inferred_task == "classification":
        class_counts = Counter(record.get(target_column) for record in labeled)
        if len(class_counts) <= 1 and labeled:
            warnings.append("Target has one observed class; classification cannot generalize.")

    return {
        "ok": True,
        "records": prepared_records,
        "feature_columns": final_feature_columns,
        "labeled_rows": labeled,
        "prediction_rows": prediction_rows,
        "task_type": inferred_task,
        "dropped_columns": all_null_columns,
        "warnings": warnings,
    }


def _expand_dates(record: Dict[str, Any]) -> Dict[str, Any]:
    expanded = dict(record)
    for key, value in list(record.items()):
        parsed = _as_datetime(value)
        if parsed is None:
            continue
        expanded.pop(key, None)
        expanded[f"{key}_year"] = parsed.year
        expanded[f"{key}_month"] = parsed.month
        expanded[f"{key}_day"] = parsed.day
        expanded[f"{key}_day_of_week"] = parsed.weekday()
        expanded[f"{key}_days_since_epoch"] = (parsed.date() - date(1970, 1, 1)).days
    return expanded


def _as_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _hashable_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)
