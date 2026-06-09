from __future__ import annotations

import math
from typing import Iterable, List


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict:
    actual = list(y_true)
    predicted = list(y_pred)
    if not actual:
        return {}
    absolute_errors = [abs(a - p) for a, p in zip(actual, predicted)]
    squared_errors = [(a - p) ** 2 for a, p in zip(actual, predicted)]
    mean_actual = sum(actual) / len(actual)
    total_variance = sum((a - mean_actual) ** 2 for a in actual)
    residual_variance = sum(squared_errors)
    return {
        "mae": sum(absolute_errors) / len(absolute_errors),
        "rmse": math.sqrt(sum(squared_errors) / len(squared_errors)),
        "r2": 1 - residual_variance / total_variance if total_variance else 0.0,
    }


def accuracy_score(y_true: Iterable[object], y_pred: Iterable[object]) -> dict:
    actual = list(y_true)
    predicted = list(y_pred)
    if not actual:
        return {}
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    return {"accuracy": correct / len(actual)}

