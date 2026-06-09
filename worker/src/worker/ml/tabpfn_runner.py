from __future__ import annotations

from collections import Counter
import json
from typing import Any, Dict, List, Optional

from .metrics import accuracy_score, regression_metrics
from .prepare_table import prepare_records


def run_tabular_predictions(
    records: List[Dict[str, Any]],
    target_column: str,
    entity_id_columns: Optional[List[str]] = None,
    task_type: str = "unknown",
    mode: str = "auto",
    test_size: float = 0.2,
    random_seed: int = 42,
) -> Dict[str, Any]:
    prepared = prepare_records(records, target_column, entity_id_columns, task_type)
    if not prepared.get("ok"):
        return prepared

    try:
        return _run_tabpfn_or_sklearn(
            prepared,
            target_column,
            entity_id_columns or [],
            mode=mode,
            test_size=test_size,
            random_seed=random_seed,
        )
    except Exception as exc:
        fallback = _run_baseline(
            prepared,
            target_column,
            entity_id_columns or [],
        )
        fallback["warnings"].append(
            f"Local sklearn model could not run; returned a dependency-free baseline instead: {_short_error(exc)}"
        )
        return fallback


def _run_tabpfn_or_sklearn(
    prepared: Dict[str, Any],
    target_column: str,
    entity_id_columns: List[str],
    mode: str,
    test_size: float,
    random_seed: int,
) -> Dict[str, Any]:
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score as skl_accuracy_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    task_type = prepared["task_type"]
    labeled = prepared["labeled_rows"]
    prediction_rows = prepared["prediction_rows"]
    feature_columns = list(prepared["feature_columns"])
    if not feature_columns:
        feature_columns = ["__moku_constant_feature"]
        for row in labeled:
            row["__moku_constant_feature"] = 1
        for row in prediction_rows:
            row["__moku_constant_feature"] = 1
        prepared["feature_columns"] = feature_columns
    if len(labeled) < 5:
        raise ValueError("At least 5 labeled rows are required.")

    train_df = pd.DataFrame(labeled)
    predict_df = pd.DataFrame(prediction_rows) if prediction_rows else pd.DataFrame()
    X, predict_X = _coerce_feature_frames(train_df, predict_df, feature_columns)
    y = train_df[target_column]

    from pandas.api.types import is_numeric_dtype

    numeric_columns = [column for column in feature_columns if is_numeric_dtype(X[column])]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_columns),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_columns),
        ],
        remainder="drop",
    )

    if mode != "tabpfn":
        if task_type == "classification":
            from sklearn.ensemble import RandomForestClassifier

            estimator = RandomForestClassifier(n_estimators=100, random_state=random_seed)
            model_name = "RandomForestClassifier"
        else:
            from sklearn.ensemble import RandomForestRegressor

            estimator = RandomForestRegressor(n_estimators=100, random_state=random_seed)
            model_name = "RandomForestRegressor"
        return _fit_and_format_predictions(
            preprocessor,
            estimator,
            model_name,
            "local sklearn",
            prepared,
            train_df,
            predict_df,
            X,
            predict_X,
            y,
            target_column,
            entity_id_columns,
            test_size,
            random_seed,
        )

    if task_type == "classification":
        tabpfn_error = None
        try:
            from tabpfn import TabPFNClassifier

            estimator = TabPFNClassifier()
            model_name = "TabPFNClassifier"
            checkpoint = "default TabPFN-3"
            try:
                return _fit_and_format_predictions(
                    preprocessor,
                    estimator,
                    model_name,
                    checkpoint,
                    prepared,
                    train_df,
                    predict_df,
                    X,
                    predict_X,
                    y,
                    target_column,
                    entity_id_columns,
                    test_size,
                    random_seed,
                )
            except Exception as exc:
                tabpfn_error = exc
        except Exception as exc:
            tabpfn_error = exc

        try:
            from sklearn.ensemble import RandomForestClassifier

            estimator = RandomForestClassifier(n_estimators=100, random_state=random_seed)
            model_name = "RandomForestClassifier fallback"
            checkpoint = "local sklearn"
            result = _fit_and_format_predictions(
                preprocessor,
                estimator,
                model_name,
                checkpoint,
                prepared,
                train_df,
                predict_df,
                X,
                predict_X,
                y,
                target_column,
                entity_id_columns,
                test_size,
                random_seed,
            )
            if tabpfn_error is not None:
                result["warnings"] = list(result.get("warnings", [])) + [
                    f"TabPFN was unavailable; used local sklearn instead: {_short_error(tabpfn_error)}"
                ]
                result["dataset"]["warnings"] = result["warnings"]
            return result
        except Exception:
            if tabpfn_error is not None:
                raise tabpfn_error
            raise
    else:
        tabpfn_error = None
        try:
            from tabpfn import TabPFNRegressor

            estimator = TabPFNRegressor()
            model_name = "TabPFNRegressor"
            checkpoint = "default TabPFN-3"
            try:
                return _fit_and_format_predictions(
                    preprocessor,
                    estimator,
                    model_name,
                    checkpoint,
                    prepared,
                    train_df,
                    predict_df,
                    X,
                    predict_X,
                    y,
                    target_column,
                    entity_id_columns,
                    test_size,
                    random_seed,
                )
            except Exception as exc:
                tabpfn_error = exc
        except Exception as exc:
            tabpfn_error = exc

        try:
            from sklearn.ensemble import RandomForestRegressor

            estimator = RandomForestRegressor(n_estimators=100, random_state=random_seed)
            model_name = "RandomForestRegressor fallback"
            checkpoint = "local sklearn"
            result = _fit_and_format_predictions(
                preprocessor,
                estimator,
                model_name,
                checkpoint,
                prepared,
                train_df,
                predict_df,
                X,
                predict_X,
                y,
                target_column,
                entity_id_columns,
                test_size,
                random_seed,
            )
            if tabpfn_error is not None:
                result["warnings"] = list(result.get("warnings", [])) + [
                    f"TabPFN was unavailable; used local sklearn instead: {_short_error(tabpfn_error)}"
                ]
                result["dataset"]["warnings"] = result["warnings"]
            return result
        except Exception:
            if tabpfn_error is not None:
                raise tabpfn_error
            raise

    raise RuntimeError("No tabular model backend was available.")


def _fit_and_format_predictions(
    preprocessor: Any,
    estimator: Any,
    model_name: str,
    checkpoint: str,
    prepared: Dict[str, Any],
    train_df: Any,
    predict_df: Any,
    X: Any,
    predict_X: Any,
    y: Any,
    target_column: str,
    entity_id_columns: List[str],
    test_size: float,
    random_seed: int,
) -> Dict[str, Any]:
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    task_type = prepared["task_type"]
    labeled = prepared["labeled_rows"]
    prediction_rows = prepared["prediction_rows"]
    feature_columns = prepared["feature_columns"]
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])

    metrics: Dict[str, Any] = {}
    if len(labeled) >= 10:
        metrics = _validation_metrics(
            pipeline,
            X,
            y,
            task_type,
            test_size,
            random_seed,
            prepared["warnings"],
        )

    if not prediction_rows and len(labeled) >= 10:
        _X_train, X_test, _y_train, _y_test = _validation_split(
            X,
            y,
            task_type,
            test_size,
            random_seed,
        )
        pipeline.fit(X, y)
        rows_for_prediction = X_test
        source_rows = train_df.loc[X_test.index].to_dict("records")
    else:
        pipeline.fit(X, y)
        rows_for_prediction = predict_X if prediction_rows else X
        source_rows = prediction_rows if prediction_rows else labeled

    predictions = _format_model_predictions(
        pipeline,
        rows_for_prediction,
        source_rows,
        entity_id_columns,
        task_type,
    )
    return {
        "ok": True,
        "task_type": task_type,
        "target_column": target_column,
        "entity_id_columns": entity_id_columns,
        "model": {"name": model_name, "checkpoint": checkpoint},
        "dataset": {
            "training_rows": len(labeled),
            "prediction_rows": len(prediction_rows),
            "feature_count": len(feature_columns),
            "dropped_columns": prepared["dropped_columns"],
            "warnings": prepared["warnings"],
        },
        "metrics": metrics,
        "predictions": predictions,
        "warnings": prepared["warnings"],
    }


def _short_error(exc: Exception, max_length: int = 240) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    text = " ".join(text.split())
    if len(text) > max_length:
        return text[: max_length - 1] + "..."
    return text


def _validation_split(
    X: Any,
    y: Any,
    task_type: str,
    test_size: float,
    random_seed: int,
) -> Any:
    from sklearn.model_selection import train_test_split

    stratify = None
    if task_type == "classification" and y.nunique() > 1 and y.value_counts().min() >= 2:
        stratify = y
    try:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_seed,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_seed,
            stratify=None,
        )


def _validation_metrics(
    pipeline: Any,
    X: Any,
    y: Any,
    task_type: str,
    test_size: float,
    random_seed: int,
    warnings: List[str],
) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score as skl_accuracy_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

    X_train, X_test, y_train, y_test = _validation_split(
        X,
        y,
        task_type,
        test_size,
        random_seed,
    )
    pipeline.fit(X_train, y_train)
    validation_predictions = pipeline.predict(X_test)
    if task_type == "classification":
        metrics: Dict[str, Any] = {
            "accuracy": float(skl_accuracy_score(y_test, validation_predictions)),
            "validation_rows": int(len(y_test)),
        }
        if hasattr(pipeline.named_steps["model"], "predict_proba") and y.nunique() == 2:
            try:
                probabilities = pipeline.predict_proba(X_test)
                classes = list(pipeline.named_steps["model"].classes_)
                if len(classes) == 2 and probabilities.shape[1] >= 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities[:, 1]))
                else:
                    warnings.append("ROC AUC was skipped because the training split contained one class.")
            except ValueError:
                warnings.append("ROC AUC was skipped because the validation split did not contain both classes.")
        return metrics
    return {
        "mae": float(mean_absolute_error(y_test, validation_predictions)),
        "rmse": float(mean_squared_error(y_test, validation_predictions) ** 0.5),
        "r2": float(r2_score(y_test, validation_predictions)),
        "validation_rows": int(len(y_test)),
    }


def _coerce_feature_frames(train_df: Any, predict_df: Any, feature_columns: List[str]) -> Any:
    import pandas as pd

    train_features = train_df[feature_columns].copy()
    predict_features = (
        predict_df[feature_columns].copy()
        if not predict_df.empty
        else pd.DataFrame(columns=feature_columns)
    )
    combined = pd.concat([train_features, predict_features], axis=0, ignore_index=True)

    for column in feature_columns:
        values = combined[column]
        numeric_values = pd.to_numeric(values, errors="coerce")
        non_null = values.notna()
        if non_null.any() and numeric_values[non_null].notna().all():
            train_features[column] = pd.to_numeric(train_features[column], errors="coerce")
            if not predict_features.empty:
                predict_features[column] = pd.to_numeric(predict_features[column], errors="coerce")
        else:
            train_features[column] = train_features[column].map(_category_value)
            if not predict_features.empty:
                predict_features[column] = predict_features[column].map(_category_value)

    return train_features, predict_features


def _category_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _format_model_predictions(
    pipeline: Any,
    X: Any,
    source_rows: List[Dict[str, Any]],
    entity_id_columns: List[str],
    task_type: str,
) -> List[Dict[str, Any]]:
    predicted_values = pipeline.predict(X)
    probabilities = None
    classes = None
    if task_type == "classification" and hasattr(pipeline.named_steps["model"], "predict_proba"):
        probabilities = pipeline.predict_proba(X)
        classes = list(pipeline.named_steps["model"].classes_)

    formatted = []
    for index, value in enumerate(predicted_values):
        row = source_rows[index] if index < len(source_rows) else {}
        item = {
            "entity_id": {column: row.get(column) for column in entity_id_columns},
            "prediction": _jsonable(value),
        }
        if probabilities is not None and classes is not None:
            class_probs = {
                str(_jsonable(classes[class_index])): float(probabilities[index][class_index])
                for class_index in range(len(classes))
            }
            item["class_probabilities"] = class_probs
            item["probability"] = max(class_probs.values()) if class_probs else None
        formatted.append(item)
    return formatted


def _run_baseline(
    prepared: Dict[str, Any],
    target_column: str,
    entity_id_columns: List[str],
) -> Dict[str, Any]:
    task_type = prepared["task_type"]
    labeled = prepared["labeled_rows"]
    prediction_rows = prepared["prediction_rows"] or labeled
    warnings = list(prepared["warnings"])

    if task_type == "regression":
        values = [float(row[target_column]) for row in labeled if row.get(target_column) is not None]
        prediction_value = sum(values) / len(values) if values else 0.0
        predictions = [
            {
                "entity_id": {column: row.get(column) for column in entity_id_columns},
                "prediction": prediction_value,
            }
            for row in prediction_rows
        ]
        metrics = regression_metrics(values, [prediction_value] * len(values)) if values else {}
        model_name = "MeanRegressor baseline"
    else:
        values = [row[target_column] for row in labeled if row.get(target_column) is not None]
        counts = Counter(values)
        prediction_value = counts.most_common(1)[0][0] if counts else None
        total = sum(counts.values()) or 1
        class_probabilities = {str(key): value / total for key, value in counts.items()}
        probability = class_probabilities.get(str(prediction_value), 0.0)
        predictions = [
            {
                "entity_id": {column: row.get(column) for column in entity_id_columns},
                "prediction": prediction_value,
                "probability": probability,
                "class_probabilities": class_probabilities,
            }
            for row in prediction_rows
        ]
        metrics = accuracy_score(values, [prediction_value] * len(values)) if values else {}
        model_name = "MajorityClassifier baseline"

    return {
        "ok": True,
        "task_type": task_type,
        "target_column": target_column,
        "entity_id_columns": entity_id_columns,
        "model": {"name": model_name, "checkpoint": "dependency-free fallback"},
        "dataset": {
            "training_rows": len(labeled),
            "prediction_rows": len(prepared["prediction_rows"]),
            "feature_count": len(prepared["feature_columns"]),
            "dropped_columns": prepared["dropped_columns"],
            "warnings": warnings,
        },
        "metrics": metrics,
        "predictions": predictions,
        "warnings": warnings,
    }


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value
