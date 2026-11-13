"""Model selection, honest evaluation, calibrated intervals and measured explanations."""

from __future__ import annotations

import gc
import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

from .adapters import Estimator
from .contracts import AugmentedPredictionResult
from .errors import SafeError
from .inference import embeddings_for, feature_frame, progress
from .models import active_tabpfn_component, available
from .splitting import folds, partitions


class SelectionError(SafeError):
    def __init__(self, candidates):
        super().__init__("no_compatible_models")
        self.candidates = candidates


def metrics(task, actual, predicted, probabilities=None):
    if task == "regression":
        result = {
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        }
        if len(actual) >= 2:
            result["r2"] = float(r2_score(actual, predicted))
    else:
        result = {
            "accuracy": float(accuracy_score(actual, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
            "log_loss": float(
                log_loss(
                    actual, probabilities, labels=np.arange(probabilities.shape[1])
                )
            ),
        }
        if probabilities.shape[1] == 2:
            result["brier"] = float(np.mean((probabilities[:, 1] - actual) ** 2))
            if len(np.unique(actual)) == 2:
                result["roc_auc"] = float(roc_auc_score(actual, probabilities[:, 1]))
    return {key: value for key, value in result.items() if np.isfinite(value)}


def loss(task, y, prediction, probabilities=None):
    return (
        float(log_loss(y, probabilities, labels=np.arange(probabilities.shape[1])))
        if task == "classification"
        else float(mean_absolute_error(y, prediction))
    )


def encoding_folds(frame, positions, task, strategy):
    positions = np.asarray(positions)
    lookup = {p: i for i, p in enumerate(positions)}
    return [
        (np.array([lookup[p] for p in t]), np.array([lookup[p] for p in v]))
        for t, v in folds(frame, positions, task, strategy, encoding=True)
    ]


def eligibility(engine, rows, features, classes, device):
    component = active_tabpfn_component() if engine == "tabpfn" else engine
    if not available(component):
        return "models_not_prepared"
    if engine == "tabpfn":
        modern = component != "tabpfn"
        max_rows = (
            (5000 if modern else 1000)
            if device == "cpu"
            else (100000 if modern else 10000)
        )
        if rows > max_rows or (not modern and (features > 500 or classes > 10)):
            return "checkpoint_limits"
    return None


def _fit(engine, task, device, frame, X, y, positions, embeddings, strategy):
    model = Estimator(
        engine, task, device, encoding_folds(frame, positions, task, strategy)
    )
    return model.fit(X.iloc[positions], y[positions], embeddings.take(positions))


def _evaluate(model, X, embeddings):
    Z = model.transform(X, embeddings)
    probabilities = model.proba_z(Z) if model.task == "classification" else None
    predicted = (
        probabilities.argmax(axis=1)
        if probabilities is not None
        else model.predict_z(Z)
    )
    if not np.isfinite(predicted).all() or (
        probabilities is not None and not np.isfinite(probabilities).all()
    ):
        raise SafeError("inference_failed")
    return predicted, probabilities, Z


def _baseline(job, frame, train, test, y, task, classes):
    if task == "classification":
        prior = np.bincount(y[train].astype(int), minlength=classes) / len(train)
        probabilities = np.tile(prior, (len(test), 1))
        return probabilities.argmax(axis=1), probabilities
    fallback = float(np.median(y[train]))
    if job.get("baseline_feature") in frame:
        values = pd.to_numeric(
            frame.iloc[test][job["baseline_feature"]], errors="coerce"
        ).to_numpy(dtype=float)
        return np.where(np.isfinite(values), values, fallback), None
    return np.full(len(test), fallback), None


def _importance(model, X, embeddings, y, positions):
    sample = np.asarray(positions)[:128]
    predicted, probabilities, Z = _evaluate(
        model, X.iloc[sample], embeddings.take(sample)
    )
    original = loss(model.task, y[sample], predicted, probabilities)
    rng = np.random.default_rng(42)
    results = []
    for col, block in list(model.features.blocks_.items())[:10]:
        if not block:
            continue
        losses = []
        for _ in range(3):
            permuted = Z.copy()
            permuted[:, block] = Z[rng.permutation(len(Z))][:, block]
            proba = model.proba_z(permuted) if model.task == "classification" else None
            pred = (
                proba.argmax(axis=1) if proba is not None else model.predict_z(permuted)
            )
            losses.append(loss(model.task, y[sample], pred, proba))
        results.append(
            dict(
                feature=col,
                effect=float(np.mean(losses) - original),
                method="grouped_permutation",
                interpretation="association",
            )
        )
    return sorted(results, key=lambda r: r["effect"], reverse=True)


def _sensitivity(
    model, X, train, positions, predictions, probabilities, influences, device, status
):
    features = [r["feature"] for r in influences[:5]]
    rows, descriptors = [], []
    for i, p in enumerate(positions[:20]):
        for col in features:
            training = X.iloc[train][col].dropna()
            if not len(training):
                continue
            baseline = (
                training.median()
                if pd.api.types.is_numeric_dtype(training)
                else training.mode().iloc[0]
            )
            row = X.iloc[[p]].copy()
            row[col] = baseline
            rows.append(row)
            descriptors.append((i, col))
    if not rows:
        return []
    perturbed = pd.concat(rows, ignore_index=True)
    # One encoder load for the whole explanation batch.
    encoded, _ = embeddings_for(perturbed, device, status)
    pred, proba, _ = _evaluate(model, perturbed, encoded)
    result = []
    for j, (i, col) in enumerate(descriptors):
        effect = (
            float(probabilities[i, int(predictions[i])] - proba[j, int(predictions[i])])
            if proba is not None
            else float(predictions[i] - pred[j])
        )
        result.append(
            dict(
                feature=col,
                effect=effect,
                method="reference_sensitivity",
                interpretation="sensitivity",
                case_id=f"case-{i + 1}",
            )
        )
    return result


def _run(job, status, device):
    task, strategy = job["task"], job["split"]
    frame = job["frame"].copy().reset_index(drop=True)
    target = job["target"]
    frame["__target"] = frame[target]
    known = np.flatnonzero(
        frame[target].notna().to_numpy() & ~frame["__score"].to_numpy()
    )
    scoring = np.flatnonzero(frame["__score"].to_numpy())
    train, calibration, test = partitions(frame, known, task, strategy)
    label_encoder = None
    y = np.zeros(len(frame), dtype=float)
    if task == "classification":
        label_encoder = LabelEncoder().fit(frame.iloc[train][target].astype(str))
        if len(label_encoder.classes_) < 2:
            raise SafeError("insufficient_classes")
        try:
            y[known] = label_encoder.transform(frame.iloc[known][target].astype(str))
        except ValueError:
            raise SafeError("insufficient_classes") from None
        y = y.astype(int)
    else:
        numeric = pd.to_numeric(frame.iloc[known][target], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(numeric).all():
            raise SafeError("invalid_outcome")
        y[known] = numeric
    frame["__target"] = y
    inner_folds = folds(frame, train, task, strategy)
    X = feature_frame(frame, job["features"], train)
    usable = [c for c in X if X.iloc[train][c].notna().any()]
    if not usable:
        raise SafeError("insufficient_data")
    X = X[usable]
    embeddings, encoder = embeddings_for(X, device, status)
    candidates, scores, devices = [], {}, {}
    requested = job.get("model", "auto")
    engines = ["tabicl", "kumo", "tabpfn"] if requested == "auto" else [requested]
    progress(status, "selecting")
    for engine in engines:
        actual_device = "cpu" if engine == "kumo" else device
        reason = eligibility(
            engine,
            len(train),
            len(usable),
            len(label_encoder.classes_) if label_encoder else 0,
            actual_device,
        )
        if reason:
            candidates.append(dict(model=engine, status="excluded", reason=reason))
            continue
        # Any accelerator retry repeats this candidate's complete fold evaluation.
        for attempt_device in (
            [actual_device, "cpu"] if actual_device == "mps" else ["cpu"]
        ):
            try:
                losses, weights = [], []
                for fit, validation in inner_folds:
                    model = _fit(
                        engine,
                        task,
                        attempt_device,
                        frame,
                        X,
                        y,
                        fit,
                        embeddings,
                        strategy,
                    )
                    pred, proba, _ = _evaluate(
                        model, X.iloc[validation], embeddings.take(validation)
                    )
                    losses.append(loss(task, y[validation], pred, proba))
                    weights.append(len(validation))
                    del model
                    gc.collect()
                score = float(np.average(losses, weights=weights))
                if not np.isfinite(score):
                    raise SafeError("inference_failed")
                candidates.append(
                    dict(
                        model=engine,
                        status="evaluated",
                        selection_score=score,
                        reason="same_model_cpu_retry"
                        if attempt_device != actual_device
                        else None,
                    )
                )
                scores[engine], devices[engine] = score, attempt_device
                break
            except Exception:
                gc.collect()
                if attempt_device == "cpu":
                    candidates.append(
                        dict(
                            model=engine,
                            status="failed",
                            reason="local_candidate_failed",
                        )
                    )
    if not scores:
        raise SelectionError(candidates)
    progress(status, "evaluating")
    model = None
    for selected in sorted(
        scores, key=lambda key: (scores[key], ["tabicl", "kumo", "tabpfn"].index(key))
    ):
        candidate = next(c for c in candidates if c["model"] == selected)
        for final_device in (
            [devices[selected], "cpu"] if devices[selected] == "mps" else ["cpu"]
        ):
            try:
                model = _fit(
                    selected,
                    task,
                    final_device,
                    frame,
                    X,
                    y,
                    train,
                    embeddings,
                    strategy,
                )
                if final_device != devices[selected]:
                    candidate["reason"] = "same_model_cpu_retry"
                candidate["status"] = "selected"
                break
            except Exception:
                gc.collect()
                if final_device == "cpu":
                    candidate["status"], candidate["reason"] = (
                        "failed",
                        "local_candidate_failed",
                    )
        if model is not None:
            break
    if model is None:
        raise SelectionError(candidates)
    held_out, test_proba, _ = _evaluate(model, X.iloc[test], embeddings.take(test))
    measurements = metrics(task, y[test], held_out, test_proba)
    base_pred, base_proba = _baseline(
        job,
        frame,
        train,
        test,
        y,
        task,
        len(label_encoder.classes_) if label_encoder else 0,
    )
    baseline = metrics(task, y[test], base_pred, base_proba)
    radius = None
    limitations = [
        "Measured influences describe associations and sensitivities, not causal effects.",
        "Grouped permutation perturbs transformed input blocks; contextual cell dependencies can make rankings approximate.",
        "Validation applies to this dataset and split; it does not establish suitability for another population or decision.",
        "Final model and all preprocessing use only the training partition.",
    ]
    if task == "classification":
        limitations.append("Class probabilities are uncalibrated model outputs.")
    elif len(calibration):
        calibrated, _, _ = _evaluate(
            model, X.iloc[calibration], embeddings.take(calibration)
        )
        residuals = np.abs(y[calibration] - calibrated)
        if strategy == "grouped":
            residuals = (
                pd.Series(residuals)
                .groupby(frame.iloc[calibration]["__group"].astype(str).to_numpy())
                .max()
                .to_numpy()
            )
        rank = int(math.ceil((len(residuals) + 1) * 0.9))
        if rank <= len(residuals) and len(residuals) >= 9:
            radius = float(np.sort(residuals)[rank - 1])
            measurements["interval_coverage"] = float(
                np.mean(np.abs(y[test] - held_out) <= radius)
            )
        else:
            limitations.append(
                "Too few independent calibration examples for a 90% prediction interval."
            )
        if strategy == "temporal":
            limitations.append(
                "Temporal interval coverage is empirical; exchangeability is not guaranteed."
            )
    progress(status, "predicting")
    positions = scoring if len(scoring) else test
    predicted, proba, _ = _evaluate(
        model, X.iloc[positions], embeddings.take(positions)
    )
    values = (
        label_encoder.inverse_transform(predicted.astype(int)).tolist()
        if label_encoder
        else predicted.astype(float).tolist()
    )
    output = []
    private = []
    for i, p in enumerate(positions):
        row = dict(case_id=f"case-{i + 1}", prediction=values[i])
        if proba is not None:
            row["probabilities"] = {
                str(c): float(proba[i, j]) for j, c in enumerate(label_encoder.classes_)
            }
        if radius is not None:
            row.update(
                lower=float(predicted[i] - radius),
                upper=float(predicted[i] + radius),
                interval_level=0.9,
            )
        if i < 20 and job.get("output", "individual") == "individual":
            output.append(row)
        local = {
            **row,
            "row_number": int(p) + 1,
            "identifiers": {
                job.get("identifier_labels", {}).get(c, c): _scalar(frame.iloc[p][c])
                for c in job["identifiers"]
            },
        }
        if not len(scoring):
            local["actual_outcome"] = _scalar(frame.iloc[p][target])
        private.append(local)
    progress(status, "explaining")
    influences = _importance(model, X, embeddings, y, test)
    if len(usable) > 10:
        limitations.append(
            "Global sensitivity checks cover the first ten input features and at most 128 test cases."
        )
    if job.get("output", "individual") == "individual":
        influences += _sensitivity(
            model, X, train, positions, predicted, proba, influences, device, status
        )
    summary = dict(count=len(values))
    if label_encoder:
        counts = pd.Series(values).value_counts()
        summary["class_counts"] = {str(k): int(v) for k, v in counts.items()}
    else:
        summary.update(
            mean=float(np.mean(values)),
            total=float(np.sum(values)),
            minimum=float(np.min(values)),
            maximum=float(np.max(values)),
        )
    quality = []
    if len(known) < 100:
        quality.append(
            "Small labeled sample; validation and explanations may be unstable."
        )
    if (
        measurements["log_loss"] >= baseline["log_loss"]
        if task == "classification"
        else measurements["mae"] >= baseline["mae"]
    ):
        quality.append(
            "The selected model did not improve on the baseline on the untouched test partition."
        )
    if len(usable) != len(job["features"]):
        quality.append("Features empty in training were excluded.")
    if X.iloc[train].isna().any().any():
        quality.append("Some training features contain missing values.")
    augmented = AugmentedPredictionResult(
        task=task,
        result_kind="predictions" if len(scoring) else "evaluation",
        target=target,
        units=job.get("units"),
        horizon=job.get("horizon"),
        as_of=job.get("as_of"),
        prediction_window_start=job.get("prediction_window_start"),
        prediction_window_end=job.get("prediction_window_end"),
        model=model.provenance(encoder),
        selection_metric="log_loss" if task == "classification" else "mae",
        candidates=candidates,
        split=strategy,
        training_rows=len(train),
        calibration_rows=len(calibration),
        test_rows=len(test),
        prediction_rows=len(positions),
        metrics=measurements,
        baseline_metrics=baseline,
        probability_calibration="uncalibrated" if label_encoder else "not_applicable",
        interval_method="split_conformal_90" if radius is not None else "unavailable",
        predictions=output,
        summary=summary,
        feature_influences=influences,
        data_quality=quality,
        limitations=limitations,
    )
    return dict(augmented=augmented.model_dump(exclude_none=True), predictions=private)


def _scalar(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def run(job, status=None):
    import torch

    progress(status, "preparing")
    device = (
        "mps"
        if os.environ.get("MOKUML_DEVICE") != "cpu"
        and torch.backends.mps.is_available()
        else "cpu"
    )
    try:
        return _run(job, status, device)
    except SafeError:
        raise
    except Exception:
        if device != "mps":
            raise
        gc.collect()
        torch.mps.empty_cache()
        return _run(job, status, "cpu")
