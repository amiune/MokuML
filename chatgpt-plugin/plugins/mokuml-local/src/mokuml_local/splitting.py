"""Independent outer partitions and training-only model/encoding folds."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    train_test_split,
)

from .errors import SafeError


def _times(frame, positions):
    values = pd.to_datetime(frame.iloc[positions]["__as_of"], errors="coerce", utc=True)
    if values.isna().any():
        raise SafeError("invalid_split")
    return values.to_numpy()


def _purge(frame, train, boundary):
    if "__label_end" not in frame:
        return np.asarray(train)
    ends = pd.to_datetime(frame.iloc[train]["__label_end"], errors="coerce", utc=True)
    if ends.isna().any():
        raise SafeError("invalid_split")
    return np.asarray(train)[(ends <= pd.Timestamp(boundary)).to_numpy()]


def _groups(frame, positions):
    if "__group" not in frame or frame.iloc[positions]["__group"].isna().any():
        raise SafeError("invalid_split")
    return frame.iloc[positions]["__group"].astype(str).to_numpy()


def partitions(frame, positions, task, strategy):
    positions = np.asarray(positions, dtype=int)
    if len(positions) < 15:
        raise SafeError("insufficient_data")
    calibration = np.array([], dtype=int)
    try:
        if strategy == "temporal":
            times = _times(frame, positions)
            unique = np.unique(times)
            if len(unique) < 8:
                raise SafeError("invalid_split")
            test_boundary = unique[max(1, int(len(unique) * 0.8))]
            test = positions[times >= test_boundary]
            development = positions[times < test_boundary]
            if task == "regression":
                boundary = unique[max(1, int(len(unique) * 0.6))]
                calibration = _purge(
                    frame,
                    positions[(times >= boundary) & (times < test_boundary)],
                    test_boundary,
                )
                development = positions[times < boundary]
            else:
                boundary = test_boundary
            train = _purge(frame, development, boundary)
        elif strategy == "grouped":
            groups = _groups(frame, positions)
            tr, te = next(
                GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
                    positions, groups=groups
                )
            )
            train, test = positions[tr], positions[te]
            if task == "regression":
                tr, ca = next(
                    GroupShuffleSplit(
                        n_splits=1, test_size=0.25, random_state=43
                    ).split(train, groups=_groups(frame, train))
                )
                train, calibration = train[tr], train[ca]
        else:
            stratify = (
                frame.iloc[positions]["__target"] if task == "classification" else None
            )
            train, test = train_test_split(
                positions, test_size=0.2, random_state=42, stratify=stratify
            )
            if task == "regression":
                train, calibration = train_test_split(
                    train, test_size=0.25, random_state=43
                )
        if (
            len(train) < 9
            or len(test) < 2
            or (task == "regression" and not len(calibration))
        ):
            raise SafeError("insufficient_data")
        return np.asarray(train), np.asarray(calibration), np.asarray(test)
    except SafeError:
        raise
    except Exception:
        raise SafeError("invalid_split") from None


def folds(frame, positions, task, strategy, encoding=False):
    """Return absolute positions; no calibration or test rows enter these folds."""
    positions = np.asarray(positions, dtype=int)
    try:
        if strategy == "temporal":
            times = _times(frame, positions)
            unique = np.unique(times)
            if len(unique) < 4:
                return [] if encoding else _invalid()
            blocks = np.array_split(unique, 4)
            result = []
            for block in blocks[1:]:
                if not len(block):
                    continue
                train = _purge(frame, positions[times < block[0]], block[0])
                validation = positions[np.isin(times, block)]
                if len(train) and len(validation):
                    result.append((train, validation))
            if not encoding and len(result) != 3:
                raise SafeError("invalid_split")
        elif strategy == "grouped":
            groups = _groups(frame, positions)
            count = min(3, len(np.unique(groups)))
            if count < 2:
                return [] if encoding else _invalid()
            if not encoding and count != 3:
                raise SafeError("invalid_split")
            result = [
                (positions[t], positions[v])
                for t, v in GroupKFold(count).split(positions, groups=groups)
            ]
        else:
            count = min(3, len(positions))
            if count < 2:
                return [] if encoding else _invalid()
            y = frame.iloc[positions]["__target"].to_numpy()
            if task == "classification" and not encoding:
                splitter = StratifiedKFold(3, shuffle=True, random_state=42)
                result = [
                    (positions[t], positions[v])
                    for t, v in splitter.split(positions, y)
                ]
            else:
                result = [
                    (positions[t], positions[v])
                    for t, v in KFold(count, shuffle=True, random_state=42).split(
                        positions
                    )
                ]
        if not encoding:
            classes = (
                set(frame.iloc[positions]["__target"])
                if task == "classification"
                else None
            )
            if len(result) != 3 or any(len(t) < 5 or not len(v) for t, v in result):
                raise SafeError("insufficient_data")
            if classes and any(
                set(frame.iloc[t]["__target"]) != classes for t, _ in result
            ):
                raise SafeError("insufficient_classes")
        return result
    except SafeError:
        raise
    except Exception:
        raise SafeError("invalid_split") from None


def _invalid():
    raise SafeError("invalid_split")
