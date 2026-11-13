import numpy as np
import pandas as pd
import pytest

from mokuml_local.errors import SafeError
from mokuml_local.features import ScopedTargetEncoder
from mokuml_local.splitting import folds, partitions


def test_grouped_patients_never_overlap_across_partitions_or_folds():
    frame = pd.DataFrame(
        {"__group": np.repeat(np.arange(50), 4), "__target": np.tile([0, 1], 100)}
    )
    train, calibration, test = partitions(
        frame, np.arange(len(frame)), "regression", "grouped"
    )

    def groups(p):
        return set(frame.iloc[p].__group)

    assert (
        not groups(train) & groups(test)
        and not groups(train) & groups(calibration)
        and not groups(calibration) & groups(test)
    )
    for t, v in folds(frame, train, "classification", "grouped"):
        assert not groups(t) & groups(v)


def test_temporal_purge_keeps_equal_cutoffs_together():
    frame = pd.DataFrame(
        {
            "__as_of": np.repeat(pd.date_range("2020-01-01", periods=100), 2),
            "__target": range(200),
        }
    )
    frame["__label_end"] = frame.__as_of + pd.Timedelta(days=4)
    train, calibration, test = partitions(
        frame, np.arange(200), "regression", "temporal"
    )
    assert frame.iloc[train].__label_end.max() <= frame.iloc[calibration].__as_of.min()
    assert frame.iloc[calibration].__label_end.max() <= frame.iloc[test].__as_of.min()
    for t, v in folds(frame, train, "regression", "temporal"):
        assert frame.iloc[t].__label_end.max() <= frame.iloc[v].__as_of.min()
        assert not set(frame.iloc[t].__as_of) & set(frame.iloc[v].__as_of)


def test_temporal_encoder_never_uses_own_or_future_labels():
    X = pd.DataFrame({"category": ["same"] * 12})
    y = np.arange(12, dtype=float)
    scoped = [(np.arange(4), np.arange(4, 8)), (np.arange(8), np.arange(8, 12))]
    first = ScopedTargetEncoder(scoped).fit_transform(X, y, "regression")
    changed = y.copy()
    changed[8:] += 100000
    second = ScopedTargetEncoder(scoped).fit_transform(X, changed, "regression")
    np.testing.assert_array_equal(first[:8], second[:8])
    np.testing.assert_array_equal(first[:4], np.zeros((4, 1)))


def test_group_target_encoder_excludes_same_patient():
    X = pd.DataFrame({"category": ["same"] * 12})
    groups = np.repeat(np.arange(3), 4)
    scoped = [
        (np.flatnonzero(groups != g), np.flatnonzero(groups == g)) for g in range(3)
    ]
    original = ScopedTargetEncoder(scoped).fit_transform(X, np.arange(12), "regression")
    y = np.arange(12, dtype=float)
    y[:4] += 100000
    changed = ScopedTargetEncoder(scoped).fit_transform(X, y, "regression")
    np.testing.assert_array_equal(original[:4], changed[:4])


def test_insufficient_temporal_history_does_not_fall_back_to_random():
    frame = pd.DataFrame({"__as_of": ["2026-01-01"] * 50, "__target": range(50)})
    with pytest.raises(SafeError):
        partitions(frame, np.arange(50), "regression", "temporal")
