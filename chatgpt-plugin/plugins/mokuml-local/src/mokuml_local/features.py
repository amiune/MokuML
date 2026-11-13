"""TILE's pinned recipe with split-aware target encoding and explanation blocks."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .vendor.tile.features import (
    CARDINALITY_THRESHOLD,
    MIN_REPEAT_RATIO,
    TargetEncoder,
    TILEFeatures,
    _encode,
    _repeat_ratio,
    _target_stats,
    components_per_column,
    string_columns,
)


class ScopedTargetEncoder(TargetEncoder):
    def __init__(self, folds):
        self.folds = folds

    def fit_transform(self, X, y, task, n_splits=3):
        self.columns_ = [
            col for col in X.columns if _repeat_ratio(X[col]) > MIN_REPEAT_RATIO
        ]
        y = np.asarray(y)
        self.n_classes_ = len(np.unique(y)) if task == "classification" else 0
        if task == "classification":
            _, y = np.unique(y, return_inverse=True)
        width = self.n_classes_ if self.n_classes_ > 2 else 1
        # The initial chronological block has no past targets. Use a fixed,
        # label-independent neutral value; never a prior from future rows.
        result = np.zeros((len(X), len(self.columns_) * width))
        self.stats_ = []
        for j, col in enumerate(self.columns_):
            codes = X[col].astype(str).to_numpy()
            self.stats_.append(_target_stats(codes, y, self.n_classes_))
            for train, validation in self.folds:
                if len(train):
                    result[validation, j * width : (j + 1) * width] = _encode(
                        codes[validation],
                        *_target_stats(codes[train], y[train], self.n_classes_),
                    )
        return result


class ScopedTILEFeatures(TILEFeatures):
    def __init__(self, window, folds):
        super().__init__(window=window, native_categories=True)
        self.encoding_folds = folds

    def fit_transform(self, X, y, embeddings, task):
        from skrub import TableVectorizer

        self.string_columns_ = string_columns(X)
        self.other_columns_ = [c for c in X.columns if c not in self.string_columns_]
        self.vectorizer_ = TableVectorizer(numeric="passthrough")
        others = X[self.other_columns_]
        numbers = (
            np.asarray(self.vectorizer_.fit_transform(others))
            if len(self.other_columns_)
            else np.empty((len(X), 0))
        )
        self.target_encoder_ = ScopedTargetEncoder(self.encoding_folds)
        encoded = self.target_encoder_.fit_transform(X[self.string_columns_], y, task)
        self.categories_ = [
            c for c in self.string_columns_ if X[c].nunique() < CARDINALITY_THRESHOLD
        ]
        bypass = numbers.shape[1] + encoded.shape[1] + len(self.categories_)
        window = tuple(max(0, bound - bypass) for bound in self.window)
        self.n_components_ = components_per_column(len(self.string_columns_), window)
        self.reducers_, reduced, widths = {}, [], []
        for col in self.string_columns_:
            block = embeddings[col]
            reducer = make_pipeline(
                StandardScaler(),
                PCA(int(min(self.n_components_, *block.shape)), random_state=42),
            )
            reduced.append(reducer.fit_transform(block))
            widths.append(reduced[-1].shape[1])
            self.reducers_[col] = reducer
        Z = self._assemble(reduced, numbers, encoded, X)
        self.n_base_ = Z.shape[1] - len(self.categories_)
        # Every transformed block belonging to one original column is permuted together.
        self.blocks_ = {col: [] for col in X.columns}
        offset = 0
        for col, width in zip(self.string_columns_, widths):
            self.blocks_[col] += list(range(offset, offset + width))
            offset += width
        if self.other_columns_:
            by_input = self.vectorizer_.output_to_input_
            for i, output in enumerate(self.vectorizer_.get_feature_names_out()):
                self.blocks_[by_input[output]].append(offset + i)
        offset += numbers.shape[1]
        width = (
            self.target_encoder_.n_classes_
            if self.target_encoder_.n_classes_ > 2
            else 1
        )
        for col in self.target_encoder_.columns_:
            self.blocks_[col] += list(range(offset, offset + width))
            offset += width
        for col in self.categories_:
            self.blocks_[col].append(offset)
            offset += 1
        return Z
