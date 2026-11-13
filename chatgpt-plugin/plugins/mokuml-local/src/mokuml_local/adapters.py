"""Pinned offline learners sharing a split-aware TILE feature pipeline."""

from __future__ import annotations

import os
from importlib.metadata import version

import numpy as np
import pandas as pd

from .features import ScopedTILEFeatures
from .models import CLASSIFIER, REGRESSOR, active_tabpfn_component, manifest, verify


class Estimator:
    def __init__(self, engine, task, device, encoding_folds):
        self.engine, self.task = engine, task
        self.device = "cpu" if engine == "kumo" else device
        self.encoding_folds = encoding_folds
        self.component = active_tabpfn_component() if engine == "tabpfn" else engine
        self.spec = manifest()[self.component]
        self.directory = verify(self.component)
        window = (
            (200, 600)
            if engine == "kumo"
            else (200, 400)
            if self.component == "tabpfn"
            else (200, 800)
        )
        self.features = ScopedTILEFeatures(window, encoding_folds)
        self.ensembles = max(1, min(8, int(os.environ.get("MOKUML_ESTIMATORS", "8"))))

    def fit(self, X, y, embeddings):
        Z = self.features.fit_transform(X, y, embeddings, self.task)
        if self.engine == "tabicl":
            from tabicl import TabICLClassifier, TabICLRegressor

            cls = TabICLClassifier if self.task == "classification" else TabICLRegressor
            checkpoint = CLASSIFIER if self.task == "classification" else REGRESSOR
            self.model = cls(
                model_path=str(self.directory / checkpoint),
                checkpoint_version=checkpoint,
                allow_auto_download=False,
                device=self.device,
                random_state=42,
                n_estimators=self.ensembles,
                n_jobs=1,
                verbose=False,
            )
            self.model.fit(self._frame(Z), y)
        elif self.engine == "tabpfn":
            from tabpfn import TabPFNClassifier, TabPFNRegressor

            cls = TabPFNClassifier if self.task == "classification" else TabPFNRegressor
            asset = (
                self.spec["classifier"]
                if self.task == "classification"
                else self.spec["regressor"]
            )
            self.model = cls(
                model_path=self.directory / asset,
                device=self.device,
                categorical_features_indices=list(
                    range(self.features.n_base_, Z.shape[1])
                ),
                random_state=42,
                n_estimators=self.ensembles,
                n_preprocessing_jobs=1,
                show_progress_bar=False,
                ignore_pretraining_limits=False,
            )
            self.model.fit(self._frame(Z), y)
        else:
            import sdm
            import torch

            self.model = sdm.models.KumoTabular(
                task=self.task, size="small", pretrained=False, device="cpu"
            )
            checkpoint = self.directory / (
                "small/classifier.pt"
                if self.task == "classification"
                else "small/regressor.pt"
            )
            self.model.models[sdm.Task(self.task)].load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
            self.model.eval()
            target = sdm.TableTensor.from_pandas(
                df=pd.DataFrame({"target": np.asarray(y)}),
                stypes={
                    "target": "categorical"
                    if self.task == "classification"
                    else "numerical"
                },
                device="cpu",
            )
            self.model.fit(
                x=self._table(Z),
                y=target,
                num_estimators=self.ensembles,
                generator=torch.Generator("cpu").manual_seed(42),
            )
        return self

    def _frame(self, Z):
        frame = pd.DataFrame(Z)
        for col in frame.columns[: self.features.n_base_]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
        for col in frame.columns[self.features.n_base_ :]:
            frame[col] = frame[col].replace("nan", np.nan)
        return frame

    def _table(self, Z):
        import sdm

        frame = self._frame(Z)
        frame.columns = [str(c) for c in frame.columns]
        stypes = {
            c: "numerical" if i < self.features.n_base_ else "categorical"
            for i, c in enumerate(frame.columns)
        }
        return sdm.TableTensor.from_pandas(df=frame, stypes=stypes, device="cpu")

    def transform(self, X, embeddings):
        return self.features.transform(X, embeddings)

    def predict_z(self, Z):
        if self.task == "classification":
            return self.proba_z(Z).argmax(axis=1)
        if self.engine == "kumo":
            return (
                self.model.predict(self._table(Z))
                .numerical.float()
                .mean(dim=-1)
                .cpu()
                .numpy()
            )
        return np.asarray(self.model.predict(self._frame(Z)), dtype=float)

    def proba_z(self, Z):
        if self.engine == "kumo":
            import sdm

            result = self.model.predict(self._table(Z))
            names = list(result.columns[sdm.Stype.numerical])
            columns = [names.index(str(i)) for i in range(len(names))]
            values = result.numerical[..., columns].float().cpu().numpy()
        else:
            values = self.model.predict_proba(self._frame(Z))
        values = np.clip(np.asarray(values, dtype=float), 1e-9, 1)
        return values / values.sum(axis=1, keepdims=True)

    def provenance(self, encoder):
        package = {
            "tabicl": "tabicl",
            "tabpfn": "tabpfn",
            "kumo": "structured-data-models",
        }[self.engine]
        if self.engine == "tabicl":
            asset = CLASSIFIER if self.task == "classification" else REGRESSOR
        elif self.engine == "tabpfn":
            asset = (
                self.spec["classifier"]
                if self.task == "classification"
                else self.spec["regressor"]
            )
        else:
            asset = (
                "small/classifier.pt"
                if self.task == "classification"
                else "small/regressor.pt"
            )
        revision = (
            self.spec.get("sources", {})
            .get(asset, {})
            .get("revision", self.spec["revision"])
        )
        return dict(
            engine=self.engine,
            version=version(package),
            checkpoint_revision=revision,
            checkpoint_name=asset,
            checkpoint_version=self.spec.get("version", "v2-20260212"),
            checkpoint_sha256=self.spec["files"][asset],
            encoder=encoder,
            device=self.device,
        )
