"""Short-record residual corrector on top of a long-record base model.

Radar rain (MRMS) only exists from 2018-07, so instead of retraining
everything on 3.5 years, a small ridge model learns the base model's
h-day residual from radar features. Where radar features are missing the
prediction falls back to the base model unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shingle import features
from shingle.models.base import HORIZONS, ConformalBandsMixin, targets
from shingle.models.direct import _make_backend


class ResidualCorrector(ConformalBandsMixin):
    name = "corrected"

    def __init__(self, base, features: list[str] | dict[int, list[str]], horizons=HORIZONS, backend: str = "ridge", coverage: float = 0.8):
        self.base = base
        self._features = features
        self.horizons = tuple(horizons)
        self.backend = backend
        self.coverage = coverage
        self.models: dict[int, object] = {}
        self.n_rows: dict[int, int] = {}

    def feature_names(self, h: int) -> list[str]:
        return list(self._features[h] if isinstance(self._features, dict) else self._features)

    def fit(self, data: dict, train_index: pd.DatetimeIndex) -> "ResidualCorrector":
        self.base.fit(data, train_index)
        base_pred = self.base.point(data, train_index)
        y = targets(data["stage"], self.horizons).loc[train_index]
        for h in self.horizons:
            X = features.build(data, self.feature_names(h), horizon=h).loc[train_index]
            resid = y[f"h{h}"] - base_pred[f"h{h}"]
            ok = X.notna().all(axis=1) & resid.notna()
            self.n_rows[h] = int(ok.sum())
            model = _make_backend(self.backend)
            model.fit(X[ok].to_numpy(), resid[ok].to_numpy())
            self.models[h] = model
        return self.calibrate(data, train_index)

    def point(self, data: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
        out = self.base.point(data, index).copy()
        for h in self.horizons:
            X = features.build(data, self.feature_names(h), horizon=h).loc[index]
            ok = X.notna().all(axis=1)
            if ok.any():
                corr = np.zeros(len(index))
                corr[ok.to_numpy()] = self.models[h].predict(X[ok].to_numpy())
                out[f"h{h}"] = out[f"h{h}"] + corr
        return out
