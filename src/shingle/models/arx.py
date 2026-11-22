"""ARX: regression of the h-day stage change on state + rain features (ridge or LightGBM)."""

from __future__ import annotations

import pandas as pd

from shingle.models.base import HORIZONS, ConformalBandsMixin
from shingle.models.direct import DirectRegressor


class ARX(ConformalBandsMixin):
    name = "arx"

    def __init__(self, features: list[str] | dict[int, list[str]], horizons=HORIZONS, coverage: float = 0.8, backend: str = "ridge", allow_missing=(), lgbm_params: dict | None = None, monotone: dict | None = None):
        self.horizons = tuple(horizons)
        self.coverage = coverage
        self.reg = DirectRegressor("stage", features, self.horizons, backend=backend, allow_missing=allow_missing,
                                   lgbm_params=lgbm_params, monotone=monotone)

    def feature_names(self, h: int) -> list[str]:
        return self.reg.feature_names(h)

    def fit(self, data: dict, train_index: pd.DatetimeIndex) -> "ARX":
        self.reg.fit(data, train_index)
        return self.calibrate(data, train_index)

    def point(self, data: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
        return self.reg.predict(data, index)
