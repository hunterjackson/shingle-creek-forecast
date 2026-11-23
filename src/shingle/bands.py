"""Likely-range bands with a learned width (docs/shingle_findings.md, 2026-11-23 follow-ups).

Calm days (|p50 - stage(t)| < CALM) keep conformal bins on |p50 - stage| calibrated on calm days only. Other days
are binned (dedicated tail bins) on a learned scale: a LightGBM model of |residual| from state, rain and GR4J
features, trained on out-of-fold perfect-rain residuals of the years before the GEFS era. Bins are calibrated on
forecast-mode (issued GEFS) residuals, so the levels are honest for live use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shingle import config, features
from shingle.evaluate import BinnedConformal

CALM = 0.05
CALM_QS = (0.25, 0.5, 0.75)
TAIL_QS = (0.3, 0.55, 0.75, 0.88, 0.95, 0.98)
LGBM = {"objective": "l1", "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 50, "verbose": -1}


def feature_names(cell: str = config.RAIN_CELL) -> list[str]:
    return ["stage", "dstage_1", "log_q", "toho_dev", f"pr_nextH__{cell}", f"pr_past3__{cell}", "gr4j_dH", "gr4j_s", "doy_sin", "doy_cos"]


def scale_features(data: dict, index: pd.DatetimeIndex, p50: pd.Series, h: int, cell: str = config.RAIN_CELL) -> pd.DataFrame:
    X = features.build(data, feature_names(cell), horizon=h).loc[index]
    X["dp50"] = p50.loc[index] - data["stage"].loc[index]
    X["adp50"] = X["dp50"].abs()
    return X


class CalmSplitBands:
    def __init__(self, calm: BinnedConformal, other: BinnedConformal, threshold: float = CALM):
        self.calm, self.other, self.threshold = calm, other, threshold

    @classmethod
    def fit(cls, residuals, adp50, scale, coverage: float = 0.8, threshold: float = CALM) -> "CalmSplitBands":
        r, a, s = (np.asarray(x, float) for x in (residuals, adp50, scale))
        ok = ~(np.isnan(r) | np.isnan(a) | np.isnan(s))
        r, a, s = r[ok], a[ok], s[ok]
        c = a < threshold
        return cls(BinnedConformal.fit_edges(r[c], a[c], CALM_QS, coverage), BinnedConformal.fit_edges(r[~c], s[~c], TAIL_QS, coverage), threshold)

    def offsets(self, adp50, scale) -> tuple[np.ndarray, np.ndarray]:
        a = np.nan_to_num(np.asarray(adp50, float), nan=0.0)
        (clo, chi), (olo, ohi) = self.calm.offsets(a), self.other.offsets(np.asarray(scale, float))
        c = a < self.threshold
        return np.where(c, clo, olo), np.where(c, chi, ohi)


class LearnedBand:
    """One horizon's band: scale model + calm/other conformal bins. Used by ConformalBandsMixin.predict()."""

    def __init__(self, model, split: CalmSplitBands, h: int, cell: str = config.RAIN_CELL):
        self.model, self.split, self.h, self.cell = model, split, h, cell

    @staticmethod
    def fit_scale(data: dict, index: pd.DatetimeIndex, p50: pd.Series, residuals: pd.Series, h: int, cell: str = config.RAIN_CELL, seed: int = 0):
        import lightgbm as lgb

        X = scale_features(data, index, p50, h, cell)
        r = residuals.reindex(index)
        ok = r.notna() & X["adp50"].notna()
        return lgb.LGBMRegressor(random_state=seed, **LGBM).fit(X[ok], r[ok].abs())

    @classmethod
    def fit(cls, scale_model, data_fc: dict, cal_index: pd.DatetimeIndex, p50_fc: pd.Series, residuals_fc: pd.Series, h: int,
            coverage: float = 0.8, cell: str = config.RAIN_CELL) -> "LearnedBand":
        band = cls(scale_model, None, h, cell)
        X = scale_features(data_fc, cal_index, p50_fc, h, cell)
        band.split = CalmSplitBands.fit(residuals_fc.reindex(cal_index), X["adp50"], band._scale(X), coverage)
        return band

    def _scale(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.model.feature_name_])

    def offsets_for(self, data: dict, index: pd.DatetimeIndex, p50: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        X = scale_features(data, index, p50, self.h, self.cell)
        return self.split.offsets(X["adp50"].to_numpy(), self._scale(X))
