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


WET_MM = 5.0
WET_QS = (1 / 3, 2 / 3)
WET_MIN_COUNT = 30


def forecast_rain_sum(data: dict, index: pd.DatetimeIndex, h: int, weights: dict[str, float]) -> pd.Series:
    """Basin-areal rain (mm) over days t+1..t+h for each issue day: the issued forecast (data['fcst']['pr'], the
    unscaled one, never by_h) when present, else the observed future rain."""
    w = pd.Series(weights, dtype=float)
    if "fcst" in data:
        fpr = data["fcst"]["pr"]
        total = sum(_areal(fpr[k], w) for k in range(1, h + 1))
    else:
        areal = _areal(data["pr"], w)
        total = sum(areal.shift(-k) for k in range(1, h + 1))
    return total.reindex(index)


def _areal(df: pd.DataFrame, w: pd.Series) -> pd.Series:
    w = w[w.index.isin(df.columns)]
    return (df[w.index] * (w / w.sum())).sum(axis=1, min_count=len(w))


class WetBand:
    """T2 (docs/divergence_test_plan.md): on days with >= WET_MM of forecast basin rain over t+1..t+h, conformal bins
    on a learned |residual| scale calibrated on such days only; every other day keeps the base (production) band.

    Forecast-mode tests 2020-23 and 2024-26: day-1 coverage on 5-20 mm days 63-64 % -> 78-85 %, dry days unchanged."""

    def __init__(self, bins: BinnedConformal, threshold: float = WET_MM):
        self.bins, self.threshold = bins, threshold
        self.base = self.scale_model = self.h = self.weights = None

    @classmethod
    def fit(cls, residuals, scale, coverage: float = 0.8, threshold: float = WET_MM) -> "WetBand":
        return cls(BinnedConformal.fit_edges(residuals, scale, WET_QS, coverage, WET_MIN_COUNT), threshold)

    def combine(self, base_lo, base_hi, wet, scale) -> tuple[np.ndarray, np.ndarray]:
        wet = np.asarray(wet, bool)
        lo, hi = self.bins.offsets(np.asarray(scale, float))
        return np.where(wet, lo, base_lo), np.where(wet, hi, base_hi)

    def attach(self, base, scale_model, h: int, weights: dict[str, float]) -> "WetBand":
        self.base, self.scale_model, self.h, self.weights = base, scale_model, h, dict(weights)
        return self

    def scale(self, data: dict, index: pd.DatetimeIndex, p50: pd.Series) -> np.ndarray:
        X = scale_features(data, index, p50, self.h)
        return self.scale_model.predict(X[self.scale_model.feature_name_])

    def wet(self, data: dict, index: pd.DatetimeIndex) -> np.ndarray:
        return (forecast_rain_sum(data, index, self.h, self.weights).fillna(0.0) >= self.threshold).to_numpy()

    def offsets_for(self, data: dict, index: pd.DatetimeIndex, p50: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(self.base, "offsets_for"):
            blo, bhi = self.base.offsets_for(data, index, p50)
        else:  # production day-1 band: bins on |p50 - stage(t)|
            blo, bhi = self.base.offsets((p50 - data["stage"].loc[index]).abs().to_numpy())
        return self.combine(blo, bhi, self.wet(data, index), self.scale(data, index, p50))
