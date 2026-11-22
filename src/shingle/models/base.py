"""Shared forecast-model interface.

Every candidate model is fit on a dict of named inputs (stage, rain, lake, ...)
plus the training issue dates, and predicts, for each issue date t, the stage
quantiles p10/p50/p90 at t+h for every horizon h. Output columns 'h{h}_p{q}'.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

HORIZONS = (1, 2, 3, 4, 5, 6, 7)
QUANTILES = (10, 50, 90)


def forecast_columns(horizons=HORIZONS, quantiles=QUANTILES) -> list[str]:
    return [f"h{h}_p{q}" for h in horizons for q in quantiles]


def targets(stage: pd.Series, horizons=HORIZONS) -> pd.DataFrame:
    """Stage at t+h for each issue date t."""
    return pd.DataFrame({f"h{h}": stage.shift(-h) for h in horizons})


class ForecastModel(Protocol):
    name: str
    horizons: tuple

    def fit(self, data: dict, train_index: pd.DatetimeIndex) -> "ForecastModel": ...

    def predict(self, data: dict, index: pd.DatetimeIndex) -> pd.DataFrame: ...


class ConformalBandsMixin:
    """calibrate()/predict() for models that provide point(data, index) -> 'h{h}' frame.

    Bands are split-conformal, binned on the band scale (default |p50 - stage(t)|),
    so the interval widens when a large change is forecast.
    """

    coverage: float = 0.8
    n_bins: int = 5

    def point(self, data: dict, index: pd.DatetimeIndex) -> pd.DataFrame:  # pragma: no cover - abstract
        raise NotImplementedError

    def band_scale(self, data: dict, index: pd.DatetimeIndex, p50: pd.Series, h: int) -> pd.Series:
        return (p50 - data["stage"].loc[index]).abs()

    def calibrate(self, data: dict, cal_index: pd.DatetimeIndex):
        from shingle.evaluate import BinnedConformal

        pred = self.point(data, cal_index)
        y = targets(data["stage"], self.horizons).loc[cal_index]
        self.bands = {}
        for h in self.horizons:
            p50 = pred[f"h{h}"]
            self.bands[h] = BinnedConformal.fit(y[f"h{h}"] - p50, self.band_scale(data, cal_index, p50, h), self.n_bins, self.coverage)
        return self

    def predict(self, data: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
        pred = self.point(data, index)
        out = {}
        for h in self.horizons:
            p50 = pred[f"h{h}"]
            lo, hi = self.bands[h].offsets(self.band_scale(data, index, p50, h).to_numpy())
            out[f"h{h}_p10"], out[f"h{h}_p50"], out[f"h{h}_p90"] = p50 + lo, p50, p50 + hi
        return pd.DataFrame(out, index=index)[forecast_columns(self.horizons)]
