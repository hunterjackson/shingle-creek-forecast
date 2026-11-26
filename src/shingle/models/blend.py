"""Per-horizon weighted average of two point-forecast models, with conformal bands.

Used to combine the direct-stage ARX (best overall/1-3 day) with the
flow+lake RatingHybrid (best low water, physically extrapolating).
An optional day1 model (models.anchor.MorningAnchor) replaces day 1 wherever it predicts.
"""

from __future__ import annotations

import pandas as pd

from shingle.models.base import HORIZONS, ConformalBandsMixin


class Blend(ConformalBandsMixin):
    name = "blend"

    def __init__(self, a, b, weights: float | dict[int, float] = 0.5, horizons=HORIZONS, coverage: float = 0.8, day1=None):
        self.a, self.b, self.day1 = a, b, day1
        self.horizons = tuple(horizons)
        self.weights = weights if isinstance(weights, dict) else {h: float(weights) for h in self.horizons}
        self.coverage = coverage

    def fit_members(self, data: dict, train_index: pd.DatetimeIndex) -> "Blend":
        """Fit the member models only (no bands): used for out-of-fold predictions."""
        self.a.fit(data, train_index)
        self.b.fit(data, train_index)
        if self.day1 is not None:
            self.day1.fit(data, train_index)
        return self

    def fit(self, data: dict, train_index: pd.DatetimeIndex) -> "Blend":
        return self.fit_members(data, train_index).calibrate(data, train_index)

    def point(self, data: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
        pa, pb = self.a.point(data, index), self.b.point(data, index)
        out = pd.DataFrame({f"h{h}": self.weights[h] * pa[f"h{h}"] + (1 - self.weights[h]) * pb[f"h{h}"] for h in self.horizons}, index=index)
        if getattr(self, "day1", None) is not None and 1 in self.horizons:  # getattr: artifacts pickled before day1
            out["h1"] = self.day1.point(data, index)["h1"].fillna(out["h1"])
        return out
