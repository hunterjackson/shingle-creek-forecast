"""Per-horizon weighted average of two point-forecast models, with conformal bands.

Used to combine the direct-stage ARX (best overall/1-3 day) with the
flow+lake RatingHybrid (best low water, physically extrapolating).
"""

from __future__ import annotations

import pandas as pd

from shingle.models.base import HORIZONS, ConformalBandsMixin


class Blend(ConformalBandsMixin):
    name = "blend"

    def __init__(self, a, b, weights: float | dict[int, float] = 0.5, horizons=HORIZONS, coverage: float = 0.8):
        self.a, self.b = a, b
        self.horizons = tuple(horizons)
        self.weights = weights if isinstance(weights, dict) else {h: float(weights) for h in self.horizons}
        self.coverage = coverage

    def fit(self, data: dict, train_index: pd.DatetimeIndex) -> "Blend":
        self.a.fit(data, train_index)
        self.b.fit(data, train_index)
        return self.calibrate(data, train_index)

    def point(self, data: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
        pa, pb = self.a.point(data, index), self.b.point(data, index)
        return pd.DataFrame({f"h{h}": self.weights[h] * pa[f"h{h}"] + (1 - self.weights[h]) * pb[f"h{h}"] for h in self.horizons}, index=index)
