"""Persistence baseline: stage(t+h) = stage(t), with conformal intervals."""

from __future__ import annotations

import pandas as pd

from shingle.models.base import HORIZONS, ConformalBandsMixin


class Persistence(ConformalBandsMixin):
    name = "persistence"

    def __init__(self, horizons=HORIZONS, coverage: float = 0.8):
        self.horizons = tuple(horizons)
        self.coverage = coverage

    def fit(self, data: dict, train_index: pd.DatetimeIndex) -> "Persistence":
        return self.calibrate(data, train_index)

    def point(self, data: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
        now = data["stage"].loc[index]
        return pd.DataFrame({f"h{h}": now for h in self.horizons}, index=index)

    def band_scale(self, data, index, p50, h):
        # p50 - now is always 0; widen bands when the creek is already moving
        return data["stage"].diff().abs().loc[index]

    @property
    def offsets(self) -> dict[int, tuple[float, float]]:
        """Representative (median-bin) offsets, kept for introspection/tests."""
        return {h: (float(b.lo[len(b.lo) // 2]), float(b.hi[len(b.hi) // 2])) for h, b in self.bands.items()}
