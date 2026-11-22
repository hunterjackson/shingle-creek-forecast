"""Physically structured hybrid: forecast creek flow and Lake Toho level, map to stage.

    stage(t+h) = CombinedRating(Q(t+h), Toho(t+h))

Low water comes from the (managed) lake forecast, floods from the flow
forecast; the rating extrapolates beyond the training range through Q.
"""

from __future__ import annotations

import pandas as pd

from shingle import rating as rating_mod
from shingle.models.base import HORIZONS, ConformalBandsMixin


class RatingHybrid(ConformalBandsMixin):
    name = "hybrid_rating"

    def __init__(self, flow_model, lake_model, horizons=HORIZONS, coverage: float = 0.8, rating_opts: dict | None = None):
        self.flow = flow_model
        self.lake = lake_model
        self.horizons = tuple(horizons)
        self.coverage = coverage
        self.rating: rating_mod.CombinedRating | None = None
        # carry today's rating residual forward: resid(t+h) ~ phi_h * resid(t)
        self.carry_residual = True
        self.phi: dict[int, float] = {}
        # passed to rating.fit_combined, e.g. {"floodplain": True, "d_max": 1.2, "g_min": 0.3}
        self.rating_opts = dict(rating_opts or {})

    def fit(self, data: dict, train_index: pd.DatetimeIndex) -> "RatingHybrid":
        d = pd.concat([data["q"], data["stage"], data["toho"]], axis=1, keys=["q", "stage", "toho"]).loc[train_index].dropna()
        self.rating = rating_mod.fit_combined(d["q"], d["stage"], d["toho"], **self.rating_opts)
        resid = self._rating_residual(data)
        for h in self.horizons:
            pair = pd.concat([resid, resid.shift(-h)], axis=1, keys=["now", "later"]).loc[train_index].dropna()
            denom = (pair["now"] ** 2).sum()
            self.phi[h] = float((pair["now"] * pair["later"]).sum() / denom) if denom > 0 else 0.0
        self.flow.fit(data, train_index)
        self.lake.fit(data, train_index)
        return self.calibrate(data, train_index)

    def _rating_residual(self, data: dict) -> pd.Series:
        """Observed stage minus rating(observed Q, observed Toho)."""
        d = pd.concat([data["q"], data["toho"]], axis=1, keys=["q", "toho"])
        ok = d.notna().all(axis=1)
        fitted = pd.Series(float("nan"), index=d.index)
        fitted[ok] = self.rating.predict_stage(d.loc[ok, "q"].to_numpy(), d.loc[ok, "toho"].to_numpy())
        return data["stage"] - fitted

    def components(self, data: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
        q = self.flow.predict(data, index)
        t = self.lake.predict(data, index)
        resid_now = self._rating_residual(data).loc[index].fillna(0.0) if self.carry_residual else None
        cols = {}
        for h in self.horizons:
            cols[f"h{h}_q"] = q[f"h{h}"]
            cols[f"h{h}_toho"] = t[f"h{h}"]
            ok = q[f"h{h}"].notna() & t[f"h{h}"].notna()
            st = pd.Series(float("nan"), index=index)
            st[ok] = self.rating.predict_stage(q.loc[ok, f"h{h}"].to_numpy(), t.loc[ok, f"h{h}"].to_numpy())
            if self.carry_residual:
                st = st + self.phi.get(h, 0.0) * resid_now
            cols[f"h{h}_stage"] = st
        return pd.DataFrame(cols, index=index)

    def point(self, data: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
        comp = self.components(data, index)
        return pd.DataFrame({f"h{h}": comp[f"h{h}_stage"] for h in self.horizons}, index=index)
