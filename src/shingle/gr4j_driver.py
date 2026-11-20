"""Calibrate GR4J on the training period and simulate it over the full record.

The simulated flow and store levels become features ('gr4j_*' in
features.py). States at day t depend only on rain up to t; simulated flow at
t+h uses rain up to t+h (observed in training, forecast when live), just like
the pr_next features.
"""

from __future__ import annotations

import pandas as pd

from home_display.shingle.models import gr4j


def areal(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()
    return (df[w.index] * w).sum(axis=1, min_count=len(w))


class GR4JDriver:
    def __init__(self, weights: dict[str, float], area_km2: float = 466.2, objective: str = "nse", warmup: int = 365, maxiter: int = 100, pet_weights: dict[str, float] | None = None):
        self.weights = weights
        self.pet_weights = pet_weights
        self.area = area_km2
        self.objective = objective
        self.warmup = warmup
        self.maxiter = maxiter
        self.params: gr4j.Params | None = None
        self.score: float | None = None

    def _inputs(self, data: dict) -> tuple[pd.Series, pd.Series]:
        P = areal(data["pr"], self.weights)
        pet_w = self.pet_weights or {c: 1.0 for c in self.weights if c in data["pet"].columns} or {c: 1.0 for c in data["pet"].columns}
        E = areal(data["pet"], pet_w)
        return P, E

    def fit(self, data: dict, train_index: pd.DatetimeIndex) -> "GR4JDriver":
        P, E = self._inputs(data)
        q = gr4j.cfs_to_mm_day(data["q"], self.area)
        df = pd.concat([P, E, q], axis=1, keys=["P", "E", "q"]).loc[train_index[0] : train_index[-1]]
        df = df.dropna(subset=["P", "E"])
        df["q"] = df["q"].interpolate(limit_direction="both")
        cal = gr4j.calibrate(df.P.to_numpy(), df.E.to_numpy(), df.q.to_numpy(), warmup=self.warmup, objective=self.objective, maxiter=self.maxiter)
        self.params, self.score = cal.params, cal.score
        return self

    def simulate(self, data: dict) -> pd.DataFrame:
        P, E = self._inputs(data)
        ok = P.notna() & E.notna()
        out = gr4j.run(P[ok].to_numpy(), E[ok].to_numpy(), self.params)
        sim = pd.DataFrame(
            {"q": gr4j.mm_day_to_cfs(out["q"], self.area), "s": out["s"], "r": out["r"]}, index=P.index[ok]
        )
        return sim.reindex(P.index)
