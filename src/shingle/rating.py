"""Stage <-> discharge ratings for converting flow (GR4J, NWM) to stage.

At Campbell a single-valued rating does not hold below ~400 cfs: the stage is
set by Lake Tohopekaliga backwater (same 50-100 cfs spans 52.1-55.2 ft). So
besides the classic free-flow power law there is a backwater form
    stage = lake + c * Q**d
i.e. downstream water surface plus the head loss needed to pass Q.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class FreeRating:
    h0: float
    a: float
    b: float

    def predict_stage(self, q: np.ndarray) -> np.ndarray:
        return self.h0 + self.a * np.clip(q, 0, None) ** self.b

    def predict_flow(self, stage: np.ndarray) -> np.ndarray:
        head = np.clip(np.asarray(stage) - self.h0, 0, None)
        return (head / self.a) ** (1 / self.b)

    def to_dict(self) -> dict:
        return {"kind": "free", **asdict(self)}


@dataclass(frozen=True)
class BackwaterRating:
    c: float
    d: float

    def predict_stage(self, q: np.ndarray, lake: np.ndarray) -> np.ndarray:
        return np.asarray(lake) + self.c * np.clip(q, 0, None) ** self.d

    def predict_flow(self, stage: np.ndarray, lake: np.ndarray) -> np.ndarray:
        head = np.clip(np.asarray(stage) - np.asarray(lake), 0, None)
        return (head / self.c) ** (1 / self.d)

    def to_dict(self) -> dict:
        return {"kind": "backwater", **asdict(self)}


def _smoothmax(a, b, k):
    return np.maximum(a, b) + np.log1p(np.exp(-k * np.abs(a - b))) / k


@dataclass(frozen=True)
class CombinedRating:
    """Smooth max of lake control (lake + c*Q^d) and channel control (h0 + a*Q^b)."""

    c: float
    d: float
    h0: float
    a: float
    b: float
    k: float

    def predict_stage(self, q: np.ndarray, lake: np.ndarray) -> np.ndarray:
        q = np.clip(np.asarray(q, float), 0, None)
        lake_ctrl = np.asarray(lake, float) + self.c * q**self.d
        chan_ctrl = self.h0 + self.a * q**self.b
        return _smoothmax(lake_ctrl, chan_ctrl, self.k)

    def predict_flow(self, stage: np.ndarray, lake: np.ndarray, q_max: float = 1e5) -> np.ndarray:
        """Invert by bisection on log(1+Q); stage is monotone in Q for a fixed lake."""
        stage = np.asarray(stage, float)
        lake = np.broadcast_to(np.asarray(lake, float), stage.shape)
        lo = np.zeros_like(stage)
        hi = np.full_like(stage, np.log1p(q_max))
        for _ in range(60):
            mid = (lo + hi) / 2
            too_high = self.predict_stage(np.expm1(mid), lake) > stage
            hi = np.where(too_high, mid, hi)
            lo = np.where(too_high, lo, mid)
        return np.expm1((lo + hi) / 2)

    def to_dict(self) -> dict:
        return {"kind": "combined", **asdict(self)}


def from_dict(d: dict) -> FreeRating | BackwaterRating | CombinedRating:
    d = dict(d)
    kind = d.pop("kind")
    return {"free": FreeRating, "backwater": BackwaterRating, "combined": CombinedRating}[kind](**d)


def _finite(*arrays):
    mask = np.all([np.isfinite(a) for a in arrays], axis=0)
    return [np.asarray(a)[mask] for a in arrays]


def fit_free(q, stage) -> FreeRating:
    q, stage = _finite(np.asarray(q, float), np.asarray(stage, float))
    f = lambda x, h0, a, b: h0 + a * x**b
    (h0, a, b), _ = curve_fit(f, q, stage, p0=[stage.min() - 0.1, 0.1, 0.5], maxfev=20000)
    return FreeRating(float(h0), float(a), float(b))


def fit_backwater(q, stage, lake) -> BackwaterRating:
    q, stage, lake = _finite(np.asarray(q, float), np.asarray(stage, float), np.asarray(lake, float))
    f = lambda x, c, d: x[1] + c * x[0] ** d
    (c, d), _ = curve_fit(f, np.vstack([q, lake]), stage, p0=[0.05, 0.6], bounds=([1e-6, 0.05], [10, 3]), maxfev=20000)
    return BackwaterRating(float(c), float(d))


def fit_combined(q, stage, lake) -> CombinedRating:
    q, stage, lake = _finite(np.asarray(q, float), np.asarray(stage, float), np.asarray(lake, float))
    bw = fit_backwater(q, stage, lake)
    hi = q >= np.quantile(q, 0.8)
    fr = fit_free(q[hi], stage[hi]) if hi.sum() > 10 else fit_free(q, stage)

    def f(x, c, d, h0, a, b, k):
        return CombinedRating(c, d, h0, a, b, k).predict_stage(x[0], x[1])

    p0 = [bw.c, bw.d, fr.h0, fr.a, fr.b, 5.0]
    lower = [1e-7, 0.05, 30.0, 1e-6, 0.05, 0.5]
    upper = [10.0, 3.0, 70.0, 10.0, 2.0, 50.0]
    p0 = list(np.clip(p0, np.array(lower) + 1e-9, np.array(upper) - 1e-9))
    params, _ = curve_fit(f, np.vstack([q, lake]), stage, p0=p0, bounds=(lower, upper), maxfev=50000)
    return CombinedRating(*(float(v) for v in params))
