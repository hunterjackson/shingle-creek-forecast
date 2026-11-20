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


def from_dict(d: dict) -> FreeRating | BackwaterRating:
    d = dict(d)
    kind = d.pop("kind")
    return {"free": FreeRating, "backwater": BackwaterRating}[kind](**d)


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
