"""GR4J daily rainfall-runoff model (Perrin, Michel & Andreassian 2003).

Four parameters:
  x1  production (soil) store capacity [mm]
  x2  groundwater exchange coefficient [mm/day] (negative = loss)
  x3  routing store capacity [mm]
  x4  unit-hydrograph time base [days]
Returns streamflow in mm/day plus internal store levels, which the hybrid
model uses as physically meaningful wetness/memory features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

CFS_TO_M3S = 0.0283168466


@dataclass(frozen=True)
class Params:
    x1: float
    x2: float
    x3: float
    x4: float


def _sh1(t: float, x4: float) -> float:
    if t <= 0:
        return 0.0
    if t < x4:
        return (t / x4) ** 2.5
    return 1.0


def _sh2(t: float, x4: float) -> float:
    if t <= 0:
        return 0.0
    if t <= x4:
        return 0.5 * (t / x4) ** 2.5
    if t < 2 * x4:
        return 1.0 - 0.5 * (2 - t / x4) ** 2.5
    return 1.0


def unit_hydrographs(x4: float) -> tuple[np.ndarray, np.ndarray]:
    n1 = int(np.ceil(x4))
    n2 = int(np.ceil(2 * x4))
    uh1 = np.array([_sh1(j, x4) - _sh1(j - 1, x4) for j in range(1, n1 + 1)])
    uh2 = np.array([_sh2(j, x4) - _sh2(j - 1, x4) for j in range(1, n2 + 1)])
    return uh1, uh2


@dataclass
class State:
    s: float  # production store [mm]
    r: float  # routing store [mm]
    q9: np.ndarray = field(default_factory=lambda: np.zeros(0))  # UH1 queue
    q1: np.ndarray = field(default_factory=lambda: np.zeros(0))  # UH2 queue

    @classmethod
    def initial(cls, params: Params, s_frac: float = 0.5, r_frac: float = 0.5) -> "State":
        uh1, uh2 = unit_hydrographs(params.x4)
        return cls(s=s_frac * params.x1, r=r_frac * params.x3, q9=np.zeros(len(uh1)), q1=np.zeros(len(uh2)))

    def total(self) -> float:
        return self.s + self.r + self.q9.sum() + self.q1.sum()


@njit(cache=True)
def _step(p, e, x1, x2, x3, uh1, uh2, s, r, q9, q1):
    """One GR4J day. Updates the UH queues q9/q1 in place; returns (q, aet, s, r, exchange)."""
    n1, n2 = uh1.shape[0], uh2.shape[0]
    if p >= e:
        pn = p - e
        tw = np.tanh(pn / x1)
        ps = x1 * (1 - (s / x1) ** 2) * tw / (1 + s / x1 * tw)
        es = 0.0
    else:
        pn = 0.0
        tw = np.tanh((e - p) / x1)
        es = s * (2 - s / x1) * tw / (1 + (1 - s / x1) * tw)
        ps = 0.0
    s = s - es + ps
    perc = s * (1 - (1 + (4.0 / 9.0 * s / x1) ** 4) ** -0.25)
    s -= perc
    pr = perc + (pn - ps)

    # unit-hydrograph queues: shift one day, add today's input, release slot 0
    for k in range(n1 - 1):
        q9[k] = q9[k + 1]
    q9[n1 - 1] = 0.0
    for k in range(n2 - 1):
        q1[k] = q1[k + 1]
    q1[n2 - 1] = 0.0
    for k in range(n1):
        q9[k] += 0.9 * pr * uh1[k]
    for k in range(n2):
        q1[k] += 0.1 * pr * uh2[k]
    q9_now, q1_now = q9[0], q1[0]
    q9[0] = 0.0
    q1[0] = 0.0

    f = x2 * (r / x3) ** 3.5
    r_new = r + q9_now + f
    f_r = f if r_new >= 0 else -(r + q9_now)
    r = max(0.0, r_new)
    qr = r * (1 - (1 + (r / x3) ** 4) ** -0.25)
    r -= qr
    qd_raw = q1_now + f
    qd = max(0.0, qd_raw)
    f_d = f if qd_raw >= 0 else -q1_now
    return qr + qd, es + min(p, e), s, r, f_r + f_d


@njit(cache=True)
def _core(P, E, x1, x2, x3, uh1, uh2, s, r, q9, q1):
    n = P.shape[0]
    q_out = np.empty(n)
    aet = np.empty(n)
    s_out = np.empty(n)
    r_out = np.empty(n)
    exch = np.empty(n)
    for t in range(n):
        q_out[t], aet[t], s, r, exch[t] = _step(P[t], E[t], x1, x2, x3, uh1, uh2, s, r, q9, q1)
        s_out[t] = s
        r_out[t] = r
    return q_out, aet, s_out, r_out, exch, s, r


@njit(cache=True)
def _forecast(P, E, Pf, Ef, x1, x2, x3, uh1, uh2, s, r, q9, q1):
    n, K = Pf.shape
    out = np.full((n, K), np.nan)
    fq9 = np.empty_like(q9)
    fq1 = np.empty_like(q1)
    for t in range(n):
        _q, _a, s, r, _x = _step(P[t], E[t], x1, x2, x3, uh1, uh2, s, r, q9, q1)
        ok = True
        for k in range(K):
            if np.isnan(Pf[t, k]) or np.isnan(Ef[t, k]):
                ok = False
        if not ok:
            continue
        fs, fr = s, r
        fq9[:] = q9
        fq1[:] = q1
        for k in range(K):
            out[t, k], _a, fs, fr, _x = _step(Pf[t, k], Ef[t, k], x1, x2, x3, uh1, uh2, fs, fr, fq9, fq1)
    return out


def run(P: np.ndarray, E: np.ndarray, params: Params, state: State | None = None) -> dict:
    uh1, uh2 = unit_hydrographs(params.x4)
    st = state or State.initial(params)
    q9 = st.q9.astype(float).copy() if len(st.q9) == len(uh1) else np.zeros(len(uh1))
    q1 = st.q1.astype(float).copy() if len(st.q1) == len(uh2) else np.zeros(len(uh2))
    initial_total = st.s + st.r + q9.sum() + q1.sum()

    q, aet, s_out, r_out, exch, s, r = _core(
        np.ascontiguousarray(P, dtype=float), np.ascontiguousarray(E, dtype=float),
        float(params.x1), float(params.x2), float(params.x3), uh1, uh2, float(st.s), float(st.r), q9, q1,
    )
    final = State(s=s, r=r, q9=q9, q1=q1)
    return {
        "q": q,
        "aet": aet,
        "s": s_out,
        "r": r_out,
        "exchange": exch,
        "state": final,
        "storage_total": np.array([final.total()]),
        "storage_total_initial": initial_total,
    }


def forecast(P: np.ndarray, E: np.ndarray, Pf: np.ndarray, Ef: np.ndarray, params: Params, state: State | None = None) -> np.ndarray:
    """Simulate history P/E; after each day t restart from the model state with the
    K-day forecast inputs Pf[t], Ef[t] (row t = days t+1..t+K). Returns flow (mm/d)
    of shape (n, K); rows with any NaN forecast input are NaN."""
    uh1, uh2 = unit_hydrographs(params.x4)
    st = state or State.initial(params)
    q9 = st.q9.astype(float).copy() if len(st.q9) == len(uh1) else np.zeros(len(uh1))
    q1 = st.q1.astype(float).copy() if len(st.q1) == len(uh2) else np.zeros(len(uh2))
    return _forecast(
        np.ascontiguousarray(P, dtype=float), np.ascontiguousarray(E, dtype=float),
        np.ascontiguousarray(Pf, dtype=float), np.ascontiguousarray(Ef, dtype=float),
        float(params.x1), float(params.x2), float(params.x3), uh1, uh2, float(st.s), float(st.r), q9, q1,
    )


def cfs_to_mm_day(q_cfs: np.ndarray, area_km2: float) -> np.ndarray:
    return q_cfs * CFS_TO_M3S * 86400 / (area_km2 * 1e6) * 1000


def mm_day_to_cfs(q_mm: np.ndarray, area_km2: float) -> np.ndarray:
    return q_mm / 1000 * (area_km2 * 1e6) / 86400 / CFS_TO_M3S


def nse(sim: np.ndarray, obs: np.ndarray) -> float:
    m = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[m], obs[m]
    return float(1 - ((sim - obs) ** 2).sum() / ((obs - obs.mean()) ** 2).sum())


def kge(sim: np.ndarray, obs: np.ndarray) -> float:
    m = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[m], obs[m]
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


BOUNDS = [(10.0, 2500.0), (-10.0, 5.0), (5.0, 1000.0), (0.5, 10.0)]


@dataclass(frozen=True)
class Calibration:
    params: Params
    score: float
    objective: str


def calibrate(
    P: np.ndarray,
    E: np.ndarray,
    q_obs: np.ndarray,
    warmup: int = 365,
    objective: str = "nse",
    bounds=BOUNDS,
    seed: int = 0,
    maxiter: int = 200,
) -> Calibration:
    """Global search (differential evolution) maximizing NSE or KGE after warmup.

    objective: 'nse', 'kge', or 'nse_sqrt' (NSE on sqrt flows - balances low/high flows).
    """
    from scipy.optimize import differential_evolution

    obs = np.asarray(q_obs, float)[warmup:]

    def score(x):
        sim = run(P, E, Params(*x))["q"][warmup:]
        if objective == "kge":
            return kge(sim, obs)
        if objective == "nse_sqrt":
            return nse(np.sqrt(sim), np.sqrt(obs))
        return nse(sim, obs)

    res = differential_evolution(lambda x: -score(x), bounds, seed=seed, maxiter=maxiter, tol=1e-7, polish=True)
    return Calibration(Params(*res.x), -float(res.fun), objective)
