"""Forecast-mode harness (issued GEFS rain), GEFS era inside the validation window: 2020-10-01 .. 2023-12-31.

Mirrors train.train(): GR4J calibrated <= 2021 (as the OOF harness), per-lead GEFS scale, raw rain h1-3 / scaled h4-7.
Point models are out-of-fold by calendar year: for year Y the production spec is fit on all of 1979-2023 except Y
(7-day gap) on observed rain, then predicts Y's issue days with issued forecasts. The per-lead scale is fitted
leave-one-year-out too. The test period (2024+) is never touched.

prep() caches: fc_data.pkl (data with gr4j, driver, hist cells, GEFS mean raw/src) and fc_models/{Y}.pkl.
"""
from __future__ import annotations

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import mt  # noqa: F401  (chdir to repo)
from shingle import config, dataset, fcst, fetch, gr4j_driver, pipeline, spec, train
from shingle.models.base import HORIZONS

START, END = pd.Timestamp("2020-10-01"), pd.Timestamp(config.VAL_END)
YEARS = (2020, 2021, 2022, 2023)
SP = mt.SP
PKL = SP + "fc_data.pkl"
HS = tuple(HORIZONS)
_S = None


def state() -> dict:
    global _S
    if _S is not None:
        return _S
    if os.path.exists(PKL):
        _S = pickle.load(open(PKL, "rb"))
        return _S
    hist = fetch.load_history("data")
    base = dataset.build_data(hist)
    idx = base["stage"].dropna().index
    w = hist["cells"]["basin_frac"][hist["cells"]["basin_frac"] > 0].to_dict()
    drv = gr4j_driver.GR4JDriver(w, maxiter=100).fit(base, idx[idx <= pd.Timestamp(config.TRAIN_END) - pd.Timedelta(days=7)])
    data = dict(base, gr4j=drv.simulate(base))
    raw, src = pipeline.gefs_raw("data")
    _S = {"data": data, "drv": drv, "tgt": hist["cells"][["lat", "lon"]], "raw": raw, "src": src}
    pickle.dump(_S, open(PKL, "wb"))
    return _S


def all_index() -> pd.DatetimeIndex:
    d = state()["data"]
    return d["stage"].dropna().loc[:END].index


def fc_index(raw=None) -> pd.DatetimeIndex:
    s = state()
    raw = s["raw"] if raw is None else raw
    ix = all_index()
    issued = raw.dropna(how="all").index
    return ix[(ix >= START) & ix.isin(issued)]


def scales(raw=None) -> dict[int, dict[int, float]]:
    """Leave-one-year-out per-lead scale, from the GEFS mean (the production quantity)."""
    s = state()
    raw = s["raw"] if raw is None else raw
    plain = fcst.build(raw, s["src"], s["tgt"], s["drv"], s["data"])
    ix = fc_index()
    return {y: fcst.fit_scale(plain["pr"], s["data"]["pr"], config.RAIN_CELL, ix[ix.year != y]) for y in YEARS}


def _fit_year(y):
    s = state()
    ix = all_index()
    start = pd.Timestamp(f"{y}-01-01")
    end = pd.Timestamp(f"{y}-12-31")
    near = ((ix < start) & (ix >= start - pd.Timedelta(days=7))) | ((ix > end) & (ix <= end + pd.Timedelta(days=7)))
    tr = ix[(ix.year != y) & ~near]
    m = spec.build_model(HS).fit(s["data"], tr)
    pickle.dump(m, open(SP + f"fc_models/{y}.pkl", "wb"))
    return y


def models() -> dict:
    os.makedirs(SP + "fc_models", exist_ok=True)
    todo = [y for y in YEARS if not os.path.exists(SP + f"fc_models/{y}.pkl")]
    if todo:
        state()
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(len(todo), mp_context=mp.get_context("fork")) as ex:
            list(ex.map(_fit_year, todo))
    return {y: pickle.load(open(SP + f"fc_models/{y}.pkl", "rb")) for y in YEARS}


def predict(raw: pd.DataFrame, scale_by_year: dict, ms: dict, index: pd.DatetimeIndex | None = None, mixed: bool = True) -> pd.DataFrame:
    """OOF-by-year forecast-mode point predictions (h1..h7) for issued rain `raw` (same layout as GEFS mean)."""
    s = state()
    index = fc_index() if index is None else index
    parts = []
    for y in YEARS:
        held = index[index.year == y]
        if len(held) == 0:
            continue
        if mixed:
            f = fcst.build_mixed(raw, s["src"], s["tgt"], s["drv"], s["data"], scale_by_year[y], train.SCALED_FROM_H, HS)
        else:
            f = fcst.build(raw, s["src"], s["tgt"], s["drv"], s["data"], scale=scale_by_year[y])
        parts.append(ms[y].point(dict(s["data"], fcst=f), held))
    return pd.concat(parts).sort_index()


def targets(index) -> pd.DataFrame:
    st = state()["data"]["stage"]
    return pd.DataFrame({f"h{h}": st.shift(-h).reindex(index) for h in HS}, index=index)


def metrics(p: pd.DataFrame, hs=HS) -> dict:
    y = targets(p.index)
    out = {}
    for h in hs:
        e = p[f"h{h}"] - y[f"h{h}"]
        ok = e.notna()
        e, yy = e[ok], y[f"h{h}"][ok]
        out[f"mae{h}"] = e.abs().mean()
        out[f"hi{h}"] = e[yy >= 59].abs().mean()
        out[f"lo{h}"] = e[yy <= mt.LOW].abs().mean()
    return out


def boot(pc: pd.DataFrame, pp: pd.DataFrame, n=2000, block=60, seed=0, subset=None, hs=HS) -> pd.DataFrame:
    """Paired moving-block bootstrap of MAE(cand) - MAE(prod), 90 % CI (60-day blocks: shorter record)."""
    rng = np.random.default_rng(seed)
    y = targets(pp.index)
    rows = []
    for h in hs:
        ec, ep = (pc[f"h{h}"] - y[f"h{h}"]).abs(), (pp[f"h{h}"] - y[f"h{h}"]).abs()
        ok = ec.notna() & ep.notna()
        if subset == "hi":
            ok &= y[f"h{h}"] >= 59
        if subset == "lo":
            ok &= y[f"h{h}"] <= mt.LOW
        dc, dp = ec[ok].to_numpy(), ep[ok].to_numpy()
        m = len(dc)
        if m == 0:
            continue
        b = min(block, m)
        nb = int(np.ceil(m / b))
        diffs = np.empty(n)
        for i in range(n):
            st = rng.integers(0, m - b + 1, nb)
            ii = (st[:, None] + np.arange(b)).ravel()[:m]
            diffs[i] = dc[ii].mean() - dp[ii].mean()
        rows.append({"h": h, "n": m, "prod": dp.mean(), "cand": dc.mean(), "pct": 100 * (dc.mean() / dp.mean() - 1),
                     "lo90": np.quantile(diffs, .05), "hi90": np.quantile(diffs, .95)})
    return pd.DataFrame(rows).set_index("h")


if __name__ == "__main__":
    import time

    t = time.time()
    s = state()
    print("state", f"{time.time() - t:.0f}s", flush=True)
    ms = models()
    print("models", f"{time.time() - t:.0f}s", flush=True)
    sc = scales()
    p = predict(s["raw"], sc, ms)
    p.to_parquet(SP + "fc_prod.parquet")
    print({k: round(v, 3) for k, v in metrics(p).items()})
    print("scales", {y: {k: round(v, 2) for k, v in d.items()} for y, d in sc.items()})
    print(f"{time.time() - t:.0f}s")
