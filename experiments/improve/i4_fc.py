"""Idea 4 in forecast mode (issued GEFS, leave-one-year-out 2020-10..2023-12): production blend vs the stage
model with the hourly rate-of-rise features at every horizon (i4_slopes.py 'arx'). Day 1 is the blend's
(not MorningAnchor) in both, so only days 2-7 matter here."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd
import fc, mt
from shingle import features, fetch, spec
from shingle.models import arx

OUT = Path(os.path.dirname(os.path.abspath(__file__)))
S = fc.state(); D = S["data"]
hist = fetch.load_history("data")
IX = D["stage"].index


def table(s):
    s = s.dropna(); s.index = s.index.tz_convert("UTC").tz_localize(None); s = s.resample("1h").mean()
    days = pd.date_range(s.index.min().normalize(), s.index.max().normalize())
    return pd.DataFrame({k: pd.Series(s.reindex(days + pd.Timedelta(hours=k)).to_numpy(), index=days - pd.Timedelta(days=1)) for k in range(0, 13)})


G = table(hist["iv"]["gauge"]["stage_ft"]); U = np.log1p(table(hist["iv"]["upstream"]["discharge_cfs"]).clip(lower=0))
am, am_u = G[[10, 11, 12]].ffill(axis=1)[12], U[[10, 11, 12]].ffill(axis=1)[12]
EXTRA = {k: v.reindex(IX) for k, v in {"slope6": am - G[6], "slope12": am - G[0], "up_slope6": am_u - U[6], "up_slope12": am_u - U[0]}.items()}
SL = list(EXTRA)
_orig = features._one
features._one = lambda data, name, horizon, index: EXTRA[name] if name in EXTRA else _orig(data, name, horizon, index)


def fit_year(y):
    ix = fc.all_index()
    st, en = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
    near = ((ix < st) & (ix >= st - pd.Timedelta(days=7))) | ((ix > en) & (ix <= en + pd.Timedelta(days=7)))
    tr = ix[(ix.year != y) & ~near]
    m = spec.build_model(fc.HS)
    fs = spec.feature_sets()
    m.a = arx.ARX({h: fs["stage"] + (spec.AM_STAGE if h <= spec.AM_MAX_H else []) + SL for h in fc.HS}, fc.HS, backend="lgbm",
                  allow_missing=spec.AM_STAGE + SL, lgbm_params=dict(spec.LGBM_PARAMS, n_jobs=8))
    m.day1 = None
    m.a.fit(S["data"], tr)
    return y, m


if __name__ == "__main__":
    prod_ms = fc.models()
    ms = dict(fit_year(y) for y in fc.YEARS)  # sequential: forked workers deadlocked in LightGBM threads
    for y in fc.YEARS:  # reuse production's fitted hybrid for the year: only the stage model differs
        ms[y].b = prod_ms[y].b
    sc = fc.scales()
    cand = fc.predict(S["raw"], sc, ms)
    prod = pd.read_parquet(mt.SP + "fc_prod.parquet")
    hs = fc.HS
    lines = ["forecast mode 2020-10..2023-12", pd.DataFrame({"prod": fc.metrics(prod), "slopes": fc.metrics(cand)}).T.round(4).to_string(),
             fc.boot(cand, prod).round(4).to_string(), "flood days:", fc.boot(cand, prod, subset="hi").round(4).to_string(),
             "low water:", fc.boot(cand, prod, subset="lo").round(4).to_string()]
    text = "\n".join(lines); (OUT / "i4_fc_report.txt").write_text(text + "\n"); print(text)
