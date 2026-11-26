"""Idea 4: rate of rise at issue time (hourly), at the gauge and upstream, as model features.

  slope6 / slope12       issue reading - gauge reading 6 / 12 h earlier (06Z / 00Z of t+1)
  up_slope6 / up_slope12 same for upstream log flow (02263800)
NaN before 2007 (allowed missing, like the morning readings).

Variants (leave-3-year-block-out, scored 2007-10..2023):
  arx      production spec + slopes in the stage model at every horizon
  arx+flow + slopes in the hybrid's flow model too
  day1     MorningAnchor (now production day 1) + slopes
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd
import mt
from shingle import features, fetch, spec, dataset
from shingle.models import anchor, arx, direct, hybrid

OUT = Path(os.path.dirname(os.path.abspath(__file__)))
D = mt.load()
hist = fetch.load_history("data")
IX = D["stage"].index


def table(s):
    s = s.dropna(); s.index = s.index.tz_convert("UTC").tz_localize(None); s = s.resample("1h").mean()
    days = pd.date_range(s.index.min().normalize(), s.index.max().normalize())
    return pd.DataFrame({k: pd.Series(s.reindex(days + pd.Timedelta(hours=k)).to_numpy(), index=days - pd.Timedelta(days=1)) for k in range(0, 13)})


G = table(hist["iv"]["gauge"]["stage_ft"])
U = np.log1p(table(hist["iv"]["upstream"]["discharge_cfs"]).clip(lower=0))
am, am_u = G[[10, 11, 12]].ffill(axis=1)[12], U[[10, 11, 12]].ffill(axis=1)[12]
EXTRA = {"slope6": am - G[6], "slope12": am - G[0], "up_slope6": am_u - U[6], "up_slope12": am_u - U[0]}
EXTRA = {k: v.reindex(IX) for k, v in EXTRA.items()}
D["stage_sofar"] = dataset.morning_mean(hist["iv"]["gauge"]["stage_ft"], IX)
SL = list(EXTRA)
_orig = features._one


def _one(data, name, horizon, index):
    if name in EXTRA:
        return EXTRA[name]
    return _orig(data, name, horizon, index)


features._one = _one
LP = dict(spec.LGBM_PARAMS, random_state=0, n_jobs=1)


def run(args):
    variant, k = args
    _, tr, held = mt.folds(mt.index())[k]
    fs = spec.feature_sets()
    if variant == "day1":
        out = {}
        for name, extra in (("base", []), ("slopes", SL)):
            m = anchor.MorningAnchor(fs["stage"] + spec.AM_STAGE + spec.DAY1_EXTRA + extra, lgbm_params=LP).fit(D, tr)
            out[name] = m.point(D, held)["h1"]
        return pd.DataFrame(out)
    stage_f = {h: fs["stage"] + (spec.AM_STAGE if h <= spec.AM_MAX_H else []) + SL for h in mt.H}
    flow_f = {h: fs["flow"] + (spec.AM_FLOW if h <= spec.AM_MAX_H else []) + (SL if variant == "arx+flow" else []) for h in mt.H}
    a = arx.ARX(stage_f, mt.H, backend="lgbm", allow_missing=spec.AM_STAGE + SL, lgbm_params=LP).fit(D, tr) if True else None
    out = {"a": a.point(D, held)}
    if variant == "arx+flow":
        b = hybrid.RatingHybrid(direct.DirectRegressor("log_q", flow_f, mt.H, backend="lgbm", allow_missing=spec.AM_FLOW + SL, lgbm_params=LP),
                                direct.DirectRegressor("toho", fs["lake"], mt.H, backend="lgbm", lgbm_params=LP), mt.H, rating_opts=spec.RATING_OPTS)
        b.fit(D, tr)
        out["b"] = b.point(D, held)
    return pd.concat(out, axis=1)


if __name__ == "__main__":
    n = len(mt.folds(mt.index()))
    ab = pd.read_parquet(mt.SP + "oof/prod_s0.parquet")
    prod = mt.blend(ab).loc["2007-10-01":]
    lines = []
    with ProcessPoolExecutor(8, mp_context=mp.get_context("fork")) as ex:
        R = {v: pd.concat(list(ex.map(run, [(v, k) for k in range(n)]))).sort_index() for v in ("day1", "arx", "arx+flow")}
    for v, r in R.items():
        r.columns = ["_".join(map(str, c)) if isinstance(c, tuple) else c for c in r.columns]
        r.to_parquet(OUT / f"i4_{v.replace('+', '_')}.parquet")
    R = {v: (r if v == "day1" else r.set_axis(pd.MultiIndex.from_tuples([tuple(c.split("_")) for c in r.columns]), axis=1)) for v, r in R.items()}
    d1 = R["day1"].loc["2007-10-01":]
    y1 = D["stage"].shift(-1).reindex(d1.index)
    ok = d1.notna().all(axis=1) & y1.notna()
    lines += ["day 1 (MorningAnchor):", f"  base {(d1.base - y1)[ok].abs().mean():.4f}  + slopes {(d1.slopes - y1)[ok].abs().mean():.4f}  "
              f"flood {(d1.base - y1)[ok & (y1 >= 59)].abs().mean():.4f} -> {(d1.slopes - y1)[ok & (y1 >= 59)].abs().mean():.4f}"]
    q = lambda c: pd.DataFrame({f"h{h}": d1.loc[ok, c] for h in mt.H})
    bt = mt.boot_diff(q("slopes"), q("base")).loc[1]
    lines += [f"  slopes vs base: {bt.pct:+.1f} %, 90 % CI {bt.lo90:+.4f}..{bt.hi90:+.4f} ft", ""]
    for v in ("arx", "arx+flow"):
        r = R[v]
        p = pd.DataFrame({f"h{h}": 0.5 * r[("a", f"h{h}")] + 0.5 * (r[("b", f"h{h}")] if v == "arx+flow" else ab[f"b_h{h}"].reindex(r.index)) for h in mt.H}).loc["2007-10-01":]
        m, mp_ = mt.metrics(p), mt.metrics(prod)
        bt = mt.boot_diff(p, prod)
        lines += [v, pd.DataFrame({"prod": mp_, v: m}).T.round(4).to_string(), bt.round(4).to_string(),
                  "flood days (obs >= 59):", mt.boot_diff(p, prod, subset="hi").round(4).to_string(),
                  "low water:", mt.boot_diff(p, prod, subset="lo").round(4).to_string(), ""]
    text = "\n".join(lines); (OUT / "i4_report.txt").write_text(text + "\n"); print(text)
