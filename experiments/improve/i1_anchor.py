"""Idea 1 (docs/project_review.md): anchor day 1 on the morning reading. Which form wins, out-of-fold?

All candidates use only inputs production already has at issue (hourly gauge values labelled <= 12Z of t+1).
Leave-3-year-block-out folds (mt.folds), scored 2007-10..2023-12 (the test period is not touched).

  prod        production blend, day 1 (oof/prod_s0)
  lin         am + b1*(prod - am) + b2*(sofar - am), b fitted on the other folds (the review's correction)
  arx_am      stage tree (production features + sofar_d) on target y - am, rows with a morning reading only
  arx_am_all  same, all rows: level = am where present, else day-t mean
  hyb_am      rating hybrid's day 1 anchored: am + (b_h1 - stage), i.e. shift its change onto the morning reading
  mix         0.5 arx_am + 0.5 lin  etc. (reported below)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd
import mt
from shingle import features, fetch, spec

OUT = Path(os.path.dirname(os.path.abspath(__file__)))
D = mt.load()
hist = fetch.load_history("data")
s = hist["iv"]["gauge"]["stage_ft"].dropna(); s.index = s.index.tz_convert("UTC").tz_localize(None); s = s.resample("1h").mean()
days = pd.date_range(s.index.min().normalize(), s.index.max().normalize())
G = pd.DataFrame({k: pd.Series(s.reindex(days + pd.Timedelta(hours=k)).to_numpy(), index=days - pd.Timedelta(days=1)) for k in range(5, 13)})
IX = D["stage"].index
AM = D["stage_am"]  # production's issue reading
SOFAR = G.mean(axis=1).reindex(IX)  # mean of 05Z..12Z of t+1 (target day's first 8 h)
SLOPE3 = (G[12] - G[9]).reindex(IX)
STAGE = D["stage"]
Y = STAGE.shift(-1)
FEATS = spec.feature_sets()["stage"] + spec.AM_STAGE
XB = features.build(D, FEATS, horizon=1)
XB["sofar_d"] = SOFAR - AM
XB["slope3"] = SLOPE3
LP = dict(spec.LGBM_PARAMS, random_state=0, n_jobs=1, verbose=-1)


def tree(Xtr, ytr, obj="regression"):
    import lightgbm as lgb
    return lgb.LGBMRegressor(**dict(LP, objective=obj)).fit(Xtr, ytr)


def run(k):
    _, tr, held = mt.folds(mt.index())[k]
    out = pd.DataFrame(index=held)
    for name, cols, rows_am, obj in [("arx_am", FEATS + ["sofar_d"], True, "regression"),
                                     ("arx_am_slope", FEATS + ["sofar_d", "slope3"], True, "regression"),
                                     ("arx_am_l1", FEATS + ["sofar_d"], True, "l1"),
                                     ("arx_am_all", FEATS + ["sofar_d"], False, "regression")]:
        lev = AM if rows_am else AM.fillna(STAGE)
        X, dy = XB[cols], Y - lev
        ok = dy.loc[tr].notna() & X.loc[tr, [c for c in cols if c not in spec.AM_STAGE + ["sofar_d", "slope3"]]].notna().all(axis=1)
        if rows_am:
            ok &= AM.loc[tr].notna() & SOFAR.loc[tr].notna()
        m = tree(X.loc[tr][ok].to_numpy(), dy.loc[tr][ok].to_numpy(), obj)
        out[name] = lev.loc[held] + m.predict(X.loc[held].to_numpy())
    return out


if __name__ == "__main__":
    n = len(mt.folds(mt.index()))
    with ProcessPoolExecutor(8, mp_context=mp.get_context("fork")) as ex:
        P = pd.concat(list(ex.map(run, range(n)))).sort_index()
    ab = pd.read_parquet(mt.SP + "oof/prod_s0.parquet")
    P["prod"] = 0.5 * ab.a_h1 + 0.5 * ab.b_h1
    P["arx_prod"] = ab.a_h1
    P["hyb_am"] = AM + (ab.b_h1 - STAGE)
    P["arx_am+hyb_am"] = 0.5 * P.arx_am + 0.5 * P.hyb_am
    P["arx_am+hyb"] = 0.5 * P.arx_am + 0.5 * ab.b_h1
    P["persist_am"] = AM
    idx = P.index[(P.index >= "2007-10-01") & AM.reindex(P.index).notna().to_numpy() & SOFAR.reindex(P.index).notna().to_numpy()
                  & Y.reindex(P.index).notna().to_numpy() & P["prod"].notna().to_numpy()]
    P = P.loc[idx]
    y, am, sf = Y[idx], AM[idx], SOFAR[idx]
    # linear correction, block-CV on 3-year blocks
    lin = pd.Series(np.nan, index=idx); yrs = sorted(set(idx.year))
    for base in ["prod", "arx_am"]:
        lin = pd.Series(np.nan, index=idx)
        X = np.column_stack([np.ones(len(idx)), (P[base] - am).to_numpy(), (sf - am).to_numpy()])
        for i in range(0, len(yrs), 3):
            te = idx.year.isin(yrs[i:i + 3])
            b, *_ = np.linalg.lstsq(X[~te], (y - am)[~te].to_numpy(), rcond=None)
            lin[te] = am[te] + X[te] @ b
        P[f"lin({base})"] = lin
    P.to_parquet(OUT / "i1_oof.parquet")
    st = STAGE[idx]
    mv, fl, lo = (y - st).abs() >= 0.10, y >= 59, y <= mt.LOW
    rows = []
    for c in P.columns:
        e = P[c] - y
        bt = mt.boot_diff(pd.DataFrame({"h1": P[c], "h3": P[c], "h5": P[c], "h7": P[c]}),
                          pd.DataFrame({"h1": P["prod"], "h3": P["prod"], "h5": P["prod"], "h7": P["prod"]})).loc[1]
        rows.append({"model": c, "MAE": e.abs().mean(), "vs prod %": 100 * (e.abs().mean() / (P["prod"] - y).abs().mean() - 1),
                     "lo90": bt.lo90, "hi90": bt.hi90, "bias": e.mean(), f"moving (n={int(mv.sum())})": e[mv].abs().mean(),
                     f"flood (n={int(fl.sum())})": e[fl].abs().mean(), f"low (n={int(lo.sum())})": e[lo].abs().mean(), "p95|e|": e.abs().quantile(.95)})
    T = pd.DataFrame(rows).set_index("model").sort_values("MAE")
    text = f"Idea 1: day-1 anchoring, {len(idx)} issue days 2007-10..2023-12, leave-3-year-block-out (MAE ft)\n\n" + T.round(4).to_string()
    (OUT / "i1_report.txt").write_text(text + "\n"); print(text)
