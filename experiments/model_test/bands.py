"""Candidate E: likely-range model vs production bands, on production OOF point forecasts (1979-2023).

Production: BinnedConformal on |p50 - stage(t)| (5 bins), calibrated on the other folds' OOF residuals.
E1: LightGBM quantile regression (10th/90th pct of residual) on state + forecast-change features, trained on
    other folds' OOF residuals, then split-conformal corrected (CQR) on a held-back slice of those folds.
E2: production bins on a different scale: |p50-stage| + forecast rain + GR4J projected change (2-D bins via
    LightGBM-predicted |residual| as the scale).
Pinball loss (q=.1,.9) averaged; coverage overall and on flood days; mean width overall / flood days.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb
import mt
from shingle import evaluate, features

d = mt.load()
oof = mt.blend(pd.read_parquet(mt.SP + "oof/prod_s0.parquet"))
ix = oof.index
y = mt.targets(ix)
C = mt.CELL

def feats(h):
    X = features.build(d, ["stage", "dstage_1", "log_q", "toho_dev", f"pr_nextH__{C}", f"pr_past3__{C}", "gr4j_dH", "gr4j_s", "doy_sin", "doy_cos"], horizon=h).loc[ix]
    X["dp50"] = oof[f"h{h}"] - d["stage"].loc[ix]
    X["adp50"] = X["dp50"].abs()
    return X

def pinball(y, q, a):
    e = y - q
    return np.mean(np.maximum(a * e, (a - 1) * e))

QP = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50, verbose=-1, n_jobs=8)
rows = []
for h in mt.H:
    X = feats(h)
    r = y[f"h{h}"] - oof[f"h{h}"]
    ok = r.notna() & X["adp50"].notna()
    X, r = X[ok], r[ok]
    yy = y[f"h{h}"][ok]
    out = {k: pd.Series(np.nan, index=r.index) for k in ("p_lo", "p_hi", "e_lo", "e_hi", "e2_lo", "e2_hi")}
    for blk, _, held in mt.folds(ix):
        te = r.index[r.index.isin(held)]
        tr = r.index[~r.index.isin(held)]
        # production
        bc = evaluate.BinnedConformal.fit(r.loc[tr], X.loc[tr, "adp50"], 5, 0.8)
        lo, hi = bc.offsets(X.loc[te, "adp50"].to_numpy())
        out["p_lo"][te], out["p_hi"][te] = lo, hi
        # E1: CQR. fit quantile models on 2/3 of training years, conformalize on the remaining 1/3
        yrs = np.array(sorted(set(tr.year)))
        cal_y = yrs[1::3]
        fit_ix, cal_ix = tr[~tr.year.isin(cal_y)], tr[tr.year.isin(cal_y)]
        ql = lgb.LGBMRegressor(objective="quantile", alpha=0.1, **QP).fit(X.loc[fit_ix], r.loc[fit_ix])
        qh = lgb.LGBMRegressor(objective="quantile", alpha=0.9, **QP).fit(X.loc[fit_ix], r.loc[fit_ix])
        cl, ch = ql.predict(X.loc[cal_ix]), qh.predict(X.loc[cal_ix])
        sc = np.maximum(cl - r.loc[cal_ix].to_numpy(), r.loc[cal_ix].to_numpy() - ch)
        qhat = np.quantile(sc, 0.8 * (1 + 1 / len(sc)))
        out["e_lo"][te], out["e_hi"][te] = ql.predict(X.loc[te]) - qhat, qh.predict(X.loc[te]) + qhat
        # E2: binned conformal on a learned |residual| scale
        sm = lgb.LGBMRegressor(objective="l1", **QP).fit(X.loc[fit_ix], r.loc[fit_ix].abs())
        sc_tr = pd.Series(sm.predict(X.loc[tr]), index=tr)
        bc2 = evaluate.BinnedConformal.fit(r.loc[tr], sc_tr, 8, 0.8)
        lo2, hi2 = bc2.offsets(sm.predict(X.loc[te]))
        out["e2_lo"][te], out["e2_hi"][te] = lo2, hi2
    fl = yy >= 59
    for tag, lo, hi in (("production", out["p_lo"], out["p_hi"]), ("E1_cqr", out["e_lo"], out["e_hi"]), ("E2_learned_scale", out["e2_lo"], out["e2_hi"])):
        inside = (r >= lo) & (r <= hi)
        rows.append({"h": h, "bands": tag, "cov": inside.mean(), "cov_flood": inside[fl].mean(), "n_flood": int(fl.sum()),
                     "width": (hi - lo).mean(), "width_flood": (hi - lo)[fl].mean(), "width_calm": (hi - lo)[X["adp50"] < 0.05].mean(),
                     "pinball": (pinball(r, lo, 0.1) + pinball(r, hi, 0.9)) / 2,
                     "pinball_flood": (pinball(r[fl], lo[fl], 0.1) + pinball(r[fl], hi[fl], 0.9)) / 2})
    # per-event flood coverage (events >= 59 in 1979-2023), production vs E1
R = pd.DataFrame(rows)
R.to_csv(mt.SP + "stage1_bands.csv", index=False)
pd.set_option("display.width", 200)
print(R.set_index(["h", "bands"]).round(3).to_string())
