"""'Train on forecast data', cheap version: a small residual corrector learned on forecast-mode errors.

The production models learn from observed rain, so issued rain is treated as truth. Here a low-capacity model learns
the forecast-mode residual (obs - p50) of the production blend from what was known at issue time, so it can learn how
far to trust the rain forecast. Leave-one-year-out over 2020-10..2023-12 (forecast-mode OOF predictions, fc_prod).

Features per horizon h: issued GEFS basin rain lead 1, sum of leads 1..h, p50 - stage(t), stage(t), dstage_1,
toho_dev, observed basin rain over t-2..t, doy sin/cos.
  ridge    standardised ridge (alpha 10)
  lgbm     LightGBM, 7 leaves, min 40 per leaf, 150 trees at 0.03, L1 objective
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
import numpy as np
import pandas as pd

import fc
import mt
from shingle import features, fcst

OUT = Path(os.path.dirname(os.path.abspath(__file__)))
pd.set_option("display.width", 250)
s = fc.state()
d, w = s["data"], s["drv"].weights
p = pd.read_parquet(mt.SP + "fc_prod.parquet")
ix = p.index
st = d["stage"].reindex(ix)
plain = fcst.build(s["raw"], s["src"], s["tgt"], s["drv"], d)["pr"]
lead = pd.DataFrame({k: fcst._areal(plain[k], w) for k in fc.HS}).reindex(ix)
past3 = fcst._areal(d["pr"], w).rolling(3).sum().reindex(ix)
base = features.build(d, ["dstage_1", "toho_dev", "doy_sin", "doy_cos"]).reindex(ix)
yt = fc.targets(ix)


def X_for(h):
    X = base.copy()
    X["lead1"] = np.log1p(lead[1])
    X["sum1h"] = np.log1p(lead[list(range(1, h + 1))].sum(axis=1))
    X["dp50"] = p[f"h{h}"] - st
    X["stage"] = st
    X["past3"] = np.log1p(past3)
    return X


def fit_predict(kind, Xtr, rtr, Xte):
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xtr, rtr)
    else:
        import lightgbm as lgb

        m = lgb.LGBMRegressor(objective="l1", num_leaves=7, min_child_samples=40, n_estimators=150, learning_rate=0.03, verbose=-1).fit(Xtr, rtr)
    return m.predict(Xte)


preds = {"prod": p}
for kind in ("ridge", "lgbm"):
    out = pd.DataFrame(index=ix, columns=p.columns, dtype=float)
    for h in fc.HS:
        X = X_for(h)
        r = yt[f"h{h}"] - p[f"h{h}"]
        ok = X.notna().all(axis=1) & r.notna()
        out[f"h{h}"] = p[f"h{h}"]
        for y in fc.YEARS:
            te = (ix.year == y) & X.notna().all(axis=1).to_numpy()
            tr = ok.to_numpy() & (ix.year != y)
            if te.any():
                out.loc[te, f"h{h}"] = p.loc[te, f"h{h}"] + fit_predict(kind, X[tr], r[tr], X[te])
    preds[kind] = out

g1 = lead[1]
o1 = fcst._areal(d["pr"], w).shift(-1).reindex(ix)
rows = []
for h in (1, 2, 3, 4, 7):
    for sub, m in (("all", pd.Series(True, index=ix)), ("GEFS lead1 >= 5", g1 >= 5), ("heavy (G or O >= 20)", (g1 >= 20) | (o1 >= 20)),
                   ("GEFS busts", (g1 >= 20) & (o1 < 10)), ("flood obs>=59", yt[f"h{h}"] >= 59)):
        for k, pr in preds.items():
            e = (pr[f"h{h}"] - yt[f"h{h}"])[m].dropna()
            rows.append({"h": h, "subset": sub, "src": k, "n": len(e), "mae": e.abs().mean()})
R = pd.DataFrame(rows)
M = R.pivot_table(index=["subset", "h"], columns="src", values="mae")[list(preds)]
N = R[R.src == "prod"].set_index(["subset", "h"])["n"]
lines = ["Residual corrector on forecast-mode errors, LOYO 2020-10..2023-12", "", "MAE:", M.assign(n=N).round(4).to_string(), "",
         "change vs prod (%):", (100 * (M.div(M["prod"], axis=0) - 1)).drop(columns="prod").round(1).to_string(), ""]
for k in ("ridge", "lgbm"):
    lines += [f"{k} vs prod, all days:", fc.boot(preds[k], p, hs=(1, 2, 3, 4, 7)).round(4).to_string(), ""]
text = "\n".join(lines)
(OUT / "corrector_screen_report.txt").write_text(text + "\n")
print(text)
