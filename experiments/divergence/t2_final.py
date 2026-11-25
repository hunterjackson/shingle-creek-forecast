"""ONE-SHOT test of the frozen T2 candidate (wet_add, days 1-3) vs production bands: 2024-01-01 .. 2026-11-12.

Frozen before running (selected on 2020-23 forecast-mode OOF, t2_bands.py):
  point model: production spec fit on <= 2023-12-31 (7-day gap), GEFS raw h1-3 / scaled h4-7 (as fc_final_bands.py)
  calibration residuals: forecast-mode OOF-by-year predictions 2020-10..2023-12 (fc_prod.parquet)
  production: day 1 |p50-stage| 5-bin conformal; day 3 E2_calmprod learned width
  wet_add: production band on dry days; on wet days (issued GEFS basin rain over leads 1..h >= 5 mm) conformal bins
           at tertile edges of the learned |residual| scale, calibrated on wet calibration days only
Evaluated at days 1 and 3 (the horizons with harness width models inside the candidate's range).
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
import numpy as np
import pandas as pd

import fc
import mt
from bands2 import TAIL_Q, EdgeConformal, pinball
from fc_bands import FEATS, scale_models
from shingle import config, evaluate, fcst, features, spec, train

OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
TEST = (pd.Timestamp("2024-01-01"), pd.Timestamp("2026-11-12"))
HS = (1, 3)
WET = 5.0
WET_Q = [1 / 3, 2 / 3]
pd.set_option("display.width", 250)

s = fc.state()
d = s["data"]
st = d["stage"]
ix_all = st.dropna().index
path = mt.SP + "fc_models/final_le2023.pkl"
if os.path.exists(path):
    m = pickle.load(open(path, "rb"))
else:
    tr = ix_all[ix_all <= pd.Timestamp(config.VAL_END) - pd.Timedelta(days=7)]
    m = spec.build_model(fc.HS).fit(d, tr)
    pickle.dump(m, open(path, "wb"))
plain = fcst.build(s["raw"], s["src"], s["tgt"], s["drv"], d)
cal_ix = fc.fc_index()
scale = fcst.fit_scale(plain["pr"], d["pr"], config.RAIN_CELL, cal_ix)
dfc = dict(d, fcst=fcst.build_mixed(s["raw"], s["src"], s["tgt"], s["drv"], d, scale, train.SCALED_FROM_H, fc.HS))
issued = s["raw"].dropna(how="all").index
te_ix = ix_all[(ix_all >= TEST[0]) & (ix_all <= TEST[1]) & ix_all.isin(issued)]
pt = m.point(dfc, te_ix)
pc = pd.read_parquet(mt.SP + "fc_prod.parquet")
sc_by_year = fc.scales()
sms = scale_models(HS)
w = s["drv"].weights
areal = pd.DataFrame({k: fcst._areal(plain["pr"][k], w) for k in fc.HS})


def X_of(index, p, dd, h):
    X = features.build(dd, FEATS, horizon=h).loc[index]
    X["dp50"] = p[f"h{h}"] - st.loc[index]
    X["adp50"] = X["dp50"].abs()
    return X


rows = []
for h in HS:
    Xc = pd.concat([X_of(pc.index[pc.index.year == y], pc,
                         dict(d, fcst=fcst.build_mixed(s["raw"], s["src"], s["tgt"], s["drv"], d, sc_by_year[y], train.SCALED_FROM_H, fc.HS)), h)
                    for y in fc.YEARS]).loc[pc.index]
    Xt = X_of(te_ix, pt, dfc, h)
    rc = st.shift(-h).reindex(pc.index) - pc[f"h{h}"]
    rt = st.shift(-h).reindex(te_ix) - pt[f"h{h}"]
    okc, okt = rc.notna(), rt.notna()
    Xc, rc, Xt, rt = Xc[okc], rc[okc], Xt[okt], rt[okt]
    yt = st.shift(-h).reindex(rt.index)
    sc_c = sms[h].predict(Xc[sms[h].feature_name_])
    sc_t = sms[h].predict(Xt[sms[h].feature_name_])
    rain_c = areal[list(range(1, h + 1))].sum(axis=1).reindex(rc.index).fillna(0.0).to_numpy()
    rain_t = areal[list(range(1, h + 1))].sum(axis=1).reindex(rt.index).fillna(0.0).to_numpy()
    wc, wt = rain_c >= WET, rain_t >= WET
    # production
    if h == 1:
        plo, phi = evaluate.BinnedConformal.fit(rc, Xc["adp50"], 5, 0.8).offsets(Xt["adp50"].to_numpy())
    else:
        cc, ct = (Xc["adp50"] < .05).to_numpy(), (Xt["adp50"] < .05).to_numpy()
        bcp = EdgeConformal.fit_edges(rc[cc], Xc["adp50"][cc], [.25, .5, .75], 0.8)
        bco = EdgeConformal.fit_edges(rc[~cc], sc_c[~cc], TAIL_Q, 0.8)
        (clo, chi), (olo, ohi) = bcp.offsets(Xt["adp50"].to_numpy()), bco.offsets(sc_t)
        plo, phi = np.where(ct, clo, olo), np.where(ct, chi, ohi)
    # wet_add
    bw = EdgeConformal.fit_edges(rc[wc], sc_c[wc], WET_Q, 0.8, min_count=30)
    wlo, whi = bw.offsets(sc_t)
    alo, ahi = np.where(wt, wlo, plo), np.where(wt, whi, phi)
    r = rt.to_numpy()
    yy = yt.to_numpy()
    subsets = {"all": np.ones(len(r), bool), "dry": ~wt, "wet 5-20": wt & (rain_t < 20), "wet>=20": rain_t >= 20, "flood": yy >= 59}
    for k, lo, hi in (("production", plo, phi), ("wet_add", alo, ahi)):
        ins = (r >= lo) & (r <= hi)
        row = {"h": h, "bands": k, "pinball": (pinball(r, lo, .1) + pinball(r, hi, .9)) / 2}
        for name, msk in subsets.items():
            row[f"n {name}"] = int(msk.sum())
            row[f"cov {name}"] = ins[msk].mean() if msk.any() else np.nan
            row[f"wid {name}"] = (hi - lo)[msk].mean() if msk.any() else np.nan
        rows.append(row)

R = pd.DataFrame(rows)
lines = ["T2 ONE-SHOT test 2024-01-01..2026-11-12: wet_add vs production bands", ""]
lines += ["Days per subset:", R[R.bands == "production"].set_index("h")[[c for c in R.columns if c.startswith("n ")]].to_string(), ""]
for kind in ("cov", "wid"):
    lines += [f"{kind}:", R.set_index(["h", "bands"])[[c for c in R.columns if c.startswith(kind + " ")]].round(3).to_string(), ""]
lines += ["pinball:", R.pivot(index="h", columns="bands", values="pinball").round(4).to_string()]
text = "\n".join(lines)
open(OUT + "t2_final_report.txt", "w").write(text + "\n")
print(text)
