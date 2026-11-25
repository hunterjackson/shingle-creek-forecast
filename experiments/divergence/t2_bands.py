"""T2 (docs/divergence_test_plan.md): a likely range that widens when rain is forecast.

T1: on forecast-mode OOF 2020-10..2023-12 the day-1 band covers 64 % of days with 5-20 mm of GEFS lead-1 basin rain
and 23 % of days with >= 20 mm (dry days 81 %).

Bands, all calibrated leave-one-year-out on the forecast-mode residuals (fc_prod.parquet), horizons 1/3/5/7:
  prod       production: day 1 |p50-stage| 5-bin conformal; h>=2 learned width (E2_calmprod = shingle.bands)
  E2         learned width at every horizon (incl. day 1)
  wet_scale  Mondrian split on forecast rain: wet = issued GEFS basin rain over leads 1..h >= WET mm. Dry days keep
             the production band (recalibrated on dry days only); wet days get their own bins on the learned scale.
  wet_rain   as wet_scale, wet-day bins on the forecast rain amount instead of the learned scale

Learned |residual| scale models: LightGBM on perfect-rain OOF residuals 1979-2019 (fc_bands.scale_models), so they
never saw 2020-23.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
import numpy as np
import pandas as pd

import fc
import mt
from bands2 import TAIL_Q, EdgeConformal, pinball
from fc_bands import fc_features, scale_models
from shingle import evaluate, fcst

OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
HS = (1, 3, 5, 7)
WET = 5.0
WET_Q = [1 / 3, 2 / 3]
pd.set_option("display.width", 250)

s = fc.state()
d = s["data"]
p = pd.read_parquet(mt.SP + "fc_prod.parquet")
sc = fc.scales()
sms = scale_models(HS)
y = fc.targets(p.index)
plain = fcst.build(s["raw"], s["src"], s["tgt"], s["drv"], d)
w = s["drv"].weights
areal = pd.DataFrame({k: fcst._areal(plain["pr"][k], w) for k in fc.HS}).reindex(p.index)


def prod_band(h, r, X, sc_, tr, te):
    """Production band for horizon h, fit on rows tr, applied to te (boolean arrays)."""
    if h == 1:
        bc = evaluate.BinnedConformal.fit(r[tr], X["adp50"][tr], 5, 0.8)
        return bc.offsets(X["adp50"][te].to_numpy())
    return e2_band(r, X, sc_, tr, te)


def e2_band(r, X, sc_, tr, te):
    a = X["adp50"]
    ctr, cte = tr & (a < .05), (a[te] < .05).to_numpy()
    bcc = EdgeConformal.fit_edges(r[ctr], a[ctr], [.25, .5, .75], 0.8)
    bco = EdgeConformal.fit_edges(r[tr & ~(a < .05)], sc_[tr & ~(a < .05)], TAIL_Q, 0.8)
    (clo, chi), (olo, ohi) = bcc.offsets(a[te].to_numpy()), bco.offsets(sc_[te].to_numpy())
    return np.where(cte, clo, olo), np.where(cte, chi, ohi)


rows, events = [], []
for h in HS:
    X = fc_features(p, h, sc)
    r = y[f"h{h}"] - p[f"h{h}"]
    ok = r.notna()
    X, r = X[ok], r[ok]
    yy = y[f"h{h}"][ok]
    sc_ = pd.Series(sms[h].predict(X[sms[h].feature_name_]), index=X.index)
    rain = areal[list(range(1, h + 1))].sum(axis=1, min_count=h).reindex(X.index).fillna(0.0)
    wet = rain >= WET
    LO = {k: pd.Series(np.nan, index=r.index) for k in ("prod", "E2", "wet_scale", "wet_rain", "wet_add")}
    HI = {k: pd.Series(np.nan, index=r.index) for k in LO}
    for yr in fc.YEARS:
        te = r.index.year == yr
        tr = ~te
        LO["prod"][te], HI["prod"][te] = prod_band(h, r, X, sc_, tr, te)
        LO["E2"][te], HI["E2"][te] = e2_band(r, X, sc_, tr, te)
        for k, var in (("wet_scale", sc_), ("wet_rain", rain)):
            ted, tew = te & ~wet.to_numpy(), te & wet.to_numpy()
            LO[k][ted], HI[k][ted] = prod_band(h, r, X, sc_, tr & ~wet.to_numpy(), ted)
            bw = EdgeConformal.fit_edges(r[tr & wet.to_numpy()], var[tr & wet.to_numpy()], WET_Q, 0.8, min_count=30)
            LO[k][tew], HI[k][tew] = bw.offsets(var[tew].to_numpy())
        tew = te & wet.to_numpy()
        LO["wet_add"][te], HI["wet_add"][te] = LO["prod"][te], HI["prod"][te]
        LO["wet_add"][tew], HI["wet_add"][tew] = LO["wet_scale"][tew], HI["wet_scale"][tew]
    subsets = {"all": pd.Series(True, index=r.index), "dry": ~wet, f"wet {WET:.0f}-20": wet & (rain < 20), "wet>=20": rain >= 20,
               "flood": yy >= 59}
    for k in LO:
        lo, hi = LO[k], HI[k]
        ins = (r >= lo) & (r <= hi)
        row = {"h": h, "bands": k, "pinball": (pinball(r.to_numpy(), lo.to_numpy(), .1) + pinball(r.to_numpy(), hi.to_numpy(), .9)) / 2}
        for name, m in subsets.items():
            row[f"n {name}"] = int(m.sum())
            row[f"cov {name}"] = ins[m].mean()
            row[f"wid {name}"] = (hi - lo)[m].mean()
        rows.append(row)
    pd.DataFrame({"rain": rain, "r": r, **{f"{k}_lo": LO[k] for k in LO}, **{f"{k}_hi": HI[k] for k in LO}}).to_parquet(OUT + f"t2_h{h}.parquet")

R = pd.DataFrame(rows)
lines = [f"T2: rain-aware likely range, forecast-mode OOF 2020-10..2023-12, LOYO calibration, wet = GEFS basin rain(1..h) >= {WET} mm", ""]
ns = R[R.bands == "prod"].set_index("h")[[c for c in R.columns if c.startswith("n ")]]
lines += ["Days per subset:", ns.to_string(), ""]
for kind in ("cov", "wid"):
    cols = [c for c in R.columns if c.startswith(kind + " ")]
    lines += [f"{kind}:", R.set_index(["h", "bands"])[cols].round(3).to_string(), ""]
P = R.pivot(index="h", columns="bands", values="pinball")
lines += ["pinball (lower is better):", P.round(4).to_string(), ""]
text = "\n".join(lines)
open(OUT + "t2_report.txt", "w").write(text + "\n")
print(text)
