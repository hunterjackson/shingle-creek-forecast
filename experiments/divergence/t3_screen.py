"""T3 screen (docs/divergence_test_plan.md): do ensemble-based hedges beat the GEFS mean on rain-forecast days?

The earlier members experiment (experiments/model_test/fc_members_report.txt) scored them over all days: the median of
member-driven stage forecasts was neutral at days 1-3, and spread-based bands lost flood coverage. The 11-25 case is a
wet-forecast day, so this screens on those days only (issued GEFS basin rain over leads 1..h >= 5 mm), 2020-10..2023-12,
10 members, OOF-by-year models from fc.py:
  prod          GEFS mean (production)
  ens_median    median over members of the member-driven stage forecasts
  rain_median   GEFS member-median rain per cell and lead, fed as the issued rain
  mean10        mean of the 10 members' rain (sanity: should be close to prod)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
import numpy as np
import pandas as pd

import fc
import fc_members
from shingle import fcst

OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
HS = (1, 2, 3)
pd.set_option("display.width", 250)

s = fc.state()
P = fc_members.run_points()
index = P["prod"].index
raws = fc_members.member_raw()
stack = np.stack([r.reindex(s["raw"].index).to_numpy() for r in raws.values()])
rain_median = pd.DataFrame(np.median(stack, 0), index=s["raw"].index, columns=s["raw"].columns)
P["rain_median"] = fc.predict(rain_median, fc.scales(), fc.models(), index)

plain = fcst.build(s["raw"], s["src"], s["tgt"], s["drv"], s["data"])
w = s["drv"].weights
areal = pd.DataFrame({k: fcst._areal(plain["pr"][k], w) for k in fc.HS}).reindex(index)
y = fc.targets(index)
rows = []
for h in HS:
    rain = areal[list(range(1, h + 1))].sum(axis=1)
    for name, m in (("all", rain.notna()), ("wet>=5", rain >= 5), ("wet>=20", rain >= 20)):
        for k in ("prod", "mean10", "ens_median", "rain_median"):
            e = (P[k][f"h{h}"] - y[f"h{h}"])[m].dropna()
            rows.append({"h": h, "subset": name, "cand": k, "n": len(e), "mae": e.abs().mean(), "bias": e.mean()})
R = pd.DataFrame(rows)
T = R.pivot_table(index=["h", "subset"], columns="cand", values=["mae", "bias"])
n = R[R.cand == "prod"].set_index(["h", "subset"])["n"]
lines = ["T3 screen: ensemble hedges on rain-forecast days, forecast-mode OOF 2020-10..2023-12 (10 members)", "",
         "n per subset:", n.to_string(), "", T.round(3).to_string(), ""]
for h in HS:
    rain = areal[list(range(1, h + 1))].sum(axis=1)
    sub = rain >= 5
    for k in ("ens_median", "rain_median"):
        b = fc.boot(P[k][sub.reindex(P[k].index, fill_value=False)], P["prod"][sub], hs=(h,))
        lines.append(f"h{h} wet>=5 {k} vs prod: " + b.round(4).to_string(header=False, index=False))
text = "\n".join(lines)
open(OUT + "t3_report.txt", "w").write(text + "\n")
print(text)
