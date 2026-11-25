"""T1 (docs/divergence_test_plan.md): forecast skill on issue days when GEFS forecasts heavy rain for day 1.

Uses the forecast-mode OOF-by-year predictions 2020-10..2023-12 (experiments/model_test/fc.py -> fc_prod.parquet).
Bands mirror production: conformal bins on |p50 - stage| per horizon (5 bins, 80 %), calibrated leave-one-year-out
so coverage is honest. (Production days 2-7 use learned-width bands; this simpler band is reported for all
horizons as a consistent yardstick, and is the production band at day 1.)

Also: issued GEFS lead-1 basin rain vs gridMET observed on those days.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
import numpy as np
import pandas as pd

import fc
import mt
from shingle import fcst
from shingle.evaluate import BinnedConformal

OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
HS = (1, 2, 3, 4)
pd.set_option("display.width", 250)

s = fc.state()
d = s["data"]
st = d["stage"]
p = pd.read_parquet(mt.SP + "fc_prod.parquet")
ix = p.index

plain = fcst.build(s["raw"], s["src"], s["tgt"], s["drv"], d)
w = s["drv"].weights
gefs1 = fcst._areal(plain["pr"][1], w).reindex(ix)                # raw issued lead-1 basin rain, mm
obs1 = fcst._areal(d["pr"], w).shift(-1).reindex(ix)              # gridMET basin rain on day t+1

y = fc.targets(ix)
res = {h: y[f"h{h}"] - p[f"h{h}"] for h in HS}
adp = {h: (p[f"h{h}"] - st.reindex(ix)).abs() for h in HS}

# leave-one-year-out conformal bands
inside = {}
width = {}
for h in HS:
    ins = pd.Series(np.nan, index=ix)
    wd = pd.Series(np.nan, index=ix)
    for yr in fc.YEARS:
        cal = (ix.year != yr) & res[h].notna()
        bc = BinnedConformal.fit(res[h][cal], adp[h][cal], 5, 0.8)
        held = ix.year == yr
        lo, hi = bc.offsets(adp[h][held].to_numpy())
        r = res[h][held].to_numpy()
        ins[held] = np.where(np.isnan(r), np.nan, (r >= lo) & (r <= hi))
        wd[held] = hi - lo
    inside[h], width[h] = ins, wd


def summary(mask: pd.Series, label: str) -> dict:
    row = {"subset": label, "n": int(mask.sum())}
    for h in HS:
        m = mask & res[h].notna()
        e = -res[h][m]  # forecast - observed
        row[f"bias{h}"] = e.mean()
        row[f"mae{h}"] = e.abs().mean()
        row[f"cov{h}"] = inside[h][m].mean()
        row[f"wid{h}"] = width[h][m].mean()
    return row


subsets = {
    "all": pd.Series(True, index=ix),
    "gefs1<5": gefs1 < 5,
    "gefs1 5-20": (gefs1 >= 5) & (gefs1 < 20),
    "gefs1>=20": gefs1 >= 20,
    "gefs1>=30": gefs1 >= 30,
    "obs1>=20": obs1 >= 20,
    "gefs1>=20 & obs1<10": (gefs1 >= 20) & (obs1 < 10),
    "gefs1>=20 & obs1>=20": (gefs1 >= 20) & (obs1 >= 20),
}
T = pd.DataFrame([summary(m.fillna(False), k) for k, m in subsets.items()]).set_index("subset")

lines = ["T1: forecast-mode OOF 2020-10..2023-12, by issued GEFS lead-1 basin rain (raw, mm)", ""]
lines.append("bias = forecast - observed (ft); cov = share inside LOYO |p50-stage| conformal 80 % band; wid = band width (ft)")
for kind in ("bias", "mae", "cov", "wid"):
    lines.append("")
    lines.append(T[["n"] + [f"{kind}{h}" for h in HS]].round(3).to_string())

wet = (gefs1 >= 20).fillna(False)
g, o = gefs1[wet], obs1[wet]
lines += ["", "GEFS lead-1 vs gridMET basin rain on gefs1>=20 days:",
          f"  n={wet.sum()}  mean gefs={g.mean():.1f}  mean obs={o.mean():.1f}  ratio={o.sum() / g.sum():.2f}",
          f"  obs < half of gefs: {(o < g / 2).mean():.0%}   obs > gefs: {(o > g).mean():.0%}",
          f"  all days: ratio obs/gefs total = {obs1.sum() / gefs1[obs1.notna()].sum():.2f}"]
bins = pd.cut(gefs1, [-0.1, 1, 5, 10, 20, 30, 200])
lines += ["", "Observed basin rain by GEFS lead-1 bin (conditional bias):",
          pd.DataFrame({"n": gefs1.groupby(bins, observed=True).size(), "gefs": gefs1.groupby(bins, observed=True).mean(),
                        "obs": obs1.groupby(bins, observed=True).mean(), "obs_med": obs1.groupby(bins, observed=True).median()}).round(1).to_string()]

ev = pd.DataFrame({"gefs1": gefs1, "obs1": obs1, "stage": st.reindex(ix)}
                  | {f"err{h}": -res[h] for h in HS} | {f"in{h}": inside[h] for h in HS})[wet]
lines += ["", "Event by event (gefs1>=20):", ev.round(2).to_string()]
text = "\n".join(lines)
open(OUT + "t1_report.txt", "w").write(text + "\n")
print(text)
