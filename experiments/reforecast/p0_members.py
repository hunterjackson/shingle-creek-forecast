"""Phase 0a (docs/forecast_training_plan.md): what does a 5-member GEFS mean cost compared with the 31-member mean?

The GEFSv12 reforecast has 5 members (c00 + p01-p04), production feeds the 31-member mean. Stand-in: the cached
perturbed members p01-p05 (no control member is cached). Forecast-mode OOF-by-year models (experiments/model_test/fc.py),
2020-10..2023-12, production rain handling (raw h1-3, per-lead scale h4-7).

  mean31          production (fc_prod.parquet)
  mean5 / mean10  mean of members 1-5 / 1-10, production per-lead scale
  mean5_rs        mean of members 1-5, per-lead scale refit on the 5-member mean (leave-one-year-out)
Also lead-1 basin rain skill of each mean vs gridMET.

This overstates the penalty a model *trained* on 5-member rain would pay (it would learn that noise); read it as an
upper bound.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
import numpy as np
import pandas as pd

import fc
import fc_members
import mt
from shingle import config, fcst

OUT = Path(os.path.dirname(os.path.abspath(__file__)))
pd.set_option("display.width", 250)
s = fc.state()
d, w = s["data"], s["drv"].weights
ms, sc31 = fc.models(), fc.scales()

raws = fc_members.member_raw()
common = sorted(set.intersection(*[set(raws[m].dropna(how="all").index) for m in range(1, 11)]))
ix = fc.fc_index()
ix = ix[ix.isin(common)]


def mean_of(members):
    return sum(raws[m].reindex(s["raw"].index) for m in members) / len(members)


raw5, raw10 = mean_of(range(1, 6)), mean_of(range(1, 11))
plain5 = fcst.build(raw5, s["src"], s["tgt"], s["drv"], d)["pr"]
sc5 = {y: fcst.fit_scale(plain5, d["pr"], config.RAIN_CELL, ix[ix.year != y]) for y in fc.YEARS}

preds = {
    "mean31": pd.read_parquet(mt.SP + "fc_prod.parquet").loc[ix],
    "mean10": fc.predict(raw10, sc31, ms, ix),
    "mean5": fc.predict(raw5, sc31, ms, ix),
    "mean5_rs": fc.predict(raw5, sc5, ms, ix),
}
for k, p in preds.items():
    p.to_parquet(OUT / f"p0_{k}.parquet")

# rain skill, lead 1 basin
obs1 = fcst._areal(d["pr"], w).shift(-1).reindex(ix)
rain = {}
for k, r in (("mean31", s["raw"]), ("mean10", raw10), ("mean5", raw5)):
    f = fcst._areal(fcst.build(r, s["src"], s["tgt"], s["drv"], d)["pr"][1], w).reindex(ix)
    ok = f.notna() & obs1.notna()
    rain[k] = {"spearman": f[ok].corr(obs1[ok], "spearman"), "r_log": np.log1p(f[ok]).corr(np.log1p(obs1[ok])),
               "obs/fc": obs1[ok].sum() / f[ok].sum(), "days fc>=10": int((f[ok] >= 10).sum()), "FAR10": ((f[ok] >= 10) & (obs1[ok] < 10)).sum() / max(1, (f[ok] >= 10).sum())}

plain31 = fcst.build(s["raw"], s["src"], s["tgt"], s["drv"], d)["pr"]
areal31 = pd.DataFrame({k: fcst._areal(plain31[k], w) for k in fc.HS}).reindex(ix)
yt = fc.targets(ix)
rows = []
for h in fc.HS:
    wet = areal31[list(range(1, h + 1))].sum(axis=1) >= 5
    for sub, m in (("all", pd.Series(True, index=ix)), ("rainy (31-mean >=5 mm)", wet), ("flood obs>=59", yt[f"h{h}"] >= 59), ("low water", yt[f"h{h}"] <= mt.LOW)):
        for k, p in preds.items():
            e = (p[f"h{h}"] - yt[f"h{h}"])[m].dropna()
            rows.append({"h": h, "subset": sub, "src": k, "n": len(e), "mae": e.abs().mean()})
R = pd.DataFrame(rows)
M = R.pivot_table(index=["subset", "h"], columns="src", values="mae")[list(preds)]
N = R[R.src == "mean31"].set_index(["subset", "h"])["n"]
lines = [f"Phase 0a: 5-member vs 31-member GEFS mean, forecast-mode OOF {ix.min().date()}..{ix.max().date()}, {len(ix)} issue days", "",
         "Lead-1 basin rain vs gridMET:", pd.DataFrame(rain).T.round(3).to_string(), "",
         "Stage MAE:", M.assign(n=N).round(4).to_string(), "",
         "change vs mean31 (%):", (100 * (M.div(M["mean31"], axis=0) - 1)).drop(columns="mean31").round(1).to_string(), ""]
for k in ("mean5", "mean5_rs", "mean10"):
    lines += [f"{k} vs mean31, all days (MAE diff, 90 % CI):", fc.boot(preds[k], preds["mean31"]).round(4).to_string(), ""]
text = "\n".join(lines)
(OUT / "p0_members_report.txt").write_text(text + "\n")
print(text)
