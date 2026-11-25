"""T4 (docs/divergence_test_plan.md): should every horizon see the same forecast rain for a given calendar day?

Production ("mixed"): horizon 1-3 models get raw GEFS for all leads; horizon 4-7 models get every lead scaled by the
per-lead total-rain ratio (~1.4, including lead 1). So the same day's rain differs between horizons, and the
issued path can jump between day 3 and day 4 (11-24 issue: 57.55 -> 58.34 with no rain in between).

Settings, forecast-mode OOF-by-year 2020-10..2023-12 (same models as fc_prod):
  mixed     production
  raw       raw GEFS for every horizon
  scaled    scaled GEFS for every horizon
  perlead   one rain series per issue day: leads 1-3 raw, leads 4-7 scaled, used by every horizon

Metrics: MAE per horizon (+ flood >=59 / low-water subsets via fc.metrics), paired block bootstrap vs mixed, and
roughness = mean |p50(h+1) - p50(h)| on transitions where issued rain on day t+h+1 is < 1 mm at the rain cell.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
import pandas as pd

import fc
import mt
from shingle import config, fcst

OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
pd.set_option("display.width", 250)
s = fc.state()
d = s["data"]
ms = fc.models()
sc = fc.scales()
ix = fc.fc_index()
HS = fc.HS


def predict_with(fcst_by_year) -> pd.DataFrame:
    parts = []
    for y in fc.YEARS:
        held = ix[ix.year == y]
        parts.append(ms[y].point(dict(d, fcst=fcst_by_year(y)), held))
    return pd.concat(parts).sort_index()


def perlead_raw(y):
    r = s["raw"].copy()
    for k in range(4, 8):
        cols = [c for c in r.columns if c[0] == k]
        r[cols] = r[cols] * sc[y][k]
    return r


def build(raw, scale=None):
    return fcst.build(raw, s["src"], s["tgt"], s["drv"], d, scale=scale)


preds = {
    "mixed": pd.read_parquet(mt.SP + "fc_prod.parquet"),
    "raw": predict_with(lambda y: build(s["raw"])),
    "scaled": predict_with(lambda y: build(s["raw"], sc[y])),
    "perlead": predict_with(lambda y: build(perlead_raw(y))),
}
for k, p in preds.items():
    p.to_parquet(OUT + f"t4_{k}.parquet")

cell = config.RAIN_CELL
mapped = build(s["raw"])["pr"]


def roughness(p: pd.DataFrame) -> pd.Series:
    out = {}
    for h in HS[:-1]:
        dry = (mapped[(h + 1, cell)].reindex(p.index) < 1.0)
        out[f"{h}->{h + 1}"] = (p[f"h{h + 1}"] - p[f"h{h}"]).abs()[dry].mean()
    return pd.Series(out)


lines = ["T4: rain consistency across horizons, forecast-mode OOF 2020-10..2023-12", ""]
M = pd.DataFrame({k: fc.metrics(p) for k, p in preds.items()}).T
lines += ["MAE all / flood (obs>=59) / low water:", M[[f"mae{h}" for h in HS]].round(4).to_string(), "",
          M[[f"hi{h}" for h in HS]].round(3).to_string(), "", M[[f"lo{h}" for h in HS]].round(4).to_string(), ""]
R = pd.DataFrame({k: roughness(p) for k, p in preds.items()}).T
R["mean"] = R.mean(axis=1)
lines += ["Roughness: mean |p50(h+1)-p50(h)| when issued rain on day t+h+1 < 1 mm (ft):", R.round(4).to_string(), ""]
for k in ("raw", "scaled", "perlead"):
    b = fc.boot(preds[k], preds["mixed"])
    lines += [f"{k} vs mixed (MAE diff, 90 % CI):", b.round(4).to_string(), ""]
text = "\n".join(lines)
open(OUT + "t4_report.txt", "w").write(text + "\n")
print(text)
