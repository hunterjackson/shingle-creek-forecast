"""T6 upper-bound screen (docs/divergence_test_plan.md): would an intraday refresh with observed rain help?

The 13Z forecast is frozen until the next morning. A rerun later on day t+1 knows part of the lead-1 rain
(12Z t+1 .. 12Z t+2). Cheap proxy before pulling any 6-hourly radar archive: lead-1 rain (every model cell, and the
GR4J forecast flow driven by it) = (1 - f) * issued GEFS + f * observed gridMET rain on t+1, with
  f = 0.25  ~ an 18Z rerun (6 of 24 h observed), f = 0.5 ~ 00Z, f = 1.0  perfect day-1 rain (upper bound).
Leads 2-7 stay as issued (scaled for h4-7 as production; lead 1 is never scaled once observed rain enters it).
Uniform-in-time mixing is crude (rain is bursty), so f=1.0 bounds what any refresh could gain.

Forecast-mode OOF-by-year models (fc.py), 2020-10..2023-12; scored on all days and on wet days (GEFS basin
rain over leads 1..h >= 5 mm). Pass to a real test only if f=0.5 cuts day-1..3 wet-day MAE clearly (> ~10 %).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
import pandas as pd

import fc
from shingle import fcst, train

OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
HS = (1, 2, 3, 4)
FS = (0.0, 0.25, 0.5, 1.0)
pd.set_option("display.width", 250)

s = fc.state()
d = s["data"]
drv = s["drv"]
ms = fc.models()
sc = fc.scales()
ix = fc.fc_index()
obs_next = d["pr"].shift(-1)  # model cells, rain on day t+1 indexed by issue day t


def patched(f_mix: float, y: int) -> dict:
    """build_mixed with lead-1 rain mixed toward the observed, in the plain and the scaled (h>=4) inputs."""
    base = fcst.build_mixed(s["raw"], s["src"], s["tgt"], drv, d, sc[y], train.SCALED_FROM_H, fc.HS)
    out = {}
    for key, part in (("plain", base), ("scaled", base["by_h"][fc.HS[-1]])):
        pr = part["pr"].copy()
        cells = [c for c in pr[1].columns]
        if f_mix > 0:
            plain1 = base["pr"][1][cells]  # unscaled issued lead 1
            mixed = (1 - f_mix) * plain1 + f_mix * obs_next[cells].reindex(pr.index)
            for c in cells:
                pr[(1, c)] = mixed[c]
        leads = sorted(pr.columns.get_level_values(0).unique())
        areal = pd.DataFrame({k: fcst._areal(pr[k], drv.weights) for k in leads}, index=pr.index)
        out[key] = {"pr": pr, "gr4j_q": drv.forecast(d, areal)}
    return dict(out["plain"], by_h={h: out["scaled"] for h in fc.HS if h >= train.SCALED_FROM_H})


preds = {}
for f_mix in FS:
    parts = []
    for y in fc.YEARS:
        held = ix[ix.year == y]
        parts.append(ms[y].point(dict(d, fcst=patched(f_mix, y)), held))
    preds[f_mix] = pd.concat(parts).sort_index()

plain = fcst.build(s["raw"], s["src"], s["tgt"], drv, d)
areal = pd.DataFrame({k: fcst._areal(plain["pr"][k], drv.weights) for k in fc.HS}).reindex(ix)
y_t = fc.targets(ix)
rows = []
for h in HS:
    wet = areal[list(range(1, h + 1))].sum(axis=1) >= 5
    for name, m in (("all", pd.Series(True, index=ix)), ("wet>=5", wet), ("wet>=20", areal[list(range(1, h + 1))].sum(axis=1) >= 20)):
        for f_mix, p in preds.items():
            e = (p[f"h{h}"] - y_t[f"h{h}"])[m].dropna()
            rows.append({"h": h, "subset": name, "f": f_mix, "n": len(e), "mae": e.abs().mean(), "bias": e.mean()})
R = pd.DataFrame(rows)
M = R.pivot_table(index=["h", "subset"], columns="f", values="mae")
B = R.pivot_table(index=["h", "subset"], columns="f", values="bias")
G = (100 * (M.div(M[0.0], axis=0) - 1)).drop(columns=0.0)
lines = ["T6 upper-bound screen: lead-1 rain mixed toward observed (f = observed share), forecast-mode OOF 2020-10..2023-12", "",
         "n:", R[R.f == 0.0].set_index(["h", "subset"])["n"].to_string(), "", "MAE (ft):", M.round(4).to_string(), "",
         "MAE change vs issued (%):", G.round(1).to_string(), "", "bias (forecast - obs):", B.round(3).to_string()]
for f_mix in (0.5, 1.0):
    for h in (1, 2, 3):
        wet = (areal[list(range(1, h + 1))].sum(axis=1) >= 5).reindex(ix, fill_value=False)
        b = fc.boot(preds[f_mix][wet], preds[0.0][wet], hs=(h,))
        lines.append(f"f={f_mix} h{h} wet>=5 vs issued: " + b.round(4).to_string(header=False, index=False))
text = "\n".join(lines)
open(OUT + "t6_report.txt", "w").write(text + "\n")
print(text)
