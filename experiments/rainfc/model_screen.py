"""Rain-forecast screen, step 2 (no new downloads): does NBM lead-1 rain improve the stage forecast?

On the 219 NBM days from fetch.py (2020-10..2023-12; selection favours wet days), forecast-mode OOF-by-year models
(fc.py) are rerun with lead-1 rain in every model cell (and the GR4J flow it drives) replaced; leads 2-7 stay GEFS.
NBM under-forecasts amounts (obs/fc 2.8 vs GEFS 1.8), so it is rescaled with ratios fitted leave-one-year-out on these days:
  nbm_g     NBM x sum(GEFS)/sum(NBM)      (GEFS level, like the raw GEFS production feeds h1-3)
  nbm_o     NBM x sum(obs)/sum(NBM)       (observed level)
  avg       (GEFS + nbm_g) / 2
For horizon 4-7 models lead 1 is further multiplied by the production per-lead scale (as GEFS is).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
import pandas as pd

import fc
import mt
from shingle import config, fcst, train

OUT = Path(os.path.dirname(os.path.abspath(__file__)))
CACHE = OUT / "cache"
pd.set_option("display.width", 250)
s = fc.state()
d, drv, tgt = s["data"], s["drv"], s["tgt"]
ms, sc = fc.models(), fc.scales()
HS = fc.HS
cell = config.RAIN_CELL

nbm = pd.concat([pd.read_parquet(p) for p in sorted((CACHE / "nbm").glob("*.parquet"))]).sort_index()
base = fcst.build(s["raw"], s["src"], tgt, drv, d)["pr"]
obs1 = d["pr"].shift(-1)
ix = fc.fc_index()
ix = ix[ix.isin(nbm.dropna(how="all").index)]


def ratio(num, den, fit):
    return float(num.reindex(fit)[cell].sum() / den.reindex(fit)[cell].sum())


def lead1_for(variant: str, y: int) -> pd.DataFrame:
    fit = ix[ix.year != y]
    g = base[1]
    if variant == "nbm_g":
        return nbm * ratio(g, nbm, fit)
    if variant == "nbm_o":
        return nbm * ratio(obs1, nbm, fit)
    if variant == "avg":
        return (g.reindex(nbm.index) + nbm * ratio(g, nbm, fit)) / 2
    raise KeyError(variant)


def inputs(lead1: pd.DataFrame, scale: dict) -> dict:
    parts = {}
    for key, scl in (("plain", None), ("scaled", scale)):
        pr = fcst.build(s["raw"], s["src"], tgt, drv, d, scale=scl)["pr"].copy()
        l1 = lead1.reindex(pr.index) * (scl.get(1, 1.0) if scl else 1.0)
        for c in pr[1].columns:
            have = l1[c].notna()
            pr.loc[have, (1, c)] = l1.loc[have, c]
        areal = pd.DataFrame({k: fcst._areal(pr[k], drv.weights) for k in HS}, index=pr.index)
        parts[key] = {"pr": pr, "gr4j_q": drv.forecast(d, areal)}
    return dict(parts["plain"], by_h={h: parts["scaled"] for h in HS if h >= train.SCALED_FROM_H})


preds = {"gefs": pd.read_parquet(mt.SP + "fc_prod.parquet").loc[ix]}
for v in ("nbm_g", "nbm_o", "avg"):
    parts = []
    for y in fc.YEARS:
        held = ix[ix.year == y]
        if len(held):
            parts.append(ms[y].point(dict(d, fcst=inputs(lead1_for(v, y), sc[y])), held))
    preds[v] = pd.concat(parts).sort_index()

yt = fc.targets(ix)
g1 = fcst._areal(base[1], drv.weights).reindex(ix)
o1 = fcst._areal(obs1, drv.weights).reindex(ix)
rows = []
for h in (1, 2, 3, 4, 7):
    for sub, m in (("all 219", pd.Series(True, index=ix)), ("gefs>=20 or obs>=20", (g1 >= 20) | (o1 >= 20)), ("gefs>=20 & obs<10", (g1 >= 20) & (o1 < 10))):
        for k, p in preds.items():
            e = (p[f"h{h}"] - yt[f"h{h}"])[m].dropna()
            rows.append({"h": h, "subset": sub, "src": k, "n": len(e), "mae": e.abs().mean()})
R = pd.DataFrame(rows)
M = R.pivot_table(index=["subset", "h"], columns="src", values="mae")[list(preds)]
G = (100 * (M.div(M["gefs"], axis=0) - 1)).drop(columns="gefs")
lines = [f"NBM lead-1 model screen, {len(ix)} days", "", "MAE:", M.round(4).to_string(), "", "change vs GEFS (%):", G.round(1).to_string(), ""]
for k in ("nbm_g", "nbm_o", "avg"):
    lines += [f"{k} vs gefs, all days (MAE diff, 90 % CI):", fc.boot(preds[k], preds["gefs"], hs=(1, 2, 3, 4, 7)).round(4).to_string(), ""]
text = "\n".join(lines)
(OUT / "model_screen_report.txt").write_text(text + "\n")
print(text)
