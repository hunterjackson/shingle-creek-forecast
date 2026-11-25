"""T4b: quantile-mapped GEFS rain instead of the flat per-lead scale.

T4 found the flat ~1.4x scale (h4-7) helps floods but costs 25-50 % MAE on low-water days at h4-7 vs raw rain.
Quantile mapping corrects the distribution instead of the total: drizzle the ensemble mean smears over many days
maps toward 0, big forecasts map up. Per lead k, fitted at the rain cell (mapped GEFS lead k at t vs gridMET at t+k),
leave-one-year-out over the GEFS era 2020-10..2023-12, applied to every source cell before fcst.build.

Variants (OOF-by-year models from fc.py, as T4):
  mixed       production (raw h1-3, flat scale h4-7)
  qm          quantile-mapped rain for every horizon
  mixed_qm    raw h1-3, quantile-mapped h4-7
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
import numpy as np
import pandas as pd

import fc
import mt
from shingle import config, fcst, train

OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
pd.set_option("display.width", 250)
s = fc.state()
d = s["data"]
ms = fc.models()
ix = fc.fc_index()
HS = fc.HS
CELL = config.RAIN_CELL
QS = np.linspace(0, 1, 401)


def build(raw):
    return fcst.build(raw, s["src"], s["tgt"], s["drv"], d)


plain = build(s["raw"])["pr"]


def fit_map(k: int, fit_ix: pd.DatetimeIndex):
    f = plain[(k, CELL)].reindex(fit_ix)
    o = d["pr"][CELL].shift(-k).reindex(fit_ix)
    ok = f.notna() & o.notna()
    fq, oq = np.quantile(f[ok], QS), np.quantile(o[ok], QS)
    # collapse ties in the forecast quantiles (many zeros) to their mean target value
    u, inv = np.unique(fq, return_inverse=True)
    ou = np.bincount(inv, weights=oq) / np.bincount(inv)
    ou = np.maximum.accumulate(ou)
    top = ou[-1] / u[-1] if u[-1] > 0 else 1.0

    def m(x):
        x = np.asarray(x, float)
        y = np.interp(x, u, ou)
        return np.where(x > u[-1], x * top, y)

    return m


def qm_raw(y: int) -> pd.DataFrame:
    fit_ix = ix[ix.year != y]
    r = s["raw"].copy()
    for k in HS:
        m = fit_map(k, fit_ix)
        cols = [c for c in r.columns if c[0] == k]
        r[cols] = r[cols].apply(lambda col: pd.Series(m(col.to_numpy()), index=col.index))
    return r


preds = {"mixed": pd.read_parquet(mt.SP + "fc_prod.parquet")}
parts_qm, parts_mq = [], []
for y in fc.YEARS:
    held = ix[ix.year == y]
    fq = build(qm_raw(y))
    parts_qm.append(ms[y].point(dict(d, fcst=fq), held))
    mix = dict(build(s["raw"]), by_h={h: fq for h in HS if h >= train.SCALED_FROM_H})
    parts_mq.append(ms[y].point(dict(d, fcst=mix), held))
preds["qm"] = pd.concat(parts_qm).sort_index()
preds["mixed_qm"] = pd.concat(parts_mq).sort_index()
for k in ("qm", "mixed_qm"):
    preds[k].to_parquet(OUT + f"t4b_{k}.parquet")

lines = ["T4b: quantile-mapped GEFS rain, forecast-mode OOF 2020-10..2023-12", ""]
M = pd.DataFrame({k: fc.metrics(p) for k, p in preds.items()}).T
for pre, lab in (("mae", "all"), ("hi", "flood obs>=59"), ("lo", "low water")):
    lines += [f"MAE {lab}:", M[[f"{pre}{h}" for h in HS]].round(4).to_string(), ""]
for k in ("qm", "mixed_qm"):
    for sub in (None, "hi", "lo"):
        b = fc.boot(preds[k], preds["mixed"], subset=sub)
        lines += [f"{k} vs mixed, subset={sub or 'all'} (MAE diff, 90 % CI):", b.round(4).to_string(), ""]
# what the mapping does, fit on all GEFS-era days
lines.append("Mapping fit on all 2020-10..2023-12 issue days (GEFS mm -> mapped mm):")
xs = np.array([0.1, 0.5, 1, 2, 5, 10, 20, 35, 60])
rows = {f"lead{k}": fit_map(k, ix)(xs).round(1) for k in HS}
lines.append(pd.DataFrame(rows, index=xs).T.to_string())
text = "\n".join(lines)
open(OUT + "t4b_report.txt", "w").write(text + "\n")
print(text)
