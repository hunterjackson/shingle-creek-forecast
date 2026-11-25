"""T6 test (docs/divergence_test_plan.md): does a 19:30Z refresh beat the frozen 13Z forecast?

At 19:30Z on day t+1 a run knows MRMS rain 12Z..18Z (hourly Pass2, ~1 h latency) and the GEFS 12Z run (complete
~18Z). Everything else (gauge 12Z reading, lakes, models) is as in the morning run, so only the rain inputs change.
Lead k covers 12Z(t+k)..12Z(t+k+1) as in production.

Variants (forecast-mode OOF-by-year models from fc.py; days from t6_fetch.py with both data sources present):
  frozen    production morning forecast: 00Z GEFS                                  (fc_prod.parquet)
  obs00     lead 1 = MRMS 12-18Z + 00Z buckets 18Z..12Z (f24, f30, f36); leads 2-7 00Z
  gefs12    12Z GEFS for every lead (lead 1 = f06..f24)
  refresh   lead 1 = MRMS 12-18Z + 12Z buckets f12..f24; leads 2-7 from 12Z
For horizon 4-7 models the per-lead scale multiplies the GEFS part only, never observed rain. MRMS hourly totals are
averaged onto model cells like the production radar fill (no extra scale; production's 24 h radar scale is ~0.98).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
import numpy as np
import pandas as pd

import fc
import mt
from shingle import basin, fcst, fetch, grid, train

OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
CACHE = Path(OUT) / "cache"
HS = fc.HS
pd.set_option("display.width", 250)

s = fc.state()
d, drv, src, tgt = s["data"], s["drv"], s["src"], s["tgt"]
ms, sc = fc.models(), fc.scales()

days = pd.read_csv(CACHE / "days.csv", parse_dates=["date"]).set_index("date")["wet"]
mr_px = pd.concat([pd.read_parquet(p) for p in sorted((CACHE / "mrms").glob("*.parquet"))]).sort_index()
mr_px = mr_px[mr_px.notna().any(axis=1)]
obs6 = grid.cell_means(mr_px, tgt, half_width=basin.GRIDMET_RES / 2)  # issue day t -> model cells, mm 12-18Z t+1
g12 = pd.concat([pd.read_parquet(p) for p in sorted((CACHE / "gefs12z").glob("*.parquet"))]).sort_index()
g00 = fetch.load_gefs("data", "mean")


def leads(buckets: pd.DataFrame, offset_h: int, first_skip: int = 0) -> pd.DataFrame:
    """(run, fhour) 6 h buckets -> issue day t = run day - 1, columns (lead, cell) daily totals. Lead k sums buckets
    ending offset_h + 24(k-1) + 6j, j = 1..4; for lead 1 the first `first_skip` buckets are left out."""
    out = {}
    for run in buckets.index.get_level_values("run").unique():
        b = buckets.xs(run, level="run")
        row = {}
        for k in HS:
            js = range(1 + (first_skip if k == 1 else 0), 5)
            fh = [offset_h + 24 * (k - 1) + 6 * j for j in js]
            ok = all(f in b.index for f in fh)
            for c in b.columns:
                row[(k, c)] = float(b.loc[fh, c].sum()) if ok else np.nan
        out[pd.Timestamp(run).normalize() - pd.Timedelta(days=1)] = row
    df = pd.DataFrame(out).T.sort_index()
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["lead", "cell"])
    return df


raw00_rest = leads(g00, 12, first_skip=1)   # 00Z: lead 1 = f24..f36 (18Z..12Z)
raw12 = leads(g12, 0)                        # 12Z: lead 1 = f06..f24
raw12_rest = leads(g12, 0, first_skip=1)     # 12Z: lead 1 = f12..f24 (18Z..12Z)

ok_days = days.index[days.index.isin(obs6.index) & days.index.isin(raw12.dropna(how="any").index)
                     & days.index.isin(raw00_rest.dropna(how="any").index)]
ix = fc.fc_index()
ix = ix[ix.isin(ok_days)]


def inputs(raw: pd.DataFrame, obs: pd.DataFrame | None, scale: dict) -> dict:
    raw = raw.reindex(s["raw"].index)
    parts = {}
    for key, scl in (("plain", None), ("scaled", scale)):
        pr = fcst.build(raw, src, tgt, drv, d, scale=scl)["pr"].copy()
        if obs is not None:
            o = obs.reindex(pr.index)
            for c in pr[1].columns:
                pr[(1, c)] = pr[(1, c)] + o[c]
        areal = pd.DataFrame({k: fcst._areal(pr[k], drv.weights) for k in HS}, index=pr.index)
        parts[key] = {"pr": pr, "gr4j_q": drv.forecast(d, areal)}
    return dict(parts["plain"], by_h={h: parts["scaled"] for h in HS if h >= train.SCALED_FROM_H})


variants = {"obs00": (raw00_rest, obs6), "gefs12": (raw12, None), "refresh": (raw12_rest, obs6)}
preds = {"frozen": pd.read_parquet(mt.SP + "fc_prod.parquet").loc[ix]}
for name, (raw, obs) in variants.items():
    parts = []
    for y in fc.YEARS:
        held = ix[ix.year == y]
        if len(held):
            parts.append(ms[y].point(dict(d, fcst=inputs(raw, obs, sc[y])), held))
    preds[name] = pd.concat(parts).sort_index()
    preds[name].to_parquet(OUT + f"t6_{name}.parquet")

y_t = fc.targets(ix)
wet = days.reindex(ix).astype(bool)
rows = []
for h in HS:
    for sub, m in (("all", pd.Series(True, index=ix)), ("wet", wet), ("dry sample", ~wet)):
        for k, p in preds.items():
            e = (p[f"h{h}"] - y_t[f"h{h}"])[m].dropna()
            rows.append({"h": h, "subset": sub, "variant": k, "n": len(e), "mae": e.abs().mean(), "bias": e.mean()})
R = pd.DataFrame(rows)
M = R.pivot_table(index=["subset", "h"], columns="variant", values="mae")[list(preds)]
G = (100 * (M.div(M["frozen"], axis=0) - 1)).drop(columns="frozen")
lines = [f"T6: 19:30Z refresh vs frozen 13Z forecast, forecast-mode OOF 2020-10..2023-12; {len(ix)} issue days "
         f"({int(wet.sum())} wet, {int((~wet).sum())} dry sample)", "", "MAE (ft):", M.round(4).to_string(), "",
         "MAE change vs frozen (%):", G.round(1).to_string(), "",
         "bias:", R.pivot_table(index=["subset", "h"], columns="variant", values="bias")[list(preds)].round(3).to_string(), ""]
for k in ("obs00", "gefs12", "refresh"):
    for sub, m in (("wet", wet), ("dry sample", ~wet)):
        b = fc.boot(preds[k][m], preds["frozen"][m], hs=(1, 2, 3, 4, 7))
        lines += [f"{k} vs frozen, {sub} (MAE diff, 90 % CI):", b.round(4).to_string(), ""]
text = "\n".join(lines)
open(OUT + "t6_report.txt", "w").write(text + "\n")
print(text)
