"""Phase 1 follow-up checks on the 1a rain-total difference (reforecast c00 ~20 % wetter vs gridMET than live c00).

  res    Is 0.25 deg GEFS wetter than 0.5 deg? Live ensemble mean at both resolutions on the cached rainfc days
         (experiments/rainfc/cache/gefs025), lead-1 basin totals.
  scale  1b with the per-lead scale refit on the reforecast itself (leave-one-pilot-year-out) instead of live's,
         i.e. what an era-matched bias correction gives. Low-water and all-day MAE vs the live-scale run and zero rain.
"""
import os
import sys
import pickle
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
import pandas as pd

import fc
import mt
from shingle import config, fcst, grid, train
from shingle.sources import gefs, gridmet

OUT = Path(os.path.dirname(os.path.abspath(__file__)))
PILOT = (2004, 2013, 2017)
pd.set_option("display.width", 250)
s = fc.state()
d, w, tgt = s["data"], s["drv"].weights, s["tgt"]
lines = ["Phase 1 follow-up checks", ""]

# res
b = pd.concat([pd.read_parquet(p) for p in sorted((OUT.parent / "rainfc" / "cache" / "gefs025").glob("*.parquet"))]).sort_index()
runs = b.index.get_level_values("run").unique()
r25 = {}
for run in runs:
    x = b.xs(run, level="run")
    if all(f in x.index for f in (18, 24, 30, 36)):
        r25[run - pd.Timedelta(days=1)] = x.loc[[18, 24, 30, 36]].sum(min_count=4)
r25 = pd.DataFrame(r25).T
src25 = gridmet.cells_from_columns(r25.columns)
a25 = fcst._areal(grid.apply_weights(r25, grid.bilinear_weights(tgt, src25)), w)
a05 = fcst._areal(grid.apply_weights(s["raw"][1], grid.bilinear_weights(tgt, s["src"])), w).reindex(a25.index)
obs1 = fcst._areal(d["pr"], w).shift(-1).reindex(a25.index)
ok = a25.notna() & a05.notna() & obs1.notna()
lines += [f"res: {int(ok.sum())} days (rain-enriched sample), lead-1 basin totals: 0.25 deg {a25[ok].sum():.0f} mm, 0.5 deg {a05[ok].sum():.0f} mm,"
          f" ratio 0.25/0.5 = {a25[ok].sum() / a05[ok].sum():.3f}; obs/fc 0.25 {obs1[ok].sum() / a25[ok].sum():.3f}, 0.5 {obs1[ok].sum() / a05[ok].sum():.3f}", ""]

# scale
rf = gefs.daily_leads(pd.concat([pd.read_parquet(p) for p in sorted((OUT / "cache" / "rf_c00").glob("*.parquet"))]).sort_index())
src_rf = gridmet.cells_from_columns(rf[1].columns)
rf_full = rf.reindex(d["stage"].index)
plain_rf = fcst.build(rf_full, src_rf, tgt, s["drv"], d)["pr"]
live_scale = fcst.fit_scale(fcst.build(s["raw"], s["src"], tgt, s["drv"], d)["pr"], d["pr"], config.RAIN_CELL, fc.fc_index())
issued = rf.dropna(how="all").index
stage_ix = d["stage"].dropna().index
pm = {y: pickle.load(open(mt.SP + f"fc_models/{y}.pkl", "rb")) for y in PILOT}
parts, scales = [], {}
for y in PILOT:
    ix = stage_ix[(stage_ix.year == y) & stage_ix.isin(issued)]
    other = issued[(issued.year != y) & issued.year.isin(PILOT)]
    sc = fcst.fit_scale(plain_rf, d["pr"], config.RAIN_CELL, other)
    scales[y] = sc
    f = fcst.build_mixed(rf_full, src_rf, tgt, s["drv"], d, sc, train.SCALED_FROM_H, fc.HS)
    parts.append(pm[y].point(dict(d, fcst=f), ix))
P = {k: pd.read_parquet(OUT / f"p1_{k}.parquet") for k in ("perfect", "c00", "zero")}
P["c00 rf-scale"] = pd.concat(parts).sort_index()
ix = P["perfect"].index
lines += ["per-lead scale (obs/fc at the rain cell): live " + str({k: round(v, 2) for k, v in live_scale.items()})]
lines += [f"  rf, fitted without {y}: " + str({k: round(v, 2) for k, v in sc.items()}) for y, sc in scales.items()]
yt = fc.targets(ix)
rows = []
for h in fc.HS:
    for sub, m in (("all", pd.Series(True, index=ix)), ("low water", yt[f"h{h}"] <= mt.LOW), ("flood obs>=59", yt[f"h{h}"] >= 59)):
        for k, p in P.items():
            e = (p[f"h{h}"] - yt[f"h{h}"])[m].dropna()
            rows.append({"subset": sub, "h": h, "src": k, "mae": e.abs().mean(), "bias": e.mean()})
R = pd.DataFrame(rows)
lines += ["", "MAE:", R.pivot_table(index=["subset", "h"], columns="src", values="mae")[list(P)].round(3).to_string(),
          "", "Bias (forecast - observed):", R.pivot_table(index=["subset", "h"], columns="src", values="bias")[list(P)].round(3).to_string()]
text = "\n".join(lines)
(OUT / "p1_checks_report.txt").write_text(text + "\n")
print(text)
