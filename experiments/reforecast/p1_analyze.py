"""Phase 1 analysis (docs/forecast_training_plan.md): pilot years 2004, 2013 (quiet), 2017, control member c00.

1a  Same forecast system?  Basin-areal rain (bilinear onto the model cells, basin weights) vs gridMET, per lead:
    obs/fc total ratio, Spearman, share of days with fc >= 10 mm (and obs >= 10 mm), false-alarm ratio at 10 mm.
    Reforecast c00 per pilot year and pooled, against live c00 2020-10..2023-12 (pooled and per year, the yardstick
    for how much these numbers move from one year to the next) and the live 31-member mean.
    Gate (fixed before the pull): obs/fc ratio within ~15 % and Spearman within ~0.05 of live c00.

1b  First look at floods in forecast mode.  Production spec, out-of-fold by year (fit on 1979-2023 minus the pilot
    year, 7-day gap; fc._fit_year), on each pilot year's issue days with
      perfect   observed future rain (how the model is trained)
      c00       reforecast control, production handling (raw h1-3, live per-lead scale h4-7)
      zero      no future rain (what the rain forecast is worth)
    and, for scale, live 2020-10..2023-12 with fc_prod (31-member mean) vs live c00.
    Flood events from flood_events.csv (frozen): crest error, timing, p50 hits, per h.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
import numpy as np
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


def load(name) -> pd.DataFrame:
    b = pd.concat([pd.read_parquet(p) for p in sorted((OUT / "cache" / name).glob("*.parquet"))]).sort_index()
    return gefs.daily_leads(b)


rf, live = load("rf_c00"), load("live_c00")
src_rf, src_live = gridmet.cells_from_columns(rf[1].columns), gridmet.cells_from_columns(live[1].columns)


def basin(raw, src) -> pd.DataFrame:
    wt = grid.bilinear_weights(tgt, src)
    return pd.DataFrame({k: fcst._areal(grid.apply_weights(raw[k], wt), w) for k in fc.HS})


obs = fcst._areal(d["pr"], w)
B = {"rf_c00": basin(rf, src_rf), "live_c00": basin(live, src_live), "live_mean31": basin(s["raw"], s["src"])}


def rain_stats(f: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    rows = {}
    for k in fc.HS:
        fk = f[k].reindex(days)
        ok_ = obs.shift(-k).reindex(days)
        m = fk.notna() & ok_.notna()
        fk, ok_ = fk[m], ok_[m]
        rows[k] = {"n": int(m.sum()), "obs/fc": ok_.sum() / fk.sum(), "spearman": fk.corr(ok_, "spearman"),
                   "fc>=10 %": 100 * (fk >= 10).mean(), "obs>=10 %": 100 * (ok_ >= 10).mean(),
                   "FAR10": ((fk >= 10) & (ok_ < 10)).sum() / max(1, (fk >= 10).sum()),
                   "POD10": ((fk >= 10) & (ok_ >= 10)).sum() / max(1, (ok_ >= 10).sum())}
    return pd.DataFrame(rows).T


live_days = pd.date_range("2020-10-01", "2023-12-31")
sets = {f"rf {y}": ("rf_c00", pd.date_range(f"{y}-01-01", f"{y}-12-31")) for y in PILOT}
sets["rf pooled"] = ("rf_c00", pd.DatetimeIndex([t for y in PILOT for t in pd.date_range(f"{y}-01-01", f"{y}-12-31")]))
for y in (2021, 2022, 2023):
    sets[f"live c00 {y}"] = ("live_c00", pd.date_range(f"{y}-01-01", f"{y}-12-31"))
sets["live c00 pooled"] = ("live_c00", live_days)
sets["live mean31 pooled"] = ("live_mean31", live_days)
RS = pd.concat({k: rain_stats(B[src], days) for k, (src, days) in sets.items()}, names=["set", "lead"])

lines = ["Phase 1: GEFSv12 reforecast control (c00), pilot years 2004, 2013, 2017", "",
         "## 1a  Basin rain vs gridMET", ""]
for col in ("obs/fc", "spearman", "fc>=10 %", "obs>=10 %", "FAR10", "POD10"):
    lines += [f"{col}:", RS[col].unstack("lead").round(3).to_string(), ""]
ref = RS.loc["live c00 pooled"]
gate = pd.DataFrame({"ratio rf/live": RS.loc["rf pooled", "obs/fc"] / ref["obs/fc"] - 1,
                     "spearman rf-live": RS.loc["rf pooled", "spearman"] - ref["spearman"]})
gate["pass"] = (gate["ratio rf/live"].abs() <= 0.15) & (gate["spearman rf-live"] >= -0.05)
lines += ["Gate 1a (rf pooled vs live c00 pooled; ratio within 15 %, Spearman not lower by > 0.05):", gate.round(3).to_string(), ""]

# ---------------- 1b
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

todo = [y for y in PILOT if not os.path.exists(mt.SP + f"fc_models/{y}.pkl")]
if todo:
    with ProcessPoolExecutor(len(todo), mp_context=mp.get_context("fork")) as ex:
        list(ex.map(fc._fit_year, todo))
import pickle

pm = {y: pickle.load(open(mt.SP + f"fc_models/{y}.pkl", "rb")) for y in PILOT}
live_scale = fcst.fit_scale(fcst.build(s["raw"], s["src"], tgt, s["drv"], d)["pr"], d["pr"], config.RAIN_CELL, fc.fc_index())
rf_full = rf.reindex(d["stage"].index)
f_rf = fcst.build_mixed(rf_full, src_rf, tgt, s["drv"], d, live_scale, train.SCALED_FROM_H, fc.HS)
zero = rf_full.where(rf_full.isna(), 0.0)
f_zero = fcst.build_mixed(zero, src_rf, tgt, s["drv"], d, live_scale, train.SCALED_FROM_H, fc.HS)
stage_ix = d["stage"].dropna().index
issued = rf.dropna(how="all").index
P = {"perfect": [], "c00": [], "zero": []}
for y in PILOT:
    ix = stage_ix[(stage_ix.year == y) & stage_ix.isin(issued)]
    P["perfect"].append(pm[y].point(d, ix))
    P["c00"].append(pm[y].point(dict(d, fcst=f_rf), ix))
    P["zero"].append(pm[y].point(dict(d, fcst=f_zero), ix))
P = {k: pd.concat(v).sort_index() for k, v in P.items()}
for k, p in P.items():
    p.to_parquet(OUT / f"p1_{k}.parquet")
pix = P["perfect"].index

# live, for scale: 31-member mean (production) vs c00, OOF 2020-23 models
lix = fc.fc_index()
lix = lix[lix.isin(live.dropna(how="all").index)]
L = {"live mean31": pd.read_parquet(mt.SP + "fc_prod.parquet").loc[lix],
     "live c00": fc.predict(live.reindex(s["raw"].index), {y: live_scale for y in fc.YEARS}, fc.models(), lix)}


def mae_table(preds: dict, ix) -> pd.DataFrame:
    yt = fc.targets(ix)
    a5 = B["rf_c00"] if "c00" in preds else B["live_c00"]
    rows = []
    for h in fc.HS:
        wet = a5[list(range(1, h + 1))].sum(axis=1).reindex(ix) >= 5
        for sub, m in (("all", pd.Series(True, index=ix)), ("rainy (c00 >= 5 mm)", wet), ("flood obs>=59", yt[f"h{h}"] >= 59),
                       ("low water", yt[f"h{h}"] <= mt.LOW)):
            for k, p in preds.items():
                e = (p.loc[ix, f"h{h}"] - yt[f"h{h}"])[m].dropna()
                rows.append({"subset": sub, "h": h, "src": k, "n": len(e), "mae": e.abs().mean()})
    R = pd.DataFrame(rows)
    M = R.pivot_table(index=["subset", "h"], columns="src", values="mae")[list(preds)]
    return M.assign(n=R[R.src == list(preds)[0]].set_index(["subset", "h"])["n"])


lines += ["## 1b  Production model in forecast mode on the pilot years (OOF by year)", "",
          f"{len(pix)} issue days. MAE (ft):", mae_table(P, pix).round(3).to_string(), "",
          "Same, per pilot year, all days:"]
for y in PILOT:
    yy = pix[pix.year == y]
    lines += [f"{y} ({len(yy)} days):", mae_table(P, yy).loc["all"].round(3).T.to_string(), ""]
lines += ["Share of the perfect-rain gain the c00 forecast recovers, (zero - c00) / (zero - perfect), all days:"]
Mall = mae_table(P, pix).loc["all"]
lines += [((Mall["zero"] - Mall["c00"]) / (Mall["zero"] - Mall["perfect"])).round(2).to_frame("share").T.to_string(), ""]
lines += ["Live 2020-10..2023-12 for comparison (31-member mean = production vs control alone):",
          mae_table(L, lix).round(3).to_string(), ""]

# flood events
ev = pd.read_csv(OUT / "flood_events.csv", parse_dates=["first", "last", "window_start", "window_end", "crest_date"])
ev = ev[ev["first"].dt.year.isin(PILOT) | (ev["first"].dt.year == 2022)]
st = d["stage"]
rows = []
for _, e in ev.iterrows():
    live_ev = e["first"].year == 2022
    preds = L if live_ev else P
    for h in fc.HS:
        tdays = pd.date_range(e["window_start"], e["window_end"])
        idays = tdays - pd.Timedelta(days=h)
        for k, p in preds.items():
            v = p[f"h{h}"].reindex(idays)
            v.index = tdays
            if v.notna().sum() == 0:
                continue
            rows.append({"event": f"{e['crest_date']:%Y-%m-%d} ({e['crest_ft']:.2f})", "h": h, "src": k,
                         "crest_err": v.max() - e["crest_ft"], "timing_d": (v.idxmax() - e["crest_date"]).days,
                         "hit": bool((v >= 59).any()), "n": int(v.notna().sum())})
E = pd.DataFrame(rows)
lines += ["Flood events (frozen list): crest error of p50 (ft), max over the window minus observed crest:",
          E.pivot_table(index=["event", "src"], columns="h", values="crest_err").round(2).to_string(), "",
          "Timing of the forecast max (days, + = late):",
          E.pivot_table(index=["event", "src"], columns="h", values="timing_d").to_string(), "",
          "p50 >= 59 ft anywhere in the window (hit):",
          E.pivot_table(index=["event", "src"], columns="h", values="hit", aggfunc="first").replace({True: "Y", False: "."}).to_string(), ""]
pe = E[~E.src.str.startswith("live")]
lines += ["Pilot events, mean |crest error| by source:", pe.assign(a=pe.crest_err.abs()).pivot_table(index="src", columns="h", values="a").round(2).to_string(), ""]

# rain forecast during the events, lead 1..3
lines += ["Basin rain around each pilot crest (mm): observed day total vs c00 lead-1 and lead-3 forecasts for that day:"]
for _, e in ev[ev["first"].dt.year.isin(PILOT)].iterrows():
    days = pd.date_range(e["crest_date"] - pd.Timedelta(days=6), e["crest_date"] + pd.Timedelta(days=1))
    t = pd.DataFrame({"obs": obs.reindex(days),
                      "c00 lead1": B["rf_c00"][1].shift(1).reindex(days),
                      "c00 lead3": B["rf_c00"][3].shift(3).reindex(days),
                      "stage": st.reindex(days)})
    lines += [f"{e['crest_date']:%Y-%m-%d}:", t.round(1).T.to_string(), ""]

text = "\n".join(lines)
(OUT / "p1_report.txt").write_text(text + "\n")
print(text)
