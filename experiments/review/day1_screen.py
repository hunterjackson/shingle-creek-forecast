"""Review screens for unexplored day-1 levers (2026-11-25). Existing OOF predictions only; test period (2024+) untouched.

A. Extra issue-time information. Production knows one morning number per gauge (last hourly mean in 10-12Z of t+1,
   minus the day-t mean). Candidates, all known by 13Z on t+1 from the hourly record (2007+):
     sofar   mean of the target day's own first 8 hours (05Z..12Z of t+1) minus the morning reading
     slope3 / slope6 / slope12   morning reading minus the reading 3 / 6 / 12 h earlier (rate of rise)
     up_*    the same slopes for the upstream gauge (Airport), log flow
   Screen as in screen.py: correlation with the production OOF residual, then a leave-year-block-out LightGBM
   residual model on those features to estimate the MAE gain at days 1-3.

B. Intraday update of day 1. The site reruns every 3 h but day 1 stays the 13Z forecast, while the target day
   (05Z t+1 .. 05Z t+2) is being observed. Estimate at cutoff hour C (labels 05Z..C observed):
     naive   (n * mean so far + (24 - n) * latest reading) / 24
     reg     year-blocked linear regression of the day mean on [p50, mean so far, latest reading]
   MAE against the frozen 13Z p50.

Predictions: perfect-rain OOF 1979-2023 (experiments/model_test/oof/prod_s0.parquet), rows 2007-10+ where the hourly
record exists; B also on forecast-mode OOF 2020-10..2023-12 (fc_prod.parquet, issued GEFS rain).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
import numpy as np
import pandas as pd

import mt
from shingle import config, fetch

OUT = Path(os.path.dirname(os.path.abspath(__file__)))
pd.set_option("display.width", 250)
hist = fetch.load_history("data")
stage = hist["gauge"]["stage_ft"]
ivg, ivu = hist["iv"]["gauge"], hist["iv"]["upstream"]


def hourly(s: pd.Series) -> pd.DataFrame:
    """Wide frame: index = issue day t, columns = UTC hours since 00Z of t+1 (-12..29), hourly means."""
    s = s.dropna()
    s.index = s.index.tz_convert("UTC").tz_localize(None)
    s = s.resample("1h").mean()
    days = pd.date_range(s.index.min().normalize(), s.index.max().normalize())
    out = {}
    for k in range(-12, 30):
        v = s.reindex(days + pd.Timedelta(hours=k))
        out[k] = pd.Series(v.to_numpy(), index=days - pd.Timedelta(days=1))
    return pd.DataFrame(out)


G = hourly(ivg["stage_ft"])
U = np.log1p(hourly(ivu["discharge_cfs"]).clip(lower=0))
am = G[[10, 11, 12]].ffill(axis=1)[12]
am_u = U[[10, 11, 12]].ffill(axis=1)[12]
F = pd.DataFrame({
    "sofar": G[list(range(5, 13))].mean(axis=1) - am,
    "slope3": am - G[9], "slope6": am - G[6], "slope12": am - G[0],
    "up_slope3": am_u - U[9], "up_slope6": am_u - U[6], "up_slope12": am_u - U[0],
})
ab = pd.read_parquet(mt.SP + "oof/prod_s0.parquet")  # the blend's two members, a_h* (stage model) and b_h* (rating hybrid)
p = pd.DataFrame({f"h{h}": 0.5 * ab[f"a_h{h}"] + 0.5 * ab[f"b_h{h}"] for h in (1, 3, 5, 7)})
y = pd.DataFrame({c: stage.shift(-int(c[1:])) for c in p.columns}).reindex(p.index)
R = (y - p)
ix = R.index[(R.index >= "2007-10-01") & F.reindex(R.index).notna().all(axis=1).to_numpy() & R["h1"].notna().to_numpy()]
lines = [f"Review screens, {len(ix)} issue days {ix.min().date()}..{ix.max().date()} (perfect-rain OOF residuals)", "",
         "## A. correlation of candidate issue-time features with the OOF residual (Spearman)"]
C = pd.DataFrame({c: [F.loc[ix, f].corr(R.loc[ix, c], "spearman") for f in F] for c in R.columns}, index=F.columns)
lines += [C.round(3).to_string(), ""]

import lightgbm as lgb

rows = []
blocks = [(a, a + 2) for a in range(2007, 2024, 3)]
for c in R.columns:
    r = R.loc[ix, c]
    ok = r.notna()
    X, r = F.loc[ix][ok], r[ok]
    corr = pd.Series(0.0, index=r.index)
    for a, b in blocks:
        te = (X.index.year >= a) & (X.index.year <= b)
        if te.any():
            m = lgb.LGBMRegressor(objective="l1", num_leaves=7, min_child_samples=60, n_estimators=200, learning_rate=0.03, verbose=-1)
            corr[te] = m.fit(X[~te], r[~te]).predict(X[te])
    yy = y.loc[r.index, c]
    for sub, msk in (("all", pd.Series(True, index=r.index)), ("rising (dstage>=.05)", stage.diff().reindex(r.index) >= 0.05),
                     ("flood obs>=59", yy >= 59), ("low water", yy <= mt.LOW)):
        rows.append({"h": c, "subset": sub, "n": int(msk.sum()), "mae prod": r[msk].abs().mean(), "mae +features": (r - corr)[msk].abs().mean()})
T = pd.DataFrame(rows)
T["change %"] = 100 * (T["mae +features"] / T["mae prod"] - 1)
lines += ["Residual model on the candidates (LightGBM l1, 7 leaves, leave-3-year-block-out):", T.round(4).to_string(index=False), ""]

# ---------------- B
lines += ["## B. day-1 estimate as the target day is observed"]


def nowcast(p50: pd.Series, label: str):
    idx = p50.dropna().index
    idx = idx[(idx >= "2007-10-01")]
    y1 = stage.shift(-1).reindex(idx)
    out = []
    for cut in (12, 15, 18, 21, 24, 27):
        hrs = list(range(5, cut + 1))
        n = len(hrs)
        sofar = G[hrs].mean(axis=1).reindex(idx)
        last = G[hrs].ffill(axis=1)[cut].reindex(idx)
        ok = y1.notna() & sofar.notna() & last.notna() & (G[hrs].reindex(idx).notna().sum(axis=1) >= n - 1)
        naive = (n * sofar + (24 - n) * last) / 24
        X = pd.DataFrame({"p50": p50.reindex(idx), "sofar": sofar, "last": last})[ok]
        reg = pd.Series(np.nan, index=X.index)
        yrs = sorted(set(X.index.year))
        for i in range(0, len(yrs), 3):
            te = X.index.year.isin(yrs[i:i + 3])
            A = np.column_stack([np.ones((~te).sum()), X[~te].to_numpy()])
            beta, *_ = np.linalg.lstsq(A, y1[ok][~te].to_numpy(), rcond=None)
            reg[te] = np.column_stack([np.ones(te.sum()), X[te].to_numpy()]) @ beta
        e0, en, er = (p50.reindex(idx) - y1)[ok], (naive - y1)[ok], reg - y1[ok]
        rise = (y1 - stage.reindex(idx))[ok].abs() >= 0.10
        out.append({"cutoff UTC": f"{cut % 24:02d}Z" + ("+1" if cut >= 24 else ""), "EST": f"{(cut - 5) % 24 + 1:02d}:00", "hours seen": n, "n": int(ok.sum()),
                    "frozen 13Z": e0.abs().mean(), "naive": en.abs().mean(), "reg": er.abs().mean(),
                    "reg vs frozen %": 100 * (er.abs().mean() / e0.abs().mean() - 1),
                    "moving days: frozen": e0[rise].abs().mean(), "moving days: reg": er[rise].abs().mean(), "n moving": int(rise.sum())})
    return [label, pd.DataFrame(out).round(4).to_string(index=False), ""]


lines += nowcast(p["h1"], "perfect-rain OOF 2007-10..2023-12 (MAE ft; 'moving days' = |day-1 change| >= 0.10 ft):")
fcp = pd.read_parquet(mt.SP + "fc_prod.parquet")
lines += nowcast(fcp["h1"], "forecast mode (issued GEFS rain) 2020-10..2023-12:")
text = "\n".join(lines)
(OUT / "day1_screen_report.txt").write_text(text + "\n")
print(text)
