"""Idea 2: update today's figure as the target day is observed. Reruns happen every 3 h (13:30Z, 16:30Z, ...).

At a run whose latest complete hour is C (UTC label), hours 05Z..C of the target day (n of 24) are observed.
  frozen   the issue-time anchored forecast (MorningAnchor OOF, experiments/improve/i1_*)
  naive    (n * mean seen + (24 - n) * latest reading) / 24
  blend    remaining hours = latest + a_n * (implied - latest), implied = the frozen forecast's implied mean for the
           hours not seen at issue, (24 * p50 - 8 * mean 05..12Z) / 16; a_n fitted on other 3-year blocks
Scored against the gauge's daily mean (perfect-rain OOF 2007-10..2023, and forecast mode 2020-10..2023).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
from pathlib import Path
import numpy as np, pandas as pd
import mt
from shingle import fetch

OUT = Path(os.path.dirname(os.path.abspath(__file__)))
hist = fetch.load_history("data")
stage = hist["gauge"]["stage_ft"]
s = hist["iv"]["gauge"]["stage_ft"].dropna(); s.index = s.index.tz_convert("UTC").tz_localize(None); s = s.resample("1h").mean()
days = pd.date_range(s.index.min().normalize(), s.index.max().normalize())
G = pd.DataFrame({k: pd.Series(s.reindex(days + pd.Timedelta(hours=k)).to_numpy(), index=days - pd.Timedelta(days=1)) for k in range(5, 29)})
sets = {"perfect-rain OOF 2007-10..2023": (pd.read_parquet(OUT / "i1_oof.parquet")["arx_am_l1"], 3),
        "forecast mode 2020-10..2023": (pd.read_parquet(OUT / "i1_fc.parquet")["l1"], 1)}
lines = []
for label, (p50, blk) in sets.items():
    rows = []
    for cut in (15, 18, 21, 24, 27):  # latest complete hour label at the 16:30Z, 19:30Z, 22:30Z, 01:30Z, 04:30Z runs
        hrs = list(range(5, cut + 1)); n = len(hrs)
        idx = p50.dropna().index
        y = stage.shift(-1).reindex(idx)
        seen = G[hrs].mean(axis=1).reindex(idx)
        last = G[hrs].ffill(axis=1)[cut].reindex(idx)
        s8 = G[list(range(5, 13))].mean(axis=1).reindex(idx)
        ok = y.notna() & seen.notna() & last.notna() & s8.notna() & G[hrs].notna().sum(axis=1).reindex(idx).ge(n - 2)
        idx = idx[ok.to_numpy()]
        y, seen, last, s8, p = y[idx], seen[idx], last[idx], s8[idx], p50[idx]
        implied = (24 * p - 8 * s8) / 16
        naive = (n * seen + (24 - n) * last) / 24
        est = pd.Series(np.nan, index=idx); yrs = sorted(set(idx.year)); al = []
        # y*24 - n*seen - (24-n)*last = (24-n) * a * (implied - last)
        lhs = 24 * y - n * seen - (24 - n) * last; x = (24 - n) * (implied - last)
        for i in range(0, len(yrs), blk):
            te = idx.year.isin(yrs[i:i + blk])
            a = float((x[~te] * lhs[~te]).sum() / (x[~te] ** 2).sum()); al.append(a)
            est[te] = naive[te] + x[te] * a / 24
        mv = (y - stage.reindex(idx)).abs() >= 0.10
        r = {"run (UTC)": f"{(cut + 1) % 24:02d}:30", "EST": f"{(cut - 4) % 24:02d}:30", "hours seen": n, "n": len(idx)}
        for k, v in {"frozen": p, "naive": naive, "blend": est}.items():
            r[k] = (v - y).abs().mean()
            r[f"{k} moving"] = (v - y)[mv].abs().mean()
        r["blend vs frozen %"] = 100 * (r["blend"] / r["frozen"] - 1); r["a"] = np.mean(al)
        rows.append(r)
    lines += [label, pd.DataFrame(rows).round(4).to_string(index=False), ""]
text = "\n".join(lines); (OUT / "i2_report.txt").write_text(text + "\n"); print(text)
