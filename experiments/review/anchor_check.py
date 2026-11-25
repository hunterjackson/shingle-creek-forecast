"""Does anchoring on the morning reading also help in forecast mode and beyond day 1?
est = morning + b1 * (production p50 - morning) + b2 * (mean of the target day's first 8 h - morning), coefficients fitted
leave-3-year-block-out (perfect-rain OOF 2007-10..2023) or leave-one-year-out (forecast mode 2020-10..2023-12)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
from pathlib import Path
import numpy as np, pandas as pd
import mt, fc
from shingle import fetch
OUT = Path(os.path.dirname(os.path.abspath(__file__)))
hist = fetch.load_history("data")
stage = hist["gauge"]["stage_ft"]
s = hist["iv"]["gauge"]["stage_ft"].dropna(); s.index = s.index.tz_convert("UTC").tz_localize(None); s = s.resample("1h").mean()
days = pd.date_range(s.index.min().normalize(), s.index.max().normalize())
G = pd.DataFrame({k: pd.Series(s.reindex(days + pd.Timedelta(hours=k)).to_numpy(), index=days - pd.Timedelta(days=1)) for k in range(5, 13)})
am_all = G[[10, 11, 12]].ffill(axis=1)[12]
sofar_all = G.mean(axis=1)
ab = pd.read_parquet(mt.SP + "oof/prod_s0.parquet")
sets = {"perfect-rain OOF 2007-10..2023": (pd.DataFrame({f"h{h}": 0.5 * ab[f"a_h{h}"] + 0.5 * ab[f"b_h{h}"] for h in (1, 3, 5, 7)}).loc["2007-10-01":], 3),
        "forecast mode (GEFS) 2020-10..2023": (pd.read_parquet(mt.SP + "fc_prod.parquet"), 1)}
lines = []
for label, (P, blk) in sets.items():
    rows = []
    for c in P.columns:
        h = int(c[1:])
        idx = P.index
        y, st, am, sf, p = stage.shift(-h).reindex(idx), stage.reindex(idx), am_all.reindex(idx), sofar_all.reindex(idx), P[c]
        ok = y.notna() & am.notna() & sf.notna() & p.notna() & st.notna()
        y, st, am, sf, p = y[ok], st[ok], am[ok], sf[ok], p[ok]
        ix = y.index; est = pd.Series(np.nan, index=ix); yrs = sorted(set(ix.year)); bs = []
        X = np.column_stack([np.ones(len(ix)), (p - am).to_numpy(), (sf - am).to_numpy()])
        for i in range(0, len(yrs), blk):
            te = ix.year.isin(yrs[i:i + blk])
            b, *_ = np.linalg.lstsq(X[~te], (y - am)[~te].to_numpy(), rcond=None); bs.append(b)
            est[te] = am[te] + X[te] @ b
        e0, e1, ea = p - y, est - y, am - y
        mv = (y - st).abs() >= 0.10 * h ** 0.5
        d = {"h": h, "n": len(ix), "prod": e0.abs().mean(), "morning persisted": ea.abs().mean(), "anchored": e1.abs().mean(),
             "change %": 100 * (e1.abs().mean() / e0.abs().mean() - 1), "moving: prod": e0[mv].abs().mean(), "moving: anchored": e1[mv].abs().mean(),
             "flood: prod": e0[y >= 59].abs().mean(), "flood: anchored": e1[y >= 59].abs().mean(), "n flood": int((y >= 59).sum()),
             "low: prod": e0[y <= mt.LOW].abs().mean(), "low: anchored": e1[y <= mt.LOW].abs().mean(), "b(p50-am)": np.mean(bs, 0)[1], "b(8h-am)": np.mean(bs, 0)[2]}
        if "forecast" in label:
            bt = fc.boot(pd.DataFrame({c: est}), pd.DataFrame({c: p}), hs=(h,))
            d["CI lo"], d["CI hi"] = bt.iloc[0].get("lo90", np.nan), bt.iloc[0].get("hi90", np.nan)
        rows.append(d)
    lines += [label, pd.DataFrame(rows).round(4).to_string(index=False), ""]
text = "\n".join(lines); (OUT / "anchor_check_report.txt").write_text(text + "\n"); print(text)
