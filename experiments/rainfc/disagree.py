"""Is GEFS-vs-NBM disagreement a signal the likely range misses? (219 NBM days, 2020-10..2023-12)

Uses the T2 leave-one-year-out bands (experiments/divergence/t2_h{h}.parquet, production 'prod' and adopted
'wet_add') and the forecast-mode residuals. Disagreement D = |log1p(GEFS) - log1p(k * NBM)| of lead-1 basin rain,
NBM brought to GEFS level (k = sum GEFS / sum NBM on these days). Coverage by D tercile, and Spearman(|resid|, D)
on wet days.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
import numpy as np
import pandas as pd

import fc
from shingle import fcst

OUT = Path(os.path.dirname(os.path.abspath(__file__)))
s = fc.state()
w = s["drv"].weights
nbm = pd.concat([pd.read_parquet(p) for p in sorted((OUT / "cache" / "nbm").glob("*.parquet"))]).sort_index()
g = fcst._areal(fcst.build(s["raw"], s["src"], s["tgt"], s["drv"], s["data"])["pr"][1], w).reindex(nbm.index)
n = fcst._areal(nbm, w)
k = g.sum() / n.sum()
D = (np.log1p(g) - np.log1p(k * n)).abs()
lines = [f"GEFS vs NBM disagreement and likely-range coverage, {len(D)} days (k={k:.2f})", ""]
for h in (1, 3):
    b = pd.read_parquet(Path(__file__).parents[1] / "divergence" / f"t2_h{h}.parquet").reindex(D.index).dropna(subset=["r"])
    d = D.reindex(b.index)
    q = pd.qcut(d, 3, labels=["low", "mid", "high"])
    for band in ("prod", "wet_add"):
        ins = (b["r"] >= b[f"{band}_lo"]) & (b["r"] <= b[f"{band}_hi"])
        cov = ins.groupby(q, observed=True).mean()
        wid = (b[f"{band}_hi"] - b[f"{band}_lo"]).groupby(q, observed=True).mean()
        lines.append(f"h{h} {band:8s} coverage by D tercile: " + "  ".join(f"{t} {cov[t]:.0%} (w {wid[t]:.2f})" for t in cov.index))
    wet = b["rain"] >= 5
    lines.append(f"h{h} wet days (n={int(wet.sum())}): Spearman(|resid|, D) = {b['r'][wet].abs().corr(d[wet], 'spearman'):+.3f}")
    lines.append("")
text = "\n".join(lines)
(OUT / "disagree_report.txt").write_text(text + "\n")
print(text)
