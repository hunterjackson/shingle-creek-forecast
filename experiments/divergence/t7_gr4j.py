"""T7 screen (docs/divergence_test_plan.md): does GR4J's flow error at issue time explain forecast residuals?

On 11-24 GR4J simulated 875 cfs against 621 observed (+40 %). GR4J only feeds features (gr4j_logq, gr4j_dH, stores);
the models also see observed log_q, so they could in principle learn the error, but not explicitly.
Screen: Spearman correlation of err = log1p(gr4j_q) - log1p(q_obs) at t with signed residuals (obs - p50), overall
and on wet-forecast days, for forecast-mode OOF 2020-10..2023-12 and perfect-rain OOF 1979-2023 (prod_s0 blend).
|r| < ~0.1 -> drop.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
import numpy as np
import pandas as pd

import fc
import mt
from shingle import fcst

OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
s = fc.state()
d = s["data"]
err = np.log1p(d["gr4j"]["q"]) - np.log1p(d["q"].clip(lower=0))
lines = ["T7 screen: Spearman r of GR4J log-flow error at t with residual (obs - p50)", ""]

p = pd.read_parquet(mt.SP + "fc_prod.parquet")
plain = fcst.build(s["raw"], s["src"], s["tgt"], s["drv"], d)
areal = pd.DataFrame({k: fcst._areal(plain["pr"][k], s["drv"].weights) for k in fc.HS}).reindex(p.index)
y = fc.targets(p.index)
lines.append(f"GR4J log error at t, 2020-23: median {err.reindex(p.index).median():+.3f}, IQR {err.reindex(p.index).quantile(.25):+.3f}..{err.reindex(p.index).quantile(.75):+.3f}")
for h in (1, 2, 3, 5, 7):
    r = y[f"h{h}"] - p[f"h{h}"]
    wet = areal[list(range(1, h + 1))].sum(axis=1) >= 5
    e = err.reindex(p.index)
    c_all = pd.concat([e, r], axis=1).dropna().corr("spearman").iloc[0, 1]
    c_wet = pd.concat([e[wet], r[wet]], axis=1).dropna().corr("spearman").iloc[0, 1]
    lines.append(f"forecast-mode h{h}: all r={c_all:+.3f}   wet>=5 r={c_wet:+.3f} (n={int(wet.sum())})")

oof = mt.blend(pd.read_parquet(mt.SP + "oof/prod_s0.parquet"))
yo = mt.targets(oof.index)
for h in mt.H:
    r = yo[f"h{h}"] - oof[f"h{h}"]
    c = pd.concat([err.reindex(oof.index), r], axis=1).dropna().corr("spearman").iloc[0, 1]
    lines.append(f"perfect-rain OOF 1979-2023 h{h}: r={c:+.3f}")
text = "\n".join(lines)
open(OUT + "t7_report.txt", "w").write(text + "\n")
print(text)
