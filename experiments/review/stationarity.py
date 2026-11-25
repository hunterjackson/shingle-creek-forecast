"""Review check: is the 1979-2023 record one regime? (a) rating residual (stage - rating(Q, Toho)) by period for a rating
fitted on <= 2021 as production; (b) runoff ratio Q/P and GR4J log-flow error by period; (c) production OOF h1/h7 error by period."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
from pathlib import Path
import numpy as np, pandas as pd
import mt, fc
from shingle import rating, spec
from shingle.models import gr4j
OUT = Path(os.path.dirname(os.path.abspath(__file__)))
s = fc.state(); d = s["data"]
df = pd.concat([d["q"], d["stage"], d["toho"]], axis=1, keys=["q", "stage", "toho"]).dropna().loc[:"2023-12-31"]
tr = df.loc[:"2021-12-31"]
r = rating.fit_combined(tr.q, tr.stage, tr.toho, **spec.RATING_OPTS)
df["resid"] = df.stage - r.predict_stage(df.q.to_numpy(), df.toho.to_numpy())
P = fc.fcst._areal(d["pr"], s["drv"].weights)
qmm = gr4j.cfs_to_mm_day(d["q"], 466.2)
g = pd.DataFrame({"P": P, "q": qmm, "lerr": np.log1p(d["gr4j"]["q"]) - np.log1p(d["q"])}).loc[:"2023-12-31"]
ab = pd.read_parquet(mt.SP + "oof/prod_s0.parquet")
e = pd.DataFrame({f"h{h}": 0.5 * ab[f"a_h{h}"] + 0.5 * ab[f"b_h{h}"] - d["stage"].shift(-h).reindex(ab.index) for h in (1, 7)})
per = lambda ix: pd.cut(ix.year, [1978, 1986, 1994, 2002, 2010, 2018, 2023], labels=["1979-86", "1987-94", "1995-02", "2003-10", "2011-18", "2019-23"])
A = df.groupby(per(df.index), observed=True)["resid"].agg(rating_resid_mean="mean", rating_resid_mae=lambda x: x.abs().mean(), n="size")
lowq = df[df.q < 100]; hiq = df[df.q > 600]
A["resid mean Q<100"] = lowq.groupby(per(lowq.index), observed=True)["resid"].mean()
A["resid mean Q>600"] = hiq.groupby(per(hiq.index), observed=True)["resid"].mean()
gg = g.dropna()
B = gg.groupby(per(gg.index), observed=True).agg(rain_mm_yr=("P", lambda x: x.mean() * 365.25), flow_mm_yr=("q", lambda x: x.mean() * 365.25), gr4j_logerr_mean=("lerr", "mean"), gr4j_logerr_mae=("lerr", lambda x: x.abs().mean()))
B["runoff ratio"] = B.flow_mm_yr / B.rain_mm_yr
C = e.groupby(per(e.index), observed=True).agg(["mean", lambda x: x.abs().mean()]); C.columns = ["h1 bias", "h1 MAE", "h7 bias", "h7 MAE"]
text = "\n".join(["Stationarity check (rating fitted <= 2021; GR4J calibrated <= 2021)", "", A.round(3).to_string(), "", B.round(3).to_string(), "", "Production OOF error by period (perfect rain):", C.round(3).to_string()])
(OUT / "stationarity_report.txt").write_text(text + "\n"); print(text)
