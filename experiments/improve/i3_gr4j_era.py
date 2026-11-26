"""Idea 3b: GR4J calibrated on recent years (2008..2021) instead of 1979..2021, as the runoff ratio rose ~1/3.
Same objective (NSE) and bounds as production; the simulated states feed the production spec via mt cfg 'sim'.
OOF leave-3-year-block-out; scored 2007-10..2023 and 2011..2023 (an era model is only meant for recent years)."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
from pathlib import Path
import numpy as np, pandas as pd
import mt
from scipy.optimize import differential_evolution
from shingle import gr4j_driver
from shingle.models import gr4j

OUT = Path(os.path.dirname(os.path.abspath(__file__)))
if __name__ == "__main__":
    d = mt.load()
    if not os.path.exists(mt.SIM_DIR + "gr4j_recent.parquet"):
        w = d["_cells"]["basin_frac"][d["_cells"]["basin_frac"] > 0].to_dict()
        P, E = gr4j_driver.GR4JDriver(w)._inputs(d)
        ok = P.notna() & E.notna(); P, E = P[ok], E[ok]
        qobs = pd.Series(gr4j.cfs_to_mm_day(d["q"], 466.2)).reindex(P.index)
        cal = (P.index >= "2007-01-01") & (P.index <= "2021-12-31")  # 2007 = warm-up year
        Pc, Ec, Oc = P[cal].to_numpy(), E[cal].to_numpy(), qobs[cal].to_numpy()
        def f(x):
            s = gr4j.run(Pc, Ec, gr4j.Params(*x))["q"][365:]; o = Oc[365:]; m = np.isfinite(o)
            return -gr4j.nse(s[m], o[m])
        r = differential_evolution(f, gr4j.BOUNDS, seed=0, maxiter=150, tol=1e-7, polish=True)
        o = gr4j.run(P.to_numpy(), E.to_numpy(), gr4j.Params(*r.x))
        os.makedirs(mt.SIM_DIR, exist_ok=True)
        pd.DataFrame({"q": gr4j.mm_day_to_cfs(o["q"], 466.2), "s": o["s"], "r": o["r"]}, index=P.index).reindex(d["stage"].index).to_parquet(mt.SIM_DIR + "gr4j_recent.parquet")
        json.dump({"params": [float(v) for v in r.x], "nse_2008_21": -r.fun}, open(OUT / "i3_gr4j_params.json", "w"))
    cand = mt.blend(mt.oof("gr4j_recent", {"sim": "gr4j_recent"}))
    prod = mt.blend(pd.read_parquet(mt.SP + "oof/prod_s0.parquet"))
    sim = mt.sim("gr4j_recent"); base = mt.sim("gr4j")
    lines = [f"GR4J recent params {json.load(open(OUT / 'i3_gr4j_params.json'))}"]
    for lab, s in (("prod", base), ("recent", sim)):
        q, o = s["q"].loc["2011":"2023"], d["q"].loc["2011":"2023"]
        lines.append(f"{lab}: 2011-23 log-flow bias {np.nanmean(np.log1p(q) - np.log1p(o)):+.3f}, NSE {gr4j.nse(q.to_numpy(), o.to_numpy()):.3f}")
    for span, start in (("2007-10..2023", "2007-10-01"), ("2011..2023", "2011-01-01")):
        mc, mp_ = mt.metrics(cand.loc[start:]), mt.metrics(prod.loc[start:])
        bt = mt.boot_diff(cand.loc[start:], prod.loc[start:])
        lines += ["", span, pd.DataFrame({"prod": mp_, "recent": mc}).T.round(4).to_string(), bt.round(4).to_string()]
    text = "\n".join(lines); (OUT / "i3_gr4j_report.txt").write_text(text + "\n"); print(text)
