"""C1: alternative rainfall-runoff simulators, calibrated <=2021 like the cached GR4J; saved as gr4j-shaped frames."""
import sys, os, time, json; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import mt, gr6j
from scipy.optimize import differential_evolution
from shingle import gr4j_driver
from shingle.models import gr4j
d = mt.load()
w = d["_cells"]["basin_frac"][d["_cells"]["basin_frac"] > 0].to_dict()
drv = gr4j_driver.GR4JDriver(w)
P, E = drv._inputs(d)
ok = P.notna() & E.notna()
P, E = P[ok], E[ok]
qobs = pd.Series(gr4j.cfs_to_mm_day(d["q"], 466.2)).reindex(P.index)
cal = (P.index <= "2021-12-31")
WU = 365
def obj_fn(kind, sim, obs):
    m = np.isfinite(obs)
    s, o = sim[m], obs[m]
    if kind == "nse": return gr4j.nse(s, o)
    if kind == "sqrt": return gr4j.nse(np.sqrt(s), np.sqrt(o))
    if kind == "log": return gr4j.nse(np.log(s + 0.01), np.log(o + 0.01))
Pc, Ec, Oc = P[cal].to_numpy(), E[cal].to_numpy(), qobs[cal].to_numpy()
def fit(model, kind, bounds):
    if model == "gr4j":
        f = lambda x: -obj_fn(kind, gr4j.run(Pc, Ec, gr4j.Params(*x))["q"][WU:], Oc[WU:])
    else:
        f = lambda x: -obj_fn(kind, gr6j.run(Pc, Ec, x)["q"][WU:], Oc[WU:])
    r = differential_evolution(f, bounds, seed=0, maxiter=150, tol=1e-7, polish=True)
    return r.x, -r.fun
def simulate(model, x):
    if model == "gr4j":
        o = gr4j.run(P.to_numpy(), E.to_numpy(), gr4j.Params(*x))
    else:
        o = gr6j.run(P.to_numpy(), E.to_numpy(), x)
    return pd.DataFrame({"q": gr4j.mm_day_to_cfs(o["q"], 466.2), "s": o["s"], "r": o["r"]}, index=P.index).reindex(d["stage"].index)
G4W = [(1.0, 2500.0), (-40.0, 5.0), (5.0, 2000.0), (0.5, 10.0)]
variants = {"gr4j_sqrt": ("gr4j", "sqrt", gr4j.BOUNDS), "gr4j_log": ("gr4j", "log", gr4j.BOUNDS), "gr4j_wide": ("gr4j", "nse", G4W),
            "gr6j": ("gr6j", "nse", gr6j.BOUNDS), "gr6j_sqrt": ("gr6j", "sqrt", gr6j.BOUNDS)}
os.makedirs(mt.SIM_DIR, exist_ok=True)
info = {}
base = d["gr4j"].reindex(d["stage"].index)
val = (d["stage"].index >= "2022-01-01") & (d["stage"].index <= "2023-12-31")
def nses(sim):
    o = d["q"].to_numpy(); s = sim["q"].to_numpy()
    tr = (d["stage"].index <= "2021-12-31") & (d["stage"].index >= "1980-01-01")
    return {"nse_cal": round(gr4j.nse(s[tr], o[tr]), 3), "nse_val": round(gr4j.nse(s[val], o[val]), 3),
            "lognse_val": round(gr4j.nse(np.log(s[val] + 1), np.log(o[val] + 1)), 3),
            "bias_val": round(float(np.nanmean(s[val]) / np.nanmean(o[val]) - 1), 3)}
info["gr4j"] = {"params": "cached x1 14.7 x2 -10 x3 674 x4 3.29", **nses(base)}
print("gr4j", info["gr4j"], flush=True)
for name, (model, kind, b) in variants.items():
    t = time.time()
    x, sc = fit(model, kind, b)
    s = simulate(model, x)
    s.to_parquet(mt.SIM_DIR + f"{name}.parquet")
    info[name] = {"params": [round(float(v), 3) for v in x], "obj": round(float(sc), 3), **nses(s)}
    print(name, f"{time.time()-t:.0f}s", info[name], flush=True)
json.dump(info, open(mt.SP + "sims_info.json", "w"), indent=1)
