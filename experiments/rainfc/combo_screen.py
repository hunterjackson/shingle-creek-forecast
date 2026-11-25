"""Can several rain forecasts be combined into one better model input? Screen on the 219 NBM days (2020-10..2023-12).

The stage models are trained on observed rain, so forecasts enter as one 'expected rain' series. A small MOS-style
model predicts observed lead-1 basin rain from GEFS and NBM, fitted leave-one-year-out:
  lin   log1p(obs) ~ a + b log1p(GEFS) + c log1p(NBM)            (smearing back-transform)
  gate  GEFS amount x P(rain >= 1 mm | NBM), logistic in log1p(NBM) (NBM decides whether, GEFS how much)
The combined basin total is spread over the cells with the GEFS pattern (NBM's where GEFS is ~dry), then brought to
GEFS level (x sum GEFS / sum combo, LOYO) because raw-level rain is what the h1-3 models do best with (T4).
Scored as rain (vs gridMET) and as stage forecasts (forecast-mode OOF models, as model_screen.py).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
import numpy as np
import pandas as pd

import fc
import mt
from shingle import fcst, train

OUT = Path(os.path.dirname(os.path.abspath(__file__)))
pd.set_option("display.width", 250)
s = fc.state()
d, drv, tgt = s["data"], s["drv"], s["tgt"]
w = drv.weights
ms, sc = fc.models(), fc.scales()
HS = fc.HS

nbm = pd.concat([pd.read_parquet(p) for p in sorted((OUT / "cache" / "nbm").glob("*.parquet"))]).sort_index()
gcell = fcst.build(s["raw"], s["src"], tgt, drv, d)["pr"][1]
ix = fc.fc_index()
ix = ix[ix.isin(nbm.dropna(how="all").index)]
G = fcst._areal(gcell, w).reindex(ix)
N = fcst._areal(nbm, w).reindex(ix)
O = fcst._areal(d["pr"].shift(-1), w).reindex(ix)


def fit_lin(tr):
    X = np.column_stack([np.ones(len(tr)), np.log1p(G[tr]), np.log1p(N[tr])])
    beta, *_ = np.linalg.lstsq(X, np.log1p(O[tr]), rcond=None)
    smear = np.mean(np.expm1(np.log1p(O[tr]) - X @ beta + np.log1p(1)) ) if False else np.mean(np.exp(np.log1p(O[tr]) - X @ beta))
    return lambda t: np.maximum(0, smear * np.exp(np.column_stack([np.ones(len(t)), np.log1p(G[t]), np.log1p(N[t])]) @ beta) - 1)


def fit_gate(tr):
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression().fit(np.log1p(N[tr]).to_numpy()[:, None], (O[tr] >= 1).astype(int))
    return lambda t: G[t].to_numpy() * lr.predict_proba(np.log1p(N[t]).to_numpy()[:, None])[:, 1]


combos = {}
for name, fitter in (("lin", fit_lin), ("gate", fit_gate)):
    out = pd.Series(np.nan, index=ix)
    for y in fc.YEARS:
        te, tr = ix[ix.year == y], ix[ix.year != y]
        if len(te) == 0:
            continue
        f = fitter(tr)
        k = G[tr].sum() / pd.Series(f(tr), index=tr).sum()  # bring to GEFS level
        out[te] = f(te) * k
    combos[name] = out

# rain skill
rows = {}
for k, f in {"gefs": G, "nbm": N * (G.sum() / N.sum()), **combos}.items():
    rows[k] = {"spearman": f.corr(O, "spearman"), "r_log": np.log1p(f).corr(np.log1p(O)), "mae(GEFS-level)": (f - O * G.sum() / O.sum()).abs().mean(),
               "heavy obs>=20: fc/obs": f[O >= 20].sum() / O[O >= 20].sum(), "busts (G>=20,O<10): mean fc": f[(G >= 20) & (O < 10)].mean()}
lines = [f"Combined rain forecasts, {len(ix)} NBM days, lead 1 basin rain", "", pd.DataFrame(rows).T.round(3).to_string(), ""]


def lead1_cells(total: pd.Series) -> pd.DataFrame:
    pat = gcell.reindex(ix)
    gsum = fcst._areal(pat, w)
    use_n = gsum < 0.5
    pat = pat.where(~use_n, nbm.reindex(ix)[pat.columns])
    psum = fcst._areal(pat, w).replace(0, np.nan)
    return pat.mul(total / psum, axis=0).fillna(0.0)


def inputs(lead1: pd.DataFrame, scale: dict) -> dict:
    parts = {}
    for key, scl in (("plain", None), ("scaled", scale)):
        pr = fcst.build(s["raw"], s["src"], tgt, drv, d, scale=scl)["pr"].copy()
        l1 = lead1.reindex(pr.index) * (scl.get(1, 1.0) if scl else 1.0)
        for c in pr[1].columns:
            have = l1[c].notna()
            pr.loc[have, (1, c)] = l1.loc[have, c]
        areal = pd.DataFrame({k: fcst._areal(pr[k], drv.weights) for k in HS}, index=pr.index)
        parts[key] = {"pr": pr, "gr4j_q": drv.forecast(d, areal)}
    return dict(parts["plain"], by_h={h: parts["scaled"] for h in HS if h >= train.SCALED_FROM_H})


preds = {"gefs": pd.read_parquet(mt.SP + "fc_prod.parquet").loc[ix]}
for name, total in combos.items():
    l1 = lead1_cells(total)
    parts = [ms[y].point(dict(d, fcst=inputs(l1, sc[y])), ix[ix.year == y]) for y in fc.YEARS if (ix.year == y).any()]
    preds[name] = pd.concat(parts).sort_index()

yt = fc.targets(ix)
res = []
for h in (1, 2, 3, 4, 7):
    for sub, m in (("all", pd.Series(True, index=ix)), ("heavy (G or O >= 20)", (G >= 20) | (O >= 20)), ("GEFS busts", (G >= 20) & (O < 10))):
        for k, p in preds.items():
            e = (p[f"h{h}"] - yt[f"h{h}"])[m].dropna()
            res.append({"h": h, "subset": sub, "src": k, "n": len(e), "mae": e.abs().mean()})
R = pd.DataFrame(res)
M = R.pivot_table(index=["subset", "h"], columns="src", values="mae")[list(preds)]
lines += ["Stage MAE:", M.round(4).to_string(), "", "change vs GEFS (%):", (100 * (M.div(M["gefs"], axis=0) - 1)).drop(columns="gefs").round(1).to_string(), ""]
for k in combos:
    lines += [f"{k} vs gefs, all days:", fc.boot(preds[k], preds["gefs"], hs=(1, 2, 3, 4, 7)).round(4).to_string(), ""]
text = "\n".join(lines)
(OUT / "combo_screen_report.txt").write_text(text + "\n")
print(text)
