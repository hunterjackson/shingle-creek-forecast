"""Why does a linear regression on the morning hours beat production's day-1 forecast at issue time? Decomposition.
All inputs are known by 13Z on t+1. 2007-10..2023-12, perfect-rain OOF (prod_s0) and forecast-mode OOF (fc_prod)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
import numpy as np, pandas as pd
import mt
from shingle import fetch
from pathlib import Path
OUT = Path(os.path.dirname(os.path.abspath(__file__)))
hist = fetch.load_history("data")
stage = hist["gauge"]["stage_ft"]
ivg = hist["iv"]["gauge"]
def hourly(s):
    s = s.dropna(); s.index = s.index.tz_convert("UTC").tz_localize(None); s = s.resample("1h").mean()
    days = pd.date_range(s.index.min().normalize(), s.index.max().normalize())
    return pd.DataFrame({k: pd.Series(s.reindex(days + pd.Timedelta(hours=k)).to_numpy(), index=days - pd.Timedelta(days=1)) for k in range(-12, 30)})
G = hourly(ivg["stage_ft"])
ab = pd.read_parquet(mt.SP + "oof/prod_s0.parquet")
P = pd.DataFrame({"blend": 0.5 * ab.a_h1 + 0.5 * ab.b_h1, "stage model": ab.a_h1, "rating hybrid": ab.b_h1})
idx = P.index[P.index >= "2007-10-01"]
y1 = stage.shift(-1).reindex(idx)
st = stage.reindex(idx)
am = G[[10, 11, 12]].ffill(axis=1)[12].reindex(idx)
sofar = G[list(range(5, 13))].mean(axis=1).reindex(idx)
h5 = G[5].reindex(idx)
# the daily mean of day t recomputed from hourly values (is the dv mean consistent with the hourly record?)
dv_from_iv = G[list(range(-19, 0)) if False else list(range(-12, 5))].mean(axis=1).reindex(idx)  # 12Z t .. 04Z t+1 (17 h of day t)
ok = y1.notna() & am.notna() & sofar.notna() & P.blend.reindex(idx).notna() & st.notna()
idx = idx[ok.to_numpy()]
y1, st, am, sofar = y1[idx], st[idx], am[idx], sofar[idx]
P = P.reindex(idx)

def cv(X):
    X = pd.DataFrame(X, index=idx); out = pd.Series(np.nan, index=idx); yrs = sorted(set(idx.year)); betas = []
    for i in range(0, len(yrs), 3):
        te = idx.year.isin(yrs[i:i + 3])
        A = np.column_stack([np.ones((~te).sum()), X[~te].to_numpy()])
        b, *_ = np.linalg.lstsq(A, (y1 - am)[~te].to_numpy(), rcond=None); betas.append(b)
        out[te] = am[te] + np.column_stack([np.ones(te.sum()), X[te].to_numpy()]) @ b
    return out, np.mean(betas, axis=0)

E = {"production blend": P.blend, "stage model alone": P["stage model"], "rating hybrid alone": P["rating hybrid"],
     "persist day-t mean": st, "persist morning reading": am, "naive (8 h seen + 16 h at morning reading)": (8 * sofar + 16 * am) / 24}
B = {}
for name, X in {"reg: morning + (blend - morning)": {"d": P.blend - am},
                "reg: + morning 8 h mean": {"d": P.blend - am, "s": sofar - am},
                "reg: no model at all (day-t mean, morning, 8 h mean)": {"t": st - am, "s": sofar - am}}.items():
    E[name], B[name] = cv(X)
moving = (y1 - st).abs() >= 0.10
big = (y1 - st).abs() >= 0.30
rows = []
for k, v in E.items():
    e = v - y1
    rows.append({"estimate": k, "MAE all": e.abs().mean(), "bias": e.mean(), f"moving >=.10 (n={int(moving.sum())})": e[moving].abs().mean(),
                 f"big >=.30 (n={int(big.sum())})": e[big].abs().mean(), "flood>=59": e[y1 >= 59].abs().mean(), "low water": e[y1 <= mt.LOW].abs().mean(), "p90 |err|": e.abs().quantile(.9)})
T = pd.DataFrame(rows).set_index("estimate")
lines = [f"Day-1 decomposition, {len(idx)} issue days 2007-10..2023-12, perfect-rain OOF", "", T.round(4).to_string(), "", "mean coefficients (intercept first; target = day-1 mean - morning reading):"]
lines += [f"  {k}: {np.round(v, 3)}" for k, v in B.items()]
e = P.blend - y1
lines += ["", "production error vs how far its forecast sits from the morning reading:",
          pd.DataFrame({"blend - morning": P.blend - am, "obs - morning": y1 - am}).describe().round(3).to_string(),
          f"corr(blend - morning, obs - morning) = {(P.blend - am).corr(y1 - am):.3f}; corr(error, blend - morning) = {e.corr(P.blend - am):.3f}",
          f"share of days where the morning reading is closer to the day-1 mean than production: {((am - y1).abs() < e.abs()).mean():.3f}"]
text = "\n".join(lines); (OUT / "day1_decomp_report.txt").write_text(text + "\n"); print(text)
