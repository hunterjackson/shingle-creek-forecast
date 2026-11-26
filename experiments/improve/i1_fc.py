"""Idea 1 in forecast mode (issued GEFS rain), 2020-10..2023-12, leave-one-year-out like fc.py.
Day-1 tree on y - am (rows with a morning reading), production stage features + sofar_d."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
from pathlib import Path
import numpy as np, pandas as pd
import fc, mt
from shingle import fcst, features, fetch, spec, train

OUT = Path(os.path.dirname(os.path.abspath(__file__)))
S = fc.state(); D = S["data"]
hist = fetch.load_history("data")
s = hist["iv"]["gauge"]["stage_ft"].dropna(); s.index = s.index.tz_convert("UTC").tz_localize(None); s = s.resample("1h").mean()
days = pd.date_range(s.index.min().normalize(), s.index.max().normalize())
G = pd.DataFrame({k: pd.Series(s.reindex(days + pd.Timedelta(hours=k)).to_numpy(), index=days - pd.Timedelta(days=1)) for k in range(5, 13)})
IX = D["stage"].index
AM, SOFAR, STAGE = D["stage_am"], G.mean(axis=1).reindex(IX), D["stage"]
Y = STAGE.shift(-1)
FEATS = spec.feature_sets()["stage"] + spec.AM_STAGE


def X_of(d):
    X = features.build(d, FEATS, horizon=1)
    X["sofar_d"] = SOFAR - AM
    return X


XO = X_of(D)
sc = fc.scales()
ix_all = fc.all_index()
held_ix = fc.fc_index()
P = {}
import lightgbm as lgb
for obj in ("regression", "l1"):
    parts = []
    for y in fc.YEARS:
        st, en = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
        near = ((ix_all < st) & (ix_all >= st - pd.Timedelta(days=7))) | ((ix_all > en) & (ix_all <= en + pd.Timedelta(days=7)))
        tr = ix_all[(ix_all.year != y) & ~near]
        dy = (Y - AM).loc[tr]
        ok = dy.notna() & SOFAR.loc[tr].notna() & XO.loc[tr, spec.feature_sets()["stage"]].notna().all(axis=1)
        m = lgb.LGBMRegressor(**dict(spec.LGBM_PARAMS, random_state=0, verbose=-1, objective=obj)).fit(XO.loc[tr][ok].to_numpy(), dy[ok].to_numpy())
        f = fcst.build_mixed(S["raw"], S["src"], S["tgt"], S["drv"], D, sc[y], train.SCALED_FROM_H, fc.HS)
        h = held_ix[held_ix.year == y]
        Xh = X_of(dict(D, fcst=f)).loc[h]
        parts.append(pd.Series(AM.loc[h].to_numpy() + m.predict(Xh.to_numpy()), index=h))
    P[obj] = pd.concat(parts)
prod = pd.read_parquet(mt.SP + "fc_prod.parquet")["h1"]
ix = prod.index[AM.reindex(prod.index).notna().to_numpy() & SOFAR.reindex(prod.index).notna().to_numpy() & Y.reindex(prod.index).notna().to_numpy()]
y = Y[ix]
rows = []
for name, p in {"prod": prod, "persist am": AM, "arx_am (L2)": P["regression"], "arx_am (L1)": P["l1"]}.items():
    p = p.reindex(ix); e = p - y
    bt = fc.boot(pd.DataFrame({"h1": p}), pd.DataFrame({"h1": prod.reindex(ix)}), hs=(1,)).iloc[0]
    rows.append({"model": name, "MAE": e.abs().mean(), "vs prod %": 100 * (e.abs().mean() / (prod[ix] - y).abs().mean() - 1),
                 "lo90": bt.get("lo90"), "hi90": bt.get("hi90"), "moving": e[(y - STAGE[ix]).abs() >= .1].abs().mean(),
                 f"flood (n={int((y >= 59).sum())})": e[y >= 59].abs().mean(), "low": e[y <= mt.LOW].abs().mean()})
text = f"Idea 1, forecast mode (issued GEFS), {len(ix)} issue days 2020-10..2023-12, leave-one-year-out\n\n" + pd.DataFrame(rows).set_index("model").round(4).to_string()
pd.DataFrame({k: v.reindex(ix) for k, v in P.items()}).to_parquet(OUT / "i1_fc.parquet")
(OUT / "i1_fc_report.txt").write_text(text + "\n"); print(text)
