"""ONE-SHOT final test of the frozen band candidate E2_calmprod vs production bands: 2024-01-01 .. 2026-11-12.

Frozen before running (selected on 1979-2023 OOF + 2020-23 forecast mode):
  point model: production spec fit on <= 2023-12-31 (7-day gap), GR4J calibrated <= 2021, GEFS raw h1-3 / scaled h4-7
  calibration residuals: forecast-mode OOF-by-year predictions 2020-10..2023-12 (fc_prod.parquet)
  production bands: |p50 - stage| 5-bin conformal
  E2_calmprod: calm days (|p50 - stage| < .05) -> |p50 - stage| bins (quartile edges) calibrated on calm days only;
               other days -> bins (tail edges .3/.55/.75/.88/.95/.98) on a learned |residual| scale; the scale model is
               LightGBM l1 trained on the perfect-rain OOF residuals of 1979-2019.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import mt, fc
from bands2 import EdgeConformal, TAIL_Q, pinball
from fc_bands import scale_models, FEATS
from shingle import config, evaluate, features, fcst, spec, train

TEST = (pd.Timestamp("2024-01-01"), pd.Timestamp("2026-11-12"))
HS = fc.HS

s = fc.state()
d = s["data"]
ix_all = d["stage"].dropna().index
tr = ix_all[ix_all <= pd.Timestamp(config.VAL_END) - pd.Timedelta(days=7)]
path = mt.SP + "fc_models/final_le2023.pkl"
if os.path.exists(path):
    import pickle; m = pickle.load(open(path, "rb"))
else:
    import pickle; m = spec.build_model(HS).fit(d, tr); pickle.dump(m, open(path, "wb"))
plain = fcst.build(s["raw"], s["src"], s["tgt"], s["drv"], d)
cal_ix = fc.fc_index()
scale = fcst.fit_scale(plain["pr"], d["pr"], config.RAIN_CELL, cal_ix)
f = fcst.build_mixed(s["raw"], s["src"], s["tgt"], s["drv"], d, scale, train.SCALED_FROM_H, HS)
dfc = dict(d, fcst=f)
issued = s["raw"].dropna(how="all").index
te_ix = ix_all[(ix_all >= TEST[0]) & (ix_all <= TEST[1]) & ix_all.isin(issued)]
pt = m.point(dfc, te_ix)
pt.to_parquet(mt.SP + "fc_final_point.parquet")
pc = pd.read_parquet(mt.SP + "fc_prod.parquet")
sc_by_year = fc.scales()
BH = (1, 3, 5, 7)
sms = scale_models(BH)
st = d["stage"]


def X_of(index, p, dd, h):
    X = features.build(dd, FEATS, horizon=h).loc[index]
    X["dp50"] = p[f"h{h}"] - st.loc[index]
    X["adp50"] = X["dp50"].abs()
    return X


rows = []
for h in BH:
    # calibration features: per year with that year's scale (as the OOF predictions were made)
    Xc = pd.concat([X_of(pc.index[pc.index.year == y], pc, dict(d, fcst=fcst.build_mixed(s["raw"], s["src"], s["tgt"], s["drv"], d, sc_by_year[y], train.SCALED_FROM_H, HS)), h)
                    for y in fc.YEARS]).loc[pc.index]
    Xt = X_of(te_ix, pt, dfc, h)
    rc = st.shift(-h).reindex(pc.index) - pc[f"h{h}"]
    rt = st.shift(-h).reindex(te_ix) - pt[f"h{h}"]
    okc, okt = rc.notna(), rt.notna()
    Xc, rc, Xt, rt = Xc[okc], rc[okc], Xt[okt], rt[okt]
    yt = st.shift(-h).reindex(rt.index)
    sc_c = sms[h].predict(Xc[sms[h].feature_name_])
    sc_t = sms[h].predict(Xt[sms[h].feature_name_])
    bc = evaluate.BinnedConformal.fit(rc, Xc["adp50"], 5, 0.8)
    plo, phi = bc.offsets(Xt["adp50"].to_numpy())
    cc, ct = (Xc["adp50"] < .05).to_numpy(), (Xt["adp50"] < .05).to_numpy()
    bcp = EdgeConformal.fit_edges(rc[cc], Xc["adp50"][cc], [.25, .5, .75], 0.8)
    bco = EdgeConformal.fit_edges(rc[~cc], sc_c[~cc], TAIL_Q, 0.8)
    (clo, chi), (olo, ohi) = bcp.offsets(Xt["adp50"].to_numpy()), bco.offsets(sc_t)
    elo, ehi = np.where(ct, clo, olo), np.where(ct, chi, ohi)
    fl = (yt >= 59).to_numpy()
    for k, lo, hi in (("production", plo, phi), ("E2_calmprod", elo, ehi)):
        r = rt.to_numpy()
        ins = (r >= lo) & (r <= hi)
        rows.append({"h": h, "bands": k, "cov": ins.mean(), "cov_flood": ins[fl].mean(), "n_flood": int(fl.sum()), "cov_calm": ins[ct].mean(),
                     "width": (hi - lo).mean(), "width_flood": (hi - lo)[fl].mean(), "width_calm": (hi - lo)[ct].mean(),
                     "pinball": (pinball(r, lo, .1) + pinball(r, hi, .9)) / 2, "pinball_flood": (pinball(r[fl], lo[fl], .1) + pinball(r[fl], hi[fl], .9)) / 2})
    pd.DataFrame({"p50": pt.loc[rt.index, f"h{h}"], "y": yt, "prod_lo": plo, "prod_hi": phi, "e2_lo": elo, "e2_hi": ehi}).to_parquet(mt.SP + f"fc_final_bands_h{h}.parquet")
R = pd.DataFrame(rows)
base = R[R.bands == "production"].set_index("h")
R["calm_width_pct"] = [100 * (w / base.loc[h, "width_calm"] - 1) for h, w in zip(R.h, R.width_calm)]
R["pinball_pct"] = [100 * (q / base.loc[h, "pinball"] - 1) for h, q in zip(R.h, R.pinball)]
R.to_csv(mt.SP + "fc_final_bands.csv", index=False)
print("point MAE test:", {k: round(v, 3) for k, v in fc.metrics(pt).items() if k.startswith("mae")})
pd.set_option("display.width", 250)
print(R.set_index(["h", "bands"]).round(3).to_string())
