"""Forecast-mode check of the tuned likely range (E2_mondrian) vs production bands, GEFS era 2020-10..2023-12.

Production: |p50 - stage| 5-bin conformal, calibrated leave-one-year-out on forecast-mode residuals (as train.py).
E2_mondrian: |residual| model (LightGBM l1) trained on the perfect-rain OOF residuals of 1979-2019 (no overlap with
2020-23); its inputs are computed from the issued forecasts here. Calibrated leave-one-year-out on forecast-mode
residuals, separately for calm (|p50 - stage| < .05: 4 scale bins) and other days (tail-edge bins).
E2_calmprod: as E2_mondrian on other days; calm days use |p50 - stage| bins calibrated on calm days only.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb
import mt, fc
from bands2 import EdgeConformal, TAIL_Q, QP, pinball
from shingle import evaluate, features, fcst, train

C = mt.CELL
FEATS = ["stage", "dstage_1", "log_q", "toho_dev", f"pr_nextH__{C}", f"pr_past3__{C}", "gr4j_dH", "gr4j_s", "doy_sin", "doy_cos"]


def scale_models(hs):
    """|residual| models from perfect-rain OOF (1979-2019), production seed-0 blend."""
    d = mt.load()
    oof = mt.blend(pd.read_parquet(mt.SP + "oof/prod_s0.parquet"))
    ix = oof.index[oof.index.year <= 2019]
    y = mt.targets(ix)
    out = {}
    for h in hs:
        X = features.build(d, FEATS, horizon=h).loc[ix]
        X["dp50"] = oof.loc[ix, f"h{h}"] - d["stage"].loc[ix]
        X["adp50"] = X["dp50"].abs()
        r = (y[f"h{h}"] - oof.loc[ix, f"h{h}"])
        ok = r.notna() & X["adp50"].notna()
        out[h] = lgb.LGBMRegressor(objective="l1", **QP).fit(X[ok], r[ok].abs())
    return out


def fc_features(p, h, sc):
    s = fc.state()
    parts = []
    for y in fc.YEARS:
        held = p.index[p.index.year == y]
        f = fcst.build_mixed(s["raw"], s["src"], s["tgt"], s["drv"], s["data"], sc[y], train.SCALED_FROM_H, fc.HS)
        parts.append(features.build(dict(s["data"], fcst=f), FEATS, horizon=h).loc[held])
    X = pd.concat(parts).loc[p.index]
    X["dp50"] = p[f"h{h}"] - s["data"]["stage"].loc[p.index]
    X["adp50"] = X["dp50"].abs()
    return X


def run(p, hs=(1, 3, 5, 7)):
    sc = fc.scales()
    sms = scale_models(hs)
    y = fc.targets(p.index)
    rows = []
    for h in hs:
        X = fc_features(p, h, sc)
        r = y[f"h{h}"] - p[f"h{h}"]
        ok = r.notna()
        X, r, yy = X[ok], r[ok], y[f"h{h}"][ok]
        s = pd.Series(sms[h].predict(X[sms[h].feature_name_]), index=X.index)
        LO = {k: pd.Series(np.nan, index=r.index) for k in ("production", "E2_mondrian", "E2_calmprod")}
        HI = {k: pd.Series(np.nan, index=r.index) for k in ("production", "E2_mondrian", "E2_calmprod")}
        for yr in fc.YEARS:
            te, tr = r.index[r.index.year == yr], r.index[r.index.year != yr]
            bc = evaluate.BinnedConformal.fit(r.loc[tr], X.loc[tr, "adp50"], 5, 0.8)
            LO["production"][te], HI["production"][te] = bc.offsets(X.loc[te, "adp50"].to_numpy())
            ctr = (X.loc[tr, "adp50"] < .05).to_numpy()
            cte = (X.loc[te, "adp50"] < .05).to_numpy()
            bcc = EdgeConformal.fit_edges(r.loc[tr][ctr], s.loc[tr][ctr], [.25, .5, .75], 0.8)
            bco = EdgeConformal.fit_edges(r.loc[tr][~ctr], s.loc[tr][~ctr], TAIL_Q, 0.8)
            (clo, chi), (olo, ohi) = bcc.offsets(s.loc[te].to_numpy()), bco.offsets(s.loc[te].to_numpy())
            LO["E2_mondrian"][te], HI["E2_mondrian"][te] = np.where(cte, clo, olo), np.where(cte, chi, ohi)
            bcp = EdgeConformal.fit_edges(r.loc[tr][ctr], X.loc[tr, "adp50"][ctr], [.25, .5, .75], 0.8)
            plo, phi = bcp.offsets(X.loc[te, "adp50"].to_numpy())
            LO["E2_calmprod"][te], HI["E2_calmprod"][te] = np.where(cte, plo, olo), np.where(cte, phi, ohi)
        fl, calm = yy >= 59, X["adp50"] < .05
        for k in LO:
            lo, hi = LO[k], HI[k]
            ins = (r >= lo) & (r <= hi)
            rows.append({"h": h, "bands": k, "cov": ins.mean(), "cov_flood": ins[fl].mean(), "n_flood": int(fl.sum()),
                         "cov_calm": ins[calm].mean(), "width": (hi - lo).mean(), "width_flood": (hi - lo)[fl].mean(),
                         "width_calm": (hi - lo)[calm].mean(), "pinball": (pinball(r, lo, .1) + pinball(r, hi, .9)) / 2,
                         "pinball_flood": (pinball(r[fl], lo[fl], .1) + pinball(r[fl], hi[fl], .9)) / 2,
                         **{f"cov_{yr}": ins[ins.index.year == yr].mean() for yr in fc.YEARS}})
    R = pd.DataFrame(rows)
    base = R[R.bands == "production"].set_index("h")
    R["calm_width_pct"] = [100 * (w / base.loc[h, "width_calm"] - 1) for h, w in zip(R.h, R.width_calm)]
    R["pinball_pct"] = [100 * (q / base.loc[h, "pinball"] - 1) for h, q in zip(R.h, R.pinball)]
    return R


if __name__ == "__main__":
    p = pd.read_parquet(mt.SP + "fc_prod.parquet")
    R = run(p)
    R.to_csv(mt.SP + "fc_bands.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.set_index(["h", "bands"]).round(3).to_string())
