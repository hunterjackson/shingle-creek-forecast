"""Follow-up to candidate E: tune the learned-scale binned bands (E2) to meet the band rule.

Band rule (docs/model_test_plan.md): overall coverage .77-.83, flood-day coverage up, calm-day width <= +20 %.
Evaluated on production OOF point forecasts 1979-2023 (observed future rain), calibrated on the other folds.

E2 in stage 1 fitted its |residual| model on 2/3 of the training folds but calibrated the bins on all of them, so
on 2/3 of the calibration rows the scale was in-sample (too confident) -> coverage .76-.77. Variants here:
  E2_split    scale model on 2/3 of training years, bins calibrated only on the held-back 1/3 (plain split conformal)
  E2_xfit     scale model cross-fitted over 3 year groups, so every training row gets an out-of-sample scale;
              bins calibrated on all training rows
  E2_xfit_tail  as E2_xfit, bin edges at scale quantiles .3/.55/.75/.88/.95/.98 (dedicated tail bins)
  E2_xfit_mix   as E2_xfit_tail, scale = learned scale, but calm days (|p50 - stage| < .05 and learned scale below
              its median) use production's |p50 - stage| bin offsets -> protects calm width
  E2_mondrian calibrated separately on calm (|p50 - stage| < .05) and other days (Mondrian conformal), each stratum
              binned on the cross-fitted learned scale (calm: 4 quantile bins; other: tail edges) -> 80 % in each
  E2_calmprod as E2_mondrian on other days; calm days use |p50 - stage| bins calibrated on calm days only
              (the forecast-mode check showed the learned scale mis-spreads calm-day width with issued rain)
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb
import mt
from shingle import evaluate, features

d = mt.load()
oof = mt.blend(pd.read_parquet(mt.SP + "oof/prod_s0.parquet"))
ix = oof.index
y = mt.targets(ix)
C = mt.CELL
QP = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50, verbose=-1, n_jobs=8)
TAIL_Q = [.3, .55, .75, .88, .95, .98]


def feats(h):
    X = features.build(d, ["stage", "dstage_1", "log_q", "toho_dev", f"pr_nextH__{C}", f"pr_past3__{C}", "gr4j_dH", "gr4j_s", "doy_sin", "doy_cos"], horizon=h).loc[ix]
    X["dp50"] = oof[f"h{h}"] - d["stage"].loc[ix]
    X["adp50"] = X["dp50"].abs()
    return X


def pinball(y, q, a):
    e = y - q
    return np.mean(np.maximum(a * e, (a - 1) * e))


class EdgeConformal(evaluate.BinnedConformal):
    @classmethod
    def fit_edges(cls, r, s, qs, coverage=0.8, min_count=60):
        r, s = np.asarray(r, float), np.asarray(s, float)
        edges = list(np.unique(np.quantile(s, qs)))
        while edges:
            lab = np.searchsorted(edges, s, side="right")
            cnt = np.bincount(lab, minlength=len(edges) + 1)
            if cnt.min() >= min_count:
                break
            edges.pop(min(int(np.argmin(cnt)), len(edges) - 1))
        edges = np.array(edges)
        lab = np.searchsorted(edges, s, side="right")
        a = (1 - coverage) / 2
        lo = np.array([np.quantile(r[lab == b], a) for b in range(len(edges) + 1)])
        hi = np.array([np.quantile(r[lab == b], 1 - a) for b in range(len(edges) + 1)])
        return cls(edges, np.minimum(lo, 0), np.maximum(hi, 0))


def xfit_scale(X, r, tr, groups=3):
    """Out-of-sample |residual| predictions for every row of tr (cross-fit over year groups) + a model on all of tr."""
    yrs = np.array(sorted(set(tr.year)))
    out = pd.Series(np.nan, index=tr)
    for g in range(groups):
        hold = tr[tr.year.isin(yrs[g::groups])]
        fit = tr[~tr.year.isin(yrs[g::groups])]
        m = lgb.LGBMRegressor(objective="l1", **QP).fit(X.loc[fit], r.loc[fit].abs())
        out[hold] = m.predict(X.loc[hold])
    full = lgb.LGBMRegressor(objective="l1", **QP).fit(X.loc[tr], r.loc[tr].abs())
    return out, full


if __name__ == "__main__":
    rows = []
    for h in mt.H:
        X = feats(h)
        r = y[f"h{h}"] - oof[f"h{h}"]
        ok = r.notna() & X["adp50"].notna()
        X, r = X[ok], r[ok]
        yy = y[f"h{h}"][ok]
        names = ["production", "E2_split", "E2_xfit", "E2_xfit_tail", "E2_xfit_mix", "E2_mondrian", "E2_calmprod"]
        LO = {k: pd.Series(np.nan, index=r.index) for k in names}
        HI = {k: pd.Series(np.nan, index=r.index) for k in names}
        for blk, _, held in mt.folds(ix):
            te = r.index[r.index.isin(held)]
            tr = r.index[~r.index.isin(held)]
            bc = evaluate.BinnedConformal.fit(r.loc[tr], X.loc[tr, "adp50"], 5, 0.8)
            plo, phi = bc.offsets(X.loc[te, "adp50"].to_numpy())
            LO["production"][te], HI["production"][te] = plo, phi
            # split
            yrs = np.array(sorted(set(tr.year)))
            cal_y = yrs[1::3]
            fit_ix, cal_ix = tr[~tr.year.isin(cal_y)], tr[tr.year.isin(cal_y)]
            sm = lgb.LGBMRegressor(objective="l1", **QP).fit(X.loc[fit_ix], r.loc[fit_ix].abs())
            b = evaluate.BinnedConformal.fit(r.loc[cal_ix], pd.Series(sm.predict(X.loc[cal_ix]), index=cal_ix), 8, 0.8)
            LO["E2_split"][te], HI["E2_split"][te] = b.offsets(sm.predict(X.loc[te]))
            # cross-fit
            s_tr, full = xfit_scale(X, r, tr)
            s_te = full.predict(X.loc[te])
            b = evaluate.BinnedConformal.fit(r.loc[tr], s_tr, 8, 0.8)
            LO["E2_xfit"][te], HI["E2_xfit"][te] = b.offsets(s_te)
            bt = EdgeConformal.fit_edges(r.loc[tr], s_tr, TAIL_Q, 0.8)
            tlo, thi = bt.offsets(s_te)
            LO["E2_xfit_tail"][te], HI["E2_xfit_tail"][te] = tlo, thi
            calm = (X.loc[te, "adp50"].to_numpy() < 0.05) & (s_te < np.median(s_tr))
            LO["E2_xfit_mix"][te] = np.where(calm, plo, tlo)
            HI["E2_xfit_mix"][te] = np.where(calm, phi, thi)
            ctr = (X.loc[tr, "adp50"] < 0.05).to_numpy()
            cte = X.loc[te, "adp50"].to_numpy() < 0.05
            bc_c = EdgeConformal.fit_edges(r.loc[tr][ctr], s_tr[ctr], [.25, .5, .75], 0.8)
            bc_o = EdgeConformal.fit_edges(r.loc[tr][~ctr], s_tr[~ctr], TAIL_Q, 0.8)
            (clo, chi), (olo, ohi) = bc_c.offsets(s_te), bc_o.offsets(s_te)
            LO["E2_mondrian"][te], HI["E2_mondrian"][te] = np.where(cte, clo, olo), np.where(cte, chi, ohi)
            bcp = EdgeConformal.fit_edges(r.loc[tr][ctr], X.loc[tr, "adp50"][ctr], [.25, .5, .75], 0.8)
            pclo, pchi = bcp.offsets(X.loc[te, "adp50"].to_numpy())
            LO["E2_calmprod"][te], HI["E2_calmprod"][te] = np.where(cte, pclo, olo), np.where(cte, pchi, ohi)
        fl = yy >= 59
        calm_days = X["adp50"] < 0.05
        for k in names:
            lo, hi = LO[k], HI[k]
            inside = (r >= lo) & (r <= hi)
            rows.append({"h": h, "bands": k, "cov": inside.mean(), "cov_flood": inside[fl].mean(), "width": (hi - lo).mean(),
                         "width_flood": (hi - lo)[fl].mean(), "width_calm": (hi - lo)[calm_days].mean(),
                         "pinball": (pinball(r, lo, .1) + pinball(r, hi, .9)) / 2,
                         "pinball_flood": (pinball(r[fl], lo[fl], .1) + pinball(r[fl], hi[fl], .9)) / 2})
        pd.DataFrame({f"{k}_{s}": (LO if s == "lo" else HI)[k] for k in names for s in ("lo", "hi")}).to_parquet(mt.SP + f"oof/bands2_h{h}.parquet")
    R = pd.DataFrame(rows)
    base = R[R.bands == "production"].set_index("h")
    R["calm_width_pct"] = [100 * (w / base.loc[h, "width_calm"] - 1) for h, w in zip(R.h, R.width_calm)]
    R["pinball_pct"] = [100 * (p / base.loc[h, "pinball"] - 1) for h, p in zip(R.h, R.pinball)]
    R["passes"] = (R["cov"].between(.77, .83)) & (R.cov_flood > R.h.map(base.cov_flood)) & (R.calm_width_pct <= 20)
    R.to_csv(mt.SP + "bands2.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.set_index(["h", "bands"]).round(3).to_string())
