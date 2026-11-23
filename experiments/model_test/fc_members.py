"""GEFS individual members as model input, forecast mode, 2020-10..2023-12 (OOF-by-year models from fc.py).

Same point models and per-lead scales as production (scale fitted on the GEFS mean; members share the mean's
total, so the total-ratio scale is the same quantity). Candidates:
  prod        GEFS ensemble-mean file (31 members), production path
  mean10      mean of the 10 fetched members fed like the GEFS mean (control: how much do 10 vs 31 members cost?)
  ens_mean    mean over members of the production prediction driven by each member's rain
  ens_median  median of the same per-member predictions
  gr4j_avg    GEFS-mean rain features, but GR4J forecast flow averaged over member-driven runs (Jensen fix)
Bands (forecast mode, LOYO calibration, h1/3/5/7):
  production  |p50 - stage| bins
  ens_spread  calm days as production (calm-only bins); other days binned on the spread (p90 - p10) of the
              per-member predictions (tail edges)
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glob, pickle
import numpy as np, pandas as pd
import mt, fc
from bands2 import EdgeConformal, TAIL_Q, pinball
from shingle import evaluate, fcst, train
from shingle.sources import gefs

OUT = mt.SP + "fc_members/"


def member_raw() -> dict[int, pd.DataFrame]:
    df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob("data/gefs/members/*.parquet"))]).sort_index()
    out = {}
    for m in sorted(df.index.get_level_values("member").unique()):
        b = df.xs(m, level="member")
        out[int(m)] = gefs.daily_leads(b)
    return out


def _gr4j_avg_predict(raws, sc, ms, index):
    s = fc.state()
    parts = []
    for y in fc.YEARS:
        held = index[index.year == y]
        if len(held) == 0:
            continue
        base = fcst.build_mixed(s["raw"], s["src"], s["tgt"], s["drv"], s["data"], sc[y], train.SCALED_FROM_H, fc.HS)
        q_plain, q_scaled = [], []
        for r in raws.values():
            f = fcst.build_mixed(r.reindex(s["raw"].index), s["src"], s["tgt"], s["drv"], s["data"], sc[y], train.SCALED_FROM_H, fc.HS)
            q_plain.append(f["gr4j_q"])
            q_scaled.append(f["by_h"][fc.HS[-1]]["gr4j_q"])
        avg = lambda L: sum(L) / len(L)
        scaled = dict(base["by_h"][fc.HS[-1]], gr4j_q=avg(q_scaled))
        f = dict(base, gr4j_q=avg(q_plain), by_h={h: scaled for h in base["by_h"]})
        parts.append(ms[y].point(dict(s["data"], fcst=f), held))
    return pd.concat(parts).sort_index()


def run_points():
    os.makedirs(OUT, exist_ok=True)
    s = fc.state()
    ms = fc.models()
    sc = fc.scales()
    raws = member_raw()
    common = sorted(set.intersection(*[set(r.dropna(how="all").index) for r in raws.values()]))
    index = fc.fc_index()
    index = index[index.isin(common)]
    preds = {}
    for m, r in raws.items():
        p = OUT + f"m{m:02d}.parquet"
        if not os.path.exists(p):
            fc.predict(r.reindex(s["raw"].index), sc, ms, index).to_parquet(p)
        preds[m] = pd.read_parquet(p)
    mean10 = sum(r.reindex(s["raw"].index) for r in raws.values()) / len(raws)
    P = {"prod": pd.read_parquet(mt.SP + "fc_prod.parquet").loc[index],
         "mean10": fc.predict(mean10, sc, ms, index)}
    stack = np.stack([preds[m].loc[index].to_numpy() for m in sorted(preds)])
    cols = preds[min(preds)].columns
    P["ens_mean"] = pd.DataFrame(stack.mean(0), index=index, columns=cols)
    P["ens_median"] = pd.DataFrame(np.median(stack, 0), index=index, columns=cols)
    P["ens_p10"] = pd.DataFrame(np.quantile(stack, .1, 0), index=index, columns=cols)
    P["ens_p90"] = pd.DataFrame(np.quantile(stack, .9, 0), index=index, columns=cols)
    P["gr4j_avg"] = _gr4j_avg_predict(raws, sc, ms, index)
    pickle.dump(P, open(OUT + "points.pkl", "wb"))
    return P


def bands(P, hs=(1, 3, 5, 7)):
    st = fc.state()["data"]["stage"]
    p = P["prod"]
    rows = []
    for h in hs:
        y = st.shift(-h).reindex(p.index)
        r = (y - p[f"h{h}"]).dropna()
        adp = (p[f"h{h}"] - st.loc[p.index]).abs().loc[r.index]
        spread = (P["ens_p90"][f"h{h}"] - P["ens_p10"][f"h{h}"]).loc[r.index]
        LO = {k: pd.Series(np.nan, index=r.index) for k in ("production", "ens_spread")}
        HI = {k: pd.Series(np.nan, index=r.index) for k in ("production", "ens_spread")}
        for yr in fc.YEARS:
            te, tr = r.index[r.index.year == yr], r.index[r.index.year != yr]
            bc = evaluate.BinnedConformal.fit(r.loc[tr], adp.loc[tr], 5, .8)
            LO["production"][te], HI["production"][te] = bc.offsets(adp.loc[te].to_numpy())
            ctr, cte = (adp.loc[tr] < .05).to_numpy(), (adp.loc[te] < .05).to_numpy()
            bcp = EdgeConformal.fit_edges(r.loc[tr][ctr], adp.loc[tr][ctr], [.25, .5, .75], .8)
            bco = EdgeConformal.fit_edges(r.loc[tr][~ctr], spread.loc[tr][~ctr], TAIL_Q, .8)
            (clo, chi), (olo, ohi) = bcp.offsets(adp.loc[te].to_numpy()), bco.offsets(spread.loc[te].to_numpy())
            LO["ens_spread"][te], HI["ens_spread"][te] = np.where(cte, clo, olo), np.where(cte, chi, ohi)
        yy = y.loc[r.index]
        fl, calm = (yy >= 59), adp < .05
        for k in LO:
            lo, hi = LO[k], HI[k]
            ins = (r >= lo) & (r <= hi)
            rows.append({"h": h, "bands": k, "cov": ins.mean(), "cov_flood": ins[fl].mean(), "n_flood": int(fl.sum()), "cov_calm": ins[calm].mean(),
                         "width": (hi - lo).mean(), "width_calm": (hi - lo)[calm].mean(), "width_flood": (hi - lo)[fl].mean(),
                         "pinball": (pinball(r, lo, .1) + pinball(r, hi, .9)) / 2, "pinball_flood": (pinball(r[fl], lo[fl], .1) + pinball(r[fl], hi[fl], .9)) / 2})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    P = pickle.load(open(OUT + "points.pkl", "rb")) if os.path.exists(OUT + "points.pkl") and "--force" not in sys.argv else run_points()
    pd.set_option("display.width", 250, "display.max_columns", 40)
    ref = fc.metrics(P["prod"])
    tab = pd.DataFrame({k: fc.metrics(v) for k, v in P.items() if not k.startswith("ens_p")}).T
    print(f"n issue days {len(P['prod'])}")
    print(tab.round(3).to_string())
    print("\n% vs prod (bootstrap 60-day blocks: * better, ! worse at 90 %)")
    rows = []
    for k, v in P.items():
        if k in ("prod",) or k.startswith("ens_p"):
            continue
        for sub in (None, "hi", "lo"):
            b = fc.boot(v, P["prod"], n=1000, subset=sub)
            for h, rr in b.iterrows():
                rows.append({"cand": k, "subset": sub or "all", "h": h,
                             "cell": f"{rr.pct:+.1f}" + ("*" if rr.hi90 < 0 else "!" if rr.lo90 > 0 else "")})
    B = pd.DataFrame(rows)
    print(B.pivot_table(index="cand", columns=["subset", "h"], values="cell", aggfunc="first").to_string())
    # crests (Ian) by horizon
    st = fc.state()["data"]["stage"]
    print("\nIan crest (2022-10-02 area): max predicted vs observed over target dates 2022-09-28..2022-10-10")
    for h in fc.HS:
        row = {}
        for k in ("prod", "mean10", "ens_mean", "ens_median", "gr4j_avg"):
            pt = P[k][f"h{h}"].copy(); pt.index = pt.index + pd.Timedelta(days=h)
            row[k] = round(pt.loc["2022-09-28":"2022-10-10"].max(), 2)
        row["obs"] = round(st.loc["2022-09-28":"2022-10-10"].max(), 2)
        print(h, row)
    R = bands(P)
    base = R[R.bands == "production"].set_index("h")
    R["calm_width_pct"] = [100 * (w / base.loc[h, "width_calm"] - 1) for h, w in zip(R.h, R.width_calm)]
    R["pinball_pct"] = [100 * (q / base.loc[h, "pinball"] - 1) for h, q in zip(R.h, R.pinball)]
    print(R.set_index(["h", "bands"]).round(3).to_string())
    R.to_csv(OUT + "bands.csv", index=False)
    tab.to_csv(OUT + "points.csv")
    B.to_csv(OUT + "boot.csv", index=False)
