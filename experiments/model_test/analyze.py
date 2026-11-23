"""Stage 0/1 report: metrics, seed noise floor, paired bootstrap vs production, per-event crests."""
import sys, os, glob; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import mt

pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)
OOF = {os.path.basename(p)[:-8]: pd.read_parquet(p) for p in sorted(glob.glob(mt.SP + "oof/*.parquet"))}
ix = OOF["prod_s0"].index
seeds = [OOF[f"prod_s{s}"] for s in range(5)]
prod = mt.blend(seeds[0])

# ---- stage 0: noise floor
seed_m = pd.DataFrame([mt.metrics(mt.blend(s)) for s in seeds])
noise = seed_m.std()
print("== Stage 0: production OOF, 5 seeds (mean, sd, sd %)")
print(pd.DataFrame({"mean": seed_m.mean(), "sd": noise, "sd_pct": 100 * noise / seed_m.mean()}).round(4).T.to_string())

# ---- post-hoc candidates
cands = {k: mt.blend(v) for k, v in OOF.items() if not k.startswith("prod_s")}
bag = sum(seeds) / 5
cands["A_seedbag5"] = mt.blend(bag)
for part in ("a", "b"):
    cands[f"only_{'stage' if part == 'a' else 'hybrid'}"] = mt.blend(seeds[0], w=1.0 if part == "a" else 0.0)

def nested_weights(df, grid=np.linspace(0, 1, 21)):
    """Per-horizon blend weight chosen on the other folds' OOF predictions (3-year blocks)."""
    y = mt.targets(df.index)
    out = {}
    for blk, _, held in mt.folds(df.index):
        other = df.index[~df.index.isin(held)]
        w = {}
        for h in mt.H:
            a, b, yy = df.loc[other, f"a_h{h}"], df.loc[other, f"b_h{h}"], y.loc[other, f"h{h}"]
            ok = yy.notna() & a.notna() & b.notna()
            maes = [((g * a[ok] + (1 - g) * b[ok]) - yy[ok]).abs().mean() for g in grid]
            w[h] = float(grid[int(np.argmin(maes))])
        out[tuple(blk)] = w
    parts = [mt.blend(df.loc[held], w=out[tuple(blk)]) for blk, _, held in mt.folds(df.index)]
    return pd.concat(parts).loc[df.index], out

cands["A_blend_w_nested"], wts = nested_weights(seeds[0])
print("\nnested per-horizon weights on stage model (first/last fold):", list(wts.values())[0], list(wts.values())[-1])

# ---- table
rows = {}
for k, p in sorted(cands.items()):
    m = mt.metrics(p)
    rows[k] = m
tab = pd.DataFrame(rows).T
ref = pd.Series(mt.metrics(prod))
print("\n== Stage 1: blend MAE (ft); production seed 0 first")
print(pd.concat([ref.rename("PRODUCTION").to_frame().T, tab]).round(4).to_string())
pct = 100 * (tab - ref) / ref
print("\n== % change vs production (negative = better); noise sd % in last row")
print(pd.concat([pct, (100 * noise / seed_m.mean()).rename("noise_sd%").to_frame().T]).round(1).to_string())

# ---- bootstrap for every candidate (all days, flood days, low water)
print("\n== paired block bootstrap, diff = cand - prod (ft), 90% CI")
summ = []
for k, p in sorted(cands.items()):
    for sub in (None, "hi", "lo"):
        b = mt.boot_diff(p, prod, n=1000, subset=sub)
        for h, r in b.iterrows():
            summ.append({"cand": k, "subset": sub or "all", "h": h, "pct": r.pct, "lo90": r.lo90, "hi90": r.hi90,
                         "sig_better": r.hi90 < 0, "sig_worse": r.lo90 > 0})
S = pd.DataFrame(summ)
S.to_csv(mt.SP + "stage1_boot.csv", index=False)
piv = S.assign(cell=lambda d: d.pct.round(1).astype(str) + np.where(d.sig_better, "*", np.where(d.sig_worse, "!", "")))
print(piv.pivot_table(index="cand", columns=["subset", "h"], values="cell", aggfunc="first").to_string())
print("(* = 90% CI below zero, i.e. better; ! = CI above zero, worse)")

# ---- per-event crest errors (17 events in 1979-2023) for h3 and h7
def crests(p, h):
    c = mt.crest_errors(p, h)
    return c.set_index(c.obs_peak_date.dt.strftime("%Y-%m-%d"))[["obs_peak", "peak_error", "timing_error_days"]]
print("\n== crest error (ft) per event, production vs selected candidates")
sel = [k for k in ("B_linear_tree", "B_mono", "B_ridge_resid", "A_seedbag5", "A_blend_w_nested", "only_stage", "only_hybrid") if k in cands]
for h in (3, 7):
    t = {"prod": crests(prod, h).peak_error}
    t.update({k: crests(cands[k], h).peak_error for k in sel})
    obsp = crests(prod, h).obs_peak
    T = pd.DataFrame(t)
    T.insert(0, "obs_peak", obsp)
    T.loc["mean |err|"] = T.abs().mean()
    print(f"-- h{h}"); print(T.round(2).to_string())
