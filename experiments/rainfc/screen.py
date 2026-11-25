"""Rain-forecast screen, step 1: lead-1 rain from GEFS 0.5 (production), GEFS 0.25, NBM 2.5 km and an NBM/GEFS mean,
scored against observed gridMET rain on t+1, 2020-10..2023-12 (sample from fetch.py).

Two scales, because the models use both: basin-areal (drives GR4J) and the single rain cell (the learned models'
pr_* features). Scale-free skill (Spearman, Pearson on log1p) matters most: the model's per-lead scale can absorb a
constant bias. Also MAE, total ratio obs/fc, and 10 mm event detection (POD, FAR, CSI).
Caveat: days were chosen on GEFS 0.5 >= 5 mm or obs >= 10 mm (+150 random), which slightly favours finding GEFS
false alarms; the 'random' rows are unbiased.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
import numpy as np
import pandas as pd

import fc
from shingle import config, fcst, grid
from shingle.sources import gridmet

OUT = Path(os.path.dirname(os.path.abspath(__file__)))
CACHE = OUT / "cache"
FH = [18, 24, 30, 36]
pd.set_option("display.width", 250)

s = fc.state()
tgt, w = s["tgt"], s["drv"].weights
days = pd.read_csv(CACHE / "days.csv", parse_dates=["date"]).set_index("date")
plain = fcst.build(s["raw"], s["src"], tgt, s["drv"], s["data"])["pr"][1]           # model cells, lead 1
g25 = pd.concat([pd.read_parquet(p) for p in sorted((CACHE / "gefs025").glob("*.parquet"))])
runs = g25.index.get_level_values("run").unique()
tot25 = pd.DataFrame({r - pd.Timedelta(days=1): g25.xs(r, level="run").reindex(FH).sum(min_count=4) for r in runs}).T
m25 = grid.apply_weights(tot25, grid.bilinear_weights(tgt, gridmet.cells_from_columns(tot25.columns)))
nbm = pd.concat([pd.read_parquet(p) for p in sorted((CACHE / "nbm").glob("*.parquet"))])
obs = s["data"]["pr"].shift(-1)

src = {"gefs0.5": plain, "gefs0.25": m25, "nbm": nbm}
cell = config.RAIN_CELL


def series(df, how):
    return fcst._areal(df, w) if how == "basin" else df[cell]


def score(f, o) -> dict:
    ok = f.notna() & o.notna()
    f, o = f[ok], o[ok]
    hit = lambda a: a >= 10
    tp, fp, fn = (hit(f) & hit(o)).sum(), (hit(f) & ~hit(o)).sum(), (~hit(f) & hit(o)).sum()
    return {"n": len(f), "spearman": f.corr(o, "spearman"), "r_log": np.log1p(f).corr(np.log1p(o)), "mae": (f - o).abs().mean(),
            "obs/fc": o.sum() / f.sum() if f.sum() else np.nan, "POD10": tp / (tp + fn) if tp + fn else np.nan,
            "FAR10": fp / (tp + fp) if tp + fp else np.nan, "CSI10": tp / (tp + fp + fn) if tp + fp + fn else np.nan}


lines = ["Rain-forecast screen, lead 1 (12Z t+1 .. 12Z t+2), vs gridMET; 2020-10..2023-12", ""]
for how in ("basin", "cell"):
    o_all = series(obs, how)
    for subset, ix in (("all sampled days", days.index), ("NBM subset", days.index[days["nbm"].to_numpy()])):
        ix = ix[ix.isin(nbm.index)] if subset == "NBM subset" else ix[ix.isin(m25.index)]
        F = {k: series(v.reindex(ix), how) for k, v in src.items() if not (k == "nbm" and subset != "NBM subset")}
        if "nbm" in F:
            F["mean(gefs0.5, nbm)"] = (F["gefs0.5"] + F["nbm"]) / 2
        T = pd.DataFrame({k: score(f, o_all.reindex(ix)) for k, f in F.items()}).T
        lines += [f"{how} rain, {subset}:", T.round(3).to_string(), ""]
# the 150 random days alone (unbiased by selection)
rand = days.index[((days["g05"] < 5) & (days["obs"] < 10)).to_numpy()]
lines.append(f"Random non-event days (n={len(rand)}): mean fc gefs0.5 {series(plain.reindex(rand), 'basin').mean():.2f} mm, "
             f"gefs0.25 {series(m25.reindex(rand), 'basin').mean():.2f}, obs {series(obs.reindex(rand), 'basin').mean():.2f}")
# heavy forecasts: how often did each source's >= 20 mm basin forecast verify?
nb_ix = days.index[days["nbm"].to_numpy()]
nb_ix = nb_ix[nb_ix.isin(nbm.index)]
o_b = series(obs, "basin").reindex(nb_ix)
lines.append("")
lines.append("Heavy forecasts on the NBM subset (basin >= 20 mm): n, median obs, share obs < 10 mm")
for k in ("gefs0.5", "gefs0.25", "nbm"):
    f = series(src[k].reindex(nb_ix), "basin")
    m = f >= 20
    lines.append(f"  {k:9s} n={int(m.sum()):3d}  median obs {o_b[m].median():5.1f}  obs<10: {(o_b[m] < 10).mean():.0%}")
text = "\n".join(lines)
(OUT / "screen_report.txt").write_text(text + "\n")
print(text)

# paired day bootstrap of the skill difference nbm - gefs0.5 (NBM subset)
rng = np.random.default_rng(0)
lines2 = ["", "Paired bootstrap (2000 day resamples), NBM subset: difference nbm - gefs0.5, 90 % CI"]
for how in ("basin", "cell"):
    f_n, f_g, o = series(nbm.reindex(nb_ix), how), series(plain.reindex(nb_ix), how), series(obs, how).reindex(nb_ix)
    ok = f_n.notna() & f_g.notna() & o.notna()
    a, b, c = f_n[ok].to_numpy(), f_g[ok].to_numpy(), o[ok].to_numpy()
    for name, fn in (("spearman", lambda x, y: pd.Series(x).corr(pd.Series(y), "spearman")), ("r_log", lambda x, y: np.corrcoef(np.log1p(x), np.log1p(y))[0, 1])):
        d = []
        for _ in range(2000):
            i = rng.integers(0, len(a), len(a))
            d.append(fn(a[i], c[i]) - fn(b[i], c[i]))
        lines2.append(f"  {how:5s} {name:8s} diff {fn(a, c) - fn(b, c):+.3f}  CI {np.quantile(d, .05):+.3f}..{np.quantile(d, .95):+.3f}")
text2 = "\n".join(lines2)
with open(OUT / "screen_report.txt", "a") as fh:
    fh.write(text2 + "\n")
print(text2)
