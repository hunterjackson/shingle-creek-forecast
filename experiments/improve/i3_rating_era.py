"""Idea 3a: era-aware rating. The rating hybrid fits one CombinedRating to all training years; the high-flow
residual drifted by ~0.5 ft (experiments/review/stationarity_report.txt). Candidates refit only the rating (and its
residual carry phi) on training years near the held-out block (|year - block centre| <= W), as production would use
the last W years. Flow and lake models unchanged. Blend = 0.5 production stage model + 0.5 candidate hybrid.
Leave-3-year-block-out, scored 2007-10..2023-12 and 1979..2023 (test period untouched)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd
import mt
from shingle import rating as rating_mod

OUT = Path(os.path.dirname(os.path.abspath(__file__)))
WINDOWS = (6, 10, 15)


def run(k):
    d = mt.load()
    blk, tr, held = mt.folds(mt.index())[k]
    _, b = mt.build({})
    b.fit(d, tr)
    out = {"prod": b.point(d, held)}
    centre = np.mean(blk)
    for W in WINDOWS:
        near = tr[np.abs(tr.year - centre) <= W]
        dd = pd.concat([d["q"], d["stage"], d["toho"]], axis=1, keys=["q", "stage", "toho"]).loc[near].dropna()
        b.rating = rating_mod.fit_combined(dd["q"], dd["stage"], dd["toho"], **b.rating_opts)
        resid = b._rating_residual(d)
        for h in b.horizons:
            pair = pd.concat([resid, resid.shift(-h)], axis=1, keys=["now", "later"]).loc[tr].dropna()
            b.phi[h] = float((pair["now"] * pair["later"]).sum() / (pair["now"] ** 2).sum())
        out[f"W{W}"] = b.point(d, held)
    return pd.concat(out, axis=1)


if __name__ == "__main__":
    n = len(mt.folds(mt.index()))
    with ProcessPoolExecutor(8, mp_context=mp.get_context("fork")) as ex:
        B = pd.concat(list(ex.map(run, range(n)))).sort_index()
    B.to_parquet(OUT / "i3_rating_oof.parquet")
    ab = pd.read_parquet(mt.SP + "oof/prod_s0.parquet")
    prod = mt.blend(ab)
    lines = []
    for span, start in (("2007-10..2023", "2007-10-01"), ("1979..2023", "1979-01-01"), ("2011..2023", "2011-01-01")):
        rows = []
        for c in ["prod"] + [f"W{W}" for W in WINDOWS]:
            p = pd.DataFrame({f"h{h}": 0.5 * ab[f"a_h{h}"] + 0.5 * B[(c, f"h{h}")] for h in mt.H}).loc[start:]
            m = mt.metrics(p)
            bt = mt.boot_diff(p, prod.loc[start:])
            rows.append({"rating": c, **{f"mae{h}": m[f"mae{h}"] for h in mt.H}, **{f"flood{h}": m[f"hi{h}"] for h in (3, 5, 7)},
                         **{f"low{h}": m[f"lo{h}"] for h in (3, 7)}, **{f"ci{h}": f"{bt.loc[h, 'lo90']:+.4f}..{bt.loc[h, 'hi90']:+.4f}" for h in (3, 5, 7)}})
        lines += [span, pd.DataFrame(rows).round(4).to_string(index=False), ""]
    text = "\n".join(lines); (OUT / "i3_rating_report.txt").write_text(text + "\n"); print(text)
