"""Why does the day-1 forecast rise too far on heavy-rain days? Two checks.

1. gridMET day alignment: lag correlation of basin gridMET rain with MRMS 24 h QPE ending 12Z on day+1
   (labelled day D = 12Z D .. 12Z D+1, the convention the code assumes) and with MRMS ending 05Z (LST day).
2. Day-1 bias in the perfect-rain OOF backtest 1979-2023 (observed future rain, prod_s0 blend), by observed rain on
   day t+1 and by rain already fallen over t-2..t.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
import numpy as np
import pandas as pd

import mt
from shingle import basin, config, fetch, grid

hist = fetch.load_history("data")
cells = hist["cells"]
w = cells["basin_frac"][cells["basin_frac"] > 0]
gm = (hist["pr"][w.index] * (w / w.sum())).sum(axis=1)
print("1. gridMET basin rain vs MRMS (Spearman r on days with rain in either, 2020-10..2023):")
for hour, label in ((12, "MRMS 12Z..12Z labelled D"), (5, "MRMS LST day (05Z..05Z) labelled D")):
    px = fetch.load_mrms("data", hour=hour)
    r = grid.cell_means(px, cells, half_width=basin.GRIDMET_RES / 2)
    rb = (r[w.index] * (w / w.sum())).sum(axis=1, min_count=1)
    row = []
    for lag in (-1, 0, 1):
        pair = pd.concat([gm, rb.shift(lag)], axis=1, keys=["gm", "mr"]).loc["2020-10-01":"2023-12-31"].dropna()
        pair = pair[(pair > 0.5).any(axis=1)]
        row.append(f"lag {lag:+d}: r={pair.corr('spearman').iloc[0, 1]:.3f}")
    print(f"   {label:38s}", "  ".join(row), " (lag +1 = MRMS day D-1 vs gridMET day D)")

d = mt.load()
oof = mt.blend(pd.read_parquet(mt.SP + "oof/prod_s0.parquet"))
y = mt.targets(oof.index)
st = d["stage"].reindex(oof.index)
pr = (d["pr"][w.index] * (w / w.sum())).sum(axis=1)
nxt = pr.shift(-1).reindex(oof.index)
past = pr.rolling(3).sum().reindex(oof.index)
e = oof["h1"] - y["h1"]
fc_rise = oof["h1"] - st
ob_rise = y["h1"] - st
print("\n2. Day-1, perfect-rain OOF 1979-2023 (forecast - observed, ft):")
rows = []
for nlab, nm in (("next <5", nxt < 5), ("next 10-25", nxt.between(10, 25)), ("next >=25", nxt >= 25)):
    for plab, pm in (("past3 <5", past < 5), ("past3 >=5", past >= 5)):
        m = nm & pm & e.notna()
        rows.append({"rain t+1": nlab, "rain t-2..t": plab, "n": int(m.sum()), "bias": e[m].mean(), "mae": e[m].abs().mean(),
                     "fc rise": fc_rise[m].mean(), "obs rise": ob_rise[m].mean(), "share fc>obs": (e[m] > 0).mean()})
print(pd.DataFrame(rows).round(3).to_string(index=False))
