"""Phase 0b: freeze the flood events used for scoring (docs/forecast_training_plan.md, "Frozen scoring").

Rule (fixed before any reforecast data is seen): an event is a run of days with observed daily stage >= 59 ft;
runs less than 7 days apart are merged (7, not 14, so the 2004 hurricanes Charley, Frances and Jeanne,
separated by 8-day dips, stay three events). Scoring window = first day - 7 .. last day + 3. Crest = max observed stage
in the event, crest_date its first occurrence. Written to flood_events.csv with the period each event belongs to.
"""
from pathlib import Path

import pandas as pd

from shingle import fetch

OUT = Path(__file__).parent
st = fetch.load_history("data")["gauge"]["stage_ft"].dropna()
above = st[st >= 59]
runs = []
for day in above.index:
    if runs and (day - runs[-1][1]).days < 7:
        runs[-1][1] = day
    else:
        runs.append([day, day])
rows = []
for a, b in runs:
    seg = st.loc[a:b]
    period = ("pre-2000 (not scored)" if a.year < 2000 else "reforecast 2000-2019" if a.year < 2020
              else "deployment fold 2020-10..2023-12" if a < pd.Timestamp("2024-01-01") else "final test 2024+")
    rows.append({"first": a.date(), "last": b.date(), "window_start": (a - pd.Timedelta(days=7)).date(),
                 "window_end": (b + pd.Timedelta(days=3)).date(), "crest_ft": round(float(seg.max()), 2),
                 "crest_date": seg.idxmax().date(), "days_ge_59": int((seg >= 59).sum()), "period": period})
df = pd.DataFrame(rows)
df.to_csv(OUT / "flood_events.csv", index=False)
print(df.to_string(index=False))
print(df.groupby("period").size())
