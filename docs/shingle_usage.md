# Shingle Creek forecast — how to run

```sh
uv run shingle fetch            # update USGS daily, gridMET, DBHYDRO caches in data/
uv run shingle train            # fit models/shingle.pkl on everything up to 8 days ago (~3 min)
uv run shingle predict          # update live inputs, write out/shingle.json, append out/forecast_log.csv
uv run shingle score            # verify logged forecasts (ours vs NWM short-range h1) as observations arrive
```

Run `predict` once a day after ~13Z (08:00 EST / 09:00 EDT), when the 12Z gauge readings and the MRMS
24 h radar total ending 12Z are published. The forecast is for issue day t = yesterday, and covers t+1 to t+7.
Retrain occasionally (e.g. monthly) so the rating, GR4J and rain-scale corrections stay current.

One-off history pulls (resumable): `fetch.fetch_iv_history`, `fetch.fetch_gefs_history` (GEFSv12 2020-10+),
`fetch.fetch_mrms_history(hour=12)`; see docs/shingle_findings.md for what each source is used for.

## Live inputs
| input | source | lag handling |
|---|---|---|
| stage/flow daily means | USGS dv | computed from 15-min IV when dv isn't out yet |
| morning reading | USGS IV at 12Z on t+1 | missing → model's missing-value branch |
| Toho / S-61 / S-59 / E. Toho | SFWMD DBHYDRO | last value carried forward ≤3 days |
| past rain | gridMET (4 km) | days after gridMET's last day from MRMS 12Z QPE × radar_scale |
| PET | gridMET | day-of-year climatology for unpublished days |
| forecast rain | GEFSv12 ensemble mean, 00Z run of t+1 | raw for h1–3, per-lead bias-scaled for h4–7 |

## Output (`out/shingle.json`)
`current` (latest 15-min stage + flood category), `flood_stages` (59/60/61/62 ft), `forecast` (7 rows of
date, p10/p50/p90, category of p50 and p90, bias-scaled basin rain in mm), `model` (train_end, what was filled).
