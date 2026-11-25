# Test plan: forecast vs. observed divergence, issue day 2026-11-24

Written 2026-11-25 ~15:30Z. Nothing here has been run yet except the diagnosis in the next section.

## What the site showed

| | Value |
|---|---|
| Forecast for 11-25 (daily mean), issued 13:39Z | 57.18 ft, likely range 57.08–57.28 |
| Gauge at 13:00Z / 15:15Z on 11-25 | 56.77 / 56.79 ft |
| Later days | 57.70, 57.55, 58.34, 58.37, 58.25, 58.02 (p90 up to 59.86, above action stage) |
| Previous issue (11-23) for the same days | 56.71, 56.63, 56.60, 56.58, 56.64, 56.63 |

## Diagnosis so far (reproduced locally with the production model)

The pipeline is doing what it was built to do; the inputs are decoded correctly. The whole gap comes
from one input: the 00Z 11-25 GEFS ensemble mean puts about 35 mm on the basin between 12Z 11-25 and
00Z 11-26. That rain had not started when the forecast was published.

| Lead-1 rain given to the model | Day 1 | Day 2 | Day 4 | Day 5 |
|---|---|---|---|---|
| 0 mm | 56.73 | 56.53 | 56.27 | 56.23 |
| ~15 mm (×0.4, about what ECMWF and NBM show) | 56.88 | 56.78 | 57.20 | 56.98 |
| ~35 mm (as issued) | 57.18 | 57.70 | 58.34 | 58.37 |
| ~70 mm (×2) | 57.33 | 57.83 | 59.10 | 59.17 |

Supporting facts:

- With no rain the model gives 56.73 for today, which matches the gauge. Gauge, lake, radar and
  gridMET inputs are all current and sensible.
- GEFS decode is right: an independent copy of the same ensemble gives a mean of 19 + 16 mm for the
  two 6-hour windows, with members from 2 to 76 mm. Other models for the same window: ECMWF ~16 mm,
  NBM ~14 mm, GFS deterministic and HRRR ~4 mm. The GEFS mean is the wet outlier.
- Both blend members react the same way (stage model 57.23, rating model 57.12 for day 1), so this
  is not one member misbehaving.

Five weaknesses this exposed, each with a test below:

1. **The day-1 range ignores rain uncertainty.** It is ±0.10 ft, while the rain scenarios above move
   day 1 by 0.6 ft. Day-1 width is binned only on |forecast change|.
2. **The forecast rests on the GEFS mean alone.** No check against other guidance, no use of spread.
3. **The seven days do not form one consistent path.** 57.70 → 57.55 → 58.34 with no rain in between.
   Days 4–7 are fed rain multiplied by 1.37–1.42 (including day-1 rain); days 1–3 get it unscaled.
   GR4J's peak flow is 1114 cfs unscaled and 1340 cfs scaled.
4. **The site's rain column shows the scaled number (52.5 mm)**, not the 38 mm that days 1–3 used.
5. **The forecast is frozen at 13Z.** The 3-hourly reruns refresh the current reading but not the
   forecast, so a rain forecast that fails by afternoon stays on the site until the next morning.

Also noted: GR4J simulates 875 cfs for 11-24 against 621 cfs observed. It is only a feature for the
learned models, but it is 40 % high going into a rain event.

## Tests

Rules as before: failing test first, then code; screen cheaply before full runs; tune on
out-of-fold and forecast-mode 2020-10..2023-12 only; the 2024+ test period gets one look for one
final candidate.

### T0. Score this event (no code change, do first)

- After the 11-25 daily values post: record actual rain (MRMS 12Z total, then gridMET) and the
  stage for 11-25..11-28 against the four scenario rows above.
- Outcome decides the weighting of everything else. If ~35 mm fell and the creek followed the
  as-issued row, the point forecast was right and only items 1, 3, 4, 5 matter. If the rain was
  under ~15 mm, item 2 matters most.

### T1. How good is the model when GEFS says heavy rain tomorrow? (item 2, screen)

- From the existing forecast-mode out-of-fold predictions, take issue days with GEFS lead-1 basin
  rain ≥ 20 mm (and ≥ 30 mm). Report n, day-1..4 bias and MAE, and 80 % coverage, against the same
  numbers for all days.
- Same days: GEFS lead-1 rain vs. gridMET observed. Bias, and how often observed < half of forecast.
- Pass to T2/T3 if wet-forecast days show a clear positive stage bias or coverage well under 80 %.
  If n < 30, say so and report event by event.

### T2. Day-1 range that knows about rain (item 1)

- Candidate A: use the learned-width band (already used for days 2–7) at day 1, restricted to days
  with forecast rain or |change| above the calm threshold. It was rejected for day 1 on flood
  coverage; retest with the restriction.
- Candidate B: add GEFS lead-1 rain (and spread, T3) as a binning variable for day 1.
- Metric: day-1 coverage on wet-forecast days (T1 subset), flood days and all days; mean width;
  pinball loss. Accept only if wet-day coverage reaches ~80 % without widening calm days.

### T3. Cheaper ways to hedge the rain forecast (item 2, screen then decide)

Screen each by correlating with day-1..3 out-of-fold forecast-mode residuals on the T1 subset:

- GEFS spread at lead 1 (the `gespr` files are already supported by the fetcher).
- Disagreement between GEFS mean and a second source. Check first which archives go back to 2020-10
  for free (ECMWF open data does not; NBM and GFS deterministic on AWS do).
- GEFS ensemble median instead of mean for lead 1 (members were ruled out as model inputs, but not
  as a replacement for a skewed mean on convective days).

Only a candidate that passes the screen gets a full pull and wiring.

### T4. One consistent path (item 3)

- Unit test first: for a single issue day, the rain each horizon's model sees for a given calendar
  day should be the same number. This fails today by design.
- Compare three settings on forecast-mode out-of-fold: (a) current mixed, (b) unscaled everywhere,
  (c) scale applied only to the leads it was fitted on for long horizons, never to leads 1–3.
  MAE per horizon, plus a new roughness metric: mean |p50(h+1) − p50(h)| on days with no forecast
  rain between h and h+1.
- Accept a setting that removes the jump without losing more than ~0.01 ft MAE at days 4–7.

### T5. Site shows the rain the model used (item 4)

- Unit test: `forecast_rain` for leads 1–3 equals the unscaled areal rain. Small fix, depends on T4's
  outcome for leads 4–7.

### T6. Intraday refresh (item 5)

- Question: does a rerun at 16Z/19Z/22Z with (a) the newest GEFS run available and (b) MRMS rain
  observed so far replacing the matching GEFS buckets beat the frozen 13Z forecast?
- Screen on the T1 subset only: replace lead-1 GEFS rain with a mix of observed-so-far and remaining
  forecast at 18Z and 00Z; compare day-1..3 MAE with the frozen forecast.
- If it helps, design the change so the forecast log still scores the first issue of each day.

### T7. GR4J running high (side issue)

- Screen: correlation of (GR4J simulated − observed flow at issue time) with out-of-fold residuals.
  If it carries signal, test it as a feature or test state updating; if not, drop it.

## Order

T0 (free, tomorrow) → T1 (an hour, existing predictions) → T4 and T5 (known defects, independent of
T1) → T2 → T3 → T6 → T7.

## Results

### T1 (run 2026-11-25) — `experiments/divergence/t1_wet.py`, `t1_report.txt`

Forecast-mode OOF 2020-10..2023-12 (1,186 issue days, rebuilt `fc_prod.parquet`; MAE 0.034 / 0.155 / 0.251 /
0.328 at days 1/3/5/7, as before). Bands: leave-one-year-out |p50 − stage| conformal, the production day-1 band.

| GEFS lead-1 basin rain | n | bias d1 | bias d2 | bias d3 | cov d1 | cov d2 | cov d3 | width d1 |
|---|---|---|---|---|---|---|---|---|
| all | 1186 | −0.00 | −0.03 | −0.07 | 78 % | 78 % | 77 % | 0.09 |
| < 5 mm | 1012 | −0.00 | −0.01 | −0.03 | 81 % | 84 % | 82 % | 0.09 |
| 5–20 mm | 161 | −0.02 | −0.15 | −0.24 | 64 % | 48 % | 47 % | 0.12 |
| ≥ 20 mm | 13 | +0.08 | −0.30 | −0.62 | 23 % | 8 % | 38 % | 0.13 |
| ≥ 30 mm | 5 | +0.18 | −0.64 | −0.70 | 20 % | 20 % | 60 % | 0.14 |

Findings:

- **The range is far too narrow whenever rain is forecast.** This is the clearest result and it does not
  rest on a handful of events: the 161 days with 5–20 mm forecast have 64 % / 48 % / 47 % coverage at
  days 1–3. All of the shortfall sits on these days; dry days are at or above 80 %. T2 goes ahead.
- **On heavy-rain days the day-1 forecast rises too early.** Bias at day 1 is positive on 9 of the 13 days
  with ≥ 20 mm forecast (+0.42 and +0.46 ft on two). Then days 2–3 come in too low. The 11-24 issue
  looks the same: +0.36 ft forecast for the day the rain was due.
- **GEFS is not wet-biased here.** Observed basin rain is 1.5× the GEFS mean overall, and 1.37× on
  ≥ 20 mm days. Only 4 of the 13 heavy-forecast days got less than half. So the 11-25 forecast (35 mm,
  with ECMWF/NBM at ~15) may simply be one of those misses. The mean is still a poor guess on any
  single day, which is what T3 tests.
- Only 13 heavy-forecast days in 3 years. Event-by-event table in `t1_report.txt`. Ian (2022-09-26/27)
  dominates the large misses.

### T4 (run 2026-11-25) — `t4_consistency.py`, `t4_report.txt`; T4b — `t4b_qmap.py`, `t4b_report.txt`

MAE at days 4 / 7 (all days; flood obs ≥ 59 ft, 14 days of one event; low water, 85 days):

| Rain setting | all d4 | all d7 | flood d4 | flood d7 | low d4 | low d7 | roughness 3→4 |
|---|---|---|---|---|---|---|---|
| mixed (production) | 0.213 | 0.328 | 1.02 | 1.75 | 0.110 | 0.153 | 0.076 |
| raw everywhere | 0.218 | 0.340 | 1.21 | 2.15 | 0.085 | 0.103 | 0.072 |
| scaled everywhere | 0.213 | 0.328 | 1.02 | 1.75 | 0.110 | 0.153 | 0.074 |
| per-lead (consistent) | 0.218 | 0.336 | 1.21 | 2.00 | 0.088 | 0.125 | 0.071 |
| quantile-mapped, h4–7 | 0.241 | 0.385 | 0.98 | 1.02 | 0.128 | 0.179 | — |

(Scaled everywhere is also 4 % worse than mixed at day 2.)

- **Production's mixed setting stays.** Every alternative is worse overall at days 4–7. Per-lead and raw
  lose 2–4 % (CIs include 0), with much larger flood misses; quantile mapping loses 13–17 % (CI excludes 0).
- **The jagged 11-24 path is not systematic.** On dry transitions, the 3→4 step averages 0.076 ft under
  mixed against 0.071 under per-lead. That difference is too small to be worth the accuracy loss.
- **Open question (not pursued here):** the flat scale costs 25–50 % on low-water days at days 4–7
  (0.153 vs 0.103 at day 7). A state-dependent choice could recover it, but it would be tuned on
  85 low-water days and one flood. That belongs in a future model-test stage, not in this fix.
- Quantile mapping inflates the ensemble-mean tail absurdly (35 mm → 250 mm at lead 4). Its flood gain
  is Ian alone, so the bootstrap CI is degenerate. Rejected.

### T5 (done 2026-11-25)

`pipeline.forecast_rain` now reports the issued GEFS basin rain without the per-lead scale. The model
inputs are unchanged. The 11-24 issue would have shown 38 mm for 11-25 instead of 52.5 mm.

### T2 development (run 2026-11-25) — `t2_bands.py`, `t2_report.txt`

Forecast-mode OOF 2020-10..2023-12, all bands calibrated leave-one-year-out. "Wet" means the issued GEFS
basin rain summed over leads 1..h is ≥ 5 mm (a fixed choice, not tuned). Flood is 14 days, all from Ian.

| Day | Band | cov all | cov dry | cov wet 5–20 | cov wet ≥ 20 | cov flood | width all | pinball |
|---|---|---|---|---|---|---|---|---|
| 1 | production | 78 % | 81 % | 64 % (161) | 23 % (13) | 6/14 | 0.093 | 0.0102 |
| 1 | learned (E2) | 78 % | 80 % | 67 % | 31 % | 3/14 | 0.094 | 0.0095 |
| 1 | wet split, dry recalibrated | 78 % | 78 % | 78 % | 69 % | 3/14 | 0.097 | 0.0098 |
| 1 | **wet_add** | 81 % | 81 % | 78 % | 69 % | 5/14 | 0.107 | 0.0097 |
| 3 | production | 77 % | 84 % | 71 % (379) | 59 % (119) | 71 % | 0.438 | 0.0440 |
| 3 | **wet_add** | 82 % | 84 % | 81 % | 76 % | 71 % | 0.514 | 0.0424 |
| 5 | production | 78 % | 85 % | 78 % | 64 % | 71 % | 0.754 | 0.0668 |
| 5 | wet_add | 81 % | 85 % | 84 % | 65 % | 64 % | 0.808 | 0.0674 |
| 7 | production | 76 % | 90 % | 79 % | 62 % | 64 % | 0.957 | 0.0919 |
| 7 | wet_add | 80 % | 90 % | 84 % | 66 % | 64 % | 1.011 | 0.0902 |

wet_add keeps every dry day on the production band. Wet days get their own conformal bins on the learned
|residual| scale, calibrated on wet days only (tertile edges).

- Recalibrating dry days without the wet days narrows their band, and Ian's flood days fall in the dry group
  at lead 1. That is why the plain wet split loses flood coverage; wet_add avoids it.
- At days 1 and 3, wet_add fixes the wet-day shortfall and improves pinball by 4–5 %. Days 5–7 gain little,
  because production's learned band already includes forecast rain.
- **Frozen candidate:** wet_add for days 1–3 (day 2 by interpolation; the width models exist for days 1/3/5/7
  in the harness), WET = 5 mm. Next: one look at 2024-01..2026-11-12, then implement if it holds.
