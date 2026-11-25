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
