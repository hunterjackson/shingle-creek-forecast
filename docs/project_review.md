# Project review: errors, faulty assumptions, unexplored improvements

2026-11-25. Read all of `src/shingle`, the workflows and the site code, and checked suspicions against the data,
the git history of the forecast log, and the existing out-of-fold predictions. Scripts and reports are in
`experiments/review/`. The 2024+ test period was not used. Nothing in production was changed by this review.

## Errors and faulty assumptions

### 1. The day-1 model under-uses the morning gauge reading (largest finding)

Assumption: giving the trees `stage_am_d` (the 7–8 am EST hourly reading, "8 am" below, minus yesterday's mean) is enough for day 1.
It is not. Carrying the 8 am reading forward unchanged beats the production day-1 forecast
(`day1_decomp.py`, 5,791 issue days 2007-10..2023-12, out-of-fold, MAE ft):

| Day-1 estimate | All | Moving days (≥ 0.10 ft) | Flood days (36) | Low water |
|---|---|---|---|---|
| Production blend | 0.0355 | 0.073 | 0.204 | 0.022 |
| Stage model alone / rating hybrid alone | 0.0347 / 0.0468 | 0.072 / 0.098 | 0.213 / 0.299 | 0.021 / 0.027 |
| 8 am reading carried forward | 0.0289 | 0.063 | 0.076 | 0.017 |
| 8 am + 0.37 × (production − 8 am) − 0.48 × (mean of midnight–8 am − 8 am) | 0.0251 | 0.048 | 0.059 | 0.018 |

- The production forecast's departure from the 8 am reading is mostly noise: its best weight is 0.37, not 1.
- Same result with issued GEFS rain, 2020-10..2023-12 (`anchor_check.py`): 0.0345 → 0.0236, −31 %
  (90 % CI −0.018..−0.006 ft), flood days 0.293 → 0.058.
- **Day 1 only.** From day 2 on the same correction gives nothing or is worse (weight on production ≈ 0.93).
- Checked for a leak: every input is an hourly value labelled ≤ 12Z of t+1, the same one production reads; the
  alignment is confirmed by the 23-hours-seen case (MAE 0.003).
- Cause not established. Candidates: trees approximate "add the morning change" in steps and cannot go beyond the
  changes seen in training; the morning reading is missing for 1979–2007, 60 % of the training rows; the 0.5 weight
  on the rating hybrid, whose day 1 is 0.047.

### 2. The track record scores a later forecast than the one issued

`forecast_log.append` replaces the rows of an issue day on every run, and the forecast job reruns every 3 hours
until the next issue day (last at 10:30Z, after day 1 has ended). The later runs use data that arrived after the
13:30Z issue (gridMET in place of the radar fill, revised gauge values, a retrained model). Git history of
`site/data/forecast_log.csv`: issue 11-22 day 1 56.74 → 56.75, issue 11-23 56.78 → 56.77, issue 11-24
57.18 → 57.15 (retrain). Small so far, but the site says "logged when it's issued", and day 2 of issue 11-24 moved
57.75 → 58.00 between two runs 17 minutes apart (cause not confirmed). The log is the only evidence untouched by model selection (item 5).

### 3. The National Water Model comparison is broken

`nwm_stage_h1` re-fetches the NWM 18-hour series on every run and averages whatever hours of day 1 are left in it.
Logged value for issue 11-22: 56.62 → 56.55 → 56.44 → 56.30 → 56.19, then blank from the 04:30Z run on, because
day 1 is no longer in the series and `None` overwrites the row. Every completed issue day in the log has an empty
NWM value; the scorecard's NWM figure (0.553 ft) rests on leftover rows. Even the first value is a partial-day mean
(the hours after the run), not the day. NWM's "day" is also the daylight-time local day, the gauge's is EST.

### 4. The 47-year record is treated as one regime, and it is not (`stationarity.py`)

| | 1979–86 | 1987–94 | 1995–02 | 2003–10 | 2011–18 | 2019–23 |
|---|---|---|---|---|---|---|
| Runoff ratio (flow / rain) | 0.24 | 0.27 | 0.30 | 0.36 | 0.29 | 0.33 |
| Rating residual at Q > 600 cfs, ft | −0.15 | −0.31 | −0.07 | −0.09 | +0.29 | +0.26 |
| GR4J log-flow bias | +0.29 | +0.32 | +0.38 | +0.15 | +0.28 | +0.04 |

The basin now turns about a third more of its rain into flow than in the 1980s, and at high flow the creek stands
about 0.5 ft higher for the same discharge and lake level than before 1995. One rating curve and one GR4J parameter
set are fitted to all of it. The rating therefore reads ~0.27 ft low at high flow today (the residual carry-over
hides part of this at short horizons).

### 5. The 2024–26 test period is no longer a one-shot test

It has been scored for the final test, the tuned range (E2), the rain-aware range (T2), the NWM head-to-head and the
numbers on the how-it-works page, and adoption decisions followed some of those looks. Its numbers, especially range
coverage, are mildly optimistic. The forward log is the clean evidence, which makes item 2 matter.

### 6. One missing input blanks the forecast (from reading the code, not reproduced)

Features must be complete for a prediction. No fallback exists when the 00Z GEFS run is missing (previous run not
used), when radar is missing for a day gridMET has not published, or when a lake series is more than 3 days stale.
The run then publishes empty forecast rows. A whole-run failure (the 16:36Z gridMET outage) keeps the old forecast.

### 7. Smaller points

- Range calibration folds: no gap at year edges, and GR4J is calibrated on all years including the held-out one.
  Slightly optimistic coverage.
- Low water is 52.41 ft on the site, 52.48 in the test harness, 52.49 in the findings log.

### Checked and found sound

Rain and gauge day conventions (gridMET and MRMS 12Z–12Z, gauge EST day); the live radar fill (tested earlier);
lake data is in by the 13:30Z run (11-24 and 11-25); GEFS lead construction; no future data in live inputs.

## Unexplored improvements

Checked against `shingle_findings.md`, `model_test_plan.md` and `divergence_test_plan.md`: none of these was tested.

| # | Idea | Targets | Evidence | Size |
|---|---|---|---|---|
| 1 | **Anchor day 1 on the morning reading**: predict the day-1 mean relative to the 8 am reading, with the mean of the target day's first 8 hours as an input | Day 1 | −29 % (perfect rain) and −31 % (issued rain) from a three-number correction; flood days −70…−80 % | Small–medium: day-1 model and its range |
| 2 | **Update today's figure as the day is observed.** The target day is 1/3 over at issue and the job reruns every 3 h, but day 1 stays the 13Z number | Day 1 as shown | Error vs the frozen forecast: −39 % by 11 am, −52 % by 2 pm, −68 % by 5 pm EST (`day1_screen.py`); the morning issue stays the scored forecast | Small |
| 3 | **Era-aware rating and runoff model**: recent-years fit or recency weights, instead of one fit to 1979–now | Floods, days 3–7 | Item 4 above; rating 0.27 ft low at high flow today | Medium |
| 4 | Rate of rise at issue time, at the gauge and upstream (hourly) | Days 1–7, rising water | Weak: −2 % at day 1, −5 % on rising days at day 5; flood days slightly worse. Mostly covered by 1 | Small |
| 5 | Latest complete GEFS run at issue time (06Z instead of 00Z) | Day 7 | Side note in T6: the 12Z run was 2–5 % better at day 7; never tested | Small |
