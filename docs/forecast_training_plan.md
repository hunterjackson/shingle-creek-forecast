# Plan: teach the model how far to trust the rain forecast

Written 2026-11-25, after the divergence work (`divergence_test_plan.md`). Nothing here has been run yet
except the fact checks listed under "Review".

## Why

The models learn from **observed** rain (1979–2026) and are then fed **forecast** rain, which they treat
as truth. On 2026-11-24 GEFS forecast ~35 mm, the model gave the full +0.36 ft response, and ~4 mm fell.
Everything tried on the 2020–23 forecast archive failed to fix this:

| Tried | Result |
|---|---|
| NBM instead of GEFS, NBM/GEFS average, learned GEFS+NBM blend | Better rain forecast, worse stage forecast (+2 to +7 %): they trim floods along with false alarms |
| Correction model trained on 3 years of forecast-mode errors | Worse everywhere (+0.3 to +16 %): three years do not generalise |
| GEFS at 0.25°, ensemble median, evening refresh | No gain |

The one thing not tried is training on a **long** record of forecasts. NOAA's GEFSv12 reforecast reruns
today's GEFS for every day of 2000–2019 (00Z, 5 members). With the live archive that is ~26 years.

## What it can and cannot buy

- **Ceiling.** With perfect future rain, MAE on 2024–26 falls 20 % at day 3, 39 % at day 5 and 46 % at
  day 7; day 1 barely moves. A model that knows its rain is a forecast cannot reach that. It can only stop
  over-trusting it. A realistic gain is a few percent of MAE on rainy days and far fewer confident busts.
- **The trade-off to watch.** A model trained on forecasts hedges: it responds less to forecast rain. That
  lowers average error and also lowers flood forecasts, which is exactly how NBM failed. So floods are
  scored on their own, and the upper range (p90) is trained to carry the flood risk the p50 gives up.
- **A gain that does not depend on the new model.** 24 years of forecast-mode residuals, with 13 flood
  events instead of 2, to calibrate the likely range. Flood coverage is the range's known weak point.

## Review: what changed from the first version of this plan

| First version | Problem found | Now |
|---|---|---|
| Pilot: compare reforecast with live GEFS over 2020–23 | They never overlap (reforecast ends 2019-12-31; live v12 starts 2020-09-23) | Compare rain statistics across eras, unpaired; test the real question, "train on reforecast, score on live", as a held-out block |
| Retrain the models on forecast rain, 2000+ | Drops 1979–99: only 5 of 18 flood events, but 1,019 of 1,675 low-water days | Keep the 47-year model; add a forecast-trained layer on top (candidate C). The straight retrain (B) stays as a comparison |
| "~60 GB control, ~300 GB five members" | Missing: live-era inputs must match. Cached members are p01–p10 for 2020–23 only, no control | Add ~15 GB: control 2020–26, p01–p04 for 2024–26 |
| Download time unknown | Measured ~9 MB/s with 24 parallel requests | Control ≈ 2 h, five members ≈ 9–10 h; only the basin crop is stored (a few MB) |
| HRRR as the next add-on | T6: even perfect day-1 rain gives nothing at day 1 and ≤ 15 % on wet days 2–3 | Demoted to last; only if the forecast-trained layer shows lead-1 accuracy is what limits it |
| Success = lower MAE | Hedging can lower MAE while missing floods | Flood metrics are a hard gate (below) |
| Looked for a pre-subsetted copy | None: dynamical.org has live GEFS only | Pull 6-h global fields from AWS by byte range, crop, discard |

Other facts checked: 5 members daily (c00, p01–p04; more on Wednesdays, ignored); rain is in 6-h buckets
matching production's lead definition; GEFS 0.25° and 0.5° score the same for this basin (rainfc screen), so
0.25° reforecast and 0.5° live inputs can be mixed.

## Phases and gates

### Phase 0 — free checks on data already on disk (about an hour)

- **0a. Price of a 5-member mean.** Live production uses the 31-member mean; the reforecast has 5. Run the
  current model in forecast mode on 2020–23 with a 5-member mean (cached p01–p05 as a stand-in). The
  10-member mean was already 3–5 % worse at days 4–7. This sets the bar the new layer must clear, and
  decides whether live should feed the 5-member mean, the 31-member mean, or both as features.
- **0b. Freeze the scoring** (next section) before any reforecast data is seen.

### Phase 1 — pilot pull, ~9 GB, ~30 min

Control member, all 7 leads, three full years: 2004 (Charley, Frances, Jeanne), 2017 (Irma), and one quiet
year.

- **1a. Same forecast system?** Rain statistics vs gridMET for these years against the live era: observed /
  forecast ratio, rank correlation, how often ≥ 10 mm is forecast. **Gate:** ratio within ~15 % and
  correlation within ~0.05 of the live-era control. If not, the reforecast is a different animal and the plan stops.
- **1b. First look at floods in forecast mode.** Run the production model on the pilot years with issued
  rain. Until now only Ian and Milton could be scored this way. This shows how much of the flood error is
  rain forecast, i.e. how much there is to win.

### Phase 2 — full pull, staged

1. Control, 2000–2019, leads 1–7: ~60 GB, ~2 h. Modelling can start on this.
2. p01–p04, 2000–2019: ~245 GB, ~8 h, in the background. Needed for the member features and the 5-member mean.
3. Live era to match: control 2020–26 (~5 GB), p01–p04 for 2024–26 (~10 GB).

### Phase 3 — candidates

All run in forecast mode on 2000–2026. A is today's model; the rest are new.

| | What | Why |
|---|---|---|
| A | Production as is, fed reforecast rain | Baseline, now with 13 flood events in forecast mode |
| A+ | A, with range and per-lead scale calibrated on 24 years | Low-risk gain, independent of B–D |
| B | Same model spec retrained on 2000+ with forecast rain in place of observed future rain | The plain "train on forecasts" idea; expected to lose at low water |
| C | Keep A. A second small model, trained on 2000+, takes A's forecast, A's forecast with zero future rain, the forecast rain by lead, and the creek state, and outputs the stage | Learns how far to trust the forecast while keeping the 47-year model's knowledge |
| D | C plus member features: share of members ≥ 10 mm, wettest member, spread | On 11-25 members ran 2–76 mm; the mean hides that |

For C and D the p10 / p90 are trained directly (quantile loss) and then calibrated, so the upper range
can stay high when the p50 hedges.

### Phase 4 — production, only if a candidate passes

Live fetch of c00 + p01–p04 (~12 MB per run), reforecast crop added to the inputs bundle, train workflow
updated, site text updated.

### Phase 5 — HRRR for day 1, optional

2018-08 onward, as one more input to the layer. Only if Phase 3 shows the layer is limited by lead-1 rain.

## Scoring, frozen before Phase 1

- **Folds.** 3-year blocks over 2000–2023 with 7-day gaps; each block is forecast by models that never saw it.
- **The deployment test.** One fold trains on reforecast years only and is scored on 2020-10..2023-12 with
  live GEFS. That is the real condition (train on reforecast, run on live) and must pass on its own.
- **Final look.** 2024-01..2026-11, once, for one frozen candidate. It has been used before (model choice,
  two range checks), so it confirms rather than selects.

| Use | Metric | Gate for adoption |
|---|---|---|
| Everyday | MAE by day ahead | Not worse overall (90 % CI) |
| Rainy days | MAE on days with ≥ 5 mm forecast | ≥ 5 % better at days 2–5, CI excluding 0 |
| Confident busts | MAE on days with ≥ 20 mm forecast and < 10 mm observed | Reported; expected to improve most |
| Floods (13 events, 2000–2023) | Crest error and timing per event; hit and false-alarm rate for ≥ 59 ft, for p50 and for p90 | p50 crest error not worse on average; p90 hit rate not lower |
| Low water | MAE on days ≤ 52.48 ft | Not worse |
| Likely range | Coverage overall, rainy days, flood days; pinball loss | Coverage ≥ today's on each; pinball not worse |

## Risks

- **Reforecast ≠ live.** Reforecast runs start from a reanalysis, live runs from the operational analysis.
  Phase 1a and the deployment-test fold exist to catch this.
- **5 members vs 31.** Phase 0a prices it; D can take both means as inputs if that helps.
- **Hedging lowers flood forecasts.** Hard gate above.
- **GEFS upgrade.** A future GEFSv13 would break the match between training and live inputs until its own
  reforecast is published. The 47-year base model (A) stays as the fallback.
- **Cost.** ~320 GB of transfer in total and a day or two of work; the phases are ordered so the cheapest
  evidence comes first and each gate can stop the spend.
