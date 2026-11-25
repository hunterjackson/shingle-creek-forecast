# Plan: teach the model how far to trust the rain forecast

Written 2026-11-25, after the divergence work (`divergence_test_plan.md`). Phase 0 is done (results and the
frozen scoring below). Phase 1 is done: rank skill matches, but the rain-total gate fails (reforecast ~20 % wetter
at leads 1, 3, 4).

**Closed 2026-11-25. Phases 2–5 will not be run.** The goal was better day-1 forecasts, and 1b shows the rain
forecast is worth almost nothing at day 1 (it recovers 22 % of a 0.004 ft gap; MAE 0.058 c00 vs 0.054 perfect), as
T6 found for perfect day-1 rain. The expected gains are at days 2–7, which the owner does not consider worth the cost.

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
- **A gain that does not depend on the new model.** 24 years of forecast-mode residuals, with 12 flood
  events instead of 1 (Ian; Milton is in the final test), to calibrate the likely range. Flood coverage is the range's known weak point.

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
| A | Production as is, fed reforecast rain | Baseline, now with 12 flood events in forecast mode |
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

## Phase 0 results (2026-11-25)

### 0a — price of a 5-member mean (`experiments/reforecast/p0_members.py`, `p0_members_report.txt`)

Today's model in forecast mode, 2020-10..2023-12 (1,182 days). The 5-member mean is a stand-in made from
cached p01–p05 (no control member is cached). Scale refit to the 5-member mean:

| MAE change vs 31-member mean | d1 | d2 | d3 | d4 | d5 | d6 | d7 |
|---|---|---|---|---|---|---|---|
| All days | +1.7 % | +0.9 % | +1.7 % | +3.3 % | +4.5 % | +5.1 % | +6.5 % |
| Rainy days | +5.4 % | +1.8 % | +2.4 % | +3.9 % | +5.3 % | +5.8 % | +6.9 % |
| 10-member mean, all days | 0.0 % | +0.3 % | +0.4 % | +2.8 % | +2.7 % | +3.8 % | +4.8 % |

The 90 % CI excludes 0 at days 1 and 3–7. Lead-1 rain skill barely changes (Spearman 0.777 vs 0.788 for the
31-member mean). The cost grows with lead time, where 31 members average out more noise. Flood days (14, all Ian)
look better with 5 members, but that is noise. This is an upper bound: a model *trained* on 5-member rain
learns that noise.

**Decision.** The 31-member mean is worth up to ~6 % at days 4–7, so it is not given up. Live, the base model A
keeps the 31-member mean. The 5 members feed only the new layer (C/D) as extra inputs, and in training years A
runs on the 5-member mean, the only option. The deployment-test fold (2020–23) scores exactly this arrangement.
Every candidate is compared with **production on the 31-member mean**, so it has to overcome the handicap.

### 0b — frozen scoring

Fixed 2026-11-25, before any reforecast data has been downloaded or looked at.

**Data periods**

| Period | Rain forecasts | Use |
|---|---|---|
| 2000-01-01..2019-12-31 | Reforecast, 5 members | Training and out-of-fold scoring |
| 2020-01-01..2020-09-30 | None (between archives) | Excluded |
| 2020-10-01..2023-12-31 | Live GEFS (31-member mean; c00 + p01–p04 for the layer) | Deployment-test fold |
| 2024-01-01..2026-11-12 | Live GEFS | Final look, once, one frozen candidate |

**Folds.** Reforecast years in blocks 2000–02, 03–05, 06–08, 09–11, 12–14, 15–17, 18–19. Each block is
predicted by models trained on the other blocks, with a 7-day gap at both edges. The **deployment-test fold**
predicts 2020-10..2023-12 with models trained on reforecast years only, fed live inputs. The base model A
keeps its own long-record training (perfect rain, OOF as now); only the new layer follows these folds.

**Flood events.** `experiments/reforecast/flood_events.csv`, generated by `p0_events.py`: runs of daily
stage ≥ 59 ft, merged when less than 7 days apart. This keeps the 2004 hurricanes, separated by 8-day dips, as
three events. 11 events in 2000–2019 (2001-09, 2002-06, 2002-12, 2003-01, 2003-08, Charley, Frances, Jeanne,
2009-05, 2011-10, Irma), Ian in the deployment fold, and Milton in the final test. Scoring window: first day − 7
to last day + 3.

**Definitions**

- *Rainy day* for horizon h: basin-areal 5-member-mean rain over leads 1..h ≥ 5 mm (the same members in every period).
- *Confident bust*: lead-1 basin 5-member-mean rain ≥ 20 mm and observed gridMET basin rain on t+1 < 10 mm.
- *Low water*: observed stage at t+h ≤ 52.48 ft.
- *Crest error* (per event, per horizon h): max p50 over the target days in the window (each issued h days
  earlier) minus the observed crest. *Timing error*: the date of that max minus crest_date.
- *Event hit* (per event, per h): any target day in the window with p50 (or p90) ≥ 59 ft.
- *False alarm event*: a run of target days with forecast ≥ 59 ft that touches no event window, counted per h.
- *Paired CI*: moving-block bootstrap, 60-day blocks, 2,000 resamples, 90 % (`fc.boot`).

**Gates** (candidate vs production on the 31-member mean; production gets the 5-member mean in 2000–2019, where nothing else exists)

| # | Use | Gate |
|---|---|---|
| G1 | All days | At no horizon 1–7 is the candidate significantly worse (lo90 of MAE diff > 0) |
| G2 | Rainy days | MAE ≥ 5 % better at each of days 2–5, with hi90 < 0 |
| G3 | Floods (11 + Ian) | Mean \|crest error\| of p50 not worse by > 0.05 ft at days 1–5; p90 event hits ≥ production's at every h |
| G4 | Low water | At no horizon is the candidate significantly worse |
| G5 | Likely range | Coverage overall ≥ production − 1 pt; rainy and flood-day coverage ≥ production; pinball ≤ production + 1 % |

- The **out-of-fold reforecast years** must pass G1–G5.
- The **deployment-test fold** must pass G1, G2, G4 and G5 on its own. Ian is reported, not gated.
- Confident busts and timing errors are reported, not gated.
- **Selection** uses only the out-of-fold years and the deployment fold. One candidate is then frozen and scored
  once on 2024-01..2026-11 (G1 and G2 point estimates and Milton reported). It is adopted if those hold.
- Phase 1a's rain-statistics gate stays as written above (ratio within ~15 %, Spearman within ~0.05 of live).

## Phase 1 results (2026-11-25)

Pulled c00 for 2004, 2013 (quiet year: no day >= 59 ft, full gauge record, typical mean stage) and 2017: ~9 GB
transferred, 1.8 MB kept. Live c00 2020-10..2023-12 for the yardstick: 3.5 GB. Scripts `p1_fetch.py`,
`p1_analyze.py`, `p1_checks.py`; reports `p1_report.txt`, `p1_checks_report.txt`.

### 1a — same forecast system? Gate fails on rain totals, passes on rank skill

| Lead | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| obs/fc, reforecast c00 (3 years) | 1.17 | 1.28 | 1.26 | 1.37 | 1.52 | 1.40 | 1.42 |
| obs/fc, live c00 (2020-10..2023-12) | 1.50 | 1.44 | 1.56 | 1.65 | 1.48 | 1.46 | 1.60 |
| difference (gate ±15 %) | **−22 %** | −11 % | **−19 %** | **−17 %** | +2 % | −4 % | −11 % |
| Spearman, reforecast − live (gate ≥ −0.05) | −0.01 | +0.01 | +0.03 | +0.03 | +0.06 | +0.03 | +0.06 |

- **Ranking of rain days: same.** The reforecast picks wet days as well as live does, at every lead.
- **Totals: the reforecast is ~20 % wetter relative to gridMET at short leads.** It under-forecasts less. Not
  year-to-year noise: at leads 1 and 3 every pilot year (1.05–1.33) is below every live year (1.40–1.70);
  at lead 2 the ranges just touch.
- **Not resolution.** Live GEFS at 0.25° and 0.5° give the same lead-1 totals (ratio 0.99).
- Likely cause: reforecast runs start from a reanalysis, live runs from the operational analysis (the risk
  listed below). A drift in gridMET between eras cannot be ruled out with this data.
- **By the frozen rule the plan stops here.** The difference is a level shift (a per-lead factor), not a loss of
  skill, so it could be corrected by rescaling reforecast rain to the live level, with the deployment-test fold
  (train on reforecast, score on live) as the judge of whether that transfers. That would be a change to a
  frozen gate and needs an explicit decision.

### 1b — production model in forecast mode on the pilot years (OOF by year, 1,031 issue days)

All-day MAE (ft) with observed future rain (perfect), c00 forecast, and no future rain (zero):

| Day | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| perfect | 0.054 | 0.130 | 0.185 | 0.230 | 0.268 | 0.297 | 0.318 |
| c00 | 0.058 | 0.138 | 0.210 | 0.292 | 0.363 | 0.426 | 0.485 |
| zero | 0.058 | 0.164 | 0.262 | 0.356 | 0.437 | 0.512 | 0.581 |
| share of perfect-rain gain recovered | 22 % | 75 % | 68 % | 51 % | 44 % | 40 % | 37 % |

- **Day 1: rain forecast is worth nothing** (as T6 found). Days 2–3 recover most of the gain; days 5–7 under half.
- **Floods** (Charley, Frances, Jeanne, Irma): mean |crest error| at days 3 / 4 / 5 is 0.44 / 1.11 / 1.84 ft with
  c00 vs 0.15 / 0.69 / 0.84 ft with perfect rain. Every event is hit (p50 ≥ 59) at every horizon by every source.
  Most of the c00 excess is over-forecasting Frances (c00 forecast 139 mm for 09-05, 107 mm fell) and Jeanne at
  days 4–7, but the model overshoots Frances at days 5–7 even with perfect rain (+1.6 to +3.2 ft).
- **Low water: forecast rain makes it worse than assuming none** (day 7 MAE 0.53 c00 vs 0.37 zero), but perfect rain
  is also biased high there (+0.24 ft at day 7) while zero is unbiased. Part of this is the model at low water, not
  the rain forecast.
- **Era-matched scale** (per-lead scale refit on the reforecast, leave-one-pilot-year-out, ~1.2–1.5 vs live's
  ~1.45–1.53): days 4–7 MAE 3–5 % lower than with live's scale; days 1–3 unchanged (raw rain there).
- Room for the layer, rainy days: c00 vs perfect MAE is 0.386 vs 0.319 at day 3 and 0.560 vs 0.391 at day 5.

## Risks

- **Reforecast ≠ live.** Reforecast runs start from a reanalysis, live runs from the operational analysis.
  Phase 1a and the deployment-test fold exist to catch this.
- **5 members vs 31.** Phase 0a prices it; D can take both means as inputs if that helps.
- **Hedging lowers flood forecasts.** Hard gate above.
- **GEFS upgrade.** A future GEFSv13 would break the match between training and live inputs until its own
  reforecast is published. The 47-year base model (A) stays as the fallback.
- **Cost.** ~320 GB of transfer in total and a day or two of work; the phases are ordered so the cheapest
  evidence comes first and each gate can stop the spend.
