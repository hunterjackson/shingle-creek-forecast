# Test plan: other model types

Written 2026-11-23. **Stages 0–1 were run the same day; results are at the end of this file** and in
`shingle_findings.md`. The expected-gain column below is the estimate made before running, kept as written.

## Why this plan exists

The production forecast uses LightGBM for all three learned parts (stage model, flow model, lake
model). The only alternative ever compared was ridge regression, on validation 2016–19. No
hyperparameter search and no blend-weight search is recorded either: the LightGBM settings and the
0.5/0.5 blend were set by hand. Accuracy is the project's main goal, so this gap should be closed
with a test, not an argument.

## What the forecast is for, and how each use is scored

| Use | What matters | Metric |
|---|---|---|
| Everyday level, days 1–7 | typical miss | MAE per day ahead |
| Flood warning (59 / 60 / 61 / 62 ft) | catching the rise, crest height and timing | MAE on days ≥ 59 ft; crest error and timing per event; hit and false-alarm rate for ≥ 59 ft |
| Low water (≤ 52.48 ft, lowest tenth of days) | the slow fall, driven by Lake Toho | MAE on low-water days |
| "Likely range" shown on the site | 8 in 10 days inside, including during floods | 80 % coverage overall and on flood days; band width; pinball loss (new) |
| Runs unattended on GitHub Actions | every 3 h on a free CPU runner, monthly retrain | forecast run ≤ 10 min (now ~3), retrain ≤ 30 min (now ~3), no GPU |

## What the data allows

| Data | Size | Consequence for model choice |
|---|---|---|
| Daily stage, flow, lakes, structures, gridMET rain | 16,637 stage days, 1979–2026 | Enough for trees and small conceptual models; small for a neural net trained on one basin |
| Flood events ≥ 59 ft | 18 events, 118 days; 8 events reach 60 ft; record 64.15 ft (Ian) | Flood results rest on a handful of events; report them event by event, not as one average |
| Low-water days | 1,675 | Well sampled |
| Morning gauge readings | 2007 onward (7,231 days) | Missing for over half the record; LightGBM handles this natively, other models need masking |
| Issued GEFS rain forecasts | 2020-10 onward (about 2,240 days) | Too short to *train* a flexible model on forecast rain; only 2 floods (Ian, Milton) can be scored with real forecasts |
| Radar rain | 2018-07 onward | Already tested as a model input: no skill. Not revisited here |

Evaluation periods and their flood content:

| Period | Days | Flood events ≥ 59 ft | Status |
|---|---|---|---|
| Out-of-fold 1979–2023 (3-year blocks, observed future rain) | 15,581 | 17 | Reusable; each flood is predicted by a model that never saw it |
| Forecast mode 2020-10..2023-12 (issued GEFS, out-of-fold by year) | 1,186 | 1 (Ian) | Partly used already for scale and band choices |
| Test 2024-01 onward | 1,056 | 1 (Milton) | Used once for the production model. One more look, for one challenger |

## Where the error is now

Test period 2024–26, MAE in ft:

| Days ahead | 1 | 3 | 5 | 7 |
|---|---|---|---|---|
| Persistence | .105 | .260 | .385 | .483 |
| Production, issued GEFS rain | .034 | .141 | .257 | .320 |
| Production, perfect future rain | .033 | .113 | .157 | .174 |
| Share of live error left with perfect rain | 97 % | 80 % | 61 % | 54 % |

Reading: a learner that is X % better with perfect rain should be about X % better live at day 1,
0.8 X at day 3 and 0.5 X at day 7. Nearly half the day-7 error is rain-forecast error that no
learner removes. The clearest known weaknesses are not in the average:

- Flood days: the likely range covered 17–50 % of flood days on test (target 80 %), 43–71 % on validation.
- The stage model cannot predict above its training range: Ian at day 7 with perfect rain, −2.33 ft
  and 7 days late. The rating-curve model was +0.78 ft; the blend −0.78 ft.
- GR4J fits this basin poorly (flow NSE 0.66, two parameters at their bounds, 1,180 cfs simulated
  against 451 observed on 2026-11-21), yet its projected change is the top input from day 3 on
  (20–45 % of gain).
- Low water at days 5–7 is only slightly better than persistence (.104–.137 against .111–.146).

## Candidates, in the order they would be run

Gains are for MAE with perfect rain unless stated. "Live" applies the shares above.

| # | Candidate | Use it targets | Expected gain | Basis for the estimate | Size |
|---|---|---|---|---|---|
| A | **Tuning controls.** Random search over LightGBM settings; CatBoost at defaults; blend weights per day ahead, chosen out-of-fold | Everyday | 0–3 % at any day ahead | Settings and weights were never searched; boosted-tree libraries usually land within a few percent of each other on tables this size | Small |
| B | **Stage model that can extrapolate.** LightGBM with linear leaves; ridge base with LightGBM on its residuals; rain inputs constrained so more rain never lowers the level | Floods | Everyday ±2 %; flood-day MAE at days 5–7 down 10–30 % | Ian: halving the stage model's −2.33 ft miss moves the blend from −0.78 to about −0.2 ft. Low confidence: 8 events reach 60 ft | Small |
| C1 | **Better conceptual runoff model** in place of GR4J: GR6J (adds a slow store for low flows and a second exchange term), GR4J fitted to log flow | Days 3–7, recession, low water | 3–8 % at days 4–7; 2–4 % live | Adding GR4J at all cut day-7 MAE 21 % (.326 → .258); a better simulator should add a fraction of that | Small |
| C2 | **LSTM runoff simulator** (basin rain and evaporation sequences → flow), used as inputs the way GR4J is | Days 3–7, floods | 0–10 % at days 4–7; 0–5 % live | Published LSTM wins come from training on hundreds of basins; here one basin, 17,492 flow days. Run only if C1 shows the simulator's quality matters | Large: adds PyTorch, per-fold retraining |
| D | **End-to-end sequence model** (LSTM or GRU: a year of history in, 7 days out, quantile loss) | All | Between 10 % worse and 5 % better; worse at days 1–2 | Day-1 error (.034 ft) is already below the median day-to-day change (.04 ft); the morning reading it relies on is missing before 2007. Run only if C2 beats C1 | Large |
| E | **Model for the likely range.** LightGBM quantile regression for the 10th and 90th percentiles, conformally corrected, with forecast rain, GEFS spread and GR4J projected change as inputs | Flood risk | MAE unchanged; flood-day coverage from 17–50 % toward 60 %+ with ≤ 20 % wider bands on ordinary days | Bands now depend only on the size of the predicted change. An earlier spread-scaled variant did not clearly help, so confidence is low | Medium |
| F | **Simpler lake model.** Lake Toho relaxing toward its regulation schedule plus rain and inflow terms (ridge), alone or blended with LightGBM | Low water | ≤ 2–5 % of RMSE, low-water days only | Measured ceiling: knowing the future lake level perfectly correlates only .17 / .27 / .31 with residuals at days 1 / 3 / 7 | Small; lowest priority |

Not planned, with reasons:

- Random forests and extra-trees: same family as LightGBM, same inability to extrapolate.
- Gaussian processes: 16,000 rows needs approximations, with no clear advantage here.
- Time-series foundation models: they cannot use the rain forecast, the lake schedule or the morning reading the way the current inputs do.
- Multi-basin LSTM: needs a forcing and flow dataset for dozens of gauges. Considered only if D is competitive.
- Using GEFS members instead of the mean: an input change, tracked separately. It attacks the larger half of day-7 error and may be worth more than anything in this plan.

## Protocol

Same harness for every candidate (`screen.oof_point`, `compare.score_models`, `train.backtest`).
New code is test-first, as usual.

**Stage 0 — reference and noise floor.** Re-run the frozen production configuration through the
harness. Train it again with 5 different seeds. The paired moving-block bootstrap (120-day blocks)
of the MAE difference between seeds is the noise floor: the smallest difference worth believing.
Add pinball loss to `evaluate`.

**Stage 1 — capacity test, observed future rain, out-of-fold 1979–2023.** All of A, B, C1, E, F.
Reference to beat: production out-of-fold MAE .054 / .143 / .234 at days 1 / 3 / 7. A candidate
moves on if it beats production by more than twice the noise floor on its target metric, and is
not worse by more than the noise floor on the others. Floods are reported per event (17 events).
C2 runs only if C1 passes; D only if C2 beats C1.

**Stage 2 — forecast mode, issued GEFS, 2020-10..2023-12.** Survivors only, combined into one or
two full configurations. Checks that the gain survives real rain forecasts and that Ian is not
made worse.

**Stage 3 — test, once.** At most one challenger, written down and committed before the run.
Paired comparison with production on the same days, 2024-01 onward.

**Stage 4 — shadow run.** The challenger runs beside production in the Actions workflow for
60 days and both are logged. This checks run time, reliability and day-to-day accuracy. It will
probably contain no flood.

## Decision rule, fixed in advance

Replace the production model only if, in stages 2 and 3 together:

1. MAE at days 4–7 improves by at least 5 %, or flood-day MAE by at least 15 %, with the 90 %
   bootstrap interval of the paired difference excluding zero; and
2. MAE at days 1–3 is not worse by more than 2 %, and low-water MAE not worse by more than 5 %; and
3. crest error is not worse for both Ian and Milton; and
4. the run-time limits above hold.

For E (likely range): overall 80 % coverage stays between 77 % and 83 %, flood-day coverage
improves in the pooled out-of-fold results, and ordinary-day width grows by no more than 20 %.

If nothing clears stage 1, the result is recorded in `shingle_findings.md` as "LightGBM confirmed
against A–F" and the question is closed.

## Known limits of this plan

- Only two floods can be scored with real rain forecasts. Flood conclusions for B, C and E rest
  mainly on the 17 out-of-fold events with perfect rain. A possible fix, not yet checked: NOAA's
  GEFS reforecast archive (2000–2019) covers 11 more flood events; pulling it only around those
  events may be affordable.
- The forecast-mode period has already informed the rain scale and band choices, so stage 2 is
  slightly optimistic for the production model. Stage 3 is the fair comparison.
- The expected-gain column is judgement. Stages 0 and 1 are small and cheap and replace most of it
  with measurements before any large item (C2, D) is started.

## Results (2026-11-23)

Harness, logs and reports: `experiments/model_test/` (`stage1_report.txt`, `stage1_bands.txt`).

| # | Candidate | Expected | Measured (out-of-fold 1979–2023) | Outcome |
|---|---|---|---|---|
| A | Tuning: seed bag, blend weights, CatBoost, 10 LightGBM settings | 0–3 % | Seed bag −0.5…−0.9 % everywhere; others trade 2–7 % better everyday/low water for 3–8 % worse floods | Only the seed bag passes stage 1; far below the 5 % bar |
| B | Stage model that can extrapolate | flood days at h5–7 −10…−30 % | Best: ridge-residual −2…−5 % on flood days at h5–7, but +3…+8 % worse at days 1–3; crest errors unchanged | Fail |
| C1 | GR6J / refit GR4J | 3–8 % at h4–7 | −1…+1 % everyday, flood days mostly worse | Fail → C2 and D not run (gated) |
| C2 | LSTM runoff simulator | 0–10 % | not run | Gate not met |
| D | End-to-end LSTM | −10…+5 % | not run | Gate not met |
| E | Quantile model for likely range | flood coverage to 60 %+, ≤ 20 % wider calm bands | Flood coverage +9…+12 points (h1, h5, h7), pinball −11…−16 %, calm width +30…+51 % at h3–7 | Fail on width rule |
| F | Simpler lake model | ≤ 2–5 % low water | 1–5 % worse | Fail |

Stage 0 noise floor (sd across 5 seeds): overall MAE 0.1–0.3 %, flood days 1.1–1.8 %, low water 0.5–0.9 %.

Decision: production model unchanged; stages 2–4 not run because no stage-1 survivor can reach the adoption
threshold. The 2024–26 test period was not used. Open follow-ups (not part of this plan): GEFS ensemble members as an
input (rain-forecast error is the larger half of day-7 error); a likely-range model with a width penalty, since CQR
showed real flood-coverage gains.

## Follow-ups (2026-11-23)
- E tuned (E2_calmprod): passes the band rule on 1979–2023 OOF; in forecast mode and on the one-shot 2024–26 test it
  improves days 3–7 (pinball −11…−14 %, Milton flood-day coverage 2→5 of 6 at h7, h7 coverage .76→.81) but lowers
  day-1 flood coverage. Candidate for production at h ≥ 2. Details in shingle_findings.md.
- GEFS individual members: every way of using them was worse than the ensemble-mean file. Ruled out.
