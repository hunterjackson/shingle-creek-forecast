# Shingle Creek forecast — data & modeling findings log

Running log of facts discovered while building the SHIF1 forecast. Newest at bottom.

## 2026-11-20 — data sources
- NWPS: SHIF1 has no NWS forecast (`fcst_not_current`). Flood stages 59/60/61/62 ft.
  Record crest 64.21 ft 2022-10-01 (Ian); 62.33 ft 2017-09 (Irma); 60.61 ft 2024-10 (Milton).
- Open-Meteo ERA5-Land archive returns **null precipitation** here → historical rain/ET from
  **gridMET** (4 km, 1979+). Reanalysis rain is far lower than gridMET in big events
  (Milton 10-09: ~70 mm reanalysis vs 150–290 mm gridMET) → forecast rain will need bias
  correction against gridMET climatology.
- NWM via NWPS: medium_range came back empty for our reach on 2026-11-20; short/long range ok.
- NWM v3 retrospective zarr: one reach 2015–2023 ≈ 700 MB of chunk reads.

## 2026-11-20 — first full fetch (1979-01-01 .. 2026-11-19)
- Campbell 02264495: discharge complete every day (USGS estimates gaps); **stage missing
  855 days** (worst 1987–1999). Stage mean 54.24, min 49.65, max 64.15 (daily mean).
- **Annual minimum stage ≈ 52.0 ft almost every year** (only 1979 50.2, 1987 50.6, 2004 49.6
  dip lower — likely Lake Toho drawdown years). Strong sign that the Toho level sets the
  low-water floor at Campbell via backwater → lake data is essential for low-water skill.
- Upstream 02263800 stage is on a local datum (3.6–11.4 ft); discharge mean 106 cfs vs 207 at
  Campbell.
- gridMET: 88 cells (11×8), no NaNs, basin-mean annual rain ~1350 mm (885–1844), ET0 ~1570 mm.
  31 cells touch the basin, 14 are >50 % inside.

## 2026-11-20 — first relevance analysis (gridMET, train ≤2015)
- Daily stage response: cross-correlation of every in-basin cell's rain with Δstage peaks at
  **lag 1 day** (r≈0.55) — at daily resolution the gauge responds within ~1 day.
- Greedy partial-corr selection (controls stage, dstage_1/3, log_q, log_q_up forced first):
  the first rain feature is always the cell just NE of the gauge (28.2750,-81.4333) —
  pr_past1 at h=1 (partial r .35), pr_next3/pr_next7 at h=3/7 (partial r .38/.41).
  After that, gains are tiny (partial r ≤ .16) and picks are scattered, often outside the basin.
- **gridMET is too spatially smooth for cell-level selection:** in-basin daily rain pairwise
  corr median 0.96 (min 0.86); still 0.96 at 10 km and 0.82 at 50 km separation; in-basin and
  out-of-basin cells score identically (mean Spearman 0.361 each); one redundancy cluster at
  r≥0.9, only 14 even at r≥0.99. Real Florida convective rain decorrelates over a few km, so
  this is a smoothing artifact of the product (daily timing from ~12 km NLDAS), not the climate.
- Implication: per-cell analysis needs radar-based rainfall (SFWMD NEXRAD ~2 km, or NOAA
  Stage IV 4 km 2002+ / MRMS 1 km). Decision pending with user.

## 2026-11-20 — stage–discharge relation at Campbell (2010+)
- Not single-valued below ~400 cfs: Q 50–100 cfs → stage 52.06–55.17 ft (IQR 53.1–54.5);
  Q 100–200 → 52.2–55.4. Tightens only above ~400 cfs (800–1600 cfs → 56.6–60.8).
  ⇒ stage is controlled by Lake Toho backwater at low/moderate flow. Rating must be
  stage = f(Q, lake). rating.py has FreeRating (h0 + a·Q^b) and BackwaterRating
  (lake + c·Q^d); fit on real data once DBHYDRO lake stage is available.

## 2026-11-20 — exploratory GR4J on real flows (basin-frac-weighted gridMET, exploratory only)
- Water balance: P 3.69, ET0 4.30, Q 1.09 mm/d (area 466.2 km²) → runoff ratio 0.29.
- Calibrated 1979–2015, validated 2016–2019 (flows, mm/d):
  | objective | x1 | x2 | x3 | x4 | train | val NSE | val KGE |
  |---|---|---|---|---|---|---|---|
  | nse | 14 | −10 (bound) | 658 | 3.3 | 0.64 | 0.79 | 0.87 |
  | nse_sqrt | 14 | −10 (bound) | 423 | 3.2 | 0.69 | 0.65 | 0.75 |
  | kge | 36 | −0.66 | 469 | 3.2 | 0.80 (KGE) | 0.61 | 0.67 |
- x1 collapses to ~14–36 mm and x2 hits its bound → GR4J structure is a poor fit for this
  urbanized (Orlando, fast runoff) + wetland/high-water-table basin. Usable as a backbone
  for the hybrid, but expect the ML corrector to do a lot of work. Lumped input here was a
  basin-frac-weighted mean — for the real model, GR4J's input should be built from the
  selected cells (decision pending, user objected to blind averaging).

## 2026-11-20 — DBHYDRO lakes + Campbell/Toho relation
- Fetched 8 named series 1979+ (kiss_stage only 2009+, s65_q 1996+). Toho stage 48.5–56.8,
  mean 53.7 ft NGVD29. Chunked requests needed (one big request → connection reset).
- **Below ~200 cfs Campbell stage ≈ Lake Toho stage**: median (Campbell − Toho) = −0.02..0.15 ft,
  corr 0.97–0.99 in every Q bin ≤ 200 cfs. Head rises with Q: median 0.6 ft @200–400 cfs,
  1.6 @400–800, 3.2 @800–1600, 5.7 ft >1600 cfs. ⇒ Campbell gauge datum is effectively
  NGVD29 and low water at Campbell *is* the (managed) Toho level.
- Ratings fit ≤2015, tested 2016–2026 (stage from observed Q + Toho):
  | rating | test MAE | RMSE | MAE ≥59 ft (n=32) | MAE low decile |
  |---|---|---|---|---|
  | free h0+aQ^b | 0.627 | 0.741 | 0.749 | – |
  | backwater toho+cQ^d | 0.263 | 0.434 | 1.512 | – |
  | **combined smooth-max** | **0.136** | **0.221** | **0.517** | **0.049** |
  Combined params c=3.04e-5 d=1.48 h0=47.68 a=0.421 b=0.457 k=1.02. Overpredicts the largest
  crests (Irma 61.85 → 63.1, Ian 63.82 → 65.4; Milton 60.56 → 60.57 exact) — floodplain
  flattening and/or estimated peak Q.
- **Modeling implication:** stage(t+h) = Rating(Q(t+h), Toho(t+h)). Forecast Q (rainfall-runoff:
  GR4J/ML) and Toho (managed lake: regulation schedule, S-61/S-59 flows, inflow) separately and
  combine through the rating → physically structured, extrapolates to floods via Q.

## 2026-11-20 — NWM v3 retrospective vs observed (reach 21476600)
- Pulled 1979-02-02..2023-01-30 daily (local-day mean, ≥24 h) in 41 s → data/nwm_retro.parquet.
- Flow skill vs USGS Campbell: NSE 0.29 / KGE 0.45 (1979–2015), 0.11 / 0.34 (2016–19),
  −0.25 / 0.23 (2020–23). **Bias +43…48 %**; corr 0.76–0.83. Peaks ~2× too high
  (2004 3310 vs 1980; 2017 5606 vs 2690; 2022 9819 vs 3750 cfs).
- NWM flow → stage through our CombinedRating with *observed* Toho, 2016–2023: MAE 0.62 ft,
  **MAE ≥59 ft 3.79 ft** (observed flow through same rating: 0.15 ft).
- ⇒ Raw NWM is a weak benchmark here; but its correlation makes a bias-corrected NWM flow a
  plausible candidate *feature*. (Exploratory GR4J w/ gridMET already reached val NSE 0.79.)

## 2026-11-20 — MRMS radar plan
- IEM archive mtarchive.geol.iastate.edu/YYYY/MM/DD/mrms/ncep/<product>/. Gauge-corrected
  daily products: GaugeCorr_QPE_24H (~2018-10 → 2020-10, 4.7 MB/file), MultiSensor_QPE_24H_Pass2
  (2020-10 → now, 3.1 MB). Before ~2018-10 only GaugeCorr_QPE_01H (0.8 MB × 24/day ≈ 24 GB
  for 2015–18) or RadarOnly_QPE_24H — skipped for now.
- Use the 24 h file stamped 05:00Z on D+1 = local-standard-time calendar day D (USGS dv uses
  LST). GRIB2 read via cfgrib + eccodes + eccodeslib (bundled binary).

## 2026-11-20 — first real model comparison (val 2016–2019; train ≤2015)
Setup: gridMET rain at the top cell 28.2750,-81.4333 (pr_past1/3/7/30, pr_nextH = *observed*
future rain, wb_past30/90); stage/lake/flow controls; leave-one-year-out cross-conformal bands.
Hybrid = RatingHybrid(flow DirectRegressor log_q, lake DirectRegressor toho) + residual carry.

| MAE ft | h1 | h3 | h7 | ≥59 h7 | low (≤52.41) h7 | cov80 h7 |
|---|---|---|---|---|---|---|
| persistence | .134 | .334 | .603 | 4.65 | .237 | .80 |
| ARX ridge | .105 | .273 | .477 | 3.88 | .292 | .80 |
| ARX lgbm | .088 | .219 | .336 | **2.10** | .191 | .78 |
| hybrid ridge | .105 | .247 | .401 | 4.19 | .185 | .79 |
| **hybrid lgbm** | **.088** | **.209** | **.326** | 2.87 | **.101** | .80 |

- Residual carry-forward fixed hybrid h1 (.171 → .088).
- Hybrid wins overall and halves low-water error; ARX-lgbm still better at flood level (few
  flood days in val — essentially Irma 2017; rating over-predicts the biggest crests).
- Caveat: future rain is *observed* (perfect forecast). Real skill at h≥3 will be lower;
  rain-realism test pending.

## 2026-11-20 — adding GR4J (lumped, basin_frac-weighted gridMET; calibrated ≤2015, train NSE .64)
| MAE ft (val 16–19) | h1 | h3 | h7 | ≥59 h7 | low h7 | Irma h3 peak err / timing |
|---|---|---|---|---|---|---|
| ARX lgbm | .088 | .219 | .336 | 2.10 | .191 | −0.73 / +2 d |
| ARX lgbm + gr4j feats | .086 | .190 | .284 | 1.80 | .172 | −0.92 / +2 d |
| hybrid lgbm | .088 | .209 | .326 | 2.87 | .101 | −0.71 / +2 d |
| **hybrid lgbm + gr4j feats** | .085 | .180 | **.258** | **1.03** | .108 | −0.49 / +3 d |
| **hybrid lgbm, gr4j offset** | **.084** | **.174** | .263 | 1.55 | .109 | +0.98 / −1 d |
| hybrid ridge, gr4j offset | .099 | .200 | .282 | 3.16 | .106 | −0.50 / +1 d |
- GR4J states/flow cut flood error at h7 from 2.9 → 1.0 ft. Offset version extrapolates
  (overshoots Irma peak +1 ft, partly the rating's known crest over-prediction) and peaks a day
  EARLY → consistent with a gridMET day-labelling shift; check with MRMS LST days.
- Still with *perfect* future rain.

## 2026-11-20 — MRMS vs gridMET (first 794 days, 2018-07 .. 2020-08, basin mean)
- Totals: MRMS ~1406 mm/yr vs gridMET 1568 (in-basin cells >50 %).
- Day alignment: corr(MRMS[t], gridMET[t+k]) = .156/.391/**.851**/.159/.142 for k=−2..2.
  MRMS LST-day rain leaks into gridMET day t−1 (r .39 vs .16 the other way) → gridMET day ≈
  07–07 local, overnight rain labelled a day early. Explains the GR4J-offset model peaking
  ~1 day early.
- Stage response: corr(rain[t−lag], Δstage[t]) MRMS lag0 .28, **lag1 .60**, lag2 .10 → real ~1-day
  response (not just a labelling artifact).
- **MRMS has real spatial structure:** in-basin pixel pairwise daily corr median .66 (min .38) vs
  gridMET .96 → per-cell relevance analysis is meaningful with radar.

## 2026-11-20 — archived rain forecasts by lead time
- Open-Meteo Previous Runs API (precipitation_previous_dayN) has lead-dependent forecasts only
  from ~2024 (best_match; gfs/ecmwf/nbm/icon also null for 2022–2023). Historical Forecast API
  is stitched short-lead only.
- Milton window 2024-10-05..09 at 28.3,-81.45: lead1 141 mm, lead3 111, lead5 47, lead7 24
  (MRMS basin mean on 10-09 alone ≈140 mm) → skill collapses beyond ~3 days.
- Plan: (1) score finalists with real lead-k forecasts 2024-01→now (test period, incl. Milton);
  (2) fit a lead-dependent forecast-error model from 2024+ and use it to degrade observed rain
  in earlier periods (and possibly in training, so models learn appropriate trust in rain).

## 2026-11-20 — bracketing the GR4J hybrid: perfect vs NO future rain (val 2016–19)
| | MAE h1/h3/h7 | ≥59 ft h1/h3/h7 | low h7 | POD≥59 h7 |
|---|---|---|---|---|
| persistence | .134/.334/.603 | .78/2.37/4.65 | .237 | .42 |
| GR4J hybrid, perfect rain | .085/.180/.258 | .51/1.18/1.03 | .108 | .50 |
| GR4J hybrid, NO future rain | .086/.242/.478 | .50/1.75/4.23 | .174 | .08 |
- The impressive 3–7 day numbers come almost entirely from perfect future rain. With no rain
  knowledge, 7-day flood error (4.2 ft) ≈ persistence; the 7-day forecasts for Irma stay flat at
  ~54.6 ft. Real skill will sit between the brackets → must measure with 2024+ archived forecasts.
- Robust regardless of rain: 1-day forecasts, low water (h7 .17 vs .24), the recession after a peak.
- Irma rise, perfect rain: 7-day-ahead forecast for 09-11 = 60.5 (obs 60.96); 3-day-ahead for 09-13
  dips to 58.1 (obs 62.28) — rise timing still imperfect even with perfect rain.
- **Uncertainty bands are broken in floods:** constant conformal offsets give ±0.1–0.3 ft during
  Irma while actual errors were 1–5 ft (calm days dominate calibration). Need state-dependent
  intervals (condition on flow regime / predicted change / forecast-rain spread) before any
  dashboard use.

## 2026-11-20 — MRMS download complete
- 3065 days 2018-06-30 .. 2026-11-19, 1320 px (44×30 @0.01°), **0 missing days**, 29 min, ~10 GB
  transferred → 91 MB of monthly parquet.
- Basin-mean annual totals 1190–1618 mm (2018 partial). Ian: 09-28 137 mm (87–195 px range),
  09-29 113 mm (37–184); Milton 10-09 140 mm (42–242).
- Split problem: radar starts mid-2018, so under the current split (val 2016–19, test 2020+) only
  ~1.5 radar years are available for selection/training. Split decision pending with user.

## 2026-11-20 — radar (MRMS, 2 km blocks) relevance analysis, train 2018-07..2021-12
- 330 blocks (2 km), 92 touch basin, 65 >50 % inside. In-basin daily corr median .75 (min .51).
  Redundancy clusters (3-day sums, in-basin): 8 @ r≥.9, 2 @ r≥.8, 1 @ r≥.7.
- Greedy partial-corr, target Δlog Q (controls forced; cum R² after controls ≈ .11–.16):
  - h1: pr_past1 @ 28.33,−81.47 (in basin, 7 km) partial r **.61** → cum R² .46
  - h3: pr_next3 @ 28.33,−81.49 (94 % in, 8 km) r **.61** → .44
  - h7: pr_next7 @ 28.41,−81.43 (in basin, 16 km upstream) r **.73** → .61
  - nearer cells for short horizons, further-upstream cell for 7-day: physically sensible.
- Stage target gives the same pattern, weaker (rain partial r .43–.50; stage confounded by Toho).
- Univariate: best cells are in-basin 2–7 km from gauge (Spearman .51 for next-3-day vs Δlog Q);
  in-basin mean .457 vs outside .434 (neighbours still correlated).
- Peak cross-corr lag = 1 day for every cell (basin ~35 km long; daily resolution can't resolve
  travel-time differences).
- After the first 1–2 rain picks, greedy selections scatter (outside cells, sign flips) →
  overfitting with ~1200 rows × ~3000 candidates. Need stability selection.

## 2026-11-20 — radar stability selection (40 block-bootstraps, 120-day blocks, target Δlog Q)
- Per 2 km cell, nothing is stable (only h1 pr_past1 @28.33,−81.47 at 58 %); neighbouring cells
  split the picks (h3: two adjacent cells 42.5 % each).
- **Pooled by redundancy region (r≥0.85 on 3-day sums → 18 regions) it is stable & physical:**
  | region | centre / in-basin / km to gauge | stable picks (pooled freq) |
  |---|---|---|
  | R12 | 28.367,−81.446 / 69 % / 12 km | pr_past1 h1 1.00; pr_next3 h3 .95; pr_past1 h3 .72; pr_next7 h7 1.00 |
  | R13 | 28.272,−81.428 / 22 % / 5 km | pr_next1 h1 .90 |
  | R18 | 28.468,−81.387 / 41 % / 23 km | pr_past3 h3 .85; pr_past1 h7 .62 |
  | R17 | 28.442,−81.471 / 88 % / 20 km | pr_past30 h3 .45, h7 .38 |
  | R3/R11 | west, outside basin | wb_past30/90 .25–.53 (suspect) |
  Lower basin → immediate; mid-basin → days 1–7; upper basin → lagged 1–3 days; upper-basin
  30-day wetness → antecedent control. Clusters at r≥0.9 (38) tell the same story.
- Design: regional (cluster-mean) radar features, keep (kind, region) with pooled freq ≥ 0.5.
  Radar only spans 2018-07+, so use it in a residual corrector on top of the long-record
  (gridMET) hybrid rather than as the sole input.

## 2026-11-20 — validation on NEW split (train ≤2021, val 2022–23 incl. Ian), GR4J recalibrated ≤2021 (NSE .66)
Radar regions persisted to data/radar_regions.json (18 regions, r≥.85, train-only; ids match the
stability analysis: R12 28.367,−81.446; R13 28.272,−81.428; R18 28.468,−81.387).
Reports: out/model_comparison_val_{perfect,nofuture}.md.

| MAE ft | h1/h3/h7 perfect rain | ≥59 h7 | low h7 | h1/h3/h7 NO future rain | ≥59 h7 | low h7 |
|---|---|---|---|---|---|---|
| persistence | .127/.326/.584 | 4.88 | .198 | same | 4.88 | .198 |
| ARX lgbm + gr4j | **.075**/.168/**.243** | 1.52 | .121 | **.074**/.241/.455 | **3.65** | .296 |
| **hybrid + gr4j** | .080/.175/.254 | **1.19** | .093 | .080/**.237**/**.440** | 4.07 | **.183** |
| hybrid gr4j-offset | .080/**.166**/.265 | 2.24 | **.089** | – | – | – |
| hybrid + gr4j + radar corr. | .082/.172/.260 | 1.27 | .160 | .081/.246/.472 | 3.93 | .295 |

Ian (crest 64.15 ft daily mean, 2022-10-01; above training max):
- perfect rain: hybrid+gr4j peak err h1 +0.21, h3 −0.43 (3 d late), **h7 +0.54 (1 d early)**;
  ARX h7 −2.33 (7 d late = no warning); **gr4j-offset overshoots +2.3/+4.0/+5.1 ft** → unconstrained
  physical extrapolation is dangerous; offset variant ruled out.
- no future rain: hybrid h1 −0.25, h3 −1.30, h7 −3.07; no week-ahead warning without rain forecasts.
- **Radar corrector adds no skill** (and hurts low water .093→.160) in either bracket. Relevance analysis
  found real regional signal, but on top of GR4J + basin-scale gridMET at daily resolution it doesn't
  transfer with 3.5 training years. Also: live future rain comes from ~10–25 km NWP grids, which can't
  resolve 2 km regions anyway. Radar's practical role: **near-real-time recent observed rain** (gridMET
  lags 1–2 days) for the live pipeline.
- Coverage in val 0.71–0.80 (only 2 val years → leave-one-year-out calibrates on 1 year; noisy).

## 2026-11-21 — lightweight data screening (screen.py) + issue-time gauge readings
Method: correlate candidates (level, 1/7-day change, 30/365-day anomaly) with *out-of-fold* residuals
of hybrid+gr4j (1979–2023, blocked 3-yr folds; OOF MAE perfect rain .070/.155/.237, no future rain
.073/.200/.375), moving-block bootstrap CI. Only candidates that explain residuals get wired in.
- **Ruled out** (operating, long record; |r| ≤ 0.11 at all horizons): USGS surficial wells (Lake Oliver,
  Palm Lake Dr, Lake Sawyer, Sea World Dr, Disney tree farm, Orlo Vista), Big Sand/Butler/Conway lakes,
  Lake Kissimmee, S-65, Okeechobee, Toho west, NWM retro flow.
- Discontinued but informative (r ≈ 0.2, unusable live): Shingle Ck at Oak Ridge Rd 02263692 + rain gauge
  (2006–10), Lake Adair wells (1984–88). Hint: better in-basin observed rain still has value.
- **USGS instantaneous values (2007+): next-morning (12Z) reading − day-t mean** → r .69/.36/.17 with
  h1/h3/h7 residuals. Implemented (fetch_iv_history, dataset.issue_reading, *_am_d features).
- First val test (train ≤2021, 2007+ rows when using am features; perfect rain):
  hybrid MAE h1 .080 → **.051** (+am), h3 .175 → .165, h7 ≈ same; low-water h3 .054 → .042.
  ARX +am: Ian peak error h1 +0.09, **h3 +0.07** (was −0.81); but hybrid +am **overshoots Ian at h3
  (+3.6 ft)** → am features in the log-Q model extrapolate badly in the record flood; needs a fix
  (e.g. carry the morning rating residual instead of feeding am deltas to the flow model, or blend
  ARX+am for h1–h3 with hybrid for h4–7).

## 2026-11-21 — floodplain rating + per-horizon morning features + blend (val 2022–23)
Rating (train ≤2021): plain CombinedRating over-predicts the highest flows (Ian from *observed* Q:
65.08/65.56/64.80 vs obs 63.82/64.15/63.86) and its lake branch (d≈1.53) explodes above the record
(6000 cfs → 71.6 ft). Floodplain break + bounds (d ≤ 1.2, g ≥ 0.3): Ian 63.94/64.15/63.76, 6k/9k cfs →
65.6/66.9 ft, train MAE .129 → .131. NB: hybrid's earlier Ian h7 "+0.54" was partly cancellation
(rating +1.4 ft high at crest, flow forecast low).
Morning features: only for h ≤ 3 and trained on the full record (LightGBM `allow_missing`, NaN before
2007) — training on 2007+ only made no-rain h7 worse.

| MAE (h1/h3/h7) | perfect rain | ≥59 h7 | low h7 | NO future rain | ≥59 h7 | low h7 | Ian h1/h3/h7 (perfect) |
|---|---|---|---|---|---|---|---|
| persistence | .127/.326/.584 | 4.88 | .198 | same | 4.88 | .198 | – |
| hyb fp (floodplain rating) | .081/.177/.260 | 1.33 | **.091** | .081/.239/.443 | 4.14 | **.179** | +.21/−.41/+.78 |
| hyb2 = hyb fp + am (h≤3) | .052/.158/.260 | 1.33 | **.091** | .052/.227/.443 | 4.14 | **.179** | +.04/+.83/+.78 |
| arx2 = arx lgbm + am (h≤3) | **.040**/.149/.243 | 1.52 | .121 | **.040**/.232/.455 | 3.65 | .296 | .00/−.13/−2.33 |
| **blend .5 arx2/hyb2** | **.040/.146/.228** | **0.87** | .104 | .041/**.224/.432** | 3.62 | .233 | −.15/+.35/−.78 |
- Morning reading halves h1 error in both brackets; blend is best overall and for floods; hybrid alone
  remains best for low water at h7. Next: weights chosen out-of-sample (not on val), real forecast rain.

## 2026-11-21 — real rain forecasts (GEFSv12 ensemble mean, AWS archive 2020-10+)
- Source: s3://noaa-gefs-pds GEFSv12 geavg/gespr 0.5°, APCP 6 h buckets fetched by byte range (~88 KB
  each, ~2.5 MB/run). Issue convention: 00Z run of day t+1 serves issue day t (forecast ~12Z); lead k =
  gridMET day t+k. Bilinear to gridMET cells; GR4J restarted from its day-t state with forecast areal rain.
- Skill at the model cell vs gridMET (2020-10..2022-10): corr lead1 .89, lead3 .59, lead5 .45, lead7 .29;
  3-day totals .83, 7-day .75. GEFS mean under-forecasts amounts (2.5 vs 3.9 mm/d, ratio ~0.64).
  GR4J forecast dH vs perfect-rain dH: corr .996/.92/.69 (h1/h3/h7).
- **Honest val 2022–23 (models fit ≤2021 on observed rain, predicted with issued GEFS forecasts; bands
  calibrated on forecast-mode residuals):**

| MAE h1/h3/h7 | raw GEFS | ≥59 h7 | low h7 | GEFS × per-lead scale | ≥59 h7 | low h7 |
|---|---|---|---|---|---|---|
| persistence | .127/.326/.584 | 4.88 | .198 | same | 4.88 | .198 |
| hyb2 | .052/.186/.398 | 2.66 | **.112** | .053/.186/.363 | 2.21 | .170 |
| arx2 | **.039/.168**/.370 | 2.54 | .130 | .041/.175/.343 | 2.11 | .188 |
| blend arx2/hyb2 | .040/.170/.368 | 2.24 | .119 | .041/.174/**.338** | **1.73** | .176 |
| (perfect rain blend) | .040/.146/.228 | 0.87 | .104 | | | |
| (no-rain blend) | .041/.224/.432 | 3.62 | .233 | | | |
  Ian with scaled GEFS: hyb fp h7 −0.17 ft (3 d late), blend −1.31; POD≥59 h7 .79 (blend), FAR 0.
- Ruled out quickly: (a) blending forecast-driven with no-rain model (oracle weight on forecast model
  0.9–1.0 → models do not over-trust forecasts); (b) quantile mapping GEFS→gridMET (h7 .392, low .218:
  inflates mid-lead forecasts); (c) ECMWF IFS HRES open data (2023+, 73-day sample: lead1–3 corr
  .49/.42/.43 vs GEFS .58/.61/.50; 3-day totals .51 vs .65; average of both ≈ GEFS) — and ECMWF ENS has
  no ensemble-mean tp (51 members × 0.8 MB per step → too heavy).
- Remaining h7 gap (.34–.37 vs .23 perfect) is forecast information loss, not model misuse.

## 2026-11-21 — packaged pipeline check + uncertainty bands (val 2022–23)
- train.train(train_end 2021-12-31, GEFS-era OOF bands) + train.backtest reproduce the scratch numbers:
  MAE with issued GEFS h1..h7 = .039/.109/.166/.230/.263/.300/.339 (perfect rain .039/.100/.144/.193/.211/.220/.223).
- 80 % bands: overall coverage .72–.80, but flood-day (≥59 ft, n=14, ≈Ian) coverage only .43–.71.
  Calibration window (2020-10..2021-12, 450 days) contains no flood.
- Tried (val): band scale + GEFS spread (additive / multiplicative / spread-only) and a wet-scenario p90
  (point forecast with mean+1σ rain): none clearly better on floods (±1 day of 14); wet p90 lifts overall
  coverage at h5–7 to .82–.85 at +0.3–0.9 ft width. Kept the simple |p50−stage|-binned bands. The final
  model's calibration window (2020-10..2023-12) includes Ian.

## 2026-11-21 — FINAL TEST (one shot): 2024-01-01 .. 2026-11-12, 1047 issue days, Milton included
Config frozen before running: spec.build_model (0.5 ARX + 0.5 RatingHybrid, deep LightGBM, floodplain rating,
morning features h≤3), GEFS raw h1–3 / per-lead-scaled h4–7, |p50−stage|-binned bands calibrated OOF by year on
forecast-mode predictions 2020-10..2023-12. Trained on ≤2023-12-31 (GR4J recalibrated ≤2023). Past rain = gridMET
archive (live runs fill the last 1–2 days with MRMS; checked separately). Script: scratchpad/final_test.py.

| MAE ft | h1 | h2 | h3 | h4 | h5 | h6 | h7 |
|---|---|---|---|---|---|---|---|
| persistence | .105 | .189 | .260 | .326 | .385 | .437 | .483 |
| model, no future rain | .036 | .114 | .180 | .248 | .301 | .346 | .388 |
| **model + issued GEFS** | **.034** | **.095** | **.141** | **.222** | **.257** | **.290** | **.320** |
| model + perfect rain (bound) | .033 | .081 | .113 | .142 | .157 | .167 | .174 |
| ≥59 ft (n=6, Milton) GEFS | .11 | .49 | .62 | 1.66 | 1.55 | 1.33 | .94 |
| ≥59 ft persistence | .67 | 1.55 | 2.39 | 3.01 | 3.54 | 4.20 | 4.67 |
| low water (≤52.49) GEFS | .021 | .045 | .064 | .089 | .104 | .118 | .137 |
| low water persistence | .034 | .060 | .078 | .094 | .111 | .130 | .146 |
| POD/FAR ≥59, GEFS | 1/0 | 1/0 | .83/0 | .50/.40 | .50/.25 | .67/0 | .67/0 |
| 80 % band coverage, GEFS | .79 | .79 | .80 | .78 | .79 | .77 | .76 |
| … on flood days (n=6) | .50 | .33 | .50 | .17 | .17 | .33 | .33 |
- vs persistence: −68 % (h1), −46 % (h3), −34 % (h7). Real-forecast skill sits ~45 % of the way from the
  no-rain to the perfect-rain bound at h7.
- Milton (crest 60.56 ft daily mean 2024-10-13): peak error h1 −0.16 (on time), h3 −0.15 (1 d early),
  h4–h7 **over** by +0.9…+1.6 ft (GEFS had big Milton totals; ×1.35–1.45 scale amplified them); perfect
  rain −0.2…−0.7. Flood-day band coverage is poor (6 days) → bands under-represent storm-forecast risk.
- Low water: clearly better than persistence at h1–h3; only marginally at h5–h7 (.104–.137 vs .111–.146).

## 2026-11-21 — live rain path check (MRMS 24 h QPE ending 12Z, mapped to gridMET cells)
- MRMS12 cell means vs gridMET at the model cell (2020-10..2025-02): daily corr .91 (vs .20 for a 1-day shift
  → same 12Z–12Z day), bias scale gridMET/MRMS = 1.013.
- Final-test artifact, 2024-01..2025-02 (417 days), ALL past rain from scaled MRMS instead of gridMET (worst case;
  live runs only fill the last 1–2 days): MAE h1/h3/h7 .033/.123/.301 vs gridMET .033/.126/.303 → no degradation.

## 2026-11-22 — production artifact + first live forecast
- `shingle train` (train_end 2026-11-13): 16 622 training days, bands calibrated OOF on 2 227 GEFS-era forecast days
  (2020-10..2026-11, includes Ian and Milton), GR4J x1 16.2 / x2 −10 / x3 872 / x4 3.32, GEFS scale 1.37–1.43,
  radar scale 0.99. ~3 min.
- `shingle predict` at 2026-11-22 01Z (issue day 11-20; all inputs live-updated in ~2 min): stage rising after
  17+25 mm on 11-17/18 (56.22 ft at 00Z); forecast p50 56.05 → peak 56.24 (11-23) → 55.75 (11-27), all normal
  category; NWM short-range h1 → 55.57 ft logged alongside. Partial IV mean for 11-21 already 56.09 → our h1
  likely ~0.05–0.07 ft low (steady rise under-forecast; S-61 releases ramping 0.8→2.2 kcfs).
