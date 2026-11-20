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
