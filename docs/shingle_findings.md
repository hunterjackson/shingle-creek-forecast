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
