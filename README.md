# Shingle Creek forecast

A 7-day water level forecast for **Shingle Creek at Campbell, FL** (USGS 02264495, NWS SHIF1), refreshed every three hours by GitHub Actions.

- **Site:** https://hunterjackson.github.io/shingle-creek-forecast/
- **Raw data** (updated by every run):
  - [`site/data/forecast.json`](https://raw.githubusercontent.com/hunterjackson/shingle-creek-forecast/main/site/data/forecast.json): current reading plus 7 days of expected level, likely range (p10–p90), flood category and forecast rain
  - [`site/data/observed.json`](https://raw.githubusercontent.com/hunterjackson/shingle-creek-forecast/main/site/data/observed.json): 45 days of daily creek and Lake Toho levels, 10 days of hourly creek levels
  - [`site/data/scorecard.json`](https://raw.githubusercontent.com/hunterjackson/shingle-creek-forecast/main/site/data/scorecard.json): verification of past forecasts against what happened (ours vs NOAA's National Water Model, day 1)
  - [`site/data/forecast_log.csv`](https://raw.githubusercontent.com/hunterjackson/shingle-creek-forecast/main/site/data/forecast_log.csv): every forecast issued

This is an experimental, independent forecast, not an official one. For flood warnings, follow the NWS Melbourne office.

## How it runs

| workflow | when | what |
|---|---|---|
| `forecast.yml` | every 3 h (13:30 UTC is the first run with a new issue day) | update inputs → `shingle predict` → `shingle site` → commit `site/data/` → deploy Pages |
| `train.yml` | 3rd of each month | `shingle update` → `shingle train` → replace the `model` and `inputs` release assets |
| `ci.yml` | push / PR | test suite |

State between runs: the trained model is the `shingle.pkl` asset on the `model` release; the input history (~75 MB of parquet caches) is the `inputs.tar.gz` asset on the `inputs` release, with an Actions cache in front of it so each run only fetches new days.

## Local use

```sh
uv sync
uv run shingle fetch     # history caches into data/
uv run shingle train     # models/shingle.pkl (~3 min)
uv run shingle predict   # out/shingle.json + forecast log
uv run shingle site      # site/data/*.json; then serve site/ with any static server
uv run pytest
```

Method, data sources and every experiment (including what was tried and rejected) are in [`docs/shingle_findings.md`](docs/shingle_findings.md); operations in [`docs/shingle_usage.md`](docs/shingle_usage.md).
