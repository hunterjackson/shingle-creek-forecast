"""Static configuration for the Shingle Creek at Campbell forecast."""

from pathlib import Path

from shingle.sources.gridmet import BBox

SITE = "02264495"  # USGS Shingle Creek at Campbell, FL (NWS SHIF1)
UPSTREAM_SITE = "02263800"  # USGS Shingle Creek at Airport nr Kissimmee (NWS SNCF1)
NWM_REACH = "21476600"
NWS_LID = "SHIF1"
OUTLET = (28.266944, -81.448056)  # (lat, lon) of SHIF1

# basin extent lon -81.521..-81.382, lat 28.238..28.558 plus ~1-2 cell buffer
BBOX = BBox(north=28.62, south=28.18, west=-81.60, east=-81.30)

HISTORY_START = "1979-01-01"  # gridMET period of record starts 1979

FLOOD_STAGES = {"action": 59.0, "minor": 60.0, "moderate": 61.0, "major": 62.0}

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
OUT_DIR = Path("out")

# Evaluation split (user decision 2026-11-20): radar starts mid-2018, so
# train <= 2021, validate 2022-2023 (Ian = validation storm, plus leave-Ian-out
# extrapolation refit), test 2024+ (Milton; real archived lead-time rain forecasts).
TRAIN_END = "2021-12-31"
VAL_END = "2023-12-31"
MRMS_START = "2018-06-30"

# gridMET cell (just NE of the gauge) picked first by partial-correlation selection at every
# horizon (docs/shingle_findings.md, 2026-11-20); the models' rain inputs come from it.
RAIN_CELL = "28.2750_-81.4333"
