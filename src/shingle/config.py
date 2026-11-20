"""Static configuration for the Shingle Creek at Campbell forecast."""

from pathlib import Path

from home_display.shingle.sources.gridmet import BBox

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
