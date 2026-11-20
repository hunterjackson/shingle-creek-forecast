"""Grid-cell geometry relative to the drainage basin."""

from __future__ import annotations

import math

import pandas as pd
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

GRIDMET_RES = 1 / 24  # degrees
EARTH_R_KM = 6371.0


def cell_basin_fraction(cells: pd.DataFrame, basin: BaseGeometry, res: float = GRIDMET_RES) -> pd.Series:
    """Fraction of each (lat, lon)-centred square cell of side `res` lying inside the basin."""
    half = res / 2
    fracs = {}
    for cid, row in cells.iterrows():
        cell = box(row.lon - half, row.lat - half, row.lon + half, row.lat + half)
        fracs[cid] = cell.intersection(basin).area / cell.area
    return pd.Series(fracs, name="basin_frac")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


def annotate_cells(
    cells: pd.DataFrame, basin: BaseGeometry, outlet: tuple[float, float], res: float = GRIDMET_RES
) -> pd.DataFrame:
    """Add basin_frac and distance (km) from the outlet gauge, given as (lat, lon)."""
    out = cells[["lat", "lon"]].copy()
    out["basin_frac"] = cell_basin_fraction(cells, basin, res)
    out["dist_outlet_km"] = [haversine_km(r.lat, r.lon, *outlet) for r in cells.itertuples()]
    return out
