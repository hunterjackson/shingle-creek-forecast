"""USGS NLDI client: upstream drainage basin polygon for a gauge."""

from __future__ import annotations

import httpx
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

NLDI = "https://api.water.usgs.gov/nldi/linked-data/nwissite"


def parse_basin(geojson: dict) -> BaseGeometry:
    geoms = [shape(f["geometry"]) for f in geojson["features"]]
    return unary_union(geoms)


def fetch_basin(site: str, client: httpx.Client | None = None) -> BaseGeometry:
    client = client or httpx.Client(timeout=60)
    resp = client.get(f"{NLDI}/USGS-{site}/basin", params={"simplified": "true"})
    resp.raise_for_status()
    return parse_basin(resp.json())
