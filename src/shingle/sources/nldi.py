"""USGS NLDI client: upstream drainage basin polygon for a gauge."""

from __future__ import annotations

import httpx
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from shingle.sources import http

NLDI = "https://api.water.usgs.gov/nldi/linked-data/nwissite"


def parse_basin(geojson: dict) -> BaseGeometry:
    geoms = [shape(f["geometry"]) for f in geojson["features"]]
    return unary_union(geoms)


async def fetch_basin_geojson(site: str, client: httpx.AsyncClient | None = None) -> dict:
    async with http.session(client, timeout=60) as c:
        resp = await http.get(c, f"{NLDI}/USGS-{site}/basin", params={"simplified": "true"})
    return resp.json()


async def fetch_basin(site: str, client: httpx.AsyncClient | None = None) -> BaseGeometry:
    return parse_basin(await fetch_basin_geojson(site, client))
