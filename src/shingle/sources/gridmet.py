"""gridMET (~4 km daily, 1979-present) client via the THREDDS NetCDF Subset Service.

Open-Meteo's ERA5-Land archive returns null precipitation for this region, so
gridMET is the historical rain/ET source. Each grid cell is kept as its own
column (no basin averaging) so that cell relevance can be analysed.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import pandas as pd
import xarray as xr

NCSS = "http://thredds.northwestknowledge.net:8080/thredds/ncss"

# short name -> (dataset file, variable name)
VARS = {
    "pr": ("agg_met_pr_1979_CurrentYear_CONUS.nc", "precipitation_amount"),
    "pet": ("agg_met_pet_1979_CurrentYear_CONUS.nc", "daily_mean_reference_evapotranspiration_grass"),
}


@dataclass(frozen=True)
class BBox:
    north: float
    south: float
    west: float
    east: float


def cell_id(lat: float, lon: float) -> str:
    return f"{lat:.4f}_{lon:.4f}"


def parse_nc(content: bytes, var: str) -> pd.DataFrame:
    """Decode an NCSS netCDF subset into a wide frame: index=date, columns=cell ids."""
    _, varname = VARS[var]
    with _open(content) as ds:
        da = ds[varname].load()
    stacked = da.stack(cell=("lat", "lon")).transpose("day", "cell")
    cols = [cell_id(lat, lon) for lat, lon in stacked["cell"].values]
    df = pd.DataFrame(stacked.values, index=pd.DatetimeIndex(stacked["day"].values).normalize(), columns=cols)
    df.index.name = "date"
    return df.astype(float)


def _open(content: bytes) -> xr.Dataset:
    # netCDF4 can open from memory; scipy's reader chokes on NCSS attributes.
    import netCDF4

    nc = netCDF4.Dataset("inmemory.nc", memory=content)
    return xr.open_dataset(xr.backends.NetCDF4DataStore(nc))


def cells_from_columns(columns) -> pd.DataFrame:
    rows = [tuple(float(x) for x in c.split("_")) for c in columns]
    cells = pd.DataFrame(rows, index=pd.Index(list(columns), name="cell_id"), columns=["lat", "lon"])
    return cells


def year_chunks(start: str, end: str) -> list[tuple[str, str]]:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    chunks = []
    for year in range(s.year, e.year + 1):
        cs = max(s, pd.Timestamp(year=year, month=1, day=1))
        ce = min(e, pd.Timestamp(year=year, month=12, day=31))
        chunks.append((cs.strftime("%Y-%m-%d"), ce.strftime("%Y-%m-%d")))
    return chunks


def fetch(var: str, bbox: BBox, start: str, end: str, client: httpx.Client | None = None) -> pd.DataFrame:
    dataset, varname = VARS[var]
    client = client or httpx.Client(timeout=180)
    frames = []
    for cs, ce in year_chunks(start, end):
        params = {
            "var": varname,
            "north": bbox.north,
            "south": bbox.south,
            "west": bbox.west,
            "east": bbox.east,
            "time_start": f"{cs}T00:00:00Z",
            "time_end": f"{ce}T00:00:00Z",
            "accept": "netcdf",
        }
        resp = client.get(f"{NCSS}/{dataset}", params=params)
        resp.raise_for_status()
        frames.append(parse_nc(resp.content, var))
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="last")].sort_index()
