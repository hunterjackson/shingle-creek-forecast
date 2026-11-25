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

from shingle.sources import http

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


UNPUBLISHED_DAYS = 7  # gridMET lags 1-2 days; a 400 for a window starting this recently means "not published yet"


async def fetch(var: str, bbox: BBox, start: str, end: str, client: httpx.AsyncClient | None = None, workers: int = 4,
                today: str | None = None) -> pd.DataFrame:
    """One request per calendar year, `workers` at a time.

    THREDDS answers 400 when time_start is past the dataset's last day. For a window starting within
    UNPUBLISHED_DAYS of `today` that just means nothing is published yet, so the result is empty; older 400s raise.
    """
    dataset, varname = VARS[var]
    recent = pd.Timestamp(today or pd.Timestamp.today().normalize()) - pd.Timedelta(days=UNPUBLISHED_DAYS)

    async def one(c: httpx.AsyncClient, cs: str, ce: str) -> pd.DataFrame:
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
        try:
            resp = await http.get(c, f"{NCSS}/{dataset}", params=params)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400 and pd.Timestamp(cs) >= recent:
                return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
            raise
        return parse_nc(resp.content, var)

    async with http.session(client, timeout=180) as c:
        frames = await http.gather((one(c, cs, ce) for cs, ce in year_chunks(start, end)), limit=workers)
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="last")].sort_index()
