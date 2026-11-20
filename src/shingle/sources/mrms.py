"""NOAA MRMS gauge-corrected radar QPE (~1 km), daily, from the Iowa Environmental Mesonet archive.

For local-standard-time day D we use the 24 h accumulation stamped D+1 05:00Z
(= midnight EST), matching the USGS daily-value day. Products:
  GaugeCorr_QPE_24H          2018-06-30 .. 2020-10-14
  MultiSensor_QPE_24H_Pass2  2020-10-13 .. present (preferred when present)
Each daily file is a ~3-5 MB CONUS grid; only the basin bbox is kept.
"""

from __future__ import annotations

import gzip
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pandas as pd
import xarray as xr

from home_display.shingle.sources.gridmet import BBox

ARCHIVE = "https://mtarchive.geol.iastate.edu"
START = pd.Timestamp("2018-06-30")
MULTISENSOR_START = pd.Timestamp("2020-10-12")
GAUGECORR_END = pd.Timestamp("2020-10-14")


def _url(product: str, stamp: pd.Timestamp) -> str:
    return (
        f"{ARCHIVE}/{stamp:%Y/%m/%d}/mrms/ncep/{product}/"
        f"{product}_00.00_{stamp:%Y%m%d}-050000.grib2.gz"
    )


def candidate_urls(day: pd.Timestamp) -> list[str]:
    day = pd.Timestamp(day).normalize()
    if day < START:
        raise ValueError(f"MRMS gauge-corrected daily QPE starts {START.date()}")
    stamp = day + pd.Timedelta(days=1)
    urls = []
    if day >= MULTISENSOR_START:
        urls.append(_url("MultiSensor_QPE_24H_Pass2", stamp))
    if day <= GAUGECORR_END or day >= MULTISENSOR_START:
        urls.append(_url("GaugeCorr_QPE_24H", stamp))
    return urls


def decode_grib(raw: bytes) -> xr.DataArray:
    """Decode a (decompressed) single-message MRMS GRIB2 into a lat/lon DataArray."""
    fd, path = tempfile.mkstemp(suffix=".grib2")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        with xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
            return ds[list(ds.data_vars)[0]].load()
    finally:
        os.unlink(path)


def crop(da: xr.DataArray, bbox: BBox) -> pd.Series:
    lon_w, lon_e = bbox.west % 360, bbox.east % 360
    sub = da.sel(latitude=slice(bbox.north, bbox.south), longitude=slice(lon_w, lon_e))
    vals = sub.values.astype(float)
    vals[vals < 0] = np.nan  # -3 no coverage, -1 missing
    lats = np.round(sub["latitude"].values, 3)
    lons = np.round(sub["longitude"].values - 360.0, 3)
    ids = [f"{la:.3f}_{lo:.3f}" for la in lats for lo in lons]
    return pd.Series(vals.ravel(), index=pd.Index(ids, name="cell_id"))


def fetch_day(day: pd.Timestamp, bbox: BBox, client: httpx.Client | None = None) -> pd.Series | None:
    client = client or httpx.Client(timeout=120)
    for url in candidate_urls(day):
        resp = client.get(url)
        if resp.status_code == 404:
            continue
        resp.raise_for_status()
        s = crop(decode_grib(gzip.decompress(resp.content)), bbox)
        s.name = pd.Timestamp(day).normalize()
        return s
    return None


def fetch(start: str, end: str, bbox: BBox, workers: int = 8) -> pd.DataFrame:
    days = pd.date_range(start, end, name="date")
    client = httpx.Client(timeout=120, limits=httpx.Limits(max_connections=workers))

    def one(day):
        for attempt in range(3):
            try:
                return fetch_day(day, bbox, client=client)
            except (httpx.TransportError, httpx.HTTPStatusError):
                if attempt == 2:
                    raise
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, days))
    rows = {d: s for d, s in zip(days, results) if s is not None}
    df = pd.DataFrame.from_dict(rows, orient="index") if rows else pd.DataFrame()
    df = df.reindex(days)
    df.index.name = "date"
    return df
