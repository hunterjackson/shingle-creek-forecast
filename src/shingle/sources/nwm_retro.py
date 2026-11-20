"""NWM v3.0 retrospective (1979-02 .. 2023-02, hourly) reader for a single reach.

The public zarr store is chunked (672 h x 30000 features), ~4-9 MB per chunk,
so reading one reach costs ~6 MB per 4 weeks of record (~700 MB for 2015-2023).
"""

from __future__ import annotations

import pandas as pd
import xarray as xr

RETRO_STORE = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"
M3S_TO_CFS = 35.314666721488590
LOCAL_TZ = "America/New_York"


def _open(store: str) -> xr.Dataset:
    opts = {"anon": True} if store.startswith("s3://") else None
    return xr.open_zarr(store, storage_options=opts, consolidated=True)


def read_reach_daily(
    feature_id: int, start: str, end: str, store: str = RETRO_STORE, min_hours: int = 1
) -> pd.Series:
    """Daily (America/New_York calendar day) mean streamflow in cfs for one reach."""
    ds = _open(store)
    if feature_id not in ds.indexes["feature_id"]:
        raise KeyError(f"feature_id {feature_id} not in store")
    # pad a day each side in UTC so local-day boundaries are covered
    t0 = pd.Timestamp(start) - pd.Timedelta(days=1)
    t1 = pd.Timestamp(end) + pd.Timedelta(days=2)
    da = ds["streamflow"].sel(feature_id=feature_id).sel(time=slice(t0, t1)).load()

    hourly = pd.Series(da.values * M3S_TO_CFS, index=pd.DatetimeIndex(da["time"].values).tz_localize("UTC"))
    local_day = hourly.index.tz_convert(LOCAL_TZ).tz_localize(None).normalize()
    grouped = hourly.groupby(local_day)
    daily = grouped.mean()[grouped.count() >= min_hours]
    daily = daily.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    daily.index.name = "date"
    daily.name = "nwm_retro_cfs"
    return daily
