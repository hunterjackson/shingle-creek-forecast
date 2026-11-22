"""Helpers that make live (morning-of) inputs look like the training inputs.

At forecast time (~13Z on day t+1): USGS daily means for day t may not be
published yet (compute them from 15-min readings), gridMET lags 1-2 days
(fill with bias-scaled MRMS 12Z-ending 24 h QPE), gridMET PET lags too
(fill with day-of-year climatology), lake/structure data can lag (carry the
last value forward a few days).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LST_OFFSET = pd.Timedelta(hours=5)  # USGS daily values use local standard time (EST = UTC-5)


def daily_from_iv(hourly: pd.DataFrame, min_hours: int = 20) -> pd.DataFrame:
    """Hourly UTC readings -> local-standard-time daily means (NaN if < min_hours valid hours)."""
    local = hourly.copy()
    local.index = (local.index.tz_convert("UTC").tz_localize(None) - LST_OFFSET)
    day = local.index.normalize()
    mean = local.groupby(day).mean()
    count = local.groupby(day).count()
    out = mean.where(count >= min_hours)
    out.index.name = "date"
    return out


def fill_rain(gridmet: pd.DataFrame, radar: pd.DataFrame, scale: float) -> pd.DataFrame:
    """Rows after gridMET's last complete day are taken from radar (same cells) times `scale`."""
    valid = gridmet.notna().all(axis=1)
    last = valid[valid].index.max()
    out = gridmet.copy()
    after = out.index > last
    r = radar.reindex(index=out.index, columns=out.columns) * scale
    out.loc[after] = r.loc[after]
    out.attrs["filled_from"] = out.index[after].min() if after.any() else None
    return out


def pet_climatology(pet: pd.DataFrame) -> pd.DataFrame:
    """Mean PET per day of year (1..366), lightly smoothed (circular 15-day window)."""
    clim = pet.groupby(pet.index.day_of_year).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
    wrapped = pd.concat([clim.iloc[-7:], clim, clim.iloc[:7]])
    return wrapped.rolling(15, center=True).mean().iloc[7:-7]


def fill_pet(pet: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    out = pet.copy()
    missing = out.isna()
    if missing.any().any():
        c = clim.reindex(out.index.day_of_year)
        c.index = out.index
        out = out.where(~missing, c[out.columns])
    return out


def ffill_recent(s: pd.Series, limit: int = 3) -> pd.Series:
    """Carry the last observation forward at most `limit` days (lake levels change slowly)."""
    return s.ffill(limit=limit)
