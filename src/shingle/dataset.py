"""Assemble the model input dict (see features.py for the keys) from fetched history."""

from __future__ import annotations

import pandas as pd

ISSUE_HOUR_UTC = 12  # forecasts are issued ~07 EST: gridMET's day and the MRMS 24 h QPE both end at 12Z


def issue_reading(hourly: pd.Series, index: pd.DatetimeIndex, hour_utc: int = ISSUE_HOUR_UTC, window_hours: int = 3) -> pd.Series:
    """Latest hourly value in (hour_utc - window_hours, hour_utc] on day t+1 (UTC), labelled t.

    A forecast for issue day t is made the next morning, so the morning
    reading is known; anything older than the window is treated as missing.
    """
    s = hourly.dropna()
    s = s[(s.index.hour > hour_utc - window_hours) & (s.index.hour <= hour_utc)]
    last = s.groupby(s.index.tz_localize(None).normalize()).last()
    last.index = last.index - pd.Timedelta(days=1)
    return last.reindex(index)


def hour_reading(hourly: pd.Series, index: pd.DatetimeIndex, hour_utc: int) -> pd.Series:
    """The hourly value labelled hour_utc on day t+1 (UTC), labelled t; NaN when that hour is missing."""
    s = hourly.dropna()
    s = s[s.index.hour == hour_utc]
    v = pd.Series(s.to_numpy(), index=s.index.tz_localize(None).normalize() - pd.Timedelta(days=1))
    return v[~v.index.duplicated(keep="last")].reindex(index)


def morning_mean(hourly: pd.Series, index: pd.DatetimeIndex, first_hour_utc: int = 5, hour_utc: int = ISSUE_HOUR_UTC) -> pd.Series:
    """Mean of the hourly values labelled first_hour_utc..hour_utc on day t+1 (UTC), labelled t.

    05Z is midnight EST, when the gauge's daily mean for t+1 starts: this is the part of the day-1 target
    already observed at issue (05Z..12Z = midnight to 8 am EST).
    """
    s = hourly.dropna()
    s = s[(s.index.hour >= first_hour_utc) & (s.index.hour <= hour_utc)]
    m = s.groupby(s.index.tz_localize(None).normalize()).mean()
    m.index = m.index - pd.Timedelta(days=1)
    return m.reindex(index)


def build_data(history: dict, rain: pd.DataFrame | None = None, pet: pd.DataFrame | None = None) -> dict:
    """Align everything on the Campbell gauge's daily index.

    rain/pet override the gridMET per-cell frames (e.g. MRMS radar cells).
    PET stays per gridMET cell; when rain cells differ from PET cells,
    water-balance features should use gridMET rain (pr_gm) explicitly.
    """
    gauge = history["gauge"]
    idx = gauge.index
    lakes = history["lakes"].reindex(idx)
    d = {
        "stage": gauge["stage_ft"],
        "q": gauge["discharge_cfs"],
        "q_up": history["upstream"]["discharge_cfs"].reindex(idx),
        "toho": lakes["toho_stage"],
        "s61_q": lakes["s61_q"],
        "s59_q": lakes["s59_q"],
        "etoho": lakes["etoho_stage"],
        "pr": (rain if rain is not None else history["pr"]).reindex(idx),
        "pet": (pet if pet is not None else history["pet"]).reindex(idx),
        "pr_gm": history["pr"].reindex(idx),
    }
    iv = history.get("iv") or {}
    if iv.get("gauge") is not None:
        d["stage_am"] = issue_reading(iv["gauge"]["stage_ft"], idx)
        d["q_am"] = issue_reading(iv["gauge"]["discharge_cfs"], idx)
        d["stage_sofar"] = morning_mean(iv["gauge"]["stage_ft"], idx)
        d["stage_h06"] = hour_reading(iv["gauge"]["stage_ft"], idx, 6)  # rate of rise into the issue reading
        d["stage_h00"] = hour_reading(iv["gauge"]["stage_ft"], idx, 0)
    if iv.get("upstream") is not None:
        d["q_up_am"] = issue_reading(iv["upstream"]["discharge_cfs"], idx)
        d["q_up_h06"] = hour_reading(iv["upstream"]["discharge_cfs"], idx, 6)
        d["q_up_h00"] = hour_reading(iv["upstream"]["discharge_cfs"], idx, 0)
    return d
