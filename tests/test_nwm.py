import asyncio
import json
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from shingle.sources import nwm

FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / f"nwm_{name}.json").read_text())


def test_parse_short_range_long_frame():
    df = nwm.parse_streamflow(load("short_range"))

    assert list(df.columns) == nwm.COLUMNS
    assert set(df["series"]) == {"short_range"}
    assert set(df["member"]) == {"det"}
    assert len(df) == 18
    assert df["reference_time"].dt.tz is not None
    assert df["valid_time"].is_monotonic_increasing
    assert (df["flow_cfs"] > 0).all()


def test_parse_long_range_keeps_ensemble_members():
    df = nwm.parse_streamflow(load("long_range"))

    assert set(df["member"]) == {"mean", "member1", "member2", "member3", "member4"}
    assert (df.groupby("member").size() == 120).all()


def test_parse_empty_series_returns_empty_frame_with_columns():
    df = nwm.parse_streamflow(load("medium_range"))

    assert df.empty
    assert list(df.columns) == nwm.COLUMNS


def test_daily_mean_groups_by_local_calendar_day():
    df = pd.DataFrame(
        {
            "series": "short_range",
            "member": "det",
            "reference_time": pd.Timestamp("2026-11-20T00:00Z"),
            # 04:00Z is still 11-19 in America/New_York (EST, UTC-5)
            "valid_time": pd.to_datetime(["2026-11-20T04:00Z", "2026-11-20T06:00Z", "2026-11-20T12:00Z"]),
            "flow_cfs": [100.0, 200.0, 400.0],
        }
    )

    daily = nwm.daily_mean(df)

    assert daily.index.name == "date"
    assert daily.loc[pd.Timestamp("2026-11-19")] == pytest.approx(100.0)
    assert daily.loc[pd.Timestamp("2026-11-20")] == pytest.approx(300.0)


def test_daily_mean_requires_single_member():
    df = nwm.parse_streamflow(load("long_range"))
    with pytest.raises(ValueError):
        nwm.daily_mean(df)


@respx.mock
def test_fetch_streamflow_requests_series():
    route = respx.get("https://api.water.noaa.gov/nwps/v1/reaches/21476600/streamflow").mock(
        return_value=httpx.Response(200, json=load("short_range"))
    )

    df = asyncio.run(nwm.fetch_streamflow("21476600", "short_range"))

    assert route.calls.last.request.url.params["series"] == "short_range"
    assert len(df) == 18
