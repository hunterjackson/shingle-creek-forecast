import asyncio
import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
import respx

from shingle.sources import usgs

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def dv_payload():
    return json.loads((FIXTURES / "usgs_dv.json").read_text())


def test_parse_dv_returns_daily_frame_with_named_columns(dv_payload):
    df = usgs.parse_dv(dv_payload)

    assert list(df.columns) == ["discharge_cfs", "stage_ft"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "date"
    assert df.index[0] == pd.Timestamp("2024-10-08")
    assert len(df) == 7
    assert df.loc["2024-10-13", "stage_ft"] == pytest.approx(60.56)
    assert df.loc["2024-10-13", "discharge_cfs"] == pytest.approx(1770.0)


def test_parse_dv_converts_no_data_sentinel_to_nan(dv_payload):
    stage_ts = next(
        ts for ts in dv_payload["value"]["timeSeries"]
        if ts["variable"]["variableCode"][0]["value"] == "00065"
    )
    stage_ts["values"][0]["value"][2]["value"] = "-999999"

    df = usgs.parse_dv(dv_payload)

    assert np.isnan(df.loc["2024-10-10", "stage_ft"])


def test_parse_dv_handles_site_with_only_stage(dv_payload):
    dv_payload["value"]["timeSeries"] = [
        ts for ts in dv_payload["value"]["timeSeries"]
        if ts["variable"]["variableCode"][0]["value"] == "00065"
    ]

    df = usgs.parse_dv(dv_payload)

    assert list(df.columns) == ["discharge_cfs", "stage_ft"]
    assert df["discharge_cfs"].isna().all()


@respx.mock
def test_fetch_dv_requests_expected_url_and_parses(dv_payload):
    route = respx.get("https://waterservices.usgs.gov/nwis/dv/").mock(
        return_value=httpx.Response(200, json=dv_payload)
    )

    df = asyncio.run(usgs.fetch_dv("02264495", start="2024-10-08", end="2024-10-14"))

    params = route.calls.last.request.url.params
    assert params["sites"] == "02264495"
    assert params["parameterCd"] == "00060,00065"
    assert params["startDT"] == "2024-10-08"
    assert params["endDT"] == "2024-10-14"
    assert params["format"] == "json"
    assert len(df) == 7


@respx.mock
def test_fetch_dv_retries_a_reset_connection(dv_payload):
    # the 2026-11-25 07:37Z forecast run failed on exactly this
    route = respx.get("https://waterservices.usgs.gov/nwis/dv/").mock(
        side_effect=[httpx.ConnectError("[Errno 104] Connection reset by peer"), httpx.Response(200, json=dv_payload)]
    )

    df = asyncio.run(usgs.fetch_dv("02264495", start="2024-10-08", end="2024-10-14"))

    assert route.call_count == 2
    assert len(df) == 7


@pytest.fixture
def iv_payload():
    return json.loads((FIXTURES / "usgs_iv.json").read_text())


def test_parse_iv_returns_hourly_means_on_utc_index(iv_payload):
    df = usgs.parse_iv(iv_payload)

    assert list(df.columns) == ["discharge_cfs", "stage_ft"]
    assert str(df.index.tz) == "UTC"
    assert df.index.name == "time"
    # 19:00-19:45 EDT = 23Z on 2024-10-09
    assert df.loc[pd.Timestamp("2024-10-09 23:00", tz="UTC"), "stage_ft"] == pytest.approx((55.83 + 55.87 + 55.89 + 55.92) / 4)
    assert df.loc[pd.Timestamp("2024-10-10 02:00", tz="UTC"), "stage_ft"] == pytest.approx(56.32)
    assert df.loc[pd.Timestamp("2024-10-10 01:00", tz="UTC"), "discharge_cfs"] == pytest.approx(512)
    assert np.isnan(df.loc[pd.Timestamp("2024-10-10 00:00", tz="UTC"), "discharge_cfs"])


def test_parse_iv_drops_no_data_values(iv_payload):
    stage_ts = next(ts for ts in iv_payload["value"]["timeSeries"] if ts["variable"]["variableCode"][0]["value"] == "00065")
    for v in stage_ts["values"][0]["value"][-1:]:
        v["value"] = "-999999"

    df = usgs.parse_iv(iv_payload)

    assert np.isnan(df.loc[pd.Timestamp("2024-10-10 02:00", tz="UTC"), "stage_ft"])


@respx.mock
def test_fetch_iv_requests_utc_window(iv_payload):
    route = respx.get("https://nwis.waterservices.usgs.gov/nwis/iv/").mock(return_value=httpx.Response(200, json=iv_payload))

    df = asyncio.run(usgs.fetch_iv("02264495", start="2024-10-09T23:00Z", end="2024-10-10T03:00Z"))

    params = route.calls.last.request.url.params
    assert params["sites"] == "02264495" and params["parameterCd"] == "00060,00065"
    assert params["startDT"] == "2024-10-09T23:00Z" and params["endDT"] == "2024-10-10T03:00Z"
    assert len(df) == 4
