import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
import respx

from home_display.shingle.sources import usgs

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

    df = usgs.fetch_dv("02264495", start="2024-10-08", end="2024-10-14")

    params = route.calls.last.request.url.params
    assert params["sites"] == "02264495"
    assert params["parameterCd"] == "00060,00065"
    assert params["startDT"] == "2024-10-08"
    assert params["endDT"] == "2024-10-14"
    assert params["format"] == "json"
    assert len(df) == 7
