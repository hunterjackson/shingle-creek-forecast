import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
import respx

from shingle.sources import dbhydro

FIXTURES = Path(__file__).parent / "fixtures"
URL = "https://insightsdata.api.sfwmd.gov/v1/insights-data/cont/data"


@pytest.fixture
def payload():
    return json.loads((FIXTURES / "dbhydro.json").read_text())


def test_parse_returns_wide_daily_frame_keyed_by_dbkey(payload):
    df = dbhydro.parse(payload)

    assert list(df.columns) == ["68918", "91625"]
    assert df.index.name == "date"
    assert list(df.index) == list(pd.date_range("2024-10-01", "2024-10-05"))
    assert df.loc["2024-10-01", "68918"] == pytest.approx(53.891)
    assert df.loc["2024-10-05", "91625"] == pytest.approx(1687.879)


def test_parse_null_and_missing_code_become_nan(payload):
    payload["timeseries"][0]["value"] = None
    payload["timeseries"][1]["code"] = "M"

    df = dbhydro.parse(payload)

    assert np.isnan(df.loc["2024-10-01", "68918"])
    assert np.isnan(df.loc["2024-10-02", "68918"])


def test_parse_rejects_non_ngvd29_stage(payload):
    payload["timeseries"][0]["units"] = "ft NAVD88"

    with pytest.raises(ValueError, match="NGVD29"):
        dbhydro.parse(payload)


def test_parse_empty_payload():
    df = dbhydro.parse({"timeseries": []})

    assert df.empty
    assert df.index.name == "date"


@respx.mock
def test_fetch_requests_ngvd29_and_dbkeys(payload):
    route = respx.get(URL).mock(return_value=httpx.Response(200, json=payload))

    df = dbhydro.fetch(["68918", "91625"], start="2024-10-01", end="2024-10-05")

    p = route.calls.last.request.url.params
    assert p["timeseriesIds"] == "68918,91625"
    assert p["datum"] == "NGVD29"
    assert p["startDate"] == "20241001"
    assert p["endDate"] == "20241005"
    assert p["format"] == "json"
    assert df.shape == (5, 2)


def test_splice_prefers_primary_and_fills_from_secondary():
    idx = pd.date_range("2026-05-08", periods=5, name="date")
    df = pd.DataFrame(
        {"68918": [1.0, 2.0, 3.0, np.nan, np.nan], "WN255": [9.0, 9.0, 3.1, 4.0, 5.0]}, index=idx
    )

    s = dbhydro.splice(df, ["68918", "WN255"])

    assert s.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_series_config_maps_names_to_dbkeys():
    assert dbhydro.SERIES["toho_stage"] == ["68918", "WN255"]
    assert dbhydro.SERIES["s61_q"] == ["91625"]
    keys = dbhydro.all_dbkeys()
    assert "06832" in keys and len(keys) == len(set(keys))


def test_named_frame_splices_each_series():
    idx = pd.date_range("2024-01-01", periods=2, name="date")
    raw = pd.DataFrame({k: 1.0 for k in dbhydro.all_dbkeys()}, index=idx)
    raw["68918"] = [np.nan, 5.0]
    raw["WN255"] = [4.0, 6.0]

    named = dbhydro.named(raw)

    assert list(named.columns) == list(dbhydro.SERIES)
    assert named["toho_stage"].tolist() == [4.0, 5.0]


@respx.mock
def test_fetch_chunks_by_key_group_and_years(payload):
    route = respx.get(URL).mock(return_value=httpx.Response(200, json=payload))

    df = dbhydro.fetch(["68918", "91625", "LQ978"], start="2000-01-01", end="2024-12-31", max_keys=2, chunk_years=10)

    # 2 key groups x 3 date chunks (2000-09, 2010-19, 2020-24)
    assert route.call_count == 6
    starts = sorted({c.request.url.params["startDate"] for c in route.calls})
    assert starts == ["20000101", "20100101", "20200101"]
    assert df.index.is_unique


@respx.mock
def test_fetch_retries_transient_connection_errors(payload):
    route = respx.get(URL).mock(
        side_effect=[httpx.ConnectError("reset"), httpx.Response(200, json=payload)]
    )

    df = dbhydro.fetch(["68918"], start="2024-10-01", end="2024-10-05")

    assert route.call_count == 2
    assert not df.empty
