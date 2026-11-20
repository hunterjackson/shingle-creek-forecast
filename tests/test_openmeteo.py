import json
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from home_display.shingle.sources import openmeteo

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def payload():
    return json.loads((FIXTURES / "openmeteo_forecast.json").read_text())


@pytest.fixture
def cells():
    return pd.DataFrame(
        {"lat": [28.3, 28.4], "lon": [-81.5, -81.4]},
        index=pd.Index(["a", "b"], name="cell_id"),
    )


def test_parse_returns_dict_of_wide_frames_keyed_by_short_var(payload):
    out = openmeteo.parse(payload, ["a", "b"])

    assert set(out) == {"pr", "pet"}
    pr = out["pr"]
    assert list(pr.columns) == ["a", "b"]
    assert pr.index.name == "date"
    assert pr.index[0] == pd.Timestamp("2026-11-20")
    assert len(pr) == 16
    assert pr.loc["2026-11-20", "a"] == pytest.approx(15.5)
    assert out["pet"].loc["2026-11-20", "a"] == pytest.approx(0.78)


def test_parse_accepts_single_location_dict(payload):
    out = openmeteo.parse(payload[0], ["a"])

    assert list(out["pr"].columns) == ["a"]


def test_parse_rejects_mismatched_cell_count(payload):
    with pytest.raises(ValueError):
        openmeteo.parse(payload, ["a"])


def test_parse_null_becomes_nan(payload):
    payload[0]["daily"]["precipitation_sum"][3] = None

    out = openmeteo.parse(payload, ["a", "b"])

    assert pd.isna(out["pr"].iloc[3, 0])


@respx.mock
def test_fetch_forecast_sends_all_cells_in_one_request(payload, cells):
    route = respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=payload)
    )

    out = openmeteo.fetch_forecast(cells)

    params = route.calls.last.request.url.params
    assert params["latitude"] == "28.3,28.4"
    assert params["longitude"] == "-81.5,-81.4"
    assert params["daily"] == "precipitation_sum,et0_fao_evapotranspiration"
    assert params["forecast_days"] == "16"
    assert params["timezone"] == "America/New_York"
    assert list(out["pr"].columns) == ["a", "b"]


@respx.mock
def test_fetch_historical_forecast_uses_dates_and_other_host(payload, cells):
    route = respx.get("https://historical-forecast-api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=payload)
    )

    openmeteo.fetch_historical_forecast(cells, start="2022-09-20", end="2022-10-05")

    params = route.calls.last.request.url.params
    assert params["start_date"] == "2022-09-20"
    assert params["end_date"] == "2022-10-05"
    assert "forecast_days" not in params
