import json
from pathlib import Path

import httpx
import numpy as np
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


@pytest.fixture
def prev_payload():
    return json.loads((FIXTURES / "openmeteo_previous_runs.json").read_text())


def test_parse_previous_runs_daily_sums_per_lead(prev_payload):
    out = openmeteo.parse_previous_runs(prev_payload, ["a", "b"], leads=(0, 1, 2))

    assert set(out) == {0, 1, 2}
    d1 = out[1]
    assert list(d1.columns) == ["a", "b"]
    assert list(d1.index) == list(pd.date_range("2024-10-08", "2024-10-10", name="date"))
    hourly = prev_payload[0]["hourly"]["precipitation_previous_day1"][:24]
    assert d1.loc["2024-10-08", "a"] == pytest.approx(sum(v or 0 for v in hourly))


def test_parse_previous_runs_day_with_all_null_is_nan(prev_payload):
    prev_payload[0]["hourly"]["precipitation_previous_day2"][:24] = [None] * 24

    out = openmeteo.parse_previous_runs(prev_payload, ["a", "b"], leads=(2,))

    assert np.isnan(out[2].loc["2024-10-08", "a"])


@respx.mock
def test_fetch_previous_runs_requests_lead_variables(prev_payload, cells):
    route = respx.get("https://previous-runs-api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=prev_payload)
    )

    openmeteo.fetch_previous_runs(cells, "2024-10-08", "2024-10-10", leads=(0, 1, 2))

    p = route.calls.last.request.url.params
    assert p["hourly"] == "precipitation,precipitation_previous_day1,precipitation_previous_day2"
    assert p["start_date"] == "2024-10-08"
