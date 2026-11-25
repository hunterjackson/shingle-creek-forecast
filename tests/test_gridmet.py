import asyncio
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from shingle.sources import gridmet

FIXTURES = Path(__file__).parent / "fixtures"
BBOX = gridmet.BBox(north=28.62, south=28.18, west=-81.60, east=-81.30)


def test_parse_nc_returns_wide_daily_frame_one_column_per_cell():
    df = gridmet.parse_nc((FIXTURES / "gridmet_pr.nc").read_bytes(), "pr")

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "date"
    assert list(df.index) == list(pd.date_range("2024-10-07", "2024-10-11"))
    assert df.shape == (5, 11 * 8)
    # NW corner cell gets Milton's 283.7 mm on 2024-10-09 (scale_factor decoded)
    assert df.loc["2024-10-09", "28.6083_-81.6000"] == pytest.approx(283.7)


def test_parse_nc_pet_variable():
    df = gridmet.parse_nc((FIXTURES / "gridmet_pet.nc").read_bytes(), "pet")

    assert df.loc["2024-10-08", "28.6083_-81.6000"] == pytest.approx(4.0)


def test_cells_table_lists_lat_lon_for_each_column():
    df = gridmet.parse_nc((FIXTURES / "gridmet_pr.nc").read_bytes(), "pr")

    cells = gridmet.cells_from_columns(df.columns)

    assert list(cells.columns) == ["lat", "lon"]
    assert cells.index.name == "cell_id"
    assert cells.loc["28.6083_-81.6000", "lat"] == pytest.approx(28.6083)
    assert cells.loc["28.6083_-81.6000", "lon"] == pytest.approx(-81.6)


def test_year_chunks_split_range_by_calendar_year():
    chunks = gridmet.year_chunks("1979-06-01", "1981-02-03")

    assert chunks == [
        ("1979-06-01", "1979-12-31"),
        ("1980-01-01", "1980-12-31"),
        ("1981-01-01", "1981-02-03"),
    ]


@respx.mock
def test_fetch_requests_bbox_subset_per_year_and_concatenates():
    body = (FIXTURES / "gridmet_pr.nc").read_bytes()
    route = respx.get(
        "http://thredds.northwestknowledge.net:8080/thredds/ncss/agg_met_pr_1979_CurrentYear_CONUS.nc"
    ).mock(return_value=httpx.Response(200, content=body))

    df = asyncio.run(gridmet.fetch("pr", BBOX, start="2023-12-30", end="2024-01-02"))

    assert route.call_count == 2
    first = route.calls[0].request.url.params
    assert first["var"] == "precipitation_amount"
    assert first["north"] == "28.62"
    assert first["west"] == "-81.6"
    assert first["time_start"] == "2023-12-30T00:00:00Z"
    assert first["time_end"] == "2023-12-31T00:00:00Z"
    assert first["accept"] == "netcdf"
    # same fixture returned twice -> duplicate dates dropped
    assert df.index.is_unique
    assert df.shape == (5, 88)


@respx.mock
def test_fetch_raises_on_http_error():
    respx.get(url__startswith="http://thredds.northwestknowledge.net").mock(
        return_value=httpx.Response(400, text="bad var")
    )
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(gridmet.fetch("pr", BBOX, start="2024-01-01", end="2024-01-02"))
