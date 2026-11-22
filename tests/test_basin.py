import json
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx
from shapely.geometry import Polygon, box

from shingle.sources import nldi
from shingle import basin

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def basin_geojson():
    return json.loads((FIXTURES / "nldi_basin.json").read_text())


def test_parse_basin_returns_polygon_with_expected_extent(basin_geojson):
    poly = nldi.parse_basin(basin_geojson)

    minx, miny, maxx, maxy = poly.bounds
    assert minx == pytest.approx(-81.5208, abs=1e-3)
    assert maxy == pytest.approx(28.5581, abs=1e-3)
    assert poly.is_valid


@respx.mock
def test_fetch_basin_hits_nldi_for_usgs_site(basin_geojson):
    route = respx.get(
        "https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-02264495/basin"
    ).mock(return_value=httpx.Response(200, json=basin_geojson))

    poly = nldi.fetch_basin("02264495")

    assert route.called
    assert poly.area > 0


def test_cell_basin_fraction_full_partial_and_outside():
    square = box(0.0, 0.0, 1.0, 1.0)
    cells = pd.DataFrame(
        {"lat": [0.5, 1.0, 3.0], "lon": [0.5, 0.5, 3.0]},
        index=pd.Index(["inside", "half", "outside"], name="cell_id"),
    )

    frac = basin.cell_basin_fraction(cells, square, res=0.5)

    assert frac["inside"] == pytest.approx(1.0)
    assert frac["half"] == pytest.approx(0.5)
    assert frac["outside"] == pytest.approx(0.0)


def test_cell_basin_fraction_handles_non_square_polygon():
    tri = Polygon([(0, 0), (1, 0), (0, 1)])
    cells = pd.DataFrame({"lat": [0.5], "lon": [0.5]}, index=pd.Index(["c"], name="cell_id"))

    frac = basin.cell_basin_fraction(cells, tri, res=1.0)

    assert frac["c"] == pytest.approx(0.5)


def test_annotate_cells_adds_fraction_and_distance_to_outlet():
    square = box(0.0, 0.0, 1.0, 1.0)
    cells = pd.DataFrame({"lat": [0.5], "lon": [0.5]}, index=pd.Index(["c"], name="cell_id"))

    out = basin.annotate_cells(cells, square, outlet=(0.0, 0.5), res=0.5)

    assert list(out.columns) == ["lat", "lon", "basin_frac", "dist_outlet_km"]
    # 0.5 deg of latitude ~ 55.6 km
    assert out.loc["c", "dist_outlet_km"] == pytest.approx(55.6, rel=0.01)
