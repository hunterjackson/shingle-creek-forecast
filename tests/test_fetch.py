import json
from pathlib import Path

import pandas as pd
import pytest

from home_display.shingle import config, fetch
from home_display.shingle.sources import dbhydro, gridmet, nldi, usgs

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fake_sources(monkeypatch):
    calls = {"usgs": [], "gridmet": [], "dbhydro": []}

    def fake_dv(site, start, end=None, client=None):
        calls["usgs"].append(site)
        return pd.DataFrame(
            {"discharge_cfs": 1.0, "stage_ft": 55.0},
            index=pd.date_range(start, end, name="date"),
        )

    def fake_gridmet(var, bbox, start, end, client=None):
        calls["gridmet"].append(var)
        idx = pd.date_range(start, end, name="date")
        return pd.DataFrame({"28.3750_-81.4583": 1.0, "28.9000_-80.0000": 2.0}, index=idx)

    def fake_dbhydro(keys, start, end, client=None):
        calls["dbhydro"].append(tuple(keys))
        idx = pd.date_range(start, end, name="date")
        return pd.DataFrame({k: 53.0 for k in keys}, index=idx)

    monkeypatch.setattr(dbhydro, "fetch", fake_dbhydro)
    basin_geojson = json.loads((FIXTURES / "nldi_basin.json").read_text())
    monkeypatch.setattr(usgs, "fetch_dv", fake_dv)
    monkeypatch.setattr(gridmet, "fetch", fake_gridmet)
    monkeypatch.setattr(nldi, "fetch_basin_geojson", lambda site, client=None: basin_geojson)
    return calls


def test_fetch_history_writes_gauges_gridmet_and_cells(tmp_path, fake_sources):
    fetch.fetch_history(tmp_path, today="2020-01-10", start="2020-01-01")

    assert set(fake_sources["usgs"]) == {config.SITE, config.UPSTREAM_SITE}
    assert set(fake_sources["gridmet"]) == {"pr", "pet"}
    for name in [
        f"usgs_{config.SITE}.parquet",
        f"usgs_{config.UPSTREAM_SITE}.parquet",
        "gridmet_pr.parquet",
        "gridmet_pet.parquet",
        "cells.parquet",
        "basin.geojson",
        "dbhydro.parquet",
    ]:
        assert (tmp_path / name).exists(), name


def test_cells_table_has_basin_fraction(tmp_path, fake_sources):
    fetch.fetch_history(tmp_path, today="2020-01-10", start="2020-01-01")

    cells = pd.read_parquet(tmp_path / "cells.parquet")
    assert cells.loc["28.3750_-81.4583", "basin_frac"] > 0.9  # basin interior
    assert cells.loc["28.9000_-80.0000", "basin_frac"] == 0.0  # far away
    assert "dist_outlet_km" in cells.columns


def test_load_history_returns_named_frames(tmp_path, fake_sources):
    fetch.fetch_history(tmp_path, today="2020-01-10", start="2020-01-01")

    h = fetch.load_history(tmp_path)

    assert set(h) == {"gauge", "upstream", "pr", "pet", "cells", "basin", "lakes"}
    assert list(h["lakes"].columns) == list(dbhydro.SERIES)
    assert h["lakes"]["toho_stage"].iloc[0] == 53.0
    assert h["gauge"]["stage_ft"].iloc[0] == 55.0
    assert h["pr"].shape == (10, 2)


def test_dbhydro_requested_in_one_call_with_all_keys(tmp_path, fake_sources):
    fetch.fetch_history(tmp_path, today="2020-01-10", start="2020-01-01")

    assert fake_sources["dbhydro"] == [tuple(dbhydro.all_dbkeys())]
