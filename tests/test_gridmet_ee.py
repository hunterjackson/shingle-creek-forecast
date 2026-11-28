import asyncio
import json

import httpx
import pandas as pd
import pytest

from shingle import config, fetch
from shingle.sources import gridmet, gridmet_ee
from test_fetch import fake_sources  # noqa: F401  (pytest fixture)

KEY = json.dumps({"type": "service_account", "client_email": "bot@proj.iam.gserviceaccount.com", "private_key": "x"})


def test_to_frame_pivots_samples_into_the_gridmet_layout():
    rows = [("20261120", "28.6083_-81.6000", 6.7), ("20261120", "28.5667_-81.6000", 6.1),
            ("20261121", "28.6083_-81.6000", 13.4), ("20261121", "28.5667_-81.6000", None)]
    df = gridmet_ee.to_frame(rows, ["28.6083_-81.6000", "28.5667_-81.6000"])
    assert list(df.columns) == ["28.6083_-81.6000", "28.5667_-81.6000"]
    assert list(df.index) == [pd.Timestamp("2026-11-20"), pd.Timestamp("2026-11-21")]
    assert df.index.name == "date"
    assert df.loc["2026-11-21", "28.6083_-81.6000"] == 13.4
    assert pd.isna(df.loc["2026-11-21", "28.5667_-81.6000"])


def test_to_frame_with_no_samples_is_empty_with_the_cell_columns():
    df = gridmet_ee.to_frame([], ["a_b"])
    assert df.empty and list(df.columns) == ["a_b"] and df.index.name == "date"


def test_configured_reads_key_and_project_from_the_environment(monkeypatch):
    monkeypatch.delenv("EE_SERVICE_ACCOUNT_KEY", raising=False)
    monkeypatch.delenv("EE_PROJECT", raising=False)
    assert not gridmet_ee.configured()
    monkeypatch.setenv("EE_SERVICE_ACCOUNT_KEY", KEY)
    monkeypatch.setenv("EE_PROJECT", "proj")
    assert gridmet_ee.configured()


def test_fetch_uses_the_band_cells_and_inclusive_end(monkeypatch):
    seen = {}

    def fake_sample(band, cells, start, end):
        seen.update(band=band, cells=list(cells.index), start=start, end=end)
        return [(d.strftime("%Y%m%d"), c, 1.0) for d in pd.date_range(start, end) for c in cells.index]

    monkeypatch.setattr(gridmet_ee, "_sample", fake_sample)
    cells = gridmet.cells_from_columns(["28.6083_-81.6000", "28.5667_-81.6000"])
    df = asyncio.run(gridmet_ee.fetch("pet", cells, "2026-11-19", "2026-11-21"))
    assert seen == {"band": "pet", "cells": list(cells.index), "start": "2026-11-19", "end": "2026-11-21"}
    assert df.shape == (3, 2)


def _down(*a, **k):
    async def go():
        raise httpx.ConnectError("All connection attempts failed")
    return go()


def test_fetch_history_falls_back_to_earth_engine_when_thredds_is_down(tmp_path, monkeypatch, capsys, fake_sources):
    asyncio.run(fetch.fetch_history(tmp_path, today="2020-01-10", start="2020-01-01"))
    monkeypatch.setattr(gridmet, "fetch", _down)
    monkeypatch.setenv("EE_SERVICE_ACCOUNT_KEY", KEY)
    monkeypatch.setenv("EE_PROJECT", "proj")
    asked = []

    async def fake_ee(var, cells, start, end):
        asked.append((var, start, end))
        idx = pd.date_range(start, end, name="date")
        return pd.DataFrame({c: 9.0 for c in cells.index}, index=idx)

    monkeypatch.setattr(gridmet_ee, "fetch", fake_ee)
    asyncio.run(fetch.fetch_history(tmp_path, today="2020-01-14", start="2020-01-01"))
    pr = pd.read_parquet(tmp_path / "gridmet_pr.parquet")
    assert pr.index.max() == pd.Timestamp("2020-01-14") and (pr.loc["2020-01-11":] == 9.0).all().all()
    assert {a[0] for a in asked} == {"pr", "pet"}
    assert "using Earth Engine" in capsys.readouterr().err


def test_fetch_history_keeps_the_cache_when_earth_engine_also_fails(tmp_path, monkeypatch, capsys, fake_sources):
    asyncio.run(fetch.fetch_history(tmp_path, today="2020-01-10", start="2020-01-01"))
    before = pd.read_parquet(tmp_path / "gridmet_pr.parquet")
    monkeypatch.setattr(gridmet, "fetch", _down)
    monkeypatch.setenv("EE_SERVICE_ACCOUNT_KEY", KEY)
    monkeypatch.setenv("EE_PROJECT", "proj")

    async def broken_ee(var, cells, start, end):
        raise RuntimeError("Earth Engine said no")

    monkeypatch.setattr(gridmet_ee, "fetch", broken_ee)
    asyncio.run(fetch.fetch_history(tmp_path, today="2020-01-14", start="2020-01-01"))
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "gridmet_pr.parquet"), before)
    err = capsys.readouterr().err
    assert "Earth Engine failed" in err and "gridMET unreachable" in err


def test_compare_reports_overlap_differences_and_last_days():
    idx = pd.date_range("2026-11-01", "2026-11-10", name="date")
    cached = pd.DataFrame({"a_b": 1.0, "c_d": 2.0}, index=idx)
    ee_df = pd.DataFrame({"a_b": 1.0, "c_d": 2.0}, index=pd.date_range("2026-11-03", "2026-11-11", name="date"))
    ee_df.loc["2026-11-05", "c_d"] = 2.3
    r = gridmet_ee.compare(cached, ee_df)
    assert r["overlap_days"] == 8 and r["cells"] == 2
    assert r["max_abs_diff"] == pytest.approx(0.3) and r["cached_last"] == "2026-11-10" and r["ee_last"] == "2026-11-11"
    assert r["missing_cells"] == 0


def test_fetch_splits_long_ranges_under_the_5000_element_limit(monkeypatch):
    calls = []

    def fake_sample(band, cells, start, end):
        calls.append((start, end))
        return [(d.strftime("%Y%m%d"), c, 1.0) for d in pd.date_range(start, end) for c in cells.index]

    monkeypatch.setattr(gridmet_ee, "_sample", fake_sample)
    cells = gridmet.cells_from_columns([f"28.{i:04d}_-81.6000" for i in range(88)])
    df = asyncio.run(gridmet_ee.fetch("pr", cells, "2026-09-01", "2026-11-27"))
    assert all((pd.Timestamp(e) - pd.Timestamp(s)).days + 1 <= 5000 // 88 for s, e in calls)
    assert calls[0][0] == "2026-09-01" and calls[-1][1] == "2026-11-27"
    assert len(df) == 88 and df.index.is_unique
