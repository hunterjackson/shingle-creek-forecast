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

    assert set(h) == {"gauge", "upstream", "pr", "pet", "cells", "basin", "lakes", "iv"}
    assert h["iv"] == {"gauge": None, "upstream": None}  # no IV files fetched yet
    assert list(h["lakes"].columns) == list(dbhydro.SERIES)
    assert h["lakes"]["toho_stage"].iloc[0] == 53.0
    assert h["gauge"]["stage_ft"].iloc[0] == 55.0
    assert h["pr"].shape == (10, 2)


def test_dbhydro_requested_in_one_call_with_all_keys(tmp_path, fake_sources):
    fetch.fetch_history(tmp_path, today="2020-01-10", start="2020-01-01")

    assert fake_sources["dbhydro"] == [tuple(dbhydro.all_dbkeys())]


def test_fetch_nwm_retro_writes_parquet_once(tmp_path, monkeypatch):
    from home_display.shingle.sources import nwm_retro

    calls = []

    def fake_read(feature_id, start, end, store=None, min_hours=1):
        calls.append((feature_id, start, end))
        return pd.Series([1.0, 2.0], index=pd.date_range(start, periods=2, name="date"), name="nwm_retro_cfs")

    monkeypatch.setattr(nwm_retro, "read_reach_daily", fake_read)

    fetch.fetch_nwm_retro(tmp_path, start="1979-02-02", end="2023-01-31")
    fetch.fetch_nwm_retro(tmp_path, start="1979-02-02", end="2023-01-31")

    assert calls == [(int(config.NWM_REACH), "1979-02-02", "2023-01-31")]
    df = pd.read_parquet(tmp_path / "nwm_retro.parquet")
    assert list(df.columns) == ["nwm_retro_cfs"]


def test_fetch_mrms_history_checkpoints_by_month_and_skips_complete_months(tmp_path, monkeypatch):
    from home_display.shingle.sources import mrms

    calls = []

    def fake_fetch(start, end, bbox, workers=8):
        calls.append((start, end))
        idx = pd.date_range(start, end, name="date")
        return pd.DataFrame({"28.500_-81.500": 1.0}, index=idx)

    monkeypatch.setattr(mrms, "fetch", fake_fetch)

    fetch.fetch_mrms_history(tmp_path, start="2024-01-15", today="2024-03-10")
    assert calls == [("2024-01-15", "2024-01-31"), ("2024-02-01", "2024-02-29"), ("2024-03-01", "2024-03-10")]
    assert sorted(p.name for p in (tmp_path / "mrms").iterdir()) == ["2024-01.parquet", "2024-02.parquet", "2024-03.parquet"]

    calls.clear()
    fetch.fetch_mrms_history(tmp_path, start="2024-01-15", today="2024-03-20")
    # Jan, Feb complete -> skipped; March (current, incomplete) refetched
    assert calls == [("2024-03-01", "2024-03-20")]

    df = fetch.load_mrms(tmp_path)
    assert df.index.min() == pd.Timestamp("2024-01-15")
    assert df.index.max() == pd.Timestamp("2024-03-20")
    assert df.index.is_unique


def test_fetch_iv_history_pulls_yearly_then_incrementally(tmp_path, monkeypatch):
    calls = []

    def fake_iv(site, start, end=None, client=None):
        calls.append((site, start, end))
        idx = pd.date_range(pd.Timestamp(start).tz_localize(None), pd.Timestamp(end).tz_localize(None), freq="6h", tz="UTC", name="time")
        return pd.DataFrame({"discharge_cfs": 100.0, "stage_ft": 55.0}, index=idx)

    monkeypatch.setattr(usgs, "fetch_iv", fake_iv)

    fetch.fetch_iv_history(tmp_path, sites=("A",), start="2021-06-01", today="2023-03-10")
    first = [c for c in calls]
    assert [c[1][:10] for c in first] == ["2021-06-01", "2022-01-01", "2023-01-01"]
    df = pd.read_parquet(tmp_path / "usgs_iv_A.parquet")
    assert df.index.is_unique and df.index.max() >= pd.Timestamp("2023-03-09", tz="UTC")

    calls.clear()
    fetch.fetch_iv_history(tmp_path, sites=("A",), start="2021-06-01", today="2023-03-20")
    assert len(calls) == 1 and calls[0][1][:10] == "2023-03-07"  # last stored day - 3
    df2 = pd.read_parquet(tmp_path / "usgs_iv_A.parquet")
    assert df2.index.is_unique and df2.index.min() == df.index.min()
    assert df2.index.max() >= pd.Timestamp("2023-03-19", tz="UTC")


def test_load_history_includes_iv_when_present(tmp_path, fake_sources, monkeypatch):
    fetch.fetch_history(tmp_path, today="2020-01-10", start="2020-01-01")
    t = pd.date_range("2020-01-01", periods=48, freq="1h", tz="UTC", name="time")
    pd.DataFrame({"discharge_cfs": 1.0, "stage_ft": 55.0}, index=t).to_parquet(tmp_path / f"usgs_iv_{config.SITE}.parquet")

    h = fetch.load_history(tmp_path)

    assert len(h["iv"]["gauge"]) == 48 and h["iv"]["upstream"] is None


def test_fetch_iv_history_retries_and_checkpoints_each_year(tmp_path, monkeypatch):
    state = {"n": 0}

    def flaky_iv(site, start, end=None, client=None):
        state["n"] += 1
        if start.startswith("2022") and state["n"] == 2:
            raise usgs.httpx.ConnectError("reset")  # first try at 2022 fails, retry works
        if start.startswith("2023"):
            raise usgs.httpx.ConnectError("down")  # 2023 never works
        idx = pd.date_range(pd.Timestamp(start).tz_localize(None), pd.Timestamp(end).tz_localize(None), freq="6h", tz="UTC", name="time")
        return pd.DataFrame({"discharge_cfs": 100.0, "stage_ft": 55.0}, index=idx)

    monkeypatch.setattr(usgs, "fetch_iv", flaky_iv)

    with pytest.raises(usgs.httpx.ConnectError):
        fetch.fetch_iv_history(tmp_path, sites=("A",), start="2021-06-01", today="2023-03-10", retries=2, retry_wait=0)

    # years completed before the failure are on disk -> a rerun resumes from there
    df = pd.read_parquet(tmp_path / "usgs_iv_A.parquet")
    assert df.index.max() >= pd.Timestamp("2022-12-31", tz="UTC")


def test_fetch_gefs_history_writes_monthly_files_and_skips_complete(tmp_path, monkeypatch):
    from home_display.shingle.sources import gefs

    calls = []

    def fake_run(run, fhours=None, bbox=None, kind="mean", client=None):
        calls.append((run, kind))
        if run == pd.Timestamp("2022-10-02"):
            return None  # missing run
        return pd.DataFrame({"28.5000_-81.5000": [1.0, 2.0]}, index=pd.Index([18, 24], name="fhour"))

    monkeypatch.setattr(gefs, "fetch_run", fake_run)

    fetch.fetch_gefs_history(tmp_path, start="2022-09-29", end="2022-10-03", kinds=("mean",), workers=2)
    assert sorted({c[0] for c in calls}) == list(pd.date_range("2022-09-29", "2022-10-03"))
    b = fetch.load_gefs(tmp_path, "mean")
    assert b.index.names == ["run", "fhour"]
    assert sorted(b.index.get_level_values("run").unique()) == [pd.Timestamp(d) for d in ("2022-09-29", "2022-09-30", "2022-10-01", "2022-10-03")]
    assert (tmp_path / "gefs" / "mean" / "2022-09.parquet").exists()

    calls.clear()  # September is complete and final -> not re-fetched; October re-fetched (not final)
    fetch.fetch_gefs_history(tmp_path, start="2022-09-29", end="2022-10-03", kinds=("mean",), workers=2, today="2022-10-04")
    assert all(c[0].month == 10 for c in calls)


def test_fetch_mrms_history_12z_goes_to_own_dir(tmp_path, monkeypatch):
    from home_display.shingle.sources import mrms

    seen = []

    def fake_fetch(start, end, bbox, workers=8, hour=5):
        seen.append(hour)
        return pd.DataFrame({"28.500_-81.500": 2.0}, index=pd.date_range(start, end, name="date"))

    monkeypatch.setattr(mrms, "fetch", fake_fetch)

    fetch.fetch_mrms_history(tmp_path, start="2024-01-01", today="2024-01-05", hour=12)

    assert seen == [12]
    assert (tmp_path / "mrms12" / "2024-01.parquet").exists() and not (tmp_path / "mrms").exists()
    assert fetch.load_mrms(tmp_path, hour=12)["28.500_-81.500"].eq(2.0).all()
