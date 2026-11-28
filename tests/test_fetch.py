import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest

from shingle import config, fetch
from shingle.sources import dbhydro, gridmet, nldi, usgs

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fake_sources(monkeypatch):
    calls = {"usgs": [], "gridmet": [], "dbhydro": []}

    async def fake_dv(site, start, end=None, client=None):
        calls["usgs"].append(site)
        return pd.DataFrame(
            {"discharge_cfs": 1.0, "stage_ft": 55.0},
            index=pd.date_range(start, end, name="date"),
        )

    async def fake_gridmet(var, bbox, start, end, client=None):
        calls["gridmet"].append(var)
        idx = pd.date_range(start, end, name="date")
        return pd.DataFrame({"28.3750_-81.4583": 1.0, "28.9000_-80.0000": 2.0}, index=idx)

    async def fake_dbhydro(keys, start, end, client=None):
        calls["dbhydro"].append(tuple(keys))
        idx = pd.date_range(start, end, name="date")
        return pd.DataFrame({k: 53.0 for k in keys}, index=idx)

    monkeypatch.setattr(dbhydro, "fetch", fake_dbhydro)
    basin_geojson = json.loads((FIXTURES / "nldi_basin.json").read_text())
    monkeypatch.setattr(usgs, "fetch_dv", fake_dv)
    monkeypatch.setattr(gridmet, "fetch", fake_gridmet)
    async def fake_basin(site, client=None):
        return basin_geojson

    monkeypatch.setattr(nldi, "fetch_basin_geojson", fake_basin)
    return calls


def test_fetch_history_writes_gauges_gridmet_and_cells(tmp_path, fake_sources):
    asyncio.run(fetch.fetch_history(tmp_path, today="2020-01-10", start="2020-01-01"))

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
    asyncio.run(fetch.fetch_history(tmp_path, today="2020-01-10", start="2020-01-01"))

    cells = pd.read_parquet(tmp_path / "cells.parquet")
    assert cells.loc["28.3750_-81.4583", "basin_frac"] > 0.9  # basin interior
    assert cells.loc["28.9000_-80.0000", "basin_frac"] == 0.0  # far away
    assert "dist_outlet_km" in cells.columns


def test_load_history_returns_named_frames(tmp_path, fake_sources):
    asyncio.run(fetch.fetch_history(tmp_path, today="2020-01-10", start="2020-01-01"))

    h = fetch.load_history(tmp_path)

    assert set(h) == {"gauge", "upstream", "pr", "pet", "cells", "basin", "lakes", "iv"}
    assert h["iv"] == {"gauge": None, "upstream": None}  # no IV files fetched yet
    assert list(h["lakes"].columns) == list(dbhydro.SERIES)
    assert h["lakes"]["toho_stage"].iloc[0] == 53.0
    assert h["gauge"]["stage_ft"].iloc[0] == 55.0
    assert h["pr"].shape == (10, 2)


def test_dbhydro_requested_in_one_call_with_all_keys(tmp_path, fake_sources):
    asyncio.run(fetch.fetch_history(tmp_path, today="2020-01-10", start="2020-01-01"))

    assert fake_sources["dbhydro"] == [tuple(dbhydro.all_dbkeys())]


def test_fetch_nwm_retro_writes_parquet_once(tmp_path, monkeypatch):
    from shingle.sources import nwm_retro

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
    from shingle.sources import mrms

    calls = []

    async def fake_fetch(start, end, bbox, workers=8):
        calls.append((start, end))
        idx = pd.date_range(start, end, name="date")
        return pd.DataFrame({"28.500_-81.500": 1.0}, index=idx)

    monkeypatch.setattr(mrms, "fetch", fake_fetch)

    asyncio.run(fetch.fetch_mrms_history(tmp_path, start="2024-01-15", today="2024-03-10"))
    assert calls == [("2024-01-15", "2024-01-31"), ("2024-02-01", "2024-02-29"), ("2024-03-01", "2024-03-10")]
    assert sorted(p.name for p in (tmp_path / "mrms").iterdir()) == ["2024-01.parquet", "2024-02.parquet", "2024-03.parquet"]

    calls.clear()
    asyncio.run(fetch.fetch_mrms_history(tmp_path, start="2024-01-15", today="2024-03-20"))
    # Jan, Feb complete -> skipped; in March only the days not cached yet are asked for
    assert calls == [("2024-03-11", "2024-03-20")]

    df = fetch.load_mrms(tmp_path)
    assert df.index.min() == pd.Timestamp("2024-01-15")
    assert df.index.max() == pd.Timestamp("2024-03-20")
    assert df.index.is_unique


def test_fetch_iv_history_pulls_yearly_then_incrementally(tmp_path, monkeypatch):
    calls = []

    async def fake_iv(site, start, end=None, client=None):
        calls.append((site, start, end))
        idx = pd.date_range(pd.Timestamp(start).tz_localize(None), pd.Timestamp(end).tz_localize(None), freq="6h", tz="UTC", name="time")
        return pd.DataFrame({"discharge_cfs": 100.0, "stage_ft": 55.0}, index=idx)

    monkeypatch.setattr(usgs, "fetch_iv", fake_iv)

    asyncio.run(fetch.fetch_iv_history(tmp_path, sites=("A",), start="2021-06-01", today="2023-03-10"))
    first = [c for c in calls]
    assert [c[1][:10] for c in first] == ["2021-06-01", "2022-01-01", "2023-01-01"]
    df = pd.read_parquet(tmp_path / "usgs_iv_A.parquet")
    assert df.index.is_unique and df.index.max() >= pd.Timestamp("2023-03-09", tz="UTC")

    calls.clear()
    asyncio.run(fetch.fetch_iv_history(tmp_path, sites=("A",), start="2021-06-01", today="2023-03-20"))
    assert len(calls) == 1 and calls[0][1][:10] == "2023-03-07"  # last stored day - 3
    df2 = pd.read_parquet(tmp_path / "usgs_iv_A.parquet")
    assert df2.index.is_unique and df2.index.min() == df.index.min()
    assert df2.index.max() >= pd.Timestamp("2023-03-19", tz="UTC")


def test_load_history_includes_iv_when_present(tmp_path, fake_sources, monkeypatch):
    asyncio.run(fetch.fetch_history(tmp_path, today="2020-01-10", start="2020-01-01"))
    t = pd.date_range("2020-01-01", periods=48, freq="1h", tz="UTC", name="time")
    pd.DataFrame({"discharge_cfs": 1.0, "stage_ft": 55.0}, index=t).to_parquet(tmp_path / f"usgs_iv_{config.SITE}.parquet")

    h = fetch.load_history(tmp_path)

    assert len(h["iv"]["gauge"]) == 48 and h["iv"]["upstream"] is None


def test_fetch_iv_history_checkpoints_each_year(tmp_path, monkeypatch):
    async def flaky_iv(site, start, end=None, client=None):
        if start.startswith("2023"):
            raise usgs.httpx.ConnectError("down")  # 2023 never works
        idx = pd.date_range(pd.Timestamp(start).tz_localize(None), pd.Timestamp(end).tz_localize(None), freq="6h", tz="UTC", name="time")
        return pd.DataFrame({"discharge_cfs": 100.0, "stage_ft": 55.0}, index=idx)

    monkeypatch.setattr(usgs, "fetch_iv", flaky_iv)

    with pytest.raises(usgs.httpx.ConnectError):
        asyncio.run(fetch.fetch_iv_history(tmp_path, sites=("A",), start="2021-06-01", today="2023-03-10"))

    # years completed before the failure are on disk -> a rerun resumes from there
    df = pd.read_parquet(tmp_path / "usgs_iv_A.parquet")
    assert df.index.max() >= pd.Timestamp("2022-12-31", tz="UTC")


def test_fetch_gefs_history_writes_monthly_files_and_skips_complete(tmp_path, monkeypatch):
    from shingle.sources import gefs

    calls = []

    async def fake_run(run, fhours=None, bbox=None, kind="mean", client=None):
        calls.append((run, kind))
        if run == pd.Timestamp("2022-10-02"):
            return None  # missing run
        return pd.DataFrame({"28.5000_-81.5000": 1.0}, index=pd.Index(gefs.FHOURS, name="fhour"))

    monkeypatch.setattr(gefs, "fetch_run", fake_run)

    asyncio.run(fetch.fetch_gefs_history(tmp_path, start="2022-09-29", end="2022-10-03", kinds=("mean",), workers=2))
    assert sorted({c[0] for c in calls}) == list(pd.date_range("2022-09-29", "2022-10-03"))
    b = fetch.load_gefs(tmp_path, "mean")
    assert b.index.names == ["run", "fhour"]
    assert sorted(b.index.get_level_values("run").unique()) == [pd.Timestamp(d) for d in ("2022-09-29", "2022-09-30", "2022-10-01", "2022-10-03")]
    assert (tmp_path / "gefs" / "mean" / "2022-09.parquet").exists()

    calls.clear()  # complete runs are not re-fetched; only the run missing from the archive is asked for again
    asyncio.run(fetch.fetch_gefs_history(tmp_path, start="2022-09-29", end="2022-10-03", kinds=("mean",), workers=2, today="2022-10-04"))
    assert [c[0] for c in calls] == [pd.Timestamp("2022-10-02")]


def test_fetch_mrms_history_12z_goes_to_own_dir(tmp_path, monkeypatch):
    from shingle.sources import mrms

    seen = []

    async def fake_fetch(start, end, bbox, workers=8, hour=5):
        seen.append(hour)
        return pd.DataFrame({"28.500_-81.500": 2.0}, index=pd.date_range(start, end, name="date"))

    monkeypatch.setattr(mrms, "fetch", fake_fetch)

    asyncio.run(fetch.fetch_mrms_history(tmp_path, start="2024-01-01", today="2024-01-05", hour=12))

    assert seen == [12]
    assert (tmp_path / "mrms12" / "2024-01.parquet").exists() and not (tmp_path / "mrms").exists()
    assert fetch.load_mrms(tmp_path, hour=12)["28.500_-81.500"].eq(2.0).all()


def test_fetch_gefs_history_merges_new_runs_into_the_month_file(tmp_path, monkeypatch):
    # Each forecast run asks only for recent runs; that used to overwrite the month
    # file and drop every earlier run of the month.
    from shingle.sources import gefs

    calls = []

    async def fake_run(run, fhours=None, bbox=None, kind="mean", client=None):
        calls.append(run)
        return pd.DataFrame({"28.5000_-81.5000": float(run.day)}, index=pd.Index(gefs.FHOURS, name="fhour"))

    monkeypatch.setattr(gefs, "fetch_run", fake_run)

    asyncio.run(fetch.fetch_gefs_history(tmp_path, start="2026-11-01", end="2026-11-20", kinds=("mean",), today="2026-11-20"))
    calls.clear()
    asyncio.run(fetch.fetch_gefs_history(tmp_path, start="2026-11-15", end="2026-11-25", kinds=("mean",), today="2026-11-25"))

    runs = fetch.load_gefs(tmp_path).index.get_level_values("run").unique()
    assert list(runs) == list(pd.date_range("2026-11-01", "2026-11-25"))
    # complete cached runs are skipped
    assert calls == list(pd.date_range("2026-11-21", "2026-11-25"))


def test_run_all_overlaps_tasks_and_finishes_the_rest_when_one_fails(capsys):
    import time

    done = []

    async def slow(name):
        await asyncio.sleep(0.2)
        done.append(name)
        return name

    async def broken():
        raise RuntimeError("source down")

    t0 = time.monotonic()
    assert asyncio.run(fetch.run_all({"a": slow("a"), "b": slow("b"), "c": slow("c")})) == {"a": "a", "b": "b", "c": "c"}
    assert time.monotonic() - t0 < 0.5  # three 0.2 s waits at the same time, not one after another

    done.clear()
    with pytest.raises(RuntimeError, match="source down"):
        asyncio.run(fetch.run_all({"bad": broken(), "a": slow("a")}, label="update"))
    assert done == ["a"]  # the healthy source still finished (and saved its cache)
    assert "update: a " in capsys.readouterr().err


def test_fetch_mrms_history_asks_again_for_days_that_came_back_empty(tmp_path, monkeypatch):
    from shingle.sources import mrms

    calls = []

    async def fake_fetch(start, end, bbox, workers=8):
        calls.append((start, end))
        idx = pd.date_range(start, end, name="date")
        df = pd.DataFrame({"28.500_-81.500": 1.0}, index=idx)
        if not calls[1:]:
            df.loc["2024-03-05"] = float("nan")  # not published yet on the first try
        return df

    monkeypatch.setattr(mrms, "fetch", fake_fetch)

    asyncio.run(fetch.fetch_mrms_history(tmp_path, start="2024-03-01", today="2024-03-08"))
    asyncio.run(fetch.fetch_mrms_history(tmp_path, start="2024-03-01", today="2024-03-09"))

    assert calls == [("2024-03-01", "2024-03-08"), ("2024-03-05", "2024-03-05"), ("2024-03-09", "2024-03-09")]
    df = fetch.load_mrms(tmp_path)
    assert df.notna().all().all() and list(df.index) == list(pd.date_range("2024-03-01", "2024-03-09"))


def test_fetch_gefs_history_refetches_partial_runs(tmp_path, monkeypatch):
    from shingle.sources import gefs

    calls = []

    async def fake_run(run, fhours=None, bbox=None, kind="mean", client=None):
        calls.append(run)
        hours = gefs.FHOURS[:5] if len(calls) == 1 else gefs.FHOURS  # first try: still uploading
        return pd.DataFrame({"28.5000_-81.5000": 1.0}, index=pd.Index(hours, name="fhour"))

    monkeypatch.setattr(gefs, "fetch_run", fake_run)

    for _ in range(3):
        asyncio.run(fetch.fetch_gefs_history(tmp_path, start="2026-11-25", end="2026-11-25", kinds=("mean",), today="2026-11-25"))

    assert len(calls) == 2  # partial, then complete; the third call sends nothing
    assert len(fetch.load_gefs(tmp_path)) == len(gefs.FHOURS)


def test_fetch_iv_history_between_revision_passes_asks_only_from_the_last_hour(tmp_path, monkeypatch):
    calls = []

    async def fake_iv(site, start, end=None, client=None):
        calls.append(start)
        idx = pd.date_range(pd.Timestamp(start).tz_localize(None), pd.Timestamp(end).tz_localize(None), freq="1h", tz="UTC", name="time")
        return pd.DataFrame({"discharge_cfs": 100.0, "stage_ft": 55.0}, index=idx[:-1])

    monkeypatch.setattr(usgs, "fetch_iv", fake_iv)
    every, t0 = pd.Timedelta(hours=20), pd.Timestamp("2023-03-10T13:00Z")

    def run(today, hours_later):
        asyncio.run(fetch.fetch_iv_history(tmp_path, sites=("A",), start="2023-03-01", today=today, revise_every=every, now=t0 + pd.Timedelta(hours=hours_later)))

    run("2023-03-10T13:00", 0)
    run("2023-03-10T16:00", 3)
    run("2023-03-11T10:00", 21)

    assert calls == ["2023-03-01T00:00Z", "2023-03-10T12:00Z", "2023-03-07T00:00Z"]


def test_fetch_history_keeps_cached_gridmet_when_server_is_down(tmp_path, fake_sources, monkeypatch, capsys):
    # gridMET's server has gone unreachable for hours at a time; the forecast fills days after gridMET's last
    # cached day from radar, so a stale gridMET cache is usable and the run should carry on with a warning.
    import httpx

    monkeypatch.delenv("EE_SERVICE_ACCOUNT_KEY", raising=False)  # no Earth Engine backup in this test
    asyncio.run(fetch.fetch_history(tmp_path, today="2020-01-10", start="2020-01-01"))
    before = pd.read_parquet(tmp_path / "gridmet_pr.parquet")

    async def down(var, bbox, start, end, client=None):
        raise httpx.ConnectError("All connection attempts failed")

    monkeypatch.setattr(gridmet, "fetch", down)
    asyncio.run(fetch.fetch_history(tmp_path, today="2020-01-14", start="2020-01-01"))
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "gridmet_pr.parquet"), before)
    assert pd.read_parquet(tmp_path / f"usgs_{config.SITE}.parquet").index.max() == pd.Timestamp("2020-01-14")
    assert "gridMET unreachable" in capsys.readouterr().err


def test_fetch_history_still_fails_when_gridmet_is_down_and_nothing_is_cached(tmp_path, fake_sources, monkeypatch):
    import httpx

    async def down(var, bbox, start, end, client=None):
        raise httpx.ConnectError("All connection attempts failed")

    monkeypatch.setattr(gridmet, "fetch", down)
    with pytest.raises(httpx.ConnectError):
        asyncio.run(fetch.fetch_history(tmp_path, today="2020-01-10", start="2020-01-01"))
