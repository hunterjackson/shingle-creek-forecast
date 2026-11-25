import asyncio
import pandas as pd
import pytest

from shingle import cache


def daily(start, end, value):
    idx = pd.date_range(start, end, name="date")
    return pd.DataFrame({"x": float(value)}, index=idx)


class FakeFetch:
    def __init__(self, value):
        self.value = value
        self.calls = []

    async def __call__(self, start, end):
        self.calls.append((start, end))
        return daily(start, end, self.value)


def test_first_call_fetches_full_range_and_writes_parquet(tmp_path):
    path = tmp_path / "s.parquet"
    fetch = FakeFetch(1)

    df = asyncio.run(cache.update_daily(path, fetch, default_start="2020-01-01", today="2020-01-10"))

    assert fetch.calls == [("2020-01-01", "2020-01-10")]
    assert path.exists()
    assert len(df) == 10
    pd.testing.assert_frame_equal(pd.read_parquet(path), df, check_freq=False)


def test_second_call_fetches_only_overlap_window_and_replaces_revised_values(tmp_path):
    path = tmp_path / "s.parquet"
    asyncio.run(cache.update_daily(path, FakeFetch(1), default_start="2020-01-01", today="2020-01-10"))
    fetch = FakeFetch(2)

    df = asyncio.run(cache.update_daily(path, fetch, default_start="2020-01-01", today="2020-01-15", overlap_days=3))

    # last cached date 01-10, re-fetch from 01-07 to pick up provisional revisions
    assert fetch.calls == [("2020-01-07", "2020-01-15")]
    assert df.index.min() == pd.Timestamp("2020-01-01")
    assert df.index.max() == pd.Timestamp("2020-01-15")
    assert df.loc["2020-01-06", "x"] == 1.0
    assert df.loc["2020-01-07", "x"] == 2.0
    assert df.index.is_unique


def test_refresh_refetches_everything(tmp_path):
    path = tmp_path / "s.parquet"
    asyncio.run(cache.update_daily(path, FakeFetch(1), default_start="2020-01-01", today="2020-01-10"))
    fetch = FakeFetch(3)

    df = asyncio.run(cache.update_daily(path, fetch, default_start="2020-01-01", today="2020-01-10", refresh=True))

    assert fetch.calls == [("2020-01-01", "2020-01-10")]
    assert (df["x"] == 3.0).all()


def test_new_columns_from_fetch_are_kept(tmp_path):
    path = tmp_path / "s.parquet"
    asyncio.run(cache.update_daily(path, FakeFetch(1), default_start="2020-01-01", today="2020-01-05"))

    async def fetch(start, end):
        d = daily(start, end, 5)
        d["y"] = 9.0
        return d

    df = asyncio.run(cache.update_daily(path, fetch, default_start="2020-01-01", today="2020-01-06", overlap_days=1))

    assert list(df.columns) == ["x", "y"]
    assert pd.isna(df.loc["2020-01-01", "y"])
    assert df.loc["2020-01-06", "y"] == 9.0


def test_cached_or_compute_only_computes_once(tmp_path):
    path = tmp_path / "thing.json"
    calls = []

    async def compute():
        calls.append(1)
        return {"a": 1}

    assert asyncio.run(cache.cached_json(path, compute)) == {"a": 1}
    assert asyncio.run(cache.cached_json(path, compute)) == {"a": 1}
    assert len(calls) == 1


def test_empty_incremental_fetch_keeps_cache(tmp_path):
    path = tmp_path / "s.parquet"
    asyncio.run(cache.update_daily(path, FakeFetch(1), default_start="2020-01-01", today="2020-01-10"))

    async def empty(start, end):
        return pd.DataFrame({"x": []}, index=pd.DatetimeIndex([], name="date"))

    df = asyncio.run(cache.update_daily(path, empty, default_start="2020-01-01", today="2020-01-11"))

    assert len(df) == 10


def test_revision_pass_runs_at_most_every_interval_and_only_new_days_in_between(tmp_path):
    path = tmp_path / "s.parquet"
    every = pd.Timedelta(hours=20)
    t0 = pd.Timestamp("2020-01-10T13:00Z")
    fetch = FakeFetch(1)

    asyncio.run(cache.update_daily(path, fetch, "2020-01-01", today="2020-01-08", revise_every=every, now=t0))
    # 3 h later: no revision due, so only days after the last cached one
    asyncio.run(cache.update_daily(path, fetch, "2020-01-01", today="2020-01-10", revise_every=every, now=t0 + pd.Timedelta(hours=3)))
    # yesterday (01-09) is cached now: nothing new can be published yet, so no request at all
    asyncio.run(cache.update_daily(path, fetch, "2020-01-01", today="2020-01-10", revise_every=every, now=t0 + pd.Timedelta(hours=6)))
    # 21 h after the last pass: the 10-day revision window again
    df = asyncio.run(cache.update_daily(path, fetch, "2020-01-01", today="2020-01-11", revise_every=every, now=t0 + pd.Timedelta(hours=21)))

    assert fetch.calls == [("2020-01-01", "2020-01-08"), ("2020-01-09", "2020-01-10"), ("2019-12-31", "2020-01-11")]
    assert df.index.is_unique and df.index.max() == pd.Timestamp("2020-01-11")


def test_without_revise_every_every_call_refetches_the_overlap(tmp_path):
    path = tmp_path / "s.parquet"
    fetch = FakeFetch(1)
    asyncio.run(cache.update_daily(path, fetch, "2020-01-01", today="2020-01-10"))
    asyncio.run(cache.update_daily(path, fetch, "2020-01-01", today="2020-01-10", overlap_days=2))

    assert fetch.calls[-1] == ("2020-01-08", "2020-01-10")
    assert not (tmp_path / "s.parquet.meta.json").exists()
