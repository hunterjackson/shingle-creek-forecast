import pandas as pd
import pytest

from home_display.shingle import cache


def daily(start, end, value):
    idx = pd.date_range(start, end, name="date")
    return pd.DataFrame({"x": float(value)}, index=idx)


class FakeFetch:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def __call__(self, start, end):
        self.calls.append((start, end))
        return daily(start, end, self.value)


def test_first_call_fetches_full_range_and_writes_parquet(tmp_path):
    path = tmp_path / "s.parquet"
    fetch = FakeFetch(1)

    df = cache.update_daily(path, fetch, default_start="2020-01-01", today="2020-01-10")

    assert fetch.calls == [("2020-01-01", "2020-01-10")]
    assert path.exists()
    assert len(df) == 10
    pd.testing.assert_frame_equal(pd.read_parquet(path), df, check_freq=False)


def test_second_call_fetches_only_overlap_window_and_replaces_revised_values(tmp_path):
    path = tmp_path / "s.parquet"
    cache.update_daily(path, FakeFetch(1), default_start="2020-01-01", today="2020-01-10")
    fetch = FakeFetch(2)

    df = cache.update_daily(path, fetch, default_start="2020-01-01", today="2020-01-15", overlap_days=3)

    # last cached date 01-10, re-fetch from 01-07 to pick up provisional revisions
    assert fetch.calls == [("2020-01-07", "2020-01-15")]
    assert df.index.min() == pd.Timestamp("2020-01-01")
    assert df.index.max() == pd.Timestamp("2020-01-15")
    assert df.loc["2020-01-06", "x"] == 1.0
    assert df.loc["2020-01-07", "x"] == 2.0
    assert df.index.is_unique


def test_refresh_refetches_everything(tmp_path):
    path = tmp_path / "s.parquet"
    cache.update_daily(path, FakeFetch(1), default_start="2020-01-01", today="2020-01-10")
    fetch = FakeFetch(3)

    df = cache.update_daily(path, fetch, default_start="2020-01-01", today="2020-01-10", refresh=True)

    assert fetch.calls == [("2020-01-01", "2020-01-10")]
    assert (df["x"] == 3.0).all()


def test_new_columns_from_fetch_are_kept(tmp_path):
    path = tmp_path / "s.parquet"
    cache.update_daily(path, FakeFetch(1), default_start="2020-01-01", today="2020-01-05")

    def fetch(start, end):
        d = daily(start, end, 5)
        d["y"] = 9.0
        return d

    df = cache.update_daily(path, fetch, default_start="2020-01-01", today="2020-01-06", overlap_days=1)

    assert list(df.columns) == ["x", "y"]
    assert pd.isna(df.loc["2020-01-01", "y"])
    assert df.loc["2020-01-06", "y"] == 9.0


def test_cached_or_compute_only_computes_once(tmp_path):
    path = tmp_path / "thing.json"
    calls = []

    def compute():
        calls.append(1)
        return {"a": 1}

    assert cache.cached_json(path, compute) == {"a": 1}
    assert cache.cached_json(path, compute) == {"a": 1}
    assert len(calls) == 1


def test_empty_incremental_fetch_keeps_cache(tmp_path):
    path = tmp_path / "s.parquet"
    cache.update_daily(path, FakeFetch(1), default_start="2020-01-01", today="2020-01-10")

    def empty(start, end):
        return pd.DataFrame({"x": []}, index=pd.DatetimeIndex([], name="date"))

    df = cache.update_daily(path, empty, default_start="2020-01-01", today="2020-01-11")

    assert len(df) == 10
