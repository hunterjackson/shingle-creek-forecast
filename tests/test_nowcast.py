import numpy as np
import pandas as pd
import pytest

from shingle import nowcast


def hourly_world(n_days=900, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2020-01-01", periods=n_days * 24, freq="1h", tz="UTC")
    s = pd.Series(55 + np.cumsum(rng.normal(0, 0.01, len(t))), index=t)
    est_day = (t - pd.Timedelta(hours=5)).tz_localize(None).normalize()
    daily = s.groupby(est_day).mean()
    daily.index.name = "date"
    return s, daily


def test_hours_table_labels_target_day_hours_by_issue_day():
    s, _ = hourly_world(5)
    G = nowcast.hours_table(s)

    assert G.loc["2020-01-01", 5] == s[pd.Timestamp("2020-01-02 05:00", tz="UTC")]
    assert G.loc["2020-01-01", 28] == s[pd.Timestamp("2020-01-03 04:00", tz="UTC")]


def test_fitted_nowcast_beats_the_frozen_forecast_and_shrinks_the_band_later_in_the_day():
    s, daily = hourly_world()
    idx = daily.index[:-2]
    y = daily.shift(-1).loc[idx]
    rng = np.random.default_rng(1)
    p50 = y + rng.normal(0, 0.05, len(idx))  # a frozen forecast with error

    nc = nowcast.fit(p50, s, daily)
    G = nowcast.hours_table(s).reindex(idx)
    err = {}
    for n in (11, 17, 23):
        est = pd.Series([nowcast.point(nc, p50[d], G.loc[d, list(range(5, 5 + n))].to_numpy()) for d in idx], index=idx)
        err[n] = (est - y).abs().mean()
    assert err[23] < err[17] < err[11] < (p50 - y).abs().mean()
    assert nc["shrink"][nc["n"].index(23)] < nc["shrink"][nc["n"].index(11)] <= 1.0


def test_apply_updates_day_one_and_keeps_band_around_the_new_value():
    nc = {"n": [9, 23], "a": [1.0, 0.0], "shrink": [1.0, 0.1]}
    seen = np.full(23, 56.0)

    p10, p50, p90 = nowcast.apply(nc, 55.0, 57.0, 59.0, seen)

    assert p50 == pytest.approx(56.0)
    assert (p10, p90) == (pytest.approx(56.0 - 0.2), pytest.approx(56.0 + 0.2))
    assert nowcast.apply(nc, 55.0, 57.0, 59.0, seen[:8]) is None  # nothing new since issue


def test_seen_hours_takes_complete_hours_of_the_target_day():
    t = pd.date_range("2024-10-10 00:00", "2024-10-10 16:00", freq="1h", tz="UTC")
    s = pd.Series(np.arange(len(t), dtype=float), index=t)

    v = nowcast.seen_hours(s, pd.Timestamp("2024-10-09"), now=pd.Timestamp("2024-10-10 16:35", tz="UTC"))

    assert list(v) == [float(h) for h in range(5, 16)]  # 05Z..15Z; the 16Z hour is not complete
