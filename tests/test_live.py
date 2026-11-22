import numpy as np
import pandas as pd
import pytest

from home_display.shingle import live


def test_daily_from_iv_uses_local_standard_time_days_and_requires_coverage():
    # EST = UTC-5: local day 2024-01-01 is 05Z Jan 1 .. 04Z Jan 2
    t = pd.date_range("2024-01-01 05:00", "2024-01-03 04:00", freq="1h", tz="UTC")
    hourly = pd.DataFrame({"stage_ft": np.r_[np.full(24, 54.0), np.full(24, 56.0)], "discharge_cfs": 100.0}, index=t)
    hourly.iloc[30:40] = np.nan  # day 2 has only 14 valid hours

    d = live.daily_from_iv(hourly, min_hours=20)

    assert d.loc["2024-01-01", "stage_ft"] == 54.0
    assert np.isnan(d.loc["2024-01-02", "stage_ft"])
    assert d.index.name == "date"


def test_fill_rain_extends_gridmet_with_scaled_radar_after_last_day():
    idx = pd.date_range("2024-01-01", periods=5, name="date")
    gm = pd.DataFrame({"c": [1.0, 2.0, 3.0, np.nan, np.nan]}, index=idx)
    radar = pd.DataFrame({"c": [9.0, 9.0, 9.0, 4.0, 6.0]}, index=idx)

    out = live.fill_rain(gm, radar, scale=1.5)

    assert out["c"].tolist() == [1.0, 2.0, 3.0, 6.0, 9.0]
    assert out.attrs["filled_from"] == pd.Timestamp("2024-01-04")


def test_fill_rain_without_radar_leaves_nan():
    idx = pd.date_range("2024-01-01", periods=3, name="date")
    gm = pd.DataFrame({"c": [1.0, np.nan, np.nan]}, index=idx)

    out = live.fill_rain(gm, pd.DataFrame({"c": [np.nan] * 3}, index=idx), scale=1.0)

    assert np.isnan(out["c"].iloc[1])


def test_pet_climatology_and_fill():
    idx = pd.date_range("2020-01-01", "2022-12-31", name="date")
    doy = idx.day_of_year.to_numpy()
    pet = pd.DataFrame({"c": 3.0 + np.sin(2 * np.pi * doy / 365.25)}, index=idx)
    clim = live.pet_climatology(pet)

    later = pd.date_range("2023-01-01", periods=10, name="date")
    full = pd.concat([pet, pd.DataFrame({"c": np.nan}, index=later)])
    out = live.fill_pet(full, clim)

    assert out.loc["2023-01-05", "c"] == pytest.approx(pet.loc["2022-01-05", "c"], abs=0.05)
    assert out.loc[:"2022-12-31"].equals(pet)


def test_ffill_recent_limits_staleness():
    idx = pd.date_range("2024-01-01", periods=6, name="date")
    s = pd.Series([1.0, 2.0, np.nan, np.nan, np.nan, np.nan], index=idx)

    out = live.ffill_recent(s, limit=2)

    assert out.tolist()[:4] == [1.0, 2.0, 2.0, 2.0]
    assert np.isnan(out.iloc[4])
