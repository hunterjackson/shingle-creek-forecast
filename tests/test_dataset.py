import numpy as np
import pandas as pd

from shingle import dataset


def test_build_data_maps_history_to_model_inputs():
    idx = pd.date_range("2020-01-01", periods=5, name="date")
    gauge = pd.DataFrame({"discharge_cfs": [1.0, 2, 3, 4, 5], "stage_ft": [54.0, 54, 54, np.nan, 54]}, index=idx)
    up = pd.DataFrame({"discharge_cfs": 0.5, "stage_ft": 4.0}, index=idx[:4])
    lakes = pd.DataFrame({"toho_stage": 53.0, "s61_q": 100.0, "s59_q": 50.0, "etoho_stage": 56.0}, index=idx)
    pr = pd.DataFrame({"c": 1.0}, index=idx)
    history = {"gauge": gauge, "upstream": up, "lakes": lakes, "pr": pr, "pet": pr * 4}

    d = dataset.build_data(history)

    assert set(d) >= {"stage", "q", "q_up", "toho", "s61_q", "s59_q", "etoho", "pr", "pet"}
    for k in ("stage", "q", "q_up", "toho"):
        assert d[k].index.equals(idx)
    assert np.isnan(d["stage"].iloc[3])
    assert np.isnan(d["q_up"].iloc[4])  # reindexed, not filled


def test_build_data_can_swap_rain_source():
    idx = pd.date_range("2020-01-01", periods=3, name="date")
    history = {
        "gauge": pd.DataFrame({"discharge_cfs": 1.0, "stage_ft": 54.0}, index=idx),
        "upstream": pd.DataFrame({"discharge_cfs": 1.0}, index=idx),
        "lakes": pd.DataFrame({"toho_stage": 53.0, "s61_q": 1.0, "s59_q": 1.0, "etoho_stage": 56.0}, index=idx),
        "pr": pd.DataFrame({"g": 1.0}, index=idx),
        "pet": pd.DataFrame({"g": 4.0}, index=idx),
    }
    radar = pd.DataFrame({"r": 2.0}, index=idx[1:])

    d = dataset.build_data(history, rain=radar)

    assert list(d["pr"].columns) == ["r"]
    assert d["pr"].index.equals(idx)


def test_issue_reading_takes_morning_value_of_next_day_labelled_today():
    t = pd.date_range("2020-01-01 00:00", "2020-01-04 23:00", freq="1h", tz="UTC")
    hourly = pd.Series(np.arange(len(t), dtype=float), index=t)
    day = pd.date_range("2020-01-01", periods=4, name="date")

    r = dataset.issue_reading(hourly, day, hour_utc=12)

    assert r.index.equals(day)
    assert r.loc["2020-01-01"] == hourly[pd.Timestamp("2020-01-02 12:00", tz="UTC")]
    assert np.isnan(r.loc["2020-01-04"])  # 2020-01-05 12Z not observed


def test_issue_reading_falls_back_to_latest_hour_in_window():
    t = pd.date_range("2020-01-01 00:00", "2020-01-03 23:00", freq="1h", tz="UTC")
    hourly = pd.Series(1.0, index=t)
    hourly[pd.Timestamp("2020-01-02 12:00", tz="UTC")] = np.nan
    hourly[pd.Timestamp("2020-01-02 11:00", tz="UTC")] = 7.0
    for h in (10, 11, 12):
        hourly[pd.Timestamp(f"2020-01-03 {h:02d}:00", tz="UTC")] = np.nan
    day = pd.date_range("2020-01-01", periods=2, name="date")

    r = dataset.issue_reading(hourly, day, hour_utc=12, window_hours=3)

    assert r.loc["2020-01-01"] == 7.0
    assert np.isnan(r.loc["2020-01-02"])  # whole window missing -> NaN, never older data


def test_build_data_adds_issue_time_readings_when_iv_present():
    idx = pd.date_range("2020-01-01", periods=3, name="date")
    t = pd.date_range("2020-01-01", "2020-01-04 23:00", freq="1h", tz="UTC")
    iv = pd.DataFrame({"discharge_cfs": 200.0, "stage_ft": 55.5}, index=t)
    history = {
        "gauge": pd.DataFrame({"discharge_cfs": 1.0, "stage_ft": 54.0}, index=idx),
        "upstream": pd.DataFrame({"discharge_cfs": 1.0}, index=idx),
        "lakes": pd.DataFrame({"toho_stage": 53.0, "s61_q": 1.0, "s59_q": 1.0, "etoho_stage": 56.0}, index=idx),
        "pr": pd.DataFrame({"g": 1.0}, index=idx),
        "pet": pd.DataFrame({"g": 4.0}, index=idx),
        "iv": {"gauge": iv, "upstream": iv * 0.5},
    }

    d = dataset.build_data(history)

    assert d["stage_am"].tolist() == [55.5] * 3
    assert d["q_am"].tolist() == [200.0] * 3
    assert d["q_up_am"].tolist() == [100.0] * 3


def test_morning_mean_averages_05z_to_12z_of_next_day_labelled_today():
    t = pd.date_range("2020-01-01 00:00", "2020-01-03 23:00", freq="1h", tz="UTC")
    hourly = pd.Series(np.arange(len(t), dtype=float), index=t)
    day = pd.date_range("2020-01-01", periods=3, name="date")

    r = dataset.morning_mean(hourly, day)

    first = hourly[pd.Timestamp("2020-01-02 05:00", tz="UTC"):pd.Timestamp("2020-01-02 12:00", tz="UTC")]
    assert len(first) == 8 and r.loc["2020-01-01"] == first.mean()
    assert np.isnan(r.loc["2020-01-03"])  # 2020-01-04 not observed


def test_build_data_adds_morning_mean_when_iv_present():
    idx = pd.date_range("2020-01-01", periods=3, name="date")
    t = pd.date_range("2020-01-01", "2020-01-04 23:00", freq="1h", tz="UTC")
    iv = pd.DataFrame({"discharge_cfs": 200.0, "stage_ft": 55.5}, index=t)
    history = {
        "gauge": pd.DataFrame({"discharge_cfs": 1.0, "stage_ft": 54.0}, index=idx),
        "upstream": pd.DataFrame({"discharge_cfs": 1.0}, index=idx),
        "lakes": pd.DataFrame({"toho_stage": 53.0, "s61_q": 1.0, "s59_q": 1.0, "etoho_stage": 56.0}, index=idx),
        "pr": pd.DataFrame({"g": 1.0}, index=idx),
        "pet": pd.DataFrame({"g": 4.0}, index=idx),
        "iv": {"gauge": iv, "upstream": iv * 0.5},
    }

    assert dataset.build_data(history)["stage_sofar"].tolist() == [55.5] * 3
