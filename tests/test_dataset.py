import numpy as np
import pandas as pd

from home_display.shingle import dataset


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
