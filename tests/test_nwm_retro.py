import numpy as np
import pandas as pd
import pytest
import xarray as xr

from home_display.shingle.sources import nwm_retro


@pytest.fixture
def store(tmp_path):
    time = pd.date_range("2020-01-01T00:00", periods=72, freq="h")
    feature_id = np.array([111, 21476600, 999])
    flow = np.zeros((72, 3))
    flow[:, 1] = 10.0  # m3/s on our reach
    flow[29:53, 1] = 20.0  # 2020-01-02 05:00Z .. 2020-01-03 04:00Z == local 01-02
    ds = xr.Dataset(
        {"streamflow": (("time", "feature_id"), flow)},
        coords={"time": time, "feature_id": feature_id},
    )
    path = tmp_path / "chrtout.zarr"
    ds.to_zarr(path, zarr_format=2, consolidated=True)
    return str(path)


def test_read_reach_daily_converts_m3s_to_cfs_and_local_days(store):
    s = nwm_retro.read_reach_daily(21476600, "2020-01-01", "2020-01-03", store=store)

    assert s.index.name == "date"
    assert s.name == "nwm_retro_cfs"
    assert s.loc["2020-01-02"] == pytest.approx(20.0 * nwm_retro.M3S_TO_CFS)
    assert s.loc["2020-01-01"] == pytest.approx(10.0 * nwm_retro.M3S_TO_CFS)
    # 00Z-04Z Jan 1 belong to local 2019-12-31, outside the requested range
    assert s.index.min() == pd.Timestamp("2020-01-01")


def test_read_reach_daily_drops_partial_days_when_requested(store):
    s = nwm_retro.read_reach_daily(21476600, "2020-01-01", "2020-01-03", store=store, min_hours=24)

    # local 01-03 only has 05Z..23Z (19 h) of data
    assert list(s.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]


def test_unknown_feature_raises(store):
    with pytest.raises(KeyError):
        nwm_retro.read_reach_daily(42, "2020-01-01", "2020-01-02", store=store)
