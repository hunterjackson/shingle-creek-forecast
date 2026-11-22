import numpy as np
import pandas as pd
import pytest

from home_display.shingle import fcst, gr4j_driver
from home_display.shingle.models import gr4j


def leads_frame(idx, cells, value_fn):
    cols = pd.MultiIndex.from_tuples([(k, c) for k in (1, 2, 3) for c in cells], names=["lead", "cell"])
    return pd.DataFrame({col: value_fn(*col) for col in cols}, index=idx)[cols]


@pytest.fixture
def setup():
    n = 400
    idx = pd.date_range("2021-01-01", periods=n, name="date")
    rng = np.random.default_rng(0)
    pr = pd.DataFrame({"g1": rng.gamma(0.4, 8, n), "g2": rng.gamma(0.4, 8, n)}, index=idx)
    data = {"stage": pd.Series(54.0, index=idx), "pr": pr, "pet": pd.DataFrame({"g1": 3.5, "g2": 3.5}, index=idx)}
    src_cells = pd.DataFrame({"lat": [28.0, 28.0, 28.5, 28.5], "lon": [-81.5, -81.0, -81.5, -81.0]},
                             index=["28.0000_-81.5000", "28.0000_-81.0000", "28.5000_-81.5000", "28.5000_-81.0000"])
    tgt_cells = pd.DataFrame({"lat": [28.25, 28.5], "lon": [-81.25, -81.5]}, index=["g1", "g2"])
    raw = leads_frame(idx, list(src_cells.index), lambda k, c: float(k) * (2.0 if c.startswith("28.5000_-81.5") else 1.0))
    drv = gr4j_driver.GR4JDriver({"g1": 0.5, "g2": 0.5})
    drv.params = gr4j.Params(300.0, -0.5, 120.0, 2.0)
    return data, raw, src_cells, tgt_cells, drv


def test_build_maps_forecast_to_model_cells_and_runs_gr4j(setup):
    data, raw, src, tgt, drv = setup

    f = fcst.build(raw, src, tgt, drv, data)

    assert set(f) == {"pr", "gr4j_q"}
    assert f["pr"].columns.names == ["lead", "cell"]
    t = data["stage"].index[50]
    assert f["pr"].loc[t, (2, "g1")] == pytest.approx(2 * (0.75 + 0.25 * 2.0))  # bilinear: 1/4 weight on the x2 point
    assert f["pr"].loc[t, (1, "g2")] == pytest.approx(2.0)  # exactly on the x2 point
    assert list(f["gr4j_q"].columns) == [1, 2, 3]
    assert f["gr4j_q"].loc[t:].notna().all().all()


def test_build_applies_per_lead_scale(setup):
    data, raw, src, tgt, drv = setup

    f = fcst.build(raw, src, tgt, drv, data, scale={1: 2.0, 2: 1.0, 3: 0.5})
    g = fcst.build(raw, src, tgt, drv, data)

    t = data["stage"].index[50]
    assert f["pr"].loc[t, (1, "g1")] == pytest.approx(2 * g["pr"].loc[t, (1, "g1")])
    assert f["pr"].loc[t, (3, "g1")] == pytest.approx(0.5 * g["pr"].loc[t, (3, "g1")])


def test_fit_scale_matches_mean_of_observed_per_lead(setup):
    data, raw, src, tgt, drv = setup
    mapped = fcst.build(raw, src, tgt, drv, data)["pr"]
    obs = data["pr"]

    scale = fcst.fit_scale(mapped, obs, "g1", data["stage"].index[:300])

    for k in (1, 2, 3):
        fc = mapped[(k, "g1")].iloc[:300]
        ob = obs["g1"].shift(-k).iloc[:300]
        ok = fc.notna() & ob.notna()
        assert scale[k] == pytest.approx(ob[ok].sum() / fc[ok].sum())


def test_build_mixed_scales_only_long_horizons(setup):
    data, raw, src, tgt, drv = setup
    scale = {1: 2.0, 2: 2.0, 3: 2.0}

    f = fcst.build_mixed(raw, src, tgt, drv, data, scale=scale, scaled_from_h=2, horizons=(1, 2, 3))
    plain = fcst.build(raw, src, tgt, drv, data)
    scaled = fcst.build(raw, src, tgt, drv, data, scale=scale)

    pd.testing.assert_frame_equal(f["pr"], plain["pr"])
    assert set(f["by_h"]) == {2, 3}
    pd.testing.assert_frame_equal(f["by_h"][3]["pr"], scaled["pr"])
    pd.testing.assert_frame_equal(f["by_h"][2]["gr4j_q"], scaled["gr4j_q"])
