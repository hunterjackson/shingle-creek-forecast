import numpy as np
import pandas as pd
import pytest

from shingle.models import base, baseline


@pytest.fixture
def stage():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-01", periods=400, name="date")
    return pd.Series(54 + np.cumsum(rng.normal(0, 0.1, 400)), index=idx)


def test_forecast_frame_columns():
    cols = base.forecast_columns(horizons=(1, 2))

    assert cols == ["h1_p10", "h1_p50", "h1_p90", "h2_p10", "h2_p50", "h2_p90"]


def test_targets_frame_shifts_stage_by_horizon(stage):
    y = base.targets(stage, horizons=(1, 3))

    assert list(y.columns) == ["h1", "h3"]
    assert y["h3"].iloc[0] == stage.iloc[3]
    assert y["h1"].iloc[-1:].isna().all()


def test_persistence_p50_is_today_and_intervals_are_calibrated(stage):
    m = baseline.Persistence(horizons=(1, 7))
    train = stage.index[:300]

    m.fit({"stage": stage}, train)
    fc = m.predict({"stage": stage}, stage.index[300:390])

    assert list(fc.columns) == base.forecast_columns((1, 7))
    assert (fc["h1_p50"] == stage.iloc[300:390]).all()
    assert (fc["h7_p90"] - fc["h7_p10"]).mean() > (fc["h1_p90"] - fc["h1_p10"]).mean()
    # empirical coverage on held-out data ~80 %
    y = base.targets(stage, (1, 7)).loc[fc.index]
    inside = ((y["h1"] >= fc["h1_p10"]) & (y["h1"] <= fc["h1_p90"])).mean()
    assert 0.65 < inside < 0.95


def test_persistence_calibrate_on_other_index_changes_offsets(stage):
    m = baseline.Persistence(horizons=(1,)).fit({"stage": stage}, stage.index[:100])
    before = m.offsets[1]

    m.calibrate({"stage": stage * 3 - 108}, stage.index[100:300])  # 3x more volatile

    assert m.offsets[1][1] > before[1]
