import numpy as np
import pandas as pd
import pytest

from shingle.models import base, blend


class Const:
    def __init__(self, v, horizons=(1, 3)):
        self.v, self.horizons, self.fitted = v, horizons, False

    def fit(self, data, idx):
        self.fitted = True
        return self

    def point(self, data, idx):
        return pd.DataFrame({f"h{h}": self.v + data["stage"].loc[idx] for h in self.horizons}, index=idx)


@pytest.fixture
def data():
    idx = pd.date_range("2000-01-01", periods=800, name="date")
    rng = np.random.default_rng(0)
    return {"stage": pd.Series(54 + np.cumsum(rng.normal(0, 0.05, 800)), index=idx)}


def test_blend_weights_members_per_horizon_and_fits_both(data):
    a, b = Const(1.0), Const(-1.0)
    m = blend.Blend(a, b, weights={1: 0.5, 3: 0.75}, horizons=(1, 3)).fit(data, data["stage"].index[:600])

    p = m.point(data, data["stage"].index[600:])

    assert a.fitted and b.fitted
    np.testing.assert_allclose(p["h1"], data["stage"].iloc[600:])            # 0.5*(+1) + 0.5*(-1)
    np.testing.assert_allclose(p["h3"], data["stage"].iloc[600:] + 0.5)      # 0.75*(+1) + 0.25*(-1)


def test_blend_predict_gives_ordered_quantile_columns(data):
    m = blend.Blend(Const(0.2), Const(-0.1), weights=0.5, horizons=(1, 3)).fit(data, data["stage"].index[:600])

    fc = m.predict(data, data["stage"].index[600:-3])

    assert list(fc.columns) == base.forecast_columns((1, 3))
    assert (fc["h1_p10"] <= fc["h1_p50"]).all() and (fc["h1_p50"] <= fc["h1_p90"]).all()


class Day1:
    def __init__(self):
        self.fitted = False

    def fit(self, data, idx):
        self.fitted = True
        return self

    def point(self, data, idx):
        v = pd.Series(100.0, index=idx)
        v.iloc[::2] = np.nan  # no morning reading on these days
        return pd.DataFrame({"h1": v})


def test_day1_model_replaces_blend_at_day_one_where_it_predicts(data):
    d1 = Day1()
    m = blend.Blend(Const(1.0), Const(-1.0), weights=0.5, horizons=(1, 3), day1=d1).fit(data, data["stage"].index[:600])
    ix = data["stage"].index[600:]

    p = m.point(data, ix)

    assert d1.fitted
    np.testing.assert_allclose(p["h1"].iloc[1::2], 100.0)
    np.testing.assert_allclose(p["h1"].iloc[::2], data["stage"].loc[ix].iloc[::2])  # falls back to the blend
    np.testing.assert_allclose(p["h3"], data["stage"].loc[ix])


def test_fit_members_fits_all_members_without_calibrating(data):
    a, b, d1 = Const(1.0), Const(-1.0), Day1()
    m = blend.Blend(a, b, horizons=(1, 3), day1=d1)

    m.fit_members(data, data["stage"].index[:600])

    assert a.fitted and b.fitted and d1.fitted and not hasattr(m, "bands")
