import numpy as np
import pandas as pd
import pytest

from shingle import gr4j_driver
from shingle.models import gr4j


@pytest.fixture
def data():
    rng = np.random.default_rng(3)
    n = 4 * 365
    idx = pd.date_range("2000-01-01", periods=n, name="date")
    P = rng.gamma(0.35, 12, n)
    E = 3.5 + 1.5 * np.sin(np.arange(n) * 2 * np.pi / 365)
    true = gr4j.Params(300.0, -0.5, 120.0, 2.0)
    q_mm = gr4j.run(P, E, true)["q"]
    area = 466.2
    return {
        "q": pd.Series(gr4j.mm_day_to_cfs(q_mm, area), index=idx),
        "pr": pd.DataFrame({"a": P, "b": P}, index=idx),
        "pet": pd.DataFrame({"a": E, "b": E}, index=idx),
    }


def test_areal_series_uses_weights():
    idx = pd.date_range("2000-01-01", periods=2)
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=idx)

    s = gr4j_driver.areal(df, {"a": 0.25, "b": 0.75})

    assert s.tolist() == [2.5, 3.5]


def test_fit_and_simulate_adds_gr4j_frame(data):
    idx = data["q"].index
    weights = {"a": 0.5, "b": 0.5}

    drv = gr4j_driver.GR4JDriver(weights, area_km2=466.2, maxiter=60).fit(data, idx[: 3 * 365])
    sim = drv.simulate(data)

    assert list(sim.columns) == ["q", "s", "r"]
    assert sim.index.equals(idx)
    obs = data["q"].iloc[400:].to_numpy()
    assert gr4j.nse(sim["q"].iloc[400:].to_numpy(), obs) > 0.95


def test_simulate_does_not_use_future_for_states(data):
    idx = data["q"].index
    drv = gr4j_driver.GR4JDriver({"a": 1.0}, area_km2=466.2, maxiter=30).fit(data, idx[:700])
    t = 900
    before = drv.simulate(data).iloc[t]
    pert = {k: v.copy() for k, v in data.items()}
    pert["pr"].iloc[t + 1 :] += 50.0

    after = drv.simulate(pert).iloc[t]
    pd.testing.assert_series_equal(before, after)


def test_forecast_restarts_from_state_with_given_areal_rain(data):
    idx = data["q"].index
    drv = gr4j_driver.GR4JDriver({"a": 0.5, "b": 0.5})
    drv.params = gr4j.Params(300.0, -0.5, 120.0, 2.0)
    sim = drv.simulate(data)
    P = gr4j_driver.areal(data["pr"], drv.weights)
    E = gr4j_driver.areal(data["pet"], {"a": 1.0, "b": 1.0})
    K = 3
    P_fc = pd.DataFrame({k: P.shift(-k) for k in range(1, K + 1)}, index=idx)
    E_fc = pd.DataFrame({k: E.shift(-k) for k in range(1, K + 1)}, index=idx)

    fc = drv.forecast(data, P_fc, E_fc)

    assert list(fc.columns) == [1, 2, 3] and fc.index.equals(idx)
    t = idx[500]
    for k in (1, 2, 3):
        assert fc.loc[t, k] == pytest.approx(sim["q"].shift(-k).loc[t], rel=1e-8)
    assert fc.iloc[-1].isna().all()


def test_forecast_default_pet_is_recent_mean(data):
    idx = data["q"].index
    drv = gr4j_driver.GR4JDriver({"a": 0.5, "b": 0.5})
    drv.params = gr4j.Params(300.0, -0.5, 120.0, 2.0)
    P_fc = pd.DataFrame({1: 0.0, 2: 0.0}, index=idx)

    fc = drv.forecast(data, P_fc)  # E_fc=None -> trailing 7-day mean PET persisted

    assert fc.loc[idx[10]:].notna().all().all()
