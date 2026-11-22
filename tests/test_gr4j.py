import numpy as np
import pytest

from shingle.models import gr4j


def test_unit_hydrograph_ordinates_sum_to_one():
    for x4 in (0.6, 1.0, 2.3, 5.0):
        uh1, uh2 = gr4j.unit_hydrographs(x4)
        assert uh1.sum() == pytest.approx(1.0)
        assert uh2.sum() == pytest.approx(1.0)
        assert len(uh2) >= len(uh1)
        assert (uh1 >= 0).all() and (uh2 >= 0).all()


def test_water_balance_closes_without_exchange():
    rng = np.random.default_rng(3)
    n = 1500
    P = rng.gamma(0.4, 10, n)
    E = np.full(n, 3.5)
    params = gr4j.Params(x1=350.0, x2=0.0, x3=90.0, x4=1.7)

    out = gr4j.run(P, E, params)

    # inputs - outputs == change in storage (production + routing + UH queues)
    lhs = P.sum() - out["aet"].sum() - out["q"].sum()
    rhs = out["storage_total"][-1] - out["storage_total_initial"]
    assert lhs == pytest.approx(rhs, abs=1e-6)


def test_no_rain_gives_monotone_nonnegative_recession():
    n = 200
    params = gr4j.Params(x1=300.0, x2=0.0, x3=80.0, x4=1.5)
    state = gr4j.State.initial(params, s_frac=0.7, r_frac=0.6)

    out = gr4j.run(np.zeros(n), np.zeros(n), params, state=state)

    q = out["q"]
    assert (q >= 0).all()
    assert (np.diff(q[5:]) <= 1e-12).all()


def test_larger_x4_delays_peak():
    P = np.zeros(40)
    P[5] = 80.0
    E = np.zeros(40)
    fast = gr4j.run(P, E, gr4j.Params(300, 0, 80, 1.0))["q"]
    slow = gr4j.run(P, E, gr4j.Params(300, 0, 80, 4.0))["q"]

    assert np.argmax(slow) > np.argmax(fast)


def test_returns_internal_states_for_hybrid_features():
    out = gr4j.run(np.ones(10), np.ones(10), gr4j.Params(300, 0, 80, 1.5))

    for key in ("s", "r", "q"):
        assert out[key].shape == (10,)


def test_final_state_allows_continuation():
    rng = np.random.default_rng(0)
    P, E = rng.gamma(0.4, 10, 100), np.full(100, 3.0)
    p = gr4j.Params(320, -0.5, 70, 2.1)

    full = gr4j.run(P, E, p)["q"]
    first = gr4j.run(P[:60], E[:60], p)
    second = gr4j.run(P[60:], E[60:], p, state=first["state"])["q"]

    np.testing.assert_allclose(np.concatenate([first["q"], second]), full)


def test_cfs_mm_conversion_round_trip():
    area_km2 = 466.2
    mm = gr4j.cfs_to_mm_day(np.array([100.0]), area_km2)

    assert mm[0] == pytest.approx(100 * 0.0283168466 * 86400 / (area_km2 * 1e6) * 1000)
    assert gr4j.mm_day_to_cfs(mm, area_km2)[0] == pytest.approx(100.0)


def test_nse_and_kge_metrics():
    obs = np.array([1.0, 2.0, 3.0, 4.0])

    assert gr4j.nse(obs, obs) == pytest.approx(1.0)
    assert gr4j.nse(np.full(4, obs.mean()), obs) == pytest.approx(0.0)
    assert gr4j.kge(obs, obs) == pytest.approx(1.0)


def test_calibrate_recovers_synthetic_flows():
    rng = np.random.default_rng(11)
    n = 3 * 365
    P = rng.gamma(0.35, 12, n)
    E = 3.5 + 1.5 * np.sin(np.arange(n) * 2 * np.pi / 365)
    true = gr4j.Params(x1=420.0, x2=-0.8, x3=110.0, x4=2.2)
    q_obs = gr4j.run(P, E, true)["q"]

    fit = gr4j.calibrate(P, E, q_obs, warmup=365, seed=0)

    sim = gr4j.run(P, E, fit.params)["q"][365:]
    assert gr4j.nse(sim, q_obs[365:]) > 0.98
    assert fit.score > 0.98


def test_unit_conversions_preserve_pandas_index():
    import pandas as pd

    s = pd.Series([100.0, 200.0], index=pd.date_range("2020-01-01", periods=2))

    out = gr4j.cfs_to_mm_day(s, 466.2)

    assert isinstance(out, pd.Series)
    assert out.index.equals(s.index)
    assert isinstance(gr4j.mm_day_to_cfs(out, 466.2), pd.Series)


def test_forecast_with_perfect_inputs_reproduces_simulation():
    rng = np.random.default_rng(3)
    n, K = 400, 7
    P = rng.gamma(0.4, 8, n)
    E = np.full(n, 3.5)
    p = gr4j.Params(x1=300.0, x2=-0.5, x3=80.0, x4=2.3)
    sim = gr4j.run(P, E, p)["q"]
    # forecast issued after day t, perfect knowledge of the next K days
    Pf = np.full((n, K), np.nan)
    Ef = np.full((n, K), 3.5)
    for t in range(n - K):
        Pf[t] = P[t + 1 : t + 1 + K]

    qf = gr4j.forecast(P, E, Pf, Ef, p)

    assert qf.shape == (n, K)
    for t in (10, 150, 300):
        np.testing.assert_allclose(qf[t], sim[t + 1 : t + 1 + K], rtol=1e-10)
    assert np.isnan(qf[n - 1]).all()  # NaN forecast input -> NaN output


def test_forecast_with_no_rain_recedes_from_current_state():
    rng = np.random.default_rng(4)
    n, K = 200, 5
    P = rng.gamma(0.4, 8, n)
    E = np.full(n, 3.5)
    p = gr4j.Params(x1=300.0, x2=0.0, x3=80.0, x4=1.5)

    qf = gr4j.forecast(P, E, np.zeros((n, K)), np.zeros((n, K)), p)

    assert (np.diff(qf[100]) <= 1e-12).all() or qf[100, -1] < qf[100].max()
    assert (qf >= 0).all()
