import numpy as np
import pytest

from home_display.shingle.models import gr4j


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
