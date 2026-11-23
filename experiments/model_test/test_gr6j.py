import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import gr6j


def _inputs(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.gamma(0.4, 8.0, n) * (rng.random(n) < 0.35)
    E = 3 + 1.5 * np.sin(np.arange(n) * 2 * np.pi / 365)
    return P, E


def test_water_balance_closes():
    P, E = _inputs()
    out = gr6j.run(P, E, [300, -1.5, 200, 2.5, 0.3, 20])
    lhs = P.sum() - out["aet"].sum() - out["q"].sum() + out["exchange"].sum()
    assert abs(lhs - (out["storage1"] - out["storage0"])) < 1e-6 * P.sum()


def test_no_exchange_conserves_and_flows_nonnegative():
    P, E = _inputs(seed=1)
    out = gr6j.run(P, E, [300, 0.0, 200, 2.5, 0.0, 20])
    assert np.all(out["q"] >= 0)
    assert abs(out["exchange"].sum()) < 1e-9
    assert abs(P.sum() - out["aet"].sum() - out["q"].sum() - (out["storage1"] - out["storage0"])) < 1e-6 * P.sum()


def test_exponential_store_slows_recession():
    # after rain stops, a larger x6 keeps more slow flow alive 60 days later
    P = np.r_[np.full(30, 20.0), np.zeros(120)]
    E = np.full(150, 1.0)
    q_small = gr6j.run(P, E, [300, 0.0, 200, 2.0, 0.0, 2.0])["q"]
    q_large = gr6j.run(P, E, [300, 0.0, 200, 2.0, 0.0, 60.0])["q"]
    assert q_large[110] > q_small[110]
