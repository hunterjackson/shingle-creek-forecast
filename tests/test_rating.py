import numpy as np
import pandas as pd
import pytest

from home_display.shingle import rating


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def test_free_rating_recovers_power_law(rng):
    q = rng.uniform(5, 3000, 2000)
    stage = 51.5 + 0.08 * q**0.55 + rng.normal(0, 0.02, q.size)

    r = rating.fit_free(q, stage)

    assert r.predict_stage(np.array([1000.0]))[0] == pytest.approx(51.5 + 0.08 * 1000**0.55, abs=0.05)
    assert r.predict_flow(r.predict_stage(np.array([500.0])))[0] == pytest.approx(500.0, rel=0.02)


def test_backwater_rating_uses_lake_level(rng):
    n = 3000
    q = rng.uniform(1, 3000, n)
    lake = rng.uniform(51.5, 56.0, n)
    stage = lake + 0.01 * q**0.8 + rng.normal(0, 0.02, n)

    r = rating.fit_backwater(q, stage, lake)

    pred = r.predict_stage(np.array([100.0, 100.0]), lake=np.array([52.0, 55.0]))
    assert pred[1] - pred[0] == pytest.approx(3.0, abs=0.05)  # same flow, lake 3 ft higher
    back = r.predict_flow(np.array([56.0]), lake=np.array([54.0]))
    assert back[0] == pytest.approx((2.0 / 0.01) ** (1 / 0.8), rel=0.05)


def test_backwater_flow_is_zero_when_stage_below_lake():
    r = rating.BackwaterRating(c=0.01, d=0.8)

    assert r.predict_flow(np.array([52.0]), lake=np.array([53.0]))[0] == 0.0


def test_fit_ignores_nans(rng):
    q = rng.uniform(5, 3000, 500)
    stage = 51.5 + 0.08 * q**0.55
    stage[:10] = np.nan

    r = rating.fit_free(q, stage)

    assert np.isfinite(r.predict_stage(np.array([100.0]))).all()


def test_rating_round_trips_through_dict():
    r = rating.FreeRating(h0=51.0, a=0.1, b=0.5)

    r2 = rating.from_dict(r.to_dict())

    assert r2 == r


def test_combined_rating_follows_lake_at_low_flow_and_channel_at_high_flow(rng):
    n = 4000
    q = np.exp(rng.uniform(np.log(5), np.log(3500), n))
    lake = rng.uniform(51.5, 56.0, n)
    lake_ctrl = lake + 0.001 * q
    chan_ctrl = 50.0 + 0.3 * q**0.45
    stage = np.maximum(lake_ctrl, chan_ctrl) + rng.normal(0, 0.03, n)

    r = rating.fit_combined(q, stage, lake)

    # low flow: tracks the lake one-for-one
    lo = r.predict_stage(np.array([20.0, 20.0]), lake=np.array([52.0, 55.0]))
    assert lo[1] - lo[0] == pytest.approx(3.0, abs=0.15)
    # high flow, low lake: channel control
    hi = r.predict_stage(np.array([3000.0]), lake=np.array([52.0]))[0]
    assert hi == pytest.approx(50.0 + 0.3 * 3000**0.45, abs=0.3)
    # inverse is consistent
    s = r.predict_stage(np.array([800.0]), lake=np.array([54.0]))
    assert r.predict_flow(s, lake=np.array([54.0]))[0] == pytest.approx(800.0, rel=0.05)


def test_combined_rating_round_trip_dict():
    r = rating.CombinedRating(c=0.001, d=1.0, h0=50.0, a=0.3, b=0.45, k=8.0)

    assert rating.from_dict(r.to_dict()) == r
