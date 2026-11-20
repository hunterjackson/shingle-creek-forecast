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
