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


def test_floodplain_flattens_channel_control_above_break():
    plain = rating.CombinedRating(c=1e-5, d=1.0, h0=50.0, a=0.3, b=0.45, k=8.0)
    fp = rating.CombinedRating(c=1e-5, d=1.0, h0=50.0, a=0.3, b=0.45, k=8.0, qb=2000.0, g=0.3)
    lake = np.array([52.0])

    # well below the break: identical
    assert fp.predict_stage(np.array([200.0]), lake)[0] == pytest.approx(plain.predict_stage(np.array([200.0]), lake)[0], abs=0.02)
    # far above: rises, but much less steeply than the unbroken power law
    s = [fp.predict_stage(np.array([q]), lake)[0] for q in (4000.0, 8000.0)]
    p = [plain.predict_stage(np.array([q]), lake)[0] for q in (4000.0, 8000.0)]
    assert s[1] > s[0]
    assert (s[1] - s[0]) < 0.6 * (p[1] - p[0])
    # inverse still consistent
    st = fp.predict_stage(np.array([5000.0]), lake)
    assert fp.predict_flow(st, lake)[0] == pytest.approx(5000.0, rel=0.02)


def test_fit_combined_floodplain_learns_flattening_and_bounds_lake_exponent(rng):
    n = 5000
    q = np.exp(rng.uniform(np.log(5), np.log(4000), n))
    lake = rng.uniform(51.5, 56.0, n)
    truth = rating.CombinedRating(c=2e-4, d=1.0, h0=50.0, a=0.3, b=0.45, k=8.0, qb=1500.0, g=0.3)
    stage = truth.predict_stage(q, lake) + rng.normal(0, 0.03, n)

    plain = rating.fit_combined(q, stage, lake)
    fp = rating.fit_combined(q, stage, lake, floodplain=True, d_max=1.2, g_min=0.3)

    top = q > 2500
    err_plain = np.abs(plain.predict_stage(q[top], lake[top]) - stage[top]).mean()
    err_fp = np.abs(fp.predict_stage(q[top], lake[top]) - stage[top]).mean()
    assert err_fp < 0.5 * err_plain
    assert fp.d <= 1.2 + 1e-9 and fp.g >= 0.3 - 1e-9
    # extrapolation stays monotone and gentle
    ext = fp.predict_stage(np.array([4000.0, 8000.0, 16000.0]), np.full(3, 56.0))
    assert np.all(np.diff(ext) > 0) and ext[-1] - ext[0] < 5.0


def test_floodplain_rating_round_trip_dict():
    r = rating.CombinedRating(c=0.001, d=1.0, h0=50.0, a=0.3, b=0.45, k=8.0, qb=2500.0, g=0.4)

    assert rating.from_dict(r.to_dict()) == r
