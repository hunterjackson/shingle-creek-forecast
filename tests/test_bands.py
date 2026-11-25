import numpy as np
import pandas as pd

from shingle import bands


def _world(n, rng):
    adp = np.where(rng.random(n) < 0.4, rng.uniform(0, 0.049, n), rng.exponential(0.5, n) + 0.05)
    s = 0.02 + adp * rng.uniform(0.2, 2.0, n)  # learned scale: informative beyond |dp50|
    calm = adp < 0.05
    resid = rng.normal(0, np.where(calm, 0.02, 0.05 + s))
    return pd.Series(resid), pd.Series(adp), pd.Series(s)


def test_calm_split_bands_cover_each_stratum():
    rng = np.random.default_rng(0)
    r, adp, s = _world(20000, rng)

    b = bands.CalmSplitBands.fit(r, adp, s, coverage=0.8)

    r2, adp2, s2 = _world(20000, rng)
    lo, hi = b.offsets(adp2.to_numpy(), s2.to_numpy())
    inside = (r2 >= lo) & (r2 <= hi)
    calm = adp2 < bands.CALM
    assert 0.76 < inside[calm].mean() < 0.84
    assert 0.76 < inside[~calm].mean() < 0.84
    big = s2 > s2.quantile(0.95)
    assert 0.74 < inside[big].mean() < 0.86


def test_calm_split_bands_calm_days_ignore_learned_scale():
    rng = np.random.default_rng(1)
    r, adp, s = _world(5000, rng)
    b = bands.CalmSplitBands.fit(r, adp, s)

    lo1, hi1 = b.offsets(np.array([0.01]), np.array([0.0]))
    lo2, hi2 = b.offsets(np.array([0.01]), np.array([5.0]))
    lo3, hi3 = b.offsets(np.array([0.5]), np.array([5.0]))

    assert lo1 == lo2 and hi1 == hi2
    assert hi3 - lo3 > 3 * (hi1 - lo1)


def _wet_world(n, rng):
    wet = rng.random(n) < 0.2
    s = rng.exponential(0.1, n) + 0.02
    resid = rng.normal(0, np.where(wet, 0.05 + 2 * s, 0.02))
    return resid, wet, s


def test_wet_band_keeps_base_band_on_dry_days_and_covers_wet_days():
    """T2 (docs/divergence_test_plan.md): days with forecast rain get their own conformal bins on the learned
    scale; dry days keep whatever band production already gives them."""
    rng = np.random.default_rng(2)
    r, wet, s = _wet_world(20000, rng)
    wb = bands.WetBand.fit(r[wet], s[wet], coverage=0.8)

    r2, wet2, s2 = _wet_world(20000, rng)
    base_lo, base_hi = np.full(len(r2), -0.026), np.full(len(r2), 0.026)  # a narrow, dry-day band
    lo, hi = wb.combine(base_lo, base_hi, wet2, s2)

    assert np.array_equal(lo[~wet2], base_lo[~wet2]) and np.array_equal(hi[~wet2], base_hi[~wet2])
    inside = (r2 >= lo) & (r2 <= hi)
    assert 0.76 < inside[wet2].mean() < 0.84
    assert (hi - lo)[wet2].mean() > 3 * 0.052


def _rain_data():
    idx = pd.date_range("2024-01-01", periods=6, name="date")
    pr = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5.0], "b": [0, 3, 6, 9, 12, 15.0]}, index=idx)
    return {"stage": pd.Series(55.0, index=idx), "pr": pr}, idx


def test_forecast_rain_sum_uses_issued_forecast_leads_1_to_h():
    data, idx = _rain_data()
    cols = pd.MultiIndex.from_tuples([(k, c) for k in (1, 2, 3) for c in ("a", "b")], names=["lead", "cell"])
    fpr = pd.DataFrame(0.0, index=idx, columns=cols)
    fpr.loc[idx[0], (1, "a")], fpr.loc[idx[0], (1, "b")] = 4.0, 8.0     # areal 6 with weights 1:1
    fpr.loc[idx[0], (2, "a")], fpr.loc[idx[0], (2, "b")] = 10.0, 10.0   # areal 10
    data["fcst"] = {"pr": fpr, "by_h": {2: {"pr": fpr * 100}}}  # by_h (scaled) must be ignored

    got = bands.forecast_rain_sum(data, idx[:1], h=2, weights={"a": 1.0, "b": 1.0})

    assert got.iloc[0] == 16.0


def test_forecast_rain_sum_falls_back_to_observed_future_rain():
    data, idx = _rain_data()
    got = bands.forecast_rain_sum(data, idx[:2], h=2, weights={"a": 1.0, "b": 1.0})
    # areal rain = mean(a, b) = 0, 2, 4, 6, ... ; issue day 0 -> days 1..2 = 2 + 4
    assert got.tolist() == [6.0, 10.0]
