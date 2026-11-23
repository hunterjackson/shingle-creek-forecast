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
