import numpy as np
import pandas as pd
import pytest

from shingle import screen


def idx(n, start="2000-01-01"):
    return pd.date_range(start, periods=n, name="date")


class Recorder:
    """Dummy model: predicts the training-period mean stage; remembers what it was fit on."""

    horizons = (1, 3)
    fits: list = []

    def fit(self, data, train_index):
        self.train_index = train_index
        self.mean = float(data["stage"].loc[train_index].mean())
        Recorder.fits.append(train_index)
        return self

    def point(self, data, index):
        assert not set(index) & set(self.train_index), "held-out rows leaked into training"
        return pd.DataFrame({f"h{h}": self.mean for h in self.horizons}, index=index)


def test_oof_point_holds_out_each_block_and_covers_index():
    Recorder.fits = []
    n = 6 * 365
    data = {"stage": pd.Series(np.arange(n, dtype=float), index=idx(n))}
    index = data["stage"].index

    pred = screen.oof_point(Recorder, data, index, block_years=2)

    assert list(pred.columns) == ["h1", "h3"]
    assert pred.index.equals(index)
    assert len(Recorder.fits) == 3  # 6 years in 2-year blocks
    # the block holding the start predicts the mean of the *other* years -> above the global mean
    assert pred["h1"].iloc[0] > data["stage"].mean()


def test_oof_point_gap_drops_rows_near_held_block():
    Recorder.fits = []
    n = 4 * 365
    data = {"stage": pd.Series(np.arange(n, dtype=float), index=idx(n))}

    screen.oof_point(Recorder, data, data["stage"].index, block_years=1, gap_days=7)

    for tr in Recorder.fits:
        # no training row within 7 days *before* a held-out block start (its targets would leak)
        years = sorted(set(data["stage"].index.year) - set(tr.year))
        start = pd.Timestamp(f"{years[0]}-01-01")
        assert not ((tr < start) & (tr >= start - pd.Timedelta(days=7))).any()


def test_residuals_are_obs_minus_pred_at_t_plus_h():
    data = {"stage": pd.Series([1.0, 2.0, 4.0, 7.0], index=idx(4))}
    pred = pd.DataFrame({"h1": [0.0, 0.0, 0.0, 0.0], "h2": [1.0, 1.0, 1.0, 1.0]}, index=idx(4))

    r = screen.residuals(data, pred)

    assert r["h1"].tolist()[:3] == [2.0, 4.0, 7.0] and np.isnan(r["h1"].iloc[3])
    assert r["h2"].tolist()[:2] == [3.0, 6.0]


def test_expand_makes_level_diffs_and_anomalies():
    s = pd.Series(np.arange(40, dtype=float), index=idx(40))

    X = screen.expand(s, "gw", diffs=(1, 7), anom=(30,))

    assert list(X.columns) == ["gw", "gw_d1", "gw_d7", "gw_anom30"]
    assert X["gw_d7"].iloc[10] == 7.0
    assert X["gw_anom30"].iloc[35] == pytest.approx(35 - np.mean(np.arange(6, 36)))


def test_residual_screen_flags_planted_signal_not_noise():
    rng = np.random.default_rng(0)
    n = 3000
    i = idx(n)
    a = pd.Series(rng.normal(size=n), index=i)
    noise = pd.Series(rng.normal(size=n), index=i)
    resid = pd.DataFrame({"h1": 0.5 * a + rng.normal(size=n), "h3": rng.normal(size=n)}, index=i)
    cands = pd.DataFrame({"a": a, "noise": noise})
    cands.iloc[:1000, 1] = np.nan  # partial record

    out = screen.residual_screen(resid, cands, n_boot=100, block_days=60, seed=0)

    assert list(out.columns) == ["candidate", "h", "n", "r", "rho", "r_lo", "r_hi", "signal"]
    row = out[(out.candidate == "a") & (out.h == 1)].iloc[0]
    assert row.r > 0.3 and row.r_lo > 0 and row.signal
    nrow = out[(out.candidate == "noise") & (out.h == 1)].iloc[0]
    assert nrow.n == 2000
    assert nrow.r_lo < 0 < nrow.r_hi and not nrow.signal
    assert not out[(out.candidate == "a") & (out.h == 3)].iloc[0].signal
    # sorted strongest first
    assert out["r"].abs().iloc[0] == out["r"].abs().max()
