import numpy as np
import pandas as pd
import pytest

from home_display.shingle import evaluate


def idx(start, n):
    return pd.date_range(start, periods=n, name="date")


def test_time_split_is_disjoint_and_ordered():
    index = pd.date_range("2014-01-01", "2021-12-31", name="date")

    sp = evaluate.time_split(index, train_end="2015-12-31", val_end="2019-12-31")

    assert sp["train"].max() == pd.Timestamp("2015-12-31")
    assert sp["val"].min() == pd.Timestamp("2016-01-01")
    assert sp["val"].max() == pd.Timestamp("2019-12-31")
    assert sp["test"].min() == pd.Timestamp("2020-01-01")
    assert len(sp["train"]) + len(sp["val"]) + len(sp["test"]) == len(index)


def test_time_split_gap_drops_days_before_each_boundary():
    index = pd.date_range("2015-12-01", "2016-01-31", name="date")

    sp = evaluate.time_split(index, train_end="2015-12-31", val_end="2016-01-20", gap_days=7)

    # targets up to 7 days ahead must not cross into the next split
    assert sp["train"].max() == pd.Timestamp("2015-12-24")
    assert sp["val"].max() == pd.Timestamp("2016-01-13")


def test_point_metrics():
    y = pd.Series([50.0, 52.0, 60.0, 61.0])
    p = pd.Series([51.0, 52.0, 59.0, 63.0])

    m = evaluate.point_metrics(y, p, high=59.0, low=51.0)

    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(np.sqrt((1 + 0 + 1 + 4) / 4))
    assert m["mae_high"] == pytest.approx(1.5)  # y >= 59 -> rows 2, 3
    assert m["mae_low"] == pytest.approx(1.0)  # y <= 51 -> row 0
    assert m["n"] == 4


def test_interval_coverage():
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    lo = pd.Series([0.0, 2.5, 2.0, 3.0])
    hi = pd.Series([2.0, 3.0, 4.0, 3.5])

    assert evaluate.coverage(y, lo, hi) == pytest.approx(0.5)


def test_threshold_skill_hits_misses_false_alarms():
    y = pd.Series([58.0, 59.5, 60.0, 58.0, 58.5])
    p = pd.Series([59.2, 59.1, 58.0, 58.0, 58.0])

    s = evaluate.threshold_skill(y, p, threshold=59.0)

    assert s == {"hits": 1, "misses": 1, "false_alarms": 1, "pod": 0.5, "far": 0.5}


def test_event_peaks_matches_obs_and_pred_peaks():
    obs = pd.Series([55, 57, 60, 61, 60, 57, 55, 55, 56, 59.5, 59, 55.0], index=idx("2022-09-25", 12))
    pred = pd.Series([55, 56, 58, 60, 60.5, 58, 55, 55, 55, 57, 58.5, 55.0], index=obs.index)

    ev = evaluate.event_peaks(obs, pred, threshold=59.0)

    assert len(ev) == 2
    first = ev.iloc[0]
    assert first["obs_peak"] == 61
    assert first["pred_peak"] == 60.5
    assert first["peak_error"] == pytest.approx(-0.5)
    assert first["timing_error_days"] == 1  # predicted peak one day late


def test_conformal_offsets_hit_target_coverage():
    rng = np.random.default_rng(0)
    resid_cal = pd.Series(rng.normal(0, 1, 5000))
    resid_new = rng.normal(0, 1, 5000)

    lo, hi = evaluate.conformal_offsets(resid_cal, coverage=0.8)

    inside = ((resid_new >= lo) & (resid_new <= hi)).mean()
    assert inside == pytest.approx(0.8, abs=0.02)
    assert lo < 0 < hi


def test_metrics_align_series_on_index():
    y = pd.Series([1.0, 2.0, 3.0], index=idx("2020-01-01", 3))
    p = pd.Series([2.0, 3.0], index=idx("2020-01-02", 2))  # perfect where overlapping

    assert evaluate.point_metrics(y, p)["mae"] == 0.0


class _StubModel:
    """p50 = 0; calibrate stores the residual offsets of the calibration rows."""

    horizons = (1,)

    def __init__(self):
        self.calibrated_on = []

    def calibrate(self, data, cal_index):
        self.calibrated_on.append(cal_index)
        r = data["stage"].shift(-1).loc[cal_index]
        self.lo, self.hi = evaluate.conformal_offsets(r, 0.8)
        return self

    def predict(self, data, index):
        return pd.DataFrame({"h1_p10": self.lo, "h1_p50": 0.0, "h1_p90": self.hi}, index=index)


def test_cross_conformal_predicts_each_year_with_bands_from_other_years():
    rng = np.random.default_rng(0)
    index = pd.date_range("2016-01-01", "2019-12-31", name="date")
    data = {"stage": pd.Series(rng.normal(0, 1, len(index)), index=index)}
    m = _StubModel()

    fc = evaluate.cross_conformal_predict(m, data, index)

    assert fc.index.equals(index)
    assert len(m.calibrated_on) == 4
    for cal in m.calibrated_on:
        assert len(set(cal.year)) == 3  # one year held out each time
    y = data["stage"].shift(-1).loc[index]
    assert 0.75 < evaluate.coverage(y, fc["h1_p10"], fc["h1_p90"]) < 0.85
