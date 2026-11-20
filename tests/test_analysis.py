import numpy as np
import pandas as pd
import pytest

from home_display.shingle import analysis


def idx(n):
    return pd.date_range("2000-01-01", periods=n, name="date")


def test_trailing_sum_includes_current_day():
    s = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}, index=idx(4))

    out = analysis.trailing_sum(s, 2)

    assert np.isnan(out["a"].iloc[0])
    assert out["a"].tolist()[1:] == [3.0, 5.0, 7.0]


def test_future_sum_excludes_current_day():
    s = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}, index=idx(4))

    out = analysis.future_sum(s, 2)

    # day t: sum of t+1..t+2
    assert out["a"].tolist()[:2] == [5.0, 7.0]
    assert out["a"].iloc[2:].isna().all()


def test_future_change():
    stage = pd.Series([1.0, 2.0, 4.0, 7.0], index=idx(4))

    out = analysis.future_change(stage, 2)

    assert out.tolist()[:2] == [3.0, 5.0]
    assert out.iloc[2:].isna().all()


def test_cross_corr_finds_planted_lag():
    rng = np.random.default_rng(0)
    n = 3000
    rain = pd.DataFrame({"fast": rng.gamma(0.3, 10, n), "slow": rng.gamma(0.3, 10, n)}, index=idx(n))
    # stage change responds to "fast" with 1-day lag and "slow" with 4-day lag
    dstage = 0.5 * rain["fast"].shift(1) + 0.5 * rain["slow"].shift(4) + rng.normal(0, 1, n)

    cc = analysis.cross_corr(rain, dstage, max_lag=7)

    assert cc.index.name == "lag"
    assert list(cc.index) == list(range(8))
    assert cc["fast"].idxmax() == 1
    assert cc["slow"].idxmax() == 4


@pytest.fixture
def planted():
    """10 cells; target depends on cell 2 and cell 7 independently. Cells 3 and 8 are
    near-copies (redundant) of 2 and 7; others are independent noise."""
    rng = np.random.default_rng(42)
    n = 4000
    base = {f"c{i}": rng.gamma(0.4, 8, n) for i in range(10)}
    base["c3"] = base["c2"] + rng.normal(0, 0.5, n)
    base["c8"] = base["c7"] + rng.normal(0, 0.5, n)
    X = pd.DataFrame(base, index=idx(n))
    y = pd.Series(1.0 * X["c2"] + 0.7 * X["c7"] + rng.normal(0, 3, n), index=X.index)
    return X, y


def test_greedy_select_picks_independent_drivers_not_redundant_copies(planted):
    X, y = planted

    sel = analysis.greedy_select(X, y, max_k=5, min_partial_r=0.05)

    assert list(sel.columns) == ["feature", "partial_r", "cum_r2"]
    picked = set(sel["feature"])
    assert {"c2", "c7"} <= picked or {"c3", "c7"} <= picked or {"c2", "c8"} <= picked
    # only one from each redundant pair
    assert not {"c2", "c3"} <= picked
    assert not {"c7", "c8"} <= picked
    assert sel["cum_r2"].is_monotonic_increasing


def test_greedy_select_respects_forced_controls(planted):
    X, y = planted
    X = X.assign(control=X["c2"])  # control already explains c2

    sel = analysis.greedy_select(X, y, max_k=3, min_partial_r=0.05, forced=["control"])

    assert sel["feature"].iloc[0] == "control"
    assert "c2" not in set(sel["feature"]) and "c3" not in set(sel["feature"])
    assert sel["feature"].iloc[1] in {"c7", "c8"}


def test_greedy_select_drops_nan_rows(planted):
    X, y = planted
    X.iloc[:10, 0] = np.nan
    y.iloc[-10:] = np.nan

    sel = analysis.greedy_select(X, y, max_k=2)

    assert len(sel) == 2


def test_redundancy_clusters_group_near_copies(planted):
    X, _ = planted

    clusters = analysis.redundancy_clusters(X, threshold=0.9)

    assert clusters["c2"] == clusters["c3"]
    assert clusters["c7"] == clusters["c8"]
    assert clusters["c2"] != clusters["c7"]
    assert clusters["c0"] != clusters["c1"]


def test_univariate_scores_rank_true_driver_highest(planted):
    X, y = planted

    scores = analysis.univariate_scores(X, y)

    assert list(scores.columns) == ["pearson", "spearman", "mutual_info"]
    assert scores["pearson"].idxmax() in {"c2", "c3"}
    assert scores.loc["c0", "pearson"] < 0.1
