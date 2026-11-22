import numpy as np
import pandas as pd
import pytest

from shingle import analysis


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


def test_stability_select_keeps_true_drivers_and_drops_noise(planted):
    X, y = planted
    # add many pure-noise candidates to provoke spurious greedy picks
    rng = np.random.default_rng(1)
    for i in range(60):
        X[f"noise{i}"] = rng.gamma(0.4, 8, len(X))

    freq = analysis.stability_select(X, y, n_boot=30, block_days=200, max_k=4, seed=0)

    assert list(freq.columns) == ["freq", "mean_partial_r"]
    top = freq[freq["freq"] >= 0.6].index
    assert ({"c2", "c3"} & set(top)) and ({"c7", "c8"} & set(top))
    assert not any(t.startswith("noise") for t in top)
    assert freq["freq"].between(0, 1).all()


def test_stability_select_excludes_forced_from_table(planted):
    X, y = planted
    X = X.assign(ctrl=X["c0"])

    freq = analysis.stability_select(X, y, n_boot=5, block_days=400, max_k=3, forced=["ctrl"], seed=0)

    assert "ctrl" not in freq.index


def test_pool_by_cluster_sums_feature_frequencies_per_kind_and_region():
    freq = pd.DataFrame(
        {"freq": [0.4, 0.45, 0.3, 0.2], "mean_partial_r": [0.6, 0.6, 0.2, 0.1]},
        index=["pr_next3__a", "pr_next3__b", "pr_past1__a", "pr_next3__c"],
    )
    clusters = pd.Series({"a": 1, "b": 1, "c": 2})

    pooled = analysis.pool_by_cluster(freq, clusters)

    assert pooled.loc["pr_next3__r1", "freq"] == pytest.approx(0.85)
    assert pooled.loc["pr_past1__r1", "freq"] == pytest.approx(0.3)
    assert pooled.loc["pr_next3__r2", "freq"] == pytest.approx(0.2)
    assert pooled["freq"].is_monotonic_decreasing
    assert pooled["freq"].max() <= 1.0
