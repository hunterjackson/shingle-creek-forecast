import numpy as np
import pandas as pd
import pytest

from home_display.shingle import features

CELL = "28.0000_-81.0000"
CELL2 = "28.0417_-81.0000"


@pytest.fixture
def data():
    rng = np.random.default_rng(5)
    n = 200
    idx = pd.date_range("2001-01-01", periods=n, name="date")
    pr = pd.DataFrame({CELL: rng.gamma(0.3, 10, n), CELL2: rng.gamma(0.3, 10, n)}, index=idx)
    pet = pd.DataFrame({CELL: 4.0, CELL2: 4.0}, index=idx)
    stage = pd.Series(54 + np.cumsum(rng.normal(0, 0.1, n)), index=idx)
    q = pd.Series(rng.uniform(50, 500, n), index=idx)
    return {"stage": stage, "q": q, "q_up": q * 0.5, "pr": pr, "pet": pet}


NAMES = [
    "stage",
    "dstage_1",
    "dstage_3",
    "log_q",
    "log_q_up",
    "doy_sin",
    "doy_cos",
    f"pr_past7__{CELL}",
    f"pr_next3__{CELL2}",
    f"wb_past30__{CELL}",
]


def test_build_returns_requested_columns_in_order(data):
    X = features.build(data, NAMES)

    assert list(X.columns) == NAMES
    assert X.index.equals(data["stage"].index)


def test_values_match_definitions(data):
    X = features.build(data, NAMES)
    t = 50

    assert X["stage"].iloc[t] == data["stage"].iloc[t]
    assert X["dstage_3"].iloc[t] == pytest.approx(data["stage"].iloc[t] - data["stage"].iloc[t - 3])
    assert X[f"pr_past7__{CELL}"].iloc[t] == pytest.approx(np.log1p(data["pr"][CELL].iloc[t - 6 : t + 1].sum()))
    assert X[f"pr_next3__{CELL2}"].iloc[t] == pytest.approx(np.log1p(data["pr"][CELL2].iloc[t + 1 : t + 4].sum()))
    wb = (data["pr"][CELL] - data["pet"][CELL]).iloc[t - 29 : t + 1].sum()
    assert X[f"wb_past30__{CELL}"].iloc[t] == pytest.approx(wb)


def test_horizon_dependent_future_rain_can_be_requested_generically(data):
    X = features.build(data, [f"pr_nextH__{CELL}"], horizon=5)

    t = 10
    assert list(X.columns) == [f"pr_nextH__{CELL}"]
    assert X.iloc[t, 0] == pytest.approx(np.log1p(data["pr"][CELL].iloc[t + 1 : t + 6].sum()))


def test_unknown_feature_raises(data):
    with pytest.raises(KeyError):
        features.build(data, ["nonsense"])


def test_no_lookahead_for_past_features(data):
    """Changing anything after day t must not change non-forecast features at t."""
    past_names = [n for n in NAMES if not n.startswith("pr_next")]
    t = 120
    before = features.build(data, past_names).iloc[t]

    perturbed = {k: v.copy() for k, v in data.items()}
    after_t = perturbed["stage"].index > perturbed["stage"].index[t]
    for k in ("stage", "q", "q_up"):
        perturbed[k][after_t] += 100.0
    for k in ("pr", "pet"):
        perturbed[k].loc[after_t] += 100.0

    after = features.build(perturbed, past_names).iloc[t]
    pd.testing.assert_series_equal(before, after)


def test_future_rain_feature_uses_only_rain_not_future_stage(data):
    name = f"pr_next3__{CELL}"
    t = 120
    before = features.build(data, [name]).iloc[t]

    perturbed = {k: v.copy() for k, v in data.items()}
    perturbed["stage"].iloc[t + 1 :] += 100.0

    after = features.build(perturbed, [name]).iloc[t]
    pd.testing.assert_series_equal(before, after)


@pytest.fixture
def lake_data(data):
    idx = data["stage"].index
    rng = np.random.default_rng(9)
    d = dict(data)
    d["toho"] = pd.Series(53.0 + np.cumsum(rng.normal(0, 0.02, len(idx))), index=idx)
    d["s61_q"] = pd.Series(rng.uniform(0, 1000, len(idx)), index=idx)
    d["s59_q"] = pd.Series(rng.uniform(0, 500, len(idx)), index=idx)
    d["etoho"] = pd.Series(56.0, index=idx)
    return d


def test_lake_features(lake_data):
    from home_display.shingle import schedule

    names = ["toho", "dtoho_3", "toho_dev", "toho_target_dH", "log_s61", "log_s59", "etoho", "log_q_d1"]
    X = features.build(lake_data, names, horizon=5)
    t = 60
    idx = lake_data["stage"].index
    target = schedule.toho_target_series(idx)

    assert X["toho"].iloc[t] == lake_data["toho"].iloc[t]
    assert X["dtoho_3"].iloc[t] == pytest.approx(lake_data["toho"].iloc[t] - lake_data["toho"].iloc[t - 3])
    assert X["toho_dev"].iloc[t] == pytest.approx(lake_data["toho"].iloc[t] - target.iloc[t])
    # schedule is known in advance, so target(t+h) - target(t) is a legitimate "future" feature
    assert X["toho_target_dH"].iloc[t] == pytest.approx(target.iloc[t + 5] - target.iloc[t])
    assert X["log_s61"].iloc[t] == pytest.approx(np.log1p(lake_data["s61_q"].iloc[t]))
    assert X["log_q_d1"].iloc[t] == pytest.approx(np.log1p(lake_data["q"].iloc[t]) - np.log1p(lake_data["q"].iloc[t - 1]))


def test_lake_features_no_lookahead(lake_data):
    names = ["toho", "dtoho_3", "toho_dev", "log_s61", "log_s59", "etoho", "log_q_d1"]
    t = 120
    before = features.build(lake_data, names).iloc[t]
    pert = {k: v.copy() for k, v in lake_data.items()}
    after_t = pert["stage"].index > pert["stage"].index[t]
    for k in ("toho", "s61_q", "s59_q", "etoho", "q"):
        pert[k][after_t] += 50.0

    after = features.build(pert, names).iloc[t]
    pd.testing.assert_series_equal(before, after)


def test_gr4j_features(data):
    from home_display.shingle.models import gr4j

    d = dict(data)
    d["gr4j"] = pd.DataFrame(
        {"q": np.linspace(10, 100, len(d["stage"])), "s": 50.0, "r": 20.0}, index=d["stage"].index
    )
    X = features.build(d, ["gr4j_logq", "gr4j_dH", "gr4j_s", "gr4j_r"], horizon=4)
    t = 30

    assert X["gr4j_logq"].iloc[t] == pytest.approx(np.log1p(d["gr4j"]["q"].iloc[t]))
    assert X["gr4j_dH"].iloc[t] == pytest.approx(np.log1p(d["gr4j"]["q"].iloc[t + 4]) - np.log1p(d["gr4j"]["q"].iloc[t]))
    assert X["gr4j_s"].iloc[t] == 50.0
