import numpy as np
import pandas as pd
import pytest

from home_display.shingle import evaluate
from home_display.shingle.models import direct


@pytest.fixture
def data():
    rng = np.random.default_rng(4)
    n = 2500
    idx = pd.date_range("2000-01-01", periods=n, name="date")
    rain = rng.gamma(0.3, 10, n)
    logq = np.empty(n)
    logq[0] = 4.0
    for t in range(1, n):
        logq[t] = logq[t - 1] + 0.03 * rain[t] - 0.1 * (logq[t - 1] - 4.0) + rng.normal(0, 0.02)
    toho = pd.Series(53 + 0.5 * np.sin(np.arange(n) / 50), index=idx)
    return {
        "stage": pd.Series(54.0, index=idx),
        "q": pd.Series(np.expm1(logq), index=idx),
        "toho": toho,
        "pr": pd.DataFrame({"c": rain}, index=idx),
        "pet": pd.DataFrame({"c": 4.0}, index=idx),
    }


def test_log_q_target_predicts_flow_level(data):
    idx = data["q"].index
    sp = evaluate.time_split(idx, "2004-12-31", "2005-06-30", gap_days=7)
    m = direct.DirectRegressor("log_q", ["log_q", "pr_nextH__c"], horizons=(1, 3)).fit(data, sp["train"])

    pred = m.predict(data, sp["test"][:-7])

    assert list(pred.columns) == ["h1", "h3"]
    truth = data["q"].shift(-3).loc[pred.index]
    persistence = data["q"].loc[pred.index]
    err_m = (np.log1p(pred["h3"]) - np.log1p(truth)).abs().mean()
    err_p = (np.log1p(persistence) - np.log1p(truth)).abs().mean()
    assert err_m < 0.5 * err_p
    assert (pred["h3"] >= 0).all()


def test_toho_target_level(data):
    idx = data["q"].index
    m = direct.DirectRegressor("toho", ["toho", "dtoho_1"], horizons=(2,)).fit(data, idx[:2000])

    pred = m.predict(data, idx[2000:2400])

    truth = data["toho"].shift(-2).loc[pred.index]
    assert (pred["h2"] - truth).abs().mean() < 0.01


def test_lgbm_backend_runs(data):
    idx = data["q"].index
    m = direct.DirectRegressor("log_q", ["log_q", "pr_nextH__c"], horizons=(1,), backend="lgbm").fit(data, idx[:2000])

    pred = m.predict(data, idx[2000:2100])

    assert pred["h1"].notna().all()


def test_unknown_target_raises():
    with pytest.raises(ValueError):
        direct.DirectRegressor("nope", ["stage"])


def test_offset_feature_makes_model_learn_correction_and_extrapolate(data):
    """With offset='gr4j_dH' the target is change minus the physical model's change,
    so an out-of-range physical signal passes straight through (trees can't cap it)."""
    idx = data["q"].index
    d = dict(data)
    lq = np.log1p(d["q"])
    # a "physical model" that knows the true future change, but 10x larger at test time
    d["gr4j"] = pd.DataFrame({"q": d["q"], "s": 0.0, "r": 0.0}, index=idx)
    m = direct.DirectRegressor("log_q", ["log_q", "gr4j_dH"], horizons=(3,), backend="lgbm", offset="gr4j_dH").fit(d, idx[:2000])

    test = idx[2100:2200]
    pred = m.predict(d, test)

    truth = d["q"].shift(-3).loc[test]
    assert (np.log1p(pred["h3"]) - np.log1p(truth)).abs().mean() < 0.02

    big = dict(d)
    big["gr4j"] = d["gr4j"].copy()
    big["gr4j"]["q"] = np.expm1(lq * 1.0 + np.where(np.arange(len(idx)) >= 2103, 2.0, 0.0))
    jump = m.predict(big, idx[2100:2101])["h3"].iloc[0]
    assert np.log1p(jump) - np.log1p(d["q"].iloc[2100]) > 1.5  # follows the +2 log jump


def test_allow_missing_keeps_rows_with_nan_in_listed_features_for_lgbm(data):
    idx = data["q"].index
    # an extra, very informative signal (tomorrow's rain) that only exists for the second half
    extra = data["pr"]["c"].shift(-1).copy()
    extra.iloc[: len(extra) // 2] = np.nan
    d = dict(data, pr=data["pr"].assign(x=extra))
    feats = ["log_q", "pr_past1__x"]  # pr_past1__x = log1p(x) at t (NaN in first half)

    strict = direct.DirectRegressor("log_q", feats, (1,), backend="lgbm").fit(d, idx[:2000])
    loose = direct.DirectRegressor("log_q", feats, (1,), backend="lgbm", allow_missing=["pr_past1__x"]).fit(d, idx[:2000])

    assert loose.n_train[1] > 1.5 * strict.n_train[1]  # early rows kept
    early = idx[100:200]
    assert strict.predict(d, early)["h1"].isna().all()
    assert loose.predict(d, early)["h1"].notna().all()  # predicts where the extra signal is missing


def test_allow_missing_rejected_for_ridge(data):
    with pytest.raises(ValueError):
        direct.DirectRegressor("log_q", ["log_q"], (1,), backend="ridge", allow_missing=["log_q"])
