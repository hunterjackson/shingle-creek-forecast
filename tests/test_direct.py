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
