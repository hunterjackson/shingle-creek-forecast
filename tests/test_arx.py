import numpy as np
import pandas as pd
import pytest

from home_display.shingle import evaluate
from home_display.shingle.models import arx, base, baseline

CELL = "28.0000_-81.0000"


@pytest.fixture
def data():
    """Reservoir-like stage driven by one cell's rain."""
    rng = np.random.default_rng(2)
    n = 3000
    idx = pd.date_range("2000-01-01", periods=n, name="date")
    rain = rng.gamma(0.3, 10, n)
    s = np.empty(n)
    s[0] = 54.0
    for t in range(1, n):
        s[t] = s[t - 1] + 0.02 * rain[t] - 0.08 * (s[t - 1] - 54.0) + rng.normal(0, 0.01)
    stage = pd.Series(s, index=idx)
    pr = pd.DataFrame({CELL: rain}, index=idx)
    return {
        "stage": stage,
        "q": pd.Series(100.0, index=idx),
        "q_up": pd.Series(50.0, index=idx),
        "pr": pr,
        "pet": pd.DataFrame({CELL: 4.0}, index=idx),
    }


FEATS = ["stage", "dstage_1", f"pr_nextH__{CELL}", f"pr_past3__{CELL}"]


def test_arx_beats_persistence_out_of_sample(data):
    idx = data["stage"].index
    sp = evaluate.time_split(idx, train_end="2006-12-31", val_end="2007-12-31", gap_days=7)
    test = sp["test"][:-7]
    horizons = (1, 3)

    m = arx.ARX(features=FEATS, horizons=horizons).fit(data, sp["train"])
    fc = m.predict(data, test)
    pers = baseline.Persistence(horizons).fit(data, sp["train"]).predict(data, test)
    y = base.targets(data["stage"], horizons).loc[test]

    assert list(fc.columns) == base.forecast_columns(horizons)
    for h in horizons:
        mae_arx = evaluate.point_metrics(y[f"h{h}"], fc[f"h{h}_p50"])["mae"]
        mae_p = evaluate.point_metrics(y[f"h{h}"], pers[f"h{h}_p50"])["mae"]
        assert mae_arx < 0.6 * mae_p


def test_arx_quantiles_ordered_and_roughly_calibrated(data):
    idx = data["stage"].index
    sp = evaluate.time_split(idx, train_end="2006-12-31", val_end="2007-12-31", gap_days=7)
    test = sp["test"][:-7]

    m = arx.ARX(features=FEATS, horizons=(3,)).fit(data, sp["train"])
    fc = m.predict(data, test)
    y = base.targets(data["stage"], (3,)).loc[test, "h3"]

    assert (fc["h3_p10"] <= fc["h3_p50"]).all() and (fc["h3_p50"] <= fc["h3_p90"]).all()
    assert 0.65 < evaluate.coverage(y, fc["h3_p10"], fc["h3_p90"]) < 0.95


def test_arx_per_horizon_feature_lists(data):
    idx = data["stage"].index
    feats = {1: ["stage", f"pr_next1__{CELL}"], 3: ["stage", f"pr_next3__{CELL}"]}

    m = arx.ARX(features=feats, horizons=(1, 3)).fit(data, idx[:2000])

    assert m.feature_names(1) == feats[1]
    assert m.feature_names(3) == feats[3]
