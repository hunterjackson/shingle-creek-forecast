import numpy as np
import pandas as pd
import pytest

from home_display.shingle import compare, evaluate
from home_display.shingle.models import baseline


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2014-01-01", "2019-12-31", name="date")
    stage = pd.Series(54 + np.cumsum(rng.normal(0, 0.05, len(idx))), index=idx)
    stage.loc["2017-09-10":"2017-09-14"] = [59.5, 60.5, 61.0, 60.2, 59.1]
    return {"stage": stage}


def test_score_models_returns_long_metrics_table(data):
    sp = evaluate.time_split(data["stage"].index, "2015-12-31", "2019-12-31", gap_days=7)
    models = {"persistence": baseline.Persistence((1, 3))}

    res = compare.score_models(models, data, sp["train"], sp["val"], horizons=(1, 3))

    m = res["metrics"]
    assert set(m.columns) >= {"model", "h", "mae", "rmse", "mae_high", "mae_low", "cov80", "pod", "far"}
    assert len(m) == 2
    assert set(res["forecasts"]) == {"persistence"}
    ev = res["events"]
    assert {"model", "h", "obs_peak", "pred_peak", "peak_error", "timing_error_days"} <= set(ev.columns)


def test_write_report_contains_tables(tmp_path, data):
    sp = evaluate.time_split(data["stage"].index, "2015-12-31", "2019-12-31", gap_days=7)
    res = compare.score_models({"persistence": baseline.Persistence((1,))}, data, sp["train"], sp["val"], horizons=(1,))
    path = tmp_path / "report.md"

    compare.write_report(res, path, title="Test comparison", notes=["perfect future rain"])

    text = path.read_text()
    assert "# Test comparison" in text
    assert "persistence" in text
    assert "perfect future rain" in text
    assert "| model" in text
