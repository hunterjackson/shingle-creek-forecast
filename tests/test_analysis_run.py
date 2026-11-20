import json

import numpy as np
import pandas as pd
import pytest

from home_display.shingle import analysis_run


def idx(n, start="2000-01-01"):
    return pd.date_range(start, periods=n, name="date")


@pytest.fixture
def history():
    rng = np.random.default_rng(1)
    n = 2500
    cells = ["28.0000_-81.0000", "28.0417_-81.0000", "28.0833_-81.0000"]
    pr = pd.DataFrame({c: rng.gamma(0.3, 10, n) for c in cells}, index=idx(n))
    pet = pd.DataFrame({c: 4.0 for c in cells}, index=idx(n))
    # reservoir-like stage: daily rise from the first cell's rain, linear recession to 54 ft
    rain0 = pr[cells[0]].to_numpy()
    s = np.empty(n)
    s[0] = 54.0
    for t in range(1, n):
        s[t] = s[t - 1] + 0.02 * rain0[t] - 0.1 * (s[t - 1] - 54.0) + rng.normal(0, 0.01)
    stage = pd.Series(s, index=idx(n))
    gauge = pd.DataFrame({"discharge_cfs": 100.0, "stage_ft": stage})
    cell_tbl = pd.DataFrame(
        {"lat": [28.0, 28.0417, 28.0833], "lon": [-81.0] * 3, "basin_frac": [1.0, 0.5, 0.0], "dist_outlet_km": [1, 5, 9]},
        index=pd.Index(cells, name="cell_id"),
    )
    return {"gauge": gauge, "upstream": gauge * 0.5, "pr": pr, "pet": pet, "cells": cell_tbl}


def test_candidate_features_naming_and_windows(history):
    X = analysis_run.candidate_features(history["pr"], history["pet"], past_windows=(1, 7), horizon=3)

    c = "28.0000_-81.0000"
    assert f"pr_past1__{c}" in X.columns
    assert f"pr_past7__{c}" in X.columns
    assert f"pr_next3__{c}" in X.columns
    assert f"wb_past30__{c}" in X.columns  # rain minus ET0 over 30 days
    # log1p transform on rain sums
    raw = history["pr"][c].rolling(7).sum()
    assert X[f"pr_past7__{c}"].iloc[10] == pytest.approx(np.log1p(raw.iloc[10]))


def test_state_controls(history):
    S = analysis_run.state_controls(history["gauge"], history["upstream"])

    assert list(S.columns) == ["stage", "dstage_1", "dstage_3", "log_q", "log_q_up"]


def test_run_restricts_to_train_period_and_finds_driver(history):
    res = analysis_run.run(history, horizons=(3,), past_windows=(1, 3), train_end="2005-12-31", max_k=4)

    assert res["train_end"] == "2005-12-31"
    sel = res["selection"][3]
    rain_feats = [f for f in sel["feature"] if f.startswith("pr_")]
    assert rain_feats[0] == "pr_next3__28.0000_-81.0000"
    assert "cross_corr" in res and "scores" in res and "clusters" in res


def test_selected_cells_json_schema(history, tmp_path):
    res = analysis_run.run(history, horizons=(3,), past_windows=(1, 3), train_end="2005-12-31", max_k=4)
    path = tmp_path / "selected_cells.json"

    analysis_run.write_selection(res, path)

    data = json.loads(path.read_text())
    assert data["train_end"] == "2005-12-31"
    assert "28.0000_-81.0000" in data["cells"]
    assert data["features"]["3"][0]  # ordered feature list per horizon
    assert all(isinstance(f, str) for f in data["features"]["3"])
