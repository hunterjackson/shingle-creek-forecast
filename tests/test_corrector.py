import numpy as np
import pandas as pd
import pytest

from home_display.shingle import evaluate
from home_display.shingle.models import arx, base, corrector


@pytest.fixture
def data():
    """Stage driven only by rain in the region near the gauge; the long-record
    'gridMET' rain blurs near and far regions together, the short-record 'radar'
    resolves them (only from 2006). The corrector should strip the far-region noise."""
    rng = np.random.default_rng(12)
    n = 3000
    idx = pd.date_range("2000-01-01", periods=n, name="date")
    near, far = rng.gamma(0.3, 10, n), rng.gamma(0.3, 10, n)
    s = np.empty(n)
    s[0] = 54.0
    for t in range(1, n):
        s[t] = s[t - 1] + 0.03 * near[t] - 0.1 * (s[t - 1] - 54.0) + rng.normal(0, 0.01)
    radar = pd.DataFrame({"r1": near, "r2": far}, index=idx)
    radar.loc[: "2005-12-31"] = np.nan
    return {
        "stage": pd.Series(s, index=idx),
        "q": pd.Series(100.0, index=idx),
        "pr": pd.DataFrame({"g": 0.5 * near + 0.5 * far}, index=idx),
        "pet": pd.DataFrame({"g": 4.0, "r1": 4.0, "r2": 4.0}, index=idx),
        "radar": radar,
    }


BASE_F = ["stage", "dstage_1", "pr_nextH__g", "pr_past3__g"]
RADAR_F = ["rad_pr_nextH__r1", "rad_pr_nextH__r2", "rad_pr_past3__r1"]


def test_corrector_improves_where_radar_exists_and_falls_back_before(data):
    idx = data["stage"].index
    sp = evaluate.time_split(idx, "2006-12-31", "2007-06-30", gap_days=7)
    test = sp["test"][:-7]
    H = (3,)

    plain = arx.ARX(BASE_F, H).fit(data, sp["train"])
    corr = corrector.ResidualCorrector(arx.ARX(BASE_F, H), RADAR_F, H).fit(data, sp["train"])
    y = base.targets(data["stage"], H).loc[test, "h3"]

    mae_plain = evaluate.point_metrics(y, plain.predict(data, test)["h3_p50"])["mae"]
    mae_corr = evaluate.point_metrics(y, corr.predict(data, test)["h3_p50"])["mae"]
    assert mae_corr < 0.8 * mae_plain

    early = idx[100:200]  # no radar yet -> identical to the base model
    pd.testing.assert_series_equal(corr.point(data, early)["h3"], corr.base.point(data, early)["h3"], check_names=False)


def test_corrector_reports_rows_used(data):
    idx = data["stage"].index
    corr = corrector.ResidualCorrector(arx.ARX(BASE_F, (1,)), RADAR_F, (1,)).fit(data, idx[:2500])

    assert 300 < corr.n_rows[1] < 400  # radar exists from 2006-01-01 (row 2191) to row 2500
