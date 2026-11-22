import numpy as np
import pandas as pd
import pytest

from home_display.shingle import evaluate, rating
from home_display.shingle.models import base, direct, hybrid

TRUE_RATING = rating.CombinedRating(c=3e-5, d=1.5, h0=47.7, a=0.42, b=0.46, k=1.0)


@pytest.fixture
def data():
    rng = np.random.default_rng(8)
    n = 3000
    idx = pd.date_range("2000-01-01", periods=n, name="date")
    rain = rng.gamma(0.3, 12, n)
    logq = np.empty(n)
    logq[0] = 4.5
    for t in range(1, n):
        logq[t] = logq[t - 1] + 0.035 * rain[t] - 0.08 * (logq[t - 1] - 4.5) + rng.normal(0, 0.02)
    q = pd.Series(np.expm1(logq), index=idx)
    toho = pd.Series(53.5 + 1.2 * np.sin(np.arange(n) * 2 * np.pi / 365) + rng.normal(0, 0.01, n), index=idx)
    stage = pd.Series(TRUE_RATING.predict_stage(q.values, toho.values) + rng.normal(0, 0.03, n), index=idx)
    return {
        "stage": stage,
        "q": q,
        "toho": toho,
        "pr": pd.DataFrame({"c": rain}, index=idx),
        "pet": pd.DataFrame({"c": 4.0}, index=idx),
    }


def make(horizons):
    flow = direct.DirectRegressor("log_q", ["log_q", "pr_nextH__c"], horizons)
    lake = direct.DirectRegressor("toho", ["toho", "dtoho_1", "dtoho_3"], horizons)
    return hybrid.RatingHybrid(flow, lake, horizons)


def test_hybrid_fits_rating_and_outputs_forecast_columns(data):
    idx = data["stage"].index
    sp = evaluate.time_split(idx, "2005-12-31", "2006-12-31", gap_days=7)
    m = make((1, 3)).fit(data, sp["train"]).calibrate(data, sp["val"])

    fc = m.predict(data, sp["test"][:-7])

    assert list(fc.columns) == base.forecast_columns((1, 3))
    assert isinstance(m.rating, rating.CombinedRating)
    assert (fc["h3_p10"] <= fc["h3_p50"]).all() and (fc["h3_p50"] <= fc["h3_p90"]).all()


def test_hybrid_beats_persistence_on_stage(data):
    from home_display.shingle.models import baseline

    idx = data["stage"].index
    sp = evaluate.time_split(idx, "2005-12-31", "2006-12-31", gap_days=7)
    test = sp["test"][:-7]
    m = make((3,)).fit(data, sp["train"]).calibrate(data, sp["val"])
    p = baseline.Persistence((3,)).fit(data, sp["train"])
    y = base.targets(data["stage"], (3,)).loc[test, "h3"]

    mae_h = evaluate.point_metrics(y, m.predict(data, test)["h3_p50"])["mae"]
    mae_p = evaluate.point_metrics(y, p.predict(data, test)["h3_p50"])["mae"]
    assert mae_h < 0.7 * mae_p


def test_hybrid_exposes_component_forecasts(data):
    idx = data["stage"].index
    m = make((1,)).fit(data, idx[:2000])

    comp = m.components(data, idx[2000:2010])

    assert list(comp.columns) == ["h1_q", "h1_toho", "h1_stage"]


def test_residual_carryforward_corrects_persistent_rating_offset(data):
    """Real ratings drift (vegetation, channel change): add an AR(1) offset to stage.

    The carry-forward should recover most of the damage, i.e. land close to the
    error achieved on drift-free data (whose floor is the flow-forecast error).
    """
    rng = np.random.default_rng(1)
    n = len(data["stage"])
    off = np.zeros(n)
    for t in range(1, n):
        off[t] = 0.98 * off[t - 1] + rng.normal(0, 0.03)
    drift = dict(data)
    drift["stage"] = data["stage"] + off
    idx = data["stage"].index
    sp = evaluate.time_split(idx, "2005-12-31", "2006-12-31", gap_days=7)
    test = sp["test"][:-7]

    def mae(d, carry):
        m = make((1,))
        m.carry_residual = carry
        m.fit(d, sp["train"])
        y = base.targets(d["stage"], (1,)).loc[test, "h1"]
        return evaluate.point_metrics(y, m.predict(d, test)["h1_p50"])["mae"], m

    floor, m_clean = mae(data, True)
    plain, _ = mae(drift, False)
    corrected, m = mae(drift, True)

    assert corrected < plain
    assert (corrected - floor) < 0.3 * (plain - floor)  # recovers >70% of the drift damage
    assert 0.8 < m.phi[1] <= 1.0
    assert abs(m_clean.phi[1]) < 0.2  # no drift -> (almost) no carry


def test_bands_widen_when_a_big_change_is_predicted(data):
    idx = data["stage"].index
    sp = evaluate.time_split(idx, "2005-12-31", "2006-12-31", gap_days=7)
    m = make((3,)).fit(data, sp["train"]).calibrate(data, sp["val"])

    fc = m.predict(data, sp["test"][:-7])

    width = fc["h3_p90"] - fc["h3_p10"]
    change = (fc["h3_p50"] - data["stage"].loc[fc.index]).abs()
    big = change > change.quantile(0.9)
    small = change < change.quantile(0.5)
    assert width[big].mean() > 1.5 * width[small].mean()


def test_hybrid_passes_rating_options_to_fit(data, monkeypatch):
    from home_display.shingle import rating as rating_mod

    seen = {}
    real = rating_mod.fit_combined

    def spy(q, stage, lake, **kw):
        seen.update(kw)
        return real(q, stage, lake, **kw)

    monkeypatch.setattr(rating_mod, "fit_combined", spy)
    m = make((1,))
    m.rating_opts = {"floodplain": True, "d_max": 1.2, "g_min": 0.3}
    m.fit(data, data["stage"].index[:1500])

    assert seen == {"floodplain": True, "d_max": 1.2, "g_min": 0.3}
    assert m.rating.qb is not None
