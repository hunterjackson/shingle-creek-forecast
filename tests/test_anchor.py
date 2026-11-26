import numpy as np
import pandas as pd
import pytest

from shingle.models import anchor


@pytest.fixture
def data():
    n = 1500
    idx = pd.date_range("2005-01-01", periods=n, name="date")
    rng = np.random.default_rng(0)
    stage = pd.Series(54 + np.cumsum(rng.normal(0, 0.05, n)), index=idx)
    nxt = stage.shift(-1).fillna(stage.iloc[-1])
    # the morning reading and the midnight-8 am mean sit between today's and tomorrow's means
    am = stage + 0.6 * (nxt - stage) + rng.normal(0, 0.005, n)
    sofar = stage + 0.4 * (nxt - stage)
    am[:400] = np.nan  # no intraday record in the early years
    sofar[:400] = np.nan
    return {"stage": stage, "stage_am": pd.Series(am, index=idx), "stage_sofar": pd.Series(sofar, index=idx)}


FEATS = ["stage", "dstage_1", "stage_am_d", "stage_sofar_d"]


def test_learns_day_one_relative_to_the_morning_reading_from_rows_that_have_it(data):
    ix = data["stage"].index
    m = anchor.MorningAnchor(FEATS, lgbm_params={"n_estimators": 200, "min_child_samples": 10}).fit(data, ix[:1200])

    p = m.point(data, ix[1200:-1])["h1"]
    y = data["stage"].shift(-1).loc[ix[1200:-1]]

    assert m.n_train == 1200 - 400
    assert (p - y).abs().mean() < 0.5 * (data["stage_am"].loc[p.index] - y).abs().mean()


def test_no_prediction_without_a_morning_reading(data):
    ix = data["stage"].index
    m = anchor.MorningAnchor(FEATS, lgbm_params={"n_estimators": 20}).fit(data, ix[:1200])

    assert m.point(data, ix[:400])["h1"].isna().all()
    d = {k: v for k, v in data.items() if k not in ("stage_am", "stage_sofar")}
    assert m.point(d, ix[1200:])["h1"].isna().all()


def test_unfitted_without_intraday_record_predicts_nothing(data):
    d = {"stage": data["stage"]}
    m = anchor.MorningAnchor(FEATS).fit(d, d["stage"].index)

    assert m.n_train == 0 and m.point(d, d["stage"].index)["h1"].isna().all()
