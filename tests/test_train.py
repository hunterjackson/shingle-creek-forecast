import numpy as np
import pandas as pd
import pytest

from shingle import dataset, train
from shingle.models import base

from synth import make_world

H = (1, 2, 4)


@pytest.fixture(scope="module")
def trained():
    hist, raw, src = make_world()
    data = dataset.build_data(hist)
    idx = data["stage"].index
    art = train.train(hist, raw, src, train_end=str(idx[3 * 365].date()), fcst_start=str(idx[365].date()), horizons=H,
                      gr4j_maxiter=15, oof_years=False)
    return hist, raw, src, art


def test_train_returns_artifact_with_everything_needed_to_predict(trained):
    _, _, _, art = trained

    assert set(art.scale) == set(range(1, 8)) and all(v > 0 for v in art.scale.values())
    assert art.gr4j.params is not None
    assert art.horizons == H
    assert set(art.model.bands) == set(H)  # bands calibrated (forecast mode)
    assert not art.pet_clim.isna().any().any()


def test_artifact_round_trips_through_disk(trained, tmp_path):
    hist, raw, src, art = trained
    path = tmp_path / "model.pkl"
    train.save(art, path)

    art2 = train.load(path)
    day = pd.Timestamp(hist["gauge"].index[3 * 365 + 50])
    a = train.predict(art, hist, raw, src, issue_day=day)
    b = train.predict(art2, hist, raw, src, issue_day=day)

    pd.testing.assert_frame_equal(a, b)


def test_predict_gives_ordered_quantiles_for_each_horizon_and_uses_the_forecast(trained):
    hist, raw, src, art = trained
    day = pd.Timestamp(hist["gauge"].index[3 * 365 + 60])

    fc = train.predict(art, hist, raw, src, issue_day=day)

    assert list(fc.columns) == ["date", "p10", "p50", "p90"]
    assert fc["date"].tolist() == [day + pd.Timedelta(days=h) for h in H]
    assert (fc["p10"] <= fc["p50"]).all() and (fc["p50"] <= fc["p90"]).all()

    wet = raw.copy()
    wet.loc[day] = wet.loc[day] + 40.0  # much more forecast rain issued that day
    fc_wet = train.predict(art, hist, wet, src, issue_day=day)
    assert fc_wet["p50"].iloc[-1] > fc["p50"].iloc[-1]


def test_backtest_matches_single_day_predict(trained):
    hist, raw, src, art = trained
    idx = hist["gauge"].index
    days = idx[3 * 365 + 40 : 3 * 365 + 50]

    bt = train.backtest(art, hist, raw, src, days)

    assert list(bt.columns) == base.forecast_columns(H)
    one = train.predict(art, hist, raw, src, issue_day=days[3])
    np.testing.assert_allclose(bt.loc[days[3], [f"h{h}_p50" for h in H]].to_numpy(float), one["p50"].to_numpy(float))


@pytest.fixture(scope="module")
def trained_oof():
    hist, raw, src = make_world(n_days=6 * 365, start="2016-01-01", iv=True)
    data = dataset.build_data(hist)
    idx = data["stage"].index
    art = train.train(hist, raw, src, train_end=str(idx[5 * 365].date()), fcst_start=str(idx[3 * 365].date()), horizons=H,
                      gr4j_maxiter=15, oof_years=True, scale_block_years=1)
    return hist, raw, src, art


def test_oof_training_uses_learned_bands_from_day_two(trained_oof):
    from shingle import bands
    from shingle.evaluate import BinnedConformal

    hist, raw, src, art = trained_oof

    def base(b):
        return b.base if isinstance(b, bands.WetBand) else b

    assert isinstance(base(art.model.bands[1]), BinnedConformal)
    assert all(isinstance(base(art.model.bands[h]), bands.LearnedBand) for h in H if h >= train.LEARNED_BANDS_FROM_H)
    day = pd.Timestamp(hist["gauge"].index[5 * 365 + 30])
    fc = train.predict(art, hist, raw, src, issue_day=day)
    assert (fc["p10"] <= fc["p50"]).all() and (fc["p50"] <= fc["p90"]).all()
    assert (fc["p90"] - fc["p10"] > 0).all()


def test_learned_bands_survive_save_and_backtest_matches_predict(trained_oof, tmp_path):
    hist, raw, src, art = trained_oof
    train.save(art, tmp_path / "m.pkl")
    art2 = train.load(tmp_path / "m.pkl")
    idx = hist["gauge"].index
    days = idx[5 * 365 + 40 : 5 * 365 + 45]

    bt = train.backtest(art2, hist, raw, src, days)
    one = train.predict(art, hist, raw, src, issue_day=days[2])

    np.testing.assert_allclose(bt.loc[days[2], [f"h{h}_p90" for h in H]].to_numpy(float), one["p90"].to_numpy(float))


def test_oof_training_widens_days_1_to_3_when_rain_is_forecast(trained_oof):
    """T2 (docs/divergence_test_plan.md): days 1-3 wrap the production band in a WetBand; later days are unchanged."""
    from shingle import bands

    hist, raw, src, art = trained_oof
    for h in H:
        b = art.model.bands[h]
        if h <= train.WET_BANDS_MAX_H:
            assert isinstance(b, bands.WetBand) and b.h == h and b.scale_model is not None
            assert set(b.weights) == set(art.gr4j.weights)
        else:
            assert not isinstance(b, bands.WetBand)

    # same issue day, issued rain forecast dry vs. soaked: only the wet one widens (day 1)
    day = pd.Timestamp(hist["gauge"].index[5 * 365 + 30])
    dry, wet = raw.copy(), raw.copy()
    dry.loc[day] = 0.0
    wet.loc[day] = 0.0
    wet.loc[day, [c for c in wet.columns if c[0] == 1]] = 40.0
    w_dry = train.predict(art, hist, dry, src, issue_day=day).set_index("date")
    w_wet = train.predict(art, hist, wet, src, issue_day=day).set_index("date")
    d1 = day + pd.Timedelta(days=1)
    b1 = art.model.bands[1]

    def width(df):
        return round(float(df.loc[d1, "p90"] - df.loc[d1, "p10"]), 9)

    assert width(w_wet) in set(np.round(b1.bins.hi - b1.bins.lo, 9))
    assert width(w_dry) in set(np.round(b1.base.hi - b1.base.lo, 9))


def test_day_one_comes_from_the_morning_anchored_model_when_readings_exist(trained_oof):
    hist, raw, src, art = trained_oof
    day = pd.Timestamp(hist["gauge"].index[5 * 365 + 30])
    data = dataset.build_data(hist)

    fc = train.predict(art, hist, raw, src, issue_day=day)

    assert art.model.day1.n_train > 0
    d = train._forecast_data(train._with_gr4j(hist, art.gr4j), hist, raw, src, art.gr4j, art.scale, art.horizons)
    want = art.model.day1.point(d, pd.DatetimeIndex([day]))["h1"].iloc[0]
    assert np.isfinite(want) and fc["p50"].iloc[0] == pytest.approx(want)
    assert abs(want - data["stage_am"].loc[day]) < 0.5
