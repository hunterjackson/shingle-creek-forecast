import numpy as np
import pandas as pd
import pytest

from home_display.shingle import dataset, train
from home_display.shingle.models import base

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
