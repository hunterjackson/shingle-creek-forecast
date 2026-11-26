from shingle import config, spec
from shingle.models import arx, blend, hybrid


def test_build_model_is_blend_of_arx_and_rating_hybrid():
    m = spec.build_model(horizons=(1, 4))

    assert isinstance(m, blend.Blend) and m.weights == {1: 0.5, 4: 0.5}
    assert isinstance(m.a, arx.ARX) and isinstance(m.b, hybrid.RatingHybrid)
    assert m.b.rating_opts == spec.RATING_OPTS


def test_morning_features_only_for_short_horizons_and_allowed_missing():
    m = spec.build_model(horizons=(1, 3, 4, 7))
    a, flow = m.a, m.b.flow

    assert "stage_am_d" in a.feature_names(1) and "stage_am_d" in a.feature_names(3)
    assert "stage_am_d" not in a.feature_names(4)
    assert "log_q_am_d" in flow.feature_names(3) and "log_q_am_d" not in flow.feature_names(7)
    assert {"stage_am_d", "log_q_am_d", "log_q_up_am_d"} <= a.reg.allow_missing
    assert a.reg.lgbm_params == spec.LGBM_PARAMS


def test_rain_features_use_configured_cell_and_nofut_strips_future():
    m = spec.build_model(horizons=(3,))
    assert f"pr_nextH__{config.RAIN_CELL}" in m.a.feature_names(3)
    assert "gr4j_dH" in m.b.flow.feature_names(3)

    n = spec.build_model(horizons=(3,), future_rain=False)
    names = n.a.feature_names(3) + n.b.flow.feature_names(3) + n.b.lake.feature_names(3)
    assert not any("next" in f for f in names) and "gr4j_dH" not in names


def test_day_one_is_anchored_on_the_morning_reading():
    from shingle.models import anchor

    m = spec.build_model(horizons=(1, 4))

    assert isinstance(m.day1, anchor.MorningAnchor)
    assert {"stage_am_d", "stage_sofar_d", f"pr_nextH__{config.RAIN_CELL}", "stage_slope6", "up_slope12"} <= set(m.day1.features)
    assert spec.build_model(horizons=(2, 4)).day1 is None
    assert not any("next" in f for f in spec.build_model(horizons=(1,), future_rain=False).day1.features)
