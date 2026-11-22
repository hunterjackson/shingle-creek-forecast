import pandas as pd
import pytest

from home_display.shingle import forecast_log


def doc(issue, p50s):
    days = pd.date_range(pd.Timestamp(issue) + pd.Timedelta(days=1), periods=len(p50s))
    return {"issue_day": issue, "forecast": [{"date": d.strftime("%Y-%m-%d"), "p10": p - 0.5, "p50": p, "p90": p + 0.5} for d, p in zip(days, p50s)]}


def test_append_writes_one_row_per_horizon_and_replaces_same_issue_day(tmp_path):
    path = tmp_path / "log.csv"

    forecast_log.append(path, doc("2024-10-09", [55.0, 56.0]), nwm_stage_h1=55.4)
    forecast_log.append(path, doc("2024-10-09", [55.1, 56.1]), nwm_stage_h1=55.5)  # rerun same morning
    forecast_log.append(path, doc("2024-10-10", [56.5, 57.0]), nwm_stage_h1=None)

    log = pd.read_csv(path)
    assert len(log) == 4
    assert log.loc[(log.issue_day == "2024-10-09") & (log.h == 1), "p50"].item() == 55.1
    assert log.loc[(log.issue_day == "2024-10-09") & (log.h == 1), "nwm_p50"].item() == 55.5
    assert pd.isna(log.loc[(log.issue_day == "2024-10-09") & (log.h == 2), "nwm_p50"].item())


def test_score_compares_model_and_nwm_with_observations(tmp_path):
    path = tmp_path / "log.csv"
    forecast_log.append(path, doc("2024-10-09", [55.0, 56.0]), nwm_stage_h1=54.0)
    forecast_log.append(path, doc("2024-10-10", [56.0, 58.0]), nwm_stage_h1=57.0)
    obs = pd.Series([55.5, 56.0, 57.0], index=pd.date_range("2024-10-10", periods=3))

    s = forecast_log.score(path, obs)

    assert s.loc[1, "n"] == 2
    assert s.loc[1, "mae"] == pytest.approx((0.5 + 0.0) / 2)
    assert s.loc[1, "nwm_mae"] == pytest.approx((1.5 + 1.0) / 2)
    assert s.loc[2, "mae"] == pytest.approx(0.5) and s.loc[2, "n"] == 2
    assert s.loc[1, "cov80"] == pytest.approx(1.0)
    obs2 = obs.iloc[:2]  # 2024-10-12 not observed yet -> only one h2 forecast verifiable
    assert forecast_log.score(path, obs2).loc[2, "n"] == 1
