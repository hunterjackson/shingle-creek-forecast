import json

import numpy as np
import pandas as pd
import pytest

from shingle import fetch, forecast_log, webdata

NOW = pd.Timestamp("2024-10-12 15:00", tz="UTC")


def test_observed_doc_keeps_recent_daily_and_hourly_windows_with_nulls():
    days = pd.date_range("2024-09-01", "2024-10-11")
    stage = pd.Series(np.linspace(55, 59, len(days)), index=days)
    stage.loc["2024-10-05"] = np.nan
    toho = pd.Series(54.0, index=days)
    hours = pd.date_range(pd.Timestamp("2024-10-01", tz="UTC"), NOW, freq="h")
    hourly = pd.Series(56.0, index=hours)

    doc = webdata.observed_doc(stage, hourly, toho, NOW, days=10, hours=48)

    assert [r["date"] for r in doc["daily"]][0] == "2024-10-03"
    assert doc["daily"][-1] == {"date": "2024-10-11", "stage": 59.0, "toho": 54.0}
    assert next(r for r in doc["daily"] if r["date"] == "2024-10-05")["stage"] is None
    assert len(doc["hourly"]) == 49
    assert doc["hourly"][-1]["time"] == "2024-10-12T15:00:00Z"


def _log(tmp_path):
    path = tmp_path / "log.csv"
    for issue, p50s, nwm in (("2024-10-09", [55.0, 56.0], 54.0), ("2024-10-10", [56.0, 58.0], 57.0)):
        days = pd.date_range(pd.Timestamp(issue) + pd.Timedelta(days=1), periods=len(p50s))
        doc = {"issue_day": issue, "forecast": [{"date": d.strftime("%Y-%m-%d"), "p10": p - 0.5, "p50": p, "p90": p + 0.5} for d, p in zip(days, p50s)]}
        forecast_log.append(path, doc, nwm_stage_h1=nwm)
    return path


def test_score_frame_matches_score_from_path(tmp_path):
    path = _log(tmp_path)
    obs = pd.Series([55.5, 56.0, 57.0], index=pd.date_range("2024-10-10", periods=3))

    a = forecast_log.score(path, obs)
    b = forecast_log.score_frame(pd.read_csv(path), obs)

    pd.testing.assert_frame_equal(a, b)


def test_scorecard_doc_has_per_horizon_stats_and_recent_verification_rows(tmp_path):
    log = pd.read_csv(_log(tmp_path))
    obs = pd.Series([55.5, 56.0, 57.0], index=pd.date_range("2024-10-10", periods=3))

    doc = webdata.scorecard_doc(log, obs, NOW, days=60)

    h1 = next(r for r in doc["by_h"] if r["h"] == 1)
    assert h1["n"] == 2 and h1["mae"] == pytest.approx(0.25) and h1["nwm_mae"] == pytest.approx(1.25)
    assert len(doc["verify"]) == 4  # every logged forecast, obs null where not yet observed
    last = [r for r in doc["verify"] if r["date"] == "2024-10-12"][0]
    assert last == {"date": "2024-10-12", "issue_day": "2024-10-10", "h": 2, "p10": 57.5, "p50": 58.0, "p90": 58.5, "obs": 57.0, "nwm_p50": None}


def test_scorecard_doc_empty_log_is_valid():
    doc = webdata.scorecard_doc(pd.DataFrame(columns=forecast_log.COLUMNS), pd.Series(dtype=float), NOW)
    assert doc == {"by_h": [], "verify": []}


def test_write_all_emits_strict_json(tmp_path):
    webdata.write_all(tmp_path, {"x": {"a": float("nan"), "b": [1.0, np.float64(np.nan)], "c": np.int64(3)}})

    assert json.loads((tmp_path / "x.json").read_text(), parse_constant=lambda c: pytest.fail(c)) == {"a": None, "b": [1.0, None], "c": 3}


def test_build_writes_forecast_observed_and_scorecard(tmp_path, monkeypatch):
    days = pd.date_range("2024-09-01", "2024-10-10")
    gauge = pd.DataFrame({"stage_ft": 55.0, "discharge_cfs": 100.0}, index=days)
    hours = pd.date_range("2024-10-09 05:00", "2024-10-12 14:00", freq="h", tz="UTC")
    iv = pd.DataFrame({"stage_ft": 56.0, "discharge_cfs": 200.0}, index=hours)
    lakes = pd.DataFrame({"toho_stage": 54.0}, index=days)
    monkeypatch.setattr(fetch, "load_history", lambda d: {"gauge": gauge, "lakes": lakes, "iv": {"gauge": iv}})
    fc = tmp_path / "shingle.json"
    fc.write_text(json.dumps({"issue_day": "2024-10-11", "forecast": []}))
    out = tmp_path / "site"

    webdata.build(tmp_path, fc, _log(tmp_path), out, now=NOW)

    f = json.loads((out / "forecast.json").read_text())
    assert f["issue_day"] == "2024-10-11" and f["generated_at"] == "2024-10-12T15:00:00Z"
    o = json.loads((out / "observed.json").read_text())
    assert o["daily"][-1] == {"date": "2024-10-11", "stage": 56.0, "toho": None}  # unpublished dv day filled from IV
    s = json.loads((out / "scorecard.json").read_text())
    assert {r["h"] for r in s["by_h"]} == {1, 2}


def test_build_without_log_writes_empty_scorecard(tmp_path, monkeypatch):
    days = pd.date_range("2024-10-01", "2024-10-10")
    monkeypatch.setattr(fetch, "load_history", lambda d: {"gauge": pd.DataFrame({"stage_ft": 55.0}, index=days),
                                                          "lakes": pd.DataFrame({"toho_stage": 54.0}, index=days), "iv": {"gauge": None}})
    fc = tmp_path / "shingle.json"
    fc.write_text(json.dumps({"issue_day": "2024-10-11"}))

    webdata.build(tmp_path, fc, tmp_path / "missing.csv", tmp_path / "site", now=NOW)

    assert json.loads((tmp_path / "site" / "scorecard.json").read_text()) == {"by_h": [], "verify": []}
