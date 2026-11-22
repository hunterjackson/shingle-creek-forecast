import json

import pandas as pd
import pytest

from home_display.shingle import cli, pipeline, train

from synth import make_world


def test_latest_issue_day_is_yesterday_after_13z_else_day_before():
    assert pipeline.latest_issue_day(pd.Timestamp("2024-10-10 13:30", tz="UTC")) == pd.Timestamp("2024-10-09")
    assert pipeline.latest_issue_day(pd.Timestamp("2024-10-10 08:00", tz="UTC")) == pd.Timestamp("2024-10-08")


@pytest.fixture(scope="module")
def world():
    hist, raw, src = make_world()
    idx = hist["gauge"].index
    art = train.train(hist, raw, src, train_end=str(idx[3 * 365].date()), fcst_start=str(idx[365].date()), horizons=(1, 2, 3, 4, 5, 6, 7),
                      gr4j_maxiter=10, oof_years=False)
    return hist, raw, src, art


def test_forecast_doc_end_to_end(world, tmp_path):
    hist, raw, src, art = world
    issue = hist["gauge"].index[3 * 365 + 30]
    h = {k: (v.loc[:issue] if isinstance(v, pd.DataFrame) and k != "cells" else v) for k, v in hist.items()}
    # 15-min feed (hourly here) through the morning after the issue day; dv for the issue day not yet published
    t = pd.date_range(h["gauge"].index[0], issue + pd.Timedelta(days=1, hours=13), freq="1h", tz="UTC")
    g = hist["gauge"].reindex(t.tz_localize(None).normalize()).set_axis(t)
    h["iv"] = {"gauge": g, "upstream": hist["upstream"].reindex(t.tz_localize(None).normalize()).set_axis(t)}
    h["gauge"] = h["gauge"].iloc[:-1]
    h["pr"] = h["pr"].iloc[:-2]        # gridMET lags two days
    radar = hist["pr"].loc[:issue]
    out = tmp_path / "shingle.json"

    doc = pipeline.forecast_doc(art, h, raw, src, radar_cells=radar, issue_day=issue, out_path=out,
                                current=(56.1, pd.Timestamp(issue) + pd.Timedelta(hours=36)))

    assert json.loads(out.read_text()) == doc
    assert len(doc["forecast"]) == 7 and doc["forecast"][0]["date"] == (issue + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    assert all(r["p10"] is not None and r["p10"] <= r["p50"] <= r["p90"] for r in doc["forecast"])
    assert all(r["rain_mm"] is not None for r in doc["forecast"])
    assert doc["current"]["stage_ft"] == 56.1
    assert doc["model"]["train_end"] == art.train_end
    assert doc["model"]["live_fill"]["rain_from"] is not None


def test_cli_routes_train_and_predict(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(pipeline, "train_from_disk", lambda data_dir, model_path, train_end: seen.append(("train", train_end)))
    monkeypatch.setattr(pipeline, "predict_from_disk", lambda data_dir, model_path, out_path, update=True: seen.append(("predict", update)))

    cli.main(["train", "--data-dir", str(tmp_path), "--train-end", "2023-12-31"])
    cli.main(["predict", "--data-dir", str(tmp_path), "--no-update"])

    assert seen == [("train", "2023-12-31"), ("predict", False)]
