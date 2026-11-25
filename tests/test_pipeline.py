import json

import pandas as pd
import pytest

from shingle import cli, pipeline, train

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
    monkeypatch.setattr(pipeline, "predict_from_disk", lambda data_dir, model_path, out_path, update=True, log_path=None: seen.append(("predict", update)))

    cli.main(["train", "--data-dir", str(tmp_path), "--train-end", "2023-12-31"])
    cli.main(["predict", "--data-dir", str(tmp_path), "--no-update"])

    assert seen == [("train", "2023-12-31"), ("predict", False)]


def test_nwm_stage_h1_converts_short_range_flow_for_the_next_local_day():
    import json
    from pathlib import Path

    from shingle import rating
    from shingle.sources import nwm

    payload = json.loads((Path(__file__).parent / "fixtures" / "nwm_short_range.json").read_text())
    df = nwm.parse_streamflow(payload)
    day1 = nwm.daily_mean(df).index[0]
    r = rating.CombinedRating(c=1e-5, d=1.0, h0=47.0, a=0.5, b=0.43, k=1.0)

    st = pipeline.nwm_stage_h1(r, issue_day=day1 - pd.Timedelta(days=1), toho=54.0, fetch=lambda: df)

    q = nwm.daily_mean(df).loc[day1]
    assert st == pytest.approx(float(r.predict_stage([q], [54.0])[0]))
    assert pipeline.nwm_stage_h1(r, issue_day=day1 + pd.Timedelta(days=5), toho=54.0, fetch=lambda: df) is None
    assert pipeline.nwm_stage_h1(r, issue_day=day1, toho=54.0, fetch=lambda: (_ for _ in ()).throw(RuntimeError("down"))) is None


def test_cli_score_prints_table(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(pipeline, "score_from_disk", lambda data_dir, log: pd.DataFrame({"n": [3], "mae": [0.1]}, index=pd.Index([1], name="h")))

    cli.main(["score", "--data-dir", str(tmp_path)])

    assert "mae" in capsys.readouterr().out


def test_forecast_rain_shows_unscaled_gefs_basin_rain():
    """The site's rain column is the issued GEFS basin rain, not the per-lead bias scale (~1.4x) that only the
    day 4-7 models use: it must match what other forecasts show and what days 1-3 were driven by."""
    from types import SimpleNamespace

    import numpy as np

    day = pd.Timestamp("2026-11-24")
    src = pd.DataFrame({"lat": [28.0, 28.0, 28.5, 28.5], "lon": [-81.5, -81.0, -81.5, -81.0]},
                       index=["28.0000_-81.5000", "28.0000_-81.0000", "28.5000_-81.5000", "28.5000_-81.0000"])
    cols = pd.MultiIndex.from_tuples([(k, c) for k in (1, 2) for c in src.index], names=["lead", "cell"])
    raw = pd.DataFrame([[10.0] * 4 + [2.0] * 4], index=pd.DatetimeIndex([day], name="date"), columns=cols)
    cells = pd.DataFrame({"lat": [28.25], "lon": [-81.25]}, index=["28.2500_-81.2500"])
    art = SimpleNamespace(gr4j=SimpleNamespace(weights={"28.2500_-81.2500": 1.0}), scale={1: 1.4, 2: 1.4})
    rain = pipeline.forecast_rain(art, cells, raw, src, day)
    assert np.allclose(rain.to_numpy(), [10.0, 2.0])
    assert list(rain.index) == [day + pd.Timedelta(days=1), day + pd.Timedelta(days=2)]
