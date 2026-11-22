from pathlib import Path

import pytest

from shingle import cli, fetch


def test_fetch_subcommand_calls_fetch_history(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(fetch, "fetch_history", lambda data_dir, refresh=False: seen.update(d=data_dir, r=refresh))

    cli.main(["fetch", "--data-dir", str(tmp_path), "--refresh"])

    assert seen == {"d": Path(tmp_path), "r": True}


def test_no_subcommand_exits_with_usage():
    with pytest.raises(SystemExit):
        cli.main([])


def test_predict_subcommand_passes_log_path(monkeypatch, tmp_path):
    from shingle import pipeline

    seen = {}
    monkeypatch.setattr(pipeline, "predict_from_disk", lambda data_dir, model, out, update=True, log_path=None: seen.update(log=log_path, update=update))

    cli.main(["predict", "--log", str(tmp_path / "log.csv"), "--no-update"])

    assert seen == {"log": tmp_path / "log.csv", "update": False}


def test_site_subcommand_builds_site_data(monkeypatch, tmp_path):
    from shingle import webdata

    seen = {}
    monkeypatch.setattr(webdata, "build", lambda data_dir, forecast, log, out_dir: seen.update(d=data_dir, f=forecast, l=log, o=out_dir))

    cli.main(["site", "--data-dir", str(tmp_path), "--forecast", "f.json", "--log", "l.csv", "--out-dir", "site/data"])

    assert seen == {"d": tmp_path, "f": Path("f.json"), "l": Path("l.csv"), "o": Path("site/data")}
