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
