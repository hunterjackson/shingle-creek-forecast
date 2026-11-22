"""JSON files behind the static forecast site (and the e-ink screens).

    forecast.json   the issued forecast (output.write_json doc) + generated_at
    observed.json   recent daily stage + Lake Toho, and hourly stage
    scorecard.json  running verification of logged forecasts (ours vs NWM h1)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from shingle import fetch, forecast_log, live


def _iso(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(x, nd: int = 2):
    return None if x is None or pd.isna(x) else round(float(x), nd)


def observed_doc(stage: pd.Series, hourly: pd.Series, toho: pd.Series, now: pd.Timestamp, days: int = 45, hours: int = 240) -> dict:
    """Daily stage/Toho for the `days` complete days before today (UTC date of `now`), hourly stage for the last `hours`."""
    today = pd.Timestamp(now).tz_convert("UTC").tz_localize(None).normalize()
    dates = pd.date_range(today - pd.Timedelta(days=days - 1), today - pd.Timedelta(days=1))
    s, t = stage.reindex(dates), toho.reindex(dates)
    daily = [{"date": d.strftime("%Y-%m-%d"), "stage": _num(s[d]), "toho": _num(t[d])} for d in dates]
    recent = hourly[hourly.index >= pd.Timestamp(now) - pd.Timedelta(hours=hours)].sort_index()
    return {"daily": daily, "hourly": [{"time": _iso(ts), "stage": _num(v)} for ts, v in recent.items()]}


def scorecard_doc(log: pd.DataFrame, observed: pd.Series, now: pd.Timestamp, days: int = 60) -> dict:
    """Per-horizon stats over the whole log + every logged forecast whose target date is within `days`."""
    if log.empty:
        return {"by_h": [], "verify": []}
    stats = forecast_log.score_frame(log, observed)
    by_h = [{"h": int(h), "n": int(r["n"]), "mae": _num(r["mae"], 3), "bias": _num(r["bias"], 3), "cov80": _num(r["cov80"], 3),
             "nwm_mae": _num(r["nwm_mae"], 3)} for h, r in stats.iterrows()]
    lg = log.copy()
    lg["date"] = pd.to_datetime(lg["date"])
    today = pd.Timestamp(now).tz_convert("UTC").tz_localize(None).normalize()
    lg = lg[lg["date"] > today - pd.Timedelta(days=days)].sort_values(["date", "h"])
    verify = [{"date": r.date.strftime("%Y-%m-%d"), "issue_day": str(r.issue_day), "h": int(r.h), "p10": _num(r.p10), "p50": _num(r.p50),
               "p90": _num(r.p90), "obs": _num(observed.get(r.date)), "nwm_p50": _num(r.nwm_p50)} for r in lg.itertuples()]
    return {"by_h": by_h, "verify": verify}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if math.isnan(o) else float(o)
    return o


def write_all(out_dir, docs: dict[str, dict]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, doc in docs.items():
        (out / f"{name}.json").write_text(json.dumps(_clean(doc), allow_nan=False, separators=(",", ":")))


def build(data_dir, forecast_path, log_path, out_dir, now: pd.Timestamp | None = None) -> None:
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    hist = fetch.load_history(data_dir)
    stage = hist["gauge"]["stage_ft"]
    iv = hist["iv"]["gauge"]
    hourly = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    if iv is not None and len(iv):
        hourly = iv["stage_ft"].dropna()
        stage = stage.combine_first(live.daily_from_iv(iv[["stage_ft"]])["stage_ft"])
    fc = json.loads(Path(forecast_path).read_text())
    fc["generated_at"] = _iso(now)
    log_path = Path(log_path)
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame(columns=forecast_log.COLUMNS)
    write_all(out_dir, {
        "forecast": fc,
        "observed": observed_doc(stage, hourly, hist["lakes"]["toho_stage"], now),
        "scorecard": scorecard_doc(log, stage, now),
    })
