"""Forecast JSON for the e-ink dashboard layer (out/shingle.json)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from shingle import config

_ORDER = ("major", "moderate", "minor", "action")


def flood_category(stage: float) -> str:
    if stage is None or pd.isna(stage):
        return "unknown"
    for cat in _ORDER:
        if stage >= config.FLOOD_STAGES[cat]:
            return cat
    return "normal"


def _r(x: float, nd: int = 2):
    return None if x is None or pd.isna(x) else round(float(x), nd)


def write_json(path, fc: pd.DataFrame, current_stage: float, current_time: pd.Timestamp, issue_day: pd.Timestamp,
               rain_mm: pd.Series | None = None, meta: dict | None = None) -> dict:
    rows = []
    for _, r in fc.iterrows():
        row = {"date": pd.Timestamp(r["date"]).strftime("%Y-%m-%d"), "p10": _r(r["p10"]), "p50": _r(r["p50"]), "p90": _r(r["p90"]),
               "category": flood_category(r["p50"]), "p90_category": flood_category(r["p90"])}
        if rain_mm is not None:
            row["rain_mm"] = _r(rain_mm.get(pd.Timestamp(r["date"]), float("nan")), 1)
        rows.append(row)
    doc = {
        "site": {"usgs": config.SITE, "nws": config.NWS_LID, "name": "Shingle Creek at Campbell, FL"},
        "issue_day": pd.Timestamp(issue_day).strftime("%Y-%m-%d"),
        "current": {"stage_ft": _r(current_stage), "time": pd.Timestamp(current_time).isoformat(), "category": flood_category(current_stage)},
        "flood_stages": {k: float(config.FLOOD_STAGES[k]) for k in ("action", "minor", "moderate", "major")},
        "forecast": rows,
        "model": dict(meta or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))
    return doc
