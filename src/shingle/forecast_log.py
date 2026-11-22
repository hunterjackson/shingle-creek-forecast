"""Running log of issued forecasts (ours + NWM where available) and their verification.

NWPS only serves the NWM short-range (18 h) series for our reach (medium/long range
come back empty), so NWM is logged for h1 only, converted to stage with our rating.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["issue_day", "h", "date", "p10", "p50", "p90", "nwm_p50"]


def append(path, doc: dict, nwm_stage_h1: float | None = None) -> pd.DataFrame:
    path = Path(path)
    rows = []
    for h, r in enumerate(doc["forecast"], start=1):
        rows.append({"issue_day": doc["issue_day"], "h": h, "date": r["date"], "p10": r["p10"], "p50": r["p50"], "p90": r["p90"],
                     "nwm_p50": nwm_stage_h1 if h == 1 else None})
    new = pd.DataFrame(rows, columns=COLUMNS)
    if path.exists():
        old = pd.read_csv(path)
        old = old[old["issue_day"] != doc["issue_day"]]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(["issue_day", "h"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new


def score(path, observed: pd.Series) -> pd.DataFrame:
    """Per-horizon MAE / bias / 80 % coverage of our forecasts and NWM (h1) against observed daily stage."""
    return score_frame(pd.read_csv(path), observed)


def score_frame(log: pd.DataFrame, observed: pd.Series) -> pd.DataFrame:
    log = log.copy()
    log["date"] = pd.to_datetime(log["date"])
    log["obs"] = log["date"].map(observed)
    log = log.dropna(subset=["obs", "p50"])
    err = log["p50"] - log["obs"]
    nerr = log["nwm_p50"] - log["obs"]
    g = pd.DataFrame({"h": log["h"], "err": err, "abs": err.abs(), "nabs": nerr.abs(),
                      "cov": ((log["obs"] >= log["p10"]) & (log["obs"] <= log["p90"])).astype(float)})
    out = g.groupby("h").agg(n=("err", "size"), mae=("abs", "mean"), bias=("err", "mean"), cov80=("cov", "mean"), nwm_mae=("nabs", "mean"))
    return out
