"""Build a model feature matrix from feature names (shared by training and live prediction).

Name grammar (matches analysis_run.candidate_features):
  stage, dstage_{k}, log_q, log_q_up, doy_sin, doy_cos
  pr_past{w}__{cell}   log1p rain sum over t-w+1..t
  pr_next{h}__{cell}   log1p rain sum over t+1..t+h  (observed in training, forecast live)
  pr_nextH__{cell}     same, with h = the `horizon` argument
  wb_past{w}__{cell}   sum of (rain - ET0) over t-w+1..t

`data` keys: stage, q, q_up (Series); pr, pet (wide per-cell frames).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from home_display.shingle import analysis

_CELL = re.compile(r"^(pr_past|pr_next|wb_past)(\d+|H)__(.+)$")
_DSTAGE = re.compile(r"^dstage_(\d+)$")


def _one(data: dict, name: str, horizon: int | None, index: pd.DatetimeIndex) -> pd.Series:
    if name == "stage":
        return data["stage"]
    if m := _DSTAGE.match(name):
        return data["stage"].diff(int(m.group(1)))
    if name == "log_q":
        return np.log1p(data["q"])
    if name == "log_q_up":
        return np.log1p(data["q_up"])
    if name in ("doy_sin", "doy_cos"):
        angle = 2 * np.pi * index.dayofyear / 365.25
        return pd.Series(np.sin(angle) if name == "doy_sin" else np.cos(angle), index=index)
    if m := _CELL.match(name):
        kind, w, cell = m.groups()
        if w == "H":
            if horizon is None:
                raise ValueError(f"{name} needs a horizon")
            w = horizon
        w = int(w)
        if kind == "pr_past":
            return np.log1p(analysis.trailing_sum(data["pr"][[cell]], w)[cell])
        if kind == "pr_next":
            return np.log1p(analysis.future_sum(data["pr"][[cell]], w)[cell])
        wb = data["pr"][[cell]] - data["pet"][[cell]]
        return analysis.trailing_sum(wb, w)[cell]
    raise KeyError(f"unknown feature {name!r}")


def build(data: dict, names: list[str], horizon: int | None = None) -> pd.DataFrame:
    index = data["stage"].index
    cols = {name: _one(data, name, horizon, index).reindex(index) for name in names}
    return pd.DataFrame(cols, index=index)[list(names)]
