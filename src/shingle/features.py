"""Build a model feature matrix from feature names (shared by training and live prediction).

Name grammar (matches analysis_run.candidate_features):
  stage, dstage_{k}, log_q, log_q_up, doy_sin, doy_cos
  pr_past{w}__{cell}   log1p rain sum over t-w+1..t
  pr_next{h}__{cell}   log1p rain sum over t+1..t+h  (observed in training, forecast live)
  pr_nextH__{cell}     same, with h = the `horizon` argument
  wb_past{w}__{cell}   sum of (rain - ET0) over t-w+1..t

  log_q_d{k}           log1p(Q) change over k days
  toho, dtoho_{k}, toho_dev (toho - regulation target), etoho
  toho_target_dH       target(t+h) - target(t)  (schedule known in advance)
  gr4j_logq, gr4j_s, gr4j_r  GR4J simulated log1p flow / store levels at t
  gr4j_dH              GR4J simulated log-flow change t -> t+h (driven by future rain)
  rad_<rain feature>    any pr_/wb_ feature computed on data['radar'] (MRMS regions)
  log_s61, log_s59     log1p structure outflows (S-61 Toho, S-59 East Toho)

`data` keys: stage, q, q_up (Series); pr, pet (wide per-cell frames);
toho, s61_q, s59_q, etoho (Series) when lake features are used.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from home_display.shingle import analysis, schedule

_CELL = re.compile(r"^(pr_past|pr_next|wb_past)(\d+|H)__(.+)$")
_DSTAGE = re.compile(r"^dstage_(\d+)$")
_DTOHO = re.compile(r"^dtoho_(\d+)$")
_LOGQ_D = re.compile(r"^log_q_d(\d+)$")


def _one(data: dict, name: str, horizon: int | None, index: pd.DatetimeIndex) -> pd.Series:
    if name.startswith("rad_"):
        # same grammar on the radar (MRMS region) frame
        radar = dict(data, pr=data["radar"], pet=data.get("radar_pet", data["pet"]))
        return _one(radar, name[4:], horizon, index)
    if name == "stage":
        return data["stage"]
    if m := _DSTAGE.match(name):
        return data["stage"].diff(int(m.group(1)))
    if name == "log_q":
        return np.log1p(data["q"])
    if m := _LOGQ_D.match(name):
        return np.log1p(data["q"]).diff(int(m.group(1)))
    if name == "toho":
        return data["toho"]
    if m := _DTOHO.match(name):
        return data["toho"].diff(int(m.group(1)))
    if name == "toho_dev":
        return data["toho"] - schedule.toho_target_series(index)
    if name == "toho_target_dH":
        # the regulation schedule is published in advance -> not lookahead
        if horizon is None:
            raise ValueError(f"{name} needs a horizon")
        later = schedule.toho_target(index + pd.Timedelta(days=horizon))
        return pd.Series(later - schedule.toho_target(index), index=index)
    if name == "gr4j_logq":
        return np.log1p(data["gr4j"]["q"])
    if name == "gr4j_dH":
        if horizon is None:
            raise ValueError(f"{name} needs a horizon")
        lq = np.log1p(data["gr4j"]["q"])
        return lq.shift(-horizon) - lq
    if name in ("gr4j_s", "gr4j_r"):
        return data["gr4j"][name.split("_")[1]]
    if name == "log_s61":
        return np.log1p(data["s61_q"].clip(lower=0))
    if name == "log_s59":
        return np.log1p(data["s59_q"].clip(lower=0))
    if name == "etoho":
        return data["etoho"]
    if name == "log_q_up":
        return np.log1p(data["q_up"])
    if name in ("doy_sin", "doy_cos"):
        angle = 2 * np.pi * index.day_of_year / 365.25
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
