"""Run the cell-relevance analysis on real history and persist the selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from home_display.shingle import analysis

PAST_WINDOWS = (1, 3, 7, 14, 30, 60)
WB_WINDOWS = (30, 90)
HORIZONS = (1, 3, 7)


def candidate_features(
    pr: pd.DataFrame,
    pet: pd.DataFrame,
    past_windows=PAST_WINDOWS,
    horizon: int = 1,
    wb_windows=WB_WINDOWS,
) -> pd.DataFrame:
    """Per-cell rain/water-balance candidates. Column = '{kind}__{cell_id}'."""
    parts = []
    for w in past_windows:
        parts.append(np.log1p(analysis.trailing_sum(pr, w)).add_prefix(f"pr_past{w}__"))
    parts.append(np.log1p(analysis.future_sum(pr, horizon)).add_prefix(f"pr_next{horizon}__"))
    wb = pr - pet
    for w in wb_windows:
        parts.append(analysis.trailing_sum(wb, w).add_prefix(f"wb_past{w}__"))
    return pd.concat(parts, axis=1)


def state_controls(gauge: pd.DataFrame, upstream: pd.DataFrame) -> pd.DataFrame:
    stage = gauge["stage_ft"]
    return pd.DataFrame(
        {
            "stage": stage,
            "dstage_1": stage.diff(1),
            "dstage_3": stage.diff(3),
            "log_q": np.log1p(gauge["discharge_cfs"]),
            "log_q_up": np.log1p(upstream["discharge_cfs"].reindex(gauge.index)),
        }
    )


def run(
    history: dict,
    horizons=HORIZONS,
    past_windows=PAST_WINDOWS,
    train_end: str = "2015-12-31",
    max_k: int = 15,
    min_partial_r: float = 0.03,
    cluster_threshold: float = 0.9,
    target: str = "stage",
) -> dict:
    """Selection uses only data <= train_end, so later evaluation periods stay untouched."""
    cut = pd.Timestamp(train_end)
    gauge = history["gauge"].loc[:cut]
    upstream = history["upstream"].loc[:cut]
    pr = history["pr"].loc[:cut]
    pet = history["pet"].loc[:cut]

    S = state_controls(gauge, upstream)
    if target == "stage":
        level = gauge["stage_ft"]
    elif target == "log_q":
        level = np.log1p(gauge["discharge_cfs"])
    else:
        raise ValueError(f"unknown target {target!r}")
    selection, scores = {}, {}
    for h in horizons:
        y = analysis.future_change(level, h)
        X = candidate_features(pr, pet, past_windows, horizon=h)
        data = S.join(X, how="inner")
        selection[h] = analysis.greedy_select(
            data, y, max_k=len(S.columns) + max_k, min_partial_r=min_partial_r, forced=list(S.columns)
        )
        nxt = X.filter(like=f"pr_next{h}__")
        scores[h] = analysis.univariate_scores(nxt.rename(columns=lambda c: c.split("__")[1]), y)

    dlevel = level.diff()
    return {
        "train_end": train_end,
        "target": target,
        "selection": selection,
        "scores": scores,
        "cross_corr": analysis.cross_corr(pr, dlevel, max_lag=14),
        "clusters": analysis.redundancy_clusters(analysis.trailing_sum(pr, 3).dropna(), cluster_threshold),
    }


def write_selection(res: dict, path: Path) -> dict:
    features = {str(h): sel["feature"].tolist() for h, sel in res["selection"].items()}
    cells = sorted({f.split("__")[1] for fs in features.values() for f in fs if "__" in f})
    out = {"train_end": res["train_end"], "cells": cells, "features": features}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out
