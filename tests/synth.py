"""Small synthetic world with every input the selected model uses (for end-to-end tests)."""

import numpy as np
import pandas as pd

from shingle import config
from shingle.sources import gridmet


def make_world(n_days: int = 4 * 365, start: str = "2018-01-01", seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_days, name="date")
    cells = pd.DataFrame(
        {"lat": [28.2750, 28.3167, 28.2750], "lon": [-81.4333, -81.4333, -81.4750], "basin_frac": [1.0, 0.6, 0.3]},
        index=[config.RAIN_CELL, "28.3167_-81.4333", "28.2750_-81.4750"],
    )
    rain = rng.gamma(0.35, 10, n_days)
    pr = pd.DataFrame({c: rain * f for c, f in zip(cells.index, (1.0, 0.9, 1.1))}, index=idx)
    pet = pd.DataFrame({c: 3.5 + 1.2 * np.sin(2 * np.pi * idx.day_of_year / 365.25) for c in cells.index}, index=idx)
    logq = np.empty(n_days)
    logq[0] = 4.5
    for t in range(1, n_days):
        logq[t] = logq[t - 1] + 0.025 * rain[t - 1] - 0.08 * (logq[t - 1] - 4.5) + rng.normal(0, 0.02)
    q = np.expm1(logq)
    toho = 53.5 + 0.8 * np.sin(2 * np.pi * idx.day_of_year / 365.25) + 0.0005 * (q - q.mean())
    stage = np.maximum(toho + 3e-5 * q**1.5, 47.2 + 0.56 * q**0.42) + rng.normal(0, 0.02, n_days)
    hist = {
        "gauge": pd.DataFrame({"discharge_cfs": q, "stage_ft": stage}, index=idx),
        "upstream": pd.DataFrame({"discharge_cfs": q * 0.5}, index=idx),
        "lakes": pd.DataFrame({"toho_stage": toho, "s61_q": q * 1.5, "s59_q": 50.0, "etoho_stage": 56.0}, index=idx),
        "pr": pr,
        "pet": pet,
        "cells": cells,
    }
    # GEFS-like raw forecasts on a 0.5 deg grid: noisy truth, issue date t, lead k = day t+k
    src = gridmet.cells_from_columns(["28.0000_-81.5000", "28.0000_-81.0000", "28.5000_-81.5000", "28.5000_-81.0000"])
    cols = pd.MultiIndex.from_tuples([(k, c) for k in range(1, 8) for c in src.index], names=["lead", "cell"])
    raw = pd.DataFrame(
        {(k, c): 0.7 * pd.Series(rain, index=idx).shift(-k) * rng.lognormal(0, 0.3, n_days) for k, c in cols}, index=idx
    )[cols]
    return hist, raw, src
