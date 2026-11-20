"""Evaluation utilities shared by every candidate model (and the NWM benchmark)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_split(
    index: pd.DatetimeIndex, train_end: str = "2015-12-31", val_end: str = "2019-12-31", gap_days: int = 0
) -> dict[str, pd.DatetimeIndex]:
    """Chronological train/val/test split.

    gap_days drops the last `gap_days` days before each boundary so that
    h-day-ahead targets of one split never fall inside the next split.
    """
    te, ve = pd.Timestamp(train_end), pd.Timestamp(val_end)
    gap = pd.Timedelta(days=gap_days)
    return {
        "train": index[index <= te - gap],
        "val": index[(index > te) & (index <= ve - gap)],
        "test": index[index > ve],
    }


def _aligned(*series):
    """Drop rows where any input is NaN; Series align on index, arrays by position."""
    if all(isinstance(s, pd.Series) for s in series):
        df = pd.concat([s.rename(i) for i, s in enumerate(series)], axis=1)
    else:
        df = pd.DataFrame({i: np.asarray(s, float) for i, s in enumerate(series)})
    df = df.dropna()
    return [df[i].astype(float) for i in range(len(series))]


def point_metrics(y, p, high: float = 59.0, low: float | None = None) -> dict:
    y, p = _aligned(y, p)
    err = p - y
    out = {
        "n": int(len(y)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias": float(err.mean()),
        "mae_high": float(err[y >= high].abs().mean()) if (y >= high).any() else float("nan"),
        "n_high": int((y >= high).sum()),
    }
    if low is not None:
        out["mae_low"] = float(err[y <= low].abs().mean()) if (y <= low).any() else float("nan")
        out["n_low"] = int((y <= low).sum())
    return out


def coverage(y, lo, hi) -> float:
    y, lo, hi = _aligned(y, lo, hi)
    return float(((y >= lo) & (y <= hi)).mean())


def threshold_skill(y, p, threshold: float) -> dict:
    y, p = _aligned(y, p)
    obs, pred = y >= threshold, p >= threshold
    hits = int((obs & pred).sum())
    misses = int((obs & ~pred).sum())
    fa = int((~obs & pred).sum())
    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": fa,
        "pod": hits / (hits + misses) if hits + misses else float("nan"),
        "far": fa / (hits + fa) if hits + fa else float("nan"),
    }


def event_peaks(obs: pd.Series, pred: pd.Series, threshold: float, window_days: int = 3) -> pd.DataFrame:
    """One row per observed exceedance event: obs vs predicted peak height and timing."""
    above = (obs >= threshold).to_numpy()
    rows = []
    i, n = 0, len(obs)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and above[j + 1]:
            j += 1
        seg = obs.iloc[i : j + 1]
        win = pred.iloc[max(0, i - window_days) : min(n, j + window_days + 1)]
        t_obs, t_pred = seg.idxmax(), win.idxmax()
        rows.append(
            {
                "start": obs.index[i],
                "end": obs.index[j],
                "obs_peak": float(seg.max()),
                "obs_peak_date": t_obs,
                "pred_peak": float(win.max()),
                "pred_peak_date": t_pred,
                "peak_error": float(win.max() - seg.max()),
                "timing_error_days": int((t_pred - t_obs) / pd.Timedelta(days=1)),
            }
        )
        i = j + 1
    return pd.DataFrame(rows)


def conformal_offsets(residuals: pd.Series, coverage: float = 0.8) -> tuple[float, float]:
    """Split-conformal: empirical residual (obs - pred) quantiles giving the target central coverage."""
    r = pd.Series(residuals).dropna().to_numpy()
    a = (1 - coverage) / 2
    return float(np.quantile(r, a)), float(np.quantile(r, 1 - a))


def cross_conformal_predict(model, data: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Leave-one-year-out conformal: each year is predicted with intervals
    calibrated on the remaining years of `index`, so reported coverage is honest."""
    parts = []
    years = sorted(set(index.year))
    for yr in years:
        held = index[index.year == yr]
        cal = index[index.year != yr]
        model.calibrate(data, cal)
        parts.append(model.predict(data, held))
    return pd.concat(parts).loc[index]
