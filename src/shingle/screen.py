"""Cheap screening of candidate data sources against a model's out-of-fold errors.

Idea: if a candidate series carries information the current model lacks, it
must correlate with that model's *out-of-sample* residuals. Correlating is
seconds of work, so dozens of sources can be ruled out before anyone builds
features, pulls full histories or retrains.

    pred  = oof_point(make_model, data, index)       # blocked leave-years-out
    resid = residuals(data, pred)                    # obs(t+h) - pred
    table = residual_screen(resid, expand(series, "gw"))
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def oof_point(make_model, data: dict, index: pd.DatetimeIndex, block_years: int = 1, gap_days: int = 7) -> pd.DataFrame:
    """Out-of-fold point forecasts: each block of years predicted by a model fit on the rest.

    gap_days removes training rows just before a held block (their h-day targets
    fall inside it).
    """
    years = sorted(set(index.year))
    blocks = [years[i : i + block_years] for i in range(0, len(years), block_years)]
    parts = []
    for blk in blocks:
        held = index[index.year.isin(blk)]
        start = pd.Timestamp(f"{blk[0]}-01-01")
        near = (index < start) & (index >= start - pd.Timedelta(days=gap_days))
        train = index[~index.year.isin(blk) & ~near]
        model = make_model().fit(data, train)
        parts.append(model.point(data, held))
    return pd.concat(parts).loc[index]


def residuals(data: dict, pred: pd.DataFrame) -> pd.DataFrame:
    """obs stage(t+h) - forecast issued at t, per 'h{h}' column."""
    stage = data["stage"]
    out = {}
    for col in pred.columns:
        h = int(col[1:])
        out[col] = stage.shift(-h).reindex(pred.index) - pred[col]
    return pd.DataFrame(out, index=pred.index)


def expand(series: pd.Series, name: str, diffs=(1, 3, 7), anom=(30, 365)) -> pd.DataFrame:
    """Level, k-day changes and departure from a trailing mean: the forms in which
    a state variable could add information."""
    cols = {name: series}
    for k in diffs:
        cols[f"{name}_d{k}"] = series.diff(k)
    for w in anom:
        cols[f"{name}_anom{w}"] = series - series.rolling(w, min_periods=w // 2).mean()
    return pd.DataFrame(cols)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    xc, yc = x - x.mean(), y - y.mean()
    den = np.sqrt((xc**2).sum() * (yc**2).sum())
    return float((xc * yc).sum() / den) if den > 0 else 0.0


def residual_screen(
    resid: pd.DataFrame, cands: pd.DataFrame, n_boot: int = 200, block_days: int = 90, ci: float = 0.9, seed: int = 0
) -> pd.DataFrame:
    """Correlation of every candidate with every horizon's residual, with a
    moving-block bootstrap CI (keeps storm/season autocorrelation). `signal` =
    CI excludes zero."""
    rng = np.random.default_rng(seed)
    a = (1 - ci) / 2
    rows = []
    for c in cands.columns:
        for col in resid.columns:
            df = pd.concat([cands[c], resid[col]], axis=1, keys=["x", "y"]).dropna()
            n = len(df)
            if n < 30:
                continue
            x, y = df["x"].to_numpy(float), df["y"].to_numpy(float)
            r = _corr(x, y)
            rho = _corr(df["x"].rank().to_numpy(), df["y"].rank().to_numpy())
            b = min(block_days, n)
            boots = []
            for _ in range(n_boot):
                starts = rng.integers(0, n - b + 1, size=int(np.ceil(n / b)))
                pos = (starts[:, None] + np.arange(b)).ravel()[:n]
                boots.append(_corr(x[pos], y[pos]))
            lo, hi = np.quantile(boots, [a, 1 - a])
            rows.append((c, int(col[1:]), n, r, rho, float(lo), float(hi), bool(lo > 0 or hi < 0)))
    out = pd.DataFrame(rows, columns=["candidate", "h", "n", "r", "rho", "r_lo", "r_hi", "signal"])
    return out.reindex(out["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)
