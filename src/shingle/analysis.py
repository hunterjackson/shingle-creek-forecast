"""Relevance analysis for per-cell rain (and other candidate drivers).

Answers: which grid cells / windows carry *independent* information about the
future stage, so the model gets informative, non-redundant inputs instead of
a basin average.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_regression


def trailing_sum(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Sum over days t-window+1 .. t (inclusive of t)."""
    return df.rolling(window, min_periods=window).sum()


def future_sum(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Sum over days t+1 .. t+window (excludes t)."""
    return trailing_sum(df, window).shift(-window)


def future_change(stage: pd.Series, horizon: int) -> pd.Series:
    return stage.shift(-horizon) - stage


def cross_corr(rain: pd.DataFrame, response: pd.Series, max_lag: int) -> pd.DataFrame:
    """Pearson corr of rain at t-lag with response at t, for lag = 0..max_lag."""
    out = {lag: rain.shift(lag).corrwith(response) for lag in range(max_lag + 1)}
    cc = pd.DataFrame(out).T
    cc.index.name = "lag"
    return cc


def univariate_scores(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> pd.DataFrame:
    data = X.join(y.rename("__y__")).dropna()
    Xc, yc = data.drop(columns="__y__"), data["__y__"]
    mi = mutual_info_regression(Xc.values, yc.values, random_state=seed)
    return pd.DataFrame(
        {
            "pearson": Xc.corrwith(yc),
            "spearman": Xc.corrwith(yc, method="spearman"),
            "mutual_info": pd.Series(mi, index=Xc.columns),
        }
    )


def _residualize(M: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Residuals of columns of M after OLS on Z (with intercept)."""
    Z1 = np.column_stack([np.ones(len(Z)), Z]) if Z.size else np.ones((len(M), 1))
    beta, *_ = np.linalg.lstsq(Z1, M, rcond=None)
    return M - Z1 @ beta


def greedy_select(
    X: pd.DataFrame,
    y: pd.Series,
    max_k: int = 10,
    min_partial_r: float = 0.0,
    forced: list[str] | None = None,
) -> pd.DataFrame:
    """Forward selection by partial correlation.

    At each step add the candidate whose residual (given already-selected
    features) correlates most with the residual target. Redundant copies of an
    already-chosen feature have ~zero partial correlation and are skipped.
    """
    forced = list(forced or [])
    data = X.join(y.rename("__y__")).dropna()
    yv = data["__y__"].to_numpy()
    sst = ((yv - yv.mean()) ** 2).sum()
    selected: list[str] = []
    rows = []

    def record(feat: str, r: float) -> None:
        selected.append(feat)
        Z = data[selected].to_numpy()
        res = _residualize(yv[:, None], Z)[:, 0]
        rows.append((feat, r, 1 - (res**2).sum() / sst))

    for feat in forced:
        Z = data[selected].to_numpy()
        ry = _residualize(yv[:, None], Z)[:, 0]
        rx = _residualize(data[[feat]].to_numpy(), Z)[:, 0]
        record(feat, float(np.corrcoef(rx, ry)[0, 1]))

    candidates = [c for c in X.columns if c not in selected]
    while candidates and len(selected) < max_k:
        Z = data[selected].to_numpy()
        ry = _residualize(yv[:, None], Z)[:, 0]
        RX = _residualize(data[candidates].to_numpy(), Z)
        sx = RX.std(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (RX - RX.mean(0)).T @ (ry - ry.mean()) / (len(ry) * sx * ry.std())
        r = np.where(sx > 1e-12, r, 0.0)
        best = int(np.argmax(np.abs(r)))
        if abs(r[best]) < min_partial_r:
            break
        record(candidates.pop(best), float(r[best]))

    return pd.DataFrame(rows, columns=["feature", "partial_r", "cum_r2"])


def redundancy_clusters(X: pd.DataFrame, threshold: float = 0.9) -> pd.Series:
    """Hierarchical (average-linkage) clusters where members have corr >= threshold."""
    corr = X.corr().fillna(0).to_numpy()
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(Z, t=1 - threshold, criterion="distance")
    return pd.Series(labels, index=X.columns, name="cluster")


def stability_select(
    X: pd.DataFrame,
    y: pd.Series,
    n_boot: int = 50,
    block_days: int = 180,
    max_k: int = 8,
    min_partial_r: float = 0.0,
    forced: list[str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Moving-block bootstrap around greedy_select.

    Contiguous blocks keep autocorrelation (storms, wet seasons) intact. A
    feature that is truly informative is picked in most resamples; features
    picked by chance (multiple comparisons) are not. Returns selection
    frequency and mean partial r for every non-forced feature ever picked.
    """
    forced = list(forced or [])
    rng = np.random.default_rng(seed)
    n = len(X)
    starts = np.arange(0, n, block_days)
    counts: dict[str, int] = {}
    rs: dict[str, list[float]] = {}
    for _ in range(n_boot):
        pos = []
        while len(pos) < n:
            s = int(rng.choice(starts))
            pos.extend(range(s, min(s + block_days, n)))
        pos = np.array(pos[:n])
        Xb = X.iloc[pos].reset_index(drop=True)
        yb = y.iloc[pos].reset_index(drop=True)
        sel = greedy_select(Xb, yb, max_k=len(forced) + max_k, min_partial_r=min_partial_r, forced=forced)
        for feat, r in zip(sel["feature"], sel["partial_r"]):
            if feat in forced:
                continue
            counts[feat] = counts.get(feat, 0) + 1
            rs.setdefault(feat, []).append(r)
    out = pd.DataFrame(
        {"freq": {f: c / n_boot for f, c in counts.items()}, "mean_partial_r": {f: float(np.mean(v)) for f, v in rs.items()}}
    )
    return out.sort_values("freq", ascending=False)
