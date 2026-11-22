"""Turn issued rain forecasts (e.g. GEFS daily leads) into model inputs (data['fcst']).

    raw   : issue date x (lead, source cell) daily rain forecasts (mm)
    -> pr : issue date x (lead, model cell), bilinear from the source grid, optional per-lead scale
    -> gr4j_q : GR4J restarted from its state at t with the forecast areal rain (cfs per lead)
"""

from __future__ import annotations

import pandas as pd

from shingle import grid


def build(raw: pd.DataFrame, src_cells: pd.DataFrame, tgt_cells: pd.DataFrame, driver, data: dict, scale: dict | None = None) -> dict:
    weights = grid.bilinear_weights(tgt_cells, src_cells)
    leads = sorted(raw.columns.get_level_values(0).unique())
    parts = {}
    for k in leads:
        mapped = grid.apply_weights(raw[k], weights)
        if scale is not None:
            mapped = mapped * scale.get(k, 1.0)
        parts[k] = mapped
    pr = pd.concat(parts, axis=1, names=["lead", "cell"]).reindex(data["stage"].index)
    areal = pd.DataFrame({k: _areal(pr[k], driver.weights) for k in leads}, index=pr.index)
    return {"pr": pr, "gr4j_q": driver.forecast(data, areal)}


def _areal(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    w = pd.Series(weights, dtype=float)
    w = w[w.index.isin(df.columns)]
    return (df[w.index] * (w / w.sum())).sum(axis=1, min_count=len(w))


def fit_scale(mapped: pd.DataFrame, obs: pd.DataFrame, cell: str, index: pd.DatetimeIndex) -> dict[int, float]:
    """Per-lead multiplicative bias correction: total observed / total forecast rain at `cell`
    for issue dates in `index` (lead k forecast compared with obs on day t+k)."""
    out = {}
    for k in sorted(mapped.columns.get_level_values(0).unique()):
        fc = mapped[(k, cell)].reindex(index)
        ob = obs[cell].shift(-k).reindex(index)
        ok = fc.notna() & ob.notna()
        out[k] = float(ob[ok].sum() / fc[ok].sum()) if fc[ok].sum() > 0 else 1.0
    return out


def build_mixed(raw, src_cells, tgt_cells, driver, data: dict, scale: dict, scaled_from_h: int = 4, horizons=tuple(range(1, 8))) -> dict:
    """Raw forecasts for short horizons, per-lead-scaled forecasts for horizons >= scaled_from_h
    (val 2022-23: raw GEFS best for h1-3, scaled for h4-7)."""
    plain = build(raw, src_cells, tgt_cells, driver, data)
    scaled = build(raw, src_cells, tgt_cells, driver, data, scale=scale)
    return dict(plain, by_h={h: scaled for h in horizons if h >= scaled_from_h})
