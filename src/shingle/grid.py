"""Grid helpers for per-cell rain frames whose columns are 'lat_lon' cell ids."""

from __future__ import annotations

import numpy as np
import pandas as pd

from shingle.sources.gridmet import cells_from_columns


def aggregate_blocks(df: pd.DataFrame, factor: int, res: float = 0.01) -> pd.DataFrame:
    """Average factor x factor blocks of pixels (NaN-aware); ids = block centres (3 dp)."""
    cells = cells_from_columns(df.columns)
    lat0, lon0 = cells["lat"].max(), cells["lon"].min()
    row = np.round((lat0 - cells["lat"]) / res).astype(int) // factor
    col = np.round((cells["lon"] - lon0) / res).astype(int) // factor
    out = {}
    for (r, c), members in cells.groupby([row, col]).groups.items():
        sub = cells.loc[members]
        cid = f"{sub['lat'].mean():.3f}_{sub['lon'].mean():.3f}"
        out[cid] = df[list(members)].mean(axis=1, skipna=True)
    return pd.DataFrame(out, index=df.index)


def nearest(targets: pd.DataFrame, sources: pd.DataFrame) -> pd.Series:
    """For each target cell, the id of the nearest source cell (planar lat/lon distance)."""
    t = targets[["lat", "lon"]].to_numpy()
    s = sources[["lat", "lon"]].to_numpy()
    d2 = ((t[:, None, :] - s[None, :, :]) ** 2).sum(axis=2)
    return pd.Series(sources.index[d2.argmin(axis=1)], index=targets.index)


def remap(src: pd.DataFrame, mapping: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({t: src[s] for t, s in mapping.items()}, index=src.index)


def region_means(df: pd.DataFrame, clusters: pd.Series, prefix: str = "r") -> pd.DataFrame:
    """Mean of member cells per region (NaN-aware); columns '{prefix}{cluster_id}'."""
    out = {}
    for cid in sorted(clusters.unique()):
        members = [c for c in clusters.index[clusters == cid] if c in df.columns]
        out[f"{prefix}{cid}"] = df[members].mean(axis=1, skipna=True)
    return pd.DataFrame(out, index=df.index)


def bilinear_weights(targets: pd.DataFrame, sources: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Bilinear weights from a regular lat/lon source grid to each target point.

    Targets outside the source grid are clamped to its edge.
    """
    lats = np.sort(sources["lat"].unique())
    lons = np.sort(sources["lon"].unique())
    ids = {(round(la, 6), round(lo, 6)): i for i, la, lo in zip(sources.index, sources["lat"], sources["lon"])}
    out = {}
    for tid, la, lo in zip(targets.index, targets["lat"], targets["lon"]):
        la = float(np.clip(la, lats[0], lats[-1]))
        lo = float(np.clip(lo, lons[0], lons[-1]))
        i = int(np.clip(np.searchsorted(lats, la) - 1, 0, len(lats) - 2))
        j = int(np.clip(np.searchsorted(lons, lo) - 1, 0, len(lons) - 2))
        fy = (la - lats[i]) / (lats[i + 1] - lats[i])
        fx = (lo - lons[j]) / (lons[j + 1] - lons[j])
        w = {}
        for di, wy in ((0, 1 - fy), (1, fy)):
            for dj, wx in ((0, 1 - fx), (1, fx)):
                sid = ids[(round(lats[i + di], 6), round(lons[j + dj], 6))]
                w[sid] = w.get(sid, 0.0) + wy * wx
        out[tid] = w
    return out


def apply_weights(df: pd.DataFrame, weights: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame({t: sum(df[s] * v for s, v in w.items()) for t, w in weights.items()}, index=df.index)


def cell_means(pixels: pd.DataFrame, cells: pd.DataFrame, half_width: float) -> pd.DataFrame:
    """Mean of fine pixels ('lat_lon' ids) falling inside each target cell (centre +- half_width)."""
    px = cells_from_columns(pixels.columns)
    out = {}
    for cid, la, lo in zip(cells.index, cells["lat"], cells["lon"]):
        inside = px.index[(np.abs(px["lat"] - la) <= half_width) & (np.abs(px["lon"] - lo) <= half_width)]
        out[cid] = pixels[inside].mean(axis=1) if len(inside) else pd.Series(np.nan, index=pixels.index)
    return pd.DataFrame(out, index=pixels.index)
