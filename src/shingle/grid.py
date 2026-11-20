"""Grid helpers for per-cell rain frames whose columns are 'lat_lon' cell ids."""

from __future__ import annotations

import numpy as np
import pandas as pd

from home_display.shingle.sources.gridmet import cells_from_columns


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
