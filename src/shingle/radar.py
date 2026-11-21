"""Radar (MRMS) rain regions: data-driven groups of cells whose rain moves together.

Stability selection showed individual 2 km cells are interchangeable with
their neighbours but *regions* are stably selected. Regions are built on
training data only and persisted so features are reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from home_display.shingle import analysis
from home_display.shingle.sources.gridmet import cells_from_columns


def build_regions(blocks: pd.DataFrame, train_end: str, threshold: float = 0.85, window: int = 3) -> pd.Series:
    """Average-linkage clusters of 3-day rain sums (train period only), r >= threshold."""
    train = blocks.loc[: pd.Timestamp(train_end)]
    return analysis.redundancy_clusters(analysis.trailing_sum(train, window).dropna(), threshold)


def save_regions(regions: pd.Series, path: Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta or {}, "regions": {k: int(v) for k, v in regions.items()}}, indent=1))


def load_regions(path: Path) -> tuple[pd.Series, dict]:
    d = json.loads(Path(path).read_text())
    return pd.Series(d["regions"], name="cluster"), d["meta"]


def region_centroids(regions: pd.Series) -> pd.DataFrame:
    cells = cells_from_columns(regions.index).join(regions.rename("cluster"))
    cen = cells.groupby("cluster")[["lat", "lon"]].mean().round(4)
    cen.index = [f"r{c}" for c in cen.index]
    return cen
