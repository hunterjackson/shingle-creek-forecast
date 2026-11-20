"""Pull and cache all historical inputs under a data directory."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from home_display.shingle import basin as basin_mod
from home_display.shingle import cache, config
from home_display.shingle.sources import dbhydro, gridmet, nldi, usgs


def fetch_history(data_dir: Path, today: str | None = None, start: str = config.HISTORY_START, refresh: bool = False) -> None:
    data_dir = Path(data_dir)

    for site in (config.SITE, config.UPSTREAM_SITE):
        cache.update_daily(
            data_dir / f"usgs_{site}.parquet",
            lambda s, e, site=site: usgs.fetch_dv(site, s, e),
            default_start=start,
            today=today,
            refresh=refresh,
        )

    for var in gridmet.VARS:
        cache.update_daily(
            data_dir / f"gridmet_{var}.parquet",
            lambda s, e, var=var: gridmet.fetch(var, config.BBOX, s, e),
            default_start=start,
            today=today,
            refresh=refresh,
        )

    cache.update_daily(
        data_dir / "dbhydro.parquet",
        lambda s, e: dbhydro.fetch(dbhydro.all_dbkeys(), s, e),
        default_start=start,
        today=today,
        refresh=refresh,
    )

    geojson = cache.cached_json(
        data_dir / "basin.geojson", lambda: nldi.fetch_basin_geojson(config.SITE), refresh=refresh
    )
    pr_cols = pd.read_parquet(data_dir / "gridmet_pr.parquet").columns
    cells = basin_mod.annotate_cells(gridmet.cells_from_columns(pr_cols), nldi.parse_basin(geojson), config.OUTLET)
    cells.to_parquet(data_dir / "cells.parquet")


def load_history(data_dir: Path) -> dict:
    data_dir = Path(data_dir)
    return {
        "gauge": pd.read_parquet(data_dir / f"usgs_{config.SITE}.parquet"),
        "upstream": pd.read_parquet(data_dir / f"usgs_{config.UPSTREAM_SITE}.parquet"),
        "pr": pd.read_parquet(data_dir / "gridmet_pr.parquet"),
        "pet": pd.read_parquet(data_dir / "gridmet_pet.parquet"),
        "cells": pd.read_parquet(data_dir / "cells.parquet"),
        "lakes": dbhydro.named(pd.read_parquet(data_dir / "dbhydro.parquet")),
        "basin": nldi.parse_basin(json.loads((data_dir / "basin.geojson").read_text())),
    }


def fetch_nwm_retro(data_dir: Path, start: str = "1979-02-02", end: str = "2023-01-31", refresh: bool = False) -> pd.DataFrame:
    """One-off pull of the NWM v3 retrospective daily flow for our reach (~3.5 GB of chunk reads)."""
    from home_display.shingle.sources import nwm_retro

    path = Path(data_dir) / "nwm_retro.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    s = nwm_retro.read_reach_daily(int(config.NWM_REACH), start, end, min_hours=24)
    df = s.to_frame()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def fetch_mrms_history(data_dir: Path, start: str | None = None, today: str | None = None, workers: int = 8) -> None:
    """Download MRMS daily QPE month by month into data/mrms/YYYY-MM.parquet (resumable).

    A month file is considered final once its last day is >= 5 days before
    `today`; otherwise it is re-fetched (latest days get revised / filled in).
    """
    from home_display.shingle.sources import mrms

    out = Path(data_dir) / "mrms"
    out.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(start) if start else mrms.START
    end_ts = pd.Timestamp(today) if today else pd.Timestamp.today().normalize() - pd.Timedelta(days=1)

    for month in pd.period_range(start_ts, end_ts, freq="M"):
        m_start = max(start_ts, month.start_time)
        m_end = min(end_ts, month.end_time.normalize())
        path = out / f"{month}.parquet"
        final = month.end_time.normalize() <= end_ts - pd.Timedelta(days=5)
        if path.exists() and final:
            existing = pd.read_parquet(path)
            if existing.index.max() >= month.end_time.normalize():
                continue
        df = mrms.fetch(m_start.strftime("%Y-%m-%d"), m_end.strftime("%Y-%m-%d"), config.BBOX, workers=workers)
        df.to_parquet(path)


def load_mrms(data_dir: Path) -> pd.DataFrame:
    parts = sorted((Path(data_dir) / "mrms").glob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "date"
    return df
