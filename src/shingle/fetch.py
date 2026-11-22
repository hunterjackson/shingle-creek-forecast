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
        "iv": {"gauge": load_iv(data_dir, config.SITE), "upstream": load_iv(data_dir, config.UPSTREAM_SITE)},
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


IV_START = "2007-01-01"  # NWIS instantaneous-value archive starts ~2007 for our gauges


def fetch_iv_history(
    data_dir: Path, sites=None, start: str = IV_START, today: str | None = None, overlap_days: int = 3,
    retries: int = 4, retry_wait: float = 10.0,
) -> None:
    """Hourly-mean instantaneous discharge/stage per site -> data/usgs_iv_{site}.parquet.

    First run pulls year by year from `start`; later runs re-fetch from the last
    stored day minus `overlap_days` (provisional values get revised). Each year
    is retried on connection errors and written to disk as soon as it arrives,
    so an interrupted pull resumes where it stopped.
    """
    import time

    import httpx

    sites = sites or (config.SITE, config.UPSTREAM_SITE)
    end_ts = pd.Timestamp(today) if today else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() + pd.Timedelta(days=1)
    for site in sites:
        path = Path(data_dir) / f"usgs_iv_{site}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        stored = pd.read_parquet(path) if path.exists() else None
        s = (stored.index.max().tz_localize(None).normalize() - pd.Timedelta(days=overlap_days)) if stored is not None else pd.Timestamp(start)
        while s < end_ts:
            e = min(pd.Timestamp(f"{s.year}-12-31 23:59"), end_ts)
            for attempt in range(retries + 1):
                try:
                    part = usgs.fetch_iv(site, s.strftime("%Y-%m-%dT%H:%MZ"), e.strftime("%Y-%m-%dT%H:%MZ"))
                    break
                except (httpx.TransportError, httpx.HTTPStatusError):
                    if attempt == retries:
                        raise
                    time.sleep(retry_wait * (attempt + 1))
            stored = (pd.concat([stored, part]) if stored is not None else part).sort_index()
            stored = stored[~stored.index.duplicated(keep="last")]
            stored.to_parquet(path)
            s = pd.Timestamp(f"{s.year + 1}-01-01")


def load_iv(data_dir: Path, site: str) -> pd.DataFrame | None:
    path = Path(data_dir) / f"usgs_iv_{site}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def fetch_gefs_history(
    data_dir: Path, start: str = "2020-10-01", end: str | None = None, kinds=("mean", "spread"),
    workers: int = 8, today: str | None = None,
) -> None:
    """00Z GEFSv12 APCP buckets per run -> data/gefs/{kind}/YYYY-MM.parquet (index run, fhour).

    Months entirely before `today` that already cover the requested runs are skipped
    (resumable); runs missing from the archive are simply absent.
    """
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    from home_display.shingle.sources import gefs

    today_ts = pd.Timestamp(today) if today else pd.Timestamp.today().normalize()
    end_ts = pd.Timestamp(end) if end else today_ts
    start_ts = pd.Timestamp(start)
    client = httpx.Client(timeout=60, limits=httpx.Limits(max_connections=workers * 2))
    for kind in kinds:
        out = Path(data_dir) / "gefs" / kind
        out.mkdir(parents=True, exist_ok=True)
        for month in pd.period_range(start_ts, end_ts, freq="M"):
            runs = pd.date_range(max(start_ts, month.start_time), min(end_ts, month.end_time.normalize()))
            path = out / f"{month}.parquet"
            if path.exists() and month.end_time.normalize() < today_ts:
                have = pd.read_parquet(path).index.get_level_values("run")
                if have.min() <= runs[0] and have.max() >= runs[-1]:
                    continue

            def one(run, kind=kind):
                df = gefs.fetch_run(run, bbox=config.BBOX, kind=kind, client=client)
                if df is None:
                    return None
                return pd.concat({run: df}, names=["run"])

            with ThreadPoolExecutor(workers) as ex:
                parts = [p for p in ex.map(one, runs) if p is not None]
            if parts:
                pd.concat(parts).sort_index().to_parquet(path)


def load_gefs(data_dir: Path, kind: str = "mean") -> pd.DataFrame:
    parts = sorted((Path(data_dir) / "gefs" / kind).glob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts]).sort_index()
    return df[~df.index.duplicated(keep="last")]
