"""Pull and cache all historical inputs under a data directory."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Awaitable

import pandas as pd

from shingle import basin as basin_mod
from shingle import cache, config
from shingle.sources import dbhydro, gridmet, http, nldi, usgs


async def run_all(tasks: dict[str, Awaitable[Any]], label: str = "") -> dict[str, Any]:
    """Await independent downloads at the same time and return {name: result}.

    Prints how long each took (stderr, so it shows in the Actions log). Every
    task runs to the end even when another fails, so caches that did update are
    kept; the first failure (in task order) is then raised.
    """
    prefix = f"{label}: " if label else ""

    async def timed(name: str, aw: Awaitable[Any]) -> Any:
        t0 = time.monotonic()
        try:
            return await aw
        finally:
            print(f"{prefix}{name} {time.monotonic() - t0:.1f}s", file=sys.stderr)

    t0 = time.monotonic()
    results = await asyncio.gather(*(timed(n, aw) for n, aw in tasks.items()), return_exceptions=True)
    if label:
        print(f"{label}: done in {time.monotonic() - t0:.1f}s", file=sys.stderr)
    for r in results:
        if isinstance(r, BaseException):
            raise r
    return dict(zip(tasks, results))


async def fetch_history(data_dir: Path, today: str | None = None, start: str = config.HISTORY_START, refresh: bool = False) -> None:
    """Update the daily caches (USGS daily values, gridMET, DBHYDRO) and the basin outline, all at once."""
    data_dir = Path(data_dir)

    def daily(name: str, fn):
        return cache.update_daily(data_dir / f"{name}.parquet", fn, default_start=start, today=today, refresh=refresh)

    tasks = {}
    for site in (config.SITE, config.UPSTREAM_SITE):
        tasks[f"usgs_{site}"] = daily(f"usgs_{site}", lambda s, e, site=site: usgs.fetch_dv(site, s, e))
    for var in gridmet.VARS:
        tasks[f"gridmet_{var}"] = daily(f"gridmet_{var}", lambda s, e, var=var: gridmet.fetch(var, config.BBOX, s, e))
    tasks["dbhydro"] = daily("dbhydro", lambda s, e: dbhydro.fetch(dbhydro.all_dbkeys(), s, e))
    tasks["basin"] = cache.cached_json(
        data_dir / "basin.geojson", lambda: nldi.fetch_basin_geojson(config.SITE), refresh=refresh
    )
    geojson = (await run_all(tasks, label="daily history"))["basin"]
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
    from shingle.sources import nwm_retro

    path = Path(data_dir) / "nwm_retro.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    s = nwm_retro.read_reach_daily(int(config.NWM_REACH), start, end, min_hours=24)
    df = s.to_frame()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def _mrms_dir(data_dir: Path, hour: int) -> Path:
    return Path(data_dir) / ("mrms" if hour == 5 else f"mrms{hour:02d}")


async def fetch_mrms_history(data_dir: Path, start: str | None = None, today: str | None = None, workers: int = 8, hour: int = 5) -> None:
    """Download MRMS daily QPE into data/mrms/YYYY-MM.parquet, one file per month (resumable).

    A month file is considered final once its last day is >= 5 days before
    `today`; otherwise it is re-fetched (latest days get revised / filled in).
    Months download at the same time; each is written as soon as it is complete.
    """
    from shingle.sources import mrms

    out = _mrms_dir(data_dir, hour)
    out.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(start) if start else mrms.START
    end_ts = pd.Timestamp(today) if today else pd.Timestamp.today().normalize() - pd.Timedelta(days=1)

    async def one(month: pd.Period) -> None:
        m_start = max(start_ts, month.start_time)
        m_end = min(end_ts, month.end_time.normalize())
        path = out / f"{month}.parquet"
        final = month.end_time.normalize() <= end_ts - pd.Timedelta(days=5)
        if path.exists() and final:
            existing = pd.read_parquet(path)
            if existing.index.max() >= month.end_time.normalize():
                return
        kw = {"hour": hour} if hour != 5 else {}
        df = await mrms.fetch(m_start.strftime("%Y-%m-%d"), m_end.strftime("%Y-%m-%d"), config.BBOX, workers=workers, **kw)
        df.to_parquet(path)

    # a long backfill has hundreds of months; a few at a time keeps the request count bounded
    await http.gather((one(m) for m in pd.period_range(start_ts, end_ts, freq="M")), limit=2)


def load_mrms(data_dir: Path, hour: int = 5) -> pd.DataFrame:
    parts = sorted(_mrms_dir(data_dir, hour).glob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "date"
    return df


IV_START = "2007-01-01"  # NWIS instantaneous-value archive starts ~2007 for our gauges


async def fetch_iv_history(
    data_dir: Path, sites=None, start: str = IV_START, today: str | None = None, overlap_days: int = 3,
) -> None:
    """Hourly-mean instantaneous discharge/stage per site -> data/usgs_iv_{site}.parquet.

    First run pulls year by year from `start`; later runs re-fetch from the last
    stored day minus `overlap_days` (provisional values get revised). Each year
    is written to disk as soon as it arrives (requests retry temporary failures
    themselves), so an interrupted pull resumes where it stopped. Sites download
    at the same time.
    """
    sites = sites or (config.SITE, config.UPSTREAM_SITE)
    end_ts = pd.Timestamp(today) if today else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() + pd.Timedelta(days=1)

    async def one(site: str) -> None:
        path = Path(data_dir) / f"usgs_iv_{site}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        stored = pd.read_parquet(path) if path.exists() else None
        s = (stored.index.max().tz_localize(None).normalize() - pd.Timedelta(days=overlap_days)) if stored is not None else pd.Timestamp(start)
        while s < end_ts:
            e = min(pd.Timestamp(f"{s.year}-12-31 23:59"), end_ts)
            part = await usgs.fetch_iv(site, s.strftime("%Y-%m-%dT%H:%MZ"), e.strftime("%Y-%m-%dT%H:%MZ"))
            stored = (pd.concat([stored, part]) if stored is not None else part).sort_index()
            stored = stored[~stored.index.duplicated(keep="last")]
            stored.to_parquet(path)
            s = pd.Timestamp(f"{s.year + 1}-01-01")

    await run_all({site: one(site) for site in sites})


def load_iv(data_dir: Path, site: str) -> pd.DataFrame | None:
    path = Path(data_dir) / f"usgs_iv_{site}.parquet"
    return pd.read_parquet(path) if path.exists() else None


async def fetch_gefs_history(
    data_dir: Path, start: str = "2020-10-01", end: str | None = None, kinds=("mean", "spread"),
    workers: int = 8, today: str | None = None, refresh_days: int = 2,
) -> None:
    """00Z GEFSv12 APCP buckets per run -> data/gefs/{kind}/YYYY-MM.parquet (index run, fhour).

    Runs already in a month file are skipped, except those from the last
    `refresh_days` before `today` (fetched while still uploading, they may be
    partial). New runs are merged into the month file, never replace it.
    Runs missing from the archive are simply absent (and asked for again).
    `workers` runs download at a time, each fetching its forecast hours in parallel.
    """
    import httpx

    from shingle.sources import gefs

    today_ts = pd.Timestamp(today) if today else pd.Timestamp.today().normalize()
    end_ts = pd.Timestamp(end) if end else today_ts
    start_ts = pd.Timestamp(start)
    fresh_from = today_ts - pd.Timedelta(days=refresh_days)

    async def one(client: httpx.AsyncClient, run: pd.Timestamp, kind: str) -> pd.DataFrame | None:
        df = await gefs.fetch_run(run, bbox=config.BBOX, kind=kind, client=client)
        return None if df is None else pd.concat({run: df}, names=["run"])

    async with httpx.AsyncClient(timeout=60, limits=httpx.Limits(max_connections=workers * 8)) as client:
        for kind in kinds:
            out = Path(data_dir) / "gefs" / kind
            out.mkdir(parents=True, exist_ok=True)
            for month in pd.period_range(start_ts, end_ts, freq="M"):
                path = out / f"{month}.parquet"
                have = pd.read_parquet(path) if path.exists() else None
                cached = set(have.index.get_level_values("run")) if have is not None else set()
                runs = pd.date_range(max(start_ts, month.start_time), min(end_ts, month.end_time.normalize()))
                runs = [r for r in runs if r not in cached or r >= fresh_from]
                if not runs:
                    continue
                parts = [p for p in await http.gather((one(client, r, kind) for r in runs), limit=workers) if p is not None]
                if not parts:
                    continue
                new = pd.concat(parts)
                if have is not None:
                    fetched = new.index.get_level_values("run").unique()
                    new = pd.concat([have[~have.index.get_level_values("run").isin(fetched)], new])
                new.sort_index().to_parquet(path)


def load_gefs(data_dir: Path, kind: str = "mean") -> pd.DataFrame:
    parts = sorted((Path(data_dir) / "gefs" / kind).glob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts]).sort_index()
    return df[~df.index.duplicated(keep="last")]
