"""Local parquet/json caching with incremental daily updates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Awaitable, Callable

import pandas as pd

DailyFetch = Callable[[str, str], Awaitable[pd.DataFrame]]  # async (start, end) -> frame


def _fmt(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")


def _now(now: pd.Timestamp | None) -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")


def _meta_path(path: Path) -> Path:
    return Path(f"{path}.meta.json")


def revision_due(path: Path, every: pd.Timedelta | None, now: pd.Timestamp | None = None) -> bool:
    """True when the cache at `path` should re-fetch its recent window to pick up revised values:
    always when `every` is None, otherwise when its last revision pass is `every` old or unknown."""
    if every is None:
        return True
    meta = _meta_path(path)
    if not meta.exists():
        return True
    revised_at = pd.Timestamp(json.loads(meta.read_text())["revised_at"])
    return _now(now) - revised_at >= every


def mark_revised(path: Path, now: pd.Timestamp | None = None) -> None:
    _meta_path(path).write_text(json.dumps({"revised_at": _now(now).isoformat()}))


async def update_daily(
    path: Path,
    fetch: DailyFetch,
    default_start: str,
    today: str | None = None,
    overlap_days: int = 10,
    refresh: bool = False,
    revise_every: pd.Timedelta | None = None,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Extend a cached daily frame up to `today`.

    Provisional data (USGS, gridMET) gets revised, so a revision pass re-fetches
    the last `overlap_days` cached days too; fetched values win over cached ones.
    With `revise_every` set, that pass runs at most that often (tracked in a
    `<file>.meta.json` beside the cache). In between, only days after the last
    cached one are asked for, and nothing at all once yesterday is cached
    (today's daily value is not published until tomorrow).
    """
    path = Path(path)
    end = pd.Timestamp(today) if today else pd.Timestamp.today().normalize()
    old = None if refresh or not path.exists() else pd.read_parquet(path)

    revising = old is None or old.empty or revision_due(path, revise_every, now)
    if old is None or old.empty:
        start = pd.Timestamp(default_start)
    elif revising:
        start = old.index.max() - pd.Timedelta(days=overlap_days)
    elif old.index.max() >= end - pd.Timedelta(days=1):
        return old
    else:
        start = old.index.max() + pd.Timedelta(days=1)

    new = await fetch(_fmt(start), _fmt(end))
    if old is None:
        df = new
    elif new.empty:
        df = old
    else:
        cols = list(old.columns) + [c for c in new.columns if c not in old.columns]
        df = pd.concat([old[old.index < new.index.min()], new]).reindex(columns=cols)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "date"

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    if revising and revise_every is not None:
        mark_revised(path, now)
    return df


async def cached_json(path: Path, compute: Callable[[], Awaitable[Any]], refresh: bool = False) -> Any:
    path = Path(path)
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    value = await compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))
    return value
