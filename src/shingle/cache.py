"""Local parquet/json caching with incremental daily updates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

DailyFetch = Callable[[str, str], pd.DataFrame]


def _fmt(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")


def update_daily(
    path: Path,
    fetch: DailyFetch,
    default_start: str,
    today: str | None = None,
    overlap_days: int = 10,
    refresh: bool = False,
) -> pd.DataFrame:
    """Extend a cached daily frame up to `today`.

    Re-fetches the last `overlap_days` cached days too, because provisional
    data (USGS, gridMET) gets revised; fetched values win over cached ones.
    """
    path = Path(path)
    end = pd.Timestamp(today) if today else pd.Timestamp.today().normalize()
    old = None if refresh or not path.exists() else pd.read_parquet(path)

    if old is None or old.empty:
        start = pd.Timestamp(default_start)
    else:
        start = old.index.max() - pd.Timedelta(days=overlap_days)

    new = fetch(_fmt(start), _fmt(end))
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
    return df


def cached_json(path: Path, compute: Callable[[], Any], refresh: bool = False) -> Any:
    path = Path(path)
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))
    return value
