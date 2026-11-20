"""SFWMD DBHYDRO daily series: Kissimmee Chain of Lakes stages and structure flows.

Uses the unauthenticated backend of DBHYDRO Insights (insightsdata.sfwmd.gov);
it is undocumented and may change. The official dataservice API needs
credentials. Always request NGVD29 - the default datum is NAVD88 (~1.05 ft
lower at S-61).

DBKEYs verified 2026-11-20 (see docs/shingle_findings.md). Where a long
composite series stopped updating (May 2026), it is spliced with the
telemetry series that continues to the present.
"""

from __future__ import annotations

import time

import httpx
import numpy as np
import pandas as pd

URL = "https://insightsdata.api.sfwmd.gov/v1/insights-data/cont/data"

# name -> DBKEYs in priority order (first non-NaN wins)
SERIES: dict[str, list[str]] = {
    "toho_stage": ["68918", "WN255"],  # Lake Toho at S61_H, 1963+ / telemetry 2009+
    "toho_west_stage": ["LQ978"],  # TOHOW west shore, 2000+
    "s61_q": ["91625"],  # Toho outflow, 1963+
    "etoho_stage": ["68890", "WN265"],  # East Lake Toho at S59_H
    "s59_q": ["91618"],  # East Toho -> Toho, 1963+
    "kiss_stage": ["WN313"],  # Lake Kissimmee at S65_H, 2009+ only
    "s65_q": ["91658"],  # Lake Kissimmee outflow, 1996-09+
    "okee_stage": ["06832"],  # Lake Okeechobee 8-gauge average, 1931+
}

MISSING_CODES = {"M", "N"}


def all_dbkeys() -> list[str]:
    return [k for keys in SERIES.values() for k in keys]


def parse(payload: dict) -> pd.DataFrame:
    rows = payload.get("timeseries", [])
    if not rows:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    recs = []
    for r in rows:
        units = r.get("units") or ""
        if units.startswith("ft") and "NGVD29" not in units:
            raise ValueError(f"stage for {r['timeseriesID']} not in NGVD29: {units!r}")
        v = r["value"]
        if v is None or (r.get("code") or "").strip() in MISSING_CODES:
            v = np.nan
        recs.append((r["timeseriesID"], pd.to_datetime(r["timestamp"], format="%m/%d/%Y %H:%M"), float(v)))
    long = pd.DataFrame(recs, columns=["key", "date", "value"])
    wide = long.pivot_table(index="date", columns="key", values="value", aggfunc="mean", dropna=False)
    order = list(dict.fromkeys(long["key"]))
    wide = wide.reindex(columns=order)
    wide.columns.name = None
    wide.index.name = "date"
    return wide.astype(float)


def _date_chunks(start: str, end: str, years: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    chunks = []
    cur = s
    while cur <= e:
        nxt = min(e, pd.Timestamp(year=(cur.year // years + 1) * years, month=1, day=1) - pd.Timedelta(days=1))
        chunks.append((cur, nxt))
        cur = nxt + pd.Timedelta(days=1)
    return chunks


def _get(client: httpx.Client, params: dict, retries: int, retry_wait: float) -> dict:
    for attempt in range(retries + 1):
        try:
            resp = client.get(URL, params=params)
            resp.raise_for_status()
            return resp.json()
        except (httpx.TransportError, httpx.HTTPStatusError):
            if attempt == retries:
                raise
            time.sleep(retry_wait * (attempt + 1))
    raise AssertionError("unreachable")


def fetch(
    dbkeys: list[str],
    start: str,
    end: str,
    client: httpx.Client | None = None,
    max_keys: int = 3,
    chunk_years: int = 10,
    retries: int = 3,
    retry_wait: float = 5.0,
) -> pd.DataFrame:
    """Fetch in (key-group x decade) chunks - one big request gets its connection reset."""
    client = client or httpx.Client(timeout=180)
    result: pd.DataFrame | None = None
    for i in range(0, len(dbkeys), max_keys):
        group = dbkeys[i : i + max_keys]
        for cs, ce in _date_chunks(start, end, chunk_years):
            params = {
                "timeseriesIds": ",".join(group),
                "reportType": "timeseries",
                "format": "json",
                "startDate": cs.strftime("%Y%m%d"),
                "endDate": ce.strftime("%Y%m%d"),
                "datum": "NGVD29",
            }
            df = parse(_get(client, params, retries, retry_wait))
            result = df if result is None else df.combine_first(result)
    result = result if result is not None else parse({})
    cols = [k for k in dbkeys if k in result.columns] + [c for c in result.columns if c not in dbkeys]
    out = result.reindex(columns=cols).sort_index()
    out.index.name = "date"
    return out


def splice(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    for k in keys:
        if k in df.columns:
            out = out.fillna(df[k])
    return out


def named(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({name: splice(raw, keys) for name, keys in SERIES.items()}, index=raw.index)
