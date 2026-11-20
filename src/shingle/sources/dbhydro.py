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


def fetch(dbkeys: list[str], start: str, end: str, client: httpx.Client | None = None) -> pd.DataFrame:
    params = {
        "timeseriesIds": ",".join(dbkeys),
        "reportType": "timeseries",
        "format": "json",
        "startDate": pd.Timestamp(start).strftime("%Y%m%d"),
        "endDate": pd.Timestamp(end).strftime("%Y%m%d"),
        "datum": "NGVD29",
    }
    client = client or httpx.Client(timeout=180)
    resp = client.get(URL, params=params)
    resp.raise_for_status()
    return parse(resp.json())


def splice(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    for k in keys:
        if k in df.columns:
            out = out.fillna(df[k])
    return out


def named(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({name: splice(raw, keys) for name, keys in SERIES.items()}, index=raw.index)
