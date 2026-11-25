"""NOAA National Water Model streamflow for a reach, via the NWPS API.

Used as a benchmark: NWM flow is converted to stage with our rating curve and
scored against the same observations as our models.
"""

from __future__ import annotations

import re

import httpx
import pandas as pd

from shingle.sources import http

NWPS = "https://api.water.noaa.gov/nwps/v1/reaches"
LOCAL_TZ = "America/New_York"
COLUMNS = ["series", "member", "reference_time", "valid_time", "flow_cfs"]


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def parse_streamflow(payload: dict) -> pd.DataFrame:
    rows = []
    for top, block in payload.items():
        if top == "reach" or not isinstance(block, dict):
            continue
        for member, ts in block.items():
            if not isinstance(ts, dict) or "data" not in ts:
                continue
            ref = pd.Timestamp(ts["referenceTime"])
            for pt in ts["data"]:
                rows.append((_snake(top), "det" if member == "series" else member, ref, pt["validTime"], pt["flow"]))
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["reference_time"] = pd.to_datetime(df["reference_time"], utc=True)
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df["flow_cfs"] = df["flow_cfs"].astype(float)
    return df.sort_values(["series", "member", "valid_time"]).reset_index(drop=True)


def daily_mean(df: pd.DataFrame) -> pd.Series:
    """Mean flow per local (America/New_York) calendar day for a single series/member."""
    if df[["series", "member", "reference_time"]].drop_duplicates().shape[0] > 1:
        raise ValueError("daily_mean expects exactly one series/member/reference_time")
    local_day = df["valid_time"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).dt.normalize()
    out = df.groupby(local_day.rename("date"))["flow_cfs"].mean()
    return out


async def fetch_streamflow(reach: str, series: str, client: httpx.AsyncClient | None = None) -> pd.DataFrame:
    async with http.session(client, timeout=60) as c:
        resp = await http.get(c, f"{NWPS}/{reach}/streamflow", params={"series": series})
    return parse_streamflow(resp.json())
