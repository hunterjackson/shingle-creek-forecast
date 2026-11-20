"""USGS NWIS daily-value (dv) client."""

from __future__ import annotations

import httpx
import numpy as np
import pandas as pd

DV_URL = "https://waterservices.usgs.gov/nwis/dv/"

PARAMS = {"00060": "discharge_cfs", "00065": "stage_ft"}


def parse_dv(payload: dict) -> pd.DataFrame:
    """Turn a NWIS dv JSON payload into a daily frame of discharge and stage."""
    series = {}
    for ts in payload["value"]["timeSeries"]:
        code = ts["variable"]["variableCode"][0]["value"]
        if code not in PARAMS:
            continue
        no_data = float(ts["variable"]["noDataValue"])
        values = ts["values"][0]["value"]
        idx = pd.to_datetime([v["dateTime"][:10] for v in values])
        vals = np.array([float(v["value"]) for v in values])
        vals[vals == no_data] = np.nan
        series[PARAMS[code]] = pd.Series(vals, index=idx)

    df = pd.DataFrame(series).reindex(columns=list(PARAMS.values()))
    df.index.name = "date"
    return df.astype(float)


def fetch_dv(site: str, start: str, end: str | None = None, client: httpx.Client | None = None) -> pd.DataFrame:
    params = {
        "sites": site,
        "parameterCd": ",".join(PARAMS),
        "startDT": start,
        "format": "json",
        "siteStatus": "all",
    }
    if end:
        params["endDT"] = end
    client = client or httpx.Client(timeout=120)
    resp = client.get(DV_URL, params=params)
    resp.raise_for_status()
    return parse_dv(resp.json())
