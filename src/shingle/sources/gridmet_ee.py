"""gridMET from Google Earth Engine (IDAHO_EPSCOR/GRIDMET), the primary gridMET source; the University of Idaho
THREDDS server (gridmet.py) tops up the newest day and covers Earth Engine failures. Same dataset and grid, so values are sampled at our cell centres and returned in the
gridmet.fetch layout (index=date, columns=cell ids, mm).

Credentials come from the environment: EE_SERVICE_ACCOUNT_KEY (the service account's JSON key) and EE_PROJECT
(the Google Cloud project registered for Earth Engine). Earth Engine's copy can lag THREDDS by a day or so.
"""

from __future__ import annotations

import asyncio
import json
import os

import pandas as pd

COLLECTION = "IDAHO_EPSCOR/GRIDMET"
SCALE_M = 4638.3  # gridMET's native 1/24 degree grid in Earth Engine
BANDS = {"pr": "pr", "pet": "eto"}  # our short name -> Earth Engine band (eto = grass reference ET, our pet)


def configured() -> bool:
    return bool(os.environ.get("EE_SERVICE_ACCOUNT_KEY") and os.environ.get("EE_PROJECT"))


def _init():
    import ee

    key = os.environ["EE_SERVICE_ACCOUNT_KEY"]
    creds = ee.ServiceAccountCredentials(json.loads(key)["client_email"], key_data=key)
    ee.Initialize(creds, project=os.environ["EE_PROJECT"])
    return ee


def _sample(band: str, cells: pd.DataFrame, start: str, end: str) -> list[tuple[str, str, float | None]]:
    """[(YYYYMMDD, cell id, value)] for every image in [start, end] at every cell centre."""
    ee = _init()
    points = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"cell": cid})
                                   for cid, r in cells.iterrows()])
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    images = ee.ImageCollection(COLLECTION).filterDate(start, end_excl).select(band)

    def per_image(img):
        day = img.get("system:index")
        return img.reduceRegions(points, ee.Reducer.first(), SCALE_M).map(lambda f: f.set("day", day))

    feats = images.map(per_image).flatten().getInfo()["features"]
    return [(f["properties"]["day"], f["properties"]["cell"], f["properties"].get("first")) for f in feats]


def to_frame(rows, cells) -> pd.DataFrame:
    cols = list(cells)
    if not rows:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="date"), dtype=float)
    long = pd.DataFrame(rows, columns=["day", "cell", "value"])
    long["day"] = pd.to_datetime(long["day"], format="%Y%m%d")
    df = long.pivot_table(index="day", columns="cell", values="value", aggfunc="first", dropna=False)
    df = df.reindex(columns=cols).astype(float)
    df.index = pd.DatetimeIndex(df.index, name="date")
    df.columns.name = None
    return df.sort_index()


async def fetch(var: str, cells: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """gridMET `var` ("pr" or "pet") for `cells` (index=cell id, columns lat/lon) over [start, end]."""
    # Earth Engine aborts a query past 5000 elements (one per cell per day), so ask in windows that fit.
    span = max(1, 4500 // max(1, len(cells)))
    rows = []
    for s in pd.date_range(start, end, freq=f"{span}D"):
        e = min(s + pd.Timedelta(days=span - 1), pd.Timestamp(end))
        rows += await asyncio.to_thread(_sample, BANDS[var], cells, s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"))
    return to_frame(rows, cells.index)


def compare(cached: pd.DataFrame, ee_df: pd.DataFrame) -> dict:
    """How Earth Engine's copy lines up with the THREDDS cache on the days and cells both have."""
    days = cached.index.intersection(ee_df.index)
    cols = [c for c in cached.columns if c in ee_df.columns]
    a, b = cached.loc[days, cols], ee_df.loc[days, cols]
    diff = (a - b).abs()
    return {
        "overlap_days": len(days), "cells": len(cols), "missing_cells": int(b.isna().all().sum()),
        "max_abs_diff": float(diff.max().max()) if len(days) else float("nan"),
        "mean_abs_diff": float(diff.stack().mean()) if len(days) else float("nan"),
        "cached_last": str(cached.index.max().date()), "ee_last": str(ee_df.index.max().date()) if len(ee_df) else None,
    }
