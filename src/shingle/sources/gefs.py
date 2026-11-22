"""NOAA GEFSv12 ensemble-mean / spread precipitation forecasts from the AWS archive.

Each forecast-hour file holds ~80 global fields; the .idx sidecar gives byte
offsets, so only the APCP message (~88 KB at 0.5 deg) is downloaded with an
HTTP Range request. APCP is a 6-hour bucket ("18-24 hour acc") in the
6-hourly files.

Timing: forecasts are issued ~07 EST (12Z) on day t+1 for issue day t (same
convention as dataset.issue_reading). The 00Z run of day t+1 is complete by
then; lead k covers gridMET day t+k = 12Z(t+k) .. 12Z(t+k+1), i.e. buckets
ending at f(12+24(k-1)+6) .. f(12+24k).
"""

from __future__ import annotations

import httpx
import numpy as np
import pandas as pd

from home_display.shingle.sources.gridmet import BBox

BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
V12_START = pd.Timestamp("2020-09-24")  # GEFSv12 (atmos/pgrb2ap5 layout, ensemble mean files)
KINDS = {"mean": "geavg", "spread": "gespr"}
RES = 0.5


def url(run: pd.Timestamp, fhour: int, kind: str = "mean") -> str:
    run = pd.Timestamp(run)
    if run < V12_START:
        raise ValueError(f"GEFSv12 archive starts {V12_START.date()}, got {run}")
    hh = f"{run.hour:02d}"
    return f"{BASE}/gefs.{run:%Y%m%d}/{hh}/atmos/pgrb2ap5/{KINDS[kind]}.t{hh}z.pgrb2a.0p50.f{fhour:03d}"


def apcp_range(idx_text: str) -> tuple[int, int, str]:
    """(start, end_exclusive, step) of the APCP message from a GRIB2 .idx listing."""
    lines = [ln for ln in idx_text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        parts = ln.split(":")
        if parts[3] == "APCP":
            if i + 1 >= len(lines):
                raise ValueError("APCP is the last message; end offset unknown")
            end = int(lines[i + 1].split(":")[1])
            return int(parts[1]), end, parts[5].split(" ")[0]
    raise KeyError("no APCP record in idx")


def grid_lons(lon0: float, ni: int, dlon: float) -> np.ndarray:
    """Longitudes of a regular grid row in -180..180 (handles grids starting at 180 and wrapping)."""
    lons = (lon0 + np.arange(ni) * dlon) % 360.0
    return np.round(np.where(lons >= 180, lons - 360, lons), 6)


def decode_crop(raw: bytes, bbox: BBox, margin: float = RES) -> pd.Series:
    """Decode one GRIB2 message (regular lat/lon, 0..360 lon) and keep grid points
    within bbox +- margin. Index 'lat_lon' (4 dp, lon in -180..180), values mm."""
    import eccodes

    h = eccodes.codes_new_from_message(raw)
    try:
        ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
        lat0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
        lat1 = eccodes.codes_get(h, "latitudeOfLastGridPointInDegrees")
        lon0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
        dlon = eccodes.codes_get(h, "iDirectionIncrementInDegrees")
        step = eccodes.codes_get(h, "stepRange")
        vals = eccodes.codes_get_values(h).reshape(nj, ni)
    finally:
        eccodes.codes_release(h)
    lats = np.linspace(lat0, lat1, nj)
    lons = grid_lons(lon0, ni, dlon)
    rows = np.where((lats >= bbox.south - margin) & (lats <= bbox.north + margin))[0]
    cols = np.where((lons >= bbox.west - margin) & (lons <= bbox.east + margin))[0]
    out = {f"{lats[r]:.4f}_{lons[c]:.4f}": float(vals[r, c]) for r in rows for c in cols}
    s = pd.Series(out, name=str(step))
    return s.where(s >= 0)


def _get(client: httpx.Client, u: str, retries: int, retry_wait: float, **kw) -> httpx.Response:
    import time

    for attempt in range(retries + 1):
        try:
            return client.get(u, **kw)
        except httpx.TransportError:
            if attempt == retries:
                raise
            time.sleep(retry_wait * (attempt + 1))


def fetch_run(
    run: pd.Timestamp, fhours=tuple(range(18, 186, 6)), bbox: BBox | None = None, kind: str = "mean",
    client: httpx.Client | None = None, retries: int = 4, retry_wait: float = 5.0,
) -> pd.DataFrame | None:
    """6-hour APCP buckets (mm) for one run: index fhour, columns grid cells. None if the run is missing.
    Transport errors (timeouts, resets) are retried."""
    client = client or httpx.Client(timeout=60)
    rows = {}
    for fh in fhours:
        u = url(run, fh, kind)
        r = _get(client, u + ".idx", retries, retry_wait)
        if r.status_code == 404:
            if not rows:
                return None
            continue
        r.raise_for_status()
        start, end, _ = apcp_range(r.text)
        d = _get(client, u, retries, retry_wait, headers={"Range": f"bytes={start}-{end - 1}"})
        d.raise_for_status()
        rows[fh] = decode_crop(d.content, bbox)
    if not rows:
        return None
    df = pd.DataFrame(rows).T
    df.index.name = "fhour"
    return df


def daily_leads(buckets: pd.DataFrame, leads=tuple(range(1, 8))) -> pd.DataFrame:
    """(run, fhour)-indexed 6 h buckets of 00Z runs -> per issue day t (= run day - 1)
    24 h totals for each lead k, columns (lead, cell). Missing bucket -> NaN."""
    runs = buckets.index.get_level_values("run").unique()
    cells = list(buckets.columns)
    out = {}
    for run in runs:
        b = buckets.xs(run, level="run")
        row = {}
        for k in leads:
            fh = [12 + 24 * (k - 1) + 6 * j for j in (1, 2, 3, 4)]
            ok = all(f in b.index for f in fh)
            for c in cells:
                row[(k, c)] = float(b.loc[fh, c].sum(min_count=4)) if ok else np.nan
        out[pd.Timestamp(run).normalize() - pd.Timedelta(days=1)] = row
    df = pd.DataFrame(out).T
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["lead", "cell"])
    df.index.name = "date"
    return df.sort_index()
