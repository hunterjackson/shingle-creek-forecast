"""Phase 1 data pull (docs/forecast_training_plan.md): GEFSv12 control member (c00) 6 h APCP, basin crop.

  rf    reforecast, s3://noaa-gefs-retrospective/GEFSv12/reforecast/YYYY/YYYYMMDD00/c00/Days:1-10/apcp_sfc_*.grib2
        0.25 deg, one file per run holding 3 h and 6 h buckets; the .idx gives byte ranges, so only the 28
        "(X-6)-X hour acc" messages with X in gefs.FHOURS (18..180) are fetched (~300 KB each).
  live  operational GEFSv12 c00, atmos/pgrb2ap5/gec00 (0.5 deg, same files layout as production's geavg),
        issue days 2020-10-01..2023-12-31: the live-era control for the 1a gate.

Issue day t uses the 00Z run of t+1 (as production). Stored per month under cache/{name}/YYYY-MM.parquet with index
(run, fhour), columns 'lat_lon' cells within the model bbox +- 0.5 deg, values mm (same layout as data/gefs).
Resumable. Usage: p1_fetch.py rf 2004 2017 2013 | p1_fetch.py live
"""
import asyncio
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import httpx
import pandas as pd

from shingle import config
from shingle.sources import gefs, http

CACHE = Path(os.path.dirname(os.path.abspath(__file__))) / "cache"
RF = "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"
REQ = asyncio.Semaphore(24)
POOL = ProcessPoolExecutor(8)
NBYTES = 0


def rf_url(run: pd.Timestamp) -> str:
    return f"{RF}/{run:%Y}/{run:%Y%m%d}00/c00/Days:1-10/apcp_sfc_{run:%Y%m%d}00_c00.grib2"


def live_url(run: pd.Timestamp, fh: int) -> str:
    return f"{gefs.BASE}/gefs.{run:%Y%m%d}/00/atmos/pgrb2ap5/gec00.t00z.pgrb2a.0p50.f{fh:03d}"


def six_hour_ranges(idx: str) -> dict[int, tuple[int, int]]:
    """fhour X -> byte range of the '(X-6)-X hour acc' APCP message, for X in gefs.FHOURS."""
    lines = [ln for ln in idx.splitlines() if ln.strip()]
    out = {}
    for i, ln in enumerate(lines[:-1]):
        p = ln.split(":")
        if p[3] != "APCP":
            continue
        a, b = p[5].split(" ")[0].split("-")
        if int(b) - int(a) == 6 and int(b) in gefs.FHOURS:
            out[int(b)] = (int(p[1]), int(lines[i + 1].split(":")[1]))
    return out


async def get(c, url, **kw):
    global NBYTES
    async with REQ:
        r = await http.get(c, url, **kw)
    NBYTES += len(r.content)
    return r


async def decode(raw: bytes, margin: float) -> pd.Series:
    return await asyncio.get_running_loop().run_in_executor(POOL, gefs.decode_crop, raw, config.BBOX, margin)


async def rf_run(c, run) -> pd.DataFrame | None:
    r = await get(c, rf_url(run) + ".idx", allow=(404, 403))
    if r.status_code != 200:
        return None
    rg = six_hour_ranges(r.text)
    if len(rg) != len(gefs.FHOURS):
        print(f"{run.date()}: only {len(rg)} 6 h buckets in idx", file=sys.stderr)

    async def one(fh):
        a, b = rg[fh]
        d = await get(c, rf_url(run), headers={"Range": f"bytes={a}-{b - 1}"})
        return fh, await decode(d.content, 0.5)

    rows = dict(await asyncio.gather(*(one(fh) for fh in sorted(rg))))
    df = pd.DataFrame(rows).T
    df.index.name = "fhour"
    return df


async def live_run(c, run) -> pd.DataFrame | None:
    async def one(fh):
        r = await get(c, live_url(run, fh) + ".idx", allow=(404,))
        if r.status_code == 404:
            return fh, None
        a, b, _ = gefs.apcp_range(r.text)
        d = await get(c, live_url(run, fh), headers={"Range": f"bytes={a}-{b - 1}"})
        return fh, await decode(d.content, gefs.RES)

    rows = {fh: s for fh, s in await asyncio.gather(*(one(fh) for fh in gefs.FHOURS)) if s is not None}
    if not rows:
        return None
    df = pd.DataFrame(rows).T
    df.index.name = "fhour"
    return df


async def pull(name, issue_days, one):
    out = CACHE / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    async with httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_connections=32)) as c:
        for month, grp in pd.Series(issue_days, index=issue_days).groupby(issue_days.to_period("M")):
            path = out / f"{month}.parquet"
            if path.exists():
                continue
            runs = [t + pd.Timedelta(days=1) for t in grp.index]
            res = await asyncio.gather(*(one(c, r) for r in runs))
            got = {r: df for r, df in zip(runs, res) if df is not None}
            if got:
                pd.concat(got, names=["run"]).to_parquet(path)
            mb = NBYTES / 1e6
            print(f"{name} {month}: {len(got)}/{len(runs)} runs, {mb:,.0f} MB so far, {mb / (time.time() - t0):.1f} MB/s", flush=True)


async def main(argv):
    if argv[0] == "rf":
        days = pd.DatetimeIndex([d for y in argv[1:] for d in pd.date_range(f"{y}-01-01", f"{y}-12-31")])
        await pull("rf_c00", days, rf_run)
    elif argv[0] == "live":
        await pull("live_c00", pd.date_range("2020-10-01", "2023-12-31"), live_run)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
