"""T6 data pull: what a 19:30Z refresh run on day t+1 would know, for selected issue days t in 2020-10..2023-12.

  MRMS MultiSensor_QPE_01H_Pass2, hourly files stamped 13Z..18Z on t+1 (= rain 12Z..18Z), summed, basin bbox
      -> cache/mrms_12to18z.parquet   (index issue day t, columns MRMS pixels 'lat_lon', mm)
  GEFS 12Z run of t+1, ensemble-mean APCP 6 h buckets f06..f168 (same decoding as production)
      -> cache/gefs12z.parquet        (index (run, fhour), columns GEFS cells, mm)

Days: every issue day with issued (00Z) GEFS basin rain over leads 1..3 >= 5 mm, plus a fixed random sample of 150
other days (seed 0) to check the refresh does no harm on dry days -> cache/days.csv.
Resumable: months already written are skipped. Only the 6 hourly radar files per day are archived for this window
(no 6 h product at IEM), ~0.43 MB each.
"""
import asyncio
import gzip
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
import httpx
import numpy as np
import pandas as pd

import fc
from shingle import config, fcst
from shingle.sources import gefs, http, mrms

CACHE = Path(os.path.dirname(os.path.abspath(__file__))) / "cache"
FHOURS_12Z = tuple(range(6, 174, 6))
HOURS = (13, 14, 15, 16, 17, 18)


def select_days() -> pd.DatetimeIndex:
    path = CACHE / "days.csv"
    if path.exists():
        return pd.DatetimeIndex(pd.read_csv(path, parse_dates=["date"])["date"])
    s = fc.state()
    ix = fc.fc_index()
    plain = fcst.build(s["raw"], s["src"], s["tgt"], s["drv"], s["data"])
    areal = pd.DataFrame({k: fcst._areal(plain["pr"][k], s["drv"].weights) for k in (1, 2, 3)}).reindex(ix)
    wet = ix[(areal.sum(axis=1) >= 5).to_numpy()]
    rest = ix.difference(wet)
    dry = pd.DatetimeIndex(np.random.default_rng(0).choice(rest, size=150, replace=False))
    days = wet.union(dry).sort_values()
    pd.DataFrame({"date": days, "wet": days.isin(wet)}).to_csv(path, index=False)
    return days


async def mrms_day(c: httpx.AsyncClient, t: pd.Timestamp) -> pd.Series | None:
    d1 = t + pd.Timedelta(days=1)
    total, n = None, 0
    for hh in HOURS:
        p = "MultiSensor_QPE_01H_Pass2"
        url = f"{mrms.ARCHIVE}/{d1:%Y/%m/%d}/mrms/ncep/{p}/{p}_00.00_{d1:%Y%m%d}-{hh:02d}0000.grib2.gz"
        r = await http.get(c, url, allow=(404,))
        if r.status_code == 404:
            return None  # an hour missing -> the day is unusable
        s = mrms.crop(mrms.decode_grib(gzip.decompress(r.content)), config.BBOX)
        total = s if total is None else total.add(s)
        n += 1
    total.name = t
    return total


async def pull_mrms(days: pd.DatetimeIndex) -> None:
    out = CACHE / "mrms"
    out.mkdir(parents=True, exist_ok=True)
    async with http.session(None, timeout=120) as c:
        for month, grp in pd.Series(days, index=days).groupby(days.to_period("M")):
            path = out / f"{month}.parquet"
            if path.exists():
                continue
            res = await http.gather((mrms_day(c, t) for t in grp.index), limit=6)
            rows = [r for r in res if r is not None]
            df = pd.DataFrame(rows) if rows else pd.DataFrame(index=pd.DatetimeIndex([]))
            df.index.name = "date"
            df.to_parquet(path)
            print(f"mrms {month}: {len(rows)}/{len(grp)} days", flush=True)


async def pull_gefs(days: pd.DatetimeIndex) -> None:
    out = CACHE / "gefs12z"
    out.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=60, limits=httpx.Limits(max_connections=64)) as c:
        for month, grp in pd.Series(days, index=days).groupby(days.to_period("M")):
            path = out / f"{month}.parquet"
            if path.exists():
                continue
            runs = [t + pd.Timedelta(days=1, hours=12) for t in grp.index]
            res = await http.gather((gefs.fetch_run(r, fhours=FHOURS_12Z, bbox=config.BBOX, client=c) for r in runs), limit=6)
            parts = [pd.concat({r: df}, names=["run"]) for r, df in zip(runs, res) if df is not None]
            if parts:
                pd.concat(parts).to_parquet(path)
            print(f"gefs12z {month}: {len(parts)}/{len(grp)} runs", flush=True)


async def main():
    CACHE.mkdir(exist_ok=True)
    days = select_days()
    print(f"{len(days)} issue days", flush=True)
    await asyncio.gather(pull_gefs(days), pull_mrms(days))


if __name__ == "__main__":
    asyncio.run(main())
