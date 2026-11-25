"""Rain-forecast screen, step 1 data: lead-1 basin rain from higher-resolution forecasts, 2020-10..2023-12.

Lead 1 of issue day t = 12Z(t+1)..12Z(t+2) from the 00Z run of t+1, as production (gefs.daily_leads):
6 h buckets ending f018, f024, f030, f036.

  GEFS 0.25 deg ensemble mean (atmos/pgrb2sp25 geavg)  -> cache/gefs025.parquet   index (run, fhour), cols lat_lon
  NBM 2.5 km core deterministic 6 h APCP               -> cache/nbm.parquet       index issue day, cols model cells
     (NBM is Lambert conformal; pixels are averaged onto the gridMET model cells like the MRMS radar fill)

Days -> cache/days.csv: issue days with GEFS 0.5 lead-1 basin rain >= 5 mm, or observed gridMET basin rain on t+1
>= 10 mm, plus 150 random others (seed 0). NBM on a subset (all days with either >= 20 mm, the rest sampled to 220).
Resumable by month.
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_test"))
import httpx
import numpy as np
import pandas as pd

import fc
from shingle import basin, config, fcst, grid
from shingle.sources import gefs, http

CACHE = Path(os.path.dirname(os.path.abspath(__file__))) / "cache"
FH = (18, 24, 30, 36)
NBM = "https://noaa-nbm-grib2-pds.s3.amazonaws.com"


def select_days() -> pd.DataFrame:
    path = CACHE / "days.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"]).set_index("date")
    s = fc.state()
    ix = fc.fc_index()
    w = s["drv"].weights
    plain = fcst.build(s["raw"], s["src"], s["tgt"], s["drv"], s["data"])
    g1 = fcst._areal(plain["pr"][1], w).reindex(ix)
    o1 = fcst._areal(s["data"]["pr"], w).shift(-1).reindex(ix)
    core = ix[((g1 >= 5) | (o1 >= 10)).to_numpy()]
    rng = np.random.default_rng(0)
    extra = pd.DatetimeIndex(rng.choice(ix.difference(core), 150, replace=False))
    days = core.union(extra).sort_values()
    heavy = days[((g1.reindex(days) >= 20) | (o1.reindex(days) >= 20)).to_numpy()]
    rest = days.difference(heavy)
    nbm = heavy.union(pd.DatetimeIndex(rng.choice(rest, max(0, 220 - len(heavy)), replace=False)))
    df = pd.DataFrame({"g05": g1.reindex(days), "obs": o1.reindex(days), "nbm": days.isin(nbm)}, index=days)
    df.index.name = "date"
    df.to_csv(path)
    return df


def gefs025_url(run: pd.Timestamp, fh: int) -> str:
    return f"{gefs.BASE}/gefs.{run:%Y%m%d}/00/atmos/pgrb2sp25/geavg.t00z.pgrb2s.0p25.f{fh:03d}"


async def gefs025_run(c, run) -> pd.DataFrame | None:
    rows = {}
    for fh in FH:
        r = await http.get(c, gefs025_url(run, fh) + ".idx", allow=(404,))
        if r.status_code == 404:
            return None
        start, end, _ = gefs.apcp_range(r.text)
        d = await http.get(c, gefs025_url(run, fh), headers={"Range": f"bytes={start}-{end - 1}"})
        rows[fh] = gefs.decode_crop(d.content, config.BBOX, margin=0.5)
    df = pd.DataFrame(rows).T
    df.index.name = "fhour"
    return df


def nbm_url(run: pd.Timestamp, fh: int) -> str:
    return f"{NBM}/blend.{run:%Y%m%d}/00/core/blend.t00z.core.f{fh:03d}.co.grib2"


def nbm_range(idx: str, fh: int) -> tuple[int, int] | None:
    lines = [ln for ln in idx.splitlines() if ln.strip()]
    want = f"APCP:surface:{fh - 6}-{fh} hour acc fcst:"
    for i, ln in enumerate(lines):
        if want in ln and ln.rstrip().endswith(want) and i + 1 < len(lines):
            return int(ln.split(":")[1]), int(lines[i + 1].split(":")[1])
    return None


_LL = {}


def nbm_decode(raw: bytes, tgt: pd.DataFrame) -> pd.Series:
    """Decode an NBM message and average its pixels onto the model cells (keyed by grid shape, lat/lon cached)."""
    import eccodes

    h = eccodes.codes_new_from_message(raw)
    try:
        ni, nj = eccodes.codes_get(h, "Nx"), eccodes.codes_get(h, "Ny")
        vals = eccodes.codes_get_values(h)
        if (ni, nj) not in _LL:
            lat = eccodes.codes_get_array(h, "latitudes")
            lon = eccodes.codes_get_array(h, "longitudes")
            lon = np.where(lon > 180, lon - 360, lon)
            b = config.BBOX
            keep = np.where((lat >= b.south - 0.05) & (lat <= b.north + 0.05) & (lon >= b.west - 0.05) & (lon <= b.east + 0.05))[0]
            _LL[(ni, nj)] = (keep, [f"{a:.4f}_{o:.4f}" for a, o in zip(lat[keep], lon[keep])])
    finally:
        eccodes.codes_release(h)
    keep, ids = _LL[(ni, nj)]
    v = vals[keep].astype(float)
    v[(v < 0) | (v > 1e4)] = np.nan
    px = pd.DataFrame([v], columns=ids)
    return grid.cell_means(px, tgt, half_width=basin.GRIDMET_RES / 2).iloc[0]


async def nbm_day(c, t, tgt) -> pd.Series | None:
    run = t + pd.Timedelta(days=1)
    total = None
    for fh in FH:
        r = await http.get(c, nbm_url(run, fh) + ".idx", allow=(404,))
        if r.status_code == 404:
            return None
        rg = nbm_range(r.text, fh)
        if rg is None:
            return None
        d = await http.get(c, nbm_url(run, fh), headers={"Range": f"bytes={rg[0]}-{rg[1] - 1}"})
        s = await asyncio.to_thread(nbm_decode, d.content, tgt)
        total = s if total is None else total + s
    total.name = t
    return total


async def pull(name, days, one, limit):
    out = CACHE / name
    out.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_connections=32)) as c:
        for month, grp in pd.Series(days, index=days).groupby(days.to_period("M")):
            path = out / f"{month}.parquet"
            if path.exists():
                continue
            res = await http.gather((one(c, t) for t in grp.index), limit=limit)
            got = [(t, r) for t, r in zip(grp.index, res) if r is not None]
            if got:
                if name == "gefs025":
                    df = pd.concat({t + pd.Timedelta(days=1): r for t, r in got}, names=["run"])
                else:
                    df = pd.DataFrame({t: r for t, r in got}).T
                    df.index.name = "date"
                df.to_parquet(path)
            print(f"{name} {month}: {len(got)}/{len(grp)}", flush=True)


async def main():
    CACHE.mkdir(exist_ok=True)
    days = select_days()
    tgt = fc.state()["tgt"]
    print(f"{len(days)} days, NBM on {int(days['nbm'].sum())}", flush=True)
    await asyncio.gather(
        pull("gefs025", days.index, lambda c, t: gefs025_run(c, t + pd.Timedelta(days=1)), 8),
        pull("nbm", days.index[days["nbm"].to_numpy()], lambda c, t: nbm_day(c, t, tgt), 4),
    )


if __name__ == "__main__":
    asyncio.run(main())
