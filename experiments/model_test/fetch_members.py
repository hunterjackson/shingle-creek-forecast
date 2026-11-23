"""GEFSv12 individual members (gep01..gepNN) APCP 6 h buckets over the basin bbox, 00Z runs.

-> data/gefs/members/YYYY-MM.parquet, index (run, member, fhour), columns grid cells (mm). Resumable by month.
Usage: uv run python experiments/model_test/fetch_members.py START END [N_MEMBERS] [FIRST_MEMBER]
"""
import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd

from shingle import config
from shingle.sources import gefs

FHOURS = tuple(range(18, 186, 6))
OUT = Path("data/gefs/members")


def member_url(run, m, fh):
    return f"{gefs.BASE}/gefs.{run:%Y%m%d}/00/atmos/pgrb2ap5/gep{m:02d}.t00z.pgrb2a.0p50.f{fh:03d}"


def main(start, end, n=10, first=1):
    OUT.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(timeout=60, limits=httpx.Limits(max_connections=48))
    members = list(range(first, first + n))

    def one(args):
        run, m, fh = args
        u = member_url(run, m, fh)
        for attempt in range(5):
            try:
                r = client.get(u + ".idx")
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                s, e, _ = gefs.apcp_range(r.text)
                d = client.get(u, headers={"Range": f"bytes={s}-{e - 1}"})
                d.raise_for_status()
                v = gefs.decode_crop(d.content, config.BBOX)
                return (run, m, fh), v
            except (httpx.TransportError, httpx.HTTPStatusError):
                time.sleep(3 * (attempt + 1))
        return None

    for month in pd.period_range(start, end, freq="M"):
        path = OUT / f"{month}.parquet"
        runs = pd.date_range(max(pd.Timestamp(start), month.start_time), min(pd.Timestamp(end), month.end_time.normalize()))
        if path.exists():
            have = pd.read_parquet(path)
            got = set(have.index.get_level_values("member"))
            if set(members) <= got:
                continue
        else:
            have = None
        t = time.time()
        with ThreadPoolExecutor(32) as ex:
            res = [x for x in ex.map(one, [(r, m, fh) for r in runs for m in members for fh in FHOURS]) if x is not None]
        df = pd.DataFrame({k: v for k, v in res}).T
        df.index.names = ["run", "member", "fhour"]
        if have is not None:
            df = pd.concat([have, df])
            df = df[~df.index.duplicated(keep="last")]
        df.sort_index().to_parquet(path)
        print(f"{month}: {len(res)} msgs in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], int(a[2]) if len(a) > 2 else 10, int(a[3]) if len(a) > 3 else 1)
