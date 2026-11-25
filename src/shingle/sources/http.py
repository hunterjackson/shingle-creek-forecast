"""Shared async HTTP helpers, used by every data source.

`get` retries a failed request when the failure is likely to be temporary:
connection errors and timeouts, rate limiting (429) and server errors (5xx).
Waits grow linearly (5, 10, 15, 20 s by default). Other HTTP errors raise at
once, except statuses the caller passes in `allow` (e.g. 404 for files that may
not exist), which are returned for the caller to handle.

Everything runs on one asyncio event loop: requests overlap while they wait on
the network, with no threads.
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator, Awaitable, Iterable, TypeVar

import httpx

RETRIES = 4
RETRY_WAIT = 5.0
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

T = TypeVar("T")


async def get(client: httpx.AsyncClient, url: str, *, allow: tuple[int, ...] = (), **kw) -> httpx.Response:
    retries, retry_wait = RETRIES, RETRY_WAIT  # read at call time so tests can zero the wait
    for attempt in range(retries + 1):
        try:
            resp = await client.get(url, **kw)
        except httpx.TransportError as exc:
            if attempt == retries:
                raise
            reason = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code not in RETRY_STATUS or attempt == retries:
                if resp.status_code not in allow:
                    resp.raise_for_status()
                return resp
            reason = f"HTTP {resp.status_code}"
        wait = retry_wait * (attempt + 1)
        print(f"retrying {url} in {wait:g}s ({reason}; attempt {attempt + 1} of {retries})", file=sys.stderr)
        await asyncio.sleep(wait)
    raise AssertionError("unreachable")


@asynccontextmanager
async def session(client: httpx.AsyncClient | None, timeout: float, **kw) -> AsyncIterator[httpx.AsyncClient]:
    """Use the caller's client, or open (and afterwards close) a new one."""
    if client is not None:
        yield client
    else:
        async with httpx.AsyncClient(timeout=timeout, **kw) as c:
            yield c


async def gather(aws: Iterable[Awaitable[T]], limit: int) -> list[T]:
    """Await all, at most `limit` at a time; results in input order."""
    sem = asyncio.Semaphore(limit)

    async def one(aw: Awaitable[T]) -> T:
        async with sem:
            return await aw

    return list(await asyncio.gather(*(one(a) for a in aws)))
