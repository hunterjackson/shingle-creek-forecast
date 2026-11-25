"""Shared GET with retries, used by every data source.

A failed request is retried when the failure is likely to be temporary:
connection errors and timeouts, rate limiting (429) and server errors (5xx).
Waits grow linearly (5, 10, 15, 20 s by default). Other HTTP errors raise at
once, except statuses the caller passes in `allow` (e.g. 404 for files that may
not exist), which are returned for the caller to handle.
"""

from __future__ import annotations

import sys
import time

import httpx

RETRIES = 4
RETRY_WAIT = 5.0
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


def get(
    client: httpx.Client,
    url: str,
    *,
    allow: tuple[int, ...] = (),
    **kw,
) -> httpx.Response:
    retries, retry_wait = RETRIES, RETRY_WAIT  # read at call time so tests can zero the wait
    for attempt in range(retries + 1):
        try:
            resp = client.get(url, **kw)
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
        time.sleep(wait)
    raise AssertionError("unreachable")
