import asyncio
import httpx
import pytest
import respx

from shingle.sources import http

URL = "https://example.test/data"


@respx.mock
def test_retries_connection_errors_and_server_errors_then_succeeds():
    route = respx.get(URL).mock(
        side_effect=[httpx.ConnectError("reset"), httpx.ReadTimeout("slow"), httpx.Response(503), httpx.Response(200, json={"ok": 1})]
    )

    resp = asyncio.run(http.get(httpx.AsyncClient(), URL))

    assert resp.json() == {"ok": 1}
    assert route.call_count == 4


@respx.mock
def test_gives_up_after_the_last_retry():
    route = respx.get(URL).mock(side_effect=httpx.ConnectError("down"))

    with pytest.raises(httpx.ConnectError):
        asyncio.run(http.get(httpx.AsyncClient(), URL))

    assert route.call_count == http.RETRIES + 1


@respx.mock
def test_server_error_on_every_attempt_raises_status_error():
    route = respx.get(URL).mock(return_value=httpx.Response(502))

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(http.get(httpx.AsyncClient(), URL))

    assert route.call_count == http.RETRIES + 1


@respx.mock
def test_client_errors_are_not_retried():
    route = respx.get(URL).mock(return_value=httpx.Response(400))

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(http.get(httpx.AsyncClient(), URL))

    assert route.call_count == 1


@respx.mock
def test_allowed_status_is_returned_without_retry():
    route = respx.get(URL).mock(return_value=httpx.Response(404))

    assert asyncio.run(http.get(httpx.AsyncClient(), URL, allow=(404,))).status_code == 404
    assert route.call_count == 1


@respx.mock
def test_rate_limit_is_retried():
    route = respx.get(URL).mock(side_effect=[httpx.Response(429), httpx.Response(200, text="x")])

    assert asyncio.run(http.get(httpx.AsyncClient(), URL)).text == "x"
    assert route.call_count == 2


def test_gather_limits_how_many_run_at_once():
    running, peak = 0, 0

    async def job(i):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return i

    assert asyncio.run(http.gather((job(i) for i in range(10)), limit=3)) == list(range(10))
    assert peak == 3
