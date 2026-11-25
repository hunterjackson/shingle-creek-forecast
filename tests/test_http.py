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

    resp = http.get(httpx.Client(), URL)

    assert resp.json() == {"ok": 1}
    assert route.call_count == 4


@respx.mock
def test_gives_up_after_the_last_retry():
    route = respx.get(URL).mock(side_effect=httpx.ConnectError("down"))

    with pytest.raises(httpx.ConnectError):
        http.get(httpx.Client(), URL)

    assert route.call_count == http.RETRIES + 1


@respx.mock
def test_server_error_on_every_attempt_raises_status_error():
    route = respx.get(URL).mock(return_value=httpx.Response(502))

    with pytest.raises(httpx.HTTPStatusError):
        http.get(httpx.Client(), URL)

    assert route.call_count == http.RETRIES + 1


@respx.mock
def test_client_errors_are_not_retried():
    route = respx.get(URL).mock(return_value=httpx.Response(400))

    with pytest.raises(httpx.HTTPStatusError):
        http.get(httpx.Client(), URL)

    assert route.call_count == 1


@respx.mock
def test_allowed_status_is_returned_without_retry():
    route = respx.get(URL).mock(return_value=httpx.Response(404))

    assert http.get(httpx.Client(), URL, allow=(404,)).status_code == 404
    assert route.call_count == 1


@respx.mock
def test_rate_limit_is_retried():
    route = respx.get(URL).mock(side_effect=[httpx.Response(429), httpx.Response(200, text="x")])

    assert http.get(httpx.Client(), URL).text == "x"
    assert route.call_count == 2
