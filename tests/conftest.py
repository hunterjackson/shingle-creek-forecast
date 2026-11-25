import pytest

from shingle.sources import http


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch):
    """Retries still happen in tests, without the real waits between them."""
    monkeypatch.setattr(http, "RETRY_WAIT", 0.0)
