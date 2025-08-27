import pytest

import app.api.dashboard_websockets as wsmod


@pytest.mark.asyncio
async def test_redis_listener_exponential_backoff(monkeypatch):
    # Make get_redis raise to force backoff path
    async def failing_get_redis():
        raise RuntimeError("redis unavailable")

    monkeypatch.setattr(wsmod, "get_redis", failing_get_redis)

    sleep_calls = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)
        # Stop after two sleeps to keep test fast
        if len(sleep_calls) >= 2:
            raise KeyboardInterrupt()

    monkeypatch.setattr(wsmod.asyncio, "sleep", fake_sleep)

    # Run the loop and stop after two backoff sleeps
    with pytest.raises(KeyboardInterrupt):
        await wsmod.websocket_manager._redis_listener_loop()

    # Expect exponential backoff: 5 then 10 seconds
    assert sleep_calls[0] == 5
    assert sleep_calls[1] == 10
