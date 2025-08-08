import asyncio
from datetime import datetime, timedelta
import pytest

from app.core.circuit_breaker import CircuitBreaker, CircuitBreakerState


@pytest.mark.asyncio
async def test_circuit_breaker_transitions_and_metrics():
    cb = CircuitBreaker(name="test", failure_threshold=2, success_threshold=1, timeout_seconds=0)

    # Define functions
    def ok():
        return "ok"

    def boom():
        raise RuntimeError("fail")

    # Start in CLOSED
    assert await cb.get_state() == CircuitBreakerState.CLOSED

    # Cause two failures to OPEN breaker
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(boom)

    # With timeout_seconds=0, implementation may flip to HALF_OPEN immediately
    assert await cb.get_state() in {CircuitBreakerState.OPEN, CircuitBreakerState.HALF_OPEN}

    # Advance failure time to allow half-open (timeout_seconds=0)
    cb._last_failure_time = datetime.utcnow() - timedelta(seconds=1)  # type: ignore[attr-defined]
    assert await cb.get_state() in {CircuitBreakerState.HALF_OPEN, CircuitBreakerState.OPEN}

    # Successful call in HALF_OPEN should close (success_threshold=1)
    result = await cb.call(ok)
    assert result == "ok"
    assert await cb.get_state() == CircuitBreakerState.CLOSED

    metrics = cb.get_metrics()
    assert metrics["name"] == "test"
    assert metrics["total_requests"] >= 3
