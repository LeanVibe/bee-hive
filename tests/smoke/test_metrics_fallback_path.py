import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_metrics_fallback_when_exporter_fails(async_test_client: AsyncClient, monkeypatch):
    # Force get_prometheus_exporter to raise so fallback is used
    import app.core.prometheus_exporter as pe  # type: ignore

    def boom():
        raise RuntimeError("exporter unavailable")

    monkeypatch.setattr(pe, "get_prometheus_exporter", boom)

    resp = await async_test_client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert "leanvibe_health_status" in text
    assert "leanvibe_uptime_seconds" in text
