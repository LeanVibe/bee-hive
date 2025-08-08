#!/usr/bin/env python3
import asyncio
import json
import os
import sys
from urllib.parse import urlparse

import httpx

try:
    import websockets
except ImportError:
    websockets = None


def to_ws_url(http_url: str) -> str:
    parsed = urlparse(http_url)
    scheme = 'wss' if parsed.scheme == 'https' else 'ws'
    return f"{scheme}://{parsed.netloc}/api/dashboard/ws/dashboard"


async def probe_ws(base_url: str, timeout: float = 5.0) -> None:
    if websockets is None:
        raise RuntimeError("websockets package not installed")
    ws_url = to_ws_url(base_url)
    async with websockets.connect(ws_url, extra_headers={"host": urlparse(base_url).netloc}) as ws:
        msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
        data = json.loads(msg)
        if not isinstance(data, dict) or 'type' not in data:
            raise AssertionError("WS initial message missing 'type'")


async def main():
    base_url = os.environ.get('CANARY_URL') or (len(sys.argv) > 1 and sys.argv[1])
    if not base_url:
        print("CANARY_URL not set; skipping probes.")
        sys.exit(0)

    headers = {"host": urlparse(base_url).netloc}

    async with httpx.AsyncClient(timeout=5.0, headers=headers) as client:
        # /health
        r = await client.get(f"{base_url}/health")
        r.raise_for_status()
        body = r.json()
        assert body.get('status') in ('healthy', 'degraded', 'unhealthy')

        # /metrics
        r = await client.get(f"{base_url}/metrics")
        r.raise_for_status()
        text = r.text
        assert '# HELP' in text and '# TYPE' in text

        # /dashboard/api/live-data
        r = await client.get(f"{base_url}/dashboard/api/live-data")
        r.raise_for_status()
        data = r.json()
        assert 'metrics' in data and isinstance(data['metrics'], dict)

    # WebSocket
    await probe_ws(base_url)

    print("All synthetic probes passed.")


if __name__ == '__main__':
    asyncio.run(main())
