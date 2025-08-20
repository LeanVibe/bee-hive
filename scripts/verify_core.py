import asyncio
import json
import os

import httpx
import websockets

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
WS_URL = BASE_URL.replace("http", "ws")


async def check_health(client: httpx.AsyncClient) -> None:
    r = await client.get(f"{BASE_URL}/health", headers={"host": "localhost:8000"})
    r.raise_for_status()
    body = r.json()
    assert "components" in body and isinstance(body["components"], dict)


async def check_metrics(client: httpx.AsyncClient) -> None:
    r = await client.get(f"{BASE_URL}/metrics", headers={"host": "localhost:8000"})
    r.raise_for_status()
    text = r.text
    assert "# HELP" in text and "# TYPE" in text


async def check_status(client: httpx.AsyncClient) -> None:
    r = await client.get(f"{BASE_URL}/status", headers={"host": "localhost:8000"})
    r.raise_for_status()
    body = r.json()
    assert isinstance(body.get("components"), dict)


async def check_ws() -> None:
    uri = f"{WS_URL}/api/dashboard/ws/dashboard"
    async with websockets.connect(uri, extra_headers={("host", "localhost:8000")}) as ws:
        # Initial
        _ = await ws.recv()
        # Ping
        await ws.send(json.dumps({"type": "ping"}))
        _ = await ws.recv()


async def check_ws_stats(client: httpx.AsyncClient) -> None:
    r = await client.get(f"{BASE_URL}/api/dashboard/websocket/stats", headers={"host": "localhost:8000"})
    r.raise_for_status()
    body = r.json()
    assert "websocket_stats" in body


async def main() -> None:
    async with httpx.AsyncClient(timeout=5.0) as client:
        await check_health(client)
        await check_metrics(client)
        await check_status(client)
        await check_ws_stats(client)
    await check_ws()
    print("verify_core: OK")


if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class VerifyCoreScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            await main()
            
            return {"status": "completed"}
    
    script_main(VerifyCoreScript)
