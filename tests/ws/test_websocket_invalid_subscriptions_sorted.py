import json
import pytest


@pytest.mark.asyncio
async def test_invalid_subscription_emits_error_and_sorted_subscriptions(test_app):
    from starlette.testclient import TestClient

    client = TestClient(test_app)

    with client.websocket_connect(
        "/api/dashboard/ws/dashboard", headers={"host": "localhost:8000"}
    ) as ws:
        # Drain initial frame
        _ = json.loads(ws.receive_text())

        # Request invalid + duplicate subscriptions
        ws.send_text(
            json.dumps(
                {
                    "type": "subscribe",
                    "subscriptions": ["unknown", "agents", "agents", "coordination", "zzz"],
                }
            )
        )

        saw_error = False
        saw_update = False

        # Read a few frames to account for background broadcasts
        for _ in range(10):
            msg = json.loads(ws.receive_text())
            mtype = msg.get("type")

            if mtype == "error":
                # Should mention invalid subscriptions (sorted)
                assert "Invalid subscription(s):" in msg.get("message", "")
                assert (
                    ", ".join(sorted(["unknown", "zzz"])) in msg.get("message", "")
                )
                saw_error = True

            if mtype == "subscription_updated":
                subs = msg.get("subscriptions", [])
                # Duplicates removed, sorted list
                assert subs == sorted(subs)
                # Should contain only known ones from request
                assert "agents" in subs and "coordination" in subs
                assert "unknown" not in subs and "zzz" not in subs
                saw_update = True

            if saw_error and saw_update:
                break

        assert saw_error and saw_update
