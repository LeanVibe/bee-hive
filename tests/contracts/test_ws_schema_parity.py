import json
from pathlib import Path


def test_ws_schema_parity():
    schema_path = Path("schemas/ws_messages.schema.json")
    ts_path = Path("mobile-pwa/src/types/ws-messages.d.ts")

    assert schema_path.exists(), "JSON schema missing"
    assert ts_path.exists(), "TypeScript definitions missing"

    schema = json.loads(schema_path.read_text())
    ts = ts_path.read_text()

    # Basic titles/types present in TS
    required_titles = [
        "ConnectionEstablished",
        "DashboardInitialized",
        "SubscriptionUpdated",
        "UpdateMessage",
        "ErrorMessage",
        "CriticalAlertMessage",
    ]
    for title in required_titles:
        assert title in ts, f"Missing TS interface for {title}"

    # Validate enum set for subscription in UpdateMessage
    subs = ["agents", "coordination", "tasks", "system", "alerts"]
    for s in subs:
        assert f'"{s}"' in ts, f"Subscription '{s}' not reflected in TS types"

    # Ensure error and critical_alert types exist
    assert 'type: "error"' in ts
    assert 'type: "critical_alert"' in ts
