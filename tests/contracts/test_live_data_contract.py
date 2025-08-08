import json
from typing import Any, Dict

import pytest
import jsonschema
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_live_dashboard_data_contract(async_test_client: AsyncClient):
    # Fetch live data from compat endpoint
    resp = await async_test_client.get("/dashboard/api/live-data")
    assert resp.status_code == 200
    data: Dict[str, Any] = resp.json()

    # Load schema and validate
    with open("schemas/live_dashboard_data.schema.json", "r") as f:
        schema = json.load(f)
    jsonschema.validate(data, schema)

    # Spot-check essential values exist (even in fallback)
    assert "metrics" in data and "system_status" in data["metrics"]
    assert isinstance(data["agent_activities"], list)
    assert isinstance(data["project_snapshots"], list)
    assert isinstance(data["conflict_snapshots"], list)
