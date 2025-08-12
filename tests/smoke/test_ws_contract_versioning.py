import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.asyncio


def test_ws_contract_info_endpoint(test_app):
    client = TestClient(test_app)
    resp = client.get('/api/dashboard/websocket/contract', headers={'host': 'localhost:8000'})
    assert resp.status_code == 200
    data = resp.json()
    assert 'current_version' in data
    assert isinstance(data.get('supported_versions'), list)


def test_ws_limits_exposes_supported_versions(test_app):
    client = TestClient(test_app)
    resp = client.get('/api/dashboard/websocket/limits', headers={'host': 'localhost:8000'})
    assert resp.status_code == 200
    data = resp.json()
    assert 'supported_versions' in data
    assert 'contract_version' in data
