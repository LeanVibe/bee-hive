import os

import pytest
from fastapi.testclient import TestClient

def test_login_me_and_protected_routes(test_app):
    client = TestClient(test_app)

    # Login with default admin credentials from auth service defaults
    payload = {
        "email": os.getenv("DEFAULT_ADMIN_EMAIL", "admin@leanvibe.com"),
        "password": os.getenv("DEFAULT_ADMIN_PASSWORD", "AdminPassword123!")
    }
    res = client.post("/api/v1/auth/login", json=payload, headers={"host": "localhost:8000"})
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data and "refresh_token" in data

    access = data["access_token"]

    # /me should work
    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {access}", "host": "localhost:8000"})
    assert me.status_code == 200
    user = me.json()
    assert user["email"] == payload["email"]

    # Protected ping should require auth
    ping_unauth = client.get("/api/v1/protected/ping", headers={"host": "localhost:8000"})
    assert ping_unauth.status_code == 401

    ping_auth = client.get("/api/v1/protected/ping", headers={"Authorization": f"Bearer {access}", "host": "localhost:8000"})
    assert ping_auth.status_code == 200

    # Admin-only route
    admin = client.get("/api/v1/protected/admin", headers={"Authorization": f"Bearer {access}", "host": "localhost:8000"})
    assert admin.status_code == 200


def test_refresh_accepts_json_and_returns_token(test_app):
    client = TestClient(test_app)
    payload = {
        "email": os.getenv("DEFAULT_ADMIN_EMAIL", "admin@leanvibe.com"),
        "password": os.getenv("DEFAULT_ADMIN_PASSWORD", "AdminPassword123!")
    }
    res = client.post("/api/v1/auth/login", json=payload, headers={"host": "localhost:8000"})
    assert res.status_code == 200
    tokens = res.json()

    refresh = client.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}, headers={"host": "localhost:8000"})
    assert refresh.status_code == 200
    body = refresh.json()
    assert "access_token" in body and body["success"] is True
