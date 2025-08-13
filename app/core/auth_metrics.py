# ruff: noqa
"""
Lightweight authentication metrics counters for REST and WebSocket.

These can be exported by Prometheus endpoints.
"""

from typing import Dict
from threading import Lock

_lock = Lock()

_counters: Dict[str, int] = {
    "auth_success_total_rest": 0,
    "auth_success_total_ws": 0,
    "auth_failure_total_rest": 0,
    "auth_failure_total_ws": 0,
}


def inc(name: str, amount: int = 1) -> None:
    with _lock:
        _counters[name] = _counters.get(name, 0) + amount


def get_all() -> Dict[str, int]:
    with _lock:
        return dict(_counters)
