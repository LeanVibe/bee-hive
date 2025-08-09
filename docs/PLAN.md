# HiveOps Reliability & WS Contract Hardening Plan (Aug 2025)

## Objective
- Protect the core user journey (mobile PWA live dashboard) by hardening WebSocket contracts and observability.
- Prevent regressions via focused, deterministic tests and CI lanes.

## Decisions
- Centralize WS error-frame creation to guarantee `timestamp` and message shape.
- Enforce schema parity between `schemas/ws_messages.schema.json` and `mobile-pwa/src/types/ws-messages.d.ts` in CI.
- Run ws + smoke lanes on PRs; full test nightly.

## Implementation Steps (detailed)

1) Centralize WS error frames (must-have)
- Helper module `app/api/ws_utils.py` with:
  - `make_error(message: str)` → `{type:"error", message, timestamp}`
  - `make_data_error(data_type: str, message: str)` → `{type:"data_error", data_type, error, timestamp}`
- Refactor `app/api/dashboard_websockets.py` to use helpers for:
  - Unknown message type
  - Invalid JSON errors in all WS endpoints
  - `request_data` unknown/exception paths
- Tests:
  - Unit tests for helpers (timestamp ISO-ish, shape)
  - Strengthen existing ws tests to assert timestamp on generic errors

Acceptance criteria:
- All generic WS errors include `timestamp`
- All data_error include `timestamp` and `error`
- WS tests and smoke tests green

2) Listener routing resilience (must-have)
- Ensure Redis listener handles `psubscribe` and `pmessage` (pattern channels) and plain `message`.
- Tests: unit test `_handle_redis_event` routing to `agents` and `system` subscriptions using an async mock for `broadcast_to_subscription`.

3) request_data happy-path contracts (must-have)
- Add a ws test for `request_data: agent_status` to assert `data_response` shape with `data` object.
- Keep the payload minimal; no backend DB dependency in tests.

4) CI lanes (near-term)
- PR: run `pytest tests/ws tests/smoke tests/unit` and schema/TS contract tests.
- Nightly: `make test`.
- (Implement in a follow-up PR to avoid CI breakage here.)

5) Documentation (ongoing)
- README lists WS invariants: error/data_error timestamps; ping/pong includes timestamp.
- This plan remains the living source of truth for reliability work.

## Done (this iteration)
- Error frames now always include `timestamp`.
- Redis listener supports `psubscribe` pattern channels and `pmessage`.
- Tests added/updated to lock down error timestamps and WS health/ping.
- README documents WS error invariants.
