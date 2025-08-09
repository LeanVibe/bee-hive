# HiveOps Reliability & WS Contract Hardening Plan (Aug 2025)

## Objective
- Protect the core user journey (mobile PWA live dashboard) by hardening WebSocket contracts and observability.
- Prevent regressions via focused, deterministic tests and CI lanes.

## Decisions
- Centralize WS error-frame creation to guarantee `timestamp` and message shape.
- Enforce schema parity between `schemas/ws_messages.schema.json` and `mobile-pwa/src/types/ws-messages.d.ts` in CI.
- Run ws + smoke lanes on PRs; full test nightly.

## Implementation Steps
1) Centralized WS error helper
- Add `app/api/ws_utils.py` with `make_error(type="error", message: str) -> dict` and `make_data_error(data_type: str, message: str) -> dict` that include `timestamp`.
- Replace inline error dicts in `app/api/dashboard_websockets.py` with helper calls.
- Tests: unit-test helper functions and re-run ws suite.

2) Enforce schema-to-TS parity in CI
- Ensure the existing script to generate TS from JSON schema is validated in tests (already partially present).
- Add a CI check to fail if types drift.

3) CI lanes
- PR workflow: run `pytest tests/ws tests/smoke` (fast, deterministic) and contracts.
- Nightly: `make test` full suite.

4) Optional: request_data happy-path contracts
- Add a test asserting shape for known `request_data` types.

## Done (this iteration)
- Error frames now always include `timestamp`.
- Redis listener supports `psubscribe` pattern channels and `pmessage`.
- Tests added/updated to lock down error timestamps and WS health/ping.
- README documents WS error invariants.
