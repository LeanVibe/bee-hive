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
 - PR: run `pytest tests/unit tests/ws tests/smoke` and schema/TS contract tests.
 - Nightly: run full backend + PWA suites and limited mutation tests.
 - Status: Implemented in `.github/workflows/ci.yml` and `.github/workflows/nightly.yml`.

5) Documentation (ongoing)
- README lists WS invariants: error/data_error timestamps; ping/pong includes timestamp.
- This plan remains the living source of truth for reliability work.

## Done (this iteration)
- Centralized WS error frames via `app/api/ws_utils.py` (`make_error`, `make_data_error`).
- All WS endpoints use helpers for invalid JSON/exception paths; unknown types use `make_error`.
- Redis listener supports `psubscribe` pattern channels and handles both `message` and `pmessage` routes.
- `request_data` happy path for `agent_status` implemented and tested.
- PR CI runs focused lanes; PWA job generates TS types from schema and fails on drift; nightly runs full suites.
- WS health endpoint and `ping`/`pong` include timestamps; tests are green.

## Next priorities (plan and breakdown)

1) Add correlation IDs for observability (must-have)
- Goal: Attach a `correlation_id` to outbound WS frames for easier tracing.
- Scope:
  - Extend `make_error`/`make_data_error` to include `correlation_id` (UUID v4).
  - Include `correlation_id` on broadcast frames (`broadcast_to_subscription`, `broadcast_to_all`).
  - Do not change schema requirements (keep as additionalProperties) to avoid breaking clients.
- Tests:
  - Unit: assert `correlation_id` present in `make_error`/`make_data_error`.
  - WS: assert `correlation_id` on generic error and on a `critical_alert` broadcast.

2) Expand `request_data` happy-path tests (high value)
- Add minimal happy-path coverage for:
  - `coordination_metrics` (contains `success_rate`, `trend`).
  - `system_health` (contains `overall_status`, `components`).
- Keep data shapes minimal and backend-agnostic.

3) Tighten schema/type contract checks (nice-to-have)
- Add a backend contract test ensuring enum values in schema are mirrored in TS types (already partly covered, keep maintained).
- Consider a pre-commit hook (optional) to run `mobile-pwa generate:schemas` and fail on drift locally.

4) Operational hygiene (nice-to-have)
- Emit structured logs for broadcast failures including `correlation_id`.
- Consider backoff in `_redis_listener_loop` reconnects (already sleeps; refine thresholds if flakiness observed).

Acceptance criteria for this tranche:
- All WS error and broadcast frames include `correlation_id`.
- New `request_data` tests for `coordination_metrics` and `system_health` green.
- All existing ws/unit/smoke tests remain green.
