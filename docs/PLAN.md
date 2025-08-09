# HiveOps Reliability & WS Contract Hardening Plan (Aug 2025)

## Objective
- Protect the core user journey (mobile PWA live dashboard) by hardening WebSocket contracts and observability.
- Prevent regressions via focused, deterministic tests and CI lanes.

## Single source of truth — Dashboard WS contract and ops

- Endpoints
  - WS multi-subscription: `/api/dashboard/ws/dashboard`
  - WS health/stats: `/api/dashboard/websocket/health`, `/api/dashboard/websocket/stats`
  - WS broadcast (admin): `/api/dashboard/websocket/broadcast`
  - WS limits: `/api/dashboard/websocket/limits` (server thresholds)
  - Prometheus WS metrics: `/api/dashboard/metrics/websockets`

- Schema & Types
  - JSON Schema: `schemas/ws_messages.schema.json`
  - TS Types: `mobile-pwa/src/types/ws-messages.d.ts` (generated in CI)
  - Message variants: `connection_established`, `dashboard_initialized`, `subscription_updated`, `*_update`, `data_response`, `error`, `data_error`, `critical_alert`
  - Additional fields allowed; `correlation_id` is present on outbound frames
  - `contract_version` is included in `connection_established` and `dashboard_initialized`

- Invariants
  - All `error` and `data_error` frames include `timestamp` and `correlation_id`
  - `pong` includes `timestamp`
  - `subscription_updated` lists are unique and sorted
  - Unknown message types and subscriptions generate `error`

- Operational limits (current defaults)
  - Rate limit: 20 msgs/sec per connection; burst 40; cooldown 5s between rate-limit errors
  - Max inbound WS message: 64KB
  - Max subscriptions per connection: 10
  - Idle disconnect: 10 minutes (disconnect_notice sent before disconnect)
  - Backpressure: disconnect after 5 consecutive send failures

- Observability
  - Counters: messages_sent_total, messages_send_failures_total, messages_received_total, messages_dropped_rate_limit_total, errors_sent_total, connections_total, disconnections_total
  - Exposed via Prometheus `/api/dashboard/metrics/websockets`
  - Structured logs include `correlation_id`, `type`, `subscription`
  - Limits endpoint: `/api/dashboard/websocket/limits` returns all thresholds and `contract_version`

- CI
  - PR: run `pytest tests/unit tests/ws tests/smoke -q`; PWA generates TS types and fails on drift
  - Nightly: full test suites and limited mutation tests

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

1) WS contract completeness and observability (must-have)
- Correlation IDs for tracing: Implemented (helpers + broadcasts) with tests.
- Schema completeness: Add explicit `data_response` and `data_error` entries to schema and regenerate TS types.
- Structured logs: Include `correlation_id`, `type`, `subscription` on send failures.
- Redis backoff: Exponential backoff on listener reconnects to improve resilience.

2) Rate limiting and flood protection (must-have)
- Goal: Protect server from client floods and misbehaving subscriptions without impacting UX.
- Scope:
  - Per-connection simple token bucket for incoming messages (e.g., 20 msgs/sec burst 40).
  - Drop or delay when exceeded; send a single `error` with reason and then silence until within limits.
  - Broadcast loop safeguards: cap per-connection queued messages; disconnect if backpressure persists.
- Tests:
  - Unit: token bucket behavior (allowance refills over time).
  - WS: send >N messages in a tight loop and assert only limited processing plus at least one rate-limit error.

3) Subscription validation and normalization (high value)
- Ensure `subscribe`/`unsubscribe` accept only known subscriptions; unknown are ignored with an `error`.
- Normalize duplicates; always return a sorted `subscriptions` list for deterministic diffs.
- Tests: subscribe to unknown + duplicates; verify error frame and sorted list.

4) CI/local parity (nice-to-have)
- Optional pre-commit hook to run `npm -w mobile-pwa run generate:schemas` and fail on drift.
- Document in CONTRIBUTING; keep CI as source of truth.

Acceptance criteria for this tranche:
- Rate limiting enforced with tests proving drop/notify behavior without flaking.
- Subscription validation errors produced for unknowns; updates return sorted lists.
- Schema and TS types include `data_response` and `data_error`; all tests green.

## Next priorities (new, higher-impact)

1) WS observability metrics (must-have)
- Add counters in manager: `messages_sent_total`, `messages_send_failures_total`, `messages_received_total`, `messages_dropped_rate_limit_total`, `errors_sent_total`, `connections_total`, `disconnections_total`.
- Expose them via Prometheus at `/api/dashboard/metrics/websockets`.
- Tests: metrics endpoint includes these names.

2) Input hardening and safety (must-have)
- Cap inbound WS message size (e.g., 64KB). On exceed: drop and send a single error message.
- Cap max subscriptions per connection (e.g., 10) with clear error for extras (future-proofing; current set is <=5).
- Ensure every outbound frame has a `correlation_id` (inject at send if missing).
- Tests: invalid large message returns an error; correlation id injection remains schema-compatible.

3) Idle timeout hygiene (nice-to-have)
- Disconnect connections idle beyond threshold with a `disconnect_notice` reason.
- Optional server-initiated ping in future if needed (skipped for now to avoid client-side impact).

## Roadmap (proposed after consolidation)

1) Backpressure and queue metrics (must-have)
- Track per-connection send failures and consecutive failure streak; disconnect after threshold
- Expose gauge for current connection count by subscription, and a counter for disconnects due to backpressure
- Tests: simulate send failures and assert disconnect path

2) Contract versioning (high value)
- Add optional `contract_version` on `connection_established` and `dashboard_initialized`; expose server version via `/health`
- Tests: presence and type only; do not gate on specific value

3) Chaos and recovery tests (high value)
- Simulate Redis disconnects and ensure listener backoff and recovery (unit with monkeypatch)
- WS reconnection strategy documented for the PWA (follow-up in PWA repo)

4) Security hardening (nice-to-have)
- Optional origin allowlist and token-based auth for WS (feature-flag); tests off by default

Acceptance criteria for next wave:
- Backpressure disconnect logic in place and covered by tests
- Version fields present where applicable without breaking schema
- Chaos test validates Redis listener resilience
