# Next 4 Epics — Detailed Plan (Aug 2025)

This document outlines four high-impact epics for HiveOps with clear, testable tasks. It builds on the current reliability work and aligns with `docs/PLAN.md` invariants and CI policy.

## Epic 1: WS Observability Metrics and Structured Logging

Objective: Provide actionable, production-grade observability for WebSocket operations.

Acceptance criteria:
- `/api/dashboard/metrics/websockets` exposes at least:
  - messages_sent_total, messages_send_failures_total, messages_received_total,
    messages_dropped_rate_limit_total, errors_sent_total, connections_total,
    disconnections_total, backpressure_disconnects_total
- Send-failure logs include: `correlation_id`, `type`, `subscription`.
- Smoke test verifies metrics endpoint includes names above.

Tasks:
- Metrics
  - Ensure counters increment at all send/receive/drop/disconnect points in `app/api/dashboard_websockets.py`.
  - Ensure `/api/dashboard/metrics/websockets` exports counters from the manager.
- Logging
  - Confirm send-failure logs include `correlation_id`, `type`, `subscription`.
  - Add guard logs for unknown message types and invalid subscriptions.
- Tests
  - `tests/smoke/test_ws_metrics_exposition.py`: assert metrics names present.
  - `tests/ws/test_websocket_broadcast_and_stats.py`: broadcast → counters increment.
  - `tests/ws/test_websocket_error_paths.py`: failure increments and error frames include `correlation_id`.

## Epic 2: Rate Limiting and Backpressure Safeguards

Objective: Protect server from floods and slow consumers without degrading UX.

Acceptance criteria:
- Per-connection token bucket: 20 rps, burst 40, single error message per cooldown when exceeded.
- Backpressure disconnect after N consecutive send failures (default 5).
- Limits visible at `/api/dashboard/websocket/limits`.

Tasks:
- Implement token bucket and enforcement in inbound handler.
- Cap send queue (implicit via failure streak) and disconnect on threshold.
- Extend limits endpoint with thresholds and `contract_version`.
- Tests
  - `tests/unit/test_rate_limit.py`: bucket refill semantics.
  - `tests/ws/test_ws_rate_limit_behavior.py`: spam client → limited processing + one rate-limit error.
  - `tests/unit/test_backpressure_disconnect.py`: send failures → disconnect.
  - `tests/smoke/test_ws_limits_endpoint.py`: thresholds present.

## Epic 3: Input Hardening and Subscription Validation

Objective: Prevent malformed, oversized, or invalid subscription inputs.

Acceptance criteria:
- Unknown subscriptions generate a single `error`; duplicates normalized; `subscription_updated` sorted and unique.
- Inbound message size capped (64KB); oversize → single `error` and drop.
- Every outbound frame includes `correlation_id`.

Tasks:
- Enforce and normalize `subscribe`/`unsubscribe` paths.
- Add inbound size check before processing.
- Inject `correlation_id` when missing just before send.
- Schema parity
  - Keep `schemas/ws_messages.schema.json` as source of truth; ensure TS regenerated in CI.
- Tests
  - `tests/ws/test_websocket_invalid_subscriptions_sorted.py`.
  - `tests/unit/test_ws_oversized_message.py`.
  - `tests/ws/test_websocket_message_contract.py`.

## Epic 4: Chaos/Recovery and Contract Versioning Rollout

Objective: Validate resilience to Redis failures and surface versioning for clients.

Acceptance criteria:
- Redis listener uses exponential backoff on failures and recovers.
- `contract_version` present in `connection_established` and `dashboard_initialized`; surfaced via `/health` and `/api/dashboard/websocket/limits`.
- PWA reconnection strategy documented with minimal test coverage.

Tasks:
- Implement backoff in Redis listener.
- Ensure version fields are included and echoed in health/limits endpoints.
- Tests
  - `tests/unit/test_redis_listener_backoff.py`.
  - `tests/ws/test_websocket_initial_frame.py` (already asserts version presence).
  - `tests/smoke/test_app_health_and_ws.py`.
- PWA
  - Add docs for reconnection behavior.

---

## Cross-Cutting: CI/Local Parity for WS Schema

Objective: Avoid drift between `schemas/ws_messages.schema.json` and PWA types while keeping local friction low.

Acceptance criteria:
- An opt-in local check exists to detect drift quickly.
- CI remains the single source of truth and fails on drift (already configured).

Tasks:
- Add `mobile-pwa/scripts/check-schema-parity.mjs` to regenerate into a temp file and compare to `src/types/ws-messages.d.ts`.
- Add `npm run check:schemas` in `mobile-pwa/package.json`.
- Document quick command in `mobile-pwa/README-TESTING.md`.

## Timeline (suggested)
- Week 1: Epics 1–2 finalize and test.
- Week 2: Epic 3 finish; land schema parity local check.
- Week 3: Epic 4 chaos tests + PWA reconnection docs.

## Epic 5: Idle Timeout Hygiene (nice-to-have)

Objective: Cleanly disconnect idle WS connections to free resources.

Acceptance criteria:
- Connections idle beyond threshold (default 10 minutes) receive a `disconnect_notice` then are closed.
- Threshold is visible in code/config; future work can expose via `/websocket/limits` if needed.

Tasks:
- Manager periodically checks last activity and disconnects idle connections.
- Test `tests/unit/test_idle_disconnect.py` validates notice and disconnect.

## Notes
- Keep tests deterministic and avoid external dependencies (DB/Redis mocked where needed).
- Prefer narrow, vertical changes and keep CI lanes fast.
