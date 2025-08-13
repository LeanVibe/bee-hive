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

4) Security hardening (high value, feature-flagged)
- Add WS auth/allowlist feature flag. Default off for dev; on in prod via env.
- CLI `doctor` should surface misconfiguration.
- Tests: only when flag enabled; otherwise skip.

Implementation notes:
- Env flags: `WS_AUTH_REQUIRED` (true/false), `WS_ALLOWED_ORIGINS` (comma list)
- Behavior: if allowlist set and `Origin` not in list, connection is closed with 4403
 - Env `WS_AUTH_TOKEN` used when `WS_AUTH_REQUIRED=true`; connections must include `Authorization: Bearer <token>` or receive 4401

## Next 4 Epics — Auth, Offline, SLOs, Governance (Sep 2025)

### Epic 1: End-to-end AuthN/AuthZ and RBAC

Objective: Introduce session-backed authentication and role-based authorization across REST and WebSocket layers with developer-friendly defaults.

Acceptance criteria:
- REST: `/api/auth/login`, `/api/auth/refresh`, `/api/auth/logout`, `/api/auth/me` implemented; returns access + refresh tokens; short-lived access tokens.
- RBAC utilities protect sensitive REST routes and WS subscriptions; denials are observable in metrics and logs.
- WS accepts Authorization header or session cookie and denies with 4401 on invalid/expired token; counters increment.
- PWA implements minimal login flow, token storage, refresh, and injects Authorization into REST and WS.

Tasks:
- Backend
  - Implement auth routes with JWT + refresh; add pydantic models and tests.
  - RBAC decorators/utilities and sample-protected endpoints.
  - WS: validate Authorization bearer on connect and re-auth on refresh.
  - Metrics: `auth_success_total`, `auth_denied_total` (exists for WS), route-level logs.
- PWA
  - `AuthService` login/logout + refresh (present) wired to backend routes; guard routes; show 401 banner.
  - Inject Authorization into WS connect and REST calls.
- Tests: unit + ws accept/deny + smoke login; docs: security guide and `.env.example`.

Status (2025-08-13):
- PR branch `auth-foundation` contains smoke tests for login/refresh/protected routes.
- CI previously failed due to Ruff and an `UnboundLocalError` in `dashboard_websockets.py`; those are now fixed and pushed.
- Remaining blocker: smoke auth tests return 404 for `/api/v1/auth/login` and `/api/v1/auth/refresh` in local/CI backend-fast lane. Coverage gate also fails (<45%).

Immediate resolution plan:
1) Ensure backend auth router is implemented and mounted under `/api/v1/auth`.
   - Add `app/api/auth.py` with routes: `POST /login`, `POST /refresh`, `POST /logout`, `GET /me`.
   - Wire into `app/main.py` with `app.include_router(auth_router, prefix="/api/v1/auth")`.
   - Keep implementation minimal and deterministic for tests (no external IdP; local JWT using `PyJWT`).
2) Align tests and contracts:
   - Keep the smoke tests' expected paths as the source of truth.
   - Return JSON shape used by tests: `{access_token, refresh_token}` for login; `{success: true, access_token}` for refresh.
3) RBAC sample and WS auth are present; ensure they remain compatible with new tokens.
4) Add focused unit tests for new auth utilities to lift coverage above 45%.

Acceptance to close Epic 1 on this branch:
- Smoke auth tests pass; coverage gate >= 45%.
- Routes exist and are mounted; minimal RBAC + WS auth continue to work.

### Epic 2: Offline-first sync and conflict resolution

Objective: Deterministic offline caching and queued updates with reconciliation on reconnect.

Acceptance criteria:
- IndexedDB caches tasks/agents/metrics with versioned schema; queue supports idempotent envelopes with `correlation_id`.
- Offline UI remains usable; upon reconnection, queued updates sync and UI reconciles; conflict policy documented.

Tasks:
- Storage schema + per-domain policies; background sync hook.
- Queue envelope + retry/backoff; reconciliation handler.
- PWA views: optimistic updates and pending/synced badges.
- Tests: unit (cache/queue), Playwright offline scenarios; docs: offline guide.

Status (2025-08-13):
- Initial offline scaffolding, optimistic CRUD, queue, reconciliation UI badges, and a basic Playwright smoke test have landed.

Remaining work (high value first):
1) Deterministic reconciliation policy
   - Define conflict resolution for task updates (last-write-wins with correlation-id tie-breaker).
   - Persist a reconciliation log in IndexedDB for user visibility (last N entries).
   - Tests: unit for reconciliation; E2E update conflict scenario.
2) Background sync/backoff
   - Exponential backoff for retrying queued envelopes (e.g., 1s → 2s → 4s, cap 30s).
   - Surface retry state in UI badges.
   - Tests: unit for backoff schedule.
3) Robust offline storage schema
   - Versioned IndexedDB schema with migrations.
   - Tests: migration path (bump version; ensure upgrade logic preserves data).

### Epic 3: WS scalability, performance SLOs, and dashboards

Objective: Define and monitor SLOs, profile fanout, and provide tuning guidance.

Acceptance criteria:
- SLOs for p95 WS send latency and drop rates defined and visualized; load scenarios reproducible.

Tasks:
- Optional message compression toggle; export message sizes.
- Export fanout and queue gauges; backpressure reason codes.
- k6/locust scripts + Make targets; Grafana dashboards; alerting on error budgets.

Status (2025-08-13):
- WS bytes counters, fanout gauge, rate limit env flags, backpressure notice, compression flag wiring, and k6 scenarios have been added incrementally.

Remaining work (to complete Epic 3):
1) Prometheus exposition for WS metrics
   - Expose `messages_*`, `errors_sent_total`, `bytes_*`, `connections/disconnections_total`, `backpressure_disconnects_total`, and current fanout as gauges/counters.
   - Endpoint: `/api/dashboard/metrics/websockets` (ensure it is registered in `app/main.py`).
   - Tests: scrape endpoint and assert metric names exist (no strict values in PR lanes).
2) Rate-limiting tests
   - Unit test token-bucket behavior in `DashboardWebSocketManager`.
   - WS integration test: burst > limit yields at least one rate-limit error and drops without flaking.
3) k6 runbook and Make targets
   - `make ws-load smoke=1` for PRs (short duration), `make ws-load soak=1` for manual.
   - Document thresholds and how to interpret.
4) Grafana dashboard JSON and docs
   - Provide starter dashboard panels for p95 send latency (approximate), error rates, and fanout.

### Epic 4: Contract governance and CI safety rails

Objective: Prevent unintentional breaking changes to WS contract and enforce migration hygiene.

Acceptance criteria:
- CI classifies schema diffs (patch/minor/major) and requires label/migration notes for major.
- PWA warns when `current_version` not in `supported_versions` and sends telemetry.

Tasks:
- Add schema diff job and PR checks; maintain `/api/dashboard/websocket/contract` (done) with policy.
- PWA version banner and optional “learn more” link to migration doc.

Status (2025-08-13): Complete
- CI schema gate: `.github/workflows/schema-governance.yml` fails PRs on version changes without labels/notes.
- Drift enforcement for TS types present in CI.
- PWA version mismatch banner implemented in `mobile-pwa/src/services/websocket.ts` and surfaced in `mobile-pwa/src/app.ts`.

Follow-ups (nice-to-have):
- Add GitHub PR template section for schema change notes and migration steps.
- Telemetry event on client-side version mismatch (count-only) to track adoption.

---

CI and Developer Experience Stabilization (cross-cutting)
1) Keep PR lanes fast and deterministic
   - PR: `ruff`, `pytest -q tests/unit tests/ws tests/smoke`, schema/TS drift, PWA typegen check.
   - Fast Lanes: changed-files Ruff + minimal backend tests.
2) Coverage gate
   - Current gate is 45% on backend; we are below in CI for this branch.
   - Action: add unit tests for auth utilities and WS helpers to raise baseline.
3) Local dev guidance
   - Document local `.env` entries and dev ports; ensure `uvicorn` and `npm run dev` steps in root README are up to date.
4) Pre-push hooks
   - Hooks flagged sensitive files in local env; recommend using `--no-verify` for local-only pushes or adjust hooks to respect `.gitignore`.

Immediate Next Actions (for the next agent)
1) Backend auth router: implement + mount to fix smoke auth 404 and pass coverage.
2) Add unit tests for auth and ws_utils to reach coverage >=45%.
3) Wire Prometheus WS metrics endpoint and basic scrape test.
4) Close out Epic 3 remaining items (rate-limit tests, docs, Makefile entries).
5) Iterate on Epic 2 remaining reconciliation/backoff tasks.
