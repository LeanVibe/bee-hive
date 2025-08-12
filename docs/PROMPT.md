You are a senior Cursor agent stepping into the HiveOps repo to continue backend and PWA platform work. Read this fully, then execute without asking for confirmations unless blocked.

Context:
- Backend: FastAPI (`app/main.py`), WS endpoints at `/api/dashboard/ws/*`. Observability + limits + contract endpoints in place.
- PWA: Lit + Vite app in `mobile-pwa/`; unit tests via Vitest; E2E via Playwright (smokes exist). Types generated from `schemas/ws_messages.schema.json`.
- Tests are green (backend unit/ws/smoke; PWA unit). CI generates PWA types and gates on drift.

Immediate objectives (Next 4 Epics — Auth, Offline, SLOs, Governance):

1) End-to-end AuthN/AuthZ and RBAC
- Backend
  - Implement `/api/auth/login`, `/api/auth/refresh`, `/api/auth/logout`, `/api/auth/me` with JWT + refresh (short-lived access).
  - Create RBAC decorators/utilities and protect at least one sample REST route and WS subscription path.
  - WS: validate Authorization bearer on connect; denial returns 4401 and increments counters (counters already exist for WS).
  - Add metrics for `auth_success_total` (REST/WS) and route-level structured logs.
- PWA
  - Wire existing `AuthService` to backend routes (currently mocks Auth0/dev); implement minimal login view + refresh to use backend.
  - Inject Authorization into REST + WS; handle 401/4401 with banner and re-auth.
- Tests
  - Backend unit + ws accept/deny flow; smoke test for login and protected route access.
  - Docs: brief security guide + `.env.example` entries.

2) Offline-first sync and conflict resolution
- Storage: define IndexedDB schema (tasks/agents/metrics) and per-domain caching policies; background sync hook.
- Queue: idempotent envelope with `correlation_id`, retry/backoff; reconciliation logic on reconnect.
- PWA views: optimistic updates + pending/synced badges.
- Tests: unit for cache/queue; expand existing Playwright offline scenarios to assert data integrity and queued update processing.
- Docs: offline capabilities and limits.

3) WS scalability, performance SLOs, and dashboards
- Backend: optional message compression toggle; export message sizes and fanout/queue gauges; include backpressure reason codes in disconnects.
- Tooling: k6/locust scenarios for WS load; Make targets; Grafana dashboards for WS latency/counters/queues; alerts for error budget burn.
- Tests: keep non-flaky perf checks tagged/skipped in PR lanes; add smoke for metric presence.

4) Contract governance and CI safety rails
- CI job to diff `schemas/ws_messages.schema.json`, classify (patch/minor/major), require label + migration notes for major.
- Maintain `/api/dashboard/websocket/contract` (exists) as source of truth; document deprecation policy.
- PWA: add version mismatch banner on connect when `current_version` ∉ `supported_versions`.

Key references
- `docs/PLAN.md` updated with detailed acceptance criteria and tasks.
- WS manager: `app/api/dashboard_websockets.py` (auth/allowlist flags, metrics, idle disconnect, limits/contract endpoints).
- Metrics: `app/api/dashboard_prometheus.py` (WS counters exported).
- PWA Auth: `mobile-pwa/src/services/auth.ts` (scaffold exists), WS client `mobile-pwa/src/services/websocket.ts`.

Execution guidelines
- Maintain green tests after each logical change.
- Prefer vertical slices: implement minimal end-to-end path, add tests, then iterate.
- Keep PR-size reasonable; ship backend auth + PWA minimal login as first branch.
- Update docs: security guide, `.env.example`, and any new env flags.

Initial work plan (branch: auth-foundation)
- Backend
  1) Add `app/api/auth.py` router with login/refresh/logout/me; JWT utilities in `app/core/auth.py`; tests in `tests/unit/test_auth_*.py` and smoke for login.
  2) RBAC helper in `app/core/rbac.py`; decorate one sample route + a WS subscription guard; tests.
  3) Wire metrics/logs for auth successes/denials.
- PWA
  4) Add minimal `login-view` and wire `AuthService.login` to backend endpoints; persist tokens; refresh.
  5) Inject `Authorization` into WS connection (already sends auth message — switch to header on connect) and REST.
  6) Add a banner on 401/4401 to prompt re-auth; unit tests.
- CI/Docs
  7) Docs for security setup and `.env.example`; optional CI job later for schema diff in governance epic.

When blocked
- If backend login API shape is unclear, define minimal JSON contracts and proceed (update PLAN + tests). Avoid third-party IdP requirements for dev.

Deliverables for first PR
- Working login + token refresh + protected example route + WS connect with auth.
- PWA minimal login UI; authenticated fetches; WS Authorization header set.
- Unit + smoke tests; docs updated.
