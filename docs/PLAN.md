# Execution Plan – Epics 2 and 3 Completion + CI Stabilization

This plan follows our prioritization protocol (Pareto, must-haves only), TDD discipline, and vertical slices. We will land high-value fixes first to unblock CI and ship user-visible reliability.

## Objectives
- Stabilize CI for PR #3 (auth-foundation): Ruff, type/lint/test, and PWA lanes.
- Complete remaining must-haves for Epic 2 (Offline-first) and Epic 3 (WS SLOs).
- Maintain governance rails from Epic 4.

## Current Status
- Epic 1: Complete
- Epic 2: Partially complete (optimistic CRUD, queue, reconciliation UI, offline e2e)
- Epic 3: Partially complete (WS metrics, fanout gauge, rate limits env, backpressure notice, k6 scenarios)
- Epic 4: Complete (CI gate, TS drift enforcement, PWA version mismatch)
- CI: Failing on Ruff in one lane and backend-fast timeout; fixed now in branch by:
  - fast-type-test: switched to `ruff check .`
  - app startup: skip heavy init when CI/SKIP_STARTUP_INIT is set
  - backend-fast: exports SKIP_STARTUP_INIT
  - tests: stabilized ws correlation timestamp test

## Workstream A: CI Stabilization (Must-have)
- Verify all CI lanes pass on PR #3 after changes.
- If any Ruff errors persist, fix code/import order rather than broad ignores.
- Ensure `ci.yml` and `ci-fast.yml` do not require external services for “fast” lanes.
- Keep PWA type drift check intact.

## Workstream B: Epic 2 — Offline-first (Must-have scope)
- Durable retry/backoff for sync queue with jitter and max attempts per op.
- Conflict resolution policy: LWW + explicit conflict surfaces when server wins.
- UX: Banner/snackbar when queue > 0; action to “sync now”.
- Tests:
  - Unit: queue state machine (pending → retrying → reconciled/failed).
  - E2E (Playwright): toggle offline, create/update/delete tasks offline, enable online, verify reconciliation and conflict hints.

## Workstream C: Epic 3 — WS SLOs (Must-have scope)
- Enforce max outbound bytes per second per-connection; drop or backoff when exceeded; increment counters.
- Track p50/p95/p99 latency distribution for send path; expose via Prometheus.
- Add soft backpressure: send `disconnect_notice` when sustained overload continues; already partly in place.
- Tests:
  - Unit: rate limiter behavior and counter increments.
  - Integration: simulated high-frequency sends obey limiter and expose metrics.
  - k6: add bps ceilings and latency thresholds in scenarios.

## Definition of Done
- Green CI across all required lanes on PR #3.
- Offline queue resiliency (retry/jitter/max-attempts), conflict surfacing, and UX affordance with tests.
- WS SLOs rate limiting and latency metrics with tests and k6 updates.
- Commit messages reference epics and requirements concisely.

## Risks and Mitigations
- Redis/DB dependency in tests → continue to mock via fixtures; ensure fast lanes set SKIP_STARTUP_INIT.
- Flaky WS tests → add minimal waits and tolerant parsing where appropriate.
- PWA schema drift → rely on existing governance job to guard.

## Next Actions (ordered)
1) Push CI fixes and merge main; re-run PR checks.
2) Shore up any remaining Ruff/import issues flagged by CI.
3) Implement Offline queue retry/jitter/max-attempts + unit tests.
4) Add PWA UX banner for queued ops count and “sync now” (if not present) + e2e.
5) Implement WS send rate limiter + counters + unit/integration tests; export latency histograms.
6) Update k6 scenarios to assert bps and latency SLOs.
7) Final pass: docs snippets for env knobs added; ensure no new secrets required.
