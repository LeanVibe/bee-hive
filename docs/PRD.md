# HiveOps Product Requirements (Rolled-Up)

## Executive Summary

A mobile-first dashboard and API/WS platform for autonomous multi-agent development, focused on pragmatic value for entrepreneurial senior engineers.

## Personas & Use Cases (condensed)

- Solo technical founder: monitor/steer agents from phone
- Small teams: backlog visibility, health, events
- Use cases: live monitoring, task triage, alerts, quick approvals

## Success Metrics (initial)

- PWA installability, stable WS connection, dashboard loads <2s
- Smoke tests pass for nav + initial data render

## Scope (Now → Next → Later)

- Now: PWA dashboard (HiveOps), REST/WS endpoints, basic observability
- Next: UX polish, dark mode, Playwright coverage, screenshots
- Later: Orchestrator, GitHub integration, prompt optimization, self-mod engine

## Non-Goals

- Server-rendered dashboards
- Hard dependency on external LLM keys for local dev

## Backlog Highlights (from archived scratchpad)

- Autonomous APIs: project mgmt, code intelligence, deployment pipeline, learning/adaptation
- Security & API hardening: rate limiting, versioning, service-to-service auth
- Data layer: DB pooling optimization, query profiling, data lifecycle; caching strategy
- Messaging: DLQ with retry/backoff, consumer lag monitoring, reliability metrics
- Scalability: distributed orchestration, Redis Cluster, read replicas, cross-region
- Context engine: compression (60–80%), temporal windows, cross-agent knowledge, <50ms retrieval
- Observability/ops: tracing, intelligent alerting; DR/backup automation (enterprise optional)
- Integrations: VS Code/CI/Helm/Terraform as adoption accelerators (optional modules)

## References

- Detailed PRDs: `docs/core/*`
- Architecture: `docs/ARCHITECTURE.md`
- Core overview: `docs/CORE.md`
