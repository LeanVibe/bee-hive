# Architecture (Implementation-Aligned)

## Overview

Backend: FastAPI (Python), Postgres (+pgvector), Redis. Mobile PWA: Lit + Vite. UI consumes REST + WebSocket.

## Key Endpoints

- Health: `GET /health`
- Dashboard WebSocket: `GET /api/dashboard/ws/dashboard`
- Compat REST: `GET /dashboard/api/live-data`

## PWA Data Flow

- Backend adapter connects WebSocket and REST; mobile-first UI renders live metrics/events

Validation

- PWA unit tests validate `BackendAdapter` fetch/cache/fallback and `BaseService` helpers
- WS message schema enforces shape for initial/updates; contract tests validate runtime messages

## Local Startup

- Infra: `docker compose up -d postgres redis`
- Backend: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
- PWA: `cd mobile-pwa && npm ci && npm run dev`

CI/CD Guardrails

- PR: focused backend tests + PWA vitest + schema→types check; coverage gate 40%
- Nightly: focused tests + Playwright smoke + mutation tests (limited scope)
- Canary: synthetic probes for /health, /metrics, live-data, and WS handshake

## Optional Enterprise (reference only)

- Kubernetes: PSS, RBAC, NetworkPolicies; HPA/VPA with custom metrics
- Secrets: External Secrets; TLS via cert-manager
- Multi-cloud IaC: AWS CFN, GCP Cloud Run, Terraform modules
- Observability: OpenTelemetry tracing, Prometheus; ML-based anomaly alerting
- DR/Backup: Cross-region backup/restore scripts; failover procedure

These are optional blueprints for production; local dev remains simple (Docker + Uvicorn + Vite).

