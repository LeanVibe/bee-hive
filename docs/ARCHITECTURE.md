# Architecture (Implementation-Aligned)

## Overview

Backend: FastAPI (Python), Postgres (+pgvector), Redis. Mobile PWA: Lit + Vite. UI consumes REST + WebSocket.

## Key Endpoints

- Health: `GET /health`
- Dashboard WebSocket: `GET /api/dashboard/ws/dashboard`
- Compat REST: `GET /dashboard/api/live-data`

## PWA Data Flow

- Backend adapter connects WebSocket and REST; mobile-first UI renders live metrics/events

## Local Startup

- Infra: `docker compose up -d postgres redis`
- Backend: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
- PWA: `cd mobile-pwa && npm ci && npm run dev`

## Optional Enterprise (reference only)

- Kubernetes: PSS, RBAC, NetworkPolicies; HPA/VPA with custom metrics
- Secrets: External Secrets; TLS via cert-manager
- Multi-cloud IaC: AWS CFN, GCP Cloud Run, Terraform modules
- Observability: OpenTelemetry tracing, Prometheus; ML-based anomaly alerting
- DR/Backup: Cross-region backup/restore scripts; failover procedure

These are optional blueprints for production; local dev remains simple (Docker + Uvicorn + Vite).

