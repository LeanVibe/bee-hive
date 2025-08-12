# HiveOps Core Overview

## Purpose

HiveOps is a mobile-first, API/WS-driven operations dashboard for autonomous multi-agent development. It targets entrepreneurial senior engineers who want real-time visibility and lightweight control without heavyweight setup.

## Value Proposition
- Autonomous agents reduce manual development time and run 24/7
- Real-time observability via WebSocket streams
- Mobile PWA for on-the-go monitoring and steering

## Core User Journey
1. Start backend (no external API keys required) and PWA
2. View live system health and agent activity on the dashboard
3. Inspect tasks/events; take corrective actions where needed

## Architecture at a Glance
- Backend: FastAPI with REST + WebSocket endpoints (no server-rendered dashboard)
- Frontend: Lit + Vite PWA (“HiveOps”)
- Infra: Postgres, Redis (docker-compose)

Key interfaces:
- WS: `/api/dashboard/ws/dashboard`
- REST (compat): `/dashboard/api/live-data`
- Health: `GET /health`

## Current Status (Aug 2025)

- Backend dashboard decommissioned; PWA-first confirmed
- Contract tests cover REST/WS core paths; WS schema enforced
- CI/Canary/Nightly guardrails in place to extend safe autonomous operation window

## Current Scope (Now → Next → Later)
- Now: API/WS endpoints, PWA dashboard, basic observability
- Next: Design tokens/dark mode, Playwright coverage, UX polish
- Later: Full agent orchestrator, GitHub integration, prompt optimization, self-mod engine

## Source of Truth
This file and `docs/ARCHITECTURE.md` + `docs/PRD.md` form the authoritative core. Subsystem PRDs in `docs/core/` are reference material. The generated navigation index lives at `docs/NAV_INDEX.md`.


