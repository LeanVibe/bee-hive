# HiveOps Mobile Dashboard Rebrand and Backend Dashboard Decommission Plan

## Objective

- Replace backend-served HTML dashboard with API/WebSocket-only endpoints.
- Rebrand and refine the mobile PWA dashboard to a polished, Silicon Valley startup aesthetic targeting entrepreneurial senior engineers.

## Current State (from docs and code)

- Vision (CLAUDE.md, docs/INDEX.md, README.md): Production-ready autonomous development platform with multi-agent orchestration, real-time monitoring, and strong mobile PWA.
- Backend exposes both API-based dashboard streams (`/api/dashboard/*`) and a legacy HTML dashboard via `app/dashboard/*` included under `/dashboard`.
- Mobile PWA (`mobile-pwa/`) is comprehensive (Lit components), already wired to `/api/dashboard/ws/*` and `/api/dashboard/*` endpoints. Branding is functional but generic.

## Gaps vs Vision

- Duplicate dashboard delivery (server-rendered HTML + PWA) adds complexity and splits focus. Vision emphasizes modern, lightweight, mobile-first experiences.
- Branding/tone could better match “SV startup for senior builders”: sharper identity, concise language, clean typography, minimal emoji.

## Decisions

- Decommission FastAPI template dashboard (HTML routes and templates under `/dashboard`). Keep API and WebSocket endpoints under `/api/dashboard/*` intact.
- Rebrand the mobile PWA:
  - Product name in UI: “HiveOps”.
  - Tone: concise, professional, experimental-friendly.
  - Palette: refined blues/slate (no heavy gradients by default), with optional glass for depth.
  - Titles and meta updated; consistent manifest and header iconography.

## Implementation Steps

1) Backend cleanup
   - Remove imports and `include_router` for `app/dashboard/coordination_dashboard.py` and `app/dashboard/simple_agent_dashboard.py` from `app/main.py`.
   - Keep all `/api/dashboard/*` routers (monitoring, websockets, prometheus, task-management) as-is.
   - Add `app/api/dashboard_compat.py` with `GET /dashboard/api/live-data` and include it in `app/main.py`.

2) Mobile PWA rebrand
   - Update `mobile-pwa/public/manifest.json`: name, short_name, description, theme/background.
   - Update `mobile-pwa/index.html`: `<title>` and meta description to reflect HiveOps brand; loader gradient refined.
   - Update `mobile-pwa/src/components/layout/app-header.ts`: logo mark (simple brand glyph), default title fallback “HiveOps”.
   - Update `mobile-pwa/src/views/dashboard-view.ts`: header title to “HiveOps”.

3) Visual polish (incremental)
   - Keep existing component structure; rely on current Tailwind config and component CSS. Future iterations can introduce CSS variables for theming.

4) QA & Validation

   - Ensure backend builds and runs without the `/dashboard` routers.
   - Verify PWA loads, connects to `/api/dashboard/ws/dashboard`, and renders updated branding.
   - Update Playwright tests for brand/title changes; run smoke across key routes and validate no endpoint regressions.

## Rollback

- Re-enable backend dashboard by restoring router includes in `app/main.py` (if needed).

## Next Iterations (post-merge)

- Typography and design token pass across Lit components.
- Dark mode polish and brand illustrations.
- Marketing-ready screenshots and docs-site alignment.
