# System Patterns

## Architecture

- Backend: FastAPI app with API/WebSocket endpoints; no server-rendered dashboard.
- Frontend: Lit + Vite PWA (“HiveOps”) consuming REST + WS.
- Compatibility layer: `/dashboard/api/live-data` maintained for PWA expectations while core data lives under `/api/dashboard/*`.

## Integration Boundaries

- WebSocket endpoint: `/api/dashboard/ws/dashboard`.
- REST live data: `/dashboard/api/live-data` (compat); prefer `/api/dashboard/*` for new features.
- Centralized backend URLs in `backend-adapter` to avoid duplication across components.

## Environment

- Local dev requires no external API keys (sandbox mode). If needed, use a local LLM via Ollama.

## Testing

- Playwright E2E smoke ensures navigation and initial data render on `/dashboard`.
- Screenshots saved under `mobile-pwa/test-results/`.
