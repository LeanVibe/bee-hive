# Getting Started (HiveOps / Bee Hive)

This guide gets you from zero to a running local stack fast. It uses only local services (no external LLM keys required) and follows the repo’s policies.

## Prerequisites
- Docker Desktop (or compatible) for Postgres and Redis
- Python 3.12
- Node.js 20.x and npm

## 1) Start infrastructure
```sh
docker compose up -d postgres redis
```

## 2) Backend API
```sh
python -m pip install --upgrade pip
pip install -e .[dev]

# Run FastAPI (hot-reload)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- Health: `GET http://localhost:8000/health`
- WebSocket (dashboard live data): `ws://localhost:8000/api/dashboard/ws/dashboard`

Notes
- CI/sandbox mode does not require external LLM keys. If you want to test LLM features later, prefer a local model (e.g., Ollama) and wire via the existing AI gateway.
- Do not reintroduce server-rendered dashboards; use API/WebSocket endpoints only.

## 3) Mobile PWA
```sh
cd mobile-pwa
npm ci
npm run dev
# Open the dev URL printed by Vite (e.g., http://localhost:5173)
```

## 4) Fast test lanes
Run focused lanes locally before committing. These complete quickly and cover core contracts and smoke checks.
```sh
# Python
pytest -q tests/smoke

# Frontend (from mobile-pwa)
npm test
```

## 5) Lint and types (recommended before PR)
```sh
# Lint (Ruff). Existing issues are being cleaned up incrementally.
ruff check .

# Types (focus on changed modules)
mypy app
```

## 6) Troubleshooting
- Port conflicts
  - Change the `--port` for uvicorn or stop other processes.
- Postgres/Redis not reachable
  - `docker compose ps` to confirm containers are healthy
  - Confirm env uses default URLs (e.g., `postgresql://...`, `redis://...`)
- WebSocket errors
  - Verify the backend is running and the WS URL is correct.

## 7) Project policies (must-read)
- No server-rendered dashboard; API and WebSocket endpoints only.
- Brand as “HiveOps” (default title in `mobile-pwa/src/components/layout/app-header.ts`).
- Keep the app bootable with `CI=true`. Avoid adding hard requirements for external API keys.

## 8) What to build next
- Focus on vertical slices with tests.
- Use `docs/TECH_DEBT_BACKLOG.md` for prioritized cleanups and `docs/CONSOLIDATION_REPORT.md` for documentation moves.
