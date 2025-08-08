# Tech Context

## Stack

- Backend: FastAPI, Structlog, SQLAlchemy; Uvicorn for dev.
- Frontend: Lit, Vite, Tailwind; Playwright for E2E.
- Infra: Postgres, Redis via docker-compose.

## Startup (local)

```sh
docker compose up -d postgres redis
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
cd mobile-pwa && npm ci && npm run dev
```

## Environment assumptions

- No external API keys required; sandbox mode.
- Local LLM optional via Ollama if needed.
