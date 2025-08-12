# PWA Testing and Reconnection Notes

This PWA uses Vitest for unit tests and Playwright for E2E. WebSocket TS types are generated from the backend schema.

## Unit tests
- Run: `npm test`
- Coverage: focused on services and components.
- Add a test: place `*.spec.ts` under `src/**/__tests__/`.

## Schema parity
- Source of truth: `../schemas/ws_messages.schema.json`
- Generate types: `npm run generate:schemas`
- Check drift (local helper): `npm run check:schemas`

## WebSocket reconnection
Backend uses rate limiting and may drop connections under backpressure. The adapter reconnects with exponential backoff.
- Initial connect: `ws://localhost:8000/api/dashboard/ws/dashboard`
- PWA `BackendAdapter`:
  - Sends a `ping` on `open`
  - On `close`/`error`, schedules reconnect with backoff up to 5s*2^n (capped)
  - Falls back to polling when max attempts reached

Recommended manual test:
1) Start backend and PWA.
2) Load the app; confirm console shows WS connect and `pong`.
3) Temporarily stop backend; observe reconnect attempts with backoff.
4) Restart backend; verify reconnection and resumed updates.
