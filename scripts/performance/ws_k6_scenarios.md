# WS Load Scenarios (k6)

- smoke: 50 VUs, 1m, subscribe agents+system
- burst: 200 VUs, 30s, rapid message bursts
- soak: 50 VUs, 15m

Environment:
- BACKEND_WS_URL=ws://localhost:18080/api/dashboard/ws/dashboard
- AUTH_HEADER=Authorization: Bearer <token>
