K6?=k6
BACKEND_WS_URL?=ws://localhost:18080/api/dashboard/ws/dashboard
ACCESS_TOKEN?=

smoke:
	$(K6) run -e BACKEND_WS_URL=$(BACKEND_WS_URL) -e ACCESS_TOKEN=$(ACCESS_TOKEN) -e VUS=20 -e DURATION=1m ws_k6_scenarios.js

burst:
	$(K6) run -e BACKEND_WS_URL=$(BACKEND_WS_URL) -e ACCESS_TOKEN=$(ACCESS_TOKEN) -e VUS=200 -e DURATION=30s ws_k6_scenarios.js

soak:
	$(K6) run -e BACKEND_WS_URL=$(BACKEND_WS_URL) -e ACCESS_TOKEN=$(ACCESS_TOKEN) -e VUS=50 -e DURATION=15m ws_k6_scenarios.js
