# Bee Hive - Monitoring Setup Guide

**Generated:** 2026-01-20T02:52:53.016824
**Project:** Bee Hive
**Location:** `leanvibe-dev/bee-hive/`

---

## Overview

This guide helps you set up monitoring for Bee Hive's real-time dashboard and WebSocket functionality.

---

## Dashboard Configuration

### Access URLs
- **Dashboard:** http://localhost:18443 (PWA)
- **API:** http://localhost:18080
- **API Docs:** http://localhost:18080/docs
- **WebSocket:** ws://localhost:18080/api/dashboard/ws/dashboard

### Port Configuration
Bee Hive uses non-standard ports to avoid conflicts:
- **API:** 18080 (instead of 8000)
- **PWA:** 18443 (instead of 3000/5173)
- **PostgreSQL:** 15432 (instead of 5432)
- **Redis:** 16379 (instead of 6379)

---

## WebSocket Setup

### Connection
```javascript
const ws = new WebSocket('ws://localhost:18080/api/dashboard/ws/dashboard');
```

### Message Format
- All messages include `correlation_id` for tracing
- Error frames include `timestamp` and `correlation_id`
- Data responses include `type`, `data_type`, `data`

### Rate Limiting
- **Rate:** 20 requests per second per connection
- **Burst:** 40 requests
- **Message Size:** 64KB cap

### Subscription Limits
- Max subscriptions enforced per connection
- Monitor subscription count

---

## Monitoring Endpoints

### Health Checks
```bash
# API Health
curl http://localhost:18080/health

# WebSocket Metrics
curl http://localhost:18080/api/dashboard/metrics/websockets
```

### Prometheus Metrics
- **Endpoint:** `/api/dashboard/metrics/websockets`
- **Format:** Prometheus text format
- **Metrics:** Connection count, message rates, errors

---

## Alerting Configuration

### Key Metrics to Monitor
- **WebSocket Connections:** Active connection count
- **Message Rate:** Messages per second
- **Error Rate:** Error frames per second
- **Latency:** Message round-trip time
- **Subscription Count:** Active subscriptions

### Alert Thresholds
- **Connection Count:** Alert if >100 concurrent connections
- **Error Rate:** Alert if >5% error rate
- **Latency:** Alert if P95 >100ms
- **Message Size:** Alert if approaching 64KB limit

---

## CLI Commands

### System Diagnostics
```bash
hive doctor  # Comprehensive diagnostics
```

### System Management
```bash
hive start           # Start the platform
hive status          # System status
hive status --watch   # Monitor system (watch mode)
hive dashboard       # Open dashboard
```

### Agent Management
```bash
hive agent deploy backend-developer  # Deploy agents
hive agent list                      # List agents
```

---

## Troubleshooting

### Dashboard Not Loading
1. Check PWA is running: `npm run dev` in `mobile-pwa/`
2. Verify port 18443 is available
3. Check browser console for errors

### WebSocket Connection Failed
1. Verify API is running on port 18080
2. Check WebSocket endpoint: `/api/dashboard/ws/dashboard`
3. Verify rate limiting not exceeded
4. Check message size <64KB

### CLI Commands Not Working
1. Verify `hive` command is in PATH
2. Check `bin/agent-hive` is executable
3. Run `hive doctor` for diagnostics

---

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Dashboard Load** | <2s | Time to interactive |
| **WebSocket Latency** | <100ms | Message round-trip time |
| **API Response** | <500ms | P95 response time |
| **Connection Stability** | >99% | Uptime percentage |

---

**Last Updated:** 2026-01-20T02:52:53.016829
