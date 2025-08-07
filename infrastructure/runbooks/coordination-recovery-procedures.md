# Multi-Agent Coordination Recovery Procedures

## Overview

This runbook provides specific procedures for diagnosing and recovering from multi-agent coordination failures in the LeanVibe Agent Hive system. Focus on maintaining the coordination success rate above 95% as identified in the strategic plan.

## Critical Coordination Metrics

| Metric | Target | Warning Threshold | Critical Threshold |
|--------|--------|------------------|-------------------|
| Coordination Success Rate | >95% | <90% | <70% |
| Agent Response Time | <5s | >10s | >30s |
| Task Queue Length | <10 | >50 | >100 |
| Active Agents | >=3 | <3 | <1 |

## Diagnosis Procedures

### 1. Quick Coordination Health Check

```bash
#!/bin/bash
# Quick coordination health assessment

echo "=== Coordination Health Check ==="

# Check coordination success rate
COORD_SUCCESS=$(curl -s http://localhost:8000/api/dashboard/metrics/coordination | grep success_rate | head -1 | awk '{print $2}')
echo "Coordination Success Rate: ${COORD_SUCCESS}%"

# Check active agents
ACTIVE_AGENTS=$(curl -s http://localhost:8000/api/dashboard/metrics/agents | grep active | head -1 | awk '{print $2}')
echo "Active Agents: ${ACTIVE_AGENTS}"

# Check task queue
QUEUE_LENGTH=$(curl -s http://localhost:8000/api/dashboard/metrics/coordination | grep queue_length | awk '{print $2}')
echo "Task Queue Length: ${QUEUE_LENGTH}"

# Determine overall health
if (( $(echo "$COORD_SUCCESS < 70" | bc -l) )); then
    echo "STATUS: CRITICAL - Immediate action required"
elif (( $(echo "$COORD_SUCCESS < 90" | bc -l) )); then
    echo "STATUS: WARNING - Investigation needed"
else
    echo "STATUS: HEALTHY"
fi
```

### 2. Detailed Coordination Analysis

```bash
#!/bin/bash
# Detailed coordination system analysis

echo "=== Detailed Coordination Analysis ==="

# Agent status breakdown
echo "--- Agent Status ---"
curl -s http://localhost:8000/api/dashboard/metrics/agents | grep -E "(active|inactive|error)" | while read line; do
    echo "  $line"
done

# Task distribution analysis  
echo "--- Task Distribution ---"
curl -s http://localhost:8000/api/dashboard/metrics/coordination | grep -E "(pending|in_progress|completed|failed)" | while read line; do
    echo "  $line"
done

# Recent coordination errors
echo "--- Recent Errors ---"
docker-compose logs api --tail=50 | grep -i "coordination.*error\|agent.*error" | tail -10

# Redis streams health
echo "--- Redis Streams Status ---"
docker exec leanvibe_redis redis-cli XINFO GROUPS agent_messages:coordination 2>/dev/null || echo "No coordination stream found"
```

## Recovery Procedures

### Scenario 1: Coordination Success Rate Below 70% (Critical)

**Immediate Actions (0-5 minutes):**

1. **Emergency agent restart:**
   ```bash
   # Stop all agents gracefully
   curl -X POST http://localhost:8000/api/v1/agents/emergency-stop
   
   # Wait for graceful shutdown
   sleep 30
   
   # Restart agent orchestrator
   curl -X POST http://localhost:8000/api/v1/agents/restart-orchestrator
   ```

2. **Clear stuck message queues:**
   ```bash
   # List Redis streams
   docker exec leanvibe_redis redis-cli KEYS "agent_messages:*"
   
   # Clear corrupted streams (if identified)
   docker exec leanvibe_redis redis-cli DEL agent_messages:coordination
   ```

3. **Verify core services:**
   ```bash
   # Check database connectivity
   curl http://localhost:8000/health | jq '.components.database'
   
   # Check Redis connectivity  
   curl http://localhost:8000/health | jq '.components.redis'
   ```

**Recovery Validation (5-10 minutes):**

```bash
# Monitor coordination recovery
watch -n 10 'curl -s http://localhost:8000/api/dashboard/metrics/coordination | grep success_rate'

# Check for new agent registrations
curl http://localhost:8000/api/v1/agents/status | jq '.active_agents'
```

### Scenario 2: Agent Communication Failures

**Symptoms:**
- Agents not responding to coordination requests
- High agent timeout rates
- Message queue backlog

**Diagnosis:**
```bash
# Check agent heartbeats
curl http://localhost:8000/api/dashboard/metrics/agents | grep stale_heartbeats

# Check Redis connection pool
docker exec leanvibe_redis redis-cli INFO clients

# Review agent communication logs
docker-compose logs api --tail=100 | grep -i "agent.*timeout\|agent.*communication"
```

**Recovery Steps:**

1. **Reset Redis connection pools:**
   ```bash
   # Force Redis connection pool refresh
   curl -X POST http://localhost:8000/api/v1/redis/refresh-pools
   
   # Restart Redis if needed
   docker-compose restart redis
   sleep 10
   ```

2. **Restart agents in sequence:**
   ```bash
   # Get list of problematic agents
   PROBLEM_AGENTS=$(curl -s http://localhost:8000/api/v1/agents/status | jq -r '.agents[] | select(.status=="error" or .last_heartbeat < now - 300) | .id')
   
   # Restart each agent
   for agent in $PROBLEM_AGENTS; do
       echo "Restarting agent: $agent"
       curl -X POST "http://localhost:8000/api/v1/agents/$agent/restart"
       sleep 5
   done
   ```

3. **Validate recovery:**
   ```bash
   # Check agent heartbeats are current
   sleep 60
   curl http://localhost:8000/api/dashboard/metrics/agents | grep stale_heartbeats
   ```

### Scenario 3: Task Distribution Failures

**Symptoms:**
- Tasks stuck in PENDING state
- Uneven task distribution
- High task failure rate

**Diagnosis:**
```bash
# Check task queue status
curl http://localhost:8000/api/dashboard/metrics/coordination | grep -E "(queue_length|long_running)"

# Check task assignment patterns
curl http://localhost:8000/api/v1/tasks/distribution-analysis

# Review task execution logs
docker-compose logs api --tail=100 | grep -i "task.*assign\|task.*fail"
```

**Recovery Steps:**

1. **Clear stuck tasks:**
   ```bash
   # Identify long-running tasks
   STUCK_TASKS=$(curl -s http://localhost:8000/api/v1/tasks/stuck | jq -r '.[]')
   
   # Cancel stuck tasks
   for task in $STUCK_TASKS; do
       curl -X POST "http://localhost:8000/api/v1/tasks/$task/cancel"
   done
   ```

2. **Rebalance task distribution:**
   ```bash
   # Trigger task rebalancing
   curl -X POST http://localhost:8000/api/v1/coordination/rebalance
   
   # Verify redistribution
   sleep 30
   curl http://localhost:8000/api/dashboard/metrics/coordination | grep queue_length
   ```

## Preventive Maintenance

### Daily Coordination Health Checks

Create a daily health check script:

```bash
#!/bin/bash
# Daily coordination health check
# Save as: infrastructure/scripts/daily-coord-check.sh

DATE=$(date '+%Y-%m-%d %H:%M:%S')
LOG_FILE="/var/log/coordination-health.log"

echo "[$DATE] Starting daily coordination health check" >> $LOG_FILE

# Check coordination success rate
SUCCESS_RATE=$(curl -s http://localhost:8000/api/dashboard/metrics/coordination | grep success_rate | awk '{print $2}')

if (( $(echo "$SUCCESS_RATE < 95" | bc -l) )); then
    echo "[$DATE] WARNING: Coordination success rate is ${SUCCESS_RATE}%" >> $LOG_FILE
    # Send alert
    curl -X POST "http://localhost:9093/api/v1/alerts" \
         -H "Content-Type: application/json" \
         -d '[{"labels":{"alertname":"DailyCoordinationCheck","severity":"warning","message":"Coordination success rate below 95%"}}]'
else
    echo "[$DATE] INFO: Coordination success rate is healthy at ${SUCCESS_RATE}%" >> $LOG_FILE
fi

# Check for agent issues
STALE_AGENTS=$(curl -s http://localhost:8000/api/dashboard/metrics/agents | grep stale_heartbeats | awk '{print $2}')
if [ "$STALE_AGENTS" -gt "0" ]; then
    echo "[$DATE] WARNING: $STALE_AGENTS agents with stale heartbeats" >> $LOG_FILE
fi

echo "[$DATE] Daily coordination health check completed" >> $LOG_FILE
```

### Weekly Coordination Optimization

```bash
#!/bin/bash
# Weekly coordination system optimization
# Save as: infrastructure/scripts/weekly-coord-optimization.sh

echo "=== Weekly Coordination Optimization ==="

# Analyze coordination patterns
echo "--- Coordination Pattern Analysis ---"
curl http://localhost:8000/api/v1/coordination/weekly-analysis

# Clean up completed tasks (older than 7 days)
echo "--- Task Cleanup ---"
curl -X DELETE http://localhost:8000/api/v1/tasks/cleanup?days=7

# Optimize Redis memory usage
echo "--- Redis Optimization ---"
docker exec leanvibe_redis redis-cli MEMORY PURGE

# Update agent performance metrics
echo "--- Agent Performance Update ---"
curl -X POST http://localhost:8000/api/v1/agents/update-performance-metrics

echo "--- Optimization Complete ---"
```

## Advanced Troubleshooting

### Deep Dive Coordination Analysis

```bash
#!/bin/bash
# Advanced coordination diagnostics

echo "=== Advanced Coordination Diagnostics ==="

# Agent communication latency analysis
echo "--- Agent Latency Analysis ---"
curl http://localhost:8000/api/v1/agents/latency-analysis | jq '.'

# Task execution performance breakdown
echo "--- Task Performance Analysis ---"
curl http://localhost:8000/api/v1/tasks/performance-analysis | jq '.'

# Redis streams detailed analysis
echo "--- Redis Streams Analysis ---"
docker exec leanvibe_redis redis-cli XINFO STREAMS agent_messages:* | head -20

# Database query performance analysis
echo "--- Database Performance Analysis ---"
docker exec leanvibe_postgres psql -U leanvibe_user -d leanvibe_agent_hive -c "
SELECT query, calls, mean_exec_time, rows, 100.0 * shared_blks_hit /
       nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 5;
"
```

### Performance Tuning Recommendations

Based on coordination analysis results:

1. **If success rate 90-95%:** Minor optimization needed
   ```bash
   # Adjust agent heartbeat interval
   curl -X PUT http://localhost:8000/api/v1/config/heartbeat-interval -d '{"interval": 30}'
   ```

2. **If success rate 70-90%:** Moderate intervention required
   ```bash
   # Increase coordination timeout
   curl -X PUT http://localhost:8000/api/v1/config/coordination-timeout -d '{"timeout": 60}'
   
   # Add more agent workers
   curl -X POST http://localhost:8000/api/v1/agents/scale-up -d '{"count": 2}'
   ```

3. **If success rate <70%:** System architecture review needed
   - Review database indexing
   - Consider Redis cluster setup  
   - Evaluate agent communication patterns
   - Check for resource constraints

## Emergency Contacts

### Coordination Failure Escalation

- **L1 Response:** DevOps on-call (PagerDuty)
- **L2 Response:** Engineering Team Lead  
- **L3 Response:** System Architect
- **Emergency Contact:** CTO (for critical business impact)

### Communication Channels

- **Incident Response:** #incident-response (Slack)
- **Coordination Alerts:** #coordination-monitoring (Slack)
- **Status Updates:** #system-status (Slack)

---

**Last Updated:** 2025-08-07
**Version:** 1.2
**Owner:** DevOps Team
**Reviewed By:** Engineering Lead