# LeanVibe Agent Hive Troubleshooting Guide

## Table of Contents

1. [Quick Diagnosis](#quick-diagnosis)
2. [Application Issues](#application-issues)
3. [Database Problems](#database-problems)
4. [Redis and Caching Issues](#redis-and-caching-issues)
5. [Agent Management Problems](#agent-management-problems)
6. [Workflow Execution Issues](#workflow-execution-issues)
7. [Performance Problems](#performance-problems)
8. [Authentication and Authorization](#authentication-and-authorization)
9. [WebSocket and Real-time Issues](#websocket-and-real-time-issues)
10. [Mobile PWA Problems](#mobile-pwa-problems)
11. [Deployment and Infrastructure](#deployment-and-infrastructure)
12. [Monitoring and Alerting](#monitoring-and-alerting)
13. [Log Analysis](#log-analysis)
14. [Emergency Procedures](#emergency-procedures)

## Quick Diagnosis

### System Health Check Script

```bash
#!/bin/bash
# health-check.sh - Quick system diagnostic

echo "=== LeanVibe Agent Hive Health Check ==="
echo "Timestamp: $(date)"
echo

# Check service status
echo "--- Service Status ---"
systemctl is-active docker && echo "✓ Docker: Running" || echo "✗ Docker: Not running"
docker-compose -f docker-compose.prod.yml ps | grep -q "Up" && echo "✓ Containers: Running" || echo "✗ Containers: Issues detected"

# Check application health
echo "--- Application Health ---"
APP_HEALTH=$(curl -s -w "%{http_code}" http://localhost:8000/health -o /tmp/health.json)
if [ "$APP_HEALTH" = "200" ]; then
    echo "✓ Application: Healthy"
    echo "  Response time: $(curl -s -w "%{time_total}s\n" http://localhost:8000/health -o /dev/null)"
else
    echo "✗ Application: Unhealthy (HTTP $APP_HEALTH)"
fi

# Check database connectivity
echo "--- Database Status ---"
if command -v psql >/dev/null 2>&1; then
    if PGPASSWORD="$DB_PASSWORD" psql -h localhost -U leanvibe_admin -d leanvibe_production -c "SELECT 1;" >/dev/null 2>&1; then
        echo "✓ Database: Connected"
    else
        echo "✗ Database: Connection failed"
    fi
else
    echo "? Database: psql not available for testing"
fi

# Check Redis connectivity
echo "--- Redis Status ---"
if command -v redis-cli >/dev/null 2>&1; then
    if redis-cli -a "$REDIS_PASSWORD" ping >/dev/null 2>&1; then
        echo "✓ Redis: Connected"
    else
        echo "✗ Redis: Connection failed"
    fi
else
    echo "? Redis: redis-cli not available for testing"
fi

# Check system resources
echo "--- System Resources ---"
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
MEM_USAGE=$(free | grep Mem | awk '{printf("%.0f", $3/$2 * 100.0)}')
LOAD_AVG=$(uptime | awk -F'load average:' '{ print $2 }' | cut -d, -f1 | xargs)

echo "Disk usage: $DISK_USAGE%"
echo "Memory usage: $MEM_USAGE%"
echo "Load average: $LOAD_AVG"

# Check recent errors
echo "--- Recent Errors ---"
ERROR_COUNT=$(journalctl -u docker --since="5 minutes ago" --grep="error\|failed\|exception" | wc -l)
echo "System errors (last 5 min): $ERROR_COUNT"

if [ -f "/opt/leanvibe/logs/app.log" ]; then
    APP_ERRORS=$(tail -100 /opt/leanvibe/logs/app.log | grep -i "error\|exception\|failed" | wc -l)
    echo "Application errors (last 100 lines): $APP_ERRORS"
fi

echo
echo "=== Health Check Complete ==="
```

### Common Issue Quick Reference

| Symptom | Likely Cause | Quick Fix |
|---------|--------------|-----------|
| 500 Internal Server Error | Database connection, application crash | Check logs, restart app |
| 502 Bad Gateway | Application not responding | Restart application container |
| Slow response times | Database queries, memory issues | Check database performance, restart |
| Agent not responding | Agent process crashed, resource exhaustion | Check agent logs, restart agent |
| Workflow stuck | Task dependencies, agent unavailability | Check workflow status, reassign tasks |
| WebSocket disconnections | Network issues, load balancer timeout | Check nginx config, network stability |
| High CPU usage | Inefficient queries, infinite loops | Check slow queries, process monitoring |
| High memory usage | Memory leaks, large datasets | Restart services, optimize queries |

## Application Issues

### Application Won't Start

#### Symptoms
- Container fails to start
- Application exits immediately
- Health check endpoints not responding

#### Diagnosis Steps

```bash
# Check container logs
docker logs leanvibe-app --tail=50

# Check if port is already in use
netstat -tlnp | grep :8000

# Verify environment variables
docker exec leanvibe-app env | grep -E "DATABASE_URL|REDIS_URL|JWT_SECRET"

# Check file permissions
docker exec leanvibe-app ls -la /app

# Test database connection from container
docker exec leanvibe-app python -c "
import asyncio
import asyncpg
import os

async def test():
    try:
        conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
        print('DB connection: OK')
        await conn.close()
    except Exception as e:
        print(f'DB connection failed: {e}')

asyncio.run(test())
"
```

#### Common Solutions

**1. Database Connection Issues:**
```bash
# Check database is running
docker-compose ps postgres

# Reset database connection
docker-compose restart postgres
docker-compose restart app

# Verify database URL format
echo $DATABASE_URL
# Should be: postgresql+asyncpg://user:pass@host:port/dbname
```

**2. Environment Variable Issues:**
```bash
# Validate .env file
cat .env.production | grep -v '^#' | grep -v '^$'

# Recreate container with fresh environment
docker-compose down
docker-compose up -d --force-recreate
```

**3. Port Conflicts:**
```bash
# Find process using port 8000
sudo lsof -i :8000

# Kill conflicting process or change port
sudo kill -9 <PID>
# OR
# Edit docker-compose.yml to use different port
```

### Application Crashes During Runtime

#### Symptoms
- Application stops responding
- Container restarts frequently
- Memory or CPU spikes before crash

#### Diagnosis Steps

```bash
# Check recent crashes
docker logs leanvibe-app --since=1h | grep -i "error\|exception\|traceback"

# Monitor resource usage
docker stats leanvibe-app

# Check system limits
docker exec leanvibe-app ulimit -a

# Analyze core dumps (if available)
docker exec leanvibe-app find /tmp -name "core.*" -ls
```

#### Solutions

**1. Memory Leaks:**
```bash
# Increase memory limits
# In docker-compose.yml:
services:
  app:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

# Monitor memory usage over time
docker exec leanvibe-app python -c "
import psutil
import time
process = psutil.Process()
for i in range(10):
    print(f'Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
    time.sleep(5)
"
```

**2. Database Connection Pool Exhaustion:**
```bash
# Check active connections
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT count(*) as active_connections, state 
FROM pg_stat_activity 
WHERE datname = 'leanvibe_production' 
GROUP BY state;
"

# Increase pool size in environment
export DATABASE_POOL_SIZE=30
export DATABASE_MAX_OVERFLOW=20
```

**3. Infinite Loops or Deadlocks:**
```bash
# Check for long-running queries
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
FROM pg_stat_activity 
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';
"

# Kill long-running queries if necessary
# docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "SELECT pg_terminate_backend(<pid>);"
```

### API Endpoints Returning Errors

#### 401 Unauthorized Errors

```bash
# Check JWT token validity
python3 -c "
import jwt
import os
from datetime import datetime

token = 'your-jwt-token-here'
secret = os.getenv('JWT_SECRET_KEY', 'your-secret')

try:
    decoded = jwt.decode(token, secret, algorithms=['HS256'])
    print('Token is valid')
    print(f'Expires: {datetime.fromtimestamp(decoded.get(\"exp\", 0))}')
    print(f'User: {decoded.get(\"sub\")}')
except jwt.ExpiredSignatureError:
    print('Token has expired')
except jwt.InvalidTokenError as e:
    print(f'Token is invalid: {e}')
"

# Test authentication endpoint
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password"}' \
  -v
```

#### 422 Validation Errors

```bash
# Check request format
curl -X POST http://localhost:8000/api/v1/agents \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"name":"test-agent","role":"developer"}' \
  -v

# Validate JSON syntax
echo '{"name":"test-agent","role":"developer"}' | python -m json.tool
```

#### 500 Internal Server Errors

```bash
# Check application logs for detailed error
docker logs leanvibe-app --tail=20 | grep -A 10 -B 5 "500\|Internal Server Error"

# Enable debug logging temporarily
docker exec leanvibe-app sed -i 's/LOG_LEVEL=INFO/LOG_LEVEL=DEBUG/' .env.production
docker-compose restart app

# Test specific endpoint
curl -X GET http://localhost:8000/api/v1/agents \
  -H "Authorization: Bearer <token>" \
  -w "Response time: %{time_total}s\nHTTP code: %{http_code}\n"
```

## Database Problems

### Connection Issues

#### Symptoms
- "Connection refused" errors
- "Too many connections" errors
- Slow database queries

#### Diagnosis

```bash
# Check PostgreSQL status
docker-compose ps postgres

# Check connection limits
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT setting FROM pg_settings WHERE name = 'max_connections';
"

# Check current connections
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT count(*) as current_connections, state, application_name
FROM pg_stat_activity 
WHERE datname = 'leanvibe_production'
GROUP BY state, application_name;
"

# Check for blocked queries
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT blocked_locks.pid AS blocked_pid,
       blocked_activity.usename AS blocked_user,
       blocking_locks.pid AS blocking_pid,
       blocking_activity.usename AS blocking_user,
       blocked_activity.query AS blocked_statement,
       blocking_activity.query AS current_statement_in_blocking_process
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
"
```

#### Solutions

**1. Increase Connection Limits:**
```bash
# Edit PostgreSQL configuration
docker exec leanvibe-postgres sed -i 's/max_connections = 100/max_connections = 200/' /var/lib/postgresql/data/postgresql.conf

# Restart PostgreSQL
docker-compose restart postgres

# Verify change
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "SHOW max_connections;"
```

**2. Optimize Connection Pool:**
```bash
# Update application environment
cat >> .env.production << EOF
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=3600
EOF

docker-compose restart app
```

### Performance Issues

#### Slow Queries

```bash
# Enable slow query logging
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
ALTER SYSTEM SET log_min_duration_statement = 1000;  -- Log queries taking > 1 second
SELECT pg_reload_conf();
"

# Check slow queries
docker exec leanvibe-postgres tail -100 /var/lib/postgresql/data/log/postgresql-*.log | grep "duration:"

# Analyze query performance
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT query, calls, total_time, mean_time, min_time, max_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
"
```

#### Missing Indexes

```bash
# Check for sequential scans on large tables
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT schemaname, tablename, seq_scan, seq_tup_read, 
       idx_scan, idx_tup_fetch,
       seq_tup_read / seq_scan as avg_seq_tup_read
FROM pg_stat_user_tables 
WHERE seq_scan > 0 
ORDER BY seq_tup_read DESC 
LIMIT 10;
"

# Create missing indexes
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
-- Add indexes for common queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_status_priority ON tasks(status, priority);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_status_capabilities ON agents(status) WHERE capabilities IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_contexts_created_at ON contexts(created_at);
"
```

### Data Integrity Issues

#### Orphaned Records

```bash
# Check for orphaned tasks (tasks without valid agents)
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT COUNT(*) as orphaned_tasks
FROM tasks t
LEFT JOIN agents a ON t.assigned_agent_id = a.id
WHERE t.assigned_agent_id IS NOT NULL AND a.id IS NULL;
"

# Clean up orphaned records
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
UPDATE tasks 
SET assigned_agent_id = NULL, status = 'pending'
WHERE assigned_agent_id NOT IN (SELECT id FROM agents);
"
```

#### Database Corruption

```bash
# Check database corruption
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT datname, checksum_failures, checksum_last_failure
FROM pg_stat_database
WHERE datname = 'leanvibe_production';
"

# Run integrity checks
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"
```

## Redis and Caching Issues

### Redis Connection Problems

#### Diagnosis

```bash
# Check Redis status
docker-compose ps redis

# Test Redis connection
redis-cli -h localhost -p 6379 -a "$REDIS_PASSWORD" ping

# Check Redis info
redis-cli -h localhost -p 6379 -a "$REDIS_PASSWORD" info

# Monitor Redis commands
redis-cli -h localhost -p 6379 -a "$REDIS_PASSWORD" monitor
```

#### Common Issues

**1. Redis Memory Full:**
```bash
# Check memory usage
redis-cli -h localhost -p 6379 -a "$REDIS_PASSWORD" info memory

# Clear cache if safe to do so
redis-cli -h localhost -p 6379 -a "$REDIS_PASSWORD" flushdb

# Increase memory limit
# Edit docker-compose.yml:
services:
  redis:
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
```

**2. Redis Slow Performance:**
```bash
# Check slow log
redis-cli -h localhost -p 6379 -a "$REDIS_PASSWORD" slowlog get 10

# Monitor keyspace
redis-cli -h localhost -p 6379 -a "$REDIS_PASSWORD" info keyspace

# Check for large keys
redis-cli -h localhost -p 6379 -a "$REDIS_PASSWORD" --bigkeys
```

### Caching Issues

#### Cache Misses

```bash
# Check cache hit ratio
redis-cli -h localhost -p 6379 -a "$REDIS_PASSWORD" info stats | grep keyspace

# Monitor cache operations in application logs
docker logs leanvibe-app | grep -i cache | tail -20
```

#### Stale Cache Data

```bash
# Clear specific cache patterns
redis-cli -h localhost -p 6379 -a "$REDIS_PASSWORD" eval "
local keys = redis.call('keys', ARGV[1])
for i=1,#keys,5000 do
  redis.call('del', unpack(keys, i, math.min(i+4999, #keys)))
end
return #keys
" 0 "agent:*"

# Set shorter TTL for problematic keys
redis-cli -h localhost -p 6379 -a "$REDIS_PASSWORD" expire "key_name" 300
```

## Agent Management Problems

### Agent Not Responding

#### Symptoms
- Agent status shows as "inactive"
- Tasks assigned to agent never complete
- Agent doesn't appear in active agents list

#### Diagnosis

```bash
# Check agent status in database
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT id, name, status, last_activity, created_at
FROM agents
WHERE status != 'active'
ORDER BY last_activity DESC;
"

# Check agent logs
docker logs leanvibe-app | grep "agent_id.*$(agent_id)" | tail -20

# Check Redis streams for agent messages
redis-cli -h localhost -p 6379 -a "$REDIS_PASSWORD" xlen "agent_messages:$(agent_id)"
```

#### Solutions

**1. Restart Unresponsive Agent:**
```bash
# Reset agent status
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
UPDATE agents 
SET status = 'active', last_activity = NOW()
WHERE id = '$(agent_id)';
"

# Clear agent message queue
redis-cli -h localhost -p 6379 -a "$REDIS_PASSWORD" del "agent_messages:$(agent_id)"

# Restart application to reinitialize agents
docker-compose restart app
```

**2. Agent Overload:**
```bash
# Check agent workload
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT a.id, a.name, a.max_concurrent_tasks,
       COUNT(t.id) as current_tasks,
       COUNT(t.id) FILTER (WHERE t.status = 'in_progress') as active_tasks
FROM agents a
LEFT JOIN tasks t ON a.id = t.assigned_agent_id
WHERE a.status = 'active'
GROUP BY a.id, a.name, a.max_concurrent_tasks
ORDER BY current_tasks DESC;
"

# Reduce agent workload
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
UPDATE agents 
SET max_concurrent_tasks = 2
WHERE max_concurrent_tasks > 3;
"
```

### Agent Assignment Issues

#### Tasks Not Being Assigned

```bash
# Check available agents for task requirements
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT a.id, a.name, a.capabilities, a.status, a.max_concurrent_tasks,
       COUNT(t.id) as current_tasks
FROM agents a
LEFT JOIN tasks t ON a.id = t.assigned_agent_id AND t.status IN ('assigned', 'in_progress')
WHERE a.status = 'active'
GROUP BY a.id, a.name, a.capabilities, a.status, a.max_concurrent_tasks;
"

# Check pending tasks
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT id, title, priority, requirements, created_at
FROM tasks
WHERE status = 'pending'
ORDER BY priority DESC, created_at ASC;
"
```

#### Capability Matching Problems

```bash
# Debug capability matching
python3 -c "
import json
import psycopg2
from psycopg2.extras import RealDictCursor

# Connect to database
conn = psycopg2.connect('postgresql://leanvibe_admin:password@localhost:5432/leanvibe_production')
cur = conn.cursor(cursor_factory=RealDictCursor)

# Get task requirements
cur.execute('SELECT id, title, requirements FROM tasks WHERE status = \"pending\" LIMIT 5')
tasks = cur.fetchall()

# Get agent capabilities
cur.execute('SELECT id, name, capabilities FROM agents WHERE status = \"active\"')
agents = cur.fetchall()

for task in tasks:
    print(f'Task: {task[\"title\"]}')
    print(f'Requirements: {task[\"requirements\"]}')
    
    for agent in agents:
        capabilities = json.loads(agent['capabilities'] or '[]')
        requirements = json.loads(task['requirements'] or '[]')
        
        match = set(requirements).issubset(set(capabilities))
        print(f'  Agent {agent[\"name\"]}: {\"✓\" if match else \"✗\"} (has: {capabilities})')
    print()

conn.close()
"
```

## Workflow Execution Issues

### Workflows Not Starting

#### Diagnosis

```bash
# Check workflow status
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT id, name, status, created_at, started_at, error_message
FROM workflows
WHERE status IN ('draft', 'failed')
ORDER BY created_at DESC;
"

# Check workflow tasks
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT w.name as workflow_name, t.title, t.status, t.requirements
FROM workflows w
JOIN workflow_tasks wt ON w.id = wt.workflow_id
JOIN tasks t ON wt.task_id = t.id
WHERE w.status = 'failed'
ORDER BY w.id, wt.order_index;
"
```

#### Solutions

**1. Fix Dependency Issues:**
```bash
# Check for circular dependencies
python3 -c "
import psycopg2
from psycopg2.extras import RealDictCursor
import json

conn = psycopg2.connect('postgresql://leanvibe_admin:password@localhost:5432/leanvibe_production')
cur = conn.cursor(cursor_factory=RealDictCursor)

def check_circular_deps(workflow_id):
    cur.execute('''
        SELECT task_id, dependencies
        FROM workflow_tasks
        WHERE workflow_id = %s
    ''', (workflow_id,))
    
    tasks = cur.fetchall()
    dependencies = {task['task_id']: json.loads(task['dependencies'] or '[]') for task in tasks}
    
    def has_cycle(node, visited, rec_stack):
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in dependencies.get(node, []):
            if neighbor not in visited:
                if has_cycle(neighbor, visited, rec_stack):
                    return True
            elif neighbor in rec_stack:
                return True
        
        rec_stack.remove(node)
        return False
    
    visited = set()
    for task_id in dependencies:
        if task_id not in visited:
            if has_cycle(task_id, visited, set()):
                return True
    return False

# Check all active workflows
cur.execute('SELECT id, name FROM workflows WHERE status = \"active\"')
workflows = cur.fetchall()

for workflow in workflows:
    if check_circular_deps(workflow['id']):
        print(f'Circular dependency detected in workflow: {workflow[\"name\"]}')

conn.close()
"
```

**2. Resource Availability:**
```bash
# Check if required agents are available
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
WITH task_requirements AS (
    SELECT DISTINCT jsonb_array_elements_text(requirements) as required_skill
    FROM tasks
    WHERE status = 'pending'
),
agent_capabilities AS (
    SELECT DISTINCT jsonb_array_elements_text(capabilities) as available_skill
    FROM agents
    WHERE status = 'active'
)
SELECT tr.required_skill, 
       CASE WHEN ac.available_skill IS NOT NULL THEN 'Available' ELSE 'Missing' END as status
FROM task_requirements tr
LEFT JOIN agent_capabilities ac ON tr.required_skill = ac.available_skill
ORDER BY status, tr.required_skill;
"
```

### Workflows Stuck or Hanging

#### Diagnosis

```bash
# Find stuck workflows
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT w.id, w.name, w.status, w.started_at,
       NOW() - w.started_at as running_time,
       COUNT(t.id) as total_tasks,
       COUNT(t.id) FILTER (WHERE t.status = 'completed') as completed_tasks,
       COUNT(t.id) FILTER (WHERE t.status IN ('assigned', 'in_progress')) as active_tasks,
       COUNT(t.id) FILTER (WHERE t.status = 'pending') as pending_tasks
FROM workflows w
LEFT JOIN workflow_tasks wt ON w.id = wt.workflow_id
LEFT JOIN tasks t ON wt.task_id = t.id
WHERE w.status = 'running' AND w.started_at < NOW() - INTERVAL '1 hour'
GROUP BY w.id, w.name, w.status, w.started_at
ORDER BY running_time DESC;
"

# Check for blocked tasks
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT t.id, t.title, t.status, t.assigned_agent_id, t.created_at,
       a.name as agent_name, a.status as agent_status
FROM tasks t
LEFT JOIN agents a ON t.assigned_agent_id = a.id
WHERE t.status IN ('assigned', 'in_progress') 
  AND t.created_at < NOW() - INTERVAL '30 minutes'
ORDER BY t.created_at;
"
```

#### Solutions

**1. Reassign Stuck Tasks:**
```bash
# Reset stuck tasks
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
UPDATE tasks 
SET status = 'pending', assigned_agent_id = NULL
WHERE status IN ('assigned', 'in_progress') 
  AND created_at < NOW() - INTERVAL '30 minutes';
"

# Restart workflow orchestrator
docker-compose restart app
```

**2. Handle Task Dependencies:**
```bash
# Check dependency chain for stuck tasks
python3 -c "
import psycopg2
from psycopg2.extras import RealDictCursor
import json

conn = psycopg2.connect('postgresql://leanvibe_admin:password@localhost:5432/leanvibe_production')
cur = conn.cursor(cursor_factory=RealDictCursor)

# Find tasks waiting on dependencies
cur.execute('''
    SELECT t.id, t.title, t.status, wt.dependencies
    FROM tasks t
    JOIN workflow_tasks wt ON t.id = wt.task_id
    WHERE t.status = 'pending' AND wt.dependencies IS NOT NULL
''')

waiting_tasks = cur.fetchall()

for task in waiting_tasks:
    dependencies = json.loads(task['dependencies'])
    print(f'Task \"{task[\"title\"]}\" waiting on:')
    
    for dep_id in dependencies:
        cur.execute('SELECT title, status FROM tasks WHERE id = %s', (dep_id,))
        dep_task = cur.fetchone()
        if dep_task:
            print(f'  - {dep_task[\"title\"]} ({dep_task[\"status\"]})')
    print()

conn.close()
"
```

## Performance Problems

### High CPU Usage

#### Diagnosis

```bash
# Monitor CPU usage
top -p $(docker inspect --format='{{.State.Pid}}' leanvibe-app)

# Check for CPU-intensive queries
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT pid, query_start, state, query
FROM pg_stat_activity
WHERE state = 'active' AND query_start < NOW() - INTERVAL '30 seconds'
ORDER BY query_start;
"

# Profile application performance
docker exec leanvibe-app python -c "
import cProfile
import pstats
import io

# Sample profiling (replace with actual problematic code)
pr = cProfile.Profile()
pr.enable()

# Your problematic function here
# result = some_expensive_function()

pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
"
```

#### Solutions

**1. Database Query Optimization:**
```bash
# Enable query statistics
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
"

# Find slow queries
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT query, calls, total_time, mean_time, rows,
       100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;
"

# Add missing indexes
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
-- Example indexes for common slow queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_status_created ON tasks(status, created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_status_last_activity ON agents(status, last_activity);
"
```

**2. Application-Level Optimization:**
```bash
# Increase worker processes
# Edit docker-compose.yml
services:
  app:
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 6

# Enable connection pooling
cat >> .env.production << EOF
DATABASE_POOL_SIZE=25
DATABASE_MAX_OVERFLOW=15
REDIS_POOL_SIZE=25
EOF

docker-compose restart app
```

### High Memory Usage

#### Diagnosis

```bash
# Monitor memory usage
docker stats leanvibe-app --no-stream

# Check for memory leaks
docker exec leanvibe-app python -c "
import psutil
import gc
import time

process = psutil.Process()
for i in range(5):
    print(f'Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
    print(f'Objects: {len(gc.get_objects())}')
    time.sleep(10)
"

# Check PostgreSQL memory usage
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT name, setting, unit FROM pg_settings WHERE name IN (
    'shared_buffers', 'work_mem', 'maintenance_work_mem', 'effective_cache_size'
);
"
```

#### Solutions

**1. Optimize PostgreSQL Memory:**
```bash
# Tune PostgreSQL memory settings
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
ALTER SYSTEM SET shared_buffers = '1GB';
ALTER SYSTEM SET work_mem = '16MB';
ALTER SYSTEM SET maintenance_work_mem = '256MB';
ALTER SYSTEM SET effective_cache_size = '3GB';
SELECT pg_reload_conf();
"
```

**2. Application Memory Management:**
```bash
# Implement memory limits
# In docker-compose.yml:
services:
  app:
    deploy:
      resources:
        limits:
          memory: 2G
    environment:
      - PYTHONHASHSEED=random
      - PYTHONUNBUFFERED=1

# Clear caches periodically
redis-cli -h localhost -p 6379 -a "$REDIS_PASSWORD" eval "
return redis.call('del', unpack(redis.call('keys', 'cache:*')))
" 0
```

### Slow Response Times

#### Diagnosis

```bash
# Measure API response times
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/api/v1/agents"

# Create curl-format.txt:
cat > curl-format.txt << EOF
     time_namelookup:  %{time_namelookup}\n
        time_connect:  %{time_connect}\n
     time_appconnect:  %{time_appconnect}\n
    time_pretransfer:  %{time_pretransfer}\n
       time_redirect:  %{time_redirect}\n
  time_starttransfer:  %{time_starttransfer}\n
                     ----------\n
          time_total:  %{time_total}\n
EOF

# Check database query performance
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del, n_tup_hot_upd, n_live_tup, n_dead_tup
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;
"
```

#### Solutions

**1. Enable Caching:**
```python
# Add caching to frequently accessed endpoints
from functools import lru_cache
import redis

redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

@lru_cache(maxsize=128)
def get_agent_capabilities(agent_id: str):
    # Cache agent capabilities
    cached = redis_client.get(f"agent_capabilities:{agent_id}")
    if cached:
        return json.loads(cached)
    
    # Fetch from database and cache
    capabilities = fetch_from_db(agent_id)
    redis_client.setex(f"agent_capabilities:{agent_id}", 300, json.dumps(capabilities))
    return capabilities
```

**2. Database Optimization:**
```bash
# Vacuum and analyze tables
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
VACUUM ANALYZE;
"

# Update table statistics
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
ANALYZE tasks;
ANALYZE agents;
ANALYZE workflows;
"
```

## Authentication and Authorization

### Login Issues

#### User Cannot Login

```bash
# Check user exists and is active
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT id, email, username, is_active, is_superuser, created_at, last_login
FROM users
WHERE email = 'user@example.com';
"

# Verify password hash
python3 -c "
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')
stored_hash = 'hash_from_database'
password = 'user_entered_password'

if pwd_context.verify(password, stored_hash):
    print('Password is correct')
else:
    print('Password is incorrect')
"

# Check authentication endpoint
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"password"}' \
  -v
```

#### Token Issues

```bash
# Validate JWT token
python3 -c "
import jwt
import json
from datetime import datetime

token = 'jwt_token_here'
secret = 'your_jwt_secret'

try:
    payload = jwt.decode(token, secret, algorithms=['HS256'])
    print('Token is valid')
    print(json.dumps(payload, indent=2))
    
    if 'exp' in payload:
        exp_time = datetime.fromtimestamp(payload['exp'])
        print(f'Expires: {exp_time}')
        if exp_time < datetime.now():
            print('Token has expired')
        else:
            print('Token is still valid')
            
except jwt.ExpiredSignatureError:
    print('Token has expired')
except jwt.InvalidTokenError as e:
    print(f'Token is invalid: {e}')
"

# Reset user password
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
UPDATE users 
SET hashed_password = '\$2b\$12\$newhashedpasswordhere'
WHERE email = 'user@example.com';
"
```

### Permission Issues

#### Access Denied Errors

```bash
# Check user roles and permissions
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
SELECT u.email, u.is_active, u.is_superuser, r.name as role_name, p.name as permission_name
FROM users u
LEFT JOIN user_roles ur ON u.id = ur.user_id
LEFT JOIN roles r ON ur.role_id = r.id
LEFT JOIN role_permissions rp ON r.id = rp.role_id
LEFT JOIN permissions p ON rp.permission_id = p.id
WHERE u.email = 'user@example.com';
"

# Grant necessary permissions
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
-- Make user a superuser temporarily
UPDATE users SET is_superuser = true WHERE email = 'user@example.com';

-- Or assign specific role
INSERT INTO user_roles (user_id, role_id)
SELECT u.id, r.id
FROM users u, roles r
WHERE u.email = 'user@example.com' AND r.name = 'admin';
"
```

## WebSocket and Real-time Issues

### WebSocket Connection Problems

#### Connection Fails

```bash
# Test WebSocket connection
wscat -c ws://localhost:8000/ws/observability

# Check nginx WebSocket configuration
docker exec leanvibe-nginx nginx -t

# Verify WebSocket headers
curl -i -N -H "Connection: Upgrade" \
     -H "Upgrade: websocket" \
     -H "Sec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==" \
     -H "Sec-WebSocket-Version: 13" \
     http://localhost:8000/ws/observability
```

#### Connection Drops

```bash
# Check nginx timeout settings
docker exec leanvibe-nginx grep -r "proxy_read_timeout\|proxy_send_timeout" /etc/nginx/

# Update nginx configuration for WebSockets
cat > nginx-websocket.conf << EOF
location /ws/ {
    proxy_pass http://leanvibe_app;
    proxy_http_version 1.1;
    proxy_set_header Upgrade \$http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host \$host;
    proxy_read_timeout 86400;
    proxy_send_timeout 86400;
}
EOF

# Reload nginx
docker exec leanvibe-nginx nginx -s reload
```

### Real-time Updates Not Working

#### Diagnosis

```bash
# Check WebSocket connections
docker exec leanvibe-app netstat -an | grep :8000

# Monitor WebSocket messages
docker logs leanvibe-app | grep -i websocket | tail -20

# Test message broadcasting
curl -X POST http://localhost:8000/api/v1/test/broadcast \
  -H "Content-Type: application/json" \
  -d '{"message":"test","type":"system_alert"}'
```

#### Solutions

```python
# Add WebSocket debugging
import logging
import websockets

logging.basicConfig(level=logging.DEBUG)
websockets_logger = logging.getLogger('websockets')
websockets_logger.setLevel(logging.DEBUG)

# Test WebSocket endpoint
async def test_websocket():
    uri = "ws://localhost:8000/ws/observability"
    async with websockets.connect(uri) as websocket:
        # Send test message
        await websocket.send('{"type":"ping"}')
        
        # Wait for response
        response = await websocket.recv()
        print(f"Received: {response}")

import asyncio
asyncio.run(test_websocket())
```

## Mobile PWA Problems

### PWA Installation Issues

#### PWA Won't Install

```bash
# Check manifest.json
curl -s http://localhost:8000/manifest.json | jq '.'

# Verify service worker registration
curl -s http://localhost:8000/sw.js | head -20

# Check PWA requirements
lighthouse http://localhost:8000 --only-categories=pwa --chrome-flags="--headless"
```

#### Solutions

```javascript
// Fix service worker registration
if ('serviceWorker' in navigator) {
  window.addEventListener('load', function() {
    navigator.serviceWorker.register('/sw.js')
      .then(function(registration) {
        console.log('ServiceWorker registration successful');
      })
      .catch(function(err) {
        console.log('ServiceWorker registration failed: ', err);
      });
  });
}

// Update manifest.json
{
  "name": "LeanVibe Agent Hive",
  "short_name": "AgentHive",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#ffffff",
  "theme_color": "#000000",
  "icons": [
    {
      "src": "/icon-192x192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "/icon-512x512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ]
}
```

### Offline Functionality Issues

#### Data Not Syncing

```javascript
// Debug offline storage
console.log('IndexedDB available:', 'indexedDB' in window);

// Check stored data
if ('indexedDB' in window) {
  const request = indexedDB.open('LeanVibeCache', 1);
  request.onsuccess = function(event) {
    const db = event.target.result;
    const transaction = db.transaction(['tasks'], 'readonly');
    const store = transaction.objectStore('tasks');
    const getAllRequest = store.getAll();
    
    getAllRequest.onsuccess = function() {
      console.log('Cached tasks:', getAllRequest.result);
    };
  };
}

// Test background sync
if ('serviceWorker' in navigator && 'sync' in window.ServiceWorkerRegistration.prototype) {
  navigator.serviceWorker.ready.then(function(registration) {
    return registration.sync.register('background-sync');
  });
}
```

## Deployment and Infrastructure

### Container Issues

#### Container Won't Start

```bash
# Check container logs
docker logs leanvibe-app --details

# Inspect container configuration
docker inspect leanvibe-app | jq '.[0].Config'

# Check resource limits
docker stats leanvibe-app --no-stream

# Verify image integrity
docker images --digests | grep leanvibe
```

#### Network Issues

```bash
# Check Docker networks
docker network ls
docker network inspect leanvibe_leanvibe-network

# Test connectivity between containers
docker exec leanvibe-app ping postgres
docker exec leanvibe-app ping redis

# Check port bindings
docker port leanvibe-app
netstat -tlnp | grep :8000
```

### Load Balancer Issues

#### 502 Bad Gateway

```bash
# Check upstream health
docker exec leanvibe-nginx nginx -t
docker logs leanvibe-nginx --tail=50

# Test backend connectivity
docker exec leanvibe-nginx curl -f http://app:8000/health

# Check nginx configuration
docker exec leanvibe-nginx cat /etc/nginx/nginx.conf | grep -A 10 "upstream"
```

#### SSL/TLS Issues

```bash
# Check certificate validity
openssl x509 -in /path/to/cert.crt -text -noout | grep -A 2 "Validity"

# Test SSL configuration
echo | openssl s_client -servername your-domain.com -connect your-domain.com:443 2>/dev/null | openssl x509 -noout -dates

# Check SSL cipher suites
nmap --script ssl-enum-ciphers -p 443 your-domain.com
```

## Monitoring and Alerting

### Prometheus Issues

#### Metrics Not Collecting

```bash
# Check Prometheus targets
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health, lastError: .lastError}'

# Verify metrics endpoint
curl -s http://localhost:8000/metrics | head -20

# Check Prometheus configuration
docker logs leanvibe-prometheus --tail=50
```

#### Missing Metrics

```bash
# Query available metrics
curl -s http://localhost:9090/api/v1/label/__name__/values | jq '.data[]' | grep leanvibe

# Check metric cardinality
curl -s "http://localhost:9090/api/v1/query?query=prometheus_tsdb_symbol_table_size_bytes"
```

### Grafana Issues

#### Dashboard Not Loading

```bash
# Check Grafana logs
docker logs leanvibe-grafana --tail=50

# Test data source connection
curl -X GET http://admin:admin@localhost:3000/api/datasources

# Verify dashboard configuration
curl -X GET http://admin:admin@localhost:3000/api/dashboards/home
```

## Log Analysis

### Centralized Logging

#### Setup Log Aggregation

```bash
# Configure structured logging
cat >> .env.production << EOF
LOG_FORMAT=json
LOG_LEVEL=INFO
STRUCTURED_LOGGING=true
EOF

# Install log forwarding (optional)
docker run -d --name=log-forwarder \
  --log-driver=fluentd \
  --log-opt fluentd-address=localhost:24224 \
  --log-opt tag="docker.{{.Name}}" \
  your-app-image
```

#### Log Analysis Scripts

```bash
#!/bin/bash
# analyze-logs.sh - Analyze application logs

LOG_FILE="/opt/leanvibe/logs/app.log"

echo "=== Log Analysis Report ==="
echo "Time period: $(tail -1 $LOG_FILE | jq -r '.timestamp') to $(head -1 $LOG_FILE | jq -r '.timestamp')"
echo

echo "--- Error Summary ---"
grep '"level":"ERROR"' $LOG_FILE | jq -r '.message' | sort | uniq -c | sort -nr | head -10

echo "--- Request Summary ---"
grep '"message":"HTTP"' $LOG_FILE | jq -r '.method + " " + .path' | sort | uniq -c | sort -nr | head -10

echo "--- Slow Queries ---"
grep '"slow_query":true' $LOG_FILE | jq -r '.query + " (" + (.duration|tostring) + "ms)"' | head -10

echo "--- Agent Activity ---"
grep '"agent_id"' $LOG_FILE | jq -r '.agent_id' | sort | uniq -c | sort -nr | head -10
```

## Emergency Procedures

### System Down Recovery

#### Complete System Recovery

```bash
#!/bin/bash
# emergency-recovery.sh - Emergency system recovery

echo "=== EMERGENCY RECOVERY PROCEDURE ==="
echo "Starting at: $(date)"

# Stop all services
echo "1. Stopping all services..."
docker-compose down

# Check system resources
echo "2. Checking system resources..."
df -h
free -h
uptime

# Start core services first
echo "3. Starting core infrastructure..."
docker-compose up -d postgres redis

# Wait for databases to be ready
echo "4. Waiting for databases..."
sleep 30

# Check database connectivity
if ! PGPASSWORD="$DB_PASSWORD" psql -h localhost -U leanvibe_admin -d leanvibe_production -c "SELECT 1;" >/dev/null 2>&1; then
    echo "ERROR: Cannot connect to database!"
    exit 1
fi

if ! redis-cli -a "$REDIS_PASSWORD" ping >/dev/null 2>&1; then
    echo "ERROR: Cannot connect to Redis!"
    exit 1
fi

# Start application
echo "5. Starting application..."
docker-compose up -d app

# Wait for application to be ready
echo "6. Waiting for application..."
for i in {1..30}; do
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        echo "Application is ready"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 10
done

# Start remaining services
echo "7. Starting remaining services..."
docker-compose up -d

# Verify all services
echo "8. Verifying services..."
docker-compose ps

# Run health checks
echo "9. Running health checks..."
curl -f http://localhost:8000/health && echo "✓ Application healthy"
curl -f http://localhost:9090/targets && echo "✓ Prometheus healthy"
curl -f http://localhost:3000/login && echo "✓ Grafana healthy"

echo "=== RECOVERY COMPLETE ==="
echo "Completed at: $(date)"
```

### Data Recovery

#### Database Recovery

```bash
#!/bin/bash
# database-recovery.sh - Recover database from backup

BACKUP_FILE="$1"
DB_NAME="leanvibe_production"

if [[ -z "$BACKUP_FILE" ]]; then
    echo "Usage: $0 <backup_file>"
    echo "Available backups:"
    ls -la /opt/leanvibe/backups/database/
    exit 1
fi

echo "=== DATABASE RECOVERY ==="
echo "Recovering from: $BACKUP_FILE"
echo "Target database: $DB_NAME"
echo

# Confirm recovery
read -p "This will overwrite the current database. Continue? (yes/no): " confirm
if [[ $confirm != "yes" ]]; then
    echo "Recovery cancelled"
    exit 1
fi

# Stop application
echo "1. Stopping application..."
docker-compose stop app

# Create backup of current state
echo "2. Creating backup of current state..."
CURRENT_BACKUP="/opt/leanvibe/backups/database/pre_recovery_$(date +%Y%m%d_%H%M%S).sql"
PGPASSWORD="$DB_PASSWORD" pg_dump -h localhost -U leanvibe_admin -d $DB_NAME > $CURRENT_BACKUP
echo "Current state backed up to: $CURRENT_BACKUP"

# Drop and recreate database
echo "3. Recreating database..."
PGPASSWORD="$DB_PASSWORD" psql -h localhost -U leanvibe_admin -c "DROP DATABASE IF EXISTS ${DB_NAME};"
PGPASSWORD="$DB_PASSWORD" psql -h localhost -U leanvibe_admin -c "CREATE DATABASE ${DB_NAME};"

# Restore from backup
echo "4. Restoring from backup..."
if [[ $BACKUP_FILE == *.gz ]]; then
    gunzip -c $BACKUP_FILE | PGPASSWORD="$DB_PASSWORD" psql -h localhost -U leanvibe_admin -d $DB_NAME
else
    PGPASSWORD="$DB_PASSWORD" psql -h localhost -U leanvibe_admin -d $DB_NAME < $BACKUP_FILE
fi

# Run migrations to ensure schema is current
echo "5. Running migrations..."
cd /opt/leanvibe/app
source venv/bin/activate
alembic upgrade head

# Restart application
echo "6. Starting application..."
docker-compose start app

# Verify recovery
echo "7. Verifying recovery..."
sleep 10
curl -f http://localhost:8000/health && echo "✓ Recovery successful"

echo "=== RECOVERY COMPLETE ==="
```

### Incident Response

#### Security Incident Response

```bash
#!/bin/bash
# security-incident-response.sh - Security incident response

echo "=== SECURITY INCIDENT RESPONSE ==="
echo "Incident detected at: $(date)"

# Immediate containment
echo "1. IMMEDIATE CONTAINMENT"

# Block suspicious IPs (if identified)
if [[ -n "$SUSPICIOUS_IP" ]]; then
    echo "Blocking IP: $SUSPICIOUS_IP"
    sudo ufw deny from $SUSPICIOUS_IP
    
    # Block in nginx if running
    echo "deny $SUSPICIOUS_IP;" | sudo tee -a /etc/nginx/conf.d/blocked_ips.conf
    sudo nginx -s reload
fi

# Revoke all sessions
echo "Revoking all active sessions..."
redis-cli -a "$REDIS_PASSWORD" eval "return redis.call('del', unpack(redis.call('keys', 'session:*')))" 0

# Force password reset for all users
docker exec leanvibe-postgres psql -U leanvibe_admin -d leanvibe_production -c "
UPDATE users SET force_password_reset = true, last_password_change = NOW();
"

# Disable API access temporarily
echo "Temporarily disabling API access..."
docker-compose stop app

echo "2. EVIDENCE COLLECTION"

# Collect logs
mkdir -p /tmp/incident_$(date +%Y%m%d_%H%M%S)
cp -r /opt/leanvibe/logs /tmp/incident_$(date +%Y%m%d_%H%M%S)/
docker logs leanvibe-app > /tmp/incident_$(date +%Y%m%d_%H%M%S)/app_container.log
docker logs leanvibe-nginx > /tmp/incident_$(date +%Y%m%d_%H%M%S)/nginx_container.log

# System state
ps aux > /tmp/incident_$(date +%Y%m%d_%H%M%S)/processes.txt
netstat -tulpn > /tmp/incident_$(date +%Y%m%d_%H%M%S)/network.txt
docker ps -a > /tmp/incident_$(date +%Y%m%d_%H%M%S)/containers.txt

echo "3. NOTIFICATION"
# Send incident notification
curl -X POST "$SLACK_WEBHOOK_URL" \
  -H 'Content-type: application/json' \
  --data '{"text":"🚨 SECURITY INCIDENT DETECTED - System contained, investigating"}'

echo "=== CONTAINMENT COMPLETE ==="
echo "Next steps:"
echo "1. Investigate collected evidence"
echo "2. Identify attack vector"
echo "3. Apply security patches"
echo "4. Restore service with enhanced monitoring"
echo "5. Update security procedures"
```

This comprehensive troubleshooting guide provides detailed procedures for diagnosing and resolving common issues in LeanVibe Agent Hive. The guide includes automated scripts, diagnostic commands, and step-by-step solutions for various operational problems.