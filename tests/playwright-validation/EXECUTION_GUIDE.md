# LeanVibe Agent Hive 2.0 - Playwright Validation Execution Guide

## Overview
This guide provides step-by-step instructions for executing the comprehensive Playwright validation suite to verify every claim about the LeanVibe Agent Hive 2.0 autonomous development platform.

## Pre-Execution Requirements

### 1. System Prerequisites
- **Node.js**: Version 18.0.0 or higher
- **Docker & Docker Compose**: For infrastructure services
- **Git**: For repository access
- **Terminal/Command Line**: For executing commands

### 2. Environment Setup
```bash
# Navigate to the validation directory
cd /Users/bogdan/work/leanvibe-dev/bee-hive/tests/playwright-validation

# Install dependencies
npm install

# Install Playwright browsers
npx playwright install
```

### 3. Infrastructure Startup
```bash
# Navigate to the main project directory
cd /Users/bogdan/work/leanvibe-dev/bee-hive

# Start all services using Docker Compose
docker compose up -d

# Verify services are running
docker compose ps

# Check service health
curl http://localhost:8000/health
curl http://localhost:3002 # Dashboard (may take a moment to start)
```

## Validation Execution Strategy

### Phase 1: Infrastructure Validation (Critical Foundation)
```bash
cd tests/playwright-validation

# Run infrastructure tests first - these must pass for system to be functional
npm run test:infrastructure

# Expected outcomes:
# ✅ FastAPI health endpoint returns 200
# ✅ PostgreSQL database connected with tables > 0
# ✅ Redis connected with streams active
# ✅ System metrics endpoint accessible
# ✅ Performance benchmarks under 5 seconds
```

### Phase 2: Multi-Agent System Validation (Core Claims)
```bash
# Run agent system tests - validates the 6 agent claim
npm run test:agents

# Expected outcomes:
# ✅ Exactly 6 agents spawned and active
# ✅ All expected roles present (Product Manager, Architect, Backend Dev, QA, DevOps)
# ✅ Agent capabilities match role requirements
# ✅ Heartbeat system functioning (30-second intervals)
# ✅ Task assignment and management working
```

### Phase 3: Dashboard Validation (User Interface Claims)
```bash
# Run dashboard tests - validates localhost:3002 dashboard claim
npm run test:dashboard

# Expected outcomes:
# ✅ Dashboard accessible at localhost:3002
# ✅ Vue.js application loads successfully
# ✅ Agent information displayed
# ✅ Real-time updates functioning
# ✅ Interactive components working
# ✅ Performance under 20 seconds load time
```

### Phase 4: API Validation (90+ Endpoint Claims)
```bash
# Run API discovery and validation tests
npm run test:api

# Expected outcomes:
# ✅ 90+ API endpoints discovered from OpenAPI schema
# ✅ Core endpoints return proper status codes
# ✅ CRUD operations functional
# ✅ Error handling proper (4xx, 5xx responses)
# ✅ Performance under 5 seconds average
```

### Phase 5: Autonomous Workflow Validation (Advanced Claims)
```bash
# Run workflow and autonomous development tests
npm run test:workflows

# Expected outcomes:
# ✅ Self-modification engine endpoints available
# ✅ GitHub integration capabilities present
# ✅ Code execution sandbox functional
# ✅ Multi-agent coordination working
# ✅ Task creation and assignment functional
# ✅ Learning and adaptation capabilities present
```

### Phase 6: Complete Integration Validation
```bash
# Run full integration test suite
npm run test:integration

# This will execute end-to-end scenarios validating:
# ✅ Complete system functionality under load
# ✅ Cross-component communication
# ✅ Data consistency across services
# ✅ Error recovery mechanisms
# ✅ Performance under realistic usage
```

## Evidence Collection and Reporting

### Automated Evidence Generation
Every test automatically collects:
- **Screenshots**: Visual proof of functionality
- **API Responses**: Complete API interaction logs
- **Performance Metrics**: Response times and resource usage
- **System States**: Database and Redis snapshots
- **Console Logs**: Application behavior logs
- **Network Activity**: Request/response patterns

### Evidence Location
```
tests/playwright-validation/reports/
├── evidence/
│   ├── infrastructure/
│   ├── multi-agent/
│   ├── dashboard/
│   ├── api/
│   └── workflows/
├── playwright-report/         # Interactive HTML report
├── validation-results.json    # Machine-readable results
└── validation-results.xml     # CI/CD compatible results
```

## Trust-Building Validation Commands

### Quick Validation (Essential Claims Only)
```bash
# Validate core claims in 10-15 minutes
npm run test:infrastructure && npm run test:agents && npm run test:dashboard
```

### Comprehensive Validation (All Claims)
```bash
# Complete validation suite - 30-45 minutes
npm run validate:claims
```

### Specific Claim Validation
```bash
# Test only agent spawning claim
npx playwright test tests/multi-agent/agent-spawning.spec.ts

# Test only API endpoint count claim
npx playwright test tests/api/endpoint-discovery.spec.ts

# Test only dashboard functionality claim
npx playwright test tests/dashboard/dashboard-functionality.spec.ts
```

## Manual Verification Steps

### Pre-Test Manual Checks
1. **Docker Services Check**:
   ```bash
   docker compose ps
   # Should show: postgres, redis, api, frontend (if applicable)
   ```

2. **API Accessibility**:
   ```bash
   curl http://localhost:8000/health
   # Should return: {"status": "healthy", "version": "2.0.0", ...}
   ```

3. **Dashboard Accessibility**:
   ```bash
   curl -I http://localhost:3002
   # Should return: HTTP/1.1 200 OK
   ```

### Post-Test Manual Verification
1. **Agent Count Verification**:
   ```bash
   curl http://localhost:8000/debug-agents | jq '.agent_count'
   # Should return: 6
   ```

2. **API Endpoint Count**:
   ```bash
   curl http://localhost:8000/openapi.json | jq '.paths | keys | length'
   # Should return: number >= 90
   ```

3. **Evidence Files Generated**:
   ```bash
   find tests/playwright-validation/reports/evidence -name "*.png" | wc -l
   # Should return: number > 20 (screenshots captured)
   ```

## Troubleshooting Common Issues

### Issue: Docker Services Not Starting
**Solution**:
```bash
# Stop all services
docker compose down

# Remove volumes if needed
docker compose down -v

# Restart services
docker compose up -d

# Check logs
docker compose logs
```

### Issue: Dashboard Not Accessible
**Solution**:
```bash
# Check if frontend service is running
docker compose ps frontend

# If not in docker-compose, dashboard might be served directly
# Check if there's a separate frontend process needed
```

### Issue: Agent Count Less Than 6
**Solution**:
```bash
# Wait longer for agent initialization (can take 30-60 seconds)
sleep 60

# Check agent logs
docker compose logs api | grep -i agent

# Restart API service if needed
docker compose restart api
```

### Issue: Tests Failing Due to Timeouts
**Solution**:
```bash
# Increase timeout multiplier
export TEST_TIMEOUT_MULTIPLIER=3

# Run tests with increased timeouts
npm run test
```

## Success Criteria Checklist

### Infrastructure Validation ✅
- [ ] All Docker services running
- [ ] PostgreSQL connected with >5 tables
- [ ] Redis connected with streams active
- [ ] API health endpoint returns "healthy"
- [ ] Metrics endpoint accessible

### Multi-Agent System ✅
- [ ] Exactly 6 agents active
- [ ] All 5 expected roles present (Product Manager, Architect, Backend Dev, QA, DevOps)
- [ ] Agent capabilities match role requirements
- [ ] Heartbeat system active (30-second intervals)
- [ ] Task assignment functional

### Dashboard Validation ✅
- [ ] Dashboard loads at localhost:3002
- [ ] Vue.js application functional
- [ ] Agent information displayed
- [ ] Interactive components working
- [ ] Load time under 20 seconds

### API Validation ✅
- [ ] 90+ endpoints discovered
- [ ] OpenAPI schema accessible
- [ ] Core CRUD operations working
- [ ] Proper error handling (4xx/5xx)
- [ ] Average response time under 5 seconds

### Autonomous Workflow ✅
- [ ] Self-modification endpoints available
- [ ] Code execution capabilities present
- [ ] Multi-agent coordination functional
- [ ] Task creation and management working
- [ ] Learning capabilities demonstrated

## Final Validation Report

Upon completion, the validation suite generates:

1. **HTML Evidence Report**: `reports/playwright-report/index.html`
   - Interactive dashboard with all test results
   - Screenshots and visual evidence
   - Performance metrics and trends
   - Pass/fail status for each claim

2. **JSON Results**: `reports/validation-results.json`
   - Machine-readable test results
   - Detailed evidence artifacts
   - Performance benchmarks
   - Error logs and debugging information

3. **Evidence Archive**: `reports/evidence/`
   - Organized by test category
   - Complete screenshot collection
   - API response logs
   - System state snapshots
   - Performance monitoring data

## Expected Validation Time

- **Quick Validation**: 10-15 minutes (core claims only)
- **Comprehensive Validation**: 30-45 minutes (all claims)
- **Evidence Review**: 15-30 minutes (manual verification)
- **Total Time**: 1-1.5 hours for complete validation

This comprehensive validation approach ensures that every claim about the LeanVibe Agent Hive 2.0 system is thoroughly tested with concrete, reproducible evidence.