# Agent Delegation Framework for Project Index Validation & Deployment

## 🎯 Overview

This framework defines how to delegate Project Index work across specialized agents to avoid context rot and maximize efficiency. Each agent has a specific domain of expertise and clear handoff protocols.

## 🤖 Agent Specialization Matrix

### Agent 1: Database & Schema Validator (db-validator-agent)
**Domain**: Database schema, migrations, data integrity
**Estimated Duration**: 1-2 hours
**Context Files**: 
- `docs/indexer/technical-specifications.md` (database schema)
- `migrations/versions/022_add_project_index_system.py`
- `app/models/project_index.py`

**Primary Tasks**:
- ✅ Validate database schema implementation
- ✅ Test all migrations work correctly
- ✅ Verify foreign key constraints and indexes
- ✅ Test data integrity and performance
- ✅ Validate enum types and constraints

**Success Criteria**:
- All 5 tables created successfully
- 19 performance indexes operational
- Migration runs without errors
- Data integrity tests pass

**Handoff Prompt**:
```
You are a database specialist validating the Project Index database schema. 

Your tasks:
1. Run database migrations and verify all tables are created
2. Test that all indexes improve query performance as expected
3. Validate foreign key constraints work correctly
4. Test enum types have correct values
5. Run data integrity tests with sample data

Use these validation commands:
- `python scripts/validate_project_index.py --api-only`
- `alembic upgrade head`
- Database query performance tests

Document any issues found and provide fixes. When complete, hand off to API Testing Agent with database validation status.
```

### Agent 2: API Testing & Validation (api-test-agent)
**Domain**: REST API endpoints, request/response validation
**Estimated Duration**: 2-3 hours
**Context Files**:
- `docs/indexer/technical-specifications.md` (API specs)
- `app/api/project_index.py`
- `tests/api/test_project_index_api.py`

**Primary Tasks**:
- ✅ Test all 8 REST API endpoints
- ✅ Validate request/response schemas
- ✅ Test error handling and edge cases
- ✅ Verify authentication and authorization
- ✅ Performance test API response times
- ✅ Test rate limiting and security

**Success Criteria**:
- All API endpoints return correct responses
- Response times < 200ms for standard operations
- Error handling works correctly
- API documentation matches implementation

**Handoff Prompt**:
```
You are an API testing specialist for the Project Index system.

Your tasks:
1. Test all REST API endpoints using the validation script
2. Verify request/response schemas match documentation
3. Test error handling and edge cases
4. Run performance tests to ensure < 200ms response times
5. Test authentication and security features

Use these tools:
- `python scripts/validate_project_index.py --api-only`
- Postman/httpx for manual testing
- API performance benchmarking

Previous agent status: [database validation results]

When complete, document API test results and hand off to WebSocket Testing Agent.
```

### Agent 3: WebSocket & Real-time Events (websocket-agent)
**Domain**: WebSocket events, real-time communication
**Estimated Duration**: 1-2 hours
**Context Files**:
- `app/project_index/websocket_events.py`
- `docs/indexer/technical-specifications.md` (WebSocket specs)

**Primary Tasks**:
- ✅ Test WebSocket connection and subscriptions
- ✅ Validate all event types are fired correctly
- ✅ Test event filtering and subscription management
- ✅ Verify real-time updates during analysis
- ✅ Test connection resilience and reconnection

**Success Criteria**:
- WebSocket connections established successfully
- All expected events are received
- Event payloads match specifications
- Real-time updates work during project analysis

**Handoff Prompt**:
```
You are a WebSocket and real-time events specialist.

Your tasks:
1. Test WebSocket connection establishment
2. Verify all project index events are fired correctly
3. Test event subscription and filtering
4. Validate event payloads match specifications
5. Test connection resilience and error handling

Use these tools:
- `python scripts/validate_project_index.py` (includes WebSocket tests)
- WebSocket client testing
- Event monitoring during project analysis

Previous agent status: [API validation results]

When complete, document WebSocket test results and hand off to Performance Testing Agent.
```

### Agent 4: Performance & Load Testing (perf-test-agent)
**Domain**: Performance benchmarking, load testing, optimization
**Estimated Duration**: 2-3 hours
**Context Files**:
- `docs/indexer/technical-specifications.md` (performance requirements)
- Performance test files in `tests/performance/`

**Primary Tasks**:
- ✅ Benchmark indexing performance for different project sizes
- ✅ Test API response times under load
- ✅ Memory usage profiling and optimization
- ✅ Concurrent analysis testing
- ✅ Database query performance optimization

**Success Criteria**:
- Small projects (< 100 files): Analysis < 30 seconds
- API responses: < 200ms for 95th percentile
- Memory usage: < 100MB for typical projects
- Concurrent operations: Support 10+ simultaneous analyses

**Handoff Prompt**:
```
You are a performance testing specialist for the Project Index system.

Your tasks:
1. Run comprehensive performance benchmarks
2. Test indexing speed for different project sizes
3. Load test API endpoints under concurrent requests
4. Profile memory usage and identify optimization opportunities
5. Test database query performance

Use these tools:
- `python scripts/validate_project_index.py` (includes performance tests)
- Memory profiling tools (psutil, memory_profiler)
- Load testing frameworks (httpx async, locust)
- Database query analysis

Previous agent status: [WebSocket validation results]

When complete, document performance results and hand off to Integration Testing Agent.
```

### Agent 5: Integration & System Testing (integration-agent)
**Domain**: End-to-end workflows, system integration
**Estimated Duration**: 2-3 hours
**Context Files**:
- Complete system documentation
- Integration test suites

**Primary Tasks**:
- ✅ Test complete project indexing workflows
- ✅ Validate integration with main bee-hive system
- ✅ Test context optimization for AI agents
- ✅ End-to-end user journey testing
- ✅ Cross-component integration validation

**Success Criteria**:
- Full workflows complete successfully
- Integration with bee-hive main app works
- AI context optimization provides value
- No regressions in existing functionality

**Handoff Prompt**:
```
You are an integration testing specialist focusing on end-to-end workflows.

Your tasks:
1. Test complete project indexing workflows from creation to context optimization
2. Validate integration with the main bee-hive application
3. Test AI context optimization features
4. Run end-to-end user journey scenarios
5. Verify no regressions in existing bee-hive functionality

Use these tools:
- `python scripts/validate_project_index.py --full`
- Manual workflow testing
- Integration test suites
- Bee-hive main application testing

Previous agent status: [Performance validation results]

When complete, document integration results and hand off to Deployment Agent.
```

### Agent 6: Deployment & Configuration (deploy-agent)
**Domain**: Production deployment, configuration management
**Estimated Duration**: 1-2 hours
**Context Files**:
- Deployment documentation
- Configuration examples

**Primary Tasks**:
- ✅ Enable Project Index for bee-hive project itself
- ✅ Create deployment documentation
- ✅ Configure monitoring and alerting
- ✅ Test multi-project installer
- ✅ Create production readiness checklist

**Success Criteria**:
- Project Index enabled for bee-hive project
- Installation script works for other projects
- Monitoring and alerting configured
- Production deployment documented

**Handoff Prompt**:
```
You are a deployment specialist responsible for production readiness.

Your tasks:
1. Enable Project Index for the bee-hive project itself using the installer
2. Test the universal installer on different project types
3. Configure monitoring and alerting
4. Create production deployment documentation
5. Validate backup and recovery procedures

Use these tools:
- `python scripts/install_project_index.py`
- `python scripts/validate_project_index.py`
- Monitoring configuration
- Documentation templates

Previous agent status: [Integration validation results]

When complete, document deployment status and provide final validation report.
```

## 📋 Agent Coordination Protocol

### Context Management Strategy
```bash
# Agent handoff procedure
# 1. Current agent completes work and runs:
/project:sleep --notes="[Agent Name] completed: [summary of work]. Issues: [any issues]. Next: [what next agent should focus on]"

# 2. Next agent starts with:
/project:wake --agent=[agent-name]

# 3. For context preservation during work:
/project:compact --preserve-context
```

### Shared Documentation System
- **Agent Handoff Log**: `.claude/agents/handoff_log.md`
- **Issue Tracking**: `.claude/agents/issues.md`
- **Validation Results**: `.claude/agents/validation_results.json`
- **Deployment Status**: `.claude/agents/deployment_status.md`

### Communication Templates

#### Task Completion Template
```markdown
# Agent Task Completion: [Agent Name]

## ✅ Completed Tasks
- [x] Task 1: [details and results]
- [x] Task 2: [details and results]

## 📊 Validation Results
- Test suite: [X/Y tests passed]
- Performance: [metrics achieved]
- Issues found: [list any issues]

## 🔄 Handoff Information
**Next Agent**: [agent-name]
**Priority Focus**: [main area of concern]
**Context Notes**: [important information for next agent]

## 📁 Key Files Modified
- [file1]: [description of changes]
- [file2]: [description of changes]

## ⚠️ Known Issues
- Issue 1: [description and impact]
- Issue 2: [description and suggested solution]

## 📝 Recommendations
- [Specific recommendations for next agent]
```

#### Issue Escalation Template
```markdown
# Issue Escalation: [Issue Title]

**Discovered by**: [Agent Name]
**Severity**: [Low/Medium/High/Critical]
**Component**: [affected component]

## Description
[Clear description of the issue]

## Impact
[How this affects the system]

## Attempted Solutions
- [Solution 1]: [result]
- [Solution 2]: [result]

## Recommended Action
[What should be done]

## Need Human Input?
[Yes/No - and why]
```

## 🚀 Immediate Action Plan

### Phase 1: Quick Validation (30 minutes)
**Agent**: Current session
**Action**: Run quick validation to check system status

```bash
# Quick system health check
python scripts/validate_project_index.py --quick

# If passing, proceed to agent delegation
# If failing, identify critical issues first
```

### Phase 2: Systematic Validation (4-6 hours)
**Agents**: Deploy 6 specialized agents in sequence
**Action**: Full system validation and optimization

1. **Deploy Database Validator** (1-2 hours)
2. **Deploy API Tester** (2-3 hours) 
3. **Deploy WebSocket Tester** (1-2 hours)
4. **Deploy Performance Tester** (2-3 hours)
5. **Deploy Integration Tester** (2-3 hours)
6. **Deploy Deployment Specialist** (1-2 hours)

### Phase 3: Production Enablement (1 hour)
**Agent**: Deployment specialist
**Action**: Enable for bee-hive and create deployment guide

```bash
# Enable for this project
python scripts/install_project_index.py . --analyze-now --wait

# Test on another project
python scripts/install_project_index.py /path/to/other/project --analyze-now

# Final validation
python scripts/validate_project_index.py --full
```

## 📊 Success Metrics Dashboard

### Technical Metrics
- **Database**: All tables, indexes, and constraints operational
- **API**: All endpoints responding < 200ms with correct data
- **WebSocket**: Real-time events delivered < 50ms
- **Performance**: Projects analyzed within time targets
- **Integration**: Full workflows complete end-to-end

### Business Metrics  
- **AI Context Quality**: Context optimization improves task accuracy
- **Developer Productivity**: Code navigation and discovery improved
- **System Reliability**: 99.9% uptime for indexing operations
- **Multi-project Support**: Installer works across project types

## 🎯 Final Deliverable

**Comprehensive Validation Report** containing:
- System health status across all components
- Performance benchmarks and optimizations
- Integration validation results
- Production deployment guide
- Multi-project installation instructions
- Monitoring and alerting configuration
- Issue tracking and resolution log

This framework ensures systematic, thorough validation while avoiding context rot through specialized agent delegation and clear handoff protocols.