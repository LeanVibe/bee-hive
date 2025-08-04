# LeanVibe Agent Hive 2.0 - Comprehensive Agentic System Testing Plan

## 🎯 Executive Summary

**Objective**: Validate all agentic system features exposed through the dashboard using Playwright MCP, ensuring real data integration and comprehensive coverage across manual testing, end-to-end automation, and unit test validation.

**System Status**: Production-ready platform with 5 specialized agents, real-time coordination, and comprehensive infrastructure.

**Testing Strategy**: Three-tier validation approach with bootstrap preparation for autonomous implementation.

---

## 📊 Agentic Features Inventory (From Documentation Analysis)

### **Core Agent System Features**
1. **Multi-Agent Coordination**
   - 5 specialized agents (Product Manager, Architect, Backend Developer, QA Engineer, DevOps Engineer)
   - Real-time agent-to-agent communication (<2ms latency)
   - Task delegation and coordination workflows
   - Agent performance scoring (target: >0.85)

2. **Agent Lifecycle Management**
   - Team activation (5-agent development teams)
   - Individual agent spawning by role
   - Agent configuration and specialization
   - Start/stop/restart operations
   - Performance monitoring and optimization

3. **Dashboard & Monitoring**
   - Real-time agent status updates (5-second intervals)
   - Performance metrics visualization
   - WebSocket-based live updates
   - Mobile-responsive PWA interface
   - System health indicators

4. **Autonomous Development Workflows**
   - End-to-end development cycles (7 phases)
   - Task assignment and progress tracking
   - Code generation and validation
   - Quality gates and error handling
   - Artifact creation and management

5. **Context & Memory Management**
   - Semantic memory with pgvector
   - Context sharing between agents
   - Context consolidation and compression
   - Cross-agent knowledge management

---

## 🧪 Three-Tier Testing Strategy

### **Tier 1: Manual Testing Procedures**
*Human-executable validation steps for each feature*

#### **1.1 Agent Management Manual Tests**

**Test Case: AM-001 - Agent Team Activation**
```
Prerequisites:
- System running (docker-compose up)
- Dashboard accessible at http://localhost:3000
- API server responsive at http://localhost:8000

Manual Steps:
1. Navigate to Agent Management tab
2. Click "Activate Agent Team" button
3. Verify progress indicator shows team activation
4. Confirm 5 agents appear with distinct roles:
   - Product Manager (capabilities: requirements_analysis, project_planning)
   - Architect (capabilities: system_design, architecture_planning)
   - Backend Developer (capabilities: api_development, database_design)
   - QA Engineer (capabilities: test_creation, quality_assurance)
   - DevOps Engineer (capabilities: deployment, infrastructure)

Expected Results:
- Team activation completes in <10 seconds
- All 5 agents show "active" status
- Performance scores >0.7 for each agent
- System readiness indicator shows "Ready"

Manual Validation:
- Verify agent badges display correct roles
- Check performance metrics update in real-time
- Confirm WebSocket connection shows live updates
```

**Test Case: AM-002 - Individual Agent Control**
```
Prerequisites:
- At least 1 agent active from AM-001

Manual Steps:
1. Select individual agent from list
2. Test control operations:
   - Pause agent → verify status changes to "paused"
   - Resume agent → verify status returns to "active"
   - Configure agent → modify specialization parameters
   - Restart agent → verify agent reinitializes

Expected Results:
- All operations complete in <5 seconds
- Status changes reflect immediately in dashboard
- Agent maintains configuration after restart
- Other agents continue operating during individual operations

Manual Validation:
- Check agent status badges update correctly
- Verify configuration persistence
- Confirm performance metrics track changes
```

#### **1.2 Multi-Agent Coordination Manual Tests**

**Test Case: MAC-001 - Agent Communication Validation**
```
Prerequisites:
- 5-agent team activated from AM-001

Manual Steps:
1. Navigate to Events/Communication tab
2. Assign complex task requiring multiple agents:
   - Task: "Design and implement user authentication system"
3. Monitor agent coordination in real-time:
   - Watch task delegation between agents
   - Observe communication events in timeline
   - Track progress across specialized roles

Expected Results:
- Product Manager initiates requirements analysis
- Architect receives and processes design task
- Backend Developer gets implementation assignment
- QA Engineer receives testing requirements
- DevOps Engineer gets deployment tasks
- All coordination happens within 30 seconds

Manual Validation:
- Verify task appears in each relevant agent's queue
- Check communication events show agent-to-agent messages
- Confirm no agent conflicts or blocking issues
```

#### **1.3 Dashboard & Real-time Features Manual Tests**

**Test Case: DRT-001 - Real-time Updates Validation**
```
Prerequisites:
- Dashboard open in browser
- Agent system operational

Manual Steps:
1. Open dashboard in two browser tabs
2. Trigger agent activity in one tab
3. Verify real-time updates in second tab:
   - Agent status changes
   - Performance metric updates
   - Task progress updates
   - System health changes

Expected Results:
- Updates appear in <2 seconds across tabs
- WebSocket connection remains stable
- No data synchronization issues
- Performance metrics update consistently

Manual Validation:
- Check timestamp accuracy of updates
- Verify no phantom or duplicate updates
- Confirm WebSocket connection stability indicator
```

### **Tier 2: Playwright End-to-End Tests**
*Automated browser-based validation of complete workflows*

#### **2.1 Core Playwright Test Suite**

**File: `tests/e2e-agentic/agent-system-validation.spec.ts`**
```typescript
import { test, expect } from '@playwright/test';

test.describe('LeanVibe Agent Hive - Agentic System Validation', () => {
  test.beforeEach(async ({ page }) => {
    // Ensure system is ready
    await page.goto('http://localhost:3000');
    await expect(page.locator('[data-testid="system-health"]')).toContainText('Healthy');
  });

  test('should activate 5-agent development team successfully', async ({ page }) => {
    // Navigate to agent management
    await page.click('[data-testid="agents-tab"]');
    
    // Activate agent team
    await page.click('[data-testid="activate-team-btn"]');
    
    // Wait for team activation progress
    await expect(page.locator('[data-testid="team-activation-progress"]')).toBeVisible();
    
    // Verify all 5 agents are activated
    await expect(page.locator('[data-testid="agent-product-manager"]')).toContainText('active');
    await expect(page.locator('[data-testid="agent-architect"]')).toContainText('active');
    await expect(page.locator('[data-testid="agent-backend-developer"]')).toContainText('active');
    await expect(page.locator('[data-testid="agent-qa-engineer"]')).toContainText('active');
    await expect(page.locator('[data-testid="agent-devops-engineer"]')).toContainText('active');
    
    // Verify performance scores
    const performanceScores = await page.locator('[data-testid^="agent-performance-"]').allTextContents();
    for (const score of performanceScores) {
      const numericScore = parseFloat(score);
      expect(numericScore).toBeGreaterThan(0.7);
    }
  });

  test('should handle multi-agent task coordination', async ({ page }) => {
    // Prerequisite: agents activated
    await page.click('[data-testid="activate-team-btn"]');
    await page.waitForSelector('[data-testid="agent-product-manager"][data-status="active"]');
    
    // Navigate to task management
    await page.click('[data-testid="tasks-tab"]');
    
    // Create complex task
    await page.click('[data-testid="create-task-btn"]');
    await page.fill('[data-testid="task-title"]', 'Implement user authentication system');
    await page.fill('[data-testid="task-description"]', 'Complete auth system with JWT, password hashing, and API endpoints');
    await page.selectOption('[data-testid="task-priority"]', 'high');
    await page.click('[data-testid="assign-to-team-btn"]');
    
    // Verify task delegation
    await expect(page.locator('[data-testid="task-delegation-progress"]')).toBeVisible();
    
    // Check each agent receives appropriate subtasks
    await expect(page.locator('[data-testid="pm-subtask"]')).toContainText('requirements analysis');
    await expect(page.locator('[data-testid="architect-subtask"]')).toContainText('system design');
    await expect(page.locator('[data-testid="backend-subtask"]')).toContainText('API implementation');
    await expect(page.locator('[data-testid="qa-subtask"]')).toContainText('test creation');
    await expect(page.locator('[data-testid="devops-subtask"]')).toContainText('deployment setup');
  });

  test('should validate real-time WebSocket updates', async ({ page }) => {
    // Open second page for cross-tab validation
    const secondPage = await page.context().newPage();
    await secondPage.goto('http://localhost:3000');
    
    // Trigger agent activity in first page
    await page.click('[data-testid="agents-tab"]');
    await page.click('[data-testid="restart-agent-btn"][data-agent="product-manager"]');
    
    // Verify update appears in second page
    await secondPage.click('[data-testid="agents-tab"]');
    await expect(secondPage.locator('[data-testid="agent-product-manager-status"]')).toContainText('restarting');
    
    // Wait for restart completion
    await expect(secondPage.locator('[data-testid="agent-product-manager-status"]')).toContainText('active', { timeout: 10000 });
    
    await secondPage.close();
  });

  test('should validate performance metrics accuracy', async ({ page }) => {
    // Navigate to performance dashboard
    await page.click('[data-testid="performance-tab"]');
    
    // Verify metrics are present and updating
    await expect(page.locator('[data-testid="cpu-usage-metric"]')).not.toBeEmpty();
    await expect(page.locator('[data-testid="memory-usage-metric"]')).not.toBeEmpty();
    await expect(page.locator('[data-testid="response-time-metric"]')).not.toBeEmpty();
    
    // Check metrics are within expected ranges
    const cpuUsage = await page.locator('[data-testid="cpu-usage-value"]').textContent();
    const memoryUsage = await page.locator('[data-testid="memory-usage-value"]').textContent();
    
    expect(parseFloat(cpuUsage!)).toBeLessThan(80); // <80% CPU
    expect(parseFloat(memoryUsage!)).toBeLessThan(85); // <85% Memory
  });
});
```

**File: `tests/e2e-agentic/autonomous-development-workflow.spec.ts`**
```typescript
import { test, expect } from '@playwright/test';

test.describe('Autonomous Development Workflow Validation', () => {
  test('should execute complete autonomous development cycle', async ({ page }) => {
    await page.goto('http://localhost:3000');
    
    // Initialize autonomous development
    await page.click('[data-testid="autonomous-dev-tab"]');
    await page.click('[data-testid="start-autonomous-cycle-btn"]');
    
    // Define development request
    await page.fill('[data-testid="dev-request-input"]', 'Create a REST API for task management with CRUD operations');
    await page.click('[data-testid="submit-dev-request-btn"]');
    
    // Monitor 7-phase development cycle
    const phases = [
      'requirements-analysis',
      'architecture-design', 
      'implementation',
      'testing',
      'code-review',
      'deployment-prep',
      'validation'
    ];
    
    for (const phase of phases) {
      await expect(page.locator(`[data-testid="phase-${phase}"]`)).toContainText('in-progress', { timeout: 30000 });
      await expect(page.locator(`[data-testid="phase-${phase}"]`)).toContainText('completed', { timeout: 120000 });
    }
    
    // Verify deliverables created
    await expect(page.locator('[data-testid="generated-artifacts"]')).toContainText('API endpoints');
    await expect(page.locator('[data-testid="generated-artifacts"]')).toContainText('test suite');
    await expect(page.locator('[data-testid="generated-artifacts"]')).toContainText('documentation');
  });
});
```

#### **2.2 Real Data Integration Tests**

**File: `tests/e2e-agentic/real-data-validation.spec.ts`**
```typescript
import { test, expect } from '@playwright/test';

test.describe('Real Data Integration Validation', () => {
  test('should validate agents connect to real backend APIs', async ({ page }) => {
    await page.goto('http://localhost:3000');
    
    // Intercept and verify real API calls
    const apiCalls: string[] = [];
    page.on('request', request => {
      if (request.url().includes('/api/')) {
        apiCalls.push(request.url());
      }
    });
    
    // Trigger agent operations
    await page.click('[data-testid="agents-tab"]');
    await page.click('[data-testid="refresh-agents-btn"]');
    
    // Verify real API endpoints called
    expect(apiCalls).toContain(expect.stringContaining('/api/agents/debug'));
    expect(apiCalls).toContain(expect.stringContaining('/api/system/health'));
    
    // Verify responses contain real data
    const response = await page.request.get('http://localhost:8000/api/agents/debug');
    const data = await response.json();
    
    expect(data.agents).toHaveLength(5);
    expect(data.agents[0]).toHaveProperty('capabilities');
    expect(data.agents[0]).toHaveProperty('performance_score');
  });

  test('should validate database integration with real data', async ({ page }) => {
    await page.goto('http://localhost:3000');
    
    // Create task that should persist to database
    await page.click('[data-testid="tasks-tab"]');
    await page.click('[data-testid="create-task-btn"]');
    await page.fill('[data-testid="task-title"]', 'Database Integration Test Task');
    await page.click('[data-testid="save-task-btn"]');
    
    // Refresh page and verify task persists
    await page.reload();
    await page.click('[data-testid="tasks-tab"]');
    await expect(page.locator('[data-testid="task-list"]')).toContainText('Database Integration Test Task');
    
    // Verify via direct API call
    const response = await page.request.get('http://localhost:8000/api/tasks');
    const tasks = await response.json();
    
    expect(tasks.some((task: any) => task.title === 'Database Integration Test Task')).toBeTruthy();
  });
});
```

### **Tier 3: Unit Tests for Logic Validation**
*Fast, isolated tests for core agentic logic*

#### **3.1 Agent Coordination Logic Tests**

**File: `tests/unit/agent-coordination.test.ts`**
```typescript
import { describe, it, expect, beforeEach, jest } from '@jest/globals';
import { AgentCoordinator } from '../../src/services/agent-coordinator';
import { AgentService } from '../../src/services/agent-service';

describe('AgentCoordinator Logic Validation', () => {
  let coordinator: AgentCoordinator;
  let mockAgentService: jest.Mocked<AgentService>;

  beforeEach(() => {
    mockAgentService = {
      getActiveAgents: jest.fn(),
      activateAgentTeam: jest.fn(),
      performBulkOperation: jest.fn(),
      getTeamComposition: jest.fn(),
    } as any;
    
    coordinator = new AgentCoordinator(mockAgentService);
  });

  it('should coordinate task delegation across agent specializations', async () => {
    // Mock 5-agent team
    mockAgentService.getActiveAgents.mockResolvedValue([
      { id: 'pm-1', role: 'product_manager', capabilities: ['requirements_analysis'] },
      { id: 'arch-1', role: 'architect', capabilities: ['system_design'] },
      { id: 'be-1', role: 'backend_developer', capabilities: ['api_development'] },
      { id: 'qa-1', role: 'qa_engineer', capabilities: ['test_creation'] },
      { id: 'devops-1', role: 'devops_engineer', capabilities: ['deployment'] },
    ]);

    const complexTask = {
      title: 'Build user authentication system',
      requirements: ['API endpoints', 'database design', 'tests', 'deployment']
    };

    const delegation = await coordinator.delegateComplexTask(complexTask);

    expect(delegation.subtasks).toHaveLength(5);
    expect(delegation.subtasks[0].assignedAgent).toBe('pm-1');
    expect(delegation.subtasks[1].assignedAgent).toBe('arch-1');
    expect(delegation.subtasks[2].assignedAgent).toBe('be-1');
    expect(delegation.subtasks[3].assignedAgent).toBe('qa-1');
    expect(delegation.subtasks[4].assignedAgent).toBe('devops-1');
  });

  it('should validate agent performance scoring logic', () => {
    const agentMetrics = {
      taskCompletionRate: 0.95,
      averageResponseTime: 1.2, // seconds
      errorRate: 0.02,
      resourceUtilization: 0.65
    };

    const performanceScore = coordinator.calculatePerformanceScore(agentMetrics);

    expect(performanceScore).toBeGreaterThan(0.85); // Target threshold
    expect(performanceScore).toBeLessThanOrEqual(1.0);
  });

  it('should handle agent failure recovery correctly', async () => {
    mockAgentService.getActiveAgents.mockResolvedValue([
      { id: 'agent-1', status: 'failed', lastHeartbeat: Date.now() - 60000 }
    ]);

    const recoveryPlan = await coordinator.generateRecoveryPlan();

    expect(recoveryPlan.actions).toContain('restart_agent');
    expect(recoveryPlan.targetAgent).toBe('agent-1');
    expect(recoveryPlan.estimatedRecoveryTime).toBeLessThan(30000); // <30s
  });
});
```

#### **3.2 Dashboard Logic Tests**

**File: `tests/unit/dashboard-service.test.ts`**
```typescript
import { describe, it, expect, beforeEach, jest } from '@jest/globals';
import { DashboardService } from '../../src/services/dashboard-service';
import { WebSocketService } from '../../src/services/websocket-service';

describe('Dashboard Service Logic Validation', () => {
  let dashboardService: DashboardService;
  let mockWebSocketService: jest.Mocked<WebSocketService>;

  beforeEach(() => {
    mockWebSocketService = {
      broadcast: jest.fn(),
      subscribeToEvents: jest.fn(),
    } as any;
    
    dashboardService = new DashboardService(mockWebSocketService);
  });

  it('should aggregate agent performance metrics correctly', () => {
    const agentMetrics = [
      { agentId: 'agent-1', cpuUsage: 0.3, memoryUsage: 0.5, responseTime: 150 },
      { agentId: 'agent-2', cpuUsage: 0.4, memoryUsage: 0.6, responseTime: 200 },
      { agentId: 'agent-3', cpuUsage: 0.2, memoryUsage: 0.4, responseTime: 100 },
    ];

    const aggregated = dashboardService.aggregateMetrics(agentMetrics);

    expect(aggregated.averageCpuUsage).toBeCloseTo(0.3);
    expect(aggregated.averageMemoryUsage).toBeCloseTo(0.5);
    expect(aggregated.averageResponseTime).toBeCloseTo(150);
    expect(aggregated.healthScore).toBeGreaterThan(0.8);
  });

  it('should validate real-time update frequency limits', () => {
    const updateConfig = {
      maxUpdatesPerSecond: 10,
      batchingThreshold: 5
    };

    dashboardService.configureUpdateLimits(updateConfig);

    // Simulate rapid updates
    const startTime = Date.now();
    for (let i = 0; i < 20; i++) {
      dashboardService.queueUpdate({ type: 'agent_status', data: {} });
    }

    const queuedUpdates = dashboardService.getQueuedUpdates();
    const timeDiff = Date.now() - startTime;

    // Should batch updates to respect rate limits
    expect(queuedUpdates.length).toBeLessThanOrEqual(10);
    expect(timeDiff).toBeLessThan(1000); // Should not introduce unnecessary delays
  });
});
```

---

## 🚀 Bootstrap Requirements for LeanVibe Agentic System

### **Environment Setup**
```bash
# 1. System Dependencies
docker --version  # Required: Docker 20.0+
node --version    # Required: Node.js 18+
python --version  # Required: Python 3.11+

# 2. Service Dependencies
docker-compose up -d postgres redis  # Core infrastructure
alembic upgrade head                  # Database migrations

# 3. API Configuration
export ANTHROPIC_API_KEY="your_key_here"  # Required for AI agents
export DATABASE_URL="postgresql://..."     # Database connection
export REDIS_URL="redis://localhost:6380"  # Message bus

# 4. Application Services
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000  # Backend API
cd mobile-pwa && npm run dev                               # Dashboard
```

### **Validation Scripts**
```bash
# Health Check Script
#!/bin/bash
echo "Validating LeanVibe Agent Hive System..."

# Check API health
curl -f http://localhost:8000/health || exit 1

# Check agent system
curl -f http://localhost:8000/api/agents/debug | jq '.agents | length' | grep -q "5" || exit 1

# Check dashboard
curl -f http://localhost:3000 || exit 1

# Check database connectivity
psql $DATABASE_URL -c "SELECT COUNT(*) FROM agents;" || exit 1

# Check Redis connectivity
redis-cli -p 6380 ping | grep -q "PONG" || exit 1

echo "✅ All systems operational"
```

### **Test Data Seeding**
```bash
# Seed script for comprehensive testing
#!/bin/bash
echo "Seeding test data for agent system validation..."

# Create test agents
curl -X POST http://localhost:8000/api/agents \
  -H "Content-Type: application/json" \
  -d '{"role": "product_manager", "capabilities": ["requirements_analysis", "project_planning"]}'

# Create test tasks
curl -X POST http://localhost:8000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{"title": "Integration Test Task", "priority": "high", "assigned_to": "product_manager"}'

# Trigger agent coordination
curl -X POST http://localhost:8000/api/coordination/activate-team \
  -H "Content-Type: application/json" \
  -d '{"team_size": 5, "project_type": "web_application"}'

echo "✅ Test data seeded successfully"
```

---

## 📋 Implementation Checklist

### **Phase 1: Infrastructure Validation (Week 1)**
- [ ] Verify system meets bootstrap requirements
- [ ] Execute health check and validation scripts
- [ ] Seed test data for comprehensive testing
- [ ] Validate all 5 agents are operational with correct capabilities

### **Phase 2: Manual Testing Execution (Week 1-2)**
- [ ] Execute all manual test cases (AM-001 through DRT-001)
- [ ] Document any deviations from expected results
- [ ] Validate real-time WebSocket functionality
- [ ] Confirm performance metrics meet targets

### **Phase 3: Playwright Test Development (Week 2-3)**
- [ ] Implement core agent system validation tests
- [ ] Develop autonomous development workflow tests
- [ ] Create real data integration validation tests
- [ ] Add performance and stability tests

### **Phase 4: Unit Test Implementation (Week 3)**
- [ ] Implement agent coordination logic tests
- [ ] Develop dashboard service logic tests
- [ ] Create WebSocket and real-time update tests
- [ ] Add error handling and recovery tests

### **Phase 5: Integration & Validation (Week 4)**
- [ ] Execute complete test suite
- [ ] Validate against real production data
- [ ] Performance benchmark validation
- [ ] Generate comprehensive test report

---

## 🎯 Success Criteria

### **Functional Validation**
- [ ] All 5 agents (PM, Architect, Backend, QA, DevOps) operational with >0.85 performance scores
- [ ] Multi-agent coordination completing complex tasks within 30 seconds
- [ ] Real-time dashboard updates within 2 seconds across multiple browser tabs
- [ ] Autonomous development cycles completing all 7 phases successfully

### **Performance Validation**
- [ ] Agent team activation completing in <10 seconds
- [ ] WebSocket connection stability >99.9% uptime during testing
- [ ] CPU usage <80%, Memory usage <85% under full agent load
- [ ] Response times <500ms for all dashboard operations

### **Integration Validation**
- [ ] All tests pass against real backend APIs (not mocks)
- [ ] Database persistence verified for all agent and task operations
- [ ] Redis Streams handling agent communication without message loss
- [ ] Error recovery completing within 30 seconds for agent failures

### **Test Coverage Validation**
- [ ] >95% code coverage for critical agent coordination logic
- [ ] 100% of dashboard features validated through Playwright tests
- [ ] All real-time features tested with multi-client scenarios
- [ ] Complete error handling and edge case coverage

---

**Next Step**: Submit this plan to Gemini CLI for strategic validation and optimization recommendations.