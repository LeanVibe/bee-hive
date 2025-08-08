# 🎯 Focused Mobile PWA Dashboard Testing Plan

**Based on actual implementation analysis - validated real data flows and production-ready components**

## 📊 Key Findings from Investigation

### ✅ Validated Implementation Status
- **Mobile-PWA Dashboard**: Most developed with 970+ line AgentService and 1000+ line AgentHealthPanel
- **Real Data Integration**: Backend-adapter connects to `/dashboard/api/live-data` endpoint
- **Live Operational Data**: Agent activities, project snapshots, performance metrics, conflict detection
- **Production Architecture**: Lit components, TypeScript, comprehensive error handling

### 🔄 Real Data Flow Validation
```
Backend API (/dashboard/api/live-data) → BackendAdapter → AgentService → AgentHealthPanel → Live UI Updates
```

---

## 🎯 Focused Testing Strategy

### **Tier 1: Real Data Flow Validation (Critical)**

#### Test Case 1.1: Backend Data Connection
- **Goal**: Validate `/dashboard/api/live-data` endpoint returns real operational data
- **Manual Test**: 
  ```bash
  curl http://localhost:8000/dashboard/api/live-data
  # Should return: metrics, agent_activities, project_snapshots, conflict_snapshots
  ```
- **Playwright Test**:
  ```typescript
  test('validates live data endpoint returns real operational data', async ({ request }) => {
    const response = await request.get('/dashboard/api/live-data');
    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    expect(data.metrics.active_agents).toBeGreaterThan(0);
    expect(data.agent_activities).toHaveLength.toBeGreaterThan(0);
  });
  ```

#### Test Case 1.2: Backend Adapter Data Transformation
- **Goal**: Verify BackendAdapter correctly transforms live data into PWA formats
- **Playwright Test**:
  ```typescript
  test('backend adapter transforms live data correctly', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForSelector('[data-testid="agent-card"]');
    
    // Verify agent data appears in UI
    const agentCards = page.locator('[data-testid="agent-card"]');
    await expect(agentCards).toHaveCount.toBeGreaterThan(0);
    await expect(agentCards.first()).toContainText(/Agent|active|busy|idle/);
  });
  ```

#### Test Case 1.3: Real-time Data Updates
- **Goal**: Validate live data polling and UI updates while system is "cooking"
- **Playwright Test**:
  ```typescript
  test('dashboard shows live updates while agents are working', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForSelector('[data-testid="live-status-indicator"]');
    
    // Check initial state
    const initialMetrics = await page.textContent('[data-testid="active-agents-count"]');
    
    // Wait for polling cycle (5 seconds)
    await page.waitForTimeout(6000);
    
    // Verify data refreshed
    const statusIndicator = page.locator('[data-testid="live-status-indicator"]');
    await expect(statusIndicator).toHaveAttribute('data-status', 'live');
  });
  ```

### **Tier 2: Agent Management Validation (High Priority)**

#### Test Case 2.1: Agent Team Activation
- **Goal**: Validate 5-agent team activation produces real agent instances
- **Manual Test**: Click "Activate 5-Agent Team" button, verify agents appear with real statuses
- **Playwright Test**:
  ```typescript
  test('team activation creates working agent instances', async ({ page }) => {
    await page.goto('/dashboard');
    await page.click('[data-testid="activate-team-button"]');
    
    // Wait for activation to complete
    await page.waitForSelector('[data-testid="agent-card"]:nth-child(5)', { timeout: 30000 });
    
    // Verify each agent has real data
    const agents = page.locator('[data-testid="agent-card"]');
    for (let i = 0; i < 5; i++) {
      const agent = agents.nth(i);
      await expect(agent.locator('[data-testid="agent-status"]')).toContainText(/active|busy|idle/);
      await expect(agent.locator('[data-testid="agent-performance"]')).toBeVisible();
    }
  });
  ```

#### Test Case 2.2: Individual Agent Control
- **Goal**: Verify individual agent spawn/deactivate operations work with real backend
- **Playwright Test**:
  ```typescript
  test('individual agent operations connect to real backend', async ({ page, request }) => {
    await page.goto('/dashboard');
    
    // Spawn backend developer agent
    await page.click('[data-testid="spawn-backend-developer"]');
    await page.waitForSelector('[data-testid="backend-developer-agent"]');
    
    // Verify agent appears in backend data
    const response = await request.get('/dashboard/api/live-data');
    const data = await response.json();
    const backendAgents = data.agent_activities.filter(a => a.name.includes('backend'));
    expect(backendAgents.length).toBeGreaterThan(0);
  });
  ```

#### Test Case 2.3: Agent Performance Monitoring
- **Goal**: Validate real performance metrics display and update
- **Playwright Test**:
  ```typescript
  test('agent performance metrics show real data', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForSelector('[data-testid="agent-card"]');
    
    const firstAgent = page.locator('[data-testid="agent-card"]').first();
    await firstAgent.click(); // Open details
    
    // Verify real performance data
    await expect(page.locator('[data-testid="cpu-usage"]')).toContainText(/%/);
    await expect(page.locator('[data-testid="memory-usage"]')).toContainText(/%/);
    await expect(page.locator('[data-testid="tasks-completed"]')).toContainText(/\d+/);
    
    // Check sparkline charts are populated
    await expect(page.locator('[data-testid="performance-chart"]')).toBeVisible();
  });
  ```

### **Tier 3: Task Management & Project Tracking (Medium Priority)**

#### Test Case 3.1: Task Kanban Board Real Data
- **Goal**: Verify Kanban board shows actual tasks from agent activities
- **Playwright Test**:
  ```typescript
  test('kanban board displays real agent tasks', async ({ page }) => {
    await page.goto('/dashboard');
    await page.click('[data-testid="tasks-tab"]');
    
    // Wait for tasks to load from live data
    await page.waitForSelector('[data-testid="kanban-column"]');
    
    const taskCards = page.locator('[data-testid="task-card"]');
    await expect(taskCards).toHaveCount.toBeGreaterThan(0);
    
    // Verify tasks have real agent assignments
    const firstTask = taskCards.first();
    await expect(firstTask.locator('[data-testid="assigned-agent"]')).toBeVisible();
    await expect(firstTask.locator('[data-testid="task-progress"]')).toBeVisible();
  });
  ```

#### Test Case 3.2: Project Snapshots Display
- **Goal**: Validate project snapshots from live data appear correctly
- **Playwright Test**:
  ```typescript
  test('project snapshots show real project data', async ({ page }) => {
    await page.goto('/dashboard');
    await page.click('[data-testid="projects-tab"]');
    
    await page.waitForSelector('[data-testid="project-card"]');
    const projects = page.locator('[data-testid="project-card"]');
    
    // Verify each project has real data
    const firstProject = projects.first();
    await expect(firstProject.locator('[data-testid="progress-percentage"]')).toContainText(/%/);
    await expect(firstProject.locator('[data-testid="participating-agents"]')).toBeVisible();
    await expect(firstProject.locator('[data-testid="completed-tasks"]')).toContainText(/\d+/);
  });
  ```

### **Tier 4: System Health & Monitoring (Medium Priority)**

#### Test Case 4.1: System Health Status
- **Goal**: Verify system health reflects real backend status
- **Playwright Test**:
  ```typescript
  test('system health shows real backend status', async ({ page }) => {
    await page.goto('/dashboard');
    
    const healthStatus = page.locator('[data-testid="system-health-status"]');
    await expect(healthStatus).toContainText(/healthy|degraded|critical/);
    
    // Check component health breakdown
    await expect(page.locator('[data-testid="healthy-components"]')).toContainText(/\d+/);
    await expect(page.locator('[data-testid="last-updated"]')).toBeVisible();
  });
  ```

#### Test Case 4.2: Event Timeline Real Data
- **Goal**: Validate event timeline shows real system events
- **Playwright Test**:
  ```typescript
  test('event timeline displays real system events', async ({ page }) => {
    await page.goto('/dashboard');
    await page.click('[data-testid="events-tab"]');
    
    await page.waitForSelector('[data-testid="event-item"]');
    const events = page.locator('[data-testid="event-item"]');
    await expect(events).toHaveCount.toBeGreaterThan(0);
    
    // Verify events have real data
    const firstEvent = events.first();
    await expect(firstEvent.locator('[data-testid="event-type"]')).toBeVisible();
    await expect(firstEvent.locator('[data-testid="event-timestamp"]')).toBeVisible();
    await expect(firstEvent.locator('[data-testid="event-severity"]')).toContainText(/info|warning|error|critical/);
  });
  ```

---

## 🚀 Implementation Priority

### **Phase 1: Core Data Validation (Execute First)**
1. Bootstrap system setup validation
2. Backend endpoint connectivity tests
3. Real data flow verification tests
4. Basic agent activation tests

### **Phase 2: Advanced Feature Testing**
1. Individual agent management tests
2. Performance monitoring validation
3. Task and project tracking tests
4. System health monitoring tests

### **Phase 3: End-to-End Scenarios**
1. Complete agent team workflow tests
2. Multi-project concurrent management
3. Error recovery and resilience tests
4. Mobile responsive behavior validation

---

## 🔧 Test Infrastructure Requirements

### **Playwright Configuration**
```typescript
// playwright.config.ts focused on mobile-pwa
export default defineConfig({
  testDir: './tests/mobile-pwa',
  projects: [
    {
      name: 'mobile-pwa-real-data',
      testDir: './tests/mobile-pwa',
      use: { 
        baseURL: 'http://localhost:3001', // Mobile PWA dev server
        extraHTTPHeaders: {
          'Accept': 'application/json'
        }
      }
    }
  ]
});
```

### **Test Data Setup**
- Ensure backend is running with live data endpoint
- Verify PostgreSQL and Redis are operational
- Confirm agent orchestrator is active
- Validate WebSocket connections work

### **Validation Checkpoints**
- ✅ Real data endpoint returns operational metrics
- ✅ Agent activities show actual agent work
- ✅ Project snapshots reflect real project states
- ✅ Performance metrics update in real-time
- ✅ UI components render data correctly

---

## 📈 Success Criteria

### **Real Data Integration Validated**
- [ ] Backend `/dashboard/api/live-data` returns operational data
- [ ] BackendAdapter correctly transforms data for PWA services
- [ ] AgentService integrates real agent management
- [ ] UI components display live agent activities and metrics

### **Production Readiness Confirmed**
- [ ] All critical user workflows function with real data
- [ ] Error handling works for network/backend failures
- [ ] Performance remains responsive under load
- [ ] Mobile experience works on actual devices

This focused testing plan prioritizes validation of the actual implemented features in mobile-pwa with emphasis on real data flows and operational capabilities while the system is actively working ("cooking").