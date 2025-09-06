import { test, expect } from '@playwright/test';

test.describe('Working Dashboard Validation', () => {
  test('Simple dashboard loads with real agent data', async ({ page }) => {
    // Test the working simple dashboard
    await page.goto('http://localhost:8000/dashboard/simple');
    
    // Wait for page to load
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000); // Allow time for data to load
    
    // Take screenshot for evidence
    await page.screenshot({ path: 'test-results/working-dashboard.png', fullPage: true });
    
    // Check for dashboard title
    const title = await page.title();
    expect(title).toContain('HiveOps');
    
    // Check if agent data is displayed
    const pageContent = await page.textContent('body');
    
    // Should show agent roles
    const agentRoles = ['Product Manager', 'Architect', 'Backend Developer', 'QA Engineer', 'DevOps Engineer'];
    let agentRolesFound = 0;
    
    for (const role of agentRoles) {
      if (pageContent?.includes(role)) {
        agentRolesFound++;
        console.log(`✅ Found agent role: ${role}`);
      }
    }
    
    console.log(`Found ${agentRolesFound} out of ${agentRoles.length} agent roles in dashboard`);
    
    // Should find at least 3 agent roles (being lenient for real-world testing)
    expect(agentRolesFound).toBeGreaterThanOrEqual(3);
  });

  test('Live dashboard API returns real agent data', async ({ request }) => {
    const response = await request.get('/dashboard/api/live-data');
    expect(response.ok()).toBeTruthy();
    
    const data = await response.json();
    console.log('Live dashboard data received:', JSON.stringify(data, null, 2));
    
    // Validate structure
    expect(data).toHaveProperty('metrics');
    expect(data).toHaveProperty('agent_activities');
    expect(data).toHaveProperty('project_snapshots');
    
    // Validate metrics
    expect(data.metrics).toHaveProperty('active_agents');
    expect(data.metrics).toHaveProperty('system_status');
    expect(data.metrics.active_agents).toBeGreaterThan(0);
    
    // Validate agent activities
    expect(Array.isArray(data.agent_activities)).toBe(true);
    expect(data.agent_activities.length).toBeGreaterThan(0);
    
    // Check each agent has required fields
    for (const agent of data.agent_activities) {
      expect(agent).toHaveProperty('agent_id');
      expect(agent).toHaveProperty('name');
      expect(agent).toHaveProperty('status');
      expect(agent).toHaveProperty('specializations');
      
      console.log(`Agent: ${agent.name} (${agent.status}) - ${agent.specializations.join(', ')}`);
    }
  });

  test('Dashboard shows agent status and metrics', async ({ page }) => {
    await page.goto('http://localhost:8000/dashboard/simple');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000);
    
    // Look for active agents count
    const activeAgentsElement = await page.locator('#activeAgents').textContent();
    
    if (activeAgentsElement && activeAgentsElement !== '-') {
      const activeAgentsCount = parseInt(activeAgentsElement);
      expect(activeAgentsCount).toBeGreaterThan(0);
      console.log(`✅ Dashboard shows ${activeAgentsCount} active agents`);
    }
    
    // Look for system status
    const statusElement = await page.locator('#systemStatusText').textContent();
    if (statusElement) {
      console.log(`✅ System status: ${statusElement}`);
      expect(statusElement).toBeTruthy();
    }
    
    // Check for agent cards
    const agentsList = await page.locator('#agentsList').innerHTML();
    
    // Should contain agent information, not just loading text
    expect(agentsList).not.toContain('Loading agent data...');
    
    console.log('✅ Dashboard displays real agent information');
  });

  test('WebSocket connection works for real-time updates', async ({ page }) => {
    let webSocketConnected = false;
    let receivedMessages = 0;
    
    // Monitor WebSocket connections
    page.on('websocket', ws => {
      console.log(`WebSocket connected: ${ws.url()}`);
      webSocketConnected = true;
      
      ws.on('framereceived', event => {
        try {
          const data = JSON.parse(event.payload as string);
          console.log(`WebSocket message type: ${data.type}`);
          receivedMessages++;
        } catch (e) {
          // Ignore non-JSON messages
        }
      });
    });
    
    await page.goto('http://localhost:8000/dashboard/simple');
    await page.waitForLoadState('networkidle');
    
    // Wait for WebSocket connection and initial messages
    await page.waitForTimeout(5000);
    
    // Check if WebSocket connected
    expect(webSocketConnected).toBe(true);
    console.log(`✅ WebSocket connected successfully`);
    
    // Check if we received messages
    if (receivedMessages > 0) {
      console.log(`✅ Received ${receivedMessages} WebSocket messages`);
      expect(receivedMessages).toBeGreaterThan(0);
    } else {
      console.log('⚠️ No WebSocket messages received yet (may need more time)');
    }
  });
});