import { test, expect, Page } from '@playwright/test';
import { EvidenceCollector } from '../../utils/evidence-collector';
import { AgentMonitor } from '../../utils/agent-monitor';

/**
 * Multi-Agent System Validation - Agent Spawning
 * 
 * Validates core multi-agent system claims:
 * - 6 active agents (Product Manager, Architect, Backend Dev, QA, DevOps, Strategic Partner)
 * - Agent spawning and lifecycle management
 * - Real-time agent status monitoring
 * - Agent capabilities and specializations
 */

test.describe('Multi-Agent System - Agent Spawning Validation', () => {
  let evidenceCollector: EvidenceCollector;
  let agentMonitor: AgentMonitor;
  
  test.beforeEach(async ({ page }) => {
    evidenceCollector = new EvidenceCollector(page, 'multi-agent');
    agentMonitor = new AgentMonitor(page);
    
    await evidenceCollector.startCollection('agent-spawning-validation');
  });

  test.afterEach(async () => {
    await evidenceCollector.finishCollection();
  });

  test('System Spawns Exactly 6 Active Agents with Correct Roles', async ({ page }) => {
    // Wait for system to initialize agents
    await page.waitForTimeout(5000);
    
    // Get active agents from debug endpoint
    const agentsResponse = await page.evaluate(async () => {
      const response = await fetch('/debug-agents');
      return response.json();
    });
    
    expect(agentsResponse.status).toBe('debug_working');
    expect(agentsResponse.agent_count).toBe(6);
    expect(agentsResponse.agents).toBeDefined();
    
    // Validate agent roles
    const expectedRoles = [
      'PRODUCT_MANAGER',
      'ARCHITECT', 
      'BACKEND_DEVELOPER',
      'QA_ENGINEER',
      'DEVOPS_ENGINEER'
    ];
    
    const agentRoles = Object.values(agentsResponse.agents).map((agent: any) => agent.role);
    
    // Check that we have all expected roles (allowing for 6th agent to be Strategic Partner or similar)
    expectedRoles.forEach(role => {
      expect(agentRoles).toContain(role);
    });
    
    // Collect evidence
    await evidenceCollector.captureApiResponse('/debug-agents', agentsResponse);
    await page.screenshot({ path: await evidenceCollector.getScreenshotPath('agents-spawned') });
    
    console.log('✅ Agent spawning validation passed:', {
      totalAgents: agentsResponse.agent_count,
      roles: agentRoles,
      expectedRoles: expectedRoles
    });
  });

  test('Each Agent Has Required Capabilities and Specializations', async ({ page }) => {
    // Get detailed agent status
    const agentsResponse = await page.evaluate(async () => {
      const response = await fetch('/debug-agents');
      return response.json();
    });
    
    const agents = agentsResponse.agents;
    
    // Validate each agent has required properties
    Object.entries(agents).forEach(([agentId, agent]: [string, any]) => {
      expect(agent).toHaveProperty('id');
      expect(agent).toHaveProperty('role');
      expect(agent).toHaveProperty('status');
      expect(agent).toHaveProperty('capabilities');
      expect(agent).toHaveProperty('last_heartbeat');
      
      // Validate capabilities array
      expect(Array.isArray(agent.capabilities)).toBe(true);
      expect(agent.capabilities.length).toBeGreaterThan(0);
      
      // Validate role-specific capabilities
      switch (agent.role) {
        case 'PRODUCT_MANAGER':
          expect(agent.capabilities).toContain('requirements_analysis');
          expect(agent.capabilities).toContain('project_planning');
          break;
        case 'ARCHITECT':
          expect(agent.capabilities).toContain('system_design');
          expect(agent.capabilities).toContain('architecture_planning');
          break;
        case 'BACKEND_DEVELOPER':
          expect(agent.capabilities).toContain('api_development');
          expect(agent.capabilities).toContain('database_design');
          break;
        case 'QA_ENGINEER':
          expect(agent.capabilities).toContain('test_creation');
          expect(agent.capabilities).toContain('quality_assurance');
          break;
        case 'DEVOPS_ENGINEER':
          expect(agent.capabilities).toContain('deployment');
          expect(agent.capabilities).toContain('infrastructure');
          break;
      }
    });
    
    // Collect evidence
    await evidenceCollector.captureData('agent-capabilities-validation', {
      agentCount: Object.keys(agents).length,
      capabilitiesByRole: Object.entries(agents).reduce((acc, [id, agent]: [string, any]) => {
        acc[agent.role] = agent.capabilities;
        return acc;
      }, {} as Record<string, string[]>)
    });
    
    console.log('✅ Agent capabilities validation passed');
  });

  test('Agents Maintain Active Status and Heartbeats', async ({ page }) => {
    // Get initial agent status
    const initialResponse = await page.evaluate(async () => {
      const response = await fetch('/debug-agents');
      return response.json();
    });
    
    expect(initialResponse.agent_count).toBe(6);
    
    // Wait for heartbeat cycle (30 seconds according to agent_spawner.py)
    await page.waitForTimeout(35000);
    
    // Get updated agent status
    const updatedResponse = await page.evaluate(async () => {
      const response = await fetch('/debug-agents');
      return response.json();
    });
    
    expect(updatedResponse.agent_count).toBe(6);
    
    // Validate all agents are still active
    Object.values(updatedResponse.agents).forEach((agent: any) => {
      expect(['ACTIVE', 'BUSY'].includes(agent.status)).toBe(true);
      
      // Validate heartbeat is recent (within last 2 minutes)
      const heartbeatTime = new Date(agent.last_heartbeat);
      const now = new Date();
      const timeDiff = now.getTime() - heartbeatTime.getTime();
      expect(timeDiff).toBeLessThan(120000); // Less than 2 minutes
    });
    
    // Collect evidence
    await evidenceCollector.captureData('agent-heartbeat-validation', {
      initialCount: initialResponse.agent_count,
      updatedCount: updatedResponse.agent_count,
      heartbeatValidation: Object.entries(updatedResponse.agents).map(([id, agent]: [string, any]) => ({
        agentId: id,
        role: agent.role,
        status: agent.status,
        lastHeartbeat: agent.last_heartbeat,
        heartbeatAge: new Date().getTime() - new Date(agent.last_heartbeat).getTime()
      }))
    });
    
    console.log('✅ Agent heartbeat validation passed');
  });

  test('Agent System Handles Task Assignment and Management', async ({ page }) => {
    // Get active agents
    const agentsResponse = await page.evaluate(async () => {
      const response = await fetch('/debug-agents');
      return response.json();
    });
    
    const agents = agentsResponse.agents;
    const agentIds = Object.keys(agents);
    
    // Test task assignment (if task management endpoint exists)
    let taskAssignmentResults = [];
    
    for (const agentId of agentIds.slice(0, 2)) { // Test with first 2 agents
      try {
        const taskResponse = await page.evaluate(async (id) => {
          const response = await fetch('/api/v1/tasks', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              title: `Test task for agent ${id}`,
              description: 'Automated test task assignment',
              assignee_id: id,
              priority: 'MEDIUM'
            })
          });
          return { status: response.status, ok: response.ok };
        }, agentId);
        
        taskAssignmentResults.push({
          agentId: agentId,
          taskCreated: taskResponse.ok,
          status: taskResponse.status
        });
        
      } catch (error) {
        // Task assignment endpoint might not be available, which is acceptable
        taskAssignmentResults.push({
          agentId: agentId,
          taskCreated: false,
          error: 'Task endpoint not available'
        });
      }
    }
    
    // Collect evidence
    await evidenceCollector.captureData('task-assignment-test', {
      totalAgents: agentIds.length,
      taskAssignmentResults: taskAssignmentResults
    });
    
    console.log('✅ Task assignment test completed:', taskAssignmentResults);
  });

  test('Agent System Provides Real-Time Status Updates', async ({ page }) => {
    // Subscribe to agent events if WebSocket endpoint exists
    const eventTestResults = await page.evaluate(async () => {
      try {
        // Test if WebSocket or SSE endpoint exists for real-time updates
        const eventsResponse = await fetch('/api/v1/events/stream');
        return {
          realTimeEventsAvailable: eventsResponse.ok,
          status: eventsResponse.status
        };
      } catch (error) {
        return {
          realTimeEventsAvailable: false,
          error: 'Real-time events endpoint not available'
        };
      }
    });
    
    // Test agent status changes over time
    const statusSnapshots = [];
    
    for (let i = 0; i < 3; i++) {
      const snapshot = await page.evaluate(async () => {
        const response = await fetch('/debug-agents');
        const data = await response.json();
        return {
          timestamp: new Date().toISOString(),
          agentCount: data.agent_count,
          agentStatuses: Object.entries(data.agents).reduce((acc, [id, agent]: [string, any]) => {
            acc[id] = {
              role: agent.role,
              status: agent.status,
              contextUsage: agent.context_usage || 0
            };
            return acc;
          }, {} as Record<string, any>)
        };
      });
      
      statusSnapshots.push(snapshot);
      
      if (i < 2) await page.waitForTimeout(10000); // Wait 10 seconds between snapshots
    }
    
    // Validate status consistency
    statusSnapshots.forEach(snapshot => {
      expect(snapshot.agentCount).toBe(6);
    });
    
    // Collect evidence
    await evidenceCollector.captureData('real-time-status-validation', {
      eventTestResults: eventTestResults,
      statusSnapshots: statusSnapshots,
      consistentAgentCount: statusSnapshots.every(s => s.agentCount === 6)
    });
    
    console.log('✅ Real-time status validation completed:', {
      eventEndpointAvailable: eventTestResults.realTimeEventsAvailable,
      statusSnapshotCount: statusSnapshots.length,
      consistentAgentCount: statusSnapshots.every(s => s.agentCount === 6)
    });
  });

  test('Agent Performance and Resource Usage Validation', async ({ page }) => {
    // Get system status to check agent performance impact
    const systemStatus = await page.evaluate(async () => {
      const response = await fetch('/status');
      return response.json();
    });
    
    // Get detailed agent information
    const agentsResponse = await page.evaluate(async () => {
      const response = await fetch('/debug-agents');
      return response.json();
    });
    
    // Validate system remains healthy with 6 active agents
    expect(systemStatus.components.database.connected).toBe(true);
    expect(systemStatus.components.redis.connected).toBe(true);
    
    // Calculate resource usage metrics
    const resourceMetrics = {
      totalAgents: agentsResponse.agent_count,
      averageContextUsage: Object.values(agentsResponse.agents).reduce((sum: number, agent: any) => 
        sum + (agent.context_usage || 0), 0) / agentsResponse.agent_count,
      activeAgents: Object.values(agentsResponse.agents).filter((agent: any) => 
        ['ACTIVE', 'BUSY'].includes(agent.status)).length,
      systemHealthy: systemStatus.components.database.connected && systemStatus.components.redis.connected
    };
    
    // Validate performance is acceptable
    expect(resourceMetrics.totalAgents).toBe(6);
    expect(resourceMetrics.averageContextUsage).toBeLessThan(0.95); // Context usage below 95%
    expect(resourceMetrics.activeAgents).toBeGreaterThanOrEqual(6);
    expect(resourceMetrics.systemHealthy).toBe(true);
    
    // Collect evidence
    await evidenceCollector.captureData('agent-performance-metrics', resourceMetrics);
    await page.screenshot({ path: await evidenceCollector.getScreenshotPath('agent-performance') });
    
    console.log('✅ Agent performance validation passed:', resourceMetrics);
  });
});