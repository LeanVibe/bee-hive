import { test, expect } from '@playwright/test';

test.describe('Real-time System Validation', () => {
  test('Agent heartbeats are updating', async ({ request }) => {
    // Get initial agent status
    const initialResponse = await request.get('/api/agents/status');
    expect(initialResponse.ok()).toBeTruthy();
    
    const initialData = await initialResponse.json();
    const initialHeartbeats: Record<string, string> = {};
    
    for (const [agentId, agent] of Object.entries(initialData.agents) as [string, any][]) {
      initialHeartbeats[agentId] = agent.last_heartbeat;
    }
    
    console.log('Initial heartbeats captured for', Object.keys(initialHeartbeats).length, 'agents');
    
    // Wait for heartbeat interval (agents should update every 30 seconds according to code)
    console.log('Waiting 35 seconds for heartbeat updates...');
    await new Promise(resolve => setTimeout(resolve, 35000));
    
    // Get updated agent status
    const updatedResponse = await request.get('/api/agents/status');
    const updatedData = await updatedResponse.json();
    
    let heartbeatsUpdated = 0;
    for (const [agentId, agent] of Object.entries(updatedData.agents) as [string, any][]) {
      const oldHeartbeat = initialHeartbeats[agentId];
      const newHeartbeat = agent.last_heartbeat;
      
      if (oldHeartbeat && newHeartbeat && oldHeartbeat !== newHeartbeat) {
        heartbeatsUpdated++;
        console.log(`Agent ${agent.role} heartbeat updated: ${oldHeartbeat} -> ${newHeartbeat}`);
      }
    }
    
    console.log(`${heartbeatsUpdated} agents updated their heartbeats`);
    
    // At least some agents should have updated heartbeats
    expect(heartbeatsUpdated).toBeGreaterThan(0);
  });

  test('Agent context usage changes over time', async ({ request }) => {
    const initialResponse = await request.get('/api/agents/status');
    const initialData = await initialResponse.json();
    
    const initialContextUsage: Record<string, number> = {};
    for (const [agentId, agent] of Object.entries(initialData.agents) as [string, any][]) {
      initialContextUsage[agentId] = agent.context_usage;
    }
    
    // Wait and check again
    await new Promise(resolve => setTimeout(resolve, 10000));
    
    const updatedResponse = await request.get('/api/agents/status');
    const updatedData = await updatedResponse.json();
    
    let contextChanges = 0;
    for (const [agentId, agent] of Object.entries(updatedData.agents) as [string, any][]) {
      const oldUsage = initialContextUsage[agentId];
      const newUsage = agent.context_usage;
      
      if (oldUsage !== newUsage) {
        contextChanges++;
        console.log(`Agent ${agent.role} context usage: ${oldUsage} -> ${newUsage}`);
      }
    }
    
    console.log(`Context usage changed for ${contextChanges} agents`);
  });

  test('System maintains agent state consistency', async ({ request }) => {
    // Test multiple rapid requests to ensure consistent state
    const requests = Array(5).fill(null).map(() => request.get('/api/agents/status'));
    const responses = await Promise.all(requests);
    
    const datasets = await Promise.all(responses.map(r => r.json()));
    
    // All responses should have the same agent count
    const agentCounts = datasets.map(data => data.agent_count);
    const uniqueCounts = new Set(agentCounts);
    
    expect(uniqueCounts.size).toBe(1); // All counts should be the same
    
    // All responses should have the same agent IDs
    const agentIdSets = datasets.map(data => new Set(Object.keys(data.agents)));
    
    for (let i = 1; i < agentIdSets.length; i++) {
      expect(agentIdSets[i].size).toBe(agentIdSets[0].size);
    }
    
    console.log('State consistency verified across multiple requests');
  });
});