import { test, expect } from '@playwright/test';

test.describe('Multi-Agent System Validation', () => {
  test('Agent system status endpoint returns valid data', async ({ request }) => {
    const agentResponse = await request.get('/api/agents/status');
    expect(agentResponse.ok()).toBeTruthy();
    
    const agentData = await agentResponse.json();
    console.log('Agent system response:', JSON.stringify(agentData, null, 2));
    
    // Validate system is active
    expect(agentData.active).toBe(true);
    expect(agentData.system_ready).toBe(true);
    
    // Validate agent count claims
    expect(agentData.agent_count).toBeGreaterThanOrEqual(5);
    expect(agentData.spawner_agents).toBeGreaterThanOrEqual(5);
    
    // Validate agents object has actual agent data
    expect(agentData.agents).toBeDefined();
    expect(Object.keys(agentData.agents).length).toBeGreaterThanOrEqual(5);
  });

  test('Individual agents have correct roles and capabilities', async ({ request }) => {
    const agentResponse = await request.get('/api/agents/status');
    const agentData = await agentResponse.json();
    
    const agents = agentData.agents;
    const roles = Object.values(agents).map((agent: any) => agent.role);
    
    // Validate required roles are present
    const expectedRoles = ['product_manager', 'architect', 'backend_developer', 'qa_engineer', 'devops_engineer'];
    
    for (const expectedRole of expectedRoles) {
      expect(roles).toContain(expectedRole);
    }
    
    // Validate each agent has required properties
    for (const [agentId, agent] of Object.entries(agents) as [string, any][]) {
      expect(agent).toHaveProperty('id');
      expect(agent).toHaveProperty('role');
      expect(agent).toHaveProperty('status');
      expect(agent).toHaveProperty('capabilities');
      expect(agent).toHaveProperty('last_heartbeat');
      
      // Validate capabilities are not empty
      expect(Array.isArray(agent.capabilities)).toBe(true);
      expect(agent.capabilities.length).toBeGreaterThan(0);
      
      console.log(`Agent ${agent.role}: ${agent.capabilities.join(', ')}`);
    }
  });

  test('Agent activation endpoint works', async ({ request }) => {
    const activateResponse = await request.post('/api/agents/activate', {
      data: { 
        team_size: 5,
        roles: ['product_manager', 'architect', 'backend_developer', 'qa_engineer', 'devops_engineer']
      }
    });
    
    // Should either succeed or indicate already active
    expect([200, 409].includes(activateResponse.status())).toBeTruthy();
    
    if (activateResponse.ok()) {
      const activateData = await activateResponse.json();
      console.log('Activation response:', JSON.stringify(activateData, null, 2));
    }
  });
});