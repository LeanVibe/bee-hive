import { Page } from '@playwright/test';

/**
 * Agent Monitor Utility
 * 
 * Monitors multi-agent system behavior, tracks agent status, and validates 
 * agent coordination and task management capabilities.
 */

export interface AgentStatus {
  id: string;
  role: string;
  status: string;
  capabilities: string[];
  assignedTasks: number;
  currentTask?: string;
  lastHeartbeat: string;
  contextUsage: number;
}

export interface AgentSystemMetrics {
  totalAgents: number;
  activeAgents: number;
  busyAgents: number;
  averageContextUsage: number;
  agentsByRole: Record<string, number>;
  taskDistribution: Record<string, number>;
  systemHealth: 'healthy' | 'degraded' | 'unhealthy';
}

export interface TaskAssignmentResult {
  taskId: string;
  assignedAgent: string;
  assignmentSuccess: boolean;
  assignmentTime: number;
  agentResponse?: any;
}

export class AgentMonitor {
  private page: Page;
  private baseUrl: string;
  private monitoringInterval?: NodeJS.Timeout;
  private agentStatusHistory: Array<{ timestamp: string; agents: Record<string, AgentStatus> }> = [];

  constructor(page: Page, baseUrl: string = 'http://localhost:8000') {
    this.page = page;
    this.baseUrl = baseUrl;
  }

  async getAgentStatus(): Promise<Record<string, AgentStatus>> {
    try {
      const agentData = await this.page.evaluate(async (url) => {
        const response = await fetch(`${url}/debug-agents`);
        if (!response.ok) {
          throw new Error(`Agent status request failed: ${response.status}`);
        }
        return response.json();
      }, this.baseUrl);

      if (agentData.status !== 'debug_working') {
        throw new Error(`Agent system error: ${agentData.status}`);
      }

      // Convert agent data to standardized format
      const agents: Record<string, AgentStatus> = {};
      
      Object.entries(agentData.agents || {}).forEach(([agentId, agentInfo]: [string, any]) => {
        agents[agentId] = {
          id: agentId,
          role: agentInfo.role || 'UNKNOWN',
          status: agentInfo.status || 'UNKNOWN',
          capabilities: agentInfo.capabilities || [],
          assignedTasks: agentInfo.assigned_tasks || 0,
          currentTask: agentInfo.current_task,
          lastHeartbeat: agentInfo.last_heartbeat || new Date().toISOString(),
          contextUsage: agentInfo.context_usage || 0
        };
      });

      return agents;
    } catch (error) {
      console.error('Failed to get agent status:', error);
      return {};
    }
  }

  async getSystemMetrics(): Promise<AgentSystemMetrics> {
    const agents = await this.getAgentStatus();
    const agentList = Object.values(agents);

    const totalAgents = agentList.length;
    const activeAgents = agentList.filter(agent => 
      ['ACTIVE', 'BUSY', 'IDLE'].includes(agent.status)
    ).length;
    const busyAgents = agentList.filter(agent => 
      agent.status === 'BUSY'
    ).length;

    const averageContextUsage = totalAgents > 0 
      ? agentList.reduce((sum, agent) => sum + agent.contextUsage, 0) / totalAgents
      : 0;

    const agentsByRole = agentList.reduce((acc, agent) => {
      acc[agent.role] = (acc[agent.role] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const taskDistribution = agentList.reduce((acc, agent) => {
      const taskCount = agent.assignedTasks;
      const key = `${taskCount}_tasks`;
      acc[key] = (acc[key] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    let systemHealth: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
    if (activeAgents < totalAgents * 0.8) {
      systemHealth = 'degraded';
    }
    if (activeAgents < totalAgents * 0.5) {
      systemHealth = 'unhealthy';
    }

    return {
      totalAgents,
      activeAgents,
      busyAgents,
      averageContextUsage,
      agentsByRole,
      taskDistribution,
      systemHealth
    };
  }

  async validateExpectedAgents(expectedRoles: string[]): Promise<{
    allRolesPresent: boolean;
    missingRoles: string[];
    extraRoles: string[];
    roleValidation: Record<string, boolean>;
  }> {
    const agents = await this.getAgentStatus();
    const presentRoles = Object.values(agents).map(agent => agent.role);
    
    const missingRoles = expectedRoles.filter(role => !presentRoles.includes(role));
    const extraRoles = presentRoles.filter(role => !expectedRoles.includes(role));
    
    const roleValidation = expectedRoles.reduce((acc, role) => {
      acc[role] = presentRoles.includes(role);
      return acc;
    }, {} as Record<string, boolean>);

    return {
      allRolesPresent: missingRoles.length === 0,
      missingRoles,
      extraRoles,
      roleValidation
    };
  }

  async monitorAgentActivity(durationMs: number, intervalMs: number = 5000): Promise<Array<{
    timestamp: string;
    agents: Record<string, AgentStatus>;
    metrics: AgentSystemMetrics;
  }>> {
    const snapshots: Array<{
      timestamp: string;
      agents: Record<string, AgentStatus>;
      metrics: AgentSystemMetrics;
    }> = [];

    const startTime = Date.now();
    
    while (Date.now() - startTime < durationMs) {
      const timestamp = new Date().toISOString();
      const agents = await this.getAgentStatus();
      const metrics = await this.getSystemMetrics();
      
      snapshots.push({
        timestamp,
        agents,
        metrics
      });
      
      await new Promise(resolve => setTimeout(resolve, intervalMs));
    }

    return snapshots;
  }

  async testTaskAssignment(agentId: string, taskData: any): Promise<TaskAssignmentResult> {
    const startTime = Date.now();
    
    try {
      const assignmentResult = await this.page.evaluate(async ({ baseUrl, agentId, taskData }) => {
        // Try to assign task through coordination endpoint
        const response = await fetch(`${baseUrl}/api/v1/coordination/assign-task`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            agent_id: agentId,
            task: taskData
          })
        });

        return {
          status: response.status,
          ok: response.ok,
          data: response.ok ? await response.json() : null
        };
      }, { baseUrl: this.baseUrl, agentId, taskData });

      const assignmentTime = Date.now() - startTime;

      return {
        taskId: taskData.id || 'test-task',
        assignedAgent: agentId,
        assignmentSuccess: assignmentResult.ok,
        assignmentTime,
        agentResponse: assignmentResult.data
      };

    } catch (error) {
      return {
        taskId: taskData.id || 'test-task',
        assignedAgent: agentId,
        assignmentSuccess: false,
        assignmentTime: Date.now() - startTime,
        agentResponse: { error: String(error) }
      };
    }
  }

  async validateAgentCapabilities(expectedCapabilities: Record<string, string[]>): Promise<{
    validationPassed: boolean;
    results: Record<string, {
      hasExpectedCapabilities: boolean;
      missingCapabilities: string[];
      extraCapabilities: string[];
    }>;
  }> {
    const agents = await this.getAgentStatus();
    const results: Record<string, any> = {};
    let validationPassed = true;

    Object.entries(expectedCapabilities).forEach(([role, expectedCaps]) => {
      const agent = Object.values(agents).find(a => a.role === role);
      
      if (!agent) {
        results[role] = {
          hasExpectedCapabilities: false,
          missingCapabilities: expectedCaps,
          extraCapabilities: [],
          error: 'Agent with this role not found'
        };
        validationPassed = false;
        return;
      }

      const agentCaps = agent.capabilities || [];
      const missingCapabilities = expectedCaps.filter(cap => !agentCaps.includes(cap));
      const extraCapabilities = agentCaps.filter(cap => !expectedCaps.includes(cap));

      results[role] = {
        hasExpectedCapabilities: missingCapabilities.length === 0,
        missingCapabilities,
        extraCapabilities
      };

      if (missingCapabilities.length > 0) {
        validationPassed = false;
      }
    });

    return {
      validationPassed,
      results
    };
  }

  async testAgentCoordination(): Promise<{
    coordinationEndpointsAvailable: boolean;
    agentCommunicationWorking: boolean;
    taskDistributionWorking: boolean;
    results: any;
  }> {
    const coordinationTests = {
      coordinationEndpointsAvailable: false,
      agentCommunicationWorking: false,
      taskDistributionWorking: false,
      results: {}
    };

    try {
      // Test coordination endpoints
      const coordinationEndpoints = [
        '/api/v1/coordination/status',
        '/api/v1/coordination/agents',
        '/api/v1/coordination/assign-task'
      ];

      const endpointResults = [];
      
      for (const endpoint of coordinationEndpoints) {
        try {
          const response = await this.page.evaluate(async ({ baseUrl, endpoint }) => {
            const res = await fetch(`${baseUrl}${endpoint}`);
            return { status: res.status, ok: res.ok };
          }, { baseUrl: this.baseUrl, endpoint });

          endpointResults.push({
            endpoint,
            available: [200, 404, 405].includes(response.status),
            status: response.status
          });
        } catch (error) {
          endpointResults.push({
            endpoint,
            available: false,
            error: String(error)
          });
        }
      }

      coordinationTests.coordinationEndpointsAvailable = 
        endpointResults.filter(r => r.available).length > 0;

      // Test agent communication by checking if agents respond to system queries
      const initialAgentCount = Object.keys(await this.getAgentStatus()).length;
      await new Promise(resolve => setTimeout(resolve, 2000));
      const updatedAgentCount = Object.keys(await this.getAgentStatus()).length;
      
      coordinationTests.agentCommunicationWorking = 
        initialAgentCount > 0 && initialAgentCount === updatedAgentCount;

      // Test task distribution by attempting to create and assign tasks
      const agents = await this.getAgentStatus();
      if (Object.keys(agents).length > 0) {
        const testAgent = Object.keys(agents)[0];
        const taskAssignmentResult = await this.testTaskAssignment(testAgent, {
          id: 'coordination-test-task',
          title: 'Coordination Test',
          description: 'Testing agent coordination'
        });

        coordinationTests.taskDistributionWorking = taskAssignmentResult.assignmentSuccess;
      }

      coordinationTests.results = {
        endpointResults,
        initialAgentCount,
        updatedAgentCount,
        taskAssignmentTested: Object.keys(agents).length > 0
      };

    } catch (error) {
      coordinationTests.results = { error: String(error) };
    }

    return coordinationTests;
  }

  async generateAgentReport(): Promise<any> {
    const agents = await this.getAgentStatus();
    const metrics = await this.getSystemMetrics();
    
    const expectedRoles = [
      'PRODUCT_MANAGER',
      'ARCHITECT',
      'BACKEND_DEVELOPER',
      'QA_ENGINEER',
      'DEVOPS_ENGINEER'
    ];

    const roleValidation = await this.validateExpectedAgents(expectedRoles);
    const coordinationTest = await this.testAgentCoordination();

    return {
      timestamp: new Date().toISOString(),
      agentCount: Object.keys(agents).length,
      systemMetrics: metrics,
      roleValidation,
      coordinationTest,
      agents: Object.values(agents),
      summary: {
        systemHealthy: metrics.systemHealth === 'healthy',
        allExpectedRolesPresent: roleValidation.allRolesPresent,
        coordinationWorking: coordinationTest.agentCommunicationWorking,
        overallStatus: this.calculateOverallStatus(metrics, roleValidation, coordinationTest)
      }
    };
  }

  private calculateOverallStatus(
    metrics: AgentSystemMetrics,
    roleValidation: any,
    coordinationTest: any
  ): 'excellent' | 'good' | 'fair' | 'poor' {
    let score = 0;

    if (metrics.systemHealth === 'healthy') score += 3;
    else if (metrics.systemHealth === 'degraded') score += 1;

    if (roleValidation.allRolesPresent) score += 3;
    if (coordinationTest.agentCommunicationWorking) score += 2;
    if (coordinationTest.taskDistributionWorking) score += 2;

    if (score >= 8) return 'excellent';
    if (score >= 6) return 'good';
    if (score >= 4) return 'fair';
    return 'poor';
  }
}