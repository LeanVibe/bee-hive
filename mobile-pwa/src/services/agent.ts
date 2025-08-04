/**
 * Agent Service for LeanVibe Agent Hive
 * 
 * Provides comprehensive agent management including:
 * - Agent activation/deactivation
 * - Real-time agent status monitoring
 * - Performance metrics tracking
 * - Team composition management
 * - Individual agent control
 */

import { BaseService } from './base-service';
import {
  AgentRole,
  AgentStatus
} from '../types/api';
import type {
  Agent,
  AgentSystemStatus,
  AgentActivationRequest,
  AgentActivationResponse,
  AgentPerformanceMetrics,
  ServiceConfig,
  Subscription,
  EventListener
} from '../types/api';

export interface AgentSummary {
  total: number;
  active: number;
  idle: number;
  busy: number;
  error: number;
  offline: number;
  byRole: Record<AgentRole, number>;
}

export interface TeamComposition {
  [role: string]: {
    count: number;
    agents: Agent[];
    capabilities: string[];
  };
}

export interface AgentActivationOptions {
  teamSize?: number;
  roles?: AgentRole[];
  autoStartTasks?: boolean;
}

export class AgentService extends BaseService {
  private pollingStopFn: (() => void) | null = null;
  private currentStatus: AgentSystemStatus | null = null;
  private agents: Map<string, Agent> = new Map();
  private performanceHistory: Map<string, AgentPerformanceMetrics[]> = new Map();

  constructor(config: Partial<ServiceConfig> = {}) {
    super({
      pollingInterval: 5000, // 5 seconds for agent status
      cacheTimeout: 3000, // 3 second cache for agent data
      ...config
    });
  }

  // ===== AGENT ACTIVATION & CONTROL =====

  /**
   * Activate the agent system with specified configuration
   */
  async activateAgentSystem(options: AgentActivationOptions = {}): Promise<AgentActivationResponse> {
    const request: AgentActivationRequest = {
      team_size: options.teamSize || 5,
      roles: options.roles || undefined,
      auto_start_tasks: options.autoStartTasks !== false
    };

    try {
      const response = await this.post<AgentActivationResponse>('/api/agents/activate', request);
      
      // Update local state
      this.updateAgentsFromResponse(response.active_agents);
      
      // Clear cache to force fresh data
      this.clearCache('agents');
      
      this.emit('agentSystemActivated', response);
      return response;

    } catch (error) {
      this.emit('agentActivationFailed', error);
      throw error;
    }
  }

  /**
   * Deactivate the entire agent system
   */
  async deactivateAgentSystem(): Promise<{ success: boolean; message: string }> {
    try {
      const response = await this.delete<{ success: boolean; message: string }>('/api/agents/deactivate');
      
      // Clear local state
      this.agents.clear();
      this.currentStatus = null;
      this.clearCache('agents');
      
      this.emit('agentSystemDeactivated', response);
      return response;

    } catch (error) {
      this.emit('agentDeactivationFailed', error);
      throw error;
    }
  }

  /**
   * Spawn a specific agent with given role
   */
  async spawnAgent(role: AgentRole): Promise<{ success: boolean; agent_id: string; role: string; message: string }> {
    try {
      const response = await this.post<{
        success: boolean;
        agent_id: string;
        role: string;
        message: string;
      }>(`/api/agents/spawn/${role}`);
      
      // Refresh agent status after spawning
      setTimeout(() => this.getAgentSystemStatus(false), 1000);
      
      this.emit('agentSpawned', response);
      return response;

    } catch (error) {
      this.emit('agentSpawnFailed', { role, error });
      throw error;
    }
  }

  // ===== AGENT STATUS & MONITORING =====

  /**
   * Get current agent system status
   */
  async getAgentSystemStatus(fromCache = true): Promise<AgentSystemStatus> {
    const cacheKey = 'agent_system_status';
    
    try {
      const status = await this.get<AgentSystemStatus>(
        '/api/agents/status',
        {},
        fromCache ? cacheKey : undefined
      );

      this.updateCurrentStatus(status);
      return status;

    } catch (error) {
      // Return fallback status if API call fails
      const fallbackStatus: AgentSystemStatus = {
        active: false,
        agent_count: 0,
        spawner_agents: 0,
        orchestrator_agents: 0,
        agents: {},
        orchestrator_agents_detail: {},
        system_ready: false,
        hybrid_integration: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };

      this.updateCurrentStatus(fallbackStatus);
      throw error;
    }
  }

  /**
   * Get all active agents
   */
  getAgents(): Agent[] {
    return Array.from(this.agents.values());
  }

  /**
   * Get specific agent by ID
   */
  getAgent(agentId: string): Agent | null {
    return this.agents.get(agentId) || null;
  }

  /**
   * Get agents by role
   */
  getAgentsByRole(role: AgentRole): Agent[] {
    return this.getAgents().filter(agent => agent.role === role);
  }

  /**
   * Get agents by status
   */
  getAgentsByStatus(status: AgentStatus): Agent[] {
    return this.getAgents().filter(agent => agent.status === status);
  }

  /**
   * Get agent summary statistics
   */
  getAgentSummary(): AgentSummary {
    const agents = this.getAgents();
    
    const summary: AgentSummary = {
      total: agents.length,
      active: 0,
      idle: 0,
      busy: 0,
      error: 0,
      offline: 0,
      byRole: {} as Record<AgentRole, number>
    };

    // Initialize role counts
    Object.values(AgentRole).forEach(role => {
      summary.byRole[role] = 0;
    });

    // Count by status and role
    agents.forEach(agent => {
      switch (agent.status) {
        case AgentStatus.ACTIVE:
          summary.active++;
          break;
        case AgentStatus.IDLE:
          summary.idle++;
          break;
        case AgentStatus.BUSY:
          summary.busy++;
          break;
        case AgentStatus.ERROR:
          summary.error++;
          break;
        case AgentStatus.OFFLINE:
          summary.offline++;
          break;
      }

      summary.byRole[agent.role]++;
    });

    return summary;
  }

  /**
   * Get team composition breakdown
   */
  getTeamComposition(): TeamComposition {
    const agents = this.getAgents();
    const composition: TeamComposition = {};

    agents.forEach(agent => {
      const roleKey = agent.role;
      
      if (!composition[roleKey]) {
        composition[roleKey] = {
          count: 0,
          agents: [],
          capabilities: []
        };
      }

      composition[roleKey].count++;
      composition[roleKey].agents.push(agent);
      
      // Collect unique capabilities
      agent.capabilities.forEach(capability => {
        if (!composition[roleKey].capabilities.includes(capability)) {
          composition[roleKey].capabilities.push(capability);
        }
      });
    });

    return composition;
  }

  /**
   * Get agent capabilities overview
   */
  async getAgentCapabilities(): Promise<{
    total_agents: number;
    roles: Record<string, { count: number; capabilities: string[] }>;
    system_capabilities: string[];
  }> {
    const cacheKey = 'agent_capabilities';
    
    return this.get<{
      total_agents: number;
      roles: Record<string, { count: number; capabilities: string[] }>;
      system_capabilities: string[];
    }>('/api/agents/capabilities', {}, cacheKey);
  }

  // ===== PERFORMANCE MONITORING =====

  /**
   * Get agent performance metrics
   */
  getAgentPerformance(agentId: string): AgentPerformanceMetrics | null {
    const agent = this.getAgent(agentId);
    return agent?.performance_metrics || null;
  }

  /**
   * Get performance history for an agent
   */
  getAgentPerformanceHistory(agentId: string, limit = 50): AgentPerformanceMetrics[] {
    const history = this.performanceHistory.get(agentId) || [];
    return history.slice(-limit);
  }

  /**
   * Get system-wide performance summary
   */
  getSystemPerformanceSummary(): {
    averageSuccessRate: number;
    totalTasksCompleted: number;
    totalTasksFailed: number;
    averageCompletionTime: number;
    systemUtilization: number;
  } {
    const agents = this.getAgents();
    
    if (agents.length === 0) {
      return {
        averageSuccessRate: 0,
        totalTasksCompleted: 0,
        totalTasksFailed: 0,
        averageCompletionTime: 0,
        systemUtilization: 0
      };
    }

    let totalCompleted = 0;
    let totalFailed = 0;
    let totalCompletionTime = 0;
    let totalSuccessRate = 0;
    let totalCpuUsage = 0;

    agents.forEach(agent => {
      const metrics = agent.performance_metrics;
      totalCompleted += metrics.tasks_completed;
      totalFailed += metrics.tasks_failed;
      totalCompletionTime += metrics.average_completion_time;
      totalSuccessRate += metrics.success_rate;
      totalCpuUsage += metrics.cpu_usage;
    });

    return {
      averageSuccessRate: totalSuccessRate / agents.length,
      totalTasksCompleted: totalCompleted,
      totalTasksFailed: totalFailed,
      averageCompletionTime: totalCompletionTime / agents.length,
      systemUtilization: totalCpuUsage / agents.length
    };
  }

  // ===== REAL-TIME MONITORING =====

  /**
   * Start real-time agent monitoring
   */
  startMonitoring(): void {
    if (this.pollingStopFn) {
      this.stopMonitoring();
    }

    this.pollingStopFn = this.startPolling(async () => {
      try {
        await this.getAgentSystemStatus(false);
      } catch (error) {
        // Polling errors are handled by base class
      }
    }, this.config.pollingInterval);

    this.emit('monitoringStarted');
  }

  /**
   * Stop real-time agent monitoring
   */
  stopMonitoring(): void {
    if (this.pollingStopFn) {
      this.pollingStopFn();
      this.pollingStopFn = null;
      this.emit('monitoringStopped');
    }
  }

  /**
   * Check if monitoring is active
   */
  isMonitoring(): boolean {
    return this.pollingStopFn !== null;
  }

  // ===== EVENT SUBSCRIPTIONS =====

  public onAgentSystemActivated(listener: EventListener<AgentActivationResponse>): Subscription {
    return this.subscribe('agentSystemActivated', listener);
  }

  public onAgentSystemDeactivated(listener: EventListener<{ success: boolean; message: string }>): Subscription {
    return this.subscribe('agentSystemDeactivated', listener);
  }

  public onAgentSpawned(listener: EventListener<{ success: boolean; agent_id: string; role: string; message: string }>): Subscription {
    return this.subscribe('agentSpawned', listener);
  }

  public onAgentStatusChanged(listener: EventListener<AgentSystemStatus>): Subscription {
    return this.subscribe('agentStatusChanged', listener);
  }

  public onAgentUpdated(listener: EventListener<Agent>): Subscription {
    return this.subscribe('agentUpdated', listener);
  }

  public onMonitoringStarted(listener: EventListener<void>): Subscription {
    return this.subscribe('monitoringStarted', listener);
  }

  public onMonitoringStopped(listener: EventListener<void>): Subscription {
    return this.subscribe('monitoringStopped', listener);
  }

  // ===== PRIVATE METHODS =====

  private updateCurrentStatus(status: AgentSystemStatus): void {
    const previousStatus = this.currentStatus;
    this.currentStatus = status;

    // Update agents from status
    this.updateAgentsFromStatus(status);

    // Emit status change event
    this.emit('agentStatusChanged', status);

    // Detect and emit individual agent changes
    if (previousStatus) {
      this.detectAgentChanges(previousStatus, status);
    }
  }

  private updateAgentsFromStatus(status: AgentSystemStatus): void {
    // Clear current agents
    this.agents.clear();

    // Update from spawner agents
    if (status.agents) {
      this.updateAgentsFromResponse(status.agents);
    }

    // Update from orchestrator agents
    if (status.orchestrator_agents_detail) {
      this.updateAgentsFromResponse(status.orchestrator_agents_detail);
    }
  }

  private updateAgentsFromResponse(agentsData: Record<string, any>): void {
    for (const [agentId, agentData] of Object.entries(agentsData)) {
      const agent: Agent = {
        id: agentId,
        role: agentData.role || AgentRole.BACKEND_DEVELOPER,
        status: agentData.status || AgentStatus.ACTIVE,
        name: agentData.name || `Agent ${agentId}`,
        capabilities: agentData.capabilities || [],
        created_at: agentData.created_at || new Date().toISOString(),
        updated_at: agentData.updated_at || new Date().toISOString(),
        last_activity: agentData.last_activity || new Date().toISOString(),
        current_task_id: agentData.current_task_id,
        performance_metrics: agentData.performance_metrics || {
          tasks_completed: 0,
          tasks_failed: 0,
          average_completion_time: 0,
          cpu_usage: 0,
          memory_usage: 0,
          success_rate: 0,
          uptime: 0
        },
        error_message: agentData.error_message
      };

      this.agents.set(agentId, agent);

      // Store performance history
      if (agent.performance_metrics) {
        const history = this.performanceHistory.get(agentId) || [];
        history.push(agent.performance_metrics);
        
        // Keep only last 100 entries
        if (history.length > 100) {
          history.shift();
        }
        
        this.performanceHistory.set(agentId, history);
      }
    }
  }

  private detectAgentChanges(previous: AgentSystemStatus, current: AgentSystemStatus): void {
    // Compare agent counts
    if (previous.agent_count !== current.agent_count) {
      this.emit('agentCountChanged', {
        previous: previous.agent_count,
        current: current.agent_count
      });
    }

    // Compare individual agents (simplified - could be more sophisticated)
    const currentAgentIds = new Set(Object.keys(current.agents || {}));
    const previousAgentIds = new Set(Object.keys(previous.agents || {}));

    // Detect new agents
    for (const agentId of currentAgentIds) {
      if (!previousAgentIds.has(agentId)) {
        const agent = this.getAgent(agentId);
        if (agent) {
          this.emit('agentAdded', agent);
        }
      }
    }

    // Detect removed agents
    for (const agentId of previousAgentIds) {
      if (!currentAgentIds.has(agentId)) {
        this.emit('agentRemoved', agentId);
      }
    }
  }

  // ===== CLEANUP =====

  public destroy(): void {
    this.stopMonitoring();
    this.agents.clear();
    this.performanceHistory.clear();
    this.currentStatus = null;
    super.destroy();
  }
}

// Singleton instance
let agentService: AgentService | null = null;

export function getAgentService(config?: Partial<ServiceConfig>): AgentService {
  if (!agentService) {
    agentService = new AgentService(config);
  }
  return agentService;
}

export function resetAgentService(): void {
  if (agentService) {
    agentService.destroy();
    agentService = null;
  }
}