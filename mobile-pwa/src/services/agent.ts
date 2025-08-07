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
  private teamActivationInProgress: boolean = false;
  private agentConfigurations: Map<string, any> = new Map();

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

  /**
   * Spawn multiple agents with specified roles (1-click team activation)
   */
  async spawnTeam(roles: AgentRole[]): Promise<{ success: boolean; agents: Record<string, { agent_id: string; role: string }>; message: string }> {
    try {
      const spawnPromises = roles.map(role => this.spawnAgent(role));
      const results = await Promise.all(spawnPromises);
      
      const agents: Record<string, { agent_id: string; role: string }> = {};
      results.forEach((result, index) => {
        if (result.success) {
          agents[result.agent_id] = {
            agent_id: result.agent_id,
            role: roles[index]
          };
        }
      });
      
      const response = {
        success: Object.keys(agents).length > 0,
        agents,
        message: `Successfully spawned ${Object.keys(agents).length} of ${roles.length} agents`
      };
      
      this.emit('teamSpawned', response);
      return response;

    } catch (error) {
      this.emit('teamSpawnFailed', { roles, error });
      throw error;
    }
  }

  /**
   * Deactivate/terminate a specific agent
   */
  async deactivateAgent(agentId: string): Promise<{ success: boolean; message: string }> {
    try {
      const response = await this.delete<{ success: boolean; message: string }>(`/api/agents/${agentId}`);
      
      // Update local state
      this.agents.delete(agentId);
      this.clearCache('agents');
      
      this.emit('agentDeactivated', { agentId, ...response });
      return response;

    } catch (error) {
      this.emit('agentDeactivationFailed', { agentId, error });
      throw error;
    }
  }

  /**
   * Configure a specific agent's settings
   */
  async configureAgent(agentId: string, config: Partial<Agent>): Promise<{ success: boolean; agent: Agent; message: string }> {
    try {
      const response = await this.put<{ success: boolean; agent: Agent; message: string }>(
        `/api/agents/${agentId}/configure`,
        config
      );
      
      // Update local state
      if (response.agent) {
        this.agents.set(agentId, response.agent);
        this.updateAgentPerformanceHistory(response.agent);
      }
      
      this.emit('agentConfigured', { agentId, config, agent: response.agent });
      return response;

    } catch (error) {
      this.emit('agentConfigurationFailed', { agentId, config, error });
      throw error;
    }
  }

  /**
   * Get specific agent configuration and capabilities
   */
  async getAgentConfiguration(agentId: string): Promise<{
    agent: Agent;
    capabilities: string[];
    configuration: Record<string, any>;
    performance_history: AgentPerformanceMetrics[];
  }> {
    const cacheKey = `agent_config_${agentId}`;
    
    try {
      const response = await this.get<{
        agent: Agent;
        capabilities: string[];
        configuration: Record<string, any>;
        performance_history: AgentPerformanceMetrics[];
      }>(`/api/agents/${agentId}/configuration`, {}, cacheKey);
      
      return response;

    } catch (error) {
      this.emit('agentConfigurationLoadFailed', { agentId, error });
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

  public onTeamSpawned(listener: EventListener<{ success: boolean; agents: Record<string, { agent_id: string; role: string }>; message: string }>): Subscription {
    return this.subscribe('teamSpawned', listener);
  }

  public onAgentDeactivated(listener: EventListener<{ agentId: string; success: boolean; message: string }>): Subscription {
    return this.subscribe('agentDeactivated', listener);
  }

  public onAgentConfigured(listener: EventListener<{ agentId: string; config: Partial<Agent>; agent: Agent }>): Subscription {
    return this.subscribe('agentConfigured', listener);
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
      this.updateAgentPerformanceHistory(agent);
    }
  }

  /**
   * Update performance history for a specific agent
   */
  private updateAgentPerformanceHistory(agent: Agent): void {
    if (agent.performance_metrics) {
      const history = this.performanceHistory.get(agent.id) || [];
      history.push(agent.performance_metrics);
      
      // Keep only last 100 entries
      if (history.length > 100) {
        history.shift();
      }
      
      this.performanceHistory.set(agent.id, history);
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

  // ===== ENHANCED AGENT MANAGEMENT =====

  /**
   * One-click team activation with enhanced configuration
   */
  async activateAgentTeam(options: {
    teamSize?: number;
    roles?: AgentRole[];
    autoStartTasks?: boolean;
    teamName?: string;
  } = {}): Promise<{
    success: boolean;
    message: string;
    activatedAgents: Agent[];
    teamComposition: Record<string, string>;
  }> {
    if (this.teamActivationInProgress) {
      throw new Error('Team activation already in progress');
    }

    this.teamActivationInProgress = true;
    
    try {
      const activationResponse = await this.activateAgentSystem({
        teamSize: options.teamSize || 5,
        roles: options.roles || [
          AgentRole.PRODUCT_MANAGER,
          AgentRole.ARCHITECT,
          AgentRole.BACKEND_DEVELOPER,
          AgentRole.FRONTEND_DEVELOPER,
          AgentRole.QA_ENGINEER
        ],
        autoStartTasks: options.autoStartTasks !== false
      });

      const activatedAgents = Object.values(activationResponse.active_agents || {});
      
      // Emit team activation event
      this.emit('teamActivated', {
        teamName: options.teamName || 'Development Team',
        agents: activatedAgents,
        composition: activationResponse.team_composition
      });

      return {
        success: activationResponse.success,
        message: `Successfully activated ${activatedAgents.length}-agent development team`,
        activatedAgents,
        teamComposition: activationResponse.team_composition || {}
      };

    } finally {
      this.teamActivationInProgress = false;
    }
  }

  /**
   * Configure individual agent with enhanced settings
   */
  async configureAgentEnhanced(agentId: string, configuration: {
    specialization?: string[];
    maxConcurrency?: number;
    priority?: 'low' | 'medium' | 'high';
    preferences?: Record<string, any>;
  }): Promise<{ success: boolean; message: string; agent: Agent | null }> {
    try {
      // Store configuration locally
      this.agentConfigurations.set(agentId, configuration);

      // In a real implementation, this would call a backend endpoint
      // For now, we'll update the local agent data
      const agent = this.getAgent(agentId);
      if (agent) {
        // Update agent configuration
        const updatedAgent = {
          ...agent,
          specialization: configuration.specialization,
          maxConcurrency: configuration.maxConcurrency,
          priority: configuration.priority,
          lastConfigured: new Date().toISOString()
        };

        this.agents.set(agentId, updatedAgent);
        this.emit('agentConfigured', { agentId, configuration, agent: updatedAgent });

        return {
          success: true,
          message: `Agent ${agent.name} configured successfully`,
          agent: updatedAgent
        };
      }

      throw new Error(`Agent ${agentId} not found`);

    } catch (error) {
      this.emit('agentConfigurationFailed', { agentId, error });
      throw error;
    }
  }

  /**
   * Get enhanced agent performance metrics with history
   */
  async getAgentPerformanceMetrics(agentId: string, timeframe: '1h' | '24h' | '7d' = '1h'): Promise<{
    current: AgentPerformanceMetrics;
    history: AgentPerformanceMetrics[];
    trends: {
      cpuTrend: 'rising' | 'falling' | 'stable';
      memoryTrend: 'rising' | 'falling' | 'stable';
      performanceTrend: 'improving' | 'declining' | 'stable';
    };
  }> {
    const agent = this.getAgent(agentId);
    if (!agent) {
      throw new Error(`Agent ${agentId} not found`);
    }

    const history = this.performanceHistory.get(agentId) || [];
    const current = history[history.length - 1] || {
      cpuUsage: 0,
      memoryUsage: 0,
      tasksCompleted: 0,
      averageTaskTime: 0,
      successRate: 1.0,
      lastUpdated: new Date().toISOString()
    };

    // Calculate trends (simplified)
    const trends = {
      cpuTrend: 'stable' as const,
      memoryTrend: 'stable' as const,
      performanceTrend: 'stable' as const
    };

    if (history.length >= 2) {
      const previous = history[history.length - 2];
      trends.cpuTrend = current.cpuUsage > previous.cpuUsage + 5 ? 'rising' : 
                      current.cpuUsage < previous.cpuUsage - 5 ? 'falling' : 'stable';
      trends.memoryTrend = current.memoryUsage > previous.memoryUsage + 5 ? 'rising' :
                          current.memoryUsage < previous.memoryUsage - 5 ? 'falling' : 'stable';
      trends.performanceTrend = current.successRate > previous.successRate + 0.05 ? 'improving' :
                               current.successRate < previous.successRate - 0.05 ? 'declining' : 'stable';
    }

    return { current, history, trends };
  }

  /**
   * Bulk agent operations
   */
  async performBulkOperation(agentIds: string[], operation: 'restart' | 'pause' | 'resume' | 'configure', options?: any): Promise<{
    success: boolean;
    results: Record<string, { success: boolean; message: string }>;
    summary: {
      total: number;
      successful: number;
      failed: number;
    };
  }> {
    const results: Record<string, { success: boolean; message: string }> = {};
    let successful = 0;
    let failed = 0;

    for (const agentId of agentIds) {
      try {
        switch (operation) {
          case 'restart':
            // In real implementation, would call restart endpoint
            results[agentId] = { success: true, message: 'Agent restarted successfully' };
            break;
          case 'pause':
            // In real implementation, would call pause endpoint
            results[agentId] = { success: true, message: 'Agent paused successfully' };
            break;
          case 'resume':
            // In real implementation, would call resume endpoint
            results[agentId] = { success: true, message: 'Agent resumed successfully' };
            break;
          case 'configure':
            if (options) {
              await this.configureAgent(agentId, options);
              results[agentId] = { success: true, message: 'Agent configured successfully' };
            } else {
              throw new Error('Configuration options required');
            }
            break;
          default:
            throw new Error(`Unknown operation: ${operation}`);
        }
        successful++;
      } catch (error) {
        results[agentId] = { 
          success: false, 
          message: error instanceof Error ? error.message : 'Operation failed' 
        };
        failed++;
      }
    }

    this.emit('bulkOperationCompleted', {
      operation,
      agentIds,
      results,
      summary: { total: agentIds.length, successful, failed }
    });

    return {
      success: failed === 0,
      results,
      summary: {
        total: agentIds.length,
        successful,
        failed
      }
    };
  }

  /**
   * Get agent team composition with role distribution
   */
  getTeamComposition(): TeamComposition {
    const agents = this.getAgents();
    const composition: TeamComposition = {};

    agents.forEach(agent => {
      const role = agent.role;
      if (!composition[role]) {
        composition[role] = {
          count: 0,
          agents: [],
          capabilities: []
        };
      }

      composition[role].count++;
      composition[role].agents.push(agent);
      
      // Add unique capabilities
      if (agent.capabilities) {
        agent.capabilities.forEach(cap => {
          if (!composition[role].capabilities.includes(cap)) {
            composition[role].capabilities.push(cap);
          }
        });
      }
    });

    return composition;
  }

  /**
   * Check if system is ready for autonomous development
   */
  isSystemReady(): {
    ready: boolean;
    requirements: {
      minAgents: boolean;
      essentialRoles: boolean;
      allAgentsHealthy: boolean;
    };
    recommendations: string[];
  } {
    const agents = this.getAgents();
    const summary = this.getAgentSummary();
    const composition = this.getTeamComposition();

    const essentialRoles = [AgentRole.PRODUCT_MANAGER, AgentRole.BACKEND_DEVELOPER, AgentRole.QA_ENGINEER];
    const hasEssentialRoles = essentialRoles.every(role => composition[role]?.count > 0);
    const minAgents = agents.length >= 3;
    const allAgentsHealthy = summary.error === 0 && summary.offline === 0;

    const recommendations: string[] = [];
    if (!minAgents) recommendations.push('Activate at least 3 agents for basic functionality');
    if (!hasEssentialRoles) recommendations.push('Ensure Product Manager, Backend Developer, and QA Engineer roles are active');
    if (!allAgentsHealthy) recommendations.push(`Address ${summary.error + summary.offline} agent health issues`);

    return {
      ready: minAgents && hasEssentialRoles && allAgentsHealthy,
      requirements: {
        minAgents,
        essentialRoles: hasEssentialRoles,
        allAgentsHealthy
      },
      recommendations
    };
  }

  // ===== CLEANUP =====

  public destroy(): void {
    this.stopMonitoring();
    this.agents.clear();
    this.performanceHistory.clear();
    this.agentConfigurations.clear();
    this.currentStatus = null;
    this.teamActivationInProgress = false;
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