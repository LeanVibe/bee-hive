/**
 * Agent Monitoring Service for Real-Time Dashboard
 * 
 * Provides WebSocket-based real-time monitoring of agent lifecycle events,
 * performance metrics, and system health for Vertical Slice 1.2.
 */

import { ref, reactive, computed } from 'vue'
import { apiClient } from './api'
import { unifiedWebSocketManager } from './unifiedWebSocketManager'

export interface AgentStatus {
  id: string
  name: string
  status: 'active' | 'idle' | 'busy' | 'offline' | 'error'
  sessionId?: string
  tasksCompleted: number
  totalTasks: number
  startTime: string
  memoryUsage: number
  performance: number
  currentActivity?: string
  lastActivity?: string
  agentType?: string
  capabilities?: string[]
  persona?: string
}

export interface AgentLifecycleEvent {
  event_type: string
  agent_id: string
  timestamp: string
  payload: {
    name?: string
    type?: string
    role?: string
    task_id?: string
    task_title?: string
    success?: boolean
    execution_time_ms?: number
    [key: string]: any
  }
}

export interface PerformanceMetrics {
  cpu_usage_percent: number
  memory_usage_mb: number
  memory_usage_percent: number
  disk_usage_percent: number
  active_connections: number
  timestamp: string
}

export interface SystemStats {
  active_agents: number
  total_agents_registered: number
  tasks_in_progress: number
  tasks_completed_today: number
  average_task_completion_time: number
  system_load: number
}

class AgentMonitoringService {
  // State management
  private state = reactive({
    agents: new Map<string, AgentStatus>(),
    recentEvents: [] as AgentLifecycleEvent[],
    performanceMetrics: null as PerformanceMetrics | null,
    systemStats: null as SystemStats | null,
    isConnected: false,
    lastUpdate: null as Date | null,
    connectionStatus: 'disconnected' as 'connected' | 'connecting' | 'disconnected' | 'error'
  })

  // WebSocket connection management
  private wsEndpoint = 'monitoring-agents'
  private perfWsEndpoint = 'monitoring-performance'
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 2000

  // Event handlers
  private eventHandlers = new Map<string, Array<(data: any) => void>>()

  constructor() {
    this.setupWebSocketEndpoints()
  }

  // Public reactive state
  public readonly agents = computed(() => Array.from(this.state.agents.values()))
  public readonly activeAgents = computed(() => 
    Array.from(this.state.agents.values()).filter(a => 
      a.status === 'active' || a.status === 'busy'
    )
  )
  public readonly recentEvents = computed(() => this.state.recentEvents)
  public readonly performanceMetrics = computed(() => this.state.performanceMetrics)
  public readonly systemStats = computed(() => this.state.systemStats)
  public readonly isConnected = computed(() => this.state.isConnected)
  public readonly connectionStatus = computed(() => this.state.connectionStatus)

  /**
   * Initialize the monitoring service
   */
  public async initialize(): Promise<void> {
    try {
      this.state.connectionStatus = 'connecting'

      // Register WebSocket endpoints
      unifiedWebSocketManager.registerEndpoint({
        id: this.wsEndpoint,
        url: this.buildWebSocketUrl('/ws/monitoring/agents'),
        component: 'GRAPH' as any,
        priority: 'high',
        autoReconnect: true,
        maxReconnectAttempts: this.maxReconnectAttempts,
        reconnectDelay: this.reconnectDelay
      })

      unifiedWebSocketManager.registerEndpoint({
        id: this.perfWsEndpoint,
        url: this.buildWebSocketUrl('/ws/monitoring/performance'),
        component: 'METRICS' as any,
        priority: 'medium',
        autoReconnect: true,
        maxReconnectAttempts: this.maxReconnectAttempts,
        reconnectDelay: this.reconnectDelay
      })

      // Connect to WebSocket endpoints
      await Promise.all([
        unifiedWebSocketManager.connect(this.wsEndpoint),
        unifiedWebSocketManager.connect(this.perfWsEndpoint)
      ])

      this.state.connectionStatus = 'connected'
      this.state.isConnected = true
      this.state.lastUpdate = new Date()

      console.log('🔌 Agent monitoring service initialized successfully')

    } catch (error) {
      console.error('❌ Failed to initialize agent monitoring service:', error)
      this.state.connectionStatus = 'error'
      this.scheduleReconnection()
    }
  }

  /**
   * Disconnect from monitoring services
   */
  public disconnect(): void {
    unifiedWebSocketManager.disconnect(this.wsEndpoint)
    unifiedWebSocketManager.disconnect(this.perfWsEndpoint)
    
    this.state.isConnected = false
    this.state.connectionStatus = 'disconnected'
    
    console.log('🔌 Agent monitoring service disconnected')
  }

  /**
   * Get agent details by ID
   */
  public async getAgentDetails(agentId: string): Promise<AgentStatus | null> {
    try {
      // Try to get from local state first
      const localAgent = this.state.agents.get(agentId)
      if (localAgent) {
        return localAgent
      }

      // Request details via WebSocket
      unifiedWebSocketManager.send(this.wsEndpoint, {
        type: 'get_agent_details',
        data: { agent_id: agentId },
        timestamp: new Date().toISOString()
      })

      // Wait for response (would be better with promises/callbacks)
      return null

    } catch (error) {
      console.error('❌ Failed to get agent details:', error)
      return null
    }
  }

  /**
   * Get performance history
   */
  public async getPerformanceHistory(duration = '1h'): Promise<any> {
    try {
      unifiedWebSocketManager.send(this.perfWsEndpoint, {
        type: 'get_performance_history',
        data: { duration },
        timestamp: new Date().toISOString()
      })

      return null // Would return promise in real implementation
    } catch (error) {
      console.error('❌ Failed to get performance history:', error)
      return null
    }
  }

  /**
   * Subscribe to specific event types
   */
  public onEvent<T = any>(eventType: string, handler: (data: T) => void): () => void {
    if (!this.eventHandlers.has(eventType)) {
      this.eventHandlers.set(eventType, [])
    }

    const handlers = this.eventHandlers.get(eventType)!
    handlers.push(handler)

    // Return unsubscribe function
    return () => {
      const index = handlers.indexOf(handler)
      if (index > -1) {
        handlers.splice(index, 1)
      }
    }
  }

  /**
   * Get current system statistics
   */
  public getSystemStatistics() {
    return {
      totalAgents: this.state.agents.size,
      activeAgents: this.activeAgents.value.length,
      idleAgents: Array.from(this.state.agents.values()).filter(a => a.status === 'idle').length,
      busyAgents: Array.from(this.state.agents.values()).filter(a => a.status === 'busy').length,
      errorAgents: Array.from(this.state.agents.values()).filter(a => a.status === 'error').length,
      totalTasksCompleted: Array.from(this.state.agents.values()).reduce((sum, a) => sum + a.tasksCompleted, 0),
      averagePerformance: this.calculateAveragePerformance(),
      totalMemoryUsage: Array.from(this.state.agents.values()).reduce((sum, a) => sum + a.memoryUsage, 0)
    }
  }

  // Private methods

  private setupWebSocketEndpoints(): void {
    // Handle agent lifecycle events
    unifiedWebSocketManager.onMessage('agent_lifecycle_event', (message, endpointId) => {
      this.handleAgentLifecycleEvent(message.data)
    })

    unifiedWebSocketManager.onMessage('agent_stats', (message, endpointId) => {
      this.handleSystemStats(message.data)
    })

    unifiedWebSocketManager.onMessage('agent_details', (message, endpointId) => {
      this.handleAgentDetails(message.data)
    })

    // Handle performance metrics
    unifiedWebSocketManager.onMessage('performance_metrics', (message, endpointId) => {
      this.handlePerformanceMetrics(message.data)
    })

    unifiedWebSocketManager.onMessage('performance_snapshot', (message, endpointId) => {
      this.handlePerformanceMetrics(message.data)
    })

    unifiedWebSocketManager.onMessage('performance_history', (message, endpointId) => {
      this.emitEvent('performance_history', message.data)
    })

    // Handle connection events
    unifiedWebSocketManager.onDashboardEvent('connection_status_changed', (event) => {
      this.handleConnectionStatusChange(event)
    })
  }

  private handleAgentLifecycleEvent(eventData: AgentLifecycleEvent): void {
    // Add to recent events
    this.state.recentEvents.unshift(eventData)
    if (this.state.recentEvents.length > 100) {
      this.state.recentEvents.pop()
    }

    // Update agent status based on event type
    const agentId = eventData.agent_id
    let agent = this.state.agents.get(agentId)

    switch (eventData.event_type) {
      case 'agent_registered':
        agent = {
          id: agentId,
          name: eventData.payload.name || `Agent-${agentId.substring(0, 8)}`,
          status: 'active',
          tasksCompleted: 0,
          totalTasks: 0,
          startTime: eventData.timestamp,
          memoryUsage: 50, // Default values
          performance: 85,
          agentType: eventData.payload.type,
          capabilities: []
        }
        this.state.agents.set(agentId, agent)
        break

      case 'agent_deregistered':
        if (agent) {
          agent.status = 'offline'
          agent.lastActivity = eventData.timestamp
        }
        break

      case 'task_assigned':
        if (agent) {
          agent.status = 'busy'
          agent.currentActivity = `Working on: ${eventData.payload.task_title || 'Task'}`
          agent.totalTasks++
        }
        break

      case 'task_completed':
        if (agent) {
          agent.status = 'active'
          agent.tasksCompleted++
          agent.currentActivity = undefined
          // Update performance based on execution time
          const executionTime = eventData.payload.execution_time_ms || 0
          if (executionTime > 0) {
            const performanceScore = Math.max(10, Math.min(100, 100 - (executionTime / 100)))
            agent.performance = Math.round((agent.performance + performanceScore) / 2)
          }
        }
        break

      case 'task_failed':
        if (agent) {
          agent.status = 'error'
          agent.currentActivity = 'Error occurred'
          agent.performance = Math.max(0, agent.performance - 5)
        }
        break

      case 'agent_heartbeat':
        if (agent) {
          agent.lastActivity = eventData.timestamp
          if (agent.status === 'offline') {
            agent.status = 'active'
          }
        }
        break
    }

    this.state.lastUpdate = new Date()
    this.emitEvent('agent_lifecycle_event', eventData)
  }

  private handleSystemStats(statsData: SystemStats): void {
    this.state.systemStats = statsData
    this.emitEvent('system_stats', statsData)
  }

  private handleAgentDetails(agentData: any): void {
    if (agentData.error) {
      console.error('❌ Agent details error:', agentData.error)
      return
    }

    const agent: AgentStatus = {
      id: agentData.id,
      name: agentData.name,
      status: this.mapAgentStatus(agentData.status),
      tasksCompleted: agentData.total_tasks_completed || 0,
      totalTasks: agentData.total_tasks_completed || 0,
      startTime: agentData.created_at || new Date().toISOString(),
      memoryUsage: agentData.current_memory_usage || 0,
      performance: Math.round((agentData.performance_score || 0) * 100),
      lastActivity: agentData.last_activity,
      agentType: agentData.agent_type
    }

    this.state.agents.set(agent.id, agent)
    this.emitEvent('agent_details', agent)
  }

  private handlePerformanceMetrics(metricsData: PerformanceMetrics): void {
    if (metricsData.error) {
      console.error('❌ Performance metrics error:', metricsData.error)
      return
    }

    this.state.performanceMetrics = metricsData
    this.emitEvent('performance_metrics', metricsData)
  }

  private handleConnectionStatusChange(event: any): void {
    const isConnected = event.data?.websocket || false
    this.state.isConnected = isConnected
    this.state.connectionStatus = isConnected ? 'connected' : 'disconnected'

    if (!isConnected) {
      this.scheduleReconnection()
    }
  }

  private mapAgentStatus(status: string): AgentStatus['status'] {
    const statusMap: Record<string, AgentStatus['status']> = {
      'ACTIVE': 'active',
      'IDLE': 'idle', 
      'BUSY': 'busy',
      'INACTIVE': 'offline',
      'ERROR': 'error',
      'SHUTTING_DOWN': 'offline'
    }
    return statusMap[status] || 'offline'
  }

  private calculateAveragePerformance(): number {
    const agents = Array.from(this.state.agents.values())
    if (agents.length === 0) return 0

    const totalPerformance = agents.reduce((sum, agent) => sum + agent.performance, 0)
    return Math.round(totalPerformance / agents.length)
  }

  private buildWebSocketUrl(path: string): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    return `${protocol}//${host}${path}`
  }

  private emitEvent(eventType: string, data: any): void {
    const handlers = this.eventHandlers.get(eventType) || []
    handlers.forEach(handler => {
      try {
        handler(data)
      } catch (error) {
        console.error(`❌ Error in event handler for ${eventType}:`, error)
      }
    })
  }

  private scheduleReconnection(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('❌ Max reconnection attempts reached')
      return
    }

    this.reconnectAttempts++
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)

    console.log(`🔄 Scheduling reconnection attempt ${this.reconnectAttempts} in ${delay}ms`)

    setTimeout(async () => {
      try {
        await this.initialize()
        this.reconnectAttempts = 0 // Reset on successful connection
      } catch (error) {
        console.error('❌ Reconnection failed:', error)
        this.scheduleReconnection()
      }
    }, delay)
  }

  /**
   * Cleanup resources
   */
  public destroy(): void {
    this.disconnect()
    this.eventHandlers.clear()
    this.state.agents.clear()
    this.state.recentEvents.length = 0
  }
}

// Create singleton instance
export const agentMonitoringService = new AgentMonitoringService()

// Vue composable for easy integration
export function useAgentMonitoring() {
  return {
    // State
    agents: agentMonitoringService.agents,
    activeAgents: agentMonitoringService.activeAgents,
    recentEvents: agentMonitoringService.recentEvents,
    performanceMetrics: agentMonitoringService.performanceMetrics,
    systemStats: agentMonitoringService.systemStats,
    isConnected: agentMonitoringService.isConnected,
    connectionStatus: agentMonitoringService.connectionStatus,

    // Methods
    initialize: agentMonitoringService.initialize.bind(agentMonitoringService),
    disconnect: agentMonitoringService.disconnect.bind(agentMonitoringService),
    getAgentDetails: agentMonitoringService.getAgentDetails.bind(agentMonitoringService),
    getPerformanceHistory: agentMonitoringService.getPerformanceHistory.bind(agentMonitoringService),
    onEvent: agentMonitoringService.onEvent.bind(agentMonitoringService),
    getSystemStatistics: agentMonitoringService.getSystemStatistics.bind(agentMonitoringService)
  }
}

export default agentMonitoringService