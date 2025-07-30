/**
 * Enhanced WebSocket Service for Real-time Team Coordination
 * 
 * Provides real-time updates for task distribution, agent status changes,
 * and performance metrics with automatic reconnection and error handling.
 */

import { ref, reactive, computed } from 'vue'
import type {
  WebSocketMessage,
  GraphUpdateMessage,
  TranscriptUpdateMessage,
  AnalysisUpdateMessage,
  WorkflowExecutionUpdateMessage,
  NodeStatusUpdateMessage,
  AgentAssignmentUpdateMessage,
  SystemMetricsUpdateMessage,
  ResourceUsageUpdateMessage
} from '@/types/coordination'

export interface CoordinationWebSocketConfig {
  baseUrl: string
  reconnectInterval: number
  maxReconnectAttempts: number
  heartbeatInterval: number
  enableDebugLogging: boolean
}

export interface TaskAssignmentUpdate {
  type: 'task_assignment'
  data: {
    task_id: string
    agent_id: string
    agent_name: string
    task_title: string
    priority: string
    assigned_at: string
    confidence_score: number
    estimated_completion?: string
  }
}

export interface TaskStatusUpdate {
  type: 'task_status'
  data: {
    task_id: string
    old_status: string
    new_status: string
    agent_id: string
    updated_at: string
    progress?: number
    notes?: string
  }
}

export interface AgentWorkloadUpdate {
  type: 'agent_workload'
  data: {
    agent_id: string
    agent_name: string
    old_workload: number
    new_workload: number
    active_tasks: number
    available_capacity: number
    updated_at: string
  }
}

export interface PerformanceMetricsUpdate {
  type: 'performance_metrics'
  data: {
    system_efficiency: number
    average_utilization: number
    task_throughput: number
    error_rate: number
    timestamp: string
    agent_metrics?: Record<string, any>
  }
}

export interface BottleneckAlert {
  type: 'bottleneck_detected'
  data: {
    id: string
    title: string
    description: string
    severity: 'low' | 'medium' | 'high' | 'critical'
    impact: number
    category: string
    affected_agents: string[]
    detected_at: string
    auto_resolve?: boolean
  }
}

export type CoordinationMessage = 
  | TaskAssignmentUpdate
  | TaskStatusUpdate
  | AgentWorkloadUpdate
  | PerformanceMetricsUpdate
  | BottleneckAlert
  | GraphUpdateMessage
  | TranscriptUpdateMessage
  | AnalysisUpdateMessage
  | WorkflowExecutionUpdateMessage
  | NodeStatusUpdateMessage
  | AgentAssignmentUpdateMessage
  | SystemMetricsUpdateMessage
  | ResourceUsageUpdateMessage

export interface WebSocketConnectionState {
  isConnected: boolean
  isConnecting: boolean
  connectionId: string | null
  lastConnectedAt: Date | null
  lastDisconnectedAt: Date | null
  reconnectAttempts: number
  latency: number
  error: string | null
}

export interface MessageHandler {
  id: string
  type: string | string[]
  handler: (message: CoordinationMessage) => void
  priority: number
}

class CoordinationWebSocketService {
  private ws: WebSocket | null = null
  private config: CoordinationWebSocketConfig
  private messageHandlers = new Map<string, MessageHandler>()
  private subscriptions = new Set<string>()
  private reconnectTimer: number | null = null
  private heartbeatTimer: number | null = null
  private lastHeartbeat: number = 0
  private messageQueue: CoordinationMessage[] = []
  private isDestroyed = false

  // Reactive state
  public readonly state = reactive<WebSocketConnectionState>({
    isConnected: false,
    isConnecting: false,
    connectionId: null,
    lastConnectedAt: null,
    lastDisconnectedAt: null,
    reconnectAttempts: 0,
    latency: 0,
    error: null
  })

  // Message statistics
  public readonly stats = reactive({
    messagesReceived: 0,
    messagesSent: 0,
    reconnections: 0,
    averageLatency: 0,
    uptime: 0
  })

  constructor(config: Partial<CoordinationWebSocketConfig> = {}) {
    this.config = {
      baseUrl: this.getWebSocketBaseUrl(),
      reconnectInterval: 3000,
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000,
      enableDebugLogging: true,
      ...config
    }

    this.setupDefaultHandlers()
  }

  /**
   * Connect to the coordination WebSocket
   */
  public async connect(connectionId?: string): Promise<void> {
    if (this.state.isConnected || this.state.isConnecting || this.isDestroyed) {
      return
    }

    this.state.isConnecting = true
    this.state.error = null

    try {
      const wsUrl = this.buildWebSocketUrl(connectionId)
      this.ws = new WebSocket(wsUrl)

      this.setupWebSocketEventHandlers()

      // Wait for connection to establish
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('WebSocket connection timeout'))
        }, 10000)

        this.ws!.onopen = () => {
          clearTimeout(timeout)
          resolve()
        }

        this.ws!.onerror = () => {
          clearTimeout(timeout)
          reject(new Error('WebSocket connection failed'))
        }
      })
    } catch (error) {
      this.state.isConnecting = false
      this.state.error = error instanceof Error ? error.message : 'Connection failed'
      this.log('Connection failed:', error)
      throw error
    }
  }

  /**
   * Disconnect from the WebSocket
   */
  public disconnect(): void {
    this.isDestroyed = true
    this.clearTimers()

    if (this.ws) {
      this.ws.close(1000, 'Manual disconnect')
      this.ws = null
    }

    this.state.isConnected = false
    this.state.isConnecting = false
    this.state.connectionId = null
    this.state.lastDisconnectedAt = new Date()
  }

  /**
   * Subscribe to specific message types or agent updates
   */
  public subscribe(subscription: string): void {
    this.subscriptions.add(subscription)
    
    if (this.state.isConnected) {
      this.sendMessage({
        type: 'subscribe',
        subscription,
        timestamp: new Date().toISOString()
      })
    }
  }

  /**
   * Unsubscribe from message types or agent updates
   */
  public unsubscribe(subscription: string): void {
    this.subscriptions.delete(subscription)
    
    if (this.state.isConnected) {
      this.sendMessage({
        type: 'unsubscribe',
        subscription,
        timestamp: new Date().toISOString()
      })
    }
  }

  /**
   * Register a message handler
   */
  public onMessage(
    id: string,
    types: string | string[],
    handler: (message: CoordinationMessage) => void,
    priority: number = 0
  ): void {
    this.messageHandlers.set(id, {
      id,
      type: Array.isArray(types) ? types : [types],
      handler,
      priority
    })
  }

  /**
   * Remove a message handler
   */
  public offMessage(id: string): void {
    this.messageHandlers.delete(id)
  }

  /**
   * Send a message through the WebSocket
   */
  public sendMessage(message: any): void {
    if (!this.state.isConnected || !this.ws) {
      this.log('Cannot send message: WebSocket not connected')
      this.messageQueue.push(message)
      return
    }

    try {
      const messageStr = JSON.stringify({
        ...message,
        timestamp: message.timestamp || new Date().toISOString(),
        connection_id: this.state.connectionId
      })

      this.ws.send(messageStr)
      this.stats.messagesSent++
      this.log('Sent message:', message.type)
    } catch (error) {
      this.log('Failed to send message:', error)
    }
  }

  /**
   * Request current task assignments for an agent
   */
  public requestAgentTasks(agentId: string): void {
    this.sendMessage({
      type: 'request_agent_tasks',
      agent_id: agentId
    })
  }

  /**
   * Request current system metrics
   */
  public requestSystemMetrics(): void {
    this.sendMessage({
      type: 'request_system_metrics'
    })
  }

  /**
   * Subscribe to task distribution events
   */
  public subscribeToTaskDistribution(): void {
    this.subscribe('task_assignments')
    this.subscribe('task_status_updates')
  }

  /**
   * Subscribe to agent performance updates
   */
  public subscribeToAgentPerformance(): void {
    this.subscribe('agent_workload')
    this.subscribe('agent_performance')
    this.subscribe('agent_status')
  }

  /**
   * Subscribe to system-wide metrics
   */
  public subscribeToSystemMetrics(): void {
    this.subscribe('performance_metrics')
    this.subscribe('bottleneck_alerts')
    this.subscribe('system_health')
  }

  /**
   * Get current connection health
   */
  public getConnectionHealth(): {
    status: 'healthy' | 'warning' | 'error'
    details: Record<string, any>
  } {
    const now = Date.now()
    const timeSinceLastHeartbeat = now - this.lastHeartbeat
    
    if (!this.state.isConnected) {
      return {
        status: 'error',
        details: {
          connected: false,
          error: this.state.error,
          reconnectAttempts: this.state.reconnectAttempts
        }
      }
    }

    if (timeSinceLastHeartbeat > this.config.heartbeatInterval * 2) {
      return {
        status: 'warning',
        details: {
          connected: true,
          stalledHeartbeat: true,
          timeSinceLastHeartbeat,
          latency: this.state.latency
        }
      }
    }

    return {
      status: 'healthy',
      details: {
        connected: true,
        latency: this.state.latency,
        uptime: this.stats.uptime,
        messagesReceived: this.stats.messagesReceived
      }
    }
  }

  // Private methods
  private setupWebSocketEventHandlers(): void {
    if (!this.ws) return

    this.ws.onopen = () => {
      this.state.isConnected = true
      this.state.isConnecting = false
      this.state.reconnectAttempts = 0
      this.state.lastConnectedAt = new Date()
      this.state.error = null

      this.log('WebSocket connected')
      this.startHeartbeat()
      this.resubscribeAll()
      this.processMessageQueue()
    }

    this.ws.onclose = (event) => {
      this.state.isConnected = false
      this.state.isConnecting = false
      this.state.lastDisconnectedAt = new Date()

      this.log('WebSocket disconnected:', event.code, event.reason)
      this.stopHeartbeat()

      if (!this.isDestroyed && event.code !== 1000) {
        this.scheduleReconnect()
      }
    }

    this.ws.onerror = (error) => {
      this.log('WebSocket error:', error)
      this.state.error = 'Connection error occurred'
    }

    this.ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as CoordinationMessage
        this.handleIncomingMessage(message)
      } catch (error) {
        this.log('Failed to parse message:', error)
      }
    }
  }

  private handleIncomingMessage(message: CoordinationMessage): void {
    this.stats.messagesReceived++
    this.log('Received message:', message.type)

    // Handle system messages
    if (this.handleSystemMessage(message)) {
      return
    }

    // Route to registered handlers
    const handlers = Array.from(this.messageHandlers.values())
      .filter(handler => 
        Array.isArray(handler.type) 
          ? handler.type.includes(message.type)
          : handler.type === message.type || handler.type === '*'
      )
      .sort((a, b) => b.priority - a.priority)

    for (const handler of handlers) {
      try {
        handler.handler(message)
      } catch (error) {
        this.log('Handler error:', error)
      }
    }
  }

  private handleSystemMessage(message: any): boolean {
    switch (message.type) {
      case 'connection_established':
        this.state.connectionId = message.connection_id
        this.log('Connection established with ID:', message.connection_id)
        return true

      case 'pong':
        const latency = Date.now() - this.lastHeartbeat
        this.state.latency = latency
        this.stats.averageLatency = (this.stats.averageLatency + latency) / 2
        return true

      case 'subscription_confirmed':
        this.log('Subscription confirmed:', message.subscription)
        return true

      case 'error':
        this.log('Server error:', message.error)
        this.state.error = message.error
        return true

      default:
        return false
    }
  }

  private setupDefaultHandlers(): void {
    // Task assignment handler
    this.onMessage('task_assignments', 'task_assignment', (message) => {
      this.log('Task assigned:', message.data)
    }, 10)

    // Agent workload handler
    this.onMessage('agent_workload', 'agent_workload', (message) => {
      this.log('Agent workload updated:', message.data)
    }, 5)

    // Performance metrics handler
    this.onMessage('performance_metrics', 'performance_metrics', (message) => {
      this.log('Performance metrics updated:', message.data)
    }, 1)

    // Bottleneck alert handler
    this.onMessage('bottleneck_alerts', 'bottleneck_detected', (message) => {
      this.log('Bottleneck detected:', message.data)
      // Could trigger notifications here
    }, 20)
  }

  private startHeartbeat(): void {
    this.stopHeartbeat()
    
    this.heartbeatTimer = setInterval(() => {
      if (this.state.isConnected && this.ws) {
        this.lastHeartbeat = Date.now()
        this.sendMessage({ type: 'ping' })
      }
    }, this.config.heartbeatInterval)
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer)
      this.heartbeatTimer = null
    }
  }

  private scheduleReconnect(): void {
    if (this.state.reconnectAttempts >= this.config.maxReconnectAttempts) {
      this.log('Max reconnect attempts reached')
      this.state.error = 'Max reconnection attempts exceeded'
      return
    }

    const delay = Math.min(
      this.config.reconnectInterval * Math.pow(2, this.state.reconnectAttempts),
      30000 // Max 30 seconds
    )

    this.log(`Scheduling reconnect in ${delay}ms (attempt ${this.state.reconnectAttempts + 1})`)

    this.reconnectTimer = setTimeout(() => {
      this.state.reconnectAttempts++
      this.stats.reconnections++
      this.connect(this.state.connectionId || undefined)
    }, delay)
  }

  private resubscribeAll(): void {
    for (const subscription of this.subscriptions) {
      this.subscribe(subscription)
    }
  }

  private processMessageQueue(): void {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift()
      if (message) {
        this.sendMessage(message)
      }
    }
  }

  private clearTimers(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer)
      this.heartbeatTimer = null
    }
  }

  private buildWebSocketUrl(connectionId?: string): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const id = connectionId || `coord-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    
    return `${protocol}//${host}/api/v1/team-coordination/ws/${id}`
  }

  private getWebSocketBaseUrl(): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${protocol}//${window.location.host}`
  }

  private log(...args: any[]): void {
    if (this.config.enableDebugLogging) {
      console.log('[CoordinationWebSocket]', ...args)
    }
  }
}

// Singleton instance
let coordinationWebSocketService: CoordinationWebSocketService | null = null

/**
 * Get or create the coordination WebSocket service
 */
export function useCoordinationWebSocket(config?: Partial<CoordinationWebSocketConfig>): CoordinationWebSocketService {
  if (!coordinationWebSocketService) {
    coordinationWebSocketService = new CoordinationWebSocketService(config)
  }
  return coordinationWebSocketService
}

/**
 * Vue composable for coordination WebSocket integration
 */
export function useCoordinationWebSocketComposable() {
  const service = useCoordinationWebSocket()

  const isConnected = computed(() => service.state.isConnected)
  const isConnecting = computed(() => service.state.isConnecting)
  const connectionError = computed(() => service.state.error)
  const connectionHealth = computed(() => service.getConnectionHealth())

  const connect = async (connectionId?: string) => {
    await service.connect(connectionId)
  }

  const disconnect = () => {
    service.disconnect()
  }

  const subscribe = (subscription: string) => {
    service.subscribe(subscription)
  }

  const unsubscribe = (subscription: string) => {
    service.unsubscribe(subscription)
  }

  const onMessage = (
    id: string,
    types: string | string[],
    handler: (message: CoordinationMessage) => void,
    priority?: number
  ) => {
    service.onMessage(id, types, handler, priority)
  }

  const offMessage = (id: string) => {
    service.offMessage(id)
  }

  const sendMessage = (message: any) => {
    service.sendMessage(message)
  }

  // Convenience methods
  const subscribeToTaskDistribution = () => {
    service.subscribeToTaskDistribution()
  }

  const subscribeToAgentPerformance = () => {
    service.subscribeToAgentPerformance()
  }

  const subscribeToSystemMetrics = () => {
    service.subscribeToSystemMetrics()
  }

  return {
    // State
    isConnected,
    isConnecting,
    connectionError,
    connectionHealth,
    state: service.state,
    stats: service.stats,

    // Methods
    connect,
    disconnect,
    subscribe,
    unsubscribe,
    onMessage,
    offMessage,
    sendMessage,

    // Convenience methods
    subscribeToTaskDistribution,
    subscribeToAgentPerformance,
    subscribeToSystemMetrics,
    requestAgentTasks: service.requestAgentTasks.bind(service),
    requestSystemMetrics: service.requestSystemMetrics.bind(service)
  }
}

export default CoordinationWebSocketService