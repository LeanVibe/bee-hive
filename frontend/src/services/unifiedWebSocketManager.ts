/**
 * Unified WebSocket Manager for Multi-Agent Coordination Dashboard
 * 
 * Manages all WebSocket connections, handles message routing, 
 * provides connection pooling and automatic reconnection.
 */

import { ref, reactive, computed } from 'vue'
import type { 
  DashboardEvent,
  DashboardComponent,
  WebSocketMessage,
  GraphUpdateMessage,
  TranscriptUpdateMessage,
  AnalysisUpdateMessage,
  ConnectionStatus
} from '@/types/coordination'

export interface WebSocketEndpoint {
  id: string
  url: string
  protocols?: string[]
  params?: Record<string, string>
  component: DashboardComponent
  priority: 'high' | 'medium' | 'low'
  autoReconnect: boolean
  maxReconnectAttempts: number
  reconnectDelay: number
}

export interface WebSocketConnection {
  id: string
  endpoint: WebSocketEndpoint
  socket: WebSocket
  status: 'connecting' | 'connected' | 'disconnected' | 'error'
  lastActivity: Date
  reconnectAttempts: number
  messageQueue: WebSocketMessage[]
  subscriptions: Set<string>
}

export interface MessageHandler {
  component: DashboardComponent
  messageTypes: string[]
  handler: (message: WebSocketMessage) => void | Promise<void>
  priority: number
}

export interface ConnectionMetrics {
  totalConnections: number
  activeConnections: number
  messagesSent: number
  messagesReceived: number
  reconnectionAttempts: number
  averageLatency: number
  lastUpdate: Date
}

class UnifiedWebSocketManager {
  // Connection management
  private connections = new Map<string, WebSocketConnection>()
  private endpoints = new Map<string, WebSocketEndpoint>()
  private messageHandlers = new Map<string, MessageHandler[]>()
  
  // State management
  private state = reactive({
    isConnected: false,
    connectionCount: 0,
    lastUpdate: null as Date | null,
    metrics: {
      totalConnections: 0,
      activeConnections: 0,
      messagesSent: 0,
      messagesReceived: 0,
      reconnectionAttempts: 0,
      averageLatency: 0,
      lastUpdate: new Date()
    } as ConnectionMetrics
  })

  // Configuration
  private config = {
    maxConnections: 10,
    messageQueueSize: 1000,
    connectionTimeout: 30000,
    heartbeatInterval: 30000,
    latencyMeasurementInterval: 60000,
    cleanupInterval: 300000, // 5 minutes
    retryBackoffMultiplier: 1.5,
    maxRetryDelay: 30000
  }

  // Internal tracking
  private heartbeatIntervals = new Map<string, NodeJS.Timeout>()
  private reconnectTimeouts = new Map<string, NodeJS.Timeout>()
  private messageCallbacks = new Map<string, Array<(message: any) => void>>()
  private latencyMeasurements: number[] = []

  // Public reactive state
  public readonly isConnected = computed(() => this.state.isConnected)
  public readonly connectionCount = computed(() => this.state.connectionCount)
  public readonly metrics = computed(() => this.state.metrics)
  public readonly connections = computed(() => Array.from(this.connections.values()))

  constructor() {
    this.initializeCleanupInterval()
    this.initializeLatencyMeasurement()
  }

  /**
   * Register a WebSocket endpoint
   */
  public registerEndpoint(endpoint: WebSocketEndpoint): void {
    this.endpoints.set(endpoint.id, endpoint)
    console.log(`WebSocket endpoint registered: ${endpoint.id}`)
  }

  /**
   * Unregister a WebSocket endpoint
   */
  public unregisterEndpoint(endpointId: string): void {
    // Disconnect if connected
    const connection = this.connections.get(endpointId)
    if (connection) {
      this.disconnect(endpointId)
    }
    
    this.endpoints.delete(endpointId)
    console.log(`WebSocket endpoint unregistered: ${endpointId}`)
  }

  /**
   * Connect to a WebSocket endpoint
   */
  public async connect(endpointId: string, params?: Record<string, string>): Promise<void> {
    const endpoint = this.endpoints.get(endpointId)
    if (!endpoint) {
      throw new Error(`WebSocket endpoint not found: ${endpointId}`)
    }

    // Check if already connected
    const existingConnection = this.connections.get(endpointId)
    if (existingConnection && existingConnection.status === 'connected') {
      console.log(`WebSocket already connected: ${endpointId}`)
      return
    }

    // Build WebSocket URL with parameters
    const url = this.buildWebSocketUrl(endpoint, params)
    
    return new Promise((resolve, reject) => {
      try {
        const socket = new WebSocket(url, endpoint.protocols)
        
        const connection: WebSocketConnection = {
          id: endpointId,
          endpoint,
          socket,
          status: 'connecting',
          lastActivity: new Date(),
          reconnectAttempts: 0,
          messageQueue: [],
          subscriptions: new Set()
        }

        this.connections.set(endpointId, connection)
        this.updateConnectionState()

        // Connection timeout
        const connectionTimeout = setTimeout(() => {
          if (connection.status === 'connecting') {
            socket.close()
            this.handleConnectionError(endpointId, new Error('Connection timeout'))
            reject(new Error('WebSocket connection timeout'))
          }
        }, this.config.connectionTimeout)

        socket.onopen = () => {
          clearTimeout(connectionTimeout)
          connection.status = 'connected'
          connection.lastActivity = new Date()
          connection.reconnectAttempts = 0
          
          this.updateConnectionState()
          this.startHeartbeat(endpointId)
          this.processMessageQueue(endpointId)
          
          console.log(`WebSocket connected: ${endpointId}`)
          resolve()
        }

        socket.onmessage = (event) => {
          this.handleMessage(endpointId, event)
        }

        socket.onerror = (error) => {
          clearTimeout(connectionTimeout)
          this.handleConnectionError(endpointId, error)
          reject(error)
        }

        socket.onclose = (event) => {
          clearTimeout(connectionTimeout)
          this.handleConnectionClose(endpointId, event)
        }

        this.state.metrics.totalConnections++
        
      } catch (error) {
        reject(error)
      }
    })
  }

  /**
   * Disconnect from a WebSocket endpoint
   */
  public disconnect(endpointId: string): void {
    const connection = this.connections.get(endpointId)
    if (!connection) {
      return
    }

    // Clear intervals and timeouts
    this.stopHeartbeat(endpointId)
    this.clearReconnectTimeout(endpointId)

    // Close socket
    if (connection.socket.readyState === WebSocket.OPEN) {
      connection.socket.close(1000, 'Manual disconnect')
    }

    // Remove connection
    this.connections.delete(endpointId)
    this.updateConnectionState()
    
    console.log(`WebSocket disconnected: ${endpointId}`)
  }

  /**
   * Disconnect all WebSocket connections
   */
  public disconnectAll(): void {
    console.log('Disconnecting all WebSocket connections')
    
    for (const endpointId of this.connections.keys()) {
      this.disconnect(endpointId)
    }
  }

  /**
   * Send message to specific endpoint
   */
  public send(endpointId: string, message: WebSocketMessage): boolean {
    const connection = this.connections.get(endpointId)
    if (!connection) {
      console.error(`WebSocket connection not found: ${endpointId}`)
      return false
    }

    if (connection.status !== 'connected') {
      // Queue message for later delivery
      if (connection.messageQueue.length < this.config.messageQueueSize) {
        connection.messageQueue.push(message)
        console.log(`Message queued for ${endpointId}: ${message.type}`)
      } else {
        console.warn(`Message queue full for ${endpointId}, dropping message`)
      }
      return false
    }

    try {
      const serialized = JSON.stringify(message)
      connection.socket.send(serialized)
      connection.lastActivity = new Date()
      this.state.metrics.messagesSent++
      return true
    } catch (error) {
      console.error(`Failed to send WebSocket message to ${endpointId}:`, error)
      return false
    }
  }

  /**
   * Broadcast message to all connected endpoints
   */
  public broadcast(message: WebSocketMessage, filter?: (endpoint: WebSocketEndpoint) => boolean): number {
    let sentCount = 0
    
    for (const [endpointId, connection] of this.connections) {
      if (connection.status === 'connected') {
        if (!filter || filter(connection.endpoint)) {
          if (this.send(endpointId, message)) {
            sentCount++
          }
        }
      }
    }
    
    return sentCount
  }

  /**
   * Register message handler
   */
  public onMessage(
    messageType: string, 
    handler: (message: WebSocketMessage, endpointId: string) => void | Promise<void>,
    component?: DashboardComponent,
    priority = 1
  ): () => void {
    if (!this.messageHandlers.has(messageType)) {
      this.messageHandlers.set(messageType, [])
    }

    const messageHandler: MessageHandler = {
      component: component || DashboardComponent.SERVICE,
      messageTypes: [messageType],
      handler: (message) => handler(message, message.correlation_id || ''),
      priority
    }

    const handlers = this.messageHandlers.get(messageType)!
    handlers.push(messageHandler)
    
    // Sort by priority (higher first)
    handlers.sort((a, b) => b.priority - a.priority)

    // Return unsubscribe function
    return () => {
      const index = handlers.indexOf(messageHandler)
      if (index > -1) {
        handlers.splice(index, 1)
      }
    }
  }

  /**
   * Subscribe to specific message types on an endpoint
   */
  public subscribe(endpointId: string, messageTypes: string[]): void {
    const connection = this.connections.get(endpointId)
    if (!connection) {
      console.error(`Cannot subscribe - connection not found: ${endpointId}`)
      return
    }

    messageTypes.forEach(type => {
      connection.subscriptions.add(type)
    })

    // Send subscription message if connected
    if (connection.status === 'connected') {
      this.send(endpointId, {
        type: 'subscribe',
        data: { message_types: messageTypes },
        timestamp: new Date().toISOString()
      })
    }
  }

  /**
   * Unsubscribe from message types on an endpoint
   */
  public unsubscribe(endpointId: string, messageTypes: string[]): void {
    const connection = this.connections.get(endpointId)
    if (!connection) {
      return
    }

    messageTypes.forEach(type => {
      connection.subscriptions.delete(type)
    })

    // Send unsubscribe message if connected
    if (connection.status === 'connected') {
      this.send(endpointId, {
        type: 'unsubscribe',
        data: { message_types: messageTypes },
        timestamp: new Date().toISOString()
      })
    }
  }

  /**
   * Get connection status for endpoint
   */
  public getConnectionStatus(endpointId: string): ConnectionStatus {
    const connection = this.connections.get(endpointId)
    if (!connection) {
      return {
        websocket: false,
        api: true,
        lastUpdate: null
      }
    }

    return {
      websocket: connection.status === 'connected',
      api: true,
      lastUpdate: connection.lastActivity,
      reconnectAttempts: connection.reconnectAttempts,
      latency: this.getAverageLatency()
    }
  }

  /**
   * Get overall connection status
   */
  public getOverallStatus(): ConnectionStatus {
    const hasActiveConnections = Array.from(this.connections.values())
      .some(conn => conn.status === 'connected')

    return {
      websocket: hasActiveConnections,
      api: true,
      lastUpdate: this.state.lastUpdate,
      reconnectAttempts: this.state.metrics.reconnectionAttempts,
      latency: this.state.metrics.averageLatency
    }
  }

  // Private methods

  private buildWebSocketUrl(endpoint: WebSocketEndpoint, params?: Record<string, string>): string {
    let url = endpoint.url
    
    const allParams = { ...endpoint.params, ...params }
    if (Object.keys(allParams).length > 0) {
      const searchParams = new URLSearchParams(allParams)
      url += (url.includes('?') ? '&' : '?') + searchParams.toString()
    }
    
    return url
  }

  private handleMessage(endpointId: string, event: MessageEvent): void {
    const connection = this.connections.get(endpointId)
    if (!connection) {
      return
    }

    connection.lastActivity = new Date()
    this.state.metrics.messagesReceived++

    try {
      const message: WebSocketMessage = JSON.parse(event.data)
      
      // Handle special message types
      if (message.type === 'pong') {
        this.handlePongMessage(endpointId, message)
        return
      }

      // Route to registered handlers
      const handlers = this.messageHandlers.get(message.type) || []
      const allHandlers = this.messageHandlers.get('*') || []
      
      const combinedHandlers = [...handlers, ...allHandlers]
        .sort((a, b) => b.priority - a.priority)

      combinedHandlers.forEach(handler => {
        try {
          handler.handler(message)
        } catch (error) {
          console.error(`Error in message handler for ${message.type}:`, error)
        }
      })

      // Update specific message type routing
      this.routeMessageByType(endpointId, message)
      
    } catch (error) {
      console.error(`Failed to parse WebSocket message from ${endpointId}:`, error)
    }
  }

  private routeMessageByType(endpointId: string, message: WebSocketMessage): void {
    switch (message.type) {
      case 'graph_update':
        this.handleGraphUpdate(endpointId, message as GraphUpdateMessage)
        break
      case 'transcript_update':
        this.handleTranscriptUpdate(endpointId, message as TranscriptUpdateMessage)
        break
      case 'analysis_update':
        this.handleAnalysisUpdate(endpointId, message as AnalysisUpdateMessage)
        break
      case 'performance_update':
        this.handlePerformanceUpdate(endpointId, message)
        break
      case 'system_alert':
        this.handleSystemAlert(endpointId, message)
        break
      default:
        // Generic message handling
        console.log(`Received message type ${message.type} from ${endpointId}`)
    }
  }

  private handleGraphUpdate(endpointId: string, message: GraphUpdateMessage): void {
    // Emit dashboard event for graph component
    this.emitDashboardEvent({
      type: 'realtime_update' as any,
      source: DashboardComponent.SERVICE,
      target: DashboardComponent.GRAPH,
      data: message.data,
      timestamp: message.timestamp
    })
  }

  private handleTranscriptUpdate(endpointId: string, message: TranscriptUpdateMessage): void {
    // Emit dashboard event for transcript component
    this.emitDashboardEvent({
      type: 'realtime_update' as any,
      source: DashboardComponent.SERVICE,
      target: DashboardComponent.TRANSCRIPT,
      data: message.data,
      timestamp: message.timestamp
    })
  }

  private handleAnalysisUpdate(endpointId: string, message: AnalysisUpdateMessage): void {
    // Emit dashboard event for analysis component
    this.emitDashboardEvent({
      type: 'realtime_update' as any,
      source: DashboardComponent.SERVICE,
      target: DashboardComponent.ANALYSIS,
      data: message.data,
      timestamp: message.timestamp
    })
  }

  private handlePerformanceUpdate(endpointId: string, message: WebSocketMessage): void {
    // Emit dashboard event for all components
    this.emitDashboardEvent({
      type: 'performance_alert' as any,
      source: DashboardComponent.SERVICE,
      target: 'all' as any,
      data: message.data,
      timestamp: message.timestamp
    })
  }

  private handleSystemAlert(endpointId: string, message: WebSocketMessage): void {
    console.warn(`System alert from ${endpointId}:`, message.data)
    
    // Emit dashboard event
    this.emitDashboardEvent({
      type: 'error_occurred' as any,  
      source: DashboardComponent.SERVICE,
      target: 'all' as any,
      data: {
        ...message.data,
        endpointId,
        severity: message.data.severity || 'medium'
      },
      timestamp: message.timestamp
    })
  }

  private handleConnectionError(endpointId: string, error: any): void {
    const connection = this.connections.get(endpointId)
    if (!connection) {
      return
    }

    connection.status = 'error'
    this.updateConnectionState()

    console.error(`WebSocket error on ${endpointId}:`, error)

    // Schedule reconnection if enabled
    if (connection.endpoint.autoReconnect && 
        connection.reconnectAttempts < connection.endpoint.maxReconnectAttempts) {
      this.scheduleReconnection(endpointId)
    }
  }

  private handleConnectionClose(endpointId: string, event: CloseEvent): void {
    const connection = this.connections.get(endpointId)
    if (!connection) {
      return
    }

    connection.status = 'disconnected'
    this.stopHeartbeat(endpointId)
    this.updateConnectionState()

    console.log(`WebSocket closed: ${endpointId}, code: ${event.code}, reason: ${event.reason}`)

    // Schedule reconnection if not manually closed
    if (event.code !== 1000 && connection.endpoint.autoReconnect && 
        connection.reconnectAttempts < connection.endpoint.maxReconnectAttempts) {
      this.scheduleReconnection(endpointId)
    }
  }

  private scheduleReconnection(endpointId: string): void {
    const connection = this.connections.get(endpointId)
    if (!connection) {
      return
    }

    connection.reconnectAttempts++
    this.state.metrics.reconnectionAttempts++

    // Calculate exponential backoff delay
    const baseDelay = connection.endpoint.reconnectDelay
    const backoffDelay = Math.min(
      baseDelay * Math.pow(this.config.retryBackoffMultiplier, connection.reconnectAttempts - 1),
      this.config.maxRetryDelay
    )

    console.log(`Scheduling reconnection for ${endpointId} in ${backoffDelay}ms (attempt ${connection.reconnectAttempts})`)

    const timeout = setTimeout(async () => {
      this.reconnectTimeouts.delete(endpointId)
      
      try {
        await this.connect(endpointId)
        console.log(`Reconnected to ${endpointId}`)
      } catch (error) {
        console.error(`Reconnection failed for ${endpointId}:`, error)
      }
    }, backoffDelay)

    this.reconnectTimeouts.set(endpointId, timeout)
  }

  private clearReconnectTimeout(endpointId: string): void {
    const timeout = this.reconnectTimeouts.get(endpointId)
    if (timeout) {
      clearTimeout(timeout)
      this.reconnectTimeouts.delete(endpointId)
    }
  }

  private startHeartbeat(endpointId: string): void {
    const connection = this.connections.get(endpointId)
    if (!connection) {
      return
    }

    const interval = setInterval(() => {
      if (connection.status === 'connected') {
        this.send(endpointId, {
          type: 'ping',
          data: { timestamp: Date.now() },
          timestamp: new Date().toISOString()
        })
      } else {
        this.stopHeartbeat(endpointId)
      }
    }, this.config.heartbeatInterval)

    this.heartbeatIntervals.set(endpointId, interval)
  }

  private stopHeartbeat(endpointId: string): void {
    const interval = this.heartbeatIntervals.get(endpointId)
    if (interval) {
      clearInterval(interval)
      this.heartbeatIntervals.delete(endpointId)
    }
  }

  private handlePongMessage(endpointId: string, message: WebSocketMessage): void {
    const sentTime = message.data.timestamp
    if (typeof sentTime === 'number') {
      const latency = Date.now() - sentTime
      this.addLatencyMeasurement(latency)
    }
  }

  private addLatencyMeasurement(latency: number): void {
    this.latencyMeasurements.push(latency)
    
    // Keep only last 100 measurements
    if (this.latencyMeasurements.length > 100) {
      this.latencyMeasurements.shift()
    }
    
    // Update average
    this.state.metrics.averageLatency = this.getAverageLatency()
  }

  private getAverageLatency(): number {
    if (this.latencyMeasurements.length === 0) {
      return 0
    }
    
    return this.latencyMeasurements.reduce((sum, latency) => sum + latency, 0) / 
           this.latencyMeasurements.length
  }

  private processMessageQueue(endpointId: string): void {
    const connection = this.connections.get(endpointId)
    if (!connection || connection.status !== 'connected') {
      return
    }

    const queue = connection.messageQueue
    while (queue.length > 0) {
      const message = queue.shift()!
      if (!this.send(endpointId, message)) {
        // Re-queue if send failed
        queue.unshift(message)
        break
      }
    }
  }

  private updateConnectionState(): void {
    const activeConnections = Array.from(this.connections.values())
      .filter(conn => conn.status === 'connected')

    this.state.isConnected = activeConnections.length > 0
    this.state.connectionCount = activeConnections.length
    this.state.lastUpdate = new Date()
    
    this.state.metrics.activeConnections = activeConnections.length
    this.state.metrics.lastUpdate = new Date()
  }

  private initializeCleanupInterval(): void {
    setInterval(() => {
      this.cleanupDeadConnections()
    }, this.config.cleanupInterval)
  }

  private initializeLatencyMeasurement(): void {
    setInterval(() => {
      // Trigger latency measurement on active connections
      for (const [endpointId, connection] of this.connections) {
        if (connection.status === 'connected') {
          this.send(endpointId, {
            type: 'ping',
            data: { timestamp: Date.now() },
            timestamp: new Date().toISOString()
          })
        }
      }
    }, this.config.latencyMeasurementInterval)
  }

  private cleanupDeadConnections(): void {
    for (const [endpointId, connection] of this.connections) {
      const timeSinceLastActivity = Date.now() - connection.lastActivity.getTime()
      
      if (timeSinceLastActivity > this.config.connectionTimeout * 2) {
        console.log(`Cleaning up dead connection: ${endpointId}`)
        this.disconnect(endpointId)
      }
    }
  }

  // Event emission for dashboard components
  private dashboardEventListeners = new Map<string, Array<(event: DashboardEvent) => void>>()

  public onDashboardEvent(eventType: string, handler: (event: DashboardEvent) => void): () => void {
    if (!this.dashboardEventListeners.has(eventType)) {
      this.dashboardEventListeners.set(eventType, [])
    }
    
    const handlers = this.dashboardEventListeners.get(eventType)!
    handlers.push(handler)
    
    return () => {
      const index = handlers.indexOf(handler)
      if (index > -1) {
        handlers.splice(index, 1)
      }
    }
  }

  private emitDashboardEvent(event: DashboardEvent): void {
    // Emit to specific event type listeners
    const handlers = this.dashboardEventListeners.get(event.type) || []
    handlers.forEach(handler => {
      try {
        handler(event)
      } catch (error) {
        console.error('Error in dashboard event handler:', error)
      }
    })

    // Emit to 'all' listeners
    const allHandlers = this.dashboardEventListeners.get('all') || []
    allHandlers.forEach(handler => {
      try {
        handler(event)
      } catch (error) {
        console.error('Error in dashboard event handler:', error)
      }
    })
  }

  /**
   * Cleanup all resources
   */
  public destroy(): void {
    this.disconnectAll()
    
    // Clear all timeouts and intervals
    this.heartbeatIntervals.forEach(interval => clearInterval(interval))
    this.reconnectTimeouts.forEach(timeout => clearTimeout(timeout))
    
    this.heartbeatIntervals.clear()
    this.reconnectTimeouts.clear()
    this.messageHandlers.clear()
    this.dashboardEventListeners.clear()
    this.endpoints.clear()
  }
}

// Singleton instance
export const unifiedWebSocketManager = new UnifiedWebSocketManager()

// Vue composable for easy integration
export function useUnifiedWebSocket() {
  return {
    // State
    isConnected: unifiedWebSocketManager.isConnected,
    connectionCount: unifiedWebSocketManager.connectionCount,
    metrics: unifiedWebSocketManager.metrics,
    connections: unifiedWebSocketManager.connections,
    
    // Connection management
    registerEndpoint: unifiedWebSocketManager.registerEndpoint.bind(unifiedWebSocketManager),
    connect: unifiedWebSocketManager.connect.bind(unifiedWebSocketManager),
    disconnect: unifiedWebSocketManager.disconnect.bind(unifiedWebSocketManager),
    disconnectAll: unifiedWebSocketManager.disconnectAll.bind(unifiedWebSocketManager),
    
    // Messaging
    send: unifiedWebSocketManager.send.bind(unifiedWebSocketManager),
    broadcast: unifiedWebSocketManager.broadcast.bind(unifiedWebSocketManager),
    onMessage: unifiedWebSocketManager.onMessage.bind(unifiedWebSocketManager),
    
    // Subscriptions
    subscribe: unifiedWebSocketManager.subscribe.bind(unifiedWebSocketManager),
    unsubscribe: unifiedWebSocketManager.unsubscribe.bind(unifiedWebSocketManager),
    
    // Status
    getConnectionStatus: unifiedWebSocketManager.getConnectionStatus.bind(unifiedWebSocketManager),
    getOverallStatus: unifiedWebSocketManager.getOverallStatus.bind(unifiedWebSocketManager),
    
    // Events
    onDashboardEvent: unifiedWebSocketManager.onDashboardEvent.bind(unifiedWebSocketManager)
  }
}

export default unifiedWebSocketManager