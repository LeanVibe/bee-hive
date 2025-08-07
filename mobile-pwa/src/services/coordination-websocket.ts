/**
 * WebSocket service for real-time coordination monitoring updates
 * Provides <100ms latency updates for dashboard components
 */

import { EventEmitter } from '../utils/event-emitter'

export interface CoordinationWebSocketMessage {
  type: string
  data: any
  timestamp: string
  source?: string
}

export interface WebSocketConnectionState {
  connected: boolean
  connecting: boolean
  error: string | null
  lastUpdate: Date | null
  reconnectAttempts: number
  latency: number
}

export class CoordinationWebSocketService extends EventEmitter {
  private static instance: CoordinationWebSocketService | null = null
  private websocket: WebSocket | null = null
  private connectionState: WebSocketConnectionState = {
    connected: false,
    connecting: false,
    error: null,
    lastUpdate: null,
    reconnectAttempts: 0,
    latency: 0
  }
  
  private reconnectTimeout: number | null = null
  private heartbeatInterval: number | null = null
  private latencyCheckInterval: number | null = null
  private maxReconnectAttempts = 10
  private reconnectDelay = 1000 // Start with 1 second
  private heartbeatFrequency = 30000 // 30 seconds
  private latencyCheckFrequency = 5000 // 5 seconds

  private constructor() {
    super()
  }

  static getInstance(): CoordinationWebSocketService {
    if (!CoordinationWebSocketService.instance) {
      CoordinationWebSocketService.instance = new CoordinationWebSocketService()
    }
    return CoordinationWebSocketService.instance
  }

  /**
   * Connect to the coordination monitoring WebSocket
   */
  async connect(): Promise<void> {
    if (this.connectionState.connected || this.connectionState.connecting) {
      return
    }

    this.connectionState.connecting = true
    this.connectionState.error = null
    this.emit('connecting')

    try {
      // Construct WebSocket URL
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = `${protocol}//${window.location.host}/api/v1/coordination-monitoring/live-dashboard`
      
      console.log('🔗 Connecting to coordination monitoring WebSocket:', wsUrl)

      this.websocket = new WebSocket(wsUrl)
      this.setupWebSocketEventHandlers()

    } catch (error) {
      console.error('❌ Failed to create WebSocket connection:', error)
      this.handleConnectionError(error)
    }
  }

  /**
   * Disconnect from the WebSocket
   */
  disconnect(): void {
    console.log('🔌 Disconnecting coordination monitoring WebSocket')
    
    this.clearTimers()
    
    if (this.websocket) {
      this.websocket.close(1000, 'User requested disconnect')
      this.websocket = null
    }

    this.connectionState = {
      connected: false,
      connecting: false,
      error: null,
      lastUpdate: null,
      reconnectAttempts: 0,
      latency: 0
    }

    this.emit('disconnected')
  }

  /**
   * Send a message through the WebSocket
   */
  send(message: CoordinationWebSocketMessage): void {
    if (!this.connectionState.connected || !this.websocket) {
      console.warn('⚠️ Cannot send message: WebSocket not connected')
      return
    }

    try {
      const messageWithTimestamp = {
        ...message,
        timestamp: new Date().toISOString()
      }
      
      this.websocket.send(JSON.stringify(messageWithTimestamp))
      
    } catch (error) {
      console.error('❌ Failed to send WebSocket message:', error)
      this.handleConnectionError(error)
    }
  }

  /**
   * Get current connection state
   */
  getConnectionState(): WebSocketConnectionState {
    return { ...this.connectionState }
  }

  /**
   * Get connection quality based on latency
   */
  getConnectionQuality(): 'excellent' | 'good' | 'poor' | 'offline' {
    if (!this.connectionState.connected) return 'offline'
    
    if (this.connectionState.latency <= 50) return 'excellent'
    if (this.connectionState.latency <= 150) return 'good'
    return 'poor'
  }

  /**
   * Request specific data update from the server
   */
  requestUpdate(dataType: string, params?: any): void {
    this.send({
      type: 'request_update',
      data: { dataType, params }
    })
  }

  private setupWebSocketEventHandlers(): void {
    if (!this.websocket) return

    this.websocket.onopen = (event) => {
      console.log('✅ Coordination monitoring WebSocket connected')
      
      this.connectionState.connected = true
      this.connectionState.connecting = false
      this.connectionState.error = null
      this.connectionState.reconnectAttempts = 0
      this.connectionState.lastUpdate = new Date()

      this.emit('connected', { event })
      this.startHeartbeat()
      this.startLatencyCheck()
    }

    this.websocket.onmessage = (event) => {
      try {
        const message: CoordinationWebSocketMessage = JSON.parse(event.data)
        
        this.connectionState.lastUpdate = new Date()
        this.emit('message', message)
        
        // Handle different message types
        this.handleIncomingMessage(message)
        
      } catch (error) {
        console.error('❌ Failed to parse WebSocket message:', error, event.data)
      }
    }

    this.websocket.onerror = (event) => {
      console.error('❌ WebSocket error:', event)
      this.handleConnectionError(new Error('WebSocket error occurred'))
    }

    this.websocket.onclose = (event) => {
      console.log('🔌 WebSocket connection closed:', event.code, event.reason)
      
      this.connectionState.connected = false
      this.connectionState.connecting = false
      this.clearTimers()
      
      this.emit('disconnected', { event })

      // Attempt to reconnect if not a clean close
      if (event.code !== 1000 && this.connectionState.reconnectAttempts < this.maxReconnectAttempts) {
        this.scheduleReconnect()
      }
    }
  }

  private handleIncomingMessage(message: CoordinationWebSocketMessage): void {
    const { type, data } = message

    switch (type) {
      case 'dashboard_update':
        this.emit('dashboard_update', data)
        break
        
      case 'success_rate_update':
        this.emit('success_rate_update', data)
        break
        
      case 'agent_status_update':
        this.emit('agent_status_update', data)
        break
        
      case 'task_distribution_update':
        this.emit('task_distribution_update', data)
        break
        
      case 'communication_health_update':
        this.emit('communication_health_update', data)
        break
        
      case 'system_alert':
        this.emit('system_alert', data)
        break
        
      case 'pong':
        this.handlePongMessage(data)
        break
        
      case 'error':
        console.error('WebSocket server error:', data)
        this.emit('error', data)
        break
        
      default:
        console.warn('Unknown WebSocket message type:', type)
    }
  }

  private handlePongMessage(data: any): void {
    if (data.timestamp) {
      const sentTime = new Date(data.timestamp).getTime()
      const receivedTime = Date.now()
      this.connectionState.latency = receivedTime - sentTime
      
      this.emit('latency_update', { latency: this.connectionState.latency })
    }
  }

  private handleConnectionError(error: any): void {
    console.error('🚨 WebSocket connection error:', error)
    
    this.connectionState.error = error?.message || 'Connection error'
    this.connectionState.connected = false
    this.connectionState.connecting = false
    
    this.emit('error', { error })
    this.clearTimers()
    
    // Attempt to reconnect
    if (this.connectionState.reconnectAttempts < this.maxReconnectAttempts) {
      this.scheduleReconnect()
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimeout) return
    
    this.connectionState.reconnectAttempts++
    const delay = Math.min(this.reconnectDelay * Math.pow(2, this.connectionState.reconnectAttempts - 1), 30000)
    
    console.log(`🔄 Scheduling WebSocket reconnect attempt ${this.connectionState.reconnectAttempts} in ${delay}ms`)
    
    this.reconnectTimeout = window.setTimeout(() => {
      this.reconnectTimeout = null
      this.connect()
    }, delay)
    
    this.emit('reconnect_scheduled', { attempt: this.connectionState.reconnectAttempts, delay })
  }

  private startHeartbeat(): void {
    if (this.heartbeatInterval) return
    
    this.heartbeatInterval = window.setInterval(() => {
      if (this.connectionState.connected) {
        this.send({
          type: 'ping',
          data: { timestamp: new Date().toISOString() }
        })
      }
    }, this.heartbeatFrequency)
  }

  private startLatencyCheck(): void {
    if (this.latencyCheckInterval) return
    
    this.latencyCheckInterval = window.setInterval(() => {
      if (this.connectionState.connected) {
        this.send({
          type: 'latency_check',
          data: { timestamp: new Date().toISOString() }
        })
      }
    }, this.latencyCheckFrequency)
  }

  private clearTimers(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout)
      this.reconnectTimeout = null
    }
    
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval)
      this.heartbeatInterval = null
    }
    
    if (this.latencyCheckInterval) {
      clearInterval(this.latencyCheckInterval)
      this.latencyCheckInterval = null
    }
  }

  /**
   * Enable mobile dashboard optimizations
   */
  enableMobileDashboardMode(): void {
    // Reduce update frequency on mobile to save battery
    if ('ontouchstart' in window) {
      this.heartbeatFrequency = 60000 // 1 minute
      this.latencyCheckFrequency = 10000 // 10 seconds
    }
    
    this.send({
      type: 'enable_mobile_mode',
      data: { enabled: true }
    })
  }

  /**
   * Subscribe to specific coordination events
   */
  subscribeToEvents(eventTypes: string[]): void {
    this.send({
      type: 'subscribe',
      data: { eventTypes }
    })
  }

  /**
   * Unsubscribe from coordination events
   */
  unsubscribeFromEvents(eventTypes: string[]): void {
    this.send({
      type: 'unsubscribe', 
      data: { eventTypes }
    })
  }
}