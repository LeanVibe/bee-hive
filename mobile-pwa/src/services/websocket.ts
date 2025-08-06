import { EventEmitter } from '../utils/event-emitter'
import { AuthService } from './auth'

export interface WebSocketMessage {
  type: string
  data?: any
  timestamp?: string
  agent_id?: string
}

export interface EventData {
  event_type: string
  agent_id?: string
  task_id?: string
  status?: string
  message?: string
  payload?: any
  timestamp: string
}

export interface MetricData {
  metric_name: string
  value: number
  labels?: Record<string, string>
  timestamp: string
}

export interface AlertData {
  severity: 'low' | 'medium' | 'high' | 'critical'
  title: string
  message: string
  source: string
  timestamp: string
}

export class WebSocketService extends EventEmitter {
  private static instance: WebSocketService
  private ws: WebSocket | null = null
  private reconnectTimer: number | null = null
  private pingTimer: number | null = null
  private isConnecting: boolean = false
  private reconnectAttempts: number = 0
  private readonly maxReconnectAttempts: number = 10
  private readonly reconnectInterval: number = 1000 // Start with 1 second
  private readonly maxReconnectInterval: number = 30000 // Max 30 seconds
  private readonly pingInterval: number = 30000 // 30 seconds
  
  private authService: AuthService
  
  static getInstance(): WebSocketService {
    if (!WebSocketService.instance) {
      WebSocketService.instance = new WebSocketService()
    }
    return WebSocketService.instance
  }
  
  constructor() {
    super()
    this.authService = AuthService.getInstance()
    
    // Listen for auth state changes
    this.authService.on('authenticated', () => this.ensureConnection())
    this.authService.on('unauthenticated', () => this.disconnect())
    
    // Handle page visibility changes
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible' && this.authService.isAuthenticated()) {
        this.ensureConnection()
      }
    })
  }
  
  async initialize(): Promise<void> {
    if (this.authService.isAuthenticated()) {
      await this.ensureConnection()
    }
  }
  
  async ensureConnection(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN || this.isConnecting) {
      return
    }
    
    await this.connect()
  }
  
  private async connect(): Promise<void> {
    if (this.isConnecting) return
    
    try {
      this.isConnecting = true
      
      // Clear any existing connection
      this.cleanup()
      
      // Determine WebSocket URL
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const host = window.location.hostname
      const port = process.env.NODE_ENV === 'development' ? ':8000' : ''
      const wsUrl = `${protocol}//${host}${port}/api/v1/ws/observability`
      
      console.log('🔌 Connecting to WebSocket:', wsUrl)
      
      this.ws = new WebSocket(wsUrl)
      
      this.ws.onopen = this.handleOpen.bind(this)
      this.ws.onmessage = this.handleMessage.bind(this)
      this.ws.onclose = this.handleClose.bind(this)
      this.ws.onerror = this.handleError.bind(this)
      
    } catch (error) {
      console.error('WebSocket connection failed:', error)
      this.isConnecting = false
      this.emit('error', error)
      this.scheduleReconnect()
    }
  }
  
  private handleOpen(event: Event): void {
    console.log('✅ WebSocket connected')
    this.isConnecting = false
    this.reconnectAttempts = 0
    
    // Send authentication message
    this.sendMessage({
      type: 'authenticate',
      data: {
        token: this.authService.getToken()
      }
    })
    
    // Start ping/pong
    this.startPing()
    
    this.emit('connected')
  }
  
  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data)
      
      // Handle different message types
      switch (message.type) {
        case 'connection':
          console.log('🔌 Connection status:', message.data)
          break
          
        case 'event':
          this.handleEventMessage(message.data)
          break
          
        case 'metric':
          this.handleMetricMessage(message.data)
          break
          
        case 'alert':
          this.handleAlertMessage(message.data)
          break
          
        case 'agent_event':
          this.handleAgentEventMessage(message)
          break
          
        case 'pong':
          // Handle pong response
          break
          
        case 'error':
          console.error('WebSocket server error:', message.data)
          this.emit('error', new Error(message.data.message || 'Server error'))
          break
          
        case 'performance_update':
          // Handle performance updates silently or emit specific event
          this.emit('performance_update', message.data)
          break
          
        case 'keepalive':
          // Handle keepalive messages silently
          this.emit('keepalive', message)
          break
          
        default:
          console.log('📡 Unknown message type:', message.type, message)
      }
      
      // Emit raw message for debugging
      this.emit('message', message)
      
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error)
    }
  }
  
  private handleEventMessage(data: EventData): void {
    this.emit('event', data)
    
    // Emit specific event types
    if (data.event_type) {
      this.emit(`event:${data.event_type}`, data)
    }
    
    // Emit agent-specific events
    if (data.agent_id) {
      this.emit(`agent:${data.agent_id}`, data)
    }
    
    // Emit task-specific events
    if (data.task_id) {
      this.emit(`task:${data.task_id}`, data)
    }
  }
  
  private handleMetricMessage(data: MetricData): void {
    this.emit('metric', data)
    
    // Emit metric-specific events
    if (data.metric_name) {
      this.emit(`metric:${data.metric_name}`, data)
    }
  }
  
  private handleAlertMessage(data: AlertData): void {
    this.emit('alert', data)
    
    // Emit severity-specific alerts
    this.emit(`alert:${data.severity}`, data)
    
    // Show critical alerts immediately
    if (data.severity === 'critical') {
      this.emit('critical_alert', data)
    }
  }
  
  private handleAgentEventMessage(message: WebSocketMessage): void {
    this.emit('agent_event', message.data)
    
    if (message.agent_id) {
      this.emit(`agent:${message.agent_id}:event`, message.data)
    }
  }
  
  private handleClose(event: CloseEvent): void {
    console.log('🔌 WebSocket disconnected:', event.code, event.reason)
    this.isConnecting = false
    this.cleanup()
    
    this.emit('disconnected', { code: event.code, reason: event.reason })
    
    // Attempt to reconnect if not a clean close
    if (event.code !== 1000 && this.authService.isAuthenticated()) {
      this.scheduleReconnect()
    }
  }
  
  private handleError(event: Event): void {
    console.error('🚨 WebSocket error:', event)
    this.isConnecting = false
    
    const error = new Error('WebSocket connection error')
    this.emit('error', error)
  }
  
  private scheduleReconnect(): void {
    if (this.reconnectTimer || this.reconnectAttempts >= this.maxReconnectAttempts) {
      return
    }
    
    const delay = Math.min(
      this.reconnectInterval * Math.pow(2, this.reconnectAttempts),
      this.maxReconnectInterval
    )
    
    console.log(`🔄 Scheduling reconnect in ${delay}ms (attempt ${this.reconnectAttempts + 1}/${this.maxReconnectAttempts})`)
    
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null
      this.reconnectAttempts++
      this.connect()
    }, delay)
  }
  
  private startPing(): void {
    this.stopPing()
    
    this.pingTimer = window.setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.sendMessage({
          type: 'ping',
          timestamp: new Date().toISOString()
        })
      }
    }, this.pingInterval)
  }
  
  private stopPing(): void {
    if (this.pingTimer) {
      clearInterval(this.pingTimer)
      this.pingTimer = null
    }
  }
  
  private cleanup(): void {
    this.stopPing()
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    
    if (this.ws) {
      // Remove event listeners to prevent memory leaks
      this.ws.onopen = null
      this.ws.onmessage = null
      this.ws.onclose = null
      this.ws.onerror = null
      
      if (this.ws.readyState === WebSocket.OPEN) {
        this.ws.close(1000, 'Client disconnect')
      }
      
      this.ws = null
    }
  }
  
  sendMessage(message: WebSocketMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      try {
        const messageWithTimestamp = {
          ...message,
          timestamp: message.timestamp || new Date().toISOString()
        }
        
        this.ws.send(JSON.stringify(messageWithTimestamp))
      } catch (error) {
        console.error('Failed to send WebSocket message:', error)
      }
    } else {
      console.warn('Cannot send message - WebSocket not connected')
    }
  }
  
  disconnect(): void {
    console.log('🔌 Disconnecting WebSocket')
    this.reconnectAttempts = this.maxReconnectAttempts // Prevent reconnection
    this.cleanup()
  }
  
  reconnect(): void {
    console.log('🔄 Manual WebSocket reconnect requested')
    this.reconnectAttempts = 0
    this.disconnect()
    
    if (this.authService.isAuthenticated()) {
      setTimeout(() => this.connect(), 1000)
    }
  }
  
  // Public getters
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }
  
  getConnectionState(): string {
    if (!this.ws) return 'disconnected'
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return 'connecting'
      case WebSocket.OPEN:
        return 'connected'
      case WebSocket.CLOSING:
        return 'closing'
      case WebSocket.CLOSED:
        return 'disconnected'
      default:
        return 'unknown'
    }
  }
  
  getReconnectAttempts(): number {
    return this.reconnectAttempts
  }
  
  // Helper methods for specific subscriptions
  subscribeToAgent(agentId: string, callback: (data: any) => void): () => void {
    const eventName = `agent:${agentId}`
    this.on(eventName, callback)
    
    // Return unsubscribe function
    return () => this.off(eventName, callback)
  }
  
  subscribeToTask(taskId: string, callback: (data: any) => void): () => void {
    const eventName = `task:${taskId}`
    this.on(eventName, callback)
    
    return () => this.off(eventName, callback)
  }
  
  subscribeToMetric(metricName: string, callback: (data: MetricData) => void): () => void {
    const eventName = `metric:${metricName}`
    this.on(eventName, callback)
    
    return () => this.off(eventName, callback)
  }
  
  subscribeToAlerts(severity: string | null, callback: (data: AlertData) => void): () => void {
    const eventName = severity ? `alert:${severity}` : 'alert'
    this.on(eventName, callback)
    
    return () => this.off(eventName, callback)
  }
}