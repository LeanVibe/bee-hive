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
  private readonly pingInterval: number = 15000 // 15 seconds for better responsiveness
  private connectionQuality: 'excellent' | 'good' | 'poor' | 'offline' = 'offline'
  private latencyHistory: number[] = []
  private lastPingTime: number = 0
  private messageCount: number = 0
  private lastMessageTime: number = 0
  private streamingConfig: Map<string, number> = new Map()
  
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
    
    // Configure high-frequency streaming for mobile dashboard
    if (this.isConnected()) {
      this.enableHighFrequencyMode()
    }
  }
  
  private async connect(): Promise<void> {
    if (this.isConnecting) return
    
    try {
      this.isConnecting = true
      
      // Clear any existing connection
      this.cleanup()
      
      // Determine WebSocket URL - Using new dashboard endpoint
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const host = window.location.hostname
      const port = process.env.NODE_ENV === 'development' ? ':8000' : ''
      const wsUrl = `${protocol}//${host}${port}/api/dashboard/ws/dashboard`
      
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
    this.connectionQuality = 'good'
    
    // Send authentication message
    this.sendMessage({
      type: 'authenticate',
      data: {
        token: this.authService.getToken(),
        client_type: 'mobile_pwa',
        features: ['real_time_streaming', 'high_frequency_updates', 'mobile_optimization']
      }
    })
    
    // Start ping/pong with quality monitoring
    this.startPing()
    
    // Request immediate status update
    this.requestAgentStatus()
    this.requestSystemMetrics()
    
    this.emit('connected')
    this.emit('connection-quality', { quality: this.connectionQuality, timestamp: new Date().toISOString() })
  }
  
  private handleMessage(event: MessageEvent): void {
    try {
      // Track message rate for quality assessment
      this.messageCount++
      this.lastMessageTime = Date.now()
      
      const message: WebSocketMessage = JSON.parse(event.data)
      
      // Handle different message types
      switch (message.type) {
        case 'connection':
          console.log('🔌 Connection status:', message.data)
          this.emit('connection-status', message.data)
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
        case 'agent-status-changed':
          this.handleAgentEventMessage(message)
          break
          
        case 'task-updated':
        case 'task-created':
        case 'task-deleted':
          this.handleTaskEventMessage(message)
          break
          
        case 'system-event':
          this.handleSystemEventMessage(message)
          break
          
        case 'metrics-updated':
          this.handleMetricsUpdateMessage(message)
          break
          
        case 'pong':
          // Handle pong response - connection quality check
          this.handlePongMessage(message)
          break
          
        case 'error':
          console.error('WebSocket server error:', message.data)
          this.emit('error', new Error(message.data.message || 'Server error'))
          break
          
        case 'performance_update':
          // Handle performance updates with enhanced dashboard streaming
          this.emit('performance_update', message.data)
          this.emit('metrics-updated', message)
          break
          
        case 'keepalive':
          // Handle keepalive messages - track connection quality
          this.emit('keepalive', message)
          this.updateConnectionQuality('excellent')
          break
          
        default:
          console.log('📡 Unknown message type:', message.type, message)
          this.emit('unknown-message', message)
      }
      
      // Emit raw message for debugging and comprehensive event handling
      this.emit('message', message)
      
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error)
      this.emit('parse-error', { error, rawData: event.data })
    }
  }

  private handleTaskEventMessage(message: WebSocketMessage): void {
    // Enhanced task event handling for dashboard streaming
    const eventType = message.type
    const taskData = message.data
    
    // Emit specific task events
    this.emit(eventType, taskData)
    
    // Emit generic task update for dashboard components
    this.emit('task-event', { type: eventType, data: taskData })
    
    // Update connection quality based on message responsiveness
    this.updateConnectionQuality('good')
  }

  private handleSystemEventMessage(message: WebSocketMessage): void {
    // Enhanced system event handling
    const eventData = message.data
    
    this.emit('system-event', eventData)
    
    // Categorize system events for dashboard prioritization
    if (eventData.severity === 'critical' || eventData.severity === 'high') {
      this.emit('critical-system-event', eventData)
    }
  }

  private handleMetricsUpdateMessage(message: WebSocketMessage): void {
    // Enhanced metrics handling for real-time dashboard updates
    const metricsData = message.data
    
    this.emit('metrics-updated', metricsData)
    
    // Emit specific metric categories for dashboard optimization
    if (metricsData.agent_metrics) {
      this.emit('agent-metrics-update', metricsData.agent_metrics)
    }
    
    if (metricsData.system_metrics) {
      this.emit('system-metrics-update', metricsData.system_metrics)
    }
    
    if (metricsData.performance_snapshot) {
      this.emit('performance-snapshot', metricsData.performance_snapshot)
    }
  }

  private handlePongMessage(message: WebSocketMessage): void {
    // Calculate connection latency for quality assessment
    const now = Date.now()
    const latency = now - this.lastPingTime
    
    // Store latency history for trend analysis
    this.latencyHistory.push(latency)
    if (this.latencyHistory.length > 10) {
      this.latencyHistory.shift()
    }
    
    // Calculate average latency
    const avgLatency = this.latencyHistory.reduce((a, b) => a + b, 0) / this.latencyHistory.length
    
    // Update connection quality based on latency and stability
    const latencyStability = this.calculateLatencyStability()
    
    if (avgLatency < 50 && latencyStability > 0.8) {
      this.updateConnectionQuality('excellent')
    } else if (avgLatency < 150 && latencyStability > 0.6) {
      this.updateConnectionQuality('good')
    } else if (avgLatency < 500 && latencyStability > 0.4) {
      this.updateConnectionQuality('poor')
    } else {
      this.updateConnectionQuality('offline')
    }
    
    this.emit('connection-latency', { 
      latency, 
      avgLatency, 
      stability: latencyStability,
      quality: this.getConnectionQuality() 
    })
  }

  private updateConnectionQuality(quality: 'excellent' | 'good' | 'poor' | 'offline'): void {
    const previousQuality = this.connectionQuality
    this.connectionQuality = quality
    
    // Emit quality change event
    this.emit('connection-quality', { 
      quality, 
      previousQuality,
      latencyHistory: [...this.latencyHistory],
      messageRate: this.calculateMessageRate(),
      timestamp: new Date().toISOString() 
    })
    
    // Adjust streaming frequency based on connection quality
    this.adjustStreamingFrequency(quality)
  }

  private getConnectionQuality(): 'excellent' | 'good' | 'poor' | 'offline' {
    if (!this.isConnected()) return 'offline'
    return this.connectionQuality
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
        this.lastPingTime = Date.now()
        this.sendMessage({
          type: 'ping',
          timestamp: new Date().toISOString(),
          client_time: this.lastPingTime
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

  // Enhanced dashboard streaming subscriptions
  subscribeToAgentMetrics(callback: (data: any) => void): () => void {
    this.on('agent-metrics-update', callback)
    return () => this.off('agent-metrics-update', callback)
  }

  subscribeToSystemMetrics(callback: (data: any) => void): () => void {
    this.on('system-metrics-update', callback)
    return () => this.off('system-metrics-update', callback)
  }

  subscribeToPerformanceSnapshots(callback: (data: any) => void): () => void {
    this.on('performance-snapshot', callback)
    return () => this.off('performance-snapshot', callback)
  }

  subscribeToConnectionQuality(callback: (data: { quality: string, timestamp: string }) => void): () => void {
    this.on('connection-quality', callback)
    return () => this.off('connection-quality', callback)
  }

  subscribeToCriticalEvents(callback: (data: any) => void): () => void {
    this.on('critical-system-event', callback)
    return () => this.off('critical-system-event', callback)
  }

  // Remote control capabilities for agent management
  sendAgentCommand(agentId: string, command: string, parameters?: any): void {
    this.sendMessage({
      type: 'agent-command',
      data: {
        agentId,
        command,
        parameters,
        timestamp: new Date().toISOString()
      }
    })
  }

  sendBulkAgentCommand(agentIds: string[], command: string, parameters?: any): void {
    this.sendMessage({
      type: 'bulk-agent-command',
      data: {
        agentIds,
        command,
        parameters,
        timestamp: new Date().toISOString()
      }
    })
  }

  requestAgentStatus(agentId?: string): void {
    this.sendMessage({
      type: 'request-agent-status',
      data: {
        agentId: agentId || 'all',
        timestamp: new Date().toISOString()
      }
    })
  }

  requestSystemMetrics(): void {
    this.sendMessage({
      type: 'request-system-metrics',
      data: {
        timestamp: new Date().toISOString()
      }
    })
  }

  sendEmergencyStop(reason?: string): void {
    this.sendMessage({
      type: 'emergency-stop',
      data: {
        reason: reason || 'Manual emergency stop from dashboard',
        timestamp: new Date().toISOString()
      }
    })
  }

  // Stream configuration for dashboard optimization
  configureStreamingFrequency(metricType: string, frequencyMs: number): void {
    this.sendMessage({
      type: 'configure-streaming',
      data: {
        metricType,
        frequencyMs,
        timestamp: new Date().toISOString()
      }
    })
  }

  enableHighFrequencyMode(): void {
    this.configureStreamingFrequency('agent-metrics', 1000)
    this.configureStreamingFrequency('system-metrics', 2000)
    this.configureStreamingFrequency('performance-snapshots', 5000)
  }

  enableLowFrequencyMode(): void {
    this.configureStreamingFrequency('agent-metrics', 10000)
    this.configureStreamingFrequency('system-metrics', 15000)
    this.configureStreamingFrequency('performance-snapshots', 30000)
  }
  
  // New methods for enhanced real-time monitoring
  private calculateLatencyStability(): number {
    if (this.latencyHistory.length < 3) return 1.0
    
    const mean = this.latencyHistory.reduce((a, b) => a + b, 0) / this.latencyHistory.length
    const variance = this.latencyHistory.reduce((sum, latency) => sum + Math.pow(latency - mean, 2), 0) / this.latencyHistory.length
    const stdDev = Math.sqrt(variance)
    
    // Normalize stability score (lower variance = higher stability)
    return Math.max(0, 1 - (stdDev / mean))
  }
  
  private calculateMessageRate(): number {
    const now = Date.now()
    const timeDiff = now - this.lastMessageTime
    
    if (timeDiff > 0) {
      return (this.messageCount * 1000) / timeDiff // messages per second
    }
    return 0
  }
  
  private adjustStreamingFrequency(quality: 'excellent' | 'good' | 'poor' | 'offline'): void {
    switch (quality) {
      case 'excellent':
        this.enableHighFrequencyMode()
        break
      case 'good':
        this.configureStreamingFrequency('agent-metrics', 2000)
        this.configureStreamingFrequency('system-metrics', 5000)
        break
      case 'poor':
        this.enableLowFrequencyMode()
        break
      case 'offline':
        // Stop all streaming to conserve resources
        this.sendMessage({ type: 'pause-streaming' })
        break
    }
  }
  
  // Real-time dashboard specific methods
  enableMobileDashboardMode(): void {
    console.log('🚀 Enabling mobile dashboard real-time mode')
    
    // Configure optimal settings for mobile dashboard
    this.sendMessage({
      type: 'configure-client',
      data: {
        mode: 'mobile_dashboard',
        real_time: true,
        priority_events: ['agent_status', 'system_alerts', 'performance_metrics'],
        update_frequency: 'high',
        compression: true // Reduce bandwidth usage
      }
    })
    
    // Subscribe to critical real-time events
    this.subscribeToCriticalRealTimeEvents()
  }
  
  private subscribeToCriticalRealTimeEvents(): void {
    // Subscribe to high-priority events that need immediate mobile notifications
    const criticalEvents = [
      'agent_disconnected',
      'system_error',
      'performance_degradation',
      'security_alert',
      'task_completion',
      'human_intervention_required'
    ]
    
    this.sendMessage({
      type: 'subscribe-events',
      data: {
        events: criticalEvents,
        priority: 'high',
        mobile_optimized: true
      }
    })
  }
  
  // Connection quality monitoring
  getConnectionStats(): {
    quality: string
    latency: number
    stability: number
    messageRate: number
    reconnectAttempts: number
  } {
    return {
      quality: this.connectionQuality,
      latency: this.latencyHistory.length > 0 ? 
        this.latencyHistory.reduce((a, b) => a + b, 0) / this.latencyHistory.length : 0,
      stability: this.calculateLatencyStability(),
      messageRate: this.calculateMessageRate(),
      reconnectAttempts: this.reconnectAttempts
    }
  }
  
  // Emergency controls for mobile dashboard
  sendEmergencyPause(): void {
    this.sendMessage({
      type: 'emergency-pause',
      data: {
        reason: 'Mobile dashboard emergency pause',
        timestamp: new Date().toISOString()
      }
    })
  }
  
  sendEmergencyResume(): void {
    this.sendMessage({
      type: 'emergency-resume', 
      data: {
        reason: 'Mobile dashboard emergency resume',
        timestamp: new Date().toISOString()
      }
    })
  }
}