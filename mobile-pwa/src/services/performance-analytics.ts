/**
 * Performance Analytics Service
 * 
 * Service for real-time performance monitoring, data fetching, and WebSocket connections
 * Integrates with the LeanVibe Agent Hive performance intelligence API
 */

import { EventEmitter } from '../utils/event-emitter'
import { backendAdapter } from './backend-adapter'
import type { PerformanceData, TimeRange } from '../components/dashboard/enhanced-performance-analytics-panel'

interface PerformanceConfig {
  autoRefreshInterval: number
  reconnectInterval: number
  maxReconnectAttempts: number
  alertThresholds: {
    api_response_time: { warning: number; critical: number }
    cpu_usage: { warning: number; critical: number }
    memory_usage: { warning: number; critical: number }
    error_rate: { warning: number; critical: number }
  }
}

interface PerformanceMetricsResponse {
  dashboard_id: string
  generated_at: string
  time_window_minutes: number
  data: {
    system_health: {
      overall_score: number
      status: string
    }
    real_time_metrics: {
      system_cpu_percent: number
      memory_usage_mb: number
      search_avg_latency_ms: number
      agent_task_completion_rate: number
      http_requests_per_second: number
      error_rate_percentage: number
      redis_memory_usage_mb: number
      database_connection_count: number
      queue_length: number
    }
    performance_alerts: Array<{
      alert_id: string
      type: string
      metric: string
      component: string
      severity: string
      time_to_threshold_hours?: number
      confidence: number
      description: string
      recommendations: string[]
    }>
    capacity_metrics: {
      cpu_utilization: number
      memory_utilization: number
      network_utilization: number
      disk_utilization: number
      agent_pool_usage: number
      database_pool_usage: number
    }
  }
  metadata: {
    version: string
    features_enabled: {
      predictions: boolean
      anomalies: boolean
      real_time: boolean
    }
  }
}

export class PerformanceAnalyticsService extends EventEmitter {
  private static instance: PerformanceAnalyticsService
  private config: PerformanceConfig
  private wsConnection: WebSocket | null = null
  private reconnectAttempts = 0
  private autoRefreshTimer: number | null = null
  private lastMetrics: PerformanceData | null = null
  private isInitialized = false
  private connectionStatus: 'connected' | 'disconnected' | 'connecting' = 'disconnected'
  
  constructor() {
    super()
    this.config = {
      autoRefreshInterval: 5000, // 5 seconds
      reconnectInterval: 3000, // 3 seconds
      maxReconnectAttempts: 10,
      alertThresholds: {
        api_response_time: { warning: 500, critical: 1000 },
        cpu_usage: { warning: 70, critical: 90 },
        memory_usage: { warning: 80, critical: 95 },
        error_rate: { warning: 1, critical: 5 }
      }
    }
  }
  
  static getInstance(): PerformanceAnalyticsService {
    if (!PerformanceAnalyticsService.instance) {
      PerformanceAnalyticsService.instance = new PerformanceAnalyticsService()
    }
    return PerformanceAnalyticsService.instance
  }
  
  async initialize(): Promise<void> {
    if (this.isInitialized) return
    
    try {
      console.log('🔧 Initializing Performance Analytics Service...')
      
      // Initial data fetch
      await this.fetchInitialData()
      
      // Set up auto-refresh
      this.startAutoRefresh()
      
      // Try to establish WebSocket connection
      this.connectWebSocket()
      
      this.isInitialized = true
      this.emit('initialized')
      
      console.log('✅ Performance Analytics Service initialized')
    } catch (error) {
      console.error('❌ Failed to initialize Performance Analytics Service:', error)
      this.emit('error', error)
      throw error
    }
  }
  
  async fetchPerformanceData(timeRange: TimeRange = '1h', forceRefresh = false): Promise<PerformanceData> {
    try {
      const timeWindowMinutes = this.convertTimeRangeToMinutes(timeRange)
      
      console.log(`📊 Fetching performance data for ${timeRange} (${timeWindowMinutes} minutes)...`)
      
      const response = await backendAdapter.get<PerformanceMetricsResponse>(
        '/api/v1/performance/dashboard/realtime',
        {
          params: {
            time_window_minutes: timeWindowMinutes,
            refresh_cache: forceRefresh,
            include_predictions: true,
            include_anomalies: true
          }
        }
      )
      
      if (!response.success || !response.data) {
        throw new Error(`Failed to fetch performance data: ${response.error?.message || 'Unknown error'}`)
      }
      
      const performanceData = this.transformApiResponse(response.data)
      this.lastMetrics = performanceData
      
      // Emit data update event
      this.emit('data-updated', performanceData)
      
      // Check for alerts and emit if needed
      if (performanceData.alerts.length > 0) {
        this.emit('alerts-updated', performanceData.alerts)
      }
      
      return performanceData
      
    } catch (error) {
      console.error('❌ Error fetching performance data:', error)
      this.emit('error', error)
      
      // Return last known data if available, otherwise empty data
      if (this.lastMetrics) {
        console.log('⚠️ Returning cached performance data due to fetch error')
        return this.lastMetrics
      }
      
      throw error
    }
  }
  
  async fetchPerformanceTrends(timeRange: TimeRange = '24h'): Promise<any> {
    try {
      const timeWindowHours = this.convertTimeRangeToHours(timeRange)
      
      const response = await backendAdapter.get(
        '/api/v1/performance/metrics/trends',
        {
          params: {
            time_window_hours: timeWindowHours,
            aggregation_interval: this.getAggregationInterval(timeRange)
          }
        }
      )
      
      if (!response.success) {
        throw new Error(`Failed to fetch performance trends: ${response.error?.message}`)
      }
      
      return response.data
      
    } catch (error) {
      console.error('❌ Error fetching performance trends:', error)
      this.emit('error', error)
      throw error
    }
  }
  
  async detectAnomalies(timeWindowHours = 1, sensitivity = 0.85): Promise<any> {
    try {
      const response = await backendAdapter.post(
        '/api/v1/performance/detect/anomalies',
        {
          time_window_hours: timeWindowHours,
          sensitivity,
          metric_filter: null
        }
      )
      
      if (!response.success) {
        throw new Error(`Failed to detect anomalies: ${response.error?.message}`)
      }
      
      return response.data
      
    } catch (error) {
      console.error('❌ Error detecting anomalies:', error)
      this.emit('error', error)
      throw error
    }
  }
  
  async getOptimizationRecommendations(component?: string): Promise<any> {
    try {
      const response = await backendAdapter.post(
        '/api/v1/performance/optimize/recommendations',
        {
          component,
          priority_level: 'all',
          include_cost_analysis: true
        }
      )
      
      if (!response.success) {
        throw new Error(`Failed to get optimization recommendations: ${response.error?.message}`)
      }
      
      return response.data
      
    } catch (error) {
      console.error('❌ Error getting optimization recommendations:', error)
      this.emit('error', error)
      throw error
    }
  }
  
  private async fetchInitialData(): Promise<void> {
    try {
      await this.fetchPerformanceData('1h')
    } catch (error) {
      // Create fallback data if initial fetch fails
      this.lastMetrics = this.createFallbackData()
      console.log('⚠️ Using fallback performance data')
    }
  }
  
  private startAutoRefresh(): void {
    if (this.autoRefreshTimer) {
      clearInterval(this.autoRefreshTimer)
    }
    
    this.autoRefreshTimer = window.setInterval(async () => {
      if (this.connectionStatus === 'disconnected') {
        try {
          await this.fetchPerformanceData('1h')
        } catch (error) {
          console.warn('⚠️ Auto-refresh failed, will retry next cycle')
        }
      }
    }, this.config.autoRefreshInterval)
  }
  
  private stopAutoRefresh(): void {
    if (this.autoRefreshTimer) {
      clearInterval(this.autoRefreshTimer)
      this.autoRefreshTimer = null
    }
  }
  
  private connectWebSocket(): void {
    try {
      const baseUrl = backendAdapter.getBaseUrl()
      const wsUrl = baseUrl.replace(/^http/, 'ws') + '/ws/performance/realtime'
      
      console.log('🔌 Connecting to performance WebSocket:', wsUrl)
      
      this.connectionStatus = 'connecting'
      this.emit('connection-status-changed', this.connectionStatus)
      
      this.wsConnection = new WebSocket(wsUrl)
      
      this.wsConnection.onopen = () => {
        console.log('✅ Performance WebSocket connected')
        this.connectionStatus = 'connected'
        this.reconnectAttempts = 0
        this.emit('connection-status-changed', this.connectionStatus)
        this.emit('websocket-connected')
      }
      
      this.wsConnection.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          this.handleWebSocketMessage(data)
        } catch (error) {
          console.error('❌ Error parsing WebSocket message:', error)
        }
      }
      
      this.wsConnection.onclose = (event) => {
        console.log('🔌 Performance WebSocket closed:', event.code, event.reason)
        this.connectionStatus = 'disconnected'
        this.emit('connection-status-changed', this.connectionStatus)
        this.emit('websocket-disconnected', event)
        
        // Attempt to reconnect if not explicitly closed
        if (event.code !== 1000 && this.reconnectAttempts < this.config.maxReconnectAttempts) {
          this.scheduleReconnect()
        }
      }
      
      this.wsConnection.onerror = (error) => {
        console.error('❌ Performance WebSocket error:', error)
        this.emit('websocket-error', error)
      }
      
    } catch (error) {
      console.error('❌ Failed to connect WebSocket:', error)
      this.connectionStatus = 'disconnected'
      this.emit('connection-status-changed', this.connectionStatus)
    }
  }
  
  private scheduleReconnect(): void {
    this.reconnectAttempts++
    const delay = Math.min(this.config.reconnectInterval * this.reconnectAttempts, 30000) // Max 30 seconds
    
    console.log(`🔄 Scheduling WebSocket reconnect attempt ${this.reconnectAttempts}/${this.config.maxReconnectAttempts} in ${delay}ms`)
    
    setTimeout(() => {
      if (this.connectionStatus === 'disconnected') {
        this.connectWebSocket()
      }
    }, delay)
  }
  
  private handleWebSocketMessage(data: any): void {
    switch (data.type) {
      case 'performance_update':
        if (data.payload) {
          const performanceData = this.transformApiResponse(data.payload)
          this.lastMetrics = performanceData
          this.emit('real-time-update', performanceData)
        }
        break
        
      case 'alert':
        this.emit('real-time-alert', data.payload)
        break
        
      case 'anomaly_detected':
        this.emit('anomaly-detected', data.payload)
        break
        
      case 'regression_detected':
        this.emit('regression-detected', data.payload)
        break
        
      default:
        console.log('📩 Unknown WebSocket message type:', data.type)
    }
  }
  
  private transformApiResponse(apiResponse: PerformanceMetricsResponse): PerformanceData {
    const metrics = apiResponse.data.real_time_metrics
    const capacity = apiResponse.data.capacity_metrics
    const alerts = apiResponse.data.performance_alerts || []
    
    return {
      timestamp: apiResponse.generated_at,
      system_metrics: {
        cpu_usage: capacity.cpu_utilization,
        memory_usage: capacity.memory_utilization,
        network_usage: capacity.network_utilization,
        disk_usage: capacity.disk_utilization
      },
      response_times: {
        api_response_time: metrics.search_avg_latency_ms,
        api_p95_response_time: metrics.search_avg_latency_ms * 1.2, // Estimate
        api_p99_response_time: metrics.search_avg_latency_ms * 1.5, // Estimate
        websocket_latency: 50, // Default estimate
        database_query_time: 25 // Default estimate
      },
      throughput: {
        requests_per_second: metrics.http_requests_per_second,
        peak_rps: metrics.http_requests_per_second * 1.3, // Estimate
        tasks_completed_per_hour: metrics.agent_task_completion_rate * 60,
        agent_operations_per_minute: metrics.agent_task_completion_rate
      },
      error_rates: {
        http_4xx_rate: metrics.error_rate_percentage * 0.7, // Estimate split
        http_5xx_rate: metrics.error_rate_percentage * 0.3, // Estimate split
        system_error_rate: metrics.error_rate_percentage * 0.1,
        total_error_rate: metrics.error_rate_percentage
      },
      capacity_metrics: {
        queue_length: metrics.queue_length,
        connection_pool_usage: capacity.database_pool_usage,
        thread_pool_usage: capacity.agent_pool_usage,
        bottlenecks: [] // Will be populated from specific analysis
      },
      alerts: alerts.map(alert => ({
        id: alert.alert_id,
        type: this.mapAlertType(alert.type),
        severity: this.mapSeverity(alert.severity),
        message: alert.description,
        timestamp: apiResponse.generated_at,
        metric: alert.metric,
        current_value: 0, // Will be filled from metrics
        threshold_value: 0, // Will be filled from thresholds
        impact_assessment: alert.recommendations.join('; ')
      }))
    }
  }
  
  private mapAlertType(type: string): 'performance' | 'threshold' | 'anomaly' | 'regression' {
    switch (type.toLowerCase()) {
      case 'anomaly': return 'anomaly'
      case 'regression': return 'regression'
      case 'threshold': return 'threshold'
      default: return 'performance'
    }
  }
  
  private mapSeverity(severity: string): 'critical' | 'warning' | 'info' {
    switch (severity.toLowerCase()) {
      case 'critical':
      case 'high': return 'critical'
      case 'warning':
      case 'medium': return 'warning'
      default: return 'info'
    }
  }
  
  private createFallbackData(): PerformanceData {
    return {
      timestamp: new Date().toISOString(),
      system_metrics: {
        cpu_usage: 65,
        memory_usage: 72,
        network_usage: 35,
        disk_usage: 45
      },
      response_times: {
        api_response_time: 245,
        api_p95_response_time: 320,
        api_p99_response_time: 450,
        websocket_latency: 25,
        database_query_time: 15
      },
      throughput: {
        requests_per_second: 850,
        peak_rps: 1100,
        tasks_completed_per_hour: 3600,
        agent_operations_per_minute: 60
      },
      error_rates: {
        http_4xx_rate: 1.2,
        http_5xx_rate: 0.3,
        system_error_rate: 0.1,
        total_error_rate: 1.6
      },
      capacity_metrics: {
        queue_length: 12,
        connection_pool_usage: 68,
        thread_pool_usage: 45,
        bottlenecks: []
      },
      alerts: []
    }
  }
  
  private convertTimeRangeToMinutes(timeRange: TimeRange): number {
    switch (timeRange) {
      case '1m': return 1
      case '5m': return 5
      case '15m': return 15
      case '1h': return 60
      case '6h': return 360
      case '24h': return 1440
      case '7d': return 10080
      default: return 60
    }
  }
  
  private convertTimeRangeToHours(timeRange: TimeRange): number {
    return Math.ceil(this.convertTimeRangeToMinutes(timeRange) / 60)
  }
  
  private getAggregationInterval(timeRange: TimeRange): string {
    switch (timeRange) {
      case '1m':
      case '5m':
      case '15m': return '1m'
      case '1h':
      case '6h': return '5m'
      case '24h': return '1h'
      case '7d': return '1h'
      default: return '5m'
    }
  }
  
  // Public API methods
  
  setAutoRefresh(enabled: boolean): void {
    if (enabled) {
      this.startAutoRefresh()
    } else {
      this.stopAutoRefresh()
    }
  }
  
  getConnectionStatus(): string {
    return this.connectionStatus
  }
  
  getLastMetrics(): PerformanceData | null {
    return this.lastMetrics
  }
  
  reconnectWebSocket(): void {
    if (this.wsConnection) {
      this.wsConnection.close()
    }
    this.reconnectAttempts = 0
    this.connectWebSocket()
  }
  
  updateConfig(newConfig: Partial<PerformanceConfig>): void {
    this.config = { ...this.config, ...newConfig }
    
    // Restart auto-refresh if interval changed
    if (newConfig.autoRefreshInterval && this.autoRefreshTimer) {
      this.startAutoRefresh()
    }
  }
  
  destroy(): void {
    console.log('🧹 Destroying Performance Analytics Service...')
    
    this.stopAutoRefresh()
    
    if (this.wsConnection) {
      this.wsConnection.close(1000, 'Service destroyed')
      this.wsConnection = null
    }
    
    this.removeAllListeners()
    this.isInitialized = false
    
    console.log('✅ Performance Analytics Service destroyed')
  }
}

// Export singleton instance
export const performanceAnalyticsService = PerformanceAnalyticsService.getInstance()