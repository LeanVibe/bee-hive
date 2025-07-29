/**
 * Real-Time Observability Event Service - VS 6.2
 * LeanVibe Agent Hive 2.0
 * 
 * High-performance event processing service for dashboard real-time updates:
 * - <1s event latency from backend to visualization
 * - Event filtering and routing for semantic intelligence
 * - Performance optimization for 1000+ events/second
 * - Integration with WebSocket manager and dashboard components
 */

import { ref, reactive, computed, watch } from 'vue'
import type { 
  DashboardEvent,
  DashboardComponent,
  WebSocketMessage,
  EventFilter,
  PerformanceMetrics
} from '@/types/coordination'
import { useUnifiedWebSocket } from './unifiedWebSocketManager'
import { api } from './api'

// Event types for dashboard streaming
export enum DashboardEventType {
  HOOK_EVENT = 'hook_event',
  WORKFLOW_UPDATE = 'workflow_update',
  SEMANTIC_INTELLIGENCE = 'semantic_intelligence',
  PERFORMANCE_METRIC = 'performance_metric',
  AGENT_STATUS = 'agent_status',
  SYSTEM_ALERT = 'system_alert',
  CONTEXT_FLOW = 'context_flow',
  INTELLIGENCE_KPI = 'intelligence_kpi'
}

// Event models
export interface ObservabilityEvent {
  id: string
  type: DashboardEventType
  source: string
  timestamp: string
  priority: number
  data: Record<string, any>
  semantic_embedding?: number[]
  semantic_concepts?: string[]
  context_references?: string[]
  latency_ms?: number
  correlation_id?: string
  visualization_hint?: string
  requires_acknowledgment?: boolean
}

export interface EventSubscription {
  id: string
  component: DashboardComponent
  eventTypes: DashboardEventType[]
  filters: EventFilter
  handler: (event: ObservabilityEvent) => void | Promise<void>
  priority: number
  active: boolean
}

export interface EventProcessingMetrics {
  events_received: number
  events_processed: number
  events_filtered: number
  events_failed: number
  average_latency_ms: number
  peak_latency_ms: number
  processing_rate_per_second: number
  buffer_size: number
  last_update: Date
}

export interface SemanticQueryRequest {
  query: string
  context_window_hours?: number
  max_results?: number
  similarity_threshold?: number
  include_context?: boolean
  include_performance?: boolean
}

export interface SemanticSearchResult {
  id: string
  event_type: string
  timestamp: string
  relevance_score: number
  content_summary: string
  agent_id?: string
  session_id?: string
  semantic_concepts: string[]
  context_references: string[]
  performance_metrics?: Record<string, any>
}

export interface ContextTrajectoryNode {
  id: string
  type: 'context' | 'event' | 'agent' | 'concept'
  label: string
  timestamp?: string
  metadata: Record<string, any>
  semantic_embedding?: number[]
  connections: string[]
}

export interface ContextTrajectoryPath {
  nodes: ContextTrajectoryNode[]
  edges: Array<{ source_id: string; target_id: string; relationship_type: string }>
  semantic_similarity: number
  path_strength: number
  temporal_flow: string[]
}

export interface IntelligenceKPI {
  name: string
  description: string
  current_value: number
  unit: string
  trend_direction: 'up' | 'down' | 'stable'
  trend_strength: number
  threshold_status: 'normal' | 'warning' | 'critical'
  historical_data: Array<{ timestamp: string; value: number; metadata?: any }>
  forecast_data?: Array<{ timestamp: string; value: number; confidence: number; is_forecast: boolean }>
}

export interface WorkflowConstellationNode {
  id: string
  type: 'agent' | 'concept' | 'session'
  label: string
  position: { x: number; y: number }
  size: number
  color: string
  metadata: Record<string, any>
}

export interface WorkflowConstellationEdge {
  source: string
  target: string
  type: 'communication' | 'semantic_flow' | 'context_sharing'
  strength: number
  frequency: number
  latency_ms?: number
  semantic_concepts: string[]
}

export interface WorkflowConstellation {
  nodes: WorkflowConstellationNode[]
  edges: WorkflowConstellationEdge[]
  semantic_flows: Array<Record<string, any>>
  temporal_data: Array<Record<string, any>>
  metadata: Record<string, any>
}

/**
 * High-performance observability event service for dashboard real-time streaming
 */
class ObservabilityEventService {
  private webSocketManager = useUnifiedWebSocket()
  private subscriptions = new Map<string, EventSubscription>()
  private eventBuffer: ObservabilityEvent[] = []
  private processingMetrics = reactive<EventProcessingMetrics>({
    events_received: 0,
    events_processed: 0,
    events_filtered: 0,
    events_failed: 0,
    average_latency_ms: 0,
    peak_latency_ms: 0,
    processing_rate_per_second: 0,
    buffer_size: 0,
    last_update: new Date()
  })

  // Configuration
  private config = {
    maxBufferSize: 1000,
    batchProcessingSize: 50,
    processingInterval: 100, // ms
    maxLatencyThreshold: 1000, // ms
    enablePerformanceMonitoring: true,
    enableSemanticProcessing: true,
    enableEventDeduplication: true
  }

  // Performance tracking
  private latencyMeasurements: number[] = []
  private eventTimestamps: number[] = []
  private processingTimer: number | null = null
  private isConnected = ref(false)
  private connectionEndpointId = 'observability_dashboard'

  // Event deduplication
  private processedEventIds = new Set<string>()
  private deduplicationWindow = 60000 // 1 minute

  constructor() {
    this.initializeWebSocketConnection()
    this.startPerformanceMonitoring()
  }

  /**
   * Initialize WebSocket connection for observability events
   */
  private async initializeWebSocketConnection() {
    try {
      // Register WebSocket endpoint for observability dashboard
      this.webSocketManager.registerEndpoint({
        id: this.connectionEndpointId,
        url: `ws://${window.location.host}/observability/dashboard/stream`,
        component: DashboardComponent.SERVICE,
        priority: 'high',
        autoReconnect: true,
        maxReconnectAttempts: 10,
        reconnectDelay: 1000
      })

      // Connect to WebSocket
      await this.webSocketManager.connect(this.connectionEndpointId)
      
      // Subscribe to all events
      this.webSocketManager.onMessage('*', (message: WebSocketMessage) => {
        this.handleIncomingEvent(message)
      }, DashboardComponent.SERVICE, 10)

      this.isConnected.value = true
      console.log('✅ Observability event service connected')

    } catch (error) {
      console.error('❌ Failed to initialize observability event service:', error)
      this.isConnected.value = false
    }
  }

  /**
   * Handle incoming WebSocket event
   */
  private async handleIncomingEvent(message: WebSocketMessage) {
    const startTime = performance.now()
    
    try {
      // Parse observability event
      const event = this.parseObservabilityEvent(message)
      if (!event) return

      // Event deduplication
      if (this.config.enableEventDeduplication && this.processedEventIds.has(event.id)) {
        this.processingMetrics.events_filtered++
        return
      }

      // Add to buffer
      this.addToBuffer(event)
      
      // Process event through subscriptions
      await this.processEventThroughSubscriptions(event)
      
      // Update metrics
      const latency = performance.now() - startTime
      this.updateProcessingMetrics(latency, true)
      
      // Mark as processed
      if (this.config.enableEventDeduplication) {
        this.processedEventIds.add(event.id)
      }

    } catch (error) {
      console.error('Failed to process observability event:', error)
      this.processingMetrics.events_failed++
      this.updateProcessingMetrics(performance.now() - startTime, false)
    }
  }

  /**
   * Parse WebSocket message into observability event
   */
  private parseObservabilityEvent(message: WebSocketMessage): ObservabilityEvent | null {
    try {
      if (message.type === 'hook_event' || message.type === 'dashboard_update') {
        return {
          id: message.data.id || `event_${Date.now()}_${Math.random()}`,
          type: this.mapMessageTypeToDashboardEventType(message.type),
          source: message.data.source || 'unknown',
          timestamp: message.timestamp || new Date().toISOString(),
          priority: message.data.priority || 5,
          data: message.data,
          semantic_embedding: message.data.semantic_embedding,
          semantic_concepts: message.data.semantic_concepts || [],
          context_references: message.data.context_references || [],
          latency_ms: message.data.latency_ms,
          correlation_id: message.data.correlation_id,
          visualization_hint: message.data.visualization_hint,
          requires_acknowledgment: message.data.requires_acknowledgment || false
        }
      }
      return null
    } catch (error) {
      console.error('Failed to parse observability event:', error)
      return null
    }
  }

  /**
   * Map WebSocket message type to dashboard event type
   */
  private mapMessageTypeToDashboardEventType(messageType: string): DashboardEventType {
    const typeMap: Record<string, DashboardEventType> = {
      'hook_event': DashboardEventType.HOOK_EVENT,
      'workflow_update': DashboardEventType.WORKFLOW_UPDATE,
      'semantic_intelligence': DashboardEventType.SEMANTIC_INTELLIGENCE,
      'performance_metric': DashboardEventType.PERFORMANCE_METRIC,
      'agent_status': DashboardEventType.AGENT_STATUS,
      'system_alert': DashboardEventType.SYSTEM_ALERT,
      'context_flow': DashboardEventType.CONTEXT_FLOW,
      'intelligence_kpi': DashboardEventType.INTELLIGENCE_KPI
    }
    
    return typeMap[messageType] || DashboardEventType.HOOK_EVENT
  }

  /**
   * Add event to processing buffer
   */
  private addToBuffer(event: ObservabilityEvent) {
    this.eventBuffer.push(event)
    this.processingMetrics.events_received++
    
    // Manage buffer size
    if (this.eventBuffer.length > this.config.maxBufferSize) {
      this.eventBuffer.shift() // Remove oldest event
    }
    
    this.processingMetrics.buffer_size = this.eventBuffer.length
  }

  /**
   * Process event through active subscriptions
   */
  private async processEventThroughSubscriptions(event: ObservabilityEvent) {
    const relevantSubscriptions = Array.from(this.subscriptions.values())
      .filter(sub => sub.active && this.eventMatchesSubscription(event, sub))
      .sort((a, b) => b.priority - a.priority) // Higher priority first

    // Process subscriptions in parallel batches
    const batchSize = 10
    for (let i = 0; i < relevantSubscriptions.length; i += batchSize) {
      const batch = relevantSubscriptions.slice(i, i + batchSize)
      
      await Promise.allSettled(
        batch.map(async subscription => {
          try {
            await subscription.handler(event)
          } catch (error) {
            console.error(`Subscription handler error for ${subscription.id}:`, error)
          }
        })
      )
    }

    this.processingMetrics.events_processed++
  }

  /**
   * Check if event matches subscription filters
   */
  private eventMatchesSubscription(event: ObservabilityEvent, subscription: EventSubscription): boolean {
    // Event type filter
    if (subscription.eventTypes.length > 0 && !subscription.eventTypes.includes(event.type)) {
      return false
    }

    // Priority filter
    if (subscription.filters.min_priority && event.priority < subscription.filters.min_priority) {
      return false
    }

    // Agent ID filter
    if (subscription.filters.agent_ids?.length && event.data.agent_id) {
      if (!subscription.filters.agent_ids.includes(event.data.agent_id)) {
        return false
      }
    }

    // Session ID filter
    if (subscription.filters.session_ids?.length && event.data.session_id) {
      if (!subscription.filters.session_ids.includes(event.data.session_id)) {
        return false
      }
    }

    // Semantic concepts filter
    if (subscription.filters.semantic_concepts?.length && event.semantic_concepts) {
      const hasMatchingConcept = subscription.filters.semantic_concepts.some(
        concept => event.semantic_concepts!.includes(concept)
      )
      if (!hasMatchingConcept) {
        return false
      }
    }

    return true
  }

  /**
   * Update processing metrics
   */
  private updateProcessingMetrics(latency: number, success: boolean) {
    // Track latency
    this.latencyMeasurements.push(latency)
    if (this.latencyMeasurements.length > 1000) {
      this.latencyMeasurements.shift()
    }

    // Update peak latency
    this.processingMetrics.peak_latency_ms = Math.max(
      this.processingMetrics.peak_latency_ms,
      latency
    )

    // Calculate average latency
    this.processingMetrics.average_latency_ms = 
      this.latencyMeasurements.reduce((sum, val) => sum + val, 0) / 
      this.latencyMeasurements.length

    // Track event timestamps for rate calculation
    const now = Date.now()
    this.eventTimestamps.push(now)
    
    // Keep only last minute of timestamps
    const oneMinuteAgo = now - 60000
    this.eventTimestamps = this.eventTimestamps.filter(ts => ts > oneMinuteAgo)
    
    // Calculate processing rate
    this.processingMetrics.processing_rate_per_second = this.eventTimestamps.length / 60

    this.processingMetrics.last_update = new Date()

    // Performance alerts
    if (latency > this.config.maxLatencyThreshold) {
      console.warn(`⚠️ High event processing latency: ${latency.toFixed(2)}ms`)
    }
  }

  /**
   * Start performance monitoring
   */
  private startPerformanceMonitoring() {
    this.processingTimer = setInterval(() => {
      // Clean up old deduplication entries
      if (this.config.enableEventDeduplication) {
        const cutoff = Date.now() - this.deduplicationWindow
        // In a real implementation, would need timestamp tracking for cleanup
      }

      // Performance logging
      if (this.config.enablePerformanceMonitoring) {
        console.debug('📊 Event processing metrics:', {
          received: this.processingMetrics.events_received,
          processed: this.processingMetrics.events_processed,
          rate: this.processingMetrics.processing_rate_per_second.toFixed(2),
          avg_latency: this.processingMetrics.average_latency_ms.toFixed(2),
          buffer_size: this.processingMetrics.buffer_size
        })
      }
    }, 30000) // Every 30 seconds
  }

  /**
   * Subscribe to specific event types with filters
   */
  public subscribe(
    component: DashboardComponent,
    eventTypes: DashboardEventType[],
    handler: (event: ObservabilityEvent) => void | Promise<void>,
    filters: Partial<EventFilter> = {},
    priority = 1
  ): string {
    const subscriptionId = `${component}_${Date.now()}_${Math.random()}`
    
    const subscription: EventSubscription = {
      id: subscriptionId,
      component,
      eventTypes,
      filters: {
        min_priority: 1,
        max_latency_ms: 1000,
        ...filters
      },
      handler,
      priority,
      active: true
    }

    this.subscriptions.set(subscriptionId, subscription)
    
    console.log(`📡 Event subscription created: ${subscriptionId}`, {
      component,
      eventTypes,
      filters
    })

    return subscriptionId
  }

  /**
   * Unsubscribe from events
   */
  public unsubscribe(subscriptionId: string): boolean {
    const success = this.subscriptions.delete(subscriptionId)
    if (success) {
      console.log(`📡 Event subscription removed: ${subscriptionId}`)
    }
    return success
  }

  /**
   * Update subscription filters
   */
  public updateSubscriptionFilters(subscriptionId: string, filters: Partial<EventFilter>): boolean {
    const subscription = this.subscriptions.get(subscriptionId)
    if (subscription) {
      subscription.filters = { ...subscription.filters, ...filters }
      console.log(`📡 Subscription filters updated: ${subscriptionId}`, filters)
      return true
    }
    return false
  }

  /**
   * Get recent events from buffer
   */
  public getRecentEvents(
    count = 50,
    eventTypes?: DashboardEventType[],
    filters?: Partial<EventFilter>
  ): ObservabilityEvent[] {
    let events = [...this.eventBuffer]

    // Apply event type filter
    if (eventTypes?.length) {
      events = events.filter(event => eventTypes.includes(event.type))
    }

    // Apply additional filters
    if (filters) {
      events = events.filter(event => {
        if (filters.min_priority && event.priority < filters.min_priority) return false
        if (filters.agent_ids?.length && !filters.agent_ids.includes(event.data.agent_id)) return false
        if (filters.session_ids?.length && !filters.session_ids.includes(event.data.session_id)) return false
        return true
      })
    }

    // Sort by timestamp (newest first) and limit
    return events
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, count)
  }

  /**
   * Perform semantic search
   */
  public async performSemanticSearch(request: SemanticQueryRequest): Promise<SemanticSearchResult[]> {
    try {
      const response = await api.post('/observability/dashboard/semantic-search', request)
      return response.data
    } catch (error) {
      console.error('Semantic search failed:', error)
      throw error
    }
  }

  /**
   * Get context trajectory
   */
  public async getContextTrajectory(params: {
    context_id?: string
    concept?: string
    agent_id?: string
    session_id?: string
    time_range_hours?: number
    max_depth?: number
  }): Promise<ContextTrajectoryPath[]> {
    try {
      const response = await api.post('/observability/dashboard/context-trajectory', params)
      return response.data
    } catch (error) {
      console.error('Context trajectory request failed:', error)
      throw error
    }
  }

  /**
   * Get intelligence KPIs
   */
  public async getIntelligenceKPIs(params: {
    kpi_names?: string[]
    time_range_hours?: number
    aggregation_interval?: string
    include_trends?: boolean
    include_forecasts?: boolean
  }): Promise<IntelligenceKPI[]> {
    try {
      const response = await api.post('/observability/dashboard/intelligence-kpis', params)
      return response.data
    } catch (error) {
      console.error('Intelligence KPIs request failed:', error)
      throw error
    }
  }

  /**
   * Get workflow constellation
   */
  public async getWorkflowConstellation(params: {
    session_ids?: string[]
    agent_ids?: string[]
    time_range_hours?: number
    include_semantic_flow?: boolean
    min_interaction_count?: number
  }): Promise<WorkflowConstellation> {
    try {
      const response = await api.post('/observability/dashboard/workflow-constellation', params)
      return response.data
    } catch (error) {
      console.error('Workflow constellation request failed:', error)
      throw error
    }
  }

  /**
   * Get performance summary
   */
  public async getPerformanceSummary(timeRangeHours = 24): Promise<any> {
    try {
      const response = await api.get('/observability/dashboard/performance-summary', {
        params: { time_range_hours: timeRangeHours }
      })
      return response.data
    } catch (error) {
      console.error('Performance summary request failed:', error)
      throw error
    }
  }

  /**
   * Get current processing metrics
   */
  public getProcessingMetrics(): EventProcessingMetrics {
    return { ...this.processingMetrics }
  }

  /**
   * Get connection status
   */
  public getConnectionStatus() {
    return {
      isConnected: this.isConnected.value,
      endpointId: this.connectionEndpointId,
      subscriptionCount: this.subscriptions.size,
      bufferSize: this.eventBuffer.length,
      metrics: this.getProcessingMetrics()
    }
  }

  /**
   * Clear event buffer
   */
  public clearBuffer(): void {
    this.eventBuffer.length = 0
    this.processingMetrics.buffer_size = 0
    console.log('📊 Event buffer cleared')
  }

  /**
   * Update service configuration
   */
  public updateConfig(newConfig: Partial<typeof this.config>): void {
    this.config = { ...this.config, ...newConfig }
    console.log('⚙️ Event service configuration updated:', newConfig)
  }

  /**
   * Disconnect and cleanup
   */
  public async disconnect(): Promise<void> {
    try {
      // Clear timer
      if (this.processingTimer) {
        clearInterval(this.processingTimer)
        this.processingTimer = null
      }

      // Clear subscriptions
      this.subscriptions.clear()

      // Disconnect WebSocket
      this.webSocketManager.disconnect(this.connectionEndpointId)
      this.isConnected.value = false

      console.log('🔌 Observability event service disconnected')
    } catch (error) {
      console.error('Failed to disconnect observability event service:', error)
    }
  }
}

// Create singleton instance
export const observabilityEventService = new ObservabilityEventService()

// Vue composable for easy integration
export function useObservabilityEvents() {
  return {
    // Service instance
    service: observabilityEventService,
    
    // Connection status
    isConnected: computed(() => observabilityEventService.getConnectionStatus().isConnected),
    
    // Metrics
    metrics: computed(() => observabilityEventService.getProcessingMetrics()),
    
    // Methods
    subscribe: observabilityEventService.subscribe.bind(observabilityEventService),
    unsubscribe: observabilityEventService.unsubscribe.bind(observabilityEventService),
    getRecentEvents: observabilityEventService.getRecentEvents.bind(observabilityEventService),
    performSemanticSearch: observabilityEventService.performSemanticSearch.bind(observabilityEventService),
    getContextTrajectory: observabilityEventService.getContextTrajectory.bind(observabilityEventService),
    getIntelligenceKPIs: observabilityEventService.getIntelligenceKPIs.bind(observabilityEventService),
    getWorkflowConstellation: observabilityEventService.getWorkflowConstellation.bind(observabilityEventService),
    getPerformanceSummary: observabilityEventService.getPerformanceSummary.bind(observabilityEventService),
    clearBuffer: observabilityEventService.clearBuffer.bind(observabilityEventService),
    updateConfig: observabilityEventService.updateConfig.bind(observabilityEventService)
  }
}

export default observabilityEventService