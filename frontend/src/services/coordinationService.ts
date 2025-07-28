/**
 * Coordination Service for Multi-Agent Dashboard Integration
 * 
 * Provides unified data management and cross-component communication
 * for the coordination dashboard with real-time synchronization.
 */

import { ref, reactive, computed, watch } from 'vue'
import { api } from './api'
import type { 
  AgentInfo, 
  SessionInfo, 
  ConversationEvent, 
  PerformanceMetrics,
  DetectedPattern,
  DashboardState 
} from '@/types/coordination'

export interface CoordinationFilter {
  sessionIds: string[]
  agentIds: string[]
  eventTypes: string[]
  timeRange: {
    start?: string
    end?: string
  }
  includeInactive: boolean
}

export interface GraphUpdateEvent {
  type: 'node_added' | 'node_updated' | 'node_removed' | 'edge_added' | 'edge_updated' | 'edge_removed'
  data: any
  timestamp: string
  sessionId: string
}

export interface TranscriptUpdateEvent {
  type: 'message_added' | 'message_updated' | 'pattern_detected'
  data: ConversationEvent | DetectedPattern
  timestamp: string
  sessionId: string
}

export interface CrossComponentEvent {
  source: 'graph' | 'transcript' | 'analysis' | 'monitoring'
  target: 'graph' | 'transcript' | 'analysis' | 'monitoring' | 'all'
  type: string
  data: any
  timestamp: string
}

class CoordinationService {
  // State management
  private state = reactive<DashboardState>({
    selectedSession: 'all',
    activeFilters: {
      sessionIds: [],
      agentIds: [],
      eventTypes: [],
      timeRange: {},
      includeInactive: false
    },
    graphData: {
      nodes: [],
      edges: [],
      stats: {},
      sessionColors: {}
    },
    transcriptData: {
      events: [],
      totalEvents: 0,
      metadata: {},
      agentSummary: {}
    },
    analysisData: {
      patterns: [],
      metrics: {},
      recommendations: []
    },
    performance: {
      responseTime: 0,
      errorRate: 0,
      throughput: 0,
      agentMetrics: {}
    },
    connections: {
      websocket: false,
      api: true,
      lastUpdate: null
    }
  })

  // Event system for cross-component communication
  private eventListeners = new Map<string, Array<(event: CrossComponentEvent) => void>>()
  private websocketConnections = new Map<string, WebSocket>()
  
  // Data cache and synchronization
  private dataCache = new Map<string, { data: any, timestamp: number, ttl: number }>()
  private syncQueue: Array<() => Promise<void>> = []
  private isSyncing = false

  // Public reactive state
  public readonly selectedSession = computed(() => this.state.selectedSession)
  public readonly activeFilters = computed(() => this.state.activeFilters)
  public readonly graphData = computed(() => this.state.graphData)
  public readonly transcriptData = computed(() => this.state.transcriptData)
  public readonly analysisData = computed(() => this.state.analysisData)
  public readonly performance = computed(() => this.state.performance)
  public readonly connections = computed(() => this.state.connections)

  constructor() {
    this.initializeWatchers()
  }

  /**
   * Initialize reactive watchers for cross-component synchronization
   */
  private initializeWatchers() {
    // Watch for session changes and sync all components
    watch(() => this.state.selectedSession, (newSession, oldSession) => {
      if (newSession !== oldSession) {
        this.emitEvent('all', 'session_changed', { 
          newSession, 
          oldSession,
          filters: this.state.activeFilters 
        })
        this.syncAllComponents()
      }
    })

    // Watch for filter changes and update components
    watch(() => this.state.activeFilters, (newFilters, oldFilters) => {
      if (JSON.stringify(newFilters) !== JSON.stringify(oldFilters)) {
        this.emitEvent('all', 'filters_changed', { 
          newFilters, 
          oldFilters 
        })
        this.debouncedSync()
      }
    }, { deep: true })
  }

  /**
   * Session Management
   */
  public async setSelectedSession(sessionId: string) {
    this.state.selectedSession = sessionId
    return this.syncAllComponents()
  }

  public getSelectedSession(): string {
    return this.state.selectedSession
  }

  public async getAvailableSessions(): Promise<SessionInfo[]> {
    const cacheKey = 'available_sessions'
    const cached = this.getCachedData(cacheKey)
    
    if (cached) {
      return cached
    }

    try {
      const response = await api.get('/coordination/sessions')
      const sessions = response.data.active_sessions.map((sessionId: string) => ({
        id: sessionId,
        label: `Session ${sessionId.substring(0, 8)}...`,
        agentCount: response.data.session_stats[sessionId]?.agents || 0,
        lastActivity: response.data.session_stats[sessionId]?.last_activity
      }))

      this.setCachedData(cacheKey, sessions, 30000) // 30 second TTL
      return sessions
    } catch (error) {
      console.error('Failed to load available sessions:', error)
      return []
    }
  }

  /**
   * Filter Management
   */
  public setFilters(filters: Partial<CoordinationFilter>) {
    Object.assign(this.state.activeFilters, filters)
  }

  public getFilters(): CoordinationFilter {
    return { ...this.state.activeFilters }
  }

  public resetFilters() {
    this.state.activeFilters = {
      sessionIds: [],
      agentIds: [],
      eventTypes: [],
      timeRange: {},
      includeInactive: false
    }
  }

  /**
   * Graph Data Management
   */
  public async getGraphData(forceRefresh = false): Promise<any> {
    const cacheKey = `graph_data_${this.state.selectedSession}_${JSON.stringify(this.state.activeFilters)}`
    
    if (!forceRefresh) {
      const cached = this.getCachedData(cacheKey)
      if (cached) {
        this.state.graphData = cached
        return cached
      }
    }

    try {
      const params = this.buildApiParams()
      const response = await api.get('/coordination/graph-data', { params })
      
      const graphData = {
        nodes: response.data.nodes,
        edges: response.data.edges,
        stats: response.data.stats,
        sessionColors: response.data.session_colors
      }

      this.state.graphData = graphData
      this.setCachedData(cacheKey, graphData, 10000) // 10 second TTL
      
      this.emitEvent('graph', 'data_updated', graphData)
      return graphData
    } catch (error) {
      console.error('Failed to load graph data:', error)
      throw error
    }
  }

  /**
   * Transcript Data Management
   */
  public async getTranscriptData(forceRefresh = false): Promise<any> {
    const sessionId = this.state.selectedSession === 'all' ? 'all' : this.state.selectedSession
    const cacheKey = `transcript_data_${sessionId}_${JSON.stringify(this.state.activeFilters)}`
    
    if (!forceRefresh) {
      const cached = this.getCachedData(cacheKey)
      if (cached) {
        this.state.transcriptData = cached
        return cached
      }
    }

    try {
      const params = this.buildTranscriptParams()
      const response = await api.get(`/coordination/transcript/${sessionId}`, { params })
      
      const transcriptData = {
        events: response.data.events,
        totalEvents: response.data.total_events,
        metadata: response.data.metadata || {},
        agentSummary: response.data.agent_summary
      }

      this.state.transcriptData = transcriptData
      this.setCachedData(cacheKey, transcriptData, 5000) // 5 second TTL
      
      this.emitEvent('transcript', 'data_updated', transcriptData)
      return transcriptData
    } catch (error) {
      console.error('Failed to load transcript data:', error)
      throw error
    }
  }

  /**
   * Analysis Data Management
   */
  public async getAnalysisData(forceRefresh = false): Promise<any> {
    const cacheKey = `analysis_data_${this.state.selectedSession}`
    
    if (!forceRefresh) {
      const cached = this.getCachedData(cacheKey)
      if (cached) {
        this.state.analysisData = cached
        return cached
      }
    }

    try {
      // Mock analysis data - replace with actual API calls
      const analysisData = {
        patterns: await this.detectPatterns(),
        metrics: await this.getPerformanceMetrics(),
        recommendations: await this.getOptimizationRecommendations()
      }

      this.state.analysisData = analysisData
      this.setCachedData(cacheKey, analysisData, 15000) // 15 second TTL
      
      this.emitEvent('analysis', 'data_updated', analysisData)
      return analysisData
    } catch (error) {
      console.error('Failed to load analysis data:', error)
      throw error
    }
  }

  /**
   * Performance Metrics
   */
  public async getPerformanceMetrics(): Promise<PerformanceMetrics> {
    const cacheKey = `performance_${this.state.selectedSession}`
    const cached = this.getCachedData(cacheKey)
    
    if (cached) {
      return cached
    }

    try {
      // Mock performance metrics - replace with actual API
      const metrics = {
        responseTime: Math.floor(Math.random() * 500) + 100,
        errorRate: Math.random() * 0.1,
        throughput: Math.floor(Math.random() * 100) + 50,
        agentMetrics: {}
      }

      this.setCachedData(cacheKey, metrics, 5000)
      this.state.performance = metrics
      
      return metrics
    } catch (error) {
      console.error('Failed to load performance metrics:', error)
      throw error
    }
  }

  /**
   * Pattern Detection
   */
  public async detectPatterns(): Promise<DetectedPattern[]> {
    try {
      // Mock pattern detection - replace with actual API
      const patterns = Math.random() > 0.7 ? [
        {
          id: 'loop-1',
          name: 'Potential Infinite Loop',
          description: 'Agents exchanging similar messages repeatedly',
          severity: 'HIGH' as const,
          occurrences: 5,
          affectedAgents: ['agent1', 'agent2'],
          confidence: 0.85
        },
        {
          id: 'bottleneck-1',
          name: 'Communication Bottleneck',
          description: 'Single agent handling excessive message load',
          severity: 'MEDIUM' as const,
          occurrences: 3,
          affectedAgents: ['agent3'],
          confidence: 0.72
        }
      ] : []

      return patterns
    } catch (error) {
      console.error('Failed to detect patterns:', error)
      return []
    }
  }

  /**
   * Cross-Component Event System
   */
  public on(eventType: string, callback: (event: CrossComponentEvent) => void) {
    if (!this.eventListeners.has(eventType)) {
      this.eventListeners.set(eventType, [])
    }
    this.eventListeners.get(eventType)!.push(callback)
  }

  public off(eventType: string, callback: (event: CrossComponentEvent) => void) {
    const listeners = this.eventListeners.get(eventType)
    if (listeners) {
      const index = listeners.indexOf(callback)
      if (index > -1) {
        listeners.splice(index, 1)
      }
    }
  }

  private emitEvent(target: string, type: string, data: any) {
    const event: CrossComponentEvent = {
      source: 'coordination_service' as any,
      target: target as any,
      type,
      data,
      timestamp: new Date().toISOString()
    }

    // Emit to specific event type listeners
    const listeners = this.eventListeners.get(type)
    if (listeners) {
      listeners.forEach(callback => {
        try {
          callback(event)
        } catch (error) {
          console.error('Error in event listener:', error)
        }
      })
    }

    // Emit to 'all' listeners
    const allListeners = this.eventListeners.get('all')
    if (allListeners) {
      allListeners.forEach(callback => {
        try {
          callback(event)
        } catch (error) {
          console.error('Error in event listener:', error)
        }
      })
    }
  }

  /**
   * WebSocket Management
   */
  public connectWebSocket(endpoint: string, sessionId?: string): Promise<WebSocket> {
    return new Promise((resolve, reject) => {
      try {
        const wsUrl = this.buildWebSocketUrl(endpoint, sessionId)
        const ws = new WebSocket(wsUrl)
        
        ws.onopen = () => {
          console.log(`WebSocket connected: ${endpoint}`)
          this.websocketConnections.set(endpoint, ws)
          this.state.connections.websocket = true
          this.state.connections.lastUpdate = new Date()
          resolve(ws)
        }

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            this.handleWebSocketMessage(endpoint, data)
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error)
          }
        }

        ws.onerror = (error) => {
          console.error(`WebSocket error on ${endpoint}:`, error)
          this.state.connections.websocket = false
          reject(error)
        }

        ws.onclose = () => {
          console.log(`WebSocket closed: ${endpoint}`)
          this.websocketConnections.delete(endpoint)
          this.state.connections.websocket = this.websocketConnections.size > 0
          
          // Attempt to reconnect after 3 seconds
          setTimeout(() => {
            this.connectWebSocket(endpoint, sessionId)
          }, 3000)
        }
      } catch (error) {
        reject(error)
      }
    })
  }

  public disconnectWebSocket(endpoint: string) {
    const ws = this.websocketConnections.get(endpoint)
    if (ws) {
      ws.close()
      this.websocketConnections.delete(endpoint)
    }
  }

  public disconnectAllWebSockets() {
    this.websocketConnections.forEach((ws, endpoint) => {
      ws.close()
    })
    this.websocketConnections.clear()
    this.state.connections.websocket = false
  }

  /**
   * Synchronization and Data Management
   */
  public async syncAllComponents(): Promise<void> {
    if (this.isSyncing) {
      return
    }

    this.isSyncing = true
    
    try {
      await Promise.all([
        this.getGraphData(true),
        this.getTranscriptData(true),
        this.getAnalysisData(true)
      ])
    } catch (error) {
      console.error('Failed to sync all components:', error)
    } finally {
      this.isSyncing = false
    }
  }

  public async refreshData(component?: 'graph' | 'transcript' | 'analysis' | 'all') {
    switch (component) {
      case 'graph':
        return this.getGraphData(true)
      case 'transcript':
        return this.getTranscriptData(true)
      case 'analysis':
        return this.getAnalysisData(true)
      default:
        return this.syncAllComponents()
    }
  }

  // Private helper methods
  private buildWebSocketUrl(endpoint: string, sessionId?: string): string {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const baseUrl = `${wsProtocol}//${window.location.host}`
    const session = sessionId || this.state.selectedSession
    return `${baseUrl}/api/v1/coordination/ws/${session}?${this.buildUrlParams()}`
  }

  private buildApiParams(): any {
    const filters = this.state.activeFilters
    return {
      session_id: this.state.selectedSession,
      event_types: filters.eventTypes.join(','),
      agent_types: filters.agentIds.join(','),
      include_inactive: filters.includeInactive
    }
  }

  private buildTranscriptParams(): any {
    const filters = this.state.activeFilters
    return {
      agent_filter: filters.agentIds.join(','),
      limit: 100,
      start_time: filters.timeRange.start,
      end_time: filters.timeRange.end,
      include_metadata: true
    }
  }

  private buildUrlParams(): string {
    const filters = this.state.activeFilters
    const params = new URLSearchParams({
      event_types: filters.eventTypes.join(','),
      agent_types: filters.agentIds.join(','),
      include_system: 'true'
    })
    return params.toString()
  }

  private handleWebSocketMessage(endpoint: string, data: any) {
    switch (data.type) {
      case 'graph_update':
        this.handleGraphUpdate(data)
        break
      case 'transcript_update':
        this.handleTranscriptUpdate(data)
        break
      case 'analysis_update':
        this.handleAnalysisUpdate(data)
        break
      case 'performance_update':
        this.handlePerformanceUpdate(data)
        break
      default:
        console.log('Unknown WebSocket message type:', data.type)
    }
  }

  private handleGraphUpdate(data: any) {
    // Update graph data and emit event
    if (data.nodes) {
      this.state.graphData.nodes = data.nodes
    }
    if (data.edges) {
      this.state.graphData.edges = data.edges
    }
    
    this.emitEvent('graph', 'realtime_update', data)
  }

  private handleTranscriptUpdate(data: any) {
    // Update transcript data and emit event
    if (data.events) {
      this.state.transcriptData.events = data.events
    }
    
    this.emitEvent('transcript', 'realtime_update', data)
  }

  private handleAnalysisUpdate(data: any) {
    // Update analysis data and emit event
    if (data.patterns) {
      this.state.analysisData.patterns = data.patterns
    }
    
    this.emitEvent('analysis', 'realtime_update', data)
  }

  private handlePerformanceUpdate(data: any) {
    // Update performance data
    Object.assign(this.state.performance, data)
    this.emitEvent('all', 'performance_update', data)
  }

  // Cache management
  private getCachedData(key: string): any | null {
    const cached = this.dataCache.get(key)
    if (cached && Date.now() - cached.timestamp < cached.ttl) {
      return cached.data
    }
    return null
  }

  private setCachedData(key: string, data: any, ttl: number) {
    this.dataCache.set(key, {
      data,
      timestamp: Date.now(),
      ttl
    })
  }

  private clearExpiredCache() {
    const now = Date.now()
    for (const [key, cached] of this.dataCache.entries()) {
      if (now - cached.timestamp >= cached.ttl) {
        this.dataCache.delete(key)
      }
    }
  }

  // Debounced sync for performance
  private debouncedSync = this.debounce(() => {
    this.syncAllComponents()
  }, 500)

  private debounce(func: Function, wait: number) {
    let timeout: NodeJS.Timeout
    return function executedFunction(...args: any[]) {
      const later = () => {
        clearTimeout(timeout)
        func(...args)
      }
      clearTimeout(timeout)
      timeout = setTimeout(later, wait)
    }
  }

  private async getOptimizationRecommendations(): Promise<any[]> {
    // Mock recommendations - replace with actual API
    return [
      {
        id: 'rec-1',
        type: 'performance',
        title: 'Optimize Agent Response Time',
        description: 'Consider reducing message processing complexity',
        priority: 'HIGH',
        estimatedImpact: '25% response time improvement'
      }
    ]
  }

  // Cleanup
  public destroy() {
    this.disconnectAllWebSockets()
    this.eventListeners.clear()
    this.dataCache.clear()
    clearInterval(this.cacheCleanupInterval)
  }

  private cacheCleanupInterval = setInterval(() => {
    this.clearExpiredCache()
  }, 60000) // Clean cache every minute
}

// Singleton instance
export const coordinationService = new CoordinationService()

// Vue composable for easy integration
export function useCoordinationService() {
  return {
    // State
    selectedSession: coordinationService.selectedSession,
    activeFilters: coordinationService.activeFilters,
    graphData: coordinationService.graphData,
    transcriptData: coordinationService.transcriptData,
    analysisData: coordinationService.analysisData,
    performance: coordinationService.performance,
    connections: coordinationService.connections,

    // Methods
    setSelectedSession: coordinationService.setSelectedSession.bind(coordinationService),
    setFilters: coordinationService.setFilters.bind(coordinationService),
    getGraphData: coordinationService.getGraphData.bind(coordinationService),
    getTranscriptData: coordinationService.getTranscriptData.bind(coordinationService),
    getAnalysisData: coordinationService.getAnalysisData.bind(coordinationService),
    refreshData: coordinationService.refreshData.bind(coordinationService),
    syncAllComponents: coordinationService.syncAllComponents.bind(coordinationService),
    connectWebSocket: coordinationService.connectWebSocket.bind(coordinationService),
    on: coordinationService.on.bind(coordinationService),
    off: coordinationService.off.bind(coordinationService)
  }
}

export default coordinationService