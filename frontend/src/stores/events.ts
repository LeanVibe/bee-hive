import { defineStore } from 'pinia'
import { ref, computed, onUnmounted } from 'vue'
import { apiClient } from '@/services/api'
import { 
  HookType, 
  SecurityRisk,
  ControlDecision
} from '@/types/hooks'
import type { 
  HookEvent as TypedHookEvent,
  SecurityAlert,
  EventFilter,
  SessionInfo,
  AgentInfo,
  HookPerformanceMetrics,
  WebSocketHookMessage
} from '@/types/hooks'

export interface AgentEvent {
  id: string
  session_id: string
  agent_id: string
  event_type: string
  payload: Record<string, any>
  latency_ms?: number
  created_at: string
}

export interface HookEvent {
  event_id: string
  event_type: string
  session_id: string
  agent_id: string
  tool_name?: string
  success?: boolean
  execution_time_ms?: number
  performance_score?: string
  timestamp: string
}

export interface EventFilters {
  session_id?: string
  agent_id?: string
  event_type?: string
  from_time?: string
  to_time?: string
}

export const useEventsStore = defineStore('events', () => {
  // State
  const events = ref<AgentEvent[]>([])
  const realtimeEvents = ref<AgentEvent[]>([])
  const hookEvents = ref<TypedHookEvent[]>([])
  const securityAlerts = ref<SecurityAlert[]>([])
  const sessions = ref<SessionInfo[]>([])
  const agents = ref<AgentInfo[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  const filters = ref<EventFilters>({})
  const hookFilters = ref<EventFilter>({})
  const pagination = ref({
    limit: 50,
    offset: 0,
    total: 0,
    hasNext: false,
    hasPrev: false,
  })
  
  // WebSocket connection
  let websocket: WebSocket | null = null
  const wsConnected = ref(false)
  const eventCallbacks = new Set<(event: TypedHookEvent) => void>()
  const securityCallbacks = new Set<(alert: SecurityAlert) => void>()
  const performanceCallbacks = new Set<(metrics: HookPerformanceMetrics) => void>()
  
  // Computed
  const allEvents = computed(() => {
    // Merge realtime events with fetched events, removing duplicates
    const combined = [...realtimeEvents.value, ...events.value]
    const unique = combined.filter((event, index, arr) => 
      arr.findIndex(e => e.id === event.id) === index
    )
    
    // Sort by created_at descending
    return unique.sort((a, b) => 
      new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    )
  })
  
  const eventStats = computed(() => {
    const stats = {
      total: allEvents.value.length,
      byType: {} as Record<string, number>,
      byStatus: {
        success: 0,
        error: 0,
        warning: 0,
      },
      latencyStats: {
        min: 0,
        max: 0,
        avg: 0,
      }
    }
    
    let totalLatency = 0
    let latencyCount = 0
    let minLatency = Infinity
    let maxLatency = 0
    
    allEvents.value.forEach(event => {
      // Count by type
      stats.byType[event.event_type] = (stats.byType[event.event_type] || 0) + 1
      
      // Count by status based on payload
      if (event.payload.success === true) {
        stats.byStatus.success++
      } else if (event.payload.error || event.payload.exception) {
        stats.byStatus.error++
      } else if (event.payload.warning) {
        stats.byStatus.warning++
      }
      
      // Calculate latency stats
      if (event.latency_ms != null) {
        totalLatency += event.latency_ms
        latencyCount++
        minLatency = Math.min(minLatency, event.latency_ms)
        maxLatency = Math.max(maxLatency, event.latency_ms)
      }
    })
    
    if (latencyCount > 0) {
      stats.latencyStats = {
        min: minLatency,
        max: maxLatency,
        avg: Math.round(totalLatency / latencyCount),
      }
    }
    
    return stats
  })
  
  const recentEvents = computed(() => 
    allEvents.value.slice(0, 10)
  )
  
  // Actions
  const fetchEvents = async (newFilters?: EventFilters) => {
    loading.value = true
    error.value = null
    
    try {
      // Update filters if provided
      if (newFilters) {
        filters.value = { ...filters.value, ...newFilters }
        pagination.value.offset = 0 // Reset pagination
      }
      
      const params = new URLSearchParams({
        limit: pagination.value.limit.toString(),
        offset: pagination.value.offset.toString(),
      })
      
      // Add filters to params
      Object.entries(filters.value).forEach(([key, value]) => {
        if (value) {
          params.append(key, value)
        }
      })
      
      const response = await apiClient.get(`/observability/events?${params}`)
      
      if (pagination.value.offset === 0) {
        // First page or filter change - replace events
        events.value = response.events
      } else {
        // Subsequent pages - append events
        events.value.push(...response.events)
      }
      
      pagination.value = {
        ...pagination.value,
        ...response.pagination,
      }
      
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to fetch events'
      console.error('Failed to fetch events:', err)
    } finally {
      loading.value = false
    }
  }
  
  const loadMoreEvents = async () => {
    if (pagination.value.hasNext && !loading.value) {
      pagination.value.offset += pagination.value.limit
      await fetchEvents()
    }
  }
  
  const refreshEvents = async () => {
    pagination.value.offset = 0
    await fetchEvents()
  }
  
  const addRealtimeEvent = (event: AgentEvent) => {
    // Add to realtime events (limited to last 100)
    realtimeEvents.value.unshift(event)
    if (realtimeEvents.value.length > 100) {
      realtimeEvents.value = realtimeEvents.value.slice(0, 100)
    }
    
    // Remove from fetched events if it exists (to avoid duplication)
    const index = events.value.findIndex(e => e.id === event.id)
    if (index > -1) {
      events.value.splice(index, 1)
    }
  }
  
  const clearRealtimeEvents = () => {
    realtimeEvents.value = []
  }
  
  const updateFilters = (newFilters: EventFilters) => {
    filters.value = newFilters
    pagination.value.offset = 0
    fetchEvents()
  }
  
  const clearFilters = () => {
    filters.value = {}
    pagination.value.offset = 0
    fetchEvents()
  }
  
  const getEventById = (id: string) => {
    return allEvents.value.find(event => event.id === id)
  }
  
  const getEventsBySession = (sessionId: string) => {
    return allEvents.value.filter(event => event.session_id === sessionId)
  }
  
  const getEventsByAgent = (agentId: string) => {
    return allEvents.value.filter(event => event.agent_id === agentId)
  }
  
  const getEventsByType = (eventType: string) => {
    return allEvents.value.filter(event => event.event_type === eventType)
  }
  
  // Hook event management
  const addHookEvent = (event: TypedHookEvent) => {
    hookEvents.value.unshift(event)
    if (hookEvents.value.length > 1000) {
      hookEvents.value = hookEvents.value.slice(0, 1000)
    }
    
    // Update session and agent tracking
    updateSessionInfo(event)
    updateAgentInfo(event)
    
    // Notify all subscribers
    eventCallbacks.forEach(callback => {
      try {
        callback(event)
      } catch (error) {
        console.error('Error in event callback:', error)
      }
    })
  }
  
  // Security alert management
  const addSecurityAlert = (alert: SecurityAlert) => {
    securityAlerts.value.unshift(alert)
    if (securityAlerts.value.length > 500) {
      securityAlerts.value = securityAlerts.value.slice(0, 500)
    }
    
    // Notify security subscribers
    securityCallbacks.forEach(callback => {
      try {
        callback(alert)
      } catch (error) {
        console.error('Error in security callback:', error)
      }
    })
  }
  
  // Session tracking
  const updateSessionInfo = (event: TypedHookEvent) => {
    if (!event.session_id) return
    
    let session = sessions.value.find(s => s.session_id === event.session_id)
    if (!session) {
      session = {
        session_id: event.session_id,
        agent_ids: [event.agent_id],
        start_time: event.timestamp,
        event_count: 0,
        error_count: 0,
        blocked_count: 0,
        status: 'active'
      }
      sessions.value.push(session)
    }
    
    // Update session info
    if (!session.agent_ids.includes(event.agent_id)) {
      session.agent_ids.push(event.agent_id)
    }
    session.event_count++
    
    if (event.hook_type === HookType.ERROR) {
      session.error_count++
    }
    
    if (event.metadata?.security_blocked) {
      session.blocked_count++
    }
    
    if (event.hook_type === HookType.STOP || event.hook_type === HookType.AGENT_STOP) {
      session.status = 'completed'
      session.end_time = event.timestamp
    }
  }
  
  // Agent tracking
  const updateAgentInfo = (event: TypedHookEvent) => {
    let agent = agents.value.find(a => a.agent_id === event.agent_id)
    if (!agent) {
      agent = {
        agent_id: event.agent_id,
        session_ids: event.session_id ? [event.session_id] : [],
        first_seen: event.timestamp,
        last_seen: event.timestamp,
        event_count: 0,
        tool_usage_count: 0,
        error_count: 0,
        blocked_count: 0,
        status: 'active'
      }
      agents.value.push(agent)
    }
    
    // Update agent info
    if (event.session_id && !agent.session_ids.includes(event.session_id)) {
      agent.session_ids.push(event.session_id)
    }
    agent.last_seen = event.timestamp
    agent.event_count++
    
    if (event.hook_type === HookType.PRE_TOOL_USE || event.hook_type === HookType.POST_TOOL_USE) {
      agent.tool_usage_count++
    }
    
    if (event.hook_type === HookType.ERROR) {
      agent.error_count++
      agent.status = 'error'
    }
    
    if (event.metadata?.security_blocked) {
      agent.blocked_count++
      if (event.metadata?.security_decision === ControlDecision.DENY) {
        agent.status = 'blocked'
      }
    }
  }
  
  const clearHookEvents = () => {
    hookEvents.value = []
  }
  
  // Real-time WebSocket connection
  const connectWebSocket = () => {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
      return // Already connected
    }
    
    try {
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = `${wsProtocol}//${window.location.host}/api/v1/websocket/observability`
      
      websocket = new WebSocket(wsUrl)
      
      websocket.onopen = () => {
        wsConnected.value = true
        console.log('WebSocket connected for real-time events')
      }
      
      websocket.onmessage = (event) => {
        try {
          const message: WebSocketHookMessage = JSON.parse(event.data)
          
          switch (message.type) {
            case 'hook_event':
              if (message.data) {
                const hookEvent: TypedHookEvent = {
                  hook_type: message.data.hook_type || HookType.NOTIFICATION,
                  agent_id: message.data.agent_id,
                  session_id: message.data.session_id,
                  timestamp: message.data.timestamp || new Date().toISOString(),
                  payload: message.data.payload || {},
                  correlation_id: message.data.correlation_id,
                  priority: message.data.priority || 5,
                  metadata: message.data.metadata || {},
                  event_id: message.data.event_id
                }
                
                addHookEvent(hookEvent)
              }
              break
              
            case 'security_alert':
              if (message.data) {
                addSecurityAlert(message.data as SecurityAlert)
              }
              break
              
            case 'performance_metric':
              if (message.data) {
                performanceCallbacks.forEach(callback => {
                  try {
                    callback(message.data as HookPerformanceMetrics)
                  } catch (error) {
                    console.error('Error in performance callback:', error)
                  }
                })
              }
              break
              
            default:
              console.log('Unknown WebSocket message type:', message.type)
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }
      
      websocket.onerror = (error) => {
        console.error('WebSocket error:', error)
        wsConnected.value = false
      }
      
      websocket.onclose = () => {
        wsConnected.value = false
        console.log('WebSocket disconnected')
        
        // Attempt to reconnect after 5 seconds
        setTimeout(() => {
          if (!wsConnected.value) {
            connectWebSocket()
          }
        }, 5000)
      }
    } catch (error) {
      console.error('Failed to connect WebSocket:', error)
    }
  }
  
  const disconnectWebSocket = () => {
    if (websocket) {
      websocket.close()
      websocket = null
      wsConnected.value = false
    }
  }
  
  // Event subscription for components
  const onEvent = (callback: (event: TypedHookEvent) => void) => {
    eventCallbacks.add(callback)
  }
  
  const offEvent = (callback: (event: TypedHookEvent) => void) => {
    eventCallbacks.delete(callback)
  }
  
  const onSecurityAlert = (callback: (alert: SecurityAlert) => void) => {
    securityCallbacks.add(callback)
  }
  
  const offSecurityAlert = (callback: (alert: SecurityAlert) => void) => {
    securityCallbacks.delete(callback)
  }
  
  const onPerformanceMetric = (callback: (metrics: HookPerformanceMetrics) => void) => {
    performanceCallbacks.add(callback)
  }
  
  const offPerformanceMetric = (callback: (metrics: HookPerformanceMetrics) => void) => {
    performanceCallbacks.delete(callback)
  }
  
  // Hook performance metrics
  const getHookPerformanceMetrics = async () => {
    try {
      const response = await apiClient.get('/observability/hook-performance')
      return response
    } catch (error) {
      console.error('Failed to fetch hook performance metrics:', error)
      throw error
    }
  }
  
  // Auto-connect WebSocket
  connectWebSocket()
  
  // Cleanup on store destruction
  onUnmounted(() => {
    disconnectWebSocket()
  })
  
  // Filtered hook events based on current filters
  const filteredHookEvents = computed(() => {
    let filtered = hookEvents.value
    
    if (hookFilters.value.agent_ids?.length) {
      filtered = filtered.filter(event => 
        hookFilters.value.agent_ids!.includes(event.agent_id)
      )
    }
    
    if (hookFilters.value.session_ids?.length) {
      filtered = filtered.filter(event => 
        event.session_id && hookFilters.value.session_ids!.includes(event.session_id)
      )
    }
    
    if (hookFilters.value.hook_types?.length) {
      filtered = filtered.filter(event => 
        hookFilters.value.hook_types!.includes(event.hook_type)
      )
    }
    
    if (hookFilters.value.from_time) {
      filtered = filtered.filter(event => 
        event.timestamp >= hookFilters.value.from_time!
      )
    }
    
    if (hookFilters.value.to_time) {
      filtered = filtered.filter(event => 
        event.timestamp <= hookFilters.value.to_time!
      )
    }
    
    if (hookFilters.value.min_priority) {
      filtered = filtered.filter(event => 
        event.priority >= hookFilters.value.min_priority!
      )
    }
    
    if (hookFilters.value.only_errors) {
      filtered = filtered.filter(event => 
        event.hook_type === HookType.ERROR ||
        event.metadata?.error ||
        event.payload?.error
      )
    }
    
    if (hookFilters.value.only_blocked) {
      filtered = filtered.filter(event => 
        event.metadata?.security_blocked === true
      )
    }
    
    if (hookFilters.value.search_query) {
      const query = hookFilters.value.search_query.toLowerCase()
      filtered = filtered.filter(event => 
        JSON.stringify(event.payload).toLowerCase().includes(query) ||
        event.agent_id.toLowerCase().includes(query) ||
        (event.session_id && event.session_id.toLowerCase().includes(query))
      )
    }
    
    return filtered.sort((a, b) => 
      new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    )
  })
  
  // Update hook filters
  const updateHookFilters = (newFilters: EventFilter) => {
    hookFilters.value = { ...hookFilters.value, ...newFilters }
  }
  
  const clearHookFilters = () => {
    hookFilters.value = {}
  }
  
  return {
    // State
    events,
    realtimeEvents,
    hookEvents,
    securityAlerts,
    sessions,
    agents,
    loading,
    error,
    filters,
    hookFilters,
    pagination,
    wsConnected,
    
    // Computed
    allEvents,
    eventStats,
    recentEvents,
    filteredHookEvents,
    
    // Actions
    fetchEvents,
    loadMoreEvents,
    refreshEvents,
    addRealtimeEvent,
    clearRealtimeEvents,
    updateFilters,
    clearFilters,
    getEventById,
    getEventsBySession,
    getEventsByAgent,
    getEventsByType,
    
    // Hook events
    addHookEvent,
    addSecurityAlert,
    clearHookEvents,
    updateHookFilters,
    clearHookFilters,
    
    // Event subscriptions
    onEvent,
    offEvent,
    onSecurityAlert,
    offSecurityAlert,
    onPerformanceMetric,
    offPerformanceMetric,
    
    // WebSocket
    connectWebSocket,
    disconnectWebSocket,
    
    // Performance
    getHookPerformanceMetrics,
  }
})