import { defineStore } from 'pinia'
import { ref, computed, onUnmounted } from 'vue'
import { apiClient } from '@/services/api'

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
  const hookEvents = ref<HookEvent[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  const filters = ref<EventFilters>({})
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
  const eventCallbacks = new Set<(event: HookEvent) => void>()
  
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
  const addHookEvent = (event: HookEvent) => {
    hookEvents.value.unshift(event)
    if (hookEvents.value.length > 1000) {
      hookEvents.value = hookEvents.value.slice(0, 1000)
    }
    
    // Notify all subscribers
    eventCallbacks.forEach(callback => {
      try {
        callback(event)
      } catch (error) {
        console.error('Error in event callback:', error)
      }
    })
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
          const message = JSON.parse(event.data)
          
          if (message.type === 'event' && message.data) {
            const hookEvent: HookEvent = {
              event_id: message.data.event_id || Date.now().toString(),
              event_type: message.data.event_type,
              session_id: message.data.session_id,
              agent_id: message.data.agent_id,
              tool_name: message.data.tool_name,
              success: message.data.success,
              execution_time_ms: message.data.execution_time_ms,
              performance_score: message.data.performance_score,
              timestamp: message.data.timestamp || new Date().toISOString()
            }
            
            addHookEvent(hookEvent)
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
  const onEvent = (callback: (event: HookEvent) => void) => {
    eventCallbacks.add(callback)
  }
  
  const offEvent = (callback: (event: HookEvent) => void) => {
    eventCallbacks.delete(callback)
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
  
  return {
    // State
    events,
    realtimeEvents,
    hookEvents,
    loading,
    error,
    filters,
    pagination,
    wsConnected,
    
    // Computed
    allEvents,
    eventStats,
    recentEvents,
    
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
    clearHookEvents,
    onEvent,
    offEvent,
    
    // WebSocket
    connectWebSocket,
    disconnectWebSocket,
    
    // Performance
    getHookPerformanceMetrics,
  }
})