import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useNotificationStore } from './notifications'

export interface WebSocketMessage {
  type: string
  data: any
  timestamp: string
}

export const useConnectionStore = defineStore('connection', () => {
  // State
  const ws = ref<WebSocket | null>(null)
  const isConnected = ref(false)
  const reconnectAttempts = ref(0)
  const maxReconnectAttempts = ref(5)
  const reconnectDelay = ref(1000)
  const lastPingTime = ref<Date | null>(null)
  const latency = ref<number | null>(null)
  
  // Computed
  const connectionStatus = computed(() => {
    if (isConnected.value) return 'connected'
    if (reconnectAttempts.value > 0) return 'reconnecting'
    return 'disconnected'
  })
  
  const shouldReconnect = computed(() => 
    reconnectAttempts.value < maxReconnectAttempts.value
  )
  
  // Actions
  const connect = () => {
    if (ws.value?.readyState === WebSocket.OPEN) {
      return
    }
    
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = `${protocol}//${window.location.host}/ws/observability`
      
      ws.value = new WebSocket(wsUrl)
      
      ws.value.onopen = () => {
        console.log('WebSocket connected')
        isConnected.value = true
        reconnectAttempts.value = 0
        
        // Start ping interval
        startPingInterval()
        
        const notificationStore = useNotificationStore()
        notificationStore.addNotification({
          type: 'success',
          title: 'Connected',
          message: 'Real-time updates enabled',
          duration: 3000,
        })
      }
      
      ws.value.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          handleMessage(message)
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }
      
      ws.value.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason)
        isConnected.value = false
        
        if (shouldReconnect.value && !event.wasClean) {
          setTimeout(reconnect, reconnectDelay.value * Math.pow(2, reconnectAttempts.value))
        }
      }
      
      ws.value.onerror = (error) => {
        console.error('WebSocket error:', error)
        
        const notificationStore = useNotificationStore()
        notificationStore.addNotification({
          type: 'error',
          title: 'Connection Error',
          message: 'Failed to connect to real-time updates',
          duration: 5000,
        })
      }
      
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
    }
  }
  
  const disconnect = () => {
    if (ws.value) {
      ws.value.close(1000, 'Manual disconnect')
      ws.value = null
      isConnected.value = false
      reconnectAttempts.value = maxReconnectAttempts.value // Prevent reconnection
    }
  }
  
  const reconnect = () => {
    if (!shouldReconnect.value) {
      console.log('Max reconnection attempts reached')
      return
    }
    
    reconnectAttempts.value++
    console.log(`Reconnection attempt ${reconnectAttempts.value}/${maxReconnectAttempts.value}`)
    
    const notificationStore = useNotificationStore()
    notificationStore.addNotification({
      type: 'info',
      title: 'Reconnecting...',
      message: `Attempt ${reconnectAttempts.value} of ${maxReconnectAttempts.value}`,
      duration: 2000,
    })
    
    connect()
  }
  
  const sendMessage = (message: any) => {
    if (ws.value?.readyState === WebSocket.OPEN) {
      ws.value.send(JSON.stringify(message))
    } else {
      console.warn('WebSocket not connected, cannot send message')
    }
  }
  
  const ping = () => {
    if (ws.value?.readyState === WebSocket.OPEN) {
      lastPingTime.value = new Date()
      sendMessage({ type: 'ping', timestamp: lastPingTime.value.toISOString() })
    }
  }
  
  const startPingInterval = () => {
    setInterval(() => {
      if (isConnected.value) {
        ping()
      }
    }, 30000) // Ping every 30 seconds
  }
  
  const handleMessage = async (message: WebSocketMessage) => {
    switch (message.type) {
      case 'pong':
        if (lastPingTime.value) {
          latency.value = Date.now() - lastPingTime.value.getTime()
        }
        break
      
      case 'event':
        // Forward to event store
        const { useEventStore } = await import('./events')
        const eventStore = useEventStore()
        eventStore.addRealtimeEvent(message.data)
        break
      
      case 'metric':
        // Forward to metrics store
        const { useMetricsStore } = await import('./metrics')
        const metricsStore = useMetricsStore()
        metricsStore.updateRealtimeMetric(message.data)
        break
      
      case 'alert':
        // Show alert notification
        const notificationStore = useNotificationStore()
        notificationStore.addNotification({
          type: message.data.severity || 'warning',
          title: message.data.title || 'System Alert',
          message: message.data.message,
          duration: message.data.persistent ? 0 : 10000,
        })
        break
      
      default:
        console.log('Unknown message type:', message.type)
    }
  }
  
  return {
    // State
    isConnected,
    reconnectAttempts,
    maxReconnectAttempts,
    latency,
    
    // Computed
    connectionStatus,
    shouldReconnect,
    
    // Actions
    connect,
    disconnect,
    reconnect,
    sendMessage,
    ping,
  }
})