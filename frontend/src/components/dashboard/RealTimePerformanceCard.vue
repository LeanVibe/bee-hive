<template>
  <div class="glass-card rounded-xl p-6">
    <!-- Header -->
    <div class="flex items-center justify-between mb-6">
      <div class="flex items-center space-x-3">
        <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
          System Performance
        </h3>
        <div class="flex items-center space-x-2">
          <div 
            class="w-2 h-2 rounded-full transition-colors duration-200"
            :class="connectionStatusColor"
          ></div>
          <span class="text-sm text-slate-600 dark:text-slate-400">
            {{ connectionStatusText }}
          </span>
        </div>
      </div>
      
      <div class="flex items-center space-x-3">
        <!-- Update frequency -->
        <select
          v-model="updateFrequency"
          @change="onUpdateFrequencyChange"
          class="input-field text-sm"
        >
          <option value={1}>1s</option>
          <option value={5}>5s</option>
          <option value={10}>10s</option>
          <option value={30}>30s</option>
        </select>
        
        <!-- Live indicator -->
        <div class="flex items-center space-x-2 text-sm text-slate-500 dark:text-slate-400">
          <div 
            class="w-2 h-2 rounded-full transition-colors duration-200"
            :class="isConnected ? 'bg-green-500 animate-pulse' : 'bg-slate-400'"
          ></div>
          <span>{{ isConnected ? 'Live' : 'Offline' }}</span>
        </div>
      </div>
    </div>

    <!-- Connection Status Banner -->
    <div 
      v-if="connectionStatus !== 'connected'"
      class="mb-4 p-3 rounded-lg border-l-4"
      :class="connectionBannerClass"
    >
      <div class="flex items-center">
        <ExclamationTriangleIcon 
          v-if="connectionStatus === 'error'"
          class="w-5 h-5 mr-2"
        />
        <ClockIcon 
          v-else-if="connectionStatus === 'connecting'"
          class="w-5 h-5 mr-2 animate-spin"
        />
        <span class="text-sm font-medium">{{ connectionBannerText }}</span>
      </div>
    </div>

    <!-- Performance Metrics Grid -->
    <div v-if="metrics" class="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      <!-- CPU Usage -->
      <div class="performance-metric">
        <div class="flex items-center justify-between mb-2">
          <span class="text-sm font-medium text-slate-600 dark:text-slate-400">CPU Usage</span>
          <span class="text-lg font-bold text-slate-900 dark:text-white">
            {{ metrics.cpu_usage_percent.toFixed(1) }}%
          </span>
        </div>
        <div class="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
          <div 
            class="h-2 rounded-full transition-all duration-500"
            :class="getCpuUsageColor(metrics.cpu_usage_percent)"
            :style="{ width: `${Math.min(100, metrics.cpu_usage_percent)}%` }"
          ></div>
        </div>
      </div>

      <!-- Memory Usage -->
      <div class="performance-metric">
        <div class="flex items-center justify-between mb-2">
          <span class="text-sm font-medium text-slate-600 dark:text-slate-400">Memory</span>
          <span class="text-lg font-bold text-slate-900 dark:text-white">
            {{ metrics.memory_usage_percent.toFixed(1) }}%
          </span>
        </div>
        <div class="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
          <div 
            class="h-2 rounded-full transition-all duration-500"
            :class="getMemoryUsageColor(metrics.memory_usage_percent)"
            :style="{ width: `${Math.min(100, metrics.memory_usage_percent)}%` }"
          ></div>
        </div>
        <div class="text-xs text-slate-500 dark:text-slate-400 mt-1">
          {{ formatMemory(metrics.memory_usage_mb) }}
        </div>
      </div>

      <!-- Disk Usage -->
      <div class="performance-metric">
        <div class="flex items-center justify-between mb-2">
          <span class="text-sm font-medium text-slate-600 dark:text-slate-400">Disk</span>
          <span class="text-lg font-bold text-slate-900 dark:text-white">
            {{ metrics.disk_usage_percent.toFixed(1) }}%
          </span>
        </div>
        <div class="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
          <div 
            class="h-2 rounded-full transition-all duration-500"
            :class="getDiskUsageColor(metrics.disk_usage_percent)"
            :style="{ width: `${Math.min(100, metrics.disk_usage_percent)}%` }"
          ></div>
        </div>
      </div>

      <!-- Active Connections -->
      <div class="performance-metric">
        <div class="flex items-center justify-between mb-2">
          <span class="text-sm font-medium text-slate-600 dark:text-slate-400">Connections</span>
          <span class="text-lg font-bold text-slate-900 dark:text-white">
            {{ metrics.active_connections }}
          </span>
        </div>
        <div class="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
          <div 
            class="h-2 rounded-full bg-blue-500 transition-all duration-500"
            :style="{ width: `${Math.min(100, (metrics.active_connections / maxConnections) * 100)}%` }"
          ></div>
        </div>
        <div class="text-xs text-slate-500 dark:text-slate-400 mt-1">
          Max: {{ maxConnections }}
        </div>
      </div>
    </div>

    <!-- Performance Trends Chart -->
    <div v-if="showChart && performanceHistory.length > 0" class="mb-6">
      <h4 class="text-sm font-medium text-slate-900 dark:text-white mb-3">Performance Trends</h4>
      <div class="h-32 bg-slate-50 dark:bg-slate-800 rounded-lg p-4">
        <canvas 
          ref="chartCanvas"
          class="w-full h-full"
        ></canvas>
      </div>
    </div>

    <!-- System Health Indicators -->
    <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
      <!-- Overall Health Score -->
      <div class="text-center p-4 rounded-lg bg-slate-50 dark:bg-slate-800">
        <div class="text-2xl font-bold mb-1" :class="getHealthScoreColor(healthScore)">
          {{ healthScore }}%
        </div>
        <div class="text-xs text-slate-600 dark:text-slate-400">Health Score</div>
      </div>

      <!-- Response Time -->
      <div class="text-center p-4 rounded-lg bg-slate-50 dark:bg-slate-800">
        <div class="text-2xl font-bold text-slate-900 dark:text-white mb-1">
          {{ averageResponseTime }}ms
        </div>
        <div class="text-xs text-slate-600 dark:text-slate-400">Avg Response</div>
      </div>

      <!-- Uptime -->
      <div class="text-center p-4 rounded-lg bg-slate-50 dark:bg-slate-800">
        <div class="text-2xl font-bold text-green-600 dark:text-green-400 mb-1">
          99.9%
        </div>
        <div class="text-xs text-slate-600 dark:text-slate-400">Uptime</div>
      </div>
    </div>

    <!-- Performance Alerts -->
    <div v-if="performanceAlerts.length > 0" class="mt-4 space-y-2">
      <h4 class="text-sm font-medium text-slate-900 dark:text-white">Performance Alerts</h4>
      <div 
        v-for="alert in performanceAlerts" 
        :key="alert.id"
        class="flex items-center justify-between p-3 rounded-lg"
        :class="getAlertClass(alert.severity)"
      >
        <div class="flex items-center space-x-2">
          <ExclamationTriangleIcon class="w-4 h-4" />
          <span class="text-sm font-medium">{{ alert.message }}</span>
        </div>
        <span class="text-xs">{{ formatTime(alert.timestamp) }}</span>
      </div>
    </div>

    <!-- Last Update -->
    <div class="mt-4 text-xs text-slate-500 dark:text-slate-400 text-center">
      Last updated: {{ lastUpdate ? formatTime(lastUpdate) : 'Never' }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { formatDistanceToNow } from 'date-fns'
import { ExclamationTriangleIcon, ClockIcon } from '@heroicons/vue/24/outline'

import { useAgentMonitoring, type PerformanceMetrics } from '@/services/agentMonitoringService'

// Props
interface Props {
  showChart?: boolean
  maxConnections?: number
}

const props = withDefaults(defineProps<Props>(), {
  showChart: true,
  maxConnections: 100
})

// Emits
const emit = defineEmits<{
  performanceAlert: [alert: any]
  metricsUpdated: [metrics: PerformanceMetrics]
}>()

// Agent monitoring service
const {
  performanceMetrics,
  isConnected,
  connectionStatus,
  onEvent
} = useAgentMonitoring()

// Local state
const updateFrequency = ref(5) // seconds
const performanceHistory = ref<PerformanceMetrics[]>([])
const lastUpdate = ref<Date | null>(null)
const chartCanvas = ref<HTMLCanvasElement | null>(null)
const performanceAlerts = ref<Array<{
  id: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  message: string
  timestamp: string
}>>([])

// Computed properties
const metrics = computed(() => performanceMetrics.value)

const connectionStatusColor = computed(() => {
  switch (connectionStatus.value) {
    case 'connected': return 'bg-green-500'
    case 'connecting': return 'bg-yellow-500 animate-pulse'
    case 'error': return 'bg-red-500'
    default: return 'bg-slate-400'
  }
})

const connectionStatusText = computed(() => {
  switch (connectionStatus.value) {
    case 'connected': return 'Connected'
    case 'connecting': return 'Connecting...'
    case 'error': return 'Connection error'
    default: return 'Disconnected'
  }
})

const connectionBannerClass = computed(() => {
  switch (connectionStatus.value) {
    case 'error': return 'bg-red-50 border-red-400 text-red-700 dark:bg-red-900/20 dark:border-red-700 dark:text-red-300'
    case 'connecting': return 'bg-yellow-50 border-yellow-400 text-yellow-700 dark:bg-yellow-900/20 dark:border-yellow-700 dark:text-yellow-300'
    default: return 'bg-slate-50 border-slate-400 text-slate-700 dark:bg-slate-900/20 dark:border-slate-700 dark:text-slate-300'
  }
})

const connectionBannerText = computed(() => {
  switch (connectionStatus.value) {
    case 'error': return 'Unable to connect to performance monitoring service'
    case 'connecting': return 'Establishing connection to performance monitoring service...'
    default: return 'Not connected to performance monitoring service'
  }
})

const healthScore = computed(() => {
  if (!metrics.value) return 0
  
  const m = metrics.value
  const cpuScore = Math.max(0, 100 - m.cpu_usage_percent)
  const memoryScore = Math.max(0, 100 - m.memory_usage_percent)
  const diskScore = Math.max(0, 100 - m.disk_usage_percent)
  
  return Math.round((cpuScore + memoryScore + diskScore) / 3)
})

const averageResponseTime = computed(() => {
  // Mock calculation - in real implementation would track actual response times
  if (!metrics.value) return 0
  
  const baseTime = 50
  const cpuFactor = metrics.value.cpu_usage_percent * 2
  const memoryFactor = metrics.value.memory_usage_percent * 1.5
  
  return Math.round(baseTime + cpuFactor + memoryFactor)
})

// Methods
const formatMemory = (mb: number) => {
  if (mb > 1024) {
    return `${(mb / 1024).toFixed(1)}GB`
  }
  return `${Math.round(mb)}MB`
}

const formatTime = (timestamp: string | Date) => {
  try {
    const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp
    return formatDistanceToNow(date, { addSuffix: true })
  } catch {
    return 'Unknown'
  }
}

const getCpuUsageColor = (usage: number) => {
  if (usage < 50) return 'bg-green-500'
  if (usage < 70) return 'bg-yellow-500'
  if (usage < 90) return 'bg-orange-500'
  return 'bg-red-500'
}

const getMemoryUsageColor = (usage: number) => {
  if (usage < 60) return 'bg-green-500'
  if (usage < 80) return 'bg-yellow-500'
  return 'bg-red-500'
}

const getDiskUsageColor = (usage: number) => {
  if (usage < 70) return 'bg-green-500'
  if (usage < 85) return 'bg-yellow-500'
  return 'bg-red-500'
}

const getHealthScoreColor = (score: number) => {
  if (score >= 90) return 'text-green-600 dark:text-green-400'
  if (score >= 70) return 'text-yellow-600 dark:text-yellow-400'
  if (score >= 50) return 'text-orange-600 dark:text-orange-400'
  return 'text-red-600 dark:text-red-400'
}

const getAlertClass = (severity: string) => {
  const classes = {
    'low': 'bg-blue-50 text-blue-700 dark:bg-blue-900/20 dark:text-blue-300',
    'medium': 'bg-yellow-50 text-yellow-700 dark:bg-yellow-900/20 dark:text-yellow-300',
    'high': 'bg-orange-50 text-orange-700 dark:bg-orange-900/20 dark:text-orange-300',
    'critical': 'bg-red-50 text-red-700 dark:bg-red-900/20 dark:text-red-300'
  }
  return classes[severity as keyof typeof classes] || classes.low
}

const onUpdateFrequencyChange = () => {
  console.log(`🔄 Update frequency changed to ${updateFrequency.value}s`)
  // In a real implementation, this would update the WebSocket subscription
}

const checkPerformanceThresholds = (metrics: PerformanceMetrics) => {
  const alerts = []
  const now = new Date().toISOString()
  
  // CPU threshold
  if (metrics.cpu_usage_percent > 90) {
    alerts.push({
      id: `cpu-${Date.now()}`,
      severity: 'critical' as const,
      message: `High CPU usage: ${metrics.cpu_usage_percent.toFixed(1)}%`,
      timestamp: now
    })
  } else if (metrics.cpu_usage_percent > 80) {
    alerts.push({
      id: `cpu-${Date.now()}`,
      severity: 'high' as const,
      message: `Elevated CPU usage: ${metrics.cpu_usage_percent.toFixed(1)}%`,
      timestamp: now
    })
  }
  
  // Memory threshold
  if (metrics.memory_usage_percent > 95) {
    alerts.push({
      id: `memory-${Date.now()}`,
      severity: 'critical' as const,
      message: `Critical memory usage: ${metrics.memory_usage_percent.toFixed(1)}%`,
      timestamp: now
    })
  } else if (metrics.memory_usage_percent > 85) {
    alerts.push({
      id: `memory-${Date.now()}`,
      severity: 'high' as const,
      message: `High memory usage: ${metrics.memory_usage_percent.toFixed(1)}%`,
      timestamp: now
    })
  }
  
  // Disk threshold
  if (metrics.disk_usage_percent > 95) {
    alerts.push({
      id: `disk-${Date.now()}`,
      severity: 'critical' as const,
      message: `Critical disk usage: ${metrics.disk_usage_percent.toFixed(1)}%`,
      timestamp: now
    })
  }
  
  // Add new alerts
  alerts.forEach(alert => {
    performanceAlerts.value.unshift(alert)
    emit('performanceAlert', alert)
  })
  
  // Keep only last 10 alerts
  if (performanceAlerts.value.length > 10) {
    performanceAlerts.value = performanceAlerts.value.slice(0, 10)
  }
}

const updatePerformanceHistory = (metrics: PerformanceMetrics) => {
  performanceHistory.value.push(metrics)
  
  // Keep only last 60 data points (5 minutes at 5s intervals)
  if (performanceHistory.value.length > 60) {
    performanceHistory.value.shift()
  }
  
  // Update chart if visible
  if (props.showChart) {
    nextTick(() => {
      drawChart()
    })
  }
}

const drawChart = () => {
  if (!chartCanvas.value || performanceHistory.value.length < 2) return
  
  const canvas = chartCanvas.value
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  
  // Set canvas size
  const rect = canvas.getBoundingClientRect()
  canvas.width = rect.width * devicePixelRatio
  canvas.height = rect.height * devicePixelRatio
  ctx.scale(devicePixelRatio, devicePixelRatio)
  
  // Clear canvas
  ctx.clearRect(0, 0, rect.width, rect.height)
  
  // Draw performance trends (simple line chart)
  const history = performanceHistory.value
  const width = rect.width
  const height = rect.height
  const padding = 20
  
  // CPU usage line
  ctx.strokeStyle = '#ef4444' // red
  ctx.lineWidth = 2
  ctx.beginPath()
  history.forEach((metrics, index) => {
    const x = padding + (index / (history.length - 1)) * (width - padding * 2)
    const y = height - padding - (metrics.cpu_usage_percent / 100) * (height - padding * 2)
    
    if (index === 0) {
      ctx.moveTo(x, y)
    } else {
      ctx.lineTo(x, y)
    }
  })
  ctx.stroke()
  
  // Memory usage line
  ctx.strokeStyle = '#3b82f6' // blue
  ctx.beginPath()
  history.forEach((metrics, index) => {
    const x = padding + (index / (history.length - 1)) * (width - padding * 2)
    const y = height - padding - (metrics.memory_usage_percent / 100) * (height - padding * 2)
    
    if (index === 0) {
      ctx.moveTo(x, y)
    } else {
      ctx.lineTo(x, y)
    }
  })
  ctx.stroke()
}

// Event handlers
let unsubscribeHandlers: Array<() => void> = []

// Lifecycle
onMounted(() => {
  // Subscribe to performance metrics updates
  unsubscribeHandlers.push(
    onEvent('performance_metrics', (metrics: PerformanceMetrics) => {
      lastUpdate.value = new Date()
      updatePerformanceHistory(metrics)
      checkPerformanceThresholds(metrics)
      emit('metricsUpdated', metrics)
    })
  )
  
  console.log('📊 Real-time performance monitoring initialized')
})

onUnmounted(() => {
  unsubscribeHandlers.forEach(unsubscribe => unsubscribe())
  unsubscribeHandlers = []
})
</script>

<style scoped>
.performance-metric {
  @apply p-4 rounded-lg bg-slate-50 dark:bg-slate-800 transition-all duration-200;
}

.performance-metric:hover {
  @apply bg-slate-100 dark:bg-slate-700;
}

.glass-card {
  backdrop-filter: blur(10px);
  background: rgba(255, 255, 255, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.2);
}

@media (prefers-color-scheme: dark) {
  .glass-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
  }
}
</style>