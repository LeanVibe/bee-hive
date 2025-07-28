<template>
  <div class="glass-card rounded-xl p-6">
    <!-- Header -->
    <div class="flex items-center justify-between mb-6">
      <div class="flex items-center space-x-3">
        <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
          Agent Performance Trends
        </h3>
        <div class="flex items-center space-x-2">
          <div 
            class="w-2 h-2 rounded-full transition-colors duration-200"
            :class="isConnected ? 'bg-green-500 animate-pulse' : 'bg-slate-400'"
          ></div>
          <span class="text-sm text-slate-600 dark:text-slate-400">
            {{ dataPoints.length }} data points
          </span>
        </div>
      </div>
      
      <div class="flex items-center space-x-3">
        <!-- Time range selector -->
        <select
          v-model="timeRange"
          @change="onTimeRangeChange"
          class="input-field text-sm"
        >
          <option value="5m">5 minutes</option>
          <option value="15m">15 minutes</option>
          <option value="1h">1 hour</option>
          <option value="4h">4 hours</option>
        </select>
        
        <!-- Metric selector -->
        <select
          v-model="selectedMetric"
          @change="onMetricChange"
          class="input-field text-sm"
        >
          <option value="performance">Performance</option>
          <option value="cpu">CPU Usage</option>
          <option value="memory">Memory Usage</option>
          <option value="tasks">Task Completion Rate</option>
        </select>
      </div>
    </div>

    <!-- Chart Container -->
    <div class="relative">
      <canvas 
        ref="chartCanvas"
        class="w-full h-64"
        @mousemove="onMouseMove"
        @mouseleave="hideTooltip"
      ></canvas>
      
      <!-- Tooltip -->
      <div
        v-if="tooltip.visible"
        class="absolute bg-slate-900 text-white text-xs rounded px-2 py-1 pointer-events-none z-10"
        :style="{ left: tooltip.x + 'px', top: tooltip.y + 'px' }"
      >
        <div>{{ tooltip.title }}</div>
        <div v-for="(value, key) in tooltip.data" :key="key" class="flex justify-between space-x-2">
          <span>{{ key }}:</span>
          <span class="font-semibold">{{ value }}</span>
        </div>
      </div>
    </div>

    <!-- Chart Legend -->
    <div class="mt-4 flex flex-wrap items-center justify-center space-x-6">
      <div 
        v-for="series in visibleSeries" 
        :key="series.name"
        class="flex items-center space-x-2 cursor-pointer"
        @click="toggleSeries(series.name)"
      >
        <div 
          class="w-3 h-3 rounded-full transition-opacity"
          :class="series.visible ? 'opacity-100' : 'opacity-30'"
          :style="{ backgroundColor: series.color }"
        ></div>
        <span 
          class="text-sm transition-opacity"
          :class="series.visible ? 'text-slate-900 dark:text-white opacity-100' : 'text-slate-500 opacity-60'"
        >
          {{ series.label }}
        </span>
      </div>
    </div>

    <!-- Chart Statistics -->
    <div class="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-4 pt-4 border-t border-slate-200 dark:border-slate-600">
      <div class="text-center">
        <div class="text-lg font-bold text-slate-900 dark:text-white">
          {{ currentStats.avgPerformance }}%
        </div>
        <div class="text-xs text-slate-600 dark:text-slate-400">Avg Performance</div>
      </div>
      <div class="text-center">
        <div class="text-lg font-bold text-slate-900 dark:text-white">
          {{ currentStats.peakPerformance }}%
        </div>
        <div class="text-xs text-slate-600 dark:text-slate-400">Peak Performance</div>
      </div>
      <div class="text-center">
        <div class="text-lg font-bold text-slate-900 dark:text-white">
          {{ currentStats.activeAgents }}
        </div>
        <div class="text-xs text-slate-600 dark:text-slate-400">Active Agents</div>
      </div>
      <div class="text-center">
        <div class="text-lg font-bold" :class="getTrendColor(currentStats.trend)">
          {{ currentStats.trend > 0 ? '+' : '' }}{{ currentStats.trend }}%
        </div>
        <div class="text-xs text-slate-600 dark:text-slate-400">Trend</div>
      </div>
    </div>

    <!-- No Data State -->
    <div v-if="dataPoints.length === 0" class="text-center py-12">
      <ChartBarIcon class="w-12 h-12 text-slate-400 mx-auto mb-4" />
      <h3 class="text-lg font-medium text-slate-900 dark:text-white mb-2">No Performance Data</h3>
      <p class="text-slate-600 dark:text-slate-400">
        Performance data will appear here when agents are active.
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { ChartBarIcon } from '@heroicons/vue/24/outline'

import { useAgentMonitoring, type AgentStatus, type PerformanceMetrics } from '@/services/agentMonitoringService'

// Props
interface Props {
  height?: number
  showLegend?: boolean
  showStats?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  height: 256,
  showLegend: true,
  showStats: true
})

// Emits
const emit = defineEmits<{
  seriesToggled: [series: string, visible: boolean]
  timeRangeChanged: [range: string]
  metricChanged: [metric: string]
}>()

// Agent monitoring service  
const {
  agents,
  performanceMetrics,
  isConnected,
  onEvent
} = useAgentMonitoring()

// Local state
const chartCanvas = ref<HTMLCanvasElement | null>(null)
const timeRange = ref('15m')
const selectedMetric = ref('performance')
const dataPoints = ref<Array<{
  timestamp: Date
  agentId: string
  agentName: string
  performance: number
  cpu: number
  memory: number
  tasks: number
}>>([])

const tooltip = ref({
  visible: false,
  x: 0,
  y: 0,
  title: '',
  data: {} as Record<string, string>
})

const visibleSeries = ref([
  { name: 'agent1', label: 'Agent Alpha', color: '#3b82f6', visible: true },
  { name: 'agent2', label: 'Agent Beta', color: '#ef4444', visible: true },
  { name: 'agent3', label: 'Agent Gamma', color: '#10b981', visible: true },
  { name: 'agent4', label: 'Agent Delta', color: '#f59e0b', visible: false },
  { name: 'avg', label: 'Average', color: '#6b7280', visible: true }
])

// Chart rendering state
let animationFrame: number | null = null
let chartContext: CanvasRenderingContext2D | null = null

// Computed properties
const maxDataPoints = computed(() => {
  const ranges = {
    '5m': 60,   // 5 minutes at 5s intervals
    '15m': 180, // 15 minutes at 5s intervals
    '1h': 720,  // 1 hour at 5s intervals
    '4h': 2880  // 4 hours at 5s intervals
  }
  return ranges[timeRange.value as keyof typeof ranges] || 180
})

const currentStats = computed(() => {
  if (dataPoints.value.length === 0) {
    return {
      avgPerformance: 0,
      peakPerformance: 0,
      activeAgents: 0,
      trend: 0
    }
  }

  const recentPoints = dataPoints.value.slice(-20) // Last 20 data points
  const performances = recentPoints.map(p => p.performance)
  
  const avgPerformance = Math.round(performances.reduce((sum, p) => sum + p, 0) / performances.length)
  const peakPerformance = Math.round(Math.max(...performances))
  const activeAgents = new Set(recentPoints.map(p => p.agentId)).size

  // Calculate trend (last 10 vs previous 10)
  let trend = 0
  if (recentPoints.length >= 20) {
    const recent10 = performances.slice(-10)
    const previous10 = performances.slice(-20, -10)
    const recentAvg = recent10.reduce((sum, p) => sum + p, 0) / recent10.length
    const previousAvg = previous10.reduce((sum, p) => sum + p, 0) / previous10.length
    trend = Math.round(((recentAvg - previousAvg) / previousAvg) * 100)
  }

  return {
    avgPerformance,
    peakPerformance,
    activeAgents,
    trend
  }
})

// Methods
const onTimeRangeChange = () => {
  // Trim data points to fit new range
  const maxPoints = maxDataPoints.value
  if (dataPoints.value.length > maxPoints) {
    dataPoints.value = dataPoints.value.slice(-maxPoints)
  }
  
  nextTick(() => {
    drawChart()
  })
  
  emit('timeRangeChanged', timeRange.value)
}

const onMetricChange = () => {
  nextTick(() => {
    drawChart()
  })
  
  emit('metricChanged', selectedMetric.value)
}

const toggleSeries = (seriesName: string) => {
  const series = visibleSeries.value.find(s => s.name === seriesName)
  if (series) {
    series.visible = !series.visible
    nextTick(() => {
      drawChart()
    })
    emit('seriesToggled', seriesName, series.visible)
  }
}

const addDataPoint = (agentData: AgentStatus) => {
  const dataPoint = {
    timestamp: new Date(),
    agentId: agentData.id,
    agentName: agentData.name,
    performance: agentData.performance,
    cpu: 50 + Math.random() * 40, // Mock CPU data
    memory: agentData.memoryUsage / 10, // Convert to percentage
    tasks: (agentData.tasksCompleted / Math.max(1, agentData.totalTasks)) * 100
  }

  dataPoints.value.push(dataPoint)

  // Trim to max data points
  if (dataPoints.value.length > maxDataPoints.value) {
    dataPoints.value.shift()
  }

  // Trigger chart redraw
  nextTick(() => {
    drawChart()
  })
}

const drawChart = () => {
  if (!chartCanvas.value || dataPoints.value.length === 0) return

  const canvas = chartCanvas.value
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  chartContext = ctx

  // Set canvas size
  const rect = canvas.getBoundingClientRect()
  canvas.width = rect.width * devicePixelRatio
  canvas.height = rect.height * devicePixelRatio
  ctx.scale(devicePixelRatio, devicePixelRatio)

  // Clear canvas
  ctx.clearRect(0, 0, rect.width, rect.height)

  // Chart dimensions
  const padding = 40
  const chartWidth = rect.width - padding * 2
  const chartHeight = rect.height - padding * 2
  const chartLeft = padding
  const chartTop = padding

  // Get data for selected metric
  const metricData = getMetricData(selectedMetric.value)
  if (metricData.length === 0) return

  // Draw grid
  drawGrid(ctx, chartLeft, chartTop, chartWidth, chartHeight)

  // Draw axes
  drawAxes(ctx, chartLeft, chartTop, chartWidth, chartHeight, metricData)

  // Draw data lines for each visible series
  drawDataLines(ctx, chartLeft, chartTop, chartWidth, chartHeight, metricData)
}

const getMetricData = (metric: string) => {
  const agentGroups = new Map<string, Array<{ timestamp: Date; value: number }>>()
  
  // Group data points by agent
  dataPoints.value.forEach(point => {
    if (!agentGroups.has(point.agentId)) {
      agentGroups.set(point.agentId, [])
    }
    
    let value = 0
    switch (metric) {
      case 'performance':
        value = point.performance
        break
      case 'cpu':
        value = point.cpu
        break
      case 'memory':
        value = point.memory
        break
      case 'tasks':
        value = point.tasks
        break
    }
    
    agentGroups.get(point.agentId)!.push({
      timestamp: point.timestamp,
      value
    })
  })

  return Array.from(agentGroups.entries()).map(([agentId, points]) => ({
    agentId,
    points
  }))
}

const drawGrid = (ctx: CanvasRenderingContext2D, left: number, top: number, width: number, height: number) => {
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)' // slate-400 with opacity
  ctx.lineWidth = 1

  // Horizontal grid lines
  for (let i = 0; i <= 4; i++) {
    const y = top + (height / 4) * i
    ctx.beginPath()
    ctx.moveTo(left, y)
    ctx.lineTo(left + width, y)
    ctx.stroke()
  }

  // Vertical grid lines
  for (let i = 0; i <= 6; i++) {
    const x = left + (width / 6) * i
    ctx.beginPath()
    ctx.moveTo(x, top)
    ctx.lineTo(x, top + height)
    ctx.stroke()
  }
}

const drawAxes = (ctx: CanvasRenderingContext2D, left: number, top: number, width: number, height: number, data: any[]) => {
  ctx.strokeStyle = 'rgba(100, 116, 139, 0.8)' // slate-500
  ctx.lineWidth = 2

  // Y-axis
  ctx.beginPath()
  ctx.moveTo(left, top)
  ctx.lineTo(left, top + height)
  ctx.stroke()

  // X-axis
  ctx.beginPath()
  ctx.moveTo(left, top + height)
  ctx.lineTo(left + width, top + height)
  ctx.stroke()

  // Y-axis labels (0-100%)
  ctx.fillStyle = 'rgba(100, 116, 139)'
  ctx.font = '12px system-ui'
  ctx.textAlign = 'right'
  ctx.textBaseline = 'middle'

  for (let i = 0; i <= 4; i++) {
    const y = top + height - (height / 4) * i
    const value = (i * 25).toString() + '%'
    ctx.fillText(value, left - 10, y)
  }

  // X-axis labels (time)
  ctx.textAlign = 'center'
  ctx.textBaseline = 'top'
  
  const now = new Date()
  const timeRangeMs = getTimeRangeMs(timeRange.value)
  
  for (let i = 0; i <= 6; i++) {
    const x = left + (width / 6) * i
    const timeOffset = (timeRangeMs / 6) * (6 - i)
    const time = new Date(now.getTime() - timeOffset)
    const label = formatTimeLabel(time)
    ctx.fillText(label, x, top + height + 10)
  }
}

const drawDataLines = (ctx: CanvasRenderingContext2D, left: number, top: number, width: number, height: number, metricData: any[]) => {
  const now = new Date()
  const timeRangeMs = getTimeRangeMs(timeRange.value)
  const startTime = now.getTime() - timeRangeMs

  // Draw line for each visible agent
  visibleSeries.value.forEach((series, seriesIndex) => {
    if (!series.visible) return

    // Find data for this series
    let seriesData: any[] = []
    
    if (series.name === 'avg') {
      // Calculate average across all agents
      const timePoints = new Map<number, number[]>()
      
      metricData.forEach(({ points }) => {
        points.forEach((point: any) => {
          const timeKey = Math.floor(point.timestamp.getTime() / 5000) * 5000 // 5-second buckets
          if (!timePoints.has(timeKey)) {
            timePoints.set(timeKey, [])
          }
          timePoints.get(timeKey)!.push(point.value)
        })
      })
      
      seriesData = Array.from(timePoints.entries()).map(([timestamp, values]) => ({
        timestamp: new Date(timestamp),
        value: values.reduce((sum, v) => sum + v, 0) / values.length
      }))
    } else {
      // Use specific agent data
      const agentData = metricData.find(({ agentId }) => agentId.includes(series.name.slice(-1)))
      seriesData = agentData ? agentData.points : []
    }

    if (seriesData.length < 2) return

    // Draw line
    ctx.strokeStyle = series.color
    ctx.lineWidth = 2
    ctx.beginPath()

    seriesData.forEach((point, index) => {
      const x = left + ((point.timestamp.getTime() - startTime) / timeRangeMs) * width
      const y = top + height - (point.value / 100) * height

      if (index === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })

    ctx.stroke()

    // Draw data points
    ctx.fillStyle = series.color
    seriesData.forEach(point => {
      const x = left + ((point.timestamp.getTime() - startTime) / timeRangeMs) * width
      const y = top + height - (point.value / 100) * height

      ctx.beginPath()
      ctx.arc(x, y, 3, 0, 2 * Math.PI)
      ctx.fill()
    })
  })
}

const getTimeRangeMs = (range: string): number => {
  const ranges = {
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000
  }
  return ranges[range as keyof typeof ranges] || ranges['15m']
}

const formatTimeLabel = (date: Date): string => {
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMinutes = Math.floor(diffMs / (1000 * 60))

  if (diffMinutes < 1) return 'now'
  if (diffMinutes < 60) return `${diffMinutes}m`
  
  const diffHours = Math.floor(diffMinutes / 60)
  return `${diffHours}h`
}

const getTrendColor = (trend: number) => {
  if (trend > 5) return 'text-green-600 dark:text-green-400'
  if (trend < -5) return 'text-red-600 dark:text-red-400'
  return 'text-slate-600 dark:text-slate-400'
}

const onMouseMove = (event: MouseEvent) => {
  // Simple tooltip implementation
  const rect = chartCanvas.value!.getBoundingClientRect()
  const x = event.clientX - rect.left
  const y = event.clientY - rect.top

  // For now, just show basic tooltip
  tooltip.value = {
    visible: true,
    x: event.clientX - rect.left + 10,
    y: event.clientY - rect.top - 30,
    title: 'Performance Data',
    data: {
      'Metric': selectedMetric.value,
      'Time Range': timeRange.value,
      'Data Points': dataPoints.value.length.toString()
    }
  }
}

const hideTooltip = () => {
  tooltip.value.visible = false
}

// Event handlers and lifecycle
let unsubscribeHandlers: Array<() => void> = []

// Watch for agent updates
watch(agents, (newAgents) => {
  // Add data points for active agents
  newAgents.forEach(agent => {
    if (agent.status === 'active' || agent.status === 'busy') {
      // Randomly add data points (in real implementation, this would be event-driven)
      if (Math.random() > 0.8) { // 20% chance per update
        addDataPoint(agent)
      }
    }
  })
}, { deep: true })

onMounted(() => {
  // Subscribe to agent lifecycle events
  unsubscribeHandlers.push(
    onEvent('agent_lifecycle_event', (event) => {
      if (event.event_type === 'task_completed' || event.event_type === 'agent_heartbeat') {
        // Find the agent and add a data point
        const agent = agents.value.find(a => a.id === event.agent_id)
        if (agent) {
          addDataPoint(agent)
        }
      }
    }),

    onEvent('performance_metrics', (metrics: PerformanceMetrics) => {
      // Add system-level data point
      const systemDataPoint = {
        timestamp: new Date(),
        agentId: 'system',
        agentName: 'System',
        performance: 100 - metrics.cpu_usage_percent,
        cpu: metrics.cpu_usage_percent,
        memory: metrics.memory_usage_percent,
        tasks: 85 // Mock task completion rate
      }
      
      dataPoints.value.push(systemDataPoint)
      
      if (dataPoints.value.length > maxDataPoints.value) {
        dataPoints.value.shift()
      }
      
      nextTick(() => {
        drawChart()
      })
    })
  )

  // Initial chart draw
  nextTick(() => {
    drawChart()
  })

  // Redraw chart on window resize
  window.addEventListener('resize', () => {
    nextTick(() => {
      drawChart()
    })
  })

  console.log('📊 Real-time agent performance chart initialized')
})

onUnmounted(() => {
  unsubscribeHandlers.forEach(unsubscribe => unsubscribe())
  unsubscribeHandlers = []

  if (animationFrame) {
    cancelAnimationFrame(animationFrame)
  }

  window.removeEventListener('resize', drawChart)
})
</script>

<style scoped>
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

canvas {
  cursor: crosshair;
}
</style>