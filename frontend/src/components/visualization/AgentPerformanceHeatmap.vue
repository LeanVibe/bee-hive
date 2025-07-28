<template>
  <div class="agent-performance-heatmap">
    <!-- Header -->
    <div class="heatmap-header flex items-center justify-between mb-4">
      <div class="flex items-center space-x-3">
        <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
          Performance Heatmap
        </h3>
        <div class="text-sm text-gray-500 dark:text-gray-400">
          {{ formatTimeRange() }}
        </div>
      </div>
      
      <div class="flex items-center space-x-2">
        <!-- Metric Selector -->
        <select
          v-model="selectedMetric"
          @change="updateHeatmap"
          class="text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
        >
          <option value="performance">Performance Score</option>
          <option value="activity">Activity Level</option>
          <option value="errors">Error Rate</option>
          <option value="latency">Response Latency</option>
          <option value="memory">Memory Usage</option>
        </select>
        
        <!-- Time Range -->
        <select
          v-model="timeRange"
          @change="updateTimeRange"
          class="text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
        >
          <option value="1h">Last Hour</option>
          <option value="6h">Last 6 Hours</option>
          <option value="24h">Last 24 Hours</option>
          <option value="7d">Last 7 Days</option>
        </select>
        
        <!-- Refresh Button -->
        <button
          @click="refreshData"
          :disabled="isLoading"
          class="p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
          :class="{ 'animate-spin': isLoading }"
        >
          <ArrowPathIcon class="w-4 h-4" />
        </button>
      </div>
    </div>
    
    <!-- Loading State -->
    <div
      v-if="isLoading"
      class="flex items-center justify-center h-64 bg-gray-50 dark:bg-gray-800 rounded-lg"
    >
      <div class="text-center">
        <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
        <div class="text-sm text-gray-600 dark:text-gray-400">Loading heatmap data...</div>
      </div>
    </div>
    
    <!-- Heatmap Container -->
    <div
      v-else
      class="heatmap-container relative bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden"
    >
      <!-- Legend -->
      <div class="heatmap-legend absolute top-4 right-4 z-10 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-3 border border-gray-200 dark:border-gray-700">
        <div class="text-xs font-medium text-gray-900 dark:text-white mb-2">
          {{ getMetricLabel() }}
        </div>
        <div class="flex items-center space-x-2">
          <div class="text-xs text-gray-600 dark:text-gray-400">Low</div>
          <div class="legend-gradient" :style="{ background: getLegendGradient() }"></div>
          <div class="text-xs text-gray-600 dark:text-gray-400">High</div>
        </div>
        <div class="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
          <span>{{ formatValue(metricRange.min) }}</span>
          <span>{{ formatValue(metricRange.max) }}</span>
        </div>
      </div>
      
      <!-- SVG Heatmap -->
      <svg
        ref="heatmapSvg"
        :width="dimensions.width"
        :height="dimensions.height"
        class="w-full h-auto"
        @mousemove="handleMouseMove"
        @mouseleave="hideTooltip"
      >
        <!-- Grid Background -->
        <g class="grid-background">
          <defs>
            <pattern
              id="grid"
              width="1"
              height="1"
              patternUnits="objectBoundingBox"
            >
              <rect width="1" height="1" fill="none" stroke="#E5E7EB" stroke-width="0.5" opacity="0.3"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </g>
        
        <!-- Time Labels (X-axis) -->
        <g class="time-labels">
          <text
            v-for="(label, index) in timeLabels"
            :key="`time-${index}`"
            :x="margin.left + (index * cellWidth) + cellWidth / 2"
            :y="dimensions.height - margin.bottom + 15"
            text-anchor="middle"
            class="text-xs fill-gray-600 dark:fill-gray-400"
          >
            {{ label }}
          </text>
        </g>
        
        <!-- Agent Labels (Y-axis) -->
        <g class="agent-labels">
          <text
            v-for="(agent, index) in sortedAgents"
            :key="`agent-${agent.agent_id}`"
            :x="margin.left - 8"
            :y="margin.top + (index * cellHeight) + cellHeight / 2"
            text-anchor="end"
            dominant-baseline="middle"
            class="text-xs fill-gray-600 dark:fill-gray-400 cursor-pointer hover:fill-gray-900 dark:hover:fill-gray-100"
            @click="selectAgent(agent)"
          >
            {{ formatAgentLabel(agent) }}
          </text>
        </g>
        
        <!-- Heatmap Cells -->
        <g class="heatmap-cells">
          <rect
            v-for="cell in heatmapCells"
            :key="`cell-${cell.agentId}-${cell.timeIndex}`"
            :x="cell.x"
            :y="cell.y"
            :width="cellWidth"
            :height="cellHeight"
            :fill="cell.color"
            :opacity="cell.opacity"
            class="heatmap-cell transition-all duration-200 hover:stroke-gray-900 dark:hover:stroke-gray-100 hover:stroke-2"
            @mouseenter="showTooltip($event, cell)"
            @click="selectCell(cell)"
          />
        </g>
        
        <!-- Performance Trend Lines -->
        <g v-if="showTrendLines" class="trend-lines">
          <path
            v-for="trend in trendLines"
            :key="`trend-${trend.agentId}`"
            :d="trend.path"
            :stroke="trend.color"
            stroke-width="2"
            fill="none"
            opacity="0.7"
            class="trend-line"
          />
        </g>
        
        <!-- Anomaly Indicators -->
        <g class="anomalies">
          <circle
            v-for="anomaly in anomalies"
            :key="`anomaly-${anomaly.id}`"
            :cx="anomaly.x"
            :cy="anomaly.y"
            :r="4"
            :fill="anomaly.severity === 'high' ? '#EF4444' : '#F59E0B'"
            stroke="#FFFFFF"
            stroke-width="2"
            class="anomaly-indicator animate-pulse cursor-pointer"
            @click="showAnomalyDetails(anomaly)"
          />
        </g>
      </svg>
      
      <!-- Tooltip -->
      <div
        v-if="tooltip.visible"
        class="tooltip absolute z-20 bg-gray-900 text-white text-xs rounded-lg p-2 pointer-events-none shadow-lg"
        :style="{ left: tooltip.x + 'px', top: tooltip.y + 'px' }"
      >
        <div class="font-semibold">{{ tooltip.agent }}</div>
        <div class="text-gray-300">{{ tooltip.time }}</div>
        <div class="mt-1">
          <span class="text-gray-400">{{ getMetricLabel() }}:</span>
          <span class="ml-1 font-medium">{{ tooltip.value }}</span>
        </div>
        <div v-if="tooltip.trend" class="text-xs text-gray-400 mt-1">
          Trend: {{ tooltip.trend }}
        </div>
      </div>
    </div>
    
    <!-- Heatmap Analytics -->
    <div class="heatmap-analytics mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
      <!-- Performance Distribution -->
      <div class="analytics-card bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
        <h4 class="text-sm font-semibold text-gray-900 dark:text-white mb-2">
          Performance Distribution
        </h4>
        <div class="space-y-2">
          <div
            v-for="bucket in performanceDistribution"
            :key="bucket.label"
            class="flex items-center justify-between text-xs"
          >
            <div class="flex items-center">
              <div
                class="w-3 h-3 rounded-full mr-2"
                :style="{ backgroundColor: bucket.color }"
              ></div>
              <span class="text-gray-600 dark:text-gray-400">{{ bucket.label }}</span>
            </div>
            <span class="font-mono text-gray-900 dark:text-white">{{ bucket.count }}</span>
          </div>
        </div>
      </div>
      
      <!-- Top Performers -->
      <div class="analytics-card bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
        <h4 class="text-sm font-semibold text-gray-900 dark:text-white mb-2">
          Top Performers
        </h4>
        <div class="space-y-2">
          <div
            v-for="performer in topPerformers"
            :key="performer.agentId"
            class="flex items-center justify-between text-xs cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700 rounded p-1"
            @click="selectAgent(performer)"
          >
            <span class="text-gray-600 dark:text-gray-400">{{ performer.name }}</span>
            <span
              class="font-mono font-semibold"
              :style="{ color: getPerformanceColor(performer.value) }"
            >
              {{ formatValue(performer.value) }}
            </span>
          </div>
        </div>
      </div>
      
      <!-- Performance Trends -->
      <div class="analytics-card bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
        <h4 class="text-sm font-semibold text-gray-900 dark:text-white mb-2">
          System Trends
        </h4>
        <div class="space-y-2 text-xs">
          <div class="flex justify-between">
            <span class="text-gray-600 dark:text-gray-400">Avg {{ getMetricLabel() }}:</span>
            <span class="font-mono text-gray-900 dark:text-white">{{ formatValue(averageMetric) }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-600 dark:text-gray-400">Peak Value:</span>
            <span class="font-mono text-gray-900 dark:text-white">{{ formatValue(peakValue) }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-600 dark:text-gray-400">Trend:</span>
            <span
              class="flex items-center font-medium"
              :class="overallTrend.direction === 'up' ? 'text-green-600' : overallTrend.direction === 'down' ? 'text-red-600' : 'text-gray-600'"
            >
              <component
                :is="getTrendIcon(overallTrend.direction)"
                class="w-3 h-3 mr-1"
              />
              {{ overallTrend.percentage }}%
            </span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-600 dark:text-gray-400">Anomalies:</span>
            <span class="font-mono text-gray-900 dark:text-white">{{ anomalies.length }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { scaleLinear, scaleTime } from 'd3-scale'
import { interpolateViridis, interpolateRdYlBu } from 'd3-scale-chromatic'
import { line, curveMonotoneX } from 'd3-shape'
import { extent, max, min } from 'd3-array'
import { 
  ArrowPathIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  MinusIcon
} from '@heroicons/vue/24/outline'

import { useSessionColors } from '@/utils/SessionColorManager'
import { useEventsStore } from '@/stores/events'
import type { AgentInfo, SessionInfo } from '@/types/hooks'

// Props
interface Props {
  agents: AgentInfo[]
  sessions: SessionInfo[]
  width?: number
  height?: number
  showTrendLines?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  width: 800,
  height: 400,
  showTrendLines: false
})

// Emits
const emit = defineEmits<{
  'agent-selected': [agent: AgentInfo]
  'cell-selected': [data: any]
  'anomaly-detected': [anomaly: any]
}>()

// Reactive state
const heatmapSvg = ref<SVGElement>()
const isLoading = ref(false)
const selectedMetric = ref<'performance' | 'activity' | 'errors' | 'latency' | 'memory'>('performance')
const timeRange = ref<'1h' | '6h' | '24h' | '7d'>('6h')
const showTrendLines = ref(false)

// Stores and utilities
const eventsStore = useEventsStore()
const { getPerformanceColor, createHeatmapScale } = useSessionColors()

// Dimensions and layout
const dimensions = reactive({
  width: props.width,
  height: props.height
})

const margin = reactive({
  top: 20,
  right: 20,
  bottom: 40,
  left: 100
})

const cellWidth = computed(() => {
  const availableWidth = dimensions.width - margin.left - margin.right
  return Math.max(20, Math.floor(availableWidth / timeLabels.value.length))
})

const cellHeight = computed(() => {
  const availableHeight = dimensions.height - margin.top - margin.bottom
  return Math.max(15, Math.floor(availableHeight / sortedAgents.value.length))
})

// Tooltip state
const tooltip = reactive({
  visible: false,
  x: 0,
  y: 0,
  agent: '',
  time: '',
  value: '',
  trend: ''
})

// Heatmap data
const heatmapData = ref<Array<{
  agentId: string
  timestamp: string
  value: number
  originalValue: any
}>>([])

const metricRange = computed(() => {
  const values = heatmapData.value.map(d => d.value)
  return {
    min: min(values) || 0,
    max: max(values) || 100
  }
})

// Time labels based on range
const timeLabels = computed(() => {
  const now = new Date()
  const labels: string[] = []
  
  switch (timeRange.value) {
    case '1h':
      for (let i = 11; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 5 * 60000) // 5-minute intervals
        labels.push(time.getHours().toString().padStart(2, '0') + ':' + 
                    time.getMinutes().toString().padStart(2, '0'))
      }
      break
    case '6h':
      for (let i = 11; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 30 * 60000) // 30-minute intervals
        labels.push(time.getHours().toString().padStart(2, '0') + ':' + 
                    time.getMinutes().toString().padStart(2, '0'))
      }
      break
    case '24h':
      for (let i = 11; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 2 * 3600000) // 2-hour intervals
        labels.push(time.getHours().toString().padStart(2, '0') + ':00')
      }
      break
    case '7d':
      for (let i = 6; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 24 * 3600000) // Daily intervals
        labels.push(time.toLocaleDateString('en-US', { weekday: 'short' }))
      }
      break
  }
  
  return labels
})

// Sorted agents by performance
const sortedAgents = computed(() => {
  return [...props.agents].sort((a, b) => {
    const aValue = calculateMetricValue(a, selectedMetric.value)
    const bValue = calculateMetricValue(b, selectedMetric.value)
    return bValue - aValue // Descending order
  })
})

// Heatmap cells
const heatmapCells = computed(() => {
  const cells: Array<{
    agentId: string
    timeIndex: number
    x: number
    y: number
    color: string
    opacity: number
    value: number
    agent: AgentInfo
    timestamp: string
  }> = []
  
  const colorScale = createHeatmapScale([metricRange.value.min, metricRange.value.max])
  
  sortedAgents.value.forEach((agent, agentIndex) => {
    timeLabels.value.forEach((_, timeIndex) => {
      const dataPoint = heatmapData.value.find(d => 
        d.agentId === agent.agent_id && 
        getTimeIndexForTimestamp(d.timestamp) === timeIndex
      )
      
      const value = dataPoint?.value ?? calculateMetricValue(agent, selectedMetric.value)
      
      cells.push({
        agentId: agent.agent_id,
        timeIndex,
        x: margin.left + timeIndex * cellWidth.value,
        y: margin.top + agentIndex * cellHeight.value,
        color: colorScale(value),
        opacity: dataPoint ? 1.0 : 0.5, // Lower opacity for estimated values
        value,
        agent,
        timestamp: dataPoint?.timestamp || new Date().toISOString()
      })
    })
  })
  
  return cells
})

// Performance distribution
const performanceDistribution = computed(() => {
  const buckets = {
    excellent: { label: '90-100%', count: 0, color: '#10B981' },
    good: { label: '70-89%', count: 0, color: '#3B82F6' },
    average: { label: '50-69%', count: 0, color: '#F59E0B' },
    poor: { label: '30-49%', count: 0, color: '#EF4444' },
    critical: { label: '0-29%', count: 0, color: '#7F1D1D' }
  }
  
  heatmapCells.value.forEach(cell => {
    const value = cell.value
    if (value >= 90) buckets.excellent.count++
    else if (value >= 70) buckets.good.count++
    else if (value >= 50) buckets.average.count++
    else if (value >= 30) buckets.poor.count++
    else buckets.critical.count++
  })
  
  return Object.values(buckets)
})

// Top performers
const topPerformers = computed(() => {
  return sortedAgents.value
    .slice(0, 5)
    .map(agent => ({
      agentId: agent.agent_id,
      name: formatAgentLabel(agent),
      value: calculateMetricValue(agent, selectedMetric.value)
    }))
})

// Analytics
const averageMetric = computed(() => {
  const values = heatmapCells.value.map(c => c.value)
  return values.length > 0 ? values.reduce((sum, val) => sum + val, 0) / values.length : 0
})

const peakValue = computed(() => metricRange.value.max)

const overallTrend = computed(() => {
  // Calculate trend over time
  const timeSlices = timeLabels.value.map((_, index) => {
    const sliceCells = heatmapCells.value.filter(c => c.timeIndex === index)
    const average = sliceCells.length > 0 
      ? sliceCells.reduce((sum, c) => sum + c.value, 0) / sliceCells.length 
      : 0
    return average
  })
  
  if (timeSlices.length < 2) {
    return { direction: 'stable' as const, percentage: 0 }
  }
  
  const firstHalf = timeSlices.slice(0, Math.floor(timeSlices.length / 2))
  const secondHalf = timeSlices.slice(Math.floor(timeSlices.length / 2))
  
  const firstAvg = firstHalf.reduce((sum, val) => sum + val, 0) / firstHalf.length
  const secondAvg = secondHalf.reduce((sum, val) => sum + val, 0) / secondHalf.length
  
  const percentage = Math.abs(((secondAvg - firstAvg) / firstAvg) * 100)
  const direction = secondAvg > firstAvg ? 'up' : secondAvg < firstAvg ? 'down' : 'stable'
  
  return { direction, percentage: Math.round(percentage) }
})

// Trend lines
const trendLines = computed(() => {
  if (!showTrendLines.value) return []
  
  return sortedAgents.value.map((agent, agentIndex) => {
    const agentCells = heatmapCells.value.filter(c => c.agentId === agent.agent_id)
    
    const lineGenerator = line<typeof agentCells[0]>()
      .x(d => d.x + cellWidth.value / 2)
      .y(d => d.y + cellHeight.value / 2)
      .curve(curveMonotoneX)
    
    return {
      agentId: agent.agent_id,
      path: lineGenerator(agentCells) || '',
      color: getPerformanceColor(calculateMetricValue(agent, selectedMetric.value))
    }
  })
})

// Anomalies detection
const anomalies = computed(() => {
  const anomaliesFound: Array<{
    id: string
    x: number
    y: number
    severity: 'high' | 'medium'
    agent: AgentInfo
    value: number
    expected: number
  }> = []
  
  // Find cells with values significantly different from their neighbors
  heatmapCells.value.forEach(cell => {
    const neighbors = heatmapCells.value.filter(c => 
      c.agentId === cell.agentId && 
      Math.abs(c.timeIndex - cell.timeIndex) <= 1 &&
      c !== cell
    )
    
    if (neighbors.length === 0) return
    
    const neighborAvg = neighbors.reduce((sum, n) => sum + n.value, 0) / neighbors.length
    const deviation = Math.abs(cell.value - neighborAvg)
    const threshold = neighborAvg * 0.3 // 30% deviation threshold
    
    if (deviation > threshold) {
      anomaliesFound.push({
        id: `${cell.agentId}-${cell.timeIndex}`,
        x: cell.x + cellWidth.value / 2,
        y: cell.y + cellHeight.value / 2,
        severity: deviation > neighborAvg * 0.5 ? 'high' : 'medium',
        agent: cell.agent,
        value: cell.value,
        expected: neighborAvg
      })
    }
  })
  
  return anomaliesFound
})

// Methods
const updateHeatmap = () => {
  generateMockData()
}

const updateTimeRange = () => {
  generateMockData()
}

const refreshData = async () => {
  isLoading.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 1000)) // Simulate API call
    generateMockData()
  } finally {
    isLoading.value = false
  }
}

const generateMockData = () => {
  heatmapData.value = []
  
  const now = new Date()
  const timeIntervals = getTimeIntervals()
  
  props.agents.forEach(agent => {
    timeIntervals.forEach((timestamp, index) => {
      // Generate realistic data with some trends and variations
      let baseValue = calculateMetricValue(agent, selectedMetric.value)
      
      // Add time-based variation
      const timeVariation = Math.sin((index / timeIntervals.length) * Math.PI * 2) * 10
      
      // Add random noise
      const noise = (Math.random() - 0.5) * 20
      
      // Add some agent-specific patterns
      const agentPattern = (agent.agent_id.charCodeAt(0) % 10) * 2
      
      const value = Math.max(0, Math.min(100, baseValue + timeVariation + noise + agentPattern))
      
      heatmapData.value.push({
        agentId: agent.agent_id,
        timestamp: timestamp.toISOString(),
        value,
        originalValue: value
      })
    })
  })
}

const getTimeIntervals = (): Date[] => {
  const now = new Date()
  const intervals: Date[] = []
  
  switch (timeRange.value) {
    case '1h':
      for (let i = 11; i >= 0; i--) {
        intervals.push(new Date(now.getTime() - i * 5 * 60000))
      }
      break
    case '6h':
      for (let i = 11; i >= 0; i--) {
        intervals.push(new Date(now.getTime() - i * 30 * 60000))
      }
      break
    case '24h':
      for (let i = 11; i >= 0; i--) {
        intervals.push(new Date(now.getTime() - i * 2 * 3600000))
      }
      break
    case '7d':
      for (let i = 6; i >= 0; i--) {
        intervals.push(new Date(now.getTime() - i * 24 * 3600000))
      }
      break
  }
  
  return intervals
}

const getTimeIndexForTimestamp = (timestamp: string): number => {
  const date = new Date(timestamp)
  const intervals = getTimeIntervals()
  
  // Find the closest time interval
  let closestIndex = 0
  let closestDiff = Math.abs(date.getTime() - intervals[0].getTime())
  
  intervals.forEach((interval, index) => {
    const diff = Math.abs(date.getTime() - interval.getTime())
    if (diff < closestDiff) {
      closestDiff = diff
      closestIndex = index
    }
  })
  
  return closestIndex
}

const calculateMetricValue = (agent: AgentInfo, metric: string): number => {
  switch (metric) {
    case 'performance':
      const errorRate = agent.error_count / Math.max(agent.event_count, 1)
      return Math.max(0, 100 - (errorRate * 100))
    case 'activity':
      return Math.min(100, agent.event_count * 2)
    case 'errors':
      return agent.error_count
    case 'latency':
      return Math.random() * 500 + 50 // Mock latency
    case 'memory':
      return Math.random() * 200 + 50 // Mock memory usage
    default:
      return 50
  }
}

const formatAgentLabel = (agent: AgentInfo): string => {
  return `Agent-${agent.agent_id.slice(-4)}`
}

const formatTimeRange = (): string => {
  const ranges = {
    '1h': 'Last Hour (5min intervals)',
    '6h': 'Last 6 Hours (30min intervals)', 
    '24h': 'Last 24 Hours (2hr intervals)',
    '7d': 'Last 7 Days (daily)'
  }
  return ranges[timeRange.value]
}

const getMetricLabel = (): string => {
  const labels = {
    performance: 'Performance Score',
    activity: 'Activity Level',
    errors: 'Error Count',
    latency: 'Response Time (ms)', 
    memory: 'Memory Usage (MB)'
  }
  return labels[selectedMetric.value]
}

const formatValue = (value: number): string => {
  switch (selectedMetric.value) {
    case 'performance':
    case 'activity':
      return `${Math.round(value)}%`
    case 'errors':
      return Math.round(value).toString()
    case 'latency':
      return `${Math.round(value)}ms`
    case 'memory':
      return `${Math.round(value)}MB`
    default:
      return Math.round(value).toString()
  }
}

const getLegendGradient = (): string => {
  return `linear-gradient(to right, ${interpolateViridis(0)}, ${interpolateViridis(0.5)}, ${interpolateViridis(1)})`
}

const getTrendIcon = (direction: string) => {
  switch (direction) {
    case 'up': return ArrowTrendingUpIcon
    case 'down': return ArrowTrendingDownIcon
    default: return MinusIcon
  }
}

// Event handlers
const handleMouseMove = (event: MouseEvent) => {
  // Mouse tracking is handled by individual cell hover events
}

const showTooltip = (event: MouseEvent, cell: any) => {
  tooltip.visible = true
  tooltip.x = event.offsetX + 10
  tooltip.y = event.offsetY - 30
  tooltip.agent = formatAgentLabel(cell.agent)
  tooltip.time = timeLabels.value[cell.timeIndex]
  tooltip.value = formatValue(cell.value)
  
  // Calculate trend
  const previousCells = heatmapCells.value.filter(c => 
    c.agentId === cell.agentId && c.timeIndex < cell.timeIndex
  )
  if (previousCells.length > 0) {
    const previousValue = previousCells[previousCells.length - 1].value
    const trend = cell.value > previousValue ? '↗ Increasing' : 
                 cell.value < previousValue ? '↘ Decreasing' : '→ Stable'
    tooltip.trend = trend
  }
}

const hideTooltip = () => {
  tooltip.visible = false
}

const selectAgent = (agent: AgentInfo) => {
  emit('agent-selected', agent)
}

const selectCell = (cell: any) => {
  emit('cell-selected', {
    agent: cell.agent,
    timestamp: cell.timestamp,
    value: cell.value,
    metric: selectedMetric.value
  })
}

const showAnomalyDetails = (anomaly: any) => {
  emit('anomaly-detected', anomaly)
}

// Lifecycle
onMounted(() => {
  generateMockData()
})

// Watchers
watch(() => props.agents, () => {
  generateMockData()
}, { deep: true })

watch([selectedMetric, timeRange], () => {
  updateHeatmap()
})
</script>

<style scoped>
.agent-performance-heatmap {
  @apply w-full;
}

.heatmap-container {
  min-height: 400px;
}

.legend-gradient {
  width: 60px;
  height: 12px;
  border-radius: 6px;
}

.heatmap-cell {
  cursor: pointer;
}

.heatmap-cell:hover {
  filter: brightness(1.1);
}

.trend-line {
  pointer-events: none;
}

.anomaly-indicator {
  cursor: pointer;
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.2));
}

.tooltip {
  max-width: 200px;
  z-index: 1000;
}

.analytics-card {
  transition: all 0.2s ease;
}

.analytics-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .agent-performance-heatmap {
    font-size: 0.75rem;
  }
  
  .heatmap-legend {
    position: relative;
    top: auto;
    right: auto;
    margin-bottom: 1rem;
  }
  
  .heatmap-analytics {
    grid-template-columns: 1fr;
  }
}
</style>