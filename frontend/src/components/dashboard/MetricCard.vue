<template>
  <div class="metric-card">
    <div class="flex items-center justify-between mb-4">
      <div class="flex items-center space-x-3">
        <div
          class="p-2 rounded-lg"
          :class="iconBackgroundColor"
        >
          <component
            :is="iconComponent"
            class="w-5 h-5"
            :class="iconColor"
          />
        </div>
        <h3 class="text-sm font-medium text-slate-600 dark:text-slate-400">
          {{ title }}
        </h3>
      </div>
      
      <!-- Trend indicator -->
      <div v-if="trend !== null" class="flex items-center space-x-1">
        <component
          :is="trendIcon"
          class="w-4 h-4"
          :class="trendColor"
        />
        <span class="text-xs font-medium" :class="trendColor">
          {{ Math.abs(trend).toFixed(1) }}%
        </span>
      </div>
    </div>
    
    <!-- Main value -->
    <div class="mb-2">
      <div class="text-2xl font-bold text-slate-900 dark:text-white">
        {{ formattedValue }}
        <span v-if="unit" class="text-lg font-normal text-slate-500 dark:text-slate-400">
          {{ unit }}
        </span>
      </div>
    </div>
    
    <!-- Secondary info -->
    <div class="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
      <span v-if="previous !== null">
        Previous: {{ formatValue(previous) }}{{ unit }}
      </span>
      <span v-if="lastUpdated">
        {{ formatTime(lastUpdated) }}
      </span>
    </div>
    
    <!-- Mini chart (optional) -->
    <div v-if="chartData && chartData.length > 0" class="mt-4">
      <canvas
        ref="chartCanvas"
        class="w-full h-8"
      ></canvas>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, onMounted, nextTick } from 'vue'
import { formatDistanceToNow } from 'date-fns'
import {
  UserGroupIcon,
  BoltIcon,
  CheckCircleIcon,
  ClockIcon,
  CpuChipIcon,
  ServerStackIcon,
  SignalIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  MinusIcon,
} from '@heroicons/vue/24/outline'

interface Props {
  title: string
  value: number
  unit?: string
  previous?: number | null
  icon?: string
  color?: 'primary' | 'success' | 'warning' | 'danger' | 'info'
  chartData?: Array<{ timestamp: string; value: number }>
  lastUpdated?: Date
}

const props = withDefaults(defineProps<Props>(), {
  unit: '',
  previous: null,
  icon: 'signal',
  color: 'primary',
  chartData: () => [],
  lastUpdated: undefined,
})

const chartCanvas = ref<HTMLCanvasElement | null>(null)

// Computed properties
const formattedValue = computed(() => formatValue(props.value))

const trend = computed(() => {
  if (props.previous === null || props.previous === 0) return null
  return ((props.value - props.previous) / props.previous) * 100
})

const trendIcon = computed(() => {
  if (trend.value === null) return MinusIcon
  return trend.value > 0 ? ArrowTrendingUpIcon : ArrowTrendingDownIcon
})

const trendColor = computed(() => {
  if (trend.value === null) return 'text-slate-400'
  return trend.value > 0 ? 'text-success-600 dark:text-success-400' : 'text-danger-600 dark:text-danger-400'
})

const iconComponent = computed(() => {
  switch (props.icon) {
    case 'users': return UserGroupIcon
    case 'lightning-bolt': return BoltIcon
    case 'check-circle': return CheckCircleIcon
    case 'clock': return ClockIcon
    case 'cpu': return CpuChipIcon
    case 'server': return ServerStackIcon
    default: return SignalIcon
  }
})

const iconColor = computed(() => {
  switch (props.color) {
    case 'success': return 'text-success-600'
    case 'warning': return 'text-warning-600'
    case 'danger': return 'text-danger-600'
    case 'info': return 'text-blue-600'
    default: return 'text-primary-600'
  }
})

const iconBackgroundColor = computed(() => {
  switch (props.color) {
    case 'success': return 'bg-success-100 dark:bg-success-900/20'
    case 'warning': return 'bg-warning-100 dark:bg-warning-900/20'
    case 'danger': return 'bg-danger-100 dark:bg-danger-900/20'
    case 'info': return 'bg-blue-100 dark:bg-blue-900/20'
    default: return 'bg-primary-100 dark:bg-primary-900/20'
  }
})

// Helper functions
const formatValue = (value: number): string => {
  if (value >= 1000000) {
    return (value / 1000000).toFixed(1) + 'M'
  }
  if (value >= 1000) {
    return (value / 1000).toFixed(1) + 'K'
  }
  if (value >= 100) {
    return Math.round(value).toString()
  }
  if (value >= 10) {
    return value.toFixed(1)
  }
  return value.toFixed(2)
}

const formatTime = (date: Date): string => {
  return formatDistanceToNow(date, { addSuffix: true })
}

const drawMiniChart = () => {
  if (!chartCanvas.value || !props.chartData || props.chartData.length === 0) {
    return
  }
  
  const canvas = chartCanvas.value
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  
  // Set canvas size
  const rect = canvas.getBoundingClientRect()
  canvas.width = rect.width * window.devicePixelRatio
  canvas.height = rect.height * window.devicePixelRatio
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio)
  
  // Clear canvas
  ctx.clearRect(0, 0, rect.width, rect.height)
  
  // Prepare data
  const data = props.chartData.map(d => d.value)
  const minValue = Math.min(...data)
  const maxValue = Math.max(...data)
  const range = maxValue - minValue || 1
  
  // Draw line
  ctx.strokeStyle = getComputedStyle(document.documentElement)
    .getPropertyValue(`--color-${props.color}-500`) || '#3b82f6'
  ctx.lineWidth = 1.5
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  
  ctx.beginPath()
  data.forEach((value, index) => {
    const x = (index / (data.length - 1)) * rect.width
    const y = rect.height - ((value - minValue) / range) * rect.height
    
    if (index === 0) {
      ctx.moveTo(x, y)
    } else {
      ctx.lineTo(x, y)
    }
  })
  ctx.stroke()
  
  // Draw fill area
  ctx.globalAlpha = 0.1
  ctx.fillStyle = ctx.strokeStyle
  ctx.lineTo(rect.width, rect.height)
  ctx.lineTo(0, rect.height)
  ctx.closePath()
  ctx.fill()
}

// Lifecycle
onMounted(async () => {
  await nextTick()
  drawMiniChart()
})
</script>