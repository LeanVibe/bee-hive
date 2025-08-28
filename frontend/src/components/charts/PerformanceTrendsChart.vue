<template>
  <div class="relative">
    <canvas 
      ref="chartCanvas" 
      :width="width" 
      :height="height"
      class="max-w-full"
    />
    
    <!-- Loading overlay -->
    <div 
      v-if="loading" 
      class="absolute inset-0 bg-white/50 dark:bg-slate-800/50 flex items-center justify-center rounded-lg"
    >
      <div class="animate-spin rounded-full h-6 w-6 border-b-2 border-indigo-500"></div>
    </div>

    <!-- No data overlay -->
    <div 
      v-if="!loading && (!data || data.length === 0)" 
      class="absolute inset-0 flex items-center justify-center text-slate-500 dark:text-slate-400"
    >
      <div class="text-center">
        <ChartBarIcon class="w-8 h-8 mx-auto mb-2 opacity-50" />
        <p class="text-sm">No performance data available</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import { ChartBarIcon } from '@heroicons/vue/24/outline'
import {
  Chart,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  type ChartConfiguration,
  type ChartData,
  type ChartOptions
} from 'chart.js'

// Register Chart.js components
Chart.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

// Types
interface PerformanceDataPoint {
  timestamp: string
  taskThroughput: number
  responseTime: number
  systemUtilization: number
  errorRate: number
  businessValue: number
}

interface Props {
  data: PerformanceDataPoint[]
  timeframe?: string
  width?: number
  height?: number
  loading?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  timeframe: '24h',
  width: 800,
  height: 300,
  loading: false
})

// Template refs
const chartCanvas = ref<HTMLCanvasElement | null>(null)

// State
let chartInstance: Chart | null = null

// Chart configuration
const createChartConfig = (): ChartConfiguration => {
  const isDark = document.documentElement.classList.contains('dark')
  
  const chartData: ChartData = {
    labels: props.data.map(point => formatTimestamp(point.timestamp, props.timeframe)),
    datasets: [
      {
        label: 'Task Throughput',
        data: props.data.map(point => point.taskThroughput),
        borderColor: isDark ? '#60A5FA' : '#3B82F6',
        backgroundColor: isDark ? '#60A5FA20' : '#3B82F620',
        fill: false,
        tension: 0.4,
        pointRadius: 3,
        pointHoverRadius: 5,
        borderWidth: 2
      },
      {
        label: 'Response Time (ms)',
        data: props.data.map(point => point.responseTime),
        borderColor: isDark ? '#F59E0B' : '#D97706',
        backgroundColor: isDark ? '#F59E0B20' : '#D9770620',
        fill: false,
        tension: 0.4,
        pointRadius: 3,
        pointHoverRadius: 5,
        borderWidth: 2,
        yAxisID: 'y1'
      },
      {
        label: 'System Utilization (%)',
        data: props.data.map(point => point.systemUtilization * 100),
        borderColor: isDark ? '#10B981' : '#059669',
        backgroundColor: isDark ? '#10B98120' : '#05966920',
        fill: true,
        tension: 0.4,
        pointRadius: 2,
        pointHoverRadius: 4,
        borderWidth: 2
      },
      {
        label: 'Error Rate (%)',
        data: props.data.map(point => point.errorRate * 100),
        borderColor: isDark ? '#EF4444' : '#DC2626',
        backgroundColor: isDark ? '#EF444420' : '#DC262620',
        fill: false,
        tension: 0.4,
        pointRadius: 2,
        pointHoverRadius: 4,
        borderWidth: 2
      },
      {
        label: 'Business Value',
        data: props.data.map(point => point.businessValue),
        borderColor: isDark ? '#8B5CF6' : '#7C3AED',
        backgroundColor: isDark ? '#8B5CF620' : '#7C3AED20',
        fill: false,
        tension: 0.4,
        pointRadius: 3,
        pointHoverRadius: 5,
        borderWidth: 2,
        yAxisID: 'y2'
      }
    ]
  }

  const options: ChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false
    },
    plugins: {
      title: {
        display: false
      },
      legend: {
        display: true,
        position: 'top',
        labels: {
          usePointStyle: true,
          color: isDark ? '#E2E8F0' : '#374151',
          font: {
            size: 12
          },
          padding: 20
        }
      },
      tooltip: {
        backgroundColor: isDark ? '#1E293B' : '#FFFFFF',
        titleColor: isDark ? '#F1F5F9' : '#111827',
        bodyColor: isDark ? '#E2E8F0' : '#374151',
        borderColor: isDark ? '#475569' : '#D1D5DB',
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: true,
        callbacks: {
          title: (context) => {
            return `Time: ${context[0].label}`
          },
          label: (context) => {
            const label = context.dataset.label || ''
            let value = context.parsed.y
            
            // Format values based on metric type
            if (label.includes('Throughput')) {
              value = Math.round(value)
            } else if (label.includes('Response Time')) {
              value = Math.round(value * 100) / 100
            } else if (label.includes('Utilization') || label.includes('Error Rate')) {
              value = Math.round(value * 10) / 10
            } else if (label.includes('Business Value')) {
              value = Math.round(value * 100) / 100
            }
            
            return `${label}: ${value}${getMetricUnit(label)}`
          }
        }
      }
    },
    scales: {
      x: {
        display: true,
        grid: {
          display: true,
          color: isDark ? '#334155' : '#F3F4F6'
        },
        ticks: {
          color: isDark ? '#94A3B8' : '#6B7280',
          font: {
            size: 11
          },
          maxTicksLimit: 8
        }
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        grid: {
          display: true,
          color: isDark ? '#334155' : '#F3F4F6'
        },
        ticks: {
          color: isDark ? '#94A3B8' : '#6B7280',
          font: {
            size: 11
          }
        },
        title: {
          display: true,
          text: 'Throughput / Utilization',
          color: isDark ? '#E2E8F0' : '#374151'
        }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        grid: {
          drawOnChartArea: false
        },
        ticks: {
          color: isDark ? '#94A3B8' : '#6B7280',
          font: {
            size: 11
          }
        },
        title: {
          display: true,
          text: 'Response Time (ms)',
          color: isDark ? '#E2E8F0' : '#374151'
        }
      },
      y2: {
        type: 'linear',
        display: false,
        position: 'right'
      }
    },
    elements: {
      point: {
        hoverBorderWidth: 3
      }
    },
    animation: {
      duration: 750,
      easing: 'easeInOutQuart'
    }
  }

  return {
    type: 'line',
    data: chartData,
    options
  }
}

// Helper functions
const formatTimestamp = (timestamp: string, timeframe: string): string => {
  const date = new Date(timestamp)
  
  switch (timeframe) {
    case '1h':
      return date.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit' 
      })
    case '24h':
      return date.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit' 
      })
    case '7d':
      return date.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric' 
      })
    case '30d':
      return date.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric' 
      })
    default:
      return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      })
  }
}

const getMetricUnit = (label: string): string => {
  if (label.includes('Throughput')) return ' ops/sec'
  if (label.includes('Response Time')) return 'ms'
  if (label.includes('Utilization') || label.includes('Error Rate')) return '%'
  if (label.includes('Business Value')) return ' pts'
  return ''
}

// Chart management
const createChart = async () => {
  await nextTick()
  
  if (!chartCanvas.value) return

  destroyChart()

  try {
    chartInstance = new Chart(chartCanvas.value, createChartConfig())
  } catch (error) {
    console.error('Failed to create chart:', error)
  }
}

const destroyChart = () => {
  if (chartInstance) {
    chartInstance.destroy()
    chartInstance = null
  }
}

const updateChart = () => {
  if (!chartInstance) {
    createChart()
    return
  }

  const config = createChartConfig()
  chartInstance.data = config.data
  chartInstance.options = config.options || {}
  chartInstance.update('none')
}

// Lifecycle
onMounted(() => {
  if (props.data && props.data.length > 0) {
    createChart()
  }
})

onBeforeUnmount(() => {
  destroyChart()
})

// Watchers
watch(() => props.data, (newData) => {
  if (newData && newData.length > 0) {
    if (chartInstance) {
      updateChart()
    } else {
      createChart()
    }
  } else {
    destroyChart()
  }
}, { deep: true })

watch(() => props.timeframe, () => {
  updateChart()
})

// Dark mode watcher
const darkModeMediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
darkModeMediaQuery.addListener(() => {
  updateChart()
})
</script>

<style scoped>
canvas {
  max-width: 100%;
  height: auto;
}
</style>