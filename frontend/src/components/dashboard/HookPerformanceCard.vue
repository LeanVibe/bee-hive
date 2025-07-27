<template>
  <div class="hook-performance-card bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
    <div class="flex justify-between items-center mb-4">
      <div>
        <h3 class="text-lg font-semibold">Hook Performance</h3>
        <p class="text-sm text-gray-600 dark:text-gray-400">
          Real-time event processing metrics
        </p>
      </div>
      <div class="flex items-center gap-2">
        <div 
          :class="[
            'w-3 h-3 rounded-full',
            wsConnected ? 'bg-green-500' : 'bg-red-500'
          ]"
        ></div>
        <span class="text-sm">
          {{ wsConnected ? 'Connected' : 'Disconnected' }}
        </span>
      </div>
    </div>

    <!-- Performance Metrics Grid -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
      <div class="bg-gray-50 dark:bg-gray-700 p-3 rounded-md">
        <div class="text-xs text-gray-500 dark:text-gray-400">Events/sec</div>
        <div class="text-lg font-semibold">{{ eventsPerSecond }}</div>
      </div>
      
      <div class="bg-gray-50 dark:bg-gray-700 p-3 rounded-md">
        <div class="text-xs text-gray-500 dark:text-gray-400">Avg Latency</div>
        <div :class="[
          'text-lg font-semibold',
          avgLatency > 150 ? 'text-red-600' : avgLatency > 100 ? 'text-orange-600' : 'text-green-600'
        ]">
          {{ avgLatency }}ms
        </div>
      </div>
      
      <div class="bg-gray-50 dark:bg-gray-700 p-3 rounded-md">
        <div class="text-xs text-gray-500 dark:text-gray-400">Success Rate</div>
        <div :class="[
          'text-lg font-semibold',
          successRate > 95 ? 'text-green-600' : successRate > 90 ? 'text-orange-600' : 'text-red-600'
        ]">
          {{ successRate }}%
        </div>
      </div>
      
      <div class="bg-gray-50 dark:bg-gray-700 p-3 rounded-md">
        <div class="text-xs text-gray-500 dark:text-gray-400">Total Events</div>
        <div class="text-lg font-semibold">{{ totalEvents }}</div>
      </div>
    </div>

    <!-- Performance Status -->
    <div class="mb-4">
      <div class="flex items-center justify-between mb-2">
        <span class="text-sm font-medium">Performance Status</span>
        <span :class="[
          'px-2 py-1 text-xs font-medium rounded-full',
          healthStatus === 'healthy' ? 'bg-green-100 text-green-800' :
          healthStatus === 'degraded' ? 'bg-orange-100 text-orange-800' :
          'bg-red-100 text-red-800'
        ]">
          {{ healthStatus }}
        </span>
      </div>
      
      <!-- Performance warnings -->
      <div v-if="warnings.length > 0" class="space-y-1">
        <div 
          v-for="warning in warnings" 
          :key="warning"
          class="text-xs text-orange-600 dark:text-orange-400 flex items-center gap-1"
        >
          ⚠️ {{ warning }}
        </div>
      </div>
      
      <!-- Performance issues -->
      <div v-if="issues.length > 0" class="space-y-1">
        <div 
          v-for="issue in issues" 
          :key="issue"
          class="text-xs text-red-600 dark:text-red-400 flex items-center gap-1"
        >
          🚨 {{ issue }}
        </div>
      </div>
    </div>

    <!-- Latency Distribution Chart -->
    <div class="mb-4">
      <div class="text-sm font-medium mb-2">Latency Distribution</div>
      <div class="h-20 bg-gray-100 dark:bg-gray-700 rounded-md p-2">
        <canvas ref="latencyChart" class="w-full h-full"></canvas>
      </div>
    </div>

    <!-- Refresh Controls -->
    <div class="flex justify-between items-center">
      <div class="text-xs text-gray-500">
        Last updated: {{ lastUpdated }}
      </div>
      <div class="flex gap-2">
        <button 
          @click="refreshMetrics"
          :disabled="loading"
          class="px-3 py-1 text-xs bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400 text-white rounded-md transition-colors"
        >
          {{ loading ? 'Loading...' : 'Refresh' }}
        </button>
        <button 
          @click="toggleAutoRefresh"
          :class="[
            'px-3 py-1 text-xs rounded-md transition-colors',
            autoRefresh 
              ? 'bg-green-500 hover:bg-green-600 text-white' 
              : 'bg-gray-200 hover:bg-gray-300 text-gray-700'
          ]"
        >
          Auto {{ autoRefresh ? 'ON' : 'OFF' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip } from 'chart.js'
import { useEventsStore } from '../../stores/events'

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip)

const eventsStore = useEventsStore()

// Component state
const loading = ref(false)
const performanceData = ref<any>(null)
const autoRefresh = ref(true)
const refreshInterval = ref<number | null>(null)
const latencyChart = ref<HTMLCanvasElement>()
let chart: ChartJS | null = null

// Computed properties
const wsConnected = computed(() => eventsStore.wsConnected)

const eventsPerSecond = computed(() => {
  if (!performanceData.value?.performance) return 0
  return Math.round(performanceData.value.performance.events_per_second || 0)
})

const avgLatency = computed(() => {
  if (!performanceData.value?.performance) return 0
  return Math.round(performanceData.value.performance.avg_processing_time_ms || 0)
})

const successRate = computed(() => {
  if (!performanceData.value?.performance) return 100
  return Math.round(performanceData.value.performance.success_rate || 100)
})

const totalEvents = computed(() => {
  if (!performanceData.value?.performance) return 0
  return performanceData.value.performance.events_processed || 0
})

const healthStatus = computed(() => {
  return performanceData.value?.health || 'unknown'
})

const warnings = computed(() => {
  return performanceData.value?.degradation?.warnings || []
})

const issues = computed(() => {
  return performanceData.value?.degradation?.issues || []
})

const lastUpdated = computed(() => {
  if (!performanceData.value?.timestamp) return 'Never'
  return new Date(performanceData.value.timestamp).toLocaleTimeString()
})

// Methods
const refreshMetrics = async () => {
  if (loading.value) return
  
  loading.value = true
  try {
    performanceData.value = await eventsStore.getHookPerformanceMetrics()
    updateLatencyChart()
  } catch (error) {
    console.error('Failed to fetch performance metrics:', error)
  } finally {
    loading.value = false
  }
}

const toggleAutoRefresh = () => {
  autoRefresh.value = !autoRefresh.value
  
  if (autoRefresh.value) {
    startAutoRefresh()
  } else {
    stopAutoRefresh()
  }
}

const startAutoRefresh = () => {
  if (refreshInterval.value) return
  
  refreshInterval.value = window.setInterval(() => {
    refreshMetrics()
  }, 5000) // Refresh every 5 seconds
}

const stopAutoRefresh = () => {
  if (refreshInterval.value) {
    clearInterval(refreshInterval.value)
    refreshInterval.value = null
  }
}

const updateLatencyChart = () => {
  if (!chart || !performanceData.value?.performance?.processing_time_percentiles) return
  
  const percentiles = performanceData.value.performance.processing_time_percentiles
  
  chart.data = {
    labels: ['P50', 'P95', 'P99'],
    datasets: [{
      label: 'Latency (ms)',
      data: [
        percentiles.p50 || 0,
        percentiles.p95 || 0, 
        percentiles.p99 || 0
      ],
      backgroundColor: [
        'rgba(34, 197, 94, 0.7)',   // Green for P50
        'rgba(249, 115, 22, 0.7)',  // Orange for P95
        'rgba(239, 68, 68, 0.7)'    // Red for P99
      ],
      borderColor: [
        'rgb(34, 197, 94)',
        'rgb(249, 115, 22)',
        'rgb(239, 68, 68)'
      ],
      borderWidth: 1
    }]
  }
  
  chart.update('none')
}

const initLatencyChart = () => {
  if (!latencyChart.value) return
  
  chart = new ChartJS(latencyChart.value, {
    type: 'bar',
    data: {
      labels: ['P50', 'P95', 'P99'],
      datasets: [{
        label: 'Latency (ms)',
        data: [0, 0, 0],
        backgroundColor: [
          'rgba(34, 197, 94, 0.7)',
          'rgba(249, 115, 22, 0.7)',
          'rgba(239, 68, 68, 0.7)'
        ],
        borderColor: [
          'rgb(34, 197, 94)',
          'rgb(249, 115, 22)',
          'rgb(239, 68, 68)'
        ],
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          callbacks: {
            label: (context) => {
              return `${context.label}: ${context.parsed.y}ms`
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            callback: (value) => `${value}ms`
          }
        }
      },
      animation: {
        duration: 0
      }
    }
  })
}

// Lifecycle
onMounted(async () => {
  initLatencyChart()
  await refreshMetrics()
  
  if (autoRefresh.value) {
    startAutoRefresh()
  }
})

onUnmounted(() => {
  stopAutoRefresh()
  
  if (chart) {
    chart.destroy()
  }
})

// Watch for auto-refresh changes
watch(autoRefresh, (newValue) => {
  if (newValue) {
    startAutoRefresh()
  } else {
    stopAutoRefresh()
  }
})
</script>

<style scoped>
/* Additional styling if needed */
</style>