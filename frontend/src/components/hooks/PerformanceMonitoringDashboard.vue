<template>
  <div class="performance-monitoring-dashboard space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h3 class="text-xl font-semibold text-gray-900 dark:text-white">
          Performance Monitoring
        </h3>
        <p class="text-sm text-gray-500 dark:text-gray-400">
          Hook lifecycle system performance metrics and analysis
        </p>
      </div>
      
      <div class="flex items-center space-x-3">
        <!-- Time range selector -->
        <select
          v-model="selectedTimeRange"
          @change="refreshData"
          class="text-sm border border-gray-300 dark:border-gray-600 rounded-md px-3 py-1.5 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
        >
          <option value="1h">Last Hour</option>
          <option value="6h">Last 6 Hours</option>
          <option value="24h">Last 24 Hours</option>
          <option value="7d">Last 7 Days</option>
        </select>

        <!-- Auto refresh toggle -->
        <button
          @click="autoRefresh = !autoRefresh"
          :class="[
            'px-3 py-1.5 text-xs rounded-full transition-colors',
            autoRefresh
              ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
              : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300'
          ]"
        >
          Auto-refresh {{ autoRefresh ? 'ON' : 'OFF' }}
        </button>

        <!-- Refresh button -->
        <button
          @click="refreshData"
          :disabled="loading"
          class="px-3 py-1.5 text-xs bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 rounded-full hover:bg-blue-200 dark:hover:bg-blue-800 transition-colors disabled:opacity-50"
        >
          {{ loading ? 'Refreshing...' : 'Refresh' }}
        </button>
      </div>
    </div>

    <!-- Performance overview cards -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <!-- Processing time -->
      <div class="performance-card">
        <div class="performance-card-header">
          <div class="performance-card-icon bg-blue-100 text-blue-600 dark:bg-blue-900 dark:text-blue-400">
            <ClockIcon class="w-5 h-5" />
          </div>
          <h4 class="performance-card-title">Avg Processing Time</h4>
        </div>
        <div class="performance-card-value text-blue-600 dark:text-blue-400">
          {{ formatProcessingTime(metrics.avg_processing_time_ms) }}
        </div>
        <div class="performance-card-change" :class="getChangeClass(processingTimeChange)">
          {{ formatChange(processingTimeChange) }}
        </div>
      </div>

      <!-- Throughput -->
      <div class="performance-card">
        <div class="performance-card-header">
          <div class="performance-card-icon bg-green-100 text-green-600 dark:bg-green-900 dark:text-green-400">
            <ArrowTrendingUpIcon class="w-5 h-5" />
          </div>
          <h4 class="performance-card-title">Throughput</h4>
        </div>
        <div class="performance-card-value text-green-600 dark:text-green-400">
          {{ formatThroughput(metrics.hooks_processed) }}
        </div>
        <div class="performance-card-change" :class="getChangeClass(throughputChange)">
          {{ formatChange(throughputChange) }}
        </div>
      </div>

      <!-- Error rate -->
      <div class="performance-card">
        <div class="performance-card-header">
          <div class="performance-card-icon bg-red-100 text-red-600 dark:bg-red-900 dark:text-red-400">
            <ExclamationTriangleIcon class="w-5 h-5" />
          </div>
          <h4 class="performance-card-title">Error Rate</h4>
        </div>
        <div class="performance-card-value text-red-600 dark:text-red-400">
          {{ formatErrorRate() }}
        </div>
        <div class="performance-card-change" :class="getChangeClass(errorRateChange)">
          {{ formatChange(errorRateChange) }}
        </div>
      </div>

      <!-- Security blocks -->
      <div class="performance-card">
        <div class="performance-card-header">
          <div class="performance-card-icon bg-orange-100 text-orange-600 dark:bg-orange-900 dark:text-orange-400">
            <ShieldExclamationIcon class="w-5 h-5" />
          </div>
          <h4 class="performance-card-title">Security Blocks</h4>
        </div>
        <div class="performance-card-value text-orange-600 dark:text-orange-400">
          {{ metrics.hooks_blocked }}
        </div>
        <div class="performance-card-change" :class="getChangeClass(securityBlocksChange)">
          {{ formatChange(securityBlocksChange) }}
        </div>
      </div>
    </div>

    <!-- Performance charts -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Processing time chart -->
      <div class="performance-chart-container">
        <div class="performance-chart-header">
          <h4 class="performance-chart-title">Processing Time Trend</h4>
          <div class="flex items-center space-x-2">
            <div class="w-3 h-3 bg-blue-500 rounded-full"></div>
            <span class="text-xs text-gray-500 dark:text-gray-400">Processing Time (ms)</span>
          </div>
        </div>
        <div class="performance-chart-content">
          <div class="chart-placeholder">
            <div class="chart-line" :style="{ height: '60%' }"></div>
            <div class="chart-line" :style="{ height: '45%' }"></div>
            <div class="chart-line" :style="{ height: '75%' }"></div>
            <div class="chart-line" :style="{ height: '30%' }"></div>
            <div class="chart-line" :style="{ height: '55%' }"></div>
            <div class="chart-line" :style="{ height: '40%' }"></div>
            <div class="chart-line" :style="{ height: '65%' }"></div>
            <div class="chart-line" :style="{ height: '50%' }"></div>
          </div>
        </div>
      </div>

      <!-- Throughput chart -->
      <div class="performance-chart-container">
        <div class="performance-chart-header">
          <h4 class="performance-chart-title">Throughput Trend</h4>
          <div class="flex items-center space-x-2">
            <div class="w-3 h-3 bg-green-500 rounded-full"></div>
            <span class="text-xs text-gray-500 dark:text-gray-400">Events per minute</span>
          </div>
        </div>
        <div class="performance-chart-content">
          <div class="chart-placeholder">
            <div class="chart-line green" :style="{ height: '80%' }"></div>
            <div class="chart-line green" :style="{ height: '65%' }"></div>
            <div class="chart-line green" :style="{ height: '90%' }"></div>
            <div class="chart-line green" :style="{ height: '55%' }"></div>
            <div class="chart-line green" :style="{ height: '75%' }"></div>
            <div class="chart-line green" :style="{ height: '60%' }"></div>
            <div class="chart-line green" :style="{ height: '85%' }"></div>
            <div class="chart-line green" :style="{ height: '70%' }"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Detailed metrics -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- System performance -->
      <div class="performance-section">
        <h4 class="performance-section-title">System Performance</h4>
        <div class="space-y-4">
          <div class="metric-row">
            <span class="metric-label">Total Hooks Processed</span>
            <span class="metric-value">{{ metrics.hooks_processed.toLocaleString() }}</span>
          </div>
          
          <div class="metric-row">
            <span class="metric-label">Performance Threshold Violations</span>
            <span class="metric-value text-orange-600 dark:text-orange-400">
              {{ metrics.performance_threshold_violations }}
            </span>
          </div>
          
          <div class="metric-row">
            <span class="metric-label">Processing Errors</span>
            <span class="metric-value text-red-600 dark:text-red-400">
              {{ metrics.processing_errors }}
            </span>
          </div>
          
          <div class="metric-row">
            <span class="metric-label">System Status</span>
            <span 
              :class="[
                'inline-flex px-2 py-1 text-xs font-semibold rounded-full',
                getSystemStatusClass()
              ]"
            >
              {{ getSystemStatus() }}
            </span>
          </div>
        </div>
      </div>

      <!-- Component performance -->
      <div class="performance-section">
        <h4 class="performance-section-title">Component Performance</h4>
        <div class="space-y-4">
          <!-- Security validator -->
          <div class="component-metric">
            <div class="component-header">
              <span class="component-name">Security Validator</span>
              <span class="component-status healthy">Healthy</span>
            </div>
            <div class="component-stats">
              <div class="component-stat">
                <span class="stat-label">Validations</span>
                <span class="stat-value">{{ metrics.security_validator_metrics?.validations_performed || 0 }}</span>
              </div>
              <div class="component-stat">
                <span class="stat-label">Avg Time</span>
                <span class="stat-value">{{ formatProcessingTime(metrics.security_validator_metrics?.avg_validation_time_ms || 0) }}</span>
              </div>
              <div class="component-stat">
                <span class="stat-label">Cache Hits</span>
                <span class="stat-value">{{ formatPercentage(getCacheHitRate()) }}</span>
              </div>
            </div>
          </div>

          <!-- Event aggregator -->
          <div class="component-metric">
            <div class="component-header">
              <span class="component-name">Event Aggregator</span>
              <span class="component-status healthy">Healthy</span>
            </div>
            <div class="component-stats">
              <div class="component-stat">
                <span class="stat-label">Events</span>
                <span class="stat-value">{{ metrics.event_aggregator_metrics?.events_aggregated || 0 }}</span>
              </div>
              <div class="component-stat">
                <span class="stat-label">Batches</span>
                <span class="stat-value">{{ metrics.event_aggregator_metrics?.batches_processed || 0 }}</span>
              </div>
              <div class="component-stat">
                <span class="stat-label">Avg Batch Size</span>
                <span class="stat-value">{{ Math.round(metrics.event_aggregator_metrics?.avg_batch_size || 0) }}</span>
              </div>
            </div>
          </div>

          <!-- WebSocket streamer -->
          <div class="component-metric">
            <div class="component-header">
              <span class="component-name">WebSocket Streamer</span>
              <span class="component-status healthy">Healthy</span>
            </div>
            <div class="component-stats">
              <div class="component-stat">
                <span class="stat-label">Connections</span>
                <span class="stat-value">{{ metrics.websocket_streamer_metrics?.active_connections || 0 }}</span>
              </div>
              <div class="component-stat">
                <span class="stat-label">Messages</span>
                <span class="stat-value">{{ metrics.websocket_streamer_metrics?.messages_sent || 0 }}</span>
              </div>
              <div class="component-stat">
                <span class="stat-label">Errors</span>
                <span class="stat-value text-red-600 dark:text-red-400">{{ metrics.websocket_streamer_metrics?.connection_errors || 0 }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useEventsStore } from '@/stores/events'
import { 
  type HookPerformanceMetrics,
  type PerformanceMonitoringDashboardProps
} from '@/types/hooks'
import {
  ClockIcon,
  ArrowTrendingUpIcon,
  ExclamationTriangleIcon,
  ShieldExclamationIcon
} from '@heroicons/vue/24/outline'

// Props
interface Props extends PerformanceMonitoringDashboardProps {}

const props = withDefaults(defineProps<Props>(), {
  refreshInterval: 5000,
  showHistoricalData: true,
  timeRange: '1h'
})

// Store
const eventsStore = useEventsStore()

// Local state
const loading = ref(false)
const autoRefresh = ref(true)
const selectedTimeRange = ref(props.timeRange)
const refreshTimer = ref<NodeJS.Timeout | null>(null)

// Mock metrics data (in real implementation, this would come from API)
const metrics = ref<HookPerformanceMetrics>({
  total_hooks_processed: 12547,
  hooks_blocked: 23,
  processing_errors: 5,
  avg_processing_time_ms: 12.5,
  performance_threshold_violations: 2,
  security_validator_metrics: {
    validations_performed: 8934,
    commands_blocked: 23,
    approvals_required: 12,
    cache_hits: 7821,
    avg_validation_time_ms: 3.2
  },
  event_aggregator_metrics: {
    events_aggregated: 12547,
    batches_processed: 125,
    aggregation_rules_applied: 342,
    avg_batch_size: 100,
    flush_operations: 125
  },
  websocket_streamer_metrics: {
    active_connections: 3,
    messages_sent: 12547,
    connection_errors: 1,
    filtered_messages: 234,
    avg_broadcast_time_ms: 1.8
  }
})

// Previous metrics for change calculation
const previousMetrics = ref<HookPerformanceMetrics | null>(null)

// Computed
const processingTimeChange = computed(() => {
  if (!previousMetrics.value) return 0
  const current = metrics.value.avg_processing_time_ms
  const previous = previousMetrics.value.avg_processing_time_ms
  return ((current - previous) / previous) * 100
})

const throughputChange = computed(() => {
  if (!previousMetrics.value) return 0
  const current = metrics.value.total_hooks_processed
  const previous = previousMetrics.value.total_hooks_processed
  return ((current - previous) / previous) * 100
})

const errorRateChange = computed(() => {
  if (!previousMetrics.value) return 0
  const currentRate = formatErrorRateValue()
  const previousRate = (previousMetrics.value.processing_errors / previousMetrics.value.total_hooks_processed) * 100
  return currentRate - previousRate
})

const securityBlocksChange = computed(() => {
  if (!previousMetrics.value) return 0
  const current = metrics.value.hooks_blocked
  const previous = previousMetrics.value.hooks_blocked
  return ((current - previous) / previous) * 100
})

// Methods
const formatProcessingTime = (timeMs: number) => {
  if (timeMs < 1) return `${(timeMs * 1000).toFixed(0)}μs`
  if (timeMs < 1000) return `${timeMs.toFixed(1)}ms`
  return `${(timeMs / 1000).toFixed(2)}s`
}

const formatThroughput = (totalHooks: number) => {
  // Calculate events per minute based on time range
  const timeRangeHours = {
    '1h': 1,
    '6h': 6,
    '24h': 24,
    '7d': 168
  }
  
  const hours = timeRangeHours[selectedTimeRange.value as keyof typeof timeRangeHours] || 1
  const eventsPerHour = totalHooks / hours
  const eventsPerMinute = eventsPerHour / 60
  
  if (eventsPerMinute < 1) return `${(eventsPerMinute * 60).toFixed(1)}/min`
  if (eventsPerMinute < 60) return `${eventsPerMinute.toFixed(1)}/min`
  return `${(eventsPerMinute / 60).toFixed(1)}/hr`
}

const formatErrorRateValue = () => {
  const total = metrics.value.total_hooks_processed
  const errors = metrics.value.processing_errors
  return total > 0 ? (errors / total) * 100 : 0
}

const formatErrorRate = () => {
  const rate = formatErrorRateValue()
  return `${rate.toFixed(2)}%`
}

const formatChange = (change: number) => {
  const sign = change >= 0 ? '+' : ''
  return `${sign}${change.toFixed(1)}%`
}

const getChangeClass = (change: number) => {
  if (change > 0) return 'text-red-600 dark:text-red-400'
  if (change < 0) return 'text-green-600 dark:text-green-400'
  return 'text-gray-500 dark:text-gray-400'
}

const formatPercentage = (value: number) => {
  return `${value.toFixed(1)}%`
}

const getCacheHitRate = () => {
  const validator = metrics.value.security_validator_metrics
  if (!validator || validator.validations_performed === 0) return 0
  return (validator.cache_hits / validator.validations_performed) * 100
}

const getSystemStatus = () => {
  const errorRate = formatErrorRateValue()
  const avgProcessingTime = metrics.value.avg_processing_time_ms
  
  if (errorRate > 5 || avgProcessingTime > 100) return 'Critical'
  if (errorRate > 2 || avgProcessingTime > 50) return 'Degraded'
  return 'Healthy'
}

const getSystemStatusClass = () => {
  const status = getSystemStatus()
  const classMap = {
    'Healthy': 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    'Degraded': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    'Critical': 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
  }
  return classMap[status as keyof typeof classMap] || 'bg-gray-100 text-gray-800'
}

const refreshData = async () => {
  loading.value = true
  
  try {
    // Store previous metrics for change calculation
    previousMetrics.value = { ...metrics.value }
    
    // In real implementation, fetch from API
    // const newMetrics = await eventsStore.getHookPerformanceMetrics()
    // metrics.value = newMetrics
    
    // Mock data update
    await new Promise(resolve => setTimeout(resolve, 500))
    
    // Simulate small changes in metrics
    metrics.value = {
      ...metrics.value,
      total_hooks_processed: metrics.value.total_hooks_processed + Math.floor(Math.random() * 50),
      avg_processing_time_ms: metrics.value.avg_processing_time_ms + (Math.random() - 0.5) * 2,
      hooks_blocked: metrics.value.hooks_blocked + (Math.random() > 0.8 ? 1 : 0),
      processing_errors: metrics.value.processing_errors + (Math.random() > 0.9 ? 1 : 0)
    }
    
  } catch (error) {
    console.error('Failed to refresh performance data:', error)
  } finally {
    loading.value = false
  }
}

const startAutoRefresh = () => {
  if (refreshTimer.value) {
    clearInterval(refreshTimer.value)
  }
  
  if (autoRefresh.value) {
    refreshTimer.value = setInterval(refreshData, props.refreshInterval)
  }
}

// Lifecycle
onMounted(() => {
  refreshData()
  startAutoRefresh()
  
  // Subscribe to performance metrics updates
  eventsStore.onPerformanceMetric((newMetrics) => {
    previousMetrics.value = { ...metrics.value }
    metrics.value = newMetrics
  })
})

onUnmounted(() => {
  if (refreshTimer.value) {
    clearInterval(refreshTimer.value)
  }
})

// Watch auto-refresh changes
watch(autoRefresh, () => {
  startAutoRefresh()
})
</script>

<style scoped>
.performance-card {
  @apply bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6;
}

.performance-card-header {
  @apply flex items-center space-x-3 mb-4;
}

.performance-card-icon {
  @apply w-10 h-10 rounded-lg flex items-center justify-center;
}

.performance-card-title {
  @apply text-sm font-medium text-gray-900 dark:text-white;
}

.performance-card-value {
  @apply text-2xl font-bold mb-2;
}

.performance-card-change {
  @apply text-sm font-medium;
}

.performance-chart-container {
  @apply bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6;
}

.performance-chart-header {
  @apply flex items-center justify-between mb-4;
}

.performance-chart-title {
  @apply text-lg font-semibold text-gray-900 dark:text-white;
}

.performance-chart-content {
  @apply h-64;
}

.chart-placeholder {
  @apply h-full flex items-end justify-between space-x-2;
}

.chart-line {
  @apply bg-blue-500 flex-1 rounded-t transition-all duration-300;
}

.chart-line.green {
  @apply bg-green-500;
}

.performance-section {
  @apply bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6;
}

.performance-section-title {
  @apply text-lg font-semibold text-gray-900 dark:text-white mb-4;
}

.metric-row {
  @apply flex justify-between items-center;
}

.metric-label {
  @apply text-sm text-gray-600 dark:text-gray-400;
}

.metric-value {
  @apply text-sm font-medium text-gray-900 dark:text-white;
}

.component-metric {
  @apply bg-gray-50 dark:bg-gray-900 rounded-lg p-4;
}

.component-header {
  @apply flex justify-between items-center mb-3;
}

.component-name {
  @apply text-sm font-semibold text-gray-900 dark:text-white;
}

.component-status {
  @apply px-2 py-1 text-xs font-medium rounded-full;
}

.component-status.healthy {
  @apply bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200;
}

.component-stats {
  @apply grid grid-cols-3 gap-4;
}

.component-stat {
  @apply text-center;
}

.stat-label {
  @apply block text-xs text-gray-500 dark:text-gray-400 mb-1;
}

.stat-value {
  @apply text-sm font-semibold text-gray-900 dark:text-white;
}

/* Responsive design */
@media (max-width: 768px) {
  .component-stats {
    @apply grid-cols-1 gap-2;
  }
  
  .performance-card-header {
    @apply flex-col items-start space-x-0 space-y-2;
  }
}
</style>