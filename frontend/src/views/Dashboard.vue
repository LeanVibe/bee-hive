<template>
  <div class="space-y-8">
    <!-- Header -->
    <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between">
      <div>
        <h1 class="text-3xl font-bold text-slate-900 dark:text-white">
          System Dashboard
        </h1>
        <p class="mt-2 text-slate-600 dark:text-slate-400">
          Real-time overview of agent activities and system health
        </p>
      </div>
      
      <div class="mt-4 sm:mt-0 flex items-center space-x-3">
        <button
          @click="refreshAll"
          :disabled="loading"
          class="btn-secondary"
          :class="{ 'opacity-50 cursor-not-allowed': loading }"
        >
          <ArrowPathIcon 
            class="w-4 h-4 mr-2" 
            :class="{ 'animate-spin': loading }"
          />
          Refresh
        </button>
        
        <div class="text-sm text-slate-500 dark:text-slate-400">
          Last updated: {{ formatTime(lastUpdated) }}
        </div>
      </div>
    </div>
    
    <!-- System Health Overview -->
    <SystemHealthCard />
    
    <!-- Key Metrics Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <MetricCard
        title="Active Sessions"
        :value="getMetricValue('active_sessions_total')"
        :previous="getMetricValue('active_sessions_total', { period: '24h' })"
        icon="users"
        color="primary"
      />
      
      <MetricCard
        title="Events/Hour"
        :value="getMetricValue('event_processor_rate_per_second') * 3600"
        unit="/hr"
        icon="lightning-bolt"
        color="success"
      />
      
      <MetricCard
        title="Tool Success Rate"
        :value="getMetricValue('tool_success_rate') * 100"
        unit="%"
        :previous="getMetricValue('tool_success_rate', { period: '24h' }) * 100"
        icon="check-circle"
        color="success"
      />
      
      <MetricCard
        title="Avg Response Time"
        :value="getMetricValue('event_processing_duration_seconds') * 1000"
        unit="ms"
        icon="clock"
        color="warning"
      />
    </div>
    
    <!-- Charts Row -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Event Timeline Chart -->
      <div class="glass-card rounded-xl p-6">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
            Event Activity
          </h3>
          <div class="flex items-center space-x-2">
            <select
              v-model="eventTimeRange"
              class="input-field text-sm"
              @change="updateEventChart"
            >
              <option value="1h">Last Hour</option>
              <option value="6h">Last 6 Hours</option>
              <option value="24h">Last 24 Hours</option>
            </select>
          </div>
        </div>
        <EventTimelineChart :timeRange="eventTimeRange" />
      </div>
      
      <!-- Performance Metrics Chart -->
      <div class="glass-card rounded-xl p-6">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
            Performance Trends
          </h3>
          <div class="flex items-center space-x-2">
            <select
              v-model="perfTimeRange"
              class="input-field text-sm"
              @change="updatePerfChart"
            >
              <option value="1h">Last Hour</option>
              <option value="6h">Last 6 Hours</option>
              <option value="24h">Last 24 Hours</option>
            </select>
          </div>
        </div>
        <PerformanceChart :timeRange="perfTimeRange" />
      </div>
    </div>
    
    <!-- Recent Activity and System Status -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Recent Events -->
      <div class="lg:col-span-2">
        <RecentEventsCard />
      </div>
      
      <!-- System Components -->
      <div>
        <SystemComponentsCard />
      </div>
    </div>
    
    <!-- Agent Status Grid -->
    <div class="glass-card rounded-xl p-6">
      <div class="flex items-center justify-between mb-6">
        <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
          Active Agents
        </h3>
        <router-link
          to="/agents"
          class="text-primary-600 dark:text-primary-400 hover:underline text-sm font-medium"
        >
          View All →
        </router-link>
      </div>
      <AgentStatusGrid />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useMetricsStore } from '@/stores/metrics'
import { useEventStore } from '@/stores/events'
import { ArrowPathIcon } from '@heroicons/vue/24/outline'
import { formatDistanceToNow } from 'date-fns'

// Components
import SystemHealthCard from '@/components/dashboard/SystemHealthCard.vue'
import MetricCard from '@/components/dashboard/MetricCard.vue'
import EventTimelineChart from '@/components/charts/EventTimelineChart.vue'
import PerformanceChart from '@/components/charts/PerformanceChart.vue'
import RecentEventsCard from '@/components/dashboard/RecentEventsCard.vue'
import SystemComponentsCard from '@/components/dashboard/SystemComponentsCard.vue'
import AgentStatusGrid from '@/components/dashboard/AgentStatusGrid.vue'

// Stores
const metricsStore = useMetricsStore()
const eventStore = useEventStore()

// Local state
const loading = ref(false)
const lastUpdated = ref<Date | null>(null)
const eventTimeRange = ref('1h')
const perfTimeRange = ref('1h')

// Auto-refresh interval
let refreshInterval: number | null = null

// Computed helpers
const getMetricValue = (name: string, labels?: Record<string, string>) => {
  return metricsStore.getMetricValue(name, labels)
}

const formatTime = (date: Date | null) => {
  if (!date) return 'Never'
  return formatDistanceToNow(date, { addSuffix: true })
}

// Actions
const refreshAll = async () => {
  loading.value = true
  
  try {
    await Promise.all([
      metricsStore.refreshAll(),
      eventStore.refreshEvents(),
    ])
    
    lastUpdated.value = new Date()
  } catch (error) {
    console.error('Failed to refresh dashboard:', error)
  } finally {
    loading.value = false
  }
}

const updateEventChart = () => {
  // Chart component will react to timeRange change
}

const updatePerfChart = () => {
  // Chart component will react to timeRange change
}

const startAutoRefresh = () => {
  // Refresh every 30 seconds
  refreshInterval = setInterval(() => {
    if (!loading.value) {
      refreshAll()
    }
  }, 30000)
}

const stopAutoRefresh = () => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
    refreshInterval = null
  }
}

// Lifecycle
onMounted(async () => {
  await refreshAll()
  startAutoRefresh()
})

onUnmounted(() => {
  stopAutoRefresh()
})
</script>