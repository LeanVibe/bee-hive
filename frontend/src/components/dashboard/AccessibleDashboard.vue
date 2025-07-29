<template>
  <div 
    class="accessible-dashboard"
    role="main"
    :aria-label="screenReaderTitle"
  >
    <!-- Skip navigation -->
    <a 
      href="#main-content" 
      class="skip-link"
      @click="focusMainContent"
    >
      Skip to main content
    </a>
    
    <!-- Screen reader announcements -->
    <div 
      aria-live="polite" 
      aria-atomic="true" 
      class="sr-only"
      ref="announceRegion"
    ></div>
    
    <!-- Dashboard header with proper heading hierarchy -->
    <header class="dashboard-header">
      <h1 
        id="dashboard-title"
        class="text-3xl font-bold text-slate-900 dark:text-white"
        tabindex="-1"
      >
        {{ title }}
      </h1>
      
      <p class="mt-2 text-slate-600 dark:text-slate-400">
        {{ subtitle }}
      </p>
      
      <!-- Status summary for screen readers -->
      <div class="sr-only">
        System status: {{ systemStatus }}. 
        {{ activeAgents }} agents active. 
        Last updated {{ lastUpdatedText }}.
      </div>
    </header>
    
    <!-- Main dashboard content -->
    <main id="main-content" tabindex="-1">
      <!-- Keyboard navigation instructions -->
      <div class="sr-only">
        Use Tab to navigate between sections. 
        Press Enter or Space to interact with controls.
        Use arrow keys within charts and data tables.
      </div>
      
      <!-- Live metrics with ARIA labels -->
      <section 
        aria-labelledby="metrics-heading"
        class="metrics-section"
      >
        <h2 
          id="metrics-heading" 
          class="text-xl font-semibold mb-4 text-slate-900 dark:text-white"
        >
          Key Metrics
        </h2>
        
        <div 
          class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
          role="group"
          aria-describedby="metrics-description"
        >
          <div id="metrics-description" class="sr-only">
            A grid of key performance metrics with current values and trends.
          </div>
          
          <AccessibleMetricCard
            v-for="(metric, index) in metrics"
            :key="metric.id"
            :title="metric.title"
            :value="metric.value"
            :unit="metric.unit"
            :previous="metric.previous"
            :trend="metric.trend"
            :icon="metric.icon"
            :color="metric.color"
            :aria-describedby="`metric-${index}-description`"
            @focus="handleMetricFocus(metric)"
            @click="handleMetricClick(metric)"
          >
            <div :id="`metric-${index}-description`" class="sr-only">
              {{ getMetricDescription(metric) }}
            </div>
          </AccessibleMetricCard>
        </div>
      </section>
      
      <!-- Charts with accessibility enhancements -->
      <section 
        aria-labelledby="charts-heading"
        class="charts-section mt-8"
      >
        <h2 
          id="charts-heading" 
          class="text-xl font-semibold mb-4 text-slate-900 dark:text-white"
        >
          Performance Charts
        </h2>
        
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <!-- Event timeline chart -->
          <div class="glass-card rounded-xl p-6">
            <div class="flex items-center justify-between mb-4">
              <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
                Event Activity
              </h3>
              
              <label for="event-time-range" class="sr-only">
                Select time range for event activity chart
              </label>
              <select
                id="event-time-range"
                v-model="eventTimeRange"
                class="input-field text-sm"
                @change="updateEventChart"
                :aria-describedby="eventChartDescription"
              >
                <option value="1h">Last Hour</option>
                <option value="6h">Last 6 Hours</option>
                <option value="24h">Last 24 Hours</option>
              </select>
            </div>
            
            <div :id="eventChartDescription" class="sr-only">
              Chart showing event activity over {{ eventTimeRange }}. 
              {{ eventChartSummary }}
            </div>
            
            <AccessibleChart
              :data="eventChartData"
              :type="'line'"
              :title="'Event Activity'"
              :time-range="eventTimeRange"
              @data-point-focus="handleChartDataFocus"
              @summary-requested="announceChartSummary"
            />
            
            <!-- Chart data table for screen readers -->
            <details class="mt-4">
              <summary class="cursor-pointer text-sm text-primary-600 dark:text-primary-400">
                View chart data as table
              </summary>
              <table class="mt-2 w-full text-sm" role="table">
                <caption class="sr-only">
                  Event activity data for {{ eventTimeRange }}
                </caption>
                <thead>
                  <tr>
                    <th scope="col" class="text-left p-2 border-b">Time</th>
                    <th scope="col" class="text-left p-2 border-b">Events</th>
                    <th scope="col" class="text-left p-2 border-b">Change</th>
                  </tr>
                </thead>
                <tbody>
                  <tr 
                    v-for="(point, index) in eventChartData" 
                    :key="index"
                    :class="{ 'bg-slate-50 dark:bg-slate-800': index % 2 === 0 }"
                  >
                    <td class="p-2 border-b">{{ formatTime(point.time) }}</td>
                    <td class="p-2 border-b">{{ point.value }}</td>
                    <td class="p-2 border-b">
                      <span :class="getTrendClass(point.trend)">
                        {{ formatTrend(point.trend) }}
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </details>
          </div>
          
          <!-- Performance metrics chart -->
          <div class="glass-card rounded-xl p-6">
            <div class="flex items-center justify-between mb-4">
              <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
                Performance Trends
              </h3>
              
              <label for="perf-time-range" class="sr-only">
                Select time range for performance trends chart
              </label>
              <select
                id="perf-time-range"
                v-model="perfTimeRange"
                class="input-field text-sm"
                @change="updatePerfChart"
                :aria-describedby="perfChartDescription"
              >
                <option value="1h">Last Hour</option>
                <option value="6h">Last 6 Hours</option>
                <option value="24h">Last 24 Hours</option>
              </select>
            </div>
            
            <div :id="perfChartDescription" class="sr-only">
              Chart showing performance trends over {{ perfTimeRange }}. 
              {{ perfChartSummary }}
            </div>
            
            <AccessibleChart
              :data="perfChartData"
              :type="'area'"
              :title="'Performance Trends'"
              :time-range="perfTimeRange"
              @data-point-focus="handleChartDataFocus"
              @summary-requested="announceChartSummary"
            />
          </div>
        </div>
      </section>
      
      <!-- Recent events with keyboard navigation -->
      <section 
        aria-labelledby="events-heading"
        class="events-section mt-8"
      >
        <h2 
          id="events-heading" 
          class="text-xl font-semibold mb-4 text-slate-900 dark:text-white"
        >
          Recent Events
        </h2>
        
        <div 
          class="glass-card rounded-xl p-6"
          role="log"
          aria-live="polite"
          aria-label="Recent system events"
        >
          <div v-if="events.length === 0" class="text-center py-8 text-slate-500">
            No recent events
          </div>
          
          <div v-else class="space-y-4">
            <div 
              v-for="(event, index) in events" 
              :key="event.id"
              class="event-item focus-ring rounded-lg p-3 transition-colors"
              :class="getEventClasses(event)"
              tabindex="0"
              role="article"
              :aria-label="getEventAriaLabel(event)"
              @click="handleEventClick(event)"
              @keydown="handleEventKeydown(event, $event)"
            >
              <div class="flex items-start space-x-3">
                <div 
                  class="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center"
                  :class="getEventIconClasses(event)"
                  :aria-hidden="true"
                >
                  <component :is="getEventIcon(event)" class="w-4 h-4" />
                </div>
                
                <div class="flex-1 min-w-0">
                  <div class="flex items-center justify-between">
                    <h4 class="font-medium text-slate-900 dark:text-white">
                      {{ event.title }}
                    </h4>
                    <time 
                      :datetime="event.timestamp"
                      class="text-sm text-slate-500 dark:text-slate-400"
                    >
                      {{ formatEventTime(event.timestamp) }}
                    </time>
                  </div>
                  
                  <p class="mt-1 text-sm text-slate-600 dark:text-slate-300">
                    {{ event.description }}
                  </p>
                  
                  <div v-if="event.metadata" class="mt-2">
                    <details class="text-xs text-slate-500">
                      <summary class="cursor-pointer hover:text-slate-700 dark:hover:text-slate-300">
                        View details
                      </summary>
                      <pre class="mt-1 p-2 bg-slate-100 dark:bg-slate-800 rounded text-xs overflow-auto">
                        {{ JSON.stringify(event.metadata, null, 2) }}
                      </pre>
                    </details>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, nextTick } from 'vue'
import { useAccessibility } from '@/composables/useAccessibility'
import { useMetricsStore } from '@/stores/metrics'
import { useEventsStore } from '@/stores/events'
import { formatDistanceToNow } from 'date-fns'
import {
  CheckCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  XCircleIcon,
} from '@heroicons/vue/24/outline'

// Import accessible components
import AccessibleMetricCard from './AccessibleMetricCard.vue'
import AccessibleChart from './AccessibleChart.vue'

// Props
interface Props {
  title?: string
  subtitle?: string
}

const props = withDefaults(defineProps<Props>(), {
  title: 'System Dashboard',
  subtitle: 'Real-time overview of agent activities and system health'
})

// Stores
const metricsStore = useMetricsStore()
const eventsStore = useEventsStore()

// Accessibility composable
const { 
  state: a11yState, 
  announce, 
  manageFocusOnRouteChange
} = useAccessibility({
  announcePageChanges: true,
  manageFocus: true,
  respectReducedMotion: true
})

// Template refs
const announceRegion = ref<HTMLElement | null>(null)

// Local state
const eventTimeRange = ref('1h')
const perfTimeRange = ref('1h')
const lastAnnouncement = ref('')

// Computed properties
const screenReaderTitle = computed(() => 
  `${props.title}: ${props.subtitle}`
)

const systemStatus = computed(() => {
  const totalEvents = events.value.length
  const errorEvents = events.value.filter(e => e.type === 'error').length
  
  if (errorEvents > 0) return 'Warning'
  if (totalEvents > 0) return 'Active'
  return 'Normal'
})

const activeAgents = computed(() => 
  metricsStore.getMetricValue('active_agents_total') || 0
)

const lastUpdatedText = computed(() => {
  const lastUpdate = metricsStore.lastUpdated
  return lastUpdate ? formatDistanceToNow(lastUpdate, { addSuffix: true }) : 'never'
})

const metrics = computed(() => [
  {
    id: 'active_sessions',
    title: 'Active Sessions',
    value: metricsStore.getMetricValue('active_sessions_total') || 0,
    previous: metricsStore.getMetricValue('active_sessions_total', { period: '24h' }) || 0,
    unit: '',
    icon: 'users',
    color: 'primary' as const,
    trend: 2.5
  },
  {
    id: 'events_per_hour',
    title: 'Events/Hour',
    value: (metricsStore.getMetricValue('event_processor_rate_per_second') || 0) * 3600,
    previous: null,
    unit: '/hr',
    icon: 'lightning-bolt',
    color: 'success' as const,
    trend: 0
  },
  {
    id: 'success_rate',
    title: 'Tool Success Rate',
    value: (metricsStore.getMetricValue('tool_success_rate') || 0) * 100,
    previous: (metricsStore.getMetricValue('tool_success_rate', { period: '24h' }) || 0) * 100,
    unit: '%',
    icon: 'check-circle',
    color: 'success' as const,
    trend: 1.2
  },
  {
    id: 'response_time',
    title: 'Avg Response Time',
    value: (metricsStore.getMetricValue('event_processing_duration_seconds') || 0) * 1000,
    previous: null,
    unit: 'ms',
    icon: 'clock',
    color: 'warning' as const,
    trend: -0.8
  }
])

const events = computed(() => eventsStore.recentEvents || [])

const eventChartData = computed(() => {
  // Mock chart data - replace with real data
  return Array.from({ length: 12 }, (_, i) => ({
    time: new Date(Date.now() - (11 - i) * 300000), // 5 min intervals
    value: Math.floor(Math.random() * 100) + 20,
    trend: Math.random() > 0.5 ? 1 : -1
  }))
})

const perfChartData = computed(() => {
  // Mock performance data - replace with real data
  return Array.from({ length: 12 }, (_, i) => ({
    time: new Date(Date.now() - (11 - i) * 300000),
    value: Math.floor(Math.random() * 50) + 100,
    trend: Math.random() > 0.5 ? 1 : -1
  }))
})

const eventChartSummary = computed(() => {
  const data = eventChartData.value
  if (data.length === 0) return 'No data available'
  
  const latest = data[data.length - 1]?.value || 0
  const average = data.reduce((sum, d) => sum + d.value, 0) / data.length
  const trend = latest > average ? 'increasing' : latest < average ? 'decreasing' : 'stable'
  
  return `Current: ${latest} events, Average: ${Math.round(average)}, Trend: ${trend}`
})

const perfChartSummary = computed(() => {
  const data = perfChartData.value
  if (data.length === 0) return 'No data available'
  
  const latest = data[data.length - 1]?.value || 0
  const average = data.reduce((sum, d) => sum + d.value, 0) / data.length
  const trend = latest > average ? 'slower' : latest < average ? 'faster' : 'stable'
  
  return `Current: ${latest}ms, Average: ${Math.round(average)}ms, Performance: ${trend}`
})

// Methods
const focusMainContent = (event: Event) => {
  event.preventDefault()
  const mainContent = document.getElementById('main-content')
  if (mainContent) {
    mainContent.focus()
    announce('Focused on main content')
  }
}

const getMetricDescription = (metric: any) => {
  const trendText = metric.trend > 0 
    ? `up ${Math.abs(metric.trend).toFixed(1)}% from previous period`
    : metric.trend < 0
    ? `down ${Math.abs(metric.trend).toFixed(1)}% from previous period`
    : 'unchanged from previous period'
  
  return `${metric.title}: ${metric.value}${metric.unit}. ${trendText}.`
}

const handleMetricFocus = (metric: any) => {
  const description = getMetricDescription(metric)
  if (description !== lastAnnouncement.value) {
    announce(description)
    lastAnnouncement.value = description
  }
}

const handleMetricClick = (metric: any) => {
  announce(`Viewing details for ${metric.title}`)
  // Emit event or navigate to detailed view
}

const updateEventChart = () => {
  announce(`Event chart updated to show ${eventTimeRange.value}`)
}

const updatePerfChart = () => {
  announce(`Performance chart updated to show ${perfTimeRange.value}`)
}

const handleChartDataFocus = (data: any) => {
  const timeText = formatTime(data.time)
  const valueText = `${data.value} at ${timeText}`
  announce(valueText)
}

const announceChartSummary = (summary: string) => {
  announce(summary)
}

const getEventClasses = (event: any) => {
  const baseClasses = 'hover:bg-slate-50 dark:hover:bg-slate-700/50'
  
  switch (event.type) {
    case 'error':
      return `${baseClasses} border-l-4 border-red-500`
    case 'warning':
      return `${baseClasses} border-l-4 border-yellow-500`
    case 'success':
      return `${baseClasses} border-l-4 border-green-500`
    default:
      return `${baseClasses} border-l-4 border-blue-500`
  }
}

const getEventIconClasses = (event: any) => {
  switch (event.type) {
    case 'error':
      return 'bg-red-100 text-red-600 dark:bg-red-900/20 dark:text-red-400'
    case 'warning':
      return 'bg-yellow-100 text-yellow-600 dark:bg-yellow-900/20 dark:text-yellow-400'
    case 'success':
      return 'bg-green-100 text-green-600 dark:bg-green-900/20 dark:text-green-400'
    default:
      return 'bg-blue-100 text-blue-600 dark:bg-blue-900/20 dark:text-blue-400'
  }
}

const getEventIcon = (event: any) => {
  switch (event.type) {
    case 'error': return XCircleIcon
    case 'warning': return ExclamationTriangleIcon
    case 'success': return CheckCircleIcon
    default: return InformationCircleIcon
  }
}

const getEventAriaLabel = (event: any) => {
  const timeText = formatEventTime(event.timestamp)
  return `${event.type} event: ${event.title}. ${event.description}. Occurred ${timeText}.`
}

const handleEventClick = (event: any) => {
  announce(`Selected event: ${event.title}`)
  // Handle event selection
}

const handleEventKeydown = (event: any, keyEvent: KeyboardEvent) => {
  if (keyEvent.key === 'Enter' || keyEvent.key === ' ') {
    keyEvent.preventDefault()
    handleEventClick(event)
  }
}

const formatTime = (time: Date) => {
  return time.toLocaleTimeString()
}

const formatEventTime = (timestamp: string | Date) => {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp
  return formatDistanceToNow(date, { addSuffix: true })
}

const getTrendClass = (trend: number) => {
  if (trend > 0) return 'text-green-600 dark:text-green-400'
  if (trend < 0) return 'text-red-600 dark:text-red-400'
  return 'text-slate-500 dark:text-slate-400'
}

const formatTrend = (trend: number) => {
  if (trend > 0) return `+${trend.toFixed(1)}%`
  if (trend < 0) return `${trend.toFixed(1)}%`
  return '0%'
}

// Lifecycle
onMounted(async () => {
  await nextTick()
  
  // Announce page load
  announce(`Dashboard loaded. ${systemStatus.value} system status.`)
  
  // Set up route change announcements
  manageFocusOnRouteChange(props.title)
  
  // Load initial data
  await Promise.all([
    metricsStore.refreshAll(),
    eventsStore.refreshEvents()
  ])
})
</script>

<style scoped>
.skip-link {
  position: absolute;
  left: -10000px;
  top: auto;
  width: 1px;
  height: 1px;
  overflow: hidden;
}

.skip-link:focus {
  position: absolute;
  left: 6px;
  top: 7px;
  width: auto;
  height: auto;
  overflow: visible;
  z-index: 999999;
  padding: 8px 16px;
  background: #000;
  color: #fff;
  text-decoration: none;
  border-radius: 3px;
}

.sr-only {
  position: absolute;
  left: -10000px;
  width: 1px;
  height: 1px;
  overflow: hidden;
}

.focus-ring {
  @apply focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2;
}

/* High contrast mode support */
@media (prefers-contrast: high) {
  .glass-card {
    border: 2px solid currentColor;
  }
  
  .event-item {
    border: 1px solid currentColor;
  }
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
</style>