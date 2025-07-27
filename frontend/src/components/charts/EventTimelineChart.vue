<template>
  <div class="event-timeline-chart">
    <div class="chart-header mb-4 flex justify-between items-center">
      <div>
        <h3 class="text-lg font-semibold">Hook Event Timeline</h3>
        <p class="text-sm text-gray-600 dark:text-gray-400">
          Real-time agent events ({{ totalEvents }} events, {{ eventsPerMinute }}/min)
        </p>
      </div>
      <div class="flex gap-2">
        <button 
          @click="togglePause" 
          :class="isPaused ? 'bg-green-500 hover:bg-green-600' : 'bg-orange-500 hover:bg-orange-600'"
          class="px-3 py-1 text-white text-xs rounded-md transition-colors"
        >
          {{ isPaused ? 'Resume' : 'Pause' }}
        </button>
        <button 
          @click="clearEvents"
          class="px-3 py-1 bg-red-500 hover:bg-red-600 text-white text-xs rounded-md transition-colors"
        >
          Clear
        </button>
      </div>
    </div>
    
    <!-- Event Type Filters -->
    <div class="flex flex-wrap gap-2 mb-4">
      <button
        v-for="eventType in eventTypes"
        :key="eventType"
        @click="toggleEventType(eventType)"
        :class="[
          'px-3 py-1 text-xs rounded-full transition-colors',
          visibleEventTypes.has(eventType) 
            ? 'bg-blue-500 text-white' 
            : 'bg-gray-200 text-gray-600 dark:bg-gray-700 dark:text-gray-300'
        ]"
      >
        {{ eventType }}
        <span class="ml-1 opacity-75">({{ eventTypeCounts[eventType] || 0 }})</span>
      </button>
    </div>
    
    <div class="chart-container" style="height: 350px;">
      <canvas ref="chartCanvas"></canvas>
    </div>
    
    <!-- Recent Events List -->
    <div class="mt-4">
      <h4 class="text-md font-medium mb-2">Recent Events</h4>
      <div class="max-h-40 overflow-y-auto space-y-1">
        <div 
          v-for="event in recentEvents.slice(0, 10)" 
          :key="event.id"
          class="flex items-center gap-3 p-2 bg-gray-50 dark:bg-gray-800 rounded text-xs"
        >
          <div 
            :class="[
              'w-2 h-2 rounded-full',
              getEventTypeColor(event.event_type)
            ]"
          ></div>
          <span class="font-mono text-gray-500">
            {{ formatTime(event.timestamp) }}
          </span>
          <span class="font-medium">{{ event.event_type }}</span>
          <span v-if="event.tool_name" class="text-gray-600">{{ event.tool_name }}</span>
          <span v-if="event.execution_time_ms" class="text-orange-600">
            {{ event.execution_time_ms }}ms
          </span>
          <span v-if="event.success === false" class="text-red-600">FAILED</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
} from 'chart.js'
import 'chartjs-adapter-date-fns'
import { useEventsStore } from '../../stores/events'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
)

const chartCanvas = ref<HTMLCanvasElement>()
let chart: ChartJS | null = null

const eventsStore = useEventsStore()

// Real-time event tracking
const recentEvents = ref<any[]>([])
const isPaused = ref(false)
const eventTypeCounts = ref<Record<string, number>>({})
const visibleEventTypes = ref(new Set(['PRE_TOOL_USE', 'POST_TOOL_USE', 'ERROR', 'AGENT_START', 'AGENT_STOP']))

// Event type configuration
const eventTypes = ['PRE_TOOL_USE', 'POST_TOOL_USE', 'ERROR', 'AGENT_START', 'AGENT_STOP']

const eventTypeColors = {
  'PRE_TOOL_USE': 'rgb(34, 197, 94)',      // Green
  'POST_TOOL_USE': 'rgb(59, 130, 246)',   // Blue  
  'ERROR': 'rgb(239, 68, 68)',            // Red
  'AGENT_START': 'rgb(168, 85, 247)',     // Purple
  'AGENT_STOP': 'rgb(107, 114, 128)'      // Gray
}

// Computed properties
const totalEvents = computed(() => recentEvents.value.length)

const eventsPerMinute = computed(() => {
  const now = new Date()
  const oneMinuteAgo = new Date(now.getTime() - 60000)
  return recentEvents.value.filter(event => 
    new Date(event.timestamp) > oneMinuteAgo
  ).length
})

// Chart data processing
const processTimelineData = () => {
  const now = new Date()
  const timeWindows: Record<string, Record<string, number>> = {}
  
  // Create 30-minute time windows
  for (let i = 0; i < 30; i++) {
    const windowTime = new Date(now.getTime() - i * 60000) // 1-minute windows
    const windowKey = windowTime.toISOString().slice(0, 16) // YYYY-MM-DDTHH:MM
    timeWindows[windowKey] = {}
    
    eventTypes.forEach(type => {
      timeWindows[windowKey][type] = 0
    })
  }
  
  // Count events in each window
  recentEvents.value.forEach(event => {
    if (!visibleEventTypes.value.has(event.event_type)) return
    
    const eventTime = new Date(event.timestamp)
    const windowKey = eventTime.toISOString().slice(0, 16)
    
    if (timeWindows[windowKey]) {
      timeWindows[windowKey][event.event_type] = (timeWindows[windowKey][event.event_type] || 0) + 1
    }
  })
  
  const sortedWindows = Object.keys(timeWindows).sort()
  
  return {
    labels: sortedWindows.map(key => new Date(key + ':00Z')),
    datasets: eventTypes
      .filter(type => visibleEventTypes.value.has(type))
      .map(eventType => ({
        label: eventType.replace('_', ' '),
        data: sortedWindows.map(window => timeWindows[window][eventType]),
        borderColor: eventTypeColors[eventType as keyof typeof eventTypeColors],
        backgroundColor: eventTypeColors[eventType as keyof typeof eventTypeColors] + '20',
        tension: 0.1,
        pointRadius: 3,
        pointHoverRadius: 5
      }))
  }
}

const updateChart = () => {
  if (!chart) return
  
  const chartData = processTimelineData()
  chart.data = chartData
  chart.update('none') // No animation for real-time updates
}

// Event handlers
const togglePause = () => {
  isPaused.value = !isPaused.value
}

const clearEvents = () => {
  recentEvents.value = []
  eventTypeCounts.value = {}
  updateChart()
}

const toggleEventType = (eventType: string) => {
  if (visibleEventTypes.value.has(eventType)) {
    visibleEventTypes.value.delete(eventType)
  } else {
    visibleEventTypes.value.add(eventType)
  }
  updateChart()
}

const addEvent = (event: any) => {
  if (isPaused.value) return
  
  // Add to recent events (keep last 1000)
  recentEvents.value.unshift(event)
  if (recentEvents.value.length > 1000) {
    recentEvents.value = recentEvents.value.slice(0, 1000)
  }
  
  // Update event type counts
  eventTypeCounts.value[event.event_type] = (eventTypeCounts.value[event.event_type] || 0) + 1
  
  // Update chart
  updateChart()
}

const getEventTypeColor = (eventType: string): string => {
  const colorMap = {
    'PRE_TOOL_USE': 'bg-green-500',
    'POST_TOOL_USE': 'bg-blue-500', 
    'ERROR': 'bg-red-500',
    'AGENT_START': 'bg-purple-500',
    'AGENT_STOP': 'bg-gray-500'
  }
  return colorMap[eventType as keyof typeof colorMap] || 'bg-gray-400'
}

const formatTime = (timestamp: string): string => {
  return new Date(timestamp).toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  })
}

// Initialize chart
onMounted(() => {
  if (!chartCanvas.value) return
  
  chart = new ChartJS(chartCanvas.value, {
    type: 'line',
    data: processTimelineData(),
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        intersect: false,
        mode: 'index'
      },
      scales: {
        x: {
          type: 'time',
          time: {
            unit: 'minute',
            displayFormats: {
              minute: 'HH:mm'
            }
          },
          title: {
            display: true,
            text: 'Time'
          }
        },
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Events per Minute'
          }
        }
      },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            usePointStyle: true,
            pointStyle: 'circle'
          }
        },
        tooltip: {
          callbacks: {
            title: (context) => {
              return new Date(context[0].parsed.x).toLocaleString()
            },
            label: (context) => {
              return `${context.dataset.label}: ${context.parsed.y} events`
            }
          }
        }
      },
      animation: {
        duration: 0 // Disable animations for real-time updates
      }
    }
  })
  
  // Subscribe to real-time events
  eventsStore.onEvent(addEvent)
  
  // Initial chart update
  updateChart()
})

// Watch for visibility changes to update chart
watch(visibleEventTypes, () => {
  updateChart()
}, { deep: true })

onUnmounted(() => {
  if (chart) {
    chart.destroy()
  }
  // Unsubscribe from events
  eventsStore.offEvent(addEvent)
})
</script>