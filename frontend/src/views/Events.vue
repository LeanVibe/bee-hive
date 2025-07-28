<template>
  <div class="p-6 space-y-6">
    <!-- Page header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-3xl font-bold text-gray-900 dark:text-white">Hook Lifecycle Events</h1>
        <p class="text-gray-600 dark:text-gray-300">Real-time agent lifecycle monitoring and security dashboard</p>
      </div>
      
      <div class="flex items-center space-x-3">
        <!-- View toggle -->
        <div class="flex items-center bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
          <button
            @click="activeView = 'timeline'"
            :class="[
              'px-3 py-1.5 text-sm font-medium rounded-md transition-colors',
              activeView === 'timeline'
                ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow'
                : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
            ]"
          >
            Timeline
          </button>
          <button
            @click="activeView = 'security'"
            :class="[
              'px-3 py-1.5 text-sm font-medium rounded-md transition-colors',
              activeView === 'security'
                ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow'
                : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
            ]"
          >
            Security
          </button>
          <button
            @click="activeView = 'performance'"
            :class="[
              'px-3 py-1.5 text-sm font-medium rounded-md transition-colors',
              activeView === 'performance'
                ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow'
                : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
            ]"
          >
            Performance
          </button>
        </div>

        <!-- Connection status -->
        <div class="flex items-center space-x-2 px-3 py-1.5 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
          <div 
            :class="[
              'w-2 h-2 rounded-full',
              eventsStore.wsConnected ? 'bg-green-500' : 'bg-red-500'
            ]"
          ></div>
          <span class="text-sm font-medium">
            {{ eventsStore.wsConnected ? 'Live' : 'Disconnected' }}
          </span>
        </div>
      </div>
    </div>

    <!-- Filter panel -->
    <EventFilterPanel
      :filters="currentFilters"
      :available-agents="eventsStore.agents"
      :available-sessions="eventsStore.sessions"
      @filters-change="handleFiltersChange"
      @clear-filters="handleClearFilters"
    />

    <!-- Main content area -->
    <div class="space-y-6">
      <!-- Timeline view -->
      <div v-if="activeView === 'timeline'" class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- Main timeline -->
        <div class="lg:col-span-2">
          <HookEventTimeline
            :height="600"
            :max-events="200"
            :auto-scroll="true"
            :initial-filters="currentFilters"
            @event-click="handleEventClick"
          />
        </div>
        
        <!-- Side panel with session info -->
        <div class="space-y-6">
          <SessionVisualization
            :sessions="eventsStore.sessions"
            :max-sessions="10"
            :show-inactive="false"
          />
          
          <!-- Quick stats -->
          <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-4">
            <h4 class="text-lg font-semibold text-gray-900 dark:text-white mb-3">
              Quick Stats
            </h4>
            <div class="space-y-3">
              <div class="flex justify-between">
                <span class="text-sm text-gray-600 dark:text-gray-400">Total Events</span>
                <span class="text-sm font-medium text-gray-900 dark:text-white">
                  {{ eventsStore.hookEvents.length }}
                </span>
              </div>
              <div class="flex justify-between">
                <span class="text-sm text-gray-600 dark:text-gray-400">Active Agents</span>
                <span class="text-sm font-medium text-gray-900 dark:text-white">
                  {{ activeAgentsCount }}
                </span>
              </div>
              <div class="flex justify-between">
                <span class="text-sm text-gray-600 dark:text-gray-400">Security Alerts</span>
                <span class="text-sm font-medium text-red-600 dark:text-red-400">
                  {{ eventsStore.securityAlerts.length }}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Security view -->
      <div v-else-if="activeView === 'security'">
        <SecurityDashboard
          :show-approval-queue="true"
          :alert-limit="15"
          :auto-refresh="true"
          :refresh-interval="30000"
        />
      </div>

      <!-- Performance view -->
      <div v-else-if="activeView === 'performance'">
        <PerformanceMonitoringDashboard
          :refresh-interval="5000"
          :show-historical-data="true"
          :time-range="'1h'"
        />
      </div>
    </div>

    <!-- Event details modal -->
    <EventDetailsModal
      :event="selectedEvent"
      :is-open="isEventModalOpen"
      :show-security-info="true"
      :show-performance-info="true"
      @close="closeEventModal"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useEventsStore } from '@/stores/events'
import { 
  type EventFilter,
  type HookEvent as TypedHookEvent
} from '@/types/hooks'
import HookEventTimeline from '@/components/hooks/HookEventTimeline.vue'
import SecurityDashboard from '@/components/hooks/SecurityDashboard.vue'
import EventFilterPanel from '@/components/hooks/EventFilterPanel.vue'
import SessionVisualization from '@/components/hooks/SessionVisualization.vue'
import EventDetailsModal from '@/components/hooks/EventDetailsModal.vue'
import PerformanceMonitoringDashboard from '@/components/hooks/PerformanceMonitoringDashboard.vue'

// Store
const eventsStore = useEventsStore()

// Local state
const activeView = ref<'timeline' | 'security' | 'performance'>('timeline')
const currentFilters = ref<EventFilter>({})
const selectedEvent = ref<TypedHookEvent | undefined>()
const isEventModalOpen = ref(false)

// Computed
const activeAgentsCount = computed(() => {
  return eventsStore.agents.filter(agent => agent.status === 'active').length
})

// Methods
const handleFiltersChange = (filters: EventFilter) => {
  currentFilters.value = filters
  eventsStore.updateHookFilters(filters)
}

const handleClearFilters = () => {
  currentFilters.value = {}
  eventsStore.clearHookFilters()
}

const handleEventClick = (event: TypedHookEvent) => {
  selectedEvent.value = event
  isEventModalOpen.value = true
}

const closeEventModal = () => {
  isEventModalOpen.value = false
  selectedEvent.value = undefined
}

// Lifecycle
onMounted(() => {
  // Connect to WebSocket for real-time events
  eventsStore.connectWebSocket()
})
</script>