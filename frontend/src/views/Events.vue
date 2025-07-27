<template>
  <div class="p-6">
    <div class="mb-6">
      <h1 class="text-3xl font-bold text-gray-900 dark:text-white">Events</h1>
      <p class="text-gray-600 dark:text-gray-300">Real-time system events and activity</p>
    </div>

    <!-- Event Filters -->
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-4 mb-6">
      <div class="flex flex-wrap gap-4">
        <select 
          v-model="selectedEventType" 
          class="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700"
        >
          <option value="">All Event Types</option>
          <option value="PreToolUse">Pre Tool Use</option>
          <option value="PostToolUse">Post Tool Use</option>
          <option value="Notification">Notification</option>
          <option value="Stop">Stop</option>
        </select>

        <select 
          v-model="selectedSeverity" 
          class="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700"
        >
          <option value="">All Severities</option>
          <option value="info">Info</option>
          <option value="warning">Warning</option>
          <option value="error">Error</option>
          <option value="critical">Critical</option>
        </select>

        <button 
          @click="refreshEvents"
          class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          Refresh
        </button>
      </div>
    </div>

    <!-- Events Table -->
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm overflow-hidden">
      <div class="overflow-x-auto">
        <table class="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead class="bg-gray-50 dark:bg-gray-700">
            <tr>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Timestamp
              </th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Event Type
              </th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Agent ID
              </th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Details
              </th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Status
              </th>
            </tr>
          </thead>
          <tbody class="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
            <tr v-for="event in filteredEvents" :key="event.id" class="hover:bg-gray-50 dark:hover:bg-gray-700">
              <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                {{ formatTimestamp(event.timestamp) }}
              </td>
              <td class="px-6 py-4 whitespace-nowrap">
                <span 
                  class="inline-flex px-2 py-1 text-xs font-semibold rounded-full"
                  :class="getEventTypeClass(event.event_type)"
                >
                  {{ event.event_type }}
                </span>
              </td>
              <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400 font-mono">
                {{ event.agent_id.substring(0, 8) }}...
              </td>
              <td class="px-6 py-4 text-sm text-gray-900 dark:text-gray-100">
                {{ getEventDetails(event) }}
              </td>
              <td class="px-6 py-4 whitespace-nowrap">
                <span 
                  class="inline-flex px-2 py-1 text-xs font-semibold rounded-full"
                  :class="getStatusClass(event.status || 'completed')"
                >
                  {{ event.status || 'completed' }}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Empty State -->
      <div v-if="filteredEvents.length === 0" class="text-center py-12">
        <p class="text-gray-500 dark:text-gray-400">No events found matching the current filters.</p>
      </div>
    </div>

    <!-- Pagination -->
    <div class="flex justify-between items-center mt-6">
      <div class="text-sm text-gray-700 dark:text-gray-300">
        Showing {{ (currentPage - 1) * pageSize + 1 }} to 
        {{ Math.min(currentPage * pageSize, totalEvents) }} of {{ totalEvents }} events
      </div>
      <div class="flex space-x-2">
        <button 
          @click="previousPage" 
          :disabled="currentPage === 1"
          class="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm disabled:opacity-50"
        >
          Previous
        </button>
        <button 
          @click="nextPage" 
          :disabled="currentPage * pageSize >= totalEvents"
          class="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm disabled:opacity-50"
        >
          Next
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { format } from 'date-fns'

interface Event {
  id: string
  timestamp: string
  event_type: string
  agent_id: string
  session_id: string
  payload: any
  status?: string
}

const events = ref<Event[]>([])
const selectedEventType = ref('')
const selectedSeverity = ref('')
const currentPage = ref(1)
const pageSize = ref(50)
const totalEvents = ref(0)

const filteredEvents = computed(() => {
  let filtered = events.value

  if (selectedEventType.value) {
    filtered = filtered.filter(event => event.event_type === selectedEventType.value)
  }

  return filtered.slice((currentPage.value - 1) * pageSize.value, currentPage.value * pageSize.value)
})

const formatTimestamp = (timestamp: string) => {
  return format(new Date(timestamp || ''), 'MMM dd, HH:mm:ss')
}

const getEventTypeClass = (eventType: string) => {
  const classes = {
    'PreToolUse': 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    'PostToolUse': 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    'Notification': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    'Stop': 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
  }
  return classes[eventType as keyof typeof classes] || 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
}

const getStatusClass = (status: string) => {
  const classes = {
    'completed': 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    'failed': 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    'pending': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
  }
  return classes[status as keyof typeof classes] || 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
}

const getEventDetails = (event: Event) => {
  if (event.event_type === 'PreToolUse' && event.payload.tool_name) {
    return `Tool: ${event.payload.tool_name}`
  }
  if (event.event_type === 'PostToolUse' && event.payload.tool_name) {
    return `Tool: ${event.payload.tool_name} (${event.payload.success ? 'Success' : 'Failed'})`
  }
  if (event.event_type === 'Notification' && event.payload.message) {
    return event.payload.message.substring(0, 50) + '...'
  }
  return 'Event details...'
}

const refreshEvents = () => {
  // Mock data generation
  const eventTypes = ['PreToolUse', 'PostToolUse', 'Notification', 'Stop']
  const mockEvents: Event[] = []

  for (let i = 0; i < 100; i++) {
    const eventType = eventTypes[Math.floor(Math.random() * eventTypes.length)]
    mockEvents.push({
      id: `event_${i}`,
      timestamp: new Date(Date.now() - Math.random() * 86400000).toISOString(),
      event_type: eventType,
      agent_id: `agent_${Math.floor(Math.random() * 10)}`,
      session_id: `session_${Math.floor(Math.random() * 20)}`,
      payload: {
        tool_name: eventType.includes('Tool') ? `Tool_${Math.floor(Math.random() * 5)}` : undefined,
        success: eventType === 'PostToolUse' ? Math.random() > 0.1 : undefined,
        message: eventType === 'Notification' ? `Test notification message ${i}` : undefined
      },
      status: Math.random() > 0.05 ? 'completed' : 'failed'
    })
  }

  events.value = mockEvents.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
  totalEvents.value = mockEvents.length
}

const previousPage = () => {
  if (currentPage.value > 1) {
    currentPage.value--
  }
}

const nextPage = () => {
  if (currentPage.value * pageSize.value < totalEvents.value) {
    currentPage.value++
  }
}

onMounted(() => {
  refreshEvents()
})
</script>