<template>
  <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6">
    <div class="flex items-center justify-between mb-4">
      <h3 class="text-lg font-semibold text-gray-900 dark:text-white">Recent Events</h3>
      <router-link 
        to="/events" 
        class="text-sm text-blue-600 hover:text-blue-500 dark:text-blue-400"
      >
        View All
      </router-link>
    </div>

    <div class="space-y-3">
      <div 
        v-for="event in recentEvents" 
        :key="event.id"
        class="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
      >
        <div class="flex items-center space-x-3">
          <div 
            class="w-2 h-2 rounded-full"
            :class="getEventStatusColor(event.type)"
          ></div>
          <div>
            <p class="text-sm font-medium text-gray-900 dark:text-gray-100">
              {{ event.title }}
            </p>
            <p class="text-xs text-gray-500 dark:text-gray-400">
              {{ formatRelativeTime(event.timestamp) }}
            </p>
          </div>
        </div>
        <span 
          class="inline-flex px-2 py-1 text-xs font-semibold rounded-full"
          :class="getEventTypeClass(event.type)"
        >
          {{ event.type }}
        </span>
      </div>
    </div>

    <!-- Empty State -->
    <div v-if="recentEvents.length === 0" class="text-center py-8">
      <div class="text-gray-400 dark:text-gray-500">
        <svg class="mx-auto h-12 w-12 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
        </svg>
        <p class="text-sm text-gray-500 dark:text-gray-400">No recent events</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { formatDistanceToNow } from 'date-fns'

interface Event {
  id: string
  title: string
  type: string
  timestamp: string
  details?: string
}

const recentEvents = ref<Event[]>([])

const getEventStatusColor = (type: string) => {
  const colors = {
    'PreToolUse': 'bg-blue-500',
    'PostToolUse': 'bg-green-500',
    'Notification': 'bg-yellow-500',
    'Error': 'bg-red-500',
    'Stop': 'bg-gray-500'
  }
  return colors[type as keyof typeof colors] || 'bg-gray-500'
}

const getEventTypeClass = (type: string) => {
  const classes = {
    'PreToolUse': 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    'PostToolUse': 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    'Notification': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    'Error': 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    'Stop': 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
  }
  return classes[type as keyof typeof classes] || 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
}

const formatRelativeTime = (timestamp: string) => {
  return formatDistanceToNow(new Date(timestamp), { addSuffix: true })
}

const loadRecentEvents = () => {
  // Mock data for recent events
  const eventTypes = ['PreToolUse', 'PostToolUse', 'Notification', 'Error', 'Stop']
  const events: Event[] = []

  for (let i = 0; i < 5; i++) {
    const type = eventTypes[Math.floor(Math.random() * eventTypes.length)]
    events.push({
      id: `event_${i}`,
      title: getEventTitle(type, i),
      type,
      timestamp: new Date(Date.now() - Math.random() * 3600000).toISOString(),
      details: `Event details for ${type} ${i}`
    })
  }

  recentEvents.value = events.sort((a, b) => 
    new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  )
}

const getEventTitle = (type: string, index: number) => {
  const titles = {
    'PreToolUse': `Started executing Tool_${index}`,
    'PostToolUse': `Completed Tool_${index} execution`,
    'Notification': `System notification ${index}`,
    'Error': `Error in agent operation ${index}`,
    'Stop': `Agent stopped operation ${index}`
  }
  return titles[type as keyof typeof titles] || `Event ${index}`
}

onMounted(() => {
  loadRecentEvents()
  
  // Refresh events every 30 seconds
  setInterval(loadRecentEvents, 30000)
})
</script>