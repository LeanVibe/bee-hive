<template>
  <div class="session-visualization bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
    <!-- Header -->
    <div class="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
      <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
        Active Sessions
      </h3>
      <div class="flex items-center space-x-2">
        <button
          @click="showInactive = !showInactive"
          :class="[
            'px-2 py-1 text-xs rounded-full transition-colors',
            showInactive
              ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
              : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300'
          ]"
        >
          {{ showInactive ? 'Hide' : 'Show' }} Inactive
        </button>
      </div>
    </div>

    <!-- Sessions list -->
    <div class="p-4">
      <div class="space-y-3 max-h-96 overflow-y-auto">
        <div
          v-for="session in displayedSessions"
          :key="session.session_id"
          class="session-item"
          :class="getSessionItemClass(session.status)"
          @click="$emit('sessionClick', session)"
        >
          <!-- Session header -->
          <div class="flex items-center justify-between mb-2">
            <div class="flex items-center space-x-3">
              <!-- Status indicator -->
              <div 
                :class="[
                  'w-3 h-3 rounded-full',
                  getStatusColor(session.status)
                ]"
              ></div>
              
              <!-- Session ID -->
              <span class="font-mono text-sm font-medium text-gray-900 dark:text-white">
                {{ session.session_id.substring(0, 8) }}...
              </span>
              
              <!-- Status badge -->
              <span 
                :class="[
                  'inline-flex px-2 py-0.5 text-xs font-semibold rounded-full',
                  getStatusBadgeClass(session.status)
                ]"
              >
                {{ session.status }}
              </span>
            </div>
            
            <!-- Duration -->
            <span class="text-xs text-gray-500 dark:text-gray-400">
              {{ formatDuration(session) }}
            </span>
          </div>

          <!-- Agent visualization -->
          <div class="flex items-center space-x-2 mb-3">
            <span class="text-xs text-gray-500 dark:text-gray-400">
              Agents:
            </span>
            <div class="flex space-x-1">
              <div
                v-for="(agentId, index) in session.agent_ids.slice(0, 5)"
                :key="agentId"
                :class="[
                  'w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium',
                  getAgentColor(agentId, index)
                ]"
                :title="agentId"
              >
                {{ agentId.substring(0, 2).toUpperCase() }}
              </div>
              <div
                v-if="session.agent_ids.length > 5"
                class="w-6 h-6 rounded-full bg-gray-200 dark:bg-gray-600 flex items-center justify-center text-xs font-medium text-gray-600 dark:text-gray-300"
              >
                +{{ session.agent_ids.length - 5 }}
              </div>
            </div>
          </div>

          <!-- Session metrics -->
          <div class="grid grid-cols-3 gap-4 text-xs">
            <div class="text-center">
              <div class="font-semibold text-gray-900 dark:text-white">
                {{ session.event_count }}
              </div>
              <div class="text-gray-500 dark:text-gray-400">
                Events
              </div>
            </div>
            <div class="text-center">
              <div class="font-semibold text-red-600 dark:text-red-400">
                {{ session.error_count }}
              </div>
              <div class="text-gray-500 dark:text-gray-400">
                Errors
              </div>
            </div>
            <div class="text-center">
              <div class="font-semibold text-orange-600 dark:text-orange-400">
                {{ session.blocked_count }}
              </div>
              <div class="text-gray-500 dark:text-gray-400">
                Blocked
              </div>
            </div>
          </div>

          <!-- Progress bar for active sessions -->
          <div v-if="session.status === 'active'" class="mt-3">
            <div class="flex items-center justify-between mb-1">
              <span class="text-xs text-gray-500 dark:text-gray-400">Activity</span>
              <span class="text-xs text-gray-500 dark:text-gray-400">
                {{ getActivityLevel(session) }}
              </span>
            </div>
            <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
              <div 
                :class="[
                  'h-1.5 rounded-full transition-all duration-300',
                  getActivityBarColor(session)
                ]"
                :style="{ width: `${getActivityPercentage(session)}%` }"
              ></div>
            </div>
          </div>
        </div>

        <!-- Empty state -->
        <div 
          v-if="displayedSessions.length === 0"
          class="text-center py-8"
        >
          <div class="w-12 h-12 mx-auto mb-3 bg-gray-100 dark:bg-gray-700 rounded-full flex items-center justify-center">
            <UsersIcon class="w-6 h-6 text-gray-400" />
          </div>
          <h4 class="text-sm font-medium text-gray-900 dark:text-white">
            No {{ showInactive ? '' : 'active' }} sessions
          </h4>
          <p class="text-xs text-gray-500 dark:text-gray-400">
            Sessions will appear here when agents start working
          </p>
        </div>
      </div>
    </div>

    <!-- Footer with session summary -->
    <div class="border-t border-gray-200 dark:border-gray-700 p-3 bg-gray-50 dark:bg-gray-900 text-xs">
      <div class="flex justify-between items-center">
        <div class="flex space-x-4">
          <span class="text-gray-600 dark:text-gray-400">
            <strong>{{ activeSessions.length }}</strong> active
          </span>
          <span class="text-gray-600 dark:text-gray-400">
            <strong>{{ completedSessions.length }}</strong> completed
          </span>
          <span v-if="errorSessions.length > 0" class="text-red-600 dark:text-red-400">
            <strong>{{ errorSessions.length }}</strong> with errors
          </span>
        </div>
        <div class="text-gray-500 dark:text-gray-400">
          Total: {{ sessions.length }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { 
  type SessionInfo,
  type SessionVisualizationProps
} from '@/types/hooks'
import { UsersIcon } from '@heroicons/vue/24/outline'
import { formatDistanceToNow } from 'date-fns'

// Props
interface Props extends SessionVisualizationProps {}

const props = withDefaults(defineProps<Props>(), {
  maxSessions: 10,
  showInactive: false,
  colorScheme: 'default'
})

// Emits
const emit = defineEmits<{
  sessionClick: [session: SessionInfo]
}>()

// Local state
const showInactive = ref(props.showInactive)

// Computed
const displayedSessions = computed(() => {
  let filtered = props.sessions
  
  if (!showInactive.value) {
    filtered = filtered.filter(session => session.status === 'active')
  }
  
  // Sort by start time (most recent first)
  filtered = filtered.sort((a, b) => 
    new Date(b.start_time).getTime() - new Date(a.start_time).getTime()
  )
  
  return filtered.slice(0, props.maxSessions)
})

const activeSessions = computed(() => 
  props.sessions.filter(session => session.status === 'active')
)

const completedSessions = computed(() => 
  props.sessions.filter(session => session.status === 'completed')
)

const errorSessions = computed(() => 
  props.sessions.filter(session => session.status === 'error')
)

// Methods
const getStatusColor = (status: string) => {
  const colorMap = {
    active: 'bg-green-500',
    completed: 'bg-blue-500',
    error: 'bg-red-500',
    terminated: 'bg-orange-500'
  }
  return colorMap[status as keyof typeof colorMap] || 'bg-gray-500'
}

const getStatusBadgeClass = (status: string) => {
  const classMap = {
    active: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    completed: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    error: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    terminated: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200'
  }
  return classMap[status as keyof typeof classMap] || 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
}

const getSessionItemClass = (status: string) => {
  const baseClasses = 'session-item p-4 border border-gray-200 dark:border-gray-600 rounded-lg cursor-pointer transition-all duration-200 hover:shadow-md'
  
  const statusClasses = {
    active: 'border-l-4 border-l-green-500 bg-green-50 dark:bg-green-900/10',
    completed: 'border-l-4 border-l-blue-500 bg-blue-50 dark:bg-blue-900/10',
    error: 'border-l-4 border-l-red-500 bg-red-50 dark:bg-red-900/10',
    terminated: 'border-l-4 border-l-orange-500 bg-orange-50 dark:bg-orange-900/10'
  }
  
  return `${baseClasses} ${statusClasses[status as keyof typeof statusClasses] || ''}`
}

const getAgentColor = (agentId: string, index: number) => {
  if (props.colorScheme === 'status') {
    // In a real implementation, you'd look up the agent's status
    return 'bg-blue-500 text-white'
  }
  
  // Default color scheme - use index-based colors
  const colors = [
    'bg-blue-500 text-white',
    'bg-green-500 text-white',
    'bg-purple-500 text-white',
    'bg-pink-500 text-white',
    'bg-indigo-500 text-white',
    'bg-yellow-500 text-white',
    'bg-red-500 text-white',
    'bg-teal-500 text-white'
  ]
  
  return colors[index % colors.length]
}

const formatDuration = (session: SessionInfo) => {
  const start = new Date(session.start_time)
  const end = session.end_time ? new Date(session.end_time) : new Date()
  
  try {
    return formatDistanceToNow(start, { addSuffix: false })
  } catch {
    return 'Unknown'
  }
}

const getActivityLevel = (session: SessionInfo) => {
  if (session.event_count === 0) return 'Idle'
  if (session.event_count < 10) return 'Low'
  if (session.event_count < 50) return 'Medium'
  return 'High'
}

const getActivityPercentage = (session: SessionInfo) => {
  // Calculate activity based on events per minute since start
  const start = new Date(session.start_time)
  const now = new Date()
  const minutesElapsed = Math.max(1, (now.getTime() - start.getTime()) / (1000 * 60))
  const eventsPerMinute = session.event_count / minutesElapsed
  
  // Scale to 0-100 where 5 events/minute = 100%
  return Math.min(100, (eventsPerMinute / 5) * 100)
}

const getActivityBarColor = (session: SessionInfo) => {
  const percentage = getActivityPercentage(session)
  
  if (percentage >= 80) return 'bg-red-500'
  if (percentage >= 60) return 'bg-orange-500'
  if (percentage >= 40) return 'bg-yellow-500'
  if (percentage >= 20) return 'bg-blue-500'
  return 'bg-gray-400'
}
</script>

<style scoped>
.session-visualization {
  min-height: 300px;
}

.session-item:hover {
  transform: translateY(-1px);
}

/* Custom scrollbar */
.space-y-3::-webkit-scrollbar {
  width: 4px;
}

.space-y-3::-webkit-scrollbar-track {
  @apply bg-gray-100 dark:bg-gray-700 rounded-full;
}

.space-y-3::-webkit-scrollbar-thumb {
  @apply bg-gray-300 dark:bg-gray-600 rounded-full;
}

.space-y-3::-webkit-scrollbar-thumb:hover {
  @apply bg-gray-400 dark:bg-gray-500;
}

/* Responsive design */
@media (max-width: 768px) {
  .session-item {
    @apply p-3;
  }
  
  .grid.grid-cols-3 {
    @apply grid-cols-1 gap-2;
  }
}
</style>