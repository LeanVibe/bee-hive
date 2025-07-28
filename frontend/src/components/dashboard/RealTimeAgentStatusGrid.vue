<template>
  <div class="glass-card rounded-xl p-6">
    <!-- Header -->
    <div class="flex items-center justify-between mb-6">
      <div class="flex items-center space-x-3">
        <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
          Live Agent Status
        </h3>
        <div class="flex items-center space-x-2">
          <div 
            class="w-2 h-2 rounded-full transition-colors duration-200"
            :class="connectionStatusColor"
          ></div>
          <span class="text-sm text-slate-600 dark:text-slate-400">
            {{ connectionStatusText }}
          </span>
        </div>
      </div>
      
      <div class="flex items-center space-x-3">
        <!-- Real-time indicator -->
        <div class="flex items-center space-x-2 text-sm text-slate-500 dark:text-slate-400">
          <div class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
          <span>Live</span>
        </div>
        
        <!-- Refresh button -->
        <button
          @click="refreshAgents"
          :disabled="!isConnected"
          class="btn-secondary btn-sm"
          :class="{ 'opacity-50 cursor-not-allowed': !isConnected }"
        >
          <ArrowPathIcon 
            class="w-4 h-4" 
            :class="{ 'animate-spin': refreshing }"
          />
        </button>
      </div>
    </div>

    <!-- Connection Status Banner -->
    <div 
      v-if="connectionStatus !== 'connected'"
      class="mb-4 p-3 rounded-lg border-l-4"
      :class="connectionBannerClass"
    >
      <div class="flex items-center">
        <ExclamationTriangleIcon 
          v-if="connectionStatus === 'error'"
          class="w-5 h-5 mr-2"
        />
        <ClockIcon 
          v-else-if="connectionStatus === 'connecting'"
          class="w-5 h-5 mr-2 animate-spin"
        />
        <span class="text-sm font-medium">{{ connectionBannerText }}</span>
      </div>
    </div>

    <!-- Agent Grid -->
    <div v-if="agents.length > 0" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 mb-6">
      <div 
        v-for="agent in agents" 
        :key="agent.id"
        class="agent-card border border-slate-200 dark:border-slate-600 rounded-lg p-4 transition-all duration-200 hover:shadow-md hover:border-primary-300 dark:hover:border-primary-600"
        :class="getAgentCardClass(agent)"
        @click="selectAgent(agent)"
      >
        <!-- Agent Header -->
        <div class="flex items-center justify-between mb-3">
          <div class="flex items-center space-x-2">
            <div 
              class="w-3 h-3 rounded-full transition-colors duration-200"
              :class="getStatusColor(agent.status)"
            ></div>
            <span class="text-sm font-medium text-slate-900 dark:text-slate-100 truncate">
              {{ agent.name }}
            </span>
          </div>
          <span 
            class="inline-flex px-2 py-1 text-xs font-semibold rounded-full transition-colors duration-200"
            :class="getStatusClass(agent.status)"
          >
            {{ agent.status }}
          </span>
        </div>

        <!-- Agent Details -->
        <div class="space-y-2 text-xs text-slate-600 dark:text-slate-400">
          <div class="flex justify-between">
            <span>ID:</span>
            <span class="font-mono">{{ agent.id.substring(0, 8) }}...</span>
          </div>
          <div class="flex justify-between">
            <span>Tasks:</span>
            <span class="font-semibold">{{ agent.tasksCompleted }}/{{ agent.totalTasks }}</span>
          </div>
          <div class="flex justify-between">
            <span>Uptime:</span>
            <span>{{ formatUptime(agent.startTime) }}</span>
          </div>
          <div class="flex justify-between">
            <span>Memory:</span>
            <span>{{ formatMemory(agent.memoryUsage) }}</span>
          </div>
          <div v-if="agent.agentType" class="flex justify-between">
            <span>Type:</span>
            <span class="capitalize">{{ agent.agentType }}</span>
          </div>
        </div>

        <!-- Current Activity -->
        <div v-if="agent.currentActivity" class="mt-3 pt-3 border-t border-slate-200 dark:border-slate-600">
          <p class="text-xs text-slate-500 dark:text-slate-400 mb-1">Current Activity:</p>
          <p class="text-xs font-medium text-slate-700 dark:text-slate-300 truncate">
            {{ agent.currentActivity }}
          </p>
        </div>

        <!-- Performance Indicator -->
        <div class="mt-3">
          <div class="flex justify-between text-xs text-slate-600 dark:text-slate-400 mb-1">
            <span>Performance</span>
            <span>{{ agent.performance }}%</span>
          </div>
          <div class="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
            <div 
              class="h-1.5 rounded-full transition-all duration-500"
              :class="getPerformanceColor(agent.performance)"
              :style="{ width: `${agent.performance}%` }"
            ></div>
          </div>
        </div>

        <!-- Last Activity -->
        <div v-if="agent.lastActivity" class="mt-2 text-xs text-slate-500 dark:text-slate-400">
          Last activity: {{ formatTime(agent.lastActivity) }}
        </div>
      </div>
    </div>

    <!-- Empty State -->
    <div v-else-if="isConnected" class="text-center py-12">
      <UserGroupIcon class="w-12 h-12 text-slate-400 mx-auto mb-4" />
      <h3 class="text-lg font-medium text-slate-900 dark:text-white mb-2">No Active Agents</h3>
      <p class="text-slate-600 dark:text-slate-400">
        Agents will appear here when they connect to the system.
      </p>
    </div>

    <!-- Loading State -->
    <div v-else class="text-center py-12">
      <div class="animate-spin w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full mx-auto mb-4"></div>
      <p class="text-slate-600 dark:text-slate-400">Connecting to agent monitoring service...</p>
    </div>

    <!-- Summary Stats -->
    <div v-if="agents.length > 0" class="pt-6 border-t border-slate-200 dark:border-slate-600">
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
        <div>
          <p class="text-2xl font-bold text-slate-900 dark:text-white">{{ (systemStats as any).active_agents || (systemStats as any).activeAgents || 0 }}</p>
          <p class="text-xs text-slate-600 dark:text-slate-400">Active</p>
        </div>
        <div>
          <p class="text-2xl font-bold text-slate-900 dark:text-white">{{ (systemStats as any).total_tasks_completed || (systemStats as any).totalTasksCompleted || 0 }}</p>
          <p class="text-xs text-slate-600 dark:text-slate-400">Tasks</p>
        </div>
        <div>
          <p class="text-2xl font-bold text-slate-900 dark:text-white">{{ (systemStats as any).average_performance || (systemStats as any).averagePerformance || 0 }}%</p>
          <p class="text-xs text-slate-600 dark:text-slate-400">Avg Performance</p>
        </div>
        <div>
          <p class="text-2xl font-bold text-slate-900 dark:text-white">{{ formatMemory((systemStats as any).total_memory_usage || (systemStats as any).totalMemoryUsage || 0) }}</p>
          <p class="text-xs text-slate-600 dark:text-slate-400">Total Memory</p>
        </div>
      </div>
    </div>

    <!-- Recent Events -->
    <div v-if="showRecentEvents && recentEvents.length > 0" class="mt-6 pt-6 border-t border-slate-200 dark:border-slate-600">
      <h4 class="text-sm font-medium text-slate-900 dark:text-white mb-3">Recent Events</h4>
      <div class="space-y-2 max-h-32 overflow-y-auto">
        <div 
          v-for="event in recentEvents.slice(0, 5)" 
          :key="`${event.agent_id}-${event.timestamp}`"
          class="flex items-center justify-between text-xs p-2 rounded bg-slate-50 dark:bg-slate-800"
        >
          <div class="flex items-center space-x-2">
            <div 
              class="w-2 h-2 rounded-full"
              :class="getEventTypeColor(event.event_type)"
            ></div>
            <span class="font-medium">{{ formatEventType(event.event_type) }}</span>
            <span class="text-slate-500">{{ event.payload.name || event.agent_id.substring(0, 8) }}</span>
          </div>
          <span class="text-slate-500">{{ formatTime(event.timestamp) }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { formatDistanceToNow } from 'date-fns'
import { 
  ArrowPathIcon, 
  UserGroupIcon, 
  ExclamationTriangleIcon, 
  ClockIcon 
} from '@heroicons/vue/24/outline'

import { useAgentMonitoring, type AgentStatus, type AgentLifecycleEvent } from '@/services/agentMonitoringService'

// Props
interface Props {
  showRecentEvents?: boolean
  autoRefresh?: boolean
  refreshInterval?: number
}

withDefaults(defineProps<Props>(), {
  showRecentEvents: true,
  autoRefresh: true,
  refreshInterval: 30000
})

// Emits
const emit = defineEmits<{
  agentSelected: [agent: AgentStatus]
  connectionStatusChanged: [status: string]
}>()

// Agent monitoring service
const {
  agents,
  activeAgents,
  recentEvents,
  systemStats: serviceSystemStats,
  isConnected,
  connectionStatus,
  initialize,
  disconnect,
  onEvent
} = useAgentMonitoring()

// Local state
const refreshing = ref(false)
const selectedAgent = ref<AgentStatus | null>(null)

// Computed properties
const connectionStatusColor = computed(() => {
  switch (connectionStatus.value) {
    case 'connected': return 'bg-green-500'
    case 'connecting': return 'bg-yellow-500 animate-pulse'
    case 'error': return 'bg-red-500'
    default: return 'bg-slate-400'
  }
})

const connectionStatusText = computed(() => {
  switch (connectionStatus.value) {
    case 'connected': return `${agents.value.length} agents`
    case 'connecting': return 'Connecting...'
    case 'error': return 'Connection error'
    default: return 'Disconnected'
  }
})

const connectionBannerClass = computed(() => {
  switch (connectionStatus.value) {
    case 'error': return 'bg-red-50 border-red-400 text-red-700 dark:bg-red-900/20 dark:border-red-700 dark:text-red-300'
    case 'connecting': return 'bg-yellow-50 border-yellow-400 text-yellow-700 dark:bg-yellow-900/20 dark:border-yellow-700 dark:text-yellow-300'
    default: return 'bg-slate-50 border-slate-400 text-slate-700 dark:bg-slate-900/20 dark:border-slate-700 dark:text-slate-300'
  }
})

const connectionBannerText = computed(() => {
  switch (connectionStatus.value) {
    case 'error': return 'Unable to connect to agent monitoring service. Retrying...'
    case 'connecting': return 'Establishing connection to agent monitoring service...'
    default: return 'Not connected to monitoring service'
  }
})

const systemStats = computed(() => {
  if (serviceSystemStats.value) {
    return serviceSystemStats.value
  }
  
  // Calculate from local data if service stats not available
  const stats = {
    activeAgents: activeAgents.value.length,
    totalTasksCompleted: agents.value.reduce((sum, a) => sum + a.tasksCompleted, 0),
    averagePerformance: agents.value.length > 0 
      ? Math.round(agents.value.reduce((sum, a) => sum + a.performance, 0) / agents.value.length)
      : 0,
    totalMemoryUsage: agents.value.reduce((sum, a) => sum + a.memoryUsage, 0)
  }
  
  return stats
})

// Methods
const refreshAgents = async () => {
  if (!isConnected.value) return
  
  refreshing.value = true
  try {
    // Service automatically updates via WebSocket, but we can trigger a refresh
    await new Promise(resolve => setTimeout(resolve, 1000)) // Simulate refresh
  } finally {
    refreshing.value = false
  }
}

const selectAgent = (agent: AgentStatus) => {
  selectedAgent.value = agent
  emit('agentSelected', agent)
}

const getStatusColor = (status: string) => {
  const colors = {
    'active': 'bg-green-500',
    'busy': 'bg-blue-500',
    'idle': 'bg-yellow-500',
    'error': 'bg-red-500',
    'offline': 'bg-slate-400'
  }
  return colors[status as keyof typeof colors] || 'bg-slate-500'
}

const getStatusClass = (status: string) => {
  const classes = {
    'active': 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    'busy': 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    'idle': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    'error': 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    'offline': 'bg-slate-100 text-slate-800 dark:bg-slate-900 dark:text-slate-200'
  }
  return classes[status as keyof typeof classes] || 'bg-slate-100 text-slate-800 dark:bg-slate-900 dark:text-slate-200'
}

const getAgentCardClass = (agent: AgentStatus) => {
  const baseClass = ''
  if (selectedAgent.value?.id === agent.id) {
    return baseClass + ' ring-2 ring-primary-500 border-primary-500'
  }
  return baseClass
}

const getPerformanceColor = (performance: number) => {
  if (performance >= 90) return 'bg-green-500'
  if (performance >= 70) return 'bg-blue-500'
  if (performance >= 50) return 'bg-yellow-500'
  return 'bg-red-500'
}

const getEventTypeColor = (eventType: string) => {
  const colors = {
    'agent_registered': 'bg-green-500',
    'agent_deregistered': 'bg-red-500',
    'task_assigned': 'bg-blue-500',
    'task_completed': 'bg-green-500',
    'task_failed': 'bg-red-500',
    'agent_heartbeat': 'bg-slate-500'
  }
  return colors[eventType as keyof typeof colors] || 'bg-slate-500'
}

const formatUptime = (startTime: string) => {
  try {
    return formatDistanceToNow(new Date(startTime))
  } catch {
    return 'Unknown'
  }
}

const formatTime = (timestamp: string) => {
  try {
    return formatDistanceToNow(new Date(timestamp), { addSuffix: true })
  } catch {
    return 'Unknown'
  }
}

const formatMemory = (mb: number) => {
  if (mb > 1024) {
    return `${(mb / 1024).toFixed(1)}GB`
  }
  return `${Math.round(mb)}MB`
}

const formatEventType = (eventType: string) => {
  return eventType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
}

// Event handlers
let unsubscribeHandlers: Array<() => void> = []

// Lifecycle
onMounted(async () => {
  try {
    await initialize()
    
    // Subscribe to events
    unsubscribeHandlers.push(
      onEvent('agent_lifecycle_event', (event: AgentLifecycleEvent) => {
        console.log('🔄 Agent lifecycle event:', event)
      }),
      
      onEvent('performance_metrics', (metrics) => {
        console.log('📊 Performance metrics update:', metrics)
      }),
      
      onEvent('system_stats', (stats) => {
        console.log('📈 System stats update:', stats)
      })
    )
    
    // Watch connection status changes
    const unwatchConnection = computed(() => connectionStatus.value)
    unwatchConnection.value // Access to make it reactive
    
  } catch (error) {
    console.error('❌ Failed to initialize agent monitoring:', error)
  }
})

onUnmounted(() => {
  // Cleanup event handlers
  unsubscribeHandlers.forEach(unsubscribe => unsubscribe())
  unsubscribeHandlers = []
  
  // Disconnect service
  disconnect()
})
</script>

<style scoped>
.agent-card {
  transition: all 0.2s ease;
}

.agent-card:hover {
  transform: translateY(-1px);
}

.glass-card {
  backdrop-filter: blur(10px);
  background: rgba(255, 255, 255, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.2);
}

@media (prefers-color-scheme: dark) {
  .glass-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
  }
}
</style>