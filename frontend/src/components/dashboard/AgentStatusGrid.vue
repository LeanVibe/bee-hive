<template>
  <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6">
    <div class="flex items-center justify-between mb-4">
      <h3 class="text-lg font-semibold text-gray-900 dark:text-white">Agent Status</h3>
      <div class="flex items-center space-x-2">
        <div class="w-2 h-2 rounded-full bg-green-500"></div>
        <span class="text-sm text-gray-600 dark:text-gray-400">{{ activeAgents }} active</span>
      </div>
    </div>

    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      <div 
        v-for="agent in agents" 
        :key="agent.id"
        class="border border-gray-200 dark:border-gray-600 rounded-lg p-4 hover:shadow-md transition-shadow"
      >
        <div class="flex items-center justify-between mb-3">
          <div class="flex items-center space-x-2">
            <div 
              class="w-3 h-3 rounded-full"
              :class="getStatusColor(agent.status)"
            ></div>
            <span class="text-sm font-medium text-gray-900 dark:text-gray-100">
              Agent {{ agent.name }}
            </span>
          </div>
          <span 
            class="inline-flex px-2 py-1 text-xs font-semibold rounded-full"
            :class="getStatusClass(agent.status)"
          >
            {{ agent.status }}
          </span>
        </div>

        <div class="space-y-2 text-xs text-gray-600 dark:text-gray-400">
          <div class="flex justify-between">
            <span>Session:</span>
            <span class="font-mono">{{ agent.sessionId?.substring(0, 8) }}...</span>
          </div>
          <div class="flex justify-between">
            <span>Tasks:</span>
            <span>{{ agent.tasksCompleted }}/{{ agent.totalTasks }}</span>
          </div>
          <div class="flex justify-between">
            <span>Uptime:</span>
            <span>{{ formatUptime(agent.startTime) }}</span>
          </div>
          <div class="flex justify-between">
            <span>Memory:</span>
            <span>{{ agent.memoryUsage }}MB</span>
          </div>
        </div>

        <!-- Current Activity -->
        <div v-if="agent.currentActivity" class="mt-3 pt-3 border-t border-gray-200 dark:border-gray-600">
          <p class="text-xs text-gray-500 dark:text-gray-400 mb-1">Current Activity:</p>
          <p class="text-xs font-medium text-gray-700 dark:text-gray-300">
            {{ agent.currentActivity }}
          </p>
        </div>

        <!-- Performance Indicator -->
        <div class="mt-3">
          <div class="flex justify-between text-xs text-gray-600 dark:text-gray-400 mb-1">
            <span>Performance</span>
            <span>{{ agent.performance }}%</span>
          </div>
          <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
            <div 
              class="h-1.5 rounded-full transition-all duration-300"
              :class="getPerformanceColor(agent.performance)"
              :style="{ width: `${agent.performance}%` }"
            ></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Summary Stats -->
    <div class="mt-6 pt-4 border-t border-gray-200 dark:border-gray-600">
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
        <div>
          <p class="text-2xl font-bold text-gray-900 dark:text-white">{{ activeAgents }}</p>
          <p class="text-xs text-gray-600 dark:text-gray-400">Active</p>
        </div>
        <div>
          <p class="text-2xl font-bold text-gray-900 dark:text-white">{{ totalTasks }}</p>
          <p class="text-xs text-gray-600 dark:text-gray-400">Tasks</p>
        </div>
        <div>
          <p class="text-2xl font-bold text-gray-900 dark:text-white">{{ averagePerformance }}%</p>
          <p class="text-xs text-gray-600 dark:text-gray-400">Avg Performance</p>
        </div>
        <div>
          <p class="text-2xl font-bold text-gray-900 dark:text-white">{{ totalMemory }}MB</p>
          <p class="text-xs text-gray-600 dark:text-gray-400">Total Memory</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { formatDistanceToNow } from 'date-fns'

interface Agent {
  id: string
  name: string
  status: 'active' | 'idle' | 'busy' | 'error'
  sessionId: string
  tasksCompleted: number
  totalTasks: number
  startTime: string
  memoryUsage: number
  performance: number
  currentActivity?: string
}

const agents = ref<Agent[]>([])

const activeAgents = computed(() => 
  agents.value.filter(a => a.status === 'active' || a.status === 'busy').length
)

const totalTasks = computed(() => 
  agents.value.reduce((sum, agent) => sum + agent.tasksCompleted, 0)
)

const averagePerformance = computed(() => {
  if (agents.value.length === 0) return 0
  const sum = agents.value.reduce((sum, agent) => sum + agent.performance, 0)
  return Math.round(sum / agents.value.length)
})

const totalMemory = computed(() => 
  agents.value.reduce((sum, agent) => sum + agent.memoryUsage, 0)
)

const getStatusColor = (status: string) => {
  const colors = {
    'active': 'bg-green-500',
    'busy': 'bg-blue-500',
    'idle': 'bg-yellow-500',
    'error': 'bg-red-500'
  }
  return colors[status as keyof typeof colors] || 'bg-gray-500'
}

const getStatusClass = (status: string) => {
  const classes = {
    'active': 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    'busy': 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    'idle': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    'error': 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
  }
  return classes[status as keyof typeof classes] || 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
}

const getPerformanceColor = (performance: number) => {
  if (performance >= 90) return 'bg-green-500'
  if (performance >= 70) return 'bg-blue-500'
  if (performance >= 50) return 'bg-yellow-500'
  return 'bg-red-500'
}

const formatUptime = (startTime: string) => {
  return formatDistanceToNow(new Date(startTime))
}

const generateMockAgents = (): Agent[] => {
  const statuses: Agent['status'][] = ['active', 'busy', 'idle']
  const activities = [
    'Processing file analysis',
    'Executing code review',
    'Generating documentation',
    'Running tests',
    'Optimizing performance',
    'Debugging issues'
  ]
  
  const agents: Agent[] = []
  
  for (let i = 1; i <= 8; i++) {
    const status = statuses[Math.floor(Math.random() * statuses.length)]
    const startTime = new Date(Date.now() - Math.random() * 86400000)
    
    agents.push({
      id: `agent_${i}`,
      name: `Alpha-${i}`,
      status,
      sessionId: `session_${Math.random().toString(36).substring(2, 10)}`,
      tasksCompleted: Math.floor(Math.random() * 50),
      totalTasks: Math.floor(Math.random() * 20) + 50,
      startTime: startTime.toISOString(),
      memoryUsage: Math.floor(Math.random() * 200) + 50,
      performance: Math.floor(Math.random() * 40) + 60,
      currentActivity: status === 'busy' ? activities[Math.floor(Math.random() * activities.length)] : undefined
    })
  }
  
  return agents
}

const refreshAgents = () => {
  agents.value = generateMockAgents()
}

onMounted(() => {
  refreshAgents()
  
  // Refresh agent data every 15 seconds
  setInterval(refreshAgents, 15000)
})
</script>