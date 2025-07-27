<template>
  <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6">
    <div class="flex items-center justify-between mb-4">
      <h3 class="text-lg font-semibold text-gray-900 dark:text-white">System Components</h3>
      <div class="flex items-center space-x-2">
        <div class="w-2 h-2 rounded-full bg-green-500"></div>
        <span class="text-sm text-gray-600 dark:text-gray-400">{{ healthyCount }}/{{ totalComponents }} healthy</span>
      </div>
    </div>

    <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
      <div 
        v-for="(component, name) in components" 
        :key="name"
        class="flex items-center justify-between p-3 border border-gray-200 dark:border-gray-600 rounded-lg"
      >
        <div class="flex items-center space-x-3">
          <div 
            class="w-3 h-3 rounded-full"
            :class="getStatusColor(component.status)"
          ></div>
          <div>
            <p class="text-sm font-medium text-gray-900 dark:text-gray-100 capitalize">
              {{ formatComponentName(name) }}
            </p>
            <p class="text-xs text-gray-500 dark:text-gray-400">
              {{ component.message || 'No additional info' }}
            </p>
          </div>
        </div>
        <span 
          class="inline-flex px-2 py-1 text-xs font-semibold rounded-full"
          :class="getStatusClass(component.status)"
        >
          {{ component.status }}
        </span>
      </div>
    </div>

    <!-- Component Health Summary -->
    <div class="mt-4 pt-4 border-t border-gray-200 dark:border-gray-600">
      <div class="flex justify-between text-sm">
        <span class="text-gray-600 dark:text-gray-400">Overall Health</span>
        <span 
          class="font-medium"
          :class="overallHealthClass"
        >
          {{ overallHealthStatus }}
        </span>
      </div>
      <div class="mt-2 w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
        <div 
          class="h-2 rounded-full transition-all duration-300"
          :class="overallHealthColor"
          :style="{ width: `${healthPercentage}%` }"
        ></div>
      </div>
    </div>

    <!-- Refresh Button -->
    <div class="mt-4 flex justify-end">
      <button 
        @click="refreshComponents"
        :disabled="loading"
        class="text-sm text-blue-600 hover:text-blue-500 dark:text-blue-400 disabled:opacity-50"
      >
        <svg v-if="loading" class="animate-spin -ml-1 mr-2 h-4 w-4 inline" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        {{ loading ? 'Refreshing...' : 'Refresh Status' }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'

interface Component {
  status: string
  message?: string
  error?: string
  lastCheck?: string
  uptime?: number
}

const components = ref<Record<string, Component>>({})
const loading = ref(false)

const healthyCount = computed(() => 
  Object.values(components.value).filter(c => c.status === 'healthy').length
)

const totalComponents = computed(() => 
  Object.keys(components.value).length
)

const healthPercentage = computed(() => 
  totalComponents.value > 0 ? Math.round((healthyCount.value / totalComponents.value) * 100) : 0
)

const overallHealthStatus = computed(() => {
  if (healthPercentage.value >= 90) return 'Excellent'
  if (healthPercentage.value >= 70) return 'Good'
  if (healthPercentage.value >= 50) return 'Fair'
  return 'Poor'
})

const overallHealthClass = computed(() => {
  if (healthPercentage.value >= 90) return 'text-green-600 dark:text-green-400'
  if (healthPercentage.value >= 70) return 'text-blue-600 dark:text-blue-400'
  if (healthPercentage.value >= 50) return 'text-yellow-600 dark:text-yellow-400'
  return 'text-red-600 dark:text-red-400'
})

const overallHealthColor = computed(() => {
  if (healthPercentage.value >= 90) return 'bg-green-500'
  if (healthPercentage.value >= 70) return 'bg-blue-500'
  if (healthPercentage.value >= 50) return 'bg-yellow-500'
  return 'bg-red-500'
})

const getStatusColor = (status: string) => {
  const colors = {
    'healthy': 'bg-green-500',
    'degraded': 'bg-yellow-500',
    'unhealthy': 'bg-red-500',
    'unknown': 'bg-gray-500'
  }
  return colors[status as keyof typeof colors] || 'bg-gray-500'
}

const getStatusClass = (status: string) => {
  const classes = {
    'healthy': 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    'degraded': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    'unhealthy': 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    'unknown': 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
  }
  return classes[status as keyof typeof classes] || 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
}

const formatComponentName = (name: string) => {
  return name.replace(/[_-]/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
}

const refreshComponents = async () => {
  loading.value = true
  
  try {
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    // Mock component data
    components.value = {
      database: {
        status: 'healthy',
        message: 'All connections active',
        lastCheck: new Date().toISOString(),
        uptime: 99.8
      },
      redis: {
        status: 'healthy',
        message: 'Cache performing well',
        lastCheck: new Date().toISOString(),
        uptime: 99.9
      },
      orchestrator: {
        status: 'healthy',
        message: 'Processing requests',
        lastCheck: new Date().toISOString(),
        uptime: 98.5
      },
      event_processor: {
        status: 'healthy',
        message: 'Queue processing normally',
        lastCheck: new Date().toISOString(),
        uptime: 99.2
      },
      hook_interceptor: {
        status: Math.random() > 0.8 ? 'degraded' : 'healthy',
        message: Math.random() > 0.8 ? 'Some hooks failing' : 'All hooks operational',
        lastCheck: new Date().toISOString(),
        uptime: 97.1
      },
      websocket_manager: {
        status: 'healthy',
        message: `${Math.floor(Math.random() * 100)} active connections`,
        lastCheck: new Date().toISOString(),
        uptime: 99.5
      }
    }
  } catch (error) {
    console.error('Failed to refresh components:', error)
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  refreshComponents()
  
  // Auto-refresh every 30 seconds
  setInterval(refreshComponents, 30000)
})
</script>