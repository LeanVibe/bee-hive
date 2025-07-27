<template>
  <div class="glass-card rounded-xl p-6">
    <div class="flex items-center justify-between mb-6">
      <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
        System Health
      </h3>
      <div class="flex items-center space-x-2">
        <div
          :class="[
            'status-dot',
            healthStatus === 'healthy' ? 'status-healthy' :
            healthStatus === 'degraded' ? 'status-warning' : 'status-danger'
          ]"
        ></div>
        <span class="text-sm font-medium capitalize">
          {{ healthStatus }}
        </span>
      </div>
    </div>
    
    <!-- Overall Health Bar -->
    <div class="mb-6">
      <div class="flex items-center justify-between mb-2">
        <span class="text-sm font-medium text-slate-700 dark:text-slate-300">
          Overall Health
        </span>
        <span class="text-sm text-slate-500 dark:text-slate-400">
          {{ healthPercentage }}%
        </span>
      </div>
      <div class="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
        <div
          class="h-2 rounded-full transition-all duration-300"
          :class="healthBarColor"
          :style="{ width: `${healthPercentage}%` }"
        ></div>
      </div>
    </div>
    
    <!-- Component Status Grid -->
    <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
      <div
        v-for="(component, name) in componentHealth"
        :key="name"
        class="p-4 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
      >
        <div class="flex items-center justify-between">
          <div class="flex items-center space-x-3">
            <component
              :is="getComponentIcon(name)"
              class="w-5 h-5"
              :class="getComponentIconColor(component.status)"
            />
            <div>
              <div class="font-medium text-slate-900 dark:text-white capitalize">
                {{ formatComponentName(name) }}
              </div>
              <div class="text-xs text-slate-500 dark:text-slate-400">
                {{ component.message || getStatusMessage(component.status) }}
              </div>
            </div>
          </div>
          <div
            :class="[
              'status-dot',
              component.status === 'healthy' ? 'status-healthy' :
              component.status === 'degraded' ? 'status-warning' : 'status-danger'
            ]"
          ></div>
        </div>
        
        <!-- Additional component info -->
        <div v-if="hasAdditionalInfo(component)" class="mt-3 space-y-1">
          <div
            v-if="component.events_processed"
            class="text-xs text-slate-600 dark:text-slate-400"
          >
            Events: {{ formatNumber(component.events_processed) }}
          </div>
          <div
            v-if="component.processing_rate"
            class="text-xs text-slate-600 dark:text-slate-400"
          >
            Rate: {{ component.processing_rate.toFixed(1) }}/s
          </div>
          <div
            v-if="component.memory_used"
            class="text-xs text-slate-600 dark:text-slate-400"
          >
            Memory: {{ component.memory_used }}
          </div>
          <div
            v-if="component.tables !== undefined"
            class="text-xs text-slate-600 dark:text-slate-400"
          >
            Tables: {{ component.tables }}
          </div>
        </div>
      </div>
    </div>
    
    <!-- Error Details -->
    <div v-if="hasErrors" class="mt-6 p-4 bg-danger-50 dark:bg-danger-900/20 rounded-lg border border-danger-200 dark:border-danger-800">
      <div class="flex items-center space-x-2 mb-2">
        <ExclamationTriangleIcon class="w-4 h-4 text-danger-600 dark:text-danger-400" />
        <span class="text-sm font-medium text-danger-800 dark:text-danger-200">
          System Issues Detected
        </span>
      </div>
      <div class="space-y-1">
        <div
          v-for="(component, name) in errorComponents"
          :key="name"
          class="text-xs text-danger-700 dark:text-danger-300"
        >
          <strong>{{ formatComponentName(name) }}:</strong> {{ component.error || component.message }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useMetricsStore } from '@/stores/metrics'
import {
  ServerIcon,
  CircleStackIcon,
  CpuChipIcon,
  EyeIcon,
  Cog6ToothIcon,
  ExclamationTriangleIcon,
} from '@heroicons/vue/24/outline'

const metricsStore = useMetricsStore()

// Computed properties
const healthStatus = computed(() => metricsStore.overallHealth)
const componentHealth = computed(() => metricsStore.componentHealth)
const healthPercentage = computed(() => metricsStore.healthPercentage)

const healthBarColor = computed(() => {
  switch (healthStatus.value) {
    case 'healthy': return 'bg-success-500'
    case 'degraded': return 'bg-warning-500'
    case 'unhealthy': return 'bg-danger-500'
    default: return 'bg-slate-300'
  }
})

const hasErrors = computed(() => {
  return Object.values(componentHealth.value).some(c => 
    c.status === 'unhealthy' || c.error
  )
})

const errorComponents = computed(() => {
  const errors: Record<string, any> = {}
  Object.entries(componentHealth.value).forEach(([name, component]) => {
    if (component.status === 'unhealthy' || component.error) {
      errors[name] = component
    }
  })
  return errors
})

// Helper functions
const getComponentIcon = (name: string) => {
  switch (name.toLowerCase()) {
    case 'database':
    case 'postgres':
      return CircleStackIcon
    case 'redis':
      return ServerIcon
    case 'orchestrator':
    case 'agent_orchestrator':
      return CpuChipIcon
    case 'observability':
    case 'event_processor':
      return EyeIcon
    default:
      return Cog6ToothIcon
  }
}

const getComponentIconColor = (status: string) => {
  switch (status) {
    case 'healthy':
      return 'text-success-600 dark:text-success-400'
    case 'degraded':
      return 'text-warning-600 dark:text-warning-400'
    case 'unhealthy':
      return 'text-danger-600 dark:text-danger-400'
    default:
      return 'text-slate-400 dark:text-slate-500'
  }
}

const formatComponentName = (name: string) => {
  return name
    .replace(/_/g, ' ')
    .replace(/([A-Z])/g, ' $1')
    .toLowerCase()
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

const getStatusMessage = (status: string) => {
  switch (status) {
    case 'healthy': return 'Operating normally'
    case 'degraded': return 'Performance issues detected'
    case 'unhealthy': return 'Service unavailable'
    default: return 'Status unknown'
  }
}

const hasAdditionalInfo = (component: any) => {
  return component.events_processed !== undefined ||
         component.processing_rate !== undefined ||
         component.memory_used !== undefined ||
         component.tables !== undefined
}

const formatNumber = (num: number) => {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M'
  }
  if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'K'
  }
  return num.toString()
}
</script>