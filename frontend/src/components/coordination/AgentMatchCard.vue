<template>
  <div 
    class="agent-match-card bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 border border-gray-200 dark:border-gray-700 hover:shadow-lg transition-all duration-200"
    :class="{
      'ring-2 ring-primary-500': isSelected,
      'opacity-75': !agent.is_available
    }"
  >
    <div class="flex items-start justify-between mb-3">
      <div class="flex items-center space-x-3">
        <div 
          class="w-10 h-10 rounded-full flex items-center justify-center text-white font-semibold"
          :style="{ backgroundColor: agentColor }"
        >
          {{ agent.agent_id.substring(0, 2).toUpperCase() }}
        </div>
        <div>
          <h3 class="font-semibold text-gray-900 dark:text-gray-100">
            {{ agent.name || agent.agent_id }}
          </h3>
          <p class="text-sm text-gray-500 dark:text-gray-400">
            {{ agent.type || 'Agent' }}
          </p>
        </div>
      </div>
      <div class="flex flex-col items-end space-y-1">
        <div 
          class="px-2 py-1 rounded-full text-xs font-medium"
          :class="{
            'bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-100': agent.is_available,
            'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-100': !agent.is_available
          }"
        >
          {{ agent.is_available ? 'Available' : 'Busy' }}
        </div>
        <div class="text-sm text-gray-500 dark:text-gray-400">
          {{ matchScore }}% match
        </div>
      </div>
    </div>
    
    <!-- Capabilities -->
    <div v-if="agent.capabilities && agent.capabilities.length > 0" class="mb-3">
      <h4 class="text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">CAPABILITIES</h4>
      <div class="flex flex-wrap gap-1">
        <span 
          v-for="capability in agent.capabilities.slice(0, 4)" 
          :key="capability"
          class="px-2 py-1 bg-blue-100 text-blue-800 dark:bg-blue-800 dark:text-blue-100 rounded text-xs"
        >
          {{ capability }}
        </span>
        <span 
          v-if="agent.capabilities.length > 4"
          class="px-2 py-1 bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300 rounded text-xs"
        >
          +{{ agent.capabilities.length - 4 }} more
        </span>
      </div>
    </div>
    
    <!-- Performance Metrics -->
    <div class="grid grid-cols-3 gap-2 mb-3 text-center">
      <div class="bg-gray-50 dark:bg-gray-700 rounded p-2">
        <div class="text-sm font-medium text-gray-900 dark:text-gray-100">
          {{ agent.performance?.success_rate || 0 }}%
        </div>
        <div class="text-xs text-gray-500 dark:text-gray-400">Success</div>
      </div>
      <div class="bg-gray-50 dark:bg-gray-700 rounded p-2">
        <div class="text-sm font-medium text-gray-900 dark:text-gray-100">
          {{ agent.performance?.avg_response_time || 0 }}ms
        </div>
        <div class="text-xs text-gray-500 dark:text-gray-400">Response</div>
      </div>
      <div class="bg-gray-50 dark:bg-gray-700 rounded p-2">
        <div class="text-sm font-medium text-gray-900 dark:text-gray-100">
          {{ agent.performance?.load || 0 }}%
        </div>
        <div class="text-xs text-gray-500 dark:text-gray-400">Load</div>
      </div>
    </div>
    
    <!-- Action Buttons -->
    <div class="flex space-x-2">
      <button
        @click="$emit('select', agent)"
        class="flex-1 bg-primary-600 hover:bg-primary-700 text-white py-2 px-3 rounded text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        :disabled="!agent.is_available"
      >
        Select
      </button>
      <button
        @click="$emit('view-details', agent)"
        class="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
      >
        Details
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Agent {
  agent_id: string
  name?: string
  type?: string
  is_available: boolean
  capabilities: string[]
  performance?: {
    success_rate: number
    avg_response_time: number
    load: number
  }
}

interface Props {
  agent: Agent
  matchScore: number
  isSelected?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  isSelected: false
})

const emit = defineEmits<{
  select: [agent: Agent]
  'view-details': [agent: Agent]
}>()

const agentColor = computed(() => {
  // Generate a color based on agent_id
  const colors = [
    '#3B82F6', '#10B981', '#F59E0B', '#EF4444', 
    '#8B5CF6', '#EC4899', '#06B6D4', '#84CC16'
  ]
  const index = props.agent.agent_id.length % colors.length
  return colors[index]
})
</script>

<style scoped>
.agent-match-card {
  transition: all 0.2s ease-in-out;
}
</style>