<template>
  <div 
    v-if="isVisible"
    class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
    @click.self="close"
  >
    <div 
      class="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] overflow-hidden"
      @click.stop
    >
      <!-- Header -->
      <div class="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
        <div>
          <h2 class="text-xl font-semibold text-gray-900 dark:text-gray-100">
            Select Agents
          </h2>
          <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Choose {{ requirements.min_agents }} - {{ requirements.max_agents }} agents for this task
          </p>
        </div>
        <button
          @click="close"
          class="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
          </svg>
        </button>
      </div>
      
      <!-- Filters -->
      <div class="p-6 border-b border-gray-200 dark:border-gray-700">
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Search Agents
            </label>
            <input
              v-model="searchQuery"
              type="text"
              placeholder="Search by name or ID..."
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Availability
            </label>
            <select
              v-model="availabilityFilter"
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="all">All Agents</option>
              <option value="available">Available Only</option>
              <option value="busy">Busy Only</option>
            </select>
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Sort By
            </label>
            <select
              v-model="sortBy"
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="match">Match Score</option>
              <option value="availability">Availability</option>
              <option value="performance">Performance</option>
              <option value="name">Name</option>
            </select>
          </div>
        </div>
      </div>
      
      <!-- Selected Agents Summary -->
      <div v-if="selectedAgents.length > 0" class="p-4 bg-blue-50 dark:bg-blue-900">
        <div class="flex items-center justify-between">
          <div class="flex items-center space-x-2">
            <svg class="w-5 h-5 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"></path>
            </svg>
            <span class="font-medium text-blue-900 dark:text-blue-100">
              {{ selectedAgents.length }} of {{ requirements.max_agents }} agents selected
            </span>
          </div>
          <button
            @click="selectedAgents = []"
            class="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 text-sm"
          >
            Clear All
          </button>
        </div>
        <div class="flex flex-wrap gap-2 mt-2">
          <div 
            v-for="agent in selectedAgents"
            :key="agent.agent_id"
            class="flex items-center space-x-2 bg-white dark:bg-gray-800 rounded-full px-3 py-1 text-sm"
          >
            <div 
              class="w-4 h-4 rounded-full"
              :style="{ backgroundColor: getAgentColor(agent.agent_id) }"
            ></div>
            <span>{{ agent.name || agent.agent_id }}</span>
            <button
              @click="deselectAgent(agent)"
              class="text-gray-400 hover:text-gray-600 ml-1"
            >
              <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
              </svg>
            </button>
          </div>
        </div>
      </div>
      
      <!-- Agent List -->
      <div class="p-6 overflow-y-auto max-h-[calc(90vh-20rem)]">
        <div v-if="filteredAndSortedAgents.length === 0" class="text-center py-12">
          <svg class="w-12 h-12 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"></path>
          </svg>
          <p class="text-gray-500 dark:text-gray-400">No agents match your criteria</p>
        </div>
        
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div 
            v-for="agent in filteredAndSortedAgents"
            :key="agent.agent_id"
            class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 border-2 transition-all duration-200 cursor-pointer"
            :class="{
              'border-primary-500 bg-primary-50 dark:bg-primary-900': isSelected(agent),
              'border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500': !isSelected(agent),
              'opacity-50': !canSelectAgent(agent)
            }"
            @click="toggleAgent(agent)"
          >
            <div class="flex items-center justify-between mb-3">
              <div class="flex items-center space-x-3">
                <div 
                  class="w-8 h-8 rounded-full flex items-center justify-center text-white text-xs font-bold"
                  :style="{ backgroundColor: getAgentColor(agent.agent_id) }"
                >
                  {{ agent.agent_id.substring(0, 2).toUpperCase() }}
                </div>
                <div>
                  <div class="font-medium text-gray-900 dark:text-gray-100">
                    {{ agent.name || agent.agent_id }}
                  </div>
                  <div class="text-xs text-gray-500 dark:text-gray-400">
                    {{ agent.type || 'Agent' }}
                  </div>
                </div>
              </div>
              <div class="text-right">
                <div 
                  class="px-2 py-1 rounded text-xs font-medium mb-1"
                  :class="{
                    'bg-green-100 text-green-800': agent.is_available,
                    'bg-gray-100 text-gray-800': !agent.is_available
                  }"
                >
                  {{ agent.is_available ? 'Available' : 'Busy' }}
                </div>
                <div class="text-xs text-gray-500">{{ getMatchScore(agent) }}% match</div>
              </div>
            </div>
            
            <!-- Quick stats -->
            <div class="grid grid-cols-3 gap-2 text-center text-xs">
              <div>
                <div class="font-medium text-gray-900 dark:text-gray-100">
                  {{ agent.performance?.success_rate || 0 }}%
                </div>
                <div class="text-gray-500">Success</div>
              </div>
              <div>
                <div class="font-medium text-gray-900 dark:text-gray-100">
                  {{ agent.performance?.avg_response_time || 0 }}ms
                </div>
                <div class="text-gray-500">Response</div>
              </div>
              <div>
                <div class="font-medium text-gray-900 dark:text-gray-100">
                  {{ agent.performance?.load || 0 }}%
                </div>
                <div class="text-gray-500">Load</div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Footer -->
      <div class="flex items-center justify-between p-6 border-t border-gray-200 dark:border-gray-700">
        <div class="text-sm text-gray-500 dark:text-gray-400">
          {{ selectedAgents.length }} / {{ requirements.max_agents }} agents selected
          <span v-if="selectedAgents.length < requirements.min_agents" class="text-red-500">
            (minimum {{ requirements.min_agents }} required)
          </span>
        </div>
        <div class="flex items-center space-x-3">
          <button
            @click="close"
            type="button"
            class="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          >
            Cancel
          </button>
          <button
            @click="confirmSelection"
            type="button"
            :disabled="!canConfirm"
            class="px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-300 disabled:text-gray-500 text-white rounded-md transition-colors"
          >
            Select Agents
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

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

interface Requirements {
  min_agents: number
  max_agents: number
  required_capabilities: Array<{ name: string; level: string }>
  preferred_capabilities: Array<{ name: string; level: string; weight?: number }>
}

interface Props {
  isVisible: boolean
  agents: Agent[]
  requirements: Requirements
  preselectedAgents?: Agent[]
}

const props = withDefaults(defineProps<Props>(), {
  preselectedAgents: () => []
})

const emit = defineEmits<{
  close: []
  select: [agents: Agent[]]
}>()

// Reactive state
const selectedAgents = ref<Agent[]>([...props.preselectedAgents])
const searchQuery = ref('')
const availabilityFilter = ref('all')
const sortBy = ref('match')

// Computed
const filteredAndSortedAgents = computed(() => {
  let filtered = props.agents

  // Filter by search query
  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    filtered = filtered.filter(agent => 
      agent.name?.toLowerCase().includes(query) ||
      agent.agent_id.toLowerCase().includes(query) ||
      agent.type?.toLowerCase().includes(query)
    )
  }

  // Filter by availability
  if (availabilityFilter.value !== 'all') {
    filtered = filtered.filter(agent => 
      availabilityFilter.value === 'available' ? agent.is_available : !agent.is_available
    )
  }

  // Sort agents
  return filtered.sort((a, b) => {
    switch (sortBy.value) {
      case 'match':
        return getMatchScore(b) - getMatchScore(a)
      case 'availability':
        return (b.is_available ? 1 : 0) - (a.is_available ? 1 : 0)
      case 'performance':
        return (b.performance?.success_rate || 0) - (a.performance?.success_rate || 0)
      case 'name':
        return (a.name || a.agent_id).localeCompare(b.name || b.agent_id)
      default:
        return 0
    }
  })
})

const canConfirm = computed(() => {
  return selectedAgents.value.length >= props.requirements.min_agents &&
         selectedAgents.value.length <= props.requirements.max_agents
})

// Methods
function getMatchScore(agent: Agent): number {
  const requiredCapabilities = props.requirements.required_capabilities.map(c => c.name)
  const agentCapabilities = agent.capabilities || []
  
  if (requiredCapabilities.length === 0) return 100
  
  const matches = requiredCapabilities.filter(req => 
    agentCapabilities.some(cap => cap.toLowerCase().includes(req.toLowerCase()))
  ).length
  
  return Math.round((matches / requiredCapabilities.length) * 100)
}

function getAgentColor(agentId: string): string {
  const colors = [
    '#3B82F6', '#10B981', '#F59E0B', '#EF4444', 
    '#8B5CF6', '#EC4899', '#06B6D4', '#84CC16'
  ]
  return colors[agentId.length % colors.length]
}

function isSelected(agent: Agent): boolean {
  return selectedAgents.value.some(selected => selected.agent_id === agent.agent_id)
}

function canSelectAgent(agent: Agent): boolean {
  return isSelected(agent) || 
         (selectedAgents.value.length < props.requirements.max_agents && agent.is_available)
}

function toggleAgent(agent: Agent) {
  if (!canSelectAgent(agent) && !isSelected(agent)) return
  
  if (isSelected(agent)) {
    deselectAgent(agent)
  } else {
    selectAgent(agent)
  }
}

function selectAgent(agent: Agent) {
  if (selectedAgents.value.length < props.requirements.max_agents) {
    selectedAgents.value.push(agent)
  }
}

function deselectAgent(agent: Agent) {
  const index = selectedAgents.value.findIndex(selected => selected.agent_id === agent.agent_id)
  if (index > -1) {
    selectedAgents.value.splice(index, 1)
  }
}

function confirmSelection() {
  if (canConfirm.value) {
    emit('select', selectedAgents.value)
    close()
  }
}

function close() {
  emit('close')
}
</script>

<style scoped>
/* Modal styles */
</style>