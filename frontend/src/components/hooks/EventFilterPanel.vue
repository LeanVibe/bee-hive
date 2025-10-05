<template>
  <div class="event-filter-panel bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
    <!-- Header -->
    <div class="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
      <div class="flex items-center space-x-3">
        <FunnelIcon class="w-5 h-5 text-gray-500 dark:text-gray-400" />
        <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
          Event Filters
        </h3>
        <span 
          v-if="activeFiltersCount > 0"
          class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
        >
          {{ activeFiltersCount }} active
        </span>
      </div>
      
      <div class="flex items-center space-x-2">
        <!-- Preset filters dropdown -->
        <select
          v-model="selectedPreset"
          @change="applyPreset"
          class="text-sm border border-gray-300 dark:border-gray-600 rounded-md px-3 py-1 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
        >
          <option value="">Custom Filter...</option>
          <option value="errors_only">Errors Only</option>
          <option value="security_blocked">Security Blocked</option>
          <option value="high_priority">High Priority</option>
          <option value="tool_usage">Tool Usage</option>
          <option value="last_hour">Last Hour</option>
        </select>

        <!-- Clear filters button -->
        <button
          @click="clearAllFilters"
          :disabled="activeFiltersCount === 0"
          class="px-3 py-1 text-sm bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Clear All
        </button>

        <!-- Collapse/expand toggle -->
        <button
          @click="isExpanded = !isExpanded"
          class="p-1.5 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition-colors"
        >
          <ChevronUpIcon v-if="isExpanded" class="w-5 h-5" />
          <ChevronDownIcon v-else class="w-5 h-5" />
        </button>
      </div>
    </div>

    <!-- Filter content -->
    <div v-show="isExpanded" class="p-4 space-y-6">
      <!-- Search query -->
      <div class="filter-group">
        <label class="filter-label">
          Search Events
        </label>
        <div class="relative">
          <MagnifyingGlassIcon class="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            v-model="localFilters.search_query"
            type="text"
            placeholder="Search in payloads, agent IDs, session IDs..."
            class="filter-input pl-10"
            @input="debouncedFilterChange"
          />
          <button
            v-if="localFilters.search_query"
            @click="localFilters.search_query = ''; emitFilterChange()"
            class="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
          >
            <XMarkIcon class="w-4 h-4" />
          </button>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <!-- Agent selection -->
        <div class="filter-group">
          <label class="filter-label">
            Agents
            <span v-if="localFilters.agent_ids?.length" class="filter-count">
              ({{ localFilters.agent_ids.length }} selected)
            </span>
          </label>
          <div class="space-y-2 max-h-32 overflow-y-auto">
            <label
              v-for="agent in availableAgents"
              :key="agent.agent_id"
              class="filter-checkbox-label"
            >
              <input
                type="checkbox"
                :value="agent.agent_id"
                :checked="localFilters.agent_ids?.includes(agent.agent_id)"
                @change="toggleAgentFilter(agent.agent_id)"
                class="filter-checkbox"
              />
              <div class="flex items-center justify-between flex-1 min-w-0">
                <span class="filter-checkbox-text">
                  {{ agent.agent_id.substring(0, 8) }}...
                </span>
                <div class="flex items-center space-x-2">
                  <span 
                    class="w-2 h-2 rounded-full"
                    :class="getAgentStatusColor(agent.status)"
                  ></span>
                  <span class="text-xs text-gray-500 dark:text-gray-400">
                    {{ agent.event_count }}
                  </span>
                </div>
              </div>
            </label>
          </div>
          <button
            v-if="availableAgents.length > 5"
            @click="showAllAgents = !showAllAgents"
            class="text-xs text-blue-600 dark:text-blue-400 hover:underline mt-2"
          >
            {{ showAllAgents ? 'Show less' : `Show all (${availableAgents.length})` }}
          </button>
        </div>

        <!-- Session selection -->
        <div class="filter-group">
          <label class="filter-label">
            Sessions
            <span v-if="localFilters.session_ids?.length" class="filter-count">
              ({{ localFilters.session_ids.length }} selected)
            </span>
          </label>
          <div class="space-y-2 max-h-32 overflow-y-auto">
            <label
              v-for="session in availableSessions"
              :key="session.session_id"
              class="filter-checkbox-label"
            >
              <input
                type="checkbox"
                :value="session.session_id"
                :checked="localFilters.session_ids?.includes(session.session_id)"
                @change="toggleSessionFilter(session.session_id)"
                class="filter-checkbox"
              />
              <div class="flex items-center justify-between flex-1 min-w-0">
                <span class="filter-checkbox-text">
                  {{ session.session_id.substring(0, 8) }}...
                </span>
                <div class="flex items-center space-x-2">
                  <span 
                    class="w-2 h-2 rounded-full"
                    :class="getSessionStatusColor(session.status)"
                  ></span>
                  <span class="text-xs text-gray-500 dark:text-gray-400">
                    {{ session.event_count }}
                  </span>
                </div>
              </div>
            </label>
          </div>
        </div>

        <!-- Hook type selection -->
        <div class="filter-group">
          <label class="filter-label">
            Event Types
            <span v-if="localFilters.hook_types?.length" class="filter-count">
              ({{ localFilters.hook_types.length }} selected)
            </span>
          </label>
          <div class="space-y-2">
            <label
              v-for="hookType in Object.values(HookType)"
              :key="hookType"
              class="filter-checkbox-label"
            >
              <input
                type="checkbox"
                :value="hookType"
                :checked="localFilters.hook_types?.includes(hookType)"
                @change="toggleHookTypeFilter(hookType)"
                class="filter-checkbox"
              />
              <div class="flex items-center justify-between flex-1">
                <span class="filter-checkbox-text">
                  {{ formatHookType(hookType) }}
                </span>
                <span 
                  class="inline-flex px-2 py-0.5 text-xs font-semibold rounded-full"
                  :class="getHookTypeBadgeClass(hookType)"
                >
                  {{ getHookTypeEventCount(hookType) }}
                </span>
              </div>
            </label>
          </div>
        </div>
      </div>

      <!-- Time range and advanced filters -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <!-- Time range -->
        <div class="filter-group">
          <label class="filter-label">From Time</label>
          <input
            v-model="localFilters.from_time"
            type="datetime-local"
            class="filter-input"
            @change="emitFilterChange"
          />
        </div>

        <div class="filter-group">
          <label class="filter-label">To Time</label>
          <input
            v-model="localFilters.to_time"
            type="datetime-local"
            class="filter-input"
            @change="emitFilterChange"
          />
        </div>

        <!-- Priority filter -->
        <div class="filter-group">
          <label class="filter-label">
            Min Priority ({{ localFilters.min_priority || 1 }})
          </label>
          <input
            v-model.number="localFilters.min_priority"
            type="range"
            min="1"
            max="10"
            class="filter-range"
            @input="emitFilterChange"
          />
          <div class="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
            <span>High (1)</span>
            <span>Low (10)</span>
          </div>
        </div>

        <!-- Risk level filter -->
        <div class="filter-group">
          <label class="filter-label">
            Risk Levels
            <span v-if="localFilters.risk_levels?.length" class="filter-count">
              ({{ localFilters.risk_levels.length }} selected)
            </span>
          </label>
          <div class="space-y-1">
            <label
              v-for="riskLevel in Object.values(SecurityRisk)"
              :key="riskLevel"
              class="filter-checkbox-label text-sm"
            >
              <input
                type="checkbox"
                :value="riskLevel"
                :checked="localFilters.risk_levels?.includes(riskLevel)"
                @change="toggleRiskLevelFilter(riskLevel)"
                class="filter-checkbox"
              />
              <span 
                class="inline-flex px-2 py-0.5 text-xs font-semibold rounded-full"
                :class="getRiskLevelBadgeClass(riskLevel)"
              >
                {{ riskLevel }}
              </span>
            </label>
          </div>
        </div>
      </div>

      <!-- Special filters -->
      <div class="flex flex-wrap gap-4">
        <label class="filter-checkbox-label">
          <input
            v-model="localFilters.only_errors"
            type="checkbox"
            class="filter-checkbox"
            @change="emitFilterChange"
          />
          <span class="filter-checkbox-text">Errors Only</span>
        </label>

        <label class="filter-checkbox-label">
          <input
            v-model="localFilters.only_blocked"
            type="checkbox"
            class="filter-checkbox"
            @change="emitFilterChange"
          />
          <span class="filter-checkbox-text">Security Blocked Only</span>
        </label>
      </div>

      <!-- Filter summary -->
      <div v-if="activeFiltersCount > 0" class="filter-summary">
        <div class="flex items-center justify-between">
          <span class="text-sm text-gray-600 dark:text-gray-400">
            {{ activeFiltersCount }} filter{{ activeFiltersCount !== 1 ? 's' : '' }} active
          </span>
          <button
            @click="saveAsPreset"
            class="text-sm text-blue-600 dark:text-blue-400 hover:underline"
          >
            Save as preset
          </button>
        </div>
        
        <div class="flex flex-wrap gap-2 mt-3">
          <span
            v-for="filterSummary in getActiveFiltersSummary()"
            :key="filterSummary.key"
            class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
          >
            {{ filterSummary.label }}
            <button
              @click="removeFilter(filterSummary.key)"
              class="ml-1.5 inline-flex items-center justify-center w-4 h-4 rounded-full text-blue-400 hover:bg-blue-200 dark:hover:bg-blue-800"
            >
              <XMarkIcon class="w-3 h-3" />
            </button>
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useEventsStore } from '@/stores/events'
import { 
  HookType, 
  SecurityRisk,
  type EventFilter,
  type EventFilterPanelProps,
  type AgentInfo,
  type SessionInfo
} from '@/types/hooks'
import {
  FunnelIcon,
  MagnifyingGlassIcon,
  XMarkIcon,
  ChevronUpIcon,
  ChevronDownIcon
} from '@heroicons/vue/24/outline'

// Props
interface Props extends EventFilterPanelProps {}

const props = defineProps<Props>()

// Emits
const emit = defineEmits<{
  filtersChange: [filters: EventFilter]
  clearFilters: []
}>()

// Store
const eventsStore = useEventsStore()

// Local state
const isExpanded = ref(true)
const showAllAgents = ref(false)
const localFilters = ref<EventFilter>({ ...props.filters })
const selectedPreset = ref('')

// Debounce timer for search input
let searchDebounceTimer: number | null = null

// Computed
const { hookEvents, agents, sessions } = eventsStore

const availableAgents = computed(() => {
  return props.availableAgents || agents.slice(0, showAllAgents.value ? agents.length : 5)
})

const availableSessions = computed(() => {
  return props.availableSessions || sessions.slice(0, 10)
})

const activeFiltersCount = computed(() => {
  let count = 0
  
  if (localFilters.value.search_query) count++
  if (localFilters.value.agent_ids?.length) count++
  if (localFilters.value.session_ids?.length) count++
  if (localFilters.value.hook_types?.length) count++
  if (localFilters.value.from_time) count++
  if (localFilters.value.to_time) count++
  if (localFilters.value.min_priority && localFilters.value.min_priority > 1) count++
  if (localFilters.value.risk_levels?.length) count++
  if (localFilters.value.only_errors) count++
  if (localFilters.value.only_blocked) count++
  
  return count
})

// Methods
const formatHookType = (hookType: HookType) => {
  const typeMap = {
    [HookType.PRE_TOOL_USE]: 'Pre Tool Use',
    [HookType.POST_TOOL_USE]: 'Post Tool Use',
    [HookType.STOP]: 'Stop',
    [HookType.NOTIFICATION]: 'Notification',
    [HookType.AGENT_START]: 'Agent Start',
    [HookType.AGENT_STOP]: 'Agent Stop',
    [HookType.ERROR]: 'Error',
    [HookType.SUBAGENT_STOP]: 'Subagent Stop'
  }
  return typeMap[hookType] || hookType
}

const getHookTypeEventCount = (hookType: HookType) => {
  return hookEvents.filter(event => event.hook_type === hookType).length
}

const getHookTypeBadgeClass = (hookType: HookType) => {
  const classMap = {
    [HookType.PRE_TOOL_USE]: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    [HookType.POST_TOOL_USE]: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    [HookType.STOP]: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    [HookType.NOTIFICATION]: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    [HookType.AGENT_START]: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
    [HookType.AGENT_STOP]: 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200',
    [HookType.ERROR]: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    [HookType.SUBAGENT_STOP]: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200'
  }
  return classMap[hookType] || 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
}

const getRiskLevelBadgeClass = (riskLevel: SecurityRisk) => {
  const classMap = {
    [SecurityRisk.CRITICAL]: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    [SecurityRisk.HIGH]: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
    [SecurityRisk.MEDIUM]: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    [SecurityRisk.LOW]: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    [SecurityRisk.SAFE]: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
  }
  return classMap[riskLevel] || 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
}

const getAgentStatusColor = (status: string) => {
  const colorMap = {
    active: 'bg-green-500',
    idle: 'bg-yellow-500',
    error: 'bg-red-500',
    blocked: 'bg-orange-500'
  }
  return colorMap[status as keyof typeof colorMap] || 'bg-gray-500'
}

const getSessionStatusColor = (status: string) => {
  const colorMap = {
    active: 'bg-green-500',
    completed: 'bg-blue-500',
    error: 'bg-red-500',
    terminated: 'bg-orange-500'
  }
  return colorMap[status as keyof typeof colorMap] || 'bg-gray-500'
}

const toggleAgentFilter = (agentId: string) => {
  if (!localFilters.value.agent_ids) {
    localFilters.value.agent_ids = []
  }
  
  const index = localFilters.value.agent_ids.indexOf(agentId)
  if (index > -1) {
    localFilters.value.agent_ids.splice(index, 1)
  } else {
    localFilters.value.agent_ids.push(agentId)
  }
  
  emitFilterChange()
}

const toggleSessionFilter = (sessionId: string) => {
  if (!localFilters.value.session_ids) {
    localFilters.value.session_ids = []
  }
  
  const index = localFilters.value.session_ids.indexOf(sessionId)
  if (index > -1) {
    localFilters.value.session_ids.splice(index, 1)
  } else {
    localFilters.value.session_ids.push(sessionId)
  }
  
  emitFilterChange()
}

const toggleHookTypeFilter = (hookType: HookType) => {
  if (!localFilters.value.hook_types) {
    localFilters.value.hook_types = []
  }
  
  const index = localFilters.value.hook_types.indexOf(hookType)
  if (index > -1) {
    localFilters.value.hook_types.splice(index, 1)
  } else {
    localFilters.value.hook_types.push(hookType)
  }
  
  emitFilterChange()
}

const toggleRiskLevelFilter = (riskLevel: SecurityRisk) => {
  if (!localFilters.value.risk_levels) {
    localFilters.value.risk_levels = []
  }
  
  const index = localFilters.value.risk_levels.indexOf(riskLevel)
  if (index > -1) {
    localFilters.value.risk_levels.splice(index, 1)
  } else {
    localFilters.value.risk_levels.push(riskLevel)
  }
  
  emitFilterChange()
}

const emitFilterChange = () => {
  emit('filtersChange', { ...localFilters.value })
}

const debouncedFilterChange = () => {
  if (searchDebounceTimer) {
    clearTimeout(searchDebounceTimer)
  }
  
  searchDebounceTimer = setTimeout(() => {
    emitFilterChange()
  }, 300)
}

const clearAllFilters = () => {
  localFilters.value = {}
  selectedPreset.value = ''
  emit('clearFilters')
}

const applyPreset = () => {
  const now = new Date()
  const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000)
  
  switch (selectedPreset.value) {
    case 'errors_only':
      localFilters.value = {
        only_errors: true,
        hook_types: [HookType.ERROR]
      }
      break
      
    case 'security_blocked':
      localFilters.value = {
        only_blocked: true,
        risk_levels: [SecurityRisk.HIGH, SecurityRisk.CRITICAL]
      }
      break
      
    case 'high_priority':
      localFilters.value = {
        min_priority: 1,
        hook_types: [HookType.STOP, HookType.ERROR]
      }
      break
      
    case 'tool_usage':
      localFilters.value = {
        hook_types: [HookType.PRE_TOOL_USE, HookType.POST_TOOL_USE]
      }
      break
      
    case 'last_hour':
      localFilters.value = {
        from_time: oneHourAgo.toISOString().slice(0, 16)
      }
      break
      
    default:
      return
  }
  
  emitFilterChange()
}

const getActiveFiltersSummary = () => {
  const summary = []
  
  if (localFilters.value.search_query) {
    summary.push({
      key: 'search_query',
      label: `Search: "${localFilters.value.search_query}"`
    })
  }
  
  if (localFilters.value.agent_ids?.length) {
    summary.push({
      key: 'agent_ids',
      label: `${localFilters.value.agent_ids.length} Agent${localFilters.value.agent_ids.length !== 1 ? 's' : ''}`
    })
  }
  
  if (localFilters.value.session_ids?.length) {
    summary.push({
      key: 'session_ids',
      label: `${localFilters.value.session_ids.length} Session${localFilters.value.session_ids.length !== 1 ? 's' : ''}`
    })
  }
  
  if (localFilters.value.hook_types?.length) {
    summary.push({
      key: 'hook_types',
      label: `${localFilters.value.hook_types.length} Event Type${localFilters.value.hook_types.length !== 1 ? 's' : ''}`
    })
  }
  
  if (localFilters.value.from_time) {
    summary.push({
      key: 'from_time',
      label: `From: ${new Date(localFilters.value.from_time).toLocaleString()}`
    })
  }
  
  if (localFilters.value.to_time) {
    summary.push({
      key: 'to_time',
      label: `To: ${new Date(localFilters.value.to_time).toLocaleString()}`
    })
  }
  
  if (localFilters.value.min_priority && localFilters.value.min_priority > 1) {
    summary.push({
      key: 'min_priority',
      label: `Priority ≥ ${localFilters.value.min_priority}`
    })
  }
  
  if (localFilters.value.risk_levels?.length) {
    summary.push({
      key: 'risk_levels',
      label: `${localFilters.value.risk_levels.length} Risk Level${localFilters.value.risk_levels.length !== 1 ? 's' : ''}`
    })
  }
  
  if (localFilters.value.only_errors) {
    summary.push({
      key: 'only_errors',
      label: 'Errors Only'
    })
  }
  
  if (localFilters.value.only_blocked) {
    summary.push({
      key: 'only_blocked',
      label: 'Blocked Only'
    })
  }
  
  return summary
}

const removeFilter = (filterKey: string) => {
  switch (filterKey) {
    case 'search_query':
      localFilters.value.search_query = undefined
      break
    case 'agent_ids':
      localFilters.value.agent_ids = undefined
      break
    case 'session_ids':
      localFilters.value.session_ids = undefined
      break
    case 'hook_types':
      localFilters.value.hook_types = undefined
      break
    case 'from_time':
      localFilters.value.from_time = undefined
      break
    case 'to_time':
      localFilters.value.to_time = undefined
      break
    case 'min_priority':
      localFilters.value.min_priority = undefined
      break
    case 'risk_levels':
      localFilters.value.risk_levels = undefined
      break
    case 'only_errors':
      localFilters.value.only_errors = undefined
      break
    case 'only_blocked':
      localFilters.value.only_blocked = undefined
      break
  }
  
  emitFilterChange()
}

const saveAsPreset = () => {
  // In a real implementation, this would save to user preferences or local storage
  console.log('Save preset:', localFilters.value)
}

// Watch for external filter changes
watch(
  () => props.filters,
  (newFilters) => {
    localFilters.value = { ...newFilters }
  },
  { deep: true }
)

// Lifecycle
onMounted(() => {
  // Initialize local filters with props
  localFilters.value = { ...props.filters }
})
</script>

<style scoped>
.filter-group {
  @apply space-y-2 min-w-0;
}

.filter-label {
  @apply block text-sm font-medium text-gray-700 dark:text-gray-300;
}

.filter-count {
  @apply text-xs font-normal text-blue-600 dark:text-blue-400;
}

.filter-input {
  @apply w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm;
}

.filter-range {
  @apply w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer;
}

.filter-range::-webkit-slider-thumb {
  @apply appearance-none w-4 h-4 bg-blue-600 dark:bg-blue-500 rounded-full cursor-pointer;
}

.filter-range::-moz-range-thumb {
  @apply w-4 h-4 bg-blue-600 dark:bg-blue-500 rounded-full cursor-pointer border-0;
}

.filter-checkbox {
  @apply w-4 h-4 text-blue-600 bg-gray-100 dark:bg-gray-700 border-gray-300 dark:border-gray-600 rounded focus:ring-blue-500 dark:focus:ring-blue-600 focus:ring-2;
}

.filter-checkbox-label {
  @apply flex items-center space-x-3 text-sm text-gray-700 dark:text-gray-300 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700 rounded px-2 py-1.5 transition-colors;
}

.filter-checkbox-text {
  @apply truncate;
}

.filter-summary {
  @apply p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700;
}

/* Responsive design */
@media (max-width: 768px) {
  .grid.grid-cols-1.md\\:grid-cols-2.lg\\:grid-cols-3 {
    @apply grid-cols-1;
  }
  
  .grid.grid-cols-1.md\\:grid-cols-2.lg\\:grid-cols-4 {
    @apply grid-cols-1;
  }
  
  .filter-checkbox-label {
    @apply px-1 py-1;
  }
}
</style>