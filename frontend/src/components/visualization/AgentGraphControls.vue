<template>
  <div class="agent-graph-controls">
    <!-- Main Controls Panel -->
    <div class="controls-panel bg-white dark:bg-gray-800 rounded-lg shadow-lg">
      <!-- Header -->
      <div class="controls-header px-4 py-3 border-b border-gray-200 dark:border-gray-700">
        <div class="flex items-center justify-between">
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
            Graph Controls
          </h3>
          <button
            @click="toggleCollapsed"
            class="p-1 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          >
            <ChevronUpIcon 
              v-if="!isCollapsed"
              class="w-5 h-5 transform transition-transform"
            />
            <ChevronDownIcon 
              v-else
              class="w-5 h-5 transform transition-transform"
            />
          </button>
        </div>
      </div>
      
      <!-- Controls Content -->
      <div 
        v-show="!isCollapsed"
        class="controls-content p-4 space-y-6"
      >
        <!-- Filter Section -->
        <div class="filter-section">
          <h4 class="text-sm font-medium text-gray-900 dark:text-white mb-3">
            Filters
          </h4>
          
          <!-- Session Filter -->
          <div class="filter-group mb-4">
            <label class="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
              Sessions
            </label>
            <div class="relative">
              <select
                v-model="filters.sessionIds"
                multiple
                class="w-full px-3 py-2 text-xs border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                :class="{ 'text-gray-500': filters.sessionIds.length === 0 }"
                @change="applyFilters"
              >
                <option value="" disabled>Select sessions...</option>
                <option
                  v-for="session in availableSessions"
                  :key="session.session_id"
                  :value="session.session_id"
                  class="flex items-center"
                >
                  {{ formatSessionLabel(session) }}
                </option>
              </select>
            </div>
            <div v-if="filters.sessionIds.length > 0" class="mt-2 flex flex-wrap gap-1">
              <span
                v-for="sessionId in filters.sessionIds"
                :key="sessionId"
                class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium"
                :style="{ 
                  backgroundColor: getSessionColor(sessionId) + '20',
                  color: getSessionColor(sessionId)
                }"
              >
                {{ formatSessionLabel(getSessionById(sessionId)) }}
                <button
                  @click="removeSessionFilter(sessionId)"
                  class="ml-1 hover:text-red-600"
                >
                  <XMarkIcon class="w-3 h-3" />
                </button>
              </span>
            </div>
          </div>
          
          <!-- Agent Status Filter -->
          <div class="filter-group mb-4">
            <label class="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
              Agent Status
            </label>
            <div class="grid grid-cols-2 gap-2">
              <label
                v-for="status in agentStatuses"
                :key="status"
                class="flex items-center"
              >
                <input
                  type="checkbox"
                  :value="status"
                  v-model="filters.agentStatuses"
                  @change="applyFilters"
                  class="rounded border-gray-300 text-blue-600 focus:ring-blue-500 focus:ring-offset-0"
                />
                <span 
                  class="ml-2 text-xs font-medium capitalize"
                  :class="getStatusTextClass(status)"
                >
                  {{ status }}
                </span>
              </label>
            </div>
          </div>
          
          <!-- Performance Range Filter -->
          <div class="filter-group mb-4">
            <label class="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
              Performance Range
            </label>
            <div class="px-3">
              <input
                type="range"
                v-model="filters.minPerformance"
                min="0"
                max="100"
                step="10"
                @input="applyFilters"
                class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
              />
              <div class="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                <span>{{ filters.minPerformance }}%</span>
                <span>100%</span>
              </div>
            </div>
          </div>
          
          <!-- Activity Level Filter -->
          <div class="filter-group mb-4">
            <label class="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
              Activity Level
            </label>
            <select
              v-model="filters.activityLevel"
              @change="applyFilters"
              class="w-full px-3 py-2 text-xs border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
            >
              <option value="">All Activity Levels</option>
              <option value="high">High Activity (50+ events)</option>
              <option value="medium">Medium Activity (10-49 events)</option>
              <option value="low">Low Activity (1-9 events)</option>
              <option value="idle">Idle (0 events)</option>
            </select>
          </div>
          
          <!-- Search Filter -->
          <div class="filter-group mb-4">
            <label class="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
              Search Agents
            </label>
            <div class="relative">
              <input
                type="text"
                v-model="filters.searchQuery"
                @input="applyFilters"
                placeholder="Search by agent ID or activity..."
                class="w-full px-3 py-2 pl-8 text-xs border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <MagnifyingGlassIcon class="absolute left-2 top-2.5 w-3 h-3 text-gray-400" />
            </div>
          </div>
          
          <!-- Filter Actions -->
          <div class="flex space-x-2">
            <button
              @click="clearAllFilters"
              class="flex-1 px-3 py-2 text-xs font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-md transition-colors"
            >
              Clear All
            </button>
            <button
              @click="saveFilterPreset"
              class="flex-1 px-3 py-2 text-xs font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-md transition-colors"
            >
              Save Preset
            </button>
          </div>
        </div>
        
        <!-- Layout Section -->
        <div class="layout-section border-t border-gray-200 dark:border-gray-700 pt-6">
          <h4 class="text-sm font-medium text-gray-900 dark:text-white mb-3">
            Layout & View
          </h4>
          
          <!-- Layout Type -->
          <div class="mb-4">
            <label class="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
              Layout Algorithm
            </label>
            <div class="grid grid-cols-1 gap-2">
              <label
                v-for="layout in layoutTypes"
                :key="layout.value"
                class="flex items-center cursor-pointer"
              >
                <input
                  type="radio"
                  :value="layout.value"
                  v-model="layoutSettings.type"
                  @change="applyLayoutChange"
                  class="text-blue-600 focus:ring-blue-500 focus:ring-offset-0"
                />
                <div class="ml-2">
                  <div class="text-xs font-medium text-gray-900 dark:text-white">
                    {{ layout.label }}
                  </div>
                  <div class="text-xs text-gray-500 dark:text-gray-400">
                    {{ layout.description }}
                  </div>
                </div>
              </label>
            </div>
          </div>
          
          <!-- Visualization Mode -->
          <div class="mb-4">
            <label class="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
              Visualization Mode
            </label>
            <select
              v-model="layoutSettings.visualizationMode"
              @change="applyLayoutChange"
              class="w-full px-3 py-2 text-xs border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
            >
              <option value="session">Color by Session</option>
              <option value="performance">Color by Performance</option>
              <option value="security">Color by Security Risk</option>
              <option value="activity">Color by Activity Level</option>
            </select>
          </div>
          
          <!-- Node Size Mode -->
          <div class="mb-4">
            <label class="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
              Node Size Based On
            </label>
            <select
              v-model="layoutSettings.nodeSizeMode"
              @change="applyLayoutChange"
              class="w-full px-3 py-2 text-xs border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
            >
              <option value="performance">Performance Score</option>
              <option value="activity">Activity Level</option>
              <option value="connections">Connection Count</option>
              <option value="uniform">Uniform Size</option>
            </select>
          </div>
          
          <!-- Link Display Options -->
          <div class="mb-4">
            <label class="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
              Link Display
            </label>
            <div class="space-y-2">
              <label class="flex items-center">
                <input
                  type="checkbox"
                  v-model="layoutSettings.showWeakLinks"
                  @change="applyLayoutChange"
                  class="rounded border-gray-300 text-blue-600 focus:ring-blue-500 focus:ring-offset-0"
                />
                <span class="ml-2 text-xs text-gray-700 dark:text-gray-300">
                  Show weak connections
                </span>
              </label>
              <label class="flex items-center">
                <input
                  type="checkbox"
                  v-model="layoutSettings.showLinkLabels"
                  @change="applyLayoutChange"
                  class="rounded border-gray-300 text-blue-600 focus:ring-blue-500 focus:ring-offset-0"
                />
                <span class="ml-2 text-xs text-gray-700 dark:text-gray-300">
                  Show link labels
                </span>
              </label>
              <label class="flex items-center">
                <input
                  type="checkbox"
                  v-model="layoutSettings.animateLinks"
                  @change="applyLayoutChange"
                  class="rounded border-gray-300 text-blue-600 focus:ring-blue-500 focus:ring-offset-0"
                />
                <span class="ml-2 text-xs text-gray-700 dark:text-gray-300">
                  Animate data flow
                </span>
              </label>
            </div>
          </div>
        </div>
        
        <!-- View Controls Section -->
        <div class="view-controls-section border-t border-gray-200 dark:border-gray-700 pt-6">
          <h4 class="text-sm font-medium text-gray-900 dark:text-white mb-3">
            View Controls
          </h4>
          
          <!-- Zoom Controls -->
          <div class="mb-4">
            <label class="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
              Zoom Level: {{ Math.round(zoomLevel * 100) }}%
            </label>
            <div class="flex items-center space-x-2">
              <button
                @click="$emit('zoom-out')"
                class="p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                title="Zoom Out"
              >
                <MinusIcon class="w-4 h-4" />
              </button>
              <div class="flex-1 px-2">
                <input
                  type="range"
                  :value="zoomLevel"
                  min="0.1"
                  max="3"
                  step="0.1"
                  @input="$emit('zoom-to', $event.target.value)"
                  class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                />
              </div>
              <button
                @click="$emit('zoom-in')"
                class="p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                title="Zoom In"
              >
                <PlusIcon class="w-4 h-4" />
              </button>
            </div>
            <div class="flex justify-center mt-2">
              <button
                @click="$emit('reset-zoom')"
                class="px-3 py-1 text-xs font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded transition-colors"
              >
                Reset View
              </button>
            </div>
          </div>
          
          <!-- Focus Controls -->
          <div class="mb-4">
            <label class="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">
              Focus On
            </label>
            <select
              v-model="focusTarget"
              @change="handleFocusChange"
              class="w-full px-3 py-2 text-xs border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
            >
              <option value="">Select agent or session...</option>
              <optgroup label="Agents">
                <option
                  v-for="agent in filteredAgents"
                  :key="`agent-${agent.agent_id}`"
                  :value="`agent-${agent.agent_id}`"
                >
                  {{ agent.name || agent.agent_id.slice(-8) }}
                </option>
              </optgroup>
              <optgroup label="Sessions">
                <option
                  v-for="session in availableSessions"
                  :key="`session-${session.session_id}`"
                  :value="`session-${session.session_id}`"
                >
                  Session {{ session.session_id.slice(-8) }}
                </option>
              </optgroup>
            </select>
          </div>
          
          <!-- Performance Controls -->
          <div class="mb-4">
            <label class="flex items-center">
              <input
                type="checkbox"
                v-model="performanceSettings.enableAdaptiveRendering"
                @change="applyPerformanceSettings"
                class="rounded border-gray-300 text-blue-600 focus:ring-blue-500 focus:ring-offset-0"
              />
              <span class="ml-2 text-xs text-gray-700 dark:text-gray-300">
                Adaptive rendering
              </span>
            </label>
            <div class="mt-2 text-xs text-gray-500 dark:text-gray-400">
              FPS: {{ Math.round(performanceStats.fps || 0) }} | 
              Nodes: {{ nodeCount }} | 
              Links: {{ linkCount }}
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Mini Stats Panel -->
    <div class="stats-panel bg-white dark:bg-gray-800 rounded-lg shadow-lg p-3 mt-4">
      <div class="text-xs text-gray-600 dark:text-gray-400 space-y-1">
        <div class="flex justify-between">
          <span>Visible Nodes:</span>
          <span class="font-mono">{{ filteredNodeCount }}</span>
        </div>
        <div class="flex justify-between">
          <span>Active Sessions:</span>
          <span class="font-mono">{{ activeSessionCount }}</span>
        </div>
        <div class="flex justify-between">
          <span>Avg Performance:</span>
          <span class="font-mono">{{ averagePerformance }}%</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, watch } from 'vue'
import {
  ChevronUpIcon,
  ChevronDownIcon,
  XMarkIcon,
  MagnifyingGlassIcon,
  PlusIcon,
  MinusIcon
} from '@heroicons/vue/24/outline'

import { useSessionColors } from '@/utils/SessionColorManager'
import { useEventsStore } from '@/stores/events'
import type { AgentInfo, SessionInfo } from '@/types/hooks'

// Props
interface Props {
  agents: AgentInfo[]
  sessions: SessionInfo[]
  zoomLevel: number
  nodeCount: number
  linkCount: number
  performanceStats: {
    fps?: number
    updateCount?: number
    renderTime?: number
  }
}

const props = defineProps<Props>()

// Emits
const emit = defineEmits<{
  'filters-changed': [filters: any]
  'layout-changed': [settings: any]
  'zoom-in': []
  'zoom-out': []
  'zoom-to': [level: number]
  'reset-zoom': []
  'focus-on': [target: string]
  'performance-settings-changed': [settings: any]
}>()

// Stores and utilities
const eventsStore = useEventsStore()
const { getSessionColor } = useSessionColors()

// Local state
const isCollapsed = ref(false)
const focusTarget = ref('')

// Filter state
const filters = reactive({
  sessionIds: [] as string[],
  agentStatuses: ['active', 'idle'] as string[],
  minPerformance: 0,
  activityLevel: '',
  searchQuery: ''
})

// Layout settings
const layoutSettings = reactive({
  type: 'force' as 'force' | 'circle' | 'grid' | 'hierarchical',
  visualizationMode: 'session' as 'session' | 'performance' | 'security' | 'activity',
  nodeSizeMode: 'performance' as 'performance' | 'activity' | 'connections' | 'uniform',
  showWeakLinks: false,
  showLinkLabels: false,
  animateLinks: true
})

// Performance settings
const performanceSettings = reactive({
  enableAdaptiveRendering: true,
  maxNodes: 100,
  simplifyAtDistance: true,
  cullOffscreen: true
})

// Constants
const agentStatuses = ['active', 'idle', 'error', 'blocked']

const layoutTypes = [
  {
    value: 'force',
    label: 'Force-Directed',
    description: 'Dynamic physics-based layout'
  },
  {
    value: 'circle',
    label: 'Circular',
    description: 'Arrange nodes in a circle'
  },
  {
    value: 'grid',
    label: 'Grid',
    description: 'Organize in a regular grid'
  },
  {
    value: 'hierarchical',
    label: 'Hierarchical',
    description: 'Tree-like structure'
  }
]

// Computed properties
const availableSessions = computed(() => props.sessions || [])

const filteredAgents = computed(() => {
  return props.agents.filter(agent => {
    // Session filter
    if (filters.sessionIds.length > 0) {
      const hasMatchingSession = agent.session_ids.some(sessionId =>
        filters.sessionIds.includes(sessionId)
      )
      if (!hasMatchingSession) return false
    }
    
    // Status filter
    if (filters.agentStatuses.length > 0 && !filters.agentStatuses.includes(agent.status)) {
      return false
    }
    
    // Performance filter
    const performance = calculateAgentPerformance(agent)
    if (performance < filters.minPerformance) {
      return false
    }
    
    // Activity level filter
    if (filters.activityLevel) {
      const activityLevel = getActivityLevel(agent)
      if (activityLevel !== filters.activityLevel) {
        return false
      }
    }
    
    // Search filter
    if (filters.searchQuery) {
      const query = filters.searchQuery.toLowerCase()
      const searchableText = [
        agent.agent_id,
        ...agent.session_ids
      ].join(' ').toLowerCase()
      
      if (!searchableText.includes(query)) {
        return false
      }
    }
    
    return true
  })
})

const filteredNodeCount = computed(() => filteredAgents.value.length)

const activeSessionCount = computed(() => {
  const activeSessions = new Set<string>()
  filteredAgents.value.forEach(agent => {
    if (agent.status === 'active') {
      agent.session_ids.forEach(sessionId => activeSessions.add(sessionId))
    }
  })
  return activeSessions.size
})

const averagePerformance = computed(() => {
  if (filteredAgents.value.length === 0) return 0
  
  const totalPerformance = filteredAgents.value.reduce((sum, agent) => {
    return sum + calculateAgentPerformance(agent)
  }, 0)
  
  return Math.round(totalPerformance / filteredAgents.value.length)
})

// Methods
const toggleCollapsed = () => {
  isCollapsed.value = !isCollapsed.value
}

const applyFilters = () => {
  emit('filters-changed', {
    ...filters,
    filteredAgents: filteredAgents.value
  })
}

const applyLayoutChange = () => {
  emit('layout-changed', { ...layoutSettings })
}

const applyPerformanceSettings = () => {
  emit('performance-settings-changed', { ...performanceSettings })
}

const clearAllFilters = () => {
  filters.sessionIds = []
  filters.agentStatuses = ['active', 'idle']
  filters.minPerformance = 0
  filters.activityLevel = ''
  filters.searchQuery = ''
  applyFilters()
}

const saveFilterPreset = () => {
  // Save filter preset to localStorage
  const preset = {
    name: `Preset ${Date.now()}`,
    filters: { ...filters },
    timestamp: new Date().toISOString()
  }
  
  const existingPresets = JSON.parse(localStorage.getItem('graphFilterPresets') || '[]')
  existingPresets.push(preset)
  localStorage.setItem('graphFilterPresets', JSON.stringify(existingPresets))
  
  // Show success message (could be implemented with a toast notification)
  console.log('Filter preset saved:', preset.name)
}

const removeSessionFilter = (sessionId: string) => {
  filters.sessionIds = filters.sessionIds.filter(id => id !== sessionId)
  applyFilters()
}

const handleFocusChange = () => {
  if (focusTarget.value) {
    emit('focus-on', focusTarget.value)
  }
}

const formatSessionLabel = (session: SessionInfo | undefined): string => {
  if (!session) return 'Unknown Session'
  return `${session.session_id.slice(-8)} (${session.status})`
}

const getSessionById = (sessionId: string): SessionInfo | undefined => {
  return availableSessions.value.find(s => s.session_id === sessionId)
}

const getStatusTextClass = (status: string): string => {
  const classes = {
    'active': 'text-green-600 dark:text-green-400',
    'idle': 'text-yellow-600 dark:text-yellow-400',
    'error': 'text-red-600 dark:text-red-400',
    'blocked': 'text-orange-600 dark:text-orange-400'
  }
  return classes[status as keyof typeof classes] || 'text-gray-600 dark:text-gray-400'
}

const calculateAgentPerformance = (agent: AgentInfo): number => {
  // Mock performance calculation based on error rate and activity
  const errorRate = agent.error_count / Math.max(agent.event_count, 1)
  const basePerformance = Math.max(0, 100 - (errorRate * 100))
  
  // Add activity bonus
  const activityBonus = Math.min(agent.event_count * 0.5, 20)
  
  return Math.min(100, Math.round(basePerformance + activityBonus))
}

const getActivityLevel = (agent: AgentInfo): string => {
  if (agent.event_count >= 50) return 'high'
  if (agent.event_count >= 10) return 'medium'
  if (agent.event_count >= 1) return 'low'
  return 'idle'
}

// Watchers
watch(() => props.agents, () => {
  // Reapply filters when agents change
  applyFilters()
}, { deep: true })

watch(() => props.sessions, () => {
  // Update available sessions
  // Remove invalid session filters
  filters.sessionIds = filters.sessionIds.filter(sessionId =>
    availableSessions.value.some(s => s.session_id === sessionId)
  )
  applyFilters()
}, { deep: true })

// Initialize with default filters
applyFilters()
</script>

<style scoped>
.agent-graph-controls {
  position: absolute;
  top: 1rem;
  left: 1rem;
  z-index: 10;
  width: 320px;
  max-height: calc(100vh - 2rem);
  overflow-y: auto;
}

.controls-panel {
  backdrop-filter: blur(10px);
  background-color: rgba(255, 255, 255, 0.95);
}

.dark .controls-panel {
  background-color: rgba(31, 41, 55, 0.95);
}

.filter-group {
  transition: all 0.2s ease;
}

.filter-group:hover {
  transform: translateY(-1px);
}

/* Custom scrollbar */
.agent-graph-controls::-webkit-scrollbar {
  width: 4px;
}

.agent-graph-controls::-webkit-scrollbar-track {
  background: transparent;
}

.agent-graph-controls::-webkit-scrollbar-thumb {
  background: rgba(156, 163, 175, 0.5);
  border-radius: 2px;
}

.agent-graph-controls::-webkit-scrollbar-thumb:hover {
  background: rgba(156, 163, 175, 0.7);
}

/* Animate collapse/expand */
.controls-content {
  transition: all 0.3s ease;
  overflow: hidden;
}

/* Range input styling */
input[type="range"]::-webkit-slider-thumb {
  appearance: none;
  height: 16px;
  width: 16px;
  border-radius: 50%;
  background: #3B82F6;
  cursor: pointer;
  border: 2px solid #FFFFFF;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

input[type="range"]::-moz-range-thumb {
  height: 16px;
  width: 16px;
  border-radius: 50%;
  background: #3B82F6;
  cursor: pointer;
  border: 2px solid #FFFFFF;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Multi-select styling */
select[multiple] {
  max-height: 120px;
}

select[multiple] option {
  padding: 0.25rem 0.5rem;
}

select[multiple] option:checked {
  background: #3B82F6 !important;
  color: white !important;
}
</style>