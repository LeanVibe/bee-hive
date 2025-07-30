<template>
  <div
    class="agent-card group"
    :class="{
      'drop-target': isDropTarget,
      'suggested-match': suggestedMatch,
      'high-confidence': suggestedMatch && suggestedMatch.match_score > 0.8,
      'medium-confidence': suggestedMatch && suggestedMatch.match_score > 0.6 && suggestedMatch.match_score <= 0.8,
      'low-confidence': suggestedMatch && suggestedMatch.match_score <= 0.6
    }"
    @click="onClick"
    @dragover.prevent="onDragOver"
    @dragleave="onDragLeave"
    @drop.prevent="onDrop"
  >
    <!-- Agent Header -->
    <div class="agent-header flex items-start justify-between mb-3">
      <div class="flex items-center space-x-3">
        <!-- Agent Avatar/Status -->
        <div class="relative">
          <div 
            class="agent-avatar w-10 h-10 rounded-full flex items-center justify-center text-white font-medium text-sm"
            :style="{ backgroundColor: agentColor }"
          >
            {{ getAgentInitials(agent.name) }}
          </div>
          <!-- Status Indicator -->
          <div 
            class="status-indicator absolute -bottom-1 -right-1 w-4 h-4 rounded-full border-2 border-white dark:border-slate-800"
            :class="getStatusIndicatorClass(agent.status)"
          ></div>
        </div>
        
        <div class="flex-1">
          <h4 class="agent-name font-medium text-sm text-slate-900 dark:text-white">
            {{ agent.name }}
          </h4>
          <p class="agent-type text-xs text-slate-500 dark:text-slate-400">
            {{ formatAgentType(agent.type) }}
          </p>
        </div>
      </div>
      
      <!-- Agent Actions -->
      <div class="agent-actions opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          @click.stop="showAgentDetails"
          class="action-button p-1 rounded hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
        >
          <InformationCircleIcon class="w-4 h-4 text-slate-400" />
        </button>
      </div>
    </div>

    <!-- Workload Progress -->
    <div class="workload-section mb-3">
      <div class="flex items-center justify-between mb-1">
        <span class="workload-label text-xs text-slate-600 dark:text-slate-400">
          Workload
        </span>
        <span class="workload-percentage text-xs font-medium text-slate-700 dark:text-slate-300">
          {{ Math.round(agent.current_workload * 100) }}%
        </span>
      </div>
      
      <!-- Workload Progress Bar -->
      <div class="workload-progress bg-slate-200 dark:bg-slate-700 rounded-full h-2 overflow-hidden">
        <div 
          class="workload-fill h-full rounded-full transition-all duration-500"
          :class="getWorkloadFillClass(agent.current_workload)"
          :style="{ width: `${Math.min(agent.current_workload * 100, 100)}%` }"
        ></div>
      </div>
    </div>

    <!-- Agent Capabilities -->
    <div v-if="agent.capabilities?.length" class="capabilities-section mb-3">
      <p class="capabilities-label text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
        Top Skills:
      </p>
      <div class="capabilities-list flex flex-wrap gap-1">
        <span 
          v-for="capability in getTopCapabilities(agent.capabilities)"
          :key="capability.name"
          class="capability-tag inline-block px-2 py-1 text-xs bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 rounded"
          :class="getCapabilityHighlight(capability.name)"
        >
          {{ capability.name }}
          <span class="confidence-score ml-1 text-slate-400">
            {{ Math.round(capability.confidence_level * 100) }}%
          </span>
        </span>
      </div>
    </div>

    <!-- Performance Metrics -->
    <div class="performance-section grid grid-cols-2 gap-2 text-xs mb-3">
      <div class="metric-item">
        <span class="metric-label text-slate-500 dark:text-slate-400">Active Tasks:</span>
        <span class="metric-value ml-1 font-medium text-slate-700 dark:text-slate-300">
          {{ agent.active_tasks }}
        </span>
      </div>
      <div class="metric-item">
        <span class="metric-label text-slate-500 dark:text-slate-400">Performance:</span>
        <span 
          class="metric-value ml-1 font-medium"
          :class="getPerformanceClass(agent.performance_score)"
        >
          {{ Math.round(agent.performance_score * 100) }}%
        </span>
      </div>
    </div>

    <!-- Suggested Match Indicator -->
    <div 
      v-if="suggestedMatch"
      class="match-indicator p-2 bg-gradient-to-r rounded-lg mb-2"
      :class="getMatchIndicatorClass(suggestedMatch.match_score)"
    >
      <div class="flex items-center justify-between">
        <div class="flex items-center space-x-2">
          <SparklesIcon class="w-4 h-4" />
          <span class="text-xs font-medium">
            Suggested Match
          </span>
        </div>
        <span class="text-xs font-bold">
          {{ Math.round(suggestedMatch.match_score * 100) }}%
        </span>
      </div>
      <p class="text-xs mt-1 opacity-90">
        {{ suggestedMatch.reasoning }}
      </p>
    </div>

    <!-- Drop Zone Indicator -->
    <div 
      v-if="isDropTarget"
      class="drop-zone-indicator absolute inset-0 bg-primary-100 dark:bg-primary-900/30 border-2 border-dashed border-primary-400 dark:border-primary-600 rounded-lg flex items-center justify-center"
    >
      <div class="text-center">
        <ArrowDownIcon class="w-8 h-8 text-primary-600 dark:text-primary-400 mx-auto mb-2 animate-bounce" />
        <p class="text-sm font-medium text-primary-700 dark:text-primary-300">
          Drop task here
        </p>
        <p class="text-xs text-primary-600 dark:text-primary-400">
          {{ suggestedMatch ? `${Math.round(suggestedMatch.match_score * 100)}% match` : 'Assign task' }}
        </p>
      </div>
    </div>

    <!-- Availability Status -->
    <div class="availability-status">
      <div 
        class="status-badge inline-flex items-center px-2 py-1 text-xs rounded-full"
        :class="getAvailabilityClass(agent.status, agent.current_workload)"
      >
        <div 
          class="status-dot w-2 h-2 rounded-full mr-1"
          :class="getStatusDotClass(agent.status)"
        ></div>
        {{ getAvailabilityText(agent.status, agent.current_workload) }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useSessionColors } from '@/utils/SessionColorManager'

// Icons
import {
  InformationCircleIcon,
  SparklesIcon,
  ArrowDownIcon
} from '@heroicons/vue/24/outline'

interface AgentCapability {
  name: string
  confidence_level: number
  description?: string
}

interface Agent {
  agent_id: string
  name: string
  type: string
  status: 'active' | 'idle' | 'busy' | 'sleeping' | 'error'
  current_workload: number
  available_capacity: number
  capabilities: AgentCapability[]
  active_tasks: number
  performance_score: number
  last_heartbeat?: string
}

interface SuggestedMatch {
  agent_id: string
  name: string
  match_score: number
  reasoning: string
  workload_impact: number
}

interface Props {
  agent: Agent
  dropTarget?: boolean
  isDropTarget?: boolean
  suggestedMatch?: SuggestedMatch | null
}

interface Emits {
  (e: 'drop', agent: Agent): void
  (e: 'dragover', agent: Agent): void
  (e: 'dragleave'): void
  (e: 'agent-click', agent: Agent): void
  (e: 'agent-details', agent: Agent): void
}

const props = withDefaults(defineProps<Props>(), {
  dropTarget: false,
  isDropTarget: false,
  suggestedMatch: null
})

const emit = defineEmits<Emits>()

const { getAgentColor } = useSessionColors()

// Computed
const agentColor = computed(() => getAgentColor(props.agent.agent_id))

// Methods
const onClick = () => {
  emit('agent-click', props.agent)
}

const onDragOver = () => {
  emit('dragover', props.agent)
}

const onDragLeave = () => {
  emit('dragleave')
}

const onDrop = () => {
  emit('drop', props.agent)
}

const showAgentDetails = () => {
  emit('agent-details', props.agent)
}

const getAgentInitials = (name: string) => {
  return name
    .split(' ')
    .map(n => n.charAt(0))
    .join('')
    .toUpperCase()
    .slice(0, 2)
}

const formatAgentType = (type: string) => {
  return type
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

const getStatusIndicatorClass = (status: string) => {
  const classes = {
    'active': 'bg-green-500',
    'idle': 'bg-yellow-500',
    'busy': 'bg-blue-500',
    'sleeping': 'bg-gray-500',
    'error': 'bg-red-500'
  }
  return classes[status] || 'bg-gray-500'
}

const getWorkloadFillClass = (workload: number) => {
  if (workload >= 0.9) {
    return 'bg-red-500'
  } else if (workload >= 0.7) {
    return 'bg-orange-500'
  } else if (workload >= 0.5) {
    return 'bg-yellow-500'
  } else {
    return 'bg-green-500'
  }
}

const getTopCapabilities = (capabilities: AgentCapability[]) => {
  return capabilities
    .sort((a, b) => b.confidence_level - a.confidence_level)
    .slice(0, 3)
}

const getCapabilityHighlight = (capabilityName: string) => {
  // This would be enhanced to highlight matching capabilities
  // based on the current task selection
  return ''
}

const getPerformanceClass = (score: number) => {
  if (score >= 0.8) {
    return 'text-green-600 dark:text-green-400'
  } else if (score >= 0.6) {
    return 'text-yellow-600 dark:text-yellow-400'
  } else {
    return 'text-red-600 dark:text-red-400'
  }
}

const getMatchIndicatorClass = (matchScore: number) => {
  if (matchScore > 0.8) {
    return 'from-green-100 to-emerald-100 dark:from-green-900/20 dark:to-emerald-900/20 text-green-700 dark:text-green-300 border border-green-200 dark:border-green-800'
  } else if (matchScore > 0.6) {
    return 'from-blue-100 to-cyan-100 dark:from-blue-900/20 dark:to-cyan-900/20 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800'
  } else {
    return 'from-yellow-100 to-orange-100 dark:from-yellow-900/20 dark:to-orange-900/20 text-yellow-700 dark:text-yellow-300 border border-yellow-200 dark:border-yellow-800'
  }
}

const getAvailabilityClass = (status: string, workload: number) => {
  if (status === 'error') {
    return 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
  } else if (status === 'sleeping') {
    return 'bg-gray-100 dark:bg-gray-900/30 text-gray-700 dark:text-gray-400'
  } else if (workload >= 0.9) {
    return 'bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400'
  } else if (workload >= 0.7) {
    return 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400'
  } else {
    return 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
  }
}

const getStatusDotClass = (status: string) => {
  return getStatusIndicatorClass(status)
}

const getAvailabilityText = (status: string, workload: number) => {
  if (status === 'error') {
    return 'Error'
  } else if (status === 'sleeping') {
    return 'Sleeping'
  } else if (workload >= 0.9) {
    return 'At Capacity'
  } else if (workload >= 0.7) {
    return 'Busy'
  } else if (workload >= 0.3) {
    return 'Available'
  } else {
    return 'Idle'
  }
}
</script>

<style scoped>
.agent-card {
  @apply relative p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 hover:shadow-md transition-all duration-200 cursor-pointer;
}

.agent-card:hover {
  @apply transform -translate-y-1 shadow-lg border-slate-300 dark:border-slate-600;
}

.agent-card.drop-target {
  @apply ring-2 ring-primary-400 dark:ring-primary-600 border-primary-300 dark:border-primary-600;
}

.agent-card.suggested-match {
  @apply ring-1 ring-blue-300 dark:ring-blue-600;
}

.agent-card.high-confidence {
  @apply ring-2 ring-green-400 dark:ring-green-600 bg-green-50/50 dark:bg-green-900/10;
}

.agent-card.medium-confidence {
  @apply ring-1 ring-blue-400 dark:ring-blue-600 bg-blue-50/50 dark:bg-blue-900/10;
}

.agent-card.low-confidence {
  @apply ring-1 ring-yellow-400 dark:ring-yellow-600 bg-yellow-50/50 dark:bg-yellow-900/10;
}

.workload-progress {
  @apply relative overflow-hidden;
}

.workload-fill {
  @apply relative;
}

.workload-fill::after {
  content: '';
  @apply absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent;
  animation: shimmer 2s infinite;
}

.capability-tag {
  @apply transition-all duration-150 hover:scale-105;
}

.match-indicator {
  @apply transition-all duration-300;
}

.drop-zone-indicator {
  @apply transition-all duration-200;
}

.status-badge {
  @apply transition-all duration-200;
}

@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

@keyframes pulse-ring {
  0% {
    transform: scale(0.8);
    opacity: 1;
  }
  100% {
    transform: scale(2);
    opacity: 0;
  }
}

.agent-card.high-confidence::before {
  content: '';
  @apply absolute -inset-1 bg-green-400/20 dark:bg-green-600/20 rounded-lg;
  animation: pulse-ring 2s infinite;
}
</style>