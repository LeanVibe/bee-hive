<template>
  <div class="assignment-confirmation-modal fixed inset-0 z-50 overflow-y-auto">
    <!-- Backdrop -->
    <div 
      class="modal-backdrop fixed inset-0 bg-black bg-opacity-50 transition-opacity"
      @click="$emit('cancel')"
    ></div>

    <!-- Modal -->
    <div class="modal-container flex items-center justify-center min-h-screen px-4 py-6">
      <div class="modal-content glass-card rounded-xl w-full max-w-lg">
        <!-- Header -->
        <div class="modal-header p-6 border-b border-slate-200 dark:border-slate-700">
          <div class="flex items-center space-x-3 mb-2">
            <div class="icon-container p-2 bg-primary-100 dark:bg-primary-900/30 rounded-lg">
              <UserPlusIcon class="w-6 h-6 text-primary-600 dark:text-primary-400" />
            </div>
            <h2 class="text-xl font-semibold text-slate-900 dark:text-white">
              Confirm Task Assignment
            </h2>
          </div>
          <p class="text-sm text-slate-600 dark:text-slate-400">
            Review the assignment details before confirming
          </p>
        </div>

        <!-- Content -->
        <div class="modal-body p-6 space-y-6">
          <!-- Task Information -->
          <div class="task-info">
            <h3 class="section-title">Task Details</h3>
            <div class="info-card">
              <div class="flex items-start justify-between mb-3">
                <div class="flex-1">
                  <h4 class="font-medium text-slate-900 dark:text-white">
                    {{ task?.task_title }}
                  </h4>
                  <p class="text-sm text-slate-600 dark:text-slate-400 mt-1">
                    {{ task?.task_description }}
                  </p>
                </div>
                <span 
                  class="priority-badge px-2 py-1 text-xs rounded-full font-medium ml-3"
                  :class="getPriorityClass(task?.priority)"
                >
                  {{ task?.priority }}
                </span>
              </div>
              
              <!-- Task Metadata -->
              <div class="task-metadata grid grid-cols-2 gap-3 text-sm">
                <div>
                  <span class="metadata-label">Type:</span>
                  <span class="metadata-value">{{ formatTaskType(task?.task_type) }}</span>
                </div>
                <div>
                  <span class="metadata-label">Effort:</span>
                  <span class="metadata-value">
                    {{ task?.estimated_effort_hours ? `${task.estimated_effort_hours}h` : 'TBD' }}
                  </span>
                </div>
              </div>

              <!-- Required Capabilities -->
              <div v-if="task?.required_capabilities?.length" class="capabilities mt-3">
                <p class="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Required Skills:
                </p>
                <div class="capability-tags flex flex-wrap gap-1">
                  <span 
                    v-for="capability in task.required_capabilities"
                    :key="capability"
                    class="capability-tag px-2 py-1 text-xs bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded"
                  >
                    {{ capability }}
                  </span>
                </div>
              </div>
            </div>
          </div>

          <!-- Agent Information -->
          <div class="agent-info">
            <h3 class="section-title">Assigned Agent</h3>
            <div class="info-card">
              <div class="flex items-center space-x-4">
                <!-- Agent Avatar -->
                <div 
                  class="agent-avatar w-12 h-12 rounded-full flex items-center justify-center text-white font-medium"
                  :style="{ backgroundColor: getAgentColor(agent?.agent_id) }"
                >
                  {{ getAgentInitials(agent?.name) }}
                </div>
                
                <!-- Agent Details -->
                <div class="flex-1">
                  <h4 class="font-medium text-slate-900 dark:text-white">
                    {{ agent?.name }}
                  </h4>
                  <p class="text-sm text-slate-600 dark:text-slate-400">
                    {{ formatAgentType(agent?.type) }}
                  </p>
                  
                  <!-- Agent Status -->
                  <div class="flex items-center space-x-2 mt-1">
                    <div 
                      class="status-indicator w-2 h-2 rounded-full"
                      :class="getStatusIndicatorClass(agent?.status)"
                    ></div>
                    <span class="text-xs text-slate-500 dark:text-slate-400">
                      {{ getStatusText(agent?.status) }}
                    </span>
                  </div>
                </div>
                
                <!-- Current Workload -->
                <div class="workload-info text-center">
                  <div class="workload-circle relative w-16 h-16">
                    <svg class="w-16 h-16 transform -rotate-90" viewBox="0 0 36 36">
                      <path
                        class="workload-bg"
                        d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="2"
                        stroke-dasharray="100, 100"
                      />
                      <path
                        class="workload-fill"
                        :class="getWorkloadColor(agent?.current_workload)"
                        d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="2"
                        :stroke-dasharray="`${(agent?.current_workload || 0) * 100}, 100`"
                      />
                    </svg>
                    <div class="workload-percentage absolute inset-0 flex items-center justify-center text-xs font-medium">
                      {{ Math.round((agent?.current_workload || 0) * 100) }}%
                    </div>
                  </div>
                  <p class="text-xs text-slate-500 dark:text-slate-400 mt-1">
                    Current Load
                  </p>
                </div>
              </div>
            </div>
          </div>

          <!-- Assignment Analysis -->
          <div class="assignment-analysis">
            <h3 class="section-title">Assignment Analysis</h3>
            <div class="info-card">
              <!-- Confidence Score -->
              <div class="confidence-section mb-4">
                <div class="flex items-center justify-between mb-2">
                  <span class="text-sm font-medium text-slate-700 dark:text-slate-300">
                    Match Confidence
                  </span>
                  <span 
                    class="confidence-score text-sm font-bold"
                    :class="getConfidenceColor(confidence)"
                  >
                    {{ Math.round(confidence * 100) }}%
                  </span>
                </div>
                
                <!-- Confidence Bar -->
                <div class="confidence-bar w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                  <div 
                    class="confidence-fill h-full rounded-full transition-all duration-500"
                    :class="getConfidenceBarColor(confidence)"
                    :style="{ width: `${confidence * 100}%` }"
                  ></div>
                </div>
              </div>

              <!-- Analysis Factors -->
              <div class="analysis-factors space-y-2">
                <div class="factor-item flex items-center justify-between text-sm">
                  <span class="factor-label">Skill Match:</span>
                  <div class="flex items-center space-x-2">
                    <div class="factor-bar w-16 bg-slate-200 dark:bg-slate-700 rounded-full h-1">
                      <div 
                        class="factor-fill h-full bg-green-500 rounded-full"
                        :style="{ width: `${skillMatchScore * 100}%` }"
                      ></div>
                    </div>
                    <span class="factor-score">{{ Math.round(skillMatchScore * 100) }}%</span>
                  </div>
                </div>
                
                <div class="factor-item flex items-center justify-between text-sm">
                  <span class="factor-label">Availability:</span>
                  <div class="flex items-center space-x-2">
                    <div class="factor-bar w-16 bg-slate-200 dark:bg-slate-700 rounded-full h-1">
                      <div 
                        class="factor-fill h-full bg-blue-500 rounded-full"
                        :style="{ width: `${availabilityScore * 100}%` }"
                      ></div>
                    </div>
                    <span class="factor-score">{{ Math.round(availabilityScore * 100) }}%</span>
                  </div>
                </div>
                
                <div class="factor-item flex items-center justify-between text-sm">
                  <span class="factor-label">Performance:</span>
                  <div class="flex items-center space-x-2">
                    <div class="factor-bar w-16 bg-slate-200 dark:bg-slate-700 rounded-full h-1">
                      <div 
                        class="factor-fill h-full bg-purple-500 rounded-full"
                        :style="{ width: `${performanceScore * 100}%` }"
                      ></div>
                    </div>
                    <span class="factor-score">{{ Math.round(performanceScore * 100) }}%</span>
                  </div>
                </div>
              </div>

              <!-- Estimated Impact -->
              <div class="impact-section mt-4 pt-4 border-t border-slate-200 dark:border-slate-700">
                <div class="grid grid-cols-2 gap-4 text-sm">
                  <div class="impact-item">
                    <span class="impact-label">Estimated Completion:</span>
                    <span class="impact-value font-medium">
                      {{ getEstimatedCompletion() }}
                    </span>
                  </div>
                  <div class="impact-item">
                    <span class="impact-label">Workload Change:</span>
                    <span 
                      class="impact-value font-medium"
                      :class="getWorkloadChangeColor(workloadImpact)"
                    >
                      {{ getWorkloadChangeText(workloadImpact) }}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Warnings/Recommendations -->
          <div v-if="hasWarnings" class="warnings-section">
            <div class="warning-card bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
              <div class="flex items-start space-x-3">
                <ExclamationTriangleIcon class="w-5 h-5 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-0.5" />
                <div>
                  <h4 class="font-medium text-yellow-800 dark:text-yellow-200 mb-1">
                    Assignment Considerations
                  </h4>
                  <ul class="text-sm text-yellow-700 dark:text-yellow-300 space-y-1">
                    <li v-for="warning in warnings" :key="warning">
                      • {{ warning }}
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Footer -->
        <div class="modal-footer flex items-center justify-end space-x-3 p-6 border-t border-slate-200 dark:border-slate-700">
          <button
            @click="$emit('cancel')"
            class="btn-secondary"
          >
            Cancel
          </button>
          <button
            @click="$emit('confirm')"
            class="btn-primary"
            :class="{ 'btn-warning': hasWarnings }"
          >
            <CheckIcon class="w-4 h-4 mr-2" />
            {{ hasWarnings ? 'Assign Anyway' : 'Confirm Assignment' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { formatDistanceToNow, addHours } from 'date-fns'
import { useSessionColors } from '@/utils/SessionColorManager'

// Icons
import {
  UserPlusIcon,
  ExclamationTriangleIcon,
  CheckIcon
} from '@heroicons/vue/24/outline'

interface Task {
  id?: string
  task_title: string
  task_description: string
  task_type: string
  priority: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW'
  required_capabilities: string[]
  estimated_effort_hours?: number
}

interface Agent {
  agent_id: string
  name: string
  type: string
  status: 'active' | 'idle' | 'busy' | 'sleeping' | 'error'
  current_workload: number
  available_capacity: number
  capabilities: Array<{ name: string; confidence_level: number }>
  performance_score: number
}

interface Props {
  task?: Task | null
  agent?: Agent | null
  confidence?: number
}

interface Emits {
  (e: 'confirm'): void
  (e: 'cancel'): void
}

const props = withDefaults(defineProps<Props>(), {
  task: null,
  agent: null,
  confidence: 0.5
})

const emit = defineEmits<Emits>()

const { getAgentColor } = useSessionColors()

// Computed values for analysis
const skillMatchScore = computed(() => {
  if (!props.task?.required_capabilities?.length || !props.agent?.capabilities?.length) {
    return 0.5
  }

  const requiredSkills = props.task.required_capabilities.map(s => s.toLowerCase())
  const agentSkills = props.agent.capabilities.map(c => c.name.toLowerCase())
  
  const matchedSkills = requiredSkills.filter(req => 
    agentSkills.some(agent => agent.includes(req) || req.includes(agent))
  )
  
  return matchedSkills.length / requiredSkills.length
})

const availabilityScore = computed(() => {
  if (!props.agent) return 0.5
  return 1 - props.agent.current_workload
})

const performanceScore = computed(() => {
  return props.agent?.performance_score || 0.5
})

const workloadImpact = computed(() => {
  if (!props.agent) return 0
  const taskImpact = (props.task?.estimated_effort_hours || 2) / 40 // Assume 40h work week
  return taskImpact / Math.max(props.agent.available_capacity, 0.1)
})

const warnings = computed(() => {
  const warnings = []
  
  if (props.agent?.current_workload >= 0.8) {
    warnings.push('Agent is near capacity - may affect response time')
  }
  
  if (skillMatchScore.value < 0.6) {
    warnings.push('Limited skill match - consider additional training or pairing')
  }
  
  if (props.agent?.status === 'busy') {
    warnings.push('Agent is currently busy with other tasks')
  }
  
  if (props.confidence < 0.6) {
    warnings.push('Low assignment confidence - consider manual review')
  }
  
  return warnings
})

const hasWarnings = computed(() => warnings.value.length > 0)

// Methods
const getAgentInitials = (name?: string) => {
  if (!name) return 'AG'
  return name
    .split(' ')
    .map(n => n.charAt(0))
    .join('')
    .toUpperCase()
    .slice(0, 2)
}

const formatTaskType = (type?: string) => {
  if (!type) return 'Unknown'
  return type
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

const formatAgentType = (type?: string) => {
  if (!type) return 'Unknown'
  return type
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

const getPriorityClass = (priority?: string) => {
  const classes = {
    'CRITICAL': 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 border border-red-200 dark:border-red-800',
    'HIGH': 'bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400 border border-orange-200 dark:border-orange-800',
    'MEDIUM': 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400 border border-yellow-200 dark:border-yellow-800',
    'LOW': 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 border border-green-200 dark:border-green-800'
  }
  return classes[priority] || classes.MEDIUM
}

const getStatusIndicatorClass = (status?: string) => {
  const classes = {
    'active': 'bg-green-500',
    'idle': 'bg-yellow-500',
    'busy': 'bg-blue-500',
    'sleeping': 'bg-gray-500',
    'error': 'bg-red-500'
  }
  return classes[status] || classes.active
}

const getStatusText = (status?: string) => {
  const texts = {
    'active': 'Active',
    'idle': 'Idle',
    'busy': 'Busy',
    'sleeping': 'Sleeping',
    'error': 'Error'
  }
  return texts[status] || 'Unknown'
}

const getWorkloadColor = (workload?: number) => {
  if (!workload) return 'text-slate-400'
  if (workload >= 0.9) return 'text-red-500'
  if (workload >= 0.7) return 'text-orange-500'
  if (workload >= 0.5) return 'text-yellow-500'
  return 'text-green-500'
}

const getConfidenceColor = (confidence: number) => {
  if (confidence >= 0.8) return 'text-green-600 dark:text-green-400'
  if (confidence >= 0.6) return 'text-blue-600 dark:text-blue-400'
  if (confidence >= 0.4) return 'text-yellow-600 dark:text-yellow-400'
  return 'text-red-600 dark:text-red-400'
}

const getConfidenceBarColor = (confidence: number) => {
  if (confidence >= 0.8) return 'bg-green-500'
  if (confidence >= 0.6) return 'bg-blue-500'
  if (confidence >= 0.4) return 'bg-yellow-500'
  return 'bg-red-500'
}

const getEstimatedCompletion = () => {
  if (!props.task?.estimated_effort_hours) return 'TBD'
  
  const hours = props.task.estimated_effort_hours
  const adjustedHours = hours / Math.max(availabilityScore.value, 0.2) // Adjust for availability
  const completionDate = addHours(new Date(), adjustedHours)
  
  return formatDistanceToNow(completionDate, { addSuffix: true })
}

const getWorkloadChangeColor = (impact: number) => {
  if (impact >= 0.5) return 'text-red-600 dark:text-red-400'
  if (impact >= 0.3) return 'text-orange-600 dark:text-orange-400'
  return 'text-green-600 dark:text-green-400'
}

const getWorkloadChangeText = (impact: number) => {
  const percentage = Math.round(impact * 100)
  if (percentage >= 50) return `+${percentage}% (High)`
  if (percentage >= 30) return `+${percentage}% (Medium)`
  return `+${percentage}% (Low)`
}
</script>

<style scoped>
.modal-backdrop {
  backdrop-filter: blur(4px);
}

.glass-card {
  @apply bg-white/90 dark:bg-slate-800/90 backdrop-blur-lg border border-slate-200/50 dark:border-slate-700/50 shadow-xl;
}

.section-title {
  @apply text-sm font-semibold text-slate-900 dark:text-white mb-3;
}

.info-card {
  @apply bg-slate-50 dark:bg-slate-800 rounded-lg p-4 border border-slate-200 dark:border-slate-700;
}

.metadata-label {
  @apply text-slate-500 dark:text-slate-400;
}

.metadata-value {
  @apply ml-1 font-medium text-slate-700 dark:text-slate-300;
}

.capability-tag {
  @apply transition-all duration-150 hover:scale-105;
}

.workload-bg {
  @apply text-slate-200 dark:text-slate-700;
}

.factor-label {
  @apply text-slate-600 dark:text-slate-400;
}

.factor-score {
  @apply text-slate-700 dark:text-slate-300 font-medium text-xs;
}

.impact-label {
  @apply text-slate-500 dark:text-slate-400;
}

.impact-value {
  @apply text-slate-700 dark:text-slate-300;
}

.btn-primary {
  @apply bg-primary-600 hover:bg-primary-700 text-white px-4 py-2 rounded-md font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 flex items-center;
}

.btn-warning {
  @apply bg-orange-600 hover:bg-orange-700 focus:ring-orange-500;
}

.btn-secondary {
  @apply bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300 px-4 py-2 rounded-md font-medium hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(-10px) scale(0.95);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

.modal-content {
  animation: slideIn 0.2s ease-out;
}
</style>