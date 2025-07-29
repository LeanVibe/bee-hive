<template>
  <div class="workflow-control-panel">
    <!-- Control Panel Header -->
    <div class="control-header bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 p-4">
      <div class="flex items-center justify-between">
        <div>
          <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
            Workflow Control Center
          </h3>
          <p class="text-sm text-slate-600 dark:text-slate-400 mt-1">
            {{ selectedExecution ? 'Managing active execution' : 'No execution selected' }}
          </p>
        </div>
        
        <!-- System Status -->
        <div class="flex items-center space-x-4">
          <div class="flex items-center space-x-2">
            <div 
              class="w-2 h-2 rounded-full"
              :class="systemHealthClass"
            ></div>
            <span class="text-sm font-medium">{{ systemStatus }}</span>
          </div>
          
          <div class="text-sm text-slate-500 dark:text-slate-400">
            {{ activeExecutions }} active executions
          </div>
        </div>
      </div>
    </div>

    <!-- Execution Selection -->
    <div class="execution-selector bg-slate-50 dark:bg-slate-900 p-4 border-b border-slate-200 dark:border-slate-700">
      <div class="flex items-center space-x-4">
        <div class="flex-1">
          <label class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
            Select Execution
          </label>
          <select
            v-model="selectedExecutionId"
            @change="onExecutionSelect"
            class="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="">Select an execution...</option>
            <option 
              v-for="execution in availableExecutions" 
              :key="execution.id" 
              :value="execution.id"
            >
              {{ execution.commandName }} - {{ execution.status }} ({{ formatTime(execution.startTime) }})
            </option>
          </select>
        </div>
        
        <button
          @click="refreshExecutions"
          :disabled="isRefreshing"
          class="px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-primary-400 text-white rounded-md font-medium transition-colors flex items-center space-x-2"
        >
          <ArrowPathIcon 
            class="w-4 h-4" 
            :class="{ 'animate-spin': isRefreshing }"
          />
          <span>Refresh</span>
        </button>
      </div>
    </div>

    <!-- Main Control Interface -->
    <div v-if="selectedExecution" class="control-interface flex-1 overflow-hidden">
      <!-- Control Actions -->
      <div class="control-actions bg-white dark:bg-slate-800 p-4 border-b border-slate-200 dark:border-slate-700">
        <div class="flex items-center justify-between">
          <div class="flex items-center space-x-3">
            <!-- Primary Controls -->
            <button
              v-for="action in primaryActions"
              :key="action.id"
              @click="executeAction(action)"
              :disabled="action.disabled || isProcessingAction"
              :class="[
                'px-4 py-2 rounded-md font-medium transition-colors flex items-center space-x-2',
                action.style === 'primary' 
                  ? 'bg-primary-600 hover:bg-primary-700 disabled:bg-primary-400 text-white'
                  : action.style === 'danger'
                  ? 'bg-red-600 hover:bg-red-700 disabled:bg-red-400 text-white'
                  : action.style === 'warning'
                  ? 'bg-yellow-600 hover:bg-yellow-700 disabled:bg-yellow-400 text-white'
                  : 'bg-slate-200 hover:bg-slate-300 disabled:bg-slate-100 text-slate-700 dark:bg-slate-600 dark:hover:bg-slate-500 dark:disabled:bg-slate-700 dark:text-slate-200'
              ]"
              :title="action.tooltip"
            >
              <component :is="action.icon" class="w-4 h-4" />
              <span>{{ action.label }}</span>
            </button>
          </div>
          
          <!-- Emergency Stop -->
          <button
            @click="showEmergencyStopDialog = true"
            :disabled="!canEmergencyStop"
            class="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-red-400 text-white rounded-md font-bold transition-colors flex items-center space-x-2 border-2 border-red-700"
          >
            <ExclamationTriangleIcon class="w-4 h-4" />
            <span>EMERGENCY STOP</span>
          </button>
        </div>
        
        <!-- Execution Progress -->
        <div class="mt-4">
          <div class="flex items-center justify-between text-sm mb-2">
            <span class="font-medium text-slate-700 dark:text-slate-300">
              Progress: {{ selectedExecution.currentStep || 'Initializing...' }}
            </span>
            <span class="text-slate-500 dark:text-slate-400">
              {{ Math.round(selectedExecution.progress || 0) }}%
            </span>
          </div>
          <div class="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
            <div 
              class="bg-primary-600 h-2 rounded-full transition-all duration-300"
              :style="{ width: `${selectedExecution.progress || 0}%` }"
            ></div>
          </div>
        </div>
      </div>

      <!-- Tabs Interface -->
      <div class="control-tabs bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700">
        <nav class="flex space-x-8 px-4" aria-label="Tabs">
          <button
            v-for="tab in controlTabs"
            :key="tab.id"
            @click="activeTab = tab.id"
            :class="[
              'whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm',
              activeTab === tab.id
                ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300 dark:text-slate-400 dark:hover:text-slate-300'
            ]"
          >
            <component :is="tab.icon" class="w-4 h-4 inline mr-2" />
            {{ tab.label }}
            <span 
              v-if="tab.badge" 
              class="ml-2 px-2 py-0.5 text-xs bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-400 rounded-full"
            >
              {{ tab.badge }}
            </span>
          </button>
        </nav>
      </div>

      <!-- Tab Content -->
      <div class="tab-content flex-1 overflow-auto">
        <!-- Execution Status Tab -->
        <div v-show="activeTab === 'status'" class="p-4 space-y-4">
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <!-- Execution Details -->
            <div class="bg-slate-50 dark:bg-slate-900 rounded-lg p-4">
              <h4 class="font-semibold text-slate-900 dark:text-white mb-3">Execution Details</h4>
              <div class="space-y-2 text-sm">
                <div class="flex justify-between">
                  <span class="text-slate-600 dark:text-slate-400">Command:</span>
                  <span class="font-medium">{{ selectedExecution.commandName }}</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-slate-600 dark:text-slate-400">Status:</span>
                  <span 
                    class="px-2 py-1 rounded-full text-xs font-medium"
                    :class="getStatusClass(selectedExecution.status)"
                  >
                    {{ selectedExecution.status }}
                  </span>
                </div>
                <div class="flex justify-between">
                  <span class="text-slate-600 dark:text-slate-400">Started:</span>
                  <span>{{ formatTime(selectedExecution.startTime) }}</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-slate-600 dark:text-slate-400">Duration:</span>
                  <span>{{ formatDuration(getExecutionDuration()) }}</span>
                </div>
                <div v-if="selectedExecution.estimatedCompletion" class="flex justify-between">
                  <span class="text-slate-600 dark:text-slate-400">Est. Completion:</span>
                  <span>{{ formatTime(selectedExecution.estimatedCompletion) }}</span>
                </div>
              </div>
            </div>

            <!-- Resource Usage -->
            <div class="bg-slate-50 dark:bg-slate-900 rounded-lg p-4">
              <h4 class="font-semibold text-slate-900 dark:text-white mb-3">Resource Usage</h4>
              <div class="space-y-3">
                <div>
                  <div class="flex justify-between text-sm mb-1">
                    <span class="text-slate-600 dark:text-slate-400">CPU Usage</span>
                    <span>{{ Math.round(resourceUsage.cpu) }}%</span>
                  </div>
                  <div class="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                    <div 
                      class="bg-blue-500 h-2 rounded-full"
                      :style="{ width: `${resourceUsage.cpu}%` }"
                    ></div>
                  </div>
                </div>
                <div>
                  <div class="flex justify-between text-sm mb-1">
                    <span class="text-slate-600 dark:text-slate-400">Memory Usage</span>
                    <span>{{ Math.round(resourceUsage.memory) }}%</span>
                  </div>
                  <div class="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                    <div 
                      class="bg-green-500 h-2 rounded-full"
                      :style="{ width: `${resourceUsage.memory}%` }"
                    ></div>
                  </div>
                </div>
                <div>
                  <div class="flex justify-between text-sm mb-1">
                    <span class="text-slate-600 dark:text-slate-400">Network I/O</span>
                    <span>{{ formatBytes(resourceUsage.networkIO) }}/s</span>
                  </div>
                  <div class="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                    <div 
                      class="bg-yellow-500 h-2 rounded-full"
                      :style="{ width: `${Math.min(resourceUsage.networkIO / 1000000 * 100, 100)}%` }"
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Step Execution Status -->
          <div class="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
            <div class="p-4 border-b border-slate-200 dark:border-slate-700">
              <h4 class="font-semibold text-slate-900 dark:text-white">Step Execution Status</h4>
            </div>
            <div class="divide-y divide-slate-200 dark:divide-slate-700">
              <div
                v-for="step in selectedExecution.steps"
                :key="step.stepId"
                class="p-4 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
              >
                <div class="flex items-center justify-between">
                  <div class="flex items-center space-x-3">
                    <div 
                      class="w-3 h-3 rounded-full"
                      :class="getStatusColor(step.status)"
                    ></div>
                    <div>
                      <div class="font-medium text-slate-900 dark:text-white">
                        {{ step.stepId }}
                      </div>
                      <div class="text-sm text-slate-500 dark:text-slate-400">
                        Agent: {{ step.agentId }}
                      </div>
                    </div>
                  </div>
                  <div class="text-right text-sm">
                    <div class="text-slate-900 dark:text-white">
                      {{ step.status }}
                    </div>
                    <div class="text-slate-500 dark:text-slate-400">
                      {{ step.duration ? formatDuration(step.duration) : '--' }}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Agent Management Tab -->
        <div v-show="activeTab === 'agents'" class="p-4">
          <AgentManagementPanel 
            :execution-id="selectedExecution.id"
            :agent-assignments="selectedExecution.agentAssignments"
            @reassign-task="handleTaskReassignment"
          />
        </div>

        <!-- Logs Tab -->
        <div v-show="activeTab === 'logs'" class="p-4">
          <LogViewer 
            :logs="selectedExecution.logs"
            :auto-scroll="true"
            :max-lines="1000"
          />
        </div>

        <!-- Performance Tab -->
        <div v-show="activeTab === 'performance'" class="p-4">
          <PerformanceMonitor 
            :execution-id="selectedExecution.id"
            :metrics="selectedExecution.metrics"
          />
        </div>
      </div>
    </div>

    <!-- Empty State -->
    <div v-else class="empty-state flex-1 flex items-center justify-center">
      <div class="text-center">
        <CogIcon class="w-16 h-16 text-slate-400 mx-auto mb-4" />
        <h3 class="text-lg font-medium text-slate-900 dark:text-white mb-2">
          No Execution Selected
        </h3>
        <p class="text-slate-600 dark:text-slate-400 mb-4">
          Select an active execution from the dropdown above to begin monitoring and control.
        </p>
        <button
          @click="refreshExecutions"
          class="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-md font-medium transition-colors"
        >
          Refresh Executions
        </button>
      </div>
    </div>

    <!-- Emergency Stop Confirmation Dialog -->
    <EmergencyStopDialog
      v-if="showEmergencyStopDialog"
      :execution="selectedExecution"
      @confirm="handleEmergencyStop"
      @cancel="showEmergencyStopDialog = false"
    />

    <!-- Action Confirmation Dialog -->
    <ActionConfirmationDialog
      v-if="showActionDialog"
      :action="pendingAction"
      :execution="selectedExecution"
      @confirm="confirmAction"
      @cancel="cancelAction"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { format, formatDistanceToNow } from 'date-fns'
import {
  ArrowPathIcon,
  PlayIcon,
  PauseIcon,
  StopIcon,
  ExclamationTriangleIcon,
  CogIcon,
  ChartBarIcon,
  ClockIcon,
  UserGroupIcon,
  DocumentTextIcon,
  BoltIcon
} from '@heroicons/vue/24/outline'

import type { 
  WorkflowExecution, 
  ExecutionStatus,
  WorkflowControlAction,
  ControlActionType
} from '@/types/workflows'
import { useWorkflowStore } from '@/stores/workflows'
import { commandManagementService } from '@/services/commandManagementService'
import { useUnifiedWebSocket } from '@/services/unifiedWebSocketManager'

// Child Components (would be implemented)
import AgentManagementPanel from './AgentManagementPanel.vue'
import LogViewer from './LogViewer.vue'
import PerformanceMonitor from './PerformanceMonitor.vue'
import EmergencyStopDialog from './EmergencyStopDialog.vue'
import ActionConfirmationDialog from './ActionConfirmationDialog.vue'

// Store
const workflowStore = useWorkflowStore()
const webSocket = useUnifiedWebSocket()

// State
const selectedExecutionId = ref<string>('')
const activeTab = ref('status')
const isRefreshing = ref(false)
const isProcessingAction = ref(false)
const showEmergencyStopDialog = ref(false)
const showActionDialog = ref(false)
const pendingAction = ref<WorkflowControlAction | null>(null)

// Mock resource usage data
const resourceUsage = ref({
  cpu: 35,
  memory: 58,
  networkIO: 1024000
})

// Computed Properties
const availableExecutions = computed(() => 
  workflowStore.executions.filter(e => ['running', 'paused'].includes(e.status))
)

const selectedExecution = computed(() => 
  selectedExecutionId.value 
    ? workflowStore.state.executions.get(selectedExecutionId.value) 
    : null
)

const activeExecutions = computed(() => 
  workflowStore.runningExecutions.length
)

const systemStatus = computed(() => {
  const health = 'healthy' // Would come from system metrics
  return health === 'healthy' ? 'System Healthy' : 'System Degraded'
})

const systemHealthClass = computed(() => 
  systemStatus.value.includes('Healthy') 
    ? 'bg-green-500 animate-pulse' 
    : 'bg-yellow-500'
)

const canEmergencyStop = computed(() => 
  selectedExecution.value && ['running', 'paused'].includes(selectedExecution.value.status)
)

const primaryActions = computed(() => {
  if (!selectedExecution.value) return []
  
  const execution = selectedExecution.value
  const actions = []
  
  if (execution.status === 'running') {
    actions.push({
      id: 'pause',
      label: 'Pause',
      icon: PauseIcon,
      style: 'secondary',
      disabled: false,
      tooltip: 'Pause execution'
    })
    actions.push({
      id: 'stop',
      label: 'Stop',
      icon: StopIcon,
      style: 'danger',
      disabled: false,
      tooltip: 'Stop execution gracefully'
    })
  } else if (execution.status === 'paused') {
    actions.push({
      id: 'resume',
      label: 'Resume',
      icon: PlayIcon,
      style: 'primary',
      disabled: false,
      tooltip: 'Resume execution'
    })
    actions.push({
      id: 'stop',
      label: 'Stop',
      icon: StopIcon,
      style: 'danger',
      disabled: false,
      tooltip: 'Stop execution'
    })
  }
  
  return actions
})

const controlTabs = computed(() => [
  {
    id: 'status',
    label: 'Status',
    icon: ClockIcon,
    badge: null
  },
  {
    id: 'agents',
    label: 'Agents',
    icon: UserGroupIcon,
    badge: selectedExecution.value?.agentAssignments.length || null
  },
  {
    id: 'logs',
    label: 'Logs',
    icon: DocumentTextIcon,
    badge: selectedExecution.value?.logs.length || null
  },
  {
    id: 'performance',
    label: 'Performance',
    icon: ChartBarIcon,
    badge: null
  }
])

// Methods
const onExecutionSelect = (): void => {
  // Subscribe to real-time updates for the selected execution
  if (selectedExecutionId.value) {
    subscribeToExecutionUpdates(selectedExecutionId.value)
  }
}

const refreshExecutions = async (): Promise<void> => {
  try {
    isRefreshing.value = true
    
    // Load active executions
    const executions = await commandManagementService.listActiveExecutions()
    
    // Update store
    executions.forEach(execution => {
      workflowStore.state.executions.set(execution.id, execution)
    })
    
  } catch (error) {
    console.error('Failed to refresh executions:', error)
  } finally {
    isRefreshing.value = false
  }
}

const executeAction = (action: any): void => {
  if (action.disabled || !selectedExecution.value) return
  
  // Show confirmation for destructive actions
  if (['stop', 'emergency_stop'].includes(action.id)) {
    pendingAction.value = {
      type: action.id as ControlActionType,
      executionId: selectedExecution.value.id,
      reason: `User requested ${action.label.toLowerCase()}`
    }
    showActionDialog.value = true
  } else {
    // Execute immediately for non-destructive actions
    performAction({
      type: action.id as ControlActionType,
      executionId: selectedExecution.value.id
    })
  }
}

const performAction = async (action: WorkflowControlAction): Promise<void> => {
  if (!selectedExecution.value) return
  
  try {
    isProcessingAction.value = true
    
    switch (action.type) {
      case 'pause':
        // Would implement pause logic
        console.log('Pausing execution:', action.executionId)
        break
        
      case 'resume':
        // Would implement resume logic
        console.log('Resuming execution:', action.executionId)
        break
        
      case 'stop':
      case 'cancel':
        await commandManagementService.cancelExecution(
          action.executionId!,
          action.reason || 'User requested stop'
        )
        break
        
      case 'emergency_stop':
        // Would implement emergency stop with immediate termination
        await commandManagementService.cancelExecution(
          action.executionId!,
          'EMERGENCY STOP: ' + (action.reason || 'Immediate termination requested')
        )
        break
    }
    
    // Refresh execution status
    await refreshExecutions()
    
  } catch (error) {
    console.error('Failed to execute action:', error)
    // Show error notification
  } finally {
    isProcessingAction.value = false
  }
}

const handleEmergencyStop = async (reason: string): Promise<void> => {
  if (!selectedExecution.value) return
  
  await performAction({
    type: 'emergency_stop',
    executionId: selectedExecution.value.id,
    reason,
    confirmation: true
  })
  
  showEmergencyStopDialog.value = false
}

const confirmAction = async (): Promise<void> => {
  if (!pendingAction.value) return
  
  await performAction(pendingAction.value)
  
  showActionDialog.value = false
  pendingAction.value = null
}

const cancelAction = (): void => {
  showActionDialog.value = false
  pendingAction.value = null
}

const handleTaskReassignment = async (taskId: string, newAgentId: string, reason: string): Promise<void> => {
  // Would implement task reassignment
  console.log('Reassigning task:', { taskId, newAgentId, reason })
}

// Real-time Updates
const subscribeToExecutionUpdates = (executionId: string): void => {
  webSocket.onMessage('execution_status_update', (message) => {
    if (message.data.execution_id === executionId) {
      updateExecutionStatus(message.data)
    }
  })
  
  webSocket.onMessage('resource_usage_update', (message) => {
    if (message.data.execution_id === executionId) {
      resourceUsage.value = message.data.usage
    }
  })
}

const updateExecutionStatus = (update: any): void => {
  const execution = workflowStore.state.executions.get(update.execution_id)
  if (execution) {
    execution.status = update.status
    execution.progress = update.progress
    execution.currentStep = update.current_step
    
    if (update.step_results) {
      execution.steps = update.step_results
    }
    
    if (update.logs) {
      execution.logs.push(...update.logs)
    }
  }
}

// Utility Methods
const formatTime = (date: Date | null): string => {
  if (!date) return '--'
  return format(date, 'MMM dd, HH:mm:ss')
}

const formatDuration = (seconds: number): string => {
  if (seconds < 60) return `${seconds}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
}

const formatBytes = (bytes: number): string => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

const getExecutionDuration = (): number => {
  if (!selectedExecution.value) return 0
  
  const start = selectedExecution.value.startTime.getTime()
  const end = selectedExecution.value.endTime?.getTime() || Date.now()
  
  return Math.floor((end - start) / 1000)
}

const getStatusClass = (status: ExecutionStatus): string => {
  const classes = {
    queued: 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200',
    starting: 'bg-blue-100 text-blue-800 dark:bg-blue-800 dark:text-blue-200',
    running: 'bg-blue-100 text-blue-800 dark:bg-blue-800 dark:text-blue-200',
    paused: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-800 dark:text-yellow-200',
    completed: 'bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-200',
    failed: 'bg-red-100 text-red-800 dark:bg-red-800 dark:text-red-200',
    cancelled: 'bg-orange-100 text-orange-800 dark:bg-orange-800 dark:text-orange-200',
    timeout: 'bg-red-100 text-red-800 dark:bg-red-800 dark:text-red-200'
  }
  return classes[status] || classes.queued
}

const getStatusColor = (status: ExecutionStatus): string => {
  const colors = {
    queued: 'bg-gray-400',
    starting: 'bg-blue-400',
    running: 'bg-blue-500',
    paused: 'bg-yellow-500',
    completed: 'bg-green-500',
    failed: 'bg-red-500',
    cancelled: 'bg-orange-500',
    timeout: 'bg-red-500'
  }
  return colors[status] || colors.queued
}

// Lifecycle
onMounted(async () => {
  await refreshExecutions()
  
  // Auto-select first available execution
  if (availableExecutions.value.length > 0 && !selectedExecutionId.value) {
    selectedExecutionId.value = availableExecutions.value[0].id
    onExecutionSelect()
  }
})

// Watchers
watch(() => selectedExecutionId.value, (newId) => {
  if (newId) {
    onExecutionSelect()
  }
})
</script>

<style scoped>
.workflow-control-panel {
  @apply flex flex-col h-full bg-white dark:bg-slate-800 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700;
}

.control-header {
  @apply shrink-0;
}

.execution-selector {
  @apply shrink-0;
}

.control-interface {
  @apply flex flex-col min-h-0;
}

.control-actions {
  @apply shrink-0;
}

.control-tabs {
  @apply shrink-0;
}

.tab-content {
  @apply min-h-0;
}

.empty-state {
  @apply bg-slate-50 dark:bg-slate-900;
}

/* Progress bar animation */
.progress-bar {
  @apply transition-all duration-300 ease-out;
}

/* Status indicator animations */
@keyframes pulse-green {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

@keyframes pulse-blue {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

.bg-green-500.animate-pulse {
  animation: pulse-green 2s ease-in-out infinite;
}

.bg-blue-500.animate-pulse {
  animation: pulse-blue 2s ease-in-out infinite;
}

/* Emergency button styling */
button:has(.text-red-600) {
  @apply ring-2 ring-red-200 dark:ring-red-800;
}

button:has(.text-red-600):hover {
  @apply ring-red-300 dark:ring-red-700 transform scale-105;
}

/* Tab transition */
.tab-content > div {
  @apply transition-opacity duration-200;
}

/* Resource usage bars */
.bg-blue-500,
.bg-green-500,
.bg-yellow-500 {
  @apply transition-all duration-500 ease-out;
}
</style>