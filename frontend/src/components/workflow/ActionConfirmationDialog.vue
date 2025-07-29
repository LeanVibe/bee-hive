<template>
  <div class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
    <div class="bg-white dark:bg-slate-800 rounded-lg shadow-xl max-w-md w-full mx-4 p-6">
      <!-- Header -->
      <div class="flex items-center mb-4">
        <component 
          :is="actionIcon" 
          class="w-6 h-6 mr-3"
          :class="actionIconClass"
        />
        <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
          Confirm {{ actionLabel }}
        </h3>
      </div>
      
      <!-- Description -->
      <div class="mb-4">
        <p class="text-slate-600 dark:text-slate-400">
          {{ actionDescription }}
        </p>
      </div>
      
      <!-- Execution Details -->
      <div class="bg-slate-50 dark:bg-slate-900 rounded-lg p-4 mb-4">
        <h4 class="font-medium text-slate-900 dark:text-white mb-2">
          Execution Details
        </h4>
        <div class="text-sm space-y-1">
          <div class="flex justify-between">
            <span class="text-slate-600 dark:text-slate-400">Command:</span>
            <span class="font-medium">{{ execution?.commandName }}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-slate-600 dark:text-slate-400">Status:</span>
            <span 
              class="px-2 py-1 rounded-full text-xs font-medium"
              :class="getStatusClass(execution?.status)"
            >
              {{ execution?.status }}
            </span>
          </div>
          <div class="flex justify-between">
            <span class="text-slate-600 dark:text-slate-400">Progress:</span>
            <span class="font-medium">{{ Math.round(execution?.progress || 0) }}%</span>
          </div>
        </div>
      </div>
      
      <!-- Reason Input (for destructive actions) -->
      <div v-if="requiresReason" class="mb-6">
        <label class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
          Reason (Optional)
        </label>
        <input
          v-model="reason"
          type="text"
          class="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          placeholder="Describe the reason for this action..."
        />
      </div>
      
      <!-- Actions -->
      <div class="flex justify-end space-x-3">
        <button
          @click="$emit('cancel')"
          class="px-4 py-2 border border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300 rounded-md hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
        >
          Cancel
        </button>
        <button
          @click="confirmAction"
          class="px-4 py-2 rounded-md font-medium transition-colors"
          :class="confirmButtonClass"
        >
          {{ actionLabel }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { 
  PlayIcon, 
  PauseIcon, 
  StopIcon,
  ExclamationTriangleIcon
} from '@heroicons/vue/24/outline'
import type { WorkflowExecution, WorkflowControlAction, ExecutionStatus } from '@/types/workflows'

interface Props {
  action: WorkflowControlAction | null
  execution: WorkflowExecution | null
}

const props = defineProps<Props>()

const emit = defineEmits<{
  confirm: []
  cancel: []
}>()

const reason = ref('')

const actionConfig = computed(() => {
  if (!props.action) return null
  
  const configs = {
    pause: {
      label: 'Pause Execution',
      description: 'This will pause the workflow execution. All running tasks will be suspended and can be resumed later.',
      icon: PauseIcon,
      iconClass: 'text-yellow-500',
      buttonClass: 'bg-yellow-600 hover:bg-yellow-700 text-white',
      requiresReason: false
    },
    resume: {
      label: 'Resume Execution',
      description: 'This will resume the paused workflow execution. All suspended tasks will continue from where they left off.',
      icon: PlayIcon,
      iconClass: 'text-green-500',
      buttonClass: 'bg-green-600 hover:bg-green-700 text-white',
      requiresReason: false
    },
    stop: {
      label: 'Stop Execution',
      description: 'This will gracefully stop the workflow execution. Running tasks will be allowed to complete, but no new tasks will be started.',
      icon: StopIcon,
      iconClass: 'text-red-500',
      buttonClass: 'bg-red-600 hover:bg-red-700 text-white',
      requiresReason: true
    },
    cancel: {
      label: 'Cancel Execution',
      description: 'This will immediately cancel the workflow execution. All running tasks will be stopped.',
      icon: ExclamationTriangleIcon,
      iconClass: 'text-red-500',
      buttonClass: 'bg-red-600 hover:bg-red-700 text-white',
      requiresReason: true
    }
  }
  
  return configs[props.action.type] || configs.stop
})

const actionLabel = computed(() => actionConfig.value?.label || 'Confirm Action')
const actionDescription = computed(() => actionConfig.value?.description || '')
const actionIcon = computed(() => actionConfig.value?.icon || StopIcon)
const actionIconClass = computed(() => actionConfig.value?.iconClass || 'text-slate-500')
const confirmButtonClass = computed(() => actionConfig.value?.buttonClass || 'bg-primary-600 hover:bg-primary-700 text-white')
const requiresReason = computed(() => actionConfig.value?.requiresReason || false)

const confirmAction = () => {
  if (props.action && reason.value.trim()) {
    props.action.reason = reason.value.trim()
  }
  emit('confirm')
}

const getStatusClass = (status: ExecutionStatus | undefined): string => {
  if (!status) return ''
  
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
</script>