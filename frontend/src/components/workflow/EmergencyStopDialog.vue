<template>
  <div class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
    <div class="bg-white dark:bg-slate-800 rounded-lg shadow-xl max-w-md w-full mx-4 p-6">
      <!-- Header -->
      <div class="flex items-center mb-4">
        <ExclamationTriangleIcon class="w-8 h-8 text-red-500 mr-3" />
        <h3 class="text-lg font-bold text-red-600 dark:text-red-400">
          EMERGENCY STOP CONFIRMATION
        </h3>
      </div>
      
      <!-- Warning Message -->
      <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-4">
        <p class="text-red-800 dark:text-red-200 font-medium mb-2">
          This action will immediately terminate the workflow execution!
        </p>
        <ul class="text-sm text-red-700 dark:text-red-300 space-y-1">
          <li>• All running tasks will be forcefully stopped</li>
          <li>• Partial work may be lost</li>
          <li>• Agents will be immediately disconnected</li>
          <li>• This action cannot be undone</li>
        </ul>
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
            <span class="text-slate-600 dark:text-slate-400">Progress:</span>
            <span class="font-medium">{{ Math.round(execution?.progress || 0) }}%</span>
          </div>
          <div class="flex justify-between">
            <span class="text-slate-600 dark:text-slate-400">Active Agents:</span>
            <span class="font-medium">{{ execution?.agentAssignments.length || 0 }}</span>
          </div>
        </div>
      </div>
      
      <!-- Reason Input -->
      <div class="mb-6">
        <label class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
          Reason for Emergency Stop (Required)
        </label>
        <textarea
          v-model="reason"
          class="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-white focus:ring-2 focus:ring-red-500 focus:border-transparent resize-none"
          rows="3"
          placeholder="Describe why an emergency stop is necessary..."
          required
        ></textarea>
        <p class="mt-1 text-xs text-slate-500 dark:text-slate-400">
          This reason will be logged for auditing purposes.
        </p>
      </div>
      
      <!-- Security Confirmation -->
      <div class="mb-6">
        <label class="flex items-start space-x-3">
          <input
            v-model="confirmed"
            type="checkbox"
            class="mt-1 w-4 h-4 text-red-600 bg-gray-100 border-gray-300 rounded focus:ring-red-500 dark:focus:ring-red-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
          />
          <span class="text-sm text-slate-700 dark:text-slate-300">
            I understand the consequences and confirm that I want to perform an emergency stop of this workflow execution.
          </span>
        </label>
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
          @click="confirmEmergencyStop"
          :disabled="!canConfirm"
          class="px-6 py-2 bg-red-600 hover:bg-red-700 disabled:bg-red-400 text-white rounded-md font-bold transition-colors border-2 border-red-700"
        >
          EMERGENCY STOP
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { ExclamationTriangleIcon } from '@heroicons/vue/24/outline'
import type { WorkflowExecution } from '@/types/workflows'

interface Props {
  execution: WorkflowExecution | null
}

const props = defineProps<Props>()

const emit = defineEmits<{
  confirm: [reason: string]
  cancel: []
}>()

const reason = ref('')
const confirmed = ref(false)

const canConfirm = computed(() => 
  reason.value.trim().length > 10 && confirmed.value
)

const confirmEmergencyStop = () => {
  if (canConfirm.value) {
    emit('confirm', reason.value.trim())
  }
}
</script>