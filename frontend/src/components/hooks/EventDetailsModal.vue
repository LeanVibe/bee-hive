<template>
  <div v-if="isVisible" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" @click.self="close">
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-4xl max-h-[90vh] overflow-hidden" @click.stop>
      <!-- Header -->
      <div class="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
        <h2 class="text-xl font-semibold text-gray-900 dark:text-gray-100">
          Event Details
        </h2>
        <button
          @click="close"
          class="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
          </svg>
        </button>
      </div>
      
      <!-- Content -->
      <div class="p-6 max-h-[calc(90vh-8rem)] overflow-y-auto">
        <div v-if="event">
          <!-- Basic Event Info -->
          <div class="mb-6">
            <h3 class="text-lg font-medium text-gray-900 dark:text-gray-100 mb-3">
              Basic Information
            </h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <span class="text-sm font-medium text-gray-500 dark:text-gray-400">Event Type</span>
                <p class="text-sm text-gray-900 dark:text-gray-100 mt-1">{{ event.hook_type || 'Unknown' }}</p>
              </div>
              <div>
                <span class="text-sm font-medium text-gray-500 dark:text-gray-400">Agent ID</span>
                <p class="text-sm text-gray-900 dark:text-gray-100 mt-1">{{ event.agent_id || 'N/A' }}</p>
              </div>
              <div>
                <span class="text-sm font-medium text-gray-500 dark:text-gray-400">Session ID</span>
                <p class="text-sm text-gray-900 dark:text-gray-100 mt-1">{{ event.session_id || 'N/A' }}</p>
              </div>
              <div>
                <span class="text-sm font-medium text-gray-500 dark:text-gray-400">Timestamp</span>
                <p class="text-sm text-gray-900 dark:text-gray-100 mt-1">{{ event.timestamp || 'N/A' }}</p>
              </div>
            </div>
          </div>

          <!-- Event Payload -->
          <div class="mb-6">
            <h3 class="text-lg font-medium text-gray-900 dark:text-gray-100 mb-3">
              Event Payload
            </h3>
            <div class="bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
              <pre class="text-xs font-mono text-gray-700 dark:text-gray-300 whitespace-pre-wrap">{{
                formatPayload(event.payload)
              }}</pre>
            </div>
          </div>

          <!-- Metadata -->
          <div v-if="event.metadata && Object.keys(event.metadata).length > 0" class="mb-6">
            <h3 class="text-lg font-medium text-gray-900 dark:text-gray-100 mb-3">
              Metadata
            </h3>
            <div class="bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
              <pre class="text-xs font-mono text-gray-700 dark:text-gray-300 whitespace-pre-wrap">{{
                formatPayload(event.metadata)
              }}</pre>
            </div>
          </div>
        </div>
        
        <!-- No event selected -->
        <div v-else class="text-center py-12">
          <div class="w-12 h-12 text-gray-400 mx-auto mb-3">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.316 16.5c-.77.833.192 2.5 1.732 2.5z"></path>
            </svg>
          </div>
          <h4 class="text-lg font-medium text-gray-900 dark:text-gray-100">
            No Event Selected
          </h4>
          <p class="text-gray-500 dark:text-gray-400">
            Select an event to view its details
          </p>
        </div>
      </div>
      
      <!-- Footer -->
      <div class="flex items-center justify-end space-x-3 p-6 border-t border-gray-200 dark:border-gray-700">
        <button
          @click="close"
          class="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-600 transition-colors"
        >
          Close
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Props {
  isVisible: boolean
  event?: any
  showSecurityInfo?: boolean
  showPerformanceInfo?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  showSecurityInfo: true,
  showPerformanceInfo: true
})

const emit = defineEmits<{
  close: []
}>()

const formatPayload = (payload: any) => {
  if (!payload) return 'No data'
  
  try {
    return JSON.stringify(payload, null, 2)
  } catch {
    return String(payload)
  }
}

const close = () => {
  emit('close')
}
</script>

<style scoped>
/* Modal styles */
</style>