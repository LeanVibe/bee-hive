<template>
  <div class="error-boundary">
    <!-- Error State -->
    <div v-if="hasError" class="error-display">
      <!-- Critical Error -->
      <div v-if="error?.type === 'network' || error?.type === 'websocket'" class="critical-error">
        <div class="glass-card rounded-xl p-6 border-l-4 border-red-500 bg-red-50 dark:bg-red-900/20">
          <div class="flex items-start">
            <ExclamationTriangleIcon class="w-6 h-6 text-red-600 dark:text-red-400 mt-1 mr-3 flex-shrink-0" />
            <div class="flex-1">
              <h3 class="text-lg font-semibold text-red-900 dark:text-red-100 mb-2">
                Connection Error
              </h3>
              <p class="text-red-700 dark:text-red-300 mb-4">
                {{ error.message }}
              </p>
              
              <!-- Recovery Actions -->
              <div class="flex items-center space-x-3">
                <button
                  @click="retry"
                  :disabled="isRetrying"
                  class="btn-error flex items-center"
                  :class="{ 'opacity-50 cursor-not-allowed': isRetrying }"
                >
                  <ArrowPathIcon 
                    class="w-4 h-4 mr-2" 
                    :class="{ 'animate-spin': isRetrying }"
                  />
                  {{ isRetrying ? 'Retrying...' : 'Retry Connection' }}
                </button>
                
                <button
                  v-if="canDismiss"
                  @click="dismiss"
                  class="btn-secondary text-sm"
                >
                  Dismiss
                </button>
                
                <span class="text-sm text-red-600 dark:text-red-400">
                  Attempt {{ retryCount }}/{{ maxRetries }}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Data Error -->
      <div v-else-if="error?.type === 'data' || error?.type === 'parsing'" class="data-error">
        <div class="glass-card rounded-xl p-6 border-l-4 border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20">
          <div class="flex items-start">
            <ExclamationTriangleIcon class="w-6 h-6 text-yellow-600 dark:text-yellow-400 mt-1 mr-3 flex-shrink-0" />
            <div class="flex-1">
              <h3 class="text-lg font-semibold text-yellow-900 dark:text-yellow-100 mb-2">
                Data Error
              </h3>
              <p class="text-yellow-700 dark:text-yellow-300 mb-4">
                {{ error.message }}
              </p>
              
              <!-- Fallback Actions -->
              <div class="flex items-center space-x-3">
                <button
                  @click="useFallback"
                  class="btn-warning flex items-center"
                >
                  <ShieldCheckIcon class="w-4 h-4 mr-2" />
                  Use Fallback Data
                </button>
                
                <button
                  @click="retry"
                  :disabled="isRetrying"
                  class="btn-secondary text-sm flex items-center"
                  :class="{ 'opacity-50 cursor-not-allowed': isRetrying }"
                >
                  <ArrowPathIcon 
                    class="w-4 h-4 mr-2" 
                    :class="{ 'animate-spin': isRetrying }"
                  />
                  Refresh Data
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Generic Error -->
      <div v-else class="generic-error">
        <div class="glass-card rounded-xl p-6 border-l-4 border-gray-500 bg-gray-50 dark:bg-gray-900/20">
          <div class="flex items-start">
            <ExclamationCircleIcon class="w-6 h-6 text-gray-600 dark:text-gray-400 mt-1 mr-3 flex-shrink-0" />
            <div class="flex-1">
              <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
                Component Error
              </h3>
              <p class="text-gray-700 dark:text-gray-300 mb-4">
                {{ error?.message || 'An unexpected error occurred' }}
              </p>
              
              <!-- Recovery Options -->
              <div class="flex items-center space-x-3">
                <button
                  v-if="fallbackComponent"
                  @click="showFallback"
                  class="btn-secondary flex items-center"
                >
                  <EyeIcon class="w-4 h-4 mr-2" />
                  Show Fallback
                </button>
                
                <button
                  @click="retry"
                  :disabled="isRetrying"
                  class="btn-secondary text-sm flex items-center"
                  :class="{ 'opacity-50 cursor-not-allowed': isRetrying }"
                >
                  <ArrowPathIcon 
                    class="w-4 h-4 mr-2" 
                    :class="{ 'animate-spin': isRetrying }"
                  />
                  Retry
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Error Details (Collapsible) -->
      <div v-if="showDetails" class="error-details mt-4">
        <div class="glass-card rounded-lg p-4 bg-slate-50 dark:bg-slate-800/50">
          <button
            @click="toggleDetails"
            class="flex items-center text-sm text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white mb-3"
          >
            <ChevronDownIcon 
              class="w-4 h-4 mr-1 transition-transform"
              :class="{ 'rotate-180': detailsExpanded }"
            />
            Error Details
          </button>
          
          <div v-if="detailsExpanded" class="space-y-3">
            <div>
              <label class="block text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">
                Error ID
              </label>
              <code class="text-xs font-mono bg-slate-200 dark:bg-slate-700 px-2 py-1 rounded">
                {{ error?.id }}
              </code>
            </div>
            
            <div>
              <label class="block text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">
                Component
              </label>
              <span class="text-xs">{{ error?.component || 'Unknown' }}</span>
            </div>
            
            <div>
              <label class="block text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">
                Timestamp
              </label>
              <span class="text-xs">{{ formatErrorTime(error?.timestamp) }}</span>
            </div>
            
            <div v-if="error?.details">
              <label class="block text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">
                Additional Details
              </label>
              <pre class="text-xs bg-slate-200 dark:bg-slate-700 p-2 rounded overflow-x-auto">{{ JSON.stringify(error.details, null, 2) }}</pre>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Fallback Component -->
    <div v-else-if="showingFallback" class="fallback-display">
      <div class="glass-card rounded-xl p-6 bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500">
        <div class="flex items-start">
          <InformationCircleIcon class="w-6 h-6 text-blue-600 dark:text-blue-400 mt-1 mr-3 flex-shrink-0" />
          <div class="flex-1">
            <h3 class="text-lg font-semibold text-blue-900 dark:text-blue-100 mb-2">
              Fallback Mode
            </h3>
            <p class="text-blue-700 dark:text-blue-300 mb-4">
              Using cached or simplified data while the main component recovers.
            </p>
            
            <div class="flex items-center space-x-3">
              <button
                @click="exitFallback"
                class="btn-info text-sm"
              >
                Try Again
              </button>
              
              <div class="text-sm text-blue-600 dark:text-blue-400">
                <ClockIcon class="w-4 h-4 inline mr-1" />
                Auto-retry in {{ Math.ceil(autoRetryCountdown) }}s
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Fallback Content -->
      <div class="fallback-content mt-4">
        <component 
          v-if="fallbackComponent"
          :is="fallbackComponent"
          :fallback-data="fallbackData"
          v-bind="$attrs"
        />
        <div v-else class="text-center text-slate-500 dark:text-slate-400 py-8">
          <div class="text-4xl mb-4">⚡</div>
          <p>Simplified view is loading...</p>
        </div>
      </div>
    </div>

    <!-- Normal Content -->
    <div v-else class="normal-content">
      <slot />
    </div>

    <!-- Loading Overlay (During Recovery) -->
    <div 
      v-if="isRetrying"
      class="absolute inset-0 bg-white/75 dark:bg-slate-900/75 flex items-center justify-center z-50 rounded-xl"
    >
      <div class="text-center">
        <div class="animate-spin w-8 h-8 border-3 border-primary-600 border-t-transparent rounded-full mx-auto mb-3"></div>
        <p class="text-sm text-slate-600 dark:text-slate-400">
          {{ retryMessage || 'Attempting recovery...' }}
        </p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { format } from 'date-fns'
import {
  ExclamationTriangleIcon,
  ExclamationCircleIcon,
  InformationCircleIcon,
  ArrowPathIcon,
  ShieldCheckIcon,
  EyeIcon, 
  ChevronDownIcon,
  ClockIcon
} from '@heroicons/vue/24/outline'

import { useDashboardErrorHandling } from '@/composables/useDashboardErrorHandling'
import type { DashboardError, DashboardComponent } from '@/types/coordination'

interface Props {
  boundaryId: string
  component: DashboardComponent
  fallbackComponent?: string
  fallbackData?: any
  showDetails?: boolean
  canDismiss?: boolean
  autoRetry?: boolean
  autoRetryInterval?: number
  maxRetries?: number
  onError?: (error: DashboardError) => void
  onRecovery?: (success: boolean) => void
}

const props = withDefaults(defineProps<Props>(), {
  showDetails: true,
  canDismiss: true,
  autoRetry: true,
  autoRetryInterval: 30000, // 30 seconds
  maxRetries: 3
})

const emit = defineEmits<{
  error: [error: DashboardError]
  recovery: [success: boolean]
  fallback: [active: boolean]
}>()

const errorHandler = useDashboardErrorHandling()

// Component state
const isRetrying = ref(false)
const showingFallback = ref(false)
const detailsExpanded = ref(false)
const retryMessage = ref('')
const autoRetryCountdown = ref(0)

// Auto-retry timer
let autoRetryTimer: number | null = null
let countdownTimer: number | null = null

// Error boundary state
const errorBoundary = computed(() => {
  return errorHandler.createErrorBoundary(props.boundaryId)
})

const hasError = computed(() => errorBoundary.value.hasError)
const error = computed(() => errorBoundary.value.error)
const retryCount = computed(() => errorBoundary.value.retryCount)

// Error handling methods
const retry = async () => {
  if (isRetrying.value) return

  isRetrying.value = true
  retryMessage.value = 'Attempting to recover...'

  try {
    const success = await errorHandler.retryErrorBoundary(props.boundaryId)
    
    if (success) {
      showingFallback.value = false
      retryMessage.value = 'Recovery successful!'
      
      // Clear retry message after delay
      setTimeout(() => {
        retryMessage.value = ''
      }, 2000)

      emit('recovery', true)
      props.onRecovery?.(true)
    } else {
      retryMessage.value = 'Recovery failed'
      emit('recovery', false)
      props.onRecovery?.(false)
      
      // Show fallback if available
      if (props.fallbackComponent || props.fallbackData) {
        setTimeout(() => {
          useFallback()
        }, 2000)
      }
    }
  } catch (error) {
    console.error('Retry failed:', error)
    retryMessage.value = 'Retry failed'
    emit('recovery', false)
    props.onRecovery?.(false)
  } finally {
    isRetrying.value = false
    
    // Clear retry message after delay
    setTimeout(() => {
      retryMessage.value = ''
    }, 3000)
  }
}

const useFallback = () => {
  showingFallback.value = true
  emit('fallback', true)
  
  // Start auto-retry countdown
  if (props.autoRetry) {
    startAutoRetryCountdown()
  }
}

const showFallback = () => {
  useFallback()
}

const exitFallback = () => {
  showingFallback.value = false
  emit('fallback', false)
  clearAutoRetryTimer()
  retry()
}

const dismiss = () => {
  if (props.canDismiss) {
    errorHandler.clearErrorBoundary(props.boundaryId)
    clearAutoRetryTimer()
  }
}

const toggleDetails = () => {
  detailsExpanded.value = !detailsExpanded.value
}

const formatErrorTime = (timestamp?: string) => {
  if (!timestamp) return 'Unknown'
  return format(new Date(timestamp), 'MMM dd, HH:mm:ss')
}

// Auto-retry functionality
const startAutoRetryCountdown = () => {
  if (!props.autoRetry) return

  autoRetryCountdown.value = props.autoRetryInterval / 1000
  
  countdownTimer = setInterval(() => {
    autoRetryCountdown.value--
    
    if (autoRetryCountdown.value <= 0) {
      clearInterval(countdownTimer!)
      countdownTimer = null
      
      // Trigger auto-retry
      exitFallback()
    }
  }, 1000)
}

const clearAutoRetryTimer = () => {
  if (countdownTimer) {
    clearInterval(countdownTimer)
    countdownTimer = null
  }
  autoRetryCountdown.value = 0
}

// Error catching for the component tree
const catchError = (error: DashboardError) => {
  console.error(`Error caught in boundary ${props.boundaryId}:`, error)
  
  errorHandler.triggerErrorBoundary(
    props.boundaryId, 
    error, 
    props.fallbackComponent
  )
  
  emit('error', error)
  props.onError?.(error)
}

// Global error listeners
let unsubscribeErrorListener: (() => void) | null = null

// Watchers
watch(() => error.value, (newError) => {
  if (newError) {
    console.log(`Error boundary ${props.boundaryId} activated:`, newError)
    
    // Auto-show fallback for certain error types
    if (newError.type === 'data' || newError.type === 'parsing') {
      if (props.fallbackComponent || props.fallbackData) {
        setTimeout(() => {
          useFallback()
        }, 1000)
      }
    }
  }
})

// Lifecycle
onMounted(() => {
  // Listen for component-specific errors
  unsubscribeErrorListener = errorHandler.onError('*', (error) => {
    if (error.component === props.component) {
      catchError(error)
    }
  })
})

onUnmounted(() => {
  clearAutoRetryTimer()
  
  if (unsubscribeErrorListener) {
    unsubscribeErrorListener()
  }
})

// Expose methods for parent components
defineExpose({
  retry,
  useFallback,
  exitFallback,
  dismiss,
  catchError,
  hasError,
  isRetrying
})
</script>

<style scoped>
.error-boundary {
  @apply relative w-full;
}

.glass-card {
  @apply bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border border-slate-200/50 dark:border-slate-700/50;
}

.btn-error {
  @apply bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-md font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2;
}

.btn-warning {
  @apply bg-yellow-600 hover:bg-yellow-700 text-white px-4 py-2 rounded-md font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:ring-offset-2;
}

.btn-info {
  @apply bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2;
}

.btn-secondary {
  @apply bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300 px-4 py-2 rounded-md font-medium hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2;
}

.border-3 {
  border-width: 3px;
}

.fallback-display {
  animation: slideIn 0.3s ease-out;
}

.error-display {
  animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>