<template>
  <div
    class="glass-card rounded-lg p-4 shadow-lg border-l-4"
    :class="[borderColor, backgroundColors]"
    role="alert"
    :aria-live="notification.type === 'error' ? 'assertive' : 'polite'"
  >
    <div class="flex items-start space-x-3">
      <!-- Icon -->
      <div class="flex-shrink-0">
        <component
          :is="iconComponent"
          class="w-5 h-5"
          :class="iconColor"
        />
      </div>
      
      <!-- Content -->
      <div class="flex-1 min-w-0">
        <div class="text-sm font-medium text-slate-900 dark:text-white">
          {{ notification.title }}
        </div>
        <div class="mt-1 text-sm text-slate-600 dark:text-slate-300">
          {{ notification.message }}
        </div>
        
        <!-- Actions -->
        <div v-if="notification.actions && notification.actions.length > 0" class="mt-3 flex space-x-2">
          <button
            v-for="(action, index) in notification.actions"
            :key="index"
            @click="$emit('action', notification, index)"
            class="text-xs font-medium px-3 py-1 rounded transition-colors"
            :class="action.style === 'primary' ? primaryButtonClasses : secondaryButtonClasses"
          >
            {{ action.label }}
          </button>
        </div>
      </div>
      
      <!-- Close button -->
      <div class="flex-shrink-0">
        <button
          @click="$emit('close', notification.id)"
          class="inline-flex text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors"
          :aria-label="`Close ${notification.title} notification`"
        >
          <XMarkIcon class="w-4 h-4" />
        </button>
      </div>
    </div>
    
    <!-- Progress bar for timed notifications -->
    <div
      v-if="notification.duration && notification.duration > 0"
      class="mt-3 w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1"
    >
      <div
        class="h-1 rounded-full transition-all ease-linear"
        :class="progressBarColor"
        :style="{ width: `${progress}%` }"
      ></div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, onUnmounted } from 'vue'
import type { Notification } from '@/stores/notifications'
import {
  CheckCircleIcon,
  ExclamationTriangleIcon,
  XCircleIcon,
  InformationCircleIcon,
  XMarkIcon,
} from '@heroicons/vue/24/outline'

interface Props {
  notification: Notification
}

const props = defineProps<Props>()

const emit = defineEmits<{
  close: [id: string]
  action: [notification: Notification, actionIndex: number]
}>()

// Progress tracking for timed notifications
const progress = ref(100)
let progressInterval: number | null = null

// Computed styles
const iconComponent = computed(() => {
  switch (props.notification.type) {
    case 'success': return CheckCircleIcon
    case 'warning': return ExclamationTriangleIcon
    case 'error': return XCircleIcon
    default: return InformationCircleIcon
  }
})

const iconColor = computed(() => {
  switch (props.notification.type) {
    case 'success': return 'text-success-600 dark:text-success-400'
    case 'warning': return 'text-warning-600 dark:text-warning-400'
    case 'error': return 'text-danger-600 dark:text-danger-400'
    default: return 'text-primary-600 dark:text-primary-400'
  }
})

const borderColor = computed(() => {
  switch (props.notification.type) {
    case 'success': return 'border-l-success-500'
    case 'warning': return 'border-l-warning-500'
    case 'error': return 'border-l-danger-500'
    default: return 'border-l-primary-500'
  }
})

const backgroundColors = computed(() => {
  switch (props.notification.type) {
    case 'success': return 'bg-success-50/90 dark:bg-success-900/20 border-success-200/50 dark:border-success-800/50'
    case 'warning': return 'bg-warning-50/90 dark:bg-warning-900/20 border-warning-200/50 dark:border-warning-800/50'
    case 'error': return 'bg-danger-50/90 dark:bg-danger-900/20 border-danger-200/50 dark:border-danger-800/50'
    default: return 'bg-primary-50/90 dark:bg-primary-900/20 border-primary-200/50 dark:border-primary-800/50'
  }
})

const progressBarColor = computed(() => {
  switch (props.notification.type) {
    case 'success': return 'bg-success-500'
    case 'warning': return 'bg-warning-500'
    case 'error': return 'bg-danger-500'
    default: return 'bg-primary-500'
  }
})

const primaryButtonClasses = computed(() => {
  switch (props.notification.type) {
    case 'success': return 'bg-success-600 hover:bg-success-700 text-white'
    case 'warning': return 'bg-warning-600 hover:bg-warning-700 text-white'
    case 'error': return 'bg-danger-600 hover:bg-danger-700 text-white'
    default: return 'bg-primary-600 hover:bg-primary-700 text-white'
  }
})

const secondaryButtonClasses = computed(() => {
  switch (props.notification.type) {
    case 'success': return 'bg-success-100 hover:bg-success-200 text-success-800 dark:bg-success-800 dark:hover:bg-success-700 dark:text-success-200'
    case 'warning': return 'bg-warning-100 hover:bg-warning-200 text-warning-800 dark:bg-warning-800 dark:hover:bg-warning-700 dark:text-warning-200'
    case 'error': return 'bg-danger-100 hover:bg-danger-200 text-danger-800 dark:bg-danger-800 dark:hover:bg-danger-700 dark:text-danger-200'
    default: return 'bg-primary-100 hover:bg-primary-200 text-primary-800 dark:bg-primary-800 dark:hover:bg-primary-700 dark:text-primary-200'
  }
})

// Lifecycle
onMounted(() => {
  // Start progress countdown for timed notifications
  if (props.notification.duration && props.notification.duration > 0) {
    const startTime = Date.now()
    const duration = props.notification.duration
    
    progressInterval = setInterval(() => {
      const elapsed = Date.now() - startTime
      const remaining = Math.max(0, duration - elapsed)
      progress.value = (remaining / duration) * 100
      
      if (remaining <= 0) {
        emit('close', props.notification.id)
      }
    }, 50) // Update every 50ms for smooth animation
  }
})

onUnmounted(() => {
  if (progressInterval) {
    clearInterval(progressInterval)
  }
})
</script>