<template>
  <div class="relative h-full overflow-hidden">
    <!-- Pull indicator -->
    <div 
      class="absolute top-0 left-0 right-0 z-10 flex items-center justify-center transition-all duration-300 ease-out"
      :style="{ 
        height: `${indicatorHeight}px`,
        transform: `translateY(${Math.max(0, pullDistance - indicatorHeight)}px)`,
        opacity: pullDistance > 0 ? 1 : 0
      }"
    >
      <div class="flex items-center space-x-2 text-slate-600 dark:text-slate-400">
        <div 
          class="w-5 h-5 transition-transform duration-200"
          :class="{
            'animate-spin': isRefreshing,
            'rotate-180': pullDistance > threshold && !isRefreshing
          }"
        >
          <component 
            :is="isRefreshing ? refreshIcon : pullIcon" 
            class="w-5 h-5"
          />
        </div>
        <span class="text-sm font-medium">
          {{ statusText }}
        </span>
      </div>
    </div>
    
    <!-- Content area -->
    <div
      ref="contentContainer"
      class="h-full overflow-y-auto"
      :style="{ transform: `translateY(${Math.max(0, pullDistance)}px)` }"
      @touchstart="handleTouchStart"
      @touchmove="handleTouchMove"
      @touchend="handleTouchEnd"
      @touchcancel="handleTouchCancel"
      @scroll="handleScroll"
    >
      <slot />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, nextTick } from 'vue'
import { ArrowDownIcon, ArrowPathIcon } from '@heroicons/vue/24/outline'

interface Props {
  threshold?: number
  disabled?: boolean
  refreshing?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  threshold: 80,
  disabled: false,
  refreshing: false
})

const emit = defineEmits<{
  refresh: []
}>()

// Icons
const pullIcon = ArrowDownIcon
const refreshIcon = ArrowPathIcon

// State
const contentContainer = ref<HTMLElement | null>(null)
const startY = ref(0)
const currentY = ref(0)
const pullDistance = ref(0)
const isTracking = ref(false)
const canPull = ref(true)
const indicatorHeight = ref(60)
const isRefreshing = ref(false)

// Computed
const { threshold } = props

const statusText = computed(() => {
  if (isRefreshing.value) {
    return 'Refreshing...'
  }
  if (pullDistance.value > threshold) {
    return 'Release to refresh'
  }
  if (pullDistance.value > 0) {
    return 'Pull to refresh'
  }
  return ''
})

// Touch handlers
const handleTouchStart = (event: TouchEvent) => {
  if (props.disabled || isRefreshing.value) return
  
  const touch = event.touches[0]
  startY.value = touch.clientY
  currentY.value = touch.clientY
  
  // Check if we can pull (at top of scroll)
  const container = contentContainer.value
  if (container) {
    canPull.value = container.scrollTop <= 0
  }
}

const handleTouchMove = (event: TouchEvent) => {
  if (!canPull.value || props.disabled || isRefreshing.value) return
  
  const touch = event.touches[0]
  const deltaY = touch.clientY - startY.value
  
  // Only handle downward pulls when at top of scroll
  if (deltaY > 0) {
    event.preventDefault()
    currentY.value = touch.clientY
    
    // Apply resistance curve - gets harder to pull as distance increases
    const resistance = Math.min(deltaY / 3, threshold * 1.5)
    pullDistance.value = resistance
    
    isTracking.value = true
    
    // Haptic feedback at threshold
    if (deltaY > threshold && 'vibrate' in navigator) {
      navigator.vibrate(25)
    }
  }
}

const handleTouchEnd = async () => {
  if (!isTracking.value || props.disabled) return
  
  const shouldRefresh = pullDistance.value > threshold
  
  if (shouldRefresh && !isRefreshing.value) {
    // Start refresh
    isRefreshing.value = true
    
    // Strong haptic feedback
    if ('vibrate' in navigator) {
      navigator.vibrate([50, 25, 50])
    }
    
    // Emit refresh event
    emit('refresh')
    
    // Keep indicator visible during refresh
    pullDistance.value = threshold
    
    // Wait for refresh to complete (will be set by parent)
    // The parent should update the `refreshing` prop
  } else {
    // Reset without refreshing
    resetPull()
  }
  
  isTracking.value = false
}

const handleTouchCancel = () => {
  resetPull()
  isTracking.value = false
}

const handleScroll = () => {
  // Update canPull based on scroll position
  const container = contentContainer.value
  if (container) {
    canPull.value = container.scrollTop <= 0
  }
}

const resetPull = () => {
  pullDistance.value = 0
  isRefreshing.value = false
  startY.value = 0
  currentY.value = 0
}

// Watch for external refreshing state changes
const stopRefreshing = async () => {
  await nextTick()
  resetPull()
}

// Expose methods
defineExpose({
  resetPull,
  stopRefreshing
})
</script>

<style scoped>
/* Smooth transitions */
.transition-transform {
  transition-property: transform;
  transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
}

/* Prevent overscroll on iOS */
.overflow-y-auto {
  -webkit-overflow-scrolling: touch;
  overscroll-behavior-y: contain;
}

/* Disable text selection during pull */
.relative {
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}
</style>