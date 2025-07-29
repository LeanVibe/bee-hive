<template>
  <div
    ref="gestureContainer"
    class="relative w-full h-full overflow-hidden"
    @touchstart="handleTouchStart"
    @touchmove="handleTouchMove"
    @touchend="handleTouchEnd"
    @touchcancel="handleTouchCancel"
  >
    <!-- Content slot -->
    <div 
      class="transition-transform duration-300 ease-out"
      :style="{ transform: `translateX(${currentTransform}px)` }"
    >
      <slot />
    </div>
    
    <!-- Swipe indicators -->
    <div 
      v-if="showLeftIndicator && leftAction"
      class="absolute left-4 top-1/2 transform -translate-y-1/2 opacity-0 transition-opacity duration-200"
      :class="{ 'opacity-100': swipeDirection === 'right' && Math.abs(currentTransform) > 50 }"
    >
      <div class="bg-success-500 text-white p-2 rounded-full shadow-lg">
        <component :is="leftAction.icon" class="w-5 h-5" />
      </div>
    </div>
    
    <div 
      v-if="showRightIndicator && rightAction"
      class="absolute right-4 top-1/2 transform -translate-y-1/2 opacity-0 transition-opacity duration-200"
      :class="{ 'opacity-100': swipeDirection === 'left' && Math.abs(currentTransform) > 50 }"
    >
      <div class="bg-primary-500 text-white p-2 rounded-full shadow-lg">
        <component :is="rightAction.icon" class="w-5 h-5" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

interface SwipeAction {
  icon: any
  label: string
  handler: () => void
}

interface Props {
  leftAction?: SwipeAction
  rightAction?: SwipeAction
  threshold?: number
  disabled?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  threshold: 100,
  disabled: false
})

const emit = defineEmits<{
  swipeLeft: []
  swipeRight: []
  swipeStart: [direction: 'left' | 'right']
  swipeEnd: []
}>()

// Touch state
const gestureContainer = ref<HTMLElement | null>(null)
const startX = ref(0)
const startY = ref(0)
const currentX = ref(0)
const currentY = ref(0)
const currentTransform = ref(0)
const isTracking = ref(false)
const swipeDirection = ref<'left' | 'right' | null>(null)

// Computed properties
const showLeftIndicator = computed(() => 
  swipeDirection.value === 'right' && Math.abs(currentTransform.value) > 20
)

const showRightIndicator = computed(() => 
  swipeDirection.value === 'left' && Math.abs(currentTransform.value) > 20
)

// Touch handlers
const handleTouchStart = (event: TouchEvent) => {
  if (props.disabled) return
  
  const touch = event.touches[0]
  startX.value = touch.clientX
  startY.value = touch.clientY
  currentX.value = touch.clientX
  currentY.value = touch.clientY
  isTracking.value = true
  currentTransform.value = 0
  swipeDirection.value = null
}

const handleTouchMove = (event: TouchEvent) => {
  if (!isTracking.value || props.disabled) return
  
  const touch = event.touches[0]
  const deltaX = touch.clientX - startX.value
  const deltaY = touch.clientY - startY.value
  
  // Determine if this is a horizontal swipe
  if (Math.abs(deltaX) > Math.abs(deltaY) && Math.abs(deltaX) > 10) {
    event.preventDefault()
    
    currentX.value = touch.clientX
    currentY.value = touch.clientY
    
    // Determine swipe direction
    if (deltaX > 0) {
      swipeDirection.value = 'right'
      if (props.leftAction) {
        currentTransform.value = Math.min(deltaX, props.threshold * 1.5)
      }
    } else {
      swipeDirection.value = 'left'
      if (props.rightAction) {
        currentTransform.value = Math.max(deltaX, -props.threshold * 1.5)
      }
    }
    
    // Emit swipe start event
    if (Math.abs(deltaX) > 20 && swipeDirection.value) {
      emit('swipeStart', swipeDirection.value)
    }
    
    // Add haptic feedback for threshold crossing
    if (Math.abs(deltaX) > props.threshold && 'vibrate' in navigator) {
      navigator.vibrate(25)
    }
  }
}

const handleTouchEnd = () => {
  if (!isTracking.value || props.disabled) return
  
  const deltaX = currentX.value - startX.value
  const absDeltaX = Math.abs(deltaX)
  
  // Check if swipe threshold was met
  if (absDeltaX > props.threshold) {
    if (deltaX > 0 && props.leftAction) {
      // Swipe right (show left action)
      emit('swipeRight')
      props.leftAction.handler()
      
      // Strong haptic feedback
      if ('vibrate' in navigator) {
        navigator.vibrate([50, 25, 50])
      }
    } else if (deltaX < 0 && props.rightAction) {
      // Swipe left (show right action)
      emit('swipeLeft')
      props.rightAction.handler()
      
      // Strong haptic feedback
      if ('vibrate' in navigator) {
        navigator.vibrate([50, 25, 50])
      }
    }
  }
  
  // Reset state
  resetGesture()
  emit('swipeEnd')
}

const handleTouchCancel = () => {
  resetGesture()
  emit('swipeEnd')
}

const resetGesture = () => {
  isTracking.value = false
  currentTransform.value = 0
  swipeDirection.value = null
  startX.value = 0
  startY.value = 0
  currentX.value = 0
  currentY.value = 0
}

// Programmatic API
defineExpose({
  resetGesture
})
</script>

<style scoped>
/* Touch optimization */
.touch-active {
  touch-action: pan-x;
}

/* Smooth transitions */
.transition-transform {
  transition-property: transform;
  transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
}

/* Prevent text selection during swipe */
.gesture-container {
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}
</style>