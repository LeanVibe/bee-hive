/**
 * Mobile Coordination Composable
 * 
 * Provides mobile-optimized interfaces and interactions for the
 * coordination dashboard with touch-friendly controls and responsive design.
 */

import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useBreakpoints } from '@/composables/useBreakpoints'

export interface MobileCoordinationConfig {
  enableSwipeGestures: boolean
  enablePullToRefresh: boolean
  compactMode: boolean
  touchFeedback: boolean
  gestureThreshold: number
}

export interface SwipeDirection {
  type: 'left' | 'right' | 'up' | 'down'
  distance: number
  velocity: number
  element: HTMLElement
}

export interface TouchGesture {
  id: string
  type: 'tap' | 'long-press' | 'swipe' | 'pinch' | 'drag'
  startPosition: { x: number; y: number }
  currentPosition: { x: number; y: number }
  startTime: number
  element: HTMLElement
  data?: any
}

export function useMobileCoordination(config: Partial<MobileCoordinationConfig> = {}) {
  const fullConfig: MobileCoordinationConfig = {
    enableSwipeGestures: true,
    enablePullToRefresh: true,
    compactMode: false,
    touchFeedback: true,
    gestureThreshold: 50,
    ...config
  }

  const { isMobile, isTablet, screenSize } = useBreakpoints()

  // State
  const isTouch = ref(false)
  const activeGestures = ref<Map<string, TouchGesture>>(new Map())
  const swipeInProgress = ref(false)
  const pullToRefreshActive = ref(false)
  const pullDistance = ref(0)
  const lastTouchTime = ref(0)
  const hapticFeedback = ref(true)

  // Computed
  const isMobileView = computed(() => isMobile.value || screenSize.value === 'sm')
  const isCompactView = computed(() => fullConfig.compactMode || isMobileView.value)
  const shouldShowMobileUI = computed(() => isMobileView.value || isTouch.value)

  // Touch gesture handling
  const handleTouchStart = (event: TouchEvent) => {
    isTouch.value = true
    lastTouchTime.value = Date.now()

    for (let i = 0; i < event.touches.length; i++) {
      const touch = event.touches[i]
      const gestureId = `touch-${touch.identifier}`

      const gesture: TouchGesture = {
        id: gestureId,
        type: 'tap',
        startPosition: { x: touch.clientX, y: touch.clientY },
        currentPosition: { x: touch.clientX, y: touch.clientY },
        startTime: Date.now(),
        element: event.target as HTMLElement
      }

      activeGestures.value.set(gestureId, gesture)
    }
  }

  const handleTouchMove = (event: TouchEvent) => {
    for (let i = 0; i < event.touches.length; i++) {
      const touch = event.touches[i]
      const gestureId = `touch-${touch.identifier}`
      const gesture = activeGestures.value.get(gestureId)

      if (gesture) {
        const deltaX = touch.clientX - gesture.startPosition.x
        const deltaY = touch.clientY - gesture.startPosition.y
        const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY)

        gesture.currentPosition = { x: touch.clientX, y: touch.clientY }

        // Determine gesture type based on movement
        if (distance > fullConfig.gestureThreshold) {
          if (Math.abs(deltaX) > Math.abs(deltaY)) {
            gesture.type = 'swipe'
            swipeInProgress.value = true
          } else if (deltaY > fullConfig.gestureThreshold) {
            gesture.type = 'drag'
            handlePullToRefresh(deltaY)
          }
        }

        // Long press detection
        if (!gesture.type && Date.now() - gesture.startTime > 500) {
          gesture.type = 'long-press'
          triggerHapticFeedback('medium')
        }
      }
    }
  }

  const handleTouchEnd = (event: TouchEvent) => {
    const remainingTouches = new Set<string>()
    
    for (let i = 0; i < event.touches.length; i++) {
      remainingTouches.add(`touch-${event.touches[i].identifier}`)
    }

    // Process ended gestures
    for (const [gestureId, gesture] of activeGestures.value) {
      if (!remainingTouches.has(gestureId)) {
        processGestureEnd(gesture)
        activeGestures.value.delete(gestureId)
      }
    }

    if (activeGestures.value.size === 0) {
      swipeInProgress.value = false
      resetPullToRefresh()
    }
  }

  const processGestureEnd = (gesture: TouchGesture) => {
    const duration = Date.now() - gesture.startTime
    const deltaX = gesture.currentPosition.x - gesture.startPosition.x
    const deltaY = gesture.currentPosition.y - gesture.startPosition.y
    const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY)
    const velocity = distance / duration

    switch (gesture.type) {
      case 'tap':
        if (duration < 200 && distance < 10) {
          handleTap(gesture)
        }
        break

      case 'swipe':
        handleSwipe({
          type: Math.abs(deltaX) > Math.abs(deltaY) 
            ? (deltaX > 0 ? 'right' : 'left')
            : (deltaY > 0 ? 'down' : 'up'),
          distance,
          velocity,
          element: gesture.element
        })
        break

      case 'long-press':
        handleLongPress(gesture)
        break

      case 'drag':
        if (pullToRefreshActive.value && deltaY > 100) {
          triggerPullToRefresh()
        }
        break
    }
  }

  const handleTap = (gesture: TouchGesture) => {
    // Add visual feedback for touch
    if (fullConfig.touchFeedback) {
      addTouchRipple(gesture.element, gesture.currentPosition)
    }

    triggerHapticFeedback('light')
  }

  const handleSwipe = (swipe: SwipeDirection) => {
    if (!fullConfig.enableSwipeGestures) return

    // Emit swipe events that components can listen to
    document.dispatchEvent(new CustomEvent('coordination-swipe', {
      detail: swipe
    }))

    triggerHapticFeedback('light')
  }

  const handleLongPress = (gesture: TouchGesture) => {
    // Show context menu or additional options
    document.dispatchEvent(new CustomEvent('coordination-long-press', {
      detail: {
        element: gesture.element,
        position: gesture.currentPosition
      }
    }))

    triggerHapticFeedback('heavy')
  }

  const handlePullToRefresh = (deltaY: number) => {
    if (!fullConfig.enablePullToRefresh) return

    pullDistance.value = Math.min(deltaY, 150)
    
    if (deltaY > 80 && !pullToRefreshActive.value) {
      pullToRefreshActive.value = true
      triggerHapticFeedback('medium')
    }
  }

  const triggerPullToRefresh = () => {
    if (pullToRefreshActive.value) {
      document.dispatchEvent(new CustomEvent('coordination-pull-refresh'))
      triggerHapticFeedback('heavy')
    }
  }

  const resetPullToRefresh = () => {
    pullToRefreshActive.value = false
    pullDistance.value = 0
  }

  const addTouchRipple = (element: HTMLElement, position: { x: number; y: number }) => {
    const ripple = document.createElement('div')
    const rect = element.getBoundingClientRect()
    
    ripple.classList.add('touch-ripple')
    ripple.style.cssText = `
      position: absolute;
      border-radius: 50%;
      background: rgba(59, 130, 246, 0.3);
      pointer-events: none;
      transform: scale(0);
      animation: ripple 0.6s linear;
      left: ${position.x - rect.left - 10}px;
      top: ${position.y - rect.top - 10}px;
      width: 20px;
      height: 20px;
      z-index: 1000;
    `

    element.style.position = 'relative'
    element.appendChild(ripple)

    // Remove ripple after animation
    setTimeout(() => {
      if (ripple.parentNode) {
        ripple.parentNode.removeChild(ripple)
      }
    }, 600)
  }

  const triggerHapticFeedback = (intensity: 'light' | 'medium' | 'heavy' = 'light') => {
    if (!fullConfig.touchFeedback || !hapticFeedback.value) return

    // Check if device supports haptic feedback
    if ('vibrate' in navigator) {
      const patterns = {
        light: [10],
        medium: [50],
        heavy: [100]
      }
      navigator.vibrate(patterns[intensity])
    }

    // Check for Web Vibration API or other haptic APIs
    if ('haptics' in navigator) {
      // @ts-ignore - Future API
      navigator.haptics?.vibrate(intensity)
    }
  }

  // Mobile-specific UI helpers
  const getMobileLayoutClasses = () => {
    if (!shouldShowMobileUI.value) return ''

    return [
      'mobile-optimized',
      isMobileView.value ? 'mobile-view' : 'tablet-view',
      isCompactView.value ? 'compact-mode' : '',
      fullConfig.touchFeedback ? 'touch-feedback-enabled' : ''
    ].filter(Boolean).join(' ')
  }

  const getTouchTargetSize = (element: 'button' | 'card' | 'icon' | 'small') => {
    if (!shouldShowMobileUI.value) return ''

    const sizes = {
      button: 'min-h-[44px] min-w-[44px]', // Apple HIG minimum
      card: 'min-h-[60px]',
      icon: 'w-6 h-6 p-2', // 24px icon with 8px padding
      small: 'min-h-[32px] min-w-[32px]'
    }

    return sizes[element] || sizes.button
  }

  const getMobileSpacing = (size: 'xs' | 'sm' | 'md' | 'lg' | 'xl') => {
    if (!shouldShowMobileUI.value) return ''

    const spacing = {
      xs: isMobileView.value ? 'gap-2' : 'gap-1',
      sm: isMobileView.value ? 'gap-3' : 'gap-2',
      md: isMobileView.value ? 'gap-4' : 'gap-3',
      lg: isMobileView.value ? 'gap-6' : 'gap-4',
      xl: isMobileView.value ? 'gap-8' : 'gap-6'
    }

    return spacing[size] || spacing.md
  }

  const getMobileTextSize = (size: 'xs' | 'sm' | 'base' | 'lg' | 'xl') => {
    if (!shouldShowMobileUI.value) return ''

    // Slightly larger text on mobile for better readability
    const textSizes = {
      xs: isMobileView.value ? 'text-sm' : 'text-xs',
      sm: isMobileView.value ? 'text-base' : 'text-sm',
      base: isMobileView.value ? 'text-lg' : 'text-base',
      lg: isMobileView.value ? 'text-xl' : 'text-lg',
      xl: isMobileView.value ? 'text-2xl' : 'text-xl'
    }

    return textSizes[size] || textSizes.base
  }

  // Responsive card configurations
  const getCardLayout = (type: 'task' | 'agent' | 'metric' | 'chart') => {
    const layouts = {
      task: {
        mobile: 'w-full mb-3 p-4',
        tablet: 'w-full sm:w-1/2 lg:w-1/3 p-3',
        desktop: 'w-1/3 lg:w-1/4 p-2'
      },
      agent: {
        mobile: 'w-full mb-4 p-4',
        tablet: 'w-full sm:w-1/2 p-3',
        desktop: 'w-1/2 lg:w-1/3 xl:w-1/4 p-2'
      },
      metric: {
        mobile: 'w-full mb-3 p-3',
        tablet: 'w-1/2 lg:w-1/4 p-2',
        desktop: 'w-1/4 lg:w-1/6 p-2'
      },
      chart: {
        mobile: 'w-full mb-6 p-4 h-64',
        tablet: 'w-full lg:w-1/2 p-4 h-80',
        desktop: 'w-1/2 xl:w-1/3 p-4 h-96'
      }
    }

    if (isMobileView.value) return layouts[type].mobile
    if (isTablet.value) return layouts[type].tablet
    return layouts[type].desktop
  }

  // Performance optimizations for mobile
  const enableMobileOptimizations = () => {
    // Reduce animation complexity on mobile
    if (isMobileView.value) {
      document.documentElement.style.setProperty('--animation-duration', '0.2s')
      document.documentElement.style.setProperty('--transition-duration', '0.15s')
    }

    // Enable hardware acceleration for smooth scrolling
    document.body.style.transform = 'translateZ(0)'
    document.body.style.backfaceVisibility = 'hidden'
    document.body.style.perspective = '1000px'
  }

  const disableMobileOptimizations = () => {
    document.documentElement.style.removeProperty('--animation-duration')
    document.documentElement.style.removeProperty('--transition-duration')
    document.body.style.removeProperty('transform')
    document.body.style.removeProperty('backfaceVisibility')
    document.body.style.removeProperty('perspective')
  }

  // Swipe navigation helpers
  const enableSwipeNavigation = (element: HTMLElement, callbacks: {
    onSwipeLeft?: () => void
    onSwipeRight?: () => void
    onSwipeUp?: () => void
    onSwipeDown?: () => void
  }) => {
    const handler = (event: CustomEvent) => {
      const swipe = event.detail as SwipeDirection
      
      if (swipe.element === element || element.contains(swipe.element)) {
        switch (swipe.type) {
          case 'left':
            callbacks.onSwipeLeft?.()
            break
          case 'right':
            callbacks.onSwipeRight?.()
            break
          case 'up':
            callbacks.onSwipeUp?.()
            break
          case 'down':
            callbacks.onSwipeDown?.()
            break
        }
      }
    }

    document.addEventListener('coordination-swipe', handler as EventListener)
    
    return () => {
      document.removeEventListener('coordination-swipe', handler as EventListener)
    }
  }

  // Setup touch event listeners
  const setupTouchListeners = () => {
    if ('ontouchstart' in window) {
      document.addEventListener('touchstart', handleTouchStart, { passive: false })
      document.addEventListener('touchmove', handleTouchMove, { passive: false })
      document.addEventListener('touchend', handleTouchEnd, { passive: false })
    }
  }

  const cleanupTouchListeners = () => {
    document.removeEventListener('touchstart', handleTouchStart)
    document.removeEventListener('touchmove', handleTouchMove)
    document.removeEventListener('touchend', handleTouchEnd)
  }

  // Lifecycle
  onMounted(() => {
    setupTouchListeners()
    enableMobileOptimizations()
  })

  onUnmounted(() => {
    cleanupTouchListeners()
    disableMobileOptimizations()
  })

  return {
    // State
    isTouch,
    isMobileView,
    isCompactView,
    shouldShowMobileUI,
    swipeInProgress,
    pullToRefreshActive,
    pullDistance,
    hapticFeedback,

    // Methods
    triggerHapticFeedback,
    getMobileLayoutClasses,
    getTouchTargetSize,
    getMobileSpacing,
    getMobileTextSize,
    getCardLayout,
    enableSwipeNavigation,

    // Events
    handleTap,
    handleSwipe,
    handleLongPress,
    resetPullToRefresh
  }
}

// CSS for mobile optimizations
export const mobileCoordinationStyles = `
@keyframes ripple {
  to {
    transform: scale(4);
    opacity: 0;
  }
}

.mobile-optimized {
  /* Smooth scrolling */
  -webkit-overflow-scrolling: touch;
  scroll-behavior: smooth;
}

.mobile-view {
  /* Larger tap targets */
  --min-tap-target: 44px;
}

.tablet-view {
  /* Medium tap targets */
  --min-tap-target: 36px;
}

.compact-mode {
  /* Reduced spacing and padding */
  --spacing-multiplier: 0.75;
}

.touch-feedback-enabled {
  /* Visual feedback for touch interactions */
  --touch-feedback-duration: 0.15s;
}

/* Touch-specific styles */
@media (hover: none) and (pointer: coarse) {
  .hover\\:scale-105:hover {
    transform: scale(1.02);
  }
  
  .hover\\:shadow-lg:hover {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  }
}

/* High DPI displays */
@media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi) {
  .mobile-optimized {
    /* Crisper text rendering */
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }
}
`