import { ref, computed, onMounted, onUnmounted } from 'vue'

export interface MobileOptimizationOptions {
  enableHapticFeedback?: boolean
  optimizeScrolling?: boolean
  handleOrientationChange?: boolean
  enableSwipeGestures?: boolean
  adaptToViewport?: boolean
}

export interface DeviceInfo {
  isMobile: boolean
  isTablet: boolean
  isDesktop: boolean
  hasTouch: boolean
  hasSafeArea: boolean
  orientation: 'portrait' | 'landscape'
  viewportWidth: number
  viewportHeight: number
  devicePixelRatio: number
  platform: 'ios' | 'android' | 'desktop' | 'unknown'
}

export function useMobileOptimization(options: MobileOptimizationOptions = {}) {
  const {
    enableHapticFeedback = true,
    optimizeScrolling = true,
    handleOrientationChange = true,
    enableSwipeGestures = true,
    adaptToViewport = true
  } = options
  
  // Device detection state
  const deviceInfo = ref<DeviceInfo>({
    isMobile: false,
    isTablet: false,
    isDesktop: true,
    hasTouch: false,
    hasSafeArea: false,
    orientation: 'portrait',
    viewportWidth: 0,
    viewportHeight: 0,
    devicePixelRatio: 1,
    platform: 'unknown'
  })
  
  const isOnline = ref(navigator.onLine)
  const connectionType = ref<string>('unknown')
  const batteryLevel = ref<number | null>(null)
  const isLowPowerMode = ref(false)
  
  // Detect device capabilities
  const detectDevice = () => {
    const userAgent = navigator.userAgent.toLowerCase()
    const hasTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0
    
    // Platform detection
    let platform: DeviceInfo['platform'] = 'unknown'
    if (/iphone|ipad|ipod/.test(userAgent)) {
      platform = 'ios'
    } else if (/android/.test(userAgent)) {
      platform = 'android'
    } else if (!hasTouch) {
      platform = 'desktop'
    }
    
    // Screen size detection
    const width = window.innerWidth
    const height = window.innerHeight
    const isMobile = width < 768 || hasTouch && width < 1024
    const isTablet = hasTouch && width >= 768 && width < 1024
    const isDesktop = !hasTouch || width >= 1024
    
    // Safe area detection (iPhone X+ style)
    const hasSafeArea = detectSafeArea()
    
    deviceInfo.value = {
      isMobile,
      isTablet,
      isDesktop,
      hasTouch,
      hasSafeArea,
      orientation: width > height ? 'landscape' : 'portrait',
      viewportWidth: width,
      viewportHeight: height,
      devicePixelRatio: window.devicePixelRatio || 1,
      platform
    }
  }
  
  // Detect safe area support
  const detectSafeArea = (): boolean => {
    const testElement = document.createElement('div')
    testElement.style.paddingTop = 'env(safe-area-inset-top)'
    document.body.appendChild(testElement)
    
    const computedStyle = window.getComputedStyle(testElement)
    const hasSafeArea = computedStyle.paddingTop !== '0px'
    
    document.body.removeChild(testElement)
    return hasSafeArea
  }
  
  // Haptic feedback
  const vibrate = (pattern?: number | number[]) => {
    if (!enableHapticFeedback || !deviceInfo.value.hasTouch) return false
    
    if ('vibrate' in navigator) {
      if (typeof pattern === 'number') {
        navigator.vibrate(pattern)
      } else if (Array.isArray(pattern)) {
        navigator.vibrate(pattern)
      } else {
        navigator.vibrate(25) // Default light feedback
      }
      return true
    }
    return false
  }
  
  // Performance-aware scrolling
  const optimizeScroll = (element: HTMLElement) => {
    if (!optimizeScrolling) return
    
    // Enable momentum scrolling on iOS
    element.style.webkitOverflowScrolling = 'touch'
    
    // Contain overscroll to prevent bounce
    element.style.overscrollBehavior = 'contain'
    
    // Use transform3d for hardware acceleration
    element.style.willChange = 'scroll-position'
    
    // Throttle scroll events for performance
    let ticking = false
    const throttledScrollHandler = () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          // Handle scroll
          ticking = false
        })
        ticking = true
      }
    }
    
    element.addEventListener('scroll', throttledScrollHandler, { passive: true })
    
    return () => {
      element.removeEventListener('scroll', throttledScrollHandler)
    }
  }
  
  // Viewport adaptation
  const adaptToViewport = () => {
    if (!adaptToViewport) return
    
    // Set CSS custom properties for viewport dimensions
    document.documentElement.style.setProperty('--vh', `${window.innerHeight * 0.01}px`)
    document.documentElement.style.setProperty('--vw', `${window.innerWidth * 0.01}px`)
    
    // Set safe area insets if supported
    if (deviceInfo.value.hasSafeArea) {
      document.documentElement.classList.add('has-safe-area')
    }
    
    // Add platform classes
    document.documentElement.classList.add(`platform-${deviceInfo.value.platform}`)
    
    if (deviceInfo.value.isMobile) {
      document.documentElement.classList.add('is-mobile')
    }
    if (deviceInfo.value.hasTouch) {
      document.documentElement.classList.add('has-touch')
    }
  }
  
  // Network detection
  const detectConnection = () => {
    const connection = (navigator as any).connection || 
                      (navigator as any).mozConnection || 
                      (navigator as any).webkitConnection
    
    if (connection) {
      connectionType.value = connection.effectiveType || 'unknown'
      
      // Detect slow connections
      const isSlowConnection = connection.effectiveType === 'slow-2g' || 
                              connection.effectiveType === '2g' ||
                              connection.saveData
      
      if (isSlowConnection) {
        document.documentElement.classList.add('slow-connection')
      }
    }
  }
  
  // Battery API
  const detectBattery = async () => {
    if ('getBattery' in navigator) {
      try {
        const battery = await (navigator as any).getBattery()
        batteryLevel.value = battery.level * 100
        isLowPowerMode.value = battery.level < 0.2
        
        // Listen for battery changes
        battery.addEventListener('levelchange', () => {
          batteryLevel.value = battery.level * 100
          isLowPowerMode.value = battery.level < 0.2
        })
      } catch {
        // Battery API not supported or permission denied
      }
    }
  }
  
  // Touch optimization
  const optimizeTouch = (element: HTMLElement) => {
    if (!deviceInfo.value.hasTouch) return
    
    // Prevent default touch behaviors that interfere with gestures
    element.style.touchAction = 'manipulation'
    
    // Improve touch responsiveness
    element.style.webkitTapHighlightColor = 'transparent'
    element.style.webkitUserSelect = 'none'
    element.style.userSelect = 'none'
    
    // Add touch feedback
    const addTouchFeedback = (event: TouchEvent) => {
      if (enableHapticFeedback && event.target instanceof HTMLElement) {
        // Light haptic feedback for touch
        vibrate(25)
        
        // Visual feedback
        event.target.style.transform = 'scale(0.98)'
        setTimeout(() => {
          event.target!.style.transform = ''
        }, 100)
      }
    }
    
    element.addEventListener('touchstart', addTouchFeedback, { passive: true })
    
    return () => {
      element.removeEventListener('touchstart', addTouchFeedback)
    }
  }
  
  // Orientation change handler
  const handleOrientationChange = () => {
    if (!handleOrientationChange) return
    
    const onOrientationChange = () => {
      setTimeout(() => {
        detectDevice()
        adaptToViewport()
      }, 100) // Delay to ensure viewport has updated
    }
    
    window.addEventListener('orientationchange', onOrientationChange)
    window.addEventListener('resize', onOrientationChange)
    
    return () => {
      window.removeEventListener('orientationchange', onOrientationChange)
      window.removeEventListener('resize', onOrientationChange)
    }
  }
  
  // Performance monitoring
  const getPerformanceMetrics = () => {
    if ('performance' in window && 'memory' in (performance as any)) {
      const memory = (performance as any).memory
      return {
        memoryUsed: memory.usedJSHeapSize,
        memoryTotal: memory.totalJSHeapSize,
        memoryLimit: memory.jsHeapSizeLimit,
        devicePixelRatio: deviceInfo.value.devicePixelRatio,
        connectionType: connectionType.value,
        batteryLevel: batteryLevel.value,
        isLowPowerMode: isLowPowerMode.value
      }
    }
    return null
  }
  
  // Computed properties
  const isMobileDevice = computed(() => deviceInfo.value.isMobile)
  const hasTouch = computed(() => deviceInfo.value.hasTouch)
  const isLandscape = computed(() => deviceInfo.value.orientation === 'landscape')
  const isPortrait = computed(() => deviceInfo.value.orientation === 'portrait')
  
  // Online/offline handlers
  const handleOnlineStatus = () => {
    const updateOnlineStatus = () => {
      isOnline.value = navigator.onLine
    }
    
    window.addEventListener('online', updateOnlineStatus)
    window.addEventListener('offline', updateOnlineStatus)
    
    return () => {
      window.removeEventListener('online', updateOnlineStatus)
      window.removeEventListener('offline', updateOnlineStatus)
    }
  }
  
  // Initialize everything
  onMounted(() => {
    detectDevice()
    detectConnection()
    detectBattery()
    adaptToViewport()
    
    const cleanupOrientation = handleOrientationChange()
    const cleanupOnline = handleOnlineStatus()
    
    onUnmounted(() => {
      cleanupOrientation?.()
      cleanupOnline?.()
    })
  })
  
  return {
    deviceInfo,
    isOnline,
    connectionType,
    batteryLevel,
    isLowPowerMode,
    
    // Computed
    isMobileDevice,
    hasTouch,
    isLandscape,
    isPortrait,
    
    // Methods
    vibrate,
    optimizeScroll,
    optimizeTouch,
    getPerformanceMetrics,
    detectDevice
  }
}