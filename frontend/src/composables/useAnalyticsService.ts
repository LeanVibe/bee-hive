import { ref, computed } from 'vue'
import { apiService } from '@/services/api'

interface AnalyticsEvent {
  event: string
  properties: Record<string, any>
  timestamp: string
  sessionId: string
  userId?: string
}

interface StepAnalytics {
  step: number
  action: 'viewed' | 'completed' | 'skipped' | 'abandoned'
  properties: Record<string, any>
  timestamp: string
  timeSpent?: number
}

interface OnboardingMetrics {
  completionRate: number
  averageTimeToComplete: number
  dropOffPoints: { step: number; dropOffRate: number }[]
  userSegmentation: Record<string, number>
}

export const useAnalyticsService = () => {
  const sessionId = ref(generateSessionId())
  const isTracking = ref(true)
  const eventQueue = ref<AnalyticsEvent[]>([])
  const isOnline = ref(navigator.onLine)

  // Generate unique session ID
  function generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`
  }

  // Track generic events
  const trackEvent = async (eventName: string, properties: Record<string, any> = {}) => {
    if (!isTracking.value) return

    const event: AnalyticsEvent = {
      event: eventName,
      properties: {
        ...properties,
        userAgent: navigator.userAgent,
        viewport: {
          width: window.innerWidth,
          height: window.innerHeight
        },
        referrer: document.referrer,
        url: window.location.href
      },
      timestamp: new Date().toISOString(),
      sessionId: sessionId.value,
      userId: getCurrentUserId()
    }

    // Queue event for batch sending
    eventQueue.value.push(event)

    // Send immediately for critical events or when queue is full
    const criticalEvents = ['onboarding_completed', 'error', 'onboarding_abandoned']
    if (criticalEvents.includes(eventName) || eventQueue.value.length >= 10) {
      await flushEvents()
    }
  }

  // Track onboarding step events
  const trackStep = async (
    step: number,
    action: StepAnalytics['action'],
    properties: Record<string, any> = {}
  ) => {
    const stepEvent: StepAnalytics = {
      step,
      action,
      properties,
      timestamp: new Date().toISOString(),
      timeSpent: properties.timeSpent
    }

    await trackEvent(`onboarding_step_${action}`, {
      step_number: step,
      step_name: getStepName(step),
      ...stepEvent.properties
    })
  }

  // Track user interactions
  const trackInteraction = async (
    element: string,
    action: string,
    properties: Record<string, any> = {}
  ) => {
    await trackEvent('user_interaction', {
      element,
      action,
      ...properties
    })
  }

  // Track performance metrics
  const trackPerformance = async (
    metric: string,
    value: number,
    properties: Record<string, any> = {}
  ) => {
    await trackEvent('performance_metric', {
      metric,
      value,
      ...properties
    })
  }

  // Track errors
  const trackError = async (
    error: Error | string,
    context: Record<string, any> = {}
  ) => {
    const errorMessage = typeof error === 'string' ? error : error.message
    const errorStack = typeof error === 'object' ? error.stack : undefined

    await trackEvent('error', {
      message: errorMessage,
      stack: errorStack,
      context,
      timestamp: new Date().toISOString()
    })
  }

  // Flush event queue to server
  const flushEvents = async () => {
    if (eventQueue.value.length === 0 || !isOnline.value) return

    try {
      const eventsToSend = [...eventQueue.value]
      eventQueue.value = [] // Clear queue optimistically

      await apiService.post('/api/analytics/events', {
        events: eventsToSend,
        batchId: generateSessionId(),
        timestamp: new Date().toISOString()
      })

    } catch (error) {
      console.error('Failed to send analytics events:', error)
      // Re-queue events if send failed
      eventQueue.value.unshift(...eventQueue.value)
      
      // Track the analytics failure
      await trackError(error as Error, {
        context: 'analytics_flush_failed',
        queueLength: eventQueue.value.length
      })
    }
  }

  // Get onboarding metrics
  const getOnboardingMetrics = async (timeRange: number = 30): Promise<OnboardingMetrics> => {
    try {
      const response = await apiService.get(`/api/analytics/onboarding/metrics`, {
        params: { days: timeRange }
      })
      return response.data
    } catch (error) {
      console.error('Failed to fetch onboarding metrics:', error)
      throw error
    }
  }

  // Real-time onboarding analytics
  const getRealtimeOnboardingData = async () => {
    try {
      const response = await apiService.get('/api/analytics/onboarding/realtime')
      return response.data
    } catch (error) {
      console.error('Failed to fetch realtime onboarding data:', error)
      throw error
    }
  }

  // A/B test tracking
  const trackABTestAssignment = async (
    testName: string,
    variant: string,
    properties: Record<string, any> = {}
  ) => {
    await trackEvent('ab_test_assignment', {
      test_name: testName,
      variant,
      ...properties
    })
  }

  // Feature usage tracking
  const trackFeatureUsage = async (
    feature: string,
    action: string,
    properties: Record<string, any> = {}
  ) => {
    await trackEvent('feature_usage', {
      feature,
      action,
      ...properties
    })
  }

  // Conversion funnel tracking
  const trackConversionStep = async (
    funnel: string,
    step: string,
    properties: Record<string, any> = {}
  ) => {
    await trackEvent('conversion_step', {
      funnel,
      step,
      ...properties
    })
  }

  // Page/component view tracking
  const trackPageView = async (
    page: string,
    properties: Record<string, any> = {}
  ) => {
    await trackEvent('page_view', {
      page,
      ...properties
    })
  }

  // User identification
  const identifyUser = async (
    userId: string,
    traits: Record<string, any> = {}
  ) => {
    await trackEvent('user_identified', {
      user_id: userId,
      traits
    })
    
    // Store user ID in session storage
    sessionStorage.setItem('analytics_user_id', userId)
  }

  // Helper functions
  function getStepName(step: number): string {
    const stepNames = {
      1: 'welcome',
      2: 'agent_creation',
      3: 'dashboard_tour',
      4: 'first_task',
      5: 'completion'
    }
    return stepNames[step as keyof typeof stepNames] || `step_${step}`
  }

  function getCurrentUserId(): string | undefined {
    return sessionStorage.getItem('analytics_user_id') || undefined
  }

  // Initialize analytics
  const initializeAnalytics = async () => {
    // Set up online/offline listeners
    window.addEventListener('online', () => {
      isOnline.value = true
      flushEvents() // Send queued events when back online
    })
    
    window.addEventListener('offline', () => {
      isOnline.value = false
    })

    // Set up periodic flush
    setInterval(() => {
      flushEvents()
    }, 30000) // Flush every 30 seconds

    // Track session start
    await trackEvent('session_start', {
      session_id: sessionId.value,
      timestamp: new Date().toISOString()
    })

    // Track page unload
    window.addEventListener('beforeunload', () => {
      // Send any remaining events synchronously
      if (eventQueue.value.length > 0) {
        navigator.sendBeacon('/api/analytics/events', JSON.stringify({
          events: eventQueue.value,
          sessionEnd: true
        }))
      }
    })
  }

  // Privacy controls
  const enableTracking = () => {
    isTracking.value = true
    localStorage.setItem('analytics_enabled', 'true')
  }

  const disableTracking = () => {
    isTracking.value = false
    eventQueue.value = []
    localStorage.setItem('analytics_enabled', 'false')
  }

  const isTrackingEnabled = computed(() => {
    const stored = localStorage.getItem('analytics_enabled')
    return stored === null ? true : stored === 'true' // Default enabled
  })

  // Initialize tracking state from localStorage
  if (typeof window !== 'undefined') {
    isTracking.value = isTrackingEnabled.value
    initializeAnalytics()
  }

  return {
    // State
    sessionId,
    isTracking,
    eventQueue,
    isOnline,
    
    // Event tracking
    trackEvent,
    trackStep,
    trackInteraction,
    trackPerformance,
    trackError,
    trackABTestAssignment,
    trackFeatureUsage,
    trackConversionStep,
    trackPageView,
    
    // User management
    identifyUser,
    
    // Data retrieval
    getOnboardingMetrics,
    getRealtimeOnboardingData,
    
    // Queue management
    flushEvents,
    
    // Privacy controls
    enableTracking,
    disableTracking,
    isTrackingEnabled,
    
    // Initialization
    initializeAnalytics
  }
}