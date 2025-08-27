import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { apiService } from '@/services/api'

interface UserData {
  name?: string
  email?: string
  role?: string
  goals?: string[]
  preferences?: Record<string, any>
  createdAgent?: {
    id: string
    name: string
    type: string
  }
  completedTasks?: string[]
}

interface OnboardingProgress {
  currentStep: number
  completedSteps: number[]
  userData: UserData
  timeSpent: number
  startedAt: Date
  completedAt?: Date
}

interface StepValidation {
  [stepNumber: number]: boolean
}

export const useOnboardingStore = defineStore('onboarding', () => {
  // State
  const isOnboarding = ref(false)
  const progress = ref<OnboardingProgress>({
    currentStep: 1,
    completedSteps: [],
    userData: {},
    timeSpent: 0,
    startedAt: new Date()
  })
  const stepValidation = ref<StepValidation>({})
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  // Getters
  const currentStep = computed(() => progress.value.currentStep)
  const userData = computed(() => progress.value.userData)
  const completionPercentage = computed(() => 
    (progress.value.completedSteps.length / 5) * 100
  )
  const hasStarted = computed(() => isOnboarding.value || progress.value.completedSteps.length > 0)
  const isComplete = computed(() => progress.value.completedSteps.length === 5)

  // Step validation logic
  const canProceedFromStep = computed(() => (step: number): boolean => {
    switch (step) {
      case 1: // Welcome step
        return !!(
          userData.value.name?.trim() &&
          userData.value.role &&
          userData.value.goals?.length
        )
      case 2: // Agent creation step
        return !!(userData.value.createdAgent?.id)
      case 3: // Dashboard tour step
        return true // Always can proceed from tour
      case 4: // First task step
        return !!(userData.value.completedTasks?.length)
      case 5: // Completion step
        return true
      default:
        return false
    }
  })

  // Actions
  const initializeOnboarding = async () => {
    isLoading.value = true
    error.value = null

    try {
      // Check if user has existing onboarding progress
      const existingProgress = await apiService.get('/api/onboarding/progress')
      
      if (existingProgress.data) {
        progress.value = {
          ...existingProgress.data,
          startedAt: new Date(existingProgress.data.startedAt)
        }
      } else {
        // Start new onboarding session
        await apiService.post('/api/onboarding/start', {
          startedAt: progress.value.startedAt.toISOString(),
          userAgent: navigator.userAgent,
          referrer: document.referrer
        })
      }

      isOnboarding.value = true
    } catch (err: any) {
      error.value = err.message || 'Failed to initialize onboarding'
      console.error('Onboarding initialization error:', err)
    } finally {
      isLoading.value = false
    }
  }

  const updateUserData = (data: Partial<UserData>) => {
    progress.value.userData = { ...progress.value.userData, ...data }
    saveProgress()
  }

  const completeStep = async (step: number, stepData?: any) => {
    if (!progress.value.completedSteps.includes(step)) {
      progress.value.completedSteps.push(step)
    }

    // Track step completion
    await apiService.post('/api/onboarding/step-completed', {
      step,
      stepData,
      timestamp: new Date().toISOString(),
      userData: progress.value.userData
    })

    saveProgress()
  }

  const saveProgress = async () => {
    try {
      await apiService.put('/api/onboarding/progress', {
        currentStep: progress.value.currentStep,
        completedSteps: progress.value.completedSteps,
        userData: progress.value.userData,
        timeSpent: Date.now() - progress.value.startedAt.getTime()
      })
    } catch (err) {
      console.error('Failed to save onboarding progress:', err)
    }
  }

  const completeOnboarding = async (completionData: {
    userData: UserData
    totalTime: number
    completedSteps: number
  }) => {
    isLoading.value = true
    error.value = null

    try {
      progress.value.completedAt = new Date()
      progress.value.userData = completionData.userData

      const response = await apiService.post('/api/onboarding/complete', {
        ...completionData,
        completedAt: progress.value.completedAt.toISOString(),
        startedAt: progress.value.startedAt.toISOString()
      })

      isOnboarding.value = false

      // Store completion status in localStorage for quick access
      localStorage.setItem('onboarding_completed', 'true')
      localStorage.setItem('onboarding_completion_date', progress.value.completedAt.toISOString())

      return response.data
    } catch (err: any) {
      error.value = err.message || 'Failed to complete onboarding'
      throw err
    } finally {
      isLoading.value = false
    }
  }

  const skipOnboarding = async () => {
    try {
      await apiService.post('/api/onboarding/skip', {
        currentStep: progress.value.currentStep,
        skippedAt: new Date().toISOString(),
        userData: progress.value.userData
      })

      isOnboarding.value = false
      localStorage.setItem('onboarding_skipped', 'true')
    } catch (err) {
      console.error('Failed to record onboarding skip:', err)
    }
  }

  const resetOnboarding = () => {
    progress.value = {
      currentStep: 1,
      completedSteps: [],
      userData: {},
      timeSpent: 0,
      startedAt: new Date()
    }
    isOnboarding.value = false
    error.value = null
    stepValidation.value = {}
    
    localStorage.removeItem('onboarding_completed')
    localStorage.removeItem('onboarding_skipped')
    localStorage.removeItem('onboarding_completion_date')
  }

  // Agent creation helpers
  const createAgent = async (agentData: {
    name: string
    type: string
    description?: string
    capabilities?: string[]
  }) => {
    try {
      const response = await apiService.post('/api/agents', agentData)
      
      progress.value.userData.createdAgent = {
        id: response.data.id,
        name: response.data.name,
        type: response.data.type
      }

      await completeStep(2, { agentCreated: true, agentData })
      
      return response.data
    } catch (err: any) {
      error.value = err.message || 'Failed to create agent'
      throw err
    }
  }

  // Task creation helpers
  const createFirstTask = async (taskData: {
    title: string
    description?: string
    agentId: string
    priority?: string
  }) => {
    try {
      const response = await apiService.post('/api/tasks', {
        ...taskData,
        agentId: progress.value.userData.createdAgent?.id
      })

      if (!progress.value.userData.completedTasks) {
        progress.value.userData.completedTasks = []
      }
      progress.value.userData.completedTasks.push(response.data.id)

      await completeStep(4, { firstTaskCreated: true, taskData })
      
      return response.data
    } catch (err: any) {
      error.value = err.message || 'Failed to create task'
      throw err
    }
  }

  // Analytics helpers
  const getOnboardingAnalytics = computed(() => ({
    timeSpent: Date.now() - progress.value.startedAt.getTime(),
    completionRate: (progress.value.completedSteps.length / 5) * 100,
    currentStep: progress.value.currentStep,
    userData: progress.value.userData,
    dropOffPoint: progress.value.completedSteps.length < 5 ? progress.value.currentStep : null
  }))

  // Check completion status on store initialization
  const checkCompletionStatus = () => {
    const isCompleted = localStorage.getItem('onboarding_completed') === 'true'
    const isSkipped = localStorage.getItem('onboarding_skipped') === 'true'
    
    if (isCompleted || isSkipped) {
      isOnboarding.value = false
    }
    
    return { isCompleted, isSkipped }
  }

  return {
    // State
    isOnboarding,
    progress,
    stepValidation,
    isLoading,
    error,
    
    // Getters
    currentStep,
    userData,
    completionPercentage,
    hasStarted,
    isComplete,
    canProceedFromStep,
    getOnboardingAnalytics,
    
    // Actions
    initializeOnboarding,
    updateUserData,
    completeStep,
    saveProgress,
    completeOnboarding,
    skipOnboarding,
    resetOnboarding,
    createAgent,
    createFirstTask,
    checkCompletionStatus
  }
})