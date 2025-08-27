<template>
  <div class="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
    <!-- Progress Header -->
    <div class="sticky top-0 z-50 bg-white/80 backdrop-blur-sm border-b border-gray-200">
      <div class="max-w-4xl mx-auto px-4 py-4">
        <div class="flex items-center justify-between">
          <div class="flex items-center space-x-4">
            <h1 class="text-2xl font-bold text-gray-900">Welcome to LeanVibe Agent Hive</h1>
            <div class="hidden sm:flex items-center text-sm text-gray-500">
              Step {{ currentStep }} of {{ totalSteps }}
            </div>
          </div>
          
          <!-- Progress Bar -->
          <div class="flex-1 max-w-md mx-8">
            <div class="w-full bg-gray-200 rounded-full h-2">
              <div 
                class="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full transition-all duration-500 ease-out"
                :style="{ width: progressPercentage + '%' }"
              ></div>
            </div>
          </div>
          
          <!-- Exit Button -->
          <button 
            @click="handleSkip"
            class="text-gray-400 hover:text-gray-600 transition-colors"
            :class="{ 'hidden': currentStep === totalSteps }"
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>
    </div>

    <!-- Main Content Area -->
    <div class="max-w-4xl mx-auto px-4 py-8">
      <div class="bg-white rounded-2xl shadow-xl overflow-hidden">
        <div class="relative min-h-[600px]">
          <!-- Step Content -->
          <Transition
            :name="transitionDirection"
            mode="out-in"
            @before-enter="onBeforeEnter"
            @after-enter="onAfterEnter"
          >
            <component
              :is="currentStepComponent"
              :key="currentStep"
              :user-data="userData"
              :step-data="stepData[currentStep]"
              @next="handleNext"
              @back="handleBack"
              @update-data="handleDataUpdate"
              @analytics-event="handleAnalyticsEvent"
              class="absolute inset-0"
            />
          </Transition>
          
          <!-- Loading Overlay -->
          <div 
            v-if="isLoading" 
            class="absolute inset-0 bg-white/80 backdrop-blur-sm flex items-center justify-center z-10"
          >
            <div class="flex flex-col items-center space-y-4">
              <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
              <p class="text-gray-600">{{ loadingMessage }}</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Mobile Navigation -->
    <div class="sm:hidden fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 px-4 py-3">
      <div class="flex items-center justify-between">
        <button
          @click="handleBack"
          :disabled="currentStep === 1"
          class="flex items-center space-x-2 px-4 py-2 rounded-lg text-gray-600 disabled:text-gray-300 disabled:cursor-not-allowed"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
          </svg>
          <span>Back</span>
        </button>
        
        <div class="text-sm text-gray-500">
          {{ currentStep }} / {{ totalSteps }}
        </div>
        
        <button
          @click="handleNext"
          :disabled="!canProceed"
          class="flex items-center space-x-2 px-4 py-2 bg-blue-500 text-white rounded-lg disabled:bg-gray-300 disabled:cursor-not-allowed"
        >
          <span>{{ isLastStep ? 'Complete' : 'Next' }}</span>
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
          </svg>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { useRouter } from 'vue-router'
import { useOnboardingStore } from '@/stores/onboarding'
import { useAnalyticsService } from '@/composables/useAnalyticsService'

// Import step components
import WelcomeStep from './WelcomeStep.vue'
import AgentCreationStep from './AgentCreationStep.vue'
import DashboardTourStep from './DashboardTourStep.vue'
import FirstTaskStep from './FirstTaskStep.vue'
import CompletionStep from './CompletionStep.vue'

// Types
interface UserData {
  name?: string
  email?: string
  role?: string
  goals?: string[]
  preferences?: Record<string, any>
}

interface StepData {
  title: string
  description: string
  canSkip: boolean
  required: boolean
}

// Composition functions
const router = useRouter()
const onboardingStore = useOnboardingStore()
const { trackEvent, trackStep } = useAnalyticsService()

// Reactive state
const currentStep = ref(1)
const totalSteps = 5
const isLoading = ref(false)
const loadingMessage = ref('')
const transitionDirection = ref('slide-left')
const startTime = ref<Date>(new Date())

const userData = ref<UserData>({})
const stepData = ref<Record<number, StepData>>({
  1: { title: 'Welcome', description: 'Get started with Agent Hive', canSkip: false, required: true },
  2: { title: 'Create Agent', description: 'Set up your first AI agent', canSkip: false, required: true },
  3: { title: 'Dashboard Tour', description: 'Explore the interface', canSkip: true, required: false },
  4: { title: 'First Task', description: 'Create your first task', canSkip: false, required: true },
  5: { title: 'Complete Setup', description: 'Finish onboarding', canSkip: false, required: true }
})

// Step components mapping
const stepComponents = {
  1: WelcomeStep,
  2: AgentCreationStep,
  3: DashboardTourStep,
  4: FirstTaskStep,
  5: CompletionStep
}

// Computed properties
const currentStepComponent = computed(() => stepComponents[currentStep.value as keyof typeof stepComponents])
const progressPercentage = computed(() => (currentStep.value / totalSteps) * 100)
const isLastStep = computed(() => currentStep.value === totalSteps)
const canProceed = computed(() => {
  // Logic to determine if user can proceed based on current step validation
  return onboardingStore.canProceedFromStep(currentStep.value)
})

// Event handlers
const handleNext = async () => {
  if (!canProceed.value) return
  
  trackStep(currentStep.value, 'completed', {
    timeSpent: Date.now() - stepStartTime.value,
    userData: userData.value
  })
  
  if (isLastStep.value) {
    await completeOnboarding()
  } else {
    transitionDirection.value = 'slide-left'
    currentStep.value++
    stepStartTime.value = Date.now()
  }
}

const handleBack = () => {
  if (currentStep.value > 1) {
    transitionDirection.value = 'slide-right'
    currentStep.value--
    stepStartTime.value = Date.now()
  }
}

const handleSkip = () => {
  const currentStepData = stepData.value[currentStep.value]
  if (currentStepData?.canSkip) {
    handleNext()
  } else {
    // Show confirmation dialog for exiting onboarding
    if (confirm('Are you sure you want to exit the setup process? You can complete it later.')) {
      trackEvent('onboarding_skipped', {
        step: currentStep.value,
        totalSteps,
        timeSpent: Date.now() - startTime.value.getTime()
      })
      router.push('/dashboard')
    }
  }
}

const handleDataUpdate = (data: Partial<UserData>) => {
  userData.value = { ...userData.value, ...data }
  onboardingStore.updateUserData(userData.value)
}

const handleAnalyticsEvent = (eventName: string, eventData: any) => {
  trackEvent(`onboarding_${eventName}`, {
    step: currentStep.value,
    ...eventData
  })
}

// Transition event handlers
const onBeforeEnter = () => {
  isLoading.value = true
  loadingMessage.value = 'Loading next step...'
}

const onAfterEnter = () => {
  isLoading.value = false
  trackEvent('onboarding_step_viewed', {
    step: currentStep.value,
    stepTitle: stepData.value[currentStep.value]?.title
  })
}

// Step completion tracking
const stepStartTime = ref<number>(Date.now())

const completeOnboarding = async () => {
  isLoading.value = true
  loadingMessage.value = 'Completing setup...'
  
  try {
    const totalTime = Date.now() - startTime.value.getTime()
    
    await onboardingStore.completeOnboarding({
      userData: userData.value,
      totalTime,
      completedSteps: totalSteps
    })
    
    trackEvent('onboarding_completed', {
      totalTime,
      completedSteps: totalSteps,
      userData: userData.value
    })
    
    // Redirect to dashboard with onboarding completion celebration
    router.push('/dashboard?onboarding=completed')
    
  } catch (error) {
    console.error('Error completing onboarding:', error)
    trackEvent('onboarding_error', {
      error: error.message,
      step: currentStep.value
    })
  } finally {
    isLoading.value = false
  }
}

// Lifecycle hooks
onMounted(async () => {
  // Initialize onboarding session
  await onboardingStore.initializeOnboarding()
  stepStartTime.value = Date.now()
  
  trackEvent('onboarding_started', {
    timestamp: startTime.value.toISOString(),
    userAgent: navigator.userAgent
  })
})

onBeforeUnmount(() => {
  // Save progress on component destroy
  onboardingStore.saveProgress({
    currentStep: currentStep.value,
    userData: userData.value,
    timeSpent: Date.now() - startTime.value.getTime()
  })
})

// Keyboard navigation
const handleKeydown = (event: KeyboardEvent) => {
  if (event.key === 'ArrowLeft' && currentStep.value > 1) {
    handleBack()
  } else if (event.key === 'ArrowRight' && canProceed.value) {
    handleNext()
  }
}

onMounted(() => {
  window.addEventListener('keydown', handleKeydown)
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', handleKeydown)
})
</script>

<style scoped>
/* Transition animations */
.slide-left-enter-active,
.slide-left-leave-active,
.slide-right-enter-active,
.slide-right-leave-active {
  transition: all 0.3s ease-in-out;
}

.slide-left-enter-from {
  opacity: 0;
  transform: translateX(30px);
}

.slide-left-leave-to {
  opacity: 0;
  transform: translateX(-30px);
}

.slide-right-enter-from {
  opacity: 0;
  transform: translateX(-30px);
}

.slide-right-leave-to {
  opacity: 0;
  transform: translateX(30px);
}

/* Mobile optimizations */
@media (max-width: 640px) {
  .min-h-screen {
    padding-bottom: 80px; /* Account for mobile navigation */
  }
}

/* Accessibility improvements */
@media (prefers-reduced-motion: reduce) {
  .slide-left-enter-active,
  .slide-left-leave-active,
  .slide-right-enter-active,
  .slide-right-leave-active {
    transition: opacity 0.2s ease;
  }
  
  .slide-left-enter-from,
  .slide-left-leave-to,
  .slide-right-enter-from,
  .slide-right-leave-to {
    transform: none;
  }
}
</style>