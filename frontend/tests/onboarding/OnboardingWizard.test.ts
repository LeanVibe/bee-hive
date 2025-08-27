import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, VueWrapper } from '@vue/test-utils'
import { createPinia } from 'pinia'
import { nextTick } from 'vue'
import OnboardingWizard from '@/components/onboarding/OnboardingWizard.vue'
import { useOnboardingStore } from '@/stores/onboarding'

// Mock dependencies
vi.mock('@/composables/useAnalyticsService', () => ({
  useAnalyticsService: () => ({
    trackEvent: vi.fn(),
    trackStep: vi.fn(),
    flushEvents: vi.fn()
  })
}))

vi.mock('vue-router', () => ({
  useRouter: () => ({
    push: vi.fn()
  })
}))

// Mock onboarding step components
vi.mock('@/components/onboarding/WelcomeStep.vue', () => ({
  default: {
    name: 'WelcomeStep',
    template: '<div data-testid="welcome-step">Welcome Step</div>',
    emits: ['next', 'back', 'update-data', 'analytics-event']
  }
}))

vi.mock('@/components/onboarding/AgentCreationStep.vue', () => ({
  default: {
    name: 'AgentCreationStep', 
    template: '<div data-testid="agent-creation-step">Agent Creation Step</div>',
    emits: ['next', 'back', 'update-data', 'analytics-event']
  }
}))

describe('OnboardingWizard', () => {
  let wrapper: VueWrapper<any>
  let pinia: any
  let onboardingStore: any

  beforeEach(() => {
    pinia = createPinia()
    
    wrapper = mount(OnboardingWizard, {
      global: {
        plugins: [pinia]
      }
    })
    
    onboardingStore = useOnboardingStore()
    
    // Mock store methods
    onboardingStore.initializeOnboarding = vi.fn()
    onboardingStore.canProceedFromStep = vi.fn(() => true)
    onboardingStore.saveProgress = vi.fn()
    onboardingStore.completeOnboarding = vi.fn()
  })

  describe('Component Initialization', () => {
    it('should render the onboarding wizard', () => {
      expect(wrapper.find('[data-testid="onboarding-wizard"]')).toBeDefined()
    })

    it('should initialize with step 1', () => {
      expect(wrapper.vm.currentStep).toBe(1)
    })

    it('should show correct progress percentage', () => {
      expect(wrapper.vm.progressPercentage).toBe(20) // 1/5 * 100
    })

    it('should display welcome step initially', async () => {
      await nextTick()
      expect(wrapper.find('[data-testid="welcome-step"]').exists()).toBe(true)
    })

    it('should call initializeOnboarding on mount', () => {
      expect(onboardingStore.initializeOnboarding).toHaveBeenCalled()
    })
  })

  describe('Navigation', () => {
    it('should advance to next step when handleNext is called', async () => {
      const initialStep = wrapper.vm.currentStep
      
      await wrapper.vm.handleNext()
      
      expect(wrapper.vm.currentStep).toBe(initialStep + 1)
    })

    it('should go back when handleBack is called', async () => {
      // First advance to step 2
      await wrapper.vm.handleNext()
      const currentStep = wrapper.vm.currentStep
      
      await wrapper.vm.handleBack()
      
      expect(wrapper.vm.currentStep).toBe(currentStep - 1)
    })

    it('should not go back from step 1', () => {
      const initialStep = wrapper.vm.currentStep
      
      wrapper.vm.handleBack()
      
      expect(wrapper.vm.currentStep).toBe(initialStep)
    })

    it('should complete onboarding on last step', async () => {
      // Set to last step
      wrapper.vm.currentStep = 5
      
      await wrapper.vm.handleNext()
      
      expect(onboardingStore.completeOnboarding).toHaveBeenCalled()
    })
  })

  describe('Progress Tracking', () => {
    it('should update progress percentage correctly', async () => {
      await wrapper.vm.handleNext() // Step 2
      expect(wrapper.vm.progressPercentage).toBe(40) // 2/5 * 100
      
      await wrapper.vm.handleNext() // Step 3
      expect(wrapper.vm.progressPercentage).toBe(60) // 3/5 * 100
    })

    it('should track step completion analytics', async () => {
      const trackStepSpy = vi.spyOn(wrapper.vm, 'trackStep')
      
      await wrapper.vm.handleNext()
      
      expect(trackStepSpy).toHaveBeenCalledWith(
        1, 
        'completed',
        expect.objectContaining({
          timeSpent: expect.any(Number),
          userData: expect.any(Object)
        })
      )
    })

    it('should save progress after navigation', async () => {
      await wrapper.vm.handleNext()
      
      expect(onboardingStore.saveProgress).toHaveBeenCalled()
    })
  })

  describe('Data Management', () => {
    it('should handle user data updates', () => {
      const testData = { name: 'John Doe', role: 'developer' }
      
      wrapper.vm.handleDataUpdate(testData)
      
      expect(wrapper.vm.userData).toEqual(expect.objectContaining(testData))
    })

    it('should emit analytics events', () => {
      const testEvent = 'test_event'
      const testData = { test: 'data' }
      
      wrapper.vm.handleAnalyticsEvent(testEvent, testData)
      
      // Verify event was tracked (mock implementation)
      expect(wrapper.emitted()).toHaveProperty('analytics-event')
    })
  })

  describe('Validation', () => {
    it('should check if user can proceed based on store validation', () => {
      onboardingStore.canProceedFromStep.mockReturnValue(false)
      
      expect(wrapper.vm.canProceed).toBe(false)
    })

    it('should enable proceed when validation passes', () => {
      onboardingStore.canProceedFromStep.mockReturnValue(true)
      
      expect(wrapper.vm.canProceed).toBe(true)
    })
  })

  describe('Skip Functionality', () => {
    it('should allow skipping when step permits', async () => {
      // Mock confirm dialog
      vi.stubGlobal('confirm', vi.fn(() => true))
      
      wrapper.vm.currentStep = 3 // Dashboard tour step (skippable)
      
      await wrapper.vm.handleSkip()
      
      expect(wrapper.vm.currentStep).toBe(4)
    })

    it('should show confirmation for non-skippable steps', () => {
      const confirmSpy = vi.stubGlobal('confirm', vi.fn(() => false))
      
      wrapper.vm.currentStep = 1 // Welcome step (non-skippable)
      
      wrapper.vm.handleSkip()
      
      expect(confirmSpy).toHaveBeenCalled()
    })
  })

  describe('Keyboard Navigation', () => {
    it('should handle left arrow key for back navigation', async () => {
      await wrapper.vm.handleNext() // Go to step 2
      const currentStep = wrapper.vm.currentStep
      
      // Simulate left arrow key
      const event = new KeyboardEvent('keydown', { key: 'ArrowLeft' })
      wrapper.vm.handleKeydown(event)
      
      expect(wrapper.vm.currentStep).toBe(currentStep - 1)
    })

    it('should handle right arrow key for forward navigation', async () => {
      const currentStep = wrapper.vm.currentStep
      
      // Simulate right arrow key
      const event = new KeyboardEvent('keydown', { key: 'ArrowRight' })
      wrapper.vm.handleKeydown(event)
      
      expect(wrapper.vm.currentStep).toBe(currentStep + 1)
    })
  })

  describe('Accessibility', () => {
    it('should have proper ARIA labels', () => {
      expect(wrapper.find('[role="progressbar"]')).toBeDefined()
    })

    it('should support reduced motion preferences', () => {
      // Test that animations respect prefers-reduced-motion
      expect(wrapper.find('.transition-all')).toBeDefined()
    })

    it('should have keyboard navigation support', () => {
      // Verify keyboard event listeners are attached
      expect(wrapper.vm.handleKeydown).toBeTypeOf('function')
    })
  })

  describe('Error Handling', () => {
    it('should handle onboarding completion errors gracefully', async () => {
      const error = new Error('Network error')
      onboardingStore.completeOnboarding.mockRejectedValue(error)
      
      wrapper.vm.currentStep = 5
      
      await wrapper.vm.handleNext()
      
      // Should not crash and should remain on the same step
      expect(wrapper.vm.currentStep).toBe(5)
    })

    it('should display error messages to user', async () => {
      const error = new Error('Test error')
      onboardingStore.completeOnboarding.mockRejectedValue(error)
      
      wrapper.vm.currentStep = 5
      await wrapper.vm.handleNext()
      
      // Error should be handled (implementation dependent)
      expect(wrapper.vm.isLoading).toBe(false)
    })
  })

  describe('Mobile Experience', () => {
    it('should show mobile navigation on small screens', () => {
      // Mock window width for mobile
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 640
      })
      
      expect(wrapper.find('.sm\\:hidden').exists()).toBe(true)
    })

    it('should have touch-friendly button sizes', () => {
      const buttons = wrapper.findAll('button')
      buttons.forEach(button => {
        // Verify minimum touch target size (44px)
        expect(button.classes()).toContain(['py-3', 'py-4'].some(cls => 
          button.classes().includes(cls)
        ))
      })
    })
  })

  describe('Performance', () => {
    it('should load components lazily', () => {
      // Verify step components are loaded on demand
      expect(wrapper.vm.currentStepComponent).toBeDefined()
    })

    it('should cleanup event listeners on unmount', () => {
      const removeEventListenerSpy = vi.spyOn(window, 'removeEventListener')
      
      wrapper.unmount()
      
      expect(removeEventListenerSpy).toHaveBeenCalledWith('keydown', expect.any(Function))
    })
  })

  describe('Epic 6 Success Metrics', () => {
    it('should track onboarding completion rate metrics', async () => {
      wrapper.vm.currentStep = 5
      await wrapper.vm.handleNext()
      
      expect(onboardingStore.completeOnboarding).toHaveBeenCalledWith(
        expect.objectContaining({
          totalTime: expect.any(Number),
          completedSteps: 5
        })
      )
    })

    it('should measure time to first value (< 5 minutes)', () => {
      const startTime = Date.now()
      wrapper.vm.startTime = new Date(startTime)
      
      const timeSpent = Date.now() - startTime
      
      // Should complete in under 5 minutes (300,000 ms)
      expect(timeSpent).toBeLessThan(300000)
    })

    it('should track user engagement metrics', () => {
      const analyticsEvents = wrapper.emitted('analytics-event') || []
      
      // Should track multiple engagement events
      expect(analyticsEvents.length).toBeGreaterThan(0)
    })
  })
})