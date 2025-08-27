import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useOnboardingStore } from '@/stores/onboarding'

// Mock API service
const mockApiService = {
  get: vi.fn(),
  post: vi.fn(), 
  put: vi.fn(),
}

vi.mock('@/services/api', () => ({
  apiService: mockApiService
}))

describe('useOnboardingStore', () => {
  let store: ReturnType<typeof useOnboardingStore>

  beforeEach(() => {
    setActivePinia(createPinia())
    store = useOnboardingStore()
    vi.clearAllMocks()
    
    // Reset localStorage
    localStorage.clear()
  })

  describe('Initial State', () => {
    it('should initialize with default values', () => {
      expect(store.isOnboarding).toBe(false)
      expect(store.currentStep).toBe(1)
      expect(store.completionPercentage).toBe(0)
      expect(store.hasStarted).toBe(false)
      expect(store.isComplete).toBe(false)
    })

    it('should have empty user data initially', () => {
      expect(store.userData).toEqual({})
    })

    it('should not be loading initially', () => {
      expect(store.isLoading).toBe(false)
      expect(store.error).toBe(null)
    })
  })

  describe('Step Validation', () => {
    it('should validate welcome step correctly', () => {
      // Invalid - missing required fields
      expect(store.canProceedFromStep(1)).toBe(false)

      // Valid - all required fields present
      store.updateUserData({
        name: 'John Doe',
        role: 'developer',
        goals: ['automate_workflows']
      })
      expect(store.canProceedFromStep(1)).toBe(true)
    })

    it('should validate agent creation step', () => {
      // Invalid - no agent created
      expect(store.canProceedFromStep(2)).toBe(false)

      // Valid - agent created
      store.updateUserData({
        createdAgent: {
          id: 'agent-123',
          name: 'Test Agent',
          type: 'workflow_automator'
        }
      })
      expect(store.canProceedFromStep(2)).toBe(true)
    })

    it('should validate dashboard tour step (always true)', () => {
      expect(store.canProceedFromStep(3)).toBe(true)
    })

    it('should validate first task step', () => {
      // Invalid - no tasks completed
      expect(store.canProceedFromStep(4)).toBe(false)

      // Valid - has completed tasks
      store.updateUserData({
        completedTasks: ['task-123']
      })
      expect(store.canProceedFromStep(4)).toBe(true)
    })

    it('should validate completion step (always true)', () => {
      expect(store.canProceedFromStep(5)).toBe(true)
    })
  })

  describe('Onboarding Initialization', () => {
    it('should initialize new onboarding session', async () => {
      mockApiService.get.mockResolvedValue({ data: null })
      mockApiService.post.mockResolvedValue({ data: { success: true } })

      await store.initializeOnboarding()

      expect(store.isOnboarding).toBe(true)
      expect(store.isLoading).toBe(false)
      expect(mockApiService.post).toHaveBeenCalledWith(
        '/api/onboarding/start',
        expect.objectContaining({
          startedAt: expect.any(String),
          userAgent: expect.any(String)
        })
      )
    })

    it('should resume existing onboarding session', async () => {
      const existingProgress = {
        data: {
          currentStep: 3,
          completedSteps: [1, 2],
          userData: { name: 'John' },
          startedAt: new Date().toISOString()
        }
      }
      
      mockApiService.get.mockResolvedValue(existingProgress)

      await store.initializeOnboarding()

      expect(store.currentStep).toBe(3)
      expect(store.progress.completedSteps).toEqual([1, 2])
      expect(store.userData.name).toBe('John')
    })

    it('should handle initialization errors', async () => {
      const error = new Error('Network error')
      mockApiService.get.mockRejectedValue(error)

      await store.initializeOnboarding()

      expect(store.error).toBe('Network error')
      expect(store.isLoading).toBe(false)
    })
  })

  describe('User Data Management', () => {
    it('should update user data', () => {
      const testData = {
        name: 'Jane Doe',
        role: 'manager',
        goals: ['scale_operations']
      }

      store.updateUserData(testData)

      expect(store.userData).toEqual(expect.objectContaining(testData))
    })

    it('should merge user data updates', () => {
      store.updateUserData({ name: 'John' })
      store.updateUserData({ role: 'developer' })

      expect(store.userData).toEqual({
        name: 'John',
        role: 'developer'
      })
    })

    it('should save progress after data update', async () => {
      mockApiService.put.mockResolvedValue({ data: { success: true } })

      await store.updateUserData({ name: 'Test User' })

      expect(mockApiService.put).toHaveBeenCalledWith(
        '/api/onboarding/progress',
        expect.objectContaining({
          userData: expect.objectContaining({ name: 'Test User' })
        })
      )
    })
  })

  describe('Step Completion', () => {
    it('should complete a step', async () => {
      mockApiService.post.mockResolvedValue({ data: { success: true } })

      await store.completeStep(1, { stepData: 'test' })

      expect(store.progress.completedSteps).toContain(1)
      expect(mockApiService.post).toHaveBeenCalledWith(
        '/api/onboarding/step-completed',
        expect.objectContaining({
          step: 1,
          stepData: { stepData: 'test' }
        })
      )
    })

    it('should not duplicate completed steps', async () => {
      mockApiService.post.mockResolvedValue({ data: { success: true } })

      await store.completeStep(1)
      await store.completeStep(1) // Complete same step again

      expect(store.progress.completedSteps.filter(step => step === 1)).toHaveLength(1)
    })
  })

  describe('Progress Tracking', () => {
    it('should calculate completion percentage correctly', () => {
      store.progress.completedSteps = [1, 2, 3]
      expect(store.completionPercentage).toBe(60) // 3/5 * 100
    })

    it('should save progress to API', async () => {
      mockApiService.put.mockResolvedValue({ data: { success: true } })

      await store.saveProgress()

      expect(mockApiService.put).toHaveBeenCalledWith(
        '/api/onboarding/progress',
        expect.objectContaining({
          currentStep: expect.any(Number),
          completedSteps: expect.any(Array),
          timeSpent: expect.any(Number)
        })
      )
    })

    it('should handle save progress errors gracefully', async () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      mockApiService.put.mockRejectedValue(new Error('Save failed'))

      await store.saveProgress()

      expect(consoleSpy).toHaveBeenCalledWith(
        'Failed to save onboarding progress:',
        expect.any(Error)
      )
      
      consoleSpy.mockRestore()
    })
  })

  describe('Agent Creation', () => {
    it('should create an agent successfully', async () => {
      const agentData = {
        name: 'Test Agent',
        type: 'workflow_automator',
        description: 'Test description',
        capabilities: ['automation', 'monitoring']
      }
      
      const mockResponse = {
        data: {
          id: 'agent-123',
          name: 'Test Agent',
          type: 'workflow_automator'
        }
      }

      mockApiService.post
        .mockResolvedValueOnce(mockResponse) // createAgent call
        .mockResolvedValueOnce({ data: { success: true } }) // completeStep call

      const result = await store.createAgent(agentData)

      expect(result).toEqual(mockResponse.data)
      expect(store.userData.createdAgent).toEqual({
        id: 'agent-123',
        name: 'Test Agent', 
        type: 'workflow_automator'
      })
      expect(mockApiService.post).toHaveBeenCalledWith('/api/agents', agentData)
    })

    it('should handle agent creation errors', async () => {
      const error = new Error('Agent creation failed')
      mockApiService.post.mockRejectedValue(error)

      await expect(store.createAgent({
        name: 'Test',
        type: 'test'
      })).rejects.toThrow('Agent creation failed')

      expect(store.error).toBe('Agent creation failed')
    })
  })

  describe('Task Creation', () => {
    it('should create first task successfully', async () => {
      // Setup: Create agent first
      store.updateUserData({
        createdAgent: { id: 'agent-123', name: 'Test Agent', type: 'test' }
      })

      const taskData = {
        title: 'Test Task',
        description: 'Test description',
        agentId: 'agent-123',
        priority: 'medium'
      }

      const mockResponse = {
        data: {
          id: 'task-123',
          title: 'Test Task'
        }
      }

      mockApiService.post
        .mockResolvedValueOnce(mockResponse) // createFirstTask call
        .mockResolvedValueOnce({ data: { success: true } }) // completeStep call

      const result = await store.createFirstTask(taskData)

      expect(result).toEqual(mockResponse.data)
      expect(store.userData.completedTasks).toContain('task-123')
    })

    it('should handle task creation errors', async () => {
      const error = new Error('Task creation failed')
      mockApiService.post.mockRejectedValue(error)

      await expect(store.createFirstTask({
        title: 'Test',
        agentId: 'agent-123'
      })).rejects.toThrow('Task creation failed')

      expect(store.error).toBe('Task creation failed')
    })
  })

  describe('Onboarding Completion', () => {
    it('should complete onboarding successfully', async () => {
      const completionData = {
        userData: { name: 'John', role: 'developer', goals: ['automate'] },
        totalTime: 240000,
        completedSteps: 5
      }

      mockApiService.post.mockResolvedValue({
        data: { success: true, sessionId: 'session-123' }
      })

      const result = await store.completeOnboarding(completionData)

      expect(store.isOnboarding).toBe(false)
      expect(localStorage.getItem('onboarding_completed')).toBe('true')
      expect(result).toEqual({ success: true, sessionId: 'session-123' })
    })

    it('should handle completion errors', async () => {
      const error = new Error('Completion failed')
      mockApiService.post.mockRejectedValue(error)

      await expect(store.completeOnboarding({
        userData: {},
        totalTime: 0,
        completedSteps: 5
      })).rejects.toThrow('Completion failed')

      expect(store.error).toBe('Completion failed')
      expect(store.isOnboarding).toBe(false) // Should still reset state
    })
  })

  describe('Skip Functionality', () => {
    it('should skip onboarding', async () => {
      mockApiService.post.mockResolvedValue({ data: { success: true } })

      await store.skipOnboarding()

      expect(store.isOnboarding).toBe(false)
      expect(localStorage.getItem('onboarding_skipped')).toBe('true')
      expect(mockApiService.post).toHaveBeenCalledWith(
        '/api/onboarding/skip',
        expect.objectContaining({
          currentStep: expect.any(Number),
          skippedAt: expect.any(String)
        })
      )
    })
  })

  describe('State Reset', () => {
    it('should reset onboarding state', () => {
      // Setup some state
      store.progress.currentStep = 3
      store.progress.completedSteps = [1, 2]
      store.updateUserData({ name: 'Test' })
      store.isOnboarding = true
      localStorage.setItem('onboarding_completed', 'true')

      store.resetOnboarding()

      expect(store.currentStep).toBe(1)
      expect(store.progress.completedSteps).toEqual([])
      expect(store.userData).toEqual({})
      expect(store.isOnboarding).toBe(false)
      expect(localStorage.getItem('onboarding_completed')).toBe(null)
    })
  })

  describe('Completion Status Check', () => {
    it('should detect completed onboarding from localStorage', () => {
      localStorage.setItem('onboarding_completed', 'true')

      const status = store.checkCompletionStatus()

      expect(status.isCompleted).toBe(true)
      expect(status.isSkipped).toBe(false)
      expect(store.isOnboarding).toBe(false)
    })

    it('should detect skipped onboarding from localStorage', () => {
      localStorage.setItem('onboarding_skipped', 'true')

      const status = store.checkCompletionStatus()

      expect(status.isCompleted).toBe(false)
      expect(status.isSkipped).toBe(true)
      expect(store.isOnboarding).toBe(false)
    })
  })

  describe('Analytics Integration', () => {
    it('should provide onboarding analytics', () => {
      store.progress.startedAt = new Date(Date.now() - 120000) // 2 minutes ago
      store.progress.completedSteps = [1, 2, 3]
      store.progress.currentStep = 4
      store.updateUserData({ name: 'Test', role: 'developer' })

      const analytics = store.getOnboardingAnalytics

      expect(analytics).toEqual({
        timeSpent: expect.any(Number),
        completionRate: 60, // 3/5 * 100
        currentStep: 4,
        userData: expect.objectContaining({ name: 'Test' }),
        dropOffPoint: 4 // Current step since not fully complete
      })
    })

    it('should not show drop-off point when completed', () => {
      store.progress.completedSteps = [1, 2, 3, 4, 5]
      
      const analytics = store.getOnboardingAnalytics

      expect(analytics.dropOffPoint).toBe(null)
    })
  })

  describe('Epic 6 Success Metrics Validation', () => {
    it('should track 90%+ completion rate target', () => {
      store.progress.completedSteps = [1, 2, 3, 4] // 80% completion
      expect(store.completionPercentage).toBe(80)

      store.progress.completedSteps = [1, 2, 3, 4, 5] // 100% completion
      expect(store.completionPercentage).toBe(100)
      expect(store.completionPercentage).toBeGreaterThan(90) // Target achieved
    })

    it('should track time-to-value (< 5 minutes)', () => {
      const startTime = Date.now() - (4 * 60 * 1000) // 4 minutes ago
      store.progress.startedAt = new Date(startTime)

      const analytics = store.getOnboardingAnalytics
      expect(analytics.timeSpent).toBeLessThan(300000) // Less than 5 minutes
    })

    it('should support user engagement tracking', () => {
      // Verify user data captures engagement signals
      store.updateUserData({
        goals: ['automate_workflows', 'improve_efficiency'],
        preferences: { notifications: true, theme: 'dark' }
      })

      expect(store.userData.goals).toHaveLength(2) // Multiple goals = higher engagement
      expect(store.userData.preferences).toBeDefined() // Preferences set = engagement
    })
  })
})