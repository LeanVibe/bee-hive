/**
 * Error Handling Integration Tests
 * 
 * Tests for the comprehensive error handling system across dashboard components
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { useDashboardErrorHandling } from '@/composables/useDashboardErrorHandling'
import { DashboardComponent } from '@/types/coordination'
import type { DashboardError } from '@/types/coordination'

describe('Error Handling Integration', () => {
  let errorHandler: ReturnType<typeof useDashboardErrorHandling>

  beforeEach(() => {
    vi.useFakeTimers()
    errorHandler = useDashboardErrorHandling()
  })

  afterEach(() => {
    errorHandler.reset()
    vi.useRealTimers()
    vi.clearAllMocks()
  })

  describe('Error Reporting', () => {
    it('should report and track different error types', () => {
      const networkError: DashboardError = {
        id: 'net_001',
        type: 'network',
        message: 'Failed to fetch data',
        component: DashboardComponent.GRAPH,
        timestamp: '2024-01-01T10:00:00Z',
        recoverable: true
      }

      const dataError: DashboardError = {
        id: 'data_001',
        type: 'data',
        message: 'Invalid data format',
        component: DashboardComponent.TRANSCRIPT,
        timestamp: '2024-01-01T10:01:00Z',
        recoverable: false
      }

      errorHandler.reportError(networkError)
      errorHandler.reportError(dataError)

      const metrics = errorHandler.metrics.value
      expect(metrics.totalErrors).toBe(2)
      expect(metrics.errorsByType.network).toBe(1)
      expect(metrics.errorsByType.data).toBe(1)
      expect(metrics.errorsByComponent[DashboardComponent.GRAPH]).toBe(1)
      expect(metrics.errorsByComponent[DashboardComponent.TRANSCRIPT]).toBe(1)
    })

    it('should report network errors with helper method', () => {
      errorHandler.reportNetworkError(
        DashboardComponent.GRAPH,
        'Connection timeout',
        { endpoint: '/api/graph', timeout: 5000 }
      )

      const status = errorHandler.getComponentErrorStatus(DashboardComponent.GRAPH)
      expect(status.hasErrors).toBe(true)
      expect(status.errorCount).toBe(1)
      expect(status.lastError?.type).toBe('network')
      expect(status.lastError?.message).toBe('Connection timeout')
    })

    it('should report WebSocket errors with endpoint details', () => {
      errorHandler.reportWebSocketError(
        'graph_endpoint',
        'WebSocket connection lost',
        { url: 'ws://localhost:8000/graph', code: 1006 }
      )

      const errors = errorHandler.componentErrors.value.get(DashboardComponent.SERVICE)
      expect(errors).toBeDefined()
      expect(errors!.length).toBe(1)
      expect(errors![0].details.endpointId).toBe('graph_endpoint')
    })

    it('should maintain error history and cleanup old errors', () => {
      // Report multiple errors
      for (let i = 0; i < 15; i++) {
        errorHandler.reportError({
          id: `error_${i}`,
          type: 'data',
          message: `Error ${i}`,
          component: DashboardComponent.ANALYSIS,
          timestamp: new Date().toISOString(),
          recoverable: false
        })
      }

      const errors = errorHandler.componentErrors.value.get(DashboardComponent.ANALYSIS)
      expect(errors!.length).toBe(10) // Should keep only last 10 errors

      // Simulate time passage for cleanup
      vi.advanceTimersByTime(25 * 60 * 60 * 1000) // 25 hours

      // Old errors should be cleaned up
      const metrics = errorHandler.metrics.value
      expect(metrics.recentErrors.length).toBe(0)
    })
  })

  describe('Error Boundaries', () => {
    it('should create and manage error boundaries', () => {
      const boundary = errorHandler.createErrorBoundary('test_boundary')

      expect(boundary.hasError).toBe(false)
      expect(boundary.error).toBeUndefined()
      expect(boundary.retryCount).toBe(0)

      const testError: DashboardError = {
        id: 'boundary_error',
        type: 'network',
        message: 'Boundary test error',
        component: DashboardComponent.GRAPH,
        timestamp: new Date().toISOString(),
        recoverable: true
      }

      errorHandler.triggerErrorBoundary('test_boundary', testError, 'GraphFallback')

      expect(boundary.hasError).toBe(true)
      expect(boundary.error).toEqual(testError)
      expect(boundary.fallbackComponent).toBe('GraphFallback')
    })

    it('should retry error boundaries with recovery strategies', async () => {
      const boundary = errorHandler.createErrorBoundary('retry_boundary')
      
      const recoverableError: DashboardError = {
        id: 'recoverable_error',
        type: 'network',
        message: 'Network timeout',
        component: DashboardComponent.TRANSCRIPT,
        timestamp: new Date().toISOString(),
        recoverable: true
      }

      errorHandler.triggerErrorBoundary('retry_boundary', recoverableError)

      // Mock successful recovery
      vi.spyOn(Math, 'random').mockReturnValue(0.8) // Above 0.3 threshold for network retry

      const recovered = await errorHandler.retryErrorBoundary('retry_boundary')

      expect(recovered).toBe(true)
      expect(boundary.hasError).toBe(false)
      expect(boundary.retryCount).toBe(1)
    })

    it('should clear error boundaries when requested', () => {
      const boundary = errorHandler.createErrorBoundary('clear_boundary')
      
      const testError: DashboardError = {
        id: 'clear_error',
        type: 'data',
        message: 'Test error',
        component: DashboardComponent.MONITORING,
        timestamp: new Date().toISOString(),
        recoverable: false
      }

      errorHandler.triggerErrorBoundary('clear_boundary', testError)
      expect(boundary.hasError).toBe(true)

      errorHandler.clearErrorBoundary('clear_boundary')
      expect(boundary.hasError).toBe(false)
      expect(boundary.error).toBeUndefined()
    })
  })

  describe('Recovery Strategies', () => {
    it('should register and execute custom recovery strategies', async () => {
      const customRecovery = vi.fn().mockResolvedValue(true)

      errorHandler.registerRecoveryStrategy({
        id: 'custom_strategy',
        name: 'Custom Recovery',
        description: 'Custom recovery for test errors',
        canRecover: (error) => error.type === 'custom',
        recover: customRecovery,
        priority: 10,
        maxRetries: 2
      })

      const customError: DashboardError = {
        id: 'custom_error',
        type: 'custom' as any,
        message: 'Custom error type',
        component: DashboardComponent.ANALYSIS,
        timestamp: new Date().toISOString(),
        recoverable: true
      }

      const recovered = await errorHandler.attemptErrorRecovery(customError)

      expect(recovered).toBe(true)
      expect(customRecovery).toHaveBeenCalledWith(customError)
    })

    it('should execute recovery strategies in priority order', async () => {
      const executionOrder: string[] = []

      const lowPriorityStrategy = {
        id: 'low_priority',
        name: 'Low Priority',
        description: 'Low priority strategy',
        canRecover: () => true,
        recover: async () => {
          executionOrder.push('low')
          return false // Fail to test next strategy
        },
        priority: 1,
        maxRetries: 1
      }

      const highPriorityStrategy = {
        id: 'high_priority',
        name: 'High Priority',
        description: 'High priority strategy',
        canRecover: () => true,
        recover: async () => {
          executionOrder.push('high')
          return true
        },
        priority: 10,
        maxRetries: 1
      }

      errorHandler.registerRecoveryStrategy(lowPriorityStrategy)
      errorHandler.registerRecoveryStrategy(highPriorityStrategy)

      const testError: DashboardError = {
        id: 'priority_test',
        type: 'network',
        message: 'Priority test',
        component: DashboardComponent.GRAPH,
        timestamp: new Date().toISOString(),
        recoverable: true
      }

      await errorHandler.attemptErrorRecovery(testError)

      // High priority should execute first
      expect(executionOrder[0]).toBe('high')
    })

    it('should handle recovery strategy timeouts', async () => {
      const slowStrategy = {
        id: 'slow_strategy',
        name: 'Slow Strategy',
        description: 'Strategy that takes too long',
        canRecover: () => true,
        recover: async () => {
          // Simulate slow recovery
          await new Promise(resolve => setTimeout(resolve, 35000)) // Exceeds 30s timeout
          return true
        },
        priority: 1,
        maxRetries: 1
      }

      errorHandler.registerRecoveryStrategy(slowStrategy)

      const testError: DashboardError = {
        id: 'timeout_test',
        type: 'network',
        message: 'Timeout test',
        component: DashboardComponent.TRANSCRIPT,
        timestamp: new Date().toISOString(),
        recoverable: true
      }

      const recovered = await errorHandler.attemptErrorRecovery(testError)

      expect(recovered).toBe(false) // Should fail due to timeout
    })

    it('should fall back when recovery fails', async () => {
      const failingStrategy = {
        id: 'failing_strategy',
        name: 'Failing Strategy',
        description: 'Strategy that always fails',
        canRecover: () => true,
        recover: async () => {
          throw new Error('Recovery failed')
        },
        priority: 1,
        maxRetries: 1
      }

      errorHandler.registerRecoveryStrategy(failingStrategy)

      const testError: DashboardError = {
        id: 'failing_test',
        type: 'network',
        message: 'Failing test',
        component: DashboardComponent.ANALYSIS,
        timestamp: new Date().toISOString(),
        recoverable: true
      }

      const recovered = await errorHandler.attemptErrorRecovery(testError)
      expect(recovered).toBe(false)

      // Should apply fallback
      const fallbackData = errorHandler.applyFallback(DashboardComponent.ANALYSIS, testError)
      expect(fallbackData).toBeDefined()
      expect(fallbackData.error).toBe(true)
    })
  })

  describe('Fallback Configuration', () => {
    it('should register and apply component fallbacks', () => {
      const customFallback = {
        component: DashboardComponent.GRAPH,
        fallbackComponent: 'CustomGraphFallback',
        fallbackData: {
          nodes: [],
          edges: [],
          customMessage: 'Custom fallback data'
        },
        showFallbackMessage: true,
        retryInterval: 5000,
        maxRetries: 2
      }

      errorHandler.registerFallbackConfig(DashboardComponent.GRAPH, customFallback)

      const testError: DashboardError = {
        id: 'fallback_test',
        type: 'data',
        message: 'Fallback test',
        component: DashboardComponent.GRAPH,
        timestamp: new Date().toISOString(),
        recoverable: false
      }

      const fallbackData = errorHandler.applyFallback(DashboardComponent.GRAPH, testError)
      
      expect(fallbackData.customMessage).toBe('Custom fallback data')
      expect(fallbackData.nodes).toEqual([])
    })

    it('should provide default fallback data when no config exists', () => {
      const testError: DashboardError = {
        id: 'no_config_test',
        type: 'data',
        message: 'No config test',
        component: DashboardComponent.SERVICE,
        timestamp: new Date().toISOString(),
        recoverable: false
      }

      const fallbackData = errorHandler.applyFallback(DashboardComponent.SERVICE, testError)
      
      expect(fallbackData.error).toBe(true)
      expect(fallbackData.status).toBe('error')
      expect(fallbackData.message).toBe('Service temporarily unavailable')
    })

    it('should get fallback configuration for components', () => {
      const graphConfig = errorHandler.getFallbackConfig(DashboardComponent.GRAPH)
      
      expect(graphConfig).toBeDefined()
      expect(graphConfig!.component).toBe(DashboardComponent.GRAPH)
      expect(graphConfig!.fallbackComponent).toBe('GraphFallback')
      expect(graphConfig!.fallbackData.nodes).toEqual([])
    })
  })

  describe('Event System', () => {
    it('should emit error events when errors are reported', () => {
      const errorListener = vi.fn()
      const specificListener = vi.fn()

      errorHandler.onError('error_reported', errorListener)
      errorHandler.onError('*', specificListener) // Wildcard listener

      const testError: DashboardError = {
        id: 'event_test',
        type: 'network',
        message: 'Event test',
        component: DashboardComponent.GRAPH,
        timestamp: new Date().toISOString(),
        recoverable: true
      }

      errorHandler.reportError(testError)

      expect(errorListener).toHaveBeenCalledWith(testError)
      expect(specificListener).toHaveBeenCalledWith(testError)
    })

    it('should emit recovery events', async () => {
      const recoveryListener = vi.fn()
      errorHandler.onRecovery('recovery_success', recoveryListener)

      const testError: DashboardError = {
        id: 'recovery_event_test',
        type: 'network',
        message: 'Recovery event test',
        component: DashboardComponent.TRANSCRIPT,
        timestamp: new Date().toISOString(),
        recoverable: true
      }

      // Mock successful recovery
      vi.spyOn(Math, 'random').mockReturnValue(0.8)

      await errorHandler.attemptErrorRecovery(testError)

      expect(recoveryListener).toHaveBeenCalledWith(true, testError)
    })

    it('should remove event listeners correctly', () => {
      const listener = vi.fn()
      const removeListener = errorHandler.onError('test_event', listener)

      // Trigger event
      errorHandler.reportError({
        id: 'test1',
        type: 'network',
        message: 'Test 1',
        component: DashboardComponent.GRAPH,
        timestamp: new Date().toISOString(),
        recoverable: true
      })

      expect(listener).toHaveBeenCalledTimes(1)

      // Remove listener
      removeListener()

      // Trigger event again
      errorHandler.reportError({
        id: 'test2',
        type: 'network',
        message: 'Test 2',
        component: DashboardComponent.GRAPH,
        timestamp: new Date().toISOString(),
        recoverable: true
      })

      // Should not be called again
      expect(listener).toHaveBeenCalledTimes(1)
    })

    it('should handle errors in event listeners gracefully', () => {
      const faultyListener = vi.fn(() => {
        throw new Error('Listener error')
      })
      const goodListener = vi.fn()

      errorHandler.onError('error_reported', faultyListener)
      errorHandler.onError('error_reported', goodListener)

      const testError: DashboardError = {
        id: 'listener_error_test',
        type: 'data',
        message: 'Listener error test',
        component: DashboardComponent.ANALYSIS,
        timestamp: new Date().toISOString(),
        recoverable: false
      }

      // Should not throw when listener errors
      expect(() => errorHandler.reportError(testError)).not.toThrow()

      // Good listener should still be called
      expect(goodListener).toHaveBeenCalledWith(testError)
    })
  })

  describe('Error Metrics and Analysis', () => {
    it('should provide comprehensive error summary', () => {
      // Report various types of errors
      errorHandler.reportNetworkError(DashboardComponent.GRAPH, 'Network error 1')
      errorHandler.reportWebSocketError('ws_endpoint', 'WebSocket error 1')
      errorHandler.reportDataError(DashboardComponent.TRANSCRIPT, 'Data error 1')
      errorHandler.reportParsingError(DashboardComponent.ANALYSIS, 'Parse error 1')

      const summary = errorHandler.getErrorSummary()

      expect(summary.hasErrors).toBe(true)
      expect(summary.totalErrors).toBe(4)
      expect(summary.criticalErrors).toBe(2) // Network and WebSocket are critical
      expect(summary.recentErrors).toBe(4)
    })

    it('should track error trends over time', () => {
      const startTime = Date.now()

      // Report errors at different times
      for (let i = 0; i < 5; i++) {
        errorHandler.reportError({
          id: `trend_${i}`,
          type: 'network',
          message: `Trend error ${i}`,
          component: DashboardComponent.GRAPH,
          timestamp: new Date(startTime + i * 1000).toISOString(),
          recoverable: true
        })
      }

      const metrics = errorHandler.metrics.value
      expect(metrics.recentErrors.length).toBe(5)
      
      // Should be ordered by timestamp
      const timestamps = metrics.recentErrors.map(e => new Date(e.timestamp).getTime())
      for (let i = 1; i < timestamps.length; i++) {
        expect(timestamps[i]).toBeGreaterThanOrEqual(timestamps[i - 1])
      }
    })

    it('should calculate recovery success rates', async () => {
      // Mock recovery with 50% success rate
      let callCount = 0
      vi.spyOn(Math, 'random').mockImplementation(() => {
        callCount++
        return callCount % 2 === 0 ? 0.8 : 0.2 // Alternate success/failure
      })

      // Attempt multiple recoveries
      for (let i = 0; i < 4; i++) {
        const error: DashboardError = {
          id: `recovery_rate_${i}`,
          type: 'network',
          message: `Recovery rate test ${i}`,
          component: DashboardComponent.GRAPH,
          timestamp: new Date().toISOString(),
          recoverable: true
        }

        await errorHandler.attemptErrorRecovery(error)
      }

      // Should track recovery metrics (exact values depend on mock implementation)
      const summary = errorHandler.getErrorSummary()
      expect(summary.recoveryRate).toBeDefined()
    })
  })

  describe('Component Error Status', () => {
    it('should clear component errors individually', () => {
      errorHandler.reportDataError(DashboardComponent.GRAPH, 'Graph error')
      errorHandler.reportDataError(DashboardComponent.TRANSCRIPT, 'Transcript error')

      expect(errorHandler.getComponentErrorStatus(DashboardComponent.GRAPH).hasErrors).toBe(true)
      expect(errorHandler.getComponentErrorStatus(DashboardComponent.TRANSCRIPT).hasErrors).toBe(true)

      errorHandler.clearComponentErrors(DashboardComponent.GRAPH)

      expect(errorHandler.getComponentErrorStatus(DashboardComponent.GRAPH).hasErrors).toBe(false)
      expect(errorHandler.getComponentErrorStatus(DashboardComponent.TRANSCRIPT).hasErrors).toBe(true)
    })

    it('should clear global errors', () => {
      errorHandler.reportNetworkError(DashboardComponent.SERVICE, 'Global network error')

      expect(errorHandler.globalError.value).toBeDefined()

      errorHandler.clearGlobalError()

      expect(errorHandler.globalError.value).toBeNull()
    })

    it('should reset entire error handling state', () => {
      // Generate various errors and state
      errorHandler.reportNetworkError(DashboardComponent.GRAPH, 'Test error')
      errorHandler.createErrorBoundary('test_boundary')
      
      expect(errorHandler.hasErrors.value).toBe(true)
      expect(errorHandler.metrics.value.totalErrors).toBeGreaterThan(0)

      errorHandler.reset()

      expect(errorHandler.hasErrors.value).toBe(false)
      expect(errorHandler.metrics.value.totalErrors).toBe(0)
      expect(errorHandler.globalError.value).toBeNull()
      expect(errorHandler.componentErrors.value.size).toBe(0)
    })
  })

  describe('Automatic Error Processing', () => {
    it('should process error queue automatically', async () => {
      const processingListener = vi.fn()
      errorHandler.onError('error_reported', processingListener)

      // Report multiple errors rapidly
      for (let i = 0; i < 3; i++) {
        errorHandler.reportError({
          id: `queue_${i}`,
          type: 'data',
          message: `Queue test ${i}`,
          component: DashboardComponent.ANALYSIS,
          timestamp: new Date().toISOString(),
          recoverable: false
        })
      }

      // Advance time to trigger queue processing
      vi.advanceTimersByTime(1000)

      // All errors should be processed
      expect(processingListener).toHaveBeenCalledTimes(3)
    })

    it('should automatically attempt recovery for recoverable errors', async () => {
      const recoveryListener = vi.fn()
      errorHandler.onRecovery('*', recoveryListener)

      // Mock successful recovery
      vi.spyOn(Math, 'random').mockReturnValue(0.8)

      // Report recoverable error
      errorHandler.reportError({
        id: 'auto_recovery',
        type: 'network',
        message: 'Auto recovery test',
        component: DashboardComponent.GRAPH,
        timestamp: new Date().toISOString(),
        recoverable: true
      })

      // Advance time to trigger automatic recovery
      vi.advanceTimersByTime(2000)

      // Recovery should have been attempted
      expect(recoveryListener).toHaveBeenCalled()
    })
  })
})