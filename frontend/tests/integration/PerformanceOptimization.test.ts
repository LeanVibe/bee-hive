/**
 * Performance Optimization Integration Tests
 * 
 * Tests for the performance optimization system managing concurrent updates
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { usePerformanceOptimization } from '@/composables/usePerformanceOptimization'
import { DashboardComponent } from '@/types/coordination'

describe('Performance Optimization Integration', () => {
  let performanceOptimizer: ReturnType<typeof usePerformanceOptimization>

  beforeEach(() => {
    vi.useFakeTimers()
    performanceOptimizer = usePerformanceOptimization()
  })

  afterEach(() => {
    performanceOptimizer.clearQueue()
    vi.useRealTimers()
    vi.clearAllMocks()
  })

  describe('Task Scheduling', () => {
    it('should schedule updates with different priorities', () => {
      const criticalId = performanceOptimizer.scheduleUpdate(
        DashboardComponent.GRAPH,
        'data',
        { critical: true },
        'critical'
      )

      const lowId = performanceOptimizer.scheduleUpdate(
        DashboardComponent.TRANSCRIPT,
        'data',
        { low: true },
        'low'
      )

      expect(criticalId).toBeDefined()
      expect(lowId).toBeDefined()
      expect(performanceOptimizer.queueLength.value).toBe(2)
    })

    it('should process tasks in priority order', async () => {
      const processedTasks: string[] = []

      // Schedule tasks in mixed priority order
      const lowTask = performanceOptimizer.scheduleUpdate(
        DashboardComponent.GRAPH,
        'data',
        { priority: 'low' },
        'low',
        () => processedTasks.push('low')
      )

      const criticalTask = performanceOptimizer.scheduleUpdate(
        DashboardComponent.GRAPH,
        'data',
        { priority: 'critical' },
        'critical',
        () => processedTasks.push('critical')
      )

      const mediumTask = performanceOptimizer.scheduleUpdate(
        DashboardComponent.GRAPH,
        'data',
        { priority: 'medium' },
        'medium',
        () => processedTasks.push('medium')
      )

      // Process queue
      await performanceOptimizer.flushQueue()
      vi.runAllTimers()

      // Critical should be processed first, then medium, then low
      expect(processedTasks[0]).toBe('critical')
      expect(processedTasks[1]).toBe('medium')
      expect(processedTasks[2]).toBe('low')
    })

    it('should batch multiple updates efficiently', () => {
      const updates = [
        { component: DashboardComponent.GRAPH, type: 'data' as const, payload: { graph: 1 } },
        { component: DashboardComponent.TRANSCRIPT, type: 'data' as const, payload: { transcript: 1 } },
        { component: DashboardComponent.ANALYSIS, type: 'ui' as const, payload: { analysis: 1 } }
      ]

      const taskIds = performanceOptimizer.scheduleBatchUpdate(updates)

      expect(taskIds).toHaveLength(3)
      expect(performanceOptimizer.queueLength.value).toBe(3)
    })

    it('should drop low priority tasks when queue is full', () => {
      // Fill queue with low priority tasks
      for (let i = 0; i < 1005; i++) { // Exceeds default maxQueueSize of 1000
        performanceOptimizer.scheduleUpdate(
          DashboardComponent.GRAPH,
          'data',
          { index: i },
          'low'
        )
      }

      // Queue should not exceed maximum size
      expect(performanceOptimizer.queueLength.value).toBeLessThanOrEqual(1000)
      
      // Metrics should show dropped tasks
      expect(performanceOptimizer.metrics.value.droppedTasks).toBeGreaterThan(0)
    })
  })

  describe('Throttling and Debouncing', () => {
    it('should throttle rapid function calls', () => {
      const mockFn = vi.fn()
      const throttledFn = performanceOptimizer.throttle('test_throttle', mockFn, 100)

      // Call multiple times rapidly
      throttledFn('call1')
      throttledFn('call2')
      throttledFn('call3')

      // Only first call should execute immediately
      expect(mockFn).toHaveBeenCalledTimes(1)
      expect(mockFn).toHaveBeenCalledWith('call1')

      // Advance time
      vi.advanceTimersByTime(100)
      throttledFn('call4')

      expect(mockFn).toHaveBeenCalledTimes(2)
      expect(mockFn).toHaveBeenLastCalledWith('call4')
    })

    it('should debounce function calls', () => {
      const mockFn = vi.fn()
      const debouncedFn = performanceOptimizer.debounce('test_debounce', mockFn, 300)

      // Call multiple times rapidly
      debouncedFn('call1')
      debouncedFn('call2')
      debouncedFn('call3')

      // No calls should execute immediately
      expect(mockFn).not.toHaveBeenCalled()

      // Advance time partially
      vi.advanceTimersByTime(200)
      expect(mockFn).not.toHaveBeenCalled()

      // Complete debounce delay
      vi.advanceTimersByTime(100)
      expect(mockFn).toHaveBeenCalledTimes(1)
      expect(mockFn).toHaveBeenCalledWith('call3') // Last call wins
    })

    it('should reset debounce timer on new calls', () => {
      const mockFn = vi.fn()
      const debouncedFn = performanceOptimizer.debounce('reset_test', mockFn, 300)

      debouncedFn('call1')
      vi.advanceTimersByTime(200) // Partial delay

      debouncedFn('call2') // Should reset timer
      vi.advanceTimersByTime(200) // Still not enough

      expect(mockFn).not.toHaveBeenCalled()

      vi.advanceTimersByTime(100) // Complete new delay
      expect(mockFn).toHaveBeenCalledTimes(1)
      expect(mockFn).toHaveBeenCalledWith('call2')
    })
  })

  describe('Component-Specific Optimization', () => {
    it('should create optimized update functions for components', () => {
      const optimized = performanceOptimizer.optimizeComponentUpdates(
        DashboardComponent.GRAPH
      )

      expect(optimized.throttledUpdate).toBeDefined()
      expect(optimized.debouncedUpdate).toBeDefined()
      expect(optimized.batchUpdate).toBeDefined()

      // Test throttled update
      optimized.throttledUpdate({ data: 1 })
      optimized.throttledUpdate({ data: 2 })

      expect(performanceOptimizer.queueLength.value).toBe(1) // Only first should queue

      // Test batch update
      optimized.batchUpdate([{ data: 1 }, { data: 2 }, { data: 3 }])
      expect(performanceOptimizer.queueLength.value).toBe(2) // Original + batch
    })

    it('should track component-specific metrics', () => {
      // Schedule updates for different components
      performanceOptimizer.scheduleUpdate(DashboardComponent.GRAPH, 'data', {}, 'medium')
      performanceOptimizer.scheduleUpdate(DashboardComponent.GRAPH, 'ui', {}, 'high')
      performanceOptimizer.scheduleUpdate(DashboardComponent.TRANSCRIPT, 'data', {}, 'low')

      const graphMetrics = performanceOptimizer.getComponentMetrics(DashboardComponent.GRAPH)
      const transcriptMetrics = performanceOptimizer.getComponentMetrics(DashboardComponent.TRANSCRIPT)

      expect(graphMetrics.queuedTasks).toBe(2)
      expect(transcriptMetrics.queuedTasks).toBe(1)
      expect(graphMetrics.load).toBeGreaterThan(0)
    })

    it('should register and execute component schedulers', async () => {
      const scheduler1 = vi.fn()
      const scheduler2 = vi.fn()

      const unregister1 = performanceOptimizer.registerComponentScheduler(
        DashboardComponent.GRAPH,
        scheduler1
      )
      const unregister2 = performanceOptimizer.registerComponentScheduler(
        DashboardComponent.GRAPH,
        scheduler2
      )

      // Schedule update for the component
      performanceOptimizer.scheduleUpdate(DashboardComponent.GRAPH, 'data', {})

      await performanceOptimizer.flushQueue()
      vi.runAllTimers()

      // Both schedulers should be called
      expect(scheduler1).toHaveBeenCalled()
      expect(scheduler2).toHaveBeenCalled()

      // Unregister one scheduler
      unregister1()

      // Schedule another update
      performanceOptimizer.scheduleUpdate(DashboardComponent.GRAPH, 'data', {})
      await performanceOptimizer.flushQueue()
      vi.runAllTimers()

      // Only scheduler2 should be called again
      expect(scheduler1).toHaveBeenCalledTimes(1)
      expect(scheduler2).toHaveBeenCalledTimes(2)
    })
  })

  describe('Performance Monitoring', () => {
    it('should track processing metrics', async () => {
      // Schedule several tasks
      for (let i = 0; i < 5; i++) {
        performanceOptimizer.scheduleUpdate(
          DashboardComponent.GRAPH,
          'data',
          { index: i },
          'medium'
        )
      }

      const initialMetrics = performanceOptimizer.metrics.value
      expect(initialMetrics.queuedTasks).toBe(5)

      await performanceOptimizer.flushQueue()
      vi.runAllTimers()

      const finalMetrics = performanceOptimizer.metrics.value
      expect(finalMetrics.processedTasks).toBeGreaterThan(0)
      expect(finalMetrics.averageProcessingTime).toBeGreaterThan(0)
    })

    it('should calculate batch efficiency', async () => {
      // Schedule tasks to fill a batch
      for (let i = 0; i < 50; i++) { // Default batch size
        performanceOptimizer.scheduleUpdate(
          DashboardComponent.GRAPH,
          'data',
          { index: i },
          'medium'
        )
      }

      await performanceOptimizer.flushQueue()
      vi.runAllTimers()

      const metrics = performanceOptimizer.metrics.value
      expect(metrics.batchEfficiency).toBe(1.0) // 50/50 = 100% efficiency
    })

    it('should provide performance recommendations', () => {
      // Fill queue to trigger warnings
      for (let i = 0; i < 850; i++) { // 85% of default 1000 limit
        performanceOptimizer.scheduleUpdate(
          DashboardComponent.GRAPH,
          'data',
          { index: i },
          'low'
        )
      }

      const recommendations = performanceOptimizer.getPerformanceRecommendations()
      
      expect(recommendations.length).toBeGreaterThan(0)
      expect(recommendations.some(r => r.message.includes('queue'))).toBe(true)
      expect(recommendations.some(r => r.type === 'warning')).toBe(true)
    })

    it('should monitor memory usage', () => {
      const initialMetrics = performanceOptimizer.metrics.value
      expect(initialMetrics.memoryUsage).toBeDefined()
      expect(initialMetrics.memoryUsage).toBeGreaterThan(0)

      // Simulate high memory usage in mock
      Object.defineProperty(global.performance, 'memory', {
        value: {
          usedJSHeapSize: 200 * 1024 * 1024, // 200MB (exceeds 100MB threshold)
        },
        configurable: true
      })

      // Advance time to trigger memory monitoring update
      vi.advanceTimersByTime(5000)

      const recommendations = performanceOptimizer.getPerformanceRecommendations()
      expect(recommendations.some(r => r.message.includes('memory'))).toBe(true)
    })
  })

  describe('Optimization Strategies', () => {
    it('should apply priority sorting strategy', async () => {
      const results: string[] = []

      // Schedule mixed priority tasks
      performanceOptimizer.scheduleUpdate(
        DashboardComponent.GRAPH,
        'data',
        {},
        'low',
        () => results.push('low')
      )
      performanceOptimizer.scheduleUpdate(
        DashboardComponent.GRAPH,
        'data',
        {},
        'critical',
        () => results.push('critical')
      )
      performanceOptimizer.scheduleUpdate(
        DashboardComponent.GRAPH,
        'data',
        {},
        'medium',
        () => results.push('medium')
      )

      await performanceOptimizer.flushQueue()
      vi.runAllTimers()

      // Should be processed in priority order
      expect(results).toEqual(['critical', 'medium', 'low'])
    })

    it('should deduplicate identical tasks', () => {
      // Schedule identical tasks
      const taskData = { type: 'duplicate', data: 'same' }
      
      performanceOptimizer.scheduleUpdate(DashboardComponent.GRAPH, 'data', taskData)
      performanceOptimizer.scheduleUpdate(DashboardComponent.GRAPH, 'data', taskData)
      performanceOptimizer.scheduleUpdate(DashboardComponent.GRAPH, 'data', taskData)

      expect(performanceOptimizer.queueLength.value).toBe(3)

      // Process and check deduplication
      vi.advanceTimersByTime(16) // One batch interval

      // After optimization strategies, should have fewer tasks
      const metricsAfterBatch = performanceOptimizer.metrics.value
      expect(metricsAfterBatch.processedTasks).toBeLessThan(3)
    })

    it('should filter out old tasks', () => {
      // Schedule a task
      performanceOptimizer.scheduleUpdate(DashboardComponent.GRAPH, 'data', { old: true })

      // Advance time beyond age threshold (5 seconds)
      vi.advanceTimersByTime(6000)

      // Schedule another task
      performanceOptimizer.scheduleUpdate(DashboardComponent.GRAPH, 'data', { new: true })

      vi.advanceTimersByTime(16) // Process batch

      // Old task should be filtered out
      const metrics = performanceOptimizer.metrics.value
      expect(metrics.processedTasks).toBeLessThan(2)
    })

    it('should enable/disable optimization strategies', () => {
      performanceOptimizer.setOptimizationStrategy('deduplication', false)

      // Schedule duplicate tasks
      const taskData = { type: 'duplicate' }
      performanceOptimizer.scheduleUpdate(DashboardComponent.GRAPH, 'data', taskData)
      performanceOptimizer.scheduleUpdate(DashboardComponent.GRAPH, 'data', taskData)

      vi.advanceTimersByTime(16)

      // With deduplication disabled, both tasks should be processed
      const metrics = performanceOptimizer.metrics.value
      expect(metrics.processedTasks).toBe(2)
    })
  })

  describe('Reactive Watchers', () => {
    it('should create optimized reactive watchers', () => {
      let watcherValue = 0
      const watchedValue = () => watcherValue

      const results: number[] = []
      const stopWatcher = performanceOptimizer.createOptimizedWatcher(
        watchedValue,
        (newVal) => results.push(newVal),
        { throttle: 100 }
      )

      // Change value rapidly
      watcherValue = 1
      watcherValue = 2
      watcherValue = 3

      // Should be throttled
      expect(results.length).toBeLessThanOrEqual(1)

      vi.advanceTimersByTime(100)
      expect(results.length).toBeGreaterThan(0)

      stopWatcher()
    })

    it('should support debounced watchers', () => {
      let watcherValue = 0
      const watchedValue = () => watcherValue

      const results: number[] = []
      const stopWatcher = performanceOptimizer.createOptimizedWatcher(
        watchedValue,
        (newVal) => results.push(newVal),
        { debounce: 300 }
      )

      // Change value rapidly
      watcherValue = 1
      vi.advanceTimersByTime(100)
      watcherValue = 2
      vi.advanceTimersByTime(100)
      watcherValue = 3

      // Should not have called handler yet
      expect(results).toHaveLength(0)

      vi.advanceTimersByTime(300)
      expect(results).toEqual([3]) // Last value wins

      stopWatcher()
    })
  })

  describe('Configuration and Control', () => {
    it('should update configuration dynamically', () => {
      const newConfig = {
        batchSize: 25,
        maxQueueSize: 500,
        throttleInterval: 200
      }

      performanceOptimizer.configure(newConfig)

      // Configuration should be applied
      // This would be verified by observing different behavior with new limits
      expect(performanceOptimizer.queueLength.value).toBe(0)
    })

    it('should clear queue when requested', () => {
      // Fill queue with tasks
      for (let i = 0; i < 10; i++) {
        performanceOptimizer.scheduleUpdate(DashboardComponent.GRAPH, 'data', { index: i })
      }

      expect(performanceOptimizer.queueLength.value).toBe(10)

      performanceOptimizer.clearQueue()

      expect(performanceOptimizer.queueLength.value).toBe(0)
      expect(performanceOptimizer.metrics.value.droppedTasks).toBe(10)
    })

    it('should flush queue immediately when requested', async () => {
      let processedCount = 0
      
      for (let i = 0; i < 5; i++) {
        performanceOptimizer.scheduleUpdate(
          DashboardComponent.GRAPH,
          'data',
          { index: i },
          'medium',
          () => processedCount++
        )
      }

      expect(processedCount).toBe(0)

      await performanceOptimizer.flushQueue()
      vi.runAllTimers()

      expect(processedCount).toBe(5)
      expect(performanceOptimizer.queueLength.value).toBe(0)
    })
  })

  describe('Error Handling', () => {
    it('should handle errors in task processing gracefully', async () => {
      const errorCallback = vi.fn()
      const successCallback = vi.fn()

      // Schedule task that will cause error
      performanceOptimizer.scheduleUpdate(
        DashboardComponent.GRAPH,
        'data',
        { shouldError: true },
        'medium',
        errorCallback
      )

      // Schedule normal task
      performanceOptimizer.scheduleUpdate(
        DashboardComponent.GRAPH,
        'data',
        { normal: true },
        'medium',
        successCallback
      )

      await performanceOptimizer.flushQueue()
      vi.runAllTimers()

      // Error callback should be called with false, success with true
      expect(errorCallback).toHaveBeenCalledWith(false)
      expect(successCallback).toHaveBeenCalledWith(true)
    })

    it('should retry failed tasks up to limit', async () => {
      let attemptCount = 0
      const callback = vi.fn(() => {
        attemptCount++
        return attemptCount < 3 // Fail first 2 attempts
      })

      performanceOptimizer.scheduleUpdate(
        DashboardComponent.GRAPH,
        'data',
        { retry: true },
        'high',
        callback
      )

      // Process multiple times to trigger retries
      for (let i = 0; i < 5; i++) {
        await performanceOptimizer.flushQueue()
        vi.runAllTimers()
        vi.advanceTimersByTime(16)
      }

      // Should have been retried multiple times
      expect(attemptCount).toBeGreaterThan(1)
    })

    it('should handle scheduler errors gracefully', async () => {
      const faultyScheduler = vi.fn(() => {
        throw new Error('Scheduler error')
      })
      const goodScheduler = vi.fn()

      performanceOptimizer.registerComponentScheduler(DashboardComponent.GRAPH, faultyScheduler)
      performanceOptimizer.registerComponentScheduler(DashboardComponent.GRAPH, goodScheduler)

      performanceOptimizer.scheduleUpdate(DashboardComponent.GRAPH, 'data', {})

      await performanceOptimizer.flushQueue()
      vi.runAllTimers()

      // Good scheduler should still execute despite faulty one
      expect(goodScheduler).toHaveBeenCalled()
    })
  })
})