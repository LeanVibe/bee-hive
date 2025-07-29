/**
 * Dashboard Performance Optimization Utilities - VS 6.2
 * LeanVibe Agent Hive 2.0
 * 
 * Performance optimization layer to achieve:
 * - <2s dashboard load times
 * - <1s event processing latency
 * - 1000+ events/second throughput
 * - Smooth 60fps animations
 */

import { ref, reactive, computed, nextTick } from 'vue'
import { debounce, throttle } from 'lodash-es'

// Performance monitoring
export interface PerformanceMetrics {
  loadTime: number
  renderTime: number
  eventLatency: number
  memoryUsage: number
  fps: number
  eventThroughput: number
  lastUpdate: Date
}

export interface OptimizationConfig {
  enableVirtualScrolling: boolean
  batchSize: number
  debounceDelay: number
  throttleDelay: number
  maxBufferSize: number
  enableLazyLoading: boolean
  enableMemoization: boolean
  enableWebWorkers: boolean
  targetFPS: number
  memoryThreshold: number
}

/**
 * Performance monitoring and optimization manager
 */
class DashboardOptimizer {
  private metrics = reactive<PerformanceMetrics>({
    loadTime: 0,
    renderTime: 0,
    eventLatency: 0,
    memoryUsage: 0,
    fps: 0,
    eventThroughput: 0,
    lastUpdate: new Date()
  })

  private config = reactive<OptimizationConfig>({
    enableVirtualScrolling: true,
    batchSize: 50,
    debounceDelay: 100,
    throttleDelay: 16, // 60fps
    maxBufferSize: 1000,
    enableLazyLoading: true,
    enableMemoization: true,
    enableWebWorkers: true,
    targetFPS: 60,
    memoryThreshold: 100 * 1024 * 1024 // 100MB
  })

  // Performance tracking
  private loadStartTime = 0
  private renderStartTimes = new Map<string, number>()
  private eventTimestamps: number[] = []
  private frameTimestamps: number[] = []
  private memoryCheckInterval: number | null = null

  // Memoization cache
  private memoCache = new Map<string, { value: any; timestamp: number; ttl: number }>()
  private readonly MEMO_TTL = 5 * 60 * 1000 // 5 minutes

  // Virtual scrolling state
  private virtualScrollContainers = new Map<string, VirtualScrollState>()

  // Web Worker manager
  private workers = new Map<string, Worker>()

  constructor() {
    this.initializePerformanceMonitoring()
  }

  /**
   * Initialize performance monitoring
   */
  private initializePerformanceMonitoring() {
    // Start performance monitoring
    this.startLoadTimeTracking()
    this.startFPSMonitoring()
    this.startMemoryMonitoring()

    // Performance observer for critical metrics
    if ('PerformanceObserver' in window) {
      this.setupPerformanceObserver()
    }

    console.log('📊 Dashboard optimizer initialized')
  }

  /**
   * Start load time tracking
   */
  startLoadTimeTracking() {
    this.loadStartTime = performance.now()
  }

  /**
   * End load time tracking
   */
  endLoadTimeTracking(componentName?: string) {
    const loadTime = performance.now() - this.loadStartTime
    this.metrics.loadTime = loadTime

    if (loadTime > 2000) {
      console.warn(`⚠️ Slow load time detected: ${loadTime.toFixed(2)}ms for ${componentName || 'dashboard'}`)
      this.suggestOptimizations('load', loadTime)
    }

    console.log(`✅ Load time: ${loadTime.toFixed(2)}ms for ${componentName || 'dashboard'}`)
  }

  /**
   * Track render performance
   */
  startRenderTracking(componentId: string) {
    this.renderStartTimes.set(componentId, performance.now())
  }

  endRenderTracking(componentId: string) {
    const startTime = this.renderStartTimes.get(componentId)
    if (!startTime) return

    const renderTime = performance.now() - startTime
    this.metrics.renderTime = renderTime
    this.renderStartTimes.delete(componentId)

    if (renderTime > 100) {
      console.warn(`⚠️ Slow render detected: ${renderTime.toFixed(2)}ms for ${componentId}`)
      this.suggestOptimizations('render', renderTime)
    }
  }

  /**
   * Track event processing latency
   */
  trackEventLatency(startTime: number) {
    const latency = performance.now() - startTime
    this.metrics.eventLatency = latency

    // Track event throughput
    this.eventTimestamps.push(Date.now())
    
    // Keep only last minute of events
    const oneMinuteAgo = Date.now() - 60000
    this.eventTimestamps = this.eventTimestamps.filter(ts => ts > oneMinuteAgo)
    this.metrics.eventThroughput = this.eventTimestamps.length / 60

    if (latency > 1000) {
      console.warn(`⚠️ High event latency: ${latency.toFixed(2)}ms`)
      this.suggestOptimizations('event', latency)
    }
  }

  /**
   * FPS monitoring
   */
  private startFPSMonitoring() {
    let lastTime = performance.now()
    let frameCount = 0

    const measureFPS = () => {
      const now = performance.now()
      frameCount++

      if (now - lastTime >= 1000) {
        this.metrics.fps = Math.round((frameCount * 1000) / (now - lastTime))
        frameCount = 0
        lastTime = now

        if (this.metrics.fps < 30) {
          console.warn(`⚠️ Low FPS detected: ${this.metrics.fps}`)
          this.suggestOptimizations('fps', this.metrics.fps)
        }
      }

      requestAnimationFrame(measureFPS)
    }

    requestAnimationFrame(measureFPS)
  }

  /**
   * Memory monitoring
   */
  private startMemoryMonitoring() {
    if (!('memory' in performance)) return

    this.memoryCheckInterval = setInterval(() => {
      const memory = (performance as any).memory
      if (memory) {
        this.metrics.memoryUsage = memory.usedJSHeapSize
        
        if (memory.usedJSHeapSize > this.config.memoryThreshold) {
          console.warn(`⚠️ High memory usage: ${(memory.usedJSHeapSize / 1024 / 1024).toFixed(2)}MB`)
          this.suggestOptimizations('memory', memory.usedJSHeapSize)
        }
      }
    }, 10000) // Check every 10 seconds
  }

  /**
   * Setup performance observer
   */
  private setupPerformanceObserver() {
    const observer = new PerformanceObserver((list) => {
      const entries = list.getEntries()
      
      entries.forEach(entry => {
        if (entry.entryType === 'measure') {
          console.log(`📏 Performance measure: ${entry.name} took ${entry.duration.toFixed(2)}ms`)
        } else if (entry.entryType === 'navigation') {
          const nav = entry as PerformanceNavigationTiming
          console.log(`🚢 Navigation timing: DOM loaded in ${nav.domContentLoadedEventEnd - nav.domContentLoadedEventStart}ms`)
        }
      })
    })

    observer.observe({ entryTypes: ['measure', 'navigation'] })
  }

  /**
   * Optimization suggestions
   */
  private suggestOptimizations(type: string, value: number) {
    const suggestions: string[] = []

    switch (type) {
      case 'load':
        if (value > 3000) {
          suggestions.push('Consider lazy loading components')
          suggestions.push('Enable code splitting')
          suggestions.push('Optimize bundle size')
        }
        break
      
      case 'render':
        if (value > 200) {
          suggestions.push('Enable virtual scrolling for large lists')
          suggestions.push('Use memoization for expensive computations')
          suggestions.push('Batch DOM updates')
        }
        break
      
      case 'event':
        if (value > 500) {
          suggestions.push('Increase batch processing size')
          suggestions.push('Enable debouncing for frequent events')
          suggestions.push('Use web workers for heavy processing')
        }
        break
      
      case 'fps':
        if (value < 30) {
          suggestions.push('Reduce animation complexity')
          suggestions.push('Use CSS transforms instead of DOM manipulation')
          suggestions.push('Enable hardware acceleration')
        }
        break
      
      case 'memory':
        if (value > this.config.memoryThreshold) {
          suggestions.push('Clear unused event buffers')
          suggestions.push('Implement object pooling')
          suggestions.push('Reduce memoization cache size')
        }
        break
    }

    if (suggestions.length > 0) {
      console.log(`💡 Optimization suggestions for ${type}:`, suggestions)
    }
  }

  /**
   * Memoization utilities
   */
  memoize<T extends (...args: any[]) => any>(
    fn: T,
    keyGenerator?: (...args: Parameters<T>) => string,
    ttl = this.MEMO_TTL
  ): T {
    if (!this.config.enableMemoization) return fn

    return ((...args: Parameters<T>) => {
      const key = keyGenerator ? keyGenerator(...args) : JSON.stringify(args)
      const cached = this.memoCache.get(key)
      
      if (cached && Date.now() - cached.timestamp < cached.ttl) {
        return cached.value
      }

      const result = fn(...args)
      this.memoCache.set(key, {
        value: result,
        timestamp: Date.now(),
        ttl
      })

      // Cleanup old entries
      if (this.memoCache.size > 1000) {
        this.cleanupMemoCache()
      }

      return result
    }) as T
  }

  /**
   * Cleanup memo cache
   */
  private cleanupMemoCache() {
    const now = Date.now()
    const toDelete: string[] = []

    this.memoCache.forEach((value, key) => {
      if (now - value.timestamp > value.ttl) {
        toDelete.push(key)
      }
    })

    toDelete.forEach(key => this.memoCache.delete(key))
  }

  /**
   * Debounced function creator
   */
  createDebouncedFunction<T extends (...args: any[]) => any>(
    fn: T,
    delay = this.config.debounceDelay
  ): T {
    return debounce(fn, delay) as T
  }

  /**
   * Throttled function creator
   */
  createThrottledFunction<T extends (...args: any[]) => any>(
    fn: T,
    delay = this.config.throttleDelay
  ): T {
    return throttle(fn, delay) as T
  }

  /**
   * Virtual scrolling implementation
   */
  createVirtualScroll<T>(
    containerId: string,
    items: T[],
    itemHeight: number,
    containerHeight: number,
    renderItem: (item: T, index: number) => any
  ) {
    if (!this.config.enableVirtualScrolling) {
      return items.map(renderItem)
    }

    const state = this.getVirtualScrollState(containerId, items.length, itemHeight, containerHeight)
    const visibleItems = items.slice(state.startIndex, state.endIndex)

    return {
      visibleItems: visibleItems.map((item, index) => renderItem(item, state.startIndex + index)),
      spacerTop: state.spacerTop,
      spacerBottom: state.spacerBottom,
      onScroll: (scrollTop: number) => this.updateVirtualScroll(containerId, scrollTop)
    }
  }

  /**
   * Get virtual scroll state
   */
  private getVirtualScrollState(
    containerId: string,
    totalItems: number,
    itemHeight: number,
    containerHeight: number
  ): VirtualScrollState {
    let state = this.virtualScrollContainers.get(containerId)
    
    if (!state) {
      state = {
        scrollTop: 0,
        startIndex: 0,
        endIndex: Math.min(Math.ceil(containerHeight / itemHeight) + 2, totalItems),
        spacerTop: 0,
        spacerBottom: (totalItems - Math.min(Math.ceil(containerHeight / itemHeight) + 2, totalItems)) * itemHeight
      }
      this.virtualScrollContainers.set(containerId, state)
    }

    return state
  }

  /**
   * Update virtual scroll state
   */
  private updateVirtualScroll(containerId: string, scrollTop: number) {
    const state = this.virtualScrollContainers.get(containerId)
    if (!state) return

    state.scrollTop = scrollTop
    // Update indices and spacers based on scroll position
    // Implementation would depend on specific use case
  }

  /**
   * Web Worker utilities
   */
  createWebWorker(
    workerId: string,
    workerScript: string | (() => void)
  ): Worker | null {
    if (!this.config.enableWebWorkers || !window.Worker) {
      return null
    }

    try {
      let worker: Worker

      if (typeof workerScript === 'string') {
        worker = new Worker(workerScript)
      } else {
        // Create worker from function
        const blob = new Blob([`(${workerScript.toString()})()`], {
          type: 'application/javascript'
        })
        worker = new Worker(URL.createObjectURL(blob))
      }

      this.workers.set(workerId, worker)
      
      worker.onerror = (error) => {
        console.error(`Web Worker error in ${workerId}:`, error)
      }

      return worker
    } catch (error) {
      console.error(`Failed to create Web Worker ${workerId}:`, error)
      return null
    }
  }

  /**
   * Batch processing utility
   */
  async processBatch<T, R>(
    items: T[],
    processor: (batch: T[]) => Promise<R[]> | R[],
    batchSize = this.config.batchSize
  ): Promise<R[]> {
    const results: R[] = []
    
    for (let i = 0; i < items.length; i += batchSize) {
      const batch = items.slice(i, i + batchSize)
      const batchResults = await processor(batch)
      results.push(...batchResults)
      
      // Allow other tasks to run
      await nextTick()
    }
    
    return results
  }

  /**
   * Lazy loading utility
   */
  createLazyLoader<T>(
    loader: () => Promise<T>,
    placeholder?: T
  ): { value: T | undefined; load: () => Promise<T>; loading: boolean } {
    let value: T | undefined = placeholder
    let loading = false
    let loaded = false

    const load = async (): Promise<T> => {
      if (loaded) return value!
      if (loading) return value!

      loading = true
      try {
        value = await loader()
        loaded = true
        return value
      } finally {
        loading = false
      }
    }

    return {
      get value() { return value },
      load,
      get loading() { return loading }
    }
  }

  /**
   * Performance measurement utilities
   */
  measurePerformance<T>(
    name: string,
    fn: () => T | Promise<T>
  ): T | Promise<T> {
    const measure = (result: T) => {
      performance.measure(name)
      return result
    }

    performance.mark(`${name}-start`)
    
    try {
      const result = fn()
      
      if (result instanceof Promise) {
        return result.then(asyncResult => {
          performance.mark(`${name}-end`)
          return measure(asyncResult)
        })
      } else {
        performance.mark(`${name}-end`)
        return measure(result)
      }
    } catch (error) {
      performance.mark(`${name}-end`)
      performance.measure(name)
      throw error
    }
  }

  /**
   * Resource cleanup
   */
  cleanup() {
    // Clear intervals
    if (this.memoryCheckInterval) {
      clearInterval(this.memoryCheckInterval)
    }

    // Clear caches
    this.memoCache.clear()
    this.virtualScrollContainers.clear()

    // Terminate workers
    this.workers.forEach(worker => worker.terminate())
    this.workers.clear()

    console.log('🧹 Dashboard optimizer cleaned up')
  }

  /**
   * Get current metrics
   */
  getMetrics(): PerformanceMetrics {
    return { ...this.metrics }
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<OptimizationConfig>) {
    Object.assign(this.config, newConfig)
    console.log('⚙️ Optimizer configuration updated:', newConfig)
  }

  /**
   * Get configuration
   */
  getConfig(): OptimizationConfig {
    return { ...this.config }
  }

  /**
   * Export performance report
   */
  exportPerformanceReport(): string {
    const report = {
      timestamp: new Date().toISOString(),
      metrics: this.getMetrics(),
      config: this.getConfig(),
      browser: {
        userAgent: navigator.userAgent,
        memory: (performance as any).memory,
        connection: (navigator as any).connection
      },
      recommendations: this.generateRecommendations()
    }

    return JSON.stringify(report, null, 2)
  }

  /**
   * Generate performance recommendations
   */
  private generateRecommendations(): string[] {
    const recommendations: string[] = []
    const metrics = this.getMetrics()

    if (metrics.loadTime > 2000) {
      recommendations.push('Optimize component loading with lazy loading and code splitting')
    }

    if (metrics.eventLatency > 1000) {
      recommendations.push('Increase batch size and enable debouncing for better event processing')
    }

    if (metrics.fps < 30) {
      recommendations.push('Reduce visual complexity and enable hardware acceleration')
    }

    if (metrics.memoryUsage > this.config.memoryThreshold) {
      recommendations.push('Implement better memory management and cleanup unused resources')
    }

    if (metrics.eventThroughput < 100) {
      recommendations.push('Consider using web workers for heavy processing tasks')
    }

    return recommendations
  }
}

// Virtual scroll state interface
interface VirtualScrollState {
  scrollTop: number
  startIndex: number
  endIndex: number
  spacerTop: number
  spacerBottom: number
}

// Export singleton instance
export const dashboardOptimizer = new DashboardOptimizer()

// Vue composable for easy integration
export function useDashboardOptimization() {
  return {
    // Core optimizer
    optimizer: dashboardOptimizer,
    
    // Metrics
    metrics: computed(() => dashboardOptimizer.getMetrics()),
    
    // Performance tracking
    startLoadTracking: dashboardOptimizer.startLoadTimeTracking.bind(dashboardOptimizer),
    endLoadTracking: dashboardOptimizer.endLoadTimeTracking.bind(dashboardOptimizer),
    startRenderTracking: dashboardOptimizer.startRenderTracking.bind(dashboardOptimizer),
    endRenderTracking: dashboardOptimizer.endRenderTracking.bind(dashboardOptimizer),
    trackEventLatency: dashboardOptimizer.trackEventLatency.bind(dashboardOptimizer),
    
    // Optimization utilities
    memoize: dashboardOptimizer.memoize.bind(dashboardOptimizer),
    createDebouncedFunction: dashboardOptimizer.createDebouncedFunction.bind(dashboardOptimizer),
    createThrottledFunction: dashboardOptimizer.createThrottledFunction.bind(dashboardOptimizer),
    createVirtualScroll: dashboardOptimizer.createVirtualScroll.bind(dashboardOptimizer),
    createWebWorker: dashboardOptimizer.createWebWorker.bind(dashboardOptimizer),
    processBatch: dashboardOptimizer.processBatch.bind(dashboardOptimizer),
    createLazyLoader: dashboardOptimizer.createLazyLoader.bind(dashboardOptimizer),
    measurePerformance: dashboardOptimizer.measurePerformance.bind(dashboardOptimizer),
    
    // Configuration
    updateConfig: dashboardOptimizer.updateConfig.bind(dashboardOptimizer),
    getConfig: dashboardOptimizer.getConfig.bind(dashboardOptimizer),
    
    // Reporting
    exportPerformanceReport: dashboardOptimizer.exportPerformanceReport.bind(dashboardOptimizer)
  }
}

// Performance monitoring decorators
export function measureRenderTime(target: any, propertyName: string, descriptor: PropertyDescriptor) {
  const method = descriptor.value

  descriptor.value = function (...args: any[]) {
    const componentName = this.constructor.name || 'Component'
    dashboardOptimizer.startRenderTracking(`${componentName}.${propertyName}`)
    
    try {
      const result = method.apply(this, args)
      
      if (result instanceof Promise) {
        return result.finally(() => {
          dashboardOptimizer.endRenderTracking(`${componentName}.${propertyName}`)
        })
      } else {
        dashboardOptimizer.endRenderTracking(`${componentName}.${propertyName}`)
        return result
      }
    } catch (error) {
      dashboardOptimizer.endRenderTracking(`${componentName}.${propertyName}`)
      throw error
    }
  }

  return descriptor
}

export function memoized(ttl = 300000) { // 5 minutes default
  return function (target: any, propertyName: string, descriptor: PropertyDescriptor) {
    const method = descriptor.value
    const memoizedMethod = dashboardOptimizer.memoize(method, undefined, ttl)
    descriptor.value = memoizedMethod
    return descriptor
  }
}

// Web Worker scripts for heavy processing
export const webWorkerScripts = {
  // Data processing worker
  dataProcessor: () => {
    self.onmessage = function(e) {
      const { type, data } = e.data
      
      switch (type) {
        case 'processEvents':
          const processed = data.map((event: any) => ({
            ...event,
            processed: true,
            timestamp: Date.now()
          }))
          self.postMessage({ type: 'eventsProcessed', data: processed })
          break
          
        case 'aggregateData':
          const aggregated = data.reduce((acc: any, item: any) => {
            const key = item.groupKey
            if (!acc[key]) acc[key] = []
            acc[key].push(item)
            return acc
          }, {})
          self.postMessage({ type: 'dataAggregated', data: aggregated })
          break
          
        default:
          self.postMessage({ type: 'error', message: 'Unknown task type' })
      }
    }
  },

  // Semantic processing worker
  semanticProcessor: () => {
    self.onmessage = function(e) {
      const { type, data } = e.data
      
      switch (type) {
        case 'calculateSimilarity':
          const similarities = data.items.map((item: any) => ({
            ...item,
            similarity: calculateCosineSimilarity(item.embedding, data.queryEmbedding)
          }))
          self.postMessage({ type: 'similaritiesCalculated', data: similarities })
          break
          
        default:
          self.postMessage({ type: 'error', message: 'Unknown task type' })
      }
    }

    function calculateCosineSimilarity(a: number[], b: number[]): number {
      const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0)
      const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0))
      const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0))
      return dotProduct / (normA * normB)
    }
  }
}

export default dashboardOptimizer