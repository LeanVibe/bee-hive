/**
 * Performance Optimization Composable
 * 
 * Optimizes concurrent real-time updates across dashboard components
 * with intelligent batching, throttling, and memory management.
 */

import { ref, reactive, computed, watch, nextTick } from 'vue'
import { DashboardComponent } from '@/types/coordination'

export interface PerformanceConfig {
  batchSize: number
  batchInterval: number
  maxQueueSize: number
  throttleInterval: number
  debounceDelay: number
  memoryThreshold: number
  gcInterval: number
  enableOptimizations: boolean
}

export interface UpdateTask {
  id: string
  component: DashboardComponent
  type: 'data' | 'ui' | 'network' | 'animation'
  priority: 'low' | 'medium' | 'high' | 'critical'
  payload: any
  timestamp: number
  retries: number
  callback?: (success: boolean) => void
}

export interface PerformanceMetrics {
  queuedTasks: number
  processedTasks: number
  droppedTasks: number
  averageProcessingTime: number
  memoryUsage: number
  frameRate: number
  batchEfficiency: number
  componentLoad: Record<DashboardComponent, number>
  lastUpdate: number
}

export interface OptimizationStrategy {
  id: string
  name: string
  description: string
  enabled: boolean
  apply: (tasks: UpdateTask[]) => UpdateTask[]
  priority: number
}

class PerformanceOptimizer {
  // Configuration
  private config: PerformanceConfig = {
    batchSize: 50,
    batchInterval: 16, // ~60fps
    maxQueueSize: 1000,
    throttleInterval: 100,
    debounceDelay: 300,
    memoryThreshold: 100 * 1024 * 1024, // 100MB
    gcInterval: 30000, // 30 seconds
    enableOptimizations: true
  }

  // State management
  private state = reactive({
    taskQueue: [] as UpdateTask[],
    processingQueue: false,
    isThrottling: false,
    metrics: {
      queuedTasks: 0,
      processedTasks: 0,
      droppedTasks: 0,
      averageProcessingTime: 0,
      memoryUsage: 0,
      frameRate: 60,
      batchEfficiency: 1.0,
      componentLoad: {} as Record<DashboardComponent, number>,
      lastUpdate: Date.now()
    } as PerformanceMetrics,
    frameTimestamps: [] as number[],
    processingTimes: [] as number[]
  })

  // Optimization strategies
  private strategies = new Map<string, OptimizationStrategy>()

  // Throttle and debounce maps
  private throttleMap = new Map<string, number>()
  private debounceMap = new Map<string, ReturnType<typeof setTimeout>>()

  // Component update schedulers
  private componentSchedulers = new Map<DashboardComponent, (() => void)[]>()

  // Memory management
  private memoryMonitorInterval: ReturnType<typeof setInterval> | null = null
  private garbageCollectionInterval: ReturnType<typeof setInterval> | null = null

  // Public reactive state
  public readonly metrics = computed(() => this.state.metrics)
  public readonly queueLength = computed(() => this.state.taskQueue.length)
  public readonly isProcessing = computed(() => this.state.processingQueue)
  public readonly isThrottling = computed(() => this.state.isThrottling)

  constructor() {
    this.initializeOptimizationStrategies()
    this.startBatchProcessor()
    this.startFrameRateMonitoring()
    this.startMemoryMonitoring()
    this.startGarbageCollection()
  }

  /**
   * Schedule an update task
   */
  public scheduleUpdate(
    component: DashboardComponent,
    type: UpdateTask['type'],
    payload: any,
    priority: UpdateTask['priority'] = 'medium',
    callback?: (success: boolean) => void
  ): string {
    const taskId = this.generateTaskId()
    
    const task: UpdateTask = {
      id: taskId,
      component,
      type,
      priority,
      payload,
      timestamp: Date.now(),
      retries: 0,
      callback
    }

    // Check if we should throttle this update
    if (this.shouldThrottle(component, type)) {
      console.log(`Update throttled for ${component}:${type}`)
      callback?.(false)
      return taskId
    }

    // Add to queue
    if (this.state.taskQueue.length >= this.config.maxQueueSize) {
      // Drop low priority tasks to make room
      this.dropLowPriorityTasks()
    }

    this.state.taskQueue.push(task)
    this.state.metrics.queuedTasks++

    // Update component load tracking
    this.updateComponentLoad(component)

    return taskId
  }

  /**
   * Schedule batched updates for multiple components
   */
  public scheduleBatchUpdate(updates: Array<{
    component: DashboardComponent
    type: UpdateTask['type']
    payload: any
    priority?: UpdateTask['priority']
  }>): string[] {
    return updates.map(update => 
      this.scheduleUpdate(
        update.component,
        update.type,
        update.payload,
        update.priority
      )
    )
  }

  /**
   * Throttle function calls by component and type
   */
  public throttle<T extends (...args: any[]) => any>(
    key: string,
    fn: T,
    interval?: number
  ): T {
    const throttleInterval = interval || this.config.throttleInterval
    
    return ((...args: any[]) => {
      const now = Date.now()
      const lastCall = this.throttleMap.get(key) || 0
      
      if (now - lastCall >= throttleInterval) {
        this.throttleMap.set(key, now)
        return fn(...args)
      }
    }) as T
  }

  /**
   * Debounce function calls
   */
  public debounce<T extends (...args: any[]) => any>(
    key: string,
    fn: T,
    delay?: number
  ): T {
    const debounceDelay = delay || this.config.debounceDelay
    
    return ((...args: any[]) => {
      const existingTimeout = this.debounceMap.get(key)
      if (existingTimeout) {
        clearTimeout(existingTimeout)
      }
      
      const timeout = setTimeout(() => {
        this.debounceMap.delete(key)
        fn(...args)
      }, debounceDelay)
      
      this.debounceMap.set(key, timeout)
    }) as T
  }

  /**
   * Register component update scheduler
   */
  public registerComponentScheduler(
    component: DashboardComponent,
    scheduler: () => void
  ): () => void {
    if (!this.componentSchedulers.has(component)) {
      this.componentSchedulers.set(component, [])
    }

    const schedulers = this.componentSchedulers.get(component)!
    schedulers.push(scheduler)

    // Return unregister function
    return () => {
      const index = schedulers.indexOf(scheduler)
      if (index > -1) {
        schedulers.splice(index, 1)
      }
    }
  }

  /**
   * Optimize updates for specific component
   */
  public optimizeComponentUpdates(component: DashboardComponent): {
    throttledUpdate: (payload: any) => void
    debouncedUpdate: (payload: any) => void
    batchUpdate: (payloads: any[]) => void
  } {
    const throttledUpdate = this.throttle(
      `${component}_update`,
      (payload: any) => {
        this.scheduleUpdate(component, 'data', payload, 'medium')
      }
    )

    const debouncedUpdate = this.debounce(
      `${component}_debounced`,
      (payload: any) => {
        this.scheduleUpdate(component, 'data', payload, 'low')
      }
    )

    const batchUpdate = (payloads: any[]) => {
      const batchId = this.generateTaskId()
      this.scheduleUpdate(component, 'data', {
        batch: true,
        batchId,
        payloads
      }, 'medium')
    }

    return {
      throttledUpdate,
      debouncedUpdate,
      batchUpdate
    }
  }

  /**
   * Create optimized reactive watcher
   */
  public createOptimizedWatcher<T>(
    source: () => T,
    callback: (newValue: T, oldValue: T) => void,
    options?: {
      throttle?: number
      debounce?: number
      immediate?: boolean
    }
  ): () => void {
    const key = `watcher_${Math.random().toString(36).substring(2, 9)}`
    
    let optimizedCallback = callback
    
    if (options?.throttle) {
      optimizedCallback = this.throttle(key, callback, options.throttle)
    } else if (options?.debounce) {
      optimizedCallback = this.debounce(key, callback, options.debounce)
    }

    return watch(source, optimizedCallback, {
      immediate: options?.immediate
    })
  }

  /**
   * Get performance recommendations
   */
  public getPerformanceRecommendations(): Array<{
    type: 'warning' | 'error' | 'info'
    message: string
    action?: string
  }> {
    const recommendations = []
    const metrics = this.state.metrics

    // Queue size warnings
    if (metrics.queuedTasks > this.config.maxQueueSize * 0.8) {
      recommendations.push({
        type: 'warning' as const,
        message: 'Update queue is getting full',
        action: 'Consider reducing update frequency or increasing batch size'
      })
    }

    // Frame rate warnings
    if (metrics.frameRate < 30) {
      recommendations.push({
        type: 'error' as const,
        message: 'Low frame rate detected',
        action: 'Enable performance optimizations or reduce visual complexity'
      })
    }

    // Memory usage warnings
    if (metrics.memoryUsage > this.config.memoryThreshold) {
      recommendations.push({
        type: 'warning' as const,
        message: 'High memory usage detected',
        action: 'Consider clearing old data or reducing cache sizes'
      })
    }

    // Component load warnings
    const highLoadComponents = Object.entries(metrics.componentLoad)
      .filter(([_, load]) => load > 100)
      .map(([component]) => component)

    if (highLoadComponents.length > 0) {
      recommendations.push({
        type: 'warning' as const,
        message: `High load on components: ${highLoadComponents.join(', ')}`,
        action: 'Consider optimizing these components or reducing update frequency'
      })
    }

    return recommendations
  }

  /**
   * Configure performance settings
   */
  public configure(config: Partial<PerformanceConfig>): void {
    Object.assign(this.config, config)
    console.log('Performance configuration updated:', this.config)
  }

  /**
   * Enable/disable specific optimization strategy
   */
  public setOptimizationStrategy(strategyId: string, enabled: boolean): void {
    const strategy = this.strategies.get(strategyId)
    if (strategy) {
      strategy.enabled = enabled
      console.log(`Optimization strategy ${strategyId} ${enabled ? 'enabled' : 'disabled'}`)
    }
  }

  /**
   * Force process queue immediately
   */
  public async flushQueue(): Promise<void> {
    if (this.state.processingQueue) {
      return
    }

    await this.processBatch()
  }

  /**
   * Clear all queued tasks
   */
  public clearQueue(): void {
    const droppedCount = this.state.taskQueue.length
    this.state.taskQueue = []
    this.state.metrics.droppedTasks += droppedCount
    console.log(`Cleared ${droppedCount} queued tasks`)
  }

  /**
   * Get component-specific metrics
   */
  public getComponentMetrics(component: DashboardComponent): {
    queuedTasks: number
    load: number
    averageProcessingTime: number
  } {
    const componentTasks = this.state.taskQueue.filter(task => task.component === component)
    const load = this.state.metrics.componentLoad[component] || 0
    
    return {
      queuedTasks: componentTasks.length,
      load,
      averageProcessingTime: this.state.metrics.averageProcessingTime
    }
  }

  // Private methods

  private initializeOptimizationStrategies(): void {
    // Priority-based sorting strategy
    this.strategies.set('priority_sort', {
      id: 'priority_sort',
      name: 'Priority Sorting',
      description: 'Sort tasks by priority before processing',
      enabled: true,
      apply: (tasks) => {
        const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 }
        return tasks.sort((a, b) => priorityOrder[b.priority] - priorityOrder[a.priority])
      },
      priority: 1
    })

    // Component batching strategy
    this.strategies.set('component_batching', {
      id: 'component_batching',
      name: 'Component Batching',
      description: 'Group tasks by component for efficient processing',
      enabled: true,
      apply: (tasks) => {
        const componentGroups = new Map<DashboardComponent, UpdateTask[]>()
        
        tasks.forEach(task => {
          if (!componentGroups.has(task.component)) {
            componentGroups.set(task.component, [])
          }
          componentGroups.get(task.component)!.push(task)
        })

        // Flatten groups back to array, maintaining component grouping
        return Array.from(componentGroups.values()).flat()
      },
      priority: 2
    })

    // Deduplication strategy
    this.strategies.set('deduplication', {
      id: 'deduplication',
      name: 'Task Deduplication',
      description: 'Remove duplicate tasks for the same component',
      enabled: true,
      apply: (tasks) => {
        const seen = new Set<string>()
        return tasks.filter(task => {
          const key = `${task.component}:${task.type}:${JSON.stringify(task.payload)}`
          if (seen.has(key)) {
            return false
          }
          seen.add(key)
          return true
        })
      },
      priority: 3
    })

    // Age-based filtering strategy
    this.strategies.set('age_filtering', {
      id: 'age_filtering',
      name: 'Age-based Filtering',
      description: 'Remove tasks that are too old to be relevant',
      enabled: true,
      apply: (tasks) => {
        const maxAge = 5000 // 5 seconds
        const now = Date.now()
        return tasks.filter(task => now - task.timestamp < maxAge)
      },
      priority: 4
    })
  }

  private startBatchProcessor(): void {
    const processLoop = async () => {
      if (this.state.taskQueue.length > 0) {
        await this.processBatch()
      }
      
      setTimeout(processLoop, this.config.batchInterval)
    }

    processLoop()
  }

  private async processBatch(): Promise<void> {
    if (this.state.processingQueue || this.state.taskQueue.length === 0) {
      return
    }

    this.state.processingQueue = true
    const startTime = performance.now()

    try {
      // Extract batch of tasks
      const batchSize = Math.min(this.config.batchSize, this.state.taskQueue.length)
      const batch = this.state.taskQueue.splice(0, batchSize)

      // Apply optimization strategies
      let optimizedBatch = batch
      if (this.config.enableOptimizations) {
        optimizedBatch = this.applyOptimizationStrategies(batch)
      }

      // Process tasks
      await this.processTasks(optimizedBatch)

      // Update metrics
      const processingTime = performance.now() - startTime
      this.updateProcessingMetrics(optimizedBatch.length, processingTime)

    } catch (error) {
      console.error('Error processing batch:', error)
    } finally {
      this.state.processingQueue = false
    }
  }

  private applyOptimizationStrategies(tasks: UpdateTask[]): UpdateTask[] {
    let optimizedTasks = tasks

    // Apply strategies in priority order
    const enabledStrategies = Array.from(this.strategies.values())
      .filter(strategy => strategy.enabled)
      .sort((a, b) => a.priority - b.priority)

    for (const strategy of enabledStrategies) {
      try {
        optimizedTasks = strategy.apply(optimizedTasks)
      } catch (error) {
        console.error(`Error applying optimization strategy ${strategy.id}:`, error)
      }
    }

    return optimizedTasks
  }

  private async processTasks(tasks: UpdateTask[]): Promise<void> {
    const processPromises = tasks.map(async (task) => {
      try {
        await this.processTask(task)
        task.callback?.(true)
        this.state.metrics.processedTasks++
      } catch (error) {
        console.error(`Error processing task ${task.id}:`, error)
        task.callback?.(false)
        
        // Retry logic for failed tasks
        if (task.retries < 3 && task.priority !== 'low') {
          task.retries++
          this.state.taskQueue.push(task)
        }
      }
    })

    await Promise.all(processPromises)
  }

  private async processTask(task: UpdateTask): Promise<void> {
    // Schedule task execution to next tick to avoid blocking
    await nextTick()

    // Execute component schedulers
    const schedulers = this.componentSchedulers.get(task.component) || []
    schedulers.forEach(scheduler => {
      try {
        scheduler()
      } catch (error) {
        console.error(`Error in component scheduler for ${task.component}:`, error)
      }
    })

    // Simulate task processing
    // In real implementation, this would trigger actual component updates
    if (task.type === 'animation') {
      // Fast processing for animations
      return Promise.resolve()
    } else {
      // Simulate some processing time
      return new Promise(resolve => setTimeout(resolve, 1))
    }
  }

  private shouldThrottle(component: DashboardComponent, type: UpdateTask['type']): boolean {
    if (!this.config.enableOptimizations) {
      return false
    }

    const key = `${component}:${type}`
    const now = Date.now()
    const lastUpdate = this.throttleMap.get(key) || 0

    return now - lastUpdate < this.config.throttleInterval
  }

  private dropLowPriorityTasks(): void {
    const lowPriorityTasks = this.state.taskQueue.filter(task => task.priority === 'low')
    const dropCount = Math.min(lowPriorityTasks.length, 10)
    
    for (let i = 0; i < dropCount; i++) {
      const index = this.state.taskQueue.findIndex(task => task.priority === 'low')
      if (index > -1) {
        const droppedTask = this.state.taskQueue.splice(index, 1)[0]
        droppedTask.callback?.(false)
        this.state.metrics.droppedTasks++
      }
    }

    if (dropCount > 0) {
      console.log(`Dropped ${dropCount} low priority tasks due to queue overflow`)
    }
  }

  private updateComponentLoad(component: DashboardComponent): void {
    if (!this.state.metrics.componentLoad[component]) {
      this.state.metrics.componentLoad[component] = 0
    }
    this.state.metrics.componentLoad[component]++
  }

  private updateProcessingMetrics(batchSize: number, processingTime: number): void {
    this.state.processingTimes.push(processingTime)
    
    // Keep only last 100 measurements
    if (this.state.processingTimes.length > 100) {
      this.state.processingTimes.shift()
    }

    // Update average
    this.state.metrics.averageProcessingTime = 
      this.state.processingTimes.reduce((sum, time) => sum + time, 0) / 
      this.state.processingTimes.length

    // Update batch efficiency
    this.state.metrics.batchEfficiency = batchSize / this.config.batchSize

    this.state.metrics.lastUpdate = Date.now()
  }

  private startFrameRateMonitoring(): void {
    const measureFrameRate = () => {
      const now = performance.now()
      this.state.frameTimestamps.push(now)

      // Keep only last 60 frames
      if (this.state.frameTimestamps.length > 60) {
        this.state.frameTimestamps.shift()
      }

      // Calculate frame rate
      if (this.state.frameTimestamps.length >= 2) {
        const timeSpan = now - this.state.frameTimestamps[0]
        const frameCount = this.state.frameTimestamps.length - 1
        this.state.metrics.frameRate = (frameCount / timeSpan) * 1000
      }

      requestAnimationFrame(measureFrameRate)
    }

    requestAnimationFrame(measureFrameRate)
  }

  private startMemoryMonitoring(): void {
    this.memoryMonitorInterval = setInterval(() => {
      if ('memory' in performance) {
        const memory = (performance as any).memory
        this.state.metrics.memoryUsage = memory.usedJSHeapSize
      }
    }, 5000) // Check every 5 seconds
  }

  private startGarbageCollection(): void {
    this.garbageCollectionInterval = setInterval(() => {
      // Clear old throttle entries
      const now = Date.now()
      for (const [key, timestamp] of this.throttleMap.entries()) {
        if (now - timestamp > this.config.throttleInterval * 10) {
          this.throttleMap.delete(key)
        }
      }

      // Reset component load counters
      for (const component of Object.keys(this.state.metrics.componentLoad)) {
        this.state.metrics.componentLoad[component as DashboardComponent] = 0
      }

    }, this.config.gcInterval)
  }

  private generateTaskId(): string {
    return `task_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`
  }

  /**
   * Destroy and cleanup all resources
   */
  public destroy(): void {
    this.clearQueue()
    
    if (this.memoryMonitorInterval) {
      clearInterval(this.memoryMonitorInterval)
    }
    
    if (this.garbageCollectionInterval) {
      clearInterval(this.garbageCollectionInterval)
    }

    // Clear all debounce timeouts
    for (const timeout of this.debounceMap.values()) {
      clearTimeout(timeout)
    }

    this.debounceMap.clear()
    this.throttleMap.clear()
    this.componentSchedulers.clear()
    this.strategies.clear()
  }
}

// Create singleton instance
const performanceOptimizer = new PerformanceOptimizer()

// Vue composable
export function usePerformanceOptimization() {
  return {
    // State
    metrics: performanceOptimizer.metrics,
    queueLength: performanceOptimizer.queueLength,
    isProcessing: performanceOptimizer.isProcessing,
    isThrottling: performanceOptimizer.isThrottling,

    // Task scheduling
    scheduleUpdate: performanceOptimizer.scheduleUpdate.bind(performanceOptimizer),
    scheduleBatchUpdate: performanceOptimizer.scheduleBatchUpdate.bind(performanceOptimizer),

    // Optimization utilities
    throttle: performanceOptimizer.throttle.bind(performanceOptimizer),
    debounce: performanceOptimizer.debounce.bind(performanceOptimizer),
    optimizeComponentUpdates: performanceOptimizer.optimizeComponentUpdates.bind(performanceOptimizer),
    createOptimizedWatcher: performanceOptimizer.createOptimizedWatcher.bind(performanceOptimizer),

    // Component management
    registerComponentScheduler: performanceOptimizer.registerComponentScheduler.bind(performanceOptimizer),
    getComponentMetrics: performanceOptimizer.getComponentMetrics.bind(performanceOptimizer),

    // Configuration and control
    configure: performanceOptimizer.configure.bind(performanceOptimizer),
    setOptimizationStrategy: performanceOptimizer.setOptimizationStrategy.bind(performanceOptimizer),
    flushQueue: performanceOptimizer.flushQueue.bind(performanceOptimizer),
    clearQueue: performanceOptimizer.clearQueue.bind(performanceOptimizer),

    // Insights
    getPerformanceRecommendations: performanceOptimizer.getPerformanceRecommendations.bind(performanceOptimizer)
  }
}

export default performanceOptimizer