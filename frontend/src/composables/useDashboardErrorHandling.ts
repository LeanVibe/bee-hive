/**
 * Dashboard Error Handling Composable
 * 
 * Comprehensive error handling, fallback mechanisms, and recovery strategies
 * for the multi-agent coordination dashboard with graceful degradation.
 */

import { ref, reactive, computed, watch } from 'vue'
import { 
  DashboardComponent,
  type DashboardError,
  type ErrorBoundaryState,
  type ConnectionStatus 
} from '@/types/coordination'

export interface ErrorRecoveryStrategy {
  id: string
  name: string
  description: string
  component?: DashboardComponent
  canRecover: (error: DashboardError) => boolean
  recover: (error: DashboardError) => Promise<boolean>
  priority: number
  maxRetries: number
}

export interface FallbackConfig {
  component: DashboardComponent
  fallbackComponent?: string
  fallbackData?: any
  showFallbackMessage: boolean
  retryInterval: number
  maxRetries: number
}

export interface ErrorMetrics {
  totalErrors: number
  errorsByType: Record<string, number>
  errorsByComponent: Record<string, number>
  recoverySuccessRate: number
  averageRecoveryTime: number
  recentErrors: DashboardError[]
}

class DashboardErrorHandler {
  // Error state management
  private state = reactive({
    globalError: null as DashboardError | null,
    componentErrors: new Map<DashboardComponent, DashboardError[]>(),
    errorBoundaries: new Map<string, ErrorBoundaryState>(),
    isRecovering: false,
    lastRecoveryAttempt: null as Date | null,
    metrics: {
      totalErrors: 0,
      errorsByType: {},
      errorsByComponent: {},
      recoverySuccessRate: 0,
      averageRecoveryTime: 0,
      recentErrors: []
    } as ErrorMetrics
  })

  // Recovery strategies and fallback configurations
  private recoveryStrategies = new Map<string, ErrorRecoveryStrategy>()
  private fallbackConfigs = new Map<DashboardComponent, FallbackConfig>()
  private errorQueue: DashboardError[] = []
  private processingQueue = false

  // Event listeners
  private errorListeners = new Map<string, Array<(error: DashboardError) => void>>()
  private recoveryListeners = new Map<string, Array<(success: boolean, error: DashboardError) => void>>()

  // Configuration
  private config = {
    maxErrorQueueSize: 100,
    errorProcessingInterval: 1000,
    maxRecoveryAttempts: 3,
    recoveryTimeout: 30000,
    fallbackTimeout: 5000,
    errorRetentionTime: 24 * 60 * 60 * 1000, // 24 hours
    metricsUpdateInterval: 60000 // 1 minute
  }

  // Public reactive state
  public readonly globalError = computed(() => this.state.globalError)
  public readonly componentErrors = computed(() => this.state.componentErrors)
  public readonly isRecovering = computed(() => this.state.isRecovering)
  public readonly metrics = computed(() => this.state.metrics)
  public readonly hasErrors = computed(() => {
    return this.state.globalError !== null || 
           Array.from(this.state.componentErrors.values()).some(errors => errors.length > 0)
  })

  constructor() {
    this.initializeDefaultRecoveryStrategies()
    this.initializeDefaultFallbackConfigs()
    this.startErrorProcessing()
    this.startMetricsCollection()
  }

  /**
   * Report an error to the error handling system
   */
  public reportError(error: DashboardError): void {
    // Add timestamp if not present
    if (!error.timestamp) {
      error.timestamp = new Date().toISOString()
    }

    console.error(`Dashboard Error [${error.component}]:`, error)

    // Add to error queue for processing
    this.errorQueue.push(error)

    // Update metrics
    this.updateErrorMetrics(error)

    // Trigger immediate processing for critical errors
    if (error.type === 'network' || error.type === 'websocket') {
      this.processErrorQueue()
    }

    // Emit error event
    this.emitErrorEvent('error_reported', error)
  }

  /**
   * Report a network error
   */
  public reportNetworkError(
    component: DashboardComponent,
    message: string,
    details?: any
  ): void {
    const error: DashboardError = {
      id: this.generateErrorId(),
      type: 'network',
      message,
      component,
      timestamp: new Date().toISOString(),
      details,
      recoverable: true
    }

    this.reportError(error)
  }

  /**
   * Report a WebSocket connection error
   */
  public reportWebSocketError(
    endpointId: string,
    message: string,
    details?: any
  ): void {
    const error: DashboardError = {
      id: this.generateErrorId(),
      type: 'websocket',
      message,
      component: DashboardComponent.SERVICE,
      timestamp: new Date().toISOString(),
      details: { endpointId, ...details },
      recoverable: true
    }

    this.reportError(error)
  }

  /**
   * Report a data parsing/processing error
   */
  public reportDataError(
    component: DashboardComponent,
    message: string,
    data?: any
  ): void {
    const error: DashboardError = {
      id: this.generateErrorId(),
      type: 'data',
      message,
      component,
      timestamp: new Date().toISOString(),
      details: { data },
      recoverable: false
    }

    this.reportError(error)
  }

  /**
   * Report a parsing error
   */
  public reportParsingError(
    component: DashboardComponent,
    message: string,
    input?: any
  ): void {
    const error: DashboardError = {
      id: this.generateErrorId(),
      type: 'parsing',
      message,
      component,
      timestamp: new Date().toISOString(),
      details: { input },
      recoverable: false
    }

    this.reportError(error)
  }

  /**
   * Create an error boundary for a component
   */
  public createErrorBoundary(boundaryId: string): ErrorBoundaryState {
    const boundary: ErrorBoundaryState = {
      hasError: false,
      error: undefined,
      fallbackComponent: undefined,
      retryCount: 0,
      lastRetry: undefined
    }

    this.state.errorBoundaries.set(boundaryId, boundary)
    return boundary
  }

  /**
   * Trigger error boundary
   */
  public triggerErrorBoundary(
    boundaryId: string,
    error: DashboardError,
    fallbackComponent?: string
  ): void {
    const boundary = this.state.errorBoundaries.get(boundaryId)
    if (!boundary) {
      console.error(`Error boundary not found: ${boundaryId}`)
      return
    }

    boundary.hasError = true
    boundary.error = error
    boundary.fallbackComponent = fallbackComponent
    boundary.lastRetry = new Date().toISOString()

    console.log(`Error boundary triggered: ${boundaryId}`, error)
  }

  /**
   * Retry error boundary
   */
  public async retryErrorBoundary(boundaryId: string): Promise<boolean> {
    const boundary = this.state.errorBoundaries.get(boundaryId)
    if (!boundary || !boundary.error) {
      return false
    }

    boundary.retryCount++
    boundary.lastRetry = new Date().toISOString()

    try {
      // Attempt recovery using registered strategies
      const recovered = await this.attemptErrorRecovery(boundary.error)
      
      if (recovered) {
        boundary.hasError = false
        boundary.error = undefined
        boundary.fallbackComponent = undefined
        return true
      }

      return false
    } catch (error) {
      console.error(`Error boundary retry failed for ${boundaryId}:`, error)
      return false
    }
  }

  /**
   * Clear error boundary
   */
  public clearErrorBoundary(boundaryId: string): void {
    const boundary = this.state.errorBoundaries.get(boundaryId)
    if (boundary) {
      boundary.hasError = false
      boundary.error = undefined
      boundary.fallbackComponent = undefined
      boundary.retryCount = 0
    }
  }

  /**
   * Register error recovery strategy
   */
  public registerRecoveryStrategy(strategy: ErrorRecoveryStrategy): void {
    this.recoveryStrategies.set(strategy.id, strategy)
    console.log(`Recovery strategy registered: ${strategy.id}`)
  }

  /**
   * Register fallback configuration
   */
  public registerFallbackConfig(component: DashboardComponent, config: FallbackConfig): void {
    this.fallbackConfigs.set(component, config)
    console.log(`Fallback config registered for component: ${component}`)
  }

  /**
   * Attempt to recover from an error
   */
  public async attemptErrorRecovery(error: DashboardError): Promise<boolean> {
    if (!error.recoverable) {
      console.log(`Error ${error.id} is not recoverable`)
      return false
    }

    this.state.isRecovering = true
    this.state.lastRecoveryAttempt = new Date()

    try {
      // Get applicable recovery strategies
      const strategies = Array.from(this.recoveryStrategies.values())
        .filter(strategy => strategy.canRecover(error))
        .sort((a, b) => b.priority - a.priority)

      for (const strategy of strategies) {
        console.log(`Attempting recovery with strategy: ${strategy.id}`)
        
        try {
          const recovered = await Promise.race([
            strategy.recover(error),
            new Promise<boolean>((_, reject) => 
              setTimeout(() => reject(new Error('Recovery timeout')), this.config.recoveryTimeout)
            )
          ])

          if (recovered) {
            console.log(`Recovery successful with strategy: ${strategy.id}`)
            this.emitRecoveryEvent('recovery_success', true, error)
            this.updateRecoveryMetrics(true)
            return true
          }
        } catch (strategyError) {
          console.error(`Recovery strategy ${strategy.id} failed:`, strategyError)
        }
      }

      console.log(`All recovery strategies failed for error: ${error.id}`)
      this.emitRecoveryEvent('recovery_failed', false, error)
      this.updateRecoveryMetrics(false)
      return false

    } finally {
      this.state.isRecovering = false
    }
  }

  /**
   * Get fallback configuration for component
   */
  public getFallbackConfig(component: DashboardComponent): FallbackConfig | null {
    return this.fallbackConfigs.get(component) || null
  }

  /**
   * Apply fallback for component
   */
  public applyFallback(component: DashboardComponent, error: DashboardError): any {
    const config = this.getFallbackConfig(component)
    if (!config) {
      console.warn(`No fallback configuration for component: ${component}`)
      return null
    }

    console.log(`Applying fallback for component: ${component}`)

    // Return fallback data if available
    if (config.fallbackData) {
      return config.fallbackData
    }

    // Generate default fallback data based on component type
    return this.generateDefaultFallbackData(component, error)
  }

  /**
   * Clear all errors for a component
   */
  public clearComponentErrors(component: DashboardComponent): void {
    this.state.componentErrors.delete(component)
    console.log(`Cleared errors for component: ${component}`)
  }

  /**
   * Clear global error
   */
  public clearGlobalError(): void {
    this.state.globalError = null
    console.log('Global error cleared')
  }

  /**
   * Get component error status
   */
  public getComponentErrorStatus(component: DashboardComponent): {
    hasErrors: boolean
    errorCount: number
    lastError?: DashboardError
  } {
    const errors = this.state.componentErrors.get(component) || []
    return {
      hasErrors: errors.length > 0,
      errorCount: errors.length,
      lastError: errors[errors.length - 1]
    }
  }

  /**
   * Register error event listener
   */
  public onError(
    eventType: string,
    handler: (error: DashboardError) => void
  ): () => void {
    if (!this.errorListeners.has(eventType)) {
      this.errorListeners.set(eventType, [])
    }

    const listeners = this.errorListeners.get(eventType)!
    listeners.push(handler)

    return () => {
      const index = listeners.indexOf(handler)
      if (index > -1) {
        listeners.splice(index, 1)
      }
    }
  }

  /**
   * Register recovery event listener
   */
  public onRecovery(
    eventType: string,
    handler: (success: boolean, error: DashboardError) => void
  ): () => void {
    if (!this.recoveryListeners.has(eventType)) {
      this.recoveryListeners.set(eventType, [])
    }

    const listeners = this.recoveryListeners.get(eventType)!
    listeners.push(handler)

    return () => {
      const index = listeners.indexOf(handler)
      if (index > -1) {
        listeners.splice(index, 1)
      }
    }
  }

  // Private methods

  private initializeDefaultRecoveryStrategies(): void {
    // Network retry strategy
    this.registerRecoveryStrategy({
      id: 'network_retry',
      name: 'Network Retry',
      description: 'Retry failed network requests',
      canRecover: (error) => error.type === 'network',
      recover: async (error) => {
        await new Promise(resolve => setTimeout(resolve, 2000))
        // This would trigger an actual retry in the real implementation
        return Math.random() > 0.3 // 70% success rate for demo
      },
      priority: 1,
      maxRetries: 3
    })

    // WebSocket reconnection strategy
    this.registerRecoveryStrategy({
      id: 'websocket_reconnect',
      name: 'WebSocket Reconnect',
      description: 'Reconnect WebSocket connections',
      canRecover: (error) => error.type === 'websocket',
      recover: async (error) => {
        const endpointId = error.details?.endpointId
        if (!endpointId) return false

        // This would trigger actual WebSocket reconnection
        console.log(`Attempting WebSocket reconnection for: ${endpointId}`)
        await new Promise(resolve => setTimeout(resolve, 1000))
        return Math.random() > 0.2 // 80% success rate for demo
      },
      priority: 2,
      maxRetries: 5
    })

    // Data refresh strategy
    this.registerRecoveryStrategy({
      id: 'data_refresh',
      name: 'Data Refresh',
      description: 'Refresh component data',
      canRecover: (error) => error.type === 'data',
      recover: async (error) => {
        console.log(`Refreshing data for component: ${error.component}`)
        // This would trigger actual data refresh
        await new Promise(resolve => setTimeout(resolve, 1500))
        return Math.random() > 0.4 // 60% success rate for demo
      },
      priority: 1,
      maxRetries: 2
    })
  }

  private initializeDefaultFallbackConfigs(): void {
    // Graph component fallback
    this.registerFallbackConfig(DashboardComponent.GRAPH, {
      component: DashboardComponent.GRAPH,
      fallbackComponent: 'GraphFallback',
      fallbackData: {
        nodes: [],
        edges: [],
        stats: { total_nodes: 0, total_edges: 0 },
        message: 'Graph data temporarily unavailable'
      },
      showFallbackMessage: true,
      retryInterval: 10000,
      maxRetries: 3
    })

    // Transcript component fallback
    this.registerFallbackConfig(DashboardComponent.TRANSCRIPT, {
      component: DashboardComponent.TRANSCRIPT,
      fallbackComponent: 'TranscriptFallback',
      fallbackData: {
        events: [],
        totalEvents: 0,
        message: 'Communication transcript temporarily unavailable'
      },
      showFallbackMessage: true,
      retryInterval: 5000,
      maxRetries: 5
    })

    // Analysis component fallback
    this.registerFallbackConfig(DashboardComponent.ANALYSIS, {
      component: DashboardComponent.ANALYSIS,
      fallbackComponent: 'AnalysisFallback',
      fallbackData: {
        patterns: [],
        metrics: {},
        recommendations: [],
        message: 'Analysis data temporarily unavailable'
      },
      showFallbackMessage: true,
      retryInterval: 15000,
      maxRetries: 2
    })

    // Monitoring component fallback
    this.registerFallbackConfig(DashboardComponent.MONITORING, {
      component: DashboardComponent.MONITORING,
      fallbackComponent: 'MonitoringFallback',
      fallbackData: {
        systemHealth: 'unknown',
        metrics: {},
        message: 'Monitoring data temporarily unavailable'
      },
      showFallbackMessage: true,
      retryInterval: 8000,
      maxRetries: 4
    })
  }

  private startErrorProcessing(): void {
    setInterval(() => {
      if (!this.processingQueue && this.errorQueue.length > 0) {
        this.processErrorQueue()
      }
    }, this.config.errorProcessingInterval)
  }

  private async processErrorQueue(): Promise<void> {
    if (this.processingQueue) return

    this.processingQueue = true

    try {
      while (this.errorQueue.length > 0) {
        const error = this.errorQueue.shift()!
        await this.processError(error)
      }
    } finally {
      this.processingQueue = false
    }
  }

  private async processError(error: DashboardError): Promise<void> {
    // Add to component errors
    if (!this.state.componentErrors.has(error.component)) {
      this.state.componentErrors.set(error.component, [])
    }

    const componentErrors = this.state.componentErrors.get(error.component)!
    componentErrors.push(error)

    // Keep only recent errors
    if (componentErrors.length > 10) {
      componentErrors.shift()
    }

    // Set as global error if critical
    if (error.type === 'network' || error.type === 'websocket') {
      this.state.globalError = error
    }

    // Attempt automatic recovery if error is recoverable
    if (error.recoverable) {
      setTimeout(async () => {
        await this.attemptErrorRecovery(error)
      }, 1000)
    }
  }

  private updateErrorMetrics(error: DashboardError): void {
    const metrics = this.state.metrics

    metrics.totalErrors++
    metrics.errorsByType[error.type] = (metrics.errorsByType[error.type] || 0) + 1
    metrics.errorsByComponent[error.component] = (metrics.errorsByComponent[error.component] || 0) + 1

    metrics.recentErrors.push(error)
    if (metrics.recentErrors.length > 20) {
      metrics.recentErrors.shift()
    }
  }

  private updateRecoveryMetrics(success: boolean): void {
    // This would update recovery success rate and average recovery time
    // Implementation would track recovery attempts and successes
  }

  private generateDefaultFallbackData(component: DashboardComponent, error: DashboardError): any {
    const fallbackDataMap: Record<DashboardComponent, any> = {}
    
    fallbackDataMap[DashboardComponent.GRAPH] = {
      nodes: [],
      edges: [],
      stats: { total_nodes: 0, total_edges: 0 },
      error: true,
      errorMessage: 'Graph visualization temporarily unavailable'
    }
    
    fallbackDataMap[DashboardComponent.TRANSCRIPT] = {
      events: [],
      totalEvents: 0,
      error: true,
      errorMessage: 'Communication transcript temporarily unavailable'
    }
    
    fallbackDataMap[DashboardComponent.ANALYSIS] = {
      patterns: [],
      metrics: {},
      recommendations: [],
      error: true,
      errorMessage: 'Analysis temporarily unavailable'
    }
    
    fallbackDataMap[DashboardComponent.MONITORING] = {
      systemHealth: 'unknown',
      metrics: {},
      error: true,
      errorMessage: 'System monitoring temporarily unavailable'
    }
    
    fallbackDataMap[DashboardComponent.SERVICE] = {
      status: 'error',
      message: 'Service temporarily unavailable'
    }

    return fallbackDataMap[component] || { error: true, errorMessage: 'Component temporarily unavailable' }
  }

  private generateErrorId(): string {
    return `error_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`
  }

  private emitErrorEvent(eventType: string, error: DashboardError): void {
    const listeners = this.errorListeners.get(eventType) || []
    const allListeners = this.errorListeners.get('*') || []
    const combinedListeners = [...listeners, ...allListeners]

    combinedListeners.forEach(listener => {
      try {
        listener(error)
      } catch (listenerError) {
        console.error('Error in error event listener:', listenerError)
      }
    })
  }

  private emitRecoveryEvent(eventType: string, success: boolean, error: DashboardError): void {
    const listeners = this.recoveryListeners.get(eventType) || []
    const allListeners = this.recoveryListeners.get('*') || []
    const combinedListeners = [...listeners, ...allListeners]

    combinedListeners.forEach(listener => {
      try {
        listener(success, error)
      } catch (listenerError) {
        console.error('Error in recovery event listener:', listenerError)
      }
    })
  }

  private startMetricsCollection(): void {
    setInterval(() => {
      // Clean up old errors
      const cutoffTime = Date.now() - this.config.errorRetentionTime
      
      for (const [component, errors] of this.state.componentErrors) {
        const filteredErrors = errors.filter(error => 
          new Date(error.timestamp).getTime() > cutoffTime
        )
        
        if (filteredErrors.length !== errors.length) {
          this.state.componentErrors.set(component, filteredErrors)
        }
      }

      // Update metrics
      this.state.metrics.recentErrors = this.state.metrics.recentErrors.filter(error =>
        new Date(error.timestamp).getTime() > cutoffTime
      )
    }, this.config.metricsUpdateInterval)
  }

  /**
   * Reset error handling state
   */
  public reset(): void {
    this.state.globalError = null
    this.state.componentErrors.clear()
    this.state.errorBoundaries.clear()
    this.state.isRecovering = false
    this.state.lastRecoveryAttempt = null
    this.state.metrics = {
      totalErrors: 0,
      errorsByType: {},
      errorsByComponent: {},
      recoverySuccessRate: 0,
      averageRecoveryTime: 0,
      recentErrors: []
    }
    this.errorQueue = []
  }

  /**
   * Get error summary
   */
  public getErrorSummary(): {
    hasErrors: boolean
    totalErrors: number
    criticalErrors: number
    recentErrors: number
    recoveryRate: number
  } {
    const criticalErrors = this.state.metrics.recentErrors.filter(
      error => error.type === 'network' || error.type === 'websocket'
    ).length

    return {
      hasErrors: this.hasErrors.value,
      totalErrors: this.state.metrics.totalErrors,
      criticalErrors,
      recentErrors: this.state.metrics.recentErrors.length,
      recoveryRate: this.state.metrics.recoverySuccessRate
    }
  }
}

// Create singleton instance
const errorHandler = new DashboardErrorHandler()

// Vue composable
export function useDashboardErrorHandling() {
  return {
    // State
    globalError: errorHandler.globalError,
    componentErrors: errorHandler.componentErrors,
    isRecovering: errorHandler.isRecovering,
    metrics: errorHandler.metrics,
    hasErrors: errorHandler.hasErrors,

    // Error reporting
    reportError: errorHandler.reportError.bind(errorHandler),
    reportNetworkError: errorHandler.reportNetworkError.bind(errorHandler),
    reportWebSocketError: errorHandler.reportWebSocketError.bind(errorHandler),
    reportDataError: errorHandler.reportDataError.bind(errorHandler),
    reportParsingError: errorHandler.reportParsingError.bind(errorHandler),

    // Error boundaries
    createErrorBoundary: errorHandler.createErrorBoundary.bind(errorHandler),
    triggerErrorBoundary: errorHandler.triggerErrorBoundary.bind(errorHandler),
    retryErrorBoundary: errorHandler.retryErrorBoundary.bind(errorHandler),
    clearErrorBoundary: errorHandler.clearErrorBoundary.bind(errorHandler),

    // Recovery and fallbacks
    attemptErrorRecovery: errorHandler.attemptErrorRecovery.bind(errorHandler),
    applyFallback: errorHandler.applyFallback.bind(errorHandler),
    getFallbackConfig: errorHandler.getFallbackConfig.bind(errorHandler),

    // Configuration
    registerRecoveryStrategy: errorHandler.registerRecoveryStrategy.bind(errorHandler),
    registerFallbackConfig: errorHandler.registerFallbackConfig.bind(errorHandler),

    // State management
    clearComponentErrors: errorHandler.clearComponentErrors.bind(errorHandler),
    clearGlobalError: errorHandler.clearGlobalError.bind(errorHandler),
    getComponentErrorStatus: errorHandler.getComponentErrorStatus.bind(errorHandler),

    // Events
    onError: errorHandler.onError.bind(errorHandler),
    onRecovery: errorHandler.onRecovery.bind(errorHandler),

    // Utilities
    getErrorSummary: errorHandler.getErrorSummary.bind(errorHandler),
    reset: errorHandler.reset.bind(errorHandler)
  }
}

export default errorHandler