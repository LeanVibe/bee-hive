export interface PerformanceMetric {
  name: string
  value: number
  timestamp: number
  labels?: Record<string, string>
}

export interface WebVitalMetric {
  name: string
  value: number
  delta: number
  id: string
  rating: 'good' | 'needs-improvement' | 'poor'
}

export interface ErrorReport {
  message: string
  stack?: string
  timestamp: number
  url: string
  line?: number
  column?: number
  userAgent: string
  userId?: string
}

export class PerformanceMonitor {
  private static instance: PerformanceMonitor
  private metrics: PerformanceMetric[] = []
  private observer: PerformanceObserver | null = null
  private sessionId: string = crypto.randomUUID()
  private startTime: number = performance.now()
  
  static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor()
    }
    return PerformanceMonitor.instance
  }
  
  startSession(): void {
    this.sessionId = crypto.randomUUID()
    this.startTime = performance.now()
    this.setupPerformanceObserver()
    
    console.log('📊 Performance monitoring started for session:', this.sessionId)
  }
  
  private setupPerformanceObserver(): void {
    if (!('PerformanceObserver' in window)) {
      console.warn('PerformanceObserver not supported')
      return
    }
    
    try {
      this.observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          this.handlePerformanceEntry(entry)
        }
      })
      
      // Observe different types of performance entries
      const observeTypes = ['navigation', 'resource', 'measure', 'mark']
      
      for (const type of observeTypes) {
        try {
          this.observer.observe({ type, buffered: true })
        } catch (e) {
          // Some types might not be supported
          console.warn(`Performance observer type ${type} not supported`)
        }
      }
      
      // Observe layout shifts and largest contentful paint if supported
      try {
        this.observer.observe({ type: 'layout-shift', buffered: true })
      } catch (e) {
        // Not supported in all browsers
      }
      
      try {
        this.observer.observe({ type: 'largest-contentful-paint', buffered: true })
      } catch (e) {
        // Not supported in all browsers
      }
      
    } catch (error) {
      console.error('Failed to setup performance observer:', error)
    }
  }
  
  private handlePerformanceEntry(entry: PerformanceEntry): void {
    const metric: PerformanceMetric = {
      name: `${entry.entryType}_${entry.name}`,
      value: entry.duration || entry.startTime,
      timestamp: Date.now(),
      labels: {
        entryType: entry.entryType,
        sessionId: this.sessionId
      }
    }
    
    // Add specific labels based on entry type
    if (entry.entryType === 'navigation') {
      const navEntry = entry as PerformanceNavigationTiming
      metric.labels = {
        ...metric.labels,
        domContentLoaded: navEntry.domContentLoadedEventEnd.toString(),
        loadComplete: navEntry.loadEventEnd.toString(),
        transferSize: navEntry.transferSize?.toString() || '0'
      }
    } else if (entry.entryType === 'resource') {
      const resourceEntry = entry as PerformanceResourceTiming
      metric.labels = {
        ...metric.labels,
        initiatorType: resourceEntry.initiatorType,
        transferSize: resourceEntry.transferSize?.toString() || '0',
        responseStatus: resourceEntry.responseStatus?.toString() || '0'
      }
    } else if (entry.entryType === 'layout-shift') {
      const layoutEntry = entry as PerformanceEntry & { value: number }
      metric.value = layoutEntry.value
      metric.labels = {
        ...metric.labels,
        hadRecentInput: 'hadRecentInput' in layoutEntry ? layoutEntry.hadRecentInput.toString() : 'false'
      }
    }
    
    this.addMetric(metric)
  }
  
  track(name: string, value: number, labels?: Record<string, string>): void {
    const metric: PerformanceMetric = {
      name,
      value,
      timestamp: Date.now(),
      labels: {
        sessionId: this.sessionId,
        ...labels
      }
    }
    
    this.addMetric(metric)
  }
  
  markStart(name: string): void {
    performance.mark(`${name}-start`)
  }
  
  markEnd(name: string): number {
    const endMark = `${name}-end`
    performance.mark(endMark)
    
    const measureName = `${name}-duration`
    performance.measure(measureName, `${name}-start`, endMark)
    
    const measure = performance.getEntriesByName(measureName)[0]
    const duration = measure?.duration || 0
    
    this.track(measureName, duration)
    
    // Clean up marks
    performance.clearMarks(`${name}-start`)
    performance.clearMarks(endMark)
    performance.clearMeasures(measureName)
    
    return duration
  }
  
  measureFunction<T>(name: string, fn: () => T): T {
    this.markStart(name)
    try {
      const result = fn()
      this.markEnd(name)
      return result
    } catch (error) {
      this.markEnd(name)
      this.reportError(error)
      throw error
    }
  }
  
  async measureAsyncFunction<T>(name: string, fn: () => Promise<T>): Promise<T> {
    this.markStart(name)
    try {
      const result = await fn()
      this.markEnd(name)
      return result
    } catch (error) {
      this.markEnd(name)
      this.reportError(error)
      throw error
    }
  }
  
  reportWebVital(metric: WebVitalMetric): void {
    console.log('🎯 Web Vital:', metric)
    
    this.track(`web_vital_${metric.name.toLowerCase()}`, metric.value, {
      rating: metric.rating,
      delta: metric.delta.toString(),
      id: metric.id
    })
    
    // Send to analytics if available
    this.sendMetricToAnalytics({
      name: `web_vital_${metric.name.toLowerCase()}`,
      value: metric.value,
      timestamp: Date.now(),
      labels: {
        rating: metric.rating,
        delta: metric.delta.toString(),
        id: metric.id,
        sessionId: this.sessionId
      }
    })
  }
  
  reportError(error: unknown): void {
    const errorReport: ErrorReport = {
      message: error instanceof Error ? error.message : 'Unknown error',
      stack: error instanceof Error ? error.stack : undefined,
      timestamp: Date.now(),
      url: window.location.href,
      userAgent: navigator.userAgent,
      userId: this.getCurrentUserId()
    }
    
    console.error('🚨 Error reported:', errorReport)
    
    this.track('error_count', 1, {
      errorType: error instanceof Error ? error.constructor.name : 'Unknown',
      message: errorReport.message
    })
    
    // Send to error tracking service
    this.sendErrorToService(errorReport)
  }
  
  private addMetric(metric: PerformanceMetric): void {
    this.metrics.push(metric)
    
    // Keep only last 1000 metrics to prevent memory issues
    if (this.metrics.length > 1000) {
      this.metrics = this.metrics.slice(-1000)
    }
    
    // Send metric to analytics
    this.sendMetricToAnalytics(metric)
  }
  
  private sendMetricToAnalytics(metric: PerformanceMetric): void {
    // In a real app, this would send to your analytics service
    if (process.env.NODE_ENV === 'development') {
      console.debug('📈 Metric:', metric)
    }
    
    // Example: Send to custom analytics endpoint
    if (navigator.sendBeacon && metric.name.includes('web_vital')) {
      const data = JSON.stringify({
        type: 'performance_metric',
        sessionId: this.sessionId,
        ...metric
      })
      
      navigator.sendBeacon('/api/v1/analytics/metrics', data)
    }
  }
  
  private sendErrorToService(error: ErrorReport): void {
    // In a real app, this would send to your error tracking service
    if (process.env.NODE_ENV === 'development') {
      console.error('🚨 Error report:', error)
    }
    
    // Example: Send to error tracking endpoint
    if (navigator.sendBeacon) {
      const data = JSON.stringify({
        type: 'error_report',
        sessionId: this.sessionId,
        ...error
      })
      
      navigator.sendBeacon('/api/v1/analytics/errors', data)
    }
  }
  
  private getCurrentUserId(): string | undefined {
    // Get user ID from auth service or localStorage
    try {
      const authData = JSON.parse(localStorage.getItem('auth_state') || '{}')
      return authData.user?.id
    } catch {
      return undefined
    }
  }
  
  // Memory monitoring
  getMemoryUsage(): any {
    if ('memory' in performance) {
      const memory = (performance as any).memory
      return {
        usedJSHeapSize: memory.usedJSHeapSize,
        totalJSHeapSize: memory.totalJSHeapSize,
        jsHeapSizeLimit: memory.jsHeapSizeLimit
      }
    }
    return null
  }
  
  trackMemoryUsage(): void {
    const memory = this.getMemoryUsage()
    if (memory) {
      this.track('memory_used_mb', Math.round(memory.usedJSHeapSize / 1024 / 1024))
      this.track('memory_total_mb', Math.round(memory.totalJSHeapSize / 1024 / 1024))
    }
  }
  
  // Network monitoring
  getConnectionInfo(): any {
    if ('connection' in navigator) {
      const connection = (navigator as any).connection
      return {
        effectiveType: connection.effectiveType,
        downlink: connection.downlink,
        rtt: connection.rtt,
        saveData: connection.saveData
      }
    }
    return null
  }
  
  trackConnectionInfo(): void {
    const connection = this.getConnectionInfo()
    if (connection) {
      this.track('network_downlink_mbps', connection.downlink || 0, {
        effectiveType: connection.effectiveType,
        saveData: connection.saveData.toString()
      })
      this.track('network_rtt_ms', connection.rtt || 0)
    }
  }
  
  // Battery monitoring
  async getBatteryInfo(): Promise<any> {
    if ('getBattery' in navigator) {
      try {
        const battery = await (navigator as any).getBattery()
        return {
          level: battery.level,
          charging: battery.charging,
          chargingTime: battery.chargingTime,
          dischargingTime: battery.dischargingTime
        }
      } catch (error) {
        return null
      }
    }
    return null
  }
  
  async trackBatteryInfo(): Promise<void> {
    const battery = await this.getBatteryInfo()
    if (battery) {
      this.track('battery_level', Math.round(battery.level * 100), {
        charging: battery.charging.toString()
      })
    }
  }
  
  // Get performance summary
  getPerformanceSummary(): any {
    const now = Date.now()
    const sessionDuration = performance.now() - this.startTime
    
    const summary = {
      sessionId: this.sessionId,
      sessionDuration: Math.round(sessionDuration),
      metricsCount: this.metrics.length,
      memory: this.getMemoryUsage(),
      connection: this.getConnectionInfo(),
      timestamp: now
    }
    
    return summary
  }
  
  // Export metrics for debugging
  exportMetrics(): PerformanceMetric[] {
    return [...this.metrics]
  }
  
  // Clear metrics
  clearMetrics(): void {
    this.metrics = []
  }
  
  // Stop monitoring
  stop(): void {
    if (this.observer) {
      this.observer.disconnect()
      this.observer = null
    }
    
    console.log('📊 Performance monitoring stopped')
    
    // Send final summary
    const summary = this.getPerformanceSummary()
    this.sendMetricToAnalytics({
      name: 'session_summary',
      value: summary.sessionDuration,
      timestamp: Date.now(),
      labels: {
        sessionId: this.sessionId,
        metricsCount: summary.metricsCount.toString()
      }
    })
  }
}