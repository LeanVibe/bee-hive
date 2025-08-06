/**
 * Performance optimization utilities for PWA
 * Targets 45+ fps on low-end Android devices
 * Phase 2: Enhanced with intelligent monitoring and predictive analytics
 */

export interface DeviceCapabilities {
  hardwareConcurrency: number
  memoryGB: number
  connectionType: string
  isLowEndDevice: boolean
  estimatedPerformanceLevel: 'high' | 'medium' | 'low'
}

export interface PerformanceMetrics {
  fps: number
  memoryUsage: number
  loadTime: number
  renderTime: number
  interactionLatency: number
  networkLatency: number
  timestamp: number
}

export interface PerformancePrediction {
  predictedFps: number
  confidenceScore: number
  riskFactors: string[]
  recommendations: string[]
  timeHorizon: number
}

export interface SystemResourceMetrics {
  cpu: {
    usage: number
    cores: number
    frequency?: number
  }
  memory: {
    used: number
    total: number
    available: number
    pressure?: 'low' | 'medium' | 'high'
  }
  network: {
    bandwidth: number
    latency: number
    connection: string
    downlink?: number
  }
  storage: {
    quota: number
    usage: number
    available: number
  }
}

export class PerformanceOptimizer {
  private static instance: PerformanceOptimizer
  private frameCount = 0
  private lastTime = 0
  private fps = 0
  private isMonitoring = false
  private deviceCapabilities?: DeviceCapabilities
  private rafId?: number
  private metricsHistory: PerformanceMetrics[] = []
  private maxHistorySize = 1000
  private alertCallbacks: ((alert: PerformanceAlert) => void)[] = []
  private resourceMetrics?: SystemResourceMetrics
  private performanceObserver?: PerformanceObserver
  private predictionModel: SimpleRegressionModel = new SimpleRegressionModel()
  
  // Performance thresholds
  private readonly thresholds = {
    fps: { critical: 20, warning: 30, target: 45 },
    memory: { critical: 90, warning: 80, target: 70 },
    loadTime: { critical: 5000, warning: 3000, target: 1000 },
    renderTime: { critical: 100, warning: 50, target: 16 },
    interactionLatency: { critical: 300, warning: 100, target: 50 }
  }
  
  static getInstance(): PerformanceOptimizer {
    if (!PerformanceOptimizer.instance) {
      PerformanceOptimizer.instance = new PerformanceOptimizer()
    }
    return PerformanceOptimizer.instance
  }
  
  async initialize(): Promise<void> {
    console.log('🚀 Initializing performance optimizer...')
    
    this.deviceCapabilities = await this.detectDeviceCapabilities()
    this.resourceMetrics = await this.gatherSystemResourceMetrics()
    this.applyDeviceSpecificOptimizations()
    this.setupAdvancedObservation()
    this.startPerformanceMonitoring()
    this.startPredictiveAnalytics()
    
    console.log(`✅ Performance optimizer initialized with enhanced monitoring`)
    console.log(`📊 Device: ${this.deviceCapabilities.estimatedPerformanceLevel} performance level`)
    console.log(`💾 Memory: ${this.resourceMetrics.memory.total}MB total, ${this.resourceMetrics.memory.used}MB used`)
  }
  
  private async detectDeviceCapabilities(): Promise<DeviceCapabilities> {
    const nav = navigator as any
    const hardwareConcurrency = nav.hardwareConcurrency || 2
    const memory = nav.deviceMemory || this.estimateMemory()
    const connection = nav.connection || {}
    const connectionType = connection.effectiveType || 'unknown'
    
    const isLowEndDevice = this.isLowEndDevice(hardwareConcurrency, memory, connectionType)
    const estimatedPerformanceLevel = this.estimatePerformanceLevel(hardwareConcurrency, memory, connectionType)
    
    return {
      hardwareConcurrency,
      memoryGB: memory,
      connectionType,
      isLowEndDevice,
      estimatedPerformanceLevel
    }
  }
  
  private estimateMemory(): number {
    const userAgent = navigator.userAgent.toLowerCase()
    if (userAgent.includes('android 4') || userAgent.includes('android 5')) return 1
    if (userAgent.includes('android 6') || userAgent.includes('android 7')) return 2
    return 4
  }
  
  private isLowEndDevice(cores: number, memory: number, connection: string): boolean {
    return cores <= 2 || memory <= 2 || connection === '2g' || connection === 'slow-2g'
  }
  
  private estimatePerformanceLevel(cores: number, memory: number, connection: string): 'high' | 'medium' | 'low' {
    let score = 0
    if (cores >= 4) score += 2; else if (cores >= 2) score += 1
    if (memory >= 4) score += 2; else if (memory >= 2) score += 1
    if (connection === '4g') score += 1
    
    if (score >= 4) return 'high'
    if (score >= 2) return 'medium'
    return 'low'
  }
  
  private applyDeviceSpecificOptimizations(): void {
    if (!this.deviceCapabilities) return
    
    if (this.deviceCapabilities.isLowEndDevice) {
      this.enableLowEndOptimizations()
    }
  }
  
  private enableLowEndOptimizations(): void {
    document.documentElement.classList.add('low-end-device')
    document.documentElement.style.setProperty('--animation-duration', '0.1s')
    document.documentElement.style.setProperty('--blur-amount', '0px')
  }
  
  private startPerformanceMonitoring(): void {
    this.isMonitoring = true
    this.lastTime = performance.now()
    this.monitorFPS()
    this.startResourceMonitoring()
  }
  
  private async gatherSystemResourceMetrics(): Promise<SystemResourceMetrics> {
    const nav = navigator as any
    const memory = (performance as any).memory || {}
    const connection = nav.connection || {}
    
    // Estimate storage quota
    let storageQuota = 0
    let storageUsage = 0
    try {
      if ('storage' in navigator && 'estimate' in navigator.storage) {
        const estimate = await navigator.storage.estimate()
        storageQuota = estimate.quota || 0
        storageUsage = estimate.usage || 0
      }
    } catch (error) {
      console.warn('Could not estimate storage:', error)
    }
    
    return {
      cpu: {
        usage: 0, // Will be updated by monitoring
        cores: nav.hardwareConcurrency || 2,
        frequency: undefined
      },
      memory: {
        used: memory.usedJSHeapSize || 0,
        total: memory.totalJSHeapSize || this.deviceCapabilities?.memoryGB || 0,
        available: memory.totalJSHeapSize - memory.usedJSHeapSize || 0,
        pressure: this.assessMemoryPressure(memory)
      },
      network: {
        bandwidth: connection.downlink || 0,
        latency: connection.rtt || 0,
        connection: connection.effectiveType || 'unknown',
        downlink: connection.downlink
      },
      storage: {
        quota: storageQuota,
        usage: storageUsage,
        available: storageQuota - storageUsage
      }
    }
  }
  
  private assessMemoryPressure(memory: any): 'low' | 'medium' | 'high' {
    if (!memory.usedJSHeapSize || !memory.totalJSHeapSize) return 'low'
    
    const usagePercent = (memory.usedJSHeapSize / memory.totalJSHeapSize) * 100
    if (usagePercent > 85) return 'high'
    if (usagePercent > 70) return 'medium'
    return 'low'
  }
  
  private setupAdvancedObservation(): void {
    if ('PerformanceObserver' in window) {
      this.performanceObserver = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          if (entry.entryType === 'measure' || entry.entryType === 'navigation') {
            this.processPerformanceEntry(entry)
          }
        })
      })
      
      try {
        this.performanceObserver.observe({ 
          entryTypes: ['measure', 'navigation', 'paint', 'largest-contentful-paint'] 
        })
      } catch (error) {
        console.warn('Performance Observer setup failed:', error)
      }
    }
  }
  
  private processPerformanceEntry(entry: PerformanceEntry): void {
    // Process navigation timing
    if (entry.entryType === 'navigation') {
      const navEntry = entry as PerformanceNavigationTiming
      const loadTime = navEntry.loadEventEnd - navEntry.navigationStart
      this.updateMetric('loadTime', loadTime)
    }
    
    // Process paint timing
    if (entry.entryType === 'paint') {
      if (entry.name === 'first-contentful-paint') {
        this.updateMetric('renderTime', entry.startTime)
      }
    }
  }
  
  private updateMetric(metricName: string, value: number): void {
    // Check if metric exceeds thresholds and trigger alerts
    const threshold = this.thresholds[metricName as keyof typeof this.thresholds]
    if (threshold) {
      if (value >= threshold.critical) {
        this.triggerAlert({
          type: 'performance',
          severity: 'critical',
          metric: metricName,
          value,
          threshold: threshold.critical,
          message: `Critical performance issue: ${metricName} = ${value}`,
          timestamp: Date.now()
        })
      } else if (value >= threshold.warning) {
        this.triggerAlert({
          type: 'performance',
          severity: 'warning',
          metric: metricName,
          value,
          threshold: threshold.warning,
          message: `Performance warning: ${metricName} = ${value}`,
          timestamp: Date.now()
        })
      }
    }
  }
  
  private startResourceMonitoring(): void {
    setInterval(async () => {
      if (!this.isMonitoring) return
      
      try {
        this.resourceMetrics = await this.gatherSystemResourceMetrics()
        
        const currentMetrics: PerformanceMetrics = {
          fps: this.fps,
          memoryUsage: this.resourceMetrics.memory.used,
          loadTime: 0, // Will be updated by performance observer
          renderTime: 0, // Will be updated by performance observer
          interactionLatency: this.measureInteractionLatency(),
          networkLatency: this.resourceMetrics.network.latency,
          timestamp: Date.now()
        }
        
        this.addMetricsToHistory(currentMetrics)
        this.updatePredictionModel(currentMetrics)
        
      } catch (error) {
        console.warn('Resource monitoring error:', error)
      }
    }, 5000) // Update every 5 seconds
  }
  
  private measureInteractionLatency(): number {
    // Measure time from last interaction to response
    // This is a simplified implementation
    return performance.now() % 100 // Placeholder
  }
  
  private addMetricsToHistory(metrics: PerformanceMetrics): void {
    this.metricsHistory.push(metrics)
    if (this.metricsHistory.length > this.maxHistorySize) {
      this.metricsHistory.shift()
    }
  }
  
  private startPredictiveAnalytics(): void {
    setInterval(() => {
      if (this.metricsHistory.length > 10) {
        const prediction = this.generatePerformancePrediction()
        if (prediction.confidenceScore > 0.7 && prediction.riskFactors.length > 0) {
          this.triggerPredictiveAlert(prediction)
        }
      }
    }, 30000) // Analyze every 30 seconds
  }
  
  private monitorFPS(): void {
    const measureFPS = (currentTime: number) => {
      this.frameCount++
      
      if (currentTime - this.lastTime >= 1000) {
        this.fps = Math.round((this.frameCount * 1000) / (currentTime - this.lastTime))
        
        if (this.fps < 30) {
          this.handlePerformanceDegradation()
        }
        
        this.frameCount = 0
        this.lastTime = currentTime
      }
      
      if (this.isMonitoring) {
        this.rafId = requestAnimationFrame(measureFPS)
      }
    }
    
    this.rafId = requestAnimationFrame(measureFPS)
  }
  
  private handlePerformanceDegradation(): void {
    console.warn(`⚠️ Performance degradation detected (${this.fps} fps)`)
    document.documentElement.style.setProperty('--animation-duration', '0s')
  }
  
  public getCurrentFPS(): number {
    return this.fps
  }
  
  public getDeviceCapabilities(): DeviceCapabilities | undefined {
    return this.deviceCapabilities
  }
  
  public getSystemResourceMetrics(): SystemResourceMetrics | undefined {
    return this.resourceMetrics
  }
  
  public getPerformanceHistory(limit?: number): PerformanceMetrics[] {
    return limit ? this.metricsHistory.slice(-limit) : [...this.metricsHistory]
  }
  
  public generatePerformancePrediction(timeHorizonMinutes = 10): PerformancePrediction {
    if (this.metricsHistory.length < 5) {
      return {
        predictedFps: this.fps,
        confidenceScore: 0,
        riskFactors: [],
        recommendations: [],
        timeHorizon: timeHorizonMinutes
      }
    }
    
    const recentMetrics = this.metricsHistory.slice(-10)
    const fpsValues = recentMetrics.map(m => m.fps)
    const memoryValues = recentMetrics.map(m => m.memoryUsage)
    
    // Simple linear regression for FPS prediction
    const predictedFps = this.predictionModel.predict(fpsValues, timeHorizonMinutes)
    const confidenceScore = this.calculatePredictionConfidence(fpsValues)
    
    const riskFactors: string[] = []
    const recommendations: string[] = []
    
    // Analyze trends for risk factors
    if (this.calculateTrend(fpsValues) < -0.5) {
      riskFactors.push('Declining FPS trend detected')
      recommendations.push('Reduce visual complexity or enable performance mode')
    }
    
    if (this.calculateTrend(memoryValues) > 0.3) {
      riskFactors.push('Increasing memory usage trend')
      recommendations.push('Clear caches or optimize memory usage')
    }
    
    if (this.resourceMetrics?.memory.pressure === 'high') {
      riskFactors.push('High memory pressure detected')
      recommendations.push('Close unused components or reduce concurrent operations')
    }
    
    return {
      predictedFps,
      confidenceScore,
      riskFactors,
      recommendations,
      timeHorizon: timeHorizonMinutes
    }
  }
  
  private calculateTrend(values: number[]): number {
    if (values.length < 2) return 0
    
    const x = values.map((_, i) => i)
    const y = values
    const n = values.length
    
    const sumX = x.reduce((a, b) => a + b, 0)
    const sumY = y.reduce((a, b) => a + b, 0)
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0)
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0)
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX)
    return slope
  }
  
  private calculatePredictionConfidence(values: number[]): number {
    const variance = this.calculateVariance(values)
    const maxVariance = 100 // Adjust based on your metrics
    return Math.max(0, Math.min(1, 1 - variance / maxVariance))
  }
  
  private calculateVariance(values: number[]): number {
    if (values.length < 2) return 0
    
    const mean = values.reduce((a, b) => a + b, 0) / values.length
    const squareDiffs = values.map(value => Math.pow(value - mean, 2))
    return squareDiffs.reduce((a, b) => a + b, 0) / values.length
  }
  
  private updatePredictionModel(metrics: PerformanceMetrics): void {
    this.predictionModel.addDataPoint({
      input: [metrics.memoryUsage, metrics.networkLatency, metrics.interactionLatency],
      output: metrics.fps
    })
  }
  
  private triggerAlert(alert: PerformanceAlert): void {
    console.warn(`🚨 Performance Alert [${alert.severity.toUpperCase()}]:`, alert.message)
    this.alertCallbacks.forEach(callback => callback(alert))
  }
  
  private triggerPredictiveAlert(prediction: PerformancePrediction): void {
    if (prediction.riskFactors.length > 0) {
      const alert: PerformanceAlert = {
        type: 'predictive',
        severity: prediction.predictedFps < 20 ? 'critical' : 'warning',
        metric: 'fps_prediction',
        value: prediction.predictedFps,
        threshold: 30,
        message: `Predicted performance degradation: ${prediction.riskFactors.join(', ')}`,
        timestamp: Date.now(),
        prediction
      }
      this.triggerAlert(alert)
    }
  }
  
  public onPerformanceAlert(callback: (alert: PerformanceAlert) => void): void {
    this.alertCallbacks.push(callback)
  }
  
  public removePerformanceAlert(callback: (alert: PerformanceAlert) => void): void {
    const index = this.alertCallbacks.indexOf(callback)
    if (index > -1) {
      this.alertCallbacks.splice(index, 1)
    }
  }
  
  public getPerformanceReport(): PerformanceReport {
    const recent = this.metricsHistory.slice(-10)
    const prediction = this.generatePerformancePrediction()
    
    return {
      currentFps: this.fps,
      averageFps: recent.reduce((sum, m) => sum + m.fps, 0) / recent.length || 0,
      memoryUsage: this.resourceMetrics?.memory.used || 0,
      memoryPressure: this.resourceMetrics?.memory.pressure || 'low',
      networkLatency: this.resourceMetrics?.network.latency || 0,
      deviceCapabilities: this.deviceCapabilities!,
      prediction,
      healthScore: this.calculateHealthScore(),
      recommendations: prediction.recommendations,
      timestamp: Date.now()
    }
  }
  
  private calculateHealthScore(): number {
    if (!this.resourceMetrics) return 0
    
    let score = 100
    
    // FPS score (40% weight)
    if (this.fps < 20) score -= 40
    else if (this.fps < 30) score -= 20
    else if (this.fps < 45) score -= 10
    
    // Memory score (30% weight)
    const memoryPercent = (this.resourceMetrics.memory.used / this.resourceMetrics.memory.total) * 100
    if (memoryPercent > 90) score -= 30
    else if (memoryPercent > 80) score -= 15
    else if (memoryPercent > 70) score -= 5
    
    // Network score (20% weight)
    if (this.resourceMetrics.network.latency > 500) score -= 20
    else if (this.resourceMetrics.network.latency > 200) score -= 10
    
    // Storage score (10% weight)
    const storagePercent = (this.resourceMetrics.storage.usage / this.resourceMetrics.storage.quota) * 100
    if (storagePercent > 95) score -= 10
    else if (storagePercent > 85) score -= 5
    
    return Math.max(0, Math.min(100, score))
  }
  
  public stop(): void {
    this.isMonitoring = false
    if (this.rafId) {
      cancelAnimationFrame(this.rafId)
    }
    if (this.performanceObserver) {
      this.performanceObserver.disconnect()
    }
  }
}

// Performance Alert interface
export interface PerformanceAlert {
  type: 'performance' | 'predictive'
  severity: 'warning' | 'critical'
  metric: string
  value: number
  threshold: number
  message: string
  timestamp: number
  prediction?: PerformancePrediction
}

// Performance Report interface
export interface PerformanceReport {
  currentFps: number
  averageFps: number
  memoryUsage: number
  memoryPressure: 'low' | 'medium' | 'high'
  networkLatency: number
  deviceCapabilities: DeviceCapabilities
  prediction: PerformancePrediction
  healthScore: number
  recommendations: string[]
  timestamp: number
}

// Simple regression model for predictions
class SimpleRegressionModel {
  private dataPoints: { input: number[], output: number }[] = []
  private maxDataPoints = 50
  
  addDataPoint(point: { input: number[], output: number }): void {
    this.dataPoints.push(point)
    if (this.dataPoints.length > this.maxDataPoints) {
      this.dataPoints.shift()
    }
  }
  
  predict(recentValues: number[], timeHorizonMinutes: number): number {
    if (recentValues.length < 3) {
      return recentValues[recentValues.length - 1] || 0
    }
    
    // Simple linear extrapolation based on recent trend
    const trend = this.calculateTrend(recentValues)
    const currentValue = recentValues[recentValues.length - 1]
    const prediction = currentValue + (trend * timeHorizonMinutes)
    
    // Clamp prediction to reasonable bounds
    return Math.max(0, Math.min(60, prediction))
  }
  
  private calculateTrend(values: number[]): number {
    if (values.length < 2) return 0
    
    const x = values.map((_, i) => i)
    const y = values
    const n = values.length
    
    const sumX = x.reduce((a, b) => a + b, 0)
    const sumY = y.reduce((a, b) => a + b, 0)
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0)
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0)
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX)
    return isNaN(slope) ? 0 : slope
  }
}

// Export alias for compatibility
export const PerformanceMonitor = PerformanceOptimizer