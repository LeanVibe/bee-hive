/**
 * Performance optimization utilities for PWA
 * Targets 45+ fps on low-end Android devices
 */

export interface DeviceCapabilities {
  hardwareConcurrency: number
  memoryGB: number
  connectionType: string
  isLowEndDevice: boolean
  estimatedPerformanceLevel: 'high' | 'medium' | 'low'
}

export class PerformanceOptimizer {
  private static instance: PerformanceOptimizer
  private frameCount = 0
  private lastTime = 0
  private fps = 0
  private isMonitoring = false
  private deviceCapabilities?: DeviceCapabilities
  private rafId?: number
  
  static getInstance(): PerformanceOptimizer {
    if (!PerformanceOptimizer.instance) {
      PerformanceOptimizer.instance = new PerformanceOptimizer()
    }
    return PerformanceOptimizer.instance
  }
  
  async initialize(): Promise<void> {
    console.log('🚀 Initializing performance optimizer...')
    
    this.deviceCapabilities = await this.detectDeviceCapabilities()
    this.applyDeviceSpecificOptimizations()
    this.startPerformanceMonitoring()
    
    console.log(`✅ Performance optimizer initialized`)
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
  
  public stop(): void {
    this.isMonitoring = false
    if (this.rafId) {
      cancelAnimationFrame(this.rafId)
    }
  }
}