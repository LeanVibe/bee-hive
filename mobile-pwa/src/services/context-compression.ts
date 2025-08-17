/**
 * Context Compression Service for LeanVibe Agent Hive
 * 
 * Provides frontend interface for the /hive:compact command with:
 * - Real-time compression progress tracking
 * - WebSocket integration for status updates  
 * - Mobile-optimized compression controls
 * - Compression metrics and history
 */

import { BaseService } from './base-service'
import { WebSocketService } from './websocket'
import type { ApiResponse } from '../types/api'

export interface CompressionOptions {
  sessionId?: string
  level?: 'light' | 'standard' | 'aggressive'
  targetTokens?: number
  preserveDecisions?: boolean
  preservePatterns?: boolean
}

export interface CompressionResult {
  success: boolean
  sessionId?: string
  compressionLevel: string
  originalTokens: number
  compressedTokens: number
  compressionRatio: number
  tokensSaved: number
  compressionTimeSeconds: number
  summary: string
  keyInsights: string[]
  decisionsMade: string[]
  patternsIdentified: string[]
  importanceScore: number
  message: string
  performanceMet: boolean
  timestamp: string
}

export interface CompressionProgress {
  sessionId?: string
  stage: 'analyzing' | 'compressing' | 'optimizing' | 'finalizing' | 'completed' | 'error'
  progress: number // 0-100
  currentStep: string
  estimatedTimeRemaining?: number
  tokensProcessed?: number
  totalTokens?: number
  error?: string
}

export interface CompressionMetrics {
  totalCompressions: number
  averageCompressionRatio: number
  totalTokensSaved: number
  averageCompressionTime: number
  successRate: number
  lastCompression?: CompressionResult
}

export interface CompressionHistory {
  results: CompressionResult[]
  metrics: CompressionMetrics
  lastUpdated: string
}

export class ContextCompressionService extends BaseService {
  private static instance: ContextCompressionService
  private webSocket: WebSocketService
  private activeCompressions = new Map<string, CompressionProgress>()
  private compressionHistory: CompressionResult[] = []
  private isMonitoring = false

  static getInstance(config?: any): ContextCompressionService {
    if (!ContextCompressionService.instance) {
      ContextCompressionService.instance = new ContextCompressionService(config)
    }
    return ContextCompressionService.instance
  }

  constructor(config?: any) {
    super(config)
    this.webSocket = WebSocketService.getInstance()
    this.setupWebSocketListeners()
  }

  private setupWebSocketListeners(): void {
    // Listen for compression progress updates
    this.webSocket.on('compression-progress', (data: CompressionProgress) => {
      this.handleCompressionProgress(data)
    })

    // Listen for compression completion
    this.webSocket.on('compression-completed', (data: CompressionResult) => {
      this.handleCompressionCompleted(data)
    })

    // Listen for compression errors
    this.webSocket.on('compression-error', (data: { sessionId?: string; error: string }) => {
      this.handleCompressionError(data)
    })
  }

  /**
   * Start a context compression operation
   */
  async compressContext(options: CompressionOptions = {}): Promise<CompressionResult> {
    try {
      // Build the hive command with options
      const commandParts = ['/hive:compact']
      
      if (options.sessionId) {
        commandParts.push(options.sessionId)
      }
      
      if (options.level) {
        commandParts.push(`--level=${options.level}`)
      }
      
      if (options.targetTokens) {
        commandParts.push(`--target-tokens=${options.targetTokens}`)
      }
      
      if (options.preserveDecisions === false) {
        commandParts.push('--no-preserve-decisions')
      }
      
      if (options.preservePatterns === false) {
        commandParts.push('--no-preserve-patterns')
      }

      const command = commandParts.join(' ')

      // Execute the hive command via API
      const response = await this.makeRequest<CompressionResult>('POST', '/api/hive/commands', {
        command,
        context: {
          source: 'mobile_dashboard',
          timestamp: new Date().toISOString()
        }
      })

      if (response.success) {
        // Store in history
        this.addToHistory(response)
        
        // Update cache
        this.updateCache('latest-compression', response)
        
        return response
      } else {
        throw new Error(response.error || 'Compression failed')
      }
    } catch (error) {
      console.error('Context compression failed:', error)
      throw error
    }
  }

  /**
   * Get compression progress for a specific session
   */
  getCompressionProgress(sessionId?: string): CompressionProgress | null {
    if (sessionId) {
      return this.activeCompressions.get(sessionId) || null
    }
    
    // Return the most recent active compression
    const activeCompression = Array.from(this.activeCompressions.values())[0]
    return activeCompression || null
  }

  /**
   * Get all active compressions
   */
  getActiveCompressions(): Map<string, CompressionProgress> {
    return new Map(this.activeCompressions)
  }

  /**
   * Get compression history and metrics
   */
  getCompressionHistory(): CompressionHistory {
    const metrics = this.calculateMetrics()
    
    return {
      results: [...this.compressionHistory].reverse(), // Most recent first
      metrics,
      lastUpdated: new Date().toISOString()
    }
  }

  /**
   * Get compression metrics summary
   */
  getCompressionMetrics(): CompressionMetrics {
    return this.calculateMetrics()
  }

  /**
   * Get recommended compression settings based on context
   */
  getRecommendedSettings(contextInfo?: {
    tokenCount?: number
    sessionType?: string
    priority?: 'speed' | 'quality' | 'aggressive'
  }): CompressionOptions {
    const defaults: CompressionOptions = {
      level: 'standard',
      preserveDecisions: true,
      preservePatterns: true
    }

    if (!contextInfo) return defaults

    // Adjust based on token count
    if (contextInfo.tokenCount) {
      if (contextInfo.tokenCount > 10000) {
        defaults.level = 'aggressive'
        defaults.targetTokens = Math.floor(contextInfo.tokenCount * 0.3) // 70% reduction
      } else if (contextInfo.tokenCount > 5000) {
        defaults.level = 'standard'
        defaults.targetTokens = Math.floor(contextInfo.tokenCount * 0.5) // 50% reduction
      } else {
        defaults.level = 'light'
      }
    }

    // Adjust based on session type
    if (contextInfo.sessionType) {
      if (contextInfo.sessionType.includes('bug') || contextInfo.sessionType.includes('error')) {
        defaults.preserveDecisions = true // Keep error resolution decisions
      } else if (contextInfo.sessionType.includes('research')) {
        defaults.preservePatterns = true // Keep learning patterns
      }
    }

    // Adjust based on priority
    if (contextInfo.priority === 'speed') {
      defaults.level = 'light'
    } else if (contextInfo.priority === 'aggressive') {
      defaults.level = 'aggressive'
      defaults.preserveDecisions = false
      defaults.preservePatterns = false
    }

    return defaults
  }

  /**
   * Cancel an active compression
   */
  async cancelCompression(sessionId?: string): Promise<boolean> {
    try {
      const response = await this.makeRequest<{ success: boolean }>('POST', '/api/hive/commands/cancel', {
        command: 'compact',
        sessionId
      })

      if (response.success && sessionId) {
        this.activeCompressions.delete(sessionId)
      }

      return response.success
    } catch (error) {
      console.error('Failed to cancel compression:', error)
      return false
    }
  }

  /**
   * Start monitoring compression progress
   */
  startMonitoring(): void {
    if (this.isMonitoring) return

    this.isMonitoring = true
    
    // Request compression status updates via WebSocket
    this.webSocket.sendMessage({
      type: 'subscribe-compression-updates',
      data: {
        mobile_optimized: true,
        timestamp: new Date().toISOString()
      }
    })
  }

  /**
   * Stop monitoring compression progress
   */
  stopMonitoring(): void {
    if (!this.isMonitoring) return

    this.isMonitoring = false
    
    this.webSocket.sendMessage({
      type: 'unsubscribe-compression-updates',
      data: {
        timestamp: new Date().toISOString()
      }
    })
  }

  private handleCompressionProgress(progress: CompressionProgress): void {
    const sessionKey = progress.sessionId || 'default'
    this.activeCompressions.set(sessionKey, progress)
    
    // Emit progress event for UI updates
    this.emit('compression-progress', progress)
  }

  private handleCompressionCompleted(result: CompressionResult): void {
    const sessionKey = result.sessionId || 'default'
    
    // Remove from active compressions
    this.activeCompressions.delete(sessionKey)
    
    // Add to history
    this.addToHistory(result)
    
    // Emit completion event
    this.emit('compression-completed', result)
    
    // Update cache
    this.updateCache('latest-compression', result)
  }

  private handleCompressionError(data: { sessionId?: string; error: string }): void {
    const sessionKey = data.sessionId || 'default'
    
    // Update progress with error state
    const errorProgress: CompressionProgress = {
      sessionId: data.sessionId,
      stage: 'error',
      progress: 0,
      currentStep: 'Compression failed',
      error: data.error
    }
    
    this.activeCompressions.set(sessionKey, errorProgress)
    
    // Emit error event
    this.emit('compression-error', data)
  }

  private addToHistory(result: CompressionResult): void {
    this.compressionHistory.push(result)
    
    // Keep only last 50 compressions
    if (this.compressionHistory.length > 50) {
      this.compressionHistory.shift()
    }
    
    // Cache the history
    this.updateCache('compression-history', this.compressionHistory)
  }

  private calculateMetrics(): CompressionMetrics {
    if (this.compressionHistory.length === 0) {
      return {
        totalCompressions: 0,
        averageCompressionRatio: 0,
        totalTokensSaved: 0,
        averageCompressionTime: 0,
        successRate: 0
      }
    }

    const successful = this.compressionHistory.filter(r => r.success)
    const totalTokensSaved = successful.reduce((sum, r) => sum + r.tokensSaved, 0)
    const totalCompressionTime = successful.reduce((sum, r) => sum + r.compressionTimeSeconds, 0)
    const totalCompressionRatio = successful.reduce((sum, r) => sum + r.compressionRatio, 0)

    return {
      totalCompressions: this.compressionHistory.length,
      averageCompressionRatio: successful.length > 0 ? totalCompressionRatio / successful.length : 0,
      totalTokensSaved,
      averageCompressionTime: successful.length > 0 ? totalCompressionTime / successful.length : 0,
      successRate: this.compressionHistory.length > 0 ? successful.length / this.compressionHistory.length : 0,
      lastCompression: this.compressionHistory[this.compressionHistory.length - 1]
    }
  }

  /**
   * Export compression data for analysis
   */
  exportCompressionData(): {
    history: CompressionResult[]
    metrics: CompressionMetrics
    exportedAt: string
  } {
    return {
      history: this.compressionHistory,
      metrics: this.calculateMetrics(),
      exportedAt: new Date().toISOString()
    }
  }

  /**
   * Clear compression history
   */
  clearHistory(): void {
    this.compressionHistory = []
    this.clearCache('compression-history')
    this.emit('history-cleared')
  }
}

// Singleton getter
export function getContextCompressionService(config?: any): ContextCompressionService {
  return ContextCompressionService.getInstance(config)
}

// Reset for testing
export function resetContextCompressionService(): void {
  // @ts-ignore - Access private static field for testing
  ContextCompressionService.instance = undefined
}