/**
 * Unit tests for Context Compression Service
 * Tests the frontend service layer for context compression functionality
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import {
  ContextCompressionService,
  getContextCompressionService,
  resetContextCompressionService,
  type CompressionOptions,
  type CompressionResult,
  type CompressionProgress,
  type CompressionMetrics
} from '../context-compression'
import { WebSocketService } from '../websocket'

// Mock WebSocket service
vi.mock('../websocket', () => ({
  WebSocketService: {
    getInstance: vi.fn(() => ({
      on: vi.fn(),
      sendMessage: vi.fn(),
      emit: vi.fn()
    }))
  }
}))

// Mock BaseService
vi.mock('../base-service', () => ({
  BaseService: class MockBaseService {
    protected makeRequest = vi.fn()
    protected updateCache = vi.fn()
    protected clearCache = vi.fn()
    protected emit = vi.fn()
    
    constructor(config?: any) {}
  }
}))

describe('ContextCompressionService', () => {
  let service: ContextCompressionService
  let mockWebSocket: any

  beforeEach(() => {
    resetContextCompressionService()
    mockWebSocket = {
      on: vi.fn(),
      sendMessage: vi.fn(),
      emit: vi.fn()
    }
    vi.mocked(WebSocketService.getInstance).mockReturnValue(mockWebSocket)
    service = new ContextCompressionService()
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  describe('Initialization', () => {
    it('should initialize with WebSocket listeners', () => {
      expect(mockWebSocket.on).toHaveBeenCalledWith('compression-progress', expect.any(Function))
      expect(mockWebSocket.on).toHaveBeenCalledWith('compression-completed', expect.any(Function))
      expect(mockWebSocket.on).toHaveBeenCalledWith('compression-error', expect.any(Function))
    })

    it('should be a singleton', () => {
      const service1 = getContextCompressionService()
      const service2 = getContextCompressionService()
      expect(service1).toBe(service2)
    })
  })

  describe('Context Compression', () => {
    const mockSuccessResponse: CompressionResult = {
      success: true,
      sessionId: 'test-session-123',
      compressionLevel: 'standard',
      originalTokens: 1000,
      compressedTokens: 400,
      compressionRatio: 0.6,
      tokensSaved: 600,
      compressionTimeSeconds: 2.5,
      summary: 'Test compression summary',
      keyInsights: ['Insight 1', 'Insight 2'],
      decisionsMade: ['Decision 1'],
      patternsIdentified: ['Pattern 1', 'Pattern 2'],
      importanceScore: 0.8,
      message: 'Compression completed successfully',
      performanceMet: true,
      timestamp: '2024-01-01T12:00:00Z'
    }

    beforeEach(() => {
      service['makeRequest'] = vi.fn().mockResolvedValue(mockSuccessResponse)
    })

    it('should compress context with default options', async () => {
      const result = await service.compressContext()

      expect(service['makeRequest']).toHaveBeenCalledWith('POST', '/api/hive/commands', {
        command: '/hive:compact',
        context: {
          source: 'mobile_dashboard',
          timestamp: expect.any(String)
        }
      })

      expect(result).toEqual(mockSuccessResponse)
    })

    it('should compress context with session ID', async () => {
      const options: CompressionOptions = {
        sessionId: 'test-session-123'
      }

      await service.compressContext(options)

      expect(service['makeRequest']).toHaveBeenCalledWith('POST', '/api/hive/commands', {
        command: '/hive:compact test-session-123',
        context: expect.any(Object)
      })
    })

    it('should compress context with compression level', async () => {
      const options: CompressionOptions = {
        level: 'aggressive'
      }

      await service.compressContext(options)

      expect(service['makeRequest']).toHaveBeenCalledWith('POST', '/api/hive/commands', {
        command: '/hive:compact --level=aggressive',
        context: expect.any(Object)
      })
    })

    it('should compress context with target tokens', async () => {
      const options: CompressionOptions = {
        targetTokens: 300
      }

      await service.compressContext(options)

      expect(service['makeRequest']).toHaveBeenCalledWith('POST', '/api/hive/commands', {
        command: '/hive:compact --target-tokens=300',
        context: expect.any(Object)
      })
    })

    it('should compress context with preserve options', async () => {
      const options: CompressionOptions = {
        preserveDecisions: false,
        preservePatterns: false
      }

      await service.compressContext(options)

      expect(service['makeRequest']).toHaveBeenCalledWith('POST', '/api/hive/commands', {
        command: '/hive:compact --no-preserve-decisions --no-preserve-patterns',
        context: expect.any(Object)
      })
    })

    it('should compress context with all options', async () => {
      const options: CompressionOptions = {
        sessionId: 'test-session-456',
        level: 'light',
        targetTokens: 500,
        preserveDecisions: true,
        preservePatterns: false
      }

      await service.compressContext(options)

      expect(service['makeRequest']).toHaveBeenCalledWith('POST', '/api/hive/commands', {
        command: '/hive:compact test-session-456 --level=light --target-tokens=500 --no-preserve-patterns',
        context: expect.any(Object)
      })
    })

    it('should handle compression failure', async () => {
      const errorResponse = {
        success: false,
        error: 'Compression failed'
      }
      service['makeRequest'] = vi.fn().mockResolvedValue(errorResponse)

      await expect(service.compressContext()).rejects.toThrow('Compression failed')
    })

    it('should handle API errors', async () => {
      service['makeRequest'] = vi.fn().mockRejectedValue(new Error('API Error'))

      await expect(service.compressContext()).rejects.toThrow('API Error')
    })

    it('should store successful compressions in history', async () => {
      service['addToHistory'] = vi.fn()
      service['updateCache'] = vi.fn()

      await service.compressContext()

      expect(service['addToHistory']).toHaveBeenCalledWith(mockSuccessResponse)
      expect(service['updateCache']).toHaveBeenCalledWith('latest-compression', mockSuccessResponse)
    })
  })

  describe('Progress Tracking', () => {
    it('should track compression progress by session ID', () => {
      const progress: CompressionProgress = {
        sessionId: 'test-session-123',
        stage: 'compressing',
        progress: 50,
        currentStep: 'Analyzing content'
      }

      service['activeCompressions'].set('test-session-123', progress)

      const retrievedProgress = service.getCompressionProgress('test-session-123')
      expect(retrievedProgress).toEqual(progress)
    })

    it('should return most recent active compression when no session ID provided', () => {
      const progress1: CompressionProgress = {
        sessionId: 'session-1',
        stage: 'compressing',
        progress: 30,
        currentStep: 'Step 1'
      }

      const progress2: CompressionProgress = {
        sessionId: 'session-2',
        stage: 'optimizing',
        progress: 70,
        currentStep: 'Step 2'
      }

      service['activeCompressions'].set('session-1', progress1)
      service['activeCompressions'].set('session-2', progress2)

      const retrievedProgress = service.getCompressionProgress()
      expect(retrievedProgress).toBeDefined()
      expect(['session-1', 'session-2']).toContain(retrievedProgress?.sessionId)
    })

    it('should return null when no active compressions', () => {
      const progress = service.getCompressionProgress('nonexistent-session')
      expect(progress).toBeNull()
    })

    it('should return all active compressions', () => {
      const progress1: CompressionProgress = {
        sessionId: 'session-1',
        stage: 'analyzing',
        progress: 25,
        currentStep: 'Extracting content'
      }

      const progress2: CompressionProgress = {
        sessionId: 'session-2',
        stage: 'finalizing',
        progress: 95,
        currentStep: 'Storing results'
      }

      service['activeCompressions'].set('session-1', progress1)
      service['activeCompressions'].set('session-2', progress2)

      const activeCompressions = service.getActiveCompressions()
      expect(activeCompressions.size).toBe(2)
      expect(activeCompressions.get('session-1')).toEqual(progress1)
      expect(activeCompressions.get('session-2')).toEqual(progress2)
    })
  })

  describe('History and Metrics', () => {
    const mockCompressionResults: CompressionResult[] = [
      {
        success: true,
        sessionId: 'session-1',
        compressionLevel: 'standard',
        originalTokens: 1000,
        compressedTokens: 400,
        compressionRatio: 0.6,
        tokensSaved: 600,
        compressionTimeSeconds: 2.0,
        summary: 'Summary 1',
        keyInsights: ['Insight 1'],
        decisionsMade: ['Decision 1'],
        patternsIdentified: ['Pattern 1'],
        importanceScore: 0.8,
        message: 'Success',
        performanceMet: true,
        timestamp: '2024-01-01T12:00:00Z'
      },
      {
        success: true,
        sessionId: 'session-2',
        compressionLevel: 'aggressive',
        originalTokens: 2000,
        compressedTokens: 600,
        compressionRatio: 0.7,
        tokensSaved: 1400,
        compressionTimeSeconds: 3.5,
        summary: 'Summary 2',
        keyInsights: ['Insight 2'],
        decisionsMade: ['Decision 2'],
        patternsIdentified: ['Pattern 2'],
        importanceScore: 0.9,
        message: 'Success',
        performanceMet: true,
        timestamp: '2024-01-01T12:05:00Z'
      }
    ]

    beforeEach(() => {
      service['compressionHistory'] = [...mockCompressionResults]
    })

    it('should calculate metrics correctly', () => {
      const metrics = service.getCompressionMetrics()

      expect(metrics.totalCompressions).toBe(2)
      expect(metrics.averageCompressionRatio).toBe(0.65) // (0.6 + 0.7) / 2
      expect(metrics.totalTokensSaved).toBe(2000) // 600 + 1400
      expect(metrics.averageCompressionTime).toBe(2.75) // (2.0 + 3.5) / 2
      expect(metrics.successRate).toBe(1.0) // 2/2
      expect(metrics.lastCompression).toEqual(mockCompressionResults[1])
    })

    it('should return compression history in reverse order', () => {
      const history = service.getCompressionHistory()

      expect(history.results).toHaveLength(2)
      expect(history.results[0]).toEqual(mockCompressionResults[1]) // Most recent first
      expect(history.results[1]).toEqual(mockCompressionResults[0])
      expect(history.lastUpdated).toBeDefined()
    })

    it('should handle empty history', () => {
      service['compressionHistory'] = []

      const metrics = service.getCompressionMetrics()
      expect(metrics.totalCompressions).toBe(0)
      expect(metrics.averageCompressionRatio).toBe(0)
      expect(metrics.totalTokensSaved).toBe(0)
      expect(metrics.averageCompressionTime).toBe(0)
      expect(metrics.successRate).toBe(0)
      expect(metrics.lastCompression).toBeUndefined()
    })

    it('should limit history to 50 entries', () => {
      const manyResults = Array.from({ length: 60 }, (_, i) => ({
        ...mockCompressionResults[0],
        sessionId: `session-${i}`,
        timestamp: `2024-01-01T${String(i).padStart(2, '0')}:00:00Z`
      }))

      service['compressionHistory'] = []
      manyResults.forEach(result => service['addToHistory'](result))

      expect(service['compressionHistory']).toHaveLength(50)
    })

    it('should clear history', () => {
      service['clearCache'] = vi.fn()
      service['emit'] = vi.fn()

      service.clearHistory()

      expect(service['compressionHistory']).toHaveLength(0)
      expect(service['clearCache']).toHaveBeenCalledWith('compression-history')
      expect(service['emit']).toHaveBeenCalledWith('history-cleared')
    })
  })

  describe('Recommended Settings', () => {
    it('should return default settings without context info', () => {
      const settings = service.getRecommendedSettings()

      expect(settings).toEqual({
        level: 'standard',
        preserveDecisions: true,
        preservePatterns: true
      })
    })

    it('should recommend aggressive compression for large content', () => {
      const settings = service.getRecommendedSettings({
        tokenCount: 15000
      })

      expect(settings.level).toBe('aggressive')
      expect(settings.targetTokens).toBe(Math.floor(15000 * 0.3))
    })

    it('should recommend standard compression for medium content', () => {
      const settings = service.getRecommendedSettings({
        tokenCount: 7000
      })

      expect(settings.level).toBe('standard')
      expect(settings.targetTokens).toBe(Math.floor(7000 * 0.5))
    })

    it('should recommend light compression for small content', () => {
      const settings = service.getRecommendedSettings({
        tokenCount: 3000
      })

      expect(settings.level).toBe('light')
      expect(settings.targetTokens).toBeUndefined()
    })

    it('should adjust settings based on session type', () => {
      const bugFixSettings = service.getRecommendedSettings({
        sessionType: 'bug_fix'
      })
      expect(bugFixSettings.preserveDecisions).toBe(true)

      const researchSettings = service.getRecommendedSettings({
        sessionType: 'research'
      })
      expect(researchSettings.preservePatterns).toBe(true)
    })

    it('should adjust settings based on priority', () => {
      const speedSettings = service.getRecommendedSettings({
        priority: 'speed'
      })
      expect(speedSettings.level).toBe('light')

      const aggressiveSettings = service.getRecommendedSettings({
        priority: 'aggressive'
      })
      expect(aggressiveSettings.level).toBe('aggressive')
      expect(aggressiveSettings.preserveDecisions).toBe(false)
      expect(aggressiveSettings.preservePatterns).toBe(false)
    })
  })

  describe('Monitoring', () => {
    it('should start monitoring compression updates', () => {
      service.startMonitoring()

      expect(mockWebSocket.sendMessage).toHaveBeenCalledWith({
        type: 'subscribe-compression-updates',
        data: {
          mobile_optimized: true,
          timestamp: expect.any(String)
        }
      })
    })

    it('should not start monitoring if already monitoring', () => {
      service['isMonitoring'] = true
      service.startMonitoring()

      expect(mockWebSocket.sendMessage).not.toHaveBeenCalled()
    })

    it('should stop monitoring compression updates', () => {
      service['isMonitoring'] = true
      service.stopMonitoring()

      expect(mockWebSocket.sendMessage).toHaveBeenCalledWith({
        type: 'unsubscribe-compression-updates',
        data: {
          timestamp: expect.any(String)
        }
      })
    })

    it('should not stop monitoring if not currently monitoring', () => {
      service['isMonitoring'] = false
      service.stopMonitoring()

      expect(mockWebSocket.sendMessage).not.toHaveBeenCalled()
    })
  })

  describe('WebSocket Event Handling', () => {
    it('should handle compression progress events', () => {
      service['emit'] = vi.fn()
      const progress: CompressionProgress = {
        sessionId: 'test-session',
        stage: 'compressing',
        progress: 75,
        currentStep: 'Optimizing content'
      }

      service['handleCompressionProgress'](progress)

      expect(service['activeCompressions'].get('test-session')).toEqual(progress)
      expect(service['emit']).toHaveBeenCalledWith('compression-progress', progress)
    })

    it('should handle compression completed events', () => {
      service['addToHistory'] = vi.fn()
      service['updateCache'] = vi.fn()
      service['emit'] = vi.fn()

      const result: CompressionResult = {
        success: true,
        sessionId: 'test-session',
        compressionLevel: 'standard',
        originalTokens: 1000,
        compressedTokens: 400,
        compressionRatio: 0.6,
        tokensSaved: 600,
        compressionTimeSeconds: 2.5,
        summary: 'Completed summary',
        keyInsights: [],
        decisionsMade: [],
        patternsIdentified: [],
        importanceScore: 0.7,
        message: 'Success',
        performanceMet: true,
        timestamp: '2024-01-01T12:00:00Z'
      }

      service['activeCompressions'].set('test-session', {
        sessionId: 'test-session',
        stage: 'compressing',
        progress: 50,
        currentStep: 'In progress'
      })

      service['handleCompressionCompleted'](result)

      expect(service['activeCompressions'].has('test-session')).toBe(false)
      expect(service['addToHistory']).toHaveBeenCalledWith(result)
      expect(service['updateCache']).toHaveBeenCalledWith('latest-compression', result)
      expect(service['emit']).toHaveBeenCalledWith('compression-completed', result)
    })

    it('should handle compression error events', () => {
      service['emit'] = vi.fn()
      const errorData = {
        sessionId: 'test-session',
        error: 'Compression failed'
      }

      service['handleCompressionError'](errorData)

      const errorProgress = service['activeCompressions'].get('test-session')
      expect(errorProgress).toEqual({
        sessionId: 'test-session',
        stage: 'error',
        progress: 0,
        currentStep: 'Compression failed',
        error: 'Compression failed'
      })
      expect(service['emit']).toHaveBeenCalledWith('compression-error', errorData)
    })
  })

  describe('Cancellation', () => {
    it('should cancel compression successfully', async () => {
      service['makeRequest'] = vi.fn().mockResolvedValue({ success: true })
      service['activeCompressions'].set('test-session', {
        sessionId: 'test-session',
        stage: 'compressing',
        progress: 50,
        currentStep: 'In progress'
      })

      const result = await service.cancelCompression('test-session')

      expect(result).toBe(true)
      expect(service['activeCompressions'].has('test-session')).toBe(false)
      expect(service['makeRequest']).toHaveBeenCalledWith('POST', '/api/hive/commands/cancel', {
        command: 'compact',
        sessionId: 'test-session'
      })
    })

    it('should handle cancellation failure', async () => {
      service['makeRequest'] = vi.fn().mockRejectedValue(new Error('Cancellation failed'))

      const result = await service.cancelCompression('test-session')

      expect(result).toBe(false)
    })
  })

  describe('Data Export', () => {
    beforeEach(() => {
      service['compressionHistory'] = [
        {
          success: true,
          sessionId: 'session-1',
          compressionLevel: 'standard',
          originalTokens: 1000,
          compressedTokens: 400,
          compressionRatio: 0.6,
          tokensSaved: 600,
          compressionTimeSeconds: 2.0,
          summary: 'Test summary',
          keyInsights: [],
          decisionsMade: [],
          patternsIdentified: [],
          importanceScore: 0.8,
          message: 'Success',
          performanceMet: true,
          timestamp: '2024-01-01T12:00:00Z'
        }
      ]
    })

    it('should export compression data', () => {
      const exportData = service.exportCompressionData()

      expect(exportData.history).toEqual(service['compressionHistory'])
      expect(exportData.metrics).toBeDefined()
      expect(exportData.exportedAt).toBeDefined()
      expect(new Date(exportData.exportedAt)).toBeInstanceOf(Date)
    })
  })
})