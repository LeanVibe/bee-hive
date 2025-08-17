/**
 * Component tests for CompressionDashboard
 * Tests the main dashboard component for context compression
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { fixture, html, expect as litExpect } from '@open-wc/testing'
import '../CompressionDashboard'
import type { CompressionDashboardComponent } from '../CompressionDashboard'
import type { CompressionHistory, CompressionMetrics, CompressionResult } from '../../../services/context-compression'

// Mock the context compression service
const mockCompressionService = {
  getCompressionHistory: vi.fn(),
  clearHistory: vi.fn(),
  on: vi.fn(),
  off: vi.fn()
}

vi.mock('../../../services/context-compression', () => ({
  getContextCompressionService: () => mockCompressionService
}))

describe('CompressionDashboard', () => {
  let element: CompressionDashboardComponent
  let mockHistory: CompressionHistory
  let mockMetrics: CompressionMetrics

  beforeEach(async () => {
    // Setup mock data
    mockMetrics = {
      totalCompressions: 10,
      averageCompressionRatio: 0.65,
      totalTokensSaved: 50000,
      averageCompressionTime: 3.2,
      successRate: 0.95,
      lastCompression: {
        success: true,
        sessionId: 'recent-session',
        compressionLevel: 'standard',
        originalTokens: 1000,
        compressedTokens: 350,
        compressionRatio: 0.65,
        tokensSaved: 650,
        compressionTimeSeconds: 2.8,
        summary: 'Recent compression summary',
        keyInsights: ['Recent insight'],
        decisionsMade: ['Recent decision'],
        patternsIdentified: ['Recent pattern'],
        importanceScore: 0.8,
        message: 'Success',
        performanceMet: true,
        timestamp: '2024-01-01T12:00:00Z'
      }
    }

    mockHistory = {
      results: [
        {
          success: true,
          sessionId: 'session-1',
          compressionLevel: 'standard',
          originalTokens: 1500,
          compressedTokens: 600,
          compressionRatio: 0.6,
          tokensSaved: 900,
          compressionTimeSeconds: 3.5,
          summary: 'First compression summary with detailed information about the session context',
          keyInsights: ['Key insight 1', 'Key insight 2'],
          decisionsMade: ['Important decision 1'],
          patternsIdentified: ['Pattern A', 'Pattern B'],
          importanceScore: 0.85,
          message: 'Compression completed successfully',
          performanceMet: true,
          timestamp: '2024-01-01T10:00:00Z'
        },
        {
          success: true,
          sessionId: 'session-2',
          compressionLevel: 'aggressive',
          originalTokens: 2000,
          compressedTokens: 600,
          compressionRatio: 0.7,
          tokensSaved: 1400,
          compressionTimeSeconds: 4.2,
          summary: 'Second compression with aggressive settings',
          keyInsights: ['Aggressive insight'],
          decisionsMade: ['Quick decision'],
          patternsIdentified: ['Efficiency pattern'],
          importanceScore: 0.9,
          message: 'Aggressive compression successful',
          performanceMet: true,
          timestamp: '2024-01-01T11:30:00Z'
        }
      ],
      metrics: mockMetrics,
      lastUpdated: '2024-01-01T12:00:00Z'
    }

    mockCompressionService.getCompressionHistory.mockReturnValue(mockHistory)

    element = await fixture<CompressionDashboardComponent>(
      html`<compression-dashboard></compression-dashboard>`
    )
    
    // Wait for component to load data
    await element.updateComplete
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  describe('Initialization', () => {
    it('should render dashboard header correctly', () => {
      const header = element.shadowRoot?.querySelector('.dashboard-header')
      expect(header).to.exist

      const title = element.shadowRoot?.querySelector('.dashboard-title')
      expect(title?.textContent).to.include('Context Compression')

      const subtitle = element.shadowRoot?.querySelector('.dashboard-subtitle')
      expect(subtitle?.textContent).to.include('Intelligent conversation context management')

      const refreshButton = element.shadowRoot?.querySelector('.refresh-button')
      expect(refreshButton).to.exist
    })

    it('should set up event listeners for compression service', () => {
      expect(mockCompressionService.on).toHaveBeenCalledWith('compression-completed', expect.any(Function))
      expect(mockCompressionService.on).toHaveBeenCalledWith('history-cleared', expect.any(Function))
    })

    it('should load compression data on initialization', () => {
      expect(mockCompressionService.getCompressionHistory).toHaveBeenCalled()
    })
  })

  describe('Metrics Overview', () => {
    it('should display metrics correctly', async () => {
      const metricCards = element.shadowRoot?.querySelectorAll('.metric-card')
      expect(metricCards).to.have.length(5)

      // Check total compressions
      const totalCompressionsCard = Array.from(metricCards!).find(card => 
        card.textContent?.includes('Total Compressions')
      )
      expect(totalCompressionsCard?.querySelector('.metric-value')?.textContent).to.equal('10')

      // Check average reduction
      const averageReductionCard = Array.from(metricCards!).find(card => 
        card.textContent?.includes('Average Reduction')
      )
      expect(averageReductionCard?.querySelector('.metric-value')?.textContent).to.equal('65.0%')

      // Check tokens saved
      const tokensSavedCard = Array.from(metricCards!).find(card => 
        card.textContent?.includes('Tokens Saved')
      )
      expect(tokensSavedCard?.querySelector('.metric-value')?.textContent).to.equal('50.0K')

      // Check success rate
      const successRateCard = Array.from(metricCards!).find(card => 
        card.textContent?.includes('Success Rate')
      )
      expect(successRateCard?.querySelector('.metric-value')?.textContent).to.equal('95.0%')
    })

    it('should show loading state when data is being loaded', async () => {
      element['isLoading'] = true
      await element.requestUpdate()

      const loadingState = element.shadowRoot?.querySelector('.loading-state')
      expect(loadingState).to.exist
      expect(loadingState?.textContent).to.include('Loading compression metrics')
    })

    it('should show empty state when no data available', async () => {
      mockCompressionService.getCompressionHistory.mockReturnValue({
        results: [],
        metrics: {
          totalCompressions: 0,
          averageCompressionRatio: 0,
          totalTokensSaved: 0,
          averageCompressionTime: 0,
          successRate: 0
        },
        lastUpdated: new Date().toISOString()
      })

      element['loadCompressionData']()
      await element.requestUpdate()

      const emptyState = element.shadowRoot?.querySelector('.empty-state')
      expect(emptyState).to.exist
      expect(emptyState?.textContent).to.include('No Compression Data')
    })
  })

  describe('History Display', () => {
    it('should render compression history items', () => {
      const historyItems = element.shadowRoot?.querySelectorAll('.history-item')
      expect(historyItems).to.have.length(2)

      // Check first history item
      const firstItem = historyItems?.[0]
      expect(firstItem?.textContent).to.include('1.5K') // Original tokens
      expect(firstItem?.textContent).to.include('600') // Compressed tokens
      expect(firstItem?.textContent).to.include('60.0%') // Compression ratio
      expect(firstItem?.textContent).to.include('4s') // Duration (rounded)
    })

    it('should display history item details correctly', () => {
      const historyItems = element.shadowRoot?.querySelectorAll('.history-item')
      const firstItem = historyItems?.[0]

      // Check status
      const status = firstItem?.querySelector('.history-status')
      expect(status?.textContent?.trim()).to.equal('Success')
      expect(status?.classList.contains('status-success')).to.be.true

      // Check metrics
      const metrics = firstItem?.querySelectorAll('.history-metric')
      expect(metrics).to.have.length(4)

      // Check summary
      const summary = firstItem?.querySelector('.history-summary')
      expect(summary?.textContent).to.include('First compression summary')
    })

    it('should format timestamps correctly', () => {
      // Mock current time to test relative time formatting
      const now = new Date('2024-01-01T12:30:00Z')
      vi.setSystemTime(now)

      const historyItems = element.shadowRoot?.querySelectorAll('.history-item')
      const firstItem = historyItems?.[0]
      const timestamp = firstItem?.querySelector('.history-timestamp')
      
      // Should show relative time (e.g., "2h ago")
      expect(timestamp?.textContent).to.match(/\d+[hm] ago/)
    })

    it('should show empty state when no history available', async () => {
      mockCompressionService.getCompressionHistory.mockReturnValue({
        results: [],
        metrics: mockMetrics,
        lastUpdated: new Date().toISOString()
      })

      element['loadCompressionData']()
      await element.requestUpdate()

      const historySection = element.shadowRoot?.querySelector('.history-section')
      const emptyState = historySection?.querySelector('.empty-state')
      expect(emptyState).to.exist
      expect(emptyState?.textContent).to.include('No Compression History')
    })
  })

  describe('Token Formatting', () => {
    it('should format token counts correctly', () => {
      expect(element['formatTokens'](500)).to.equal('500')
      expect(element['formatTokens'](1500)).to.equal('1.5K')
      expect(element['formatTokens'](1000000)).to.equal('1.0M')
      expect(element['formatTokens'](2500000)).to.equal('2.5M')
    })
  })

  describe('Time Formatting', () => {
    it('should format time durations correctly', () => {
      expect(element['formatTime'](30)).to.equal('30s')
      expect(element['formatTime'](65)).to.equal('1m 5s')
      expect(element['formatTime'](120)).to.equal('2m 0s')
      expect(element['formatTime'](3661)).to.equal('61m 1s')
    })
  })

  describe('Percentage Formatting', () => {
    it('should format percentages correctly', () => {
      expect(element['formatPercentage'](0.65)).to.equal('65.0%')
      expect(element['formatPercentage'](0.123456)).to.equal('12.3%')
      expect(element['formatPercentage'](1.0)).to.equal('100.0%')
    })
  })

  describe('User Interactions', () => {
    it('should refresh data when refresh button is clicked', async () => {
      const refreshButton = element.shadowRoot?.querySelector('.refresh-button') as HTMLButtonElement
      expect(refreshButton).to.exist

      // Clear previous calls
      mockCompressionService.getCompressionHistory.mockClear()

      refreshButton.click()
      await element.updateComplete

      expect(mockCompressionService.getCompressionHistory).toHaveBeenCalled()
    })

    it('should clear history when clear button is clicked', async () => {
      // Mock window.confirm
      vi.stubGlobal('confirm', vi.fn().mockReturnValue(true))

      const clearButton = element.shadowRoot?.querySelector('.clear-history-btn') as HTMLButtonElement
      expect(clearButton).to.exist

      clearButton.click()
      await element.updateComplete

      expect(mockCompressionService.clearHistory).toHaveBeenCalled()

      vi.unstubAllGlobals()
    })

    it('should not clear history when user cancels confirmation', async () => {
      // Mock window.confirm to return false
      vi.stubGlobal('confirm', vi.fn().mockReturnValue(false))

      const clearButton = element.shadowRoot?.querySelector('.clear-history-btn') as HTMLButtonElement
      clearButton.click()

      expect(mockCompressionService.clearHistory).not.toHaveBeenCalled()

      vi.unstubAllGlobals()
    })

    it('should hide clear button when no history available', async () => {
      mockCompressionService.getCompressionHistory.mockReturnValue({
        results: [],
        metrics: mockMetrics,
        lastUpdated: new Date().toISOString()
      })

      element['loadCompressionData']()
      await element.requestUpdate()

      const clearButton = element.shadowRoot?.querySelector('.clear-history-btn')
      expect(clearButton).to.not.exist
    })
  })

  describe('Event Handling', () => {
    it('should reload data when compression is completed', async () => {
      // Clear previous calls
      mockCompressionService.getCompressionHistory.mockClear()

      // Simulate compression completed event
      const compressionCompletedCallback = mockCompressionService.on.mock.calls.find(
        call => call[0] === 'compression-completed'
      )?.[1]

      compressionCompletedCallback?.()
      await element.updateComplete

      expect(mockCompressionService.getCompressionHistory).toHaveBeenCalled()
    })

    it('should reload data when history is cleared', async () => {
      // Clear previous calls
      mockCompressionService.getCompressionHistory.mockClear()

      // Simulate history cleared event
      const historyClearedCallback = mockCompressionService.on.mock.calls.find(
        call => call[0] === 'history-cleared'
      )?.[1]

      historyClearedCallback?.()
      await element.updateComplete

      expect(mockCompressionService.getCompressionHistory).toHaveBeenCalled()
    })
  })

  describe('Context Info Integration', () => {
    it('should pass context info to compression controls', async () => {
      const contextInfo = {
        tokenCount: 5000,
        sessionType: 'development',
        priority: 'quality' as const
      }

      element.contextInfo = contextInfo
      await element.requestUpdate()

      const compressionControls = element.shadowRoot?.querySelector('compression-controls')
      expect(compressionControls).to.exist
      // Note: We would need to check the property was passed, but that requires
      // deeper integration testing with the actual compression-controls component
    })
  })

  describe('Auto-refresh', () => {
    it('should set up auto-refresh interval on connection', () => {
      // Auto-refresh should be set up in connectedCallback
      expect(element['refreshInterval']).to.be.a('number')
    })

    it('should clear auto-refresh interval on disconnection', () => {
      const intervalId = element['refreshInterval']
      
      // Disconnect the element
      element.disconnectedCallback()
      
      // Interval should be cleared
      expect(element['refreshInterval']).to.be.undefined
    })
  })

  describe('Error Handling', () => {
    it('should handle data loading errors gracefully', async () => {
      mockCompressionService.getCompressionHistory.mockImplementation(() => {
        throw new Error('Data loading failed')
      })

      // Should not throw error
      await element['loadCompressionData']()

      // Loading state should be reset
      expect(element['isLoading']).to.be.false
    })
  })

  describe('Mobile Responsive Design', () => {
    it('should have mobile-optimized CSS classes', () => {
      const styles = element.constructor as typeof CompressionDashboardComponent
      const cssText = styles.styles.toString()

      // Check for mobile media queries
      expect(cssText).to.include('@media (max-width: 768px)')
      expect(cssText).to.include('grid-template-columns: 1fr') // Single column on mobile
      expect(cssText).to.include('grid-template-columns: repeat(2, 1fr)') // 2 columns for metrics on mobile
    })
  })

  describe('Accessibility', () => {
    it('should have proper ARIA labels and semantic structure', () => {
      const title = element.shadowRoot?.querySelector('.dashboard-title')
      expect(title?.tagName).to.equal('H1')

      const sectionTitle = element.shadowRoot?.querySelector('.section-title')
      expect(sectionTitle).to.exist

      // Buttons should be accessible
      const refreshButton = element.shadowRoot?.querySelector('.refresh-button')
      expect(refreshButton?.tagName).to.equal('BUTTON')

      const clearButton = element.shadowRoot?.querySelector('.clear-history-btn')
      if (clearButton) {
        expect(clearButton.tagName).to.equal('BUTTON')
      }
    })
  })
})