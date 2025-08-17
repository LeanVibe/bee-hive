/**
 * Context Compression Dashboard Component
 * 
 * Comprehensive dashboard for context compression management with:
 * - Compression history and analytics
 * - Real-time progress monitoring
 * - Mobile-optimized responsive design
 * - Project Index integration
 */

import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'
import { getContextCompressionService, type CompressionHistory, type CompressionResult, type CompressionMetrics } from '../../services/context-compression'
import './CompressionProgress'
import './CompressionControls'

@customElement('compression-dashboard')
export class CompressionDashboardComponent extends LitElement {
  static styles = css`
    :host {
      display: block;
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
    }

    .dashboard-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 2px solid var(--border-color, #e0e0e0);
    }

    .dashboard-title {
      font-size: 24px;
      font-weight: 700;
      color: var(--text-primary, #1a1a1a);
      margin: 0;
    }

    .dashboard-subtitle {
      font-size: 14px;
      color: var(--text-secondary, #666);
      margin-top: 4px;
    }

    .refresh-button {
      padding: 10px 16px;
      background: transparent;
      border: 2px solid var(--primary, #6366f1);
      color: var(--primary, #6366f1);
      border-radius: 8px;
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .refresh-button:hover {
      background: var(--primary, #6366f1);
      color: white;
    }

    .dashboard-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-bottom: 24px;
    }

    .dashboard-section {
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    .metrics-overview {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }

    .metric-card {
      background: var(--surface-color, #ffffff);
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
      border: 1px solid var(--border-color, #e0e0e0);
      text-align: center;
    }

    .metric-value {
      font-size: 28px;
      font-weight: 700;
      color: var(--primary, #6366f1);
      margin-bottom: 8px;
    }

    .metric-label {
      font-size: 14px;
      color: var(--text-secondary, #666);
      font-weight: 500;
    }

    .metric-trend {
      font-size: 12px;
      margin-top: 8px;
      padding: 4px 8px;
      border-radius: 4px;
      font-weight: 500;
    }

    .trend-positive {
      background: #e8f5e8;
      color: #2e7d32;
    }

    .trend-negative {
      background: #ffebee;
      color: #c62828;
    }

    .trend-neutral {
      background: #f5f5f5;
      color: #666;
    }

    .history-section {
      background: var(--surface-color, #ffffff);
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
      border: 1px solid var(--border-color, #e0e0e0);
    }

    .section-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }

    .section-title {
      font-size: 18px;
      font-weight: 600;
      color: var(--text-primary, #1a1a1a);
    }

    .clear-history-btn {
      padding: 6px 12px;
      background: transparent;
      border: 1px solid var(--error, #dc2626);
      color: var(--error, #dc2626);
      border-radius: 6px;
      font-size: 12px;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .clear-history-btn:hover {
      background: var(--error, #dc2626);
      color: white;
    }

    .history-list {
      display: flex;
      flex-direction: column;
      gap: 12px;
      max-height: 400px;
      overflow-y: auto;
    }

    .history-item {
      padding: 16px;
      background: var(--surface-variant, #f8f9fa);
      border-radius: 8px;
      border: 1px solid var(--border-color, #e0e0e0);
      transition: all 0.2s ease;
    }

    .history-item:hover {
      background: #f0f9ff;
      border-color: #0ea5e9;
    }

    .history-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }

    .history-timestamp {
      font-size: 12px;
      color: var(--text-secondary, #666);
      font-weight: 500;
    }

    .history-status {
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .status-success {
      background: #e8f5e8;
      color: #2e7d32;
    }

    .status-error {
      background: #ffebee;
      color: #c62828;
    }

    .history-details {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 8px;
    }

    .history-metric {
      text-align: center;
    }

    .history-metric-value {
      font-size: 14px;
      font-weight: 600;
      color: var(--text-primary, #1a1a1a);
    }

    .history-metric-label {
      font-size: 10px;
      color: var(--text-secondary, #666);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .history-summary {
      font-size: 13px;
      color: var(--text-secondary, #666);
      line-height: 1.4;
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px solid var(--border-color, #e0e0e0);
    }

    .empty-state {
      text-align: center;
      padding: 40px 20px;
      color: var(--text-secondary, #666);
    }

    .empty-icon {
      font-size: 48px;
      margin-bottom: 16px;
      opacity: 0.5;
    }

    .empty-title {
      font-size: 16px;
      font-weight: 600;
      margin-bottom: 8px;
    }

    .empty-description {
      font-size: 14px;
      line-height: 1.5;
    }

    .loading-state {
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 40px;
      gap: 12px;
      color: var(--text-secondary, #666);
    }

    .spinner {
      width: 20px;
      height: 20px;
      border: 2px solid #e0e0e0;
      border-top: 2px solid var(--primary, #6366f1);
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }

    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    /* Mobile optimizations */
    @media (max-width: 768px) {
      :host {
        padding: 16px 8px;
      }

      .dashboard-grid {
        grid-template-columns: 1fr;
        gap: 16px;
      }

      .dashboard-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 12px;
      }

      .refresh-button {
        align-self: stretch;
        justify-content: center;
      }

      .metrics-overview {
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
      }

      .metric-card {
        padding: 16px;
      }

      .metric-value {
        font-size: 24px;
      }

      .history-details {
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
      }

      .section-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 8px;
      }

      .clear-history-btn {
        align-self: stretch;
        text-align: center;
      }
    }
  `

  @property({ type: Object })
  contextInfo?: {
    tokenCount?: number
    sessionType?: string
    priority?: 'speed' | 'quality' | 'aggressive'
  }

  @state()
  private compressionHistory: CompressionHistory | null = null

  @state()
  private isLoading = true

  @state()
  private lastRefresh = new Date()

  private compressionService = getContextCompressionService()
  private refreshInterval?: number

  connectedCallback() {
    super.connectedCallback()
    this.loadCompressionData()
    
    // Subscribe to compression events
    this.compressionService.on('compression-completed', () => {
      this.loadCompressionData()
    })

    this.compressionService.on('history-cleared', () => {
      this.loadCompressionData()
    })

    // Auto-refresh every 30 seconds
    this.refreshInterval = window.setInterval(() => {
      this.loadCompressionData()
    }, 30000)
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval)
    }
  }

  private async loadCompressionData() {
    try {
      this.isLoading = true
      this.compressionHistory = this.compressionService.getCompressionHistory()
      this.lastRefresh = new Date()
    } catch (error) {
      console.error('Failed to load compression data:', error)
    } finally {
      this.isLoading = false
    }
  }

  private async handleRefresh() {
    await this.loadCompressionData()
  }

  private async handleClearHistory() {
    if (confirm('Are you sure you want to clear all compression history? This action cannot be undone.')) {
      this.compressionService.clearHistory()
    }
  }

  private formatTimestamp(isoString: string): string {
    const date = new Date(isoString)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / (1000 * 60))
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays < 7) return `${diffDays}d ago`
    
    return date.toLocaleDateString()
  }

  private formatTokens(tokens: number): string {
    if (tokens < 1000) return tokens.toString()
    if (tokens < 1000000) return `${(tokens / 1000).toFixed(1)}K`
    return `${(tokens / 1000000).toFixed(1)}M`
  }

  private formatTime(seconds: number): string {
    if (seconds < 60) return `${Math.round(seconds)}s`
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = Math.round(seconds % 60)
    return `${minutes}m ${remainingSeconds}s`
  }

  private formatPercentage(ratio: number): string {
    return `${(ratio * 100).toFixed(1)}%`
  }

  private getTrendClass(current: number, previous: number): string {
    if (Math.abs(current - previous) < 0.01) return 'trend-neutral'
    return current > previous ? 'trend-positive' : 'trend-negative'
  }

  private renderMetricsOverview() {
    if (!this.compressionHistory || this.isLoading) {
      return html`
        <div class="loading-state">
          <div class="spinner"></div>
          Loading compression metrics...
        </div>
      `
    }

    const metrics = this.compressionHistory.metrics
    const hasHistory = this.compressionHistory.results.length > 0

    if (!hasHistory) {
      return html`
        <div class="empty-state">
          <div class="empty-icon">📊</div>
          <div class="empty-title">No Compression Data</div>
          <div class="empty-description">
            Start your first compression to see analytics and metrics here.
          </div>
        </div>
      `
    }

    return html`
      <div class="metrics-overview">
        <div class="metric-card">
          <div class="metric-value">${metrics.totalCompressions}</div>
          <div class="metric-label">Total Compressions</div>
          <div class="metric-trend trend-neutral">All time</div>
        </div>

        <div class="metric-card">
          <div class="metric-value">${this.formatPercentage(metrics.averageCompressionRatio)}</div>
          <div class="metric-label">Average Reduction</div>
          <div class="metric-trend trend-positive">Efficiency</div>
        </div>

        <div class="metric-card">
          <div class="metric-value">${this.formatTokens(metrics.totalTokensSaved)}</div>
          <div class="metric-label">Tokens Saved</div>
          <div class="metric-trend trend-positive">Memory Freed</div>
        </div>

        <div class="metric-card">
          <div class="metric-value">${this.formatTime(metrics.averageCompressionTime)}</div>
          <div class="metric-label">Average Time</div>
          <div class="metric-trend trend-neutral">Performance</div>
        </div>

        <div class="metric-card">
          <div class="metric-value">${this.formatPercentage(metrics.successRate)}</div>
          <div class="metric-label">Success Rate</div>
          <div class="metric-trend ${metrics.successRate > 0.9 ? 'trend-positive' : 'trend-negative'}">
            Reliability
          </div>
        </div>
      </div>
    `
  }

  private renderHistoryList() {
    if (!this.compressionHistory || this.isLoading) return ''

    const results = this.compressionHistory.results

    if (results.length === 0) {
      return html`
        <div class="empty-state">
          <div class="empty-icon">📜</div>
          <div class="empty-title">No Compression History</div>
          <div class="empty-description">
            Completed compressions will appear here with detailed metrics and insights.
          </div>
        </div>
      `
    }

    return html`
      <div class="history-list">
        ${results.map(result => this.renderHistoryItem(result))}
      </div>
    `
  }

  private renderHistoryItem(result: CompressionResult) {
    return html`
      <div class="history-item">
        <div class="history-header">
          <div class="history-timestamp">
            ${this.formatTimestamp(result.timestamp)}
          </div>
          <div class="history-status ${result.success ? 'status-success' : 'status-error'}">
            ${result.success ? 'Success' : 'Error'}
          </div>
        </div>

        <div class="history-details">
          <div class="history-metric">
            <div class="history-metric-value">${this.formatTokens(result.originalTokens)}</div>
            <div class="history-metric-label">Original</div>
          </div>
          <div class="history-metric">
            <div class="history-metric-value">${this.formatTokens(result.compressedTokens)}</div>
            <div class="history-metric-label">Compressed</div>
          </div>
          <div class="history-metric">
            <div class="history-metric-value">${this.formatPercentage(result.compressionRatio)}</div>
            <div class="history-metric-label">Reduction</div>
          </div>
          <div class="history-metric">
            <div class="history-metric-value">${this.formatTime(result.compressionTimeSeconds)}</div>
            <div class="history-metric-label">Duration</div>
          </div>
        </div>

        ${result.summary ? html`
          <div class="history-summary">
            ${result.summary}
          </div>
        ` : ''}
      </div>
    `
  }

  render() {
    return html`
      <div class="dashboard-header">
        <div>
          <h1 class="dashboard-title">Context Compression</h1>
          <div class="dashboard-subtitle">
            Intelligent conversation context management and optimization
          </div>
        </div>
        <button class="refresh-button" @click=${this.handleRefresh}>
          <span>🔄</span>
          Refresh
        </button>
      </div>

      ${this.renderMetricsOverview()}

      <div class="dashboard-grid">
        <div class="dashboard-section">
          <compression-controls .contextInfo=${this.contextInfo}></compression-controls>
        </div>

        <div class="dashboard-section">
          <compression-progress></compression-progress>
        </div>
      </div>

      <div class="history-section">
        <div class="section-header">
          <div class="section-title">Compression History</div>
          ${this.compressionHistory?.results.length ? html`
            <button class="clear-history-btn" @click=${this.handleClearHistory}>
              Clear History
            </button>
          ` : ''}
        </div>
        ${this.renderHistoryList()}
      </div>
    `
  }
}