/**
 * Mobile Context Compression Widget
 * 
 * Compact, mobile-optimized widget for quick compression access with:
 * - Touch-friendly quick actions
 * - Real-time compression status
 * - Smart compression suggestions
 * - Swipe gestures for mobile interaction
 */

import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'
import { getContextCompressionService, type CompressionProgress, type CompressionMetrics } from '../../services/context-compression'

@customElement('compression-widget')
export class CompressionWidgetComponent extends LitElement {
  static styles = css`
    :host {
      display: block;
      position: relative;
      background: var(--surface-color, #ffffff);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
      border: 1px solid var(--border-color, #e0e0e0);
      user-select: none;
      touch-action: manipulation;
    }

    .widget-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }

    .widget-title {
      font-size: 16px;
      font-weight: 600;
      color: var(--text-primary, #1a1a1a);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .status-indicator {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--success, #22c55e);
      animation: pulse 2s infinite;
    }

    .status-indicator.active {
      background: var(--warning, #f59e0b);
      animation: pulse 1s infinite;
    }

    .status-indicator.error {
      background: var(--error, #ef4444);
      animation: none;
    }

    @keyframes pulse {
      0% { opacity: 1; }
      50% { opacity: 0.5; }
      100% { opacity: 1; }
    }

    .widget-menu {
      position: relative;
    }

    .menu-button {
      width: 32px;
      height: 32px;
      border: none;
      background: transparent;
      border-radius: 50%;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--text-secondary, #666);
      transition: all 0.2s ease;
    }

    .menu-button:hover {
      background: var(--surface-variant, #f5f5f5);
      color: var(--text-primary, #1a1a1a);
    }

    .quick-stats {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-bottom: 16px;
    }

    .stat-item {
      text-align: center;
      padding: 8px;
      background: var(--surface-variant, #f8f9fa);
      border-radius: 8px;
      border: 1px solid var(--border-color, #e0e0e0);
    }

    .stat-value {
      font-size: 16px;
      font-weight: 700;
      color: var(--primary, #6366f1);
      margin-bottom: 2px;
    }

    .stat-label {
      font-size: 10px;
      color: var(--text-secondary, #666);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .progress-section {
      margin-bottom: 16px;
    }

    .progress-bar-mini {
      width: 100%;
      height: 6px;
      background: var(--surface-variant, #f5f5f5);
      border-radius: 3px;
      overflow: hidden;
      margin-bottom: 8px;
    }

    .progress-fill {
      height: 100%;
      background: linear-gradient(90deg, #6366f1, #8b5cf6);
      border-radius: 3px;
      transition: width 0.3s ease;
      position: relative;
    }

    .progress-fill.pulsing::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);
      animation: shimmer 1.5s infinite;
    }

    @keyframes shimmer {
      0% { transform: translateX(-100%); }
      100% { transform: translateX(100%); }
    }

    .progress-text {
      font-size: 12px;
      color: var(--text-secondary, #666);
      text-align: center;
    }

    .quick-actions {
      display: flex;
      gap: 8px;
    }

    .action-button {
      flex: 1;
      padding: 12px 8px;
      border: none;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      text-align: center;
      min-height: 44px; /* iOS recommended touch target */
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 2px;
    }

    .btn-primary {
      background: var(--primary, #6366f1);
      color: white;
    }

    .btn-primary:hover {
      background: #5855eb;
      transform: translateY(-1px);
    }

    .btn-primary:disabled {
      background: var(--surface-variant, #e0e0e0);
      color: var(--text-secondary, #999);
      cursor: not-allowed;
      transform: none;
    }

    .btn-secondary {
      background: var(--surface-variant, #f5f5f5);
      color: var(--text-secondary, #666);
      border: 1px solid var(--border-color, #e0e0e0);
    }

    .btn-secondary:hover {
      background: #e5e7eb;
      border-color: var(--text-secondary, #666);
    }

    .action-icon {
      font-size: 14px;
    }

    .action-text {
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .compression-suggestion {
      margin-top: 12px;
      padding: 8px;
      background: linear-gradient(135deg, #fefce8, #fef3c7);
      border: 1px solid #f59e0b;
      border-radius: 8px;
      font-size: 11px;
      color: #92400e;
      line-height: 1.3;
    }

    .suggestion-title {
      font-weight: 600;
      margin-bottom: 2px;
    }

    .expandable-content {
      max-height: 0;
      overflow: hidden;
      transition: max-height 0.3s ease;
    }

    .expandable-content.expanded {
      max-height: 200px;
    }

    .swipe-indicator {
      position: absolute;
      top: 50%;
      right: 8px;
      transform: translateY(-50%);
      color: var(--text-secondary, #999);
      font-size: 12px;
      opacity: 0.5;
    }

    /* Mobile-specific optimizations */
    @media (max-width: 768px) {
      :host {
        padding: 12px;
        margin: 8px;
        border-radius: 12px;
      }

      .quick-stats {
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
      }

      .stat-item {
        padding: 6px;
      }

      .stat-value {
        font-size: 14px;
      }

      .quick-actions {
        gap: 6px;
      }

      .action-button {
        padding: 10px 6px;
        min-height: 40px;
      }
    }

    /* Touch feedback */
    @media (hover: none) {
      .action-button:active {
        transform: scale(0.98);
      }

      .menu-button:active {
        background: var(--surface-variant, #f5f5f5);
      }
    }
  `

  @property({ type: Boolean })
  expanded = false

  @state()
  private progress: CompressionProgress | null = null

  @state()
  private metrics: CompressionMetrics | null = null

  @state()
  private isCompressing = false

  @state()
  private suggestion: string | null = null

  private compressionService = getContextCompressionService()
  private unsubscribeProgress?: () => void = undefined
  private unsubscribeCompletion?: () => void = undefined

  connectedCallback() {
    super.connectedCallback()
    
    // Load initial data
    this.loadMetrics()
    this.checkCompressionProgress()
    this.generateSuggestion()

    // Subscribe to compression events
    const progressHandler = (progress: CompressionProgress) => {
      this.progress = progress
      this.isCompressing = progress.stage !== 'completed' && progress.stage !== 'error'
      this.requestUpdate()
    }
    this.compressionService.on('compression-progress', progressHandler)
    this.unsubscribeProgress = () => this.compressionService.off('compression-progress', progressHandler)

    const completionHandler = () => {
      this.loadMetrics()
      this.isCompressing = false
      this.progress = null
      this.requestUpdate()
    }
    this.compressionService.on('compression-completed', completionHandler)
    this.unsubscribeCompletion = () => this.compressionService.off('compression-completed', completionHandler)
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    this.unsubscribeProgress?.()
    this.unsubscribeCompletion?.()
  }

  private loadMetrics() {
    this.metrics = this.compressionService.getCompressionMetrics()
  }

  private checkCompressionProgress() {
    this.progress = this.compressionService.getCompressionProgress()
    this.isCompressing = this.progress ? 
      (this.progress.stage !== 'completed' && this.progress.stage !== 'error') : false
  }

  private generateSuggestion() {
    // Generate contextual compression suggestion
    if (!this.metrics || this.metrics.totalCompressions === 0) {
      this.suggestion = 'Try your first compression to optimize memory usage'
      return
    }

    if (this.metrics.successRate < 0.8) {
      this.suggestion = 'Consider using lighter compression levels for better reliability'
    } else if (this.metrics.averageCompressionRatio < 0.3) {
      this.suggestion = 'Increase compression level for better space savings'
    } else {
      this.suggestion = `Great efficiency! Average ${(this.metrics.averageCompressionRatio * 100).toFixed(0)}% reduction`
    }
  }

  private async handleQuickCompress() {
    if (this.isCompressing) return

    try {
      // Start compression with smart defaults
      await this.compressionService.compressContext({
        level: 'standard',
        preserveDecisions: true,
        preservePatterns: true
      })

      this.dispatchEvent(new CustomEvent('compression-started', {
        detail: { type: 'quick' },
        bubbles: true
      }))
    } catch (error) {
      console.error('Quick compression failed:', error)
    }
  }

  private handleOpenDashboard() {
    this.dispatchEvent(new CustomEvent('open-compression-dashboard', {
      bubbles: true
    }))
  }

  private handleToggleExpanded() {
    this.expanded = !this.expanded
  }

  private formatTokens(tokens: number): string {
    if (tokens < 1000) return tokens.toString()
    if (tokens < 1000000) return `${(tokens / 1000).toFixed(1)}K`
    return `${(tokens / 1000000).toFixed(1)}M`
  }

  private formatPercentage(ratio: number): string {
    return `${(ratio * 100).toFixed(0)}%`
  }

  private getStatusIndicatorClass(): string {
    if (this.progress?.stage === 'error') return 'error'
    if (this.isCompressing) return 'active'
    return ''
  }

  private getProgressPercentage(): number {
    return this.progress?.progress || 0
  }

  private renderProgress() {
    if (!this.progress || !this.isCompressing) return ''

    return html`
      <div class="progress-section">
        <div class="progress-bar-mini">
          <div 
            class="progress-fill ${this.isCompressing ? 'pulsing' : ''}"
            style="width: ${this.getProgressPercentage()}%"
          ></div>
        </div>
        <div class="progress-text">
          ${this.progress.currentStep || 'Processing...'} (${this.getProgressPercentage()}%)
        </div>
      </div>
    `
  }

  private renderQuickStats() {
    if (!this.metrics) {
      return html`
        <div class="quick-stats">
          <div class="stat-item">
            <div class="stat-value">-</div>
            <div class="stat-label">Total</div>
          </div>
          <div class="stat-item">
            <div class="stat-value">-</div>
            <div class="stat-label">Saved</div>
          </div>
          <div class="stat-item">
            <div class="stat-value">-</div>
            <div class="stat-label">Efficiency</div>
          </div>
        </div>
      `
    }

    return html`
      <div class="quick-stats">
        <div class="stat-item">
          <div class="stat-value">${this.metrics.totalCompressions}</div>
          <div class="stat-label">Total</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">${this.formatTokens(this.metrics.totalTokensSaved)}</div>
          <div class="stat-label">Saved</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">${this.formatPercentage(this.metrics.averageCompressionRatio)}</div>
          <div class="stat-label">Efficiency</div>
        </div>
      </div>
    `
  }

  render() {
    return html`
      <div class="widget-header">
        <div class="widget-title">
          <div class="status-indicator ${this.getStatusIndicatorClass()}"></div>
          Context Compression
        </div>
        <div class="widget-menu">
          <button class="menu-button" @click=${this.handleToggleExpanded}>
            ${this.expanded ? '−' : '+'}
          </button>
        </div>
      </div>

      ${this.renderQuickStats()}
      ${this.renderProgress()}

      <div class="quick-actions">
        <button 
          class="action-button btn-primary"
          @click=${this.handleQuickCompress}
          ?disabled=${this.isCompressing}
        >
          <div class="action-icon">🗜️</div>
          <div class="action-text">
            ${this.isCompressing ? 'Compressing' : 'Quick Compress'}
          </div>
        </button>
        
        <button 
          class="action-button btn-secondary"
          @click=${this.handleOpenDashboard}
        >
          <div class="action-icon">📊</div>
          <div class="action-text">Dashboard</div>
        </button>
      </div>

      <div class="expandable-content ${this.expanded ? 'expanded' : ''}">
        ${this.suggestion ? html`
          <div class="compression-suggestion">
            <div class="suggestion-title">💡 Smart Tip</div>
            ${this.suggestion}
          </div>
        ` : ''}
      </div>

      ${!this.expanded ? html`<div class="swipe-indicator">⋯</div>` : ''}
    `
  }
}