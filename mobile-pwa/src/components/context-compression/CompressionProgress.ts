/**
 * Real-time Context Compression Progress Component
 * 
 * Provides visual feedback for active compression operations with:
 * - Animated progress indicators
 * - Real-time status updates via WebSocket
 * - Mobile-optimized responsive design
 * - Compression metrics display
 */

import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'
import { getContextCompressionService, type CompressionProgress, type CompressionResult } from '../../services/context-compression'

@customElement('compression-progress')
export class CompressionProgressComponent extends LitElement {
  static styles = css`
    :host {
      display: block;
      background: var(--surface-color, #ffffff);
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
      border: 1px solid var(--border-color, #e0e0e0);
    }

    .progress-container {
      width: 100%;
    }

    .progress-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }

    .progress-title {
      font-size: 16px;
      font-weight: 600;
      color: var(--text-primary, #1a1a1a);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .progress-status {
      font-size: 12px;
      padding: 4px 8px;
      border-radius: 6px;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .status-analyzing { background: #e3f2fd; color: #1976d2; }
    .status-compressing { background: #f3e5f5; color: #7b1fa2; }
    .status-optimizing { background: #fff3e0; color: #f57c00; }
    .status-finalizing { background: #e8f5e8; color: #388e3c; }
    .status-completed { background: #e8f5e8; color: #388e3c; }
    .status-error { background: #ffebee; color: #d32f2f; }

    .progress-bar-container {
      position: relative;
      width: 100%;
      height: 8px;
      background: var(--surface-variant, #f5f5f5);
      border-radius: 4px;
      overflow: hidden;
      margin-bottom: 12px;
    }

    .progress-bar {
      height: 100%;
      background: linear-gradient(90deg, #6366f1, #8b5cf6);
      border-radius: 4px;
      transition: width 0.3s ease;
      position: relative;
    }

    .progress-bar.pulsing::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
      animation: progress-shimmer 1.5s infinite;
    }

    @keyframes progress-shimmer {
      0% { transform: translateX(-100%); }
      100% { transform: translateX(100%); }
    }

    .progress-details {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 16px;
    }

    .detail-item {
      background: var(--surface-variant, #f8f9fa);
      padding: 12px;
      border-radius: 8px;
      border: 1px solid var(--border-color, #e0e0e0);
    }

    .detail-label {
      font-size: 12px;
      color: var(--text-secondary, #666);
      font-weight: 500;
      margin-bottom: 4px;
    }

    .detail-value {
      font-size: 14px;
      color: var(--text-primary, #1a1a1a);
      font-weight: 600;
    }

    .current-step {
      margin-top: 12px;
      padding: 12px;
      background: var(--surface-variant, #f8f9fa);
      border-radius: 8px;
      border-left: 4px solid var(--primary, #6366f1);
    }

    .step-text {
      font-size: 14px;
      color: var(--text-primary, #1a1a1a);
      margin-bottom: 4px;
    }

    .time-remaining {
      font-size: 12px;
      color: var(--text-secondary, #666);
    }

    .cancel-button {
      margin-top: 16px;
      padding: 8px 16px;
      background: transparent;
      border: 1px solid var(--error, #dc2626);
      color: var(--error, #dc2626);
      border-radius: 6px;
      font-size: 14px;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .cancel-button:hover {
      background: var(--error, #dc2626);
      color: white;
    }

    .completion-summary {
      margin-top: 16px;
      padding: 16px;
      background: linear-gradient(135deg, #e8f5e8, #f1f8e9);
      border-radius: 8px;
      border: 1px solid #4caf50;
    }

    .summary-title {
      font-size: 14px;
      font-weight: 600;
      color: #2e7d32;
      margin-bottom: 8px;
    }

    .summary-metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
      gap: 8px;
    }

    .metric-item {
      text-align: center;
    }

    .metric-value {
      font-size: 16px;
      font-weight: 700;
      color: #1b5e20;
    }

    .metric-label {
      font-size: 11px;
      color: #388e3c;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .error-message {
      margin-top: 16px;
      padding: 12px;
      background: #ffebee;
      border: 1px solid #f44336;
      border-radius: 8px;
      color: #c62828;
      font-size: 14px;
    }

    .spinner {
      width: 16px;
      height: 16px;
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
        padding: 16px;
        margin: 8px;
      }

      .progress-details {
        grid-template-columns: 1fr;
        gap: 8px;
      }

      .summary-metrics {
        grid-template-columns: repeat(2, 1fr);
      }

      .detail-item {
        padding: 8px;
      }
    }
  `

  @property({ type: String })
  sessionId?: string

  @state()
  private progress: CompressionProgress | null = null

  @state()
  private completionResult: CompressionResult | null = null

  @state()
  private isVisible = false

  private compressionService = getContextCompressionService()
  private unsubscribeProgress?: () => void = undefined
  private unsubscribeCompletion?: () => void = undefined

  connectedCallback() {
    super.connectedCallback()
    
    // Subscribe to compression events
    const progressHandler = (progress: CompressionProgress) => {
      if (!this.sessionId || progress.sessionId === this.sessionId) {
        this.progress = progress
        this.isVisible = true
        this.requestUpdate()
      }
    }
    this.compressionService.on('compression-progress', progressHandler)
    this.unsubscribeProgress = () => this.compressionService.off('compression-progress', progressHandler)

    const completionHandler = (result: CompressionResult) => {
      if (!this.sessionId || result.sessionId === this.sessionId) {
        this.completionResult = result
        this.progress = null
        this.requestUpdate()
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
          this.isVisible = false
          this.requestUpdate()
        }, 5000)
      }
    }
    this.compressionService.on('compression-completed', completionHandler)
    this.unsubscribeCompletion = () => this.compressionService.off('compression-completed', completionHandler)

    // Check for existing progress
    const existingProgress = this.compressionService.getCompressionProgress(this.sessionId)
    if (existingProgress) {
      this.progress = existingProgress
      this.isVisible = true
    }
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    this.unsubscribeProgress?.()
    this.unsubscribeCompletion?.()
  }

  private async handleCancelCompression() {
    if (this.progress) {
      const success = await this.compressionService.cancelCompression(this.sessionId)
      if (success) {
        this.progress = null
        this.isVisible = false
        this.requestUpdate()
      }
    }
  }

  private formatTime(seconds: number): string {
    if (seconds < 60) {
      return `${Math.round(seconds)}s`
    }
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = Math.round(seconds % 60)
    return `${minutes}m ${remainingSeconds}s`
  }

  private formatTokens(tokens: number): string {
    if (tokens < 1000) return tokens.toString()
    if (tokens < 1000000) return `${(tokens / 1000).toFixed(1)}K`
    return `${(tokens / 1000000).toFixed(1)}M`
  }

  private getProgressPercentage(): number {
    return this.progress?.progress || 0
  }

  private renderProgressIndicator() {
    if (!this.progress) return ''

    const percentage = this.getProgressPercentage()
    const isActive = this.progress.stage !== 'completed' && this.progress.stage !== 'error'

    return html`
      <div class="progress-bar-container">
        <div 
          class="progress-bar ${isActive ? 'pulsing' : ''}"
          style="width: ${percentage}%"
        ></div>
      </div>
    `
  }

  private renderProgressDetails() {
    if (!this.progress) return ''

    return html`
      <div class="progress-details">
        <div class="detail-item">
          <div class="detail-label">Progress</div>
          <div class="detail-value">${this.getProgressPercentage()}%</div>
        </div>
        <div class="detail-item">
          <div class="detail-label">Stage</div>
          <div class="detail-value">${this.progress.stage}</div>
        </div>
        ${this.progress.tokensProcessed && this.progress.totalTokens ? html`
          <div class="detail-item">
            <div class="detail-label">Tokens</div>
            <div class="detail-value">
              ${this.formatTokens(this.progress.tokensProcessed)} / ${this.formatTokens(this.progress.totalTokens)}
            </div>
          </div>
        ` : ''}
        ${this.progress.estimatedTimeRemaining ? html`
          <div class="detail-item">
            <div class="detail-label">ETA</div>
            <div class="detail-value">${this.formatTime(this.progress.estimatedTimeRemaining)}</div>
          </div>
        ` : ''}
      </div>
    `
  }

  private renderCurrentStep() {
    if (!this.progress?.currentStep) return ''

    return html`
      <div class="current-step">
        <div class="step-text">${this.progress.currentStep}</div>
        ${this.progress.estimatedTimeRemaining ? html`
          <div class="time-remaining">
            ~${this.formatTime(this.progress.estimatedTimeRemaining)} remaining
          </div>
        ` : ''}
      </div>
    `
  }

  private renderError() {
    if (!this.progress?.error) return ''

    return html`
      <div class="error-message">
        <strong>Compression Error:</strong> ${this.progress.error}
      </div>
    `
  }

  private renderCompletionSummary() {
    if (!this.completionResult) return ''

    const compressionPercentage = (this.completionResult.compressionRatio * 100).toFixed(1)

    return html`
      <div class="completion-summary">
        <div class="summary-title">Compression Completed Successfully!</div>
        <div class="summary-metrics">
          <div class="metric-item">
            <div class="metric-value">${compressionPercentage}%</div>
            <div class="metric-label">Reduced</div>
          </div>
          <div class="metric-item">
            <div class="metric-value">${this.formatTokens(this.completionResult.tokensSaved)}</div>
            <div class="metric-label">Tokens Saved</div>
          </div>
          <div class="metric-item">
            <div class="metric-value">${this.formatTime(this.completionResult.compressionTimeSeconds)}</div>
            <div class="metric-label">Time Taken</div>
          </div>
          <div class="metric-item">
            <div class="metric-value">${this.completionResult.keyInsights.length}</div>
            <div class="metric-label">Key Insights</div>
          </div>
        </div>
      </div>
    `
  }

  render() {
    if (!this.isVisible && !this.progress && !this.completionResult) {
      return html``
    }

    const isActive = this.progress && this.progress.stage !== 'completed' && this.progress.stage !== 'error'

    return html`
      <div class="progress-container">
        <div class="progress-header">
          <div class="progress-title">
            ${isActive ? html`<div class="spinner"></div>` : ''}
            Context Compression
          </div>
          ${this.progress ? html`
            <div class="progress-status status-${this.progress.stage}">
              ${this.progress.stage}
            </div>
          ` : ''}
        </div>

        ${this.renderProgressIndicator()}
        ${this.renderProgressDetails()}
        ${this.renderCurrentStep()}
        ${this.renderError()}
        ${this.renderCompletionSummary()}

        ${isActive ? html`
          <button class="cancel-button" @click=${this.handleCancelCompression}>
            Cancel Compression
          </button>
        ` : ''}
      </div>
    `
  }
}