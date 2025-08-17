/**
 * Context Compression Controls Component
 * 
 * Provides user interface for initiating and configuring compression with:
 * - Smart compression level selection
 * - Real-time compression previews
 * - Mobile-optimized touch controls
 * - Guided compression recommendations
 */

import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'
import { getContextCompressionService, type CompressionOptions } from '../../services/context-compression'
import './CompressionHelp'

@customElement('compression-controls')
export class CompressionControlsComponent extends LitElement {
  static styles = css`
    :host {
      display: block;
      background: var(--surface-color, #ffffff);
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
      border: 1px solid var(--border-color, #e0e0e0);
    }

    .controls-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
    }

    .controls-title {
      font-size: 18px;
      font-weight: 600;
      color: var(--text-primary, #1a1a1a);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .smart-mode-toggle {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 14px;
      color: var(--text-secondary, #666);
    }

    .toggle-switch {
      position: relative;
      width: 44px;
      height: 24px;
      background: var(--surface-variant, #e0e0e0);
      border-radius: 12px;
      cursor: pointer;
      transition: background 0.3s ease;
    }

    .toggle-switch.active {
      background: var(--primary, #6366f1);
    }

    .toggle-handle {
      position: absolute;
      top: 2px;
      left: 2px;
      width: 20px;
      height: 20px;
      background: white;
      border-radius: 50%;
      transition: transform 0.3s ease;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    .toggle-switch.active .toggle-handle {
      transform: translateX(20px);
    }

    .session-selector {
      margin-bottom: 20px;
    }

    .form-group {
      margin-bottom: 16px;
    }

    .form-label {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 14px;
      font-weight: 500;
      color: var(--text-primary, #1a1a1a);
      margin-bottom: 8px;
    }

    .form-input {
      width: 100%;
      padding: 12px;
      border: 2px solid var(--border-color, #e0e0e0);
      border-radius: 8px;
      font-size: 14px;
      transition: border-color 0.2s ease;
      background: var(--surface-color, #ffffff);
      color: var(--text-primary, #1a1a1a);
      box-sizing: border-box;
    }

    .form-input:focus {
      outline: none;
      border-color: var(--primary, #6366f1);
    }

    .compression-levels {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-bottom: 20px;
    }

    .level-option {
      padding: 16px;
      border: 2px solid var(--border-color, #e0e0e0);
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s ease;
      text-align: center;
      background: var(--surface-color, #ffffff);
    }

    .level-option:hover {
      border-color: var(--primary, #6366f1);
      transform: translateY(-2px);
    }

    .level-option.selected {
      border-color: var(--primary, #6366f1);
      background: rgba(99, 102, 241, 0.05);
    }

    .level-name {
      font-size: 14px;
      font-weight: 600;
      color: var(--text-primary, #1a1a1a);
      margin-bottom: 4px;
    }

    .level-description {
      font-size: 12px;
      color: var(--text-secondary, #666);
      line-height: 1.3;
    }

    .level-percentage {
      font-size: 16px;
      font-weight: 700;
      color: var(--primary, #6366f1);
      margin-top: 8px;
    }

    .advanced-options {
      margin-bottom: 20px;
    }

    .advanced-toggle {
      display: flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
      margin-bottom: 16px;
      color: var(--text-secondary, #666);
      font-size: 14px;
    }

    .advanced-toggle compression-help {
      margin-left: auto;
    }

    .advanced-content {
      display: none;
      padding: 16px;
      background: var(--surface-variant, #f8f9fa);
      border-radius: 8px;
      border: 1px solid var(--border-color, #e0e0e0);
    }

    .advanced-content.visible {
      display: block;
    }

    .checkbox-group {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .checkbox-item {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .checkbox {
      width: 16px;
      height: 16px;
      accent-color: var(--primary, #6366f1);
    }

    .checkbox-label {
      font-size: 14px;
      color: var(--text-primary, #1a1a1a);
      cursor: pointer;
    }

    .target-tokens-group {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: end;
    }

    .tokens-input {
      min-width: 0;
    }

    .tokens-suggestion {
      padding: 8px 12px;
      background: var(--primary, #6366f1);
      color: white;
      border: none;
      border-radius: 6px;
      font-size: 12px;
      cursor: pointer;
      white-space: nowrap;
    }

    .compression-preview {
      margin-bottom: 20px;
      padding: 16px;
      background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
      border-radius: 8px;
      border: 1px solid #0ea5e9;
    }

    .preview-title {
      font-size: 14px;
      font-weight: 600;
      color: #0369a1;
      margin-bottom: 12px;
    }

    .preview-metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
      gap: 12px;
    }

    .preview-metric {
      text-align: center;
    }

    .preview-value {
      font-size: 16px;
      font-weight: 700;
      color: #0c4a6e;
    }

    .preview-label {
      font-size: 11px;
      color: #0369a1;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .action-buttons {
      display: flex;
      gap: 12px;
    }

    .btn {
      flex: 1;
      padding: 14px 20px;
      border: none;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      min-height: 48px; /* Better touch targets for mobile */
    }

    .btn-primary {
      background: var(--primary, #6366f1);
      color: white;
    }

    .btn-primary:hover {
      background: #5855eb;
      transform: translateY(-2px);
    }

    .btn-primary:disabled {
      background: var(--surface-variant, #e0e0e0);
      color: var(--text-secondary, #999);
      cursor: not-allowed;
      transform: none;
    }

    .btn-secondary {
      background: transparent;
      color: var(--text-secondary, #666);
      border: 2px solid var(--border-color, #e0e0e0);
    }

    .btn-secondary:hover {
      background: var(--surface-variant, #f5f5f5);
      border-color: var(--text-secondary, #666);
    }

    .recommendations {
      margin-bottom: 20px;
      padding: 16px;
      background: #fefce8;
      border: 1px solid #facc15;
      border-radius: 8px;
    }

    .recommendation-title {
      font-size: 14px;
      font-weight: 600;
      color: #a16207;
      margin-bottom: 8px;
    }

    .recommendation-text {
      font-size: 13px;
      color: #a16207;
      line-height: 1.4;
    }

    /* Mobile optimizations */
    @media (max-width: 768px) {
      :host {
        padding: 16px;
        margin: 8px;
      }

      .compression-levels {
        grid-template-columns: 1fr;
        gap: 8px;
      }

      .level-option {
        padding: 12px;
      }

      .preview-metrics {
        grid-template-columns: repeat(2, 1fr);
      }

      .action-buttons {
        flex-direction: column;
      }

      .target-tokens-group {
        grid-template-columns: 1fr;
        gap: 8px;
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
  private selectedLevel: 'light' | 'standard' | 'aggressive' = 'standard'

  @state()
  private sessionId = ''

  @state()
  private targetTokens: number | null = null

  @state()
  private preserveDecisions = true

  @state()
  private preservePatterns = true

  @state()
  private showAdvanced = false

  @state()
  private smartMode = true

  @state()
  private isCompressing = false

  private compressionService = getContextCompressionService()

  connectedCallback() {
    super.connectedCallback()
    
    // Load smart recommendations if context info is available
    if (this.contextInfo && this.smartMode) {
      this.loadSmartRecommendations()
    }
  }

  private loadSmartRecommendations() {
    if (!this.contextInfo) return

    const recommendations = this.compressionService.getRecommendedSettings(this.contextInfo)
    
    this.selectedLevel = recommendations.level || 'standard'
    this.targetTokens = recommendations.targetTokens || null
    this.preserveDecisions = recommendations.preserveDecisions ?? true
    this.preservePatterns = recommendations.preservePatterns ?? true

    this.requestUpdate()
  }

  private toggleSmartMode() {
    this.smartMode = !this.smartMode
    
    if (this.smartMode && this.contextInfo) {
      this.loadSmartRecommendations()
    }
  }

  private selectLevel(level: 'light' | 'standard' | 'aggressive') {
    this.selectedLevel = level
    
    // Auto-adjust target tokens based on level if in smart mode
    if (this.smartMode && this.contextInfo?.tokenCount) {
      const tokenCount = this.contextInfo.tokenCount
      switch (level) {
        case 'light':
          this.targetTokens = Math.floor(tokenCount * 0.8) // 20% reduction
          break
        case 'standard':
          this.targetTokens = Math.floor(tokenCount * 0.5) // 50% reduction
          break
        case 'aggressive':
          this.targetTokens = Math.floor(tokenCount * 0.3) // 70% reduction
          break
      }
    }
  }

  private toggleAdvanced() {
    this.showAdvanced = !this.showAdvanced
  }

  private updateSessionId(event: Event) {
    const input = event.target as HTMLInputElement
    this.sessionId = input.value
  }

  private updateTargetTokens(event: Event) {
    const input = event.target as HTMLInputElement
    this.targetTokens = input.value ? parseInt(input.value) : null
  }

  private applySuggestedTokens() {
    if (this.contextInfo?.tokenCount) {
      const multiplier = this.selectedLevel === 'light' ? 0.8 : 
                        this.selectedLevel === 'standard' ? 0.5 : 0.3
      this.targetTokens = Math.floor(this.contextInfo.tokenCount * multiplier)
    }
  }

  private async startCompression() {
    if (this.isCompressing) return

    this.isCompressing = true

    try {
      const options: CompressionOptions = {
        level: this.selectedLevel,
        preserveDecisions: this.preserveDecisions,
        preservePatterns: this.preservePatterns
      }

      if (this.sessionId.trim()) {
        options.sessionId = this.sessionId.trim()
      }

      if (this.targetTokens) {
        options.targetTokens = this.targetTokens
      }

      await this.compressionService.compressContext(options)
      
      // Emit event for parent components
      this.dispatchEvent(new CustomEvent('compression-started', {
        detail: options,
        bubbles: true
      }))

    } catch (error) {
      console.error('Failed to start compression:', error)
      
      this.dispatchEvent(new CustomEvent('compression-error', {
        detail: { error: error.message },
        bubbles: true
      }))
    } finally {
      this.isCompressing = false
    }
  }

  private resetToDefaults() {
    this.selectedLevel = 'standard'
    this.sessionId = ''
    this.targetTokens = null
    this.preserveDecisions = true
    this.preservePatterns = true
    this.showAdvanced = false
  }

  private getRecommendations(): string | null {
    if (!this.smartMode || !this.contextInfo) return null

    const { tokenCount, sessionType } = this.contextInfo

    if (tokenCount && tokenCount > 10000) {
      return `Large context detected (${this.formatTokens(tokenCount)} tokens). Aggressive compression recommended to reduce memory usage.`
    }

    if (sessionType?.includes('bug') || sessionType?.includes('error')) {
      return 'Error resolution session detected. Consider preserving decisions to maintain troubleshooting context.'
    }

    if (sessionType?.includes('research') || sessionType?.includes('learning')) {
      return 'Research session detected. Pattern preservation recommended to maintain learning insights.'
    }

    return null
  }

  private formatTokens(tokens: number): string {
    if (tokens < 1000) return tokens.toString()
    if (tokens < 1000000) return `${(tokens / 1000).toFixed(1)}K`
    return `${(tokens / 1000000).toFixed(1)}M`
  }

  private getEstimatedReduction(): number {
    switch (this.selectedLevel) {
      case 'light': return 0.2
      case 'standard': return 0.5
      case 'aggressive': return 0.7
      default: return 0.5
    }
  }

  private renderLevelOptions() {
    const levels = [
      {
        key: 'light' as const,
        name: 'Light',
        description: 'Minimal compression, preserves most context',
        percentage: '20%'
      },
      {
        key: 'standard' as const,
        name: 'Standard',
        description: 'Balanced compression with good preservation',
        percentage: '50%'
      },
      {
        key: 'aggressive' as const,
        name: 'Aggressive',
        description: 'Maximum compression for storage efficiency',
        percentage: '70%'
      }
    ]

    return html`
      <div class="compression-levels">
        ${levels.map(level => html`
          <div 
            class="level-option ${this.selectedLevel === level.key ? 'selected' : ''}"
            @click=${() => this.selectLevel(level.key)}
          >
            <div class="level-name">${level.name}</div>
            <div class="level-description">${level.description}</div>
            <div class="level-percentage">${level.percentage}</div>
          </div>
        `)}
      </div>
    `
  }

  private renderAdvancedOptions() {
    return html`
      <div class="advanced-options">
        <div class="advanced-toggle" @click=${this.toggleAdvanced}>
          <span>${this.showAdvanced ? '▼' : '▶'}</span>
          Advanced Options
          <compression-help topic="options" trigger="icon"></compression-help>
        </div>
        
        <div class="advanced-content ${this.showAdvanced ? 'visible' : ''}">
          <div class="form-group">
            <label class="form-label">Target Token Count</label>
            <div class="target-tokens-group">
              <input 
                type="number" 
                class="form-input tokens-input"
                placeholder="Optional target token count"
                .value=${this.targetTokens?.toString() || ''}
                @input=${this.updateTargetTokens}
              />
              ${this.contextInfo?.tokenCount ? html`
                <button class="tokens-suggestion" @click=${this.applySuggestedTokens}>
                  Suggest: ${this.formatTokens(Math.floor(this.contextInfo.tokenCount * (1 - this.getEstimatedReduction())))}
                </button>
              ` : ''}
            </div>
          </div>

          <div class="form-group">
            <div class="checkbox-group">
              <div class="checkbox-item">
                <input 
                  type="checkbox" 
                  id="preserve-decisions"
                  class="checkbox"
                  .checked=${this.preserveDecisions}
                  @change=${(e: Event) => this.preserveDecisions = (e.target as HTMLInputElement).checked}
                />
                <label for="preserve-decisions" class="checkbox-label">
                  Preserve decision points and conclusions
                </label>
              </div>
              
              <div class="checkbox-item">
                <input 
                  type="checkbox" 
                  id="preserve-patterns"
                  class="checkbox"
                  .checked=${this.preservePatterns}
                  @change=${(e: Event) => this.preservePatterns = (e.target as HTMLInputElement).checked}
                />
                <label for="preserve-patterns" class="checkbox-label">
                  Preserve learning patterns and insights
                </label>
              </div>
            </div>
          </div>
        </div>
      </div>
    `
  }

  private renderCompressionPreview() {
    if (!this.contextInfo?.tokenCount) return ''

    const originalTokens = this.contextInfo.tokenCount
    const estimatedReduction = this.getEstimatedReduction()
    const targetTokens = this.targetTokens || Math.floor(originalTokens * (1 - estimatedReduction))
    const savedTokens = originalTokens - targetTokens
    const savedPercentage = ((savedTokens / originalTokens) * 100).toFixed(1)

    return html`
      <div class="compression-preview">
        <div class="preview-title">Compression Preview</div>
        <div class="preview-metrics">
          <div class="preview-metric">
            <div class="preview-value">${this.formatTokens(originalTokens)}</div>
            <div class="preview-label">Original</div>
          </div>
          <div class="preview-metric">
            <div class="preview-value">${this.formatTokens(targetTokens)}</div>
            <div class="preview-label">After</div>
          </div>
          <div class="preview-metric">
            <div class="preview-value">${this.formatTokens(savedTokens)}</div>
            <div class="preview-label">Saved</div>
          </div>
          <div class="preview-metric">
            <div class="preview-value">${savedPercentage}%</div>
            <div class="preview-label">Reduction</div>
          </div>
        </div>
      </div>
    `
  }

  render() {
    const recommendations = this.getRecommendations()

    return html`
      <div class="controls-header">
        <div class="controls-title">
          Context Compression
          <compression-help topic="general" trigger="icon"></compression-help>
        </div>
        <div class="smart-mode-toggle">
          Smart Mode
          <div class="toggle-switch ${this.smartMode ? 'active' : ''}" @click=${this.toggleSmartMode}>
            <div class="toggle-handle"></div>
          </div>
        </div>
      </div>

      ${recommendations ? html`
        <div class="recommendations">
          <div class="recommendation-title">💡 Smart Recommendation</div>
          <div class="recommendation-text">${recommendations}</div>
        </div>
      ` : ''}

      <div class="session-selector">
        <div class="form-group">
          <label class="form-label">Session ID (Optional)</label>
          <input 
            type="text" 
            class="form-input"
            placeholder="Leave empty for current context"
            .value=${this.sessionId}
            @input=${this.updateSessionId}
          />
        </div>
      </div>

      <div class="form-group">
        <label class="form-label">
          Compression Level
          <compression-help topic="levels" trigger="icon"></compression-help>
        </label>
        ${this.renderLevelOptions()}
      </div>

      ${this.renderAdvancedOptions()}
      ${this.renderCompressionPreview()}

      <div class="action-buttons">
        <button 
          class="btn btn-secondary"
          @click=${this.resetToDefaults}
          ?disabled=${this.isCompressing}
        >
          Reset
        </button>
        <button 
          class="btn btn-primary"
          @click=${this.startCompression}
          ?disabled=${this.isCompressing}
        >
          ${this.isCompressing ? 'Compressing...' : 'Start Compression'}
        </button>
      </div>
    `
  }
}