/**
 * Context Compression Help Component
 * 
 * Provides comprehensive help, tooltips, and guided tour for compression features with:
 * - Interactive tooltips for all compression options
 * - Step-by-step guided tour
 * - Command help and usage examples
 * - Best practices and tips
 */

import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'

export interface HelpTopic {
  id: string
  title: string
  content: string
  examples?: string[]
  tips?: string[]
}

@customElement('compression-help')
export class CompressionHelpComponent extends LitElement {
  static styles = css`
    :host {
      display: block;
      position: relative;
    }

    .help-trigger {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 20px;
      height: 20px;
      background: var(--primary, #6366f1);
      color: white;
      border-radius: 50%;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      border: none;
      outline: none;
    }

    .help-trigger:hover {
      background: #5855eb;
      transform: scale(1.1);
    }

    .help-trigger:focus {
      box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.3);
    }

    .tooltip {
      position: absolute;
      bottom: calc(100% + 8px);
      left: 50%;
      transform: translateX(-50%);
      background: #1f2937;
      color: white;
      padding: 12px 16px;
      border-radius: 8px;
      font-size: 13px;
      line-height: 1.4;
      max-width: 280px;
      width: max-content;
      z-index: 1000;
      opacity: 0;
      visibility: hidden;
      transition: all 0.2s ease;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .tooltip.visible {
      opacity: 1;
      visibility: visible;
    }

    .tooltip::after {
      content: '';
      position: absolute;
      top: 100%;
      left: 50%;
      transform: translateX(-50%);
      border: 6px solid transparent;
      border-top-color: #1f2937;
    }

    .tooltip-content {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .tooltip-title {
      font-weight: 600;
      font-size: 14px;
      color: #f3f4f6;
    }

    .tooltip-text {
      color: #d1d5db;
    }

    .tooltip-examples {
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px solid #374151;
    }

    .tooltip-example {
      background: rgba(0, 0, 0, 0.3);
      padding: 6px 8px;
      border-radius: 4px;
      font-family: 'Monaco', 'Menlo', monospace;
      font-size: 11px;
      color: #9ca3af;
      margin-bottom: 4px;
    }

    .help-modal {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0, 0, 0, 0.5);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 10000;
      opacity: 0;
      visibility: hidden;
      transition: all 0.3s ease;
      padding: 20px;
    }

    .help-modal.visible {
      opacity: 1;
      visibility: visible;
    }

    .help-modal-content {
      background: var(--surface-color, #ffffff);
      border-radius: 16px;
      padding: 24px;
      max-width: 600px;
      width: 100%;
      max-height: 90vh;
      overflow-y: auto;
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }

    .help-modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 2px solid var(--border-color, #e0e0e0);
    }

    .help-modal-title {
      font-size: 20px;
      font-weight: 700;
      color: var(--text-primary, #1a1a1a);
    }

    .help-close-button {
      width: 32px;
      height: 32px;
      border: none;
      background: transparent;
      color: var(--text-secondary, #666);
      cursor: pointer;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s ease;
    }

    .help-close-button:hover {
      background: var(--surface-variant, #f5f5f5);
      color: var(--text-primary, #1a1a1a);
    }

    .help-sections {
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    .help-section {
      padding: 20px;
      background: var(--surface-variant, #f8f9fa);
      border-radius: 12px;
      border: 1px solid var(--border-color, #e0e0e0);
    }

    .help-section-title {
      font-size: 16px;
      font-weight: 600;
      color: var(--text-primary, #1a1a1a);
      margin-bottom: 12px;
    }

    .help-section-content {
      color: var(--text-secondary, #666);
      line-height: 1.6;
      margin-bottom: 16px;
    }

    .help-examples-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .help-example {
      background: #1f2937;
      color: #d1d5db;
      padding: 12px 16px;
      border-radius: 8px;
      font-family: 'Monaco', 'Menlo', monospace;
      font-size: 13px;
      overflow-x: auto;
    }

    .help-tips {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 16px;
    }

    .help-tip {
      display: flex;
      align-items: flex-start;
      gap: 8px;
      padding: 8px 12px;
      background: #fefce8;
      border: 1px solid #facc15;
      border-radius: 6px;
      font-size: 13px;
      color: #a16207;
    }

    .help-tip-icon {
      font-size: 14px;
      margin-top: 1px;
    }

    /* Mobile optimizations */
    @media (max-width: 768px) {
      .tooltip {
        max-width: 250px;
        font-size: 12px;
        padding: 10px 12px;
      }

      .help-modal {
        padding: 16px;
      }

      .help-modal-content {
        padding: 20px;
        border-radius: 12px;
      }

      .help-section {
        padding: 16px;
      }
    }
  `

  @property({ type: String })
  topic = 'general'

  @property({ type: String })
  trigger = 'icon' // 'icon' | 'text' | 'button'

  @property({ type: String })
  position = 'top' // 'top' | 'bottom' | 'left' | 'right'

  @state()
  private showTooltip = false

  @state()
  private showModal = false

  private helpTopics: Record<string, HelpTopic> = {
    general: {
      id: 'general',
      title: 'Context Compression',
      content: 'Context compression reduces conversation memory usage while preserving important information like decisions, insights, and key patterns.',
      examples: [
        '/hive:compact',
        '/hive:compact --level=standard',
        '/hive:compact session123 --target-tokens=5000'
      ],
      tips: [
        'Use "standard" level for balanced compression and preservation',
        'Preserve decisions for debugging sessions',
        'Preserve patterns for learning-heavy conversations'
      ]
    },
    levels: {
      id: 'levels',
      title: 'Compression Levels',
      content: 'Different compression levels offer trade-offs between space savings and information preservation.',
      examples: [
        'Light: 20% reduction, minimal data loss',
        'Standard: 50% reduction, balanced approach',
        'Aggressive: 70% reduction, maximum savings'
      ],
      tips: [
        'Start with "standard" for most use cases',
        'Use "light" for critical conversations',
        'Use "aggressive" for storage optimization'
      ]
    },
    options: {
      id: 'options',
      title: 'Compression Options',
      content: 'Customize compression behavior with advanced options to match your specific needs.',
      examples: [
        '--preserve-decisions: Keep decision points',
        '--preserve-patterns: Keep learning insights',
        '--target-tokens=N: Set specific token target'
      ],
      tips: [
        'Preserve decisions for debugging workflows',
        'Preserve patterns for research sessions',
        'Set token targets for memory constraints'
      ]
    },
    command: {
      id: 'command',
      title: 'Command Usage',
      content: 'The /hive:compact command provides flexible compression with real-time progress tracking.',
      examples: [
        '/hive:compact',
        '/hive:compact --level=aggressive --target-tokens=3000',
        '/hive:compact session123 --preserve-decisions'
      ],
      tips: [
        'Omit session ID to compress current context',
        'Combine options for fine-tuned control',
        'Monitor progress in real-time via dashboard'
      ]
    }
  }

  private handleTriggerClick() {
    if (this.trigger === 'icon' || this.trigger === 'text') {
      this.showTooltip = !this.showTooltip
    } else {
      this.showModal = true
    }
  }

  private handleTriggerMouseEnter() {
    if (this.trigger === 'icon' || this.trigger === 'text') {
      this.showTooltip = true
    }
  }

  private handleTriggerMouseLeave() {
    if (this.trigger === 'icon' || this.trigger === 'text') {
      this.showTooltip = false
    }
  }

  private handleModalClose() {
    this.showModal = false
  }

  private handleModalBackdropClick(e: Event) {
    if (e.target === e.currentTarget) {
      this.handleModalClose()
    }
  }

  private renderTrigger() {
    const topic = this.helpTopics[this.topic] || this.helpTopics.general

    switch (this.trigger) {
      case 'text':
        return html`
          <span 
            class="help-text-trigger"
            @click=${this.handleTriggerClick}
            @mouseenter=${this.handleTriggerMouseEnter}
            @mouseleave=${this.handleTriggerMouseLeave}
            style="color: var(--primary, #6366f1); cursor: pointer; text-decoration: underline;"
          >
            ${topic.title}
          </span>
        `
      
      case 'button':
        return html`
          <button 
            class="help-button-trigger"
            @click=${this.handleTriggerClick}
            style="padding: 8px 16px; background: var(--primary, #6366f1); color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 14px;"
          >
            Help
          </button>
        `
      
      default: // icon
        return html`
          <button 
            class="help-trigger"
            @click=${this.handleTriggerClick}
            @mouseenter=${this.handleTriggerMouseEnter}
            @mouseleave=${this.handleTriggerMouseLeave}
            aria-label="Show help for ${topic.title}"
          >
            ?
          </button>
        `
    }
  }

  private renderTooltip() {
    const topic = this.helpTopics[this.topic] || this.helpTopics.general

    return html`
      <div class="tooltip ${this.showTooltip ? 'visible' : ''}">
        <div class="tooltip-content">
          <div class="tooltip-title">${topic.title}</div>
          <div class="tooltip-text">${topic.content}</div>
          
          ${topic.examples && topic.examples.length > 0 ? html`
            <div class="tooltip-examples">
              ${topic.examples.map(example => html`
                <div class="tooltip-example">${example}</div>
              `)}
            </div>
          ` : ''}
        </div>
      </div>
    `
  }

  private renderModal() {
    const allTopics = Object.values(this.helpTopics)

    return html`
      <div 
        class="help-modal ${this.showModal ? 'visible' : ''}"
        @click=${this.handleModalBackdropClick}
      >
        <div class="help-modal-content" @click=${(e: Event) => e.stopPropagation()}>
          <div class="help-modal-header">
            <h2 class="help-modal-title">Context Compression Help</h2>
            <button class="help-close-button" @click=${this.handleModalClose}>
              ✕
            </button>
          </div>

          <div class="help-sections">
            ${allTopics.map(topic => html`
              <div class="help-section">
                <h3 class="help-section-title">${topic.title}</h3>
                <div class="help-section-content">${topic.content}</div>

                ${topic.examples && topic.examples.length > 0 ? html`
                  <div class="help-examples-list">
                    ${topic.examples.map(example => html`
                      <div class="help-example">${example}</div>
                    `)}
                  </div>
                ` : ''}

                ${topic.tips && topic.tips.length > 0 ? html`
                  <div class="help-tips">
                    ${topic.tips.map(tip => html`
                      <div class="help-tip">
                        <div class="help-tip-icon">💡</div>
                        <div>${tip}</div>
                      </div>
                    `)}
                  </div>
                ` : ''}
              </div>
            `)}
          </div>
        </div>
      </div>
    `
  }

  render() {
    return html`
      ${this.renderTrigger()}
      ${this.trigger !== 'button' ? this.renderTooltip() : ''}
      ${this.renderModal()}
    `
  }
}

// Guided Tour Component
@customElement('compression-guided-tour')
export class CompressionGuidedTourComponent extends LitElement {
  static styles = css`
    :host {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 20000;
      pointer-events: none;
    }

    .tour-overlay {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0, 0, 0, 0.7);
      opacity: 0;
      transition: opacity 0.3s ease;
      pointer-events: auto;
    }

    .tour-overlay.visible {
      opacity: 1;
    }

    .tour-spotlight {
      position: absolute;
      border: 4px solid var(--primary, #6366f1);
      border-radius: 8px;
      box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.3);
      pointer-events: none;
      transition: all 0.3s ease;
    }

    .tour-tooltip {
      position: absolute;
      background: white;
      border-radius: 12px;
      padding: 20px;
      max-width: 320px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
      pointer-events: auto;
      z-index: 20001;
    }

    .tour-tooltip-title {
      font-size: 16px;
      font-weight: 600;
      color: var(--text-primary, #1a1a1a);
      margin-bottom: 8px;
    }

    .tour-tooltip-content {
      font-size: 14px;
      color: var(--text-secondary, #666);
      line-height: 1.5;
      margin-bottom: 16px;
    }

    .tour-tooltip-actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .tour-progress {
      font-size: 12px;
      color: var(--text-secondary, #666);
    }

    .tour-buttons {
      display: flex;
      gap: 8px;
    }

    .tour-button {
      padding: 8px 16px;
      border: none;
      border-radius: 6px;
      font-size: 14px;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .tour-button.primary {
      background: var(--primary, #6366f1);
      color: white;
    }

    .tour-button.secondary {
      background: transparent;
      color: var(--text-secondary, #666);
      border: 1px solid var(--border-color, #e0e0e0);
    }

    .tour-button:hover {
      transform: translateY(-1px);
    }
  `

  @state()
  private isActive = false

  @state()
  private currentStep = 0

  private tourSteps = [
    {
      target: 'compression-widget',
      title: 'Welcome to Context Compression',
      content: 'This widget provides quick access to compress your conversation context, saving memory while preserving important information.'
    },
    {
      target: '.compression-levels',
      title: 'Choose Compression Level',
      content: 'Select from Light (20%), Standard (50%), or Aggressive (70%) compression based on your needs.'
    },
    {
      target: '.advanced-options',
      title: 'Advanced Options',
      content: 'Customize compression behavior by preserving decisions, patterns, or setting specific token targets.'
    },
    {
      target: '.action-buttons',
      title: 'Start Compression',
      content: 'Click "Start Compression" to begin the process. You can monitor progress in real-time.'
    }
  ]

  startTour() {
    this.isActive = true
    this.currentStep = 0
  }

  endTour() {
    this.isActive = false
    this.currentStep = 0
  }

  nextStep() {
    if (this.currentStep < this.tourSteps.length - 1) {
      this.currentStep++
    } else {
      this.endTour()
    }
  }

  previousStep() {
    if (this.currentStep > 0) {
      this.currentStep--
    }
  }

  render() {
    if (!this.isActive) return html``

    const currentStep = this.tourSteps[this.currentStep]

    return html`
      <div class="tour-overlay ${this.isActive ? 'visible' : ''}" @click=${this.endTour}></div>
      
      ${currentStep ? html`
        <div class="tour-tooltip" style="top: 200px; left: 50%; transform: translateX(-50%);">
          <div class="tour-tooltip-title">${currentStep.title}</div>
          <div class="tour-tooltip-content">${currentStep.content}</div>
          
          <div class="tour-tooltip-actions">
            <div class="tour-progress">
              Step ${this.currentStep + 1} of ${this.tourSteps.length}
            </div>
            
            <div class="tour-buttons">
              ${this.currentStep > 0 ? html`
                <button class="tour-button secondary" @click=${this.previousStep}>
                  Previous
                </button>
              ` : ''}
              
              <button class="tour-button secondary" @click=${this.endTour}>
                Skip Tour
              </button>
              
              <button class="tour-button primary" @click=${this.nextStep}>
                ${this.currentStep === this.tourSteps.length - 1 ? 'Finish' : 'Next'}
              </button>
            </div>
          </div>
        </div>
      ` : ''}
    `
  }
}