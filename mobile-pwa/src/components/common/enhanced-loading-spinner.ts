import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'

export type LoadingSize = 'small' | 'medium' | 'large'
export type LoadingVariant = 'spinner' | 'pulse' | 'skeleton' | 'dots'

@customElement('enhanced-loading-spinner')
export class EnhancedLoadingSpinner extends LitElement {
  @property({ type: String }) size: LoadingSize = 'medium'
  @property({ type: String }) variant: LoadingVariant = 'spinner'
  @property({ type: String }) message: string = ''
  @property({ type: Boolean }) overlay: boolean = false
  @property({ type: String }) color: string = ''
  @state() private loadingMessages: string[] = [
    'Loading dashboard data...',
    'Fetching agent status...',
    'Synchronizing with backend...',
    'Updating real-time metrics...',
    'Processing system health...'
  ]
  @state() private currentMessageIndex: number = 0

  static styles = css`
    :host {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: var(--space-3, 0.75rem);
    }

    .loading-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: var(--space-4, 1rem);
      padding: var(--space-4, 1rem);
    }

    .loading-overlay {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(255, 255, 255, 0.9);
      backdrop-filter: blur(4px);
      z-index: 9999;
      display: flex;
      align-items: center;
      justify-content: center;
      color-scheme: light;
    }

    .theme-dark .loading-overlay {
      background: rgba(15, 23, 42, 0.9);
      color-scheme: dark;
    }

    /* Spinner variants */
    .spinner {
      border-radius: 50%;
      border: 3px solid var(--color-border, #e2e8f0);
      border-top-color: var(--spinner-color, var(--color-primary, #1e40af));
      animation: spin 1s linear infinite;
    }

    .spinner.small {
      width: 24px;
      height: 24px;
      border-width: 2px;
    }

    .spinner.medium {
      width: 40px;
      height: 40px;
      border-width: 3px;
    }

    .spinner.large {
      width: 56px;
      height: 56px;
      border-width: 4px;
    }

    /* Pulse variant */
    .pulse {
      border-radius: 50%;
      background: var(--spinner-color, var(--color-primary, #1e40af));
      animation: pulse 1.5s ease-in-out infinite;
    }

    .pulse.small {
      width: 16px;
      height: 16px;
    }

    .pulse.medium {
      width: 24px;
      height: 24px;
    }

    .pulse.large {
      width: 32px;
      height: 32px;
    }

    /* Dots variant */
    .dots {
      display: flex;
      gap: var(--space-2, 0.5rem);
    }

    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--spinner-color, var(--color-primary, #1e40af));
      animation: dots 1.4s ease-in-out infinite both;
    }

    .dots.small .dot {
      width: 6px;
      height: 6px;
    }

    .dots.medium .dot {
      width: 8px;
      height: 8px;
    }

    .dots.large .dot {
      width: 12px;
      height: 12px;
    }

    .dot:nth-child(1) { animation-delay: -0.32s; }
    .dot:nth-child(2) { animation-delay: -0.16s; }
    .dot:nth-child(3) { animation-delay: 0s; }

    /* Skeleton variant */
    .skeleton-container {
      display: flex;
      flex-direction: column;
      gap: var(--space-3, 0.75rem);
      width: 100%;
      max-width: 300px;
    }

    .skeleton-line {
      height: 16px;
      background: var(--color-surface-secondary, #f1f5f9);
      border-radius: var(--radius-md, 0.375rem);
      position: relative;
      overflow: hidden;
    }

    .skeleton-line::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: linear-gradient(
        90deg,
        transparent,
        rgba(255, 255, 255, 0.4),
        transparent
      );
      animation: skeleton-shimmer 1.5s ease-in-out infinite;
    }

    .skeleton-line.short { width: 75%; }
    .skeleton-line.medium { width: 85%; }
    .skeleton-line.long { width: 95%; }

    /* Loading message */
    .loading-message {
      font-size: var(--text-sm, 0.875rem);
      color: var(--color-text-secondary, #64748b);
      text-align: center;
      font-weight: 500;
      opacity: 0.8;
      animation: fade-in-out 2s ease-in-out infinite;
    }

    .loading-progress {
      width: 200px;
      height: 4px;
      background: var(--color-border, #e2e8f0);
      border-radius: 2px;
      overflow: hidden;
      margin-top: var(--space-2, 0.5rem);
    }

    .loading-progress-bar {
      height: 100%;
      background: var(--spinner-color, var(--color-primary, #1e40af));
      border-radius: 2px;
      animation: progress 3s ease-in-out infinite;
    }

    /* Animations */
    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    @keyframes pulse {
      0%, 100% {
        opacity: 1;
        transform: scale(1);
      }
      50% {
        opacity: 0.3;
        transform: scale(1.2);
      }
    }

    @keyframes dots {
      0%, 80%, 100% {
        transform: scale(0);
        opacity: 0.5;
      }
      40% {
        transform: scale(1);
        opacity: 1;
      }
    }

    @keyframes skeleton-shimmer {
      0% { transform: translateX(-100%); }
      100% { transform: translateX(100%); }
    }

    @keyframes fade-in-out {
      0%, 100% { opacity: 0.6; }
      50% { opacity: 1; }
    }

    @keyframes progress {
      0% { transform: translateX(-100%); }
      50% { transform: translateX(0%); }
      100% { transform: translateX(100%); }
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
      .loading-container {
        padding: var(--space-6, 1.5rem);
      }

      .loading-message {
        font-size: var(--text-base, 1rem);
      }
    }

    /* Reduced motion support */
    @media (prefers-reduced-motion: reduce) {
      .spinner,
      .pulse,
      .dot,
      .skeleton-line::after,
      .loading-message,
      .loading-progress-bar {
        animation: none;
      }

      .pulse {
        opacity: 0.7;
      }

      .dot {
        opacity: 0.7;
      }
    }

    /* High contrast mode */
    @media (prefers-contrast: high) {
      .spinner {
        border-width: 4px;
      }

      .loading-message {
        color: var(--color-text, #0f172a);
        opacity: 1;
      }
    }
  `

  connectedCallback() {
    super.connectedCallback()
    if (this.message) {
      this.startMessageRotation()
    }
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    this.stopMessageRotation()
  }

  private messageInterval?: number

  private startMessageRotation() {
    if (this.loadingMessages.length <= 1) return
    
    this.messageInterval = window.setInterval(() => {
      this.currentMessageIndex = (this.currentMessageIndex + 1) % this.loadingMessages.length
    }, 2000)
  }

  private stopMessageRotation() {
    if (this.messageInterval) {
      clearInterval(this.messageInterval)
    }
  }

  private renderSpinner() {
    return html`
      <div 
        class="spinner ${this.size}"
        style="--spinner-color: ${this.color}"
        role="status"
        aria-label="Loading"
      ></div>
    `
  }

  private renderPulse() {
    return html`
      <div 
        class="pulse ${this.size}"
        style="--spinner-color: ${this.color}"
        role="status"
        aria-label="Loading"
      ></div>
    `
  }

  private renderDots() {
    return html`
      <div 
        class="dots ${this.size}"
        role="status"
        aria-label="Loading"
      >
        <div class="dot" style="--spinner-color: ${this.color}"></div>
        <div class="dot" style="--spinner-color: ${this.color}"></div>
        <div class="dot" style="--spinner-color: ${this.color}"></div>
      </div>
    `
  }

  private renderSkeleton() {
    return html`
      <div class="skeleton-container" role="status" aria-label="Loading content">
        <div class="skeleton-line long"></div>
        <div class="skeleton-line medium"></div>
        <div class="skeleton-line short"></div>
      </div>
    `
  }

  private renderLoadingIndicator() {
    switch (this.variant) {
      case 'pulse':
        return this.renderPulse()
      case 'dots':
        return this.renderDots()
      case 'skeleton':
        return this.renderSkeleton()
      default:
        return this.renderSpinner()
    }
  }

  private getCurrentMessage(): string {
    if (this.message) {
      return this.message
    }
    return this.loadingMessages[this.currentMessageIndex] || 'Loading...'
  }

  render() {
    const content = html`
      <div class="loading-container">
        ${this.renderLoadingIndicator()}
        
        ${this.message || this.loadingMessages.length > 0 ? html`
          <div class="loading-message" aria-live="polite">
            ${this.getCurrentMessage()}
          </div>
          
          ${this.variant !== 'skeleton' ? html`
            <div class="loading-progress">
              <div class="loading-progress-bar" style="--spinner-color: ${this.color}"></div>
            </div>
          ` : ''}
        ` : ''}
      </div>
    `

    if (this.overlay) {
      return html`
        <div class="loading-overlay">
          ${content}
        </div>
      `
    }

    return content
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'enhanced-loading-spinner': EnhancedLoadingSpinner
  }
}