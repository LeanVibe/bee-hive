import { LitElement, html, css } from 'lit'
import { customElement, state, property } from 'lit/decorators.js'

interface BeforeInstallPromptEvent extends Event {
  prompt(): Promise<void>
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>
}

export type InstallSource = 'banner' | 'menu' | 'settings' | 'onboarding'

@customElement('pwa-install-prompt')
export class PWAInstallPrompt extends LitElement {
  @state() private deferredPrompt: BeforeInstallPromptEvent | null = null
  @state() private isInstallable: boolean = false
  @state() private isInstalled: boolean = false
  @state() private showPrompt: boolean = false
  @state() private installStep: 'prompt' | 'installing' | 'success' | 'error' = 'prompt'
  @state() private errorMessage: string = ''
  @property({ type: String }) source: InstallSource = 'banner'
  @property({ type: Boolean }) compact: boolean = false
  @property({ type: Boolean }) autoShow: boolean = true

  static styles = css`
    :host {
      display: block;
      position: relative;
    }

    .install-prompt {
      background: var(--glass-bg, rgba(255, 255, 255, 0.95));
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid var(--color-border, #e2e8f0);
      border-radius: var(--radius-xl, 1rem);
      padding: var(--space-6, 1.5rem);
      box-shadow: var(--shadow-xl, 0 20px 25px -5px rgba(0, 0, 0, 0.1));
      position: relative;
      overflow: hidden;
      max-width: 400px;
      margin: 0 auto;
    }

    .install-prompt.compact {
      padding: var(--space-4, 1rem);
      max-width: 300px;
    }

    .install-prompt::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 4px;
      background: linear-gradient(90deg, var(--color-primary), var(--color-accent, #8b5cf6));
    }

    .prompt-header {
      display: flex;
      align-items: flex-start;
      gap: var(--space-4, 1rem);
      margin-bottom: var(--space-4, 1rem);
    }

    .app-icon {
      width: 56px;
      height: 56px;
      background: linear-gradient(135deg, var(--color-primary), var(--color-primary-light));
      border-radius: var(--radius-xl, 1rem);
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: 700;
      font-size: 1.5rem;
      box-shadow: var(--shadow-md, 0 4px 6px -1px rgba(0, 0, 0, 0.1));
      flex-shrink: 0;
    }

    .compact .app-icon {
      width: 40px;
      height: 40px;
      font-size: 1.125rem;
    }

    .prompt-content {
      flex: 1;
    }

    .app-name {
      font-size: var(--text-lg, 1.125rem);
      font-weight: 700;
      color: var(--color-text, #0f172a);
      margin: 0 0 var(--space-1, 0.25rem) 0;
    }

    .compact .app-name {
      font-size: var(--text-base, 1rem);
    }

    .app-description {
      font-size: var(--text-sm, 0.875rem);
      color: var(--color-text-secondary, #334155);
      line-height: 1.5;
      margin: 0;
    }

    .compact .app-description {
      font-size: var(--text-xs, 0.75rem);
    }

    .features-list {
      margin: var(--space-4, 1rem) 0;
      padding: 0;
      list-style: none;
    }

    .compact .features-list {
      margin: var(--space-3, 0.75rem) 0;
    }

    .feature-item {
      display: flex;
      align-items: center;
      gap: var(--space-2, 0.5rem);
      margin-bottom: var(--space-2, 0.5rem);
      font-size: var(--text-sm, 0.875rem);
      color: var(--color-text-secondary, #334155);
    }

    .compact .feature-item {
      font-size: var(--text-xs, 0.75rem);
      margin-bottom: var(--space-1, 0.25rem);
    }

    .feature-icon {
      width: 16px;
      height: 16px;
      color: var(--color-success, #10b981);
      flex-shrink: 0;
    }

    .compact .feature-icon {
      width: 14px;
      height: 14px;
    }

    .prompt-actions {
      display: flex;
      gap: var(--space-3, 0.75rem);
      margin-top: var(--space-6, 1.5rem);
    }

    .compact .prompt-actions {
      margin-top: var(--space-4, 1rem);
      gap: var(--space-2, 0.5rem);
    }

    .install-button {
      flex: 1;
      background: linear-gradient(135deg, var(--color-primary), var(--color-primary-dark));
      color: white;
      border: none;
      border-radius: var(--radius-lg, 0.5rem);
      padding: var(--space-3, 0.75rem) var(--space-4, 1rem);
      font-size: var(--text-sm, 0.875rem);
      font-weight: 600;
      cursor: pointer;
      transition: all var(--transition-normal, 0.3s);
      position: relative;
      overflow: hidden;
      min-height: 44px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: var(--space-2, 0.5rem);
    }

    .install-button:hover {
      transform: translateY(-1px);
      box-shadow: var(--shadow-lg, 0 10px 15px -3px rgba(0, 0, 0, 0.1));
    }

    .install-button:active {
      transform: translateY(0);
    }

    .install-button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }

    .dismiss-button {
      background: transparent;
      color: var(--color-text-muted, #64748b);
      border: 1px solid var(--color-border, #e2e8f0);
      border-radius: var(--radius-lg, 0.5rem);
      padding: var(--space-3, 0.75rem) var(--space-4, 1rem);
      font-size: var(--text-sm, 0.875rem);
      font-weight: 500;
      cursor: pointer;
      transition: all var(--transition-normal, 0.3s);
      min-height: 44px;
    }

    .dismiss-button:hover {
      background: var(--color-surface-secondary, #f1f5f9);
      border-color: var(--color-border-focus, #3b82f6);
    }

    .close-button {
      position: absolute;
      top: var(--space-4, 1rem);
      right: var(--space-4, 1rem);
      background: none;
      border: none;
      color: var(--color-text-muted, #64748b);
      cursor: pointer;
      padding: var(--space-1, 0.25rem);
      border-radius: var(--radius-md, 0.375rem);
      transition: all var(--transition-normal, 0.3s);
      width: 32px;
      height: 32px;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .close-button:hover {
      background: var(--color-surface-secondary, #f1f5f9);
      color: var(--color-text, #0f172a);
    }

    .loading-spinner {
      width: 20px;
      height: 20px;
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-top: 2px solid white;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }

    .success-icon {
      color: var(--color-success, #10b981);
      animation: bounceIn 0.5s ease-out;
    }

    .error-message {
      background: var(--color-error-light, #fef2f2);
      color: var(--color-error, #dc2626);
      border: 1px solid var(--color-error, #dc2626);
      border-radius: var(--radius-md, 0.375rem);
      padding: var(--space-3, 0.75rem);
      margin-top: var(--space-4, 1rem);
      font-size: var(--text-sm, 0.875rem);
    }

    .platform-hint {
      margin-top: var(--space-4, 1rem);
      padding: var(--space-3, 0.75rem);
      background: var(--color-info-light, #f0f9ff);
      border: 1px solid var(--color-info, #0284c7);
      border-radius: var(--radius-md, 0.375rem);
      font-size: var(--text-xs, 0.75rem);
      color: var(--color-info-dark, #075985);
    }

    /* Mobile optimizations */
    @media (max-width: 768px) {
      .install-prompt {
        margin: var(--space-4, 1rem);
        max-width: none;
      }

      .prompt-actions {
        flex-direction: column;
      }

      .install-button,
      .dismiss-button {
        min-height: 48px;
        font-size: var(--text-base, 1rem);
      }
    }

    /* Animations */
    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    @keyframes bounceIn {
      0% {
        opacity: 0;
        transform: scale(0.3);
      }
      50% {
        opacity: 1;
        transform: scale(1.05);
      }
      70% {
        transform: scale(0.9);
      }
      100% {
        opacity: 1;
        transform: scale(1);
      }
    }

    /* Reduced motion */
    @media (prefers-reduced-motion: reduce) {
      .install-button,
      .dismiss-button,
      .close-button {
        transition: none;
      }

      .loading-spinner,
      .success-icon {
        animation: none;
      }

      .install-button:hover {
        transform: none;
      }
    }

    /* Hidden state */
    :host([hidden]) {
      display: none !important;
    }
  `

  connectedCallback() {
    super.connectedCallback()
    this.initializeInstallPrompt()
  }

  private async initializeInstallPrompt() {
    // Check if already installed
    if (window.matchMedia('(display-mode: standalone)').matches ||
        (window.navigator as any).standalone === true) {
      this.isInstalled = true
      return
    }

    // Listen for beforeinstallprompt event
    window.addEventListener('beforeinstallprompt', this.handleBeforeInstallPrompt.bind(this))
    
    // Listen for app installed event
    window.addEventListener('appinstalled', this.handleAppInstalled.bind(this))

    // Check if we can show the prompt based on user preferences
    if (this.autoShow && this.shouldShowPrompt()) {
      this.showPrompt = true
    }
  }

  private handleBeforeInstallPrompt(event: Event) {
    event.preventDefault()
    this.deferredPrompt = event as BeforeInstallPromptEvent
    this.isInstallable = true

    // Show prompt if auto-show is enabled
    if (this.autoShow && this.shouldShowPrompt()) {
      this.showPrompt = true
    }

    // Dispatch event for parent components
    this.dispatchEvent(new CustomEvent('install-available', {
      detail: { canInstall: true },
      bubbles: true,
      composed: true
    }))
  }

  private handleAppInstalled(event: Event) {
    this.isInstalled = true
    this.showPrompt = false
    this.deferredPrompt = null

    // Clear install prompts from localStorage
    localStorage.removeItem('pwa-install-dismissed')
    localStorage.removeItem('pwa-install-reminded')

    // Dispatch success event
    this.dispatchEvent(new CustomEvent('install-success', {
      detail: { source: this.source },
      bubbles: true,
      composed: true
    }))

    // Show success message briefly
    this.installStep = 'success'
    setTimeout(() => {
      this.showPrompt = false
      this.installStep = 'prompt'
    }, 3000)
  }

  private shouldShowPrompt(): boolean {
    // Don't show if already installed
    if (this.isInstalled) return false

    // Check if user has dismissed recently
    const dismissed = localStorage.getItem('pwa-install-dismissed')
    if (dismissed) {
      const dismissedTime = parseInt(dismissed)
      const oneDayAgo = Date.now() - (24 * 60 * 60 * 1000)
      if (dismissedTime > oneDayAgo) {
        return false
      }
    }

    // Check if user has been reminded recently
    const reminded = localStorage.getItem('pwa-install-reminded')
    if (reminded) {
      const remindedTime = parseInt(reminded)
      const oneWeekAgo = Date.now() - (7 * 24 * 60 * 60 * 1000)
      if (remindedTime > oneWeekAgo) {
        return false
      }
    }

    return true
  }

  private async handleInstall() {
    if (!this.deferredPrompt) {
      this.showPlatformInstructions()
      return
    }

    try {
      this.installStep = 'installing'

      // Show the install prompt
      await this.deferredPrompt.prompt()

      // Wait for user choice
      const choiceResult = await this.deferredPrompt.userChoice

      if (choiceResult.outcome === 'accepted') {
        console.log('User accepted PWA install')
        
        // Track install
        this.trackInstallEvent('accepted')
        
        // The appinstalled event will handle the success state
      } else {
        console.log('User dismissed PWA install')
        this.installStep = 'prompt'
        this.handleDismiss()
        
        // Track dismissal
        this.trackInstallEvent('dismissed')
      }

      this.deferredPrompt = null

    } catch (error) {
      console.error('Install failed:', error)
      this.installStep = 'error'
      this.errorMessage = 'Installation failed. Please try again.'
      
      // Reset after 3 seconds
      setTimeout(() => {
        this.installStep = 'prompt'
        this.errorMessage = ''
      }, 3000)
    }
  }

  private showPlatformInstructions() {
    // For browsers that don't support the install prompt API
    const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent)
    const isSafari = /Safari/.test(navigator.userAgent) && !/Chrome/.test(navigator.userAgent)
    
    let instructions = ''
    
    if (isIOS && isSafari) {
      instructions = 'To install: Tap the Share button, then "Add to Home Screen"'
    } else if (/Chrome/.test(navigator.userAgent)) {
      instructions = 'To install: Look for the install icon in the address bar'
    } else {
      instructions = 'To install: Check your browser menu for "Install App" or "Add to Home Screen"'
    }

    // Show platform-specific instructions
    this.dispatchEvent(new CustomEvent('show-instructions', {
      detail: { instructions, platform: this.getPlatform() },
      bubbles: true,
      composed: true
    }))
  }

  private getPlatform(): string {
    const userAgent = navigator.userAgent
    
    if (/iPad|iPhone|iPod/.test(userAgent)) return 'ios'
    if (/Android/.test(userAgent)) return 'android'
    if (/Windows/.test(userAgent)) return 'windows'
    if (/Mac/.test(userAgent)) return 'mac'
    
    return 'unknown'
  }

  private handleDismiss() {
    // Store dismissal timestamp
    localStorage.setItem('pwa-install-dismissed', Date.now().toString())
    
    this.showPrompt = false

    // Dispatch dismiss event
    this.dispatchEvent(new CustomEvent('install-dismissed', {
      detail: { source: this.source },
      bubbles: true,
      composed: true
    }))
  }

  private handleClose() {
    this.showPrompt = false
  }

  private trackInstallEvent(outcome: 'accepted' | 'dismissed') {
    // Track with analytics if available
    if ((window as any).gtag) {
      (window as any).gtag('event', 'pwa_install_prompt', {
        outcome,
        source: this.source,
        platform: this.getPlatform()
      })
    }

    // Custom tracking event
    this.dispatchEvent(new CustomEvent('install-tracked', {
      detail: { outcome, source: this.source, platform: this.getPlatform() },
      bubbles: true,
      composed: true
    }))
  }

  public showInstallPrompt() {
    if (this.isInstallable && !this.isInstalled) {
      this.showPrompt = true
    }
  }

  public hideInstallPrompt() {
    this.showPrompt = false
  }

  private renderInstallButton() {
    switch (this.installStep) {
      case 'installing':
        return html`
          <button class="install-button" disabled>
            <div class="loading-spinner"></div>
            Installing...
          </button>
        `
      case 'success':
        return html`
          <button class="install-button" disabled>
            <svg class="success-icon" width="20" height="20" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/>
            </svg>
            Installed!
          </button>
        `
      case 'error':
        return html`
          <button class="install-button" @click=${this.handleInstall}>
            Try Again
          </button>
        `
      default:
        return html`
          <button class="install-button" @click=${this.handleInstall}>
            <svg width="20" height="20" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd"/>
            </svg>
            Install App
          </button>
        `
    }
  }

  render() {
    if (!this.showPrompt || this.isInstalled) {
      return html``
    }

    return html`
      <div class="install-prompt ${this.compact ? 'compact' : ''}">
        <button class="close-button" @click=${this.handleClose} aria-label="Close install prompt">
          <svg width="16" height="16" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"/>
          </svg>
        </button>

        <div class="prompt-header">
          <div class="app-icon">HO</div>
          <div class="prompt-content">
            <h3 class="app-name">HiveOps</h3>
            <p class="app-description">
              Install the app for faster access and better mobile experience
            </p>
          </div>
        </div>

        ${!this.compact ? html`
          <ul class="features-list">
            <li class="feature-item">
              <svg class="feature-icon" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/>
              </svg>
              Works offline with cached data
            </li>
            <li class="feature-item">
              <svg class="feature-icon" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/>
              </svg>
              Push notifications for critical alerts
            </li>
            <li class="feature-item">
              <svg class="feature-icon" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/>
              </svg>
              Faster loading and better performance
            </li>
          </ul>
        ` : ''}

        <div class="prompt-actions">
          ${this.renderInstallButton()}
          <button class="dismiss-button" @click=${this.handleDismiss}>
            Not Now
          </button>
        </div>

        ${this.errorMessage ? html`
          <div class="error-message">
            ${this.errorMessage}
          </div>
        ` : ''}

        ${!this.deferredPrompt && !this.compact ? html`
          <div class="platform-hint">
            ${this.getPlatform() === 'ios' 
              ? 'On iOS: Tap Share → Add to Home Screen'
              : 'Look for the install icon in your browser address bar'
            }
          </div>
        ` : ''}
      </div>
    `
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'pwa-install-prompt': PWAInstallPrompt
  }
}