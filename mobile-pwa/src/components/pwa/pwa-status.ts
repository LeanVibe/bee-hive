import { LitElement, html, css } from 'lit'
import { customElement, state, property } from 'lit/decorators.js'

export interface PWACapabilities {
  isInstalled: boolean
  isOnline: boolean
  hasServiceWorker: boolean
  hasPushSupport: boolean
  hasNotificationSupport: boolean
  hasBackgroundSync: boolean
  storageQuota: number
  usedStorage: number
  cacheStatus: 'healthy' | 'degraded' | 'empty'
}

@customElement('pwa-status')
export class PWAStatus extends LitElement {
  @property({ type: Boolean }) expanded: boolean = false
  @property({ type: Boolean }) showDetails: boolean = false
  @state() private capabilities: PWACapabilities = {
    isInstalled: false,
    isOnline: navigator.onLine,
    hasServiceWorker: 'serviceWorker' in navigator,
    hasPushSupport: 'PushManager' in window,
    hasNotificationSupport: 'Notification' in window,
    hasBackgroundSync: 'serviceWorker' in navigator && 'sync' in window.ServiceWorkerRegistration.prototype,
    storageQuota: 0,
    usedStorage: 0,
    cacheStatus: 'empty'
  }
  @state() private lastUpdateTime: Date | null = null
  @state() private updateAvailable: boolean = false
  @state() private isUpdating: boolean = false

  static styles = css`
    :host {
      display: block;
    }

    .pwa-status {
      background: var(--color-surface, #ffffff);
      border: 1px solid var(--color-border, #e2e8f0);
      border-radius: var(--radius-lg, 0.5rem);
      padding: var(--space-4, 1rem);
      transition: all var(--transition-normal, 0.3s);
    }

    .pwa-status.expanded {
      border-radius: var(--radius-xl, 1rem);
      box-shadow: var(--shadow-lg, 0 10px 15px -3px rgba(0, 0, 0, 0.1));
    }

    .status-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      cursor: pointer;
      padding: var(--space-2, 0.5rem);
      border-radius: var(--radius-md, 0.375rem);
      transition: background-color var(--transition-normal, 0.3s);
    }

    .status-header:hover {
      background: var(--color-surface-secondary, #f8fafc);
    }

    .status-info {
      display: flex;
      align-items: center;
      gap: var(--space-3, 0.75rem);
    }

    .status-icon {
      width: 24px;
      height: 24px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }

    .status-icon.online {
      background: var(--color-success-light, #d1fae5);
      color: var(--color-success, #10b981);
    }

    .status-icon.offline {
      background: var(--color-error-light, #fef2f2);
      color: var(--color-error, #ef4444);
    }

    .status-icon.degraded {
      background: var(--color-warning-light, #fef3c7);
      color: var(--color-warning, #f59e0b);
    }

    .status-pulse {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      border-radius: 50%;
      border: 2px solid currentColor;
      opacity: 0;
      animation: pulse 2s infinite;
    }

    .status-text {
      display: flex;
      flex-direction: column;
      gap: var(--space-1, 0.25rem);
    }

    .status-title {
      font-size: var(--text-sm, 0.875rem);
      font-weight: 600;
      color: var(--color-text, #0f172a);
      margin: 0;
    }

    .status-subtitle {
      font-size: var(--text-xs, 0.75rem);
      color: var(--color-text-muted, #64748b);
      margin: 0;
    }

    .expand-icon {
      width: 16px;
      height: 16px;
      color: var(--color-text-muted, #64748b);
      transition: transform var(--transition-normal, 0.3s);
    }

    .expand-icon.expanded {
      transform: rotate(180deg);
    }

    .status-details {
      margin-top: var(--space-4, 1rem);
      border-top: 1px solid var(--color-border, #e2e8f0);
      padding-top: var(--space-4, 1rem);
      opacity: 0;
      max-height: 0;
      overflow: hidden;
      transition: all var(--transition-normal, 0.3s);
    }

    .status-details.visible {
      opacity: 1;
      max-height: 500px;
    }

    .capabilities-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: var(--space-4, 1rem);
      margin-bottom: var(--space-6, 1.5rem);
    }

    .capability-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: var(--space-3, 0.75rem);
      background: var(--color-surface-secondary, #f8fafc);
      border-radius: var(--radius-md, 0.375rem);
      border: 1px solid var(--color-border, #e2e8f0);
    }

    .capability-label {
      font-size: var(--text-sm, 0.875rem);
      color: var(--color-text, #0f172a);
      font-weight: 500;
    }

    .capability-status {
      display: flex;
      align-items: center;
      gap: var(--space-1, 0.25rem);
      font-size: var(--text-xs, 0.75rem);
      font-weight: 600;
    }

    .capability-status.enabled {
      color: var(--color-success, #10b981);
    }

    .capability-status.disabled {
      color: var(--color-error, #ef4444);
    }

    .capability-status.partial {
      color: var(--color-warning, #f59e0b);
    }

    .storage-info {
      margin-bottom: var(--space-4, 1rem);
    }

    .storage-bar {
      width: 100%;
      height: 8px;
      background: var(--color-border, #e2e8f0);
      border-radius: var(--radius-full, 9999px);
      overflow: hidden;
      margin: var(--space-2, 0.5rem) 0;
    }

    .storage-used {
      height: 100%;
      background: linear-gradient(90deg, var(--color-primary), var(--color-primary-light));
      border-radius: var(--radius-full, 9999px);
      transition: width 0.5s ease;
    }

    .storage-text {
      display: flex;
      justify-content: space-between;
      font-size: var(--text-xs, 0.75rem);
      color: var(--color-text-muted, #64748b);
    }

    .actions-section {
      display: flex;
      gap: var(--space-2, 0.5rem);
      flex-wrap: wrap;
    }

    .action-button {
      padding: var(--space-2, 0.5rem) var(--space-3, 0.75rem);
      background: var(--color-primary, #1e40af);
      color: white;
      border: none;
      border-radius: var(--radius-md, 0.375rem);
      font-size: var(--text-xs, 0.75rem);
      font-weight: 500;
      cursor: pointer;
      transition: all var(--transition-normal, 0.3s);
      min-height: 36px;
    }

    .action-button:hover {
      background: var(--color-primary-dark, #1e3a8a);
      transform: translateY(-1px);
    }

    .action-button:disabled {
      background: var(--color-text-muted, #64748b);
      cursor: not-allowed;
      transform: none;
    }

    .action-button.secondary {
      background: transparent;
      color: var(--color-primary, #1e40af);
      border: 1px solid var(--color-primary, #1e40af);
    }

    .action-button.secondary:hover {
      background: var(--color-primary-alpha, rgba(30, 64, 175, 0.1));
    }

    .update-indicator {
      display: flex;
      align-items: center;
      gap: var(--space-2, 0.5rem);
      padding: var(--space-3, 0.75rem);
      background: var(--color-info-light, #f0f9ff);
      border: 1px solid var(--color-info, #0284c7);
      border-radius: var(--radius-md, 0.375rem);
      margin-bottom: var(--space-4, 1rem);
      font-size: var(--text-sm, 0.875rem);
      color: var(--color-info-dark, #075985);
    }

    .loading-spinner {
      width: 16px;
      height: 16px;
      border: 2px solid var(--color-border, #e2e8f0);
      border-top: 2px solid var(--color-primary, #1e40af);
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }

    /* Mobile optimizations */
    @media (max-width: 768px) {
      .capabilities-grid {
        grid-template-columns: 1fr;
      }

      .actions-section {
        flex-direction: column;
      }

      .action-button {
        min-height: 44px;
        font-size: var(--text-sm, 0.875rem);
      }
    }

    /* Animations */
    @keyframes pulse {
      0% {
        opacity: 0;
        transform: scale(1);
      }
      50% {
        opacity: 0.7;
        transform: scale(1.2);
      }
      100% {
        opacity: 0;
        transform: scale(1.4);
      }
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    /* Reduced motion */
    @media (prefers-reduced-motion: reduce) {
      .status-pulse,
      .loading-spinner {
        animation: none;
      }

      .expand-icon,
      .action-button {
        transition: none;
      }
    }
  `

  connectedCallback() {
    super.connectedCallback()
    this.initializePWAStatus()
    this.setupEventListeners()
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    this.removeEventListeners()
  }

  private async initializePWAStatus() {
    // Check if app is installed
    this.capabilities.isInstalled = 
      window.matchMedia('(display-mode: standalone)').matches ||
      (window.navigator as any).standalone === true

    // Get storage information
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      const estimate = await navigator.storage.estimate()
      this.capabilities.storageQuota = estimate.quota || 0
      this.capabilities.usedStorage = estimate.usage || 0
    }

    // Check cache status
    await this.updateCacheStatus()

    // Check for service worker updates
    if (this.capabilities.hasServiceWorker) {
      const registration = await navigator.serviceWorker.ready
      this.checkForUpdates(registration)
    }

    this.requestUpdate()
  }

  private setupEventListeners() {
    window.addEventListener('online', this.handleOnlineStatus.bind(this))
    window.addEventListener('offline', this.handleOnlineStatus.bind(this))
    
    if (this.capabilities.hasServiceWorker) {
      navigator.serviceWorker.addEventListener('message', this.handleServiceWorkerMessage.bind(this))
    }
  }

  private removeEventListeners() {
    window.removeEventListener('online', this.handleOnlineStatus.bind(this))
    window.removeEventListener('offline', this.handleOnlineStatus.bind(this))
    
    if (this.capabilities.hasServiceWorker) {
      navigator.serviceWorker.removeEventListener('message', this.handleServiceWorkerMessage.bind(this))
    }
  }

  private handleOnlineStatus() {
    this.capabilities = {
      ...this.capabilities,
      isOnline: navigator.onLine
    }
  }

  private handleServiceWorkerMessage(event: MessageEvent) {
    const { type, data } = event.data

    switch (type) {
      case 'cache-updated':
        this.updateCacheStatus()
        break
      case 'update-available':
        this.updateAvailable = true
        break
      case 'update-installing':
        this.isUpdating = true
        break
      case 'update-installed':
        this.isUpdating = false
        this.updateAvailable = false
        this.lastUpdateTime = new Date()
        break
    }
  }

  private async updateCacheStatus() {
    if (!this.capabilities.hasServiceWorker) return

    try {
      // Send message to service worker to get cache status
      const channel = new MessageChannel()
      const promise = new Promise<any>((resolve) => {
        channel.port1.onmessage = (event) => resolve(event.data)
      })

      navigator.serviceWorker.controller?.postMessage(
        { type: 'get-cache-status' },
        [channel.port2]
      )

      const status = await promise
      
      if (status) {
        this.capabilities.cacheStatus = status.isHealthy ? 'healthy' : 
                                      status.cacheSize > 0 ? 'degraded' : 'empty'
        this.capabilities.usedStorage = status.cacheSize || 0
      }
    } catch (error) {
      console.error('Failed to get cache status:', error)
      this.capabilities.cacheStatus = 'empty'
    }
  }

  private async checkForUpdates(registration: ServiceWorkerRegistration) {
    // Check for updates every 30 seconds
    setInterval(async () => {
      try {
        await registration.update()
      } catch (error) {
        console.error('Failed to check for updates:', error)
      }
    }, 30000)

    // Listen for waiting service worker
    if (registration.waiting) {
      this.updateAvailable = true
    }

    registration.addEventListener('updatefound', () => {
      const newWorker = registration.installing
      if (newWorker) {
        newWorker.addEventListener('statechange', () => {
          if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
            this.updateAvailable = true
          }
        })
      }
    })
  }

  private toggleExpanded() {
    this.expanded = !this.expanded
  }

  private async applyUpdate() {
    if (!this.updateAvailable) return

    this.isUpdating = true

    try {
      // Send skip waiting message to service worker
      navigator.serviceWorker.controller?.postMessage({ type: 'SKIP_WAITING' })
      
      // Reload the page to activate new service worker
      setTimeout(() => {
        window.location.reload()
      }, 1000)

    } catch (error) {
      console.error('Failed to apply update:', error)
      this.isUpdating = false
    }
  }

  private async clearCache() {
    if (!this.capabilities.hasServiceWorker) return

    try {
      const channel = new MessageChannel()
      const promise = new Promise<any>((resolve) => {
        channel.port1.onmessage = (event) => resolve(event.data)
      })

      navigator.serviceWorker.controller?.postMessage(
        { type: 'cleanup-cache' },
        [channel.port2]
      )

      await promise
      await this.updateCacheStatus()

      // Dispatch event for parent components
      this.dispatchEvent(new CustomEvent('cache-cleared', {
        bubbles: true,
        composed: true
      }))

    } catch (error) {
      console.error('Failed to clear cache:', error)
    }
  }

  private getStatusColor(): 'online' | 'offline' | 'degraded' {
    if (!this.capabilities.isOnline) return 'offline'
    if (this.capabilities.cacheStatus === 'empty') return 'degraded'
    return 'online'
  }

  private getStatusText(): string {
    if (!this.capabilities.isOnline) return 'Offline'
    if (!this.capabilities.hasServiceWorker) return 'Limited Features'
    return this.capabilities.isInstalled ? 'App Ready' : 'Web App'
  }

  private getStatusSubtext(): string {
    if (!this.capabilities.isOnline) return 'Using cached data'
    if (this.updateAvailable) return 'Update available'
    if (this.lastUpdateTime) return `Updated ${this.formatRelativeTime(this.lastUpdateTime)}`
    return `${this.capabilities.cacheStatus} cache`
  }

  private formatRelativeTime(date: Date): string {
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const minutes = Math.floor(diff / 60000)
    
    if (minutes < 1) return 'just now'
    if (minutes < 60) return `${minutes}m ago`
    
    const hours = Math.floor(minutes / 60)
    if (hours < 24) return `${hours}h ago`
    
    const days = Math.floor(hours / 24)
    return `${days}d ago`
  }

  private formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B'
    
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  render() {
    const statusColor = this.getStatusColor()
    const usagePercent = this.capabilities.storageQuota > 0 
      ? (this.capabilities.usedStorage / this.capabilities.storageQuota) * 100 
      : 0

    return html`
      <div class="pwa-status ${this.expanded ? 'expanded' : ''}">
        <div class="status-header" @click=${this.toggleExpanded}>
          <div class="status-info">
            <div class="status-icon ${statusColor}">
              <svg width="16" height="16" fill="currentColor" viewBox="0 0 20 20">
                ${statusColor === 'online' 
                  ? html`<path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/>`
                  : statusColor === 'offline'
                  ? html`<path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"/>`
                  : html`<path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>`
                }
              </svg>
              ${statusColor === 'online' ? html`<div class="status-pulse"></div>` : ''}
            </div>
            <div class="status-text">
              <p class="status-title">${this.getStatusText()}</p>
              <p class="status-subtitle">${this.getStatusSubtext()}</p>
            </div>
          </div>
          <svg class="expand-icon ${this.expanded ? 'expanded' : ''}" 
               fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd"/>
          </svg>
        </div>

        <div class="status-details ${this.expanded ? 'visible' : ''}">
          ${this.updateAvailable || this.isUpdating ? html`
            <div class="update-indicator">
              ${this.isUpdating ? html`
                <div class="loading-spinner"></div>
                Updating app...
              ` : html`
                <svg width="16" height="16" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clip-rule="evenodd"/>
                </svg>
                App update available
              `}
            </div>
          ` : ''}

          <div class="capabilities-grid">
            <div class="capability-item">
              <span class="capability-label">Service Worker</span>
              <span class="capability-status ${this.capabilities.hasServiceWorker ? 'enabled' : 'disabled'}">
                ${this.capabilities.hasServiceWorker ? 'Active' : 'Disabled'}
              </span>
            </div>
            <div class="capability-item">
              <span class="capability-label">Push Notifications</span>
              <span class="capability-status ${this.capabilities.hasPushSupport ? 'enabled' : 'disabled'}">
                ${this.capabilities.hasPushSupport ? 'Supported' : 'Not Available'}
              </span>
            </div>
            <div class="capability-item">
              <span class="capability-label">Background Sync</span>
              <span class="capability-status ${this.capabilities.hasBackgroundSync ? 'enabled' : 'disabled'}">
                ${this.capabilities.hasBackgroundSync ? 'Supported' : 'Not Available'}
              </span>
            </div>
            <div class="capability-item">
              <span class="capability-label">Offline Mode</span>
              <span class="capability-status ${this.capabilities.cacheStatus !== 'empty' ? 'enabled' : 'disabled'}">
                ${this.capabilities.cacheStatus !== 'empty' ? 'Ready' : 'Limited'}
              </span>
            </div>
          </div>

          ${this.capabilities.storageQuota > 0 ? html`
            <div class="storage-info">
              <div class="storage-text">
                <span>Storage Used</span>
                <span>${this.formatBytes(this.capabilities.usedStorage)} / ${this.formatBytes(this.capabilities.storageQuota)}</span>
              </div>
              <div class="storage-bar">
                <div class="storage-used" style="width: ${usagePercent}%"></div>
              </div>
            </div>
          ` : ''}

          <div class="actions-section">
            ${this.updateAvailable && !this.isUpdating ? html`
              <button class="action-button" @click=${this.applyUpdate}>
                Apply Update
              </button>
            ` : ''}
            
            ${this.capabilities.hasServiceWorker ? html`
              <button class="action-button secondary" @click=${this.clearCache}>
                Clear Cache
              </button>
            ` : ''}

            <button class="action-button secondary" @click=${() => window.location.reload()}>
              Refresh App
            </button>
          </div>
        </div>
      </div>
    `
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'pwa-status': PWAStatus
  }
}