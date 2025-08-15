import { LitElement, html, css } from 'lit'
import { customElement, state, property } from 'lit/decorators.js'
import { WebSocketService } from '../../services/websocket'

export interface ConnectionStats {
  quality: 'excellent' | 'good' | 'poor' | 'offline'
  latency: number
  stability: number
  messageRate: number
  reconnectAttempts: number
  isConnected: boolean
  lastMessageTime?: Date
  connectionState: string
}

@customElement('connection-monitor')
export class ConnectionMonitor extends LitElement {
  @property({ type: Boolean }) compact: boolean = false
  @property({ type: Boolean }) showControls: boolean = true
  @property({ type: Boolean }) autoReconnect: boolean = true
  @state() private stats: ConnectionStats = {
    quality: 'offline',
    latency: 0,
    stability: 0,
    messageRate: 0,
    reconnectAttempts: 0,
    isConnected: false,
    connectionState: 'disconnected'
  }
  @state() private lastUpdate: Date = new Date()
  @state() private showDetails: boolean = false
  @state() private reconnecting: boolean = false
  @state() private connectionHistory: Array<{timestamp: Date, quality: string, latency: number}> = []

  private wsService: WebSocketService = WebSocketService.getInstance()
  private updateInterval?: number
  private unsubscribeCallbacks: Array<() => void> = []

  static styles = css`
    :host {
      display: block;
    }

    .connection-monitor {
      background: var(--color-surface, #ffffff);
      border: 1px solid var(--color-border, #e2e8f0);
      border-radius: var(--radius-lg, 0.5rem);
      padding: var(--space-4, 1rem);
      transition: all var(--transition-normal, 0.3s);
    }

    .connection-monitor.compact {
      padding: var(--space-3, 0.75rem);
    }

    .connection-monitor.excellent {
      border-left: 4px solid var(--color-success, #10b981);
      background: linear-gradient(135deg, var(--color-surface, #ffffff), rgba(16, 185, 129, 0.02));
    }

    .connection-monitor.good {
      border-left: 4px solid var(--color-info, #0284c7);
      background: linear-gradient(135deg, var(--color-surface, #ffffff), rgba(2, 132, 199, 0.02));
    }

    .connection-monitor.poor {
      border-left: 4px solid var(--color-warning, #f59e0b);
      background: linear-gradient(135deg, var(--color-surface, #ffffff), rgba(245, 158, 11, 0.02));
    }

    .connection-monitor.offline {
      border-left: 4px solid var(--color-error, #ef4444);
      background: linear-gradient(135deg, var(--color-surface, #ffffff), rgba(239, 68, 68, 0.02));
    }

    .monitor-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: var(--space-3, 0.75rem);
    }

    .connection-status {
      display: flex;
      align-items: center;
      gap: var(--space-3, 0.75rem);
    }

    .status-indicator {
      width: 32px;
      height: 32px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
      flex-shrink: 0;
    }

    .compact .status-indicator {
      width: 24px;
      height: 24px;
    }

    .status-indicator.excellent {
      background: var(--color-success-light, #d1fae5);
      color: var(--color-success, #10b981);
    }

    .status-indicator.good {
      background: var(--color-info-light, #f0f9ff);
      color: var(--color-info, #0284c7);
    }

    .status-indicator.poor {
      background: var(--color-warning-light, #fef3c7);
      color: var(--color-warning, #f59e0b);
    }

    .status-indicator.offline {
      background: var(--color-error-light, #fef2f2);
      color: var(--color-error, #ef4444);
    }

    .status-pulse {
      position: absolute;
      top: -2px;
      left: -2px;
      right: -2px;
      bottom: -2px;
      border-radius: 50%;
      border: 2px solid currentColor;
      opacity: 0;
      animation: pulse 2s infinite;
    }

    .status-indicator.excellent .status-pulse,
    .status-indicator.good .status-pulse {
      animation: pulse 2s infinite;
    }

    .status-info {
      flex: 1;
      min-width: 0;
    }

    .status-title {
      font-size: var(--text-sm, 0.875rem);
      font-weight: 700;
      color: var(--color-text, #0f172a);
      margin: 0 0 var(--space-1, 0.25rem) 0;
    }

    .compact .status-title {
      font-size: var(--text-xs, 0.75rem);
    }

    .status-subtitle {
      font-size: var(--text-xs, 0.75rem);
      color: var(--color-text-muted, #64748b);
      margin: 0;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .toggle-details {
      background: none;
      border: 1px solid var(--color-border, #e2e8f0);
      border-radius: var(--radius-md, 0.375rem);
      color: var(--color-text-muted, #64748b);
      cursor: pointer;
      padding: var(--space-1, 0.25rem);
      width: 32px;
      height: 32px;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all var(--transition-normal, 0.3s);
    }

    .toggle-details:hover {
      background: var(--color-surface-secondary, #f8fafc);
      border-color: var(--color-border-focus, #3b82f6);
    }

    .toggle-icon {
      width: 16px;
      height: 16px;
      transition: transform var(--transition-normal, 0.3s);
    }

    .toggle-icon.expanded {
      transform: rotate(180deg);
    }

    .monitor-details {
      margin-top: var(--space-4, 1rem);
      border-top: 1px solid var(--color-border, #e2e8f0);
      padding-top: var(--space-4, 1rem);
      opacity: 0;
      max-height: 0;
      overflow: hidden;
      transition: all var(--transition-normal, 0.3s);
    }

    .monitor-details.visible {
      opacity: 1;
      max-height: 400px;
    }

    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: var(--space-4, 1rem);
      margin-bottom: var(--space-6, 1.5rem);
    }

    .stat-item {
      text-align: center;
      padding: var(--space-3, 0.75rem);
      background: var(--color-surface-secondary, #f8fafc);
      border-radius: var(--radius-md, 0.375rem);
      border: 1px solid var(--color-border, #e2e8f0);
    }

    .stat-value {
      font-size: var(--text-lg, 1.125rem);
      font-weight: 700;
      color: var(--color-text, #0f172a);
      margin: 0 0 var(--space-1, 0.25rem) 0;
    }

    .stat-label {
      font-size: var(--text-xs, 0.75rem);
      color: var(--color-text-muted, #64748b);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin: 0;
    }

    .connection-graph {
      height: 120px;
      background: var(--color-surface-secondary, #f8fafc);
      border: 1px solid var(--color-border, #e2e8f0);
      border-radius: var(--radius-md, 0.375rem);
      margin-bottom: var(--space-4, 1rem);
      position: relative;
      overflow: hidden;
    }

    .graph-line {
      position: absolute;
      bottom: 0;
      width: 2px;
      background: var(--color-primary, #1e40af);
      transition: height 0.3s ease;
      border-radius: 1px 1px 0 0;
    }

    .graph-line.excellent { background: var(--color-success, #10b981); }
    .graph-line.good { background: var(--color-info, #0284c7); }
    .graph-line.poor { background: var(--color-warning, #f59e0b); }
    .graph-line.offline { background: var(--color-error, #ef4444); }

    .monitor-controls {
      display: flex;
      gap: var(--space-2, 0.5rem);
      flex-wrap: wrap;
    }

    .control-button {
      padding: var(--space-2, 0.5rem) var(--space-3, 0.75rem);
      background: var(--color-primary, #1e40af);
      color: white;
      border: none;
      border-radius: var(--radius-md, 0.375rem);
      font-size: var(--text-xs, 0.75rem);
      font-weight: 600;
      cursor: pointer;
      transition: all var(--transition-normal, 0.3s);
      min-height: 36px;
      display: flex;
      align-items: center;
      gap: var(--space-1, 0.25rem);
    }

    .control-button:hover {
      background: var(--color-primary-dark, #1e3a8a);
      transform: translateY(-1px);
    }

    .control-button:disabled {
      background: var(--color-text-muted, #64748b);
      cursor: not-allowed;
      transform: none;
    }

    .control-button.secondary {
      background: transparent;
      color: var(--color-primary, #1e40af);
      border: 1px solid var(--color-primary, #1e40af);
    }

    .control-button.secondary:hover {
      background: var(--color-primary-alpha, rgba(30, 64, 175, 0.1));
    }

    .control-button.danger {
      background: var(--color-error, #ef4444);
    }

    .control-button.danger:hover {
      background: var(--color-error-dark, #dc2626);
    }

    .loading-spinner {
      width: 14px;
      height: 14px;
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-top: 2px solid white;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }

    .reconnect-status {
      margin-top: var(--space-3, 0.75rem);
      padding: var(--space-3, 0.75rem);
      background: var(--color-warning-light, #fef3c7);
      border: 1px solid var(--color-warning, #f59e0b);
      border-radius: var(--radius-md, 0.375rem);
      font-size: var(--text-sm, 0.875rem);
      color: var(--color-warning-dark, #92400e);
      display: flex;
      align-items: center;
      gap: var(--space-2, 0.5rem);
    }

    /* Mobile optimizations */
    @media (max-width: 768px) {
      .stats-grid {
        grid-template-columns: repeat(2, 1fr);
      }

      .monitor-controls {
        flex-direction: column;
      }

      .control-button {
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
        transform: scale(1.1);
      }
      100% {
        opacity: 0;
        transform: scale(1.2);
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

      .control-button,
      .toggle-details {
        transition: none;
      }
    }
  `

  connectedCallback() {
    super.connectedCallback()
    this.setupWebSocketListeners()
    this.startStatsUpdate()
    this.updateStats()
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    this.cleanup()
  }

  private setupWebSocketListeners() {
    // Connection state changes
    const onConnected = () => {
      this.updateStats()
      this.reconnecting = false
    }

    const onDisconnected = () => {
      this.updateStats()
      if (this.autoReconnect && this.wsService.getReconnectAttempts() === 0) {
        this.reconnecting = true
      }
    }

    const onConnectionQuality = (event: any) => {
      this.updateConnectionHistory(event.quality, event.latency || 0)
      this.updateStats()
    }

    const onMessage = () => {
      this.stats.lastMessageTime = new Date()
      this.lastUpdate = new Date()
    }

    // Subscribe to events
    this.wsService.on('connected', onConnected)
    this.wsService.on('disconnected', onDisconnected)
    this.wsService.on('connection-quality', onConnectionQuality)
    this.wsService.on('message', onMessage)

    // Store unsubscribe callbacks
    this.unsubscribeCallbacks = [
      () => this.wsService.off('connected', onConnected),
      () => this.wsService.off('disconnected', onDisconnected),
      () => this.wsService.off('connection-quality', onConnectionQuality),
      () => this.wsService.off('message', onMessage)
    ]
  }

  private updateConnectionHistory(quality: string, latency: number) {
    const entry = {
      timestamp: new Date(),
      quality,
      latency
    }

    this.connectionHistory.push(entry)
    
    // Keep only last 50 entries
    if (this.connectionHistory.length > 50) {
      this.connectionHistory.shift()
    }
  }

  private startStatsUpdate() {
    // Update stats every 2 seconds
    this.updateInterval = window.setInterval(() => {
      this.updateStats()
    }, 2000)
  }

  private updateStats() {
    const wsStats = this.wsService.getConnectionStats()
    
    this.stats = {
      ...wsStats,
      isConnected: this.wsService.isConnected(),
      connectionState: this.wsService.getConnectionState(),
      lastMessageTime: this.stats.lastMessageTime
    }

    this.lastUpdate = new Date()
  }

  private cleanup() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval)
    }

    this.unsubscribeCallbacks.forEach(unsubscribe => unsubscribe())
    this.unsubscribeCallbacks = []
  }

  private toggleDetails() {
    this.showDetails = !this.showDetails
  }

  private async handleReconnect() {
    this.reconnecting = true
    
    try {
      this.wsService.reconnect()
      
      // Reset reconnecting state after a delay
      setTimeout(() => {
        this.reconnecting = false
      }, 3000)

    } catch (error) {
      console.error('Manual reconnect failed:', error)
      this.reconnecting = false
    }
  }

  private handleDisconnect() {
    this.wsService.disconnect()
    this.reconnecting = false
  }

  private handleEnableHighFrequency() {
    this.wsService.enableHighFrequencyMode()
  }

  private handleEnableLowFrequency() {
    this.wsService.enableLowFrequencyMode()
  }

  private getStatusIcon() {
    if (this.reconnecting) {
      return html`<div class="loading-spinner"></div>`
    }

    switch (this.stats.quality) {
      case 'excellent':
        return html`
          <svg width="20" height="20" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/>
          </svg>
        `
      case 'good':
        return html`
          <svg width="20" height="20" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
          </svg>
        `
      case 'poor':
        return html`
          <svg width="20" height="20" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
          </svg>
        `
      case 'offline':
      default:
        return html`
          <svg width="20" height="20" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"/>
          </svg>
        `
    }
  }

  private getStatusText(): string {
    if (this.reconnecting) return 'Reconnecting...'
    
    const statusTexts = {
      excellent: 'Excellent Connection',
      good: 'Good Connection', 
      poor: 'Poor Connection',
      offline: 'Disconnected'
    }

    return statusTexts[this.stats.quality] || 'Unknown'
  }

  private getStatusSubtext(): string {
    if (this.reconnecting) {
      return `Attempt ${this.stats.reconnectAttempts + 1}`
    }

    if (this.stats.isConnected) {
      return `${Math.round(this.stats.latency)}ms • ${this.stats.messageRate.toFixed(1)} msg/s`
    }

    return this.stats.connectionState
  }

  private formatLatency(latency: number): string {
    return `${Math.round(latency)}ms`
  }

  private formatStability(stability: number): string {
    return `${Math.round(stability * 100)}%`
  }

  private formatMessageRate(rate: number): string {
    return `${rate.toFixed(1)}/s`
  }

  private renderConnectionGraph() {
    const maxHeight = 100
    const width = 200
    const entryWidth = width / Math.max(this.connectionHistory.length, 20)

    return html`
      <div class="connection-graph">
        ${this.connectionHistory.map((entry, index) => {
          const height = Math.max(10, (entry.latency / 500) * maxHeight)
          const left = index * entryWidth
          
          return html`
            <div 
              class="graph-line ${entry.quality}"
              style="height: ${height}px; left: ${left}px;"
              title="${entry.quality} - ${this.formatLatency(entry.latency)}"
            ></div>
          `
        })}
      </div>
    `
  }

  render() {
    const qualityClass = this.stats.quality

    return html`
      <div class="connection-monitor ${qualityClass} ${this.compact ? 'compact' : ''}">
        <div class="monitor-header">
          <div class="connection-status">
            <div class="status-indicator ${qualityClass}">
              ${this.getStatusIcon()}
              ${this.stats.isConnected ? html`<div class="status-pulse"></div>` : ''}
            </div>
            <div class="status-info">
              <h3 class="status-title">${this.getStatusText()}</h3>
              <p class="status-subtitle">${this.getStatusSubtext()}</p>
            </div>
          </div>
          
          ${!this.compact ? html`
            <button 
              class="toggle-details" 
              @click=${this.toggleDetails}
              title="Toggle connection details"
              aria-label="Toggle connection details"
            >
              <svg class="toggle-icon ${this.showDetails ? 'expanded' : ''}" 
                   fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd"/>
              </svg>
            </button>
          ` : ''}
        </div>

        ${this.showDetails && !this.compact ? html`
          <div class="monitor-details visible">
            <div class="stats-grid">
              <div class="stat-item">
                <p class="stat-value">${this.formatLatency(this.stats.latency)}</p>
                <p class="stat-label">Latency</p>
              </div>
              <div class="stat-item">
                <p class="stat-value">${this.formatStability(this.stats.stability)}</p>
                <p class="stat-label">Stability</p>
              </div>
              <div class="stat-item">
                <p class="stat-value">${this.formatMessageRate(this.stats.messageRate)}</p>
                <p class="stat-label">Message Rate</p>
              </div>
              <div class="stat-item">
                <p class="stat-value">${this.stats.reconnectAttempts}</p>
                <p class="stat-label">Reconnects</p>
              </div>
            </div>

            ${this.connectionHistory.length > 0 ? this.renderConnectionGraph() : ''}

            ${this.showControls ? html`
              <div class="monitor-controls">
                ${!this.stats.isConnected ? html`
                  <button 
                    class="control-button" 
                    @click=${this.handleReconnect}
                    ?disabled=${this.reconnecting}
                  >
                    ${this.reconnecting ? html`<div class="loading-spinner"></div>` : ''}
                    Reconnect
                  </button>
                ` : html`
                  <button 
                    class="control-button danger" 
                    @click=${this.handleDisconnect}
                  >
                    Disconnect
                  </button>
                `}
                
                ${this.stats.isConnected ? html`
                  <button 
                    class="control-button secondary" 
                    @click=${this.handleEnableHighFrequency}
                  >
                    High Frequency
                  </button>
                  <button 
                    class="control-button secondary" 
                    @click=${this.handleEnableLowFrequency}
                  >
                    Low Frequency
                  </button>
                ` : ''}
              </div>
            ` : ''}

            ${this.reconnecting ? html`
              <div class="reconnect-status">
                <div class="loading-spinner" style="border-color: currentColor; border-top-color: transparent;"></div>
                Attempting to reconnect (${this.stats.reconnectAttempts + 1}/${15})
              </div>
            ` : ''}
          </div>
        ` : ''}
      </div>
    `
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'connection-monitor': ConnectionMonitor
  }
}