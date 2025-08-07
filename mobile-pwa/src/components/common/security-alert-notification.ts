/**
 * Security Alert Notification Component
 * 
 * Provides real-time security alert notifications with different severity levels,
 * emergency controls, and risk heat map visualization.
 */

import { LitElement, html, css } from 'lit'
import { customElement, property, state } from 'lit/decorators.js'

export interface SecurityNotification {
  id: string
  type: 'critical' | 'high' | 'medium' | 'low' | 'emergency'
  title: string
  message: string
  timestamp: string
  actions?: {
    label: string
    action: string
    severity: 'primary' | 'secondary' | 'danger'
  }[]
  autoClose?: number // ms, 0 = no auto close
  persistent?: boolean
  component?: string
  metadata?: Record<string, any>
}

@customElement('security-alert-notification')
export class SecurityAlertNotification extends LitElement {
  @property({ type: Object }) declare notification: SecurityNotification
  @property({ type: Boolean }) declare compact: boolean
  @property({ type: Boolean }) declare mobile: boolean

  @state() private isVisible = true
  @state() private timeRemaining = 0
  @state() private autoCloseTimer: number | null = null

  static styles = css`
    :host {
      display: block;
      position: relative;
      margin-bottom: 1rem;
      z-index: 1000;
    }

    .notification {
      display: flex;
      align-items: flex-start;
      padding: 1rem;
      border-radius: 0.5rem;
      border-left: 4px solid;
      background: white;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
      transition: all 0.3s ease;
      position: relative;
      overflow: hidden;
    }

    .notification.critical {
      border-left-color: #dc2626;
      background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
      animation: criticalPulse 2s ease-in-out infinite;
    }

    .notification.high {
      border-left-color: #f59e0b;
      background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    }

    .notification.medium {
      border-left-color: #2563eb;
      background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    }

    .notification.low {
      border-left-color: #10b981;
      background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    }

    .notification.emergency {
      border-left-color: #991b1b;
      background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
      color: white;
      animation: emergencyFlash 1s ease-in-out infinite;
    }

    .notification.closing {
      transform: translateX(100%);
      opacity: 0;
    }

    .notification-icon {
      flex-shrink: 0;
      width: 24px;
      height: 24px;
      margin-right: 0.75rem;
    }

    .notification-icon.critical,
    .notification-icon.emergency {
      color: #dc2626;
    }

    .notification-icon.high {
      color: #f59e0b;
    }

    .notification-icon.medium {
      color: #2563eb;
    }

    .notification-icon.low {
      color: #10b981;
    }

    .notification.emergency .notification-icon {
      color: white;
    }

    .notification-content {
      flex: 1;
      min-width: 0;
    }

    .notification-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.25rem;
    }

    .notification-title {
      font-size: 0.875rem;
      font-weight: 600;
      color: #374151;
    }

    .notification.emergency .notification-title {
      color: white;
    }

    .notification-severity {
      padding: 0.125rem 0.5rem;
      border-radius: 0.25rem;
      font-size: 0.625rem;
      font-weight: 700;
      text-transform: uppercase;
    }

    .notification-severity.critical {
      background: #fecaca;
      color: #dc2626;
    }

    .notification-severity.high {
      background: #fed7aa;
      color: #d97706;
    }

    .notification-severity.medium {
      background: #bfdbfe;
      color: #2563eb;
    }

    .notification-severity.low {
      background: #bbf7d0;
      color: #059669;
    }

    .notification-severity.emergency {
      background: rgba(255, 255, 255, 0.2);
      color: white;
    }

    .notification-message {
      font-size: 0.75rem;
      line-height: 1.4;
      color: #6b7280;
      margin-bottom: 0.75rem;
    }

    .notification.emergency .notification-message {
      color: rgba(255, 255, 255, 0.9);
    }

    .notification-metadata {
      display: flex;
      gap: 1rem;
      font-size: 0.625rem;
      color: #9ca3af;
      margin-bottom: 0.75rem;
    }

    .notification.emergency .notification-metadata {
      color: rgba(255, 255, 255, 0.7);
    }

    .notification-actions {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      margin-bottom: 0.75rem;
    }

    .action-button {
      padding: 0.375rem 0.75rem;
      border-radius: 0.25rem;
      font-size: 0.75rem;
      font-weight: 500;
      border: none;
      cursor: pointer;
      transition: all 0.2s;
    }

    .action-button.primary {
      background: #2563eb;
      color: white;
    }

    .action-button.primary:hover {
      background: #1d4ed8;
    }

    .action-button.secondary {
      background: #f3f4f6;
      color: #374151;
      border: 1px solid #d1d5db;
    }

    .action-button.secondary:hover {
      background: #e5e7eb;
    }

    .action-button.danger {
      background: #dc2626;
      color: white;
    }

    .action-button.danger:hover {
      background: #b91c1c;
    }

    .notification.emergency .action-button.primary {
      background: rgba(255, 255, 255, 0.2);
      color: white;
      border: 1px solid rgba(255, 255, 255, 0.3);
    }

    .notification.emergency .action-button.primary:hover {
      background: rgba(255, 255, 255, 0.3);
    }

    .notification-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.625rem;
      color: #9ca3af;
    }

    .notification.emergency .notification-footer {
      color: rgba(255, 255, 255, 0.7);
    }

    .notification-time {
      font-weight: 500;
    }

    .auto-close-indicator {
      display: flex;
      align-items: center;
      gap: 0.25rem;
    }

    .close-timer {
      width: 40px;
      height: 4px;
      background: rgba(0, 0, 0, 0.1);
      border-radius: 2px;
      overflow: hidden;
    }

    .close-timer-fill {
      height: 100%;
      background: #dc2626;
      transition: width 0.1s linear;
    }

    .notification.emergency .close-timer {
      background: rgba(255, 255, 255, 0.2);
    }

    .notification.emergency .close-timer-fill {
      background: white;
    }

    .close-button {
      position: absolute;
      top: 0.5rem;
      right: 0.5rem;
      background: none;
      border: none;
      cursor: pointer;
      color: #6b7280;
      padding: 0.25rem;
      border-radius: 0.25rem;
      transition: all 0.2s;
    }

    .close-button:hover {
      background: rgba(0, 0, 0, 0.1);
    }

    .notification.emergency .close-button {
      color: rgba(255, 255, 255, 0.7);
    }

    .notification.emergency .close-button:hover {
      background: rgba(255, 255, 255, 0.1);
    }

    @keyframes criticalPulse {
      0%, 100% { 
        transform: scale(1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
      }
      50% { 
        transform: scale(1.02);
        box-shadow: 0 8px 15px -3px rgba(220, 38, 38, 0.3);
      }
    }

    @keyframes emergencyFlash {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.8; }
    }

    /* Mobile responsive */
    @media (max-width: 768px) {
      .notification {
        padding: 0.75rem;
      }

      .notification-actions {
        flex-direction: column;
      }

      .action-button {
        width: 100%;
        text-align: center;
      }

      .notification-footer {
        flex-direction: column;
        gap: 0.25rem;
        align-items: flex-start;
      }
    }

    /* Compact mode */
    :host([compact]) .notification {
      padding: 0.5rem 0.75rem;
    }

    :host([compact]) .notification-content {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    :host([compact]) .notification-message {
      margin-bottom: 0;
    }

    :host([compact]) .notification-actions,
    :host([compact]) .notification-metadata,
    :host([compact]) .notification-footer {
      display: none;
    }
  `

  constructor() {
    super()
    this.compact = false
    this.mobile = false
  }

  connectedCallback() {
    super.connectedCallback()
    this.setupAutoClose()
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    if (this.autoCloseTimer) {
      clearInterval(this.autoCloseTimer)
    }
  }

  private setupAutoClose() {
    if (!this.notification?.autoClose || this.notification.autoClose === 0 || this.notification.persistent) {
      return
    }

    this.timeRemaining = this.notification.autoClose / 1000
    
    this.autoCloseTimer = setInterval(() => {
      this.timeRemaining -= 0.1
      this.requestUpdate()

      if (this.timeRemaining <= 0) {
        this.handleClose()
      }
    }, 100) as any
  }

  private handleClose() {
    if (this.autoCloseTimer) {
      clearInterval(this.autoCloseTimer)
      this.autoCloseTimer = null
    }

    this.isVisible = false
    
    // Add closing animation class
    const notification = this.shadowRoot?.querySelector('.notification')
    notification?.classList.add('closing')

    // Remove after animation
    setTimeout(() => {
      this.dispatchEvent(new CustomEvent('notification-closed', {
        detail: { notification: this.notification },
        bubbles: true,
        composed: true
      }))
    }, 300)
  }

  private handleActionClick(action: string) {
    this.dispatchEvent(new CustomEvent('notification-action', {
      detail: { 
        notification: this.notification,
        action
      },
      bubbles: true,
      composed: true
    }))
  }

  private getNotificationIcon() {
    switch (this.notification?.type) {
      case 'critical':
      case 'emergency':
        return html`
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"/>
        `
      case 'high':
        return html`
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
        `
      case 'medium':
        return html`
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
        `
      case 'low':
        return html`
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
        `
      default:
        return html`
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M20.618 5.984A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>
        `
    }
  }

  render() {
    if (!this.isVisible || !this.notification) {
      return html``
    }

    const timeAgo = this.getTimeAgo(this.notification.timestamp)
    const progressPercentage = this.notification.autoClose 
      ? Math.max(0, (this.timeRemaining / (this.notification.autoClose / 1000)) * 100)
      : 0

    return html`
      <div class="notification ${this.notification.type}">
        <svg class="notification-icon ${this.notification.type}" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          ${this.getNotificationIcon()}
        </svg>

        <div class="notification-content">
          <div class="notification-header">
            <div class="notification-title">${this.notification.title}</div>
            <span class="notification-severity ${this.notification.type}">
              ${this.notification.type}
            </span>
          </div>

          <div class="notification-message">${this.notification.message}</div>

          ${this.notification.metadata && !this.compact ? html`
            <div class="notification-metadata">
              ${this.notification.component ? html`<span>Component: ${this.notification.component}</span>` : ''}
              ${Object.entries(this.notification.metadata).map(([key, value]) => html`
                <span>${key}: ${value}</span>
              `)}
            </div>
          ` : ''}

          ${this.notification.actions && this.notification.actions.length > 0 && !this.compact ? html`
            <div class="notification-actions">
              ${this.notification.actions.map(action => html`
                <button 
                  class="action-button ${action.severity}"
                  @click=${() => this.handleActionClick(action.action)}
                >
                  ${action.label}
                </button>
              `)}
            </div>
          ` : ''}

          ${!this.compact ? html`
            <div class="notification-footer">
              <div class="notification-time">${timeAgo}</div>
              ${this.notification.autoClose && this.notification.autoClose > 0 ? html`
                <div class="auto-close-indicator">
                  <span>Auto-close in ${Math.ceil(this.timeRemaining)}s</span>
                  <div class="close-timer">
                    <div class="close-timer-fill" style="width: ${progressPercentage}%"></div>
                  </div>
                </div>
              ` : ''}
            </div>
          ` : ''}
        </div>

        ${!this.notification.persistent ? html`
          <button class="close-button" @click=${this.handleClose} title="Close notification">
            <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
            </svg>
          </button>
        ` : ''}
      </div>
    `
  }

  private getTimeAgo(timestamp: string): string {
    const now = Date.now()
    const time = new Date(timestamp).getTime()
    const diffInSeconds = Math.floor((now - time) / 1000)

    if (diffInSeconds < 60) {
      return `${diffInSeconds}s ago`
    } else if (diffInSeconds < 3600) {
      return `${Math.floor(diffInSeconds / 60)}m ago`
    } else if (diffInSeconds < 86400) {
      return `${Math.floor(diffInSeconds / 3600)}h ago`
    } else {
      return new Date(timestamp).toLocaleDateString()
    }
  }
}