import { LitElement, html, css } from 'lit'
import { customElement, state, property } from 'lit/decorators.js'
import { WebSocketService } from '../services/websocket'

interface CriticalNotification {
  id: string
  type: 'system_failure' | 'security_violation' | 'resource_exhaustion' | 'architectural_conflict' | 'agent_deadlock'
  priority: 'critical' | 'high'
  title: string
  message: string
  context: Record<string, any>
  actions: NotificationAction[]
  timestamp: string
  acknowledged: boolean
  escalated: boolean
  estimatedResolutionTime?: string
}

interface NotificationAction {
  id: string
  label: string
  command: string
  priority: 'primary' | 'secondary' | 'danger'
  requiresConfirmation?: boolean
}

interface NotificationPreferences {
  enabled: boolean
  criticalOnly: boolean
  quietHours: { start: string; end: string } | null
  soundEnabled: boolean
  vibrationEnabled: boolean
  showPreview: boolean
  maxNotifications: number
  autoAcknowledgeAfter: number // minutes
}

@customElement('mobile-notification-system')
export class MobileNotificationSystem extends LitElement {
  @property({ type: Boolean }) declare enabled: boolean
  @property({ type: String }) declare role: 'developer' | 'manager' | 'architect'

  @state() private declare notifications: CriticalNotification[]
  @state() private declare preferences: NotificationPreferences
  @state() private declare permissionStatus: 'default' | 'granted' | 'denied'
  @state() private declare serviceWorkerReady: boolean
  @state() private declare connectionStatus: 'connected' | 'disconnected' | 'reconnecting'
  @state() private declare showSettings: boolean
  @state() private declare unreadCount: number

  private websocketService: WebSocketService
  private notificationQueue: CriticalNotification[] = []
  private acknowledgmentTimers: Map<string, number> = new Map()
  private lastNotificationSound = 0
  private soundCooldown = 5000 // 5 seconds between sounds

  static styles = css`
    :host {
      position: fixed;
      top: 0;
      right: 0;
      z-index: 2000;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    .notification-container {
      position: fixed;
      top: 1rem;
      right: 1rem;
      max-width: 380px;
      z-index: 2001;
    }

    .notification-badge {
      position: fixed;
      top: 1rem;
      right: 1rem;
      background: #ef4444;
      color: white;
      border-radius: 50%;
      width: 24px;
      height: 24px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.75rem;
      font-weight: 700;
      z-index: 2002;
      animation: pulse 2s infinite;
      cursor: pointer;
      box-shadow: 0 2px 8px rgba(239, 68, 68, 0.4);
    }

    .notification-card {
      background: white;
      border-radius: 16px;
      margin-bottom: 0.75rem;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
      border: 1px solid #e5e7eb;
      overflow: hidden;
      animation: slideIn 0.3s ease-out;
      backdrop-filter: blur(10px);
    }

    .notification-card.critical {
      border-left: 4px solid #ef4444;
      background: linear-gradient(135deg, #ffffff 0%, #fef2f2 100%);
    }

    .notification-card.high {
      border-left: 4px solid #f59e0b;
      background: linear-gradient(135deg, #ffffff 0%, #fffbeb 100%);
    }

    .notification-header {
      padding: 1rem 1rem 0.5rem;
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
    }

    .notification-type {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .notification-icon {
      font-size: 1.25rem;
    }

    .notification-meta {
      flex: 1;
      min-width: 0;
    }

    .notification-title {
      font-weight: 700;
      color: #111827;
      font-size: 0.9rem;
      line-height: 1.3;
      margin-bottom: 0.25rem;
    }

    .notification-timestamp {
      font-size: 0.75rem;
      color: #6b7280;
    }

    .notification-priority {
      padding: 0.25rem 0.5rem;
      border-radius: 6px;
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      margin-left: 0.5rem;
    }

    .notification-priority.critical {
      background: #fecaca;
      color: #dc2626;
    }

    .notification-priority.high {
      background: #fed7aa;
      color: #ea580c;
    }

    .notification-content {
      padding: 0 1rem 1rem;
    }

    .notification-message {
      font-size: 0.875rem;
      color: #374151;
      line-height: 1.4;
      margin-bottom: 1rem;
    }

    .notification-context {
      background: #f9fafb;
      border-radius: 8px;
      padding: 0.75rem;
      margin-bottom: 1rem;
      font-size: 0.8rem;
      color: #6b7280;
      border: 1px solid #e5e7eb;
    }

    .context-item {
      display: flex;
      justify-content: space-between;
      margin-bottom: 0.25rem;
    }

    .context-item:last-child {
      margin-bottom: 0;
    }

    .context-key {
      font-weight: 600;
      color: #374151;
    }

    .notification-actions {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
    }

    .notification-action {
      padding: 0.5rem 1rem;
      border-radius: 8px;
      border: none;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      flex: 1;
      min-width: 0;
    }

    .notification-action.primary {
      background: #3b82f6;
      color: white;
    }

    .notification-action.primary:hover {
      background: #2563eb;
    }

    .notification-action.secondary {
      background: #f3f4f6;
      color: #374151;
    }

    .notification-action.secondary:hover {
      background: #e5e7eb;
    }

    .notification-action.danger {
      background: #ef4444;
      color: white;
    }

    .notification-action.danger:hover {
      background: #dc2626;
    }

    .notification-dismiss {
      background: none;
      border: none;
      color: #6b7280;
      cursor: pointer;
      padding: 0.25rem;
      border-radius: 4px;
      transition: all 0.2s;
    }

    .notification-dismiss:hover {
      background: #f3f4f6;
      color: #374151;
    }

    .settings-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.7);
      z-index: 2100;
      display: flex;
      align-items: center;
      justify-content: center;
      backdrop-filter: blur(5px);
    }

    .settings-panel {
      background: white;
      border-radius: 20px;
      padding: 2rem;
      max-width: 90%;
      width: 400px;
      max-height: 80vh;
      overflow-y: auto;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }

    .settings-header {
      text-align: center;
      margin-bottom: 2rem;
    }

    .settings-title {
      font-size: 1.5rem;
      font-weight: 700;
      color: #111827;
      margin-bottom: 0.5rem;
    }

    .settings-section {
      margin-bottom: 2rem;
    }

    .settings-section-title {
      font-weight: 600;
      color: #111827;
      margin-bottom: 1rem;
      font-size: 1rem;
    }

    .setting-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.75rem 0;
      border-bottom: 1px solid #f3f4f6;
    }

    .setting-item:last-child {
      border-bottom: none;
    }

    .setting-label {
      flex: 1;
      font-size: 0.9rem;
      color: #374151;
    }

    .setting-description {
      font-size: 0.8rem;
      color: #6b7280;
      margin-top: 0.25rem;
    }

    .toggle-switch {
      position: relative;
      width: 48px;
      height: 24px;
      background: #d1d5db;
      border-radius: 12px;
      cursor: pointer;
      transition: background 0.2s;
    }

    .toggle-switch.active {
      background: #3b82f6;
    }

    .toggle-switch::after {
      content: '';
      position: absolute;
      top: 2px;
      left: 2px;
      width: 20px;
      height: 20px;
      background: white;
      border-radius: 50%;
      transition: transform 0.2s;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    .toggle-switch.active::after {
      transform: translateX(24px);
    }

    .time-input {
      padding: 0.5rem;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      font-size: 0.9rem;
      width: 80px;
    }

    .number-input {
      padding: 0.5rem;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      font-size: 0.9rem;
      width: 80px;
      text-align: center;
    }

    .settings-actions {
      display: flex;
      gap: 1rem;
      justify-content: center;
      margin-top: 2rem;
    }

    .settings-button {
      padding: 0.75rem 1.5rem;
      border-radius: 8px;
      border: none;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }

    .settings-button.primary {
      background: #3b82f6;
      color: white;
    }

    .settings-button.primary:hover {
      background: #2563eb;
    }

    .settings-button.secondary {
      background: #f3f4f6;
      color: #374151;
    }

    .settings-button.secondary:hover {
      background: #e5e7eb;
    }

    .connection-status {
      position: fixed;
      bottom: 1rem;
      right: 1rem;
      padding: 0.5rem 1rem;
      border-radius: 8px;
      font-size: 0.8rem;
      font-weight: 600;
      z-index: 2003;
    }

    .connection-status.connected {
      background: #d1fae5;
      color: #065f46;
    }

    .connection-status.disconnected {
      background: #fee2e2;
      color: #991b1b;
    }

    .connection-status.reconnecting {
      background: #fef3c7;
      color: #92400e;
    }

    /* Animations */
    @keyframes slideIn {
      0% {
        opacity: 0;
        transform: translateX(100%);
      }
      100% {
        opacity: 1;
        transform: translateX(0);
      }
    }

    @keyframes pulse {
      0%, 100% {
        transform: scale(1);
        opacity: 1;
      }
      50% {
        transform: scale(1.1);
        opacity: 0.8;
      }
    }

    /* Mobile optimizations */
    @media (max-width: 768px) {
      .notification-container {
        max-width: calc(100vw - 2rem);
        right: 1rem;
        left: 1rem;
      }

      .settings-panel {
        width: calc(100vw - 2rem);
        margin: 1rem;
      }
    }

    /* Accessibility */
    @media (prefers-reduced-motion: reduce) {
      * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
      }
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
      .notification-card {
        background: #1f2937;
        border-color: #374151;
        color: #f9fafb;
      }

      .notification-title {
        color: #f9fafb;
      }

      .notification-message {
        color: #d1d5db;
      }

      .notification-context {
        background: #374151;
        border-color: #4b5563;
        color: #d1d5db;
      }

      .settings-panel {
        background: #1f2937;
        color: #f9fafb;
      }

      .settings-title {
        color: #f9fafb;
      }
    }
  `

  constructor() {
    super()
    this.enabled = true
    this.role = 'developer'
    this.notifications = []
    this.preferences = this.getDefaultPreferences()
    this.permissionStatus = 'default'
    this.serviceWorkerReady = false
    this.connectionStatus = 'disconnected'
    this.showSettings = false
    this.unreadCount = 0
    this.websocketService = WebSocketService.getInstance()
  }

  connectedCallback() {
    super.connectedCallback()
    this.initializeNotificationSystem()
    this.setupWebSocketListeners()
    this.loadPreferences()
    this.checkPermissions()
    this.registerServiceWorker()
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    this.cleanupTimers()
  }

  private getDefaultPreferences(): NotificationPreferences {
    return {
      enabled: true,
      criticalOnly: true,
      quietHours: null,
      soundEnabled: true,
      vibrationEnabled: true,
      showPreview: true,
      maxNotifications: 5,
      autoAcknowledgeAfter: 30
    }
  }

  private async initializeNotificationSystem() {
    if (!this.enabled) return

    try {
      // Load existing notifications from API
      const response = await fetch('/api/hive/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: '/hive:notifications --mobile --critical-only'
        })
      })

      if (response.ok) {
        const result = await response.json()
        if (result.success && result.result.notifications) {
          this.notifications = result.result.notifications
          this.updateUnreadCount()
        }
      }
    } catch (error) {
      console.warn('Failed to load existing notifications:', error)
    }
  }

  private setupWebSocketListeners() {
    this.websocketService.on('notification', (data) => {
      this.handleIncomingNotification(data)
    })

    this.websocketService.on('system_alert', (data) => {
      this.handleSystemAlert(data)
    })

    this.websocketService.on('connect', () => {
      this.connectionStatus = 'connected'
    })

    this.websocketService.on('disconnect', () => {
      this.connectionStatus = 'disconnected'
    })

    this.websocketService.on('reconnecting', () => {
      this.connectionStatus = 'reconnecting'
    })
  }

  private async checkPermissions() {
    if ('Notification' in window) {
      this.permissionStatus = Notification.permission
      
      if (this.permissionStatus === 'default') {
        // Request permission on first critical notification
        this.permissionStatus = await Notification.requestPermission()
      }
    }
  }

  private async registerServiceWorker() {
    if ('serviceWorker' in navigator) {
      try {
        const registration = await navigator.serviceWorker.register('/sw.js')
        this.serviceWorkerReady = true
        
        // Listen for messages from service worker
        navigator.serviceWorker.addEventListener('message', (event) => {
          if (event.data.type === 'notification-click') {
            this.handleNotificationClick(event.data.notificationId)
          }
        })
      } catch (error) {
        console.warn('Service worker registration failed:', error)
      }
    }
  }

  private handleIncomingNotification(data: any) {
    if (!this.preferences.enabled) return

    const notification: CriticalNotification = {
      id: data.id || this.generateId(),
      type: data.type || 'system_failure',
      priority: data.priority || 'high',
      title: data.title,
      message: data.message,
      context: data.context || {},
      actions: data.actions || [],
      timestamp: new Date().toISOString(),
      acknowledged: false,
      escalated: false,
      estimatedResolutionTime: data.estimatedResolutionTime
    }

    // Filter based on preferences
    if (this.preferences.criticalOnly && notification.priority !== 'critical') {
      return
    }

    // Check quiet hours
    if (this.isQuietHours()) {
      this.notificationQueue.push(notification)
      return
    }

    this.addNotification(notification)
  }

  private handleSystemAlert(data: any) {
    // System alerts are always critical
    const notification: CriticalNotification = {
      id: this.generateId(),
      type: 'system_failure',
      priority: 'critical',
      title: 'System Alert',
      message: data.message || 'Critical system event detected',
      context: data,
      actions: [
        {
          id: 'investigate',
          label: 'Investigate',
          command: '/hive:investigate-alert',
          priority: 'primary'
        },
        {
          id: 'escalate',
          label: 'Escalate',
          command: '/hive:escalate-alert',
          priority: 'danger'
        }
      ],
      timestamp: new Date().toISOString(),
      acknowledged: false,
      escalated: false
    }

    this.addNotification(notification)
  }

  private addNotification(notification: CriticalNotification) {
    // Remove oldest notifications if exceeding max
    if (this.notifications.length >= this.preferences.maxNotifications) {
      this.notifications = this.notifications.slice(-(this.preferences.maxNotifications - 1))
    }

    this.notifications = [...this.notifications, notification]
    this.updateUnreadCount()

    // Show native notification if permission granted
    if (this.permissionStatus === 'granted' && this.preferences.showPreview) {
      this.showNativeNotification(notification)
    }

    // Play sound
    if (this.preferences.soundEnabled) {
      this.playNotificationSound(notification.priority)
    }

    // Vibrate
    if (this.preferences.vibrationEnabled && 'vibrate' in navigator) {
      const pattern = notification.priority === 'critical' ? [200, 100, 200] : [100]
      navigator.vibrate(pattern)
    }

    // Auto-acknowledge timer
    if (this.preferences.autoAcknowledgeAfter > 0) {
      const timer = window.setTimeout(() => {
        this.acknowledgeNotification(notification.id)
      }, this.preferences.autoAcknowledgeAfter * 60 * 1000)
      
      this.acknowledgmentTimers.set(notification.id, timer)
    }
  }

  private showNativeNotification(notification: CriticalNotification) {
    if (this.serviceWorkerReady && 'serviceWorker' in navigator) {
      // Use service worker for persistent notifications
      navigator.serviceWorker.ready.then(registration => {
        registration.showNotification(notification.title, {
          body: notification.message,
          icon: '/icons/icon-192x192.png',
          badge: '/icons/badge-72x72.png',
          tag: notification.id,
          requireInteraction: notification.priority === 'critical',
          data: { notificationId: notification.id },
          actions: notification.actions.slice(0, 2).map(action => ({
            action: action.id,
            title: action.label
          }))
        })
      })
    } else {
      // Fallback to regular notifications
      new Notification(notification.title, {
        body: notification.message,
        icon: '/icons/icon-192x192.png',
        tag: notification.id,
        requireInteraction: notification.priority === 'critical'
      })
    }
  }

  private playNotificationSound(priority: 'critical' | 'high') {
    const now = Date.now()
    if (now - this.lastNotificationSound < this.soundCooldown) return

    this.lastNotificationSound = now
    
    // Use different sounds for different priorities
    const soundFile = priority === 'critical' ? '/sounds/critical.mp3' : '/sounds/notification.mp3'
    
    const audio = new Audio(soundFile)
    audio.volume = 0.7
    audio.play().catch(() => {
      // Fallback to system beep
      const beep = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSYFJHfH8N2QQAoUXrTp66hVFApGn+DyvmEcBjiR1/LNeSYFJHfH8N2QQAoUXrTp66hVFApGn+DyvmEcBjmT2fLFeycBK4LN9tieQwYY...')
      beep.play().catch(() => {})
    })
  }

  private isQuietHours(): boolean {
    if (!this.preferences.quietHours) return false

    const now = new Date()
    const currentTime = now.getHours() * 60 + now.getMinutes()
    
    const [startHour, startMin] = this.preferences.quietHours.start.split(':').map(Number)
    const [endHour, endMin] = this.preferences.quietHours.end.split(':').map(Number)
    
    const startTime = startHour * 60 + startMin
    const endTime = endHour * 60 + endMin
    
    if (startTime < endTime) {
      return currentTime >= startTime && currentTime <= endTime
    } else {
      // Overnight quiet hours
      return currentTime >= startTime || currentTime <= endTime
    }
  }

  private updateUnreadCount() {
    this.unreadCount = this.notifications.filter(n => !n.acknowledged).length
  }

  private async executeNotificationAction(notificationId: string, action: NotificationAction) {
    if (action.requiresConfirmation) {
      const confirmed = confirm(`Are you sure you want to: ${action.label}?`)
      if (!confirmed) return
    }

    try {
      const response = await fetch('/api/hive/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: action.command,
          context: { notificationId }
        })
      })

      const result = await response.json()
      if (result.success) {
        this.acknowledgeNotification(notificationId)
        this.showFeedback(`Action completed: ${action.label}`)
      } else {
        this.showFeedback(`Action failed: ${action.label}`, 'error')
      }
    } catch (error) {
      this.showFeedback(`Action failed: ${action.label}`, 'error')
    }
  }

  private acknowledgeNotification(id: string) {
    const index = this.notifications.findIndex(n => n.id === id)
    if (index !== -1) {
      this.notifications[index].acknowledged = true
      this.updateUnreadCount()
      
      // Clear auto-acknowledge timer
      const timer = this.acknowledgmentTimers.get(id)
      if (timer) {
        clearTimeout(timer)
        this.acknowledgmentTimers.delete(id)
      }
      
      this.requestUpdate()
    }
  }

  private dismissNotification(id: string) {
    this.notifications = this.notifications.filter(n => n.id !== id)
    this.updateUnreadCount()
    
    // Clear timer
    const timer = this.acknowledgmentTimers.get(id)
    if (timer) {
      clearTimeout(timer)
      this.acknowledgmentTimers.delete(id)
    }
  }

  private handleNotificationClick(notificationId: string) {
    // Bring app to foreground and focus on notification
    window.focus()
    const notification = this.notifications.find(n => n.id === notificationId)
    if (notification) {
      this.acknowledgeNotification(notificationId)
    }
  }

  private showFeedback(message: string, type: 'success' | 'error' = 'success') {
    // Dispatch event for feedback system
    this.dispatchEvent(new CustomEvent('show-feedback', {
      detail: { message, type },
      bubbles: true,
      composed: true
    }))
  }

  private generateId(): string {
    return Math.random().toString(36).substr(2, 9)
  }

  private loadPreferences() {
    const saved = localStorage.getItem('mobile-notification-preferences')
    if (saved) {
      try {
        this.preferences = { ...this.preferences, ...JSON.parse(saved) }
      } catch (error) {
        console.warn('Failed to load notification preferences:', error)
      }
    }
  }

  private savePreferences() {
    localStorage.setItem('mobile-notification-preferences', JSON.stringify(this.preferences))
  }

  private cleanupTimers() {
    this.acknowledgmentTimers.forEach(timer => clearTimeout(timer))
    this.acknowledgmentTimers.clear()
  }

  private formatTimestamp(timestamp: string): string {
    const date = new Date(timestamp)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    
    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    
    const diffHours = Math.floor(diffMins / 60)
    if (diffHours < 24) return `${diffHours}h ago`
    
    return date.toLocaleDateString()
  }

  openSettings() {
    this.showSettings = true
  }

  closeSettings() {
    this.showSettings = false
    this.savePreferences()
  }

  render() {
    if (!this.enabled) return html``

    return html`
      ${this.unreadCount > 0 ? html`
        <div class="notification-badge" @click=${this.openSettings}>
          ${this.unreadCount}
        </div>
      ` : ''}
      
      <div class="notification-container">
        ${this.notifications
          .filter(n => !n.acknowledged)
          .slice(-3) // Show only last 3 unacknowledged
          .map(notification => this.renderNotification(notification))}
      </div>
      
      ${this.showSettings ? this.renderSettings() : ''}
      
      ${this.connectionStatus !== 'connected' ? html`
        <div class="connection-status ${this.connectionStatus}">
          ${this.connectionStatus === 'reconnecting' ? 'Reconnecting...' :
            this.connectionStatus === 'disconnected' ? 'Offline' : ''}
        </div>
      ` : ''}
    `
  }

  private renderNotification(notification: CriticalNotification) {
    const priorityIcon = {
      'system_failure': '⚠️',
      'security_violation': '🔒',
      'resource_exhaustion': '📈',
      'architectural_conflict': '🏧',
      'agent_deadlock': '🤖'
    }

    return html`
      <div class="notification-card ${notification.priority}">
        <div class="notification-header">
          <div class="notification-type">
            <span class="notification-icon">
              ${priorityIcon[notification.type] || '⚠️'}
            </span>
            <div class="notification-meta">
              <div class="notification-title">${notification.title}</div>
              <div class="notification-timestamp">
                ${this.formatTimestamp(notification.timestamp)}
              </div>
            </div>
          </div>
          <div class="notification-priority ${notification.priority}">
            ${notification.priority}
          </div>
          <button class="notification-dismiss" @click=${() => this.dismissNotification(notification.id)}>
            ✕
          </button>
        </div>
        
        <div class="notification-content">
          <div class="notification-message">${notification.message}</div>
          
          ${Object.keys(notification.context).length > 0 ? html`
            <div class="notification-context">
              ${Object.entries(notification.context).map(([key, value]) => html`
                <div class="context-item">
                  <span class="context-key">${key}:</span>
                  <span>${String(value).substring(0, 50)}${String(value).length > 50 ? '...' : ''}</span>
                </div>
              `)}
            </div>
          ` : ''}
          
          ${notification.actions.length > 0 ? html`
            <div class="notification-actions">
              ${notification.actions.map(action => html`
                <button 
                  class="notification-action ${action.priority}"
                  @click=${() => this.executeNotificationAction(notification.id, action)}
                >
                  ${action.label}
                </button>
              `)}
            </div>
          ` : ''}
        </div>
      </div>
    `
  }

  private renderSettings() {
    return html`
      <div class="settings-overlay" @click=${this.closeSettings}>
        <div class="settings-panel" @click=${(e: Event) => e.stopPropagation()}>
          <div class="settings-header">
            <div class="settings-title">🔔 Notification Settings</div>
          </div>
          
          <div class="settings-section">
            <div class="settings-section-title">General</div>
            
            <div class="setting-item">
              <div class="setting-label">
                Enable Notifications
                <div class="setting-description">Receive mobile notifications for critical events</div>
              </div>
              <div class="toggle-switch ${this.preferences.enabled ? 'active' : ''}" 
                   @click=${() => { this.preferences.enabled = !this.preferences.enabled }}>
              </div>
            </div>
            
            <div class="setting-item">
              <div class="setting-label">
                Critical Only
                <div class="setting-description">Only show critical priority notifications</div>
              </div>
              <div class="toggle-switch ${this.preferences.criticalOnly ? 'active' : ''}" 
                   @click=${() => { this.preferences.criticalOnly = !this.preferences.criticalOnly }}>
              </div>
            </div>
          </div>
          
          <div class="settings-section">
            <div class="settings-section-title">Feedback</div>
            
            <div class="setting-item">
              <div class="setting-label">
                Sound Alerts
                <div class="setting-description">Play sound for new notifications</div>
              </div>
              <div class="toggle-switch ${this.preferences.soundEnabled ? 'active' : ''}" 
                   @click=${() => { this.preferences.soundEnabled = !this.preferences.soundEnabled }}>
              </div>
            </div>
            
            <div class="setting-item">
              <div class="setting-label">
                Vibration
                <div class="setting-description">Vibrate device for notifications</div>
              </div>
              <div class="toggle-switch ${this.preferences.vibrationEnabled ? 'active' : ''}" 
                   @click=${() => { this.preferences.vibrationEnabled = !this.preferences.vibrationEnabled }}>
              </div>
            </div>
          </div>
          
          <div class="settings-section">
            <div class="settings-section-title">Behavior</div>
            
            <div class="setting-item">
              <div class="setting-label">
                Max Notifications
                <div class="setting-description">Maximum number of visible notifications</div>
              </div>
              <input type="number" class="number-input" min="1" max="10"
                     .value=${this.preferences.maxNotifications}
                     @change=${(e: InputEvent) => {
                       this.preferences.maxNotifications = parseInt((e.target as HTMLInputElement).value) || 5
                     }}>
            </div>
            
            <div class="setting-item">
              <div class="setting-label">
                Auto-acknowledge (minutes)
                <div class="setting-description">Automatically dismiss after this time</div>
              </div>
              <input type="number" class="number-input" min="0" max="60"
                     .value=${this.preferences.autoAcknowledgeAfter}
                     @change=${(e: InputEvent) => {
                       this.preferences.autoAcknowledgeAfter = parseInt((e.target as HTMLInputElement).value) || 30
                     }}>
            </div>
          </div>
          
          <div class="settings-actions">
            <button class="settings-button secondary" @click=${this.closeSettings}>
              Cancel
            </button>
            <button class="settings-button primary" @click=${this.closeSettings}>
              Save Settings
            </button>
          </div>
        </div>
      </div>
    `
  }
}
