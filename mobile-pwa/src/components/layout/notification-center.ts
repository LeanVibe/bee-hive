import { LitElement, html, css } from 'lit'
import { customElement, state } from 'lit/decorators.js'
import { getNotificationService, NotificationService } from '../../services'
import '../common/loading-spinner'

export interface NotificationItem {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message: string
  timestamp: string
  read: boolean
  actions?: Array<{
    label: string
    action: () => void
    style?: 'primary' | 'secondary' | 'danger'
  }>
}

@customElement('notification-center')
export class NotificationCenter extends LitElement {
  private notificationService: NotificationService = getNotificationService()
  private isMobile: boolean = this.detectMobile()
  private isIOS: boolean = this.detectIOS()

  // Reactive properties defined using static properties to avoid class field shadowing
  static properties = {
    notifications: { state: true },
    isOpen: { state: true },
    isLoading: { state: true },
    unreadCount: { state: true },
    permissionState: { state: true },
    fcmSupported: { state: true },
    isOnline: { state: true }
  }

  // Declare property types for TypeScript
  declare notifications: NotificationItem[]
  declare isOpen: boolean
  declare isLoading: boolean
  declare unreadCount: number
  declare permissionState: NotificationPermission
  declare fcmSupported: boolean
  declare isOnline: boolean

  // Initialize properties in constructor
  constructor() {
    super()
    this.notifications = []
    this.isOpen = false
    this.isLoading = false
    this.unreadCount = 0
    this.permissionState = 'default'
    this.fcmSupported = false
    this.isOnline = navigator.onLine
  }

  private detectMobile(): boolean {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
  }

  private detectIOS(): boolean {
    return /iPad|iPhone|iPod/.test(navigator.userAgent)
  }

  static styles = css`
    :host {
      position: relative;
      display: inline-block;
    }

    .notification-trigger {
      position: relative;
      background: none;
      border: none;
      color: #6b7280;
      cursor: pointer;
      padding: 0.5rem;
      border-radius: 0.5rem;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .notification-trigger:hover {
      background: #f3f4f6;
      color: #374151;
    }

    .notification-badge {
      position: absolute;
      top: 0.25rem;
      right: 0.25rem;
      background: #ef4444;
      color: white;
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.125rem 0.375rem;
      border-radius: 0.75rem;
      min-width: 1.25rem;
      height: 1.25rem;
      display: flex;
      align-items: center;
      justify-content: center;
      transform: scale(0);
      transition: transform 0.2s ease;
    }

    .notification-badge.show {
      transform: scale(1);
    }

    .notification-panel {
      position: absolute;
      top: 100%;
      right: 0;
      width: 400px;
      max-width: 90vw;
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 0.75rem;
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
      z-index: 1000;
      opacity: 0;
      visibility: hidden;
      transform: translateY(-10px) scale(0.95);
      transition: all 0.2s ease;
      max-height: 600px;
      overflow: hidden;
    }

    .notification-panel.open {
      opacity: 1;
      visibility: visible;
      transform: translateY(0) scale(1);
    }

    .panel-header {
      padding: 1.5rem;
      border-bottom: 1px solid #e5e7eb;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .panel-title {
      font-size: 1.125rem;
      font-weight: 600;
      color: #111827;
      margin: 0;
    }

    .header-actions {
      display: flex;
      gap: 0.5rem;
    }

    .action-button {
      background: none;
      border: none;
      color: #6b7280;
      cursor: pointer;
      padding: 0.25rem;
      border-radius: 0.375rem;
      transition: all 0.2s ease;
      font-size: 0.75rem;
    }

    .action-button:hover {
      background: #f3f4f6;
      color: #374151;
    }

    .action-button.primary {
      background: #3b82f6;
      color: white;
      padding: 0.5rem 1rem;
      font-weight: 500;
    }

    .action-button.primary:hover {
      background: #2563eb;
    }

    .notifications-list {
      max-height: 400px;
      overflow-y: auto;
    }

    .notification-item {
      padding: 1rem 1.5rem;
      border-bottom: 1px solid #f3f4f6;
      transition: all 0.2s ease;
      cursor: pointer;
      position: relative;
    }

    .notification-item:hover {
      background: #f9fafb;
    }

    .notification-item:last-child {
      border-bottom: none;
    }

    .notification-item.unread {
      background: #eff6ff;
      border-left: 4px solid #3b82f6;
    }

    .notification-item.unread:hover {
      background: #dbeafe;
    }

    .notification-header {
      display: flex;
      align-items: flex-start;
      gap: 0.75rem;
      margin-bottom: 0.5rem;
    }

    .notification-icon {
      width: 20px;
      height: 20px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      margin-top: 0.125rem;
    }

    .notification-icon.success {
      background: #d1fae5;
      color: #065f46;
    }

    .notification-icon.error {
      background: #fee2e2;
      color: #991b1b;
    }

    .notification-icon.warning {
      background: #fef3c7;
      color: #92400e;
    }

    .notification-icon.info {
      background: #dbeafe;
      color: #1d4ed8;
    }

    .notification-content {
      flex: 1;
      min-width: 0;
    }

    .notification-title {
      font-size: 0.875rem;
      font-weight: 600;
      color: #111827;
      margin: 0 0 0.25rem 0;
      line-height: 1.4;
    }

    .notification-message {
      font-size: 0.875rem;
      color: #6b7280;
      margin: 0;
      line-height: 1.4;
    }

    .notification-time {
      font-size: 0.75rem;
      color: #9ca3af;
      margin-top: 0.5rem;
    }

    .notification-actions {
      display: flex;
      gap: 0.5rem;
      margin-top: 0.75rem;
    }

    .notification-action {
      background: none;
      border: 1px solid #d1d5db;
      color: #374151;
      cursor: pointer;
      padding: 0.375rem 0.75rem;
      border-radius: 0.375rem;
      font-size: 0.75rem;
      font-weight: 500;
      transition: all 0.2s ease;
    }

    .notification-action:hover {
      background: #f9fafb;
      border-color: #9ca3af;
    }

    .notification-action.primary {
      background: #3b82f6;
      border-color: #3b82f6;
      color: white;
    }

    .notification-action.primary:hover {
      background: #2563eb;
      border-color: #2563eb;
    }

    .notification-action.danger {
      background: #ef4444;
      border-color: #ef4444;
      color: white;
    }

    .notification-action.danger:hover {
      background: #dc2626;
      border-color: #dc2626;
    }

    .empty-state {
      padding: 3rem 1.5rem;
      text-align: center;
      color: #6b7280;
    }

    .empty-icon {
      width: 48px;
      height: 48px;
      color: #d1d5db;
      margin: 0 auto 1rem;
    }

    .empty-title {
      font-size: 0.875rem;
      font-weight: 500;
      color: #374151;
      margin: 0 0 0.5rem 0;
    }

    .empty-message {
      font-size: 0.75rem;
      margin: 0;
    }

    .loading-state {
      padding: 2rem;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 1rem;
    }

    /* Mobile-first responsive design */
    @media (max-width: 640px) {
      .notification-panel {
        width: 320px;
        right: -50px;
        max-height: 70vh; /* Better mobile viewport usage */
      }

      .notification-item {
        padding: 0.75rem 1rem;
      }

      .notification-actions {
        flex-direction: column;
        gap: 0.375rem;
      }

      .notification-action {
        width: 100%;
        justify-content: center;
      }

      .panel-header {
        padding: 1rem;
      }

      .panel-title {
        font-size: 1rem;
      }
    }

    /* iPhone optimization */
    @media (max-width: 393px) {
      .notification-panel {
        width: 300px;
        right: -30px;
      }
    }

    /* Animation for new notifications */
    @keyframes slideIn {
      from {
        transform: translateX(100%);
        opacity: 0;
      }
      to {
        transform: translateX(0);
        opacity: 1;
      }
    }

    .notification-item.new {
      animation: slideIn 0.3s ease-out;
    }

    /* Overlay for mobile */
    .notification-overlay {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0, 0, 0, 0.1);
      z-index: 999;
      opacity: 0;
      visibility: hidden;
      transition: all 0.2s ease;
    }

    .notification-overlay.show {
      opacity: 1;
      visibility: visible;
    }

    /* FCM Permission Request Styles */
    .permission-request {
      padding: 1.5rem;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      border-radius: 0.75rem;
      margin: 1rem;
      box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }

    .permission-title {
      font-size: 1rem;
      font-weight: 600;
      margin: 0 0 0.5rem 0;
    }

    .permission-message {
      font-size: 0.875rem;
      opacity: 0.9;
      margin: 0 0 1rem 0;
      line-height: 1.4;
    }

    .permission-actions {
      display: flex;
      gap: 0.75rem;
    }

    .permission-button {
      background: rgba(255, 255, 255, 0.2);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.3);
      color: white;
      cursor: pointer;
      padding: 0.625rem 1.25rem;
      border-radius: 0.5rem;
      font-size: 0.875rem;
      font-weight: 500;
      transition: all 0.2s ease;
      flex: 1;
    }

    .permission-button:hover {
      background: rgba(255, 255, 255, 0.3);
      transform: translateY(-1px);
    }

    .permission-button.primary {
      background: rgba(255, 255, 255, 0.95);
      color: #667eea;
    }

    .permission-button.primary:hover {
      background: white;
    }

    /* Network status indicator */
    .network-status {
      padding: 0.5rem 1rem;
      background: #f97316;
      color: white;
      font-size: 0.75rem;
      text-align: center;
      border-radius: 0 0 0.75rem 0.75rem;
    }

    .network-status.online {
      background: #10b981;
    }

    /* FCM status indicator */
    .fcm-status {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem 1rem;
      background: #f3f4f6;
      border-bottom: 1px solid #e5e7eb;
      font-size: 0.75rem;
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #6b7280;
    }

    .status-dot.enabled {
      background: #10b981;
    }

    .status-dot.disabled {
      background: #ef4444;
    }

    @media (max-width: 640px) {
      .notification-panel {
        position: fixed;
        top: 4rem;
        right: 1rem;
        left: 1rem;
        width: auto;
        max-width: none;
      }

      .permission-actions {
        flex-direction: column;
      }

      .permission-button {
        width: 100%;
      }
    }
  `

  connectedCallback() {
    super.connectedCallback()
    this.initializeNotificationCenter()
  }

  disconnectedCallback() {
    super.disconnectedCallback()
    document.removeEventListener('click', this.handleDocumentClick)
    window.removeEventListener('online', this.handleNetworkChange)
    window.removeEventListener('offline', this.handleNetworkChange)
  }

  private async initializeNotificationCenter() {
    // Initialize notification service
    await this.notificationService.initialize()
    
    // Get initial state
    const stats = this.notificationService.getNotificationStats()
    this.permissionState = stats.permission
    this.fcmSupported = stats.isSupported && stats.isPushSupported
    this.isOnline = stats.isOnline
    
    console.log('📱 Notification Center initialized:', { stats, isMobile: this.isMobile, isIOS: this.isIOS })
    
    this.loadNotifications()
    this.setupEventListeners()
  }

  private setupEventListeners() {
    // Listen for clicks outside to close panel
    document.addEventListener('click', this.handleDocumentClick)
    
    // Listen for network changes
    window.addEventListener('online', this.handleNetworkChange)
    window.addEventListener('offline', this.handleNetworkChange)
    
    // Listen for notification service events
    this.notificationService.on('notification_shown', this.handleNotificationShown)
    this.notificationService.on('permission_changed', this.handlePermissionChanged)
    this.notificationService.on('fcm_token_received', this.handleFCMTokenReceived)
    this.notificationService.on('notification_queued', this.handleNotificationQueued)
  }

  private handleNetworkChange = () => {
    this.isOnline = navigator.onLine
    console.log(`🌐 Network status changed: ${this.isOnline ? 'Online' : 'Offline'}`)
  }

  private handleNotificationShown = (notification: any) => {
    console.log('🔔 Notification shown:', notification)
    // Add to local notifications list for UI
    this.addNotificationToUI(notification)
  }

  private handlePermissionChanged = (permission: NotificationPermission) => {
    this.permissionState = permission
    console.log('🔔 Permission changed:', permission)
  }

  private handleFCMTokenReceived = (token: string) => {
    console.log('🔔 FCM token received:', token.substring(0, 20) + '...')
  }

  private handleNotificationQueued = (notification: any) => {
    console.log('📦 Notification queued:', notification)
    // Show visual indicator that notification is queued
    this.addQueuedNotificationToUI(notification)
  }

  private handleDocumentClick = (event: Event) => {
    const target = event.target as HTMLElement
    if (!this.contains(target)) {
      this.isOpen = false
    }
  }

  private async loadNotifications() {
    this.isLoading = true
    
    try {
      // For now, use demo notifications
      // In a real app, this would load from the notification service
      this.notifications = this.getDemoNotifications()
      this.updateUnreadCount()
    } catch (error) {
      console.error('Failed to load notifications:', error)
    } finally {
      this.isLoading = false
    }
  }

  private getDemoNotifications(): NotificationItem[] {
    return [
      {
        id: '1',
        type: 'success',
        title: 'Agent Team Activated',
        message: 'Successfully activated 5 agents for autonomous development',
        timestamp: new Date(Date.now() - 300000).toISOString(), // 5 minutes ago
        read: false,
        actions: [
          {
            label: 'View Agents',
            action: () => this.handleViewAgents(),
            style: 'primary'
          }
        ]
      },
      {
        id: '2',
        type: 'info',
        title: 'New Task Assigned',
        message: 'Task "Implement user authentication" has been assigned to Developer Agent',
        timestamp: new Date(Date.now() - 900000).toISOString(), // 15 minutes ago
        read: false,
        actions: [
          {
            label: 'View Task',
            action: () => this.handleViewTask('task-001'),
            style: 'secondary'
          }
        ]
      },
      {
        id: '3',
        type: 'warning',
        title: 'Agent Performance Alert',
        message: 'Tester Agent performance has decreased by 15% in the last hour',
        timestamp: new Date(Date.now() - 1800000).toISOString(), // 30 minutes ago
        read: true,
        actions: [
          {
            label: 'Check Agent',
            action: () => this.handleCheckAgent('agent-003'),
            style: 'secondary'
          },
          {
            label: 'Restart Agent',
            action: () => this.handleRestartAgent('agent-003'),
            style: 'primary'
          }
        ]
      },
      {
        id: '4',
        type: 'error',
        title: 'Task Execution Failed',
        message: 'Failed to execute task "Database migration" - Error: Connection timeout',
        timestamp: new Date(Date.now() - 3600000).toISOString(), // 1 hour ago
        read: true,
        actions: [
          {
            label: 'Retry Task',
            action: () => this.handleRetryTask('task-002'),
            style: 'primary'
          },
          {
            label: 'View Details',
            action: () => this.handleViewTaskDetails('task-002'),
            style: 'secondary'
          }
        ]
      },
      {
        id: '5',
        type: 'success',
        title: 'Deployment Completed',
        message: 'Successfully deployed version 1.2.0 to staging environment',
        timestamp: new Date(Date.now() - 7200000).toISOString(), // 2 hours ago
        read: true
      }
    ]
  }

  private updateUnreadCount() {
    this.unreadCount = this.notifications.filter(n => !n.read).length
  }

  private handleTogglePanel() {
    this.isOpen = !this.isOpen
  }

  private async handleMarkAllRead() {
    try {
      // Update all notifications to read
      this.notifications = this.notifications.map(n => ({ ...n, read: true }))
      this.updateUnreadCount()
      
      console.log('Marked all notifications as read')
    } catch (error) {
      console.error('Failed to mark notifications as read:', error)
    }
  }

  private async handleClearAll() {
    try {
      this.notifications = []
      this.updateUnreadCount()
      
      console.log('Cleared all notifications')
    } catch (error) {
      console.error('Failed to clear notifications:', error)
    }
  }

  private async handleNotificationClick(notification: NotificationItem) {
    if (!notification.read) {
      // Mark as read
      const index = this.notifications.findIndex(n => n.id === notification.id)
      if (index >= 0) {
        this.notifications[index] = { ...notification, read: true }
        this.updateUnreadCount()
        this.requestUpdate()
      }
    }
  }

  private formatTime(timestamp: string): string {
    const date = new Date(timestamp)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / (1000 * 60))
    
    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    
    const diffHours = Math.floor(diffMins / 60)
    if (diffHours < 24) return `${diffHours}h ago`
    
    const diffDays = Math.floor(diffHours / 24)
    return `${diffDays}d ago`
  }

  // Action handlers (these would integrate with the actual app navigation)
  private handleViewAgents() {
    this.isOpen = false
    window.location.hash = '/agents'
  }

  private handleViewTask(taskId: string) {
    this.isOpen = false
    window.location.hash = `/tasks?task=${taskId}`
  }

  private handleCheckAgent(agentId: string) {
    this.isOpen = false
    window.location.hash = `/agents?agent=${agentId}`
  }

  private handleRestartAgent(agentId: string) {
    this.isOpen = false
    console.log('Restarting agent:', agentId)
    // This would trigger an agent restart
  }

  private handleRetryTask(taskId: string) {
    this.isOpen = false
    console.log('Retrying task:', taskId)
    // This would trigger a task retry
  }

  private handleViewTaskDetails(taskId: string) {
    this.isOpen = false
    window.location.hash = `/tasks?task=${taskId}&details=true`
  }

  // FCM-specific methods
  private async handleRequestPermission() {
    try {
      const permission = await this.notificationService.requestPermission()
      this.permissionState = permission
      
      if (permission === 'granted') {
        // Subscribe to push notifications
        await this.notificationService.subscribeToPush()
        
        // Show success notification
        this.notifications.unshift({
          id: crypto.randomUUID(),
          type: 'success',
          title: '🔔 Notifications Enabled',
          message: 'You will now receive real-time updates from Agent Hive',
          timestamp: new Date().toISOString(),
          read: false
        })
        this.updateUnreadCount()
      }
    } catch (error) {
      console.error('Failed to request notification permission:', error)
      // Show error notification
      this.notifications.unshift({
        id: crypto.randomUUID(),
        type: 'error',
        title: '❌ Notification Setup Failed',
        message: 'Unable to enable notifications. Please check your browser settings.',
        timestamp: new Date().toISOString(),
        read: false
      })
      this.updateUnreadCount()
    }
  }

  private handleDismissPermission() {
    console.log('📵 User dismissed notification permission request')
  }

  private addNotificationToUI(notification: any) {
    const uiNotification: NotificationItem = {
      id: notification.id || crypto.randomUUID(),
      type: this.mapNotificationTypeToUI(notification.category),
      title: notification.title,
      message: notification.body,
      timestamp: new Date(notification.timestamp).toISOString(),
      read: false
    }
    
    this.notifications.unshift(uiNotification)
    if (this.notifications.length > 50) {
      this.notifications = this.notifications.slice(0, 50) // Keep only recent 50
    }
    this.updateUnreadCount()
  }

  private addQueuedNotificationToUI(notification: any) {
    const uiNotification: NotificationItem = {
      id: notification.id || crypto.randomUUID(),
      type: 'warning',
      title: '📦 Queued: ' + notification.title,
      message: notification.body + ' (Will deliver when online)',
      timestamp: new Date(notification.timestamp).toISOString(),
      read: false
    }
    
    this.notifications.unshift(uiNotification)
    this.updateUnreadCount()
  }

  private mapNotificationTypeToUI(category: string): NotificationItem['type'] {
    switch (category) {
      case 'success':
      case 'task':
        return 'success'
      case 'error':
      case 'critical':
        return 'error'
      case 'warning':
      case 'agent':
        return 'warning'
      default:
        return 'info'
    }
  }

  private renderNotificationIcon(type: NotificationItem['type']) {
    switch (type) {
      case 'success':
        return html`
          <svg width="12" height="12" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/>
          </svg>
        `
      case 'error':
        return html`
          <svg width="12" height="12" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"/>
          </svg>
        `
      case 'warning':
        return html`
          <svg width="12" height="12" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
          </svg>
        `
      case 'info':
        return html`
          <svg width="12" height="12" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/>
          </svg>
        `
    }
  }

  private renderStatusIndicators() {
    return html`
      ${!this.isOnline ? html`
        <div class="network-status">
          📵 Offline - Notifications will be queued
        </div>
      ` : html`
        <div class="network-status online">
          🌐 Online - Live notifications enabled
        </div>
      `}
      
      ${this.fcmSupported ? html`
        <div class="fcm-status">
          <div class="status-dot ${this.permissionState === 'granted' ? 'enabled' : 'disabled'}"></div>
          <span>
            Push notifications: ${this.permissionState === 'granted' ? 'Enabled' : 
                                this.permissionState === 'denied' ? 'Blocked' : 'Not configured'}
            ${this.isMobile ? '📱' : '💻'}
          </span>
        </div>
      ` : ''}
    `
  }

  private renderPermissionRequest() {
    if (this.permissionState === 'granted' || this.permissionState === 'denied') {
      return ''
    }

    return html`
      <div class="permission-request">
        <div class="permission-title">
          ${this.isMobile ? '📱 Enable Mobile Notifications' : '🔔 Enable Notifications'}
        </div>
        <div class="permission-message">
          ${this.isMobile 
            ? 'Get instant alerts about agent activities, task completions, and system updates on your mobile device.'
            : 'Stay informed about agent activities, build status, and system alerts in real-time.'
          }
          ${this.isIOS ? ' For iOS users: Add to Home Screen for best experience.' : ''}
        </div>
        <div class="permission-actions">
          <button 
            class="permission-button primary" 
            @click=${this.handleRequestPermission}
          >
            ${this.isMobile ? '📱 Enable Alerts' : '🔔 Allow Notifications'}
          </button>
          <button 
            class="permission-button" 
            @click=${this.handleDismissPermission}
          >
            Not Now
          </button>
        </div>
      </div>
    `
  }

  render() {
    return html`
      <div class="notification-overlay ${this.isOpen ? 'show' : ''}" @click=${this.handleTogglePanel}></div>
      
      <button class="notification-trigger" @click=${this.handleTogglePanel}>
        <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-3.405-3.405A5.982 5.982 0 0118 9.75V9a6 6 0 10-12 0v.75c0 2.123.8 4.057 2.405 5.595L5 17h5m0 0v1a3 3 0 106 0v-1m-6 0h6"/>
        </svg>
        <div class="notification-badge ${this.unreadCount > 0 ? 'show' : ''}">
          ${this.unreadCount}
        </div>
      </button>

      <div class="notification-panel ${this.isOpen ? 'open' : ''}">
        ${this.renderStatusIndicators()}
        ${this.renderPermissionRequest()}
        
        <div class="panel-header">
          <h3 class="panel-title">Notifications</h3>
          <div class="header-actions">
            ${this.unreadCount > 0 ? html`
              <button class="action-button" @click=${this.handleMarkAllRead}>
                Mark all read
              </button>
            ` : ''}
            <button class="action-button" @click=${this.handleClearAll}>
              Clear all
            </button>
          </div>
        </div>

        <div class="notifications-list">
          ${this.isLoading ? html`
            <div class="loading-state">
              <loading-spinner size="medium"></loading-spinner>
              <span>Loading notifications...</span>
            </div>
          ` : this.notifications.length === 0 ? html`
            <div class="empty-state">
              <svg class="empty-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-3.405-3.405A5.982 5.982 0 0118 9.75V9a6 6 0 10-12 0v.75c0 2.123.8 4.057 2.405 5.595L5 17h5m0 0v1a3 3 0 106 0v-1m-6 0h6"/>
              </svg>
              <div class="empty-title">No notifications</div>
              <div class="empty-message">You're all caught up!</div>
            </div>
          ` : this.notifications.map(notification => html`
            <div 
              class="notification-item ${notification.read ? '' : 'unread'}"
              @click=${() => this.handleNotificationClick(notification)}
            >
              <div class="notification-header">
                <div class="notification-icon ${notification.type}">
                  ${this.renderNotificationIcon(notification.type)}
                </div>
                <div class="notification-content">
                  <div class="notification-title">${notification.title}</div>
                  <div class="notification-message">${notification.message}</div>
                  <div class="notification-time">${this.formatTime(notification.timestamp)}</div>
                  
                  ${notification.actions && notification.actions.length > 0 ? html`
                    <div class="notification-actions">
                      ${notification.actions.map(action => html`
                        <button 
                          class="notification-action ${action.style || 'secondary'}"
                          @click=${(e: Event) => {
                            e.stopPropagation()
                            action.action()
                          }}
                        >
                          ${action.label}
                        </button>
                      `)}
                    </div>
                  ` : ''}
                </div>
              </div>
            </div>
          `)}
        </div>
      </div>
    `
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'notification-center': NotificationCenter
  }
}