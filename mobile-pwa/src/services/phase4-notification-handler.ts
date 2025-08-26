/**
 * Phase 4: Mobile PWA Development - Critical Agent Event Notifications
 * 
 * Enterprise-grade notification handler for real-time agent monitoring
 * and system alerts with mobile-first design and offline support.
 */

import { NotificationService, NotificationData } from './notification'
import { EventEmitter } from '../utils/event-emitter'

export interface AgentEvent {
  id: string
  agentId: string
  agentName: string
  type: 'error' | 'warning' | 'success' | 'status_change' | 'performance' | 'security'
  severity: 'low' | 'medium' | 'high' | 'critical'
  title: string
  message: string
  metadata?: Record<string, any>
  timestamp: number
  resolved?: boolean
  actionRequired?: boolean
}

export interface SystemAlert {
  id: string
  type: 'build_failure' | 'deployment_error' | 'performance_degradation' | 'security_breach' | 'resource_limit' | 'service_down'
  severity: 'low' | 'medium' | 'high' | 'critical'
  service: string
  title: string
  message: string
  impact: 'none' | 'minor' | 'major' | 'severe'
  metadata?: Record<string, any>
  timestamp: number
  resolved?: boolean
  actionRequired?: boolean
  estimatedResolution?: number // minutes
}

export class Phase4NotificationHandler extends EventEmitter {
  private static instance: Phase4NotificationHandler
  private notificationService: NotificationService
  private isInitialized: boolean = false
  private eventQueue: AgentEvent[] = []
  private alertQueue: SystemAlert[] = []
  private isMobile: boolean = this.detectMobile()
  private isOffline: boolean = !navigator.onLine
  
  // Performance tracking for notification delivery
  private deliveryStats = {
    sent: 0,
    failed: 0,
    queued: 0,
    avgDeliveryTime: 0
  }

  static getInstance(): Phase4NotificationHandler {
    if (!Phase4NotificationHandler.instance) {
      Phase4NotificationHandler.instance = new Phase4NotificationHandler()
    }
    return Phase4NotificationHandler.instance
  }

  private constructor() {
    super()
    this.notificationService = NotificationService.getInstance()
    this.setupNetworkMonitoring()
  }

  private detectMobile(): boolean {
    const userAgent = navigator.userAgent
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(userAgent)
    const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0
    return isMobile || isTouchDevice
  }

  async initialize(): Promise<void> {
    try {
      console.log('🎯 Initializing Phase 4 Notification Handler...')

      // Initialize base notification service first
      await this.notificationService.initialize()

      // Setup event listeners
      this.setupEventListeners()

      this.isInitialized = true
      console.log('✅ Phase 4 Notification Handler initialized')

    } catch (error) {
      console.error('❌ Failed to initialize Phase 4 notification handler:', error)
      throw error
    }
  }

  /**
   * Handle critical agent events with intelligent routing and mobile optimization
   */
  async handleAgentEvent(event: AgentEvent): Promise<void> {
    try {
      const startTime = performance.now()

      // Queue if offline
      if (this.isOffline) {
        await this.queueAgentEvent(event)
        return
      }

      // Create notification based on event type and severity
      const notification = await this.createAgentEventNotification(event)

      // Send notification
      await this.notificationService.showNotification(notification)

      // Update delivery stats
      const deliveryTime = performance.now() - startTime
      this.updateDeliveryStats('sent', deliveryTime)

      // Emit success event
      this.emit('agent_event_processed', event)

      console.log(`📱 Agent event notification sent: ${event.title} (${deliveryTime.toFixed(2)}ms)`)

    } catch (error) {
      console.error('❌ Failed to handle agent event:', error)
      this.updateDeliveryStats('failed')
      
      // Queue for retry
      await this.queueAgentEvent(event)
      this.emit('agent_event_failed', { event, error })
    }
  }

  /**
   * Handle critical system alerts with immediate delivery for high-severity issues
   */
  async handleSystemAlert(alert: SystemAlert): Promise<void> {
    try {
      const startTime = performance.now()

      // Critical alerts bypass throttling
      const bypassThrottling = alert.severity === 'critical'

      // Queue if offline (unless critical)
      if (this.isOffline && !bypassThrottling) {
        await this.queueSystemAlert(alert)
        return
      }

      // Create notification
      const notification = await this.createSystemAlertNotification(alert)

      // Send notification
      await this.notificationService.showNotification(notification)

      // For critical alerts, also try to wake the screen/app
      if (alert.severity === 'critical' && this.isMobile) {
        await this.triggerCriticalAlertActions(alert)
      }

      // Update stats
      const deliveryTime = performance.now() - startTime
      this.updateDeliveryStats('sent', deliveryTime)

      this.emit('system_alert_processed', alert)

      console.log(`🚨 System alert notification sent: ${alert.title} (${deliveryTime.toFixed(2)}ms)`)

    } catch (error) {
      console.error('❌ Failed to handle system alert:', error)
      this.updateDeliveryStats('failed')
      
      // Queue for retry
      await this.queueSystemAlert(alert)
      this.emit('system_alert_failed', { alert, error })
    }
  }

  /**
   * Create mobile-optimized notification for agent events
   */
  private async createAgentEventNotification(event: AgentEvent): Promise<Partial<NotificationData>> {
    const iconMap = {
      error: '🚫',
      warning: '⚠️',
      success: '✅',
      status_change: '🔄',
      performance: '📊',
      security: '🔒'
    }

    const priorityMap = {
      low: 'normal' as const,
      medium: 'normal' as const,
      high: 'high' as const,
      critical: 'high' as const
    }

    // Mobile-optimized title and body
    const title = this.isMobile 
      ? `${iconMap[event.type]} ${event.agentName}`
      : `${iconMap[event.type]} Agent ${event.agentName} - ${event.title}`
    
    const body = this.isMobile 
      ? event.message.substring(0, 80) + (event.message.length > 80 ? '...' : '')
      : event.message

    return {
      id: event.id,
      title,
      body,
      icon: '/icons/icon-192x192.png',
      badge: '/icons/icon-96x96.png',
      tag: `agent-${event.agentId}-${event.type}`,
      priority: priorityMap[event.severity],
      category: 'agent',
      requireInteraction: event.severity === 'critical' || event.actionRequired,
      data: {
        type: 'agent_event',
        agentId: event.agentId,
        agentName: event.agentName,
        eventType: event.type,
        severity: event.severity,
        timestamp: event.timestamp,
        metadata: event.metadata
      }
    }
  }

  /**
   * Create mobile-optimized notification for system alerts
   */
  private async createSystemAlertNotification(alert: SystemAlert): Promise<Partial<NotificationData>> {
    const iconMap = {
      build_failure: '🔥',
      deployment_error: '🚨',
      performance_degradation: '📉',
      security_breach: '🔓',
      resource_limit: '⚡',
      service_down: '💥'
    }

    const priorityMap = {
      low: 'normal' as const,
      medium: 'normal' as const,
      high: 'high' as const,
      critical: 'high' as const
    }

    // Mobile-optimized content
    const title = this.isMobile 
      ? `${iconMap[alert.type]} ${alert.service}`
      : `${iconMap[alert.type]} ${alert.service} - ${alert.title}`
    
    const body = this.isMobile 
      ? alert.message.substring(0, 80) + (alert.message.length > 80 ? '...' : '')
      : alert.message

    return {
      id: alert.id,
      title,
      body,
      icon: '/icons/icon-192x192.png',
      badge: '/icons/icon-96x96.png',
      tag: `system-${alert.type}-${alert.service}`,
      priority: priorityMap[alert.severity],
      category: 'system',
      requireInteraction: alert.severity === 'critical' || alert.actionRequired,
      data: {
        type: 'system_alert',
        alertType: alert.type,
        service: alert.service,
        severity: alert.severity,
        impact: alert.impact,
        timestamp: alert.timestamp,
        estimatedResolution: alert.estimatedResolution,
        metadata: alert.metadata
      }
    }
  }

  /**
   * Trigger additional actions for critical alerts on mobile
   */
  private async triggerCriticalAlertActions(alert: SystemAlert): Promise<void> {
    try {
      // Vibrate if supported (mobile only)
      if ('vibrate' in navigator && this.isMobile) {
        navigator.vibrate([200, 100, 200, 100, 400])
      }

      // Try to wake the screen by playing a silent audio (mobile hack)
      if (this.isMobile) {
        const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmciPjCN2va8dCU=')
        
        try {
          await audio.play()
        } catch (e) {
          // Ignore audio errors
        }
      }

    } catch (error) {
      console.warn('Failed to trigger critical alert actions:', error)
    }
  }

  /**
   * Queue events for offline processing
   */
  private async queueAgentEvent(event: AgentEvent): Promise<void> {
    this.eventQueue.push(event)
    this.updateDeliveryStats('queued')
    
    console.log(`📦 Queued agent event: ${event.title}`)
    this.emit('agent_event_queued', event)
  }

  private async queueSystemAlert(alert: SystemAlert): Promise<void> {
    this.alertQueue.push(alert)
    this.updateDeliveryStats('queued')
    
    console.log(`📦 Queued system alert: ${alert.title}`)
    this.emit('system_alert_queued', alert)
  }

  /**
   * Setup event listeners and network monitoring
   */
  private setupEventListeners(): void {
    // Listen to notification service events
    this.notificationService.on('notification_click', (data) => {
      this.handleNotificationAction(data)
    })
  }

  private setupNetworkMonitoring(): void {
    window.addEventListener('online', () => {
      console.log('🌐 Phase 4 Notifications: Online')
      this.isOffline = false
    })
    
    window.addEventListener('offline', () => {
      console.log('📵 Phase 4 Notifications: Offline')
      this.isOffline = true
    })
    
    this.isOffline = !navigator.onLine
  }

  /**
   * Utility methods
   */
  private handleNotificationAction(data: any): void {
    console.log('📱 Notification action:', data)
    this.emit('notification_action', data)
  }

  private updateDeliveryStats(type: 'sent' | 'failed' | 'queued', deliveryTime?: number): void {
    this.deliveryStats[type]++
    
    if (type === 'sent' && deliveryTime !== undefined) {
      const total = this.deliveryStats.avgDeliveryTime * (this.deliveryStats.sent - 1)
      this.deliveryStats.avgDeliveryTime = (total + deliveryTime) / this.deliveryStats.sent
    }
  }

  /**
   * Public API methods
   */
  
  async requestPermissions(): Promise<boolean> {
    try {
      const permission = await this.notificationService.requestPermission()
      return permission === 'granted'
    } catch (error) {
      console.error('Failed to request notification permissions:', error)
      return false
    }
  }

  async subscribeToPushNotifications(): Promise<boolean> {
    try {
      const subscription = await this.notificationService.subscribeToPush()
      return subscription !== null
    } catch (error) {
      console.error('Failed to subscribe to push notifications:', error)
      return false
    }
  }

  getDeliveryStats() {
    return { ...this.deliveryStats }
  }

  getQueueStatus() {
    return {
      events: this.eventQueue.length,
      alerts: this.alertQueue.length,
      offline: this.isOffline
    }
  }

  getNotificationStats() {
    return this.notificationService.getNotificationStats()
  }

  async clearQueues(): Promise<void> {
    this.eventQueue = []
    this.alertQueue = []
    console.log('🗑️ Cleared notification queues')
  }
}

// Factory function for service consistency
export function getPhase4NotificationHandler(): Phase4NotificationHandler {
  return Phase4NotificationHandler.getInstance()
}

/**
 * Convenience functions for common notification scenarios
 */

export async function notifyAgentError(agentId: string, agentName: string, error: string, metadata?: any): Promise<void> {
  const handler = getPhase4NotificationHandler()
  await handler.handleAgentEvent({
    id: crypto.randomUUID(),
    agentId,
    agentName,
    type: 'error',
    severity: 'high',
    title: 'Agent Error',
    message: error,
    metadata,
    timestamp: Date.now(),
    actionRequired: true
  })
}

export async function notifyBuildFailure(buildId: string, service: string, error: string, metadata?: any): Promise<void> {
  const handler = getPhase4NotificationHandler()
  await handler.handleSystemAlert({
    id: crypto.randomUUID(),
    type: 'build_failure',
    severity: 'high',
    service,
    title: 'Build Failed',
    message: error,
    impact: 'major',
    metadata: { buildId, ...metadata },
    timestamp: Date.now(),
    actionRequired: true,
    estimatedResolution: 30
  })
}

export async function notifyCriticalSystemIssue(service: string, issue: string, impact: SystemAlert['impact'], metadata?: any): Promise<void> {
  const handler = getPhase4NotificationHandler()
  await handler.handleSystemAlert({
    id: crypto.randomUUID(),
    type: 'service_down',
    severity: 'critical',
    service,
    title: 'Critical System Issue',
    message: issue,
    impact,
    metadata,
    timestamp: Date.now(),
    actionRequired: true,
    estimatedResolution: 15
  })
}

export async function notifyTaskCompletion(agentId: string, agentName: string, taskTitle: string, metadata?: any): Promise<void> {
  const handler = getPhase4NotificationHandler()
  await handler.handleAgentEvent({
    id: crypto.randomUUID(),
    agentId,
    agentName,
    type: 'success',
    severity: 'low',
    title: 'Task Completed',
    message: `Task "${taskTitle}" completed successfully`,
    metadata,
    timestamp: Date.now(),
    actionRequired: false
  })
}