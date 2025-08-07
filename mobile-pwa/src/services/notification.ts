import { EventEmitter } from '../utils/event-emitter'
import { initializeApp } from 'firebase/app'
import { getMessaging, getToken, onMessage, isSupported } from 'firebase/messaging'

export interface NotificationData {
  id: string
  title: string
  body: string
  icon?: string
  badge?: string
  image?: string
  tag?: string
  data?: any
  actions?: NotificationAction[]
  timestamp: number
  priority: 'low' | 'normal' | 'high'
  category: string
  requireInteraction?: boolean
}

export interface NotificationAction {
  action: string
  title: string
  icon?: string
}

export interface PushSubscriptionData {
  endpoint: string
  keys: {
    p256dh: string
    auth: string
  }
}

export class NotificationService extends EventEmitter {
  private static instance: NotificationService
  private registration: ServiceWorkerRegistration | null = null
  private pushSubscription: PushSubscription | null = null
  private permission: NotificationPermission = 'default'
  private isInitialized: boolean = false
  private messaging: any = null
  private fcmToken: string | null = null
  private offlineQueue: NotificationData[] = []
  private isOnline: boolean = navigator.onLine
  private isMobile: boolean = this.detectMobile()
  private iosInstallPromptShown: boolean = false
  
  // FCM configuration using environment variables with proper fallbacks
  private readonly fcmConfig = {
    apiKey: process.env.VITE_FIREBASE_API_KEY || 'demo-key',
    authDomain: process.env.VITE_FIREBASE_AUTH_DOMAIN || 'agent-hive-demo.firebaseapp.com',
    projectId: process.env.VITE_FIREBASE_PROJECT_ID || 'agent-hive-demo',
    storageBucket: process.env.VITE_FIREBASE_STORAGE_BUCKET || 'agent-hive-demo.appspot.com',
    messagingSenderId: process.env.VITE_FIREBASE_MESSAGING_SENDER_ID || '123456789',
    appId: process.env.VITE_FIREBASE_APP_ID || '1:123456789:web:abcdef',
    measurementId: process.env.VITE_FIREBASE_MEASUREMENT_ID || 'G-DEMO123456',
    vapidKey: process.env.VITE_FIREBASE_VAPID_KEY || 'demo-vapid-key'
  }
  
  static getInstance(): NotificationService {
    if (!NotificationService.instance) {
      NotificationService.instance = new NotificationService()
    }
    return NotificationService.instance
  }
  
  private detectMobile(): boolean {
    const userAgent = navigator.userAgent
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(userAgent)
    const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0
    return isMobile || isTouchDevice
  }
  
  private isIOS(): boolean {
    return /iPad|iPhone|iPod/.test(navigator.userAgent)
  }
  
  private isInStandaloneMode(): boolean {
    return (window.navigator as any).standalone === true ||
           window.matchMedia('(display-mode: standalone)').matches
  }
  
  async initialize(): Promise<void> {
    try {
      console.log('🔔 Initializing notification service...')
      console.log(`📱 Mobile device: ${this.isMobile}, iOS: ${this.isIOS()}`)
      
      // Check if notifications are supported
      if (!('Notification' in window)) {
        console.warn('This browser does not support notifications')
        return
      }
      
      // Check if service workers are supported
      if (!('serviceWorker' in navigator)) {
        console.warn('This browser does not support service workers')
        return
      }
      
      // Get current permission status
      this.permission = Notification.permission
      
      // Setup online/offline monitoring
      this.setupNetworkMonitoring()
      
      // Skip service worker registration in development mode
      if (process.env.NODE_ENV === 'development') {
        console.log('🔧 Development mode: Skipping notification service worker registration')
      } else {
        // Register service worker if not already registered
        await this.registerServiceWorker()
      }
      
      // Setup message handling
      this.setupMessageHandling()
      
      // In development mode, skip Firebase initialization to avoid hanging
      if (process.env.NODE_ENV === 'development' && this.fcmConfig.apiKey === 'demo-key') {
        console.log('🔧 Development mode: Skipping Firebase Cloud Messaging initialization')
      } else {
        // Initialize Firebase Cloud Messaging
        await this.initializeFirebaseMessaging()
      }
      
      // Initialize push messaging if supported
      if ('PushManager' in window) {
        await this.initializePushMessaging()
      }
      
      // Load offline notification queue
      await this.loadOfflineQueue()
      
      // Process queued notifications if online
      if (this.isOnline) {
        await this.processOfflineQueue()
      }
      
      // Show iOS install prompt if appropriate
      this.handleIOSInstallPrompt()
      
      this.isInitialized = true
      console.log('✅ Notification service initialized')
      
    } catch (error) {
      console.error('❌ Failed to initialize notification service:', error)
      // Don't throw the error in development mode - just log it and continue
      if (process.env.NODE_ENV === 'development') {
        console.warn('🔧 Development mode: Continuing despite notification service error')
        this.isInitialized = true
      } else {
        throw error
      }
    }
  }
  
  async requestPermission(): Promise<NotificationPermission> {
    try {
      if (this.permission === 'granted') {
        return this.permission
      }
      
      // Request permission
      const permission = await Notification.requestPermission()
      this.permission = permission
      
      if (permission === 'granted') {
        console.log('✅ Notification permission granted')
        
        // Initialize push messaging after permission is granted
        if ('PushManager' in window && this.registration) {
          await this.initializePushMessaging()
        }
      } else {
        console.log('❌ Notification permission denied')
      }
      
      this.emit('permission_changed', permission)
      return permission
      
    } catch (error) {
      console.error('Failed to request notification permission:', error)
      throw error
    }
  }
  
  async showNotification(data: Partial<NotificationData>): Promise<void> {
    try {
      if (this.permission !== 'granted') {
        console.warn('Cannot show notification - permission not granted')
        
        // Queue notification for offline/no-permission scenarios
        if (!this.isOnline) {
          await this.queueNotification(data)
        }
        return
      }
      
      const notification: NotificationData = {
        id: data.id || crypto.randomUUID(),
        title: data.title || 'Agent Hive',
        body: data.body || '',
        icon: data.icon || '/icons/icon-192x192.png',
        badge: data.badge || '/icons/icon-96x96.png',
        tag: data.tag,
        data: data.data,
        actions: data.actions,
        timestamp: Date.now(),
        priority: data.priority || 'normal',
        category: data.category || 'general',
        requireInteraction: data.requireInteraction || false
      }
      
      // Apply mobile-specific optimizations
      if (this.isMobile) {
        notification.requireInteraction = notification.priority === 'high'
        // Shorter body text for mobile
        if (notification.body.length > 100) {
          notification.body = notification.body.substring(0, 97) + '...'
        }
      }
      
      // Show notification via service worker for better reliability
      if (this.registration) {
        await this.registration.showNotification(notification.title, {
          body: notification.body,
          icon: notification.icon,
          badge: notification.badge,
          tag: notification.tag,
          data: notification.data,
          actions: notification.actions,
          requireInteraction: notification.requireInteraction,
          timestamp: notification.timestamp,
        })
      } else {
        // Fallback to browser notification
        new Notification(notification.title, {
          body: notification.body,
          icon: notification.icon,
          tag: notification.tag,
          data: notification.data,
        })
      }
      
      this.emit('notification_shown', notification)
      
    } catch (error) {
      console.error('Failed to show notification:', error)
      // Queue notification for retry if offline
      if (!this.isOnline) {
        await this.queueNotification(data)
      }
      throw error
    }
  }
  
  async subscribeToPush(): Promise<PushSubscriptionData | null> {
    try {
      if (!this.registration || !this.fcmConfig.vapidKey) {
        console.warn('Cannot subscribe to push - service worker or VAPID key not available')
        return null
      }
      
      // Check if already subscribed
      const existingSubscription = await this.registration.pushManager.getSubscription()
      if (existingSubscription) {
        this.pushSubscription = existingSubscription
        return this.subscriptionToData(existingSubscription)
      }
      
      // Subscribe to push notifications
      const subscription = await this.registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: this.urlBase64ToUint8Array(this.fcmConfig.vapidKey)
      })
      
      this.pushSubscription = subscription
      
      // Send subscription to server
      await this.sendSubscriptionToServer(subscription)
      
      console.log('✅ Push subscription created')
      this.emit('push_subscribed', subscription)
      
      return this.subscriptionToData(subscription)
      
    } catch (error) {
      console.error('Failed to subscribe to push notifications:', error)
      throw error
    }
  }
  
  async unsubscribeFromPush(): Promise<void> {
    try {
      if (this.pushSubscription) {
        await this.pushSubscription.unsubscribe()
        
        // Notify server
        await this.removeSubscriptionFromServer()
        
        this.pushSubscription = null
        console.log('✅ Push subscription removed')
        this.emit('push_unsubscribed')
      }
    } catch (error) {
      console.error('Failed to unsubscribe from push notifications:', error)
      throw error
    }
  }
  
  private async registerServiceWorker(): Promise<void> {
    try {
      // In development mode, skip service worker registration if sw.js is not accessible
      if (process.env.NODE_ENV === 'development') {
        try {
          const response = await fetch('/sw.js', { method: 'HEAD' })
          if (!response.ok) {
            console.log('🔧 Development mode: Service worker file not available, skipping registration')
            return
          }
        } catch (fetchError) {
          console.log('🔧 Development mode: Service worker file not accessible, skipping registration')
          return
        }
      }
      
      this.registration = await navigator.serviceWorker.register('/sw.js')
      console.log('✅ Service worker registered')
      
      // In development mode, don't wait for service worker ready to avoid hanging
      if (process.env.NODE_ENV === 'development') {
        console.log('🔧 Development mode: Skipping service worker ready wait')
      } else {
        // Wait for service worker to be ready
        await navigator.serviceWorker.ready
      }
      
    } catch (error) {
      console.error('Service worker registration failed:', error)
      // Don't throw in development mode
      if (process.env.NODE_ENV === 'development') {
        console.warn('🔧 Development mode: Continuing without service worker registration')
      } else {
        throw error
      }
    }
  }
  
  private setupMessageHandling(): void {
    // Listen for messages from service worker
    navigator.serviceWorker.addEventListener('message', (event) => {
      const { type, data } = event.data
      
      switch (type) {
        case 'notification-click':
          this.handleNotificationClick(data)
          break
        case 'notification-close':
          this.handleNotificationClose(data)
          break
        case 'push-received':
          this.handlePushReceived(data)
          break
        default:
          console.log('Unknown service worker message:', type, data)
      }
    })
    
    // Listen for notification events
    if (this.registration) {
      this.registration.addEventListener('notificationclick', (event) => {
        this.handleNotificationClick(event.notification.data)
      })
    }
  }
  
  private async initializeFirebaseMessaging(): Promise<void> {
    try {
      // Check if Firebase Messaging is supported
      const messagingSupported = await isSupported()
      if (!messagingSupported) {
        console.log('Firebase Messaging is not supported in this browser')
        return
      }
      
      // Initialize Firebase app
      const firebaseApp = initializeApp(this.fcmConfig)
      
      // Get messaging instance
      this.messaging = getMessaging(firebaseApp)
      
      // Setup foreground message handling
      onMessage(this.messaging, (payload) => {
        console.log('📨 Foreground FCM message received:', payload)
        this.handleForegroundMessage(payload)
      })
      
      console.log('✅ Firebase Cloud Messaging initialized')
      
    } catch (error) {
      console.error('Failed to initialize Firebase Messaging:', error)
    }
  }
  
  private async initializePushMessaging(): Promise<void> {
    try {
      if (!this.registration || this.permission !== 'granted') {
        return
      }
      
      // Check for existing subscription
      const existingSubscription = await this.registration.pushManager.getSubscription()
      if (existingSubscription) {
        this.pushSubscription = existingSubscription
        console.log('✅ Existing push subscription found')
      }
      
      // Get FCM token if Firebase messaging is available
      if (this.messaging) {
        await this.getFCMToken()
      }
      
    } catch (error) {
      console.error('Failed to initialize push messaging:', error)
    }
  }
  
  private async getFCMToken(): Promise<string | null> {
    try {
      if (!this.messaging) {
        return null
      }
      
      const token = await getToken(this.messaging, {
        vapidKey: this.fcmConfig.vapidKey
      })
      
      if (token) {
        this.fcmToken = token
        console.log('✅ FCM token obtained:', token)
        
        // Send token to server
        await this.sendFCMTokenToServer(token)
        this.emit('fcm_token_received', token)
        
        return token
      } else {
        console.log('No FCM registration token available')
        return null
      }
      
    } catch (error) {
      console.error('Failed to get FCM token:', error)
      return null
    }
  }
  
  private handleForegroundMessage(payload: any): void {
    console.log('Handling foreground FCM message:', payload)
    
    // Extract notification data
    const notification = payload.notification
    const data = payload.data
    
    // Show notification if app is in foreground
    if (notification) {
      this.showNotification({
        title: notification.title,
        body: notification.body,
        icon: notification.icon || '/icons/icon-192x192.png',
        data: data,
        tag: data?.tag,
        category: data?.category || 'fcm',
        priority: data?.priority || 'normal'
      })
    }
    
    this.emit('fcm_message_received', payload)
  }
  
  private async sendFCMTokenToServer(token: string): Promise<void> {
    try {
      const response = await fetch('/api/v1/notifications/fcm-token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
        },
        body: JSON.stringify({
          token,
          device_type: 'web',
          topics: ['build.failed', 'agent.error', 'task.completed', 'human.approval.request']
        })
      })
      
      if (!response.ok) {
        throw new Error('Failed to register FCM token with server')
      }
      
      console.log('✅ FCM token sent to server')
      
    } catch (error) {
      console.error('Failed to send FCM token to server:', error)
    }
  }
  
  private async sendSubscriptionToServer(subscription: PushSubscription): Promise<void> {
    try {
      const subscriptionData = this.subscriptionToData(subscription)
      
      const response = await fetch('/api/v1/notifications/subscribe', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}` // Get from auth service
        },
        body: JSON.stringify({
          subscription: subscriptionData,
          topics: ['build.failed', 'agent.error', 'task.completed', 'human.approval.request']
        })
      })
      
      if (!response.ok) {
        throw new Error('Failed to register push subscription with server')
      }
      
      console.log('✅ Push subscription sent to server')
      
    } catch (error) {
      console.error('Failed to send subscription to server:', error)
      throw error
    }
  }
  
  private async removeSubscriptionFromServer(): Promise<void> {
    try {
      const response = await fetch('/api/v1/notifications/unsubscribe', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
        }
      })
      
      if (!response.ok) {
        console.warn('Failed to remove subscription from server')
      }
      
    } catch (error) {
      console.error('Failed to remove subscription from server:', error)
    }
  }
  
  private subscriptionToData(subscription: PushSubscription): PushSubscriptionData {
    const key = subscription.getKey('p256dh')
    const auth = subscription.getKey('auth')
    
    return {
      endpoint: subscription.endpoint,
      keys: {
        p256dh: key ? btoa(String.fromCharCode(...new Uint8Array(key))) : '',
        auth: auth ? btoa(String.fromCharCode(...new Uint8Array(auth))) : ''
      }
    }
  }
  
  private urlBase64ToUint8Array(base64String: string): Uint8Array {
    const padding = '='.repeat((4 - base64String.length % 4) % 4)
    const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/')
    const rawData = window.atob(base64)
    const outputArray = new Uint8Array(rawData.length)
    
    for (let i = 0; i < rawData.length; ++i) {
      outputArray[i] = rawData.charCodeAt(i)
    }
    
    return outputArray
  }
  
  private handleNotificationClick(data: any): void {
    console.log('Notification clicked:', data)
    this.emit('notification_click', data)
    
    // Handle different notification types
    if (data?.action) {
      this.emit(`notification_action:${data.action}`, data)
    }
    
    // Focus the app window
    if ('clients' in self) {
      // This would be in the service worker context
    } else {
      window.focus()
    }
  }
  
  private handleNotificationClose(data: any): void {
    console.log('Notification closed:', data)
    this.emit('notification_close', data)
  }
  
  private handlePushReceived(data: any): void {
    console.log('Push message received:', data)
    this.emit('push_received', data)
  }
  
  // Public getters
  getPermission(): NotificationPermission {
    return this.permission
  }
  
  isSupported(): boolean {
    return 'Notification' in window && 'serviceWorker' in navigator
  }
  
  isPushSupported(): boolean {
    return 'PushManager' in window
  }
  
  isSubscribedToPush(): boolean {
    return this.pushSubscription !== null
  }
  
  getPushSubscription(): PushSubscription | null {
    return this.pushSubscription
  }
  
  // Helper methods for common notification types
  async showBuildFailedNotification(buildInfo: any): Promise<void> {
    await this.showNotification({
      title: '🚨 Build Failed',
      body: `Build #${buildInfo.id} failed: ${buildInfo.error}`,
      category: 'build',
      priority: 'high',
      requireInteraction: true,
      tag: `build-${buildInfo.id}`,
      data: { type: 'build_failed', buildId: buildInfo.id },
      actions: [
        { action: 'view_logs', title: 'View Logs' },
        { action: 'retry_build', title: 'Retry Build' }
      ]
    })
  }
  
  async showAgentErrorNotification(agentInfo: any): Promise<void> {
    await this.showNotification({
      title: '⚠️ Agent Error',
      body: `Agent ${agentInfo.name} encountered an error: ${agentInfo.error}`,
      category: 'agent',
      priority: 'high',
      tag: `agent-${agentInfo.id}`,
      data: { type: 'agent_error', agentId: agentInfo.id }
    })
  }
  
  async showTaskCompletedNotification(taskInfo: any): Promise<void> {
    await this.showNotification({
      title: '✅ Task Completed',
      body: `Task "${taskInfo.title}" has been completed successfully`,
      category: 'task',
      priority: 'normal',
      tag: `task-${taskInfo.id}`,
      data: { type: 'task_completed', taskId: taskInfo.id }
    })
  }
  
  async showApprovalRequestNotification(requestInfo: any): Promise<void> {
    await this.showNotification({
      title: '👤 Approval Needed',
      body: `${requestInfo.title} requires your approval`,
      category: 'approval',
      priority: 'high',
      requireInteraction: true,
      tag: `approval-${requestInfo.id}`,
      data: { type: 'approval_request', requestId: requestInfo.id },
      actions: [
        { action: 'approve', title: 'Approve' },
        { action: 'reject', title: 'Reject' },
        { action: 'view_details', title: 'View Details' }
      ]
    })
  }

  // Network monitoring for offline support
  private setupNetworkMonitoring(): void {
    window.addEventListener('online', () => {
      console.log('🌐 Network: Online')
      this.isOnline = true
      this.processOfflineQueue()
    })
    
    window.addEventListener('offline', () => {
      console.log('📵 Network: Offline')
      this.isOnline = false
    })
    
    // Also check using navigator.onLine
    this.isOnline = navigator.onLine
  }
  
  // Offline notification queuing
  private async queueNotification(data: Partial<NotificationData>): Promise<void> {
    const notification: NotificationData = {
      id: data.id || crypto.randomUUID(),
      title: data.title || 'Agent Hive',
      body: data.body || '',
      icon: data.icon || '/icons/icon-192x192.png',
      badge: data.badge || '/icons/icon-96x96.png',
      tag: data.tag,
      data: data.data,
      actions: data.actions,
      timestamp: Date.now(),
      priority: data.priority || 'normal',
      category: data.category || 'general',
      requireInteraction: data.requireInteraction || false
    }
    
    this.offlineQueue.push(notification)
    await this.saveOfflineQueue()
    
    console.log(`📦 Queued notification for offline: ${notification.title}`)
    this.emit('notification_queued', notification)
  }
  
  private async loadOfflineQueue(): Promise<void> {
    try {
      const stored = localStorage.getItem('notification_queue')
      if (stored) {
        this.offlineQueue = JSON.parse(stored)
        console.log(`📦 Loaded ${this.offlineQueue.length} queued notifications`)
      }
    } catch (error) {
      console.error('Failed to load offline notification queue:', error)
      this.offlineQueue = []
    }
  }
  
  private async saveOfflineQueue(): Promise<void> {
    try {
      localStorage.setItem('notification_queue', JSON.stringify(this.offlineQueue))
    } catch (error) {
      console.error('Failed to save offline notification queue:', error)
    }
  }
  
  private async processOfflineQueue(): Promise<void> {
    if (this.offlineQueue.length === 0) return
    
    console.log(`📦 Processing ${this.offlineQueue.length} queued notifications`)
    
    const notifications = [...this.offlineQueue]
    this.offlineQueue = []
    await this.saveOfflineQueue()
    
    for (const notification of notifications) {
      try {
        await this.showNotification(notification)
        await new Promise(resolve => setTimeout(resolve, 500)) // Throttle notifications
      } catch (error) {
        console.error('Failed to process queued notification:', error)
        // Re-queue failed notifications
        this.offlineQueue.push(notification)
      }
    }
    
    if (this.offlineQueue.length > 0) {
      await this.saveOfflineQueue()
    }
  }
  
  // iOS install prompt handling
  private handleIOSInstallPrompt(): void {
    if (!this.isIOS() || this.isInStandaloneMode() || this.iosInstallPromptShown) {
      return
    }
    
    // Show iOS install prompt after a delay
    setTimeout(() => {
      if (this.permission === 'granted') {
        this.showIOSInstallPrompt()
        this.iosInstallPromptShown = true
      }
    }, 10000) // Show after 10 seconds
  }
  
  private async showIOSInstallPrompt(): Promise<void> {
    await this.showNotification({
      title: '📱 Install Agent Hive',
      body: 'Add to Home Screen for the best experience with push notifications',
      category: 'install',
      priority: 'normal',
      requireInteraction: true,
      tag: 'ios-install-prompt',
      data: { type: 'ios_install_prompt' },
      actions: [
        { action: 'install_guide', title: 'Show How' },
        { action: 'dismiss_install', title: 'Not Now' }
      ]
    })
  }
  
  // Enhanced mobile notification methods
  async showCriticalMobileAlert(title: string, body: string, data?: any): Promise<void> {
    if (!this.isMobile) {
      return this.showNotification({ title, body, data, priority: 'high' })
    }
    
    // Mobile-optimized critical alert
    await this.showNotification({
      title: `🚨 ${title}`,
      body,
      data,
      priority: 'high',
      requireInteraction: true,
      category: 'critical',
      actions: [
        { action: 'view_dashboard', title: 'Open Dashboard' },
        { action: 'dismiss', title: 'OK' }
      ]
    })
  }
  
  async showMobileTaskUpdate(taskInfo: any): Promise<void> {
    const title = taskInfo.status === 'completed' ? '✅ Task Complete' : '🔄 Task Update'
    const body = this.isMobile 
      ? `${taskInfo.title}` // Shorter for mobile
      : `Task "${taskInfo.title}" status: ${taskInfo.status}`
    
    await this.showNotification({
      title,
      body,
      category: 'task',
      priority: taskInfo.status === 'completed' ? 'normal' : 'low',
      tag: `task-${taskInfo.id}`,
      data: { type: 'task_update', taskId: taskInfo.id, status: taskInfo.status }
    })
  }
  
  // Get notification statistics
  getNotificationStats(): {
    permission: NotificationPermission
    isSupported: boolean
    isPushSupported: boolean
    isSubscribed: boolean
    isMobile: boolean
    isIOS: boolean
    isOnline: boolean
    queueLength: number
    fcmToken: string | null
  } {
    return {
      permission: this.permission,
      isSupported: this.isSupported(),
      isPushSupported: this.isPushSupported(),
      isSubscribed: this.isSubscribedToPush(),
      isMobile: this.isMobile,
      isIOS: this.isIOS(),
      isOnline: this.isOnline,
      queueLength: this.offlineQueue.length,
      fcmToken: this.fcmToken
    }
  }
  
  // Clear notification queue (for testing/debugging)
  async clearNotificationQueue(): Promise<void> {
    this.offlineQueue = []
    await this.saveOfflineQueue()
    console.log('🗑️ Cleared notification queue')
  }
}

// Factory function for service consistency
export function getNotificationService(): NotificationService {
  return NotificationService.getInstance()
}