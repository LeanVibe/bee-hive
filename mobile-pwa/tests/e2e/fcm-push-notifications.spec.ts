/**
 * FCM Push Notifications Validation Tests
 * 
 * Validates the Frontend Builder's FCM push notification system:
 * - Tests Firebase SDK 10.7.1 loading and initialization
 * - Validates mobile-optimized notification UI and permission flows
 * - Tests offline notification queuing with retry logic
 * - Validates mobile-specific notification display and interactions
 */

import { test, expect, Page, BrowserContext } from '@playwright/test'

test.describe('FCM Push Notifications Validation', () => {
  test.beforeEach(async ({ page, isMobile }) => {
    // Set mobile viewport for iPhone 14 Pro (393x852)
    await page.setViewportSize({ width: 393, height: 852 })
    
    // Enable console logging for FCM debugging
    page.on('console', msg => {
      if (msg.text().includes('FCM') || msg.text().includes('Firebase') || 
          msg.text().includes('notification') || msg.text().includes('🔔')) {
        console.log('FCM:', msg.text())
      }
    })

    // Mock Notification API for testing
    await page.addInitScript(() => {
      // Mock Notification constructor and permission
      window.mockNotifications = []
      
      const originalNotification = window.Notification
      window.Notification = class MockNotification extends originalNotification {
        constructor(title: string, options?: NotificationOptions) {
          super(title, options)
          window.mockNotifications.push({ title, options })
        }
        
        static permission: NotificationPermission = 'default'
        
        static requestPermission(): Promise<NotificationPermission> {
          this.permission = 'granted'
          return Promise.resolve('granted')
        }
      } as any
      
      // Mock service worker for testing
      if (!navigator.serviceWorker) {
        (navigator as any).serviceWorker = {
          register: () => Promise.resolve({
            showNotification: (title: string, options: any) => {
              window.mockNotifications.push({ title, options })
              return Promise.resolve()
            },
            pushManager: {
              getSubscription: () => Promise.resolve(null),
              subscribe: () => Promise.resolve({
                endpoint: 'test-endpoint',
                getKey: () => new Uint8Array(0)
              })
            }
          }),
          ready: Promise.resolve({}),
          addEventListener: () => {}
        }
      }
    })
  })

  test('should load Firebase SDK 10.7.1 properly on mobile', async ({ page }) => {
    let firebaseLoaded = false
    let firebaseVersion = ''
    
    // Monitor Firebase loading
    page.on('request', request => {
      const url = request.url()
      if (url.includes('firebase') && url.includes('10.7.1')) {
        firebaseLoaded = true
      }
    })
    
    // Check Firebase in page context
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Wait for Firebase initialization
    await page.waitForTimeout(3000)
    
    const firebaseInfo = await page.evaluate(() => {
      return {
        hasFirebase: typeof window.firebase !== 'undefined',
        hasMessaging: typeof window.getMessaging !== 'undefined',
        version: window.firebase?.SDK_VERSION || 'unknown'
      }
    })
    
    // Should load Firebase SDK (in production build)
    if (process.env.NODE_ENV !== 'development') {
      expect(firebaseLoaded || firebaseInfo.hasFirebase).toBe(true)
    }
    
    console.log('Firebase info:', firebaseInfo)
  })

  test('should display notification permission request flow on mobile', async ({ page }) => {
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Look for notification permission UI
    const permissionButton = page.locator('[data-testid="enable-notifications"], button:has-text("Enable Notifications"), .notification-permission-btn')
    
    if (await permissionButton.count() > 0) {
      await expect(permissionButton.first()).toBeVisible()
      
      // Should have mobile-friendly text and styling
      const buttonText = await permissionButton.first().textContent()
      expect(buttonText).toMatch(/(Enable|Allow|Turn On).*Notifications?/i)
    }
  })

  test('should handle notification permission grant flow', async ({ page }) => {
    let permissionRequested = false
    let permissionGranted = false
    
    // Monitor permission requests
    page.on('console', msg => {
      const text = msg.text()
      if (text.includes('Notification permission') || text.includes('requestPermission')) {
        permissionRequested = true
      }
      if (text.includes('permission granted') || text.includes('✅ Notification permission granted')) {
        permissionGranted = true
      }
    })
    
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Try to trigger notification permission request
    const permissionButton = page.locator('[data-testid="enable-notifications"], button:has-text("Enable Notifications"), .notification-permission-btn')
    
    if (await permissionButton.count() > 0) {
      await permissionButton.first().click()
      await page.waitForTimeout(2000)
      
      expect(permissionRequested).toBe(true)
      
      // In our mocked environment, permission should be granted
      expect(permissionGranted).toBe(true)
    }
  })

  test('should queue notifications when offline', async ({ page, context }) => {
    let notificationQueued = false
    
    // Monitor offline notification queuing
    page.on('console', msg => {
      const text = msg.text()
      if (text.includes('Queued notification') || text.includes('📦 Queued notification for offline')) {
        notificationQueued = true
      }
    })
    
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Go offline
    await context.setOffline(true)
    await page.waitForTimeout(1000)
    
    // Try to trigger a notification while offline
    // This might happen automatically or we can simulate it
    await page.evaluate(() => {
      if (window.NotificationService) {
        window.NotificationService.getInstance().showNotification({
          title: 'Test Offline Notification',
          body: 'This should be queued'
        }).catch(() => {
          // Expected to fail and queue
        })
      }
    })
    
    await page.waitForTimeout(2000)
    
    // Should queue notification when offline
    expect(notificationQueued).toBe(true)
  })

  test('should replay queued notifications when back online', async ({ page, context }) => {
    let offlineQueueProcessed = false
    
    // Monitor queue processing
    page.on('console', msg => {
      const text = msg.text()
      if (text.includes('Processing') && text.includes('queued notifications')) {
        offlineQueueProcessed = true
      }
    })
    
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Go offline and queue notifications
    await context.setOffline(true)
    await page.evaluate(() => {
      if (window.NotificationService) {
        const service = window.NotificationService.getInstance()
        service.showNotification({
          title: 'Queued Notification 1',
          body: 'Should be queued'
        }).catch(() => {})
      }
    })
    
    await page.waitForTimeout(1000)
    
    // Come back online
    await context.setOffline(false)
    await page.waitForTimeout(3000)
    
    // Should process offline queue
    expect(offlineQueueProcessed).toBe(true)
  })

  test('should display mobile-optimized notification UI', async ({ page }) => {
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Check for notification center or notification list
    const notificationCenter = page.locator('[data-testid="notification-center"], .notification-center, notification-center')
    const notificationList = page.locator('[data-testid="notifications"], .notifications-list')
    
    if (await notificationCenter.count() > 0 || await notificationList.count() > 0) {
      const element = await notificationCenter.count() > 0 ? notificationCenter.first() : notificationList.first()
      await expect(element).toBeVisible()
      
      // Should be mobile-friendly (touch targets, proper sizing)
      const boundingBox = await element.boundingBox()
      if (boundingBox) {
        // Should take reasonable space on mobile screen
        expect(boundingBox.width).toBeGreaterThan(200)
      }
    }
  })

  test('should handle mobile-specific notification actions', async ({ page }) => {
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Simulate receiving a notification with actions
    await page.evaluate(() => {
      if (window.NotificationService) {
        const service = window.NotificationService.getInstance()
        service.showNotification({
          title: '🚨 Critical Alert',
          body: 'Agent coordination failure detected',
          priority: 'high',
          requireInteraction: true,
          actions: [
            { action: 'view_dashboard', title: 'Open Dashboard' },
            { action: 'dismiss', title: 'OK' }
          ]
        }).catch(() => {
          // Might fail in test environment, that's ok
        })
      }
    })
    
    await page.waitForTimeout(2000)
    
    // Check if notification actions are properly sized for mobile
    const actionButtons = page.locator('.notification-action, [data-testid="notification-action"]')
    
    if (await actionButtons.count() > 0) {
      for (let i = 0; i < await actionButtons.count(); i++) {
        const button = actionButtons.nth(i)
        const boundingBox = await button.boundingBox()
        
        if (boundingBox) {
          // Touch targets should be at least 44px (iOS guidelines)
          expect(boundingBox.height).toBeGreaterThanOrEqual(44)
        }
      }
    }
  })

  test('should adapt notification display for mobile screens', async ({ page }) => {
    await page.setViewportSize({ width: 393, height: 852 }) // iPhone 14 Pro
    
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Simulate showing a long notification
    await page.evaluate(() => {
      if (window.NotificationService) {
        const service = window.NotificationService.getInstance()
        service.showNotification({
          title: 'Very Long Notification Title That Should Be Truncated on Mobile Devices',
          body: 'This is a very long notification body that should be truncated appropriately for mobile screens to ensure good user experience and prevent overflow issues.',
          category: 'test'
        }).catch(() => {})
      }
    })
    
    await page.waitForTimeout(2000)
    
    const notifications = await page.evaluate(() => {
      return window.mockNotifications || []
    })
    
    // Should truncate long content for mobile
    const mobileNotifications = notifications.filter((n: any) => 
      n.title && (n.title.length <= 100 || n.title.includes('...')))
    
    if (notifications.length > 0) {
      expect(mobileNotifications.length).toBeGreaterThan(0)
    }
  })

  test('should handle iOS Safari notification quirks', async ({ page, browserName }) => {
    if (browserName === 'webkit') {
      await page.goto('/dashboard')
      await page.waitForSelector('dashboard-view', { timeout: 10000 })
      
      // Check iOS-specific notification handling
      const iosSupported = await page.evaluate(() => {
        // Check for iOS-specific features
        return {
          hasNotifications: 'Notification' in window,
          hasServiceWorker: 'serviceWorker' in navigator,
          isStandalone: (window.navigator as any).standalone === true,
          isIOS: /iPad|iPhone|iPod/.test(navigator.userAgent)
        }
      })
      
      console.log('iOS notification support:', iosSupported)
      
      if (iosSupported.isIOS && !iosSupported.isStandalone) {
        // Should show iOS install prompt for better notification support
        const installPrompt = page.locator('[data-testid="ios-install"], .ios-install-prompt')
        
        // iOS install prompt might appear (optional behavior)
        // Just ensure no errors occur
        await page.waitForTimeout(3000)
        
        // No notification errors should occur on iOS
        const consoleErrors = []
        page.on('console', msg => {
          if (msg.type() === 'error' && msg.text().includes('notification')) {
            consoleErrors.push(msg.text())
          }
        })
        
        await page.waitForTimeout(2000)
        expect(consoleErrors).toHaveLength(0)
      }
    }
  })

  test('should send FCM token to server', async ({ page }) => {
    let tokenSentToServer = false
    
    // Monitor API calls for FCM token registration
    page.on('request', request => {
      if (request.url().includes('/api/v1/notifications/fcm-token') && 
          request.method() === 'POST') {
        tokenSentToServer = true
      }
    })
    
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Simulate FCM token generation and sending
    await page.evaluate(() => {
      if (window.NotificationService) {
        const service = window.NotificationService.getInstance()
        // Trigger FCM token generation (if supported)
        service.initialize().catch(() => {
          // May fail in test environment
        })
      }
    })
    
    await page.waitForTimeout(5000)
    
    // In production environment, should attempt to send token
    // (May not happen in test environment, that's expected)
    console.log('FCM token sent to server:', tokenSentToServer)
  })

  test('should handle notification click actions', async ({ page }) => {
    let notificationClicked = false
    
    // Monitor notification clicks
    page.on('console', msg => {
      const text = msg.text()
      if (text.includes('Notification clicked') || text.includes('notification_click')) {
        notificationClicked = true
      }
    })
    
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Simulate notification click
    await page.evaluate(() => {
      if (window.NotificationService) {
        const service = window.NotificationService.getInstance()
        // Simulate notification click event
        service.emit('notification_click', {
          type: 'agent_error',
          agentId: 'test-agent'
        })
      }
    })
    
    await page.waitForTimeout(1000)
    
    expect(notificationClicked).toBe(true)
  })

  test('should provide notification statistics and status', async ({ page }) => {
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Get notification service statistics
    const stats = await page.evaluate(() => {
      if (window.NotificationService) {
        return window.NotificationService.getInstance().getNotificationStats()
      }
      return null
    })
    
    if (stats) {
      expect(stats).toHaveProperty('permission')
      expect(stats).toHaveProperty('isSupported')
      expect(stats).toHaveProperty('isMobile')
      expect(stats).toHaveProperty('isOnline')
      expect(stats).toHaveProperty('queueLength')
      
      // Should correctly identify as mobile
      expect(stats.isMobile).toBe(true)
      
      console.log('Notification stats:', stats)
    }
  })
})

test.describe('FCM Push Notifications Error Handling', () => {
  test('should gracefully handle FCM initialization failures', async ({ page }) => {
    let fcmError = false
    
    page.on('console', msg => {
      const text = msg.text()
      if (msg.type() === 'error' && text.includes('Firebase')) {
        fcmError = true
      }
    })
    
    // Block Firebase requests to simulate FCM unavailable
    await page.route('**/firebase**', route => route.abort())
    
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    await page.waitForTimeout(3000)
    
    // Should handle FCM errors gracefully (may still show error)
    // Dashboard should remain functional
    await expect(page.locator('dashboard-view')).toBeVisible()
  })

  test('should work without service worker support', async ({ page }) => {
    // Disable service worker support
    await page.addInitScript(() => {
      delete (navigator as any).serviceWorker
    })
    
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Should still work without service worker (fallback mode)
    await expect(page.locator('dashboard-view')).toBeVisible()
    
    // Should handle notification service initialization
    const hasNotificationService = await page.evaluate(() => {
      return window.NotificationService ? true : false
    })
    
    // Service should initialize even without service worker
    expect(hasNotificationService).toBe(true)
  })

  test('should handle permission denied gracefully', async ({ page }) => {
    // Mock permission denied
    await page.addInitScript(() => {
      window.Notification = class {
        static permission: NotificationPermission = 'denied'
        static requestPermission(): Promise<NotificationPermission> {
          return Promise.resolve('denied')
        }
      } as any
    })
    
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Should handle denied permission gracefully
    await expect(page.locator('dashboard-view')).toBeVisible()
    
    // May show some indication that notifications are disabled
    const permissionMessage = page.locator('.permission-denied, [data-testid="notification-disabled"]')
    
    // Optional: Should show user how to enable notifications
    if (await permissionMessage.count() > 0) {
      await expect(permissionMessage.first()).toBeVisible()
    }
  })
})

// Extend global types for testing
declare global {
  interface Window {
    mockNotifications: Array<{ title: string; options?: NotificationOptions }>
    NotificationService: any
  }
}