import { Page, expect } from '@playwright/test'
import { mockTasks, mockAgents, mockEvents, mockSystemHealth, mockPerformanceMetrics } from '../fixtures/test-data'

/**
 * Test helper utilities for common testing operations
 */
export class TestHelpers {
  
  /**
   * Wait for element to be visible with custom timeout
   */
  static async waitForElement(page: Page, selector: string, timeout: number = 10000) {
    await page.waitForSelector(selector, { state: 'visible', timeout })
  }

  /**
   * Wait for element to be hidden
   */
  static async waitForElementToBeHidden(page: Page, selector: string, timeout: number = 10000) {
    await page.waitForSelector(selector, { state: 'hidden', timeout })
  }

  /**
   * Wait for network requests to complete
   */
  static async waitForNetworkIdle(page: Page, timeout: number = 5000) {
    await page.waitForLoadState('networkidle', { timeout })
  }

  /**
   * Disable API mocks to test against real backend
   */
  static async disableAPIMocks(page: Page) {
    // Remove any existing route handlers that mock API calls
    await page.unroute('**/api/**')
    
    // Set a flag in the page context to indicate real backend testing
    await page.addInitScript(() => {
      window.__TEST_REAL_BACKEND__ = true
    })
  }

  /**
   * Verify the backend is accessible and ready
   */
  static async verifyBackendConnection(page: Page, baseUrl: string = 'http://localhost:8000') {
    try {
      const response = await page.request.get(`${baseUrl}/health`)
      return response.status() === 200
    } catch (error) {
      return false
    }
  }

  /**
   * Simulate slow network conditions
   */
  static async simulateSlowNetwork(page: Page) {
    const client = await page.context().newCDPSession(page)
    await client.send('Network.emulateNetworkConditions', {
      offline: false,
      downloadThroughput: 1000, // 1kb/s
      uploadThroughput: 1000,
      latency: 500 // 500ms latency
    })
  }

  /**
   * Simulate offline network conditions
   */
  static async simulateOfflineNetwork(page: Page) {
    await page.context().setOffline(true)
  }

  /**
   * Restore normal network conditions
   */
  static async restoreNetwork(page: Page) {
    await page.context().setOffline(false)
    const client = await page.context().newCDPSession(page)
    await client.send('Network.emulateNetworkConditions', {
      offline: false,
      downloadThroughput: -1, // No throttling
      uploadThroughput: -1,
      latency: 0
    })
  }

  /**
   * Mock API responses for dashboard data
   */
  static async mockDashboardAPI(page: Page) {
    // Mock tasks API
    await page.route('**/api/v1/tasks', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockTasks)
      })
    })

    // Mock agents API
    await page.route('**/api/v1/agents', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockAgents)
      })
    })

    // Mock events API
    await page.route('**/api/v1/events', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockEvents)
      })
    })

    // Mock system health API
    await page.route('**/api/v1/health', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockSystemHealth)
      })
    })

    // Mock performance metrics API
    await page.route('**/api/v1/metrics', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockPerformanceMetrics)
      })
    })
  }

  /**
   * Mock API error responses
   */
  static async mockAPIErrors(page: Page) {
    await page.route('**/api/v1/**', async route => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Internal Server Error' })
      })
    })
  }

  /**
   * Mock WebSocket connections
   */
  static async mockWebSocket(page: Page) {
    await page.addInitScript(() => {
      // Mock WebSocket for real-time updates
      class MockWebSocket {
        url: string
        readyState: number = WebSocket.CONNECTING
        onopen: ((event: Event) => void) | null = null
        onmessage: ((event: MessageEvent) => void) | null = null
        onclose: ((event: CloseEvent) => void) | null = null
        onerror: ((event: Event) => void) | null = null

        constructor(url: string) {
          this.url = url
          setTimeout(() => {
            this.readyState = WebSocket.OPEN
            if (this.onopen) {
              this.onopen(new Event('open'))
            }
            
            // Send mock updates periodically
            this.sendMockUpdates()
          }, 100)
        }

        send(data: string) {
          console.log('WebSocket send:', data)
        }

        close() {
          this.readyState = WebSocket.CLOSED
          if (this.onclose) {
            this.onclose(new CloseEvent('close'))
          }
        }

        private sendMockUpdates() {
          const updates = [
            { type: 'task-updated', data: { id: 'task-1', status: 'in-progress' } },
            { type: 'agent-status', data: { id: 'agent-1', status: 'active' } },
            { type: 'system-health', data: { overall: 'healthy' } }
          ]

          updates.forEach((update, index) => {
            setTimeout(() => {
              if (this.onmessage && this.readyState === WebSocket.OPEN) {
                this.onmessage(new MessageEvent('message', { data: JSON.stringify(update) }))
              }
            }, (index + 1) * 2000)
          })
        }
      }

      // Replace global WebSocket
      (window as any).WebSocket = MockWebSocket
    })
  }

  /**
   * Take screenshot with timestamp
   */
  static async takeTimestampedScreenshot(page: Page, name: string) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
    await page.screenshot({ 
      path: `test-results/screenshots/${name}-${timestamp}.png`,
      fullPage: true
    })
  }

  /**
   * Verify loading states
   */
  static async verifyLoadingState(page: Page, elementSelector: string) {
    // Should show loading indicator
    await expect(page.locator('.spinner, .loading')).toBeVisible()
    
    // Loading should complete
    await page.waitForSelector('.spinner, .loading', { state: 'hidden', timeout: 10000 })
    
    // Content should be visible
    await expect(page.locator(elementSelector)).toBeVisible()
  }

  /**
   * Verify error handling
   */
  static async verifyErrorHandling(page: Page, expectedErrorMessage?: string) {
    const errorElement = page.locator('.error-state, .error-message, [data-testid="error"]')
    await expect(errorElement).toBeVisible()
    
    if (expectedErrorMessage) {
      await expect(errorElement).toContainText(expectedErrorMessage)
    }
  }

  /**
   * Generate unique test ID
   */
  static generateTestId(): string {
    return `test-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
  }

  /**
   * Enhanced WebSocket connection testing
   */
  static async waitForWebSocketConnection(page: Page, timeout: number = 10000): Promise<boolean> {
    return new Promise((resolve) => {
      let wsConnected = false
      
      const timeoutId = setTimeout(() => {
        if (!wsConnected) {
          console.log('WebSocket connection timeout')
          resolve(false)
        }
      }, timeout)
      
      page.on('websocket', ws => {
        ws.on('open', () => {
          wsConnected = true
          clearTimeout(timeoutId)
          console.log('WebSocket connected successfully')
          resolve(true)
        })
      })
      
      // Check if already connected
      page.evaluate(() => {
        if (window.websocketConnected === true) {
          return true
        }
        return false
      }).then(connected => {
        if (connected) {
          wsConnected = true
          clearTimeout(timeoutId)
          resolve(true)
        }
      })
    })
  }

  /**
   * Validate Core Web Vitals
   */
  static async validateCoreWebVitals(page: Page): Promise<any> {
    return await page.evaluate(() => {
      return new Promise((resolve) => {
        const vitals: any = {}
        let completedObservers = 0
        const totalObservers = 3
        
        const checkComplete = () => {
          completedObservers++
          if (completedObservers === totalObservers) {
            resolve(vitals)
          }
        }
        
        // First Contentful Paint
        const fcpObserver = new PerformanceObserver((list) => {
          const entries = list.getEntries()
          const fcpEntry = entries.find(entry => entry.name === 'first-contentful-paint')
          if (fcpEntry) {
            vitals.fcp = fcpEntry.startTime
          }
          checkComplete()
        })
        fcpObserver.observe({ entryTypes: ['paint'] })
        
        // Largest Contentful Paint
        const lcpObserver = new PerformanceObserver((list) => {
          const entries = list.getEntries()
          const lastEntry = entries[entries.length - 1]
          vitals.lcp = lastEntry.startTime
          checkComplete()
        })
        lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] })
        
        // Cumulative Layout Shift
        let clsValue = 0
        const clsObserver = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (!(entry as any).hadRecentInput) {
              clsValue += (entry as any).value
            }
          }
          vitals.cls = clsValue
          checkComplete()
        })
        clsObserver.observe({ entryTypes: ['layout-shift'] })
        
        // Set timeout to ensure resolution
        setTimeout(() => {
          resolve(vitals)
        }, 5000)
      })
    })
  }

  /**
   * Check PWA installability
   */
  static async checkPWAInstallability(page: Page): Promise<any> {
    return await page.evaluate(async () => {
      const pwaInfo: any = {
        hasManifest: false,
        hasServiceWorker: false,
        isInstallable: false,
        manifestValid: false
      }
      
      // Check manifest
      const manifestLink = document.querySelector('link[rel="manifest"]')
      if (manifestLink) {
        pwaInfo.hasManifest = true
        
        try {
          const manifestUrl = manifestLink.getAttribute('href')
          if (manifestUrl) {
            const response = await fetch(manifestUrl)
            const manifest = await response.json()
            
            pwaInfo.manifestValid = !!(
              manifest.name && 
              manifest.start_url && 
              manifest.display && 
              manifest.icons && 
              manifest.icons.length > 0
            )
          }
        } catch (error) {
          console.error('Manifest validation error:', error)
        }
      }
      
      // Check service worker
      if ('serviceWorker' in navigator) {
        try {
          const registration = await navigator.serviceWorker.getRegistration()
          pwaInfo.hasServiceWorker = !!registration
        } catch (error) {
          console.error('Service worker check error:', error)
        }
      }
      
      // Check for install prompt
      pwaInfo.isInstallable = !!(window as any).deferredPrompt
      
      return pwaInfo
    })
  }

  /**
   * Simulate user interactions for testing
   */
  static async simulateUserBehavior(page: Page, scenario: 'browsing' | 'task-creation' | 'agent-monitoring'): Promise<void> {
    switch (scenario) {
      case 'browsing':
        await page.mouse.move(100, 100)
        await page.waitForTimeout(500)
        await page.mouse.move(300, 200)
        await page.waitForTimeout(300)
        await page.keyboard.press('Tab')
        await page.waitForTimeout(200)
        await page.evaluate(() => window.scrollTo(0, 100))
        await page.waitForTimeout(300)
        break
        
      case 'task-creation':
        const taskButton = page.locator('[data-testid="new-task"], .new-task-button, button:has-text("New Task")')
        if (await taskButton.count() > 0) {
          await taskButton.first().click()
          await this.waitForAnimations(page)
          
          const descriptionField = page.locator('input[name="description"], textarea')
          if (await descriptionField.count() > 0) {
            await descriptionField.first().fill('Simulated task creation test')
          }
        }
        break
        
      case 'agent-monitoring':
        const agentCards = page.locator('.agent-card, .agent-status')
        const count = await agentCards.count()
        
        for (let i = 0; i < Math.min(count, 3); i++) {
          await agentCards.nth(i).hover()
          await page.waitForTimeout(200)
        }
        break
    }
  }

  /**
   * Calculate color contrast ratio
   */
  static calculateColorContrast(color1: string, color2: string): number {
    const getLuminance = (color: string): number => {
      const rgb = color.match(/\d+/g)
      if (!rgb) return 0
      
      const [r, g, b] = rgb.map(c => {
        const channel = parseInt(c) / 255
        return channel <= 0.03928 ? channel / 12.92 : Math.pow((channel + 0.055) / 1.055, 2.4)
      })
      
      return 0.2126 * r + 0.7152 * g + 0.0722 * b
    }
    
    const l1 = getLuminance(color1)
    const l2 = getLuminance(color2)
    const lighter = Math.max(l1, l2)
    const darker = Math.min(l1, l2)
    
    return (lighter + 0.05) / (darker + 0.05)
  }

  /**
   * Comprehensive accessibility check
   */
  static async checkAccessibility(page: Page): Promise<any> {
    return await page.evaluate(() => {
      const accessibility = {
        hasProperHeadings: false,
        hasSkipLinks: false,
        hasLandmarks: false,
        formLabelsCount: 0,
        unlabeledImages: 0,
        focusableElements: 0,
        ariaElements: 0
      }
      
      // Check heading structure
      const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6')
      accessibility.hasProperHeadings = headings.length > 0
      
      // Check skip links
      accessibility.hasSkipLinks = !!document.querySelector('a[href="#main"], a[href="#content"], .skip-link')
      
      // Check landmarks
      const landmarks = document.querySelectorAll('[role="banner"], [role="navigation"], [role="main"], [role="contentinfo"]')
      accessibility.hasLandmarks = landmarks.length > 0
      
      // Check form labels
      const formElements = document.querySelectorAll('input, select, textarea')
      formElements.forEach(element => {
        const hasLabel = element.labels?.length > 0 ||
                        element.hasAttribute('aria-label') ||
                        element.hasAttribute('aria-labelledby')
        if (hasLabel) {
          accessibility.formLabelsCount++
        }
      })
      
      // Check images for alt text
      const images = document.querySelectorAll('img')
      images.forEach(img => {
        if (!img.hasAttribute('alt') && !img.hasAttribute('role')) {
          accessibility.unlabeledImages++
        }
      })
      
      // Count focusable elements
      const focusable = document.querySelectorAll(
        'button, a, input, select, textarea, [tabindex="0"], [role="button"]'
      )
      accessibility.focusableElements = focusable.length
      
      // Count ARIA elements
      const ariaElements = document.querySelectorAll('[aria-label], [aria-labelledby], [role]')
      accessibility.ariaElements = ariaElements.length
      
      return accessibility
    })
  }

  /**
   * Monitor memory usage
   */
  static async monitorMemoryUsage(page: Page): Promise<any> {
    return await page.evaluate(() => {
      if ('memory' in performance) {
        const memory = (performance as any).memory
        return {
          usedJSHeapSize: memory.usedJSHeapSize,
          totalJSHeapSize: memory.totalJSHeapSize,
          jsHeapSizeLimit: memory.jsHeapSizeLimit,
          usagePercent: (memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100
        }
      }
      return null
    })
  }

  /**
   * Test mobile touch targets
   */
  static async validateTouchTargets(page: Page, minimumSize: number = 44): Promise<any> {
    const touchTargets = await page.locator('button, a, input, select, textarea, [role="button"], [tabindex="0"]').all()
    
    const results = {
      total: touchTargets.length,
      compliant: 0,
      nonCompliant: 0,
      undersizedElements: []
    }
    
    for (let i = 0; i < Math.min(touchTargets.length, 20); i++) {
      const element = touchTargets[i]
      const boundingBox = await element.boundingBox()
      
      if (boundingBox) {
        const minDimension = Math.min(boundingBox.width, boundingBox.height)
        
        if (minDimension >= minimumSize) {
          results.compliant++
        } else {
          results.nonCompliant++
          results.undersizedElements.push({
            size: { width: boundingBox.width, height: boundingBox.height },
            text: await element.textContent()
          })
        }
      }
    }
    
    return results
  }

  /**
   * Test WebSocket message handling
   */
  static async testWebSocketMessages(page: Page, timeout: number = 10000): Promise<any> {
    return new Promise((resolve) => {
      const messages: any[] = []
      const connections: any[] = []
      
      const timeoutId = setTimeout(() => {
        resolve({ messages, connections, timeout: true })
      }, timeout)
      
      page.on('websocket', ws => {
        const connection = {
          url: ws.url(),
          state: 'connecting',
          messagesReceived: 0,
          messagesSent: 0
        }
        
        connections.push(connection)
        
        ws.on('open', () => {
          connection.state = 'open'
        })
        
        ws.on('close', () => {
          connection.state = 'closed'
        })
        
        ws.on('framereceived', data => {
          connection.messagesReceived++
          try {
            const message = JSON.parse(data)
            messages.push({ type: 'received', message, timestamp: Date.now() })
          } catch (e) {
            messages.push({ type: 'received', raw: data, timestamp: Date.now() })
          }
        })
        
        ws.on('framesent', data => {
          connection.messagesSent++
          try {
            const message = JSON.parse(data)
            messages.push({ type: 'sent', message, timestamp: Date.now() })
          } catch (e) {
            messages.push({ type: 'sent', raw: data, timestamp: Date.now() })
          }
        })
      })
      
      // Check if we have enough data after some time
      setTimeout(() => {
        if (messages.length > 0 || connections.length > 0) {
          clearTimeout(timeoutId)
          resolve({ messages, connections, timeout: false })
        }
      }, 3000)
    })
  }

  /**
   * Wait for animations to complete
   */
  static async waitForAnimations(page: Page) {
    await page.waitForTimeout(500) // Allow for CSS transitions
  }

  /**
   * Verify responsive design breakpoints
   */
  static async testResponsiveBreakpoint(page: Page, width: number, height: number) {
    await page.setViewportSize({ width, height })
    await this.waitForAnimations(page)
  }

  /**
   * Test keyboard navigation
   */
  static async testKeyboardNavigation(page: Page, startElement: string, expectedElements: string[]) {
    await page.locator(startElement).focus()
    
    for (const element of expectedElements) {
      await page.keyboard.press('Tab')
      await expect(page.locator(element)).toBeFocused()
    }
  }

  /**
   * Test touch interactions
   */
  static async testTouchInteraction(page: Page, element: string, action: 'tap' | 'swipe' | 'pinch') {
    const locator = page.locator(element)
    
    switch (action) {
      case 'tap':
        await locator.tap()
        break
      case 'swipe':
        await locator.hover()
        await page.mouse.down()
        await page.mouse.move(100, 0)
        await page.mouse.up()
        break
      case 'pinch':
        // Simulate pinch gesture for zoom
        await page.touchscreen.tap(100, 100)
        await page.touchscreen.tap(200, 200)
        break
    }
  }

  /**
   * Monitor console errors during test execution
   */
  static async monitorConsoleErrors(page: Page): Promise<string[]> {
    const errors: string[] = []
    
    page.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(msg.text())
      }
    })
    
    page.on('pageerror', error => {
      errors.push(error.message)
    })
    
    return errors
  }

  /**
   * Verify accessibility requirements
   */
  static async verifyAccessibility(page: Page, element?: string) {
    const target = element ? page.locator(element) : page
    
    // Check for ARIA labels
    const interactiveElements = await target.locator('button, input, select, a, [tabindex]').all()
    
    for (const el of interactiveElements) {
      const ariaLabel = await el.getAttribute('aria-label')
      const title = await el.getAttribute('title')
      const textContent = await el.textContent()
      
      // Interactive elements should have accessible names
      if (!ariaLabel && !title && !textContent?.trim()) {
        console.warn('Interactive element missing accessible name:', await el.innerHTML())
      }
    }
  }

  /**
   * Simulate user typing with realistic delays
   */
  static async typeRealistic(page: Page, selector: string, text: string) {
    const element = page.locator(selector)
    await element.focus()
    
    for (const char of text) {
      await element.type(char)
      await page.waitForTimeout(Math.random() * 100 + 50) // 50-150ms per character
    }
  }

  /**
   * Wait for condition with polling
   */
  static async waitForCondition(
    condition: () => Promise<boolean>,
    timeout: number = 10000,
    interval: number = 500
  ): Promise<void> {
    const startTime = Date.now()
    
    while (Date.now() - startTime < timeout) {
      if (await condition()) {
        return
      }
      await new Promise(resolve => setTimeout(resolve, interval))
    }
    
    throw new Error(`Condition not met within ${timeout}ms`)
  }
}