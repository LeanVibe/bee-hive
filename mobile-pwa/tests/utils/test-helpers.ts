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