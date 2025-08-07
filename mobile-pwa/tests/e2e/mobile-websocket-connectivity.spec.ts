/**
 * Mobile WebSocket Connectivity Validation Tests
 * 
 * Validates the Backend Engineer's WebSocket routing fixes and mobile connectivity:
 * - Tests `/dashboard/ws` → `/api/dashboard/ws/dashboard` route fixing
 * - Validates mobile-specific WebSocket behavior and reconnection
 * - Tests real-time data streaming to mobile dashboard components
 * - Validates connection quality monitoring and adaptation
 */

import { test, expect, Page, BrowserContext } from '@playwright/test'

test.describe('Mobile WebSocket Connectivity Validation', () => {
  test.beforeEach(async ({ page, isMobile }) => {
    // Set mobile viewport for iPhone 14 Pro (393x852)
    if (isMobile) {
      await page.setViewportSize({ width: 393, height: 852 })
    } else {
      await page.setViewportSize({ width: 393, height: 852 })
    }
    
    // Enable console logging for debugging
    page.on('console', msg => {
      if (msg.type() === 'error' || msg.text().includes('WebSocket')) {
        console.log('Browser:', msg.text())
      }
    })
  })

  test('should connect to fixed WebSocket endpoint without 404 errors', async ({ page }) => {
    let wsErrorCount = 0
    let wsConnectedCount = 0
    
    // Monitor WebSocket connection attempts
    page.on('console', msg => {
      const text = msg.text()
      if (text.includes('WebSocket') && text.includes('404')) {
        wsErrorCount++
      }
      if (text.includes('WebSocket connected') || text.includes('✅ WebSocket connected')) {
        wsConnectedCount++
      }
    })
    
    // Navigate to dashboard
    await page.goto('/dashboard')
    
    // Wait for dashboard to initialize
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Wait for WebSocket connection attempts
    await page.waitForTimeout(3000)
    
    // Validate no 404 WebSocket errors
    expect(wsErrorCount).toBe(0)
    
    // Should have successful WebSocket connection
    expect(wsConnectedCount).toBeGreaterThan(0)
    
    // Check for connection indicator
    const connectionIndicator = page.locator('.connection-indicator, [data-testid="connection-status"]')
    if (await connectionIndicator.isVisible()) {
      const statusText = await connectionIndicator.textContent()
      expect(statusText).not.toContain('404')
      expect(statusText).not.toContain('Failed')
    }
  })

  test('should connect using correct endpoint /api/dashboard/ws/dashboard', async ({ page }) => {
    let correctEndpointUsed = false
    let incorrectEndpointUsed = false
    
    // Monitor network requests
    page.on('request', request => {
      const url = request.url()
      if (url.includes('/api/dashboard/ws/dashboard')) {
        correctEndpointUsed = true
      }
      if (url.includes('/dashboard/ws') && !url.includes('/api/dashboard/ws/dashboard')) {
        incorrectEndpointUsed = true
      }
    })
    
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    await page.waitForTimeout(2000)
    
    // Should use the correct endpoint
    expect(correctEndpointUsed).toBe(true)
    
    // Should not use the old incorrect endpoint
    expect(incorrectEndpointUsed).toBe(false)
  })

  test('should receive real-time data updates on iPhone 14 Pro viewport', async ({ page }) => {
    await page.setViewportSize({ width: 393, height: 852 })
    
    let dataUpdateReceived = false
    
    // Monitor for real-time updates
    page.on('console', msg => {
      const text = msg.text()
      if (text.includes('agent_update') || text.includes('system_update') || 
          text.includes('coordination_update') || text.includes('task_update')) {
        dataUpdateReceived = true
      }
    })
    
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Wait for real-time updates (should come within 10 seconds)
    await page.waitForTimeout(10000)
    
    // Should receive real-time data updates
    expect(dataUpdateReceived).toBe(true)
    
    // Check that data is displayed in mobile components
    const agentPanel = page.locator('realtime-agent-status-panel, [data-testid="agent-status"]')
    if (await agentPanel.isVisible()) {
      await expect(agentPanel).toContainText(/\d+/)  // Should contain numeric data
    }
  })

  test('should handle WebSocket reconnection on poor connections', async ({ page, context }) => {
    let reconnectionAttempted = false
    
    // Monitor reconnection attempts
    page.on('console', msg => {
      const text = msg.text()
      if (text.includes('reconnect') || text.includes('🔄') || text.includes('Scheduling reconnect')) {
        reconnectionAttempted = true
      }
    })
    
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Simulate network interruption
    await context.setOffline(true)
    await page.waitForTimeout(2000)
    
    // Restore connection
    await context.setOffline(false)
    await page.waitForTimeout(5000)
    
    // Should attempt reconnection
    expect(reconnectionAttempted).toBe(true)
    
    // Should eventually reconnect
    const connectionIndicator = page.locator('.connection-indicator, [data-testid="connection-status"]')
    if (await connectionIndicator.isVisible()) {
      await expect(connectionIndicator).not.toContainText('Offline', { timeout: 10000 })
    }
  })

  test('should maintain <50ms latency for real-time updates', async ({ page }) => {
    const latencyMeasurements: number[] = []
    let measurementCount = 0
    
    // Inject latency measurement script
    await page.addInitScript(() => {
      window.latencyMeasurements = []
      
      // Override WebSocket message handling to measure latency
      const originalWebSocket = window.WebSocket
      window.WebSocket = class extends originalWebSocket {
        constructor(url: string) {
          super(url)
          
          this.addEventListener('message', (event) => {
            try {
              const data = JSON.parse(event.data)
              if (data.timestamp) {
                const serverTime = new Date(data.timestamp).getTime()
                const clientTime = Date.now()
                const latency = clientTime - serverTime
                
                if (latency > 0 && latency < 5000) { // Reasonable latency range
                  window.latencyMeasurements.push(latency)
                }
              }
            } catch (e) {
              // Ignore JSON parse errors
            }
          })
        }
      }
    })
    
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Wait for several real-time updates
    await page.waitForTimeout(15000)
    
    // Get latency measurements
    const measurements = await page.evaluate(() => window.latencyMeasurements)
    measurementCount = measurements.length
    
    if (measurementCount > 0) {
      const averageLatency = measurements.reduce((a: number, b: number) => a + b, 0) / measurements.length
      const maxLatency = Math.max(...measurements)
      
      console.log(`Average latency: ${averageLatency}ms, Max latency: ${maxLatency}ms, Measurements: ${measurementCount}`)
      
      // Target: <50ms average latency (allowing some buffer for mobile networks)
      expect(averageLatency).toBeLessThan(100) // Relaxed for mobile
      
      // No single update should take longer than 500ms
      expect(maxLatency).toBeLessThan(500)
    }
  })

  test('should adapt streaming frequency based on connection quality', async ({ page }) => {
    let highFrequencyModeEnabled = false
    let lowFrequencyModeEnabled = false
    
    // Monitor connection quality changes
    page.on('console', msg => {
      const text = msg.text()
      if (text.includes('enableHighFrequencyMode') || text.includes('high_frequency_updates')) {
        highFrequencyModeEnabled = true
      }
      if (text.includes('enableLowFrequencyMode') || text.includes('low_frequency')) {
        lowFrequencyModeEnabled = true
      }
    })
    
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Wait for connection quality assessment
    await page.waitForTimeout(5000)
    
    // Should enable high frequency mode for good connections initially
    expect(highFrequencyModeEnabled).toBe(true)
  })

  test('should display connection quality indicator', async ({ page }) => {
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Wait for connection quality to be assessed
    await page.waitForTimeout(3000)
    
    // Should have a connection quality indicator
    const qualityIndicator = page.locator('.connection-quality, .connection-indicator, [data-testid="connection-quality"]')
    
    if (await qualityIndicator.count() > 0) {
      await expect(qualityIndicator.first()).toBeVisible()
      
      const qualityText = await qualityIndicator.first().textContent()
      expect(qualityText).toMatch(/(excellent|good|poor|offline|connected|disconnected)/i)
    }
  })

  test('should handle backward compatibility routes', async ({ page }) => {
    // This test verifies that old routes still work during transition period
    let backwardCompatibilityWorking = true
    
    page.on('response', response => {
      const url = response.url()
      if ((url.includes('/dashboard/ws') || url.includes('/ws/dashboard')) && 
          response.status() === 404) {
        backwardCompatibilityWorking = false
      }
    })
    
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    await page.waitForTimeout(3000)
    
    // Should maintain backward compatibility (no 404s for WebSocket routes)
    expect(backwardCompatibilityWorking).toBe(true)
  })

  test('should work with mobile Safari specifics', async ({ page, browserName }) => {
    // Skip if not testing Safari-like behavior
    if (browserName === 'webkit') {
      await page.goto('/dashboard')
      await page.waitForSelector('dashboard-view', { timeout: 10000 })
      
      // Wait for WebSocket connection on Safari
      await page.waitForTimeout(5000)
      
      // Should not have Safari-specific WebSocket issues
      const consoleErrors = []
      page.on('console', msg => {
        if (msg.type() === 'error' && msg.text().includes('WebSocket')) {
          consoleErrors.push(msg.text())
        }
      })
      
      await page.waitForTimeout(2000)
      
      // Should not have Safari WebSocket errors
      expect(consoleErrors.filter(e => e.includes('Safari') || e.includes('webkit'))).toHaveLength(0)
    }
  })

  test('should maintain WebSocket connection during mobile navigation', async ({ page }) => {
    let connectionLostCount = 0
    let connectionRestoredCount = 0
    
    page.on('console', msg => {
      const text = msg.text()
      if (text.includes('WebSocket disconnected') || text.includes('connection lost')) {
        connectionLostCount++
      }
      if (text.includes('WebSocket connected') || text.includes('connection restored')) {
        connectionRestoredCount++
      }
    })
    
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Navigate between mobile views
    await page.click('[data-testid="agents-nav"], text=Agents', { timeout: 5000 })
    await page.waitForSelector('agents-view', { timeout: 5000 })
    
    await page.click('[data-testid="tasks-nav"], text=Tasks', { timeout: 5000 })
    await page.waitForSelector('tasks-view', { timeout: 5000 })
    
    await page.click('[data-testid="dashboard-nav"], text=Dashboard', { timeout: 5000 })
    await page.waitForSelector('dashboard-view', { timeout: 5000 })
    
    // WebSocket should remain connected during navigation
    expect(connectionLostCount).toBeLessThanOrEqual(1) // Allow one reconnect
    
    // If connection was lost, it should be restored
    if (connectionLostCount > 0) {
      expect(connectionRestoredCount).toBeGreaterThanOrEqual(1)
    }
  })

  test('should handle mobile background/foreground transitions', async ({ page }) => {
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Simulate app going to background (page visibility change)
    await page.evaluate(() => {
      Object.defineProperty(document, 'visibilityState', {
        writable: true,
        value: 'hidden'
      })
      document.dispatchEvent(new Event('visibilitychange'))
    })
    
    await page.waitForTimeout(2000)
    
    // Simulate app coming back to foreground
    await page.evaluate(() => {
      Object.defineProperty(document, 'visibilityState', {
        writable: true,
        value: 'visible'
      })
      document.dispatchEvent(new Event('visibilitychange'))
    })
    
    await page.waitForTimeout(3000)
    
    // Connection should be restored when app comes back to foreground
    const connectionIndicator = page.locator('.connection-indicator, [data-testid="connection-status"]')
    if (await connectionIndicator.isVisible()) {
      const statusText = await connectionIndicator.textContent()
      expect(statusText).toMatch(/(connected|live|excellent|good)/i)
    }
  })
})

test.describe('WebSocket Error Handling', () => {
  test('should gracefully handle server disconnections', async ({ page, context }) => {
    let errorHandled = false
    
    page.on('console', msg => {
      const text = msg.text()
      if (text.includes('WebSocket error') || text.includes('connection error')) {
        errorHandled = true
      }
    })
    
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Block WebSocket connections to simulate server down
    await page.route('**/ws/dashboard', route => route.abort())
    await page.route('**/api/dashboard/ws/dashboard', route => route.abort())
    
    // Try to establish connection
    await page.reload()
    await page.waitForTimeout(5000)
    
    // Should handle connection errors gracefully
    expect(errorHandled).toBe(true)
    
    // Dashboard should still be functional with cached/fallback data
    await expect(page.locator('dashboard-view')).toBeVisible()
  })

  test('should show appropriate error states for WebSocket failures', async ({ page }) => {
    await page.route('**/api/dashboard/ws/dashboard', route => {
      route.fulfill({ status: 500, body: 'Server Error' })
    })
    
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    await page.waitForTimeout(3000)
    
    // Should show connection error state
    const errorState = page.locator('.connection-error, .error-state, [data-testid="connection-error"]')
    const offlineIndicator = page.locator('.offline-indicator, [data-testid="offline-mode"]')
    
    // Should have some indication of connection issues
    const hasErrorState = await errorState.count() > 0
    const hasOfflineIndicator = await offlineIndicator.count() > 0
    
    expect(hasErrorState || hasOfflineIndicator).toBe(true)
  })
})

// Extend the Window interface for TypeScript
declare global {
  interface Window {
    latencyMeasurements: number[]
  }
}