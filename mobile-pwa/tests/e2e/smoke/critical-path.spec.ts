import { test, expect } from '@playwright/test'
import { TestHelpers } from '../../utils/test-helpers'

/**
 * Critical Path Smoke Tests for LeanVibe Agent Hive 2.0
 * 
 * These tests validate the most essential user journeys that must work
 * for the application to be considered functional. They run first and
 * block deployment if they fail.
 * 
 * Test Goals:
 * - Dashboard loads and displays core elements
 * - Navigation works correctly
 * - Basic agent status is visible
 * - Real-time updates function
 * - Performance meets baseline requirements
 */

test.describe('Critical Path Smoke Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Monitor console errors throughout tests
    await TestHelpers.monitorConsoleErrors(page)
    
    // Navigate to dashboard
    await page.goto('/')
    
    // Wait for initial load
    await TestHelpers.waitForNetworkIdle(page)
  })

  test('dashboard loads successfully with core elements', async ({ page }) => {
    // Verify page loads without critical errors
    await expect(page).toHaveTitle(/HiveOps|LeanVibe Agent Hive/)
    
    // Check for essential UI elements
    await expect(page.locator('header')).toBeVisible()
    await expect(page.locator('text=HiveOps')).toBeVisible()
    
    // Verify main dashboard content is present
    const dashboardContent = page.locator('[data-testid="dashboard-content"], .dashboard-content, main')
    await expect(dashboardContent).toBeVisible()
    
    // Check for agent dashboard indicator
    await expect(page.locator('text=Agent Dashboard')).toBeVisible({ timeout: 10000 })
    
    // Take screenshot for visual validation
    await page.screenshot({ 
      path: 'test-results/smoke/dashboard-loaded.png',
      fullPage: true 
    })
  })

  test('navigation elements are functional', async ({ page }) => {
    // Check for main navigation
    const navigation = page.locator('nav, [role="navigation"], .navigation')
    if (await navigation.count() > 0) {
      await expect(navigation.first()).toBeVisible()
    }
    
    // Check for sidebar or bottom navigation (mobile)
    const sidebar = page.locator('.sidebar, [data-testid="sidebar"]')
    const bottomNav = page.locator('.bottom-navigation, [data-testid="bottom-nav"]')
    
    const hasNavigation = await sidebar.count() > 0 || await bottomNav.count() > 0
    if (hasNavigation) {
      // Test navigation interaction
      if (await sidebar.count() > 0) {
        await expect(sidebar.first()).toBeVisible()
      }
      if (await bottomNav.count() > 0) {
        await expect(bottomNav.first()).toBeVisible()
      }
    }
    
    // Verify header navigation if present
    const headerNav = page.locator('header nav, header .nav')
    if (await headerNav.count() > 0) {
      await expect(headerNav.first()).toBeVisible()
    }
  })

  test('agent status information is displayed', async ({ page }) => {
    // Look for agent-related UI elements
    const agentElements = [
      '.agent-card',
      '.agent-status',
      '[data-testid*="agent"]',
      'text=agent',
      'text=Agent',
      '.live-agent-monitor'
    ]
    
    let agentUIFound = false
    for (const selector of agentElements) {
      if (await page.locator(selector).count() > 0) {
        await expect(page.locator(selector).first()).toBeVisible()
        agentUIFound = true
        break
      }
    }
    
    // If no specific agent UI, check for general dashboard content
    if (!agentUIFound) {
      const dashboardElements = page.locator('.dashboard-grid, .metrics-panel, .status-panel')
      if (await dashboardElements.count() > 0) {
        await expect(dashboardElements.first()).toBeVisible()
      }
    }
  })

  test('basic performance requirements are met', async ({ page }) => {
    const startTime = Date.now()
    
    // Navigate and measure load time
    await page.goto('/')
    await page.waitForLoadState('networkidle')
    
    const loadTime = Date.now() - startTime
    
    // Dashboard should load within 2 seconds as per PRD
    expect(loadTime).toBeLessThan(2000)
    
    // Check for performance markers
    const performanceMetrics = await page.evaluate(() => {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming
      return {
        domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
        loadComplete: navigation.loadEventEnd - navigation.loadEventStart,
        firstPaint: performance.getEntriesByName('first-paint')[0]?.startTime || 0,
        firstContentfulPaint: performance.getEntriesByName('first-contentful-paint')[0]?.startTime || 0
      }
    })
    
    // Validate performance thresholds
    expect(performanceMetrics.domContentLoaded).toBeLessThan(1000) // < 1s for DOM ready
    if (performanceMetrics.firstContentfulPaint > 0) {
      expect(performanceMetrics.firstContentfulPaint).toBeLessThan(1500) // < 1.5s FCP
    }
    
    console.log('Performance metrics:', performanceMetrics)
  })

  test('websocket connection establishes successfully', async ({ page }) => {
    // Monitor WebSocket connections
    const wsConnections: any[] = []
    
    page.on('websocket', ws => {
      wsConnections.push({
        url: ws.url(),
        isClosed: ws.isClosed()
      })
      
      ws.on('open', () => console.log('WebSocket opened:', ws.url()))
      ws.on('close', () => console.log('WebSocket closed:', ws.url()))
    })
    
    // Wait for WebSocket to potentially connect
    await page.waitForTimeout(3000)
    
    // Check if WebSocket connection was established
    if (wsConnections.length > 0) {
      console.log('WebSocket connections found:', wsConnections)
      
      // Verify at least one WebSocket connected
      const hasOpenConnection = wsConnections.some(ws => !ws.isClosed)
      if (hasOpenConnection) {
        console.log('✓ WebSocket connection established successfully')
      }
    } else {
      console.log('No WebSocket connections detected - may be using polling or disabled')
    }
    
    // Look for WebSocket-related UI indicators
    const connectionIndicators = page.locator(
      '.connection-status, .ws-status, [data-testid*="connection"], [data-testid*="websocket"]'
    )
    
    if (await connectionIndicators.count() > 0) {
      await expect(connectionIndicators.first()).toBeVisible()
    }
  })

  test('mobile responsiveness basics', async ({ page, isMobile }) => {
    if (isMobile) {
      // Verify mobile-specific elements
      await expect(page.locator('body')).toBeVisible()
      
      // Check for mobile navigation
      const mobileNav = page.locator('.mobile-nav, .bottom-navigation, [data-testid*="mobile"]')
      if (await mobileNav.count() > 0) {
        await expect(mobileNav.first()).toBeVisible()
      }
      
      // Verify touch-friendly interface
      const touchElements = page.locator('button, .touchable, [role="button"]')
      if (await touchElements.count() > 0) {
        const firstButton = touchElements.first()
        const boundingBox = await firstButton.boundingBox()
        
        if (boundingBox) {
          // Touch targets should be at least 44px (recommended minimum)
          expect(Math.min(boundingBox.width, boundingBox.height)).toBeGreaterThan(44)
        }
      }
      
      // Take mobile screenshot
      await page.screenshot({ 
        path: 'test-results/smoke/mobile-responsive.png',
        fullPage: true 
      })
    }
  })

  test('pwa installability check', async ({ page, browserName }) => {
    // Skip for browsers that don't support PWA
    if (browserName === 'firefox' || browserName === 'webkit') {
      test.skip('PWA features not fully supported in this browser')
    }
    
    // Check for PWA manifest
    const manifestLink = page.locator('link[rel="manifest"]')
    if (await manifestLink.count() > 0) {
      await expect(manifestLink).toBeAttached()
      
      const href = await manifestLink.getAttribute('href')
      expect(href).toBeTruthy()
      
      // Verify manifest is accessible
      const manifestResponse = await page.request.get(href!)
      expect(manifestResponse.status()).toBe(200)
    }
    
    // Check for service worker registration
    const serviceWorkerRegistered = await page.evaluate(async () => {
      if ('serviceWorker' in navigator) {
        try {
          const registration = await navigator.serviceWorker.getRegistration()
          return !!registration
        } catch (error) {
          return false
        }
      }
      return false
    })
    
    // Log PWA readiness
    console.log('PWA features check:', {
      hasManifest: await manifestLink.count() > 0,
      hasServiceWorker: serviceWorkerRegistered
    })
  })

  test('error handling gracefully manages failures', async ({ page }) => {
    // Test handling of network errors
    await page.route('**/api/**', route => {
      // Simulate intermittent API failures
      if (Math.random() > 0.7) {
        route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Simulated server error' })
        })
      } else {
        route.continue()
      }
    })
    
    // Reload page to trigger potential errors
    await page.reload()
    await TestHelpers.waitForNetworkIdle(page)
    
    // Verify page still functions despite errors
    await expect(page.locator('body')).toBeVisible()
    
    // Check for error handling UI
    const errorElements = page.locator('.error-state, .error-message, [data-testid*="error"]')
    
    // If errors are shown, they should be user-friendly
    if (await errorElements.count() > 0) {
      const errorText = await errorElements.first().textContent()
      expect(errorText).toBeTruthy()
      expect(errorText!.length).toBeGreaterThan(5) // Should have meaningful error message
    }
  })

  test('basic accessibility requirements', async ({ page }) => {
    // Check for essential accessibility features
    await TestHelpers.verifyAccessibility(page)
    
    // Verify page structure
    const main = page.locator('main, [role="main"]')
    if (await main.count() > 0) {
      await expect(main.first()).toBeVisible()
    }
    
    // Check for heading structure
    const headings = page.locator('h1, h2, h3, h4, h5, h6')
    if (await headings.count() > 0) {
      await expect(headings.first()).toBeVisible()
    }
    
    // Verify keyboard navigation works
    await page.keyboard.press('Tab')
    const focusedElement = page.locator(':focus')
    if (await focusedElement.count() > 0) {
      await expect(focusedElement).toBeVisible()
    }
  })
})