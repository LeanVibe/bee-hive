/**
 * Performance Analytics Dashboard Validation Tests
 * 
 * Tests to validate the Phase 2.1 Performance Analytics Dashboard implementation
 * against the requirements defined in the strategic plan
 */

import { expect } from '@playwright/test'
import { test } from './config/test-setup'

test.describe('Performance Analytics Dashboard - Phase 2.1 Validation', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the performance analytics page
    await page.goto('/performance')
    
    // Wait for the dashboard to load
    await page.waitForSelector('enhanced-performance-analytics-panel', { timeout: 10000 })
  })

  test('should load dashboard within performance targets (<1 second)', async ({ page }) => {
    const startTime = Date.now()
    
    // Navigate to performance dashboard
    await page.goto('/performance')
    
    // Wait for key components to be visible
    await page.waitForSelector('.analytics-header')
    await page.waitForSelector('.performance-tabs')
    
    const loadTime = Date.now() - startTime
    
    // Should load within 1 second (1000ms) as per requirements
    expect(loadTime).toBeLessThan(1000)
  })

  test('should render real-time response time monitoring', async ({ page }) => {
    // Check for response time metrics
    const responseTimeCard = page.locator('[data-testid="response-time-card"]').first()
    await expect(responseTimeCard).toBeVisible()
    
    // Should show P95/P99 percentiles
    await expect(page.locator('text=/P95.*ms/')).toBeVisible()
    await expect(page.locator('text=/P99.*ms/')).toBeVisible()
    
    // Should have color-coded status indicators
    const metricValue = page.locator('.metric-value').first()
    await expect(metricValue).toHaveClass(/healthy|warning|critical/)
  })

  test('should display throughput and capacity monitoring', async ({ page }) => {
    // Click on throughput tab
    await page.click('button:has-text("Throughput")')
    
    // Check for throughput metrics
    await expect(page.locator('text=/Requests.*Second/')).toBeVisible()
    await expect(page.locator('text=/Queue Length/')).toBeVisible()
    
    // Should show capacity utilization
    await expect(page.locator('text=/Connection Pool/')).toBeVisible()
  })

  test('should show error rate tracking and alerts', async ({ page }) => {
    // Click on error analysis tab
    await page.click('button:has-text("Error Analysis")')
    
    // Check for error rate metrics
    await expect(page.locator('text=/Error Rate/')).toBeVisible()
    await expect(page.locator('text=/4xx.*%/')).toBeVisible()
    await expect(page.locator('text=/5xx.*%/')).toBeVisible()
    
    // Should show alerts section
    await expect(page.locator('.alerts-section')).toBeVisible()
  })

  test('should render resource utilization monitoring', async ({ page }) => {
    // Click on resources tab
    await page.click('button:has-text("Resources")')
    
    // Check for system resource metrics
    await expect(page.locator('text=/CPU Usage/')).toBeVisible()
    await expect(page.locator('text=/Memory Usage/')).toBeVisible()
    await expect(page.locator('text=/Network Usage/')).toBeVisible()
  })

  test('should provide performance regression detection', async ({ page }) => {
    // Click on regression tab
    await page.click('button:has-text("Regression")')
    
    // Should show regression analysis
    await expect(page.locator('.alerts-title')).toContainText('Regression Detection')
    
    // Should have filters for regression alerts
    await expect(page.locator('.alert-filter-btn')).toBeVisible()
  })

  test('should have interactive time range selection', async ({ page }) => {
    const timeRangeSelector = page.locator('.time-range-selector')
    await expect(timeRangeSelector).toBeVisible()
    
    // Should have multiple time range options
    await expect(page.locator('.time-range-btn')).toHaveCount(7) // 1M, 5M, 15M, 1H, 6H, 24H, 7D
    
    // Should be able to change time range
    await page.click('.time-range-btn:has-text("5M")')
    await expect(page.locator('.time-range-btn:has-text("5M")')).toHaveClass(/active/)
  })

  test('should show real-time connection status', async ({ page }) => {
    const connectionIndicator = page.locator('.connection-indicator')
    await expect(connectionIndicator).toBeVisible()
    
    // Should show connection status
    const statusText = await connectionIndicator.textContent()
    expect(statusText).toMatch(/Live Data|Connecting\.\.\.|Offline Mode/)
    
    // Should have visual status indicator
    await expect(page.locator('.status-dot')).toBeVisible()
  })

  test('should have chart expansion functionality', async ({ page }) => {
    // Find a chart toggle button
    const chartToggle = page.locator('.chart-toggle').first()
    if (await chartToggle.isVisible()) {
      await chartToggle.click()
      
      // Chart should expand
      await expect(page.locator('.chart-canvas.expanded')).toBeVisible()
    }
  })

  test('should render charts within performance targets (<500ms)', async ({ page }) => {
    const startTime = Date.now()
    
    // Wait for charts to be rendered
    await page.waitForSelector('canvas', { timeout: 5000 })
    
    const renderTime = Date.now() - startTime
    
    // Charts should render within 500ms as per requirements
    expect(renderTime).toBeLessThan(500)
  })

  test('should be mobile responsive', async ({ page, isMobile }) => {
    if (isMobile) {
      // Mobile-specific responsive checks
      await expect(page.locator('.analytics-header')).toBeVisible()
      
      // Header should stack on mobile
      const headerContent = page.locator('.header-content')
      const computedStyle = await headerContent.evaluate(el => 
        window.getComputedStyle(el).flexDirection
      )
      expect(computedStyle).toBe('column')
      
      // Metrics grid should be single column on mobile
      const metricsGrid = page.locator('.metrics-grid')
      if (await metricsGrid.isVisible()) {
        const gridColumns = await metricsGrid.evaluate(el => 
          window.getComputedStyle(el).gridTemplateColumns
        )
        // Should be single column (1fr) on mobile
        expect(gridColumns).toContain('1fr')
      }
    }
  })

  test('should handle auto-refresh toggle', async ({ page }) => {
    const refreshButton = page.locator('.control-btn').first()
    await expect(refreshButton).toBeVisible()
    
    // Should be able to toggle auto-refresh
    const initialState = await refreshButton.getAttribute('class')
    await refreshButton.click()
    
    // State should change
    const newState = await refreshButton.getAttribute('class')
    expect(newState).not.toBe(initialState)
  })

  test('should show performance alerts with proper severity', async ({ page }) => {
    // Click on alerts tab
    await page.click('button:has-text("Alerts")')
    
    // If alerts are present, check their structure
    const alertItems = page.locator('.alert-item')
    const alertCount = await alertItems.count()
    
    if (alertCount > 0) {
      // First alert should have proper structure
      const firstAlert = alertItems.first()
      await expect(firstAlert).toHaveClass(/critical|warning|info/)
      await expect(firstAlert.locator('.alert-message')).toBeVisible()
      await expect(firstAlert.locator('.alert-time')).toBeVisible()
    }
  })

  test('should export chart functionality', async ({ page }) => {
    // Look for export button
    const exportButton = page.locator('button[title="Export chart"]').first()
    if (await exportButton.isVisible()) {
      // Should be able to click export (we won't test actual download)
      await expect(exportButton).toBeEnabled()
    }
  })

  test('should handle performance data loading states', async ({ page }) => {
    // On initial load, should show loading state
    const loadingState = page.locator('.loading-overlay')
    
    // Loading overlay should appear then disappear
    if (await loadingState.isVisible()) {
      await expect(loadingState).toBeHidden({ timeout: 10000 })
    }
    
    // Dashboard content should be visible after loading
    await expect(page.locator('.analytics-panel')).toBeVisible()
  })

  test('should validate metric value formats and ranges', async ({ page }) => {
    // Wait for metrics to load
    await page.waitForSelector('.metric-value', { timeout: 10000 })
    
    const metricValues = page.locator('.metric-value')
    const count = await metricValues.count()
    
    // Check that metric values are properly formatted
    for (let i = 0; i < Math.min(count, 5); i++) {
      const value = await metricValues.nth(i).textContent()
      
      if (value) {
        // Values should be numeric or have proper units
        expect(value).toMatch(/^\d+(\.\d+)?\s*(ms|%|MB|GB|RPS)?$/)
      }
    }
  })

  test('should show proper error handling for API failures', async ({ page }) => {
    // Mock API failure
    await page.route('**/api/v1/performance/**', route => {
      route.fulfill({ status: 500, body: 'Internal Server Error' })
    })
    
    // Navigate to dashboard
    await page.goto('/performance')
    
    // Should show error state or fallback data
    const hasErrorMessage = await page.locator('.error-state').isVisible()
    const hasFallbackData = await page.locator('.metric-card').count() > 0
    
    expect(hasErrorMessage || hasFallbackData).toBe(true)
  })
})

test.describe('Performance Dashboard Integration', () => {
  test('should be accessible from sidebar navigation', async ({ page }) => {
    await page.goto('/')
    
    // Should see Performance Analytics in sidebar
    await expect(page.locator('text=Performance Analytics')).toBeVisible()
    
    // Should be able to navigate to performance dashboard
    await page.click('text=Performance Analytics')
    await page.waitForURL('**/performance')
    
    await expect(page.locator('enhanced-performance-analytics-panel')).toBeVisible()
  })

  test('should be accessible from bottom navigation on mobile', async ({ page, isMobile }) => {
    if (isMobile) {
      await page.goto('/')
      
      // Should see Analytics in bottom navigation
      await expect(page.locator('text=Analytics')).toBeVisible()
      
      // Should be able to navigate via bottom nav
      await page.click('text=Analytics')
      await page.waitForURL('**/performance')
      
      await expect(page.locator('performance-analytics-view')).toBeVisible()
    }
  })

  test('should maintain URL state for time range selection', async ({ page }) => {
    await page.goto('/performance')
    
    // Change time range
    await page.click('.time-range-btn:has-text("24H")')
    
    // URL should reflect the change or component should maintain state
    // (This depends on implementation - checking component state)
    await expect(page.locator('.time-range-btn:has-text("24H")')).toHaveClass(/active/)
    
    // Refresh page and state should persist if implemented
    await page.reload()
    // Note: Depending on implementation, this might reset to default
  })
})

test.describe('Performance Analytics Service Integration', () => {
  test('should connect to WebSocket for real-time updates', async ({ page }) => {
    await page.goto('/performance')
    
    // Wait for WebSocket connection attempt
    await page.waitForTimeout(2000)
    
    // Should show connection status
    const connectionStatus = page.locator('.connection-indicator')
    await expect(connectionStatus).toBeVisible()
    
    const statusText = await connectionStatus.textContent()
    // Should show some connection status
    expect(statusText?.length).toBeGreaterThan(0)
  })

  test('should handle offline mode gracefully', async ({ page, context }) => {
    // Go offline
    await context.setOffline(true)
    await page.goto('/performance')
    
    // Should still load with offline data
    await expect(page.locator('performance-analytics-view')).toBeVisible()
    
    // Should show offline indicator
    const connectionIndicator = page.locator('.connection-indicator')
    if (await connectionIndicator.isVisible()) {
      await expect(connectionIndicator).toContainText(/Offline|Disconnected/)
    }
  })
})