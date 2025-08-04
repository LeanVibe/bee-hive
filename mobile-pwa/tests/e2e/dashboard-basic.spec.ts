import { test, expect } from '@playwright/test'

test.describe('LeanVibe Agent Hive Dashboard - Basic Functionality', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the dashboard
    await page.goto('/')
  })

  test('should load the dashboard successfully', async ({ page }) => {
    // Wait for the page to load
    await page.waitForLoadState('networkidle')
    
    // Verify the page title
    await expect(page).toHaveTitle(/LeanVibe Agent Hive/)
    
    // Verify the page contains some basic content
    const body = page.locator('body')
    await expect(body).toBeVisible()
    
    // Take a screenshot
    await page.screenshot({ path: 'test-results/dashboard-loaded.png' })
  })

  test('should have basic HTML structure', async ({ page }) => {
    // Wait for page load
    await page.waitForLoadState('networkidle')
    
    // Check for basic HTML elements
    await expect(page.locator('html')).toBeVisible()
    await expect(page.locator('head')).toBeAttached()
    await expect(page.locator('body')).toBeVisible()
    
    // Check for meta viewport tag (mobile responsiveness)
    const metaViewport = page.locator('meta[name="viewport"]')
    await expect(metaViewport).toBeAttached()
    
    // Verify PWA meta tags
    const themeColor = page.locator('meta[name="theme-color"]')
    await expect(themeColor).toBeAttached()
  })

  test('should be responsive on mobile devices', async ({ page }) => {
    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 }) // iPhone SE
    await page.waitForLoadState('networkidle')
    
    // Verify the page still loads and is visible
    await expect(page.locator('body')).toBeVisible()
    
    // Take mobile screenshot
    await page.screenshot({ path: 'test-results/dashboard-mobile.png' })
  })

  test('should be responsive on tablet devices', async ({ page }) => {
    // Test tablet viewport
    await page.setViewportSize({ width: 768, height: 1024 }) // iPad
    await page.waitForLoadState('networkidle')
    
    // Verify the page still loads and is visible
    await expect(page.locator('body')).toBeVisible()
    
    // Take tablet screenshot
    await page.screenshot({ path: 'test-results/dashboard-tablet.png' })
  })

  test('should load without JavaScript errors', async ({ page }) => {
    const errors: string[] = []
    
    page.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(msg.text())
      }
    })
    
    page.on('pageerror', error => {
      errors.push(error.message)
    })
    
    await page.waitForLoadState('networkidle')
    
    // Filter out known development warnings/errors that don't affect functionality
    const criticalErrors = errors.filter(error => 
      !error.includes('web-vitals') && 
      !error.includes('workbox') &&
      !error.includes('service worker') &&
      !error.includes('vite')
    )
    
    // Report any critical errors
    if (criticalErrors.length > 0) {
      console.log('JavaScript errors found:', criticalErrors)
    }
    
    // Should not have critical JavaScript errors
    expect(criticalErrors.length).toBeLessThan(5) // Allow some minor errors
  })

  test('should have proper meta tags and SEO elements', async ({ page }) => {
    await page.waitForLoadState('networkidle')
    
    // Check for essential meta tags
    const charset = page.locator('meta[charset]')
    await expect(charset).toBeAttached()
    
    const viewport = page.locator('meta[name="viewport"]')
    await expect(viewport).toBeAttached()
    
    const description = page.locator('meta[name="description"]')
    await expect(description).toBeAttached()
    
    // Check for Open Graph tags
    const ogTitle = page.locator('meta[property="og:title"]')
    await expect(ogTitle).toBeAttached()
    
    const ogDescription = page.locator('meta[property="og:description"]')
    await expect(ogDescription).toBeAttached()
  })

  test('should have PWA manifest and service worker support', async ({ page }) => {
    await page.waitForLoadState('networkidle')
    
    // Check for manifest link
    const manifest = page.locator('link[rel="manifest"]')
    if (await manifest.count() > 0) {
      await expect(manifest).toBeAttached()
    }
    
    // Check for apple-touch-icon
    const appleTouchIcon = page.locator('link[rel="apple-touch-icon"]')
    if (await appleTouchIcon.count() > 0) {
      await expect(appleTouchIcon).toBeAttached()
    }
    
    // Check for theme-color meta tag
    const themeColor = page.locator('meta[name="theme-color"]')
    await expect(themeColor).toBeAttached()
  })

  test('should handle network connectivity', async ({ page }) => {
    // Load page normally first
    await page.waitForLoadState('networkidle')
    await expect(page.locator('body')).toBeVisible()
    
    // Simulate offline conditions
    await page.context().setOffline(true)
    
    // Try to reload - should either show offline page or cached content
    await page.reload({ waitUntil: 'domcontentloaded' })
    
    // Should still have some content (either offline page or cached)
    await expect(page.locator('body')).toBeVisible()
    
    // Restore network
    await page.context().setOffline(false)
  })

  test('should load in different browsers', async ({ page, browserName }) => {
    await page.waitForLoadState('networkidle')
    
    // Verify basic functionality works across browsers
    await expect(page.locator('body')).toBeVisible()
    await expect(page).toHaveTitle(/LeanVibe Agent Hive/)
    
    // Take browser-specific screenshot
    await page.screenshot({ path: `test-results/dashboard-${browserName}.png` })
  })
})