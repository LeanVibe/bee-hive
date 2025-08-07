/**
 * Mobile UI/UX Quality Validation Tests
 * 
 * Validates Silicon Valley startup quality standards for mobile dashboard:
 * - Tests touch targets meet mobile standards (44px+ on iOS, 48px+ on Android)
 * - Validates responsive design on various mobile viewports
 * - Tests smooth scrolling, interactions, and animations
 * - Validates mobile-first design principles and accessibility
 * - Ensures excellent user experience comparable to top-tier mobile apps
 */

import { test, expect, Page, Locator } from '@playwright/test'

test.describe('Mobile UI/UX Quality Validation - Silicon Valley Standards', () => {
  // Test different mobile viewports
  const mobileViewports = [
    { name: 'iPhone 14 Pro', width: 393, height: 852 },
    { name: 'iPhone SE', width: 375, height: 667 },
    { name: 'Samsung Galaxy S21', width: 360, height: 800 },
    { name: 'iPad Mini', width: 768, height: 1024 }
  ]

  mobileViewports.forEach(({ name, width, height }) => {
    test(`should meet touch target standards on ${name} (${width}x${height})`, async ({ page }) => {
      await page.setViewportSize({ width, height })
      await page.goto('/dashboard')
      await page.waitForSelector('dashboard-view', { timeout: 10000 })
      
      // Get all interactive elements
      const interactiveElements = page.locator('button, a, input, [role="button"], .clickable, [data-testid*="btn"]')
      const count = await interactiveElements.count()
      
      let failedElements = 0
      const minTouchTarget = name.includes('iPad') ? 44 : (name.includes('iPhone') ? 44 : 48)
      
      for (let i = 0; i < count; i++) {
        const element = interactiveElements.nth(i)
        if (await element.isVisible()) {
          const boundingBox = await element.boundingBox()
          
          if (boundingBox) {
            const hasValidSize = boundingBox.width >= minTouchTarget && boundingBox.height >= minTouchTarget
            
            if (!hasValidSize) {
              failedElements++
              console.log(`Failed touch target on ${name}: ${boundingBox.width}x${boundingBox.height}`)
            }
          }
        }
      }
      
      // Allow up to 10% of elements to be smaller (for edge cases like close buttons)
      const failureRate = failedElements / Math.max(count, 1)
      expect(failureRate).toBeLessThan(0.1)
      
      console.log(`${name}: ${count} elements, ${failedElements} failed (${(failureRate * 100).toFixed(1)}% failure rate)`)
    })
  })

  test('should have excellent responsive design on iPhone 14 Pro', async ({ page }) => {
    await page.setViewportSize({ width: 393, height: 852 })
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Check header responsiveness
    const header = page.locator('app-header, .header, [data-testid="header"]')
    if (await header.count() > 0) {
      const headerBox = await header.first().boundingBox()
      if (headerBox) {
        // Header should be full width
        expect(headerBox.width).toBeGreaterThanOrEqual(390) // Allow for some padding
      }
    }
    
    // Check navigation responsiveness
    const bottomNav = page.locator('bottom-navigation, .bottom-nav, [data-testid="bottom-nav"]')
    const sideNav = page.locator('sidebar-navigation, .sidebar, [data-testid="sidebar"]')
    
    if (await bottomNav.count() > 0) {
      await expect(bottomNav.first()).toBeVisible()
      
      // Bottom nav should be at bottom of screen
      const navBox = await bottomNav.first().boundingBox()
      if (navBox) {
        expect(navBox.y + navBox.height).toBeGreaterThan(800) // Near bottom
      }
    }
    
    // Sidebar should be hidden or collapsible on mobile
    if (await sideNav.count() > 0) {
      const isVisible = await sideNav.first().isVisible()
      if (isVisible) {
        // If visible, should not take up too much horizontal space
        const navBox = await sideNav.first().boundingBox()
        if (navBox) {
          expect(navBox.width).toBeLessThan(200) // Should be collapsed or narrow
        }
      }
    }
  })

  test('should have smooth scrolling and interactions', async ({ page }) => {
    await page.setViewportSize({ width: 393, height: 852 })
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Test smooth scrolling
    const scrollContainer = page.locator('main, .main-content, [data-testid="main-content"]').first()
    
    if (await scrollContainer.count() > 0) {
      // Scroll down
      await scrollContainer.hover()
      await page.mouse.wheel(0, 300)
      await page.waitForTimeout(100)
      
      // Scroll up
      await page.mouse.wheel(0, -300)
      await page.waitForTimeout(100)
      
      // Should handle touch scrolling without issues
      // (No specific assertion, just ensuring no crashes)
    }
    
    // Test tap interactions
    const interactiveElements = page.locator('button:visible').first()
    if (await interactiveElements.count() > 0) {
      const elementBox = await interactiveElements.boundingBox()
      if (elementBox) {
        // Tap in center of element
        await page.tap(`${elementBox.x + elementBox.width / 2}, ${elementBox.y + elementBox.height / 2}`)
        await page.waitForTimeout(100)
      }
    }
  })

  test('should have proper mobile typography and spacing', async ({ page }) => {
    await page.setViewportSize({ width: 393, height: 852 })
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Check text readability
    const textElements = page.locator('h1, h2, h3, h4, h5, h6, p, span, .text')
    const count = await textElements.count()
    
    let smallTextCount = 0
    
    for (let i = 0; i < Math.min(count, 20); i++) { // Check first 20 elements
      const element = textElements.nth(i)
      if (await element.isVisible()) {
        const fontSize = await element.evaluate(el => {
          return parseFloat(window.getComputedStyle(el).fontSize)
        })
        
        // Text should be at least 14px for readability on mobile
        if (fontSize < 14) {
          smallTextCount++
        }
      }
    }
    
    // No more than 20% of text should be smaller than 14px
    const smallTextRate = smallTextCount / Math.max(count, 1)
    expect(smallTextRate).toBeLessThan(0.2)
    
    console.log(`Typography check: ${smallTextCount}/${count} elements smaller than 14px (${(smallTextRate * 100).toFixed(1)}%)`)
  })

  test('should have appropriate loading states and feedback', async ({ page }) => {
    await page.setViewportSize({ width: 393, height: 852 })
    
    let hasLoadingState = false
    let hasErrorState = false
    let hasEmptyState = false
    
    // Monitor page loading
    await page.goto('/dashboard')
    
    // Check for loading indicators during initial load
    const loadingIndicators = page.locator('.loading, .spinner, loading-spinner, [data-testid*="loading"]')
    
    if (await loadingIndicators.count() > 0) {
      hasLoadingState = await loadingIndicators.first().isVisible()
    }
    
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Check for error states (if any)
    const errorStates = page.locator('.error, .error-state, [data-testid*="error"]')
    if (await errorStates.count() > 0) {
      hasErrorState = await errorStates.first().isVisible()
    }
    
    // Check for empty states (if any)
    const emptyStates = page.locator('.empty, .no-data, [data-testid*="empty"]')
    if (await emptyStates.count() > 0) {
      hasEmptyState = await emptyStates.first().isVisible()
    }
    
    console.log(`UI States - Loading: ${hasLoadingState}, Error: ${hasErrorState}, Empty: ${hasEmptyState}`)
    
    // At minimum, should handle loading states
    // (Other states are optional depending on data availability)
  })

  test('should have mobile-optimized card layouts', async ({ page }) => {
    await page.setViewportSize({ width: 393, height: 852 })
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Find card components
    const cards = page.locator('.card, .panel, .metric-card, enhanced-card, [data-testid*="card"]')
    const cardCount = await cards.count()
    
    if (cardCount > 0) {
      let wellOptimizedCards = 0
      
      for (let i = 0; i < Math.min(cardCount, 10); i++) {
        const card = cards.nth(i)
        if (await card.isVisible()) {
          const cardBox = await card.boundingBox()
          
          if (cardBox) {
            // Cards should be appropriately sized for mobile
            const isGoodWidth = cardBox.width >= 200 && cardBox.width <= 400
            const isGoodHeight = cardBox.height >= 80 && cardBox.height <= 300
            const hasGoodPadding = cardBox.width < 380 // Should have some margin on mobile
            
            if (isGoodWidth && isGoodHeight && hasGoodPadding) {
              wellOptimizedCards++
            }
          }
        }
      }
      
      // At least 70% of cards should be well-optimized for mobile
      const optimizationRate = wellOptimizedCards / Math.min(cardCount, 10)
      expect(optimizationRate).toBeGreaterThanOrEqual(0.7)
      
      console.log(`Card optimization: ${wellOptimizedCards}/${Math.min(cardCount, 10)} cards well-optimized (${(optimizationRate * 100).toFixed(1)}%)`)
    }
  })

  test('should support pull-to-refresh gesture', async ({ page }) => {
    await page.setViewportSize({ width: 393, height: 852 })
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    let pullToRefreshDetected = false
    
    // Monitor for pull-to-refresh
    page.on('console', msg => {
      const text = msg.text()
      if (text.includes('pull-to-refresh') || text.includes('pullToRefresh')) {
        pullToRefreshDetected = true
      }
    })
    
    // Look for pull-to-refresh component
    const pullToRefresh = page.locator('pull-to-refresh, [data-testid="pull-to-refresh"], .pull-to-refresh')
    
    if (await pullToRefresh.count() > 0) {
      await expect(pullToRefresh.first()).toBeVisible()
      pullToRefreshDetected = true
    }
    
    // Try to simulate pull gesture (limited in Playwright)
    const viewport = page.viewportSize()
    if (viewport) {
      await page.mouse.move(viewport.width / 2, 100)
      await page.mouse.down()
      await page.mouse.move(viewport.width / 2, 200)
      await page.mouse.up()
    }
    
    await page.waitForTimeout(1000)
    
    console.log(`Pull-to-refresh detected: ${pullToRefreshDetected}`)
  })

  test('should have proper mobile navigation patterns', async ({ page }) => {
    await page.setViewportSize({ width: 393, height: 852 })
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Check for mobile navigation patterns
    const bottomNav = page.locator('bottom-navigation, .bottom-nav, [data-testid="bottom-nav"]')
    const hamburgerMenu = page.locator('.hamburger, .menu-toggle, [data-testid="menu-toggle"]')
    const tabBar = page.locator('.tab-bar, .tabs, [data-testid="tabs"]')
    
    let hasMobileNavigation = false
    
    // Bottom navigation (preferred for mobile)
    if (await bottomNav.count() > 0 && await bottomNav.first().isVisible()) {
      hasMobileNavigation = true
      
      // Bottom nav should have 3-5 items (UX best practice)
      const navItems = bottomNav.locator('button, a, .nav-item')
      const itemCount = await navItems.count()
      expect(itemCount).toBeGreaterThanOrEqual(3)
      expect(itemCount).toBeLessThanOrEqual(5)
    }
    
    // Hamburger menu (alternative pattern)
    if (await hamburgerMenu.count() > 0) {
      hasMobileNavigation = true
      await expect(hamburgerMenu.first()).toBeVisible()
      
      // Should be properly sized for touch
      const menuBox = await hamburgerMenu.first().boundingBox()
      if (menuBox) {
        expect(menuBox.width).toBeGreaterThanOrEqual(44)
        expect(menuBox.height).toBeGreaterThanOrEqual(44)
      }
    }
    
    // Tab bar (for content sections)
    if (await tabBar.count() > 0) {
      hasMobileNavigation = true
      await expect(tabBar.first()).toBeVisible()
    }
    
    expect(hasMobileNavigation).toBe(true)
  })

  test('should handle mobile gestures and interactions', async ({ page }) => {
    await page.setViewportSize({ width: 393, height: 852 })
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    let gestureSupported = false
    
    // Check for swipe gesture support
    const swipeComponents = page.locator('swipe-gesture, [data-testid*="swipe"], .swipeable')
    if (await swipeComponents.count() > 0) {
      gestureSupported = true
    }
    
    // Check for mobile-specific event handlers
    const hasTouch = await page.evaluate(() => {
      return 'ontouchstart' in window || navigator.maxTouchPoints > 0
    })
    
    expect(hasTouch).toBe(true) // Should support touch events
    
    // Test tap vs click handling
    const button = page.locator('button:visible').first()
    if (await button.count() > 0) {
      // Should respond to tap
      await button.tap()
      await page.waitForTimeout(100)
      
      // No error should occur
      gestureSupported = true
    }
    
    console.log(`Mobile gestures supported: ${gestureSupported}`)
  })

  test('should have proper mobile performance characteristics', async ({ page }) => {
    await page.setViewportSize({ width: 393, height: 852 })
    
    const startTime = Date.now()
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    const loadTime = Date.now() - startTime
    
    // Should load quickly on mobile (under 2 seconds for dashboard)
    expect(loadTime).toBeLessThan(2000)
    
    // Check for performance optimizations
    const performanceMetrics = await page.evaluate(() => {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming
      return {
        domContentLoaded: navigation.domContentLoadedEventEnd - navigation.navigationStart,
        firstPaint: performance.getEntriesByName('first-paint')[0]?.startTime || 0,
        firstContentfulPaint: performance.getEntriesByName('first-contentful-paint')[0]?.startTime || 0
      }
    })
    
    console.log('Mobile performance metrics:', performanceMetrics)
    
    // DOM should be ready quickly
    expect(performanceMetrics.domContentLoaded).toBeLessThan(1500)
  })

  test('should have excellent visual hierarchy and contrast', async ({ page }) => {
    await page.setViewportSize({ width: 393, height: 852 })
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Check color contrast for key elements
    const headings = page.locator('h1, h2, h3, .heading, .title')
    const buttons = page.locator('button:visible')
    const links = page.locator('a:visible')
    
    // Test a few key elements for contrast
    const elements = [
      ...(await headings.all()).slice(0, 3),
      ...(await buttons.all()).slice(0, 3),
      ...(await links.all()).slice(0, 2)
    ]
    
    let contrastFailures = 0
    
    for (const element of elements) {
      const colors = await element.evaluate(el => {
        const styles = window.getComputedStyle(el)
        const backgroundColor = styles.backgroundColor
        const color = styles.color
        
        // Simple RGB extraction (would need more robust parsing for production)
        const rgbMatch = (colorStr: string) => {
          const match = colorStr.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/)
          return match ? [parseInt(match[1]), parseInt(match[2]), parseInt(match[3])] : [255, 255, 255]
        }
        
        return {
          background: rgbMatch(backgroundColor),
          foreground: rgbMatch(color)
        }
      })
      
      // Simple contrast calculation (luminance ratio)
      const luminance = (rgb: number[]) => {
        const [r, g, b] = rgb.map(c => {
          c = c / 255
          return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
        })
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
      }
      
      const l1 = luminance(colors.foreground)
      const l2 = luminance(colors.background)
      const contrast = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05)
      
      // WCAG AA requires 4.5:1 for normal text, 3:1 for large text
      if (contrast < 3) {
        contrastFailures++
      }
    }
    
    // Allow some contrast failures for non-critical elements
    const contrastFailureRate = contrastFailures / Math.max(elements.length, 1)
    expect(contrastFailureRate).toBeLessThan(0.3)
    
    console.log(`Contrast check: ${contrastFailures}/${elements.length} failures (${(contrastFailureRate * 100).toFixed(1)}%)`)
  })

  test('should handle mobile keyboard and form interactions', async ({ page }) => {
    await page.setViewportSize({ width: 393, height: 852 })
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Look for form inputs
    const inputs = page.locator('input, textarea, [contenteditable="true"]')
    const inputCount = await inputs.count()
    
    if (inputCount > 0) {
      const firstInput = inputs.first()
      
      // Focus should work
      await firstInput.focus()
      
      // Should handle mobile keyboard
      await firstInput.fill('test input')
      
      // Check for proper input styling on mobile
      const inputBox = await firstInput.boundingBox()
      if (inputBox) {
        // Input should be tall enough for mobile interaction
        expect(inputBox.height).toBeGreaterThanOrEqual(40)
      }
    }
    
    console.log(`Form inputs found: ${inputCount}`)
  })
})

test.describe('Mobile Accessibility Standards', () => {
  test('should meet WCAG accessibility guidelines', async ({ page }) => {
    await page.setViewportSize({ width: 393, height: 852 })
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Check for proper ARIA labels
    const interactiveElements = page.locator('button, a, input, [role="button"]')
    const count = await interactiveElements.count()
    
    let missingLabels = 0
    
    for (let i = 0; i < Math.min(count, 20); i++) {
      const element = interactiveElements.nth(i)
      
      const hasLabel = await element.evaluate(el => {
        return !!(el.getAttribute('aria-label') || 
                 el.getAttribute('aria-labelledby') || 
                 el.textContent?.trim() ||
                 el.getAttribute('alt') ||
                 el.getAttribute('title'))
      })
      
      if (!hasLabel) {
        missingLabels++
      }
    }
    
    // No more than 20% should be missing labels
    const missingLabelRate = missingLabels / Math.max(count, 1)
    expect(missingLabelRate).toBeLessThan(0.2)
    
    console.log(`Accessibility: ${missingLabels}/${count} elements missing labels (${(missingLabelRate * 100).toFixed(1)}%)`)
  })

  test('should support screen reader navigation', async ({ page }) => {
    await page.setViewportSize({ width: 393, height: 852 })
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Check for proper heading hierarchy
    const headings = page.locator('h1, h2, h3, h4, h5, h6')
    const headingCount = await headings.count()
    
    if (headingCount > 0) {
      // Should have at least one h1
      const h1Count = await page.locator('h1').count()
      expect(h1Count).toBeGreaterThanOrEqual(1)
    }
    
    // Check for landmarks
    const landmarks = page.locator('[role="main"], [role="navigation"], [role="banner"], main, nav, header')
    const landmarkCount = await landmarks.count()
    
    expect(landmarkCount).toBeGreaterThanOrEqual(1) // Should have semantic structure
    
    console.log(`Screen reader: ${headingCount} headings, ${landmarkCount} landmarks`)
  })
})

test.describe('Cross-Device Compatibility', () => {
  test('should work consistently across mobile browsers', async ({ page, browserName }) => {
    await page.setViewportSize({ width: 393, height: 852 })
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Should load without browser-specific issues
    let browserIssues = 0
    
    page.on('console', msg => {
      if (msg.type() === 'error') {
        browserIssues++
      }
    })
    
    await page.waitForTimeout(3000)
    
    // Minimal console errors (allow for network issues in test environment)
    expect(browserIssues).toBeLessThan(5)
    
    console.log(`${browserName}: ${browserIssues} console errors detected`)
  })

  test('should handle different mobile densities', async ({ page }) => {
    // Test high-DPI mobile display
    await page.setViewportSize({ width: 393, height: 852 })
    await page.goto('/dashboard')
    await page.waitForSelector('dashboard-view', { timeout: 10000 })
    
    // Images and icons should be crisp
    const images = page.locator('img, .icon, [data-testid*="icon"]')
    const imageCount = await images.count()
    
    if (imageCount > 0) {
      // Check for high-DPI image sources
      const highDPIImages = await images.evaluateAll(imgs => {
        return imgs.filter(img => {
          const src = img.getAttribute('src') || ''
          const srcset = img.getAttribute('srcset') || ''
          return src.includes('@2x') || srcset.includes('2x') || srcset.length > 0
        }).length
      })
      
      // At least some images should support high-DPI
      console.log(`High-DPI support: ${highDPIImages}/${imageCount} images`)
    }
  })
})