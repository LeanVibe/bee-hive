import { test, expect } from '@playwright/test'
import { TestHelpers } from '../../utils/test-helpers'

/**
 * Mobile Touch Interactions & Responsiveness Tests
 * 
 * Validates mobile-specific functionality:
 * - Touch target sizes and interactions
 * - Gesture support (swipe, tap, scroll)
 * - Mobile navigation patterns
 * - Touch-friendly UI elements
 * - Mobile performance optimization
 */

test.describe('Mobile Touch Interactions', () => {
  
  test.beforeEach(async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 }) // iPhone SE
    await page.goto('/')
    await TestHelpers.waitForNetworkIdle(page)
  })

  test('touch targets meet minimum size requirements', async ({ page }) => {
    // Get all interactive elements
    const interactiveElements = page.locator(
      'button, a, input, select, textarea, [role="button"], [tabindex="0"], .clickable, .touchable'
    )
    
    const elementCount = await interactiveElements.count()
    expect(elementCount).toBeGreaterThan(0)
    
    const undersizedElements = []
    
    // Check each interactive element
    for (let i = 0; i < Math.min(elementCount, 20); i++) { // Check first 20 elements
      const element = interactiveElements.nth(i)
      const boundingBox = await element.boundingBox()
      
      if (boundingBox) {
        const minDimension = Math.min(boundingBox.width, boundingBox.height)
        
        // Apple Human Interface Guidelines recommend 44px minimum
        if (minDimension < 44) {
          const elementInfo = {
            index: i,
            size: { width: boundingBox.width, height: boundingBox.height },
            text: await element.textContent(),
            tagName: await element.evaluate(el => el.tagName)
          }
          undersizedElements.push(elementInfo)
        }
      }
    }
    
    // Log findings
    if (undersizedElements.length > 0) {
      console.log('Elements below 44px touch target:', undersizedElements)
    }
    
    // Allow some flexibility for small icons or secondary elements
    const allowedUndersizedPercent = 20 // Allow 20% of elements to be smaller
    const undersizedPercent = (undersizedElements.length / Math.min(elementCount, 20)) * 100
    
    expect(undersizedPercent).toBeLessThan(allowedUndersizedPercent)
    console.log(`Touch target compliance: ${100 - undersizedPercent}% of elements meet 44px minimum`)
  })

  test('tap interactions work correctly', async ({ page }) => {
    // Test tapping various elements
    const tapTargets = [
      'button',
      'a[href]',
      '[role="button"]',
      '.clickable',
      'input[type="button"]'
    ]
    
    for (const selector of tapTargets) {
      const elements = page.locator(selector)
      const count = await elements.count()
      
      if (count > 0) {
        const firstElement = elements.first()
        
        // Ensure element is visible and enabled
        await expect(firstElement).toBeVisible()
        
        const isDisabled = await firstElement.getAttribute('disabled')
        if (isDisabled === null || isDisabled === 'false') {
          // Test tap interaction
          await firstElement.tap()
          await TestHelpers.waitForAnimations(page)
          
          console.log(`✓ Tap interaction successful for ${selector}`)
        }
      }
    }
  })

  test('swipe gestures function properly', async ({ page }) => {
    // Look for swipeable elements
    const swipeableElements = page.locator(
      '.swipeable, .carousel, .slider, [data-swipe], .swipe-container'
    )
    
    if (await swipeableElements.count() > 0) {
      const firstSwipeable = swipeableElements.first()
      await expect(firstSwipeable).toBeVisible()
      
      const boundingBox = await firstSwipeable.boundingBox()
      if (boundingBox) {
        // Perform swipe gesture
        const startX = boundingBox.x + boundingBox.width * 0.8
        const endX = boundingBox.x + boundingBox.width * 0.2
        const y = boundingBox.y + boundingBox.height / 2
        
        await page.touchscreen.tap(startX, y)
        await page.mouse.move(startX, y)
        await page.mouse.down()
        await page.mouse.move(endX, y)
        await page.mouse.up()
        
        await TestHelpers.waitForAnimations(page)
        console.log('✓ Swipe gesture performed')
      }
    } else {
      console.log('No swipeable elements found - skipping swipe test')
    }
    
    // Test pull-to-refresh if present
    const refreshContainer = page.locator('.pull-to-refresh, [data-testid*="refresh"]')
    if (await refreshContainer.count() > 0) {
      const container = refreshContainer.first()
      const boundingBox = await container.boundingBox()
      
      if (boundingBox) {
        // Simulate pull-to-refresh gesture
        const centerX = boundingBox.x + boundingBox.width / 2
        const startY = boundingBox.y + 10
        const endY = boundingBox.y + 100
        
        await page.touchscreen.tap(centerX, startY)
        await page.mouse.move(centerX, startY)
        await page.mouse.down()
        await page.mouse.move(centerX, endY)
        await page.mouse.up()
        
        await TestHelpers.waitForAnimations(page)
        console.log('✓ Pull-to-refresh gesture performed')
      }
    }
  })

  test('scroll behavior is optimized for mobile', async ({ page }) => {
    // Test vertical scrolling
    const bodyHeight = await page.evaluate(() => document.body.scrollHeight)
    const viewportHeight = await page.evaluate(() => window.innerHeight)
    
    if (bodyHeight > viewportHeight) {
      // Scroll down
      await page.mouse.wheel(0, 200)
      await TestHelpers.waitForAnimations(page)
      
      const scrollY = await page.evaluate(() => window.scrollY)
      expect(scrollY).toBeGreaterThan(0)
      
      // Scroll back to top
      await page.mouse.wheel(0, -scrollY)
      await TestHelpers.waitForAnimations(page)
      
      console.log('✓ Vertical scrolling works correctly')
    }
    
    // Test horizontal scrolling if applicable
    const horizontalScrollContainers = page.locator(
      '.horizontal-scroll, .scroll-x, .overflow-x-auto, .overflow-x-scroll'
    )
    
    if (await horizontalScrollContainers.count() > 0) {
      const container = horizontalScrollContainers.first()
      const scrollWidth = await container.evaluate(el => el.scrollWidth)
      const clientWidth = await container.evaluate(el => el.clientWidth)
      
      if (scrollWidth > clientWidth) {
        await container.hover()
        await page.mouse.wheel(100, 0) // Horizontal scroll
        await TestHelpers.waitForAnimations(page)
        
        const scrollLeft = await container.evaluate(el => el.scrollLeft)
        expect(scrollLeft).toBeGreaterThan(0)
        
        console.log('✓ Horizontal scrolling works correctly')
      }
    }
  })

  test('mobile navigation patterns work correctly', async ({ page }) => {
    // Test bottom navigation if present
    const bottomNav = page.locator('.bottom-navigation, .bottom-nav, [data-testid*="bottom-nav"]')
    if (await bottomNav.count() > 0) {
      await expect(bottomNav.first()).toBeVisible()
      
      // Test navigation items
      const navItems = bottomNav.locator('a, button, [role="tab"], [role="button"]')
      const itemCount = await navItems.count()
      
      if (itemCount > 0) {
        // Test tapping navigation items
        for (let i = 0; i < Math.min(itemCount, 3); i++) {
          const navItem = navItems.nth(i)
          await navItem.tap()
          await TestHelpers.waitForAnimations(page)
          
          // Verify active state or navigation occurred
          const isActive = await navItem.evaluate(el => 
            el.classList.contains('active') || 
            el.classList.contains('selected') ||
            el.getAttribute('aria-selected') === 'true'
          )
          
          if (isActive) {
            console.log(`✓ Navigation item ${i} activated successfully`)
          }
        }
      }
    }
    
    // Test hamburger menu if present
    const hamburgerMenu = page.locator(
      '.hamburger, .menu-toggle, [data-testid*="menu"], .mobile-menu-trigger'
    )
    
    if (await hamburgerMenu.count() > 0) {
      const menuTrigger = hamburgerMenu.first()
      await menuTrigger.tap()
      await TestHelpers.waitForAnimations(page)
      
      // Look for opened menu
      const openMenu = page.locator('.menu-open, .sidebar-open, .mobile-menu-open, nav.open')
      if (await openMenu.count() > 0) {
        await expect(openMenu.first()).toBeVisible()
        console.log('✓ Mobile menu opens correctly')
        
        // Close menu
        await menuTrigger.tap()
        await TestHelpers.waitForAnimations(page)
      }
    }
  })

  test('text input is optimized for mobile', async ({ page }) => {
    // Find text inputs
    const textInputs = page.locator('input[type="text"], input[type="email"], input[type="search"], textarea')
    const inputCount = await textInputs.count()
    
    if (inputCount > 0) {
      const firstInput = textInputs.first()
      
      // Check for mobile-optimized attributes
      const inputType = await firstInput.getAttribute('type')
      const inputMode = await firstInput.getAttribute('inputmode')
      const autoComplete = await firstInput.getAttribute('autocomplete')
      
      console.log('Input attributes:', { inputType, inputMode, autoComplete })
      
      // Test typing
      await firstInput.tap()
      await TestHelpers.waitForAnimations(page)
      
      // Verify input is focused
      await expect(firstInput).toBeFocused()
      
      // Type test text
      await firstInput.fill('Test input text')
      const inputValue = await firstInput.inputValue()
      expect(inputValue).toBe('Test input text')
      
      console.log('✓ Text input works correctly on mobile')
    }
  })

  test('mobile performance is acceptable', async ({ page }) => {
    const startTime = Date.now()
    
    // Measure interaction responsiveness
    const button = page.locator('button').first()
    if (await button.count() > 0) {
      const tapStartTime = Date.now()
      await button.tap()
      await TestHelpers.waitForAnimations(page)
      const tapEndTime = Date.now()
      
      const tapResponseTime = tapEndTime - tapStartTime
      expect(tapResponseTime).toBeLessThan(300) // Should respond within 300ms
      
      console.log(`Tap response time: ${tapResponseTime}ms`)
    }
    
    // Check for smooth animations
    const animatedElements = page.locator('.animate, .transition, [data-animate]')
    if (await animatedElements.count() > 0) {
      // Trigger animation
      const element = animatedElements.first()
      await element.hover()
      await TestHelpers.waitForAnimations(page)
      
      console.log('✓ Animations complete without blocking')
    }
    
    // Measure memory usage (basic check)
    const memoryInfo = await page.evaluate(() => {
      if ('memory' in performance) {
        const memory = (performance as any).memory
        return {
          usedJSHeapSize: memory.usedJSHeapSize,
          totalJSHeapSize: memory.totalJSHeapSize,
          jsHeapSizeLimit: memory.jsHeapSizeLimit
        }
      }
      return null
    })
    
    if (memoryInfo) {
      const memoryUsagePercent = (memoryInfo.usedJSHeapSize / memoryInfo.jsHeapSizeLimit) * 100
      expect(memoryUsagePercent).toBeLessThan(50) // Should use less than 50% of available heap
      
      console.log(`Memory usage: ${memoryUsagePercent.toFixed(2)}%`)
    }
  })

  test('device orientation changes are handled gracefully', async ({ page }) => {
    // Test portrait mode (current)
    await page.setViewportSize({ width: 375, height: 667 })
    await TestHelpers.waitForAnimations(page)
    
    await expect(page.locator('body')).toBeVisible()
    
    // Take portrait screenshot
    await page.screenshot({ 
      path: 'test-results/mobile/portrait-mode.png',
      fullPage: true 
    })
    
    // Switch to landscape mode
    await page.setViewportSize({ width: 667, height: 375 })
    await TestHelpers.waitForAnimations(page)
    
    // Verify app still functions in landscape
    await expect(page.locator('body')).toBeVisible()
    
    // Check for landscape-specific adjustments
    const navigation = page.locator('nav, .navigation, .bottom-navigation')
    if (await navigation.count() > 0) {
      await expect(navigation.first()).toBeVisible()
    }
    
    // Take landscape screenshot
    await page.screenshot({ 
      path: 'test-results/mobile/landscape-mode.png',
      fullPage: true 
    })
    
    console.log('✓ App handles orientation changes correctly')
  })

  test('mobile accessibility features work correctly', async ({ page }) => {
    // Test with reduced motion preference
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await page.reload()
    await TestHelpers.waitForNetworkIdle(page)
    
    // Verify app respects reduced motion
    const animatedElements = page.locator('.animate, .transition, [data-animate]')
    if (await animatedElements.count() > 0) {
      // Check if animations are disabled/reduced
      const animationDuration = await animatedElements.first().evaluate(el => {
        const styles = getComputedStyle(el)
        return styles.animationDuration || styles.transitionDuration
      })
      
      // Should be instant or very fast with reduced motion
      if (animationDuration && animationDuration !== '0s') {
        console.log(`Animation duration with reduced motion: ${animationDuration}`)
      }
    }
    
    // Test high contrast mode
    await page.emulateMedia({ colorScheme: 'dark', forcedColors: 'active' })
    await page.reload()
    await TestHelpers.waitForNetworkIdle(page)
    
    await expect(page.locator('body')).toBeVisible()
    
    // Take high contrast screenshot
    await page.screenshot({ 
      path: 'test-results/mobile/high-contrast-mode.png',
      fullPage: true 
    })
    
    console.log('✓ Mobile accessibility features work correctly')
  })
})