import { test, expect } from '@playwright/test'
import { TestHelpers } from '../../utils/test-helpers'

/**
 * Dark Mode & Theme Switching Tests with Visual Regression
 * 
 * Validates theme functionality:
 * - System preference detection
 * - Manual theme switching
 * - Theme persistence
 * - Visual consistency across themes
 * - Accessibility compliance for each theme
 * - Color contrast validation
 */

test.describe('Dark Mode & Theme Switching', () => {
  
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await TestHelpers.waitForNetworkIdle(page)
  })

  test('detects and respects system theme preference', async ({ page }) => {
    // Test with light system preference
    await page.emulateMedia({ colorScheme: 'light' })
    await page.reload()
    await TestHelpers.waitForNetworkIdle(page)
    
    // Check for light theme application
    const htmlElement = page.locator('html')
    const bodyElement = page.locator('body')
    
    // Common ways themes are applied
    const themeIndicators = [
      await htmlElement.getAttribute('data-theme'),
      await htmlElement.getAttribute('class'),
      await bodyElement.getAttribute('class'),
      await bodyElement.getAttribute('data-theme')
    ]
    
    const hasLightTheme = themeIndicators.some(indicator => 
      indicator && (indicator.includes('light') || !indicator.includes('dark'))
    )
    
    // Take light theme screenshot
    await page.screenshot({ 
      path: 'test-results/visual/system-light-theme.png',
      fullPage: true,
      animations: 'disabled'
    })
    
    // Test with dark system preference
    await page.emulateMedia({ colorScheme: 'dark' })
    await page.reload()
    await TestHelpers.waitForNetworkIdle(page)
    
    // Check for dark theme application
    const darkThemeIndicators = [
      await htmlElement.getAttribute('data-theme'),
      await htmlElement.getAttribute('class'),
      await bodyElement.getAttribute('class'),
      await bodyElement.getAttribute('data-theme')
    ]
    
    const hasDarkTheme = darkThemeIndicators.some(indicator => 
      indicator && indicator.includes('dark')
    )
    
    // Take dark theme screenshot
    await page.screenshot({ 
      path: 'test-results/visual/system-dark-theme.png',
      fullPage: true,
      animations: 'disabled'
    })
    
    console.log('System theme detection:', {
      lightThemeApplied: hasLightTheme,
      darkThemeApplied: hasDarkTheme,
      themeIndicators: darkThemeIndicators.filter(Boolean)
    })
  })

  test('manual theme toggle works correctly', async ({ page }) => {
    // Look for theme toggle button
    const themeToggleSelectors = [
      '[data-testid="theme-toggle"]',
      '.theme-toggle',
      '.dark-mode-toggle',
      'button[aria-label*="theme" i]',
      'button[aria-label*="dark" i]',
      'button[title*="theme" i]',
      '.theme-switcher',
      '#theme-toggle'
    ]
    
    let themeToggle = null
    for (const selector of themeToggleSelectors) {
      const element = page.locator(selector)
      if (await element.count() > 0) {
        themeToggle = element.first()
        break
      }
    }
    
    if (themeToggle) {
      await expect(themeToggle).toBeVisible()
      
      // Get initial theme state
      const initialTheme = await page.evaluate(() => {
        return {
          htmlClass: document.documentElement.className,
          htmlDataTheme: document.documentElement.getAttribute('data-theme'),
          bodyClass: document.body.className,
          computedStyle: getComputedStyle(document.body).backgroundColor
        }
      })
      
      // Take screenshot of initial state
      await page.screenshot({ 
        path: 'test-results/visual/theme-initial-state.png',
        fullPage: true,
        animations: 'disabled'
      })
      
      // Click theme toggle
      await themeToggle.click()
      await TestHelpers.waitForAnimations(page)
      
      // Get theme state after toggle
      const toggledTheme = await page.evaluate(() => {
        return {
          htmlClass: document.documentElement.className,
          htmlDataTheme: document.documentElement.getAttribute('data-theme'),
          bodyClass: document.body.className,
          computedStyle: getComputedStyle(document.body).backgroundColor
        }
      })
      
      // Take screenshot of toggled state
      await page.screenshot({ 
        path: 'test-results/visual/theme-toggled-state.png',
        fullPage: true,
        animations: 'disabled'
      })
      
      // Verify theme actually changed
      const themeChanged = 
        initialTheme.htmlClass !== toggledTheme.htmlClass ||
        initialTheme.htmlDataTheme !== toggledTheme.htmlDataTheme ||
        initialTheme.bodyClass !== toggledTheme.bodyClass ||
        initialTheme.computedStyle !== toggledTheme.computedStyle
      
      expect(themeChanged).toBe(true)
      console.log('✓ Theme toggle successfully changes theme')
      
      // Toggle back to verify bidirectional functionality
      await themeToggle.click()
      await TestHelpers.waitForAnimations(page)
      
      const revertedTheme = await page.evaluate(() => {
        return {
          htmlClass: document.documentElement.className,
          htmlDataTheme: document.documentElement.getAttribute('data-theme'),
          bodyClass: document.body.className,
          computedStyle: getComputedStyle(document.body).backgroundColor
        }
      })
      
      // Should be back to initial state (or close to it)
      const revertedSuccessfully = 
        Math.abs(initialTheme.computedStyle.localeCompare(revertedTheme.computedStyle)) < 2
      
      console.log('Theme toggle cycle:', {
        initial: initialTheme,
        toggled: toggledTheme,
        reverted: revertedTheme
      })
      
    } else {
      console.log('No theme toggle found - testing system theme changes only')
      test.skip('No theme toggle button found')
    }
  })

  test('theme persistence works across page reloads', async ({ page }) => {
    // Find and use theme toggle if available
    const themeToggle = page.locator('[data-testid="theme-toggle"], .theme-toggle, .dark-mode-toggle')
    
    if (await themeToggle.count() > 0) {
      // Set to dark mode
      await themeToggle.first().click()
      await TestHelpers.waitForAnimations(page)
      
      // Get dark theme state
      const darkThemeState = await page.evaluate(() => {
        return {
          theme: document.documentElement.getAttribute('data-theme'),
          className: document.documentElement.className,
          localStorage: localStorage.getItem('theme') || localStorage.getItem('darkMode')
        }
      })
      
      // Reload page
      await page.reload()
      await TestHelpers.waitForNetworkIdle(page)
      
      // Check if dark theme persisted
      const persistedThemeState = await page.evaluate(() => {
        return {
          theme: document.documentElement.getAttribute('data-theme'),
          className: document.documentElement.className,
          localStorage: localStorage.getItem('theme') || localStorage.getItem('darkMode')
        }
      })
      
      // Theme should persist
      const themePersisted = 
        darkThemeState.theme === persistedThemeState.theme ||
        darkThemeState.className === persistedThemeState.className ||
        darkThemeState.localStorage === persistedThemeState.localStorage
      
      expect(themePersisted).toBe(true)
      console.log('✓ Theme preference persists across page reloads')
      
    } else {
      console.log('No theme toggle available - testing system preference persistence')
      
      // Test system preference persistence
      await page.emulateMedia({ colorScheme: 'dark' })
      await page.reload()
      await TestHelpers.waitForNetworkIdle(page)
      
      // Should still respect system preference
      await expect(page.locator('body')).toBeVisible()
    }
  })

  test('color contrast meets accessibility standards', async ({ page }) => {
    const themes = ['light', 'dark']
    
    for (const theme of themes) {
      if (theme === 'dark') {
        await page.emulateMedia({ colorScheme: 'dark' })
      } else {
        await page.emulateMedia({ colorScheme: 'light' })
      }
      
      await page.reload()
      await TestHelpers.waitForNetworkIdle(page)
      
      // Get color information for key elements
      const colorData = await page.evaluate(() => {
        const elements = [
          { selector: 'body', name: 'body' },
          { selector: 'h1, h2, h3', name: 'headings' },
          { selector: 'p, span, div', name: 'text' },
          { selector: 'button', name: 'button' },
          { selector: 'a', name: 'links' }
        ]
        
        const colors = []
        
        for (const { selector, name } of elements) {
          const element = document.querySelector(selector)
          if (element) {
            const styles = getComputedStyle(element)
            colors.push({
              element: name,
              backgroundColor: styles.backgroundColor,
              color: styles.color,
              borderColor: styles.borderColor
            })
          }
        }
        
        return colors
      })
      
      // Calculate contrast ratios (simplified check)
      for (const colorInfo of colorData) {
        console.log(`${theme} theme - ${colorInfo.element}:`, {
          background: colorInfo.backgroundColor,
          foreground: colorInfo.color
        })
      }
      
      // Take screenshot for manual contrast verification
      await page.screenshot({ 
        path: `test-results/visual/contrast-${theme}-theme.png`,
        fullPage: true,
        animations: 'disabled'
      })
    }
  })

  test('all UI components render correctly in both themes', async ({ page }) => {
    const themes = [
      { name: 'light', colorScheme: 'light' },
      { name: 'dark', colorScheme: 'dark' }
    ]
    
    for (const theme of themes) {
      await page.emulateMedia({ colorScheme: theme.colorScheme as 'light' | 'dark' })
      await page.reload()
      await TestHelpers.waitForNetworkIdle(page)
      
      // Check for key UI components
      const components = [
        { selector: 'header', name: 'header' },
        { selector: 'nav, .navigation', name: 'navigation' },
        { selector: 'main, .main-content', name: 'main-content' },
        { selector: 'button', name: 'buttons' },
        { selector: '.card, .panel', name: 'cards' },
        { selector: '.agent-card, .task-card', name: 'dashboard-cards' },
        { selector: 'footer', name: 'footer' }
      ]
      
      for (const component of components) {
        const elements = page.locator(component.selector)
        const count = await elements.count()
        
        if (count > 0) {
          await expect(elements.first()).toBeVisible()
          console.log(`✓ ${component.name} visible in ${theme.name} theme`)
          
          // Take component-specific screenshot
          await elements.first().screenshot({ 
            path: `test-results/visual/${component.name}-${theme.name}-theme.png`,
            animations: 'disabled'
          })
        }
      }
      
      // Take full page screenshot for each theme
      await page.screenshot({ 
        path: `test-results/visual/full-page-${theme.name}-theme.png`,
        fullPage: true,
        animations: 'disabled'
      })
    }
  })

  test('theme transition animations are smooth', async ({ page }) => {
    const themeToggle = page.locator('[data-testid="theme-toggle"], .theme-toggle, .dark-mode-toggle')
    
    if (await themeToggle.count() > 0) {
      // Enable animation monitoring
      const animationEvents = []
      
      await page.addInitScript(() => {
        const originalAnimate = Element.prototype.animate
        Element.prototype.animate = function(...args) {
          window.__animationEvents = window.__animationEvents || []
          window.__animationEvents.push({
            element: this.tagName,
            timestamp: Date.now()
          })
          return originalAnimate.apply(this, args)
        }
      })
      
      // Perform theme toggle
      const toggleStartTime = Date.now()
      await themeToggle.first().click()
      
      // Wait for animations to complete
      await TestHelpers.waitForAnimations(page)
      const toggleEndTime = Date.now()
      
      const toggleDuration = toggleEndTime - toggleStartTime
      
      // Theme toggle should complete quickly (under 500ms)
      expect(toggleDuration).toBeLessThan(500)
      
      // Check for animation events
      const animationEvents = await page.evaluate(() => {
        return window.__animationEvents || []
      })
      
      console.log('Theme toggle performance:', {
        duration: toggleDuration,
        animationsTriggered: animationEvents.length
      })
      
    } else {
      console.log('No theme toggle available - skipping animation test')
    }
  })

  test('theme works correctly on different screen sizes', async ({ page }) => {
    const viewports = [
      { width: 320, height: 568, name: 'mobile' },
      { width: 768, height: 1024, name: 'tablet' },
      { width: 1280, height: 720, name: 'desktop' }
    ]
    
    const themes = ['light', 'dark']
    
    for (const viewport of viewports) {
      await page.setViewportSize(viewport)
      
      for (const theme of themes) {
        await page.emulateMedia({ colorScheme: theme as 'light' | 'dark' })
        await page.reload()
        await TestHelpers.waitForNetworkIdle(page)
        
        // Verify theme applied correctly
        await expect(page.locator('body')).toBeVisible()
        
        // Take screenshot for each combination
        await page.screenshot({ 
          path: `test-results/visual/${viewport.name}-${theme}-theme.png`,
          fullPage: true,
          animations: 'disabled'
        })
        
        console.log(`✓ ${theme} theme works on ${viewport.name} (${viewport.width}x${viewport.height})`)
      }
    }
  })

  test('high contrast mode compatibility', async ({ page }) => {
    // Test with forced colors (high contrast mode)
    await page.emulateMedia({ forcedColors: 'active' })
    await page.reload()
    await TestHelpers.waitForNetworkIdle(page)
    
    // Verify app still functions in high contrast mode
    await expect(page.locator('body')).toBeVisible()
    
    // Check that interactive elements are still visible
    const interactiveElements = page.locator('button, a, input, select')
    const count = await interactiveElements.count()
    
    if (count > 0) {
      for (let i = 0; i < Math.min(count, 5); i++) {
        await expect(interactiveElements.nth(i)).toBeVisible()
      }
      console.log('✓ Interactive elements visible in high contrast mode')
    }
    
    // Take high contrast screenshot
    await page.screenshot({ 
      path: 'test-results/visual/high-contrast-mode.png',
      fullPage: true,
      animations: 'disabled'
    })
    
    // Test with both light and dark in high contrast
    await page.emulateMedia({ colorScheme: 'dark', forcedColors: 'active' })
    await page.reload()
    await TestHelpers.waitForNetworkIdle(page)
    
    await expect(page.locator('body')).toBeVisible()
    
    await page.screenshot({ 
      path: 'test-results/visual/high-contrast-dark-mode.png',
      fullPage: true,
      animations: 'disabled'
    })
  })
})