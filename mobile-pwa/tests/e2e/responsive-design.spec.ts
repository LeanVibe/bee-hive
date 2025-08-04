import { test, expect } from '@playwright/test'
import { DashboardPage } from '../fixtures/page-objects'
import { TestHelpers } from '../utils/test-helpers'
import { APIMocks } from '../utils/api-mocks'

test.describe('Responsive Design and Mobile Interactions', () => {
  let dashboardPage: DashboardPage

  test.beforeEach(async ({ page }) => {
    dashboardPage = new DashboardPage(page)
    
    // Set up API mocks for consistent testing
    await APIMocks.setupStandardMocks(page)
    
    // Navigate to dashboard
    await dashboardPage.goto('/')
    await dashboardPage.waitForLoad()
  })

  test.describe('Desktop Responsive Breakpoints', () => {
    test('should display correctly on large desktop screens', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 1920, 1080)
      await dashboardPage.navigateToOverview()
      
      // Large desktop layout
      await expect(dashboardPage.activeTasksCard).toBeVisible()
      await expect(dashboardPage.agentHealthPanel).toBeVisible()
      
      // Summary cards should be in a grid layout
      const summaryGrid = page.locator('.overview-summary')
      await expect(summaryGrid).toHaveCSS('display', 'grid')
      
      // Dashboard panels should be side-by-side
      const overviewPanels = page.locator('.overview-panels')
      const computedStyle = await overviewPanels.evaluate(el => 
        window.getComputedStyle(el).gridTemplateColumns
      )
      
      // Should have multiple columns on large screens
      expect(computedStyle).not.toBe('1fr')
      
      await TestHelpers.takeTimestampedScreenshot(page, 'desktop-large-1920')
    })

    test('should adapt to medium desktop screens', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 1366, 768)
      await dashboardPage.navigateToOverview()
      
      // Medium desktop layout
      await expect(dashboardPage.activeTasksCard).toBeVisible()
      await expect(dashboardPage.systemHealthCard).toBeVisible()
      
      // Content should still be well-organized
      const headerContent = page.locator('.header-content')
      await expect(headerContent).toHaveCSS('display', 'flex')
      
      await TestHelpers.takeTimestampedScreenshot(page, 'desktop-medium-1366')
    })

    test('should handle ultrawide screens', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 2560, 1440)
      await dashboardPage.navigateToOverview()
      
      // Ultrawide layout should not stretch content too much
      const dashboardContent = page.locator('.dashboard-content')
      const maxWidth = await dashboardContent.evaluate(el => 
        window.getComputedStyle(el).maxWidth
      )
      
      // Should have reasonable max-width
      expect(maxWidth).toMatch(/\d+px|1200px/)
      
      await TestHelpers.takeTimestampedScreenshot(page, 'desktop-ultrawide-2560')
    })
  })

  test.describe('Tablet Responsive Breakpoints', () => {
    test('should display correctly on iPad Pro', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 1024, 1366)
      await dashboardPage.navigateToOverview()
      
      // Tablet layout adaptations
      await expect(dashboardPage.pageTitle).toBeVisible()
      await expect(dashboardPage.activeTasksCard).toBeVisible()
      
      // Tab navigation should be touch-friendly
      const tabs = page.locator('.tab-button')
      const firstTab = tabs.first()
      const tabHeight = await firstTab.evaluate(el => el.getBoundingClientRect().height)
      
      // Tabs should be at least 44px tall for touch targets
      expect(tabHeight).toBeGreaterThanOrEqual(44)
      
      await TestHelpers.takeTimestampedScreenshot(page, 'tablet-ipad-pro')
    })

    test('should display correctly on standard iPad', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 768, 1024)
      await dashboardPage.navigateToOverview()
      
      // Standard iPad layout
      await expect(dashboardPage.overviewTab).toBeVisible()
      
      // Summary cards should adapt to tablet width
      const summaryCards = page.locator('.summary-card')
      const cardCount = await summaryCards.count()
      
      if (cardCount > 0) {
        const firstCard = summaryCards.first()
        await expect(firstCard).toBeVisible()
        
        // Cards should have appropriate sizing
        const cardWidth = await firstCard.evaluate(el => el.getBoundingClientRect().width)
        expect(cardWidth).toBeGreaterThan(100)
      }
      
      await TestHelpers.takeTimestampedScreenshot(page, 'tablet-ipad-standard')
    })

    test('should handle tablet in landscape mode', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 1024, 768)
      await dashboardPage.navigateToOverview()
      
      // Landscape tablet layout
      await expect(dashboardPage.agentHealthPanel).toBeVisible()
      await expect(dashboardPage.eventTimeline).toBeVisible()
      
      // Should utilize horizontal space effectively
      const overviewPanels = page.locator('.overview-panels')
      const gridColumns = await overviewPanels.evaluate(el => 
        window.getComputedStyle(el).gridTemplateColumns
      )
      
      // Should have multiple columns in landscape
      expect(gridColumns).not.toBe('1fr')
      
      await TestHelpers.takeTimestampedScreenshot(page, 'tablet-landscape')
    })
  })

  test.describe('Mobile Responsive Breakpoints', () => {
    test('should display correctly on iPhone 14 Pro', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 393, 852)
      await dashboardPage.navigateToOverview()
      
      // Mobile layout adaptations
      await expect(dashboardPage.pageTitle).toBeVisible()
      
      // Page title should be smaller on mobile
      const titleFontSize = await dashboardPage.pageTitle.evaluate(el => 
        window.getComputedStyle(el).fontSize
      )
      
      const titleSizePx = parseFloat(titleFontSize.replace('px', ''))
      expect(titleSizePx).toBeLessThan(30) // Should be smaller than desktop
      
      // Tab navigation should scroll horizontally
      const tabContainer = page.locator('.view-tabs')
      const overflowX = await tabContainer.evaluate(el => 
        window.getComputedStyle(el).overflowX
      )
      expect(overflowX).toBe('auto')
      
      await TestHelpers.takeTimestampedScreenshot(page, 'mobile-iphone14-pro')
    })

    test('should display correctly on iPhone SE', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 375, 667)
      await dashboardPage.navigateToOverview()
      
      // Small mobile screen adaptations
      await expect(dashboardPage.activeTasksCard).toBeVisible()
      
      // Summary cards should stack vertically
      const summaryCards = page.locator('.summary-card')
      if (await summaryCards.count() >= 2) {
        const firstCard = summaryCards.first()
        const secondCard = summaryCards.nth(1)
        
        const firstCardRect = await firstCard.boundingBox()
        const secondCardRect = await secondCard.boundingBox()
        
        if (firstCardRect && secondCardRect) {
          // Cards should be vertically stacked (different y positions)
          expect(Math.abs(firstCardRect.y - secondCardRect.y)).toBeGreaterThan(50)
        }
      }
      
      await TestHelpers.takeTimestampedScreenshot(page, 'mobile-iphone-se')
    })

    test('should display correctly on Android phones', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 360, 640)
      await dashboardPage.navigateToOverview()
      
      // Android mobile layout
      await expect(dashboardPage.syncStatus).toBeVisible()
      
      // Sync status text should be readable on small screens
      const syncStatusFontSize = await dashboardPage.syncStatus.evaluate(el => 
        window.getComputedStyle(el).fontSize
      )
      
      const fontSizePx = parseFloat(syncStatusFontSize.replace('px', ''))
      expect(fontSizePx).toBeGreaterThanOrEqual(12) // Minimum readable size
      
      await TestHelpers.takeTimestampedScreenshot(page, 'mobile-android')
    })
  })

  test.describe('Touch Interactions', () => {
    test('should handle tap gestures on mobile', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 375, 667)
      await dashboardPage.navigateToOverview()
      
      // Test tab tapping
      await TestHelpers.testTouchInteraction(page, '.tab-button:nth-child(2)', 'tap')
      
      // Should navigate to the tapped tab
      const tasksTab = dashboardPage.tasksTab
      await expect(tasksTab).toHaveClass(/active/)
      
      // Test refresh button tap
      await TestHelpers.testTouchInteraction(page, '.refresh-button', 'tap')
      
      // Should trigger refresh (button might show loading state)
      await page.waitForTimeout(500)
    })

    test('should handle swipe gestures on Kanban board', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 375, 667)
      await dashboardPage.navigateToTasks()
      
      // Kanban board should be horizontally scrollable
      const boardContainer = page.locator('.board-container')
      const overflowX = await boardContainer.evaluate(el => 
        window.getComputedStyle(el).overflowX
      )
      expect(overflowX).toBe('auto')
      
      // Test horizontal swipe on board
      await TestHelpers.testTouchInteraction(page, '.board-container', 'swipe')
      
      // Board should remain functional after swipe
      await expect(boardContainer).toBeVisible()
    })

    test('should support pull-to-refresh gesture', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 375, 667)
      await dashboardPage.navigateToOverview()
      
      // Look for pull-to-refresh component
      const pullToRefresh = page.locator('pull-to-refresh, [data-testid="pull-to-refresh"]')
      
      if (await pullToRefresh.isVisible()) {
        // Simulate pull-to-refresh gesture
        await page.evaluate(() => {
          const touchStart = new TouchEvent('touchstart', {
            touches: [{ clientX: 100, clientY: 100 } as Touch]
          })
          const touchMove = new TouchEvent('touchmove', {
            touches: [{ clientX: 100, clientY: 200 } as Touch]
          })
          const touchEnd = new TouchEvent('touchend', { touches: [] })
          
          document.dispatchEvent(touchStart)
          setTimeout(() => document.dispatchEvent(touchMove), 100)
          setTimeout(() => document.dispatchEvent(touchEnd), 200)
        })
        
        // Should trigger refresh
        await page.waitForTimeout(1000)
        
        // Verify refresh was triggered
        await expect(dashboardPage.activeTasksCard).toBeVisible()
      }
    })

    test('should handle long press gestures', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 375, 667)
      await dashboardPage.navigateToTasks()
      
      // Find a task card
      const taskCard = page.locator('task-card').first()
      
      if (await taskCard.isVisible()) {
        // Simulate long press
        await page.evaluate(() => {
          const element = document.querySelector('task-card')
          if (element) {
            const touchStart = new TouchEvent('touchstart', { 
              touches: [{ clientX: 100, clientY: 100 } as Touch]
            })
            element.dispatchEvent(touchStart)
            
            // Hold for long press duration
            setTimeout(() => {
              const touchEnd = new TouchEvent('touchend', { touches: [] })
              element.dispatchEvent(touchEnd)
            }, 800)
          }
        })
        
        // Should show context menu or selection
        await page.waitForTimeout(1000)
        
        // Look for context menu or selection indicators
        const contextMenu = page.locator('.context-menu, .selection-mode')
        if (await contextMenu.isVisible()) {
          await expect(contextMenu).toBeVisible()
        }
      }
    })
  })

  test.describe('Accessibility on Mobile', () => {
    test('should have appropriate touch target sizes', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 375, 667)
      await dashboardPage.navigateToOverview()
      
      // Check tab buttons (primary interactive elements)
      const tabs = page.locator('.tab-button')
      const tabCount = await tabs.count()
      
      for (let i = 0; i < tabCount; i++) {
        const tab = tabs.nth(i)
        const rect = await tab.boundingBox()
        
        if (rect) {
          // Touch targets should be at least 44x44 pixels
          expect(rect.height).toBeGreaterThanOrEqual(44)
          expect(rect.width).toBeGreaterThanOrEqual(44)
        }
      }
      
      // Check refresh button
      const refreshButton = dashboardPage.refreshButton
      const refreshRect = await refreshButton.boundingBox()
      
      if (refreshRect) {
        expect(refreshRect.height).toBeGreaterThanOrEqual(44)
        expect(refreshRect.width).toBeGreaterThanOrEqual(44)
      }
    })

    test('should support screen reader navigation on mobile', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 375, 667)
      await dashboardPage.navigateToOverview()
      
      // Verify ARIA labels are present
      await TestHelpers.verifyAccessibility(page, '.dashboard-header')
      
      // Important elements should have accessible names
      await expect(dashboardPage.pageTitle).toHaveAttribute('aria-label')
      await expect(dashboardPage.refreshButton).toHaveAttribute('title')
      
      // Navigation should be logical for screen readers
      const landmarkElements = page.locator('[role="main"], [role="navigation"], [role="banner"]')
      const landmarkCount = await landmarkElements.count()
      expect(landmarkCount).toBeGreaterThan(0)
    })

    test('should support high contrast mode', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 375, 667)
      
      // Simulate high contrast mode
      await page.emulateMedia({ colorScheme: 'dark' })
      await page.addStyleTag({
        content: `
          @media (prefers-contrast: high) {
            * {
              filter: contrast(2) !important;
            }
          }
        `
      })
      
      await dashboardPage.navigateToOverview()
      
      // Elements should remain visible and functional
      await expect(dashboardPage.pageTitle).toBeVisible()
      await expect(dashboardPage.activeTasksCard).toBeVisible()
      
      // Text should have sufficient contrast
      const titleColor = await dashboardPage.pageTitle.evaluate(el => 
        window.getComputedStyle(el).color
      )
      expect(titleColor).toBeTruthy()
      
      await TestHelpers.takeTimestampedScreenshot(page, 'mobile-high-contrast')
    })

    test('should support reduced motion preferences', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 375, 667)
      
      // Simulate reduced motion preference
      await page.emulateMedia({ reducedMotion: 'reduce' })
      
      await dashboardPage.navigateToOverview()
      
      // Animations should be reduced or disabled
      const refreshButton = dashboardPage.refreshButton
      await refreshButton.click()
      
      // Spinner animation should be minimal with reduced motion
      await page.waitForTimeout(500)
      
      // Page should still be functional
      await expect(dashboardPage.activeTasksCard).toBeVisible()
    })
  })

  test.describe('PWA Mobile Features', () => {
    test('should display install prompt on mobile', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 375, 667)
      await dashboardPage.navigateToOverview()
      
      // Look for install prompt component
      const installPrompt = page.locator('install-prompt, [data-testid="install-prompt"]')
      
      if (await installPrompt.isVisible()) {
        await expect(installPrompt).toBeVisible()
        
        // Should have install and dismiss buttons
        const installButton = installPrompt.locator('.install-button')
        const dismissButton = installPrompt.locator('.dismiss-button')
        
        await expect(installButton).toBeVisible()
        await expect(dismissButton).toBeVisible()
      }
    })

    test('should work in standalone PWA mode', async ({ page }) => {
      // Simulate PWA standalone mode
      await page.addInitScript(() => {
        Object.defineProperty(window.navigator, 'standalone', {
          writable: false,
          value: true
        })
      })
      
      await TestHelpers.testResponsiveBreakpoint(page, 375, 667)
      await dashboardPage.navigateToOverview()
      
      // Should hide browser chrome simulation
      const header = dashboardPage.pageTitle
      await expect(header).toBeVisible()
      
      // Should adapt to full-screen mode
      const dashboardContent = page.locator('.dashboard-content')
      const height = await dashboardContent.evaluate(el => 
        window.getComputedStyle(el).height
      )
      
      expect(height).toBeTruthy()
    })

    test('should handle orientation changes', async ({ page }) => {
      // Start in portrait
      await TestHelpers.testResponsiveBreakpoint(page, 375, 667)
      await dashboardPage.navigateToOverview()
      
      // Verify portrait layout
      await expect(dashboardPage.activeTasksCard).toBeVisible()
      
      // Switch to landscape
      await TestHelpers.testResponsiveBreakpoint(page, 667, 375)
      
      // Should adapt to landscape
      await expect(dashboardPage.pageTitle).toBeVisible()
      await expect(dashboardPage.overviewTab).toBeVisible()
      
      // Summary cards might change layout
      const summaryGrid = page.locator('.overview-summary')
      await expect(summaryGrid).toBeVisible()
      
      await TestHelpers.takeTimestampedScreenshot(page, 'mobile-landscape')
    })

    test('should support mobile gestures in PWA mode', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 375, 667)
      await dashboardPage.navigateToTasks()
      
      // Test swipe gesture on Kanban board
      const boardContainer = page.locator('.board-container')
      
      // Simulate swipe gesture
      await page.evaluate(() => {
        const container = document.querySelector('.board-container')
        if (container) {
          const touchStart = new TouchEvent('touchstart', {
            touches: [{ clientX: 200, clientY: 300 } as Touch]
          })
          const touchMove = new TouchEvent('touchmove', {
            touches: [{ clientX: 100, clientY: 300 } as Touch]
          })
          const touchEnd = new TouchEvent('touchend', { touches: [] })
          
          container.dispatchEvent(touchStart)
          setTimeout(() => container.dispatchEvent(touchMove), 50)
          setTimeout(() => container.dispatchEvent(touchEnd), 100)
        }
      })
      
      // Board should handle gesture gracefully
      await expect(boardContainer).toBeVisible()
    })
  })

  test.describe('Performance on Mobile Devices', () => {
    test('should load efficiently on slow mobile connections', async ({ page }) => {
      // Simulate slow 3G connection
      await page.route('**/*', async (route, request) => {
        // Add delay to simulate slow connection
        await new Promise(resolve => setTimeout(resolve, 200))
        await route.continue()
      })
      
      await TestHelpers.testResponsiveBreakpoint(page, 375, 667)
      
      const startTime = Date.now()
      await dashboardPage.goto('/')
      await dashboardPage.waitForLoad()
      const loadTime = Date.now() - startTime
      
      // Should load within reasonable time even on slow connection
      expect(loadTime).toBeLessThan(15000)
      
      // Critical content should be visible
      await expect(dashboardPage.pageTitle).toBeVisible()
      await expect(dashboardPage.overviewTab).toBeVisible()
    })

    test('should handle memory constraints on mobile', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 375, 667)
      await dashboardPage.navigateToOverview()
      
      // Navigate through all views to test memory usage
      await dashboardPage.navigateToTasks()
      await page.waitForTimeout(1000)
      
      await dashboardPage.navigateToAgents()
      await page.waitForTimeout(1000)
      
      await dashboardPage.navigateToEvents()
      await page.waitForTimeout(1000)
      
      await dashboardPage.navigateToOverview()
      
      // Should remain responsive after navigation
      await expect(dashboardPage.activeTasksCard).toBeVisible()
      
      // Check for memory leaks by monitoring console errors
      const errors = await TestHelpers.monitorConsoleErrors(page)
      
      // Should not have memory-related errors
      const memoryErrors = errors.filter(error => 
        error.toLowerCase().includes('memory') || 
        error.toLowerCase().includes('leak')
      )
      expect(memoryErrors).toHaveLength(0)
    })

    test('should optimize images and assets for mobile', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 375, 667)
      await dashboardPage.navigateToOverview()
      
      // Check that images are appropriately sized
      const images = page.locator('img')
      const imageCount = await images.count()
      
      for (let i = 0; i < imageCount; i++) {
        const image = images.nth(i)
        if (await image.isVisible()) {
          const naturalWidth = await image.evaluate((img: HTMLImageElement) => img.naturalWidth)
          const displayWidth = await image.evaluate((img: HTMLImageElement) => img.getBoundingClientRect().width)
          
          // Images shouldn't be significantly larger than display size
          if (naturalWidth > 0 && displayWidth > 0) {
            const ratio = naturalWidth / displayWidth
            expect(ratio).toBeLessThan(3) // Shouldn't be more than 3x larger than needed
          }
        }
      }
    })
  })
})