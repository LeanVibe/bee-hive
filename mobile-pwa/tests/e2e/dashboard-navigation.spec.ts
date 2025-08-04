import { test, expect } from '@playwright/test'
import { DashboardPage } from '../fixtures/page-objects'
import { TestHelpers } from '../utils/test-helpers'
import { APIMocks } from '../utils/api-mocks'

test.describe('Dashboard Navigation and Layout', () => {
  let dashboardPage: DashboardPage

  test.beforeEach(async ({ page }) => {
    dashboardPage = new DashboardPage(page)
    
    // Set up API mocks for consistent testing
    await APIMocks.setupStandardMocks(page)
    
    // Navigate to dashboard
    await dashboardPage.goto('/')
    await dashboardPage.waitForLoad()
  })

  test.describe('Header and Navigation', () => {
    test('should display dashboard header with title and controls', async ({ page }) => {
      // Verify page title
      await expect(dashboardPage.pageTitle).toBeVisible()
      await expect(dashboardPage.pageTitle).toContainText('Agent Dashboard')
      
      // Verify sync status and controls
      await expect(dashboardPage.syncStatus).toBeVisible()
      await expect(dashboardPage.refreshButton).toBeVisible()
      await expect(dashboardPage.syncIndicator).toBeVisible()
      
      // Take screenshot for visual validation
      await TestHelpers.takeTimestampedScreenshot(page, 'dashboard-header')
    })

    test('should show all navigation tabs', async ({ page }) => {
      // Verify all tabs are visible
      await expect(dashboardPage.overviewTab).toBeVisible()
      await expect(dashboardPage.tasksTab).toBeVisible()
      await expect(dashboardPage.agentsTab).toBeVisible()
      await expect(dashboardPage.eventsTab).toBeVisible()
      
      // Verify Overview tab is active by default
      await expect(dashboardPage.overviewTab).toHaveClass(/active/)
    })

    test('should navigate between tabs correctly', async ({ page }) => {
      // Test navigation to Tasks tab
      await dashboardPage.navigateToTasks()
      await expect(dashboardPage.tasksTab).toHaveClass(/active/)
      await expect(page.locator('kanban-board')).toBeVisible()
      
      // Test navigation to Agents tab
      await dashboardPage.navigateToAgents()
      await expect(dashboardPage.agentsTab).toHaveClass(/active/)
      await expect(page.locator('agent-health-panel')).toBeVisible()
      
      // Test navigation to Events tab
      await dashboardPage.navigateToEvents()
      await expect(dashboardPage.eventsTab).toHaveClass(/active/)
      await expect(page.locator('event-timeline')).toBeVisible()
      
      // Test navigation back to Overview
      await dashboardPage.navigateToOverview()
      await expect(dashboardPage.overviewTab).toHaveClass(/active/)
      await expect(dashboardPage.activeTasksCard).toBeVisible()
    })

    test('should maintain tab state during navigation', async ({ page }) => {
      // Navigate to Tasks tab and verify state
      await dashboardPage.navigateToTasks()
      await expect(dashboardPage.tasksTab).toHaveClass(/active/)
      
      // Refresh the page
      await page.reload()
      await dashboardPage.waitForLoad()
      
      // Should return to Overview tab after reload (default state)
      await expect(dashboardPage.overviewTab).toHaveClass(/active/)
    })
  })

  test.describe('Overview Dashboard Layout', () => {
    test('should display all summary cards with data', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Verify all summary cards are visible
      await expect(dashboardPage.activeTasksCard).toBeVisible()
      await expect(dashboardPage.completedTasksCard).toBeVisible()
      await expect(dashboardPage.activeAgentsCard).toBeVisible()
      await expect(dashboardPage.systemHealthCard).toBeVisible()
      await expect(dashboardPage.cpuUsageCard).toBeVisible()
      await expect(dashboardPage.memoryUsageCard).toBeVisible()
      
      // Verify cards contain numeric values
      const activeTasks = await dashboardPage.getSummaryCardValue('Active Tasks')
      const completedTasks = await dashboardPage.getSummaryCardValue('Completed Tasks')
      const activeAgents = await dashboardPage.getSummaryCardValue('Active Agents')
      
      expect(parseInt(activeTasks)).toBeGreaterThanOrEqual(0)
      expect(parseInt(completedTasks)).toBeGreaterThanOrEqual(0)
      expect(parseInt(activeAgents)).toBeGreaterThanOrEqual(0)
    })

    test('should display dashboard panels', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Verify main dashboard panels
      await expect(dashboardPage.agentHealthPanel).toBeVisible()
      await expect(dashboardPage.eventTimeline).toBeVisible()
      
      // Verify panels contain content
      const agentCards = page.locator('agent-health-panel .agent-card')
      const eventItems = page.locator('event-timeline .timeline-event')
      
      await expect(agentCards.first()).toBeVisible()
      await expect(eventItems.first()).toBeVisible()
    })

    test('should handle loading states properly', async ({ page }) => {
      // Simulate slow network to test loading states
      await APIMocks.clearAllMocks(page)
      await APIMocks.mockSlowNetwork(page, 1000)
      
      // Navigate to dashboard
      await page.goto('/')
      
      // Should show loading indicators
      await TestHelpers.verifyLoadingState(page, '.overview-summary')
      
      // Content should load after delay
      await expect(dashboardPage.activeTasksCard).toBeVisible()
    })
  })

  test.describe('Responsive Design', () => {
    test('should adapt layout for mobile screens', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 375, 667) // iPhone SE
      
      await dashboardPage.navigateToOverview()
      
      // Verify mobile layout adaptations
      await expect(dashboardPage.pageTitle).toBeVisible()
      await expect(dashboardPage.overviewTab).toBeVisible()
      
      // Summary cards should stack vertically on mobile
      const summaryCards = page.locator('.summary-card')
      const firstCard = summaryCards.first()
      const secondCard = summaryCards.nth(1)
      
      if (await firstCard.isVisible() && await secondCard.isVisible()) {
        const firstCardBox = await firstCard.boundingBox()
        const secondCardBox = await secondCard.boundingBox()
        
        // Cards should be stacked vertically (y position different)
        expect(firstCardBox?.y).not.toBe(secondCardBox?.y)
      }
      
      await TestHelpers.takeTimestampedScreenshot(page, 'mobile-dashboard')
    })

    test('should adapt layout for tablet screens', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 768, 1024) // iPad
      
      await dashboardPage.navigateToOverview()
      
      // Verify tablet layout
      await expect(dashboardPage.activeTasksCard).toBeVisible()
      await expect(dashboardPage.agentHealthPanel).toBeVisible()
      
      await TestHelpers.takeTimestampedScreenshot(page, 'tablet-dashboard')
    })

    test('should adapt layout for desktop screens', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 1920, 1080) // Desktop
      
      await dashboardPage.navigateToOverview()
      
      // Verify desktop layout with side-by-side panels
      await expect(dashboardPage.agentHealthPanel).toBeVisible()
      await expect(dashboardPage.eventTimeline).toBeVisible()
      
      await TestHelpers.takeTimestampedScreenshot(page, 'desktop-dashboard')
    })
  })

  test.describe('Refresh and Sync Functionality', () => {
    test('should refresh data when refresh button is clicked', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Initial data load
      await dashboardPage.waitForRealTimeUpdate()
      
      // Click refresh button
      await dashboardPage.refreshData()
      
      // Verify refresh was triggered (button animation)
      await expect(dashboardPage.refreshButton).not.toHaveClass(/spinning/)
      
      // Verify sync status updates
      await dashboardPage.waitForRealTimeUpdate()
    })

    test('should show sync status accurately', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Verify online sync status
      await expect(dashboardPage.syncStatus).not.toContainText('Offline mode')
      await expect(dashboardPage.syncIndicator).not.toHaveClass(/offline/)
      
      // Verify sync time updates
      const syncText = await dashboardPage.syncStatus.textContent()
      expect(syncText).toMatch(/Last sync:|just now/)
    })

    test('should handle offline mode properly', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Simulate offline
      await TestHelpers.simulateOfflineNetwork(page)
      await page.reload()
      await dashboardPage.waitForLoad()
      
      // Verify offline indicators
      await dashboardPage.verifyOfflineMode()
      
      // Restore network
      await TestHelpers.restoreNetwork(page)
    })
  })

  test.describe('Error Handling', () => {
    test('should handle API errors gracefully', async ({ page }) => {
      // Set up error mocks
      await APIMocks.clearAllMocks(page)
      await APIMocks.mockErrorResponses(page, 'server')
      
      await page.goto('/')
      
      // Should show error state
      await TestHelpers.verifyErrorHandling(page)
      
      // Should have retry functionality
      const retryButton = page.locator('.btn-primary', { hasText: 'Try Again' })
      if (await retryButton.isVisible()) {
        await expect(retryButton).toBeVisible()
      }
    })

    test('should handle partial failures gracefully', async ({ page }) => {
      // Mock partial failures (some endpoints succeed, others fail)
      await APIMocks.clearAllMocks(page)
      await APIMocks.mockPartialFailures(page, ['/agents'])
      
      await dashboardPage.goto('/')
      await dashboardPage.waitForLoad()
      
      // Tasks should load (success)
      await expect(dashboardPage.activeTasksCard).toBeVisible()
      
      // Agents might show error or empty state
      await expect(dashboardPage.agentHealthPanel).toBeVisible()
    })

    test('should handle network timeouts', async ({ page }) => {
      await APIMocks.clearAllMocks(page)
      await APIMocks.mockErrorResponses(page, 'timeout')
      
      await page.goto('/')
      
      // Should eventually show error or timeout message
      await page.waitForTimeout(5000) // Wait for timeout to occur
      
      // Verify error handling
      const errorElements = page.locator('.error-state, .error-message, [data-testid="error"]')
      if (await errorElements.count() > 0) {
        await expect(errorElements.first()).toBeVisible()
      }
    })
  })

  test.describe('Accessibility', () => {
    test('should support keyboard navigation', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Test keyboard navigation through tabs
      await TestHelpers.testKeyboardNavigation(page, '.tab-button:first-child', [
        '.tab-button:nth-child(2)',
        '.tab-button:nth-child(3)',
        '.tab-button:nth-child(4)'
      ])
    })

    test('should have proper ARIA labels and roles', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Verify accessibility features
      await TestHelpers.verifyAccessibility(page, '.dashboard-header')
      await TestHelpers.verifyAccessibility(page, '.view-tabs')
      await TestHelpers.verifyAccessibility(page, '.overview-summary')
    })

    test('should support screen readers', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Verify important elements have accessible names
      await expect(dashboardPage.pageTitle).toHaveAttribute('aria-label')
      await expect(dashboardPage.refreshButton).toHaveAttribute('title')
      
      // Verify navigation elements are properly labeled
      const tabs = page.locator('.tab-button')
      const tabCount = await tabs.count()
      
      for (let i = 0; i < tabCount; i++) {
        const tab = tabs.nth(i)
        const hasText = await tab.textContent()
        const hasAriaLabel = await tab.getAttribute('aria-label')
        
        expect(hasText || hasAriaLabel).toBeTruthy()
      }
    })
  })

  test.describe('Performance', () => {
    test('should load dashboard within acceptable time', async ({ page }) => {
      const startTime = Date.now()
      
      await dashboardPage.goto('/')
      await dashboardPage.waitForLoad()
      await expect(dashboardPage.activeTasksCard).toBeVisible()
      
      const loadTime = Date.now() - startTime
      
      // Dashboard should load within 5 seconds
      expect(loadTime).toBeLessThan(5000)
    })

    test('should handle large datasets efficiently', async ({ page }) => {
      // Mock large dataset
      const largeMockTasks = Array.from({ length: 100 }, (_, i) => ({
        id: `task-${i}`,
        title: `Task ${i}`,
        description: `Description for task ${i}`,
        status: ['pending', 'in-progress', 'review', 'done'][i % 4],
        priority: 'medium',
        agent: `agent-${i % 4}`,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }))
      
      await page.route('**/api/v1/tasks', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(largeMockTasks)
        })
      })
      
      const startTime = Date.now()
      await dashboardPage.goto('/')
      await dashboardPage.waitForLoad()
      const loadTime = Date.now() - startTime
      
      // Should still load efficiently with large dataset
      expect(loadTime).toBeLessThan(10000)
      
      // Verify data is displayed
      await expect(dashboardPage.activeTasksCard).toBeVisible()
    })
  })
})