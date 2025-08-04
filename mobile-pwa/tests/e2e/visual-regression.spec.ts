import { test, expect } from '@playwright/test'
import { DashboardPage } from '../fixtures/page-objects'
import { TestHelpers } from '../utils/test-helpers'
import { APIMocks } from '../utils/api-mocks'

test.describe('Visual Regression Testing', () => {
  let dashboardPage: DashboardPage

  test.beforeEach(async ({ page }) => {
    dashboardPage = new DashboardPage(page)
    
    // Set up consistent mock data for visual testing
    await APIMocks.setupStandardMocks(page)
    
    // Navigate to dashboard
    await dashboardPage.goto('/')
    await dashboardPage.waitForLoad()
  })

  test.describe('Dashboard Views Screenshots', () => {
    test('should match overview dashboard layout', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Wait for all content to load
      await expect(dashboardPage.activeTasksCard).toBeVisible()
      await expect(dashboardPage.agentHealthPanel).toBeVisible()
      await expect(dashboardPage.eventTimeline).toBeVisible()
      
      // Take full page screenshot
      await expect(page).toHaveScreenshot('dashboard-overview.png', {
        fullPage: true,
        animations: 'disabled'
      })
    })

    test('should match tasks Kanban board layout', async ({ page }) => {
      await dashboardPage.navigateToTasks()
      
      // Wait for Kanban board to load
      await expect(page.locator('kanban-board')).toBeVisible()
      await expect(page.locator('kanban-column')).toHaveCount(4)
      
      // Take screenshot of Kanban board
      await expect(page).toHaveScreenshot('dashboard-tasks.png', {
        fullPage: true,
        animations: 'disabled'
      })
    })

    test('should match agents management view', async ({ page }) => {
      await dashboardPage.navigateToAgents()
      
      // Wait for agents to load
      await expect(page.locator('agent-health-panel')).toBeVisible()
      await expect(page.locator('.agent-card')).toHaveCount.greaterThan(0)
      
      // Take screenshot of agents view
      await expect(page).toHaveScreenshot('dashboard-agents.png', {
        fullPage: true,
        animations: 'disabled'
      })
    })

    test('should match events timeline view', async ({ page }) => {
      await dashboardPage.navigateToEvents()
      
      // Wait for events to load
      await expect(page.locator('event-timeline')).toBeVisible()
      await expect(page.locator('.timeline-event')).toHaveCount.greaterThan(0)
      
      // Take screenshot of events view
      await expect(page).toHaveScreenshot('dashboard-events.png', {
        fullPage: true,
        animations: 'disabled'
      })
    })
  })

  test.describe('Component Screenshots', () => {
    test('should match summary cards appearance', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Screenshot of summary cards section
      const summarySection = page.locator('.overview-summary')
      await expect(summarySection).toBeVisible()
      
      await expect(summarySection).toHaveScreenshot('summary-cards.png', {
        animations: 'disabled'
      })
    })

    test('should match agent health panel appearance', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Screenshot of agent health panel
      const agentPanel = page.locator('agent-health-panel')
      await expect(agentPanel).toBeVisible()
      
      await expect(agentPanel).toHaveScreenshot('agent-health-panel.png', {
        animations: 'disabled'
      })
    })

    test('should match event timeline appearance', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Screenshot of event timeline
      const eventTimeline = page.locator('event-timeline')
      await expect(eventTimeline).toBeVisible()
      
      await expect(eventTimeline).toHaveScreenshot('event-timeline.png', {
        animations: 'disabled'
      })
    })

    test('should match task cards appearance', async ({ page }) => {
      await dashboardPage.navigateToTasks()
      
      // Screenshot of a task card
      const taskCard = page.locator('task-card').first()
      await expect(taskCard).toBeVisible()
      
      await expect(taskCard).toHaveScreenshot('task-card.png', {
        animations: 'disabled'
      })
    })
  })

  test.describe('Responsive Visual Testing', () => {
    test('should match mobile layout', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 375, 667)
      await dashboardPage.navigateToOverview()
      
      // Wait for mobile layout to settle
      await TestHelpers.waitForAnimations(page)
      
      await expect(page).toHaveScreenshot('dashboard-mobile.png', {
        fullPage: true,
        animations: 'disabled'
      })
    })

    test('should match tablet layout', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 768, 1024)
      await dashboardPage.navigateToOverview()
      
      // Wait for tablet layout to settle
      await TestHelpers.waitForAnimations(page)
      
      await expect(page).toHaveScreenshot('dashboard-tablet.png', {
        fullPage: true,
        animations: 'disabled'
      })
    })

    test('should match desktop layout', async ({ page }) => {
      await TestHelpers.testResponsiveBreakpoint(page, 1920, 1080)
      await dashboardPage.navigateToOverview()
      
      // Wait for desktop layout to settle
      await TestHelpers.waitForAnimations(page)
      
      await expect(page).toHaveScreenshot('dashboard-desktop.png', {
        fullPage: true,
        animations: 'disabled'
      })
    })
  })

  test.describe('State-based Visual Testing', () => {
    test('should match loading states', async ({ page }) => {
      // Mock slow loading
      await APIMocks.mockSlowNetwork(page, 3000)
      
      await page.goto('/')
      
      // Capture loading state
      const loadingElement = page.locator('.loading-state, .spinner')
      if (await loadingElement.isVisible()) {
        await expect(loadingElement).toHaveScreenshot('loading-state.png', {
          animations: 'disabled'
        })
      }
    })

    test('should match error states', async ({ page }) => {
      await APIMocks.mockErrorResponses(page, 'server')
      
      await page.goto('/')
      
      // Wait for error state
      await TestHelpers.verifyErrorHandling(page)
      
      const errorElement = page.locator('.error-state')
      if (await errorElement.isVisible()) {
        await expect(errorElement).toHaveScreenshot('error-state.png', {
          animations: 'disabled'
        })
      }
    })

    test('should match empty states', async ({ page }) => {
      // Mock empty data responses
      await page.route('**/api/v1/tasks', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([])
        })
      })
      
      await page.route('**/api/v1/agents', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([])
        })
      })
      
      await page.route('**/api/v1/events', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([])
        })
      })
      
      await dashboardPage.goto('/')
      await dashboardPage.waitForLoad()
      await dashboardPage.navigateToTasks()
      
      // Capture empty state
      const emptyState = page.locator('.empty-state, [data-testid="no-tasks"]')
      if (await emptyState.isVisible()) {
        await expect(emptyState).toHaveScreenshot('empty-state.png', {
          animations: 'disabled'
        })
      }
    })

    test('should match offline mode appearance', async ({ page }) => {
      await TestHelpers.simulateOfflineNetwork(page)
      
      await dashboardPage.goto('/')
      await dashboardPage.waitForLoad()
      
      // Wait for offline indicators
      await TestHelpers.waitForCondition(async () => {
        const offlineIndicator = page.locator('.offline-indicator, .sync-indicator.offline')
        return await offlineIndicator.isVisible()
      })
      
      await expect(page).toHaveScreenshot('dashboard-offline.png', {
        fullPage: true,
        animations: 'disabled'
      })
    })
  })

  test.describe('Modal and Overlay Screenshots', () => {
    test('should match task edit modal appearance', async ({ page }) => {
      await dashboardPage.navigateToTasks()
      
      // Open task edit modal
      const firstTask = page.locator('task-card').first()
      if (await firstTask.isVisible()) {
        await firstTask.click()
        
        const modal = page.locator('task-edit-modal')
        if (await modal.isVisible()) {
          await expect(modal).toHaveScreenshot('task-edit-modal.png', {
            animations: 'disabled'
          })
        }
      }
    })

    test('should match agent config modal appearance', async ({ page }) => {
      await dashboardPage.navigateToAgents()
      
      // Try to open agent config modal
      const configButton = page.locator('.config-button, [data-testid="configure-agent"]').first()
      if (await configButton.isVisible()) {
        await configButton.click()
        
        const modal = page.locator('agent-config-modal')
        if (await modal.isVisible()) {
          await expect(modal).toHaveScreenshot('agent-config-modal.png', {
            animations: 'disabled'
          })
        }
      }
    })

    test('should match notification appearances', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Look for notification center
      const notificationCenter = page.locator('notification-center')
      if (await notificationCenter.isVisible()) {
        await expect(notificationCenter).toHaveScreenshot('notification-center.png', {
          animations: 'disabled'
        })
      }
    })
  })

  test.describe('Theme and Style Variations', () => {
    test('should match dark mode appearance', async ({ page }) => {
      // Enable dark mode if supported
      await page.emulateMedia({ colorScheme: 'dark' })
      
      await dashboardPage.navigateToOverview()
      await TestHelpers.waitForAnimations(page)
      
      await expect(page).toHaveScreenshot('dashboard-dark-mode.png', {
        fullPage: true,
        animations: 'disabled'
      })
    })

    test('should match high contrast mode', async ({ page }) => {
      // Enable high contrast mode
      await page.emulateMedia({ colorScheme: 'dark' })
      await page.addStyleTag({
        content: `
          @media (prefers-contrast: high) {
            * {
              filter: contrast(1.5) !important;
            }
          }
        `
      })
      
      await dashboardPage.navigateToOverview()
      await TestHelpers.waitForAnimations(page)
      
      await expect(page).toHaveScreenshot('dashboard-high-contrast.png', {
        fullPage: true,
        animations: 'disabled'
      })
    })

    test('should match focus states', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Focus on first tab
      await dashboardPage.overviewTab.focus()
      
      // Screenshot focused element
      await expect(dashboardPage.overviewTab).toHaveScreenshot('focused-tab.png', {
        animations: 'disabled'
      })
    })
  })

  test.describe('Data Variation Screenshots', () => {
    test('should match dashboard with high task counts', async ({ page }) => {
      // Mock high task count data
      const manyTasks = Array.from({ length: 20 }, (_, i) => ({
        id: `task-${i}`,
        title: `Task ${i}`,
        description: `Description for task ${i}`,
        status: ['pending', 'in-progress', 'review', 'done'][i % 4],
        priority: ['high', 'medium', 'low'][i % 3],
        agent: `agent-${i % 3}`,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }))
      
      await page.route('**/api/v1/tasks', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(manyTasks)
        })
      })
      
      await dashboardPage.goto('/')
      await dashboardPage.waitForLoad()
      await dashboardPage.navigateToTasks()
      
      await expect(page).toHaveScreenshot('dashboard-many-tasks.png', {
        fullPage: true,
        animations: 'disabled'
      })
    })

    test('should match dashboard with various agent states', async ({ page }) => {
      // Already covered by standard mocks which include various agent states
      await dashboardPage.navigateToAgents()
      
      await expect(page).toHaveScreenshot('dashboard-agent-states.png', {
        fullPage: true,
        animations: 'disabled'
      })
    })

    test('should match dashboard with different priority distributions', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Screenshot showing priority distribution
      const summaryCards = page.locator('.overview-summary')
      await expect(summaryCards).toHaveScreenshot('priority-distribution.png', {
        animations: 'disabled'
      })
    })
  })
})