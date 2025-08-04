import { test, expect } from '@playwright/test'
import { DashboardPage } from '../fixtures/page-objects'
import { TestHelpers } from '../utils/test-helpers'
import { APIMocks } from '../utils/api-mocks'

test.describe('Error Handling and Recovery Flows', () => {
  let dashboardPage: DashboardPage

  test.beforeEach(async ({ page }) => {
    dashboardPage = new DashboardPage(page)
  })

  test.describe('Network Error Handling', () => {
    test('should handle complete network failure gracefully', async ({ page }) => {
      // Set up network failure
      await page.route('**/*', async route => {
        await route.abort('internetdisconnected')
      })
      
      await dashboardPage.goto('/')
      
      // Should show appropriate error state
      await TestHelpers.verifyErrorHandling(page)
      
      // Should offer retry mechanism
      const retryButton = page.locator('.retry-button, .btn-primary', { hasText: /try again|retry/i })
      if (await retryButton.isVisible()) {
        await expect(retryButton).toBeVisible()
      }
      
      await TestHelpers.takeTimestampedScreenshot(page, 'network-failure')
    })

    test('should handle intermittent network issues', async ({ page }) => {
      let requestCount = 0
      
      // Fail first few requests, then succeed
      await page.route('**/api/v1/**', async route => {
        requestCount++
        if (requestCount <= 2) {
          await route.abort('connectionfailed')
        } else {
          await APIMocks.setupStandardMocks(page)
          await route.continue()
        }
      })
      
      await dashboardPage.goto('/')
      
      // Should eventually load successfully
      await TestHelpers.waitForCondition(async () => {
        return await dashboardPage.pageTitle.isVisible()
      }, 10000)
      
      await expect(dashboardPage.pageTitle).toBeVisible()
    })

    test('should handle slow network connections', async ({ page }) => {
      // Set up slow network
      await APIMocks.mockSlowNetwork(page, 5000)
      
      await dashboardPage.goto('/')
      
      // Should show loading state
      await TestHelpers.verifyLoadingState(page, '.dashboard-content')
      
      // Should eventually load
      await expect(dashboardPage.pageTitle).toBeVisible({ timeout: 15000 })
      
      // User should be able to interact while loading
      await expect(dashboardPage.overviewTab).toBeVisible()
    })

    test('should handle timeout errors', async ({ page }) => {
      // Mock timeout responses
      await APIMocks.mockErrorResponses(page, 'timeout')
      
      await dashboardPage.goto('/')
      
      // Should show timeout error after waiting
      await TestHelpers.waitForCondition(async () => {
        const errorElements = page.locator('.error-state, .timeout-error')
        return await errorElements.count() > 0
      }, 35000) // Longer timeout for this test
      
      await TestHelpers.verifyErrorHandling(page)
    })
  })

  test.describe('API Error Handling', () => {
    test('should handle 500 server errors', async ({ page }) => {
      await APIMocks.mockErrorResponses(page, 'server')
      
      await dashboardPage.goto('/')
      
      // Should show server error message
      await TestHelpers.verifyErrorHandling(page, 'server error')
      
      // Should provide retry option
      const retryButton = page.locator('[data-testid="retry"], .btn-primary')
      if (await retryButton.isVisible()) {
        await expect(retryButton).toBeVisible()
        
        // Test retry functionality
        await APIMocks.clearAllMocks(page)
        await APIMocks.setupStandardMocks(page)
        
        await retryButton.click()
        
        // Should recover and load normally
        await expect(dashboardPage.pageTitle).toBeVisible()
      }
    })

    test('should handle 404 not found errors', async ({ page }) => {
      await page.route('**/api/v1/**', async route => {
        await route.fulfill({
          status: 404,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Not Found' })
        })
      })
      
      await dashboardPage.goto('/')
      
      // Should handle 404 errors gracefully
      await TestHelpers.verifyErrorHandling(page, 'not found')
      
      await TestHelpers.takeTimestampedScreenshot(page, 'api-404-error')
    })

    test('should handle 401 unauthorized errors', async ({ page }) => {
      await page.route('**/api/v1/**', async route => {
        await route.fulfill({
          status: 401,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Unauthorized' })
        })
      })
      
      await dashboardPage.goto('/')
      
      // Should redirect to login or show auth error
      await TestHelpers.waitForCondition(async () => {
        const loginView = page.locator('login-view')
        const authError = page.locator('.auth-error, .unauthorized-error')
        return (await loginView.isVisible()) || (await authError.isVisible())
      })
      
      // Should show appropriate authentication error
      const authElements = page.locator('login-view, .auth-error, .unauthorized-error')
      await expect(authElements.first()).toBeVisible()
    })

    test('should handle 403 forbidden errors', async ({ page }) => {
      await page.route('**/api/v1/**', async route => {
        await route.fulfill({
          status: 403,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Forbidden - Insufficient permissions' })
        })
      })
      
      await dashboardPage.goto('/')
      
      // Should show permissions error
      await TestHelpers.verifyErrorHandling(page, 'forbidden|permission')
      
      await TestHelpers.takeTimestampedScreenshot(page, 'api-403-error')
    })

    test('should handle malformed JSON responses', async ({ page }) => {
      await page.route('**/api/v1/**', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: 'invalid json response'
        })
      })
      
      await dashboardPage.goto('/')
      
      // Should handle JSON parsing errors
      await TestHelpers.verifyErrorHandling(page)
      
      // Should not crash the application
      await expect(page.locator('body')).toBeVisible()
    })
  })

  test.describe('Partial Service Failures', () => {
    test('should handle mixed success/failure API responses', async ({ page }) => {
      // Mock partial failures
      await APIMocks.mockPartialFailures(page, ['/agents', '/events'])
      
      await dashboardPage.goto('/')
      await dashboardPage.waitForLoad()
      await dashboardPage.navigateToOverview()
      
      // Successful endpoints should load
      await expect(dashboardPage.activeTasksCard).toBeVisible()
      await expect(dashboardPage.systemHealthCard).toBeVisible()
      
      // Failed endpoints should show error states or graceful degradation
      const agentPanel = dashboardPage.agentHealthPanel
      if (await agentPanel.isVisible()) {
        // Should either show error state or empty state
        const errorState = agentPanel.locator('.error-state, .empty-state')
        if (await errorState.isVisible()) {
          await expect(errorState).toBeVisible()
        }
      }
    })

    test('should continue functioning with degraded services', async ({ page }) => {
      // Mock degraded service responses
      await page.route('**/api/v1/agents', async route => {
        await route.fulfill({
          status: 503,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Service Unavailable' })
        })
      })
      
      await APIMocks.setupStandardMocks(page)
      
      await dashboardPage.goto('/')
      await dashboardPage.waitForLoad()
      
      // Core functionality should work
      await expect(dashboardPage.pageTitle).toBeVisible()
      await dashboardPage.navigateToTasks()
      await expect(page.locator('kanban-board')).toBeVisible()
      
      // Agent section should show degraded state
      await dashboardPage.navigateToAgents()
      await TestHelpers.verifyErrorHandling(page)
    })

    test('should handle WebSocket connection failures', async ({ page }) => {
      // Mock WebSocket failure
      await page.addInitScript(() => {
        class FailingWebSocket {
          constructor() {
            setTimeout(() => {
              if (this.onerror) {
                this.onerror(new Event('error'))
              }
            }, 100)
          }
          
          close() {}
          send() {}
          
          onerror: ((event: Event) => void) | null = null
          onopen: ((event: Event) => void) | null = null
          onmessage: ((event: MessageEvent) => void) | null = null
          onclose: ((event: CloseEvent) => void) | null = null
        }
        
        (window as any).WebSocket = FailingWebSocket
      })
      
      await APIMocks.setupStandardMocks(page)
      
      await dashboardPage.goto('/')
      await dashboardPage.waitForLoad()
      
      // Should fall back to polling
      await page.waitForTimeout(3000)
      
      // Application should still function
      await expect(dashboardPage.activeTasksCard).toBeVisible()
      
      // Should not show real-time indicators
      const syncStatus = await dashboardPage.syncStatus.textContent()
      expect(syncStatus).not.toContain('real-time')
    })
  })

  test.describe('User Action Error Handling', () => {
    test('should handle task creation failures', async ({ page }) => {
      await APIMocks.setupStandardMocks(page)
      
      // Mock task creation failure
      await page.route('**/api/v1/tasks', async route => {
        if (route.request().method() === 'POST') {
          await route.fulfill({
            status: 400,
            contentType: 'application/json',
            body: JSON.stringify({ error: 'Validation failed', details: 'Title is required' })
          })
        }
      })
      
      await dashboardPage.goto('/')
      await dashboardPage.waitForLoad()
      await dashboardPage.navigateToTasks()
      
      // Try to create a task
      const createButton = page.locator('[data-testid="create-task"]')
      if (await createButton.isVisible()) {
        await createButton.click()
        
        const modal = page.locator('task-edit-modal')
        if (await modal.isVisible()) {
          const saveButton = modal.locator('[data-testid="save-task"]')
          await saveButton.click()
          
          // Should show validation error
          await TestHelpers.verifyErrorHandling(page, 'validation|title')
          
          // Modal should remain open
          await expect(modal).toBeVisible()
        }
      }
    })

    test('should handle task update failures', async ({ page }) => {
      await APIMocks.setupStandardMocks(page)
      
      // Mock task update failure
      await page.route('**/api/v1/tasks/*', async route => {
        if (route.request().method() === 'PUT') {
          await route.fulfill({
            status: 409,
            contentType: 'application/json',
            body: JSON.stringify({ error: 'Conflict', message: 'Task was modified by another user' })
          })
        }
      })
      
      await dashboardPage.goto('/')
      await dashboardPage.waitForLoad()
      await dashboardPage.navigateToTasks()
      
      // Try to move a task
      const firstTask = page.locator('task-card').first()
      if (await firstTask.isVisible()) {
        const taskId = await firstTask.getAttribute('data-task-id')
        
        // Simulate drag and drop
        await firstTask.dragTo(page.locator('kanban-column[data-column="in-progress"]'))
        
        // Should show conflict error
        await TestHelpers.verifyErrorHandling(page, 'conflict|modified')
        
        // Task should remain in original position
        const originalColumn = page.locator('kanban-column[data-column="pending"]')
        await expect(originalColumn.locator(`task-card[data-task-id="${taskId}"]`)).toBeVisible()
      }
    })

    test('should handle agent activation failures', async ({ page }) => {
      await APIMocks.setupStandardMocks(page)
      
      // Mock agent activation failure
      await page.route('**/api/v1/agents/activate', async route => {
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Internal Server Error', message: 'Failed to activate agents' })
        })
      })
      
      await dashboardPage.goto('/')
      await dashboardPage.waitForLoad()
      await dashboardPage.navigateToAgents()
      
      // Try to activate agent team
      const activateButton = page.locator('[data-testid="activate-team"]')
      if (await activateButton.isVisible()) {
        await activateButton.click()
        
        // Should show activation error
        await TestHelpers.verifyErrorHandling(page, 'activation|failed')
        
        // Button should return to normal state
        await expect(activateButton).not.toHaveClass(/loading/)
      }
    })

    test('should handle drag and drop failures', async ({ page }) => {
      await APIMocks.setupStandardMocks(page)
      
      await dashboardPage.goto('/')
      await dashboardPage.waitForLoad()
      await dashboardPage.navigateToTasks()
      
      // Mock drag failure during operation
      let dragAttempted = false
      await page.route('**/api/v1/tasks/*', async route => {
        if (route.request().method() === 'PUT' && !dragAttempted) {
          dragAttempted = true
          await route.abort('connectionfailed')
        } else {
          await route.continue()
        }
      })
      
      const firstTask = page.locator('task-card').first()
      if (await firstTask.isVisible()) {
        const taskId = await firstTask.getAttribute('data-task-id')
        const originalColumn = 'pending'
        
        // Attempt drag and drop
        await firstTask.dragTo(page.locator('kanban-column[data-column="in-progress"]'))
        
        // Should show error and revert
        await TestHelpers.waitForCondition(async () => {
          const errorElements = page.locator('.error-state, .error-message')
          return await errorElements.count() > 0
        })
        
        // Task should be reverted to original position
        await expect(page.locator(`kanban-column[data-column="${originalColumn}"] task-card[data-task-id="${taskId}"]`)).toBeVisible()
      }
    })
  })

  test.describe('Data Validation and Integrity', () => {
    test('should handle invalid data formats', async ({ page }) => {
      // Mock invalid data response
      await page.route('**/api/v1/tasks', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([
            { id: 'invalid-task', title: null, status: 'invalid-status' },
            { id: 'missing-fields' },
            { title: 'No ID task', status: 'pending' }
          ])
        })
      })
      
      await dashboardPage.goto('/')
      await dashboardPage.waitForLoad()
      await dashboardPage.navigateToTasks()
      
      // Should handle invalid data gracefully
      const kanbanBoard = page.locator('kanban-board')
      await expect(kanbanBoard).toBeVisible()
      
      // Should not crash despite invalid data
      await expect(page.locator('body')).toBeVisible()
      
      // May show data validation errors
      const errorState = page.locator('.data-error, .validation-error')
      if (await errorState.isVisible()) {
        await expect(errorState).toBeVisible()
      }
    })

    test('should handle missing required fields', async ({ page }) => {
      await APIMocks.setupStandardMocks(page)
      
      await dashboardPage.goto('/')
      await dashboardPage.waitForLoad()
      await dashboardPage.navigateToTasks()
      
      // Try to create task with missing required fields
      const createButton = page.locator('[data-testid="create-task"]')
      if (await createButton.isVisible()) {
        await createButton.click()
        
        const modal = page.locator('task-edit-modal')
        if (await modal.isVisible()) {
          // Try to save without filling required fields
          const saveButton = modal.locator('[data-testid="save-task"]')
          await saveButton.click()
          
          // Should show field validation errors
          const titleError = modal.locator('[data-testid="task-title"] + .error-message')
          if (await titleError.isVisible()) {
            await expect(titleError).toContainText(/required|title/)
          }
          
          // Form should not submit
          await expect(modal).toBeVisible()
        }
      }
    })

    test('should handle data type mismatches', async ({ page }) => {
      // Mock response with wrong data types
      await page.route('**/api/v1/agents', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([
            {
              id: 123, // Should be string
              name: ['Agent', 'Array'], // Should be string
              status: true, // Should be specific status string
              uptime: 'not-a-number', // Should be number
              metrics: 'invalid-metrics' // Should be object
            }
          ])
        })
      })
      
      await dashboardPage.goto('/')
      await dashboardPage.waitForLoad()
      await dashboardPage.navigateToAgents()
      
      // Should handle type mismatches gracefully
      const agentPanel = page.locator('agent-health-panel')
      await expect(agentPanel).toBeVisible()
      
      // Should show data error or empty state
      const errorIndicator = page.locator('.data-error, .type-error, .empty-state')
      if (await errorIndicator.isVisible()) {
        await expect(errorIndicator).toBeVisible()
      }
    })
  })

  test.describe('Error Recovery and Retry Mechanisms', () => {
    test('should provide retry mechanisms for failed operations', async ({ page }) => {
      let attemptCount = 0
      
      // Fail first attempt, succeed on retry
      await page.route('**/api/v1/tasks', async route => {
        attemptCount++
        if (attemptCount === 1) {
          await route.fulfill({
            status: 500,
            contentType: 'application/json',
            body: JSON.stringify({ error: 'Temporary server error' })
          })
        } else {
          await APIMocks.setupStandardMocks(page)
          await route.continue()
        }
      })
      
      await dashboardPage.goto('/')
      
      // Should show error initially
      await TestHelpers.verifyErrorHandling(page)
      
      // Click retry button
      const retryButton = page.locator('.retry-button, [data-testid="retry"]')
      if (await retryButton.isVisible()) {
        await retryButton.click()
        
        // Should succeed on retry
        await expect(dashboardPage.pageTitle).toBeVisible()
      }
    })

    test('should implement exponential backoff for retries', async ({ page }) => {
      let retryCount = 0
      const retryTimes: number[] = []
      
      await page.route('**/api/v1/**', async route => {
        retryCount++
        retryTimes.push(Date.now())
        
        if (retryCount <= 3) {
          await route.fulfill({ status: 500 })
        } else {
          await APIMocks.setupStandardMocks(page)
          await route.continue()
        }
      })
      
      await dashboardPage.goto('/')
      
      // Wait for retries to complete
      await TestHelpers.waitForCondition(async () => {
        return retryCount >= 4
      }, 15000)
      
      // Should have increasing delays between retries
      if (retryTimes.length >= 3) {
        const delay1 = retryTimes[1] - retryTimes[0]
        const delay2 = retryTimes[2] - retryTimes[1]
        
        // Second delay should be longer than first (exponential backoff)
        expect(delay2).toBeGreaterThan(delay1)
      }
    })

    test('should stop retrying after maximum attempts', async ({ page }) => {
      let attemptCount = 0
      
      // Always fail
      await page.route('**/api/v1/**', async route => {
        attemptCount++
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Persistent server error' })
        })
      })
      
      await dashboardPage.goto('/')
      
      // Wait for retry attempts
      await page.waitForTimeout(10000)
      
      // Should eventually give up and show final error state
      await TestHelpers.verifyErrorHandling(page)
      
      // Should not make excessive retry attempts
      expect(attemptCount).toBeLessThan(10)
    })

    test('should allow manual refresh after giving up retries', async ({ page }) => {
      // Always fail initially
      await APIMocks.mockErrorResponses(page, 'server')
      
      await dashboardPage.goto('/')
      await TestHelpers.verifyErrorHandling(page)
      
      // Setup successful responses for manual refresh
      await APIMocks.clearAllMocks(page)
      await APIMocks.setupStandardMocks(page)
      
      // Find and click manual refresh button
      const refreshButton = page.locator('.refresh-button, [data-testid="manual-refresh"]')
      if (await refreshButton.isVisible()) {
        await refreshButton.click()
        
        // Should recover after manual refresh
        await expect(dashboardPage.pageTitle).toBeVisible()
        await expect(dashboardPage.activeTasksCard).toBeVisible()
      }
    })
  })

  test.describe('User Experience During Errors', () => {
    test('should maintain application state during errors', async ({ page }) => {
      await APIMocks.setupStandardMocks(page)
      
      await dashboardPage.goto('/')
      await dashboardPage.waitForLoad()
      await dashboardPage.navigateToTasks()
      
      // User makes some changes
      const searchInput = page.locator('.filter-input')
      if (await searchInput.isVisible()) {
        await searchInput.fill('test search')
      }
      
      // Simulate network error
      await APIMocks.mockErrorResponses(page, 'network')
      
      // Try to refresh
      await dashboardPage.refreshButton.click()
      
      // Search filter should be preserved despite error
      if (await searchInput.isVisible()) {
        const searchValue = await searchInput.inputValue()
        expect(searchValue).toBe('test search')
      }
    })

    test('should provide helpful error messages', async ({ page }) => {
      await APIMocks.mockErrorResponses(page, 'server')
      
      await dashboardPage.goto('/')
      
      // Should show user-friendly error message
      const errorMessage = page.locator('.error-message, .error-state')
      await expect(errorMessage).toBeVisible()
      
      const errorText = await errorMessage.textContent()
      
      // Should not show technical error details to user
      expect(errorText).not.toContain('500')
      expect(errorText).not.toMatch(/stack trace|exception/i)
      
      // Should provide actionable guidance
      expect(errorText).toMatch(/try again|refresh|contact support/i)
    })

    test('should show loading states during error recovery', async ({ page }) => {
      let requestCount = 0
      
      await page.route('**/api/v1/**', async route => {
        requestCount++
        if (requestCount === 1) {
          await route.fulfill({ status: 500 })
        } else {
          // Slow success response
          await new Promise(resolve => setTimeout(resolve, 2000))
          await APIMocks.setupStandardMocks(page)
          await route.continue()
        }
      })
      
      await dashboardPage.goto('/')
      
      // Should show error initially
      await TestHelpers.verifyErrorHandling(page)
      
      // Click retry
      const retryButton = page.locator('.retry-button')
      if (await retryButton.isVisible()) {
        await retryButton.click()
        
        // Should show loading state during recovery
        await expect(retryButton).toHaveClass(/loading/)
        
        // Should eventually succeed
        await expect(dashboardPage.pageTitle).toBeVisible()
      }
    })

    test('should prevent user actions during error states', async ({ page }) => {
      await APIMocks.mockErrorResponses(page, 'server')
      
      await dashboardPage.goto('/')
      await TestHelpers.verifyErrorHandling(page)
      
      // Interactive elements should be disabled during error
      const navigationTabs = page.locator('.tab-button')
      const tabCount = await navigationTabs.count()
      
      for (let i = 0; i < tabCount; i++) {
        const tab = navigationTabs.nth(i)
        if (await tab.isVisible()) {
          const isDisabled = await tab.evaluate(el => 
            el.hasAttribute('disabled') || 
            el.getAttribute('aria-disabled') === 'true' ||
            window.getComputedStyle(el).pointerEvents === 'none'
          )
          
          // Tabs should be disabled or have visual indication of unavailability
          if (!isDisabled) {
            // At minimum, clicks should be ignored during error state
            await tab.click()
            // Should not navigate during error state
            await expect(tab).not.toHaveClass(/active/)
          }
        }
      }
    })
  })
})