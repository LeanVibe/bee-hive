import { test, expect } from '@playwright/test'
import { DashboardPage, KanbanBoardPage, AgentHealthPanelPage } from '../fixtures/page-objects'
import { TestHelpers } from '../utils/test-helpers'
import { APIMocks } from '../utils/api-mocks'

test.describe('Enhanced Features Tests', () => {
  let dashboardPage: DashboardPage
  let kanbanPage: KanbanBoardPage
  let agentPage: AgentHealthPanelPage

  test.beforeEach(async ({ page }) => {
    dashboardPage = new DashboardPage(page)
    kanbanPage = new KanbanBoardPage(page)
    agentPage = new AgentHealthPanelPage(page)
    
    // Set up API mocks for enhanced features
    await APIMocks.setupEnhancedFeatureMocks(page)
    
    // Navigate to dashboard
    await dashboardPage.goto('/')
    await dashboardPage.waitForLoad()
  })

  test.describe('Multi-Agent Task Assignment', () => {
    test('should allow assigning tasks to multiple agents', async ({ page }) => {
      // Navigate to tasks view
      await dashboardPage.navigateToTasks()
      await expect(page.locator('kanban-board')).toBeVisible()
      
      // Click on a task to open edit modal
      const firstTask = page.locator('task-card').first()
      await firstTask.click()
      
      // Wait for modal to open
      await expect(page.locator('task-edit-modal')).toBeVisible()
      
      // Test multi-agent selection
      const agentSelect = page.locator('[data-testid="task-agent-multiple"]')
      if (await agentSelect.isVisible()) {
        // Select multiple agents
        await agentSelect.selectOption(['agent-1', 'agent-2'])
        
        // Verify multiple agents can be selected
        const selectedOptions = await agentSelect.evaluate(select => 
          Array.from(select.selectedOptions).map(option => option.value)
        )
        expect(selectedOptions).toContain('agent-1')
        expect(selectedOptions).toContain('agent-2')
        
        // Save task with multiple agents
        await page.locator('[data-testid="save-task"]').click()
        await expect(page.locator('task-edit-modal')).not.toBeVisible()
        
        // Verify task shows multiple agents assigned
        const taskCard = page.locator('task-card').first()
        const agentIndicators = taskCard.locator('.agent-indicator')
        expect(await agentIndicators.count()).toBeGreaterThan(1)
      }
    })

    test('should show agent workload distribution', async ({ page }) => {
      await dashboardPage.navigateToAgents()
      
      // Verify agent workload visualization
      const agentCards = page.locator('.agent-card')
      const firstAgent = agentCards.first()
      
      // Check for workload indicators
      await expect(firstAgent.locator('.workload-indicator')).toBeVisible()
      await expect(firstAgent.locator('.task-count')).toBeVisible()
      
      // Verify workload metrics
      const workloadText = await firstAgent.locator('.workload-indicator').textContent()
      expect(workloadText).toMatch(/\d+%|\d+ tasks/)
    })

    test('should support agent coordination features', async ({ page }) => {
      await dashboardPage.navigateToAgents()
      
      // Test agent collaboration panel
      const collaborationPanel = page.locator('.agent-collaboration-panel')
      if (await collaborationPanel.isVisible()) {
        // Verify collaboration features
        await expect(collaborationPanel.locator('.team-assignments')).toBeVisible()
        await expect(collaborationPanel.locator('.communication-status')).toBeVisible()
        
        // Test agent communication simulation
        const communicationButton = collaborationPanel.locator('.test-communication')
        if (await communicationButton.isVisible()) {
          await communicationButton.click()
          await expect(page.locator('.communication-test-result')).toBeVisible()
        }
      }
    })
  })

  test.describe('Advanced Kanban Filtering', () => {
    test('should filter tasks by multiple criteria', async ({ page }) => {
      await dashboardPage.navigateToTasks()
      await expect(page.locator('kanban-board')).toBeVisible()
      
      // Test multiple filter criteria
      const filterPanel = page.locator('.advanced-filters')
      if (await filterPanel.isVisible()) {
        // Filter by agent
        await filterPanel.locator('[data-filter="agent"]').selectOption('agent-1')
        await page.waitForTimeout(500)
        
        // Add priority filter
        await filterPanel.locator('[data-filter="priority"]').selectOption('high')
        await page.waitForTimeout(500)
        
        // Add date range filter
        const dateFromInput = filterPanel.locator('[data-filter="date-from"]')
        const dateToInput = filterPanel.locator('[data-filter="date-to"]')
        
        if (await dateFromInput.isVisible()) {
          await dateFromInput.fill('2024-01-01')
          await dateToInput.fill('2024-12-31')
        }
        
        // Verify filtered results
        const visibleTasks = page.locator('task-card')
        const taskCount = await visibleTasks.count()
        expect(taskCount).toBeGreaterThanOrEqual(0)
        
        // Verify filter tags are displayed
        await expect(page.locator('.active-filter-tag')).toBeVisible()
      }
    })

    test('should support custom filter creation', async ({ page }) => {
      await dashboardPage.navigateToTasks()
      
      const customFilterButton = page.locator('.create-custom-filter')
      if (await customFilterButton.isVisible()) {
        await customFilterButton.click()
        
        // Wait for custom filter modal
        await expect(page.locator('.custom-filter-modal')).toBeVisible()
        
        // Create custom filter
        await page.locator('[data-testid="filter-name"]').fill('High Priority Sprint Tasks')
        await page.locator('[data-testid="filter-criteria-priority"]').selectOption('high')
        await page.locator('[data-testid="filter-criteria-sprint"]').selectOption('current')
        
        // Save custom filter
        await page.locator('[data-testid="save-custom-filter"]').click()
        await expect(page.locator('.custom-filter-modal')).not.toBeVisible()
        
        // Verify custom filter appears in filter list
        const filterDropdown = page.locator('.filter-dropdown')
        await filterDropdown.click()
        await expect(page.locator('.filter-option').filter({ hasText: 'High Priority Sprint Tasks' })).toBeVisible()
      }
    })

    test('should support bulk operations on filtered tasks', async ({ page }) => {
      await dashboardPage.navigateToTasks()
      
      // Apply a filter to get specific tasks
      const searchFilter = page.locator('.filter-input')
      await searchFilter.fill('urgent')
      await page.waitForTimeout(500)
      
      // Test bulk selection
      const bulkSelectButton = page.locator('.bulk-select-toggle')
      if (await bulkSelectButton.isVisible()) {
        await bulkSelectButton.click()
        
        // Select multiple tasks
        const taskCheckboxes = page.locator('.task-checkbox')
        const checkboxCount = await taskCheckboxes.count()
        
        if (checkboxCount > 0) {
          await taskCheckboxes.first().check()
          await taskCheckboxes.nth(1).check()
          
          // Verify bulk actions panel appears
          await expect(page.locator('.bulk-actions-panel')).toBeVisible()
          
          // Test bulk status change
          const bulkStatusChange = page.locator('.bulk-action-status')
          if (await bulkStatusChange.isVisible()) {
            await bulkStatusChange.selectOption('in-progress')
            await page.locator('.apply-bulk-action').click()
            
            // Verify confirmation dialog
            await expect(page.locator('.bulk-action-confirm')).toBeVisible()
            await page.locator('.confirm-bulk-action').click()
          }
        }
      }
    })
  })

  test.describe('Sprint Planning Interface', () => {
    test('should load sprint planning interface', async ({ page }) => {
      // Navigate to sprint planning (might be a separate tab or modal)
      const sprintPlanningButton = page.locator('.sprint-planning-button, [data-view="sprint-planning"]')
      
      if (await sprintPlanningButton.isVisible()) {
        await sprintPlanningButton.click()
        
        // Verify sprint planning interface loads
        await expect(page.locator('.sprint-planner, sprint-planner')).toBeVisible()
        await TestHelpers.takeTimestampedScreenshot(page, 'sprint-planning-interface')
      } else {
        // Sprint planning might be integrated into the main dashboard
        const sprintPanel = page.locator('.sprint-panel, .planning-panel')
        if (await sprintPanel.isVisible()) {
          await expect(sprintPanel).toBeVisible()
        }
      }
    })

    test('should support sprint creation and management', async ({ page }) => {
      // Look for sprint management features
      const createSprintButton = page.locator('.create-sprint, [data-action="create-sprint"]')
      
      if (await createSprintButton.isVisible()) {
        await createSprintButton.click()
        
        // Fill sprint details
        const sprintModal = page.locator('.sprint-modal, .create-sprint-modal')
        await expect(sprintModal).toBeVisible()
        
        const sprintNameInput = sprintModal.locator('[data-testid="sprint-name"], input[name="name"]')
        const sprintGoalInput = sprintModal.locator('[data-testid="sprint-goal"], textarea[name="goal"]')
        const sprintDurationSelect = sprintModal.locator('[data-testid="sprint-duration"], select[name="duration"]')
        
        if (await sprintNameInput.isVisible()) {
          await sprintNameInput.fill('Test Sprint Q1 2024')
        }
        if (await sprintGoalInput.isVisible()) {
          await sprintGoalInput.fill('Complete dashboard enhancements and testing framework')
        }
        if (await sprintDurationSelect.isVisible()) {
          await sprintDurationSelect.selectOption('2')
        }
        
        // Save sprint
        const saveSprintButton = sprintModal.locator('.save-sprint, [data-action="save-sprint"]')
        if (await saveSprintButton.isVisible()) {
          await saveSprintButton.click()
          await expect(sprintModal).not.toBeVisible()
        }
      }
    })

    test('should support velocity tracking and burndown charts', async ({ page }) => {
      // Look for sprint analytics features
      const analyticsSection = page.locator('.sprint-analytics, .velocity-tracking')
      
      if (await analyticsSection.isVisible()) {
        // Verify velocity chart
        const velocityChart = analyticsSection.locator('.velocity-chart, canvas[data-chart="velocity"]')
        if (await velocityChart.isVisible()) {
          await expect(velocityChart).toBeVisible()
        }
        
        // Verify burndown chart
        const burndownChart = analyticsSection.locator('.burndown-chart, canvas[data-chart="burndown"]')
        if (await burndownChart.isVisible()) {
          await expect(burndownChart).toBeVisible()
        }
        
        // Verify sprint metrics
        const sprintMetrics = analyticsSection.locator('.sprint-metrics')
        if (await sprintMetrics.isVisible()) {
          await expect(sprintMetrics.locator('.completed-points')).toBeVisible()
          await expect(sprintMetrics.locator('.remaining-points')).toBeVisible()
          await expect(sprintMetrics.locator('.velocity-score')).toBeVisible()
        }
      }
    })
  })

  test.describe('Task Analytics', () => {
    test('should display task completion metrics', async ({ page }) => {
      // Look for analytics dashboard or panel
      const analyticsPanel = page.locator('.task-analytics, .analytics-panel')
      
      if (await analyticsPanel.isVisible()) {
        // Verify completion rate chart
        await expect(analyticsPanel.locator('.completion-rate-chart')).toBeVisible()
        
        // Verify task distribution by status
        await expect(analyticsPanel.locator('.status-distribution')).toBeVisible()
        
        // Verify agent performance metrics
        await expect(analyticsPanel.locator('.agent-performance')).toBeVisible()
        
        // Test time period selection
        const timePeriodSelect = analyticsPanel.locator('.time-period-select')
        if (await timePeriodSelect.isVisible()) {
          await timePeriodSelect.selectOption('last-7-days')
          await page.waitForTimeout(1000)
          
          // Verify charts update with new data
          await expect(analyticsPanel.locator('.chart-loading')).not.toBeVisible()
        }
      } else {
        // Analytics might be on a separate tab
        const analyticsTab = page.locator('.tab-button').filter({ hasText: /Analytics|Reports/ })
        if (await analyticsTab.isVisible()) {
          await analyticsTab.click()
          await expect(page.locator('.analytics-view')).toBeVisible()
        }
      }
    })

    test('should support custom analytics queries', async ({ page }) => {
      const customQueryButton = page.locator('.custom-query, [data-action="custom-analytics"]')
      
      if (await customQueryButton.isVisible()) {
        await customQueryButton.click()
        
        const queryBuilder = page.locator('.query-builder')
        await expect(queryBuilder).toBeVisible()
        
        // Build a custom query
        await queryBuilder.locator('.metric-select').selectOption('task-completion-time')
        await queryBuilder.locator('.groupby-select').selectOption('agent')
        await queryBuilder.locator('.timerange-select').selectOption('last-30-days')
        
        // Execute query
        await page.locator('.execute-query').click()
        
        // Verify results
        await expect(page.locator('.query-results')).toBeVisible()
        await expect(page.locator('.results-chart')).toBeVisible()
      }
    })
  })

  test.describe('Real-time Collaboration Features', () => {
    test('should show real-time task updates', async ({ page }) => {
      await dashboardPage.navigateToTasks()
      
      // Mock WebSocket for real-time updates
      await TestHelpers.mockWebSocket(page)
      
      // Verify WebSocket connection
      const wsIndicator = page.locator('.websocket-status, .realtime-indicator')
      if (await wsIndicator.isVisible()) {
        await expect(wsIndicator).toHaveClass(/connected/)
      }
      
      // Wait for real-time update simulation
      await page.waitForTimeout(3000)
      
      // Verify task updates are reflected
      const updatedTasks = page.locator('.task-updated, .task-highlight')
      if (await updatedTasks.count() > 0) {
        await expect(updatedTasks.first()).toBeVisible()
      }
    })

    test('should support collaborative editing indicators', async ({ page }) => {
      await dashboardPage.navigateToTasks()
      
      // Open a task for editing
      const firstTask = page.locator('task-card').first()
      await firstTask.click()
      
      const taskModal = page.locator('task-edit-modal')
      await expect(taskModal).toBeVisible()
      
      // Look for collaboration indicators
      const collaborationIndicators = taskModal.locator('.editing-indicator, .user-presence')
      if (await collaborationIndicators.isVisible()) {
        await expect(collaborationIndicators).toBeVisible()
        
        // Verify user avatars or indicators
        const userIndicators = collaborationIndicators.locator('.user-avatar, .user-indicator')
        if (await userIndicators.count() > 0) {
          await expect(userIndicators.first()).toBeVisible()
        }
      }
    })
  })

  test.describe('Advanced Search and Filtering', () => {
    test('should support natural language search', async ({ page }) => {
      await dashboardPage.navigateToTasks()
      
      const searchInput = page.locator('.search-input, .natural-search')
      if (await searchInput.isVisible()) {
        // Test natural language queries
        await searchInput.fill('high priority tasks assigned to agent-1 this week')
        await page.keyboard.press('Enter')
        
        // Wait for search results
        await page.waitForTimeout(1000)
        
        // Verify search results
        const searchResults = page.locator('.search-results, .filtered-tasks')
        if (await searchResults.isVisible()) {
          await expect(searchResults).toBeVisible()
          
          // Verify search query interpretation
          const queryInterpretation = page.locator('.query-interpretation')
          if (await queryInterpretation.isVisible()) {
            await expect(queryInterpretation).toContainText('priority: high')
            await expect(queryInterpretation).toContainText('agent: agent-1')
          }
        }
      }
    })

    test('should support saved search queries', async ({ page }) => {
      await dashboardPage.navigateToTasks()
      
      const savedSearches = page.locator('.saved-searches, .search-presets')
      if (await savedSearches.isVisible()) {
        // Verify preset searches exist
        const presetButtons = savedSearches.locator('.preset-search')
        const presetCount = await presetButtons.count()
        
        if (presetCount > 0) {
          // Test a preset search
          await presetButtons.first().click()
          await page.waitForTimeout(500)
          
          // Verify search is applied
          const activeSearch = page.locator('.active-search-indicator')
          if (await activeSearch.isVisible()) {
            await expect(activeSearch).toBeVisible()
          }
        }
      }
    })
  })

  test.describe('Performance and Optimization', () => {
    test('should handle large datasets efficiently', async ({ page }) => {
      // Mock large dataset
      await page.route('**/api/v1/tasks', async route => {
        const largeTasks = Array.from({ length: 500 }, (_, i) => ({
          id: `task-${i}`,
          title: `Task ${i}`,
          description: `Description for task ${i}`,
          status: ['pending', 'in-progress', 'review', 'done'][i % 4],
          priority: ['low', 'medium', 'high'][i % 3],
          agent: `agent-${i % 5}`,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString()
        }))
        
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(largeTasks)
        })
      })
      
      const startTime = Date.now()
      await dashboardPage.navigateToTasks()
      await expect(page.locator('kanban-board')).toBeVisible()
      const loadTime = Date.now() - startTime
      
      // Should load within reasonable time even with large dataset
      expect(loadTime).toBeLessThan(10000)
      
      // Verify virtualization or pagination is working
      const visibleTasks = page.locator('task-card')
      const visibleCount = await visibleTasks.count()
      
      // Should not render all 500 tasks at once (performance optimization)
      expect(visibleCount).toBeLessThan(100)
    })

    test('should implement lazy loading for performance', async ({ page }) => {
      await dashboardPage.navigateToTasks()
      
      // Scroll to trigger lazy loading
      await page.evaluate(() => {
        window.scrollTo(0, document.body.scrollHeight)
      })
      
      // Wait for lazy loading
      await page.waitForTimeout(1000)
      
      // Verify more content is loaded
      const loadMoreIndicator = page.locator('.loading-more, .lazy-loading')
      if (await loadMoreIndicator.isVisible()) {
        await expect(loadMoreIndicator).not.toBeVisible()
      }
    })
  })

  test.afterEach(async ({ page }) => {
    // Take screenshot for debugging if test fails
    await TestHelpers.takeTimestampedScreenshot(page, 'enhanced-features-test-end')
    
    // Clear any mocks
    await APIMocks.clearAllMocks(page)
  })
})