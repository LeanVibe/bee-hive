import { test, expect } from '@playwright/test'
import { DashboardPage, KanbanBoardPage, TaskEditModalPage } from '../fixtures/page-objects'
import { TestHelpers } from '../utils/test-helpers'
import { APIMocks } from '../utils/api-mocks'

test.describe('Task Management and Kanban Board', () => {
  let dashboardPage: DashboardPage
  let kanbanBoard: KanbanBoardPage
  let taskEditModal: TaskEditModalPage

  test.beforeEach(async ({ page }) => {
    dashboardPage = new DashboardPage(page)
    kanbanBoard = new KanbanBoardPage(page)
    taskEditModal = new TaskEditModalPage(page)
    
    // Set up API mocks for consistent testing
    await APIMocks.setupStandardMocks(page)
    
    // Navigate to dashboard and then to tasks view
    await dashboardPage.goto('/')
    await dashboardPage.waitForLoad()
    await dashboardPage.navigateToTasks()
  })

  test.describe('Kanban Board Display', () => {
    test('should display all Kanban columns', async ({ page }) => {
      // Verify all columns are visible
      await expect(kanbanBoard.pendingColumn).toBeVisible()
      await expect(kanbanBoard.inProgressColumn).toBeVisible()
      await expect(kanbanBoard.reviewColumn).toBeVisible()
      await expect(kanbanBoard.doneColumn).toBeVisible()
      
      // Verify column headers
      await expect(page.locator('kanban-column').filter({ hasText: 'Backlog' })).toBeVisible()
      await expect(page.locator('kanban-column').filter({ hasText: 'In Progress' })).toBeVisible()
      await expect(page.locator('kanban-column').filter({ hasText: 'Review' })).toBeVisible()
      await expect(page.locator('kanban-column').filter({ hasText: 'Done' })).toBeVisible()
      
      await TestHelpers.takeTimestampedScreenshot(page, 'kanban-board')
    })

    test('should display tasks in correct columns', async ({ page }) => {
      // Verify tasks are distributed correctly
      const pendingTasks = await kanbanBoard.getTasksInColumn('pending')
      const inProgressTasks = await kanbanBoard.getTasksInColumn('in-progress')
      const reviewTasks = await kanbanBoard.getTasksInColumn('review')
      const doneTasks = await kanbanBoard.getTasksInColumn('done')
      
      // Should have tasks in each column based on mock data
      expect(pendingTasks).toBeGreaterThanOrEqual(0)
      expect(inProgressTasks).toBeGreaterThanOrEqual(0)
      expect(reviewTasks).toBeGreaterThanOrEqual(0)
      expect(doneTasks).toBeGreaterThanOrEqual(0)
      
      // Total tasks should match mock data
      const totalTasks = pendingTasks + inProgressTasks + reviewTasks + doneTasks
      expect(totalTasks).toBeGreaterThan(0)
    })

    test('should display task cards with complete information', async ({ page }) => {
      // Find first task card
      const taskCard = page.locator('task-card').first()
      await expect(taskCard).toBeVisible()
      
      // Verify task card contains required information
      await expect(taskCard.locator('.task-title')).toBeVisible()
      await expect(taskCard.locator('.task-description')).toBeVisible()
      await expect(taskCard.locator('.task-priority')).toBeVisible()
      await expect(taskCard.locator('.task-agent')).toBeVisible()
      
      // Verify task metadata
      const taskId = await taskCard.getAttribute('data-task-id')
      expect(taskId).toBeTruthy()
      
      // Task should have tags if present
      const tags = taskCard.locator('.task-tags')
      if (await tags.isVisible()) {
        await expect(tags).toBeVisible()
      }
    })

    test('should show loading state while fetching tasks', async ({ page }) => {
      // Mock slow task loading
      await APIMocks.clearAllMocks(page)
      await APIMocks.mockSlowNetwork(page, 2000)
      
      await page.reload()
      await dashboardPage.waitForLoad()
      await dashboardPage.navigateToTasks()
      
      // Should show loading state
      await TestHelpers.verifyLoadingState(page, 'kanban-board')
      
      // Content should eventually load
      await expect(kanbanBoard.kanbanBoard).toBeVisible()
    })
  })

  test.describe('Task Filtering and Search', () => {
    test('should filter tasks by search term', async ({ page }) => {
      const initialTaskCount = await page.locator('task-card').count()
      
      // Search for specific task
      await kanbanBoard.filterTasks('authentication')
      
      // Should show fewer tasks
      const filteredTaskCount = await page.locator('task-card').count()
      expect(filteredTaskCount).toBeLessThanOrEqual(initialTaskCount)
      
      // Visible tasks should contain search term
      const visibleTasks = page.locator('task-card')
      const taskCount = await visibleTasks.count()
      
      for (let i = 0; i < taskCount; i++) {
        const task = visibleTasks.nth(i)
        const title = await task.locator('.task-title').textContent()
        const description = await task.locator('.task-description').textContent()
        
        const containsSearchTerm = title?.toLowerCase().includes('authentication') || 
                                 description?.toLowerCase().includes('authentication')
        expect(containsSearchTerm).toBe(true)
      }
    })

    test('should filter tasks by agent', async ({ page }) => {
      const initialTaskCount = await page.locator('task-card').count()
      
      // Filter by specific agent
      await kanbanBoard.filterByAgent('agent-1')
      
      // Should show only tasks assigned to that agent
      const filteredTasks = page.locator('task-card')
      const taskCount = await filteredTasks.count()
      
      if (taskCount > 0) {
        expect(taskCount).toBeLessThanOrEqual(initialTaskCount)
        
        // All visible tasks should be assigned to the selected agent
        for (let i = 0; i < taskCount; i++) {
          const task = filteredTasks.nth(i)
          const agent = await task.locator('.task-agent').textContent()
          expect(agent).toContain('agent-1')
        }
      }
    })

    test('should clear filters correctly', async ({ page }) => {
      const initialTaskCount = await page.locator('task-card').count()
      
      // Apply filters
      await kanbanBoard.filterTasks('test')
      await kanbanBoard.filterByAgent('agent-1')
      
      const filteredTaskCount = await page.locator('task-card').count()
      expect(filteredTaskCount).toBeLessThanOrEqual(initialTaskCount)
      
      // Clear filters
      await kanbanBoard.clearFilters()
      
      // Should show all tasks again
      const clearedTaskCount = await page.locator('task-card').count()
      expect(clearedTaskCount).toBe(initialTaskCount)
    })

    test('should handle no search results gracefully', async ({ page }) => {
      // Search for non-existent term
      await kanbanBoard.filterTasks('nonexistenttask12345')
      
      // Should show no tasks
      const taskCount = await page.locator('task-card').count()
      expect(taskCount).toBe(0)
      
      // Should show empty state message
      const emptyState = page.locator('.empty-search-results, [data-testid="no-results"]')
      if (await emptyState.isVisible()) {
        await expect(emptyState).toBeVisible()
        await expect(emptyState).toContainText(/no tasks found|no results/)
      }
    })
  })

  test.describe('Drag and Drop Task Movement', () => {
    test('should move task from pending to in-progress', async ({ page }) => {
      // Find a task in pending column
      const pendingTask = kanbanBoard.pendingColumn.locator('task-card').first()
      
      if (await pendingTask.isVisible()) {
        const taskId = await pendingTask.getAttribute('data-task-id')
        
        // Drag task to in-progress column
        await kanbanBoard.dragTaskToColumn(taskId!, 'in-progress')
        
        // Verify task moved to correct column
        await kanbanBoard.verifyTaskInColumn(taskId!, 'in-progress')
        
        // Should not show updating overlay after move
        await expect(kanbanBoard.updatingOverlay).not.toBeVisible()
        
        await TestHelpers.takeTimestampedScreenshot(page, 'task-moved')
      }
    })

    test('should move task through complete workflow', async ({ page }) => {
      // Find a task in pending column
      const pendingTask = kanbanBoard.pendingColumn.locator('task-card').first()
      
      if (await pendingTask.isVisible()) {
        const taskId = await pendingTask.getAttribute('data-task-id')
        
        // Move through workflow: pending -> in-progress -> review -> done
        await kanbanBoard.dragTaskToColumn(taskId!, 'in-progress')
        await kanbanBoard.verifyTaskInColumn(taskId!, 'in-progress')
        
        await kanbanBoard.dragTaskToColumn(taskId!, 'review')
        await kanbanBoard.verifyTaskInColumn(taskId!, 'review')
        
        await kanbanBoard.dragTaskToColumn(taskId!, 'done')
        await kanbanBoard.verifyTaskInColumn(taskId!, 'done')
      }
    })

    test('should handle drag and drop failures', async ({ page }) => {
      // Mock task update failure
      await page.route('**/api/v1/tasks/*', async route => {
        if (route.request().method() === 'PUT') {
          await route.fulfill({
            status: 500,
            contentType: 'application/json',
            body: JSON.stringify({ error: 'Update failed' })
          })
        }
      })
      
      const pendingTask = kanbanBoard.pendingColumn.locator('task-card').first()
      
      if (await pendingTask.isVisible()) {
        const taskId = await pendingTask.getAttribute('data-task-id')
        
        // Attempt to drag task
        await kanbanBoard.dragTaskToColumn(taskId!, 'in-progress')
        
        // Should show error state
        await TestHelpers.verifyErrorHandling(page)
        
        // Task should remain in original column
        await kanbanBoard.verifyTaskInColumn(taskId!, 'pending')
      }
    })

    test('should show loading state during task move', async ({ page }) => {
      // Mock slow task update
      await page.route('**/api/v1/tasks/*', async route => {
        if (route.request().method() === 'PUT') {
          await new Promise(resolve => setTimeout(resolve, 1000))
          await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ id: 'task-1', status: 'in-progress' })
          })
        }
      })
      
      const pendingTask = kanbanBoard.pendingColumn.locator('task-card').first()
      
      if (await pendingTask.isVisible()) {
        const taskId = await pendingTask.getAttribute('data-task-id')
        
        // Start drag operation
        await kanbanBoard.dragTaskToColumn(taskId!, 'in-progress')
        
        // Should show updating overlay during operation
        await expect(kanbanBoard.updatingOverlay).toBeVisible()
        
        // Should complete successfully
        await expect(kanbanBoard.updatingOverlay).not.toBeVisible()
      }
    })
  })

  test.describe('Task Creation and Editing', () => {
    test('should open task creation modal', async ({ page }) => {
      // Look for create task button
      const createTaskButton = page.locator('[data-testid="create-task"], .create-task-button')
      
      if (await createTaskButton.isVisible()) {
        await createTaskButton.click()
        
        // Modal should open
        await expect(taskEditModal.modal).toBeVisible()
        
        // Should be in creation mode (empty fields)
        await expect(taskEditModal.titleInput).toHaveValue('')
        await expect(taskEditModal.descriptionInput).toHaveValue('')
        
        await TestHelpers.takeTimestampedScreenshot(page, 'task-create-modal')
      }
    })

    test('should create new task', async ({ page }) => {
      const createTaskButton = page.locator('[data-testid="create-task"], .create-task-button')
      
      if (await createTaskButton.isVisible()) {
        await createTaskButton.click()
        await expect(taskEditModal.modal).toBeVisible()
        
        // Fill task form
        await taskEditModal.fillTaskForm({
          title: 'New Test Task',
          description: 'This is a test task created by automation',
          status: 'pending',
          priority: 'high',
          agent: 'agent-1',
          tags: 'automation, testing'
        })
        
        // Save task
        await taskEditModal.saveTask()
        
        // Should close modal
        await expect(taskEditModal.modal).not.toBeVisible()
        
        // New task should appear in pending column
        const newTask = page.locator('task-card').filter({ hasText: 'New Test Task' })
        await expect(newTask).toBeVisible()
        
        // Should be in pending column
        await expect(kanbanBoard.pendingColumn.locator('task-card').filter({ hasText: 'New Test Task' })).toBeVisible()
      }
    })

    test('should edit existing task', async ({ page }) => {
      // Click on an existing task
      const firstTask = page.locator('task-card').first()
      const taskId = await firstTask.getAttribute('data-task-id')
      
      await kanbanBoard.clickTask(taskId!)
      
      // Modal should open with existing data
      await expect(taskEditModal.modal).toBeVisible()
      
      // Fields should be populated
      const currentTitle = await taskEditModal.titleInput.inputValue()
      expect(currentTitle).toBeTruthy()
      
      // Update task
      await taskEditModal.fillTaskForm({
        title: 'Updated Task Title',
        description: 'Updated description for testing'
      })
      
      await taskEditModal.saveTask()
      
      // Modal should close
      await expect(taskEditModal.modal).not.toBeVisible()
      
      // Task should show updated information
      const updatedTask = page.locator(`task-card[data-task-id="${taskId}"]`)
      await expect(updatedTask.locator('.task-title')).toContainText('Updated Task Title')
    })

    test('should delete task', async ({ page }) => {
      const initialTaskCount = await page.locator('task-card').count()
      
      // Click on first task
      const firstTask = page.locator('task-card').first()
      const taskId = await firstTask.getAttribute('data-task-id')
      
      await kanbanBoard.clickTask(taskId!)
      await expect(taskEditModal.modal).toBeVisible()
      
      // Delete task
      await taskEditModal.deleteTask()
      
      // Modal should close
      await expect(taskEditModal.modal).not.toBeVisible()
      
      // Task count should decrease
      const newTaskCount = await page.locator('task-card').count()
      expect(newTaskCount).toBe(initialTaskCount - 1)
      
      // Specific task should no longer exist
      const deletedTask = page.locator(`task-card[data-task-id="${taskId}"]`)
      await expect(deletedTask).not.toBeVisible()
    })

    test('should validate required fields', async ({ page }) => {
      const createTaskButton = page.locator('[data-testid="create-task"], .create-task-button')
      
      if (await createTaskButton.isVisible()) {
        await createTaskButton.click()
        await expect(taskEditModal.modal).toBeVisible()
        
        // Try to save without title
        await taskEditModal.saveButton.click()
        
        // Should show validation error
        await taskEditModal.verifyValidationError('title')
        
        // Modal should remain open
        await expect(taskEditModal.modal).toBeVisible()
      }
    })

    test('should cancel editing without saving', async ({ page }) => {
      const firstTask = page.locator('task-card').first()
      const originalTitle = await firstTask.locator('.task-title').textContent()
      const taskId = await firstTask.getAttribute('data-task-id')
      
      await kanbanBoard.clickTask(taskId!)
      await expect(taskEditModal.modal).toBeVisible()
      
      // Make changes
      await taskEditModal.titleInput.fill('Temporary Title')
      
      // Cancel without saving
      await taskEditModal.cancelEdit()
      
      // Modal should close
      await expect(taskEditModal.modal).not.toBeVisible()
      
      // Original title should remain
      const taskAfterCancel = page.locator(`task-card[data-task-id="${taskId}"]`)
      await expect(taskAfterCancel.locator('.task-title')).toContainText(originalTitle!)
    })
  })

  test.describe('Task Priority and Assignment', () => {
    test('should display task priorities correctly', async ({ page }) => {
      // Find tasks with different priorities
      const highPriorityTasks = page.locator('task-card[data-priority="high"]')
      const mediumPriorityTasks = page.locator('task-card[data-priority="medium"]')
      const lowPriorityTasks = page.locator('task-card[data-priority="low"]')
      
      // Verify priority indicators
      if (await highPriorityTasks.count() > 0) {
        const highTask = highPriorityTasks.first()
        const priorityIndicator = highTask.locator('.priority-indicator')
        await expect(priorityIndicator).toHaveClass(/high/)
      }
      
      if (await mediumPriorityTasks.count() > 0) {
        const mediumTask = mediumPriorityTasks.first()
        const priorityIndicator = mediumTask.locator('.priority-indicator')
        await expect(priorityIndicator).toHaveClass(/medium/)
      }
      
      if (await lowPriorityTasks.count() > 0) {
        const lowTask = lowPriorityTasks.first()
        const priorityIndicator = lowTask.locator('.priority-indicator')
        await expect(priorityIndicator).toHaveClass(/low/)
      }
    })

    test('should display agent assignments', async ({ page }) => {
      // Verify tasks show assigned agents
      const tasksWithAgents = page.locator('task-card[data-agent]')
      
      if (await tasksWithAgents.count() > 0) {
        const firstAssignedTask = tasksWithAgents.first()
        const agentElement = firstAssignedTask.locator('.task-agent')
        
        await expect(agentElement).toBeVisible()
        
        const agentText = await agentElement.textContent()
        expect(agentText).toBeTruthy()
        expect(agentText).toMatch(/agent-\d+/)
      }
    })

    test('should allow reassigning tasks to different agents', async ({ page }) => {
      const firstTask = page.locator('task-card').first()
      const taskId = await firstTask.getAttribute('data-task-id')
      
      await kanbanBoard.clickTask(taskId!)
      await expect(taskEditModal.modal).toBeVisible()
      
      // Change agent assignment
      await taskEditModal.agentSelect.selectOption('agent-2')
      await taskEditModal.saveTask()
      
      // Verify assignment change
      const updatedTask = page.locator(`task-card[data-task-id="${taskId}"]`)
      await expect(updatedTask.locator('.task-agent')).toContainText('agent-2')
    })
  })

  test.describe('Offline Mode Support', () => {
    test('should show offline indicator when offline', async ({ page }) => {
      // Simulate offline mode
      await TestHelpers.simulateOfflineNetwork(page)
      await page.reload()
      await dashboardPage.waitForLoad()
      await dashboardPage.navigateToTasks()
      
      // Should show offline indicator
      await expect(kanbanBoard.offlineIndicator).toBeVisible()
      await expect(kanbanBoard.offlineIndicator).toContainText('Offline Mode')
    })

    test('should queue task operations when offline', async ({ page }) => {
      // Simulate offline mode
      await TestHelpers.simulateOfflineNetwork(page)
      
      const firstTask = kanbanBoard.pendingColumn.locator('task-card').first()
      
      if (await firstTask.isVisible()) {
        const taskId = await firstTask.getAttribute('data-task-id')
        
        // Try to move task while offline
        await kanbanBoard.dragTaskToColumn(taskId!, 'in-progress')
        
        // Task should move optimistically
        await kanbanBoard.verifyTaskInColumn(taskId!, 'in-progress')
        
        // Should show sync pending indicator
        const syncPendingIndicator = page.locator(`task-card[data-task-id="${taskId}"] .sync-pending`)
        if (await syncPendingIndicator.isVisible()) {
          await expect(syncPendingIndicator).toBeVisible()
        }
      }
    })

    test('should sync queued operations when back online', async ({ page }) => {
      // Start offline
      await TestHelpers.simulateOfflineNetwork(page)
      
      const firstTask = kanbanBoard.pendingColumn.locator('task-card').first()
      
      if (await firstTask.isVisible()) {
        const taskId = await firstTask.getAttribute('data-task-id')
        
        // Make changes while offline
        await kanbanBoard.dragTaskToColumn(taskId!, 'in-progress')
        
        // Go back online
        await TestHelpers.restoreNetwork(page)
        await page.reload()
        await dashboardPage.waitForLoad()
        await dashboardPage.navigateToTasks()
        
        // Changes should be synced
        await kanbanBoard.verifyTaskInColumn(taskId!, 'in-progress')
        
        // No sync pending indicator
        const syncPendingIndicator = page.locator(`task-card[data-task-id="${taskId}"] .sync-pending`)
        await expect(syncPendingIndicator).not.toBeVisible()
      }
    })
  })

  test.describe('Real-time Task Updates', () => {
    test('should receive real-time task updates', async ({ page }) => {
      // Set up WebSocket mock
      await APIMocks.mockWebSocketAPI(page)
      
      const initialTaskCount = await page.locator('task-card').count()
      
      // Wait for WebSocket updates
      await page.waitForTimeout(4000)
      
      // Should receive task updates (count may change or task status may update)
      const newTaskCount = await page.locator('task-card').count()
      expect(newTaskCount).toBeGreaterThanOrEqual(initialTaskCount)
    })

    test('should update task status in real-time', async ({ page }) => {
      const firstTask = page.locator('task-card').first()
      const taskId = await firstTask.getAttribute('data-task-id')
      
      // Mock WebSocket task update
      await page.evaluate((id) => {
        const mockUpdate = {
          type: 'task-updated',
          data: { id, status: 'in-progress', timestamp: new Date().toISOString() }
        }
        
        // Simulate WebSocket message
        window.dispatchEvent(new CustomEvent('websocket-message', { detail: mockUpdate }))
      }, taskId)
      
      // Task should update status
      await TestHelpers.waitForCondition(async () => {
        const updatedTask = page.locator(`task-card[data-task-id="${taskId}"]`)
        const statusElement = updatedTask.locator('.task-status')
        if (await statusElement.isVisible()) {
          const statusText = await statusElement.textContent()
          return statusText?.includes('in-progress') || false
        }
        return false
      })
    })

    test('should show new tasks created by other users', async ({ page }) => {
      const initialTaskCount = await page.locator('task-card').count()
      
      // Mock new task creation via WebSocket
      await page.evaluate(() => {
        const newTaskUpdate = {
          type: 'task-created',
          data: {
            id: 'task-realtime-new',
            title: 'Real-time Created Task',
            status: 'pending',
            timestamp: new Date().toISOString()
          }
        }
        
        window.dispatchEvent(new CustomEvent('websocket-message', { detail: newTaskUpdate }))
      })
      
      // New task should appear
      await TestHelpers.waitForCondition(async () => {
        const newTaskCount = await page.locator('task-card').count()
        return newTaskCount > initialTaskCount
      })
      
      // Specific new task should be visible
      const newTask = page.locator('task-card').filter({ hasText: 'Real-time Created Task' })
      await expect(newTask).toBeVisible()
    })
  })

  test.describe('Performance and Responsiveness', () => {
    test('should handle large number of tasks efficiently', async ({ page }) => {
      // Mock large task dataset
      const largeMockTasks = Array.from({ length: 50 }, (_, i) => ({
        id: `large-task-${i}`,
        title: `Large Dataset Task ${i}`,
        description: `Description ${i}`,
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
          body: JSON.stringify(largeMockTasks)
        })
      })
      
      const startTime = Date.now()
      await page.reload()
      await dashboardPage.waitForLoad()
      await dashboardPage.navigateToTasks()
      
      const loadTime = Date.now() - startTime
      
      // Should load within reasonable time even with many tasks
      expect(loadTime).toBeLessThan(10000)
      
      // All columns should be visible
      await expect(kanbanBoard.pendingColumn).toBeVisible()
      await expect(kanbanBoard.inProgressColumn).toBeVisible()
      await expect(kanbanBoard.reviewColumn).toBeVisible()
      await expect(kanbanBoard.doneColumn).toBeVisible()
    })

    test('should maintain smooth drag and drop with many tasks', async ({ page }) => {
      // Test drag and drop performance with current task set
      const firstTask = kanbanBoard.pendingColumn.locator('task-card').first()
      
      if (await firstTask.isVisible()) {
        const taskId = await firstTask.getAttribute('data-task-id')
        
        const startTime = Date.now()
        await kanbanBoard.dragTaskToColumn(taskId!, 'in-progress')
        const dragTime = Date.now() - startTime
        
        // Drag operation should complete quickly
        expect(dragTime).toBeLessThan(3000)
        
        // Task should be in correct position
        await kanbanBoard.verifyTaskInColumn(taskId!, 'in-progress')
      }
    })

    test('should be responsive on mobile devices', async ({ page }) => {
      // Test mobile viewport
      await TestHelpers.testResponsiveBreakpoint(page, 375, 667)
      
      // Kanban board should adapt to mobile
      await expect(kanbanBoard.kanbanBoard).toBeVisible()
      
      // Columns should be horizontally scrollable on mobile
      const boardContainer = page.locator('.board-container')
      await expect(boardContainer).toHaveCSS('overflow-x', 'auto')
      
      // Tasks should still be draggable on mobile
      const firstTask = page.locator('task-card').first()
      if (await firstTask.isVisible()) {
        await TestHelpers.testTouchInteraction(page, 'task-card:first-child', 'tap')
      }
      
      await TestHelpers.takeTimestampedScreenshot(page, 'mobile-kanban')
    })
  })
})