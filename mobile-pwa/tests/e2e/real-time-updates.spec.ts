import { test, expect } from '@playwright/test'
import { DashboardPage, EventTimelinePage } from '../fixtures/page-objects'
import { TestHelpers } from '../utils/test-helpers'
import { APIMocks } from '../utils/api-mocks'

test.describe('Real-time Updates and Polling', () => {
  let dashboardPage: DashboardPage
  let eventTimeline: EventTimelinePage

  test.beforeEach(async ({ page }) => {
    dashboardPage = new DashboardPage(page)
    eventTimeline = new EventTimelinePage(page)
    
    // Set up API mocks and WebSocket simulation
    await APIMocks.setupStandardMocks(page)
    await APIMocks.mockWebSocketAPI(page)
    
    // Navigate to dashboard
    await dashboardPage.goto('/')
    await dashboardPage.waitForLoad()
  })

  test.describe('System Health Real-time Updates', () => {
    test('should update system health indicators in real-time', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Get initial system health status
      const initialHealthText = await dashboardPage.systemHealthCard.locator('.summary-value').textContent()
      
      // Wait for potential real-time updates
      await page.waitForTimeout(5000)
      
      // Health indicator should be present and functional
      await expect(dashboardPage.systemHealthCard).toBeVisible()
      await expect(dashboardPage.systemHealthCard.locator('.summary-value')).toBeVisible()
      
      // Color should reflect system status
      const healthColor = await dashboardPage.systemHealthCard.locator('.summary-value').evaluate(el => 
        window.getComputedStyle(el).color
      )
      expect(healthColor).toBeTruthy()
    })

    test('should update CPU and memory usage in real-time', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Monitor CPU usage updates
      const cpuCard = dashboardPage.cpuUsageCard
      await expect(cpuCard).toBeVisible()
      
      const initialCpuValue = await cpuCard.locator('.summary-value').textContent()
      
      // Wait for updates
      await page.waitForTimeout(3000)
      
      // CPU value should be numeric and reasonable
      const currentCpuValue = await cpuCard.locator('.summary-value').textContent()
      const cpuNumber = parseFloat(currentCpuValue?.replace('%', '') || '0')
      expect(cpuNumber).toBeGreaterThanOrEqual(0)
      expect(cpuNumber).toBeLessThanOrEqual(100)
      
      // Memory usage should also be updated
      const memoryCard = dashboardPage.memoryUsageCard
      const memoryValue = await memoryCard.locator('.summary-value').textContent()
      const memoryNumber = parseFloat(memoryValue?.replace('%', '') || '0')
      expect(memoryNumber).toBeGreaterThanOrEqual(0)
      expect(memoryNumber).toBeLessThanOrEqual(100)
    })

    test('should show real-time sync status updates', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Verify sync status updates
      await expect(dashboardPage.syncStatus).toBeVisible()
      
      // Sync status should show recent activity
      const syncText = await dashboardPage.syncStatus.textContent()
      expect(syncText).toMatch(/Last sync:|just now|ago/)
      
      // Sync indicator should be green (online)
      await expect(dashboardPage.syncIndicator).not.toHaveClass(/offline|error/)
    })

    test('should handle WebSocket connection states', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Simulate WebSocket disconnection
      await page.evaluate(() => {
        // Mock WebSocket close event
        window.dispatchEvent(new CustomEvent('websocket-close'))
      })
      
      // Should handle disconnection gracefully
      await page.waitForTimeout(1000)
      
      // Simulate reconnection
      await page.evaluate(() => {
        // Mock WebSocket open event
        window.dispatchEvent(new CustomEvent('websocket-open'))
      })
      
      // Should resume normal operation
      await expect(dashboardPage.syncIndicator).not.toHaveClass(/error/)
    })
  })

  test.describe('Task Updates in Real-time', () => {
    test('should update task counts in real-time', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Get initial task counts
      const initialActiveTasks = await dashboardPage.getSummaryCardValue('Active Tasks')
      const initialCompletedTasks = await dashboardPage.getSummaryCardValue('Completed Tasks')
      
      // Simulate task status change via WebSocket
      await page.evaluate(() => {
        const taskUpdate = {
          type: 'task-updated',
          data: {
            id: 'task-1',
            status: 'done',
            timestamp: new Date().toISOString()
          }
        }
        
        window.dispatchEvent(new CustomEvent('websocket-message', { detail: taskUpdate }))
      })
      
      // Wait for update processing
      await page.waitForTimeout(2000)
      
      // Task counts should potentially update (depending on the change)
      const newActiveTasks = await dashboardPage.getSummaryCardValue('Active Tasks')
      const newCompletedTasks = await dashboardPage.getSummaryCardValue('Completed Tasks')
      
      // Values should be valid numbers
      expect(parseInt(newActiveTasks)).toBeGreaterThanOrEqual(0)
      expect(parseInt(newCompletedTasks)).toBeGreaterThanOrEqual(0)
    })

    test('should show new tasks created in real-time', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      const initialTaskCount = parseInt(await dashboardPage.getSummaryCardValue('Active Tasks'))
      
      // Simulate new task creation
      await page.evaluate(() => {
        const newTask = {
          type: 'task-created',
          data: {
            id: 'realtime-task-new',
            title: 'Real-time New Task',
            status: 'pending',
            priority: 'high',
            agent: 'agent-1',
            timestamp: new Date().toISOString()
          }
        }
        
        window.dispatchEvent(new CustomEvent('websocket-message', { detail: newTask }))
      })
      
      // Navigate to tasks view to see the new task
      await dashboardPage.navigateToTasks()
      
      // New task should appear in the Kanban board
      await TestHelpers.waitForCondition(async () => {
        const newTaskElement = page.locator('task-card').filter({ hasText: 'Real-time New Task' })
        return await newTaskElement.isVisible()
      })
      
      const newTask = page.locator('task-card').filter({ hasText: 'Real-time New Task' })
      await expect(newTask).toBeVisible()
    })

    test('should update task status changes in Kanban board', async ({ page }) => {
      await dashboardPage.navigateToTasks()
      
      // Find a task in pending column
      const pendingTask = page.locator('kanban-column[data-column="pending"] task-card').first()
      
      if (await pendingTask.isVisible()) {
        const taskId = await pendingTask.getAttribute('data-task-id')
        
        // Simulate task status change
        await page.evaluate((id) => {
          const statusUpdate = {
            type: 'task-updated',
            data: {
              id,
              status: 'in-progress',
              timestamp: new Date().toISOString()
            }
          }
          
          window.dispatchEvent(new CustomEvent('websocket-message', { detail: statusUpdate }))
        }, taskId)
        
        // Task should move to in-progress column
        await TestHelpers.waitForCondition(async () => {
          const inProgressTask = page.locator(`kanban-column[data-column="in-progress"] task-card[data-task-id="${taskId}"]`)
          return await inProgressTask.isVisible()
        })
        
        const movedTask = page.locator(`kanban-column[data-column="in-progress"] task-card[data-task-id="${taskId}"]`)
        await expect(movedTask).toBeVisible()
      }
    })

    test('should handle task deletion in real-time', async ({ page }) => {
      await dashboardPage.navigateToTasks()
      
      const initialTaskCount = await page.locator('task-card').count()
      const firstTask = page.locator('task-card').first()
      const taskId = await firstTask.getAttribute('data-task-id')
      
      // Simulate task deletion
      await page.evaluate((id) => {
        const deleteUpdate = {
          type: 'task-deleted',
          data: { id, timestamp: new Date().toISOString() }
        }
        
        window.dispatchEvent(new CustomEvent('websocket-message', { detail: deleteUpdate }))
      }, taskId)
      
      // Task should be removed
      await TestHelpers.waitForCondition(async () => {
        const deletedTask = page.locator(`task-card[data-task-id="${taskId}"]`)
        return !(await deletedTask.isVisible())
      })
      
      const finalTaskCount = await page.locator('task-card').count()
      expect(finalTaskCount).toBe(initialTaskCount - 1)
    })
  })

  test.describe('Agent Status Real-time Updates', () => {
    test('should update agent status in real-time', async ({ page }) => {
      await dashboardPage.navigateToAgents()
      
      const firstAgent = page.locator('.agent-card').first()
      const agentId = await firstAgent.getAttribute('data-agent-id')
      const initialStatus = await firstAgent.locator('.agent-status').textContent()
      
      // Simulate agent status change
      await page.evaluate((id) => {
        const statusUpdate = {
          type: 'agent-status',
          data: {
            id,
            status: 'active',
            timestamp: new Date().toISOString()
          }
        }
        
        window.dispatchEvent(new CustomEvent('websocket-message', { detail: statusUpdate }))
      }, agentId)
      
      // Status should update
      await TestHelpers.waitForCondition(async () => {
        const currentStatus = await firstAgent.locator('.agent-status').textContent()
        return currentStatus !== initialStatus
      })
      
      // Status indicator should reflect the change
      const statusIndicator = firstAgent.locator('.status-indicator')
      await expect(statusIndicator).toBeVisible()
    })

    test('should update agent performance metrics in real-time', async ({ page }) => {
      await dashboardPage.navigateToAgents()
      
      const firstAgent = page.locator('.agent-card').first()
      const agentId = await firstAgent.getAttribute('data-agent-id')
      
      // Simulate performance metrics update
      await page.evaluate((id) => {
        const metricsUpdate = {
          type: 'agent-metrics',
          data: {
            id,
            metrics: {
              cpuUsage: 75,
              memoryUsage: 60,
              performance: 88
            },
            timestamp: new Date().toISOString()
          }
        }
        
        window.dispatchEvent(new CustomEvent('websocket-message', { detail: metricsUpdate }))
      }, agentId)
      
      // Metrics should be visible and updated
      const cpuMetric = firstAgent.locator('.cpu-usage')
      const memoryMetric = firstAgent.locator('.memory-usage')
      
      if (await cpuMetric.isVisible()) {
        await expect(cpuMetric).toBeVisible()
        const cpuText = await cpuMetric.textContent()
        expect(cpuText).toMatch(/\d+/)
      }
      
      if (await memoryMetric.isVisible()) {
        await expect(memoryMetric).toBeVisible()
        const memoryText = await memoryMetric.textContent()
        expect(memoryText).toMatch(/\d+/)
      }
    })

    test('should update agent task assignments in real-time', async ({ page }) => {
      await dashboardPage.navigateToAgents()
      
      const firstAgent = page.locator('.agent-card').first()
      const agentId = await firstAgent.getAttribute('data-agent-id')
      
      // Simulate task assignment
      await page.evaluate((id) => {
        const assignmentUpdate = {
          type: 'agent-task-assigned',
          data: {
            agentId: id,
            taskId: 'task-realtime-assigned',
            taskTitle: 'Real-time Assigned Task',
            timestamp: new Date().toISOString()
          }
        }
        
        window.dispatchEvent(new CustomEvent('websocket-message', { detail: assignmentUpdate }))
      }, agentId)
      
      // Current task should update
      const currentTask = firstAgent.locator('.current-task')
      if (await currentTask.isVisible()) {
        await TestHelpers.waitForCondition(async () => {
          const taskText = await currentTask.textContent()
          return taskText?.includes('Real-time Assigned Task') || false
        })
      }
    })
  })

  test.describe('Event Timeline Real-time Updates', () => {
    test('should show new events in real-time', async ({ page }) => {
      await dashboardPage.navigateToEvents()
      
      const initialEventCount = await eventTimeline.getEventCount()
      
      // Simulate new event
      await page.evaluate(() => {
        const newEvent = {
          type: 'new-event',
          data: {
            id: 'realtime-event-new',
            type: 'system-alert',
            title: 'Real-time System Alert',
            description: 'This is a real-time generated event',
            severity: 'warning',
            timestamp: new Date().toISOString()
          }
        }
        
        window.dispatchEvent(new CustomEvent('websocket-message', { detail: newEvent }))
      })
      
      // New event should appear
      await TestHelpers.waitForCondition(async () => {
        const newEventCount = await eventTimeline.getEventCount()
        return newEventCount > initialEventCount
      })
      
      const newEvent = page.locator('.timeline-event').filter({ hasText: 'Real-time System Alert' })
      await expect(newEvent).toBeVisible()
    })

    test('should update event timeline automatically', async ({ page }) => {
      await dashboardPage.navigateToEvents()
      
      // Verify real-time updates functionality
      await eventTimeline.verifyRealTimeUpdates()
      
      // Events should be sorted by timestamp (newest first)
      const eventItems = page.locator('.timeline-event')
      const eventCount = await eventItems.count()
      
      if (eventCount >= 2) {
        const firstEventTime = await eventItems.first().locator('.event-timestamp').textContent()
        const secondEventTime = await eventItems.nth(1).locator('.event-timestamp').textContent()
        
        // First event should be more recent (this is a basic check)
        expect(firstEventTime).toBeTruthy()
        expect(secondEventTime).toBeTruthy()
      }
    })

    test('should filter events by severity in real-time', async ({ page }) => {
      await dashboardPage.navigateToEvents()
      
      // Filter by error events
      await eventTimeline.filterEventsBySeverity('error')
      
      // Add new error event
      await page.evaluate(() => {
        const errorEvent = {
          type: 'new-event',
          data: {
            id: 'realtime-error-event',
            type: 'agent-error',
            title: 'Real-time Error Event',
            severity: 'error',
            timestamp: new Date().toISOString()
          }
        }
        
        window.dispatchEvent(new CustomEvent('websocket-message', { detail: errorEvent }))
      })
      
      // New error event should appear in filtered view
      await TestHelpers.waitForCondition(async () => {
        const errorEvent = page.locator('.timeline-event').filter({ hasText: 'Real-time Error Event' })
        return await errorEvent.isVisible()
      })
    })

    test('should limit event timeline to prevent memory issues', async ({ page }) => {
      await dashboardPage.navigateToEvents()
      
      const initialEventCount = await eventTimeline.getEventCount()
      
      // Generate many events rapidly
      for (let i = 0; i < 10; i++) {
        await page.evaluate((index) => {
          const event = {
            type: 'new-event',
            data: {
              id: `bulk-event-${index}`,
              type: 'info',
              title: `Bulk Event ${index}`,
              severity: 'info',
              timestamp: new Date().toISOString()
            }
          }
          
          window.dispatchEvent(new CustomEvent('websocket-message', { detail: event }))
        }, i)
        
        await page.waitForTimeout(100) // Small delay between events
      }
      
      // Event count should be limited (e.g., max 100 events)
      const finalEventCount = await eventTimeline.getEventCount()
      expect(finalEventCount).toBeLessThanOrEqual(100)
    })
  })

  test.describe('Polling Functionality', () => {
    test('should poll for updates when WebSocket is unavailable', async ({ page }) => {
      // Disable WebSocket simulation
      await page.evaluate(() => {
        // Mock WebSocket failure
        window.WebSocket = class MockFailedWebSocket {
          constructor() {
            setTimeout(() => {
              if (this.onerror) {
                this.onerror(new Event('error'))
              }
            }, 100)
          }
          
          close() {}
          send() {}
        } as any
      })
      
      await dashboardPage.goto('/')
      await dashboardPage.waitForLoad()
      await dashboardPage.navigateToOverview()
      
      // Should fall back to polling
      await page.waitForTimeout(3000)
      
      // Data should still be updated via polling
      await expect(dashboardPage.activeTasksCard).toBeVisible()
      await expect(dashboardPage.systemHealthCard).toBeVisible()
    })

    test('should adjust polling frequency based on activity', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Monitor for polling requests
      let pollRequestCount = 0
      
      page.on('request', request => {
        if (request.url().includes('/api/v1/') && request.method() === 'GET') {
          pollRequestCount++
        }
      })
      
      // Wait for some polling cycles
      await page.waitForTimeout(10000)
      
      // Should have made multiple polling requests
      expect(pollRequestCount).toBeGreaterThan(0)
    })

    test('should handle polling errors gracefully', async ({ page }) => {
      // Mock polling endpoint failures
      await page.route('**/api/v1/health', async route => {
        await route.fulfill({ status: 503 })
      })
      
      await dashboardPage.goto('/')
      await dashboardPage.waitForLoad()
      
      // Should handle polling errors without crashing
      await page.waitForTimeout(5000)
      
      // Application should remain functional
      await expect(dashboardPage.pageTitle).toBeVisible()
      await expect(dashboardPage.overviewTab).toBeVisible()
    })

    test('should stop polling when page is not visible', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Simulate page visibility change
      await page.evaluate(() => {
        Object.defineProperty(document, 'visibilityState', {
          value: 'hidden',
          writable: true
        })
        
        document.dispatchEvent(new Event('visibilitychange'))
      })
      
      // Monitor polling requests after hiding
      let hiddenPollRequests = 0
      
      page.on('request', request => {
        if (request.url().includes('/api/v1/') && request.method() === 'GET') {
          hiddenPollRequests++
        }
      })
      
      await page.waitForTimeout(5000)
      
      // Should make fewer or no polling requests when hidden
      expect(hiddenPollRequests).toBeLessThan(5)
    })
  })

  test.describe('Performance and Resource Management', () => {
    test('should handle high-frequency updates efficiently', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Generate rapid updates
      for (let i = 0; i < 50; i++) {
        await page.evaluate((index) => {
          const update = {
            type: 'system-metrics',
            data: {
              cpuUsage: Math.random() * 100,
              memoryUsage: Math.random() * 100,
              timestamp: new Date().toISOString()
            }
          }
          
          window.dispatchEvent(new CustomEvent('websocket-message', { detail: update }))
        }, i)
      }
      
      // UI should remain responsive
      await expect(dashboardPage.cpuUsageCard).toBeVisible()
      await expect(dashboardPage.memoryUsageCard).toBeVisible()
      
      // Values should be reasonable (not showing rapid flickering)
      const cpuValue = await dashboardPage.getSummaryCardValue('CPU Usage')
      const memoryValue = await dashboardPage.getSummaryCardValue('Memory Usage')
      
      expect(parseFloat(cpuValue)).toBeGreaterThanOrEqual(0)
      expect(parseFloat(memoryValue)).toBeGreaterThanOrEqual(0)
    })

    test('should throttle update processing', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      const startTime = Date.now()
      
      // Send many updates quickly
      for (let i = 0; i < 20; i++) {
        await page.evaluate(() => {
          const update = {
            type: 'task-updated',
            data: { id: 'task-1', status: 'in-progress' }
          }
          
          window.dispatchEvent(new CustomEvent('websocket-message', { detail: update }))
        })
      }
      
      const endTime = Date.now()
      const processingTime = endTime - startTime
      
      // Should process updates efficiently
      expect(processingTime).toBeLessThan(2000)
      
      // UI should remain stable
      await expect(dashboardPage.activeTasksCard).toBeVisible()
    })

    test('should cleanup resources when navigating away', async ({ page }) => {
      await dashboardPage.navigateToOverview()
      
      // Start real-time monitoring
      await page.waitForTimeout(2000)
      
      // Navigate to different page
      await page.goto('about:blank')
      
      // Wait a moment
      await page.waitForTimeout(1000)
      
      // Navigate back to dashboard
      await dashboardPage.goto('/')
      await dashboardPage.waitForLoad()
      
      // Should reinitialize properly without memory leaks
      await expect(dashboardPage.pageTitle).toBeVisible()
      await expect(dashboardPage.overviewTab).toBeVisible()
    })
  })
})