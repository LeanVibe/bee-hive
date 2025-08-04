import { Page, Route } from '@playwright/test'
import { mockTasks, mockAgents, mockEvents, mockSystemHealth, mockPerformanceMetrics } from '../fixtures/test-data'

/**
 * API mocking utilities for comprehensive testing scenarios
 */
export class APIMocks {
  
  /**
   * Set up all standard API mocks for normal operation
   */
  static async setupStandardMocks(page: Page) {
    await this.mockTasksAPI(page)
    await this.mockAgentsAPI(page)
    await this.mockEventsAPI(page)
    await this.mockSystemHealthAPI(page)
    await this.mockMetricsAPI(page)
    await this.mockWebSocketAPI(page)
  }

  /**
   * Mock Tasks API endpoints
   */
  static async mockTasksAPI(page: Page) {
    // GET /api/v1/tasks
    await page.route('**/api/v1/tasks', async (route: Route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(mockTasks)
        })
      }
    })

    // POST /api/v1/tasks (create task)
    await page.route('**/api/v1/tasks', async (route: Route) => {
      if (route.request().method() === 'POST') {
        const postData = JSON.parse(route.request().postData() || '{}')
        const newTask = {
          id: `task-${Date.now()}`,
          ...postData,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString()
        }
        
        await route.fulfill({
          status: 201,
          contentType: 'application/json',
          body: JSON.stringify(newTask)
        })
      }
    })

    // PUT /api/v1/tasks/:id (update task)
    await page.route('**/api/v1/tasks/*', async (route: Route) => {
      if (route.request().method() === 'PUT') {
        const taskId = route.request().url().split('/').pop()
        const updateData = JSON.parse(route.request().postData() || '{}')
        
        const existingTask = mockTasks.find(t => t.id === taskId)
        if (!existingTask) {
          await route.fulfill({ status: 404 })
          return
        }

        const updatedTask = {
          ...existingTask,
          ...updateData,
          updatedAt: new Date().toISOString()
        }
        
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(updatedTask)
        })
      }
    })

    // DELETE /api/v1/tasks/:id
    await page.route('**/api/v1/tasks/*', async (route: Route) => {
      if (route.request().method() === 'DELETE') {
        await route.fulfill({ status: 204 })
      }
    })
  }

  /**
   * Mock Agents API endpoints
   */
  static async mockAgentsAPI(page: Page) {
    // GET /api/v1/agents
    await page.route('**/api/v1/agents', async (route: Route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(mockAgents)
        })
      }
    })

    // POST /api/v1/agents/activate (activate agent team)
    await page.route('**/api/v1/agents/activate', async (route: Route) => {
      if (route.request().method() === 'POST') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ 
            message: 'Agent team activation initiated',
            activatedAgents: 5
          })
        })
      }
    })

    // POST /api/v1/agents/:id/deactivate
    await page.route('**/api/v1/agents/*/deactivate', async (route: Route) => {
      if (route.request().method() === 'POST') {
        const agentId = route.request().url().split('/')[5]
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ 
            message: `Agent ${agentId} deactivated`,
            agentId
          })
        })
      }
    })

    // PUT /api/v1/agents/:id/config
    await page.route('**/api/v1/agents/*/config', async (route: Route) => {
      if (route.request().method() === 'PUT') {
        const agentId = route.request().url().split('/')[5]
        const configData = JSON.parse(route.request().postData() || '{}')
        
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ 
            agentId,
            config: configData,
            updated: true
          })
        })
      }
    })
  }

  /**
   * Mock Events API endpoints
   */
  static async mockEventsAPI(page: Page) {
    // GET /api/v1/events
    await page.route('**/api/v1/events**', async (route: Route) => {
      if (route.request().method() === 'GET') {
        const url = new URL(route.request().url())
        const limit = parseInt(url.searchParams.get('limit') || '50')
        const severity = url.searchParams.get('severity')
        
        let events = [...mockEvents]
        
        if (severity) {
          events = events.filter(e => e.severity === severity)
        }
        
        events = events.slice(0, limit)
        
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(events)
        })
      }
    })
  }

  /**
   * Mock System Health API endpoints
   */
  static async mockSystemHealthAPI(page: Page) {
    // GET /api/v1/health
    await page.route('**/api/v1/health', async (route: Route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockSystemHealth)
      })
    })

    // GET /api/v1/health/summary
    await page.route('**/api/v1/health/summary', async (route: Route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockSystemHealth)
      })
    })
  }

  /**
   * Mock Metrics API endpoints
   */
  static async mockMetricsAPI(page: Page) {
    // GET /api/v1/metrics
    await page.route('**/api/v1/metrics**', async (route: Route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockPerformanceMetrics)
      })
    })
  }

  /**
   * Mock WebSocket connections
   */
  static async mockWebSocketAPI(page: Page) {
    await page.addInitScript(() => {
      class MockWebSocket extends EventTarget {
        static CONNECTING = 0
        static OPEN = 1
        static CLOSING = 2
        static CLOSED = 3

        url: string
        readyState: number = MockWebSocket.CONNECTING
        protocol: string = ''
        extensions: string = ''
        bufferedAmount: number = 0

        onopen: ((event: Event) => void) | null = null
        onmessage: ((event: MessageEvent) => void) | null = null
        onclose: ((event: CloseEvent) => void) | null = null
        onerror: ((event: Event) => void) | null = null

        private updateInterval?: number

        constructor(url: string, protocols?: string | string[]) {
          super()
          this.url = url
          
          // Simulate connection
          setTimeout(() => {
            this.readyState = MockWebSocket.OPEN
            const openEvent = new Event('open')
            this.onopen?.(openEvent)
            this.dispatchEvent(openEvent)
            
            // Start sending mock updates
            this.startMockUpdates()
          }, 100)
        }

        send(data: string | ArrayBuffer | Blob): void {
          if (this.readyState !== MockWebSocket.OPEN) {
            throw new Error('WebSocket is not open')
          }
          console.log('WebSocket send:', data)
        }

        close(code?: number, reason?: string): void {
          if (this.readyState === MockWebSocket.CLOSED || this.readyState === MockWebSocket.CLOSING) {
            return
          }
          
          this.readyState = MockWebSocket.CLOSING
          
          setTimeout(() => {
            this.readyState = MockWebSocket.CLOSED
            const closeEvent = new CloseEvent('close', { code: code || 1000, reason: reason || '' })
            this.onclose?.(closeEvent)
            this.dispatchEvent(closeEvent)
            
            if (this.updateInterval) {
              clearInterval(this.updateInterval)
            }
          }, 10)
        }

        private startMockUpdates() {
          const mockUpdates = [
            { type: 'task-updated', data: { id: 'task-1', status: 'in-progress', timestamp: new Date().toISOString() } },
            { type: 'task-created', data: { id: 'task-new', title: 'New Task', status: 'pending', timestamp: new Date().toISOString() } },
            { type: 'agent-status', data: { id: 'agent-1', status: 'active', timestamp: new Date().toISOString() } },
            { type: 'system-health', data: { overall: 'healthy', timestamp: new Date().toISOString() } },
            { type: 'new-event', data: { id: 'event-new', type: 'task-completed', severity: 'info', timestamp: new Date().toISOString() } }
          ]

          let updateIndex = 0
          this.updateInterval = window.setInterval(() => {
            if (this.readyState === MockWebSocket.OPEN) {
              const update = mockUpdates[updateIndex % mockUpdates.length]
              const messageEvent = new MessageEvent('message', { data: JSON.stringify(update) })
              
              this.onmessage?.(messageEvent)
              this.dispatchEvent(messageEvent)
              
              updateIndex++
            }
          }, 3000) // Send update every 3 seconds
        }
      }

      // Replace the global WebSocket
      ;(window as any).WebSocket = MockWebSocket
    })
  }

  /**
   * Mock error responses for testing error handling
   */
  static async mockErrorResponses(page: Page, errorType: 'network' | 'server' | 'timeout' = 'server') {
    const handler = async (route: Route) => {
      switch (errorType) {
        case 'network':
          await route.abort('internetdisconnected')
          break
        case 'timeout':
          // Delay response to trigger timeout
          await new Promise(resolve => setTimeout(resolve, 31000))
          await route.fulfill({ status: 408 })
          break
        case 'server':
        default:
          await route.fulfill({
            status: 500,
            contentType: 'application/json',
            body: JSON.stringify({ 
              error: 'Internal Server Error',
              message: 'Something went wrong on our end'
            })
          })
          break
      }
    }

    await page.route('**/api/v1/**', handler)
  }

  /**
   * Mock slow network conditions
   */
  static async mockSlowNetwork(page: Page, delayMs: number = 2000) {
    await page.route('**/api/v1/**', async (route: Route) => {
      await new Promise(resolve => setTimeout(resolve, delayMs))
      
      // Then use standard mock
      const url = route.request().url()
      if (url.includes('/tasks')) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(mockTasks)
        })
      } else if (url.includes('/agents')) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(mockAgents)
        })
      } else {
        await route.continue() // Let other requests through
      }
    })
  }

  /**
   * Mock partial failures (some endpoints succeed, others fail)
   */
  static async mockPartialFailures(page: Page, failingEndpoints: string[] = ['/agents', '/events']) {
    await page.route('**/api/v1/**', async (route: Route) => {
      const url = route.request().url()
      const shouldFail = failingEndpoints.some(endpoint => url.includes(endpoint))
      
      if (shouldFail) {
        await route.fulfill({
          status: 503,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Service Unavailable' })
        })
      } else if (url.includes('/tasks')) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(mockTasks)
        })
      } else if (url.includes('/health')) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(mockSystemHealth)
        })
      } else {
        await route.continue()
      }
    })
  }

  /**
   * Remove all mocked routes (for clean up)
   */
  static async clearAllMocks(page: Page) {
    await page.unroute('**/api/v1/**')
    await page.unroute('**/ws/**')
  }

  /**
   * Mock authentication responses
   */
  static async mockAuthAPI(page: Page, shouldSucceed: boolean = true) {
    await page.route('**/api/v1/auth/**', async (route: Route) => {
      if (shouldSucceed) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            token: 'mock-jwt-token',
            user: { id: 'user-1', email: 'test@example.com', role: 'admin' }
          })
        })
      } else {
        await route.fulfill({
          status: 401,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Invalid credentials' })
        })
      }
    })
  }

  /**
   * Set up mocks for enhanced features testing
   */
  static async setupEnhancedFeatureMocks(page: Page) {
    // Set up standard mocks first
    await this.setupStandardMocks(page)
    
    // Mock multi-agent task assignments
    await this.mockMultiAgentFeatures(page)
    
    // Mock advanced filtering and search
    await this.mockAdvancedSearchFeatures(page)
    
    // Mock sprint planning features
    await this.mockSprintPlanningFeatures(page)
    
    // Mock analytics features
    await this.mockAnalyticsFeatures(page)
    
    // Mock collaboration features
    await this.mockCollaborationFeatures(page)
  }

  /**
   * Mock multi-agent task assignment features
   */
  static async mockMultiAgentFeatures(page: Page) {
    // Mock agent workload distribution
    await page.route('**/api/v1/agents/workload', async (route: Route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          { agentId: 'agent-1', activeTasks: 5, capacity: 10, utilization: 50 },
          { agentId: 'agent-2', activeTasks: 8, capacity: 10, utilization: 80 },
          { agentId: 'agent-3', activeTasks: 3, capacity: 10, utilization: 30 },
          { agentId: 'agent-4', activeTasks: 7, capacity: 10, utilization: 70 },
          { agentId: 'agent-5', activeTasks: 6, capacity: 10, utilization: 60 }
        ])
      })
    })

    // Mock agent collaboration status
    await page.route('**/api/v1/agents/collaboration', async (route: Route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          activeCollaborations: 3,
          teamAssignments: [
            { taskId: 'task-1', agents: ['agent-1', 'agent-2'], status: 'active' },
            { taskId: 'task-5', agents: ['agent-2', 'agent-3', 'agent-4'], status: 'active' }
          ],
          communicationHealth: 'excellent'
        })
      })
    })
  }

  /**
   * Mock advanced search and filtering features
   */
  static async mockAdvancedSearchFeatures(page: Page) {
    // Mock natural language search
    await page.route('**/api/v1/search/natural', async (route: Route) => {
      const searchQuery = JSON.parse(route.request().postData() || '{}').query
      
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          query: searchQuery,
          interpretation: {
            filters: [
              { type: 'priority', value: 'high' },
              { type: 'agent', value: 'agent-1' },
              { type: 'timeframe', value: 'this week' }
            ]
          },
          results: mockTasks.slice(0, 3)
        })
      })
    })

    // Mock saved searches
    await page.route('**/api/v1/search/saved', async (route: Route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          { id: 'search-1', name: 'High Priority This Week', query: 'priority:high created:week' },
          { id: 'search-2', name: 'My Agent Tasks', query: 'agent:current status:active' },
          { id: 'search-3', name: 'Overdue Items', query: 'due:overdue' }
        ])
      })
    })

    // Mock bulk operations
    await page.route('**/api/v1/tasks/bulk', async (route: Route) => {
      const bulkAction = JSON.parse(route.request().postData() || '{}')
      
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          affectedTasks: bulkAction.taskIds.length,
          action: bulkAction.action,
          message: `${bulkAction.action} applied to ${bulkAction.taskIds.length} tasks`
        })
      })
    })
  }

  /**
   * Mock sprint planning features
   */
  static async mockSprintPlanningFeatures(page: Page) {
    // Mock sprints API
    await page.route('**/api/v1/sprints', async (route: Route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([
            {
              id: 'sprint-1',
              name: 'Q1 2024 Sprint',
              goal: 'Complete dashboard enhancements',
              startDate: '2024-01-01',
              endDate: '2024-01-14',
              status: 'active',
              totalPoints: 55,
              completedPoints: 32,
              velocity: 28
            }
          ])
        })
      } else if (route.request().method() === 'POST') {
        const sprintData = JSON.parse(route.request().postData() || '{}')
        await route.fulfill({
          status: 201,
          contentType: 'application/json',
          body: JSON.stringify({
            id: `sprint-${Date.now()}`,
            ...sprintData,
            status: 'planning',
            createdAt: new Date().toISOString()
          })
        })
      }
    })

    // Mock velocity tracking
    await page.route('**/api/v1/analytics/velocity', async (route: Route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          currentVelocity: 28,
          averageVelocity: 25,
          trend: 'increasing',
          history: [
            { sprint: 'Sprint 1', velocity: 22 },
            { sprint: 'Sprint 2', velocity: 26 },
            { sprint: 'Sprint 3', velocity: 24 },
            { sprint: 'Sprint 4', velocity: 28 }
          ]
        })
      })
    })

    // Mock burndown data
    await page.route('**/api/v1/analytics/burndown', async (route: Route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          sprintId: 'sprint-1',
          ideal: [55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0],
          actual: [55, 52, 48, 42, 38, 35, 30, 28, 23, 18, 12, 8],
          days: Array.from({ length: 12 }, (_, i) => `Day ${i + 1}`)
        })
      })
    })
  }

  /**
   * Mock analytics and reporting features
   */
  static async mockAnalyticsFeatures(page: Page) {
    // Mock task completion analytics
    await page.route('**/api/v1/analytics/completion', async (route: Route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          completionRate: 78,
          averageCompletionTime: 4.2,
          statusDistribution: {
            pending: 15,
            'in-progress': 25,
            review: 12,
            done: 48
          },
          agentPerformance: [
            { agentId: 'agent-1', completedTasks: 12, averageTime: 3.8 },
            { agentId: 'agent-2', completedTasks: 15, averageTime: 4.1 },
            { agentId: 'agent-3', completedTasks: 9, averageTime: 5.2 },
            { agentId: 'agent-4', completedTasks: 11, averageTime: 3.9 },
            { agentId: 'agent-5', completedTasks: 13, averageTime: 4.5 }
          ]
        })
      })
    })

    // Mock custom analytics queries
    await page.route('**/api/v1/analytics/query', async (route: Route) => {
      const queryData = JSON.parse(route.request().postData() || '{}')
      
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          query: queryData,
          results: [
            { label: 'Agent 1', value: 4.2 },
            { label: 'Agent 2', value: 3.8 },
            { label: 'Agent 3', value: 5.1 },
            { label: 'Agent 4', value: 3.9 },
            { label: 'Agent 5', value: 4.5 }
          ],
          chartType: 'bar',
          totalRecords: 150
        })
      })
    })
  }

  /**
   * Mock collaboration and real-time features
   */
  static async mockCollaborationFeatures(page: Page) {
    // Mock user presence
    await page.route('**/api/v1/presence', async (route: Route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          { userId: 'user-1', status: 'active', lastSeen: new Date().toISOString() },
          { userId: 'user-2', status: 'editing', taskId: 'task-1', lastSeen: new Date().toISOString() }
        ])
      })
    })

    // Mock collaborative editing status
    await page.route('**/api/v1/tasks/*/editing', async (route: Route) => {
      const taskId = route.request().url().split('/').slice(-2, -1)[0]
      
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          taskId,
          currentEditors: [
            { userId: 'user-1', name: 'Test User', joinedAt: new Date().toISOString() }
          ],
          lockStatus: 'unlocked'
        })
      })
    })
  }
}