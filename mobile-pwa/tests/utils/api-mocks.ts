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
}