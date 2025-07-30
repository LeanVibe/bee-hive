/**
 * Enhanced Coordination Dashboard Integration Tests
 * 
 * Comprehensive test suite for the new coordination features including
 * task distribution, performance analytics, WebSocket integration,
 * and mobile responsiveness.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { mount, VueWrapper } from '@vue/test-utils'
import { nextTick } from 'vue'
import { createPinia } from 'pinia'

// Components
import TaskDistributionInterface from '@/components/coordination/TaskDistributionInterface.vue'
import PerformanceAnalyticsViewer from '@/components/coordination/PerformanceAnalyticsViewer.vue'
import AgentCapabilityMatcher from '@/components/coordination/AgentCapabilityMatcher.vue'
import TaskCard from '@/components/coordination/TaskCard.vue'
import AgentCard from '@/components/coordination/AgentCard.vue'

// Services
import { useCoordinationWebSocket } from '@/services/coordinationWebSocketService'
import { api } from '@/services/api'

// Test utilities
import { mockAgent, mockTask, mockPerformanceMetrics, createMockWebSocket } from '../utils/coordinationMocks'

// Mock external dependencies
vi.mock('@/services/api')
vi.mock('chart.js', () => ({
  Chart: vi.fn().mockImplementation(() => ({
    destroy: vi.fn(),
    update: vi.fn(),
    data: { datasets: [] },
    options: {}
  })),
  registerables: []
}))

describe('Enhanced Coordination Dashboard', () => {
  let wrapper: VueWrapper<any>
  let pinia: any
  let mockWebSocket: any

  beforeEach(() => {
    pinia = createPinia()

    // Mock WebSocket
    mockWebSocket = createMockWebSocket()
    global.WebSocket = vi.fn(() => mockWebSocket)

    // Mock API responses
    vi.mocked(api.get).mockImplementation((url) => {
      switch (true) {
        case url.includes('/team-coordination/agents'):
          return Promise.resolve({ data: [mockAgent()] })
        case url.includes('/tasks'):
          return Promise.resolve({ data: { tasks: [mockTask()] } })
        case url.includes('/team-coordination/metrics'):
          return Promise.resolve({ data: mockPerformanceMetrics() })
        default:
          return Promise.resolve({ data: {} })
      }
    })

    vi.mocked(api.post).mockResolvedValue({ data: { success: true } })
  })

  afterEach(() => {
    wrapper?.unmount()
    vi.clearAllMocks()
  })

  describe('Task Distribution Interface', () => {
    beforeEach(async () => {
      wrapper = mount(TaskDistributionInterface, {
        global: {
          plugins: [pinia]
        }
      })
      await nextTick()
    })

    it('should render task distribution interface correctly', () => {
      expect(wrapper.find('[data-testid="task-distribution-interface"]').exists()).toBe(true)
      expect(wrapper.find('h2').text()).toContain('Task Distribution Center')
    })

    it('should load tasks and agents on mount', async () => {
      await nextTick()
      expect(api.get).toHaveBeenCalledWith('/tasks', expect.any(Object))
      expect(api.get).toHaveBeenCalledWith('/team-coordination/agents')
    })

    it('should display task cards with correct information', async () => {
      const taskCard = wrapper.findComponent(TaskCard)
      expect(taskCard.exists()).toBe(true)
      expect(taskCard.props('task')).toEqual(expect.objectContaining({
        task_title: expect.any(String),
        priority: expect.any(String)
      }))
    })

    it('should display agent cards with workload information', async () => {
      const agentCard = wrapper.findComponent(AgentCard)
      expect(agentCard.exists()).toBe(true)
      expect(agentCard.props('agent')).toEqual(expect.objectContaining({
        agent_id: expect.any(String),
        current_workload: expect.any(Number)
      }))
    })

    it('should handle drag and drop task assignment', async () => {
      const taskCard = wrapper.findComponent(TaskCard)
      const agentCard = wrapper.findComponent(AgentCard)

      // Simulate drag start
      await taskCard.vm.$emit('drag-start', mockTask())
      expect(wrapper.vm.draggedTask).toBeTruthy()

      // Simulate drop on agent
      await agentCard.vm.$emit('drop', mockAgent())
      expect(wrapper.vm.showAssignmentModal).toBe(true)
    })

    it('should show task creation modal when clicking new task', async () => {
      const newTaskButton = wrapper.find('[data-testid="new-task-button"]')
      await newTaskButton.trigger('click')
      expect(wrapper.vm.showTaskModal).toBe(true)
    })

    it('should generate matching suggestions for selected task', async () => {
      const task = mockTask()
      await wrapper.vm.selectTask(task)
      
      expect(wrapper.vm.selectedTask).toEqual(task)
      expect(wrapper.vm.matchingSuggestions.length).toBeGreaterThan(0)
    })

    it('should filter tasks by priority', async () => {
      const prioritySelect = wrapper.find('[data-testid="priority-filter"]')
      await prioritySelect.setValue('HIGH')

      const filteredTasks = wrapper.vm.filteredTasks
      expect(filteredTasks.every((task: any) => task.priority === 'HIGH')).toBe(true)
    })

    it('should refresh agents when refresh button is clicked', async () => {
      const refreshButton = wrapper.find('[data-testid="refresh-button"]')
      await refreshButton.trigger('click')

      expect(api.get).toHaveBeenCalledWith('/team-coordination/agents')
    })
  })

  describe('Performance Analytics Viewer', () => {
    beforeEach(async () => {
      wrapper = mount(PerformanceAnalyticsViewer, {
        global: {
          plugins: [pinia]
        }
      })
      await nextTick()
    })

    it('should render performance analytics interface', () => {
      expect(wrapper.find('[data-testid="performance-analytics"]').exists()).toBe(true)
      expect(wrapper.find('h2').text()).toContain('Performance Analytics')
    })

    it('should load performance metrics on mount', async () => {
      await nextTick()
      expect(api.get).toHaveBeenCalledWith('/team-coordination/metrics', expect.any(Object))
    })

    it('should display key performance metrics', async () => {
      const metricCards = wrapper.findAllComponents({ name: 'MetricCard' })
      expect(metricCards.length).toBeGreaterThan(0)
      
      const systemEfficiencyCard = metricCards.find(card => 
        card.props('title') === 'System Efficiency'
      )
      expect(systemEfficiencyCard.exists()).toBe(true)
    })

    it('should create charts when data is loaded', async () => {
      await wrapper.vm.updateCharts()
      expect(wrapper.vm.completionChartInstance).toBeTruthy()
    })

    it('should handle time range changes', async () => {
      const timeRangeSelect = wrapper.find('[data-testid="time-range-select"]')
      await timeRangeSelect.setValue('7d')

      expect(wrapper.vm.selectedTimeRange).toBe('7d')
      expect(api.get).toHaveBeenCalledWith('/team-coordination/metrics', 
        expect.objectContaining({
          params: expect.objectContaining({
            time_range_hours: 168
          })
        })
      )
    })

    it('should detect and display bottlenecks', async () => {
      wrapper.vm.bottlenecks = [
        {
          id: 'bt-1',
          title: 'High CPU Usage',
          severity: 'high',
          impact: 0.8
        }
      ]
      await nextTick()

      const bottleneckItems = wrapper.findAll('[data-testid="bottleneck-item"]')
      expect(bottleneckItems.length).toBe(1)
    })

    it('should enable auto-refresh when toggle is activated', async () => {
      const autoRefreshButton = wrapper.find('[data-testid="auto-refresh-toggle"]')
      await autoRefreshButton.trigger('click')

      expect(wrapper.vm.autoRefresh).toBe(true)
      expect(wrapper.vm.refreshInterval).toBeTruthy()
    })

    it('should export analytics data', async () => {
      const exportButton = wrapper.find('[data-testid="export-button"]')
      
      // Mock URL.createObjectURL
      global.URL.createObjectURL = vi.fn(() => 'blob:url')
      global.URL.revokeObjectURL = vi.fn()

      await exportButton.trigger('click')
      
      expect(global.URL.createObjectURL).toHaveBeenCalled()
    })
  })

  describe('Agent Capability Matcher', () => {
    beforeEach(async () => {
      wrapper = mount(AgentCapabilityMatcher, {
        global: {
          plugins: [pinia]
        }
      })
      await nextTick()
    })

    it('should render capability matcher interface', () => {
      expect(wrapper.find('[data-testid="capability-matcher"]').exists()).toBe(true)
      expect(wrapper.find('h2').text()).toContain('Agent Capability Matcher')
    })

    it('should calculate agent matches for task requirements', async () => {
      const taskRequirements = {
        title: 'Test Task',
        requiredCapabilities: [
          { name: 'Vue.js', importance: 'high' },
          { name: 'TypeScript', importance: 'medium' }
        ]
      }

      wrapper.vm.taskRequirements = taskRequirements
      await wrapper.vm.calculateMatches()

      expect(wrapper.vm.matches.length).toBeGreaterThan(0)
      expect(wrapper.vm.matches[0]).toEqual(
        expect.objectContaining({
          agent: expect.any(Object),
          overallMatch: expect.any(Number),
          capabilityMatches: expect.any(Object)
        })
      )
    })

    it('should switch between different view modes', async () => {
      // Test grid view
      const gridButton = wrapper.find('[data-testid="grid-view-button"]')
      await gridButton.trigger('click')
      expect(wrapper.vm.viewMode).toBe('grid')

      // Test matrix view
      const matrixButton = wrapper.find('[data-testid="matrix-view-button"]')
      await matrixButton.trigger('click')
      expect(wrapper.vm.viewMode).toBe('matrix')

      // Test radar view
      const radarButton = wrapper.find('[data-testid="radar-view-button"]')
      await radarButton.trigger('click')
      expect(wrapper.vm.viewMode).toBe('radar')
    })

    it('should create radar charts in radar view mode', async () => {
      wrapper.vm.viewMode = 'radar'
      wrapper.vm.matches = [
        {
          agent: mockAgent(),
          overallMatch: 0.8,
          capabilityMatches: {}
        }
      ]

      await nextTick()
      await wrapper.vm.updateRadarCharts()

      expect(wrapper.vm.radarCharts.size).toBeGreaterThan(0)
    })

    it('should handle agent selection and assignment', async () => {
      const agent = mockAgent()
      const matchData = {
        agent,
        overallMatch: 0.8,
        capabilityMatches: {}
      }

      await wrapper.vm.selectAgent(agent, matchData)

      expect(wrapper.vm.selectedAgent).toEqual(agent)
      expect(wrapper.vm.selectedMatch).toEqual(matchData)
      expect(wrapper.vm.showAgentModal).toBe(true)
    })
  })

  describe('WebSocket Integration', () => {
    let webSocketService: any

    beforeEach(() => {
      webSocketService = useCoordinationWebSocket()
    })

    it('should establish WebSocket connection', async () => {
      await webSocketService.connect('test-connection')
      
      expect(global.WebSocket).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/team-coordination/ws/')
      )
      expect(webSocketService.state.isConnected).toBe(true)
    })

    it('should handle incoming task assignment messages', async () => {
      const messageHandler = vi.fn()
      webSocketService.onMessage('test', 'task_assignment', messageHandler)

      const taskAssignmentMessage = {
        type: 'task_assignment',
        data: {
          task_id: 'task-1',
          agent_id: 'agent-1',
          task_title: 'Test Task'
        }
      }

      // Simulate message reception
      mockWebSocket.onmessage({ data: JSON.stringify(taskAssignmentMessage) })

      expect(messageHandler).toHaveBeenCalledWith(taskAssignmentMessage)
    })

    it('should handle agent workload updates', async () => {
      const messageHandler = vi.fn()
      webSocketService.onMessage('test', 'agent_workload', messageHandler)

      const workloadMessage = {
        type: 'agent_workload',
        data: {
          agent_id: 'agent-1',
          new_workload: 0.8,
          active_tasks: 3
        }
      }

      mockWebSocket.onmessage({ data: JSON.stringify(workloadMessage) })

      expect(messageHandler).toHaveBeenCalledWith(workloadMessage)
    })

    it('should handle performance metrics updates', async () => {
      const messageHandler = vi.fn()
      webSocketService.onMessage('test', 'performance_metrics', messageHandler)

      const metricsMessage = {
        type: 'performance_metrics',
        data: {
          system_efficiency: 0.85,
          task_throughput: 45
        }
      }

      mockWebSocket.onmessage({ data: JSON.stringify(metricsMessage) })

      expect(messageHandler).toHaveBeenCalledWith(metricsMessage)
    })

    it('should reconnect on connection loss', async () => {
      await webSocketService.connect()
      
      // Simulate connection loss
      mockWebSocket.onclose({ code: 1006, reason: 'Connection lost' })

      expect(webSocketService.state.isConnected).toBe(false)
      
      // Should attempt reconnection
      await new Promise(resolve => setTimeout(resolve, 100))
      expect(global.WebSocket).toHaveBeenCalledTimes(2)
    })

    it('should send heartbeat messages', async () => {
      await webSocketService.connect()
      
      // Fast-forward time to trigger heartbeat
      vi.advanceTimersByTime(30000)

      expect(mockWebSocket.send).toHaveBeenCalledWith(
        expect.stringContaining('"type":"ping"')
      )
    })
  })

  describe('Mobile Responsiveness', () => {
    beforeEach(() => {
      // Mock mobile viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375
      })
      
      Object.defineProperty(window, 'innerHeight', {
        writable: true,
        configurable: true,
        value: 667
      })

      wrapper = mount(TaskDistributionInterface, {
        global: {
          plugins: [pinia]
        }
      })
    })

    it('should adapt layout for mobile screens', async () => {
      await nextTick()
      
      const container = wrapper.find('.distribution-layout')
      expect(container.classes()).toContain('grid-cols-1')
    })

    it('should enable touch gestures on mobile', async () => {
      const taskCard = wrapper.findComponent(TaskCard)
      
      // Simulate touch events
      const touchStart = new TouchEvent('touchstart', {
        touches: [{ clientX: 100, clientY: 100, identifier: 0 } as Touch]
      })
      
      await taskCard.element.dispatchEvent(touchStart)
      expect(wrapper.vm.isTouch).toBe(true)
    })

    it('should show mobile-optimized spacing and sizing', async () => {
      const cards = wrapper.findAll('.glass-card')
      
      cards.forEach(card => {
        expect(card.element.classList.toString()).toMatch(/p-\d+/)
      })
    })
  })

  describe('Error Handling', () => {
    it('should handle API errors gracefully', async () => {
      vi.mocked(api.get).mockRejectedValue(new Error('Network error'))

      wrapper = mount(TaskDistributionInterface, {
        global: {
          plugins: [pinia]
        }
      })

      await nextTick()

      // Should not crash and should show error state
      expect(wrapper.vm.loading).toBe(false)
      expect(wrapper.find('[data-testid="error-message"]').exists()).toBe(true)
    })

    it('should handle WebSocket connection errors', async () => {
      const webSocketService = useCoordinationWebSocket()
      
      mockWebSocket.onerror(new Error('Connection failed'))
      
      expect(webSocketService.state.error).toBeTruthy()
      expect(webSocketService.state.isConnected).toBe(false)
    })

    it('should handle malformed WebSocket messages', async () => {
      const webSocketService = useCoordinationWebSocket()
      const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      
      // Send malformed JSON
      mockWebSocket.onmessage({ data: 'invalid-json' })
      
      expect(errorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Failed to parse message'),
        expect.any(Error)
      )
      
      errorSpy.mockRestore()
    })

    it('should recover from chart creation failures', async () => {
      // Mock Chart.js to throw error
      const ChartMock = vi.fn().mockImplementation(() => {
        throw new Error('Canvas context error')
      })
      
      vi.doMock('chart.js', () => ({ Chart: ChartMock, registerables: [] }))

      wrapper = mount(PerformanceAnalyticsViewer, {
        global: {
          plugins: [pinia]
        }
      })

      await nextTick()

      // Should handle chart creation error gracefully
      expect(wrapper.vm.loading).toBe(false)
    })
  })

  describe('Performance Optimization', () => {
    it('should virtualize large lists of tasks', async () => {
      // Create large number of mock tasks
      const largeTasks = Array.from({ length: 1000 }, (_, i) => mockTask({ id: `task-${i}` }))
      
      vi.mocked(api.get).mockResolvedValue({ data: { tasks: largeTasks } })

      wrapper = mount(TaskDistributionInterface, {
        global: {
          plugins: [pinia]
        }
      })

      await nextTick()

      // Should only render visible tasks
      const renderedTasks = wrapper.findAllComponents(TaskCard)
      expect(renderedTasks.length).toBeLessThan(largeTasks.length)
    })

    it('should debounce search inputs', async () => {
      wrapper = mount(PerformanceAnalyticsViewer, {
        global: {
          plugins: [pinia]
        }
      })

      const searchInput = wrapper.find('[data-testid="search-input"]')
      
      // Rapid typing
      await searchInput.setValue('a')
      await searchInput.setValue('ag')
      await searchInput.setValue('age')
      await searchInput.setValue('agen')
      await searchInput.setValue('agent')

      // Should debounce the search
      await new Promise(resolve => setTimeout(resolve, 300))
      
      expect(wrapper.vm.searchQuery).toBe('agent')
    })

    it('should lazy load chart components', async () => {
      wrapper = mount(PerformanceAnalyticsViewer, {
        global: {
          plugins: [pinia]
        }
      })

      // Charts should not be created until data is loaded
      expect(wrapper.vm.completionChartInstance).toBeNull()

      await wrapper.vm.loadChartData()
      await wrapper.vm.updateCharts()

      expect(wrapper.vm.completionChartInstance).toBeTruthy()
    })
  })

  describe('Accessibility', () => {
    it('should have proper ARIA labels', async () => {
      wrapper = mount(TaskDistributionInterface, {
        global: {
          plugins: [pinia]
        }
      })

      const buttons = wrapper.findAll('button')
      buttons.forEach(button => {
        expect(
          button.attributes('aria-label') || 
          button.text() || 
          button.find('span[class*="sr-only"]').exists()
        ).toBeTruthy()
      })
    })

    it('should support keyboard navigation', async () => {
      wrapper = mount(TaskDistributionInterface, {
        global: {
          plugins: [pinia]
        }
      })

      const firstTask = wrapper.findComponent(TaskCard)
      await firstTask.trigger('keydown', { key: 'Enter' })

      expect(wrapper.vm.selectedTask).toBeTruthy()
    })

    it('should provide screen reader announcements', async () => {
      const webSocketService = useCoordinationWebSocket()
      
      const taskAssignmentMessage = {
        type: 'task_assignment',
        data: {
          task_title: 'Test Task',
          agent_name: 'Test Agent'
        }
      }

      mockWebSocket.onmessage({ data: JSON.stringify(taskAssignmentMessage) })

      // Should announce task assignment
      const announcement = document.querySelector('[aria-live="polite"]')
      expect(announcement?.textContent).toContain('assigned to')
    })
  })
})

describe('Integration Scenarios', () => {
  it('should handle complete task assignment workflow', async () => {
    const wrapper = mount(TaskDistributionInterface, {
      global: {
        plugins: [createTestingPinia({ createSpy: vi.fn })]
      }
    })

    await nextTick()

    // 1. Load initial data
    expect(api.get).toHaveBeenCalledWith('/tasks', expect.any(Object))
    expect(api.get).toHaveBeenCalledWith('/team-coordination/agents')

    // 2. Select a task
    const task = mockTask()
    await wrapper.vm.selectTask(task)
    expect(wrapper.vm.selectedTask).toEqual(task)

    // 3. Get matching suggestions
    expect(wrapper.vm.matchingSuggestions.length).toBeGreaterThan(0)

    // 4. Assign task to suggested agent
    const suggestion = wrapper.vm.matchingSuggestions[0]
    await wrapper.vm.assignTaskToAgent(task, suggestion)
    expect(wrapper.vm.showAssignmentModal).toBe(true)

    // 5. Confirm assignment
    await wrapper.vm.confirmAssignment()
    expect(api.post).toHaveBeenCalledWith('/team-coordination/tasks/distribute', 
      expect.objectContaining({
        task_title: task.task_title,
        target_agent_id: suggestion.agent_id
      })
    )
  })

  it('should sync data across multiple components', async () => {
    const webSocketService = useCoordinationWebSocket()
    
    const distributionWrapper = mount(TaskDistributionInterface, {
      global: {
        plugins: [createTestingPinia({ createSpy: vi.fn })]
      }
    })

    const analyticsWrapper = mount(PerformanceAnalyticsViewer, {
      global: {
        plugins: [createTestingPinia({ createSpy: vi.fn })]
      }
    })

    await nextTick()

    // Simulate task assignment via WebSocket
    const assignmentMessage = {
      type: 'task_assignment',
      data: {
        task_id: 'task-1',
        agent_id: 'agent-1'
      }
    }

    mockWebSocket.onmessage({ data: JSON.stringify(assignmentMessage) })

    // Both components should update
    expect(distributionWrapper.vm.loadTasks).toHaveBeenCalled()
    expect(analyticsWrapper.vm.refreshAnalytics).toHaveBeenCalled()
  })
})