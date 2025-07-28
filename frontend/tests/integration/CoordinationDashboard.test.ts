/**
 * Coordination Dashboard Integration Tests
 * 
 * End-to-end integration tests for the complete CORE-4 dashboard functionality
 * covering agent graph, transcript manager, analysis tools, and monitoring components.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { mount, VueWrapper } from '@vue/test-utils'
import { createRouter, createWebHistory } from 'vue-router'
import { createPinia } from 'pinia'
import CoordinationDashboard from '@/views/CoordinationDashboard.vue'
import { DashboardComponent } from '@/types/coordination'
import type { Router } from 'vue-router'

// Mock data for testing
const mockAgentGraphData = {
  nodes: [
    {
      id: 'agent_001',
      type: 'agent',
      name: 'Test Agent 1',
      status: 'active',
      metadata: { session_id: 'session_123' },
      last_updated: '2024-01-01T10:00:00Z'
    },
    {
      id: 'agent_002', 
      type: 'agent',
      name: 'Test Agent 2',
      status: 'active',
      metadata: { session_id: 'session_123' },
      last_updated: '2024-01-01T10:01:00Z'
    }
  ],
  edges: [
    {
      id: 'edge_001',
      source: 'agent_001',
      target: 'agent_002',
      type: 'communication',
      weight: 0.8
    }
  ],
  stats: {
    total_nodes: 2,
    total_edges: 1,
    active_agents: 2
  }
}

const mockTranscriptEvents = [
  {
    id: 'event_001',
    source_agent: 'agent_001',
    target_agent: 'agent_002',
    event_type: 'tool_call',
    session_id: 'session_123',
    timestamp: '2024-01-01T10:00:00Z',
    content: { tool: 'test_tool', parameters: { param1: 'value1' } }
  },
  {
    id: 'event_002',
    source_agent: 'agent_002',
    target_agent: 'agent_001',
    event_type: 'response',
    session_id: 'session_123',
    timestamp: '2024-01-01T10:00:30Z',
    content: { result: 'success', data: 'test data' }
  }
]

const mockAnalysisData = {
  patterns: [
    {
      id: 'pattern_001',
      pattern_type: 'communication_loop',
      severity: 'medium',
      affectedAgents: ['agent_001', 'agent_002'],
      description: 'Circular communication pattern detected'
    }
  ],
  metrics: {
    averageResponseTime: 250,
    throughput: 45,
    errorRate: 0.02
  },
  recommendations: [
    {
      type: 'optimization',
      message: 'Consider caching frequently accessed data',
      priority: 'medium'
    }
  ]
}

const mockMonitoringData = {
  systemHealth: 'healthy',
  components: {
    database: { status: 'healthy', latency: 15 },
    redis: { status: 'healthy', latency: 3 },
    websocket: { status: 'healthy', connections: 42 }
  },
  recentEvents: [
    {
      id: 'sys_event_001',
      type: 'info',
      message: 'System operating normally',
      timestamp: '2024-01-01T10:00:00Z'
    }
  ]
}

describe('CoordinationDashboard Integration', () => {
  let wrapper: VueWrapper<any>
  let router: Router
  let pinia: ReturnType<typeof createPinia>

  beforeEach(async () => {
    // Create router instance
    router = createRouter({
      history: createWebHistory(),
      routes: [
        { path: '/coordination', component: CoordinationDashboard },
        { path: '/coordination/:tab', component: CoordinationDashboard }
      ]
    })

    // Create Pinia store
    pinia = createPinia()

    // Mock API calls
    vi.spyOn(global, 'fetch').mockImplementation(async (url) => {
      const urlStr = url.toString()
      
      if (urlStr.includes('/api/agents/graph')) {
        return new Response(JSON.stringify(mockAgentGraphData))
      } else if (urlStr.includes('/api/events')) {
        return new Response(JSON.stringify({ events: mockTranscriptEvents }))
      } else if (urlStr.includes('/api/analysis')) {
        return new Response(JSON.stringify(mockAnalysisData))
      } else if (urlStr.includes('/api/monitoring')) {
        return new Response(JSON.stringify(mockMonitoringData))
      }
      
      return new Response('{}')
    })

    // Mock WebSocket connections
    const mockWebSocket = {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      send: vi.fn(),
      close: vi.fn(),
      readyState: 1,
    }
    
    vi.stubGlobal('WebSocket', vi.fn(() => mockWebSocket))

    // Mount component
    wrapper = mount(CoordinationDashboard, {
      global: {
        plugins: [router, pinia],
        stubs: {
          AgentGraphVisualization: {
            template: '<div data-testid="agent-graph">Agent Graph</div>',
            props: ['graphData', 'selectedSession']
          },
          TranscriptManager: {
            template: '<div data-testid="transcript-manager">Transcript Manager</div>',
            props: ['events', 'selectedSession']
          },
          AnalysisTools: {
            template: '<div data-testid="analysis-tools">Analysis Tools</div>',
            props: ['analysisData', 'selectedSession']
          },
          SystemMonitoring: {
            template: '<div data-testid="system-monitoring">System Monitoring</div>',
            props: ['monitoringData']
          }
        }
      }
    })

    await router.isReady()
  })

  afterEach(() => {
    wrapper?.unmount()
    vi.restoreAllMocks()
  })

  describe('Component Integration', () => {
    it('should render all CORE-4 components', async () => {
      expect(wrapper.find('[data-testid="coordination-dashboard"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="agent-graph"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="transcript-manager"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="analysis-tools"]').exists()).toBe(true)
      expect(wrapper.find('[data-testid="system-monitoring"]').exists()).toBe(true)
    })

    it('should display session selector with options', async () => {
      const sessionSelector = wrapper.find('[data-testid="session-selector"]')
      expect(sessionSelector.exists()).toBe(true)
      
      // Should include "All Sessions" option
      const options = sessionSelector.findAll('option')
      expect(options.length).toBeGreaterThan(0)
      expect(options[0].text()).toBe('All Sessions')
    })

    it('should show connection status indicator', async () => {
      const connectionStatus = wrapper.find('[data-testid="connection-status"]')
      expect(connectionStatus.exists()).toBe(true)
      
      // Should show connected state initially
      expect(connectionStatus.classes()).toContain('connected')
    })

    it('should display breadcrumb navigation', async () => {
      const breadcrumbs = wrapper.find('[data-testid="breadcrumbs"]')
      expect(breadcrumbs.exists()).toBe(true)
      expect(breadcrumbs.text()).toContain('Dashboard')
    })
  })

  describe('Tab Navigation', () => {
    it('should switch between tabs correctly', async () => {
      // Check initial tab (Graph)
      expect(wrapper.find('[data-testid="tab-graph"]').classes()).toContain('active')
      
      // Click on Transcript tab
      await wrapper.find('[data-testid="tab-transcript"]').trigger('click')
      await wrapper.vm.$nextTick()
      
      expect(wrapper.find('[data-testid="tab-transcript"]').classes()).toContain('active')
      expect(wrapper.find('[data-testid="tab-graph"]').classes()).not.toContain('active')
    })

    it('should maintain tab state when switching sessions', async () => {
      // Switch to Analysis tab
      await wrapper.find('[data-testid="tab-analysis"]').trigger('click')
      await wrapper.vm.$nextTick()
      
      // Change session
      const sessionSelect = wrapper.find('[data-testid="session-selector"] select')
      await sessionSelect.setValue('session_123')
      await wrapper.vm.$nextTick()
      
      // Analysis tab should still be active
      expect(wrapper.find('[data-testid="tab-analysis"]').classes()).toContain('active')
    })

    it('should update URL when switching tabs', async () => {
      await wrapper.find('[data-testid="tab-monitoring"]').trigger('click')
      await wrapper.vm.$nextTick()
      
      expect(router.currentRoute.value.path).toBe('/coordination/monitoring')
    })
  })

  describe('Cross-Component Communication', () => {
    it('should filter data across components when session changes', async () => {
      const sessionSelect = wrapper.find('[data-testid="session-selector"] select')
      await sessionSelect.setValue('session_123')
      await wrapper.vm.$nextTick()
      
      // All components should receive filtered data
      const agentGraph = wrapper.findComponent({ name: 'AgentGraphVisualization' })
      const transcript = wrapper.findComponent({ name: 'TranscriptManager' })
      const analysis = wrapper.findComponent({ name: 'AnalysisTools' })
      
      expect(agentGraph.props('selectedSession')).toBe('session_123')
      expect(transcript.props('selectedSession')).toBe('session_123')
      expect(analysis.props('selectedSession')).toBe('session_123')
    })

    it('should update breadcrumbs based on navigation context', async () => {
      // Navigate to specific agent (simulated)
      await wrapper.vm.navigation.navigateFromGraphNode(
        mockAgentGraphData.nodes[0],
        'investigate'
      )
      await wrapper.vm.$nextTick()
      
      const breadcrumbs = wrapper.find('[data-testid="breadcrumbs"]')
      expect(breadcrumbs.text()).toContain('Agent agent_001')
    })

    it('should correlate data between components using session colors', async () => {
      await wrapper.vm.$nextTick()
      
      // Check that session colors are applied consistently
      const colorElements = wrapper.findAll('[data-session-color]')
      expect(colorElements.length).toBeGreaterThan(0)
      
      // All elements with the same session should have the same color
      const sessionColors = new Set()
      colorElements.forEach(el => {
        if (el.attributes('data-session-id') === 'session_123') {
          sessionColors.add(el.attributes('data-session-color'))
        }
      })
      expect(sessionColors.size).toBe(1) // Should all have the same color
    })
  })

  describe('Real-time Updates', () => {
    it('should establish WebSocket connections for all components', async () => {
      await wrapper.vm.$nextTick()
      
      // Should have established WebSocket connections
      expect(global.WebSocket).toHaveBeenCalledTimes(4) // One for each component
      
      // Verify connection endpoints
      const calls = (global.WebSocket as any).mock.calls
      expect(calls.some((call: any) => call[0].includes('graph'))).toBe(true)
      expect(calls.some((call: any) => call[0].includes('transcript'))).toBe(true)
      expect(calls.some((call: any) => call[0].includes('analysis'))).toBe(true)
      expect(calls.some((call: any) => call[0].includes('monitoring'))).toBe(true)
    })

    it('should handle WebSocket disconnections gracefully', async () => {
      const mockWebSocket = (global.WebSocket as any).mock.results[0].value
      
      // Simulate connection loss
      mockWebSocket.readyState = 3 // CLOSED
      const disconnectHandler = mockWebSocket.addEventListener.mock.calls
        .find((call: any) => call[0] === 'close')?.[1]
      
      if (disconnectHandler) {
        disconnectHandler()
        await wrapper.vm.$nextTick()
      }
      
      // Connection status should show disconnected
      const connectionStatus = wrapper.find('[data-testid="connection-status"]')
      expect(connectionStatus.classes()).toContain('disconnected')
    })

    it('should batch real-time updates for performance', async () => {
      const performanceOptimizer = wrapper.vm.performanceOptimizer
      
      // Schedule multiple updates rapidly
      for (let i = 0; i < 10; i++) {
        performanceOptimizer.scheduleUpdate(
          DashboardComponent.GRAPH,
          'data',
          { update: i },
          'medium'
        )
      }
      
      await new Promise(resolve => setTimeout(resolve, 100))
      
      // Updates should be batched
      expect(performanceOptimizer.metrics.value.processedTasks).toBeLessThan(10)
      expect(performanceOptimizer.metrics.value.batchEfficiency).toBeGreaterThan(0.5)
    })
  })

  describe('Error Handling', () => {
    it('should display error boundaries when components fail', async () => {
      const errorHandler = wrapper.vm.errorHandler
      
      // Simulate component error
      errorHandler.reportError({
        id: 'test_error',
        type: 'network',
        message: 'Failed to load graph data',
        component: DashboardComponent.GRAPH,
        timestamp: new Date().toISOString(),
        recoverable: true
      })
      
      await wrapper.vm.$nextTick()
      
      // Error boundary should be visible
      const errorBoundary = wrapper.find('[data-testid="error-boundary-graph"]')
      expect(errorBoundary.exists()).toBe(true)
      expect(errorBoundary.text()).toContain('Failed to load graph data')
    })

    it('should provide fallback data when components fail', async () => {
      const errorHandler = wrapper.vm.errorHandler
      
      // Get fallback data for graph component
      const fallbackData = errorHandler.applyFallback(
        DashboardComponent.GRAPH,
        {
          id: 'test_error',
          type: 'data',
          message: 'Graph data error',
          component: DashboardComponent.GRAPH,
          timestamp: new Date().toISOString(),
          recoverable: false
        }
      )
      
      expect(fallbackData).toBeDefined()
      expect(fallbackData.nodes).toEqual([])
      expect(fallbackData.edges).toEqual([])
      expect(fallbackData.error).toBe(true)
    })

    it('should attempt error recovery for recoverable errors', async () => {
      const errorHandler = wrapper.vm.errorHandler
      
      const recoverableError = {
        id: 'recoverable_error',
        type: 'network',
        message: 'Network timeout',
        component: DashboardComponent.TRANSCRIPT,
        timestamp: new Date().toISOString(),
        recoverable: true
      }
      
      // Attempt recovery
      const recovered = await errorHandler.attemptErrorRecovery(recoverableError)
      
      // Should attempt recovery (mocked to succeed 70% of the time)
      expect(typeof recovered).toBe('boolean')
    })
  })

  describe('Performance Optimization', () => {
    it('should throttle rapid updates', async () => {
      const performanceOptimizer = wrapper.vm.performanceOptimizer
      
      // Create throttled update function
      const throttledUpdate = performanceOptimizer.throttle(
        'test_throttle',
        vi.fn(),
        100
      )
      
      // Call multiple times rapidly
      throttledUpdate()
      throttledUpdate()
      throttledUpdate()
      
      // Only first call should execute
      expect(throttledUpdate).toHaveBeenCalledTimes(1)
    })

    it('should provide performance recommendations', async () => {
      const performanceOptimizer = wrapper.vm.performanceOptimizer
      
      // Simulate high queue usage
      for (let i = 0; i < 800; i++) {
        performanceOptimizer.scheduleUpdate(
          DashboardComponent.GRAPH,
          'data',
          { test: i },
          'low'
        )
      }
      
      const recommendations = performanceOptimizer.getPerformanceRecommendations()
      
      expect(recommendations.length).toBeGreaterThan(0)
      expect(recommendations.some((r: any) => r.message.includes('queue'))).toBe(true)
    })

    it('should optimize component updates', async () => {
      const performanceOptimizer = wrapper.vm.performanceOptimizer
      
      const optimized = performanceOptimizer.optimizeComponentUpdates(
        DashboardComponent.GRAPH
      )
      
      expect(optimized.throttledUpdate).toBeDefined()
      expect(optimized.debouncedUpdate).toBeDefined()
      expect(optimized.batchUpdate).toBeDefined()
      
      // Test batch update
      optimized.batchUpdate([{ data: 1 }, { data: 2 }, { data: 3 }])
      
      await new Promise(resolve => setTimeout(resolve, 50))
      
      expect(performanceOptimizer.queueLength.value).toBeGreaterThan(0)
    })
  })

  describe('Navigation Integration', () => {
    it('should support contextual navigation between components', async () => {
      const navigation = wrapper.vm.navigation
      
      // Navigate from graph node to transcript
      navigation.navigateFromGraphNode(
        mockAgentGraphData.nodes[0],
        'investigate'
      )
      
      await wrapper.vm.$nextTick()
      
      // Should switch to transcript tab
      expect(wrapper.find('[data-testid="tab-transcript"]').classes()).toContain('active')
      
      // Should update context
      const context = navigation.getCurrentContext()
      expect(context.agentId).toBe('agent_001')
    })

    it('should maintain navigation history', async () => {
      const navigation = wrapper.vm.navigation
      
      // Navigate through different tabs
      navigation.navigateToTab('transcript')
      await wrapper.vm.$nextTick()
      
      navigation.navigateToTab('analysis')
      await wrapper.vm.$nextTick()
      
      // Check history
      const history = navigation.navigationHistory.value
      expect(history.length).toBeGreaterThan(0)
      expect(history[history.length - 1].component).toBe(DashboardComponent.ANALYSIS)
    })

    it('should support going back in navigation', async () => {
      const navigation = wrapper.vm.navigation
      
      // Navigate to different tab
      navigation.navigateToTab('monitoring')
      await wrapper.vm.$nextTick()
      
      const currentTab = navigation.activeTab.value
      expect(currentTab).toBe('monitoring')
      
      // Go back
      navigation.goBack()
      await wrapper.vm.$nextTick()
      
      // Should return to previous tab
      expect(navigation.activeTab.value).toBe('graph')
    })
  })

  describe('Data Synchronization', () => {
    it('should synchronize data across all components', async () => {
      const coordinationService = wrapper.vm.coordinationService
      
      // Update session filter
      coordinationService.setSessionFilter('session_123')
      await wrapper.vm.$nextTick()
      
      // All components should receive filtered data
      const state = coordinationService.state.value
      expect(state.selectedSession).toBe('session_123')
      expect(state.filteredGraphData).toBeDefined()
      expect(state.filteredTranscriptEvents).toBeDefined()
    })

    it('should handle concurrent data updates', async () => {
      const coordinationService = wrapper.vm.coordinationService
      
      // Simulate concurrent updates
      const updates = [
        coordinationService.updateGraphData(mockAgentGraphData),
        coordinationService.updateTranscriptEvents(mockTranscriptEvents),
        coordinationService.updateAnalysisData(mockAnalysisData),
        coordinationService.updateMonitoringData(mockMonitoringData)
      ]
      
      await Promise.all(updates)
      
      // All data should be updated
      const state = coordinationService.state.value
      expect(state.graphData).toBeDefined()
      expect(state.transcriptEvents).toBeDefined()
      expect(state.analysisData).toBeDefined()
      expect(state.monitoringData).toBeDefined()
    })

    it('should maintain data consistency during updates', async () => {
      const coordinationService = wrapper.vm.coordinationService
      
      // Start with initial data
      coordinationService.updateGraphData(mockAgentGraphData)
      
      const initialState = coordinationService.state.value
      const initialNodeCount = initialState.graphData?.nodes?.length || 0
      
      // Update with new data
      const newGraphData = {
        ...mockAgentGraphData,
        nodes: [...mockAgentGraphData.nodes, {
          id: 'agent_003',
          type: 'agent',
          name: 'Test Agent 3',
          status: 'active',
          metadata: { session_id: 'session_456' },
          last_updated: '2024-01-01T10:02:00Z'
        }]
      }
      
      coordinationService.updateGraphData(newGraphData)
      await wrapper.vm.$nextTick()
      
      const updatedState = coordinationService.state.value
      const updatedNodeCount = updatedState.graphData?.nodes?.length || 0
      
      expect(updatedNodeCount).toBe(initialNodeCount + 1)
    })
  })
})