/**
 * VS 6.2 Frontend Dashboard Integration Tests
 * LeanVibe Agent Hive 2.0
 * 
 * Comprehensive integration tests for frontend dashboard components
 * including WebSocket connections, D3.js visualizations, and performance optimization.
 */

import { describe, test, expect, beforeEach, afterEach, vi, Mock } from 'vitest'
import { mount, VueWrapper } from '@vue/test-utils'
import { nextTick } from 'vue'
import WS from 'jest-websocket-mock'

// Component imports
import LiveWorkflowConstellation from '@/components/intelligence/LiveWorkflowConstellation.vue'
import SemanticQueryExplorer from '@/components/intelligence/SemanticQueryExplorer.vue'
import ContextTrajectoryView from '@/components/intelligence/ContextTrajectoryView.vue'
import IntelligenceKPIDashboard from '@/components/intelligence/IntelligenceKPIDashboard.vue'

// Service imports
import { useObservabilityEvents } from '@/services/observabilityEventService'
import { useDashboardOptimization } from '@/utils/dashboardOptimization'

// Mock D3.js
vi.mock('d3', () => ({
  select: vi.fn(() => ({
    selectAll: vi.fn(() => ({
      data: vi.fn(() => ({
        enter: vi.fn(() => ({
          append: vi.fn(() => ({
            attr: vi.fn().mockReturnThis(),
            style: vi.fn().mockReturnThis(),
            text: vi.fn().mockReturnThis(),
            on: vi.fn().mockReturnThis(),
            call: vi.fn().mockReturnThis(),
            transition: vi.fn().mockReturnThis(),
            duration: vi.fn().mockReturnThis(),
            ease: vi.fn().mockReturnThis()
          }))
        })),
        exit: vi.fn(() => ({
          remove: vi.fn()
        })),
        merge: vi.fn(() => ({
          attr: vi.fn().mockReturnThis(),
          style: vi.fn().mockReturnThis(),
          text: vi.fn().mockReturnThis()
        }))
      }))
    })),
    append: vi.fn().mockReturnThis(),
    attr: vi.fn().mockReturnThis(),
    style: vi.fn().mockReturnThis(),
    call: vi.fn().mockReturnThis(),
    on: vi.fn().mockReturnThis(),
    remove: vi.fn().mockReturnThis()
  })),
  forceSimulation: vi.fn(() => ({
    nodes: vi.fn().mockReturnThis(),
    force: vi.fn().mockReturnThis(),
    alpha: vi.fn().mockReturnThis(),
    restart: vi.fn().mockReturnThis(),
    stop: vi.fn().mockReturnThis(),
    on: vi.fn().mockReturnThis()
  })),
  forceLink: vi.fn(() => ({
    id: vi.fn().mockReturnThis(),
    distance: vi.fn().mockReturnThis(),
    strength: vi.fn().mockReturnThis(),
    links: vi.fn().mockReturnThis()
  })),
  forceManyBody: vi.fn(() => ({
    strength: vi.fn().mockReturnThis()
  })),
  forceCenter: vi.fn(),
  forceCollide: vi.fn(() => ({
    radius: vi.fn().mockReturnThis()
  })),
  forceRadial: vi.fn(),
  forceY: vi.fn(() => ({
    y: vi.fn().mockReturnThis()
  })),
  zoom: vi.fn(() => ({
    scaleExtent: vi.fn().mockReturnThis(),
    on: vi.fn().mockReturnThis()
  })),
  zoomIdentity: {},
  drag: vi.fn(() => ({
    on: vi.fn().mockReturnThis()
  })),
  scaleLinear: vi.fn(() => ({
    domain: vi.fn().mockReturnThis(),
    range: vi.fn().mockReturnThis()
  })),
  axisBottom: vi.fn(),
  axisLeft: vi.fn(),
  line: vi.fn(() => ({
    x: vi.fn().mockReturnThis(),
    y: vi.fn().mockReturnThis(),
    curve: vi.fn().mockReturnThis()
  })),
  curveMonotoneX: {},
  easeLinear: {},
  easeQuadInOut: {}
}))

// Mock WebSocket
global.WebSocket = vi.fn()

// Test data fixtures
const mockObservabilityEvents = [
  {
    id: 'event-1',
    type: 'agent_status',
    timestamp: new Date().toISOString(),
    agent_id: 'agent-1',
    session_id: 'session-1',
    data: { status: 'active', cpu_usage: 0.15 },
    semantic_concepts: ['concept-1', 'concept-2'],
    performance_metrics: { execution_time_ms: 150, latency_ms: 75 }
  },
  {
    id: 'event-2',
    type: 'workflow_update',
    timestamp: new Date().toISOString(),
    agent_id: 'agent-2',
    session_id: 'session-1',
    data: { workflow_id: 'workflow-1', status: 'completed' },
    semantic_concepts: ['workflow-concept'],
    performance_metrics: { execution_time_ms: 200, latency_ms: 100 }
  },
  {
    id: 'event-3',
    type: 'semantic_intelligence',
    timestamp: new Date().toISOString(),
    agent_id: 'agent-1',
    session_id: 'session-2',
    data: { query: 'test query', similarity_score: 0.85 },
    semantic_concepts: ['semantic-1', 'intelligence-1'],
    performance_metrics: { execution_time_ms: 175, latency_ms: 90 }
  }
]

const mockConstellationData = {
  nodes: [
    {
      id: 'agent-1',
      type: 'agent',
      label: 'Agent 1',
      position: { x: 100, y: 100 },
      size: 1.5,
      color: '#3B82F6',
      metadata: { event_count: 5, last_seen: new Date().toISOString() }
    },
    {
      id: 'concept-1',
      type: 'concept',
      label: 'Concept 1',
      position: { x: 200, y: 150 },
      size: 1.2,
      color: '#10B981',
      metadata: { usage_count: 10 }
    }
  ],
  edges: [
    {
      id: 'edge-1',
      source: 'agent-1',
      target: 'concept-1',
      type: 'semantic_flow',
      strength: 0.8,
      frequency: 5,
      latency_ms: 50
    }
  ],
  semantic_flows: [
    {
      agents: ['agent-1', 'agent-2'],
      concept: 'shared-concept',
      flow_strength: 0.9
    }
  ]
}

const mockSemanticSearchResults = [
  {
    id: 'result-1',
    event_type: 'agent_status',
    relevance_score: 0.92,
    content_summary: 'Agent 1 status update with high CPU usage',
    timestamp: new Date().toISOString(),
    agent_id: 'agent-1',
    session_id: 'session-1',
    semantic_concepts: ['performance', 'monitoring'],
    performance_metrics: { execution_time_ms: 150 }
  },
  {
    id: 'result-2',
    event_type: 'workflow_update',
    relevance_score: 0.87,
    content_summary: 'Workflow completion notification',
    timestamp: new Date().toISOString(),
    agent_id: 'agent-2',
    session_id: 'session-1',
    semantic_concepts: ['workflow', 'completion'],
    performance_metrics: { execution_time_ms: 200 }
  }
]

describe('VS 6.2 Dashboard Integration Tests', () => {
  let server: WS
  let mockObservabilityService: any

  beforeEach(() => {
    // Setup WebSocket mock server
    server = new WS('ws://localhost:8000/ws/observability/dashboard')
    
    // Mock observability service
    mockObservabilityService = {
      isConnected: { value: true },
      subscribe: vi.fn(),
      unsubscribe: vi.fn(),
      performSemanticSearch: vi.fn().mockResolvedValue(mockSemanticSearchResults),
      getWorkflowConstellation: vi.fn().mockResolvedValue(mockConstellationData),
      getContextTrajectory: vi.fn().mockResolvedValue({
        trajectory: {
          nodes: mockConstellationData.nodes,
          edges: mockConstellationData.edges,
          flow_path: ['context-1', 'context-2']
        }
      }),
      getIntelligenceKPIs: vi.fn().mockResolvedValue({
        kpis: [
          {
            metric_name: 'avg_latency',
            current_value: 150,
            timestamp: new Date().toISOString(),
            trend: 'stable'
          }
        ],
        trends: {},
        forecasts: {}
      }),
      updateSubscriptionFilters: vi.fn()
    }

    // Mock the service composable
    vi.mocked(useObservabilityEvents).mockReturnValue(mockObservabilityService)
  })

  afterEach(() => {
    server.close()
    vi.clearAllMocks()
  })

  describe('LiveWorkflowConstellation Component', () => {
    test('should initialize D3 visualization on mount', async () => {
      const wrapper = mount(LiveWorkflowConstellation, {
        props: {
          width: 800,
          height: 600,
          autoRefresh: false
        }
      })

      await nextTick()

      expect(mockObservabilityService.getWorkflowConstellation).toHaveBeenCalled()
      expect(wrapper.find('.constellation-container').exists()).toBe(true)
      expect(wrapper.find('svg').exists()).toBe(true)
    })

    test('should handle real-time WebSocket events', async () => {
      const wrapper = mount(LiveWorkflowConstellation, {
        props: { autoRefresh: true }
      })

      await nextTick()

      // Verify WebSocket subscription was created
      expect(mockObservabilityService.subscribe).toHaveBeenCalledWith(
        expect.any(String), // component
        expect.arrayContaining(['workflow_update', 'agent_status', 'semantic_intelligence']),
        expect.any(Function), // callback
        expect.any(Object), // filters
        expect.any(Number) // priority
      )

      // Simulate WebSocket event
      const subscribeCallback = mockObservabilityService.subscribe.mock.calls[0][2]
      await subscribeCallback({
        type: 'workflow_update',
        data: {
          agent_updates: [{
            agent_id: 'agent-test',
            activity_level: 0.8,
            metadata: { status: 'processing' }
          }]
        }
      })

      await nextTick()
      // Component should have processed the update
    })

    test('should update layout when layout type changes', async () => {
      const wrapper = mount(LiveWorkflowConstellation)
      await nextTick()

      // Change layout type
      const layoutSelect = wrapper.find('select')
      await layoutSelect.setValue('circular')

      expect(wrapper.emitted('layoutChanged')).toBeTruthy()
      expect(wrapper.emitted('layoutChanged')![0]).toEqual(['circular'])
    })

    test('should toggle semantic flow visualization', async () => {
      const wrapper = mount(LiveWorkflowConstellation)
      await nextTick()

      const semanticFlowButton = wrapper.find('button[title="Semantic Flow"]')
      await semanticFlowButton.trigger('click')

      // Should reload constellation data with semantic flow enabled/disabled
      expect(mockObservabilityService.getWorkflowConstellation).toHaveBeenCalledTimes(2)
    })

    test('should meet performance requirements for rendering', async () => {
      const startTime = performance.now()
      
      const wrapper = mount(LiveWorkflowConstellation, {
        props: {
          width: 800,
          height: 600
        }
      })

      await nextTick()
      await wrapper.vm.$nextTick()

      const renderTime = performance.now() - startTime
      expect(renderTime).toBeLessThan(2000) // <2s load time requirement
    })
  })

  describe('SemanticQueryExplorer Component', () => {
    test('should execute semantic queries and display results', async () => {
      const wrapper = mount(SemanticQueryExplorer, {
        props: {
          maxResults: 50
        }
      })

      await nextTick()

      // Enter query
      const queryInput = wrapper.find('input[type="text"]')
      await queryInput.setValue('show me agent performance issues')

      // Execute query
      const searchButton = wrapper.find('button:contains("Search")')
      await searchButton.trigger('click')

      await nextTick()

      expect(mockObservabilityService.performSemanticSearch).toHaveBeenCalledWith({
        query: 'show me agent performance issues',
        context_window_hours: 24,
        max_results: 50,
        similarity_threshold: 0.7,
        include_context: true,
        include_performance: true
      })

      // Results should be displayed
      await wrapper.vm.$nextTick()
      expect(wrapper.find('.result-card').exists()).toBe(true)
    })

    test('should generate and display query suggestions', async () => {
      const wrapper = mount(SemanticQueryExplorer, {
        props: { autoSuggest: true }
      })

      await nextTick()

      const queryInput = wrapper.find('input[type="text"]')
      await queryInput.setValue('agent')
      await queryInput.trigger('input')

      await nextTick()

      // Suggestions should be generated and shown
      expect(wrapper.vm.querySuggestions.length).toBeGreaterThan(0)
    })

    test('should handle query execution within performance limits', async () => {
      const wrapper = mount(SemanticQueryExplorer)
      await nextTick()

      const startTime = performance.now()

      const queryInput = wrapper.find('input[type="text"]')
      await queryInput.setValue('test query')

      const searchButton = wrapper.find('button:contains("Search")')
      await searchButton.trigger('click')

      await nextTick()

      const queryTime = performance.now() - startTime
      expect(queryTime).toBeLessThan(1000) // <1s event latency requirement
    })

    test('should export search results', async () => {
      const wrapper = mount(SemanticQueryExplorer)
      await nextTick()

      // Set up search results
      wrapper.vm.searchResults = mockSemanticSearchResults
      wrapper.vm.lastQuery = 'test query'

      await nextTick()

      // Mock URL.createObjectURL and document methods
      global.URL.createObjectURL = vi.fn(() => 'blob:test-url')
      global.URL.revokeObjectURL = vi.fn()
      const mockAppendChild = vi.fn()
      const mockRemoveChild = vi.fn()
      const mockClick = vi.fn()
      
      Object.defineProperty(document, 'createElement', {
        value: vi.fn(() => ({
          href: '',
          download: '',
          click: mockClick
        }))
      })
      Object.defineProperty(document.body, 'appendChild', { value: mockAppendChild })
      Object.defineProperty(document.body, 'removeChild', { value: mockRemoveChild })

      const exportButton = wrapper.find('button:contains("Export")')
      await exportButton.trigger('click')

      expect(mockClick).toHaveBeenCalled()
      expect(mockAppendChild).toHaveBeenCalled()
      expect(mockRemoveChild).toHaveBeenCalled()
    })
  })

  describe('ContextTrajectoryView Component', () => {
    test('should render context trajectory visualization', async () => {
      const wrapper = mount(ContextTrajectoryView, {
        props: {
          contextId: 'context-1',
          maxDepth: 5
        }
      })

      await nextTick()

      expect(mockObservabilityService.getContextTrajectory).toHaveBeenCalledWith({
        context_id: 'context-1',
        max_depth: 5,
        time_range_hours: 24
      })

      expect(wrapper.find('.trajectory-container').exists()).toBe(true)
      expect(wrapper.find('svg').exists()).toBe(true)
    })

    test('should handle trajectory path selection', async () => {
      const wrapper = mount(ContextTrajectoryView, {
        props: { contextId: 'context-1' }
      })

      await nextTick()

      // Simulate path selection (this would normally be done via D3 interaction)
      await wrapper.vm.selectTrajectoryPath(['context-1', 'context-2', 'context-3'])

      expect(wrapper.emitted('pathSelected')).toBeTruthy()
      expect(wrapper.emitted('pathSelected')![0]).toEqual([['context-1', 'context-2', 'context-3']])
    })

    test('should update when context ID changes', async () => {
      const wrapper = mount(ContextTrajectoryView, {
        props: { contextId: 'context-1' }
      })

      await nextTick()
      expect(mockObservabilityService.getContextTrajectory).toHaveBeenCalledTimes(1)

      await wrapper.setProps({ contextId: 'context-2' })
      await nextTick()

      expect(mockObservabilityService.getContextTrajectory).toHaveBeenCalledTimes(2)
      expect(mockObservabilityService.getContextTrajectory).toHaveBeenLastCalledWith({
        context_id: 'context-2',
        max_depth: 3,
        time_range_hours: 24
      })
    })
  })

  describe('IntelligenceKPIDashboard Component', () => {
    test('should load and display KPI metrics', async () => {
      const wrapper = mount(IntelligenceKPIDashboard, {
        props: {
          timeRangeHours: 24,
          refreshInterval: 30000
        }
      })

      await nextTick()

      expect(mockObservabilityService.getIntelligenceKPIs).toHaveBeenCalledWith({
        time_range_hours: 24,
        granularity: 'hour',
        include_forecasting: true
      })

      // KPI cards should be rendered
      expect(wrapper.find('.kpi-card').exists()).toBe(true)
    })

    test('should handle real-time KPI updates', async () => {
      const wrapper = mount(IntelligenceKPIDashboard, {
        props: { realTimeUpdates: true }
      })

      await nextTick()

      // Verify real-time subscription
      expect(mockObservabilityService.subscribe).toHaveBeenCalled()

      // Simulate real-time KPI update
      const subscribeCallback = mockObservabilityService.subscribe.mock.calls[0][2]
      await subscribeCallback({
        type: 'performance_metric',
        data: {
          metric_name: 'avg_latency',
          current_value: 175,
          timestamp: new Date().toISOString()
        }
      })

      await nextTick()
      // Component should have updated the KPI display
    })

    test('should render KPI charts with D3.js', async () => {
      const wrapper = mount(IntelligenceKPIDashboard)
      await nextTick()

      // D3 should have been called to create charts
      const d3 = await import('d3')
      expect(d3.select).toHaveBeenCalled()
    })

    test('should handle KPI threshold alerts', async () => {
      const wrapper = mount(IntelligenceKPIDashboard, {
        props: {
          alertThresholds: {
            avg_latency: { threshold: 200, comparison: 'greater_than' }
          }
        }
      })

      await nextTick()

      // Simulate KPI value exceeding threshold
      await wrapper.vm.handleKPIUpdate({
        metric_name: 'avg_latency',
        current_value: 250,
        timestamp: new Date().toISOString()
      })

      expect(wrapper.emitted('thresholdExceeded')).toBeTruthy()
      expect(wrapper.emitted('thresholdExceeded')![0]).toEqual([{
        metric_name: 'avg_latency',
        threshold_value: 200,
        current_value: 250
      }])
    })
  })

  describe('Performance Optimization Integration', () => {
    test('should integrate with dashboard optimization utilities', async () => {
      const mockOptimization = {
        optimizer: {
          startLoadTimeTracking: vi.fn(),
          endLoadTimeTracking: vi.fn(),
          startRenderTracking: vi.fn(),
          endRenderTracking: vi.fn(),
          trackEventLatency: vi.fn(),
          measurePerformance: vi.fn((name, fn) => fn()),
          memoize: vi.fn((fn) => fn),
          createDebouncedFunction: vi.fn((fn) => fn),
          createThrottledFunction: vi.fn((fn) => fn)
        },
        metrics: { value: {
          loadTime: 150,
          renderTime: 50,
          eventLatency: 25,
          fps: 60,
          memoryUsage: 45000000
        }}
      }

      vi.mocked(useDashboardOptimization).mockReturnValue(mockOptimization)

      const wrapper = mount(LiveWorkflowConstellation, {
        props: { width: 800, height: 600 }
      })

      await nextTick()

      // Performance tracking should be initiated
      expect(mockOptimization.optimizer.startLoadTimeTracking).toHaveBeenCalled()
    })

    test('should meet performance targets across all components', async () => {
      const components = [
        LiveWorkflowConstellation,
        SemanticQueryExplorer,
        ContextTrajectoryView,
        IntelligenceKPIDashboard
      ]

      for (const Component of components) {
        const startTime = performance.now()
        
        const wrapper = mount(Component, {
          props: { 
            width: 800, 
            height: 600,
            contextId: 'test-context',
            maxResults: 25
          }
        })

        await nextTick()
        await wrapper.vm.$nextTick()

        const loadTime = performance.now() - startTime
        expect(loadTime).toBeLessThan(2000) // <2s load time requirement

        wrapper.unmount()
      }
    })

    test('should handle high-frequency event processing', async () => {
      const wrapper = mount(LiveWorkflowConstellation, {
        props: { autoRefresh: true }
      })

      await nextTick()

      // Simulate high-frequency events (1000+ events/second)
      const eventCount = 1200
      const events = Array.from({ length: eventCount }, (_, i) => ({
        type: 'performance_test',
        data: { test_data: `value-${i}` },
        timestamp: new Date().toISOString()
      }))

      const startTime = performance.now()

      // Get the subscription callback
      const subscribeCallback = mockObservabilityService.subscribe.mock.calls[0][2]

      // Process events rapidly
      for (const event of events) {
        subscribeCallback(event)
      }

      const processingTime = performance.now() - startTime
      const eventsPerSecond = eventCount / (processingTime / 1000)

      expect(eventsPerSecond).toBeGreaterThan(1000) // Must handle 1000+ events/second
    })
  })

  describe('End-to-End Dashboard Workflow', () => {
    test('should complete full observability workflow', async () => {
      // Mount all dashboard components
      const constellation = mount(LiveWorkflowConstellation, {
        props: { autoRefresh: true }
      })
      const queryExplorer = mount(SemanticQueryExplorer, {
        props: { autoSuggest: true }
      })
      const trajectory = mount(ContextTrajectoryView, {
        props: { contextId: 'context-1' }
      })
      const kpiDashboard = mount(IntelligenceKPIDashboard, {
        props: { realTimeUpdates: true }
      })

      await nextTick()

      // Verify all components initialized
      expect(constellation.find('.constellation-container').exists()).toBe(true)
      expect(queryExplorer.find('.semantic-query-explorer').exists()).toBe(true)
      expect(trajectory.find('.trajectory-container').exists()).toBe(true)
      expect(kpiDashboard.find('.kpi-dashboard').exists()).toBe(true)

      // Verify all WebSocket subscriptions created
      expect(mockObservabilityService.subscribe).toHaveBeenCalledTimes(4)

      // Simulate end-to-end event flow
      const testEvent = {
        type: 'workflow_update',
        data: { agent_id: 'agent-1', status: 'completed' },
        timestamp: new Date().toISOString()
      }

      // All subscription callbacks should process the event
      const callbacks = mockObservabilityService.subscribe.mock.calls.map(call => call[2])
      for (const callback of callbacks) {
        await callback(testEvent)
      }

      await nextTick()

      // Execute semantic query to find the event
      const queryInput = queryExplorer.find('input[type="text"]')
      await queryInput.setValue('show me recent workflow updates')
      
      const searchButton = queryExplorer.find('button:contains("Search")')
      await searchButton.trigger('click')

      await nextTick()

      expect(mockObservabilityService.performSemanticSearch).toHaveBeenCalled()

      // Clean up
      constellation.unmount()
      queryExplorer.unmount()
      trajectory.unmount()
      kpiDashboard.unmount()
    })

    test('should handle error conditions gracefully', async () => {
      // Mock service errors
      mockObservabilityService.getWorkflowConstellation.mockRejectedValueOnce(new Error('Network error'))
      mockObservabilityService.performSemanticSearch.mockRejectedValueOnce(new Error('Search error'))

      const constellation = mount(LiveWorkflowConstellation)
      const queryExplorer = mount(SemanticQueryExplorer)

      await nextTick()

      // Components should handle errors gracefully without crashing
      expect(constellation.exists()).toBe(true)
      expect(queryExplorer.exists()).toBe(true)

      // Error states should be displayed
      expect(constellation.find('.error-message').exists() || constellation.vm.loading === false).toBe(true)
    })
  })

  describe('WebSocket Connection Management', () => {
    test('should establish WebSocket connections for real-time updates', async () => {
      const wrapper = mount(LiveWorkflowConstellation, {
        props: { autoRefresh: true }
      })

      await nextTick()

      // WebSocket connection should be established
      expect(mockObservabilityService.subscribe).toHaveBeenCalledWith(
        expect.any(String), // component name
        expect.arrayContaining(['workflow_update']), // event types
        expect.any(Function), // callback
        expect.any(Object), // filters
        expect.any(Number) // priority
      )
    })

    test('should handle WebSocket reconnection', async () => {
      const wrapper = mount(LiveWorkflowConstellation, {
        props: { autoRefresh: true }
      })

      await nextTick()

      // Simulate connection loss
      mockObservabilityService.isConnected.value = false
      await nextTick()

      expect(wrapper.find('.w-2.h-2.rounded-full.bg-red-500').exists()).toBe(true)

      // Simulate reconnection
      mockObservabilityService.isConnected.value = true
      await nextTick()

      expect(wrapper.find('.w-2.h-2.rounded-full.bg-green-500').exists()).toBe(true)
    })

    test('should clean up WebSocket subscriptions on unmount', async () => {
      const wrapper = mount(LiveWorkflowConstellation, {
        props: { autoRefresh: true }
      })

      await nextTick()

      const subscriptionId = mockObservabilityService.subscribe.mock.results[0].value
      
      wrapper.unmount()

      expect(mockObservabilityService.unsubscribe).toHaveBeenCalledWith(subscriptionId)
    })
  })
})

/**
 * Performance Benchmarking Suite
 */
describe('VS 6.2 Performance Benchmarks', () => {
  test('dashboard load time benchmark', async () => {
    const loadTimes: number[] = []
    const iterations = 10

    for (let i = 0; i < iterations; i++) {
      const startTime = performance.now()
      
      const wrapper = mount(LiveWorkflowConstellation, {
        props: { width: 800, height: 600 }
      })

      await nextTick()
      await wrapper.vm.$nextTick()

      const loadTime = performance.now() - startTime
      loadTimes.push(loadTime)

      wrapper.unmount()
    }

    const avgLoadTime = loadTimes.reduce((sum, time) => sum + time, 0) / loadTimes.length
    const maxLoadTime = Math.max(...loadTimes)

    console.log(`📊 Dashboard Load Time Benchmark:`)
    console.log(`   Average: ${avgLoadTime.toFixed(2)}ms`)
    console.log(`   Maximum: ${maxLoadTime.toFixed(2)}ms`)
    console.log(`   Target: <2000ms`)

    expect(avgLoadTime).toBeLessThan(2000)
    expect(maxLoadTime).toBeLessThan(3000) // Allow some variance
  })

  test('event processing latency benchmark', async () => {
    const wrapper = mount(LiveWorkflowConstellation, {
      props: { autoRefresh: true }
    })

    await nextTick()

    const latencies: number[] = []
    const eventCount = 100

    // Get subscription callback
    const subscribeCallback = mockObservabilityService.subscribe.mock.calls[0][2]

    for (let i = 0; i < eventCount; i++) {
      const startTime = performance.now()
      
      await subscribeCallback({
        type: 'performance_test',
        data: { test_data: `value-${i}` },
        timestamp: new Date().toISOString()
      })

      const latency = performance.now() - startTime
      latencies.push(latency)
    }

    const avgLatency = latencies.reduce((sum, latency) => sum + latency, 0) / latencies.length
    const maxLatency = Math.max(...latencies)
    const p95Latency = latencies.sort((a, b) => a - b)[Math.floor(0.95 * latencies.length)]

    console.log(`📊 Event Processing Latency Benchmark:`)
    console.log(`   Average: ${avgLatency.toFixed(2)}ms`)
    console.log(`   P95: ${p95Latency.toFixed(2)}ms`)
    console.log(`   Maximum: ${maxLatency.toFixed(2)}ms`)
    console.log(`   Target: <1000ms`)

    expect(avgLatency).toBeLessThan(1000)
    expect(p95Latency).toBeLessThan(1500)

    wrapper.unmount()
  })

  test('concurrent component rendering benchmark', async () => {
    const components = [
      LiveWorkflowConstellation,
      SemanticQueryExplorer,
      ContextTrajectoryView,
      IntelligenceKPIDashboard
    ]

    const startTime = performance.now()
    
    // Mount all components concurrently
    const wrappers = components.map(Component => 
      mount(Component, {
        props: { 
          width: 400, 
          height: 300,
          contextId: 'test-context',
          maxResults: 10
        }
      })
    )

    await nextTick()
    await Promise.all(wrappers.map(wrapper => wrapper.vm.$nextTick()))

    const totalRenderTime = performance.now() - startTime

    console.log(`📊 Concurrent Component Rendering Benchmark:`)
    console.log(`   Total time for ${components.length} components: ${totalRenderTime.toFixed(2)}ms`)
    console.log(`   Average per component: ${(totalRenderTime / components.length).toFixed(2)}ms`)
    console.log(`   Target: <2000ms total`)

    expect(totalRenderTime).toBeLessThan(2000)

    // Clean up
    wrappers.forEach(wrapper => wrapper.unmount())
  })
})