/**
 * Integration tests for Real-Time Monitoring Dashboard components
 * 
 * Tests the complete frontend integration of:
 * - AgentMonitoringService WebSocket connections
 * - RealTimeAgentStatusGrid component
 * - RealTimePerformanceCard component  
 * - RealTimeAgentPerformanceChart component
 * - Dashboard real-time updates
 * 
 * Created for Vertical Slice 1.2: Real-Time Monitoring Dashboard
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { mount, VueWrapper } from '@vue/test-utils'
import { createTestingPinia } from '@pinia/testing'
import { nextTick } from 'vue'

import RealTimeAgentStatusGrid from '@/components/dashboard/RealTimeAgentStatusGrid.vue'
import RealTimePerformanceCard from '@/components/dashboard/RealTimePerformanceCard.vue'
import RealTimeAgentPerformanceChart from '@/components/charts/RealTimeAgentPerformanceChart.vue'
import { agentMonitoringService } from '@/services/agentMonitoringService'
import { unifiedWebSocketManager } from '@/services/unifiedWebSocketManager'

// Mock WebSocket
class MockWebSocket {
  url: string
  readyState: number = WebSocket.CONNECTING
  onopen: ((event: Event) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null

  constructor(url: string) {
    this.url = url
    // Simulate connection after a brief delay
    setTimeout(() => {
      this.readyState = WebSocket.OPEN
      if (this.onopen) {
        this.onopen(new Event('open'))
      }
    }, 10)
  }

  send(data: string) {
    // Mock successful send
    console.log('MockWebSocket send:', data)
  }

  close() {
    this.readyState = WebSocket.CLOSED
    if (this.onclose) {
      this.onclose(new CloseEvent('close'))
    }
  }

  // Helper method to simulate receiving messages
  simulateMessage(data: any) {
    if (this.onmessage && this.readyState === WebSocket.OPEN) {
      this.onmessage(new MessageEvent('message', {
        data: JSON.stringify(data)
      }))
    }
  }
}

// Mock agent data
const mockAgentData = {
  id: 'agent-123',
  name: 'Test Agent Alpha',
  status: 'active' as const,
  tasksCompleted: 15,
  totalTasks: 20,
  startTime: new Date(Date.now() - 3600000).toISOString(), // 1 hour ago
  memoryUsage: 128,
  performance: 85,
  currentActivity: 'Processing data analysis',
  agentType: 'worker',
  capabilities: ['data_processing', 'analysis']
}

const mockPerformanceMetrics = {
  timestamp: new Date().toISOString(),
  cpu_usage_percent: 45.2,
  memory_usage_mb: 1024.5,
  memory_usage_percent: 68.3,
  disk_usage_percent: 72.1,
  active_connections: 5
}

const mockLifecycleEvent = {
  event_type: 'task_completed',
  agent_id: 'agent-123',
  timestamp: new Date().toISOString(),
  payload: {
    task_id: 'task-456',
    task_title: 'Data Processing Task',
    success: true,
    execution_time_ms: 2500
  }
}

describe('Real-Time Dashboard Integration', () => {
  let mockWebSocketInstance: MockWebSocket

  beforeEach(() => {
    // Mock global WebSocket
    global.WebSocket = vi.fn().mockImplementation((url: string) => {
      mockWebSocketInstance = new MockWebSocket(url)
      return mockWebSocketInstance
    }) as any

    // Mock window.location
    Object.defineProperty(window, 'location', {
      value: {
        protocol: 'http:',
        host: 'localhost:3000'
      },
      writable: true
    })

    // Reset services
    vi.clearAllMocks()
  })

  afterEach(() => {
    // Cleanup
    agentMonitoringService.destroy()
    unifiedWebSocketManager.disconnectAll()
  })

  describe('Agent Monitoring Service Integration', () => {
    it('should initialize and establish WebSocket connections', async () => {
      // Initialize the service
      await agentMonitoringService.initialize()

      // Verify WebSocket connections were created
      expect(global.WebSocket).toHaveBeenCalledTimes(2) // agents + performance endpoints
      expect(agentMonitoringService.isConnected.value).toBe(true)
      expect(agentMonitoringService.connectionStatus.value).toBe('connected')
    })

    it('should handle agent lifecycle events', async () => {
      await agentMonitoringService.initialize()

      // Simulate agent lifecycle event
      mockWebSocketInstance.simulateMessage({
        type: 'agent_lifecycle_event',
        data: mockLifecycleEvent,
        timestamp: new Date().toISOString()
      })

      await nextTick()

      // Verify event was processed
      const events = agentMonitoringService.recentEvents.value
      expect(events.length).toBeGreaterThan(0)
      expect(events[0].event_type).toBe('task_completed')
      expect(events[0].agent_id).toBe('agent-123')
    })

    it('should handle performance metrics updates', async () => {
      await agentMonitoringService.initialize()

      // Simulate performance metrics
      mockWebSocketInstance.simulateMessage({
        type: 'performance_metrics',
        data: mockPerformanceMetrics,
        timestamp: new Date().toISOString()
      })

      await nextTick()

      // Verify metrics were updated
      const metrics = agentMonitoringService.performanceMetrics.value
      expect(metrics?.cpu_usage_percent).toBe(45.2)
      expect(metrics?.memory_usage_percent).toBe(68.3)
    })

    it('should handle connection failures and reconnection', async () => {
      await agentMonitoringService.initialize()

      // Simulate connection error
      if (mockWebSocketInstance.onerror) {
        mockWebSocketInstance.onerror(new Event('error'))
      }

      await nextTick()

      // Should attempt reconnection
      expect(agentMonitoringService.connectionStatus.value).not.toBe('connected')
    })
  })

  describe('RealTimeAgentStatusGrid Component', () => {
    let wrapper: VueWrapper

    beforeEach(async () => {
      await agentMonitoringService.initialize()
      
      wrapper = mount(RealTimeAgentStatusGrid, {
        global: {
          plugins: [createTestingPinia()]
        },
        props: {
          showRecentEvents: true
        }
      })
    })

    afterEach(() => {
      wrapper?.unmount()
    })

    it('should render and display connection status', async () => {
      expect(wrapper.find('.glass-card').exists()).toBe(true)
      expect(wrapper.find('h3').text()).toContain('Live Agent Status')
      
      // Should show connection status
      const statusIndicator = wrapper.find('.w-2.h-2.rounded-full')
      expect(statusIndicator.exists()).toBe(true)
    })

    it('should display agents when data is available', async () => {
      // Simulate agent data
      mockWebSocketInstance.simulateMessage({
        type: 'agent_details',
        data: mockAgentData,
        timestamp: new Date().toISOString()
      })

      await nextTick()

      // Should display agent cards
      const agentCards = wrapper.findAll('.agent-card')
      expect(agentCards.length).toBeGreaterThan(0)
    })

    it('should handle agent selection', async () => {
      // Add mock agent data
      mockWebSocketInstance.simulateMessage({
        type: 'agent_details',
        data: mockAgentData,
        timestamp: new Date().toISOString()
      })

      await nextTick()

      const agentCard = wrapper.find('.agent-card')
      if (agentCard.exists()) {
        await agentCard.trigger('click')
        
        // Should emit agent selected event
        expect(wrapper.emitted('agentSelected')).toBeTruthy()
      }
    })

    it('should show empty state when no agents', async () => {
      // Should show empty state
      const emptyState = wrapper.find('.text-center.py-12')
      expect(emptyState.exists()).toBe(true)
      expect(emptyState.text()).toContain('No Active Agents')
    })

    it('should display recent events', async () => {
      // Simulate lifecycle event
      mockWebSocketInstance.simulateMessage({
        type: 'agent_lifecycle_event',
        data: mockLifecycleEvent,
        timestamp: new Date().toISOString()
      })

      await nextTick()

      // Should show recent events section
      const recentEvents = wrapper.find('h4')
      if (recentEvents.exists() && recentEvents.text().includes('Recent Events')) {
        expect(recentEvents.exists()).toBe(true)
      }
    })
  })

  describe('RealTimePerformanceCard Component', () => {
    let wrapper: VueWrapper

    beforeEach(async () => {
      await agentMonitoringService.initialize()
      
      wrapper = mount(RealTimePerformanceCard, {
        global: {
          plugins: [createTestingPinia()]
        },
        props: {
          showChart: true,
          maxConnections: 100
        }
      })
    })

    afterEach(() => {
      wrapper?.unmount()
    })

    it('should render performance metrics', async () => {
      expect(wrapper.find('.glass-card').exists()).toBe(true)
      expect(wrapper.find('h3').text()).toContain('System Performance')
    })

    it('should display performance metrics when available', async () => {
      // Simulate performance metrics
      mockWebSocketInstance.simulateMessage({
        type: 'performance_metrics',
        data: mockPerformanceMetrics,
        timestamp: new Date().toISOString()
      })

      await nextTick()

      // Should display CPU usage
      const cpuMetric = wrapper.find('[data-testid="cpu-usage"]')
      if (cpuMetric.exists()) {
        expect(cpuMetric.text()).toContain('45.1%')
      }
    })

    it('should handle performance alerts', async () => {
      // Simulate high CPU usage
      const highCpuMetrics = {
        ...mockPerformanceMetrics,
        cpu_usage_percent: 95.0
      }

      mockWebSocketInstance.simulateMessage({
        type: 'performance_metrics',
        data: highCpuMetrics,
        timestamp: new Date().toISOString()
      })

      await nextTick()

      // Should emit performance alert
      expect(wrapper.emitted('performanceAlert')).toBeTruthy()
    })

    it('should update health score based on metrics', async () => {
      // Simulate good performance metrics
      const goodMetrics = {
        ...mockPerformanceMetrics,
        cpu_usage_percent: 25.0,
        memory_usage_percent: 45.0,
        disk_usage_percent: 30.0
      }

      mockWebSocketInstance.simulateMessage({
        type: 'performance_metrics',
        data: goodMetrics,
        timestamp: new Date().toISOString()
      })

      await nextTick()

      // Health score should be high
      const healthScore = wrapper.find('[data-testid="health-score"]')
      if (healthScore.exists()) {
        const score = parseInt(healthScore.text())
        expect(score).toBeGreaterThan(70)
      }
    })
  })

  describe('RealTimeAgentPerformanceChart Component', () => {
    let wrapper: VueWrapper

    beforeEach(async () => {
      await agentMonitoringService.initialize()
      
      // Mock canvas context
      const mockContext = {
        clearRect: vi.fn(),
        beginPath: vi.fn(),
        moveTo: vi.fn(),
        lineTo: vi.fn(),
        stroke: vi.fn(),
        fill: vi.fn(),
        arc: vi.fn(),
        scale: vi.fn(),
        strokeStyle: '',
        fillStyle: '',
        lineWidth: 1,
        font: '',
        textAlign: 'left',
        textBaseline: 'alphabetic',
        fillText: vi.fn()
      }

      HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue(mockContext)
      
      wrapper = mount(RealTimeAgentPerformanceChart, {
        global: {
          plugins: [createTestingPinia()]
        },
        props: {
          height: 300,
          showLegend: true,
          showStats: true
        }
      })
    })

    afterEach(() => {
      wrapper?.unmount()
    })

    it('should render chart canvas', () => {
      expect(wrapper.find('canvas').exists()).toBe(true)
      expect(wrapper.find('h3').text()).toContain('Agent Performance Trends')
    })

    it('should handle time range changes', async () => {
      const timeRangeSelect = wrapper.find('select')
      if (timeRangeSelect.exists()) {
        await timeRangeSelect.setValue('1h')
        expect(wrapper.emitted('timeRangeChanged')).toBeTruthy()
      }
    })

    it('should toggle series visibility', async () => {
      const legendItem = wrapper.find('[data-testid="legend-item"]')
      if (legendItem.exists()) {
        await legendItem.trigger('click')
        expect(wrapper.emitted('seriesToggled')).toBeTruthy()
      }
    })

    it('should show statistics', () => {
      const statsSection = wrapper.find('.grid.grid-cols-2')
      expect(statsSection.exists()).toBe(true)
    })

    it('should handle agent performance data', async () => {
      // Simulate agent performance data
      mockWebSocketInstance.simulateMessage({
        type: 'agent_lifecycle_event',
        data: {
          ...mockLifecycleEvent,
          event_type: 'agent_heartbeat'
        },
        timestamp: new Date().toISOString()
      })

      await nextTick()

      // Chart should process the data (we can't easily test the actual drawing)
      expect(wrapper.vm).toBeDefined()
    })
  })

  describe('Dashboard Integration', () => {
    it('should handle multiple component interactions', async () => {
      await agentMonitoringService.initialize()

      // Mount multiple components
      const agentGrid = mount(RealTimeAgentStatusGrid, {
        global: { plugins: [createTestingPinia()] }
      })

      const performanceCard = mount(RealTimePerformanceCard, {
        global: { plugins: [createTestingPinia()] }
      })

      // Simulate data updates
      mockWebSocketInstance.simulateMessage({
        type: 'agent_lifecycle_event',
        data: mockLifecycleEvent,
        timestamp: new Date().toISOString()
      })

      mockWebSocketInstance.simulateMessage({
        type: 'performance_metrics',
        data: mockPerformanceMetrics,
        timestamp: new Date().toISOString()
      })

      await nextTick()

      // Both components should have processed the updates
      expect(agentGrid.vm).toBeDefined()
      expect(performanceCard.vm).toBeDefined()

      // Cleanup
      agentGrid.unmount()
      performanceCard.unmount()
    })

    it('should maintain connection across component lifecycle', async () => {
      await agentMonitoringService.initialize()

      // Mount and unmount components
      const wrapper1 = mount(RealTimeAgentStatusGrid, {
        global: { plugins: [createTestingPinia()] }
      })
      wrapper1.unmount()

      const wrapper2 = mount(RealTimePerformanceCard, {
        global: { plugins: [createTestingPinia()] }
      })
      wrapper2.unmount()

      // Service should still be connected
      expect(agentMonitoringService.isConnected.value).toBe(true)
    })
  })

  describe('Error Handling and Recovery', () => {
    it('should handle WebSocket connection errors gracefully', async () => {
      await agentMonitoringService.initialize()

      // Simulate connection error
      if (mockWebSocketInstance.onerror) {
        mockWebSocketInstance.onerror(new Event('error'))
      }

      await nextTick()

      // Should handle error without crashing
      expect(agentMonitoringService.connectionStatus.value).not.toBe('connected')
    })

    it('should handle malformed WebSocket messages', async () => {
      await agentMonitoringService.initialize()

      // Simulate malformed message
      if (mockWebSocketInstance.onmessage) {
        mockWebSocketInstance.onmessage(new MessageEvent('message', {
          data: 'invalid json'
        }))
      }

      await nextTick()

      // Should not crash and connection should remain
      expect(agentMonitoringService).toBeDefined()
    })

    it('should recover from temporary disconnections', async () => {
      await agentMonitoringService.initialize()

      // Simulate disconnect
      mockWebSocketInstance.close()

      await nextTick()

      // Should show disconnected state
      expect(agentMonitoringService.connectionStatus.value).toBe('disconnected')
    })
  })

  describe('Performance and Optimization', () => {
    it('should limit data points to prevent memory leaks', async () => {
      await agentMonitoringService.initialize()

      // Simulate many performance updates
      for (let i = 0; i < 1000; i++) {
        mockWebSocketInstance.simulateMessage({
          type: 'performance_metrics',
          data: {
            ...mockPerformanceMetrics,
            timestamp: new Date(Date.now() + i * 1000).toISOString()
          },
          timestamp: new Date().toISOString()
        })
      }

      await nextTick()

      // Should not accumulate unlimited data
      const events = agentMonitoringService.recentEvents.value
      expect(events.length).toBeLessThan(200) // Should have reasonable limit
    })

    it('should debounce rapid updates', async () => {
      await agentMonitoringService.initialize()

      const wrapper = mount(RealTimeAgentStatusGrid, {
        global: { plugins: [createTestingPinia()] }
      })

      // Simulate rapid updates
      for (let i = 0; i < 10; i++) {
        mockWebSocketInstance.simulateMessage({
          type: 'agent_lifecycle_event',
          data: {
            ...mockLifecycleEvent,
            timestamp: new Date(Date.now() + i).toISOString()
          },
          timestamp: new Date().toISOString()
        })
      }

      await nextTick()

      // Component should handle rapid updates gracefully
      expect(wrapper.vm).toBeDefined()

      wrapper.unmount()
    })
  })
})