/**
 * Coordination Service Integration Tests
 * 
 * Tests for the cross-component communication and data synchronization service
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { DashboardComponent, type GraphData, type ConversationEvent, type AnalysisData, type MonitoringData } from '@/types/coordination'

// Mock CoordinationService class for testing
class CoordinationService {
  private state = {
    selectedSession: 'all',
    graphData: null as GraphData | null,
    transcriptEvents: [] as ConversationEvent[],
    analysisData: null as AnalysisData | null,
    monitoringData: null as MonitoringData | null,
    filteredGraphData: null as GraphData | null,
    filteredTranscriptEvents: [] as ConversationEvent[]
  }

  private listeners = new Map<string, Function[]>()

  constructor(private webSocketManager?: any) {}

  getState() {
    return this.state
  }

  updateGraphData(data: GraphData) {
    this.state.graphData = data
    this.updateFilteredData()
    this.emit('graph_updated', data)
  }

  updateTranscriptEvents(events: ConversationEvent[]) {
    this.state.transcriptEvents = events
    this.updateFilteredData()
    this.emit('transcript_updated', events)
  }

  updateAnalysisData(data: AnalysisData) {
    this.state.analysisData = data
    this.emit('analysis_updated', data)
  }

  updateMonitoringData(data: MonitoringData) {
    this.state.monitoringData = data
    this.emit('monitoring_updated', data)
  }

  setSessionFilter(sessionId: string) {
    this.state.selectedSession = sessionId
    this.updateFilteredData()
    this.emit('session_changed', sessionId)
  }

  getAvailableSessions(): string[] {
    const sessions = new Set<string>()
    this.state.transcriptEvents.forEach(event => {
      sessions.add(event.session_id)
    })
    return Array.from(sessions)
  }

  initializeRealTimeUpdates() {
    // Mock WebSocket initialization
  }

  handleConnectionError() {
    // Mock error handling
  }

  invalidateCache() {
    this.state.graphData = null
    this.state.transcriptEvents = []
    this.state.analysisData = null
    this.state.monitoringData = null
    this.updateFilteredData()
  }

  on(event: string, listener: Function) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, [])
    }
    this.listeners.get(event)!.push(listener)
    
    return () => {
      const listeners = this.listeners.get(event)
      if (listeners) {
        const index = listeners.indexOf(listener)
        if (index > -1) {
          listeners.splice(index, 1)
        }
      }
    }
  }

  emit(event: string, data: any) {
    const listeners = this.listeners.get(event) || []
    listeners.forEach(listener => {
      try {
        listener(data)
      } catch (error) {
        console.error('Error in event listener:', error)
      }
    })
  }

  cleanup() {
    this.listeners.clear()
    this.state = {
      selectedSession: 'all',
      graphData: null,
      transcriptEvents: [],
      analysisData: null,
      monitoringData: null,
      filteredGraphData: null,
      filteredTranscriptEvents: []
    }
  }

  private updateFilteredData() {
    if (this.state.selectedSession === 'all') {
      this.state.filteredGraphData = this.state.graphData
      this.state.filteredTranscriptEvents = this.state.transcriptEvents
    } else {
      // Filter graph data
      if (this.state.graphData) {
        this.state.filteredGraphData = {
          ...this.state.graphData,
          nodes: this.state.graphData.nodes.filter(node => 
            node.metadata.session_id === this.state.selectedSession
          )
        }
      }

      // Filter transcript events
      this.state.filteredTranscriptEvents = this.state.transcriptEvents.filter(event =>
        event.session_id === this.state.selectedSession
      )
    }
  }
}

describe('CoordinationService Integration', () => {
  let service: CoordinationService
  let mockWebSocketManager: any

  beforeEach(() => {
    mockWebSocketManager = {
      connect: vi.fn(),
      disconnect: vi.fn(),
      subscribe: vi.fn(),
      unsubscribe: vi.fn(),
      send: vi.fn(),
      isConnected: vi.fn(() => true),
      getConnectionStatus: vi.fn(() => ({ connected: true, lastSeen: new Date() })),
      on: vi.fn(),
      off: vi.fn()
    }

    service = new CoordinationService(mockWebSocketManager)
  })

  afterEach(() => {
    service.cleanup()
    vi.clearAllMocks()
  })

  describe('Data Synchronization', () => {
    it('should initialize with empty state', () => {
      const state = service.getState()
      
      expect(state.selectedSession).toBe('all')
      expect(state.graphData).toBeNull()
      expect(state.transcriptEvents).toEqual([])
      expect(state.analysisData).toBeNull()
      expect(state.monitoringData).toBeNull()
    })

    it('should update graph data and trigger listeners', async () => {
      const mockData: GraphData = {
        nodes: [
          {
            id: 'agent_001',
            type: 'agent',
            name: 'Test Agent',
            status: 'active',
            metadata: { session_id: 'session_123' },
            last_updated: '2024-01-01T10:00:00Z'
          }
        ],
        edges: [],
        stats: { total_nodes: 1, total_edges: 0 }
      }

      const listener = vi.fn()
      service.on('graph_updated', listener)

      service.updateGraphData(mockData)

      expect(service.getState().graphData).toEqual(mockData)
      expect(listener).toHaveBeenCalledWith(mockData)
    })

    it('should filter data by session', () => {
      const mockEvents: ConversationEvent[] = [
        {
          id: 'event_001',
          source_agent: 'agent_001',
          target_agent: 'agent_002',
          event_type: 'tool_call',
          session_id: 'session_123',
          timestamp: '2024-01-01T10:00:00Z',
          content: {}
        },
        {
          id: 'event_002',
          source_agent: 'agent_003',
          target_agent: 'agent_004',
          event_type: 'tool_call',
          session_id: 'session_456',
          timestamp: '2024-01-01T10:01:00Z',
          content: {}
        }
      ]

      service.updateTranscriptEvents(mockEvents)
      service.setSessionFilter('session_123')

      const state = service.getState()
      expect(state.filteredTranscriptEvents).toHaveLength(1)
      expect(state.filteredTranscriptEvents[0].session_id).toBe('session_123')
    })

    it('should handle concurrent updates safely', async () => {
      const updates = []
      
      // Simulate concurrent updates
      for (let i = 0; i < 10; i++) {
        updates.push(
          service.updateTranscriptEvents([
            {
              id: `event_${i}`,
              source_agent: 'agent_001',
              target_agent: 'agent_002',
              event_type: 'tool_call',
              session_id: 'session_123',
              timestamp: new Date().toISOString(),
              content: { update: i }
            }
          ])
        )
      }

      await Promise.all(updates)

      const state = service.getState()
      expect(state.transcriptEvents).toHaveLength(10)
    })

    it('should maintain data consistency during rapid updates', async () => {
      const listener = vi.fn()
      service.on('transcript_updated', listener)

      // Rapid fire updates
      for (let i = 0; i < 5; i++) {
        service.updateTranscriptEvents([
          {
            id: `event_${i}`,
            source_agent: 'agent_001',
            target_agent: 'agent_002',
            event_type: 'tool_call',
            session_id: 'session_123',
            timestamp: new Date().toISOString(),
            content: { batch: i }
          }
        ])
      }

      // Should have called listener for each update
      expect(listener).toHaveBeenCalledTimes(5)
      
      const finalState = service.getState()
      expect(finalState.transcriptEvents).toHaveLength(5)
    })
  })

  describe('Session Management', () => {
    it('should extract unique sessions from data', () => {
      const mockEvents: ConversationEvent[] = [
        {
          id: 'event_001',
          source_agent: 'agent_001',
          target_agent: 'agent_002',
          event_type: 'tool_call',
          session_id: 'session_123',
          timestamp: '2024-01-01T10:00:00Z',
          content: {}
        },
        {
          id: 'event_002',
          source_agent: 'agent_003',
          target_agent: 'agent_004',
          event_type: 'tool_call',
          session_id: 'session_456',
          timestamp: '2024-01-01T10:01:00Z',
          content: {}
        },
        {
          id: 'event_003',
          source_agent: 'agent_001',
          target_agent: 'agent_003',
          event_type: 'response',
          session_id: 'session_123',
          timestamp: '2024-01-01T10:02:00Z',
          content: {}
        }
      ]

      service.updateTranscriptEvents(mockEvents)

      const sessions = service.getAvailableSessions()
      expect(sessions).toHaveLength(2)
      expect(sessions).toContain('session_123')
      expect(sessions).toContain('session_456')
    })

    it('should notify when session filter changes', () => {
      const listener = vi.fn()
      service.on('session_changed', listener)

      service.setSessionFilter('session_123')

      expect(listener).toHaveBeenCalledWith('session_123')
    })

    it('should update filtered data when session changes', () => {
      const mockGraphData: GraphData = {
        nodes: [
          {
            id: 'agent_001',
            type: 'agent',
            name: 'Agent 1',
            status: 'active',
            metadata: { session_id: 'session_123' },
            last_updated: '2024-01-01T10:00:00Z'
          },
          {
            id: 'agent_002',
            type: 'agent',
            name: 'Agent 2',
            status: 'active',
            metadata: { session_id: 'session_456' },
            last_updated: '2024-01-01T10:00:00Z'
          }
        ],
        edges: [],
        stats: { total_nodes: 2, total_edges: 0 }
      }

      service.updateGraphData(mockGraphData)
      service.setSessionFilter('session_123')

      const state = service.getState()
      expect(state.filteredGraphData?.nodes).toHaveLength(1)
      expect(state.filteredGraphData?.nodes[0].metadata.session_id).toBe('session_123')
    })
  })

  describe('WebSocket Integration', () => {
    it('should establish WebSocket connections for real-time updates', () => {
      service.initializeRealTimeUpdates()

      // Should subscribe to all relevant channels
      expect(mockWebSocketManager.subscribe).toHaveBeenCalledWith('graph_updates')
      expect(mockWebSocketManager.subscribe).toHaveBeenCalledWith('transcript_updates')
      expect(mockWebSocketManager.subscribe).toHaveBeenCalledWith('analysis_updates')
      expect(mockWebSocketManager.subscribe).toHaveBeenCalledWith('monitoring_updates')
    })

    it('should handle WebSocket messages correctly', () => {
      service.initializeRealTimeUpdates()

      // Simulate WebSocket message
      const onCall = mockWebSocketManager.on.mock.calls.find(
        (call: any) => call[0] === 'message'
      )
      expect(onCall).toBeDefined()

      const messageHandler = onCall[1]
      
      // Test graph update message
      messageHandler({
        channel: 'graph_updates',
        data: {
          nodes: [{
            id: 'agent_new',
            type: 'agent',
            name: 'New Agent',
            status: 'active',
            metadata: { session_id: 'session_789' },
            last_updated: '2024-01-01T10:00:00Z'
          }],
          edges: [],
          stats: { total_nodes: 1, total_edges: 0 }
        }
      })

      const state = service.getState()
      expect(state.graphData?.nodes).toHaveLength(1)
      expect(state.graphData?.nodes[0].id).toBe('agent_new')
    })

    it('should handle WebSocket disconnections gracefully', () => {
      service.initializeRealTimeUpdates()

      // Simulate connection error
      const onCall = mockWebSocketManager.on.mock.calls.find(
        (call: any) => call[0] === 'error'
      )
      expect(onCall).toBeDefined()

      const errorHandler = onCall[1]
      errorHandler(new Error('Connection lost'))

      // Should handle error gracefully (no throws)
      expect(() => service.getState()).not.toThrow()
    })
  })

  describe('Event System', () => {
    it('should support event subscription and emission', () => {
      const listener1 = vi.fn()
      const listener2 = vi.fn()

      service.on('test_event', listener1)
      service.on('test_event', listener2)

      service.emit('test_event', { test: 'data' })

      expect(listener1).toHaveBeenCalledWith({ test: 'data' })
      expect(listener2).toHaveBeenCalledWith({ test: 'data' })
    })

    it('should support event unsubscription', () => {
      const listener = vi.fn()

      const unsubscribe = service.on('test_event', listener)
      unsubscribe()

      service.emit('test_event', { test: 'data' })

      expect(listener).not.toHaveBeenCalled()
    })

    it('should handle errors in event listeners gracefully', () => {
      const faultyListener = vi.fn(() => {
        throw new Error('Listener error')
      })
      const goodListener = vi.fn()

      service.on('test_event', faultyListener)
      service.on('test_event', goodListener)

      // Should not throw when listener errors
      expect(() => service.emit('test_event', {})).not.toThrow()
      
      // Good listener should still be called
      expect(goodListener).toHaveBeenCalled()
    })
  })

  describe('Cache Management', () => {
    it('should cache data efficiently', () => {
      const mockData: GraphData = {
        nodes: [],
        edges: [],
        stats: { total_nodes: 0, total_edges: 0 }
      }

      // First update should cache data
      service.updateGraphData(mockData)
      expect(service.getState().graphData).toEqual(mockData)

      // Same data should not trigger update
      const listener = vi.fn()
      service.on('graph_updated', listener)
      
      service.updateGraphData(mockData)
      expect(listener).not.toHaveBeenCalled()
    })

    it('should expire cached data appropriately', async () => {
      const mockAnalysisData: AnalysisData = {
        patterns: [],
        metrics: { averageResponseTime: 100, throughput: 50, errorRate: 0.01 },
        recommendations: []
      }

      service.updateAnalysisData(mockAnalysisData)
      
      // Wait for cache expiration (simulate)
      await new Promise(resolve => setTimeout(resolve, 50))
      
      const state = service.getState()
      expect(state.analysisData).toEqual(mockAnalysisData)
    })

    it('should handle cache invalidation correctly', () => {
      const mockData: GraphData = {
        nodes: [{
          id: 'agent_001',
          type: 'agent',
          name: 'Test Agent',
          status: 'active',
          metadata: { session_id: 'session_123' },
          last_updated: '2024-01-01T10:00:00Z'
        }],
        edges: [],
        stats: { total_nodes: 1, total_edges: 0 }
      }

      service.updateGraphData(mockData)
      service.invalidateCache()

      // Cache should be cleared
      const state = service.getState()
      expect(state.graphData).toBeNull()
    })
  })

  describe('Error Handling', () => {
    it('should handle malformed data gracefully', () => {
      const malformedData = {
        nodes: 'invalid',
        edges: null,
        stats: undefined
      } as any

      expect(() => service.updateGraphData(malformedData)).not.toThrow()
      
      // Should maintain previous valid state
      const state = service.getState()
      expect(state.graphData).toBeNull() // Should remain null if no valid data was set
    })

    it('should validate data before updates', () => {
      const invalidEvent = {
        id: '', // Invalid empty ID
        source_agent: 'agent_001',
        target_agent: 'agent_002',
        event_type: 'invalid_type',
        session_id: 'session_123',
        timestamp: 'invalid_date',
        content: null
      } as any

      const listener = vi.fn()
      service.on('transcript_updated', listener)

      service.updateTranscriptEvents([invalidEvent])

      // Should not update with invalid data
      expect(listener).not.toHaveBeenCalled()
      
      const state = service.getState()
      expect(state.transcriptEvents).toHaveLength(0)
    })

    it('should recover from WebSocket errors', () => {
      service.initializeRealTimeUpdates()

      // Simulate WebSocket error
      mockWebSocketManager.isConnected.mockReturnValue(false)

      // Should attempt reconnection
      service.handleConnectionError()
      
      expect(mockWebSocketManager.connect).toHaveBeenCalled()
    })
  })

  describe('Performance', () => {
    it('should handle large datasets efficiently', () => {
      const largeDataset = {
        nodes: Array.from({ length: 1000 }, (_, i) => ({
          id: `agent_${i}`,
          type: 'agent',
          name: `Agent ${i}`,
          status: 'active',
          metadata: { session_id: `session_${i % 10}` },
          last_updated: '2024-01-01T10:00:00Z'
        })),
        edges: Array.from({ length: 500 }, (_, i) => ({
          id: `edge_${i}`,
          source: `agent_${i}`,
          target: `agent_${i + 1}`,
          type: 'communication',
          weight: Math.random()
        })),
        stats: { total_nodes: 1000, total_edges: 500 }
      }

      const startTime = performance.now()
      service.updateGraphData(largeDataset)
      const endTime = performance.now()

      // Should complete within reasonable time (< 100ms)
      expect(endTime - startTime).toBeLessThan(100)

      const state = service.getState()
      expect(state.graphData?.nodes).toHaveLength(1000)
    })

    it('should throttle rapid updates', async () => {
      const listener = vi.fn()
      service.on('transcript_updated', listener)

      // Fire 10 updates in rapid succession
      for (let i = 0; i < 10; i++) {
        service.updateTranscriptEvents([{
          id: `event_${i}`,
          source_agent: 'agent_001',
          target_agent: 'agent_002',
          event_type: 'tool_call',
          session_id: 'session_123',
          timestamp: new Date().toISOString(),
          content: { rapid: i }
        }])
      }

      // Should throttle updates
      expect(listener.mock.calls.length).toBeLessThan(10)
    })
  })
})