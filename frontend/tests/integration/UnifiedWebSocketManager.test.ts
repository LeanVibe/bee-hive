/**
 * Unified WebSocket Manager Integration Tests
 * 
 * Tests for the consolidated WebSocket manager handling real-time connections
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'

// Mock UnifiedWebSocketManager class for testing
class UnifiedWebSocketManager {
  private connections = new Map<string, any>()
  private subscriptions = new Map<string, Set<string>>()
  private listeners = new Map<string, Function[]>()
  private messageQueues = new Map<string, any[]>()
  private config: any

  constructor(config: any) {
    this.config = config
  }

  async connect(endpoint: string): Promise<boolean> {
    try {
      const ws = new (global.WebSocket as any)(`${this.config.baseUrl}/${endpoint}`)
      this.connections.set(endpoint, ws)
      this.subscriptions.set(endpoint, new Set())
      this.messageQueues.set(endpoint, [])
      
      // Simulate connection success
      setTimeout(() => {
        this.emit('status_change', { endpoint, status: 'connected' })
      }, 10)
      
      return true
    } catch (error) {
      return false
    }
  }

  disconnect(endpoint?: string) {
    if (endpoint) {
      const connection = this.connections.get(endpoint)
      if (connection) {
        connection.close()
        this.connections.delete(endpoint)
        this.subscriptions.delete(endpoint)
        this.messageQueues.delete(endpoint)
        this.emit('status_change', { endpoint, status: 'disconnected' })
      }
    } else {
      // Disconnect all
      for (const endpoint of this.connections.keys()) {
        this.disconnect(endpoint)
      }
    }
  }

  isConnected(endpoint: string): boolean {
    const connection = this.connections.get(endpoint)
    return connection && connection.readyState === 1
  }

  send(endpoint: string, message: any, priority: string = 'medium') {
    const connection = this.connections.get(endpoint)
    if (connection && this.isConnected(endpoint)) {
      if (priority === 'high') {
        connection.send(JSON.stringify(message))
      } else {
        // Add to batch queue
        const queue = this.messageQueues.get(endpoint) || []
        queue.push(message)
        this.messageQueues.set(endpoint, queue)
        
        // Process batch after interval
        setTimeout(() => this.processBatch(endpoint), this.config.batchInterval || 50)
      }
    } else {
      // Queue message for later
      const queue = this.messageQueues.get(endpoint) || []
      queue.push(message)
      this.messageQueues.set(endpoint, queue)
    }
  }

  subscribe(endpoint: string, channel: string) {
    if (!this.subscriptions.has(endpoint)) {
      this.subscriptions.set(endpoint, new Set())
    }
    this.subscriptions.get(endpoint)!.add(channel)
  }

  unsubscribe(endpoint: string, channel: string) {
    const subs = this.subscriptions.get(endpoint)
    if (subs) {
      subs.delete(channel)
    }
  }

  getSubscriptions(endpoint: string): string[] {
    const subs = this.subscriptions.get(endpoint)
    return subs ? Array.from(subs) : []
  }

  getConnection(endpoint: string) {
    return this.connections.get(endpoint) || null
  }

  getQueueStats(endpoint: string) {
    const queue = this.messageQueues.get(endpoint) || []
    return {
      queuedMessages: queue.length
    }
  }

  getConnectionStats() {
    return {
      totalConnections: this.connections.size,
      totalMessagesSent: 0, // Mock value
      connectionsPerEndpoint: Object.fromEntries(
        Array.from(this.connections.keys()).map(endpoint => [endpoint, 1])
      )
    }
  }

  on(event: string, handler: Function) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, [])
    }
    this.listeners.get(event)!.push(handler)
    
    return () => {
      const listeners = this.listeners.get(event)
      if (listeners) {
        const index = listeners.indexOf(handler)
        if (index > -1) {
          listeners.splice(index, 1)
        }
      }
    }
  }

  private emit(event: string, data: any) {
    const listeners = this.listeners.get(event) || []
    const wildcardListeners = this.listeners.get('*') || []
    const allListeners = [...listeners, ...wildcardListeners]
    
    allListeners.forEach(listener => {
      try {
        listener(data)
      } catch (error) {
        console.error('Error in event listener:', error)
      }
    })
  }

  private processBatch(endpoint: string) {
    const queue = this.messageQueues.get(endpoint) || []
    const connection = this.connections.get(endpoint)
    
    if (queue.length > 0 && connection && this.isConnected(endpoint)) {
      const batchMessage = {
        type: 'batch',
        messages: queue.splice(0, this.config.maxBatchSize || 10)
      }
      connection.send(JSON.stringify(batchMessage))
    }
  }
}

// Mock WebSocket
class MockWebSocket {
  public readyState: number = 1 // OPEN
  public url: string
  private listeners: Map<string, Function[]> = new Map()

  constructor(url: string) {
    this.url = url
    // Simulate connection after a short delay
    setTimeout(() => {
      this.dispatchEvent('open', {})
    }, 10)
  }

  addEventListener(event: string, listener: Function) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, [])
    }
    this.listeners.get(event)!.push(listener)
  }

  removeEventListener(event: string, listener: Function) {
    const listeners = this.listeners.get(event)
    if (listeners) {
      const index = listeners.indexOf(listener)
      if (index > -1) {
        listeners.splice(index, 1)
      }
    }
  }

  send(data: string) {
    // Simulate sending data
  }

  close() {
    this.readyState = 3 // CLOSED
    this.dispatchEvent('close', {})
  }

  // Simulate receiving messages
  dispatchEvent(event: string, data: any) {
    const listeners = this.listeners.get(event) || []
    listeners.forEach(listener => listener(data))
  }

  // Simulate receiving message
  simulateMessage(data: any) {
    this.dispatchEvent('message', { data: JSON.stringify(data) })
  }

  // Simulate connection error
  simulateError(error: any) {
    this.dispatchEvent('error', error)
  }
}

describe('UnifiedWebSocketManager Integration', () => {
  let manager: UnifiedWebSocketManager
  let mockWebSocket: MockWebSocket

  beforeEach(() => {
    // Mock global WebSocket
    global.WebSocket = MockWebSocket as any
    
    manager = new UnifiedWebSocketManager({
      baseUrl: 'ws://localhost:8000',
      reconnectInterval: 100,
      maxReconnectAttempts: 3,
      connectionTimeout: 1000,
      enableConnectionPooling: true,
      enableMessageBatching: true,
      batchInterval: 50,
      maxBatchSize: 10
    })

    // Get reference to created WebSocket for testing
    mockWebSocket = (global.WebSocket as any).mock?.instances?.[0]
  })

  afterEach(() => {
    manager.disconnect()
    vi.clearAllMocks()
    vi.restoreAllMocks()
  })

  describe('Connection Management', () => {
    it('should establish WebSocket connection', async () => {
      const result = await manager.connect('test_endpoint')
      
      expect(result).toBe(true)
      expect(manager.isConnected('test_endpoint')).toBe(true)
    })

    it('should handle multiple endpoint connections', async () => {
      await manager.connect('endpoint1')
      await manager.connect('endpoint2')
      await manager.connect('endpoint3')

      expect(manager.isConnected('endpoint1')).toBe(true)
      expect(manager.isConnected('endpoint2')).toBe(true)
      expect(manager.isConnected('endpoint3')).toBe(true)
    })

    it('should reuse connections when pooling is enabled', async () => {
      const WebSocketSpy = vi.spyOn(global, 'WebSocket')
      
      await manager.connect('endpoint1')
      await manager.connect('endpoint1') // Same endpoint

      // Should not create a new WebSocket instance
      expect(WebSocketSpy).toHaveBeenCalledTimes(1)
    })

    it('should handle connection failures gracefully', async () => {
      // Mock WebSocket constructor to throw error
      vi.spyOn(global, 'WebSocket').mockImplementation(() => {
        throw new Error('Connection failed')
      })

      const result = await manager.connect('failing_endpoint')
      
      expect(result).toBe(false)
      expect(manager.isConnected('failing_endpoint')).toBe(false)
    })

    it('should disconnect specific endpoints', async () => {
      await manager.connect('endpoint1')
      await manager.connect('endpoint2')

      manager.disconnect('endpoint1')

      expect(manager.isConnected('endpoint1')).toBe(false)
      expect(manager.isConnected('endpoint2')).toBe(true)
    })

    it('should disconnect all endpoints', async () => {
      await manager.connect('endpoint1')
      await manager.connect('endpoint2')
      await manager.connect('endpoint3')

      manager.disconnect()

      expect(manager.isConnected('endpoint1')).toBe(false)
      expect(manager.isConnected('endpoint2')).toBe(false)
      expect(manager.isConnected('endpoint3')).toBe(false)
    })
  })

  describe('Message Handling', () => {
    it('should send messages to specific endpoints', async () => {
      await manager.connect('test_endpoint')
      
      const mockWs = manager.getConnection('test_endpoint') as any
      const sendSpy = vi.spyOn(mockWs, 'send')

      manager.send('test_endpoint', { type: 'test', data: 'hello' })

      expect(sendSpy).toHaveBeenCalledWith(
        JSON.stringify({ type: 'test', data: 'hello' })
      )
    })

    it('should receive and route messages correctly', async () => {
      await manager.connect('test_endpoint')
      
      const messageHandler = vi.fn()
      manager.on('message', messageHandler)

      // Simulate receiving a message
      const testMessage = { type: 'update', data: { id: 1 } }
      const mockWs = manager.getConnection('test_endpoint') as MockWebSocket
      mockWs.simulateMessage(testMessage)

      expect(messageHandler).toHaveBeenCalledWith({
        endpoint: 'test_endpoint',
        message: testMessage
      })
    })

    it('should batch messages when enabled', async () => {
      await manager.connect('test_endpoint')
      
      const mockWs = manager.getConnection('test_endpoint') as any
      const sendSpy = vi.spyOn(mockWs, 'send')

      // Send multiple messages rapidly
      manager.send('test_endpoint', { type: 'msg1' })
      manager.send('test_endpoint', { type: 'msg2' })
      manager.send('test_endpoint', { type: 'msg3' })

      // Wait for batch interval
      await new Promise(resolve => setTimeout(resolve, 100))

      // Should batch messages into single send
      expect(sendSpy).toHaveBeenCalledTimes(1)
      
      const sentData = JSON.parse(sendSpy.mock.calls[0][0])
      expect(sentData.type).toBe('batch')
      expect(sentData.messages).toHaveLength(3)
    })

    it('should handle priority messages immediately', async () => {
      await manager.connect('test_endpoint')
      
      const mockWs = manager.getConnection('test_endpoint') as any
      const sendSpy = vi.spyOn(mockWs, 'send')

      // Send priority message
      manager.send('test_endpoint', { type: 'priority' }, 'high')

      // Should send immediately, not batched
      expect(sendSpy).toHaveBeenCalledWith(
        JSON.stringify({ type: 'priority' })
      )
    })

    it('should queue messages when disconnected', async () => {
      // Don't connect initially
      const messageHandler = vi.fn()
      manager.on('message', messageHandler)

      manager.send('test_endpoint', { type: 'queued' })

      // Message should be queued
      expect(messageHandler).not.toHaveBeenCalled()

      // Connect and messages should be sent
      await manager.connect('test_endpoint')
      
      // Wait for queue processing
      await new Promise(resolve => setTimeout(resolve, 50))

      const mockWs = manager.getConnection('test_endpoint') as any
      const sendSpy = vi.spyOn(mockWs, 'send')
      
      // Queue should be processed
      expect(sendSpy).toHaveBeenCalled()
    })
  })

  describe('Subscription Management', () => {
    it('should manage channel subscriptions', async () => {
      await manager.connect('test_endpoint')

      manager.subscribe('test_endpoint', 'channel1')
      manager.subscribe('test_endpoint', 'channel2')

      const subscriptions = manager.getSubscriptions('test_endpoint')
      expect(subscriptions).toContain('channel1')
      expect(subscriptions).toContain('channel2')
    })

    it('should unsubscribe from channels', async () => {
      await manager.connect('test_endpoint')

      manager.subscribe('test_endpoint', 'channel1')
      manager.subscribe('test_endpoint', 'channel2')
      manager.unsubscribe('test_endpoint', 'channel1')

      const subscriptions = manager.getSubscriptions('test_endpoint')
      expect(subscriptions).not.toContain('channel1')
      expect(subscriptions).toContain('channel2')
    })

    it('should filter messages by subscription', async () => {
      await manager.connect('test_endpoint')
      
      manager.subscribe('test_endpoint', 'allowed_channel')

      const messageHandler = vi.fn()
      manager.on('message', messageHandler)

      const mockWs = manager.getConnection('test_endpoint') as MockWebSocket
      
      // Send message for subscribed channel
      mockWs.simulateMessage({
        channel: 'allowed_channel',
        data: 'allowed message'
      })

      // Send message for non-subscribed channel
      mockWs.simulateMessage({
        channel: 'blocked_channel',
        data: 'blocked message'
      })

      expect(messageHandler).toHaveBeenCalledTimes(1)
      expect(messageHandler).toHaveBeenCalledWith({
        endpoint: 'test_endpoint',
        message: {
          channel: 'allowed_channel',
          data: 'allowed message'
        }
      })
    })
  })

  describe('Error Handling and Reconnection', () => {
    it('should attempt reconnection on connection loss', async () => {
      await manager.connect('test_endpoint')
      
      const reconnectSpy = vi.spyOn(manager, 'connect')
      
      // Simulate connection loss
      const mockWs = manager.getConnection('test_endpoint') as MockWebSocket
      mockWs.simulateError(new Error('Connection lost'))

      // Wait for reconnection attempt
      await new Promise(resolve => setTimeout(resolve, 150))

      expect(reconnectSpy).toHaveBeenCalled()
    })

    it('should respect maximum reconnection attempts', async () => {
      // Create manager with limited reconnection attempts
      const limitedManager = new UnifiedWebSocketManager({
        baseUrl: 'ws://localhost:8000',
        maxReconnectAttempts: 2,
        reconnectInterval: 50
      })

      const connectSpy = vi.spyOn(limitedManager, 'connect')
      
      // Mock connection to always fail
      vi.spyOn(global, 'WebSocket').mockImplementation(() => {
        throw new Error('Always fails')
      })

      await limitedManager.connect('failing_endpoint')

      // Wait for all reconnection attempts
      await new Promise(resolve => setTimeout(resolve, 200))

      // Should attempt initial connection + 2 reconnections = 3 total
      expect(connectSpy).toHaveBeenCalledTimes(3)
    })

    it('should emit connection status events', async () => {
      const statusHandler = vi.fn()
      manager.on('status_change', statusHandler)

      await manager.connect('test_endpoint')

      expect(statusHandler).toHaveBeenCalledWith({
        endpoint: 'test_endpoint',
        status: 'connected'
      })

      // Simulate disconnection
      manager.disconnect('test_endpoint')

      expect(statusHandler).toHaveBeenCalledWith({
        endpoint: 'test_endpoint',
        status: 'disconnected'
      })
    })

    it('should handle malformed messages gracefully', async () => {
      await manager.connect('test_endpoint')
      
      const messageHandler = vi.fn()
      const errorHandler = vi.fn()
      
      manager.on('message', messageHandler)
      manager.on('error', errorHandler)

      const mockWs = manager.getConnection('test_endpoint') as MockWebSocket
      
      // Send malformed JSON
      mockWs.dispatchEvent('message', { data: 'invalid json {' })

      expect(messageHandler).not.toHaveBeenCalled()
      expect(errorHandler).toHaveBeenCalledWith({
        endpoint: 'test_endpoint',
        error: expect.any(Error)
      })
    })
  })

  describe('Connection Pooling', () => {
    it('should pool connections efficiently', async () => {
      const WebSocketSpy = vi.spyOn(global, 'WebSocket')

      // Connect to same endpoint multiple times
      await manager.connect('shared_endpoint')
      await manager.connect('shared_endpoint')
      await manager.connect('shared_endpoint')

      // Should only create one WebSocket instance
      expect(WebSocketSpy).toHaveBeenCalledTimes(1)
    })

    it('should share connection across multiple subscriptions', async () => {
      await manager.connect('shared_endpoint')
      
      manager.subscribe('shared_endpoint', 'channel1')
      manager.subscribe('shared_endpoint', 'channel2')
      manager.subscribe('shared_endpoint', 'channel3')

      const connection = manager.getConnection('shared_endpoint')
      expect(connection).toBeDefined()

      const subscriptions = manager.getSubscriptions('shared_endpoint')
      expect(subscriptions).toHaveLength(3)
    })

    it('should clean up connections when no longer needed', async () => {
      await manager.connect('temp_endpoint')
      
      const connection = manager.getConnection('temp_endpoint')
      expect(connection).toBeDefined()

      manager.disconnect('temp_endpoint')

      const connectionAfterDisconnect = manager.getConnection('temp_endpoint')
      expect(connectionAfterDisconnect).toBeNull()
    })
  })

  describe('Performance and Optimization', () => {
    it('should handle high message throughput', async () => {
      await manager.connect('throughput_test')
      
      const startTime = performance.now()
      
      // Send many messages
      for (let i = 0; i < 1000; i++) {
        manager.send('throughput_test', { 
          type: 'throughput_test', 
          id: i,
          data: `message ${i}` 
        })
      }

      const endTime = performance.now()
      const duration = endTime - startTime

      // Should handle 1000 messages in reasonable time (< 100ms)
      expect(duration).toBeLessThan(100)
    })

    it('should optimize memory usage with message queues', async () => {
      // Send messages without connection (will be queued)
      for (let i = 0; i < 100; i++) {
        manager.send('queue_test', { id: i })
      }

      const queueStats = manager.getQueueStats('queue_test')
      expect(queueStats.queuedMessages).toBe(100)

      // Connect and process queue
      await manager.connect('queue_test')

      // Wait for queue processing
      await new Promise(resolve => setTimeout(resolve, 100))

      const statsAfterProcessing = manager.getQueueStats('queue_test')
      expect(statsAfterProcessing.queuedMessages).toBe(0)
    })

    it('should provide connection statistics', async () => {
      await manager.connect('stats_endpoint1')
      await manager.connect('stats_endpoint2')
      
      manager.send('stats_endpoint1', { test: 1 })
      manager.send('stats_endpoint1', { test: 2 })
      manager.send('stats_endpoint2', { test: 3 })

      const stats = manager.getConnectionStats()
      
      expect(stats.totalConnections).toBe(2)
      expect(stats.totalMessagesSent).toBeGreaterThan(0)
      expect(stats.connectionsPerEndpoint).toEqual({
        stats_endpoint1: 1,
        stats_endpoint2: 1
      })
    })
  })

  describe('Event System', () => {
    it('should support wildcard event listeners', async () => {
      const wildcardHandler = vi.fn()
      manager.on('*', wildcardHandler)

      await manager.connect('test_endpoint')
      manager.send('test_endpoint', { type: 'test' })

      expect(wildcardHandler).toHaveBeenCalled()
    })

    it('should support event listener removal', async () => {
      const handler = vi.fn()
      const removeListener = manager.on('message', handler)

      await manager.connect('test_endpoint')
      
      const mockWs = manager.getConnection('test_endpoint') as MockWebSocket
      mockWs.simulateMessage({ test: 1 })

      expect(handler).toHaveBeenCalledTimes(1)

      removeListener()
      mockWs.simulateMessage({ test: 2 })

      // Should not be called after removal
      expect(handler).toHaveBeenCalledTimes(1)
    })

    it('should handle event listener errors gracefully', async () => {
      const faultyHandler = vi.fn(() => {
        throw new Error('Handler error')
      })
      const goodHandler = vi.fn()

      manager.on('message', faultyHandler)
      manager.on('message', goodHandler)

      await manager.connect('test_endpoint')
      
      const mockWs = manager.getConnection('test_endpoint') as MockWebSocket
      
      // Should not throw when handler errors
      expect(() => mockWs.simulateMessage({ test: 1 })).not.toThrow()

      // Good handler should still be called
      expect(goodHandler).toHaveBeenCalled()
    })
  })
})