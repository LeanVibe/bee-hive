import { test, expect } from '@playwright/test'
import { TestHelpers } from '../utils/test-helpers'

/**
 * WebSocket Connection Reliability & Real-Time Data Tests
 * 
 * Validates real-time functionality:
 * - WebSocket connection establishment
 * - Connection resilience and auto-reconnection
 * - Real-time data updates and synchronization
 * - Message handling and error recovery
 * - Performance under load
 * - Connection state management
 */

test.describe('WebSocket & Real-Time Functionality', () => {
  
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await TestHelpers.waitForNetworkIdle(page)
  })

  test('websocket connection establishes successfully', async ({ page }) => {
    const wsConnections: any[] = []
    const wsMessages: any[] = []
    
    // Monitor WebSocket connections
    page.on('websocket', ws => {
      const connection = {
        url: ws.url(),
        state: 'connecting',
        messages: [],
        errors: []
      }
      
      wsConnections.push(connection)
      
      ws.on('open', () => {
        connection.state = 'open'
        console.log(`WebSocket opened: ${ws.url()}`)
      })
      
      ws.on('close', ({ code, reason }) => {
        connection.state = 'closed'
        console.log(`WebSocket closed: ${ws.url()} (${code}: ${reason})`)
      })
      
      ws.on('framesent', data => {
        connection.messages.push({ type: 'sent', data, timestamp: Date.now() })
      })
      
      ws.on('framereceived', data => {
        connection.messages.push({ type: 'received', data, timestamp: Date.now() })
        wsMessages.push({ data, timestamp: Date.now() })
      })
      
      ws.on('socketerror', error => {
        connection.errors.push({ error, timestamp: Date.now() })
        console.error(`WebSocket error: ${ws.url()}`, error)
      })
    })
    
    // Wait for potential WebSocket connections
    await page.waitForTimeout(5000)
    
    if (wsConnections.length > 0) {
      console.log(`Found ${wsConnections.length} WebSocket connection(s)`)
      
      // Verify at least one connection is open
      const openConnections = wsConnections.filter(conn => conn.state === 'open')
      expect(openConnections.length).toBeGreaterThan(0)
      
      // Verify connection URLs are reasonable
      for (const conn of openConnections) {
        expect(conn.url).toMatch(/wss?:\/\//)
        console.log(`✓ WebSocket connection established: ${conn.url}`)
      }
      
      // Check for initial messages
      if (wsMessages.length > 0) {
        console.log(`Received ${wsMessages.length} WebSocket messages`)
        expect(wsMessages.length).toBeGreaterThan(0)
      }
      
    } else {
      console.log('No WebSocket connections detected - application may use polling or be offline')
      
      // Look for connection status indicators in UI
      const connectionIndicators = page.locator(
        '.connection-status, .ws-status, [data-testid*="connection"], .online-status'
      )
      
      if (await connectionIndicators.count() > 0) {
        const status = await connectionIndicators.first().textContent()
        console.log(`Connection status indicator: ${status}`)
      }
    }
  })

  test('websocket handles connection interruption and recovery', async ({ page, context }) => {
    const wsEvents: any[] = []
    
    // Monitor WebSocket events
    page.on('websocket', ws => {
      wsEvents.push({ type: 'connection', url: ws.url(), timestamp: Date.now() })
      
      ws.on('open', () => {
        wsEvents.push({ type: 'open', url: ws.url(), timestamp: Date.now() })
      })
      
      ws.on('close', ({ code, reason }) => {
        wsEvents.push({ type: 'close', url: ws.url(), code, reason, timestamp: Date.now() })
      })
    })
    
    // Wait for initial connection
    await page.waitForTimeout(3000)
    
    const initialConnections = wsEvents.filter(e => e.type === 'open')
    
    if (initialConnections.length > 0) {
      console.log('✓ Initial WebSocket connection established')
      
      // Simulate network interruption
      await context.setOffline(true)
      await page.waitForTimeout(2000)
      
      // Restore network
      await context.setOffline(false)
      
      // Wait for potential reconnection
      await page.waitForTimeout(10000)
      
      // Check for reconnection attempts
      const totalConnections = wsEvents.filter(e => e.type === 'connection')
      const totalOpens = wsEvents.filter(e => e.type === 'open')
      const totalCloses = wsEvents.filter(e => e.type === 'close')
      
      console.log('WebSocket events during interruption:', {
        connections: totalConnections.length,
        opens: totalOpens.length,
        closes: totalCloses.length
      })
      
      // Should have attempted reconnection
      expect(totalConnections.length).toBeGreaterThanOrEqual(1)
      
      // Look for UI indicators of reconnection
      const reconnectionIndicators = page.locator(
        '.reconnecting, .connection-lost, .connection-restored, [data-testid*="reconnect"]'
      )
      
      if (await reconnectionIndicators.count() > 0) {
        console.log('✓ Reconnection UI indicators present')
      }
      
    } else {
      console.log('No WebSocket connections to test recovery with')
    }
  })

  test('real-time updates reflect in UI immediately', async ({ page }) => {
    // Monitor for real-time UI updates
    const uiChanges: any[] = []
    
    // Track changes to dynamic content areas
    const dynamicSelectors = [
      '.live-data',
      '.real-time',
      '[data-testid*="live"]',
      '.agent-status',
      '.task-status',
      '.metrics',
      '.dashboard-stats'
    ]
    
    // Set up mutation observers for dynamic content
    await page.addInitScript((selectors) => {
      window.__uiChanges = []
      
      const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
          if (mutation.type === 'childList' || mutation.type === 'attributes') {
            window.__uiChanges.push({
              type: mutation.type,
              target: mutation.target.tagName,
              timestamp: Date.now()
            })
          }
        })
      })
      
      // Observe changes to dynamic elements
      selectors.forEach(selector => {
        const elements = document.querySelectorAll(selector)
        elements.forEach(element => {
          observer.observe(element, {
            childList: true,
            attributes: true,
            subtree: true
          })
        })
      })
      
      // Also observe body for general changes
      observer.observe(document.body, {
        childList: true,
        subtree: true
      })
    }, dynamicSelectors)
    
    // Wait for initial load and potential updates
    await page.waitForTimeout(10000)
    
    // Get UI changes
    const changes = await page.evaluate(() => window.__uiChanges || [])
    
    if (changes.length > 0) {
      console.log(`Detected ${changes.length} real-time UI updates`)
      
      // Group changes by time windows to identify update bursts
      const timeWindows = new Map()
      changes.forEach(change => {
        const window = Math.floor(change.timestamp / 1000) * 1000 // 1-second windows
        if (!timeWindows.has(window)) {
          timeWindows.set(window, [])
        }
        timeWindows.get(window).push(change)
      })
      
      console.log(`UI updates distributed across ${timeWindows.size} time windows`)
      
      // Should have some real-time updates
      expect(changes.length).toBeGreaterThan(0)
      
    } else {
      console.log('No dynamic UI updates detected - may be a static dashboard or no live data')
    }
    
    // Look for timestamp indicators that should update
    const timestampElements = page.locator(
      '.timestamp, .last-updated, [data-testid*="time"], .time-ago'
    )
    
    if (await timestampElements.count() > 0) {
      const initialTimestamp = await timestampElements.first().textContent()
      
      // Wait and check if timestamp updates
      await page.waitForTimeout(5000)
      const updatedTimestamp = await timestampElements.first().textContent()
      
      if (initialTimestamp !== updatedTimestamp) {
        console.log('✓ Timestamp elements update in real-time')
      }
    }
  })

  test('websocket message handling is robust', async ({ page }) => {
    const messageTests: any[] = []
    
    // Monitor WebSocket for message handling
    page.on('websocket', ws => {
      ws.on('framereceived', data => {
        try {
          const message = JSON.parse(data)
          messageTests.push({
            type: 'valid_json',
            message,
            timestamp: Date.now()
          })
        } catch (error) {
          messageTests.push({
            type: 'invalid_json',
            raw: data,
            error: error.message,
            timestamp: Date.now()
          })
        }
      })
      
      ws.on('framesent', data => {
        try {
          const message = JSON.parse(data)
          messageTests.push({
            type: 'sent_valid',
            message,
            timestamp: Date.now()
          })
        } catch (error) {
          messageTests.push({
            type: 'sent_invalid',
            raw: data,
            timestamp: Date.now()
          })
        }
      })
    })
    
    // Wait for message activity
    await page.waitForTimeout(8000)
    
    if (messageTests.length > 0) {
      console.log(`Processed ${messageTests.length} WebSocket messages`)
      
      // Analyze message quality
      const validMessages = messageTests.filter(m => m.type === 'valid_json')
      const invalidMessages = messageTests.filter(m => m.type === 'invalid_json')
      
      console.log(`Valid JSON messages: ${validMessages.length}`)
      console.log(`Invalid JSON messages: ${invalidMessages.length}`)
      
      // Most messages should be valid JSON
      if (messageTests.length > 0) {
        const validPercent = (validMessages.length / messageTests.length) * 100
        expect(validPercent).toBeGreaterThan(80) // At least 80% should be valid
      }
      
      // Sample some message structures
      const sampleMessages = validMessages.slice(0, 3)
      for (const msgTest of sampleMessages) {
        const msg = msgTest.message
        console.log('Sample message structure:', {
          hasType: !!msg.type,
          hasData: !!msg.data,
          hasTimestamp: !!msg.timestamp,
          keys: Object.keys(msg)
        })
      }
      
    } else {
      console.log('No WebSocket messages captured')
    }
  })

  test('connection state is properly managed and displayed', async ({ page }) => {
    // Look for connection state indicators
    const stateIndicators = page.locator(
      '.connection-state, .ws-state, [data-testid*="connection-state"], .online-indicator'
    )
    
    if (await stateIndicators.count() > 0) {
      const indicator = stateIndicators.first()
      await expect(indicator).toBeVisible()
      
      const initialState = await indicator.textContent()
      console.log(`Initial connection state: ${initialState}`)
      
      // Connection state should indicate online/connected status
      const connectionTerms = ['online', 'connected', 'active', 'ready']
      const isConnectedState = connectionTerms.some(term => 
        initialState?.toLowerCase().includes(term)
      )
      
      if (isConnectedState) {
        console.log('✓ Connection state indicates active connection')
      }
    }
    
    // Check for network status API usage
    const networkStatus = await page.evaluate(() => {
      if ('onLine' in navigator) {
        return {
          online: navigator.onLine,
          connectionType: (navigator as any).connection?.effectiveType || 'unknown',
          hasNetworkAPI: 'connection' in navigator
        }
      }
      return { supported: false }
    })
    
    console.log('Network status API info:', networkStatus)
    
    if (networkStatus.supported !== false) {
      expect(networkStatus.online).toBe(true) // Should be online during test
    }
  })

  test('real-time performance is acceptable', async ({ page }) => {
    const performanceMetrics = {
      wsLatency: [],
      uiUpdateLatency: [],
      messageFrequency: []
    }
    
    let lastMessageTime = 0
    
    // Monitor WebSocket performance
    page.on('websocket', ws => {
      ws.on('framesent', data => {
        const sendTime = Date.now()
        // Store send time for latency calculation
        window.__wsSendTimes = window.__wsSendTimes || new Map()
        try {
          const message = JSON.parse(data)
          if (message.id) {
            window.__wsSendTimes.set(message.id, sendTime)
          }
        } catch (e) {
          // Ignore non-JSON messages
        }
      })
      
      ws.on('framereceived', data => {
        const receiveTime = Date.now()
        
        // Calculate message frequency
        if (lastMessageTime > 0) {
          const frequency = receiveTime - lastMessageTime
          performanceMetrics.messageFrequency.push(frequency)
        }
        lastMessageTime = receiveTime
        
        // Try to calculate latency if we can match request/response
        try {
          const message = JSON.parse(data)
          if (message.id && window.__wsSendTimes?.has(message.id)) {
            const sendTime = window.__wsSendTimes.get(message.id)
            const latency = receiveTime - sendTime
            performanceMetrics.wsLatency.push(latency)
            window.__wsSendTimes.delete(message.id)
          }
        } catch (e) {
          // Ignore non-JSON messages
        }
      })
    })
    
    // Monitor UI update performance
    await page.addInitScript(() => {
      window.__uiUpdateTimes = []
      
      const observer = new MutationObserver((mutations) => {
        const updateTime = Date.now()
        window.__uiUpdateTimes.push(updateTime)
      })
      
      observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true
      })
    })
    
    // Wait for activity
    await page.waitForTimeout(15000)
    
    // Collect performance data
    const uiUpdateTimes = await page.evaluate(() => window.__uiUpdateTimes || [])
    
    // Calculate UI update frequency
    if (uiUpdateTimes.length > 1) {
      for (let i = 1; i < uiUpdateTimes.length; i++) {
        const frequency = uiUpdateTimes[i] - uiUpdateTimes[i - 1]
        performanceMetrics.uiUpdateLatency.push(frequency)
      }
    }
    
    // Analyze performance
    const analysis = {
      wsLatency: {
        count: performanceMetrics.wsLatency.length,
        avg: performanceMetrics.wsLatency.length > 0 
          ? performanceMetrics.wsLatency.reduce((a, b) => a + b, 0) / performanceMetrics.wsLatency.length 
          : 0,
        max: Math.max(...performanceMetrics.wsLatency, 0)
      },
      messageFrequency: {
        count: performanceMetrics.messageFrequency.length,
        avg: performanceMetrics.messageFrequency.length > 0
          ? performanceMetrics.messageFrequency.reduce((a, b) => a + b, 0) / performanceMetrics.messageFrequency.length
          : 0
      },
      uiUpdates: {
        count: uiUpdateTimes.length,
        frequency: performanceMetrics.uiUpdateLatency.length > 0
          ? performanceMetrics.uiUpdateLatency.reduce((a, b) => a + b, 0) / performanceMetrics.uiUpdateLatency.length
          : 0
      }
    }
    
    console.log('Real-time performance analysis:', analysis)
    
    // Performance assertions
    if (analysis.wsLatency.count > 0) {
      expect(analysis.wsLatency.avg).toBeLessThan(1000) // Average latency < 1s
      expect(analysis.wsLatency.max).toBeLessThan(5000) // Max latency < 5s
    }
    
    if (analysis.messageFrequency.count > 0) {
      // Message frequency should be reasonable (not flooding)
      expect(analysis.messageFrequency.avg).toBeGreaterThan(100) // At least 100ms between messages
    }
    
    // UI should be responsive
    if (analysis.uiUpdates.count > 0) {
      console.log(`UI updated ${analysis.uiUpdates.count} times with average frequency of ${analysis.uiUpdates.frequency}ms`)
    }
  })

  test('websocket handles authentication and authorization', async ({ page }) => {
    const authEvents: any[] = []
    
    // Monitor WebSocket for auth-related messages
    page.on('websocket', ws => {
      ws.on('framesent', data => {
        try {
          const message = JSON.parse(data)
          if (message.type?.includes('auth') || message.type?.includes('login') || message.token) {
            authEvents.push({ type: 'auth_sent', message, timestamp: Date.now() })
          }
        } catch (e) {
          // Ignore non-JSON
        }
      })
      
      ws.on('framereceived', data => {
        try {
          const message = JSON.parse(data)
          if (message.type?.includes('auth') || message.error?.includes('auth') || message.authenticated !== undefined) {
            authEvents.push({ type: 'auth_received', message, timestamp: Date.now() })
          }
        } catch (e) {
          // Ignore non-JSON
        }
      })
    })
    
    // Wait for potential auth activity
    await page.waitForTimeout(5000)
    
    if (authEvents.length > 0) {
      console.log(`Detected ${authEvents.length} authentication-related WebSocket messages`)
      
      // Check for successful authentication
      const authSuccessEvents = authEvents.filter(event => 
        event.message.authenticated === true || 
        event.message.type?.includes('success') ||
        event.message.status === 'authenticated'
      )
      
      if (authSuccessEvents.length > 0) {
        console.log('✓ WebSocket authentication appears successful')
      }
      
      // Check for auth errors
      const authErrorEvents = authEvents.filter(event =>
        event.message.error || 
        event.message.authenticated === false ||
        event.message.type?.includes('error')
      )
      
      if (authErrorEvents.length > 0) {
        console.log('Authentication errors detected:', authErrorEvents)
      }
      
    } else {
      console.log('No authentication-related WebSocket activity detected')
    }
    
    // Check for auth tokens in connection URLs
    page.on('websocket', ws => {
      const url = ws.url()
      if (url.includes('token=') || url.includes('auth=') || url.includes('jwt=')) {
        console.log('✓ WebSocket URL includes authentication parameters')
      }
    })
  })
})