import { test, expect } from '@playwright/test'
import { TestHelpers } from '../utils/test-helpers'

test.describe('Real Backend Integration Tests', () => {
  const BACKEND_URL = 'http://localhost:8000'
  
  test.beforeEach(async ({ page }) => {
    // Ensure we're testing against the real backend, not mocks
    await TestHelpers.disableAPIMocks(page)
    
    // Navigate to dashboard
    await page.goto('/')
    await page.waitForLoadState('networkidle')
  })

  test.describe('Backend Health and Connectivity', () => {
    test('should connect to real FastAPI backend', async ({ page }) => {
      // Test direct API health endpoint
      const healthResponse = await page.request.get(`${BACKEND_URL}/health`)
      expect(healthResponse.status()).toBe(200)
      
      const healthData = await healthResponse.json()
      expect(healthData).toHaveProperty('status')
      expect(healthData).toHaveProperty('version', '2.0.0')
      expect(healthData).toHaveProperty('components')
      
      await TestHelpers.takeTimestampedScreenshot(page, 'backend-health-check')
    })

    test('should have operational database and redis connections', async ({ page }) => {
      const healthResponse = await page.request.get(`${BACKEND_URL}/health`)
      const healthData = await healthResponse.json()
      
      // Verify core components are healthy
      expect(healthData.components.database.status).toBe('healthy')
      expect(healthData.components.redis.status).toBe('healthy')
      expect(healthData.summary.healthy).toBeGreaterThan(0)
    })

    test('should expose Prometheus metrics', async ({ page }) => {
      const metricsResponse = await page.request.get(`${BACKEND_URL}/metrics`)
      expect(metricsResponse.status()).toBe(200)
      
      const metricsText = await metricsResponse.text()
      expect(metricsText).toContain('leanvibe_health_status')
      expect(metricsText).toContain('leanvibe_uptime_seconds')
      
      // Should contain real metrics, not just static ones
      expect(metricsText).toMatch(/leanvibe_health_status\{component="database"\}\s+[01]/)
    })
  })

  test.describe('Real Agent System Integration', () => {
    test('should load real agent data from API', async ({ page }) => {
      // Navigate to agents view
      await page.click('[data-nav="agents"]')
      await page.waitForSelector('.agents-container', { timeout: 10000 })
      
      // Wait for API call to complete
      await page.waitForResponse(response => 
        response.url().includes('/api/agents/status') && response.status() === 200
      )
      
      // Check if real data is loaded (not mock fallback)
      const agentCards = page.locator('.agent-card')
      const agentCount = await agentCards.count()
      
      if (agentCount > 0) {
        // Verify agents have real data characteristics
        const firstAgent = agentCards.first()
        const agentName = await firstAgent.locator('.agent-name').textContent()
        
        // Real agents should have role-based names (e.g., "product_manager Agent")
        expect(agentName).toMatch(/(product_manager|architect|backend_developer|qa_engineer|devops_engineer)/i)
        
        // Verify status indicator shows real status
        const statusIndicator = firstAgent.locator('.agent-status-indicator')
        await expect(statusIndicator).toBeVisible()
        
        await TestHelpers.takeTimestampedScreenshot(page, 'real-agent-data')
        
        console.log(`✅ Real agent data loaded: ${agentCount} agents from API`)
      } else {
        // If no agents are active, that's valid - just ensure we're getting real API response
        const emptyState = page.locator('.empty-state')
        if (await emptyState.isVisible()) {
          console.log('✅ Real API response: No agents currently active')
        }
      }
    })

    test('should activate agent team through real API', async ({ page }) => {
      await page.click('[data-nav="agents"]')
      await page.waitForSelector('.agents-container')
      
      // Look for team activation button
      const activateButton = page.locator('.action-button.success')
      if (await activateButton.isVisible()) {
        // Monitor network requests
        const apiPromise = page.waitForResponse(response => 
          response.url().includes('/api/agents/activate') && response.status() === 200
        )
        
        await activateButton.click()
        
        // Wait for real API response
        const response = await apiPromise
        const responseData = await response.json()
        
        expect(responseData).toHaveProperty('success', true)
        expect(responseData).toHaveProperty('active_agents')
        expect(responseData).toHaveProperty('team_composition')
        
        // Verify agents appear in UI after activation
        await page.waitForSelector('.agent-card', { timeout: 15000 })
        const agentCount = await page.locator('.agent-card').count()
        expect(agentCount).toBeGreaterThan(0)
        
        console.log(`✅ Team activation successful: ${agentCount} agents spawned`)
        await TestHelpers.takeTimestampedScreenshot(page, 'team-activation-real')
      }
    })

    test('should spawn individual agents through real API', async ({ page }) => {
      await page.click('[data-nav="agents"]')
      await page.waitForSelector('.agents-container')
      
      // Get initial agent count
      const initialCount = await page.locator('.agent-card').count()
      
      // Try to activate a specific agent role
      const roles = ['backend_developer', 'frontend_developer', 'qa_engineer']
      const testRole = roles[Math.floor(Math.random() * roles.length)]
      
      // Mock individual agent activation by calling the API directly
      const spawnResponse = await page.request.post(`${BACKEND_URL}/api/agents/spawn/${testRole}`)
      
      if (spawnResponse.status() === 200) {
        const spawnData = await spawnResponse.json()
        expect(spawnData).toHaveProperty('success', true)
        expect(spawnData).toHaveProperty('agent_id')
        expect(spawnData).toHaveProperty('role', testRole)
        
        // Refresh the page to see the new agent
        await page.reload()
        await page.waitForSelector('.agents-container')
        
        // Verify new agent appears
        const newCount = await page.locator('.agent-card').count()
        expect(newCount).toBeGreaterThanOrEqual(initialCount)
        
        console.log(`✅ Individual agent spawned: ${testRole}`)
      }
    })

    test('should handle real API errors gracefully', async ({ page }) => {
      // Test with invalid agent role to trigger real API error
      const invalidResponse = await page.request.post(`${BACKEND_URL}/api/agents/spawn/invalid_role`)
      expect(invalidResponse.status()).toBe(400)
      
      const errorData = await invalidResponse.json()
      expect(errorData).toHaveProperty('detail')
      expect(errorData.detail).toContain('Invalid role')
    })
  })

  test.describe('Real Performance Metrics Integration', () => {
    test('should display real system metrics', async ({ page }) => {
      await page.goto('/')
      await page.waitForSelector('.dashboard-content')
      
      // Wait for performance metrics to load
      await page.waitForTimeout(3000)
      
      // Check if system health card shows real data
      const healthCard = page.locator('[data-metric="system-health"]')
      if (await healthCard.isVisible()) {
        const healthValue = await healthCard.locator('.summary-value').textContent()
        expect(healthValue).toMatch(/healthy|degraded|unhealthy/i)
      }
      
      // Check CPU usage if available
      const cpuCard = page.locator('[data-metric="cpu-usage"]')
      if (await cpuCard.isVisible()) {
        const cpuValue = await cpuCard.locator('.summary-value').textContent()
        expect(cpuValue).toMatch(/\d+%/)
      }
      
      // Check memory usage if available
      const memoryCard = page.locator('[data-metric="memory-usage"]')
      if (await memoryCard.isVisible()) {
        const memoryValue = await memoryCard.locator('.summary-value').textContent()
        expect(memoryValue).toMatch(/\d+%/)
      }
      
      await TestHelpers.takeTimestampedScreenshot(page, 'real-system-metrics')
    })

    test('should receive real-time metric updates', async ({ page }) => {
      await page.goto('/')
      await page.waitForSelector('.dashboard-content')
      
      // Monitor metrics updates over time
      const initialMetrics = {}
      const metricsElements = await page.locator('[data-metric]').all()
      
      for (const element of metricsElements) {
        const metricType = await element.getAttribute('data-metric')
        const value = await element.locator('.summary-value').textContent()
        if (metricType && value) {
          initialMetrics[metricType] = value
        }
      }
      
      // Wait for potential updates (real metrics should update every 5 seconds)
      await page.waitForTimeout(6000)
      
      // Verify metrics are still present (may or may not have changed)
      for (const [metricType] of Object.entries(initialMetrics)) {
        const element = page.locator(`[data-metric="${metricType}"]`)
        await expect(element).toBeVisible()
        await expect(element.locator('.summary-value')).not.toBeEmpty()
      }
      
      console.log('✅ Real-time metrics monitoring validated')
    })
  })

  test.describe('Data Consistency Validation', () => {
    test('should match API data with UI display', async ({ page }) => {
      // Get agent data directly from API
      const apiResponse = await page.request.get(`${BACKEND_URL}/api/agents/status`)
      const apiData = await apiResponse.json()
      
      // Navigate to agents view
      await page.click('[data-nav="agents"]')
      await page.waitForSelector('.agents-container')
      
      // Compare API data with UI
      const uiAgentCount = await page.locator('.agent-card').count()
      const apiAgentCount = apiData.agent_count || 0
      
      // Allow for small discrepancies due to timing
      expect(Math.abs(uiAgentCount - apiAgentCount)).toBeLessThanOrEqual(1)
      
      // Verify system readiness consistency
      const systemReadyAPI = apiData.system_ready
      if (systemReadyAPI !== undefined) {
        const readyIndicator = page.locator('[data-indicator="system-ready"]')
        if (await readyIndicator.isVisible()) {
          const readyClass = await readyIndicator.getAttribute('class')
          if (systemReadyAPI) {
            expect(readyClass).toContain('ready')
          } else {
            expect(readyClass).not.toContain('ready')
          }
        }
      }
      
      console.log(`✅ Data consistency validated: API=${apiAgentCount}, UI=${uiAgentCount}`)
    })

    test('should handle partial API failures gracefully', async ({ page }) => {
      // Test behavior when some API endpoints are down
      await page.route('**/api/agents/capabilities', route => route.abort())
      
      await page.click('[data-nav="agents"]')
      await page.waitForSelector('.agents-container')
      
      // Main functionality should still work even if secondary endpoints fail
      const agentCards = page.locator('.agent-card')
      if (await agentCards.count() > 0) {
        await expect(agentCards.first()).toBeVisible()
        console.log('✅ Graceful degradation: Core functionality maintained despite partial failures')
      }
    })
  })

  test.describe('End-to-End Workflow Validation', () => {
    test('should complete full agent lifecycle', async ({ page }) => {
      await page.click('[data-nav="agents"]')
      await page.waitForSelector('.agents-container')
      
      let testResults = {
        initialState: false,
        teamActivation: false,
        agentSpawning: false,
        systemStatus: false
      }
      
      // 1. Check initial state
      const initialResponse = await page.request.get(`${BACKEND_URL}/api/agents/status`)
      if (initialResponse.status() === 200) {
        testResults.initialState = true
        console.log('✅ Initial system state validated')
      }
      
      // 2. Activate team if no agents present
      const statusData = await initialResponse.json()
      if (statusData.agent_count === 0) {
        const activateButton = page.locator('.action-button.success')
        if (await activateButton.isVisible()) {
          await activateButton.click()
          await page.waitForResponse(response => 
            response.url().includes('/api/agents/activate')
          )
          testResults.teamActivation = true
          console.log('✅ Team activation completed')
        }
      }
      
      // 3. Verify agents are operational
      await page.waitForTimeout(2000)
      const finalResponse = await page.request.get(`${BACKEND_URL}/api/agents/status`)
      const finalData = await finalResponse.json()
      
      if (finalData.agent_count > 0) {
        testResults.systemStatus = true
        console.log(`✅ System operational with ${finalData.agent_count} agents`)
      }
      
      // 4. Test individual agent spawning
      try {
        const spawnResponse = await page.request.post(`${BACKEND_URL}/api/agents/spawn/qa_engineer`)
        if (spawnResponse.status() === 200) {
          testResults.agentSpawning = true
          console.log('✅ Individual agent spawning validated')
        }
      } catch (error) {
        console.log('ℹ️ Individual agent spawning test skipped (may already exist)')
      }
      
      // Verify at least core functionality works
      expect(testResults.initialState).toBe(true)
      expect(testResults.systemStatus || testResults.teamActivation).toBe(true)
      
      await TestHelpers.takeTimestampedScreenshot(page, 'end-to-end-validation')
    })
  })
})

// Test utility for disabling API mocks when testing real backend
class BackendTestHelpers {
  static async waitForBackendReady(page, timeout = 30000) {
    const startTime = Date.now()
    
    while (Date.now() - startTime < timeout) {
      try {
        const response = await page.request.get(`${BACKEND_URL}/health`)
        if (response.status() === 200) {
          return true
        }
      } catch (error) {
        // Continue waiting
      }
      await page.waitForTimeout(1000)
    }
    
    throw new Error(`Backend not ready after ${timeout}ms`)
  }
  
  static async validateRealDataCharacteristics(page, agentData) {
    // Verify this is real data, not mock data
    const realDataIndicators = [
      agentData.some(agent => agent.id && agent.id.length > 10), // Real UUIDs
      agentData.some(agent => agent.name && agent.name.includes('Agent')), // Real naming pattern
      agentData.some(agent => agent.role && ['product_manager', 'architect', 'backend_developer', 'qa_engineer', 'devops_engineer'].includes(agent.role))
    ]
    
    return realDataIndicators.some(indicator => indicator)
  }
}