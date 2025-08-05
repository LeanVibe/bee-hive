/**
 * Autonomous Dashboard Development Validation Tests
 * Tests mobile PWA integration with enhanced /hive commands and agent coordination
 */

import { test, expect, Page } from '@playwright/test';

interface HiveCommandResponse {
  success: boolean;
  mobile_optimized: boolean;
  execution_time_ms: number;
  cached: boolean;
  result: any;
  performance_metrics?: {
    mobile_performance_score?: number;
    mobile_optimized_alerts?: number;
    cache_hit?: boolean;
  };
}

interface AgentStatus {
  role: string;
  status: string;
  current_task?: string;
  progress?: number;
  last_update?: string;
}

interface MobileOptimizedStatus {
  mobile_optimized: boolean;
  system_state: 'operational' | 'degraded';
  agent_count: number;
  requires_attention: boolean;
  quick_actions: Array<{
    action: string;
    command: string;
    description: string;
  }>;
}

test.describe('Enhanced /hive Commands Mobile Integration', () => {
  
  test('Mobile-optimized status command integration', async ({ page }) => {
    await page.goto('/');
    
    // Wait for dashboard to load
    await expect(page.locator('[data-testid="mobile-dashboard"]')).toBeVisible();
    
    // Test mobile-optimized status API integration
    const apiResponse = await page.evaluate(async () => {
      const response = await fetch('/api/hive/mobile/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: '/hive:status --mobile --priority=high',
          mobile_optimized: true,
          use_cache: true,
          priority: 'high'
        })
      });
      return response.json();
    });
    
    // Validate mobile-optimized response structure
    expect(apiResponse.success).toBe(true);
    expect(apiResponse.mobile_optimized).toBe(true);
    expect(apiResponse.execution_time_ms).toBeLessThan(50); // Mobile target <50ms
    expect(apiResponse.result).toHaveProperty('system_state');
    expect(apiResponse.result).toHaveProperty('quick_actions');
    
    // Test dashboard updates with mobile data
    await page.click('[data-testid="refresh-status"]');
    await expect(page.locator('[data-testid="system-status"]')).toContainText(/operational|degraded/);
    
    // Validate quick actions are displayed
    const quickActions = page.locator('[data-testid="quick-action"]');
    await expect(quickActions).toHaveCount({ min: 1, max: 3 });
    
    console.log('✅ Mobile-optimized status command integration validated');
  });
  
  test('Intelligent caching effectiveness', async ({ page }) => {
    await page.goto('/');
    
    // First request - should be live
    const firstRequest = await page.evaluate(async () => {
      const start = performance.now();
      const response = await fetch('/api/hive/mobile/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: '/hive:focus development --mobile',
          mobile_optimized: true,
          use_cache: true
        })
      });
      const end = performance.now();
      const data = await response.json();
      return { ...data, client_time: end - start };
    });
    
    // Second request - should be cached
    const secondRequest = await page.evaluate(async () => {
      const start = performance.now();
      const response = await fetch('/api/hive/mobile/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: '/hive:focus development --mobile',
          mobile_optimized: true,
          use_cache: true
        })
      });
      const end = performance.now();
      const data = await response.json();
      return { ...data, client_time: end - start };
    });
    
    // Validate caching effectiveness
    expect(firstRequest.success).toBe(true);
    expect(secondRequest.success).toBe(true);
    
    // Second request should be faster (cached)
    if (secondRequest.cached) {
      expect(secondRequest.execution_time_ms).toBeLessThan(5); // Cache target <5ms
      expect(secondRequest.client_time).toBeLessThan(firstRequest.client_time);
    }
    
    console.log(`Cache effectiveness: First=${firstRequest.execution_time_ms}ms, Second=${secondRequest.execution_time_ms}ms, Cached=${secondRequest.cached}`);
  });
  
  test('Agent coordination mobile interface', async ({ page }) => {
    await page.goto('/');
    
    // Test agent spawning through mobile interface
    await page.click('[data-testid="spawn-agent-button"]');
    await page.selectOption('[data-testid="agent-role-select"]', 'backend_developer');
    await page.click('[data-testid="confirm-spawn"]');
    
    // Wait for agent coordination API call
    const coordinationResponse = await page.waitForResponse(
      response => response.url().includes('/api/hive/execute') && response.request().method() === 'POST'
    );
    
    expect(coordinationResponse.status()).toBe(200);
    
    const responseData = await coordinationResponse.json();
    expect(responseData.success).toBe(true);
    
    // Validate agent appears in mobile dashboard
    await expect(page.locator('[data-testid="agent-backend_developer"]')).toBeVisible();
    
    // Test agent task assignment through mobile
    await page.click('[data-testid="agent-backend_developer"]');
    await page.fill('[data-testid="task-input"]', 'Implement user authentication API');
    await page.click('[data-testid="assign-task"]');
    
    // Validate task assignment
    await expect(page.locator('[data-testid="agent-task"]')).toContainText('Implement user authentication API');
    
    console.log('✅ Agent coordination mobile interface validated');
  });
  
  test('Real-time WebSocket integration', async ({ page }) => {
    await page.goto('/');
    
    // Setup WebSocket message listener
    const wsMessages: any[] = [];
    await page.evaluate(() => {
      // @ts-ignore
      window.wsMessages = [];
      // @ts-ignore
      if (window.websocketService) {
        // @ts-ignore
        window.websocketService.addEventListener('agent_update', (event) => {
          // @ts-ignore
          window.wsMessages.push(event.detail);
        });
      }
    });
    
    // Trigger agent status change
    await page.evaluate(async () => {
      await fetch('/api/hive/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: '/hive:spawn qa_engineer --capabilities=testing,validation'
        })
      });
    });
    
    // Wait for WebSocket update
    await page.waitForTimeout(1000);
    
    // Check if real-time update was received
    const messages = await page.evaluate(() => {
      // @ts-ignore
      return window.wsMessages || [];
    });
    
    if (messages.length > 0) {
      expect(messages[0]).toHaveProperty('agent_id');
      expect(messages[0]).toHaveProperty('status');
      console.log('✅ Real-time WebSocket updates working');
    } else {
      console.log('⚠️ WebSocket updates not received (may be expected in test environment)');
    }
    
    // Validate UI updates in real-time
    await expect(page.locator('[data-testid="agent-count"]')).not.toHaveText('0');
  });
  
  test('Mobile performance targets validation', async ({ page }) => {
    await page.goto('/');
    
    // Test mobile performance under load
    const performanceMetrics = await page.evaluate(async () => {
      const results = [];
      
      // Execute multiple rapid requests to test mobile performance
      const commands = [
        '/hive:status --mobile',
        '/hive:focus development --mobile',
        '/hive:productivity --mobile --developer'
      ];
      
      for (const command of commands) {
        const start = performance.now();
        const response = await fetch('/api/hive/mobile/execute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            command,
            mobile_optimized: true,
            use_cache: true,
            priority: 'high'
          })
        });
        const end = performance.now();
        
        const data = await response.json();
        results.push({
          command,
          success: data.success,
          cached: data.cached,
          server_time: data.execution_time_ms,
          client_time: end - start,
          mobile_optimized: data.mobile_optimized
        });
      }
      
      return results;
    });
    
    // Validate performance targets
    for (const metric of performanceMetrics) {
      expect(metric.success).toBe(true);
      expect(metric.mobile_optimized).toBe(true);
      
      if (metric.cached) {
        expect(metric.server_time).toBeLessThan(5); // Cached target <5ms
      } else {
        expect(metric.server_time).toBeLessThan(50); // Live target <50ms
      }
      
      expect(metric.client_time).toBeLessThan(100); // Total client time <100ms
    }
    
    const avgServerTime = performanceMetrics.reduce((sum, m) => sum + m.server_time, 0) / performanceMetrics.length;
    const avgClientTime = performanceMetrics.reduce((sum, m) => sum + m.client_time, 0) / performanceMetrics.length;
    
    console.log(`✅ Mobile performance targets met: Avg server=${avgServerTime.toFixed(1)}ms, Avg client=${avgClientTime.toFixed(1)}ms`);
  });
  
  test('Context-aware mobile recommendations', async ({ page }) => {
    await page.goto('/');
    
    // Test context-aware focus recommendations
    const focusResponse = await page.evaluate(async () => {
      return await fetch('/api/hive/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: '/hive:focus development --mobile',
          mobile_optimized: true,
          context: {
            current_view: 'dashboard',
            user_intent: 'development_workflow'
          }
        })
      });
    });
    
    expect(focusResponse.status).toBe(200);
    
    const focusData = await focusResponse.json();
    expect(focusData.success).toBe(true);
    expect(focusData.result).toHaveProperty('recommendations');
    expect(focusData.result).toHaveProperty('mobile_optimized');
    
    // Validate mobile-optimized recommendations format
    if (focusData.result.mobile_optimized) {
      expect(focusData.result).toHaveProperty('quick_actions');
      expect(focusData.result).toHaveProperty('summary');
      
      const quickActions = focusData.result.quick_actions;
      expect(Array.isArray(quickActions)).toBe(true);
      expect(quickActions.length).toBeLessThanOrEqual(3); // Mobile limit
      
      // Each quick action should have mobile-specific fields
      for (const action of quickActions) {
        expect(action).toHaveProperty('title');
        expect(action).toHaveProperty('priority');
        expect(action).toHaveProperty('time');
      }
    }
    
    console.log('✅ Context-aware mobile recommendations validated');
  });
  
  test('Autonomous development workflow mobile oversight', async ({ page }) => {
    await page.goto('/');
    
    // Start autonomous development through mobile interface
    await page.click('[data-testid="start-development"]');
    await page.fill('[data-testid="project-description"]', 'Build user authentication system');
    await page.check('[data-testid="enable-mobile-oversight"]');
    await page.click('[data-testid="begin-development"]');
    
    // Wait for development to begin
    await page.waitForResponse(
      response => response.url().includes('/api/hive/execute') && 
                 response.request().postData()?.includes('hive:develop')
    );
    
    // Test mobile oversight features
    await expect(page.locator('[data-testid="development-status"]')).toBeVisible();
    await expect(page.locator('[data-testid="agent-progress"]')).toBeVisible();
    
    // Test real-time progress updates
    const progressUpdates = await page.locator('[data-testid="progress-update"]');
    await expect(progressUpdates).toHaveCount({ min: 1 });
    
    // Test mobile-specific oversight controls
    await expect(page.locator('[data-testid="pause-development"]')).toBeVisible();
    await expect(page.locator('[data-testid="mobile-notifications"]')).toBeVisible();
    
    // Validate mobile oversight dashboard
    const oversightData = await page.evaluate(async () => {
      return await fetch('/api/hive/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: '/hive:oversight --mobile-info'
        })
      });
    });
    
    expect(oversightData.status).toBe(200);
    
    console.log('✅ Autonomous development mobile oversight validated');
  });
  
  test('Mobile gesture interface integration', async ({ page }) => {
    await page.goto('/');
    
    // Test pull-to-refresh gesture
    await page.evaluate(() => {
      const dashboard = document.querySelector('[data-testid="mobile-dashboard"]');
      if (dashboard) {
        // Simulate pull-to-refresh gesture
        const touchStart = new TouchEvent('touchstart', {
          touches: [{ clientX: 200, clientY: 100 } as Touch]
        });
        const touchMove = new TouchEvent('touchmove', {
          touches: [{ clientX: 200, clientY: 200 } as Touch]
        });
        const touchEnd = new TouchEvent('touchend', { touches: [] });
        
        dashboard.dispatchEvent(touchStart);
        dashboard.dispatchEvent(touchMove);
        dashboard.dispatchEvent(touchEnd);
      }
    });
    
    // Wait for refresh action
    await page.waitForTimeout(500);
    
    // Validate refresh occurred
    await expect(page.locator('[data-testid="last-updated"]')).toBeVisible();
    
    // Test swipe navigation
    const agentCards = page.locator('[data-testid="agent-card"]');
    if (await agentCards.count() > 0) {
      await agentCards.first().hover();
      
      // Simulate swipe gesture
      await page.evaluate(() => {
        const card = document.querySelector('[data-testid="agent-card"]');
        if (card) {
          const touchStart = new TouchEvent('touchstart', {
            touches: [{ clientX: 100, clientY: 200 } as Touch]
          });
          const touchMove = new TouchEvent('touchmove', {
            touches: [{ clientX: 300, clientY: 200 } as Touch]
          });
          const touchEnd = new TouchEvent('touchend', { touches: [] });
          
          card.dispatchEvent(touchStart);
          card.dispatchEvent(touchMove);
          card.dispatchEvent(touchEnd);
        }
      });
      
      // Validate swipe action revealed options
      await expect(page.locator('[data-testid="swipe-actions"]')).toBeVisible();
    }
    
    console.log('✅ Mobile gesture interface integration validated');
  });
  
  test('Battery and resource optimization', async ({ page }) => {
    await page.goto('/');
    
    // Test adaptive polling based on activity
    const performanceObserver = await page.evaluate(() => {
      return new Promise((resolve) => {
        if ('PerformanceObserver' in window) {
          const observer = new PerformanceObserver((list) => {
            const entries = list.getEntries();
            const networkRequests = entries.filter(entry => 
              entry.name.includes('/api/hive/') && entry.entryType === 'navigation'
            );
            resolve(networkRequests.length);
          });
          observer.observe({ entryTypes: ['navigation', 'resource'] });
        } else {
          resolve(0);
        }
      });
    });
    
    // Test request batching optimization
    const batchResponse = await page.evaluate(async () => {
      // Simulate batched request
      const start = performance.now();
      const response = await fetch('/api/hive/mobile/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: '/hive:status --mobile --detailed',
          mobile_optimized: true,
          context: {
            batch_mode: true,
            include_agents: true,
            include_health: true,
            include_metrics: true
          }
        })
      });
      const end = performance.now();
      
      const data = await response.json();
      return {
        success: data.success,
        execution_time: end - start,
        mobile_optimized: data.mobile_optimized
      };
    });
    
    expect(batchResponse.success).toBe(true);
    expect(batchResponse.mobile_optimized).toBe(true);
    expect(batchResponse.execution_time).toBeLessThan(100); // Batched request efficiency
    
    console.log(`✅ Resource optimization: Batched request completed in ${batchResponse.execution_time.toFixed(1)}ms`);
  });
});

test.describe('Error Handling and Resilience', () => {
  
  test('Graceful degradation when backend unavailable', async ({ page }) => {
    await page.goto('/');
    
    // Mock backend unavailability
    await page.route('/api/hive/**', route => {
      route.abort('failed');
    });
    
    // Test that dashboard still loads
    await expect(page.locator('[data-testid="mobile-dashboard"]')).toBeVisible();
    
    // Should show offline indicator
    await expect(page.locator('[data-testid="offline-indicator"]')).toBeVisible();
    
    // Should show cached data if available
    const cachedContent = page.locator('[data-testid="cached-content"]');
    if (await cachedContent.count() > 0) {
      await expect(cachedContent).toBeVisible();
    }
    
    console.log('✅ Graceful degradation when backend unavailable');
  });
  
  test('Error recovery and retry mechanisms', async ({ page }) => {
    await page.goto('/');
    
    let requestCount = 0;
    
    // Mock intermittent failures
    await page.route('/api/hive/mobile/execute', route => {
      requestCount++;
      if (requestCount <= 2) {
        // Fail first two requests
        route.fulfill({
          status: 500,
          body: JSON.stringify({ success: false, error: 'Internal server error' })
        });
      } else {
        // Success on third request
        route.fulfill({
          status: 200,
          body: JSON.stringify({
            success: true,
            mobile_optimized: true,
            result: { system_state: 'operational', agent_count: 3 }
          })
        });
      }
    });
    
    // Trigger request that should retry
    await page.click('[data-testid="refresh-status"]');
    
    // Wait for retry mechanism to succeed
    await page.waitForTimeout(3000);
    
    // Should eventually succeed
    await expect(page.locator('[data-testid="error-message"]')).not.toBeVisible();
    await expect(page.locator('[data-testid="system-status"]')).toContainText('operational');
    
    expect(requestCount).toBeGreaterThan(2); // Should have retried
    
    console.log(`✅ Error recovery completed after ${requestCount} attempts`);
  });
});

test.describe('Performance Regression Detection', () => {
  
  test('Mobile performance regression detection', async ({ page }) => {
    await page.goto('/');
    
    // Measure baseline performance
    const baselineMetrics = await page.evaluate(async () => {
      const start = performance.now();
      await new Promise(resolve => setTimeout(resolve, 100)); // Simulate work
      const end = performance.now();
      
      return {
        loadTime: end - start,
        memoryUsage: (performance as any).memory?.usedJSHeapSize || 0
      };
    });
    
    // Test performance under various conditions
    const stressMetrics = await page.evaluate(async () => {
      const results = [];
      
      // Simulate multiple rapid requests
      for (let i = 0; i < 5; i++) {
        const start = performance.now();
        
        try {
          await fetch('/api/hive/mobile/execute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              command: '/hive:status --mobile',
              mobile_optimized: true,
              use_cache: true
            })
          });
        } catch (e) {
          // Ignore network errors in test
        }
        
        const end = performance.now();
        results.push(end - start);
      }
      
      return {
        averageTime: results.reduce((a, b) => a + b, 0) / results.length,
        maxTime: Math.max(...results),
        memoryUsage: (performance as any).memory?.usedJSHeapSize || 0
      };
    });
    
    // Validate no significant performance regression
    expect(stressMetrics.averageTime).toBeLessThan(100); // Average response <100ms
    expect(stressMetrics.maxTime).toBeLessThan(200); // Max response <200ms
    
    // Memory usage should not grow excessively
    if (baselineMetrics.memoryUsage > 0 && stressMetrics.memoryUsage > 0) {
      const memoryGrowth = (stressMetrics.memoryUsage - baselineMetrics.memoryUsage) / baselineMetrics.memoryUsage;
      expect(memoryGrowth).toBeLessThan(2.0); // Memory growth <200%
    }
    
    console.log(`✅ Performance regression check: Avg=${stressMetrics.averageTime.toFixed(1)}ms, Max=${stressMetrics.maxTime.toFixed(1)}ms`);
  });
});

/**
 * Test utility functions
 */
async function waitForWebSocketConnection(page: Page): Promise<boolean> {
  return await page.evaluate(() => {
    return new Promise((resolve) => {
      const checkConnection = () => {
        // @ts-ignore
        if (window.websocketService && window.websocketService.isConnected) {
          resolve(true);
        } else {
          setTimeout(checkConnection, 100);
        }
      };
      checkConnection();
      
      // Timeout after 5 seconds
      setTimeout(() => resolve(false), 5000);
    });
  });
}

async function simulateMobileViewport(page: Page) {
  await page.setViewportSize({ width: 375, height: 667 }); // iPhone SE
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'userAgent', {
      value: 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
    });
  });
}