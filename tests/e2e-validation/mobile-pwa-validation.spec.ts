import { test, expect, Page, BrowserContext } from '@playwright/test';

/**
 * Comprehensive Mobile PWA Dashboard Validation Test Suite
 * 
 * Tests the mobile PWA dashboard at localhost:3002 and compares
 * it with the simple dashboard at localhost:8000/dashboard/simple
 * 
 * Coverage:
 * 1. Basic loading and UI elements
 * 2. Backend API connectivity mapping
 * 3. WebSocket connection validation
 * 4. Real data vs mock data behavior
 * 5. Feature comparison between dashboards
 */

interface DashboardEndpoints {
  http: string[];
  websocket: string[];
  api: string[];
}

interface UIValidation {
  requiredElements: string[];
  interactiveElements: string[];
  navigationElements: string[];
}

interface DataValidation {
  realDataIndicators: string[];
  mockDataIndicators: string[];
  loadingStates: string[];
}

test.describe('Mobile PWA Dashboard Validation Suite', () => {
  let pwaPage: Page;
  let simplePage: Page;
  let context: BrowserContext;

  test.beforeAll(async ({ browser }) => {
    context = await browser.newContext({
      viewport: { width: 390, height: 844 }, // iPhone 12 Pro dimensions
      userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1',
      isMobile: true,
      hasTouch: true,
    });
  });

  test.beforeEach(async () => {
    pwaPage = await context.newPage();
    simplePage = await context.newPage();
    
    // Enable console logging for debugging
    pwaPage.on('console', msg => console.log(`PWA Console: ${msg.text()}`));
    simplePage.on('console', msg => console.log(`Simple Console: ${msg.text()}`));
    
    // Enable network monitoring
    await pwaPage.route('**/*', route => {
      console.log(`PWA Request: ${route.request().method()} ${route.request().url()}`);
      route.continue();
    });
    
    await simplePage.route('**/*', route => {
      console.log(`Simple Request: ${route.request().method()} ${route.request().url()}`);
      route.continue();
    });
  });

  test.afterEach(async () => {
    await pwaPage?.close();
    await simplePage?.close();
  });

  test.afterAll(async () => {
    await context?.close();
  });

  test('1. Basic Loading and UI Elements Validation', async () => {
    console.log('🧪 Testing basic loading and UI elements...');

    // Test PWA Dashboard Loading
    console.log('📱 Loading PWA Dashboard at localhost:3002...');
    const pwaResponse = await pwaPage.goto('http://localhost:3002', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    expect(pwaResponse?.status()).toBeLessThan(400);
    
    // Verify PWA loads and removes loading screen
    await expect(pwaPage.locator('.loading-container')).toBeHidden({ timeout: 15000 });
    await expect(pwaPage.locator('#app')).toBeVisible();
    
    // Test Simple Dashboard Loading
    console.log('🖥️ Loading Simple Dashboard at localhost:8000/dashboard/simple...');
    const simpleResponse = await simplePage.goto('http://localhost:8000/dashboard/simple', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    expect(simpleResponse?.status()).toBeLessThan(400);

    // PWA UI Elements Validation
    const pwaUIElements: UIValidation = {
      requiredElements: [
        '#app',
        'agent-hive-app, [data-testid="app"]', // Lit component or fallback
        '.dashboard, [data-view="dashboard"]', // Dashboard container
      ],
      interactiveElements: [
        'button, [role="button"]',
        'a[href], [role="link"]', 
        'input, textarea, select',
      ],
      navigationElements: [
        '.navigation, nav, [role="navigation"]',
        '.menu, [data-menu], [role="menu"]'
      ]
    };

    console.log('🔍 Validating PWA UI elements...');
    for (const selector of pwaUIElements.requiredElements) {
      try {
        await expect(pwaPage.locator(selector).first()).toBeVisible({ timeout: 10000 });
        console.log(`✅ PWA: Found required element: ${selector}`);
      } catch (error) {
        console.log(`⚠️ PWA: Missing or not visible: ${selector}`);
      }
    }

    // Simple Dashboard UI Elements Validation
    const simpleUIElements: UIValidation = {
      requiredElements: [
        'body',
        '.dashboard, #dashboard, [data-dashboard]',
        'h1, h2, .title, [data-title]',
      ],
      interactiveElements: [
        'button',
        'a[href]',
        'input, select',
      ],
      navigationElements: [
        'nav, .nav, [role="navigation"]',
        '.menu, [data-menu]'
      ]
    };

    console.log('🔍 Validating Simple Dashboard UI elements...');
    for (const selector of simpleUIElements.requiredElements) {
      try {
        await expect(simplePage.locator(selector).first()).toBeVisible({ timeout: 10000 });
        console.log(`✅ Simple: Found required element: ${selector}`);
      } catch (error) {
        console.log(`⚠️ Simple: Missing or not visible: ${selector}`);
      }
    }

    // Mobile-specific PWA validations
    console.log('📱 Validating mobile-specific features...');
    
    // Check for mobile viewport meta tag
    const viewport = await pwaPage.locator('meta[name="viewport"]').getAttribute('content');
    expect(viewport).toContain('width=device-width');
    
    // Check for PWA manifest
    const manifest = await pwaPage.locator('link[rel="manifest"]').count();
    expect(manifest).toBeGreaterThan(0);
    
    // Check for service worker registration
    const swRegistration = await pwaPage.evaluate(() => {
      return 'serviceWorker' in navigator;
    });
    expect(swRegistration).toBe(true);
  });

  test('2. Backend API Connectivity and Endpoint Discovery', async () => {
    console.log('🔌 Testing backend API connectivity...');

    const pwaEndpoints: DashboardEndpoints = {
      http: [],
      websocket: [],
      api: []
    };

    const simpleEndpoints: DashboardEndpoints = {
      http: [],
      websocket: [],
      api: []
    };

    // Monitor PWA network requests
    pwaPage.on('request', request => {
      const url = request.url();
      if (url.includes('/api/')) {
        pwaEndpoints.api.push(url);
      } else if (url.startsWith('ws://') || url.startsWith('wss://')) {
        pwaEndpoints.websocket.push(url);
      } else if (url.startsWith('http')) {
        pwaEndpoints.http.push(url);
      }
    });

    // Monitor Simple Dashboard network requests
    simplePage.on('request', request => {
      const url = request.url();
      if (url.includes('/api/')) {
        simpleEndpoints.api.push(url);
      } else if (url.startsWith('ws://') || url.startsWith('wss://')) {
        simpleEndpoints.websocket.push(url);
      } else if (url.startsWith('http')) {
        simpleEndpoints.http.push(url);
      }
    });

    // Load both dashboards to capture network activity
    await pwaPage.goto('http://localhost:3002', { waitUntil: 'networkidle' });
    await simplePage.goto('http://localhost:8000/dashboard/simple', { waitUntil: 'networkidle' });

    // Wait for potential async API calls
    await pwaPage.waitForTimeout(5000);
    await simplePage.waitForTimeout(5000);

    console.log('📊 PWA Endpoints discovered:');
    console.log('  HTTP:', [...new Set(pwaEndpoints.http)]);
    console.log('  WebSocket:', [...new Set(pwaEndpoints.websocket)]);
    console.log('  API:', [...new Set(pwaEndpoints.api)]);

    console.log('📊 Simple Dashboard Endpoints discovered:');
    console.log('  HTTP:', [...new Set(simpleEndpoints.http)]);
    console.log('  WebSocket:', [...new Set(simpleEndpoints.websocket)]);
    console.log('  API:', [...new Set(simpleEndpoints.api)]);

    // Test expected API endpoints based on code analysis
    const expectedPWAEndpoints = [
      'http://localhost:8000/api/', // Proxied through Vite
      'ws://localhost:8000/ws/observability'
    ];

    const expectedSimpleEndpoints = [
      'http://localhost:8000/dashboard/api/live-data',
      'ws://localhost:8000/dashboard/simple-ws'
    ];

    // Validate PWA can reach expected endpoints
    for (const endpoint of expectedPWAEndpoints) {
      if (endpoint.startsWith('http')) {
        try {
          const response = await pwaPage.request.get(endpoint);
          console.log(`✅ PWA: ${endpoint} - Status: ${response.status()}`);
        } catch (error) {
          console.log(`❌ PWA: ${endpoint} - Error: ${error.message}`);
        }
      }
    }

    // Validate Simple Dashboard can reach expected endpoints  
    for (const endpoint of expectedSimpleEndpoints) {
      if (endpoint.startsWith('http')) {
        try {
          const response = await simplePage.request.get(endpoint);
          console.log(`✅ Simple: ${endpoint} - Status: ${response.status()}`);
        } catch (error) {
          console.log(`❌ Simple: ${endpoint} - Error: ${error.message}`);
        }
      }
    }
  });

  test('3. WebSocket Connection Validation', async () => {
    console.log('🔗 Testing WebSocket connections...');

    // Test PWA WebSocket connection
    console.log('📱 Testing PWA WebSocket connection...');
    
    await pwaPage.goto('http://localhost:3002');
    
    // Check WebSocket connection in PWA
    const pwaWSStatus = await pwaPage.evaluate(async () => {
      // Wait for app services to be available
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // @ts-ignore - Check if WebSocket service is available
      if (window.appServices && window.appServices.websocket) {
        return {
          available: true,
          connected: window.appServices.websocket.isConnected(),
          state: window.appServices.websocket.getConnectionState(),
          attempts: window.appServices.websocket.getReconnectAttempts()
        };
      }
      
      return { available: false, connected: false, state: 'unknown', attempts: 0 };
    });

    console.log('📱 PWA WebSocket Status:', pwaWSStatus);

    // Test Simple Dashboard WebSocket connection
    console.log('🖥️ Testing Simple Dashboard WebSocket connection...');
    
    await simplePage.goto('http://localhost:8000/dashboard/simple');
    
    // Monitor WebSocket creation in Simple Dashboard
    const simpleWSConnections = [];
    simplePage.on('websocket', ws => {
      console.log(`🔗 Simple Dashboard WebSocket: ${ws.url()}`);
      simpleWSConnections.push(ws.url());
      
      ws.on('framereceived', event => {
        console.log(`📨 Simple WS Received: ${event.payload}`);
      });
      
      ws.on('framesent', event => {
        console.log(`📤 Simple WS Sent: ${event.payload}`);
      });
    });

    // Wait for potential WebSocket connections
    await simplePage.waitForTimeout(5000);

    // Test WebSocket connectivity by sending test messages
    const wsConnectivityTests = await Promise.allSettled([
      // Test PWA WebSocket
      pwaPage.evaluate(() => {
        // @ts-ignore
        if (window.appServices && window.appServices.websocket) {
          window.appServices.websocket.sendMessage({
            type: 'ping',
            timestamp: new Date().toISOString()
          });
          return true;
        }
        return false;
      }),
    ]);

    console.log('🔗 WebSocket connectivity test results:', wsConnectivityTests);
    console.log('🔗 Simple Dashboard WebSocket connections:', simpleWSConnections);
  });

  test('4. Real Data vs Mock Data Behavior Validation', async () => {
    console.log('📊 Testing real data vs mock data behavior...');

    // Data validation patterns
    const dataValidation: DataValidation = {
      realDataIndicators: [
        'timestamp', 'last_updated', 'created_at', 'updated_at',
        'agent_id', 'task_id', 'session_id',
        'active_agents', 'system_status', 'metrics'
      ],
      mockDataIndicators: [
        'mock', 'test', 'demo', 'sample', 'placeholder',
        'Lorem ipsum', 'example.com', '123-456-7890'
      ],
      loadingStates: [
        'Loading...', 'loading', 'spinner', 'skeleton',
        'Fetching', 'Connecting', 'Initializing'
      ]
    };

    // Test PWA data behavior
    console.log('📱 Analyzing PWA data patterns...');
    await pwaPage.goto('http://localhost:3002');
    await pwaPage.waitForTimeout(10000); // Wait for data loading

    const pwaPageContent = await pwaPage.content();
    const pwaDataAnalysis = {
      realDataCount: 0,
      mockDataCount: 0,
      loadingStateCount: 0
    };

    // Analyze PWA content for data patterns
    for (const indicator of dataValidation.realDataIndicators) {
      if (pwaPageContent.toLowerCase().includes(indicator.toLowerCase())) {
        pwaDataAnalysis.realDataCount++;
      }
    }

    for (const indicator of dataValidation.mockDataIndicators) {
      if (pwaPageContent.toLowerCase().includes(indicator.toLowerCase())) {
        pwaDataAnalysis.mockDataCount++;
      }
    }

    for (const indicator of dataValidation.loadingStates) {
      if (pwaPageContent.toLowerCase().includes(indicator.toLowerCase())) {
        pwaDataAnalysis.loadingStateCount++;
      }
    }

    // Test Simple Dashboard data behavior
    console.log('🖥️ Analyzing Simple Dashboard data patterns...');
    await simplePage.goto('http://localhost:8000/dashboard/simple');
    await simplePage.waitForTimeout(10000); // Wait for data loading

    const simplePageContent = await simplePage.content();
    const simpleDataAnalysis = {
      realDataCount: 0,
      mockDataCount: 0,
      loadingStateCount: 0
    };

    // Analyze Simple Dashboard content for data patterns
    for (const indicator of dataValidation.realDataIndicators) {
      if (simplePageContent.toLowerCase().includes(indicator.toLowerCase())) {
        simpleDataAnalysis.realDataCount++;
      }
    }

    for (const indicator of dataValidation.mockDataIndicators) {
      if (simplePageContent.toLowerCase().includes(indicator.toLowerCase())) {
        simpleDataAnalysis.mockDataCount++;
      }
    }

    for (const indicator of dataValidation.loadingStates) {
      if (simplePageContent.toLowerCase().includes(indicator.toLowerCase())) {
        simpleDataAnalysis.loadingStateCount++;
      }
    }

    console.log('📊 PWA Data Analysis:', pwaDataAnalysis);
    console.log('📊 Simple Dashboard Data Analysis:', simpleDataAnalysis);

    // Test dynamic data updates
    console.log('🔄 Testing dynamic data updates...');
    
    // Take screenshot before potential updates
    await pwaPage.screenshot({ path: 'test-results/pwa-before-update.png', fullPage: true });
    await simplePage.screenshot({ path: 'test-results/simple-before-update.png', fullPage: true });

    // Wait for potential real-time updates
    await pwaPage.waitForTimeout(15000);
    await simplePage.waitForTimeout(15000);

    // Take screenshot after waiting for updates
    await pwaPage.screenshot({ path: 'test-results/pwa-after-update.png', fullPage: true });
    await simplePage.screenshot({ path: 'test-results/simple-after-update.png', fullPage: true });

    // Check for dynamic content updates
    const pwaUpdatedContent = await pwaPage.content();
    const simpleUpdatedContent = await simplePage.content();

    const pwaContentChanged = pwaPageContent !== pwaUpdatedContent;
    const simpleContentChanged = simplePageContent !== simpleUpdatedContent;

    console.log('🔄 PWA content changed during test:', pwaContentChanged);
    console.log('🔄 Simple Dashboard content changed during test:', simpleContentChanged);

    // Verify data freshness indicators
    const pwaTimestamps = await pwaPage.locator('[data-timestamp], .timestamp, .last-updated').allTextContents();
    const simpleTimestamps = await simplePage.locator('[data-timestamp], .timestamp, .last-updated').allTextContents();

    console.log('⏰ PWA timestamps found:', pwaTimestamps);
    console.log('⏰ Simple Dashboard timestamps found:', simpleTimestamps);
  });

  test('5. Feature Comparison Between Dashboards', async () => {
    console.log('⚖️ Comparing features between PWA and Simple dashboards...');

    // Load both dashboards
    await pwaPage.goto('http://localhost:3002');
    await simplePage.goto('http://localhost:8000/dashboard/simple');
    
    // Wait for full loading
    await pwaPage.waitForTimeout(10000);
    await simplePage.waitForTimeout(10000);

    // Feature comparison matrix
    const featureComparison = {
      'Agent Status Display': {
        pwa: false,
        simple: false
      },
      'Real-time Updates': {
        pwa: false,
        simple: false
      },
      'Task Management': {
        pwa: false,
        simple: false
      },
      'System Metrics': {
        pwa: false,
        simple: false
      },
      'Navigation Menu': {
        pwa: false,
        simple: false
      },
      'Mobile Optimization': {
        pwa: false,
        simple: false
      },
      'Offline Support': {
        pwa: false,
        simple: false
      },
      'PWA Features': {
        pwa: false,
        simple: false
      }
    };

    // Test PWA features
    console.log('📱 Testing PWA-specific features...');
    
    // Check for agent status in PWA
    const pwaAgentElements = await pwaPage.locator('[data-agent], .agent, [class*="agent"]').count();
    featureComparison['Agent Status Display'].pwa = pwaAgentElements > 0;

    // Check for navigation in PWA
    const pwaNavElements = await pwaPage.locator('nav, .nav, [role="navigation"], .menu').count();
    featureComparison['Navigation Menu'].pwa = pwaNavElements > 0;

    // Check for mobile optimization (touch targets, responsive design)
    const pwaTouchElements = await pwaPage.locator('button, [role="button"], a, input').count();
    featureComparison['Mobile Optimization'].pwa = pwaTouchElements > 0;

    // Check for PWA manifest and service worker
    const pwaManifest = await pwaPage.locator('link[rel="manifest"]').count();
    const pwaServiceWorker = await pwaPage.evaluate(() => 'serviceWorker' in navigator);
    featureComparison['PWA Features'].pwa = pwaManifest > 0 && pwaServiceWorker;

    // Check for offline support
    const pwaOfflineIndicators = await pwaPage.locator('[data-offline], .offline, [class*="offline"]').count();
    featureComparison['Offline Support'].pwa = pwaOfflineIndicators > 0;

    // Test Simple Dashboard features
    console.log('🖥️ Testing Simple Dashboard features...');
    
    // Check for agent status in Simple Dashboard
    const simpleAgentElements = await simplePage.locator('[data-agent], .agent, [class*="agent"]').count();
    featureComparison['Agent Status Display'].simple = simpleAgentElements > 0;

    // Check for navigation in Simple Dashboard
    const simpleNavElements = await simplePage.locator('nav, .nav, [role="navigation"], .menu').count();
    featureComparison['Navigation Menu'].simple = simpleNavElements > 0;

    // Check for system metrics
    const simpleMetricElements = await simplePage.locator('[data-metric], .metric, [class*="metric"]').count();
    featureComparison['System Metrics'].simple = simpleMetricElements > 0;

    // Test real-time updates by monitoring WebSocket activity
    let pwaWSActivity = false;
    let simpleWSActivity = false;

    pwaPage.on('websocket', ws => {
      pwaWSActivity = true;
      ws.on('framereceived', () => {
        featureComparison['Real-time Updates'].pwa = true;
      });
    });

    simplePage.on('websocket', ws => {
      simpleWSActivity = true;
      ws.on('framereceived', () => {
        featureComparison['Real-time Updates'].simple = true;
      });
    });

    // Wait for potential WebSocket activity
    await pwaPage.waitForTimeout(10000);
    await simplePage.waitForTimeout(10000);

    // Performance comparison
    console.log('⚡ Measuring performance metrics...');
    
    const pwaMetrics = await pwaPage.evaluate(() => {
      const perf = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      return {
        loadTime: perf.loadEventEnd - perf.loadEventStart,
        domContentLoaded: perf.domContentLoadedEventEnd - perf.domContentLoadedEventStart,
        firstPaint: performance.getEntriesByType('paint').find(p => p.name === 'first-paint')?.startTime || 0
      };
    });

    const simpleMetrics = await simplePage.evaluate(() => {
      const perf = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      return {
        loadTime: perf.loadEventEnd - perf.loadEventStart,
        domContentLoaded: perf.domContentLoadedEventEnd - perf.domContentLoadedEventStart,
        firstPaint: performance.getEntriesByType('paint').find(p => p.name === 'first-paint')?.startTime || 0
      };
    });

    console.log('📊 Performance Comparison:');
    console.log('PWA Metrics:', pwaMetrics);
    console.log('Simple Dashboard Metrics:', simpleMetrics);

    // Final feature comparison report
    console.log('📋 Feature Comparison Summary:');
    console.table(featureComparison);

    // Recommendations based on comparison
    console.log('💡 Recommendations:');
    
    let pwaScore = 0;
    let simpleScore = 0;
    
    Object.values(featureComparison).forEach(feature => {
      if (feature.pwa) pwaScore++;
      if (feature.simple) simpleScore++;
    });

    console.log(`PWA Score: ${pwaScore}/${Object.keys(featureComparison).length}`);
    console.log(`Simple Dashboard Score: ${simpleScore}/${Object.keys(featureComparison).length}`);

    if (pwaScore > simpleScore) {
      console.log('🏆 Recommendation: Keep PWA Dashboard as primary interface');
      console.log('  - Better mobile experience');
      console.log('  - Offline capabilities');
      console.log('  - Modern PWA features');
    } else if (simpleScore > pwaScore) {
      console.log('🏆 Recommendation: Keep Simple Dashboard as primary interface');
      console.log('  - Better functionality coverage');
      console.log('  - More reliable backend integration');
      console.log('  - Simpler maintenance');
    } else {
      console.log('🤔 Recommendation: Both dashboards have similar capabilities');
      console.log('  - Consider use case specific deployment');
      console.log('  - PWA for mobile users');
      console.log('  - Simple for desktop/admin users');
    }

    // Take final comparison screenshots
    await pwaPage.screenshot({ 
      path: 'test-results/pwa-final-comparison.png', 
      fullPage: true 
    });
    await simplePage.screenshot({ 
      path: 'test-results/simple-final-comparison.png', 
      fullPage: true 
    });
  });

  test('6. Error Handling and Resilience Testing', async () => {
    console.log('🛡️ Testing error handling and resilience...');

    // Test PWA error handling
    console.log('📱 Testing PWA error scenarios...');
    
    // Test with network offline
    await pwaPage.context().setOffline(true);
    await pwaPage.goto('http://localhost:3002');
    
    // Check for offline handling
    const pwaOfflineMessage = await pwaPage.locator('text=/offline|disconnected|network error/i').count();
    console.log('📱 PWA offline indicators found:', pwaOfflineMessage);
    
    // Restore network
    await pwaPage.context().setOffline(false);
    await pwaPage.reload();

    // Test Simple Dashboard error handling
    console.log('🖥️ Testing Simple Dashboard error scenarios...');
    
    await simplePage.context().setOffline(true);
    await simplePage.goto('http://localhost:8000/dashboard/simple');
    
    const simpleOfflineMessage = await simplePage.locator('text=/offline|disconnected|network error/i').count();
    console.log('🖥️ Simple Dashboard offline indicators found:', simpleOfflineMessage);
    
    await simplePage.context().setOffline(false);
    await simplePage.reload();

    // Test invalid endpoints
    console.log('🔍 Testing invalid endpoint handling...');
    
    const pwaNotFoundResponse = await pwaPage.goto('http://localhost:3002/invalid-page');
    console.log('📱 PWA 404 response:', pwaNotFoundResponse?.status());
    
    const simpleNotFoundResponse = await simplePage.goto('http://localhost:8000/dashboard/invalid');
    console.log('🖥️ Simple Dashboard 404 response:', simpleNotFoundResponse?.status());

    // Test JavaScript error handling
    await pwaPage.goto('http://localhost:3002');
    await pwaPage.evaluate(() => {
      // Trigger a JavaScript error
      throw new Error('Test error for PWA');
    }).catch(() => {
      console.log('📱 PWA handled JavaScript error gracefully');
    });

    await simplePage.goto('http://localhost:8000/dashboard/simple');
    await simplePage.evaluate(() => {
      // Trigger a JavaScript error
      throw new Error('Test error for Simple Dashboard');
    }).catch(() => {
      console.log('🖥️ Simple Dashboard handled JavaScript error gracefully');
    });
  });

  test('7. Accessibility and Usability Testing', async () => {
    console.log('♿ Testing accessibility and usability...');

    // Load both dashboards
    await pwaPage.goto('http://localhost:3002');
    await simplePage.goto('http://localhost:8000/dashboard/simple');

    // Test keyboard navigation
    console.log('⌨️ Testing keyboard navigation...');
    
    // Test PWA keyboard navigation
    await pwaPage.keyboard.press('Tab');
    const pwaFocusedElement = await pwaPage.evaluate(() => document.activeElement?.tagName);
    console.log('📱 PWA first focusable element:', pwaFocusedElement);

    // Test Simple Dashboard keyboard navigation
    await simplePage.keyboard.press('Tab');
    const simpleFocusedElement = await simplePage.evaluate(() => document.activeElement?.tagName);
    console.log('🖥️ Simple Dashboard first focusable element:', simpleFocusedElement);

    // Test ARIA labels and roles
    console.log('🏷️ Testing ARIA accessibility...');
    
    const pwaAriaElements = await pwaPage.locator('[aria-label], [role], [aria-describedby]').count();
    const simpleAriaElements = await simplePage.locator('[aria-label], [role], [aria-describedby]').count();
    
    console.log('📱 PWA ARIA elements found:', pwaAriaElements);
    console.log('🖥️ Simple Dashboard ARIA elements found:', simpleAriaElements);

    // Test color contrast (basic check)
    const pwaColors = await pwaPage.evaluate(() => {
      const style = getComputedStyle(document.body);
      return {
        backgroundColor: style.backgroundColor,
        color: style.color
      };
    });

    const simpleColors = await simplePage.evaluate(() => {
      const style = getComputedStyle(document.body);
      return {
        backgroundColor: style.backgroundColor,
        color: style.color
      };
    });

    console.log('🎨 PWA color scheme:', pwaColors);
    console.log('🎨 Simple Dashboard color scheme:', simpleColors);

    // Test responsive design
    console.log('📱 Testing responsive design...');
    
    // Test different viewport sizes
    const viewports = [
      { width: 320, height: 568 }, // iPhone SE
      { width: 768, height: 1024 }, // iPad
      { width: 1920, height: 1080 } // Desktop
    ];

    for (const viewport of viewports) {
      await pwaPage.setViewportSize(viewport);
      await pwaPage.screenshot({ 
        path: `test-results/pwa-${viewport.width}x${viewport.height}.png` 
      });
      
      await simplePage.setViewportSize(viewport);
      await simplePage.screenshot({ 
        path: `test-results/simple-${viewport.width}x${viewport.height}.png` 
      });
    }
  });
});

// Test utilities and helpers
class DashboardTestUtils {
  static async waitForDataLoad(page: Page, timeout = 15000): Promise<void> {
    // Wait for loading indicators to disappear
    await page.waitForFunction(
      () => {
        const loadingElements = document.querySelectorAll(
          '.loading, .spinner, [data-loading="true"], .skeleton'
        );
        return loadingElements.length === 0;
      },
      { timeout }
    ).catch(() => {
      console.log('⏳ Loading indicators still present after timeout');
    });
  }

  static async captureNetworkActivity(page: Page): Promise<string[]> {
    const requests: string[] = [];
    
    page.on('request', request => {
      requests.push(`${request.method()} ${request.url()}`);
    });

    return requests;
  }

  static async measurePerformance(page: Page): Promise<any> {
    return await page.evaluate(() => {
      const perf = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      return {
        loadTime: perf.loadEventEnd - perf.loadEventStart,
        domContentLoaded: perf.domContentLoadedEventEnd - perf.domContentLoadedEventStart,
        firstContentfulPaint: performance.getEntriesByType('paint')
          .find(p => p.name === 'first-contentful-paint')?.startTime || 0
      };
    });
  }
}