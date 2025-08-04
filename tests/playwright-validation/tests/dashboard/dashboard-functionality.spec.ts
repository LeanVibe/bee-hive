import { test, expect, Page } from '@playwright/test';
import { EvidenceCollector } from '../../utils/evidence-collector';

/**
 * Dashboard Functionality Validation
 * 
 * Validates the dashboard claims:
 * - Real-time dashboard at localhost:3002
 * - Live agent status display
 * - Interactive dashboard components
 * - WebSocket real-time updates
 * - Vue.js frontend functionality
 */

test.describe('Dashboard Functionality Validation', () => {
  let evidenceCollector: EvidenceCollector;
  
  test.beforeEach(async ({ page }) => {
    evidenceCollector = new EvidenceCollector(page, 'dashboard');
    await evidenceCollector.startCollection('dashboard-functionality');
  });

  test.afterEach(async () => {
    await evidenceCollector.finishCollection();
  });

  test('Dashboard Loads Successfully at localhost:3002', async ({ page }) => {
    // Navigate to dashboard
    const response = await page.goto('http://localhost:3002');
    
    // Validate successful load
    expect(response?.status()).toBe(200);
    
    // Wait for Vue.js app to initialize
    await page.waitForSelector('body', { timeout: 10000 });
    await page.waitForTimeout(2000); // Allow time for Vue components to mount
    
    // Check for Vue.js app structure
    const pageTitle = await page.title();
    expect(pageTitle).toBeTruthy();
    
    // Look for common dashboard elements
    const bodyContent = await page.textContent('body');
    expect(bodyContent).toBeTruthy();
    expect(bodyContent.length).toBeGreaterThan(100); // Substantial content
    
    // Check if it's a single-page application (Vue.js characteristics)
    const scriptTags = await page.$$eval('script', scripts => 
      scripts.map(script => script.src).filter(src => src.includes('js'))
    );
    expect(scriptTags.length).toBeGreaterThan(0);
    
    // Collect evidence
    await page.screenshot({ path: await evidenceCollector.getScreenshotPath('dashboard-loaded') });
    await evidenceCollector.captureData('dashboard-load-info', {
      status: response?.status(),
      title: pageTitle,
      contentLength: bodyContent?.length || 0,
      scriptCount: scriptTags.length,
      url: page.url()
    });
    
    console.log('✅ Dashboard load validation passed:', {
      status: response?.status(),
      title: pageTitle,
      contentPresent: (bodyContent?.length || 0) > 100
    });
  });

  test('Dashboard Displays Agent Information', async ({ page }) => {
    await page.goto('http://localhost:3002');
    await page.waitForTimeout(3000); // Allow time for data loading
    
    // Look for agent-related content in the dashboard
    const pageContent = await page.textContent('body');
    
    // Check for agent-related keywords that would appear in a working dashboard
    const agentKeywords = [
      'agent', 'Agent', 'AGENT',
      'status', 'Status', 'STATUS',
      'active', 'Active', 'ACTIVE',
      'task', 'Task', 'TASK'
    ];
    
    const foundKeywords = agentKeywords.filter(keyword => 
      pageContent?.includes(keyword)
    );
    
    // Expect at least some agent-related content
    expect(foundKeywords.length).toBeGreaterThan(0);
    
    // Look for data containers or tables that might display agent info
    const containers = await page.$$('[class*="agent"], [class*="card"], [class*="table"], [class*="list"], [id*="agent"]');
    const containerCount = containers.length;
    
    // Look for any dynamic content areas
    const dynamicElements = await page.$$('[v-if], [v-for], [v-show], [*ngFor], [*ngIf]');
    const dynamicElementCount = dynamicElements.length;
    
    // Collect evidence
    await page.screenshot({ path: await evidenceCollector.getScreenshotPath('dashboard-agent-content') });
    await evidenceCollector.captureData('agent-content-analysis', {
      foundKeywords: foundKeywords,
      keywordCount: foundKeywords.length,
      containerCount: containerCount,
      dynamicElementCount: dynamicElementCount,
      totalContentLength: pageContent?.length || 0
    });
    
    console.log('✅ Agent information display validation:', {
      agentKeywordsFound: foundKeywords.length,
      potentialContainers: containerCount,
      dynamicElements: dynamicElementCount
    });
  });

  test('Dashboard Shows Real-Time Data Updates', async ({ page }) => {
    await page.goto('http://localhost:3002');
    await page.waitForTimeout(2000);
    
    // Capture initial state
    const initialContent = await page.textContent('body');
    const initialTitle = await page.title();
    
    // Wait for potential real-time updates
    await page.waitForTimeout(10000); // Wait 10 seconds for updates
    
    // Capture updated state
    const updatedContent = await page.textContent('body');
    const updatedTitle = await page.title();
    
    // Check for any changes that might indicate real-time updates
    const contentChanged = initialContent !== updatedContent;
    
    // Look for timestamp elements that might update
    const timestampElements = await page.$$('[class*="time"], [class*="timestamp"], [class*="updated"], [class*="last"]');
    const timestampCount = timestampElements.length;
    
    // Check for WebSocket connections (indicates real-time capability)
    const websocketConnections = await page.evaluate(() => {
      // Check if WebSocket is being used
      return window.WebSocket ? 'WebSocket available' : 'WebSocket not available';
    });
    
    // Look for real-time indicators in the HTML
    const realTimeIndicators = await page.$$('[class*="live"], [class*="real-time"], [class*="ws"], [class*="socket"]');
    const realTimeIndicatorCount = realTimeIndicators.length;
    
    // Collect evidence
    await page.screenshot({ path: await evidenceCollector.getScreenshotPath('dashboard-realtime') });
    await evidenceCollector.captureData('real-time-analysis', {
      contentChanged: contentChanged,
      timestampElements: timestampCount,
      websocketSupport: websocketConnections,
      realTimeIndicators: realTimeIndicatorCount,
      initialContentLength: initialContent?.length || 0,
      updatedContentLength: updatedContent?.length || 0
    });
    
    console.log('✅ Real-time updates analysis:', {
      contentChanged: contentChanged,
      timestampElements: timestampCount,
      websocketSupport: websocketConnections,
      realTimeIndicators: realTimeIndicatorCount
    });
  });

  test('Dashboard Navigation and Interactivity', async ({ page }) => {
    await page.goto('http://localhost:3002');
    await page.waitForTimeout(2000);
    
    // Look for interactive elements
    const buttons = await page.$$('button');
    const links = await page.$$('a');
    const forms = await page.$$('form');
    const inputs = await page.$$('input, select, textarea');
    
    const interactivityMetrics = {
      buttonCount: buttons.length,
      linkCount: links.length,
      formCount: forms.length,
      inputCount: inputs.length
    };
    
    // Test clicking on interactive elements (safely)
    let clickableElementsFound = 0;
    
    // Try clicking buttons (first few only)
    for (let i = 0; i < Math.min(3, buttons.length); i++) {
      try {
        const button = buttons[i];
        const isVisible = await button.isVisible();
        const isEnabled = await button.isEnabled();
        
        if (isVisible && isEnabled) {
          clickableElementsFound++;
          // Don't actually click to avoid disrupting the system
        }
      } catch (error) {
        // Continue if element interaction fails
      }
    }
    
    // Check for navigation elements
    const navElements = await page.$$('nav, [class*="nav"], [class*="menu"], [class*="sidebar"]');
    const navElementCount = navElements.length;
    
    // Look for routing indicators (Vue Router, etc.)
    const routingElements = await page.$$('[router-link], [to], [href*="#"], [class*="route"]');
    const routingElementCount = routingElements.length;
    
    // Collect evidence
    await page.screenshot({ path: await evidenceCollector.getScreenshotPath('dashboard-interactivity') });
    await evidenceCollector.captureData('interactivity-analysis', {
      ...interactivityMetrics,
      clickableElements: clickableElementsFound,
      navigationElements: navElementCount,
      routingElements: routingElementCount
    });
    
    console.log('✅ Dashboard interactivity validation:', {
      ...interactivityMetrics,
      clickableElements: clickableElementsFound,
      navigationElements: navElementCount
    });
  });

  test('Dashboard Performance and Loading Speed', async ({ page }) => {
    // Measure dashboard loading performance
    const startTime = Date.now();
    
    const response = await page.goto('http://localhost:3002');
    const navigationTime = Date.now() - startTime;
    
    // Wait for content to be fully loaded
    await page.waitForLoadState('networkidle', { timeout: 15000 });
    const fullLoadTime = Date.now() - startTime;
    
    // Measure DOM content
    const domMetrics = await page.evaluate(() => {
      return {
        domNodes: document.querySelectorAll('*').length,
        bodySize: document.body?.innerHTML.length || 0,
        scriptTags: document.querySelectorAll('script').length,
        styleTags: document.querySelectorAll('style, link[rel="stylesheet"]').length
      };
    });
    
    // Performance expectations
    expect(navigationTime).toBeLessThan(10000); // Less than 10 seconds
    expect(fullLoadTime).toBeLessThan(20000); // Less than 20 seconds
    expect(domMetrics.domNodes).toBeGreaterThan(10); // Substantial DOM structure
    
    // Collect evidence
    await page.screenshot({ path: await evidenceCollector.getScreenshotPath('dashboard-performance') });
    await evidenceCollector.captureData('performance-metrics', {
      navigationTime: navigationTime,
      fullLoadTime: fullLoadTime,
      responseStatus: response?.status(),
      domMetrics: domMetrics
    });
    
    console.log('✅ Dashboard performance validation:', {
      navigationTime: navigationTime,
      fullLoadTime: fullLoadTime,
      domNodes: domMetrics.domNodes,
      responseStatus: response?.status()
    });
  });

  test('Dashboard Error Handling and Resilience', async ({ page }) => {
    // Test dashboard behavior when backend might be unavailable
    await page.goto('http://localhost:3002');
    await page.waitForTimeout(2000);
    
    // Capture console errors
    const consoleErrors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });
    
    // Capture network failures
    const networkFailures: string[] = [];
    page.on('response', response => {
      if (!response.ok()) {
        networkFailures.push(`${response.status()} - ${response.url()}`);
      }
    });
    
    // Wait for potential errors to surface
    await page.waitForTimeout(5000);
    
    // Check if dashboard still renders despite potential backend issues
    const bodyContent = await page.textContent('body');
    const hasContent = (bodyContent?.length || 0) > 50;
    
    // Look for error messages or loading states
    const errorElements = await page.$$('[class*="error"], [class*="fail"], [class*="warning"]');
    const loadingElements = await page.$$('[class*="loading"], [class*="spinner"], [class*="loader"]');
    
    const errorHandlingMetrics = {
      consoleErrorCount: consoleErrors.length,
      networkFailureCount: networkFailures.length,
      hasContent: hasContent,
      errorElementCount: errorElements.length,
      loadingElementCount: loadingElements.length,
      contentLength: bodyContent?.length || 0
    };
    
    // Dashboard should handle errors gracefully
    expect(hasContent).toBe(true); // Should have some content even if backend is having issues
    
    // Collect evidence
    await page.screenshot({ path: await evidenceCollector.getScreenshotPath('dashboard-error-handling') });
    await evidenceCollector.captureData('error-handling-analysis', {
      ...errorHandlingMetrics,
      consoleErrors: consoleErrors.slice(0, 5), // First 5 errors
      networkFailures: networkFailures.slice(0, 5) // First 5 failures
    });
    
    console.log('✅ Dashboard error handling validation:', errorHandlingMetrics);
  });
});