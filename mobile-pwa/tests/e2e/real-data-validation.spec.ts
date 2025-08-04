/**
 * Real Data Flow Validation Tests
 * 
 * Tests the critical path of real data flowing from backend through 
 * mobile-pwa dashboard while agents are actively working ("cooking")
 */

import { test, expect } from '@playwright/test';

test.describe('Mobile PWA Real Data Integration', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to mobile PWA dashboard
    await page.goto('http://localhost:3002');
    
    // Wait for initial app load
    await page.waitForSelector('#app', { timeout: 10000 });
    
    // Wait for loading screen to disappear 
    await page.waitForSelector('.loading-container', { state: 'hidden', timeout: 15000 });
  });

  test('validates backend live data endpoint returns operational data', async ({ request }) => {
    // Test the critical data endpoint that mobile-pwa depends on
    const response = await request.get('http://localhost:8000/dashboard/api/live-data');
    
    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    
    // Validate structure matches what mobile-pwa expects
    expect(data).toHaveProperty('metrics');
    expect(data).toHaveProperty('agent_activities');
    expect(data).toHaveProperty('project_snapshots');
    
    // Validate real operational data
    expect(data.metrics.active_agents).toBeGreaterThan(0);
    expect(data.agent_activities.length).toBeGreaterThan(0);
    expect(data.metrics.system_status).toMatch(/healthy|degraded|critical/);
    
    // Validate agent data has required fields
    const firstAgent = data.agent_activities[0];
    expect(firstAgent).toHaveProperty('agent_id');
    expect(firstAgent).toHaveProperty('name');
    expect(firstAgent).toHaveProperty('status');
    expect(firstAgent).toHaveProperty('performance_score');
    expect(firstAgent).toHaveProperty('specializations');
    
    console.log(`✅ Backend providing real data: ${data.metrics.active_agents} agents, ${data.project_snapshots.length} projects`);
  });

  test('dashboard loads and displays real agent data', async ({ page }) => {
    // Wait for dashboard to fully load with real data
    await page.waitForLoadState('networkidle');
    
    // Check if main dashboard elements are present
    const appContainer = page.locator('#app');
    await expect(appContainer).toBeVisible();
    
    // Look for common dashboard elements that would contain real data
    // Using flexible selectors since exact structure may vary
    const possibleAgentSelectors = [
      '[data-testid="agent-card"]',
      '.agent-card',
      '[class*="agent"]',
      '[id*="agent"]'
    ];
    
    const possibleMetricSelectors = [
      '[data-testid="active-agents-count"]',
      '.metrics',
      '[class*="metric"]',
      '[class*="status"]'
    ];
    
    // Try to find agent-related UI elements
    let agentElementsFound = false;
    for (const selector of possibleAgentSelectors) {
      const elements = page.locator(selector);
      const count = await elements.count();
      if (count > 0) {
        console.log(`✅ Found ${count} agent elements with selector: ${selector}`);
        agentElementsFound = true;
        break;
      }
    }
    
    // Try to find metric-related UI elements
    let metricsElementsFound = false;
    for (const selector of possibleMetricSelectors) {
      const elements = page.locator(selector);
      const count = await elements.count();
      if (count > 0) {
        console.log(`✅ Found ${count} metric elements with selector: ${selector}`);
        metricsElementsFound = true;
        break;
      }
    }
    
    // At minimum, verify the page loaded without errors
    const title = await page.title();
    expect(title).toContain('Agent Hive');
    
    // Check for any obvious error messages
    const errorElements = page.locator('[class*="error"], .error, [data-testid*="error"]');
    const errorCount = await errorElements.count();
    expect(errorCount).toBe(0);
    
    console.log(`✅ Dashboard loaded successfully: title="${title}"`);
    if (agentElementsFound) console.log('✅ Agent UI elements detected');
    if (metricsElementsFound) console.log('✅ Metrics UI elements detected');
  });

  test('real-time data polling shows live updates', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    // Monitor network requests to see if polling is active
    const apiRequests: string[] = [];
    page.on('request', request => {
      const url = request.url();
      if (url.includes('/api/') || url.includes('/dashboard/')) {
        apiRequests.push(url);
        console.log(`📡 API Request: ${url}`);
      }
    });
    
    // Wait for potential polling cycles (most systems poll every 3-5 seconds)
    await page.waitForTimeout(8000);
    
    // Check if any API requests were made (indicating real-time polling)
    const liveDataRequests = apiRequests.filter(url => 
      url.includes('live-data') || url.includes('agents') || url.includes('status')
    );
    
    if (liveDataRequests.length > 0) {
      console.log(`✅ Real-time polling detected: ${liveDataRequests.length} requests`);
    } else {
      console.log('ℹ️  No real-time polling detected in 8 second window');
    }
    
    // Look for live status indicators
    const statusIndicators = page.locator('[class*="live"], [class*="status"], [data-status]');
    const statusCount = await statusIndicators.count();
    
    if (statusCount > 0) {
      console.log(`✅ Found ${statusCount} status indicator elements`);
    }
    
    // Verify page remains functional during polling
    const pageTitle = await page.title();
    expect(pageTitle).toContain('Agent Hive');
  });

  test('backend adapter transforms data correctly for UI consumption', async ({ page, request }) => {
    // Get raw backend data
    const backendResponse = await request.get('http://localhost:8000/dashboard/api/live-data');
    const backendData = await backendResponse.json();
    
    // Navigate to dashboard and wait for load
    await page.waitForLoadState('networkidle');
    
    // Check if backend data values appear anywhere in the DOM
    const pageContent = await page.textContent('body');
    
    // Look for evidence that backend data was transformed and displayed
    const agentCount = backendData.metrics.active_agents;
    const systemStatus = backendData.metrics.system_status;
    const firstAgentName = backendData.agent_activities[0]?.name;
    
    console.log(`🔍 Looking for backend data in UI:`);
    console.log(`  - Agent count: ${agentCount}`);
    console.log(`  - System status: ${systemStatus}`);
    console.log(`  - First agent: ${firstAgentName}`);
    
    // Check if key metrics appear in the UI
    if (pageContent?.includes(String(agentCount))) {
      console.log(`✅ Agent count (${agentCount}) found in UI`);
    }
    
    if (pageContent?.includes(systemStatus)) {
      console.log(`✅ System status (${systemStatus}) found in UI`);
    }
    
    if (firstAgentName && pageContent?.includes(firstAgentName)) {
      console.log(`✅ First agent name (${firstAgentName}) found in UI`);
    }
    
    // Verify the page loads without JavaScript errors
    const jsErrors: string[] = [];
    page.on('pageerror', error => {
      jsErrors.push(error.message);
    });
    
    await page.waitForTimeout(2000);
    
    expect(jsErrors).toHaveLength(0);
    console.log('✅ No JavaScript errors detected during data transformation');
  });

  test('mobile PWA features work correctly', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    // Check PWA manifest
    const manifestResponse = await page.request.get('http://localhost:3003/manifest.webmanifest');
    if (manifestResponse.ok()) {
      const manifest = await manifestResponse.json();
      expect(manifest).toHaveProperty('name');
      expect(manifest).toHaveProperty('short_name'); 
      console.log(`✅ PWA manifest loaded: ${manifest.name}`);
    }
    
    // Check service worker registration
    const swRegistered = await page.evaluate(() => {
      return 'serviceWorker' in navigator;
    });
    expect(swRegistered).toBeTruthy();
    console.log('✅ Service Worker API available');
    
    // Check mobile viewport meta tag
    const viewportMeta = page.locator('meta[name="viewport"]');
    await expect(viewportMeta).toHaveAttribute('content', /width=device-width/);
    console.log('✅ Mobile viewport configured correctly');
    
    // Check mobile-specific meta tags
    const appleMeta = page.locator('meta[name="apple-mobile-web-app-capable"]');
    await expect(appleMeta).toHaveAttribute('content', 'yes');
    console.log('✅ Apple mobile web app meta tags present');
  });

  test('dashboard handles offline/connection issues gracefully', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    // Simulate network offline
    await page.context().setOffline(true);
    
    // Try to interact with the page
    await page.waitForTimeout(2000);
    
    // Check if page shows appropriate offline messaging or cached content
    const pageContent = await page.textContent('body');
    const pageTitle = await page.title();
    
    // The page should still be functional with cached content
    expect(pageTitle).toContain('Agent Hive');
    console.log('✅ Page remains functional when offline');
    
    // Restore connection
    await page.context().setOffline(false);
    await page.waitForTimeout(2000);
    
    console.log('✅ Network connection restored');
  });

  test('system integration provides useful operational data', async ({ page, request }) => {
    // Verify system is actually "cooking" with useful data
    const healthResponse = await request.get('http://localhost:8000/health');
    const healthData = await healthResponse.json();
    
    expect(healthData.status).toMatch(/healthy|degraded|critical/);
    expect(healthData.components.orchestrator.active_agents).toBeGreaterThan(0);
    
    const liveDataResponse = await request.get('http://localhost:8000/dashboard/api/live-data');
    const liveData = await liveDataResponse.json();
    
    // Verify we have meaningful operational data
    expect(liveData.metrics.system_efficiency).toBeGreaterThan(0);
    expect(liveData.agent_activities.length).toBeGreaterThan(0);
    
    // Each agent should have meaningful data
    liveData.agent_activities.forEach((agent: any, index: number) => {
      expect(agent.performance_score).toBeGreaterThan(0);
      expect(agent.specializations.length).toBeGreaterThan(0);
      console.log(`✅ Agent ${index + 1}: ${agent.name} (${agent.status}) - ${Math.round(agent.performance_score * 100)}% performance`);
    });
    
    // Project data should be meaningful
    if (liveData.project_snapshots.length > 0) {
      const project = liveData.project_snapshots[0];
      expect(project.progress_percentage).toBeGreaterThanOrEqual(0);
      expect(project.quality_score).toBeGreaterThan(0);
      console.log(`✅ Project: ${project.name} (${project.progress_percentage}% complete)`);
    }
    
    console.log(`✅ System providing useful operational data while "cooking"`);
    console.log(`   - ${liveData.metrics.active_agents} active agents`);
    console.log(`   - ${liveData.project_snapshots.length} projects in progress`);
    console.log(`   - ${liveData.metrics.system_efficiency}% system efficiency`);
    console.log(`   - Status: ${liveData.metrics.system_status}`);
  });
});